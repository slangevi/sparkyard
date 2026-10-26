#!/usr/bin/env python3
"""Adaptive vLLM launcher for the sparkyard stack (Python 3, stdlib only).

Computes --gpu-memory-utilization from model weights + a KV-cache estimate +
/proc/meminfo, enforcing the GB10 SYSTEM_RAM_CEILING crash-guard, then
exec's `docker run ... vllm serve ...`. Runs inside the minimal llama-swap
container; delivered via the /app mount and invoked as
`python3 /app/scripts/launch.py <vllm flags...>`.

Numeric behavior is faithful to the original bash launcher: every value that
the bash printed with "%.2f" before reuse is rounded here the same way (g2),
so a cross-launcher parity harness sees identical gpu-memory-utilization.

Env contract (same names/defaults as the original):
  required: MODEL_PATH MODEL_HOST_PATH CONTAINER_NAME IMAGE PORT HOST
  tunable:  MAX_MODEL_LEN=131072 MAX_NUM_SEQS=10 KV_DTYPE_BYTES=1
            GMEM_MIN=0.55 GMEM_MAX=0.92 SAFETY_GIB=4 CUDA_OVERHEAD_GIB=6.3
            SYSTEM_RAM_CEILING_GIB=117.81 KV_BATCH_REALISTIC=4 GMEM_FREE_BUFFER_GIB=5
  optional: GMEM_OVERRIDE (number in (0,1) -> static; "adaptive"/unset -> compute)
            EXTRA_DOCKER_ARGS PRE_LAUNCH_CMD VLLM_SERVE_PREFIX(default "vllm serve")
            RECLAIM_CONTAINERS (space/comma-separated container names) RECLAIM_WAIT_S=90
              RECLAIM_MARGIN_GIB=10 RECLAIM_BUSY_CPU=20 RECLAIM_BUSY_WAIT_S=300
              when free memory is short — below GMEM_MIN, or below this model's
              share plus RECLAIM_MARGIN_GIB (vLLM's real footprint exceeds
              gmem*total by about that much) — restart those of them that are
              running and idle, wait for the memory to come back, and size again.
              A container above RECLAIM_BUSY_CPU percent is waited for (up to
              RECLAIM_BUSY_WAIT_S) and left alone if it stays busy, so a job in
              flight is never killed. For co-tenants that cache models in unified
              memory and do not hand it back on request.
  cluster:  (two-node tensor-parallel, set by the rendered cmd) NODES=2 WORKER_SSH_HOST
            WORKER_SSH_USER WORKER_CX7_IP HEAD_CX7_IP MASTER_PORT ETH_IF IB_HCA;
            CLUSTER_SSH_DIR=/app/cluster holds id_ed25519 + known_hosts for the ssh
            to the worker. Rank 0 runs here, rank 1 (--headless) on the worker.
            Preflight (ssh, image layers, weights on the worker; stale `-w` removed),
            then both nodes are sized at tp=2 and both get the smaller gmem.
            EXTRA_PATHS (optional, comma-separated, relative to LLM_ROOT_PATH):
            other paths the flags reference (e.g. a draft model) that preflight
            also requires on the worker.
            The worker starts detached over ssh; the head runs as a child (not
            exec) so SIGTERM/SIGINT are forwarded and the worker is stopped when
            the head exits. The worker is polled (docker container inspect over ssh, ~5 s);
            if it is gone the head is SIGTERMed instead of waiting out the
            torch.distributed timeout. The head API binds to CLUSTER_API_HOST, else the IP
            of host.docker.internal (never 0.0.0.0).
            `--stop [PID]` (needs only CONTAINER_NAME + the cluster env) first
            SIGTERMs the launcher PID (llama-swap's ${PID}) so a launch still in
            preflight/sizing aborts before starting anything, then stops the
            worker's <name>-w and the local <name>, tolerating all being gone.
  test-only: SPARKYARD_MEMINFO_PATH (override /proc/meminfo), SPARKYARD_LAUNCH_PRINT=1
Pass `--print` (or SPARKYARD_LAUNCH_PRINT=1) to print gmem + docker argv and exit
without launching (cluster mode: head argv, `--- worker ---`, worker argv; it
still makes the read-only ssh/docker preflight calls and reads the worker's
meminfo). Remaining args are forwarded verbatim after the vllm flags.
"""
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NoReturn

GIB = 1073741824


def g2(x):
    """Round to 2 decimals exactly as the bash awk `printf "%.2f"` did, so
    intermediate values feed the next step identically."""
    return float(f"{float(x):.2f}")


def _fatal(msg) -> NoReturn:
    sys.stderr.write(msg + "\n")
    sys.exit(1)


def _env_num(name, default, cast):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        _fatal(f"[launch] FATAL: env {name}={raw!r} is not a valid {cast.__name__}")


@dataclass
class Params:
    max_model_len: int
    max_num_seqs: int
    kv_dtype_bytes: int
    gmin: float
    gmax: float
    safety: float
    cuda_overhead: float
    ceiling: float
    kv_batch_realistic: int
    free_buffer: float
    weights: float
    tp: int = 1


@dataclass
class ClusterEnv:
    worker_host: str
    worker_user: str
    worker_ip: str
    head_ip: str
    master_port: str
    eth_if: str
    ib_hca: str
    ssh_dir: str


@dataclass
class LaunchEnv:
    model_path: str
    model_host_path: str
    container_name: str
    image: str
    port: str
    host: str
    max_model_len: int
    max_num_seqs: int
    extra_docker_args: str
    pre_launch_cmd: str
    vllm_serve_prefix: str


def weights_gib(model_host_path):
    total = 0
    d = Path(model_host_path)
    if d.is_dir():
        for f in d.glob("*.safetensors"):  # maxdepth 1
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return g2(total / GIB)


def _extract(cfg, key):
    """Prefer a nested text_config block (multimodal models nest the LLM cfg
    there); else top-level. Return the int value or 0. Accepts JSON ints or
    floats (some configs use 128.0); excludes bool (isinstance(True,int) is True)."""
    def _num(v):
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    tc = cfg.get("text_config")
    if isinstance(tc, dict):
        v = _num(tc.get(key))
        if v is not None:
            return v
    v = _num(cfg.get(key))
    return v if v is not None else 0


def _extract_raw(cfg, key):
    """Like _extract but returns the raw value (lists, etc.), or None."""
    tc = cfg.get("text_config")
    if isinstance(tc, dict) and key in tc:
        return tc[key]
    return cfg.get(key)


def _attention_layers(cfg, n_layers):
    """Number of layers that actually hold a KV cache.

    Hybrid models (Qwen3.5/3.6/3.8 `qwen3_5`) interleave linear-attention
    (Gated DeltaNet) layers, which carry a small fixed recurrent state instead
    of a per-token KV cache, with a full-attention layer every
    `full_attention_interval`. Counting all `num_hidden_layers` overestimates
    the KV cache by that interval. Dense models fall through to n_layers."""
    types = _extract_raw(cfg, "layer_types")
    if isinstance(types, list) and types:
        n = sum(1 for t in types if t == "full_attention")
        if n:
            return n
    interval = _extract(cfg, "full_attention_interval")
    if interval > 1:
        n = n_layers // interval
        if n:
            return n
    # nemotron_h: '*' = attention, 'M' = Mamba, '-' = MLP.
    pattern = _extract_raw(cfg, "hybrid_override_pattern")
    if isinstance(pattern, str):
        n = pattern.count("*")
        if n:
            return n
    # nemotron_h / nemotron_h_puzzle: a per-block list of
    # 'mamba' | 'moe' | 'attention'; only the attention blocks hold a KV cache.
    blocks = _extract_raw(cfg, "layers_block_type")
    if isinstance(blocks, list) and blocks:
        n = sum(1 for b in blocks if b == "attention")
        if n:
            return n
    return n_layers


def parse_config(model_host_path):
    cfgp = Path(model_host_path) / "config.json"
    if not cfgp.is_file():
        _fatal(f"[auto-gmem] FATAL: config.json not found at {cfgp}")
    try:
        cfg = json.loads(cfgp.read_text())
    except (ValueError, OSError) as e:
        _fatal(f"[auto-gmem] FATAL: could not read {cfgp}: {e}")
    n_layers = _extract(cfg, "num_hidden_layers")
    if n_layers == 0:
        blocks = _extract_raw(cfg, "layers_block_type")
        if isinstance(blocks, list):
            n_layers = len(blocks)
    n_heads = _extract(cfg, "num_attention_heads")
    n_kv = _extract(cfg, "num_key_value_heads")
    hidden = _extract(cfg, "hidden_size")
    head_dim = _extract(cfg, "head_dim")
    if n_kv == 0:
        n_kv = n_heads
    if head_dim == 0 and n_heads != 0:
        head_dim = int(hidden / n_heads)
    if n_layers == 0 or head_dim == 0 or n_kv == 0:
        _fatal(f"[auto-gmem] FATAL: could not parse layers/heads/head_dim from {cfgp}")
    return n_layers, n_heads, n_kv, head_dim, _attention_layers(cfg, n_layers)


def read_meminfo(path=None):
    path = path or os.environ.get("SPARKYARD_MEMINFO_PATH", "/proc/meminfo")
    try:
        text = Path(path).read_text()
    except OSError as e:
        _fatal(f"[auto-gmem] FATAL: could not read meminfo at {path}: {e}")
    return parse_meminfo(text)


def parse_meminfo(text):
    """(MemTotal, MemAvailable, MemFree) in kB from /proc/meminfo text."""
    total = avail = free = 0
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "MemTotal:":
            total = int(parts[1])
        elif parts[0] == "MemAvailable:":
            avail = int(parts[1])
        elif parts[0] == "MemFree:":
            free = int(parts[1])
    return total, avail, free


def compute_gmem(cfg_vals, params, meminfo):
    """Return (gmem: float|None, mode: str, diag: dict). mode in
    {sized, fallback, ERR}. Faithful to the bash awk pipeline."""
    n_layers, n_heads, n_kv, head_dim = cfg_vals[:4]
    n_attn = cfg_vals[4] if len(cfg_vals) > 4 else n_layers
    p = params
    mem_total_kb, mem_avail_kb, mem_free_kb = meminfo

    kv_batch = min(p.max_num_seqs, p.kv_batch_realistic)
    # Per-node share under tensor parallelism: weights and KV both split by tp.
    weights = g2(p.weights / p.tp)
    kv = g2((2 * n_attn * n_kv * head_dim * p.max_model_len * kv_batch * p.kv_dtype_bytes) / GIB / p.tp)
    need = g2(weights + kv + p.safety)

    mt_raw = g2(mem_total_kb / 1048576)
    ma_raw = g2(mem_avail_kb / 1048576)
    mf_raw = g2(mem_free_kb / 1048576)
    headroom = g2(max(0.0, mt_raw - p.ceiling))
    mt_capped = g2(min(mt_raw, p.ceiling))
    total = g2(mt_capped - p.cuda_overhead)
    free = g2(max(0.0, ma_raw - headroom - p.cuda_overhead))

    diag = {"layers": n_layers, "attn_layers": n_attn,
            "kv_heads": n_kv, "head_dim": head_dim,
            "ctx": p.max_model_len, "batch": p.max_num_seqs, "kvb": p.kv_dtype_bytes,
            "kv_batch": kv_batch, "tp": p.tp, "weights": weights, "kv": kv, "safety": p.safety,
            "need": need, "mt": mt_raw, "mf": mf_raw, "ma": ma_raw, "ceiling": p.ceiling,
            "headroom": headroom, "free": free, "total": total}

    if total <= 0:
        return None, "ERR", dict(diag, u_cap=0.0)
    u_cap = (free - p.free_buffer) / total
    if u_cap < 0:
        u_cap = 0.0
    if u_cap < p.gmin:
        return None, "ERR", dict(diag, u_cap=g2(u_cap))
    if need <= free:
        u = need / total
        u = max(p.gmin, min(p.gmax, u))
        if u > u_cap:
            u = u_cap
        mode = "sized"
    else:
        u = p.gmax
        if u > u_cap:
            u = u_cap
        if u < p.gmin:
            u = p.gmin
        mode = "fallback"
    return g2(u), mode, diag


_CONTAINER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_reclaim(raw):
    names = [n for n in re.split(r"[\s,]+", raw or "") if n]
    for n in names:
        if not _CONTAINER_NAME.match(n):
            _fatal(f"[reclaim] FATAL: RECLAIM_CONTAINERS entry {n!r} is not a container name")
    return names


def _docker(argv):
    r = subprocess.run(argv, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def short_of_memory(gmem, mode, diag, margin):
    """Short if sizing failed outright, or if free memory does not cover this
    model's share plus `margin` GiB of the overhead vLLM uses beyond it."""
    if mode == "ERR" or gmem is None:
        return True
    return diag["free"] < gmem * diag["total"] + margin


def parse_cpu(text):
    m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)%\s*", text or "")
    return float(m.group(1)) if m else None


def _wait_until_idle(name, run, sleep, busy_cpu, busy_wait_s, poll=5):
    """True once `name` is at or below busy_cpu percent CPU (or its CPU cannot
    be read); False if it stays busy for busy_wait_s seconds."""
    waited = 0
    while True:
        cpu = parse_cpu(run(["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}", name]))
        if cpu is None or cpu <= busy_cpu:
            return True
        if waited >= busy_wait_s:
            return False
        if waited == 0:
            sys.stderr.write(f"[reclaim] {name} is busy ({cpu:.0f}% CPU): waiting up to {busy_wait_s}s\n")
        sleep(poll)
        waited += poll


def reclaim_and_recompute(names, cfg_vals, params, read=None, run=_docker,
                          sleep=time.sleep, wait_s=90, busy_cpu=20.0, busy_wait_s=300):
    """Restart each listed container that is running and idle (a busy one is
    waited for, then left alone), then poll meminfo once a second (up to
    wait_s) until compute_gmem stops reporting ERR. Returns compute_gmem's
    result from the last read."""
    read = read or read_meminfo
    restarted = []
    for n in names:
        if run(["docker", "inspect", "-f", "{{.State.Running}}", n]) != "true":
            continue
        if not _wait_until_idle(n, run, sleep, busy_cpu, busy_wait_s):
            sys.stderr.write(f"[reclaim] {n} stayed busy; not restarting it\n")
            continue
        sys.stderr.write(f"[reclaim] free memory short: restarting {n}\n")
        run(["docker", "restart", n])
        restarted.append(n)
    result = compute_gmem(cfg_vals, params, read())
    if not restarted:
        return result
    waited = 0
    while result[1] == "ERR" and waited < wait_s:
        sleep(1)
        waited += 1
        result = compute_gmem(cfg_vals, params, read())
    sys.stderr.write(f"[reclaim] after restarting {' '.join(restarted)}: "
                     f"free={result[2]['free']:.2f}GiB ({'ok' if result[1] != 'ERR' else 'still short'})\n")
    return result

# Must stay identical to tools/sparkyard/validate.py EXTRA_PATH_PART_RE (this
# script cannot import the package; a test compares the two).
_EXTRA_PART = re.compile(r"^[A-Za-z0-9_.+@=][A-Za-z0-9_.+@=-]*$")


def parse_extra_paths(raw):
    """EXTRA_PATHS (rendered for nodes: 2 models): comma-separated paths relative
    to LLM_ROOT_PATH that the model's flags reference besides its own weights
    (e.g. a speculative-decoding draft model). Conservative charset; no leading
    '/', no '.'/'..' or empty components."""
    out = []
    for rel in (raw or "").split(","):
        if not rel:
            continue
        parts = rel.split("/")
        if any(not _EXTRA_PART.match(x) or x in (".", "..") for x in parts):
            _fatal(f"[cluster] FATAL: EXTRA_PATHS entry {rel!r} is not a plain relative path "
                   f"under LLM_ROOT_PATH (letters, digits, '._+@=-'; no leading '/', no '..')")
        out.append(rel)
    return out


def ssh_argv(c, remote):
    """ssh to the worker: key-only, no prompts, pinned host key, no ssh config."""
    return ["ssh", "-i", f"{c.ssh_dir}/id_ed25519",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", f"UserKnownHostsFile={c.ssh_dir}/known_hosts",
            "-o", "StrictHostKeyChecking=yes",
            "-l", c.worker_user, c.worker_host, "--", shlex.join(remote)]


def nccl_env(c, node_ip):
    """`-e K=V` pairs pinning NCCL/Gloo/UCX to the CX7 interface on one node."""
    pairs = [("VLLM_HOST_IP", node_ip),
             ("NCCL_SOCKET_IFNAME", c.eth_if), ("GLOO_SOCKET_IFNAME", c.eth_if),
             ("UCX_NET_DEVICES", c.eth_if), ("TP_SOCKET_IFNAME", c.eth_if),
             ("NCCL_IB_HCA", c.ib_hca), ("NCCL_IB_DISABLE", "0"),
             ("NCCL_IGNORE_CPU_AFFINITY", "1")]
    out = []
    for k, v in pairs:
        out += ["-e", f"{k}={v}"]
    return out


def cluster_env_from_environ():
    return ClusterEnv(
        worker_host=_req("WORKER_SSH_HOST"), worker_user=_req("WORKER_SSH_USER"),
        worker_ip=_req("WORKER_CX7_IP"), head_ip=_req("HEAD_CX7_IP"),
        master_port=_req("MASTER_PORT"), eth_if=_req("ETH_IF"), ib_hca=_req("IB_HCA"),
        ssh_dir=os.environ.get("CLUSTER_SSH_DIR") or "/app/cluster")


def build_argv(gmem_str, env, passthrough, cluster=None, rank=0, api_host=None):
    llm_root = os.environ.get("LLM_ROOT_PATH")
    if not llm_root:
        _fatal("[launch] FATAL: required env LLM_ROOT_PATH is unset")
    worker = cluster is not None and rank == 1
    base = [
        "docker", "run"] + (["-d"] if worker else []) + [
        "--rm", "--name", env.container_name + ("-w" if worker else ""),
        "--runtime", "nvidia", "--gpus", "all", "--ipc=host",
    ]
    if cluster is None:
        base += ["--network", "container:llama-swap"]
    else:
        base += ["--network", "host", "--device", "/dev/infiniband",
                 "--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1"]
    base += [
        "-e", "NVIDIA_DISABLE_FORWARD_COMPATIBILITY=1",
        "-e", "VLLM_MARLIN_USE_ATOMIC_ADD=1",
    ]
    if cluster is not None:
        base += nccl_env(cluster, cluster.worker_ip if worker else cluster.head_ip)
    base += ["-v", f"{llm_root}:/models"]
    # vLLM keeps torch.compile artifacts and FlashInfer's GEMM autotune results
    # under /root/.cache/vllm. The container is --rm, so without this mount every
    # cold load redoes them (~7 min for an NVFP4 27B on the GB10). A named volume
    # (created by docker on first use) keeps root-owned files out of the host tree.
    # Entries are keyed by model/config hash, so one volume serves every model.
    # VLLM_CACHE_VOLUME overrides the name; set it empty to disable the mount.
    cache_vol = os.environ.get("VLLM_CACHE_VOLUME", "sparkyard-vllm-cache")
    if cache_vol:
        base += ["-v", f"{cache_vol}:/root/.cache/vllm"]
    extra = shlex.split(env.extra_docker_args) if env.extra_docker_args else []
    prefix = shlex.split(env.vllm_serve_prefix) if env.vllm_serve_prefix else []
    if cluster is not None and rank == 0:
        if not api_host:
            _fatal("[launch] FATAL: cluster head needs an API host (CLUSTER_API_HOST or "
                   "host.docker.internal); refusing to bind 0.0.0.0")
        host = api_host
    else:
        host = env.host
    cluster_args = []
    if cluster is not None:
        cluster_args = ["--tensor-parallel-size", "2", "--nnodes", "2",
                        "--node-rank", str(rank), "--master-addr", cluster.head_ip,
                        "--master-port", cluster.master_port]
        if worker:
            cluster_args.append("--headless")
    vllm_args = prefix + [
        env.model_path,
        "--host", host, "--port", env.port,
        "--gpu-memory-utilization", gmem_str,
        "--max-model-len", str(env.max_model_len),
        "--max-num-seqs", str(env.max_num_seqs),
    ] + cluster_args + list(passthrough)
    if env.pre_launch_cmd:
        quoted = " ".join(shlex.quote(t) for t in vllm_args)
        return base + extra + ["--entrypoint", "/bin/bash", env.image, "-c",
                               f"{env.pre_launch_cmd} && exec {quoted}"]
    return base + extra + [env.image] + vllm_args


def _run(argv):
    """Run argv, capturing (rc, stdout, stderr). The injectable runner for the
    cluster path. No subprocess timeout: ssh calls carry ConnectTimeout, and a
    worker `docker run` may legitimately pull for a long time."""
    r = subprocess.run(argv, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def _run_poll(argv, timeout=15):
    """_run with a timeout, for the worker liveness poll: a hung ssh must not
    stall the head's wait loop. A timeout reads as an ssh failure (rc 255)."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 255, "", "worker poll timed out"
    return r.returncode, r.stdout, r.stderr


def _target(c):
    return f"{c.worker_user}@{c.worker_host}"


def _why(out, err):
    text = (err or "").strip() or (out or "").strip()
    return f": {text}" if text else ""


def image_fingerprint(layers_json):
    """Short storage-backend-independent identity of an image: the ordered layer
    diff IDs are content-addressed and equal on classic overlay2 and containerd
    stores, unlike `.Id` (config digest vs manifest digest)."""
    text = (layers_json or "").strip()
    return hashlib.sha256(text.encode()).hexdigest()[:12] if text else ""


def cluster_preflight(c, env, llm_root, run=_run, model=None, remove_stale=True,
                      extra_paths=()):
    """Fail fast, naming the worker and the fix, before anything starts.
    `extra_paths` (from EXTRA_PATHS) must also exist on the worker."""
    t = _target(c)
    rc, out, err = run(ssh_argv(c, ["true"]))
    if rc != 0:
        _fatal(f"[cluster] FATAL: cannot ssh to {t} (rc={rc}){_why(out, err)} — "
               f"run sparkyard cluster check")
    inspect = ["docker", "image", "inspect", "-f", "{{json .RootFS.Layers}}", env.image]
    lrc, lout, lerr = run(inspect)
    rrc, rout, rerr = run(ssh_argv(c, inspect))
    head_id = image_fingerprint(lout) if lrc == 0 else ""
    worker_id = image_fingerprint(rout) if rrc == 0 else ""
    if rrc != 0 and "permission denied" in f"{rerr}\n{rout}".lower():
        _fatal(f"[cluster] FATAL: docker permission denied on worker {t}{_why(rout, rerr)} — "
               f"on the worker run `sudo usermod -aG docker {c.worker_user}`, then log in again")
    if not head_id or head_id != worker_id:
        detail = "" if rrc == 0 else _why(rout, rerr)
        _fatal(f"[cluster] FATAL: image {env.image} differs between head and worker {t} "
               f"(head={head_id or 'missing'} worker={worker_id or 'missing'}){detail} — "
               f"run sparkyard cluster sync")
    prefix = "/models/"
    if not env.model_path.startswith(prefix):
        _fatal(f"[cluster] FATAL: MODEL_PATH {env.model_path} is not under {prefix}")
    cfg = f"{llm_root.rstrip('/')}/{env.model_path[len(prefix):].strip('/')}/config.json"
    rc, out, err = run(ssh_argv(c, ["test", "-f", cfg]))
    if rc != 0:
        name = model or Path(env.model_path).name
        _fatal(f"[cluster] FATAL: model weights missing on worker {t}: {cfg} not found — "
               f"run sparkyard cluster sync {name}")
    for rel in extra_paths:
        p = f"{llm_root.rstrip('/')}/{rel}"
        rc, out, err = run(ssh_argv(c, ["test", "-e", p]))
        if rc != 0:
            name = model or env.container_name
            _fatal(f"[cluster] FATAL: worker {c.worker_host} is missing {p} (referenced by "
                   f"this model's flags) — run sparkyard cluster sync {name}")
    if remove_stale:
        # A launcher killed by SIGKILL (e.g. llama-swap's unloadTimeout escalation)
        # leaves the detached worker, and possibly the head container, behind.
        run(["docker", "rm", "-f", env.container_name])
        run(ssh_argv(c, ["docker", "rm", "-f", f"{env.container_name}-w"]))


def read_worker_meminfo(c, run=_run):
    rc, out, err = run(ssh_argv(c, ["cat", "/proc/meminfo"]))
    mi = parse_meminfo(out) if rc == 0 else (0, 0, 0)
    if rc != 0 or mi[0] == 0:
        _fatal(f"[cluster] FATAL: cannot read /proc/meminfo on worker {_target(c)} "
               f"(rc={rc}){_why('', err)} — run sparkyard cluster check")
    return mi


def size_cluster(cfg_vals, params, head_mi, worker_mi):
    """Size both nodes at tp=2; both get the smaller gmem. Each diag also
    carries that node's own `gmem` and `mode`."""
    p = replace(params, tp=2)
    gh, mh, dh = compute_gmem(cfg_vals, p, head_mi)
    gw, mw, dw = compute_gmem(cfg_vals, p, worker_mi)
    dh = dict(dh, gmem=gh, mode=mh)
    dw = dict(dw, gmem=gw, mode=mw)
    if "ERR" in (mh, mw):
        return None, "ERR", dh, dw
    mode = "fallback" if "fallback" in (mh, mw) else "sized"
    return min(gh, gw), mode, dh, dw


def install_abort(signal_mod=signal):
    """Install SIGTERM/SIGINT handlers that record the signal and forward it to
    the head once it runs (state["proc"]). Returns the shared state dict."""
    state = {"proc": None, "sig": None}

    def handler(sig, frame):
        state["sig"] = sig
        if state["proc"] is not None:
            state["proc"].send_signal(sig)

    signal_mod.signal(signal_mod.SIGTERM, handler)
    signal_mod.signal(signal_mod.SIGINT, handler)
    return state


_LOG_THREADS = []


def _stream_logs(argv, popen):
    """Best-effort: run argv (ssh ... docker logs -f <name>-w) and copy its output
    to stderr, each line prefixed "[worker] ". Returns the process or None."""
    try:
        proc = popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, errors="replace")
    except Exception as e:  # noqa: BLE001 — streaming must never affect the launch
        sys.stderr.write(f"[cluster] note: cannot stream worker logs: {e}\n")
        return None

    def pump():
        try:
            for line in proc.stdout:
                sys.stderr.write("[worker] " + (line if line.endswith("\n") else line + "\n"))
                sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass

    t = threading.Thread(target=pump, name="worker-logs", daemon=True)
    t.start()
    _LOG_THREADS.append(t)
    return proc


def _stop_logs(proc):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _join_log_threads(timeout=None):
    """Test helper: wait for log pump threads to drain."""
    while _LOG_THREADS:
        _LOG_THREADS.pop().join(timeout)


def _worker_status(run, poll_argv):
    """"alive", "gone" or "unknown" from `ssh ... docker container inspect -f
    '{{.State.Running}}' <name>-w` (last stdout line, so shell noise before it
    is ignored). The worker runs with --rm, so once it exits
    inspect fails with "No such object": that is the death signal. Anything
    else that fails (ssh rc 255, a runner error) is "unknown"."""
    try:
        rc, out, err = run(poll_argv)
    except OSError:
        return "unknown"
    if rc == 0:
        lines = (out or "").strip().splitlines()
        last = lines[-1].strip() if lines else ""
        return {"true": "alive", "false": "gone"}.get(last, "unknown")
    text = f"{out or ''}\n{err or ''}".lower()
    if "no such object" in text or "no such container" in text:
        return "gone"
    return "unknown"


WORKER_POLL_MISSES = 3


def _wait_head(proc, state, run, poll_argv, interval, worker_name, signal_mod):
    """Wait for the head, polling the worker every `interval` s. If the worker
    is gone (or the poll fails WORKER_POLL_MISSES times in a row), SIGTERM the
    head and wait for it; returns True in that case. Once a stop signal has
    arrived (already forwarded to the head) polling stops."""
    misses = 0
    while True:
        try:
            proc.wait(timeout=interval)
            return False
        except subprocess.TimeoutExpired:
            pass
        if state["sig"] is not None:
            proc.wait()
            return False
        status = _worker_status(run, poll_argv)
        if state["sig"] is not None:
            continue
        if status == "alive":
            misses = 0
            continue
        if status == "unknown":
            misses += 1
            if misses < WORKER_POLL_MISSES:
                continue
        sys.stderr.write(f"[cluster] worker {worker_name} exited before/while serving — "
                         f"stopping head\n")
        sys.stderr.flush()
        proc.send_signal(signal_mod.SIGTERM)
        proc.wait()
        return True


def run_cluster(head_argv, worker_argv, stop_worker_argv, run=_run,
                popen=subprocess.Popen, signal_mod=signal, state=None,
                logs_argv=None, log_popen=subprocess.Popen, poll_argv=None,
                poll_interval=5, worker_name="worker", poll_run=None):
    """Start the worker detached, then the head as a child. Forward SIGTERM/
    SIGINT to the head; whenever the head ends, stop the worker. Returns the
    head's exit code (128+N if it died of signal N). `state` comes from
    install_abort() when the caller installed the handlers earlier (so a stop
    during preflight/sizing is seen here); otherwise they are installed now.
    `logs_argv`, if given, is streamed to stderr as "[worker] ..." lines while
    the worker runs. `poll_argv`, if given, is run every `poll_interval` s
    while the head runs; when it shows the worker gone, the head is stopped
    (without it a head whose worker died waits out the torch.distributed
    timeout, ~10 min) and a non-zero code is returned."""
    if state is None:
        # Installed before the worker starts, so a stop during `docker run -d`
        # still reaches the cleanup below instead of killing the launcher.
        state = install_abort(signal_mod)
    if state["sig"] is not None:
        sys.stderr.write(f"[cluster] aborted (signal {state['sig']}) before starting the worker\n")
        return 128 + state["sig"]
    rc, out, err = run(worker_argv)
    if rc != 0:
        # ssh can fail (e.g. 255) after the remote `docker run -d` succeeded.
        try:
            run(stop_worker_argv)
        except OSError:
            pass
        _fatal(f"[cluster] FATAL: worker container failed to start (rc={rc}){_why(out, err)}")
    logs = _stream_logs(logs_argv, log_popen) if logs_argv else None
    try:
        if state["sig"] is not None:
            sys.stderr.write(f"[cluster] aborted (signal {state['sig']}) before starting the head\n")
            return 128 + state["sig"]
        proc = popen(head_argv)
        state["proc"] = proc
        if state["sig"] is not None:
            proc.send_signal(state["sig"])
        died = False
        if poll_argv is None:
            proc.wait()
        else:
            died = _wait_head(proc, state, poll_run or run, poll_argv, poll_interval,
                              worker_name, signal_mod)
        code = proc.returncode
        code = 128 - code if code < 0 else code
        return (code or 1) if died else code
    finally:
        run(stop_worker_argv)
        _stop_logs(logs)


def stop_cluster(c, container_name, run=_run, pid=None, kill=os.kill):
    """cmdStop for a two-node model. llama-swap runs cmdStop *instead of*
    SIGTERM, so first SIGTERM the launcher (pid = llama-swap's ${PID}): a launch
    still in preflight/sizing then aborts before starting anything, and a
    running one forwards it to the head. Then stop worker, then head; all may
    be gone."""
    if pid and pid.isdigit() and int(pid) > 1 and int(pid) != os.getpid():
        try:
            kill(int(pid), signal.SIGTERM)
        except OSError:
            pass
    for argv in (ssh_argv(c, ["docker", "stop", f"{container_name}-w"]),
                 ["docker", "stop", container_name]):
        try:
            run(argv)
        except OSError:
            pass
    return 0


def _served_name(passthrough):
    if "--served-model-name" in passthrough:
        i = passthrough.index("--served-model-name")
        if i + 1 < len(passthrough):
            return passthrough[i + 1]
    return None


def _log_plan(prefix, gmem, mode, d, cfg=True):
    """The sizing lines; prefix is "[auto-gmem] " (single-node, unchanged) or
    "[auto-gmem] head: " / "[auto-gmem] worker: " in cluster mode."""
    if cfg:
        sys.stderr.write(
            f"{prefix}cfg: layers={d['layers']} attn_layers={d['attn_layers']} "
            f"kv_heads={d['kv_heads']} "
            f"head_dim={d['head_dim']} ctx={d['ctx']} batch={d['batch']} "
            f"kvb={d['kvb']} kv_batch_used={d['kv_batch']}\n")
        sys.stderr.write(
            f"{prefix}weights={d['weights']:.2f}GiB kv={d['kv']:.2f}GiB "
            f"safety={d['safety']:.0f}GiB → need={d['need']:.2f}GiB\n")
    sys.stderr.write(
        f"{prefix}system: MemTotal={d['mt']:.2f}GiB MemFree={d['mf']:.2f}GiB "
        f"MemAvail={d['ma']:.2f}GiB ceiling={d['ceiling']:.2f}GiB "
        f"headroom_reserved={d['headroom']:.2f}GiB\n")
    sys.stderr.write(
        f"{prefix}free={d['free']:.2f}GiB total={d['total']:.2f}GiB "
        f"(CUDA view, capped) → gpu_memory_utilization={gmem:.2f} [{mode}]\n")


def _req(name):
    v = os.environ.get(name)
    if not v:
        _fatal(f"[launch] FATAL: required env {name} is unset")
    return v


def _params(env):
    return Params(
        max_model_len=env.max_model_len,
        max_num_seqs=env.max_num_seqs,
        kv_dtype_bytes=_env_num("KV_DTYPE_BYTES", 1, int),
        gmin=_env_num("GMEM_MIN", 0.55, float),
        gmax=_env_num("GMEM_MAX", 0.92, float),
        safety=_env_num("SAFETY_GIB", 4.0, float),
        cuda_overhead=_env_num("CUDA_OVERHEAD_GIB", 6.3, float),
        ceiling=_env_num("SYSTEM_RAM_CEILING_GIB", 117.81, float),
        kv_batch_realistic=_env_num("KV_BATCH_REALISTIC", 4, int),
        free_buffer=_env_num("GMEM_FREE_BUFFER_GIB", 5.0, float),
        weights=weights_gib(env.model_host_path),
    )


def _size_local(cfg_vals, params, head):
    """Today's sizing + reclaim on this node. `head` prefixes the FATAL in
    cluster mode; single-node text is unchanged."""
    gmem, mode, d = compute_gmem(cfg_vals, params, read_meminfo())
    reclaim = parse_reclaim(os.environ.get("RECLAIM_CONTAINERS"))
    margin = _env_num("RECLAIM_MARGIN_GIB", 10.0, float)
    if reclaim and short_of_memory(gmem, mode, d, margin):
        gmem, mode, d = reclaim_and_recompute(
            reclaim, cfg_vals, params, wait_s=_env_num("RECLAIM_WAIT_S", 90, int),
            busy_cpu=_env_num("RECLAIM_BUSY_CPU", 20.0, float),
            busy_wait_s=_env_num("RECLAIM_BUSY_WAIT_S", 300, int))
        if mode != "ERR" and short_of_memory(gmem, mode, d, margin):
            sys.stderr.write(f"[reclaim] WARNING: still under the {margin:.0f} GiB margin "
                             f"(free={d['free']:.2f}GiB); launching anyway\n")
    if mode == "ERR":
        _fatal(f"[auto-gmem] FATAL: {'head: ' if head else ''}free VRAM below GMEM_MIN floor — "
               f"cap={d['u_cap']:.2f} gmin={params.gmin:.2f} free={d['free']:.2f}")
    return gmem, mode, d


def _gmem_override():
    override = os.environ.get("GMEM_OVERRIDE", "")
    if override and override != "adaptive":
        try:
            v = float(override)
        except ValueError:
            v = 0.0
        if not (0 < v < 1):
            _fatal(f"[auto-gmem] FATAL: GMEM_OVERRIDE='{override}' is not a number in (0,1) and not 'adaptive'")
        sys.stderr.write(f"[auto-gmem] GMEM_OVERRIDE={override} — bypassing adaptive calculation\n")
        return override
    return None


def _launch_env():
    return LaunchEnv(
        model_path=_req("MODEL_PATH"),
        model_host_path=_req("MODEL_HOST_PATH"),
        container_name=_req("CONTAINER_NAME"),
        image=_req("IMAGE"),
        port=_req("PORT"),
        host=_req("HOST"),
        max_model_len=_env_num("MAX_MODEL_LEN", 131072, int),
        max_num_seqs=_env_num("MAX_NUM_SEQS", 10, int),
        extra_docker_args=os.environ.get("EXTRA_DOCKER_ARGS", ""),
        pre_launch_cmd=os.environ.get("PRE_LAUNCH_CMD", ""),
        # `-` semantics: unset -> "vllm serve"; set-but-empty -> "" (no prefix)
        vllm_serve_prefix=os.environ.get("VLLM_SERVE_PREFIX", "vllm serve"),
    )


def main_cluster(env, passthrough, print_mode):
    if print_mode:
        return _main_cluster(env, passthrough, True, None)
    # Before preflight/sizing/reclaim: a stop arriving now must abort the launch
    # before any container starts (checked in run_cluster), not be lost.
    prev = [(s, signal.getsignal(s)) for s in (signal.SIGTERM, signal.SIGINT)]
    state = install_abort(signal)
    try:
        return _main_cluster(env, passthrough, False, state)
    finally:
        for s, h in prev:
            if h is not None:
                signal.signal(s, h)


def _main_cluster(env, passthrough, print_mode, state):
    c = cluster_env_from_environ()
    llm_root = _req("LLM_ROOT_PATH")
    extra_paths = parse_extra_paths(os.environ.get("EXTRA_PATHS"))
    # `_run` is looked up at call time so tests can monkeypatch launch._run.
    # --print runs only the read-only preflight calls (ssh true, image layers,
    # weights test): no stale-container `docker rm -f` on the worker.
    cluster_preflight(c, env, llm_root, run=_run, model=_served_name(passthrough),
                      remove_stale=not print_mode, extra_paths=extra_paths)
    gmem_str = _gmem_override()
    if gmem_str is None:
        params = _params(env)
        cfg_vals = parse_config(env.model_host_path)
        worker_mi = read_worker_meminfo(c, run=_run)
        p2 = replace(params, tp=2)
        # Worker first: a short worker must not cost the head's co-tenants a restart.
        _, mw, dw = compute_gmem(cfg_vals, p2, worker_mi)
        if mw == "ERR":
            _fatal(f"[auto-gmem] FATAL: worker {c.worker_host} short of memory — "
                   f"free={dw['free']:.2f}GiB need={dw['need']:.2f}GiB "
                   f"(cap={dw['u_cap']:.2f} gmin={params.gmin:.2f})")
        # Reclaim runs on the head only, against the head's own tp=2 plan; the
        # worker is sized once from its meminfo (no remote reclaim).
        _size_local(cfg_vals, p2, head=True)
        gmem, mode, dh, dw = size_cluster(cfg_vals, params, read_meminfo(), worker_mi)
        if mode == "ERR":  # head memory dropped between the two reads
            _fatal(f"[auto-gmem] FATAL: head: free VRAM below GMEM_MIN floor — "
                   f"cap={dh['u_cap']:.2f} gmin={params.gmin:.2f} free={dh['free']:.2f}")
        _log_plan("[auto-gmem] head: ", dh["gmem"], dh["mode"], dh)
        _log_plan("[auto-gmem] worker: ", dw["gmem"], dw["mode"], dw, cfg=False)
        sys.stderr.write(f"[auto-gmem] cluster: gpu_memory_utilization={gmem:.2f} "
                         f"(min of head {dh['gmem']:.2f}, worker {dw['gmem']:.2f}) [{mode}]\n")
        if mode == "fallback":
            sys.stderr.write("[auto-gmem] NOTE: estimate exceeded free VRAM; using "
                             "clamped gmax — vLLM will trim KV at startup if needed\n")
        gmem_str = f"{gmem:.2f}"

    api_host = os.environ.get("CLUSTER_API_HOST")
    if not api_host:
        try:
            api_host = socket.gethostbyname("host.docker.internal")
        except OSError:
            _fatal("[cluster] FATAL: cannot resolve host.docker.internal inside llama-swap — "
                   "add extra_hosts host-gateway (see docker-compose.yml)")
    head = build_argv(gmem_str, env, passthrough, cluster=c, rank=0, api_host=api_host)
    worker = build_argv(gmem_str, env, passthrough, cluster=c, rank=1)
    if print_mode:
        sys.stdout.write(f"gpu_memory_utilization={gmem_str}\n")
        sys.stdout.write("\n".join(head) + "\n")
        sys.stdout.write("--- worker ---\n")
        sys.stdout.write("\n".join(worker) + "\n")
        return 0
    stop = ssh_argv(c, ["docker", "stop", f"{env.container_name}-w"])
    logs = ssh_argv(c, ["docker", "logs", "-f", f"{env.container_name}-w"])
    poll = ssh_argv(c, ["docker", "container", "inspect", "-f", "{{.State.Running}}",
                        f"{env.container_name}-w"])
    sys.exit(run_cluster(head, ssh_argv(c, worker), stop, run=_run, state=state,
                         logs_argv=logs, poll_argv=poll, poll_run=_run_poll,
                         worker_name=f"{env.container_name}-w"))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--stop" in argv:
        # cmdStop for two-node models: only CONTAINER_NAME + the cluster env.
        i = argv.index("--stop")
        pid = argv[i + 1] if i + 1 < len(argv) else None
        return stop_cluster(cluster_env_from_environ(), _req("CONTAINER_NAME"), run=_run,
                            pid=pid, kill=os.kill)
    print_mode = os.environ.get("SPARKYARD_LAUNCH_PRINT") == "1" or "--print" in argv
    passthrough = [a for a in argv if a != "--print"]

    env = _launch_env()
    nodes = os.environ.get("NODES") or "1"
    if nodes not in ("1", "2"):
        _fatal(f"[launch] FATAL: NODES={nodes!r} must be 1 or 2")
    if nodes == "2":
        return main_cluster(env, passthrough, print_mode)

    gmem_str = _gmem_override()
    if gmem_str is None:
        params = _params(env)
        cfg_vals = parse_config(env.model_host_path)
        gmem, mode, d = _size_local(cfg_vals, params, head=False)
        _log_plan("[auto-gmem] ", gmem, mode, d)
        if mode == "fallback":
            sys.stderr.write("[auto-gmem] NOTE: estimate exceeded free VRAM; using "
                             "clamped gmax — vLLM will trim KV at startup if needed\n")
        gmem_str = f"{gmem:.2f}"

    out = build_argv(gmem_str, env, passthrough)
    if print_mode:
        sys.stdout.write(f"gpu_memory_utilization={gmem_str}\n")
        sys.stdout.write("\n".join(out) + "\n")
        return
    os.execvp(out[0], out)


if __name__ == "__main__":
    sys.exit(main())
