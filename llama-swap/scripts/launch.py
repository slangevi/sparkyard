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
  test-only: SPARKYARD_MEMINFO_PATH (override /proc/meminfo), SPARKYARD_LAUNCH_PRINT=1
Pass `--print` (or SPARKYARD_LAUNCH_PRINT=1) to print gmem + docker argv and exit
without launching. Remaining args are forwarded verbatim after the vllm flags.
"""
import json
import os
import shlex
import sys
from dataclasses import dataclass
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


def parse_config(model_host_path):
    cfgp = Path(model_host_path) / "config.json"
    if not cfgp.is_file():
        _fatal(f"[auto-gmem] FATAL: config.json not found at {cfgp}")
    try:
        cfg = json.loads(cfgp.read_text())
    except (ValueError, OSError) as e:
        _fatal(f"[auto-gmem] FATAL: could not read {cfgp}: {e}")
    n_layers = _extract(cfg, "num_hidden_layers")
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
    return n_layers, n_heads, n_kv, head_dim


def read_meminfo(path=None):
    path = path or os.environ.get("SPARKYARD_MEMINFO_PATH", "/proc/meminfo")
    try:
        text = Path(path).read_text()
    except OSError as e:
        _fatal(f"[auto-gmem] FATAL: could not read meminfo at {path}: {e}")
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
    n_layers, n_heads, n_kv, head_dim = cfg_vals
    p = params
    mem_total_kb, mem_avail_kb, mem_free_kb = meminfo

    kv_batch = min(p.max_num_seqs, p.kv_batch_realistic)
    kv = g2((2 * n_layers * n_kv * head_dim * p.max_model_len * kv_batch * p.kv_dtype_bytes) / GIB)
    need = g2(p.weights + kv + p.safety)

    mt_raw = g2(mem_total_kb / 1048576)
    ma_raw = g2(mem_avail_kb / 1048576)
    mf_raw = g2(mem_free_kb / 1048576)
    headroom = g2(max(0.0, mt_raw - p.ceiling))
    mt_capped = g2(min(mt_raw, p.ceiling))
    total = g2(mt_capped - p.cuda_overhead)
    free = g2(max(0.0, ma_raw - headroom - p.cuda_overhead))

    diag = {"layers": n_layers, "kv_heads": n_kv, "head_dim": head_dim,
            "ctx": p.max_model_len, "batch": p.max_num_seqs, "kvb": p.kv_dtype_bytes,
            "kv_batch": kv_batch, "weights": p.weights, "kv": kv, "safety": p.safety,
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


def build_argv(gmem_str, env, passthrough):
    llm_root = os.environ.get("LLM_ROOT_PATH")
    if not llm_root:
        _fatal("[launch] FATAL: required env LLM_ROOT_PATH is unset")
    base = [
        "docker", "run", "--rm", "--name", env.container_name,
        "--runtime", "nvidia", "--gpus", "all", "--ipc=host",
        "--network", "container:llama-swap",
        "-e", "NVIDIA_DISABLE_FORWARD_COMPATIBILITY=1",
        "-e", "VLLM_MARLIN_USE_ATOMIC_ADD=1",
        "-v", f"{llm_root}:/models",
    ]
    extra = shlex.split(env.extra_docker_args) if env.extra_docker_args else []
    prefix = shlex.split(env.vllm_serve_prefix) if env.vllm_serve_prefix else []
    vllm_args = prefix + [
        env.model_path,
        "--host", env.host, "--port", env.port,
        "--gpu-memory-utilization", gmem_str,
        "--max-model-len", str(env.max_model_len),
        "--max-num-seqs", str(env.max_num_seqs),
    ] + list(passthrough)
    if env.pre_launch_cmd:
        quoted = " ".join(shlex.quote(t) for t in vllm_args)
        return base + extra + ["--entrypoint", "/bin/bash", env.image, "-c",
                               f"{env.pre_launch_cmd} && exec {quoted}"]
    return base + extra + [env.image] + vllm_args


def _req(name):
    v = os.environ.get(name)
    if not v:
        _fatal(f"[launch] FATAL: required env {name} is unset")
    return v


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    print_mode = os.environ.get("SPARKYARD_LAUNCH_PRINT") == "1" or "--print" in argv
    passthrough = [a for a in argv if a != "--print"]

    env = LaunchEnv(
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

    override = os.environ.get("GMEM_OVERRIDE", "")
    if override and override != "adaptive":
        try:
            v = float(override)
        except ValueError:
            v = 0.0
        if not (0 < v < 1):
            _fatal(f"[auto-gmem] FATAL: GMEM_OVERRIDE='{override}' is not a number in (0,1) and not 'adaptive'")
        sys.stderr.write(f"[auto-gmem] GMEM_OVERRIDE={override} — bypassing adaptive calculation\n")
        gmem_str = override
    else:
        params = Params(
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
        cfg_vals = parse_config(env.model_host_path)
        gmem, mode, d = compute_gmem(cfg_vals, params, read_meminfo())
        if mode == "ERR":
            _fatal(f"[auto-gmem] FATAL: free VRAM below GMEM_MIN floor — "
                   f"cap={d['u_cap']:.2f} gmin={params.gmin:.2f} free={d['free']:.2f}")
        sys.stderr.write(
            f"[auto-gmem] cfg: layers={d['layers']} kv_heads={d['kv_heads']} "
            f"head_dim={d['head_dim']} ctx={d['ctx']} batch={d['batch']} "
            f"kvb={d['kvb']} kv_batch_used={d['kv_batch']}\n")
        sys.stderr.write(
            f"[auto-gmem] weights={d['weights']:.2f}GiB kv={d['kv']:.2f}GiB "
            f"safety={d['safety']:.0f}GiB → need={d['need']:.2f}GiB\n")
        sys.stderr.write(
            f"[auto-gmem] system: MemTotal={d['mt']:.2f}GiB MemFree={d['mf']:.2f}GiB "
            f"MemAvail={d['ma']:.2f}GiB ceiling={d['ceiling']:.2f}GiB "
            f"headroom_reserved={d['headroom']:.2f}GiB\n")
        sys.stderr.write(
            f"[auto-gmem] free={d['free']:.2f}GiB total={d['total']:.2f}GiB "
            f"(CUDA view, capped) → gpu_memory_utilization={gmem:.2f} [{mode}]\n")
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
    main()
