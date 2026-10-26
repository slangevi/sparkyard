"""Two-node cluster operator commands: setup-key / check / sync / suggest.

Host-side preparation and verification for what llama-swap's launch.py needs on
the worker (dedicated ssh key, same image layers, same weights path). `run`/`popen`
are injected so tests never ssh, rsync or touch docker. sparkyard never edits
the worker's authorized_keys: setup-key only prints the ssh-copy-id command."""
import hashlib
import os
import re
import shlex
import subprocess

KEY_NAME = "id_ed25519"
NVIDIA_HINT = ("sudo nvidia-ctk runtime configure --runtime=docker --set-as-default "
               "&& sudo systemctl restart docker")


def ssh_base(settings, host):
    """ssh argv up to and including the target host. Same option set as
    llama-swap/scripts/launch.py ssh_argv, key/known_hosts under cluster_dir."""
    d = settings.cluster_dir
    return ["ssh", "-i", f"{d}/{KEY_NAME}",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", f"UserKnownHostsFile={d}/known_hosts",
            "-o", "StrictHostKeyChecking=yes",
            "-l", settings.cluster.worker.ssh_user, host]


def _remote(settings, remote, run, host=None):
    w = settings.cluster.worker
    argv = ssh_base(settings, host or w.ssh_host) + ["--", shlex.join(remote)]
    return run(argv, capture_output=True, text=True)


def _local(argv, run):
    return run(argv, capture_output=True, text=True)


def _out(p):
    return (getattr(p, "stdout", "") or "").strip()


def _scanned_hosts(text):
    """Host names/IPs that an ssh-keyscan output carries keys for."""
    hosts = set()
    for ln in (text or "").splitlines():
        parts = ln.split()
        if len(parts) < 3 or parts[0].startswith("#"):
            continue
        for h in parts[0].split(","):
            if h.startswith("[") and "]" in h:
                h = h[1:h.index("]")]
            hosts.add(h)
    return hosts


def setup_key(settings, run=subprocess.run):
    c = settings.cluster
    d = settings.cluster_dir
    w = c.worker
    os.makedirs(d, mode=0o700, exist_ok=True)
    os.chmod(d, 0o700)
    key = os.path.join(d, KEY_NAME)
    if not os.path.exists(key):
        p = run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "sparkyard-llama-swap",
                 "-f", key])
        if p.returncode != 0:
            print("✗ ssh-keygen failed")
            return 1
    scan = run(["ssh-keyscan", "-t", "ed25519", w.ssh_host, w.cx7_ip],
               capture_output=True, text=True)
    if scan.returncode != 0 or not _out(scan):
        print(f"✗ ssh-keyscan returned no host keys for {w.ssh_host} / {w.cx7_ip}")
        return 1
    # launch.py ssh's to ssh_host and sync/rsync use cx7_ip, both with
    # StrictHostKeyChecking=yes: a key missing for either fails later, obscurely.
    missing = [h for h in (w.ssh_host, w.cx7_ip) if h not in _scanned_hosts(scan.stdout)]
    if missing:
        print(f"✗ ssh-keyscan returned no ed25519 host key for {', '.join(missing)} — "
              f"check that sshd on the worker is reachable at both {w.ssh_host} and "
              f"{w.cx7_ip}, then re-run")
        return 1
    with open(os.path.join(d, "known_hosts"), "w") as f:
        f.write(scan.stdout if scan.stdout.endswith("\n") else scan.stdout + "\n")
    fp = run(["ssh-keygen", "-lf", key + ".pub"], capture_output=True, text=True)
    print(f"Key fingerprint: {_out(fp)}")
    print(f"Run once:  ssh-copy-id -i {key}.pub {w.ssh_user}@{w.ssh_host}")
    print("Optional hardening: on the worker, prefix that key's line in "
          f"~{w.ssh_user}/.ssh/authorized_keys with")
    print(f'  from="<head LAN IP>,{c.head_cx7_ip}" ssh-ed25519 AAAA... sparkyard-llama-swap')
    print("  (the head's address as seen from the worker on the LAN, plus its CX7 IP "
          "for sync) — the key reaches a docker-group (root-equivalent) user.")
    return 0


def _local_weights(settings, path):
    d = os.path.join(settings.llm_root, path)
    total = 0
    try:
        for n in os.listdir(d):
            fp = os.path.join(d, n)
            if n.endswith(".safetensors") and os.path.isfile(fp):
                total += os.stat(fp).st_size
    except OSError:
        return None
    return total


def _remote_weights(settings, path, run):
    d = f"{settings.llm_root}/{path}"
    p = _remote(settings, ["find", d, "-maxdepth", "1", "-name", "*.safetensors",
                           "-printf", "%s\\n"], run)
    if p.returncode != 0:
        return None
    return sum(int(x) for x in _out(p).split() if x.isdigit())


def _tp_models(models):
    return [m for m in models if m.nodes == 2]


def _images(models):
    seen = []
    for m in models:
        if m.image not in seen:
            seen.append(m.image)
    return seen


def _image_id(argv_prefix_remote, settings, image, run):
    """Short fingerprint of the image's layer diff IDs. `.Id` differs between
    the containerd and overlay2 image stores for identical images; layers don't."""
    cmd = ["docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", image]
    p = _remote(settings, cmd, run) if argv_prefix_remote else _local(cmd, run)
    if p.returncode != 0 or not _out(p):
        return None
    return hashlib.sha256(_out(p).encode()).hexdigest()[:12]


def check(settings, models, run=subprocess.run):
    c = settings.cluster
    w = c.worker
    ok = True

    def report(passed, label, detail=""):
        nonlocal ok
        ok = ok and passed
        print(f"{'PASS' if passed else 'FAIL'}  {label}  {detail}".rstrip())

    p = _remote(settings, ["true"], run)
    if p.returncode != 0:
        report(False, "ssh", f"{w.ssh_user}@{w.ssh_host}: {_out(p) or (getattr(p, 'stderr', '') or '').strip()}")
        return 1
    report(True, "ssh", f"{w.ssh_user}@{w.ssh_host}")

    p = _remote(settings, ["true"], run, host=w.cx7_ip)
    report(p.returncode == 0, "ssh cx7",
           f"{w.ssh_user}@{w.cx7_ip}" + ("" if p.returncode == 0 else
                                         f": {(getattr(p, 'stderr', '') or '').strip() or _out(p)}"))

    p = _remote(settings, ["command", "-v", "rsync"], run)
    report(p.returncode == 0, "worker rsync", _out(p) or "rsync not found (sudo apt install rsync)")

    p = _remote(settings, ["docker", "info", "--format", "{{.ServerVersion}}"], run)
    report(p.returncode == 0, "worker docker", _out(p))

    p = _remote(settings, ["docker", "info", "--format", "{{json .Runtimes}}"], run)
    has_rt = p.returncode == 0 and "nvidia" in _out(p)
    report(has_rt, "worker nvidia runtime", "" if has_rt else NVIDIA_HINT)

    tp = _tp_models(models)
    for img in _images(tp):
        h = _image_id(False, settings, img, run)
        r = _image_id(True, settings, img, run)
        report(bool(h) and h == r, f"image {img}",
               f"head={h or 'missing'} worker={r or 'missing'}")

    for node, remote in (("head", False), ("worker", True)):
        cmd = ["cat", f"/sys/class/net/{c.eth_if}/operstate"]
        p = _remote(settings, cmd, run) if remote else _local(cmd, run)
        report(p.returncode == 0 and _out(p) == "up", f"iface {c.eth_if} {node}", _out(p))
    for hca in [h for h in c.ib_hca.split(",") if h]:
        for node, remote in (("head", False), ("worker", True)):
            cmd = ["cat", f"/sys/class/infiniband/{hca}/ports/1/state"]
            p = _remote(settings, cmd, run) if remote else _local(cmd, run)
            report(p.returncode == 0 and "ACTIVE" in _out(p), f"hca {hca} {node}", _out(p))

    p = _local(["ping", "-c1", "-W2", "-I", c.eth_if, w.cx7_ip], run)
    report(p.returncode == 0, "cx7 ping", w.cx7_ip)

    for m in tp:
        path = m.raw["path"]
        hs = _local_weights(settings, path)
        rs = _remote_weights(settings, path, run)
        report(hs is not None and hs == rs, f"weights {m.name}",
               f"head={hs if hs is not None else 'missing'} worker={rs if rs is not None else 'missing'}")
    return 0 if ok else 1


_IB_LINE = re.compile(r"^(\S+)\s+port\s+\d+\s+==>\s+(\S+)\s+\((\w+)\)")


def suggest(run=subprocess.run):
    try:
        p = run(["ibdev2netdev"], capture_output=True, text=True)
    except (FileNotFoundError, OSError):
        p = None
    if p is None or p.returncode != 0:
        return ("# ibdev2netdev not found; see the cluster: example in "
                "docs/superpowers/specs/2026-10-25-sparkyard-two-node-vllm-design.md")
    up = [m.groups() for m in (_IB_LINE.match(ln.strip()) for ln in (p.stdout or "").splitlines())
          if m and m.group(3) == "Up"]
    eth = next((net for _, net, _ in up if "P" not in net), "")
    hcas = ",".join(h for h, _, _ in up)
    return ("# add to settings.local.yaml:\n"
            "# cluster:\n"
            "#   head_cx7_ip: <this node's CX7 IP>\n"
            f"#   eth_if: {eth}\n"
            f"#   ib_hca: {hcas}\n"
            "#   workers:\n"
            "#     - {name: spark-2, ssh_host: <ip>, ssh_user: <user>, cx7_ip: <CX7 IP>}")


def sync(settings, models, names, run=subprocess.run, popen=subprocess.Popen):
    c = settings.cluster
    w = c.worker
    tp = _tp_models(models)
    if names:
        by = {m.name: m for m in models}
        for n in names:
            if n not in by:
                print(f"✗ unknown model: {n}")
                return 1
            if by[n].nodes != 2:
                print(f"✗ {n} is not a nodes: 2 model")
                return 1
        targets = [by[n] for n in names]
    else:
        targets = tp

    for img in _images(targets):
        h = _image_id(False, settings, img, run)
        r = _image_id(True, settings, img, run)
        if h and h == r:
            print(f"✓ image {img} already on worker")
            continue
        if not h:
            print(f"✗ image {img} not found on head")
            return 1
        print(f"→ streaming image {img} to {w.name}")
        save = popen(["docker", "save", img], stdout=subprocess.PIPE)
        load = popen(ssh_base(settings, w.cx7_ip) + ["--", "docker load"],
                     stdin=save.stdout)
        save.stdout.close()
        rcs = (load.wait(), save.wait())
        if any(rcs):
            print(f"✗ image transfer failed (docker save rc={rcs[1]}, docker load rc={rcs[0]})")
            return 1

    e = shlex.join(ssh_base(settings, w.cx7_ip)[:-1])
    for m in targets:
        path = m.raw["path"]
        src = f"{settings.llm_root}/{path}/"
        parent = os.path.dirname(f"{settings.llm_root}/{path}")
        p = _remote(settings, ["mkdir", "-p", parent], run)
        if p.returncode != 0:
            print(f"✗ mkdir {parent} on worker failed")
            return 1
        print(f"→ rsync {m.name}")
        r = run(["rsync", "-a", "--info=progress2", "-e", e, src,
                 f"{w.ssh_user}@{w.cx7_ip}:{src}"])
        if r.returncode != 0:
            print(f"✗ rsync {m.name} failed")
            return 1
    print("✓ sync complete")
    return 0
