"""Operator commands behind the `sparkyard` CLI: init / secrets / build / bench /
start / stop. Thin shell-outs to the existing scripts + docker compose, with an
injected `run` for testability (mirrors update.py); init's config seeding is pure."""
import os
import shutil
import subprocess
import time


def secrets(root, *, run=subprocess.run):
    return run(["bash", "scripts/gen-secrets.sh"], cwd=root).returncode


def build(root, *, run=subprocess.run):
    return run(["docker", "compose", "build"], cwd=root).returncode


def start(root, *, run=subprocess.run):
    return run(["docker", "compose", "up", "-d"], cwd=root).returncode


# Services that consume generated config from a bind mount. `docker compose up -d`
# cannot see those files change, so it leaves the old config running — a render is
# not live until these restart.
RELOAD_SERVICES = ("llama-swap", "litellm")


def reload(root, *, run=subprocess.run, sleep=time.sleep, attempts=60):
    """Restart the services that read generated config, then wait for health.

    `render` writes files that are bind-mounted into running containers, so the
    stack keeps serving the previous config until this runs. Waits on EVERY
    restarted service: waiting on llama-swap alone races LiteLLM, which answers
    "connection reset by peer" while it is still coming up."""
    rc = run(["docker", "restart", *RELOAD_SERVICES], cwd=root).returncode
    if rc != 0:
        return rc
    for svc in RELOAD_SERVICES:
        for _ in range(attempts):
            p = run(["docker", "inspect", "-f", "{{.State.Health.Status}}", svc],
                    cwd=root, capture_output=True, text=True)
            out = (getattr(p, "stdout", "") or "").strip()
            # A service without a healthcheck reports nothing; treat as ready.
            if out in ("healthy", "", "<no value>"):
                break
            sleep(1)
        else:
            print(f"✗ {svc} did not become healthy")
            return 1
    return 0


def stop(root, *, run=subprocess.run):
    # `down` (not `stop`): the canonical inverse of `up -d`. Named volumes
    # (litellm-db, open-webui) persist — no `-v`.
    return run(["docker", "compose", "down"], cwd=root).returncode


def bench(root, mode=None, base_url=None, model=None, *, run=subprocess.run):
    """Run the benchmark sweep. `model` scopes it; unscoped, bench.sh loads every
    discovered model in turn, which on unified memory is a memory event rather
    than merely a slow one."""
    env = dict(os.environ, MODE=mode or "quality")
    if base_url:
        env["BASE_URL"] = base_url
    if model:
        env["MODELS"] = " ".join(model)
    return run(["bash", "scripts/bench.sh"], cwd=root, env=env).returncode


_SEEDS = [("settings.example.yaml", "settings.local.yaml"),
          ("models.example.yaml", "models.yaml")]


def init(root, *, run=subprocess.run, copy=shutil.copy, exists=os.path.exists):
    """Seed the gitignored working files from the committed examples (idempotent),
    then scaffold secrets. Does NOT build a venv or offer a global install — you
    already have `sparkyard` (that's how you ran this)."""
    for example, target in _SEEDS:
        tp = os.path.join(root, target)
        if exists(tp):
            print(f"• {target} exists — leaving it")
        else:
            copy(os.path.join(root, example), tp)
            print(f"→ created {target} (edit it)")
    rc = secrets(root, run=run)
    print("\nNext steps:")
    print("  1. edit settings.local.yaml + models.yaml + secrets.env (HF_TOKEN)")
    print("  2. sparkyard render")
    print("  3. sparkyard build")
    print("  4. sparkyard start        # docker compose up -d")
    return rc
