"""Operator commands behind the `sparkyard` CLI: init / secrets / build / bench /
start / stop. Thin shell-outs to the existing scripts + docker compose, with an
injected `run` for testability (mirrors update.py); init's config seeding is pure."""
import os
import shutil
import subprocess


def secrets(root, *, run=subprocess.run):
    return run(["bash", "scripts/gen-secrets.sh"], cwd=root).returncode


def build(root, *, run=subprocess.run):
    return run(["docker", "compose", "build"], cwd=root).returncode


def start(root, *, run=subprocess.run):
    return run(["docker", "compose", "up", "-d"], cwd=root).returncode


def stop(root, *, run=subprocess.run):
    # `down` (not `stop`): the canonical inverse of `up -d`. Named volumes
    # (litellm-db, open-webui) persist — no `-v`.
    return run(["docker", "compose", "down"], cwd=root).returncode


def bench(root, mode=None, base_url=None, *, run=subprocess.run):
    env = dict(os.environ, MODE=mode or "quality")
    if base_url:
        env["BASE_URL"] = base_url
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
