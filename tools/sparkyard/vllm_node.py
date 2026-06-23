"""`vllm-node` subcommand: clone eugr/spark-vllm-docker, check out the pinned
vLLM ref, and build the vllm-node serving image(s). Backs `make vllm-node`.

Planning (build_plan) is pure and unit-tested; side effects (git/docker) live in
run(), which is dependency-injected for testability."""
import os
import shutil
import subprocess
import sys
from collections import namedtuple

# Default image set when no --variant is given. base + tf5 share the pinned ref.
DEFAULT_VARIANTS = ["base", "tf5"]

# A unit of work: a human label, a working dir (None = run from CWD), and argv.
Step = namedtuple("Step", "description cwd argv")

# Per-variant build-and-copy.sh argv (ref appended for the ref-pinned variants).
_REF_VARIANTS = {
    "base": [],
    "tf5": ["--tf5"],
}


def build_plan(cfg, variants, ref, clone_exists):
    """Return the ordered Steps to build `variants`. Pure: no git/docker/fs calls.

    cfg: VllmBuild (upstream, clone_path, vllm_ref). clone_exists: bool."""
    steps = []
    if clone_exists:
        steps.append(Step("fetch upstream", cfg.clone_path, ["git", "fetch"]))
    else:
        steps.append(Step("clone upstream", None,
                          ["git", "clone", cfg.upstream, cfg.clone_path]))

    # `ref` is a vLLM commit, NOT a spark-vllm-docker commit: build-and-copy.sh
    # checks vLLM out itself via --vllm-ref (below). The tooling clone stays at its
    # cloned HEAD — never `git checkout <ref>` here (that ref isn't in this repo and
    # the checkout would abort the build on a fresh clone). mxfp4 tracks its own ref.
    for v in variants:
        if v == "mxfp4":
            argv = ["./build-and-copy.sh", "--exp-mxfp4"]
        else:
            argv = ["./build-and-copy.sh", *_REF_VARIANTS[v], "--vllm-ref", ref]
        steps.append(Step(f"build {v}", cfg.clone_path, argv))
    return steps


def _default_exec_step(step):
    """Exec a Step, streaming child output (no capture). Returns the exit code."""
    print(f"→ {step.description}: {' '.join(step.argv)}"
          + (f"  (in {step.cwd})" if step.cwd else ""))
    return subprocess.run(step.argv, cwd=step.cwd).returncode


def run(args, settings, exists=os.path.exists, which=shutil.which, exec_step=None):
    """Build the vllm-node image(s). Returns 0 on success, 1 on failure.

    Side-effecting deps are injected (exists/which/exec_step) for testability."""
    exec_step = exec_step or _default_exec_step
    cfg = settings.vllm
    variants = [args.variant] if args.variant else list(DEFAULT_VARIANTS)
    ref = args.vllm_ref or cfg.vllm_ref
    clone_exists = exists(os.path.join(cfg.clone_path, ".git"))
    plan = build_plan(cfg, variants, ref, clone_exists)

    if args.dry_run:
        print(f"# plan: variants={variants} ref={ref} clone={cfg.clone_path}")
        for step in plan:
            loc = f" (in {step.cwd})" if step.cwd else ""
            print(f"  [{step.description}] {' '.join(step.argv)}{loc}")
        return 0

    for tool in ("git", "docker"):
        if which(tool) is None:
            print(f"✗ '{tool}' not found on PATH — install it and retry.", file=sys.stderr)
            return 1

    print(f"… building vLLM image(s) {', '.join(variants)} at ref {ref} — this can take ~30 min.")
    for step in plan:
        rc = exec_step(step)
        if rc != 0:
            print(f"✗ step failed ({step.description}); aborting.", file=sys.stderr)
            return 1

    if ref != cfg.vllm_ref:
        print(f"note: built at ref {ref} (settings pin is {cfg.vllm_ref}). "
              f"Update settings vllm_ref + vllm/VLLM_NODE_PROVENANCE.md to match.")
    print(f"✓ vllm-node build complete: {', '.join(variants)}")
    return 0
