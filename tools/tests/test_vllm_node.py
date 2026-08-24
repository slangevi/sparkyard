import types

from sparkyard import vllm_node
from sparkyard.settings import VllmBuild

CFG = VllmBuild(
    upstream="https://github.com/eugr/spark-vllm-docker",
    clone_path="/repo/vllm/build/spark-vllm-docker",
    vllm_ref="7852e50e4",
)


def _argvs(plan):
    return [step.argv for step in plan]


def test_clone_when_absent_then_builds_base():
    plan = vllm_node.build_plan(CFG, ["base"], "7852e50e4", clone_exists=False)
    argvs = _argvs(plan)
    assert argvs[0] == ["git", "clone", CFG.upstream, CFG.clone_path]
    assert ["./build-and-copy.sh", "--vllm-ref", "7852e50e4"] in argvs


def test_fetch_when_clone_present():
    plan = vllm_node.build_plan(CFG, ["base"], "7852e50e4", clone_exists=True)
    assert plan[0].argv == ["git", "fetch"]
    assert plan[0].cwd == CFG.clone_path


def test_build_steps_run_in_clone_dir():
    plan = vllm_node.build_plan(CFG, ["base"], "7852e50e4", clone_exists=True)
    build = [s for s in plan if s.argv[0] == "./build-and-copy.sh"][0]
    assert build.cwd == CFG.clone_path


def test_ref_override_threads_into_build():
    plan = vllm_node.build_plan(CFG, ["base"], "abc1234", clone_exists=True)
    argvs = _argvs(plan)
    assert ["./build-and-copy.sh", "--vllm-ref", "abc1234"] in argvs
    assert not any(a[:2] == ["git", "checkout"] for a in argvs)


def test_mxfp4_has_no_ref_and_no_checkout():
    plan = vllm_node.build_plan(CFG, ["mxfp4"], "7852e50e4", clone_exists=True)
    argvs = _argvs(plan)
    assert ["./build-and-copy.sh", "--exp-mxfp4"] in argvs
    assert not any(a[:2] == ["git", "checkout"] for a in argvs)  # mxfp4 tracks its own ref
    assert not any("--vllm-ref" in a for a in argvs)


def test_build_plan_never_checks_out_vllm_ref_in_tooling_clone():
    # Regression: `ref` is a vLLM commit; the spark-vllm-docker tooling clone does
    # NOT contain it. build-and-copy.sh checks vLLM out itself via --vllm-ref, so
    # build_plan must never `git checkout <ref>` in the tooling clone — doing so
    # aborts the build ("pathspec did not match") on a fresh clone.
    for clone_exists in (True, False):
        argvs = _argvs(vllm_node.build_plan(CFG, ["base", "mxfp4"], "7852e50e4",
                                            clone_exists=clone_exists))
        assert not any(a[:2] == ["git", "checkout"] for a in argvs)
        assert ["./build-and-copy.sh", "--vllm-ref", "7852e50e4"] in argvs


def test_default_variants_is_base_only():
    # tf5 was "base + transformers v5". Upstream made v5 the default and reduced
    # --tf5 to a tag alias ("no longer alter[s] dependency resolution"), so the two
    # images build byte-equivalent — verified: same vllm, torch, transformers 5.15.1
    # and flashinfer. Building it cost ~30 min of Rust compilation for a duplicate.
    assert vllm_node.DEFAULT_VARIANTS == ["base"]


def test_tf5_is_no_longer_a_buildable_variant():
    assert "tf5" not in vllm_node._REF_VARIANTS


def _args(variant=None, vllm_ref=None, dry_run=False):
    return types.SimpleNamespace(variant=variant, vllm_ref=vllm_ref, dry_run=dry_run)


class _S:
    vllm = CFG


def test_print_runs_no_steps_and_returns_zero():
    calls = []
    rc = vllm_node.run(_args(dry_run=True), _S(),
                       exists=lambda p: True,
                       which=lambda t: "/usr/bin/" + t,
                       exec_step=lambda step: calls.append(step) or 0)
    assert rc == 0
    assert calls == []  # dry-run executes nothing


def test_missing_git_returns_one_and_runs_nothing():
    calls = []
    rc = vllm_node.run(_args(), _S(),
                       exists=lambda p: True,
                       which=lambda t: None,  # nothing on PATH
                       exec_step=lambda step: calls.append(step) or 0)
    assert rc == 1
    assert calls == []


def test_exec_runs_steps_in_order_until_failure():
    seen = []

    def exec_step(step):
        seen.append(step.description)
        return 1 if step.description == "build base" else 0

    rc = vllm_node.run(_args(), _S(),
                       exists=lambda p: True,           # clone present -> fetch
                       which=lambda t: "/usr/bin/" + t,
                       exec_step=exec_step)
    assert rc == 1
    assert seen == ["fetch upstream", "update tooling clone", "build base"]  # stops at the failing step


def test_clone_existence_checks_dot_git():
    probed = []
    vllm_node.run(_args(dry_run=True), _S(),
                  exists=lambda p: probed.append(p) or True,
                  which=lambda t: "/usr/bin/" + t,
                  exec_step=lambda step: 0)
    assert probed == [CFG.clone_path + "/.git"]


def test_variant_arg_selects_single_variant():
    seen = []
    vllm_node.run(_args(variant="mxfp4"), _S(),
                  exists=lambda p: True,
                  which=lambda t: "/usr/bin/" + t,
                  exec_step=lambda step: seen.append(step.description) or 0)
    assert "build mxfp4" in seen
    assert "build base" not in seen


# --- the tooling clone must actually advance after fetching ------------------
# `git fetch` alone leaves the clone at its old HEAD, so the build uses whatever
# spark-vllm-docker was cloned months ago. Observed in the field: a clone 101
# commits behind lacked "Fix flashinfer build issues", and the build died at
# `uv pip install` on an unsatisfiable flashinfer/nvidia-cutlass-dsl conflict.

def test_build_plan_advances_tooling_clone_after_fetch():
    plan = vllm_node.build_plan(CFG, ["base"], "7852e50e4", clone_exists=True)
    argvs = _argvs(plan)
    ff = [a for a in argvs if a[:2] == ["git", "merge"]]
    assert ff, f"no fast-forward step in plan: {argvs}"
    assert "--ff-only" in ff[0], "clone must advance by fast-forward only"
    # and it must happen after the fetch, before any build
    i_fetch = argvs.index(["git", "fetch"])
    i_ff = argvs.index(ff[0])
    i_build = next(i for i, a in enumerate(argvs) if a[0] == "./build-and-copy.sh")
    assert i_fetch < i_ff < i_build


def test_fresh_clone_needs_no_fast_forward():
    # A just-cloned repo is already at the default branch tip.
    argvs = _argvs(vllm_node.build_plan(CFG, ["base"], "7852e50e4", clone_exists=False))
    assert not any(a[:2] == ["git", "merge"] for a in argvs)


def test_advancing_the_clone_is_not_a_vllm_ref_checkout():
    # Guard the PR #11 regression: advancing the tooling clone must not reintroduce
    # `git checkout <vllm ref>`, which aborts on a fresh clone.
    argvs = _argvs(vllm_node.build_plan(CFG, ["base"], "abc1234", clone_exists=True))
    assert not any(a[:2] == ["git", "checkout"] for a in argvs)
    assert not any("abc1234" in a for a in argvs if a[0] == "git")


# --- --use-wheels: build the runner from prebuilt, pre-resolved wheels -------
# Source builds take ~30 min and can fail on dependency conflicts that the
# published wheel set has already resolved. --use-wheels built in 11:15 and
# installed the same flashinfer 0.6.18 that the source path could not satisfy.

def test_use_wheels_threads_into_build():
    plan = vllm_node.build_plan(CFG, ["base"], "7852e50e4", clone_exists=True,
                                use_wheels=True)
    build = [a for a in _argvs(plan) if a[0] == "./build-and-copy.sh"][0]
    assert "--use-wheels" in build


def test_use_wheels_defaults_off():
    plan = vllm_node.build_plan(CFG, ["base"], "7852e50e4", clone_exists=True)
    build = [a for a in _argvs(plan) if a[0] == "./build-and-copy.sh"][0]
    assert "--use-wheels" not in build


def test_use_wheels_applies_to_every_variant():
    plan = vllm_node.build_plan(CFG, ["base", "mxfp4"], "7852e50e4", clone_exists=True,
                                use_wheels=True)
    builds = [a for a in _argvs(plan) if a[0] == "./build-and-copy.sh"]
    assert len(builds) == 2
    assert all("--use-wheels" in b for b in builds)
