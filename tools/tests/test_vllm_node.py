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


def test_clone_when_absent_then_base_and_tf5():
    plan = vllm_node.build_plan(CFG, ["base", "tf5"], "7852e50e4", clone_exists=False)
    argvs = _argvs(plan)
    assert argvs[0] == ["git", "clone", CFG.upstream, CFG.clone_path]
    assert ["./build-and-copy.sh", "--vllm-ref", "7852e50e4"] in argvs
    assert ["./build-and-copy.sh", "--tf5", "--vllm-ref", "7852e50e4"] in argvs


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
        argvs = _argvs(vllm_node.build_plan(CFG, ["base", "tf5"], "7852e50e4",
                                            clone_exists=clone_exists))
        assert not any(a[:2] == ["git", "checkout"] for a in argvs)
        assert ["./build-and-copy.sh", "--vllm-ref", "7852e50e4"] in argvs


def test_default_variants_constant():
    assert vllm_node.DEFAULT_VARIANTS == ["base", "tf5"]


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
    assert seen == ["fetch upstream", "build base"]  # stops at the failing step


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
