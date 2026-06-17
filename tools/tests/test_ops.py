import types

from sparkyard import ops


def _fake_run(calls):
    def run(argv, cwd=None, env=None):
        calls.append({"argv": argv, "cwd": cwd, "env": env})
        return types.SimpleNamespace(returncode=0)
    return run


def test_secrets_runs_gen_secrets_script():
    calls = []
    assert ops.secrets("/repo", run=_fake_run(calls)) == 0
    assert calls[0]["argv"] == ["bash", "scripts/gen-secrets.sh"]
    assert calls[0]["cwd"] == "/repo"


def test_build_runs_compose_build():
    calls = []
    assert ops.build("/repo", run=_fake_run(calls)) == 0
    assert calls[0]["argv"] == ["docker", "compose", "build"] and calls[0]["cwd"] == "/repo"


def test_start_runs_compose_up_detached():
    calls = []
    assert ops.start("/repo", run=_fake_run(calls)) == 0
    assert calls[0]["argv"] == ["docker", "compose", "up", "-d"] and calls[0]["cwd"] == "/repo"


def test_stop_runs_compose_down():
    calls = []
    assert ops.stop("/repo", run=_fake_run(calls)) == 0
    assert calls[0]["argv"] == ["docker", "compose", "down"] and calls[0]["cwd"] == "/repo"


def test_bench_sets_mode_and_base_url_env():
    calls = []
    ops.bench("/repo", mode="speed", base_url="http://x", run=_fake_run(calls))
    c = calls[0]
    assert c["argv"] == ["bash", "scripts/bench.sh"] and c["cwd"] == "/repo"
    assert c["env"]["MODE"] == "speed" and c["env"]["BASE_URL"] == "http://x"


def test_bench_defaults_quality_and_omits_base_url():
    calls = []
    ops.bench("/repo", run=_fake_run(calls))
    c = calls[0]
    assert c["env"]["MODE"] == "quality" and "BASE_URL" not in c["env"]


def test_init_seeds_missing_then_runs_secrets(tmp_path):
    (tmp_path / "settings.example.yaml").write_text("llm_root: /x\n")
    (tmp_path / "models.example.yaml").write_text("models: []\n")
    calls = []
    assert ops.init(str(tmp_path), run=_fake_run(calls)) == 0
    assert (tmp_path / "settings.local.yaml").read_text() == "llm_root: /x\n"
    assert (tmp_path / "models.yaml").read_text() == "models: []\n"
    assert [c["argv"] for c in calls] == [["bash", "scripts/gen-secrets.sh"]]  # secrets once


def test_init_idempotent_leaves_existing(tmp_path):
    (tmp_path / "settings.example.yaml").write_text("EX\n")
    (tmp_path / "models.example.yaml").write_text("EX\n")
    (tmp_path / "settings.local.yaml").write_text("MINE\n")
    (tmp_path / "models.yaml").write_text("MINE\n")
    ops.init(str(tmp_path), run=_fake_run([]))
    assert (tmp_path / "settings.local.yaml").read_text() == "MINE\n"
    assert (tmp_path / "models.yaml").read_text() == "MINE\n"


def test_init_propagates_nonzero_secrets_rc(tmp_path):
    (tmp_path / "settings.example.yaml").write_text("x\n")
    (tmp_path / "models.example.yaml").write_text("x\n")
    def failing_run(argv, cwd=None, env=None):
        return types.SimpleNamespace(returncode=1)
    assert ops.init(str(tmp_path), run=failing_run) == 1
