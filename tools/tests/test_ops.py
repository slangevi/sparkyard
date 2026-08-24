import types

from sparkyard import ops


def _fake_run(calls, health="healthy"):
    """Models subprocess.run, including stdout for captured calls — without it a
    health-poll bug (stdout never captured, so 'unhealthy' read as ready) passes."""
    def run(argv, cwd=None, env=None, **kw):
        calls.append({"argv": argv, "cwd": cwd, "env": env, "kw": kw})
        out = health if "inspect" in argv else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
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


# --- reload: the step that `render` always needs and `start` never does ------
# llama-swap/config.yaml is bind-mounted, so `docker compose up -d` sees no
# service change and leaves the old config running. Every render therefore
# needed a manual `docker restart llama-swap litellm`, and waiting on llama-swap's
# health alone raced LiteLLM ("Connection reset by peer").

RELOAD_SERVICES = ["llama-swap", "litellm"]


def test_reload_restarts_the_config_consuming_services():
    calls = []
    assert ops.reload("/repo", run=_fake_run(calls)) == 0
    restart = [c for c in calls if "restart" in c["argv"]]
    assert restart, f"no restart issued: {[c['argv'] for c in calls]}"
    for svc in RELOAD_SERVICES:
        assert svc in restart[0]["argv"], f"{svc} not restarted"
    assert restart[0]["cwd"] == "/repo"


def test_reload_waits_for_health_after_restarting():
    calls = []
    ops.reload("/repo", run=_fake_run(calls))
    argvs = [" ".join(c["argv"]) for c in calls]
    waited = [a for a in argvs if "inspect" in a or "Health" in a]
    assert waited, f"reload must wait for health, got: {argvs}"


def test_reload_health_wait_covers_every_restarted_service():
    calls = []
    ops.reload("/repo", run=_fake_run(calls))
    blob = " ".join(" ".join(c["argv"]) for c in calls)
    for svc in RELOAD_SERVICES:
        assert blob.count(svc) >= 2, f"{svc} restarted but not health-waited"


def test_reload_polls_until_healthy_not_just_once():
    # Regression: the health poll read p.stdout, but subprocess.run does not
    # capture by default, so stdout was always None and every service looked
    # ready immediately. The inspect call must request capture.
    calls = []
    ops.reload("/repo", run=_fake_run(calls), sleep=lambda _s: None)
    ins = [c for c in calls if "inspect" in c["argv"]]
    assert ins, "no health inspect issued"
    assert ins[0]["kw"].get("capture_output"), "inspect must capture stdout"
    assert ins[0]["kw"].get("text"), "inspect must decode stdout"


def test_reload_fails_when_a_service_never_becomes_healthy():
    calls = []
    rc = ops.reload("/repo", run=_fake_run(calls, health="starting"),
                    sleep=lambda _s: None, attempts=3)
    assert rc != 0, "a service stuck in 'starting' must not report success"


def test_reload_returns_nonzero_when_restart_fails():
    def run(argv, cwd=None, env=None):
        return types.SimpleNamespace(returncode=1 if "restart" in argv else 0)
    assert ops.reload("/repo", run=run) != 0
