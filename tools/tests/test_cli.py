import pytest
from sparkyard import cli


def test_add_model_parser_accepts_gguf_file(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.addmodel, "run", lambda args: captured.update(vars(args)) or 0)
    rc = cli.main(["add-model", "org/M-GGUF", "--gguf-file", "Q4_K_M"])
    assert rc == 0
    assert captured["gguf_file"] == "Q4_K_M"


def test_vllm_node_dispatches_with_parsed_args(monkeypatch, tmp_path):
    settings = tmp_path / "settings.local.yaml"
    settings.write_text("llm_root: /x\nrepo_path: /y\n")
    captured = {}

    def fake_run(args, s, **kw):
        captured["variant"] = args.variant
        captured["vllm_ref"] = args.vllm_ref
        captured["dry_run"] = args.dry_run
        captured["clone_path"] = s.vllm.clone_path
        return 0

    import sparkyard.vllm_node as vn
    monkeypatch.setattr(vn, "run", fake_run)
    rc = cli.main(["--settings", str(settings), "vllm-node",
                   "--variant", "mxfp4", "--vllm-ref", "abc1234", "--print"])
    assert rc == 0
    assert captured == {"variant": "mxfp4", "vllm_ref": "abc1234",
                        "dry_run": True, "clone_path": "/y/vllm/build/spark-vllm-docker"}


def test_vllm_node_rejects_unknown_variant():
    with pytest.raises(SystemExit):
        cli.main(["vllm-node", "--variant", "bogus"])


def test_vllm_node_handles_malformed_settings_yaml(tmp_path, capsys):
    bad = tmp_path / "settings.local.yaml"
    bad.write_text("llm_root: [unclosed\n")   # invalid YAML
    rc = cli.main(["--settings", str(bad), "vllm-node", "--print"])
    assert rc == 1
    assert "✗" in capsys.readouterr().err


def test_update_dispatches_with_check(monkeypatch, tmp_path):
    import sparkyard.update as upd
    captured = {}
    def fake_run(root, settings, *, check, notes=False, model=None, components=None, deps=None):
        captured["root"] = root; captured["check"] = check
        return 0
    monkeypatch.setattr(upd, "run", fake_run)
    # a discoverable checkout so autodiscovery resolves
    (tmp_path / "models.example.yaml").write_text("models: []\n")
    (tmp_path / "settings.local.yaml").write_text("llm_root: /x\nrepo_path: /y\n")
    monkeypatch.chdir(tmp_path)
    from sparkyard import cli
    assert cli.main(["update", "--check"]) == 0
    assert captured["check"] is True and captured["root"] == str(tmp_path)


def test_start_dispatches_to_ops(monkeypatch, tmp_path):
    import sparkyard.ops as ops
    captured = {}
    monkeypatch.setattr(ops, "start", lambda root: captured.update(root=root) or 0)
    (tmp_path / "models.example.yaml").write_text("models: []\n")
    monkeypatch.chdir(tmp_path)
    from sparkyard import cli
    assert cli.main(["start"]) == 0
    assert captured["root"] == str(tmp_path)


def test_bench_dispatches_with_flags(monkeypatch, tmp_path):
    import sparkyard.ops as ops
    captured = {}
    monkeypatch.setattr(ops, "bench",
                        lambda root, mode, base_url: captured.update(root=root, mode=mode, url=base_url) or 0)
    (tmp_path / "models.example.yaml").write_text("models: []\n")
    monkeypatch.chdir(tmp_path)
    from sparkyard import cli
    assert cli.main(["bench", "--mode", "speed", "--base-url", "http://x"]) == 0
    assert captured == {"root": str(tmp_path), "mode": "speed", "url": "http://x"}


def test_update_notes_flag_threads_through(monkeypatch, tmp_path):
    import sparkyard.update as upd
    captured = {}
    def fake_run(root, settings, *, check, notes=False, model=None, components=None, deps=None):
        captured.update(check=check, notes=notes, model=model)
        return 0
    monkeypatch.setattr(upd, "run", fake_run)
    (tmp_path / "models.example.yaml").write_text("models: []\n")
    (tmp_path / "settings.local.yaml").write_text("llm_root: /x\nrepo_path: /y\n")
    monkeypatch.chdir(tmp_path)
    from sparkyard import cli
    assert cli.main(["update", "--check", "--notes", "--model", "m1"]) == 0
    assert captured == {"check": True, "notes": True, "model": "m1"}


def test_update_forwards_positional_components(monkeypatch, tmp_path):
    import sparkyard.update as upd
    captured = {}
    def fake_run(root, settings, *, check, notes=False, model=None, components=None, deps=None):
        captured.update(components=components, check=check)
        return 0
    monkeypatch.setattr(upd, "run", fake_run)
    (tmp_path / "models.example.yaml").write_text("models: []\n")
    (tmp_path / "settings.local.yaml").write_text("llm_root: /x\nrepo_path: /y\n")
    monkeypatch.chdir(tmp_path)
    from sparkyard import cli
    assert cli.main(["update", "litellm", "open-webui", "--check"]) == 0
    assert captured["components"] == ("litellm", "open-webui")
    assert captured["check"] is True
