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
