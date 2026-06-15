from sparkyard import cli


def test_add_model_parser_accepts_gguf_file(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.addmodel, "run", lambda args: captured.update(vars(args)) or 0)
    rc = cli.main(["add-model", "org/M-GGUF", "--gguf-file", "Q4_K_M"])
    assert rc == 0
    assert captured["gguf_file"] == "Q4_K_M"
