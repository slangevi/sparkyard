import os
import yaml
import pytest
from sparkyard import addmodel


def test_entry_to_yaml_is_indented_list_item():
    text = addmodel.entry_to_yaml({"name": "X", "engine": "vllm"})
    assert text.startswith("  - name: X\n")
    assert "\n    engine: vllm\n" in text


def test_append_model_adds_one_and_preserves_comments(tmp_path):
    src = tmp_path / "models.yaml"
    src.write_text(
        "# HEADER COMMENT\n"
        "defaults: {}\n"
        "models:\n"
        "  - name: A\n"
        "    engine: vllm\n"
        "    container: a\n"
    )
    addmodel.append_model(str(src), {"name": "B", "engine": "vllm", "container": "b"})
    out = src.read_text()
    assert "# HEADER COMMENT" in out
    data = yaml.safe_load(out)
    assert [m["name"] for m in data["models"]] == ["A", "B"]


def test_append_refuses_when_models_not_last_key(tmp_path):
    src = tmp_path / "models.yaml"
    src.write_text("models:\n  - name: A\n    engine: vllm\n    container: a\ndefaults: {}\n")
    with pytest.raises(addmodel.AppendError):
        addmodel.append_model(str(src), {"name": "B", "engine": "vllm", "container": "b"})
    assert "name: B" not in src.read_text()


def test_append_flow_style_models_raises_appenderror(tmp_path):
    src = tmp_path / "models.yaml"
    src.write_text("defaults: {}\nmodels: []\n")  # flow-style empty list
    with pytest.raises(addmodel.AppendError):
        addmodel.append_model(str(src), {"name": "B", "engine": "vllm", "container": "b"})
    assert "name: B" not in src.read_text()  # unchanged


def test_hf_token_read_from_secrets_next_to_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / "settings.local.yaml").write_text("llm_root: /x\nrepo_path: /y\nhome: /z\nnamespace: n\n")
    (tmp_path / "secrets.env").write_text('FOO=bar\nHF_TOKEN="hf_TESTTOKEN123"\n')
    tok = addmodel._hf_token(str(tmp_path / "settings.local.yaml"))
    assert tok == "hf_TESTTOKEN123"   # found beside settings, surrounding quotes stripped


def test_hf_token_none_when_blank(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / "settings.local.yaml").write_text("llm_root: /x\nrepo_path: /y\nhome: /z\nnamespace: n\n")
    (tmp_path / "secrets.env").write_text("HF_TOKEN=\n")
    assert addmodel._hf_token(str(tmp_path / "settings.local.yaml")) is None
