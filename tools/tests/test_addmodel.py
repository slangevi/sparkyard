import os
import types
import yaml
import pytest
from sparkyard import addmodel, download


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


# A fully-valid base models.yaml — both engines' defaults present so load()/validate()
# pass after an entry is appended. `models:` stays the LAST top-level key (append needs it).
_BASE_MODELS = (
    "defaults:\n"
    "  vllm:\n"
    "    image: vllm-node:latest\n"
    "    gmem_min: 0.12\n"
    "    gmem_max: 0.85\n"
    "  llamacpp:\n"
    "    image: sparkyard/llama-cpp-spark:latest\n"
    "models:\n"
    "  - name: A\n"
    "    engine: vllm\n"
    "    container: a\n"
    "    path: vllm/o/A\n"
    "    max_model_len: 1\n"
    "    max_num_seqs: 1\n"
    "    kv_dtype_bytes: 1\n"
)


def _args(tmp_path, repo, **over):
    models = tmp_path / "models.yaml"
    if not models.exists():
        models.write_text(_BASE_MODELS)
    (tmp_path / "settings.local.yaml").write_text("llm_root: /llm\nrepo_path: /repo\nhome: /h\n")
    base = dict(repo=repo, name=None, dry_run=True, yes=False, download=False, gguf_file=None,
                models=str(models), settings=str(tmp_path / "settings.local.yaml"),
                llama_swap_out=str(tmp_path / "ls.yaml"), litellm_out=str(tmp_path / "ll.yaml"),
                env_out=str(tmp_path / ".env"))
    base.update(over)
    return types.SimpleNamespace(**base)


def test_gguf_dry_run_pattern_match(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(addmodel, "fetch_repo_metadata",
                        lambda repo: (None, ["m-Q4_K_M.gguf", "m-Q8_0.gguf"]))
    args = _args(tmp_path, "org/M-GGUF", gguf_file="Q4_K_M")
    rc = addmodel.run(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "engine: llamacpp" in out
    assert "gguf/org/M-GGUF/m-Q4_K_M.gguf" in out
    assert "WARNING" in out  # no config.json -> ctx fallback warning


def test_gguf_pattern_ambiguous_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(addmodel, "fetch_repo_metadata",
                        lambda repo: (None, ["m-Q4_K_M.gguf", "m-Q4_K_S.gguf"]))
    args = _args(tmp_path, "org/M-GGUF", gguf_file="Q4")
    assert addmodel.run(args) == 2


def test_gguf_no_flag_non_tty_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(addmodel, "fetch_repo_metadata",
                        lambda repo: (None, ["m-Q4_K_M.gguf", "m-Q8_0.gguf"]))
    args = _args(tmp_path, "org/M-GGUF")
    assert addmodel.run(args, isatty=lambda: False) == 2


def test_gguf_interactive_menu_picks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(addmodel, "fetch_repo_metadata",
                        lambda repo: (None, ["m-Q4_K_M.gguf", "m-Q8_0.gguf"]))
    args = _args(tmp_path, "org/M-GGUF")
    rc = addmodel.run(args, input_fn=lambda prompt: "2", isatty=lambda: True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "m-Q8_0.gguf" in out  # picked option 2 (sorted: Q4_K_M=1, Q8_0=2)


def test_gguf_append_render_and_download(tmp_path, monkeypatch):
    monkeypatch.setattr(addmodel, "fetch_repo_metadata",
                        lambda repo: ({"max_position_embeddings": 16384}, ["m-Q4_K_M.gguf"]))
    dl = []
    monkeypatch.setattr(download, "run", lambda models, settings, token, only=None: dl.append(only) or 0)
    args = _args(tmp_path, "org/M-GGUF", gguf_file="Q4_K_M", dry_run=False, yes=True, download=True)
    rc = addmodel.run(args)
    assert rc == 0
    data = yaml.safe_load(open(args.models))
    g = [m for m in data["models"] if m["engine"] == "llamacpp"][0]
    assert g["ctx_size"] == 16384            # inferred from config
    assert g["gguf"] == "gguf/org/M-GGUF/m-Q4_K_M.gguf"
    assert dl == [g["name"]]                  # --download delegated to download.run(only=name)


def test_gguf_entry_round_trips_through_validate_and_render(tmp_path, monkeypatch):
    from sparkyard.render import load, render_llama_swap
    monkeypatch.setattr(addmodel, "fetch_repo_metadata",
                        lambda repo: (None, ["Foo-Q4_K_M.gguf"]))
    # _args writes a fully-valid base models.yaml (vllm + llamacpp defaults); download
    # defaults to False so no fetch happens.
    args = _args(tmp_path, "org/Foo-GGUF", gguf_file="Q4_K_M", dry_run=False, yes=True)
    assert addmodel.run(args) == 0
    # load() runs validate() (raises on any error) + resolves placeholders
    settings, loaded = load(args.models, args.settings)
    assert any(m.engine == "llamacpp" for m in loaded)
    out = render_llama_swap(loaded)
    assert "Foo-GGUF" in out and "/models/gguf/org/Foo-GGUF/Foo-Q4_K_M.gguf" in out


def test_vllm_route_dry_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(addmodel, "fetch_repo_metadata",
                        lambda repo: ({"model_type": "llama", "max_position_embeddings": 4096},
                                      ["model.safetensors"]))
    args = _args(tmp_path, "org/Plain")
    rc = addmodel.run(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "engine: vllm" in out


def test_no_config_no_gguf_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(addmodel, "fetch_repo_metadata",
                        lambda repo: (None, ["README.md"]))
    args = _args(tmp_path, "org/Empty")
    assert addmodel.run(args) == 1
