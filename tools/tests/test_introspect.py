import io
import json
import contextlib
import pytest
from sparkyard import introspect


def _fake_opener(config, siblings):
    """opener(url) -> context-manager with .read() serving fixtures."""
    def opener(url):
        if url.endswith("/config.json"):
            body = json.dumps(config).encode()
        else:  # api/models/<repo>
            body = json.dumps({"siblings": [{"rfilename": f} for f in siblings]}).encode()
        return contextlib.nullcontext(io.BytesIO(body))
    return opener


def test_fetch_and_detect_gguf():
    op = _fake_opener({"model_type": "llama"}, ["model-Q4_K_M.gguf", "README.md"])
    config, files = introspect.fetch_repo_metadata("org/m", opener=op)
    assert any(f.endswith(".gguf") for f in files)
    entry, hints, is_gguf = introspect.derive_entry("org/m", config, files)
    assert is_gguf and entry is None


def test_derive_standard_fp8_model():
    config = {"model_type": "qwen3_5_moe", "architectures": ["Qwen3_5MoeForConditionalGeneration"],
              "max_position_embeddings": 131072}
    entry, hints, is_gguf = introspect.derive_entry("Alibaba/Qwen3.5-X", config, ["model.safetensors"])
    assert not is_gguf
    assert entry["engine"] == "vllm"
    assert entry["image"] == "vllm-node:latest"
    assert entry["path"] == "vllm/Alibaba/Qwen3.5-X"
    assert entry["max_model_len"] == 131072
    assert entry["kv_dtype_bytes"] == 1
    assert "--kv-cache-dtype fp8" in entry["vllm_flags"]
    assert hints == []


def test_derive_mamba_hint():
    config = {"model_type": "qwen3_next", "architectures": ["Qwen3NextForCausalLM"],
              "max_position_embeddings": 262144}
    entry, hints, _ = introspect.derive_entry("Alibaba/Coder-Next", config, ["model.safetensors"])
    assert entry["image"] == "vllm-node:latest"  # conservative baseline stays consistent
    assert any("mamba" in h.lower() for h in hints)


def test_derive_mxfp4_image():
    entry, hints, _ = introspect.derive_entry("openai/gpt-oss-120b", {"model_type": "gpt_oss"}, ["x.safetensors"])
    assert "mxfp4" in entry["image"]
    assert any("mxfp4" in h.lower() for h in hints)


def test_derive_max_len_fallback_when_missing():
    entry, _, _ = introspect.derive_entry("org/m", {"model_type": "llama"}, ["x.safetensors"])
    assert entry["max_model_len"] == 131072


def test_derive_entry_includes_hf_repo():
    from sparkyard.introspect import derive_entry
    cfg = {"num_hidden_layers": 32, "num_attention_heads": 32, "max_position_embeddings": 4096}
    entry, _hints, is_gguf = derive_entry("nvidia/Model-X", cfg, files=["config.json"])
    assert is_gguf is False
    assert entry["hf_repo"] == "nvidia/Model-X"
    assert entry["path"] == "vllm/nvidia/Model-X"


def test_fetch_metadata_config_optional_for_gguf_repo():
    # opener that 404s config.json but serves siblings -> config is None, files present
    def opener(url):
        if url.endswith("/config.json"):
            raise RuntimeError("404")
        body = json.dumps({"siblings": [{"rfilename": "model-Q4_K_M.gguf"}]}).encode()
        return contextlib.nullcontext(io.BytesIO(body))
    config, files = introspect.fetch_repo_metadata("org/m", opener=opener)
    assert config is None
    assert files == ["model-Q4_K_M.gguf"]


def test_fetch_metadata_raises_when_repo_unlistable():
    def opener(url):
        raise RuntimeError("network down")
    with pytest.raises(introspect.IntrospectError):
        introspect.fetch_repo_metadata("org/m", opener=opener)


def test_is_gguf_repo():
    assert introspect.is_gguf_repo(["a.safetensors", "model-Q4_K_M.gguf"]) is True
    assert introspect.is_gguf_repo(["a.safetensors", "config.json"]) is False


def test_infer_ctx_size_from_config():
    assert introspect.infer_ctx_size({"max_position_embeddings": 32768}) == (32768, True)
    assert introspect.infer_ctx_size({"text_config": {"max_position_embeddings": 8192}}) == (8192, True)


def test_infer_ctx_size_fallback():
    assert introspect.infer_ctx_size(None) == (8192, False)
    assert introspect.infer_ctx_size({"model_type": "llama"}) == (8192, False)


def test_is_gguf_repo_empty_list():
    assert introspect.is_gguf_repo([]) is False


def test_infer_ctx_size_custom_default():
    assert introspect.infer_ctx_size(None, default=4096) == (4096, False)


def test_derive_entry_raises_when_no_config_and_no_gguf():
    import pytest
    with pytest.raises(introspect.IntrospectError):
        introspect.derive_entry("org/m", None, ["model.safetensors"])


def test_derive_gguf_entry_shape_inferred_ctx():
    entry, hints = introspect.derive_gguf_entry(
        "Qwen/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        {"max_position_embeddings": 32768})
    assert entry["engine"] == "llamacpp"
    assert entry["container"] == "llamacpp-qwen2-5-3b-instruct-gguf"
    assert entry["hf_repo"] == "Qwen/Qwen2.5-3B-Instruct-GGUF"
    assert entry["mount"] == "{llm_root}/gguf:/models/gguf"
    assert entry["gguf"] == "gguf/Qwen/Qwen2.5-3B-Instruct-GGUF/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    assert entry["ctx_size"] == 32768
    assert entry["no_mmap"] is True and entry["unified_memory"] is True
    assert entry["llamacpp_flags"] == ["--jinja"]
    assert any("inferred" in h for h in hints)


def test_derive_gguf_entry_fallback_warns():
    entry, hints = introspect.derive_gguf_entry("bartowski/Foo-GGUF", "Foo-Q4_K_M.gguf", None)
    assert entry["ctx_size"] == 8192
    assert any("WARNING" in h for h in hints)


def test_derive_gguf_entry_name_override_and_subdir_hint():
    entry, hints = introspect.derive_gguf_entry(
        "org/Repo", "Q4_K_M/Foo-Q4_K_M.gguf", None, name="MyModel")
    assert entry["name"] == "MyModel"
    assert entry["container"] == "llamacpp-mymodel"
    assert any("subdir" in h.lower() for h in hints)
