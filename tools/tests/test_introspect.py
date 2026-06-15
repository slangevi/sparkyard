import io
import json
import contextlib
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
