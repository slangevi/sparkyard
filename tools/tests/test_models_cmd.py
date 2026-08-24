import json
import os
import types

from sparkyard.model import load_models
from sparkyard import models_cmd


def _settings(root):
    return types.SimpleNamespace(llm_root=str(root))


RAW = {"defaults": {}, "models": [
    {"name": "Here", "engine": "vllm", "container": "h", "path": "vllm/o/Here",
     "max_model_len": 4096, "max_num_seqs": 4, "kv_dtype_bytes": 1,
     "gmem": {"min": 0.3, "max": 0.5}, "image": "vllm-node:latest",
     "aliases": ["short"]},
    {"name": "Gone", "engine": "vllm", "container": "g", "path": "vllm/o/Gone",
     "max_model_len": 8192, "max_num_seqs": 2, "kv_dtype_bytes": 1,
     "gmem": {"min": 0.3, "max": 0.5}, "image": "vllm-node:latest"},
    {"name": "Gguf", "engine": "llamacpp", "container": "q",
     "gguf": "ollama/o/m.gguf", "ctx_size": 2048},
]}


def test_table_lists_every_model_with_engine_and_weight_state(tmp_path):
    os.makedirs(tmp_path / "vllm" / "o" / "Here")
    out = models_cmd.render(load_models(RAW), _settings(tmp_path))
    assert "Here" in out and "Gone" in out and "Gguf" in out
    assert "vllm" in out and "llamacpp" in out
    assert "short" in out, "aliases must be visible — they are what callers use"


def test_on_disk_filter_hides_models_without_weights(tmp_path):
    os.makedirs(tmp_path / "vllm" / "o" / "Here")
    out = models_cmd.render(load_models(RAW), _settings(tmp_path), on_disk=True)
    assert "Here" in out and "Gone" not in out


def test_engine_filter(tmp_path):
    out = models_cmd.render(load_models(RAW), _settings(tmp_path), engine="llamacpp")
    assert "Gguf" in out and "Here" not in out


def test_json_output_is_machine_readable(tmp_path):
    os.makedirs(tmp_path / "vllm" / "o" / "Here")
    data = json.loads(models_cmd.render(load_models(RAW), _settings(tmp_path),
                                        as_json=True))
    by = {m["name"]: m for m in data}
    assert by["Here"]["on_disk"] is True and by["Gone"]["on_disk"] is False
    assert by["Here"]["engine"] == "vllm"
    assert by["Here"]["aliases"] == ["short"]
    assert by["Gguf"]["engine"] == "llamacpp"
