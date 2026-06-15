import os
import types
from sparkyard.model import load_models
from sparkyard import doctor


def _settings(root):
    return types.SimpleNamespace(llm_root=str(root))


def test_doctor_reports_present_and_missing(tmp_path):
    os.makedirs(tmp_path / "vllm" / "org" / "Present")
    raw = {"defaults": {}, "models": [
        {"name": "Present", "engine": "vllm", "container": "p", "path": "vllm/org/Present",
         "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1,
         "gmem": {"min": 0.1, "max": 0.2}, "image": "vllm-node:latest"},
        {"name": "Gone", "engine": "vllm", "container": "g", "path": "vllm/org/Gone",
         "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1,
         "gmem": {"min": 0.1, "max": 0.2}, "image": "vllm-node:latest"},
    ]}
    lines, summary = doctor.check(load_models(raw), _settings(tmp_path))
    text = "\n".join(lines)
    assert "Present" in text and "Gone" in text
    assert "MISSING" in text
    assert summary == "1/2 models have weights on disk"


def test_doctor_mamba_headsup_on_non_tf5(tmp_path):
    os.makedirs(tmp_path / "vllm" / "m")
    raw = {"defaults": {}, "models": [
        {"name": "M", "engine": "vllm", "container": "m", "path": "vllm/m",
         "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1,
         "gmem": {"min": 0.1, "max": 0.2}, "image": "vllm-node:latest",
         "vllm_flags": ["--mamba-ssm-cache-dtype float32"]},
    ]}
    lines, _ = doctor.check(load_models(raw), _settings(tmp_path))
    assert any("mamba" in ln.lower() and "tf5" in ln.lower() for ln in lines)
