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


def _vllm(name, path, **kw):
    e = {"name": name, "engine": "vllm", "container": name.lower(), "path": path,
         "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1,
         "gmem": {"min": 0.1, "max": 0.2}, "image": "vllm-node:latest"}
    e.update(kw)
    return e


# --- fastsafetensors: measured +32GiB peak on a 21.8GiB model (86 vs 54) -----
# It buffers the whole weight set while CUDA reserves the gmem budget. On a
# discrete GPU that is host RAM nobody notices; on unified memory both come out
# of the same pool and it can take the box down mid-load.

def test_doctor_warns_on_fastsafetensors(tmp_path):
    os.makedirs(tmp_path / "vllm" / "f")
    raw = {"defaults": {}, "models": [
        _vllm("F", "vllm/f", vllm_flags=["--load-format fastsafetensors"])]}
    lines, _ = doctor.check(load_models(raw), _settings(tmp_path))
    assert any("fastsafetensors" in ln for ln in lines), lines


def test_doctor_silent_without_fastsafetensors(tmp_path):
    os.makedirs(tmp_path / "vllm" / "g")
    raw = {"defaults": {}, "models": [
        _vllm("G", "vllm/g", vllm_flags=["--enable-prefix-caching"])]}
    lines, _ = doctor.check(load_models(raw), _settings(tmp_path))
    assert not any("fastsafetensors" in ln for ln in lines)


def test_doctor_warns_when_weights_exceed_the_gmem_budget(tmp_path):
    # gmem.max caps what vLLM may reserve; weights alone must fit inside it with
    # room for a KV cache. 40GiB of weights under a 0.2 budget cannot load.
    d = tmp_path / "vllm" / "big"; os.makedirs(d)
    with open(d / "model.safetensors", "wb") as fh:
        fh.truncate(40 * 1024**3)
    raw = {"defaults": {}, "models": [_vllm("Big", "vllm/big")]}
    lines, _ = doctor.check(load_models(raw), _settings(tmp_path))
    assert any("budget" in ln.lower() for ln in lines), lines


def test_doctor_no_budget_warning_when_weights_fit(tmp_path):
    d = tmp_path / "vllm" / "small"; os.makedirs(d)
    with open(d / "model.safetensors", "wb") as fh:
        fh.truncate(1 * 1024**3)
    raw = {"defaults": {}, "models": [
        _vllm("Small", "vllm/small", gmem={"min": 0.4, "max": 0.6})]}
    lines, _ = doctor.check(load_models(raw), _settings(tmp_path))
    assert not any("budget" in ln.lower() for ln in lines), lines
