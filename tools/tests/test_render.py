import os
import re
import yaml
from sparkyard.render import load, render_llama_swap

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
MODELS = os.path.join(FIXTURES, "models.yaml")
SETTINGS = os.path.join(FIXTURES, "settings.local.yaml")


def _render():
    _settings, models = load(MODELS, SETTINGS)
    return render_llama_swap(models)


def test_output_is_valid_yaml():
    doc = yaml.safe_load(_render())
    assert "models" in doc
    assert "Nemotron-3-Nano-4B-FP8" in doc["models"]


def test_no_blank_lines_inside_cmd_blocks():
    out = _render()
    lines = out.splitlines()
    in_cmd = False
    cmd_indent = 0
    for line in lines:
        stripped = line.strip()
        if re.match(r"\s*cmd: >\s*$", line):
            in_cmd = True
            cmd_indent = len(line) - len(line.lstrip())
            continue
        if in_cmd:
            if stripped and (len(line) - len(line.lstrip())) <= cmd_indent:
                in_cmd = False
            elif stripped == "":
                raise AssertionError("blank line inside a cmd: > block")


def test_vllm_basic_matches_golden():
    out = _render()
    block = out.split("Nemotron-3-Nano-4B-FP8:", 1)[1]
    block = "  Nemotron-3-Nano-4B-FP8:" + block.split('cmdStop: "docker stop vllm-nano4b-${PORT}"')[0] \
        + 'cmdStop: "docker stop vllm-nano4b-${PORT}"'
    with open(os.path.join(FIXTURES, "expected", "llama-swap.vllm-basic.yaml")) as f:
        expected = f.read().rstrip("\n")
    assert block.rstrip("\n") == expected


def test_vllm_advanced_has_overrides():
    out = _render()
    assert "GMEM_OVERRIDE=0.7069" in out
    assert "EXTRA_DOCKER_ARGS='-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e OMP_NUM_THREADS=4'" in out
    assert "--served-model-name Qwen3.6-35B-A3B-FP8 qwen3.6-35b-a3b" in out


def test_llamacpp_block_rendered():
    out = _render()
    assert "docker run --rm --name llamacpp-qwen27b-${PORT}" in out
    assert "-m /models/ollama/Alibaba/Qwen3.6-27B-heretic/model-NVFP4-MLP-Only.gguf" in out
    assert "-v /data/LLMs/ollama:/models/ollama" in out
    assert "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in out.split("llamacpp-qwen27b")[1].split("cmdStop")[0]


from sparkyard.render import render_litellm


def test_litellm_is_valid_yaml_with_expected_models():
    _s, models = load(MODELS, SETTINGS)
    doc = yaml.safe_load(render_litellm(models))
    names = [e["model_name"] for e in doc["model_list"]]
    assert "Nemotron-3-Nano-4B-FP8" in names
    assert "Qwen3.6-35B-A3B-FP8" in names
    assert "qwen3.6-35b-a3b" in names  # alias becomes its own entry


def test_litellm_never_inlines_the_key():
    _s, models = load(MODELS, SETTINGS)
    out = render_litellm(models)
    assert "os.environ/LITELLM_MASTER_KEY" in out
    assert "sk-" not in out  # no literal key value


def test_litellm_passes_through_params():
    _s, models = load(MODELS, SETTINGS)
    doc = yaml.safe_load(render_litellm(models))
    entry = next(e for e in doc["model_list"] if e["model_name"] == "Qwen3.6-35B-A3B-FP8")
    p = entry["litellm_params"]
    assert p["model"] == "openai/Qwen3.6-35B-A3B-FP8"
    assert p["api_base"] == "http://llama-swap:8080/v1"
    assert p["supports_reasoning"] is True
    assert p["max_tokens"] == 4096


from sparkyard.render import render_compose_env


def test_compose_env_matches_golden():
    settings, _models = load(MODELS, SETTINGS)
    out = render_compose_env(settings)
    with open(os.path.join(FIXTURES, "expected", "compose.env")) as f:
        expected = f.read()
    assert out == expected


import subprocess
import sys


def test_cli_validate_ok():
    result = subprocess.run(
        [sys.executable, "-m", "sparkyard.cli", "--models", MODELS,
         "--settings", SETTINGS, "validate"],
        cwd=os.path.dirname(os.path.dirname(__file__)),  # tools/
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "3 models valid" in result.stdout


def test_cli_validate_fails_closed_on_bad_engine(tmp_path):
    bad = tmp_path / "models.yaml"
    bad.write_text(
        "defaults: {}\n"
        "models:\n"
        "  - name: X\n"
        "    engine: sglang\n"
        "    container: x\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "sparkyard.cli", "--models", str(bad),
         "--settings", SETTINGS, "render", "--llama-swap-out", str(tmp_path / "ls.yaml"),
         "--litellm-out", str(tmp_path / "ll.yaml"), "--env-out", str(tmp_path / ".env")],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "sglang" in result.stderr
    assert not (tmp_path / "ls.yaml").exists()


def test_cli_friendly_error_on_malformed_yaml(tmp_path):
    bad = tmp_path / "models.yaml"
    bad.write_text("models:\n  - name: X\n    engine: vllm\n  bad: : :\n")
    r = subprocess.run([sys.executable, "-m", "sparkyard.cli", "--models", str(bad),
                        "--settings", SETTINGS, "validate"],
                       cwd=os.path.dirname(os.path.dirname(__file__)),
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "valid YAML" in r.stderr or "✗" in r.stderr


def test_cli_friendly_error_on_missing_settings_key(tmp_path):
    s = tmp_path / "settings.yaml"
    s.write_text("llm_root: /x\n")  # missing repo_path
    r = subprocess.run([sys.executable, "-m", "sparkyard.cli", "--models", MODELS,
                        "--settings", str(s), "validate"],
                       cwd=os.path.dirname(os.path.dirname(__file__)),
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr


def test_absolute_chat_template_not_prefixed():
    from sparkyard.model import load_models
    raw = {"defaults": {"vllm": {"image": "i", "gmem_min": 0.5, "gmem_max": 0.8, "safety_gib": 6}},
           "models": [{"name": "M", "engine": "vllm", "container": "m", "path": "p",
                       "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1,
                       "chat_template": "/templates/ct.jinja"}]}
    out = render_llama_swap(load_models(raw))
    assert "--chat-template /templates/ct.jinja" in out
    assert "/models//templates" not in out


def test_json_and_macro_flags_round_trip():
    # Guards the two historical bugs: a flag with ": " (JSON) breaking YAML,
    # and ${VAR} macros being wrongly resolved/escaped. Both must survive verbatim
    # through render -> yaml.safe_load.
    from sparkyard.model import load_models
    raw = {
        "defaults": {"vllm": {"image": "vllm-node:latest", "gmem_min": 0.5,
                              "gmem_max": 0.8, "safety_gib": 6}},
        "models": [{
            "name": "M", "engine": "vllm", "container": "m", "path": "x/y",
            "max_model_len": 1024, "max_num_seqs": 1, "kv_dtype_bytes": 1,
            "vllm_flags": [
                "--limit-mm-per-prompt '{\"image\": 2}'",
                "--tensor-parallel-size ${tensor_parallel}",
            ],
        }],
    }
    out = render_llama_swap(load_models(raw))
    cmd = yaml.safe_load(out)["models"]["M"]["cmd"]   # must parse as YAML
    assert "--limit-mm-per-prompt '{\"image\": 2}'" in cmd
    assert "--tensor-parallel-size ${tensor_parallel}" in cmd


def test_cli_render_default_env_out_is_dotenv(tmp_path):
    import shutil
    shutil.copy(MODELS, tmp_path / "models.yaml")
    shutil.copy(MODELS, tmp_path / "models.example.yaml")   # marker for repo-root autodiscovery
    shutil.copy(SETTINGS, tmp_path / "settings.local.yaml")
    tools_dir = os.path.dirname(os.path.dirname(__file__))  # tools/
    r = subprocess.run(
        [sys.executable, "-m", "sparkyard.cli", "render",
         "--llama-swap-out", str(tmp_path / "ls.yaml"),
         "--litellm-out", str(tmp_path / "ll.yaml")],   # NOTE: no --env-out
        cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": tools_dir},
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / ".env").exists()              # default is now .env
    assert not (tmp_path / ".env.sparkyard").exists()
