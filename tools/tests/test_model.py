import os
import yaml
from sparkyard.model import load_models

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load():
    with open(os.path.join(FIXTURES, "models.yaml")) as f:
        raw = yaml.safe_load(f)
    return load_models(raw)


def test_count_and_names():
    models = _load()
    assert [m.name for m in models] == [
        "Nemotron-3-Nano-4B-FP8",
        "Qwen3.6-35B-A3B-FP8",
        "Qwen3.6-27B-uncensored-heretic-NVFP4",
    ]


def test_engine_defaults_applied():
    m = _load()[0]
    assert m.image == "vllm-node:latest"
    assert m.check_endpoint == "/health"


def test_per_model_overrides_win():
    m = _load()[1]
    assert m.ttl == 600
    assert m.gmem_override == 0.7069
    assert m.gmem_min == 0.55


def test_container_path_and_host_path():
    m = _load()[0]
    assert m.model_path == "/models/vllm/Nvidia/Nemotron-3-Nano-4B-FP8"
    assert m.model_host_path == "/models/vllm/Nvidia/Nemotron-3-Nano-4B-FP8"


def test_served_names_include_aliases():
    m = _load()[1]
    assert m.served_names == ["Qwen3.6-35B-A3B-FP8", "qwen3.6-35b-a3b"]


def test_no_aliases_means_just_name():
    m = _load()[0]
    assert m.served_names == ["Nemotron-3-Nano-4B-FP8"]


def test_llamacpp_fields():
    m = _load()[2]
    assert m.engine == "llamacpp"
    assert m.ctx_size == 65536
    assert m.parallel == 2
    assert m.unified_memory is False
    assert m.n_gpu_layers == 99


def test_chat_template_path_absolute_vs_relative():
    raw = {"defaults": {}, "models": [
        {"name": "A", "engine": "vllm", "container": "a", "path": "p",
         "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1,
         "chat_template": "vllm/x/ct.jinja"},
        {"name": "B", "engine": "vllm", "container": "b", "path": "p",
         "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1,
         "chat_template": "/templates/ct.jinja"},
    ]}
    ms = load_models(raw)
    assert ms[0].chat_template_path == "/models/vllm/x/ct.jinja"
    assert ms[1].chat_template_path == "/templates/ct.jinja"


def test_hf_repo_optional(tmp_path):
    raw = {"defaults": {}, "models": [
        {"name": "A", "engine": "vllm", "container": "a", "path": "vllm/x/A",
         "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1, "hf_repo": "org/A"},
        {"name": "B", "engine": "vllm", "container": "b", "path": "vllm/x/B",
         "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1},
    ]}
    ms = load_models(raw)
    assert ms[0].hf_repo == "org/A"
    assert ms[1].hf_repo is None


# --- extra_paths (draft models etc. referenced from flags) ---

def _m(**raw):
    base = {"name": "X", "engine": "vllm", "path": "vllm/Org/X"}
    base.update(raw)
    return load_models({"defaults": {}, "models": [base]})[0]


DSPARK = ('--speculative-config \'{"method":"dspark","model":"/models/vllm/RadixArk/'
          'Qwen3.8-27B-DSpark","num_speculative_tokens":7}\'')


def test_extra_paths_from_speculative_config():
    assert _m(vllm_flags=[DSPARK]).extra_paths == ["vllm/RadixArk/Qwen3.8-27B-DSpark"]


def test_extra_paths_none():
    assert _m().extra_paths == []
    assert _m(vllm_flags=["--trust-remote-code"]).extra_paths == []


def test_extra_paths_all_sources_deduped_in_order():
    m = _m(vllm_flags=[DSPARK, "--foo /models/a/b/", "--bar=/models/a/b"],
           extra_docker_args="-v /models/c,d -e X=/models/e/f.json",
           pre_launch_cmd="cp /models/g/h /tmp && ls /models/vllm/RadixArk/Qwen3.8-27B-DSpark")
    assert m.extra_paths == ["vllm/RadixArk/Qwen3.8-27B-DSpark", "a/b", "c", "e/f.json", "g/h"]


def test_extra_paths_excludes_own_path_and_chat_templates():
    m = _m(vllm_flags=["--x /models/vllm/Org/X", "--y /models/vllm/Org/X/sub",
                       "--chat-template /models/t/a.jinja", "--chat-template=/models/t/b.jinja"],
           chat_template="t/c.jinja", pre_launch_cmd="cat /models/t/c.jinja")
    assert m.extra_paths == []


def test_extra_paths_ignores_models_inside_other_paths():
    assert _m(extra_docker_args="-v /data/models/x:/opt/y").extra_paths == []


def test_extra_paths_escaped_json_stops_at_backslash():
    m = _m(vllm_flags=['--speculative-config {\\"model\\":\\"/models/vllm/D\\"}'])
    assert m.vllm_flags[0] == '--speculative-config {\\"model\\":\\"/models/vllm/D\\"}'
    assert m.extra_paths == ["vllm/D"]


def test_extra_paths_skip_bind_mount_targets_in_docker_args():
    m = _m(extra_docker_args="-v /host/cache:/models/cache:ro -v /x:/opt/y -e A=/models/real/p")
    assert m.extra_paths == ["real/p"]
