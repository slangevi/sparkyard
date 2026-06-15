import os
import yaml
from sparkyard.model import load_models
from sparkyard.validate import validate

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _models():
    with open(os.path.join(FIXTURES, "models.yaml")) as f:
        return load_models(yaml.safe_load(f))


def test_valid_fixture_has_no_errors():
    assert validate(_models()) == []


def test_duplicate_served_name_detected():
    models = _models()
    models[0].raw["aliases"] = ["Qwen3.6-35B-A3B-FP8"]
    errors = validate(models)
    assert any("Qwen3.6-35B-A3B-FP8" in e and "duplicate" in e.lower() for e in errors)


def test_unknown_engine_detected():
    models = _models()
    models[0].engine = "sglang"
    errors = validate(models)
    assert any("sglang" in e for e in errors)


def test_gmem_min_greater_than_max_detected():
    models = _models()
    models[1].raw["gmem"] = {"min": 0.9, "max": 0.5}
    errors = validate(models)
    assert any("gmem" in e.lower() for e in errors)


def test_missing_required_vllm_field_detected():
    models = _models()
    del models[0].raw["max_model_len"]
    errors = validate(models)
    assert any("max_model_len" in e for e in errors)


def test_raw_engine_now_rejected():
    raw = {"defaults": {}, "models": [{
        "name": "R", "engine": "raw", "container": "r",
        "cmd": "x ${PORT}", "cmd_stop": "docker stop r-${PORT}",
    }]}
    errors = validate(load_models(raw))
    assert any("raw" in e and "unknown engine" in e.lower() for e in errors)


def test_vllm_missing_gmem_detected():
    raw = {"defaults": {}, "models": [{
        "name": "V", "engine": "vllm", "container": "v", "path": "x",
        "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1, "image": "img",
    }]}
    errors = validate(load_models(raw))
    assert any("gmem" in e for e in errors)


def test_vllm_missing_image_detected():
    raw = {"defaults": {}, "models": [{
        "name": "V", "engine": "vllm", "container": "v", "path": "x",
        "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1,
        "gmem": {"min": 0.1, "max": 0.2},
    }]}
    errors = validate(load_models(raw))
    assert any("image" in e for e in errors)


def test_single_quote_in_extra_docker_args_detected():
    raw = {"defaults": {"vllm": {"image": "img", "gmem_min": 0.1, "gmem_max": 0.2}},
           "models": [{"name": "V", "engine": "vllm", "container": "v", "path": "x",
                       "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1,
                       "extra_docker_args": "-e FOO='bar'"}]}
    errors = validate(load_models(raw))
    assert any("single quote" in e for e in errors)


def _vllm(**over):
    base = {"name": "V", "engine": "vllm", "container": "v", "path": "p",
            "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1,
            "gmem": {"min": 0.1, "max": 0.2}, "image": "vllm-node:latest"}
    base.update(over)
    return {"defaults": {}, "models": [base]}


def test_kv_dtype_bytes_out_of_range_detected():
    errors = validate(load_models(_vllm(kv_dtype_bytes=4)))
    assert any("kv_dtype_bytes" in e for e in errors)


def test_mxfp4_flags_require_mxfp4_image():
    errors = validate(load_models(_vllm(vllm_flags=["--quantization mxfp4"])))
    assert any("mxfp4" in e for e in errors)
    ok = validate(load_models(_vllm(vllm_flags=["--quantization mxfp4"], image="vllm-node-mxfp4")))
    assert not any("mxfp4 image" in e for e in ok)


def test_kv_cache_dtype_fp8_requires_kv_dtype_bytes_1():
    errors = validate(load_models(_vllm(vllm_flags=["--kv-cache-dtype fp8"], kv_dtype_bytes=2)))
    assert any("kv_dtype_bytes: 1" in e for e in errors)


def test_kv_cache_dtype_nonfp8_requires_2():
    errors = validate(load_models(_vllm(vllm_flags=["--kv-cache-dtype auto"], kv_dtype_bytes=1)))
    assert any("kv_dtype_bytes: 2" in e for e in errors)


def test_fp8_variant_kv_cache_dtype_accepted():
    # fp8_e5m2 is 1-byte; kv_dtype_bytes:1 must NOT error
    errors = validate(load_models(_vllm(vllm_flags=["--kv-cache-dtype fp8_e5m2"], kv_dtype_bytes=1)))
    assert not any("kv_dtype_bytes" in e for e in errors)


def test_hf_repo_absent_is_ok():
    raw = {"defaults": {"vllm": {"image": "i", "gmem_min": 0.1, "gmem_max": 0.2}},
           "models": [{"name": "V", "engine": "vllm", "container": "v", "path": "x",
                       "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1}]}
    assert validate(load_models(raw)) == []


def test_hf_repo_good_is_ok():
    raw = {"defaults": {"vllm": {"image": "i", "gmem_min": 0.1, "gmem_max": 0.2}},
           "models": [{"name": "V", "engine": "vllm", "container": "v", "path": "x",
                       "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1,
                       "hf_repo": "nvidia/Model-X"}]}
    assert validate(load_models(raw)) == []


def test_hf_repo_malformed_detected():
    for bad in ("noslash", "too/many/slashes", "", 123):
        raw = {"defaults": {"vllm": {"image": "i", "gmem_min": 0.1, "gmem_max": 0.2}},
               "models": [{"name": "V", "engine": "vllm", "container": "v", "path": "x",
                           "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1,
                           "hf_repo": bad}]}
        errors = validate(load_models(raw))
        assert any("hf_repo" in e for e in errors), f"expected hf_repo error for {bad!r}"


def test_name_must_be_safe_charset():
    # YAML auto-typing: `name: yes` -> bool True, `name: 1.5` -> float; both non-str.
    errors = validate(load_models(_vllm(name=True)))
    assert any("name must be" in e for e in errors)
    errors = validate(load_models(_vllm(name="bad name!")))
    assert any("name must be" in e for e in errors)


def test_alias_must_be_safe_charset_and_list():
    errors = validate(load_models(_vllm(aliases="oops")))  # bare string, not a list
    assert any("aliases" in e.lower() and "list" in e.lower() for e in errors)
    errors = validate(load_models(_vllm(aliases=["bad alias!"])))
    assert any("alias" in e.lower() for e in errors)


def test_vllm_flags_must_be_a_list():
    errors = validate(load_models(_vllm(vllm_flags="--foo bar")))
    assert any("vllm_flags" in e and "list" in e.lower() for e in errors)


def test_llamacpp_flags_must_be_a_list():
    errors = validate(load_models(_vllm(llamacpp_flags="--jinja")))
    assert any("llamacpp_flags" in e and "list" in e.lower() for e in errors)


def test_litellm_must_be_a_mapping():
    errors = validate(load_models(_vllm(litellm=True)))
    assert any("litellm" in e.lower() and "mapping" in e.lower() for e in errors)


def test_yaml_special_name_rejected():
    # charset-valid strings that a YAML parser would re-type (bool/null/float/int)
    for bad in ("true", "no", "null", "1.5", "123"):
        errors = validate(load_models(_vllm(name=bad)))
        assert any("YAML-special" in e for e in errors), bad
    # a normal name with digits/dots/hyphens is fine
    assert not any("YAML-special" in e for e in validate(load_models(_vllm(name="Qwen2.5-3B-Instruct"))))


def test_yaml_special_alias_rejected():
    errors = validate(load_models(_vllm(aliases=["yes"])))
    assert any("YAML-special" in e for e in errors)


def test_litellm_passthrough_cannot_override_reserved_keys():
    for key in ("model", "api_base", "api_key"):
        errors = validate(load_models(_vllm(litellm={key: "x"})))
        assert any("litellm" in e.lower() and key in e for e in errors), key


def test_gmem_override_must_be_in_unit_interval():
    errors = validate(load_models(_vllm(gmem={"min": 0.1, "max": 0.2, "override": 1.5})))
    assert any("override" in e.lower() for e in errors)
    ok = validate(load_models(_vllm(gmem={"min": 0.1, "max": 0.2, "override": 0.7})))
    assert not any("override" in e.lower() for e in ok)
    errors = validate(load_models(_vllm(gmem={"min": 0.1, "max": 0.2, "override": True})))
    assert any("override" in e.lower() for e in errors)
