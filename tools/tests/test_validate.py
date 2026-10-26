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


# --- groups (llama-swap routing.router.settings.groups) ---
# Membership is validated fail-closed: llama-swap silently ignores a group whose
# member is not a model ID, which reads as "my model never stays loaded".

def _groups_ok():
    return {"resident": {"swap": False, "exclusive": False, "persistent": True,
                         "members": ["Nemotron-3-Nano-4B-FP8"]}}


def test_valid_groups_have_no_errors():
    assert validate(_models(), _groups_ok()) == []


def test_group_member_must_be_a_known_model():
    groups = {"resident": {"members": ["Nope-7B"]}}
    errors = validate(_models(), groups)
    assert any("Nope-7B" in e and "resident" in e for e in errors)


def test_group_member_may_not_be_an_alias():
    # llama-swap groups key off model IDs, not the aliases LiteLLM serves.
    models = _models()
    models[0].raw["aliases"] = ["shorthand"]
    errors = validate(models, {"resident": {"members": ["shorthand"]}})
    assert any("shorthand" in e for e in errors)


def test_model_in_two_groups_detected():
    name = "Nemotron-3-Nano-4B-FP8"
    groups = {"a": {"members": [name]}, "b": {"members": [name]}}
    errors = validate(_models(), groups)
    assert any(name in e and "more than one group" in e for e in errors)


def test_group_members_must_be_a_non_empty_list():
    errors = validate(_models(), {"resident": {"members": []}})
    assert any("resident" in e and "members" in e for e in errors)


def test_group_flags_must_be_boolean():
    groups = {"resident": {"persistent": "yes",
                           "members": ["Nemotron-3-Nano-4B-FP8"]}}
    errors = validate(_models(), groups)
    assert any("persistent" in e for e in errors)


def test_reclaim_containers_must_be_a_list_of_container_names():
    assert not validate(load_models(_vllm(reclaim_containers=["comfyui", "my_app.2"])))
    for bad in ("comfyui", ["ok", "bad name"], ["x;rm -rf /"], ["-flag"], [""], [3]):
        errors = validate(load_models(_vllm(reclaim_containers=bad)))
        assert any("reclaim_containers" in e for e in errors), bad


# --- nodes / cluster ---
import pytest
from sparkyard.settings import Cluster, ClusterWorker
from sparkyard.validate import validate_cluster


def _cluster(**over):
    kw = dict(head_cx7_ip="10.0.0.1", eth_if="enp1s0f0np0", ib_hca="rocep1s0f0",
              workers=[ClusterWorker("w", "spark-2", "scott", "10.0.0.2")], master_port=29501)
    kw.update(over)
    return Cluster(**kw)


def _two_models(**over):
    a = _vllm(**over)["models"][0]
    b = dict(a, name="V2", container="v2")
    return load_models({"defaults": {}, "models": [a, b]})


@pytest.mark.parametrize("val", [3, True, "2"])
def test_nodes_must_be_1_or_2(val):
    errors = validate(load_models(_vllm(nodes=val)), cluster=_cluster())
    assert any("nodes must be 1 or 2" in e for e in errors)


def test_nodes_requires_cluster():
    errors = validate(load_models(_vllm(nodes=2)))
    assert any("nodes: 2 needs a cluster: block in settings.local.yaml" in e for e in errors)


def test_nodes_llamacpp_rejected():
    raw = {"defaults": {}, "models": [{"name": "L", "engine": "llamacpp", "container": "l",
           "gguf": "g", "mount": "m", "ctx_size": 1, "image": "i", "nodes": 2}]}
    errors = validate(load_models(raw), cluster=_cluster())
    assert any("nodes > 1 is only supported for engine vllm" in e for e in errors)


@pytest.mark.parametrize("flag", ["--tensor-parallel-size 2", "--tensor-parallel-size=2",
                                  "-tp 2", "--headless", "--distributed-executor-backend ray",
                                  "--master-port=1"])
def test_cluster_reserved_flags_rejected(flag):
    errors = validate(load_models(_vllm(nodes=2, vllm_flags=[flag])), cluster=_cluster())
    assert any("is set by the cluster launcher" in e for e in errors)


def test_single_node_tp_flag_still_allowed():
    errors = validate(load_models(_vllm(vllm_flags=["--tensor-parallel-size 1"])))
    assert errors == []


def test_valid_two_node_model_has_no_errors():
    assert validate(load_models(_vllm(nodes=2)), cluster=_cluster()) == []


def test_two_cluster_models_in_coloading_group_rejected():
    groups = {"g": {"swap": False, "members": ["V", "V2"]}}
    errors = validate(_two_models(nodes=2), groups, _cluster())
    assert any("cannot be loaded together" in e for e in errors)


@pytest.mark.parametrize("g", [{"members": ["V", "V2"]}, {"swap": True, "members": ["V", "V2"]}])
def test_two_cluster_models_in_swap_group_ok(g):
    errors = validate(_two_models(nodes=2), {"g": g}, _cluster())
    assert not any("cannot be loaded together" in e for e in errors)


@pytest.mark.parametrize("workers", [[], [ClusterWorker("a", "h", "u", "10.0.0.2"),
                                          ClusterWorker("b", "h2", "u", "10.0.0.3")]])
def test_validate_cluster_one_worker(workers):
    assert any("exactly one worker" in e for e in validate_cluster(_cluster(workers=workers)))


def test_validate_cluster_ip_and_port():
    bad = _cluster(workers=[ClusterWorker("w", "h", "u", "not-an-ip")])
    assert any("cx7_ip" in e for e in validate_cluster(bad))
    assert any("master_port" in e for e in validate_cluster(_cluster(master_port=0)))
    assert validate_cluster(_cluster()) == []


def test_two_cluster_models_in_separate_nonexclusive_groups_rejected():
    groups = {"a": {"exclusive": False, "members": ["V"]},
              "b": {"exclusive": False, "members": ["V2"]}}
    errors = validate(_two_models(nodes=2), groups, _cluster())
    assert any("cannot be loaded together" in e for e in errors)


def test_cluster_models_nonexclusive_plus_default_group_rejected():
    # A non-exclusive group does not unload the (exclusive) default group, so
    # V (in 'a') and V2 (in the default group 'b') can be resident together.
    groups = {"a": {"exclusive": False, "members": ["V"]}, "b": {"members": ["V2"]}}
    errors = validate(_two_models(nodes=2), groups, _cluster())
    assert any("cannot be loaded together" in e for e in errors)


def test_cluster_model_nonexclusive_plus_ungrouped_rejected():
    groups = {"a": {"exclusive": False, "members": ["V"]}}
    errors = validate(_two_models(nodes=2), groups, _cluster())
    assert any("cannot be loaded together" in e and "V2" in e for e in errors)


def test_cluster_models_persistent_plus_default_group_rejected():
    # A persistent group is never unloaded, so a second nodes: 2 model loaded
    # anywhere else co-resides with it.
    groups = {"p": {"persistent": True, "members": ["V"]}}
    errors = validate(_two_models(nodes=2), groups, _cluster())
    assert any("cannot be loaded together" in e for e in errors)


def test_cluster_models_both_in_nonexclusive_swapping_group_ok():
    groups = {"a": {"exclusive": False, "members": ["V", "V2"]}}
    errors = validate(_two_models(nodes=2), groups, _cluster())
    assert not any("cannot be loaded together" in e for e in errors)


def test_cluster_models_both_in_nonexclusive_nonswapping_group_rejected():
    groups = {"a": {"exclusive": False, "swap": False, "members": ["V", "V2"]}}
    errors = validate(_two_models(nodes=2), groups, _cluster())
    assert any("cannot be loaded together" in e for e in errors)


def test_single_cluster_model_in_persistent_group_with_single_node_members_ok():
    a = _vllm(nodes=2)["models"][0]
    b = dict(a, name="Small", container="small")
    b.pop("nodes")
    models = load_models({"defaults": {}, "models": [a, b]})
    groups = {"resident": {"swap": False, "exclusive": False, "persistent": True,
                           "members": ["V", "Small"]}}
    assert validate(models, groups, _cluster()) == []


def test_cluster_model_listed_twice_not_coload():
    groups = {"g": {"swap": False, "members": ["V", "V"]}}
    errors = validate(_two_models(nodes=2), groups, _cluster())
    assert not any("cannot be loaded together" in e for e in errors)


@pytest.mark.parametrize("fld,val", [("eth_if", ""), ("eth_if", "  "), ("ib_hca", ""),
                                     ("eth_if", None)])
def test_validate_cluster_fields_nonempty(fld, val):
    assert any(f"cluster: {fld} must be a non-empty string" in e
               for e in validate_cluster(_cluster(**{fld: val})))


@pytest.mark.parametrize("fld", ["ssh_host", "ssh_user"])
@pytest.mark.parametrize("val", ["", "   ", None])
def test_validate_cluster_worker_fields_nonempty(fld, val):
    kw = dict(name="w", ssh_host="spark-2", ssh_user="scott", cx7_ip="10.0.0.2")
    kw[fld] = val
    errors = validate_cluster(_cluster(workers=[ClusterWorker(**kw)]))
    assert any(f"cluster: worker {fld} must be a non-empty string" in e for e in errors)


@pytest.mark.parametrize("val", ["enp1s0 f0", "enp'1", 'e"0', "-oProxyCommand=x", "e0;rm", "e0\n"])
def test_validate_cluster_eth_if_charset(val):
    assert any("cluster: eth_if" in e and "invalid" in e
               for e in validate_cluster(_cluster(eth_if=val)))


@pytest.mark.parametrize("val", ["rocep1s0f0, roceP2p1s0f0", "rocep1s0f0,", "-x", "a,'b"])
def test_validate_cluster_ib_hca_charset(val):
    assert any("cluster: ib_hca" in e and "invalid" in e
               for e in validate_cluster(_cluster(ib_hca=val)))


@pytest.mark.parametrize("fld", ["ssh_host", "ssh_user"])
@pytest.mark.parametrize("val", ["-oProxyCommand=touch", "spark 2", "a'b", 'a"b', "a`b`"])
def test_validate_cluster_worker_charset(fld, val):
    kw = dict(name="w", ssh_host="spark-2", ssh_user="scott", cx7_ip="10.0.0.2")
    kw[fld] = val
    errors = validate_cluster(_cluster(workers=[ClusterWorker(**kw)]))
    assert any(f"cluster: worker {fld}" in e and "invalid" in e for e in errors)


def test_validate_cluster_charset_accepts_real_values():
    c = _cluster(eth_if="enP2p1s0f0np0", ib_hca="rocep1s0f0,roceP2p1s0f0",
                 workers=[ClusterWorker("w", "spark-2.lan", "scott_x", "10.0.0.2")])
    assert validate_cluster(c) == []
    c = _cluster(workers=[ClusterWorker("w", "fe80::1", "a.b-c", "10.0.0.2")])
    assert validate_cluster(c) == []
