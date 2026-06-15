import pytest
from sparkyard.placeholders import resolve


def test_resolves_string_token():
    assert resolve("{llm_root}/vllm/x", {"llm_root": "/data/LLMs"}) == "/data/LLMs/vllm/x"


def test_resolves_inside_list_and_dict():
    mapping = {"repo_path": "/repo", "namespace": "acme"}
    value = {"args": ["{repo_path}/mod:/mod:ro", "ghcr.io/{namespace}/img"]}
    assert resolve(value, mapping) == {
        "args": ["/repo/mod:/mod:ro", "ghcr.io/acme/img"]
    }


def test_leaves_non_strings_untouched():
    assert resolve(131072, {}) == 131072
    assert resolve(True, {}) is True
    assert resolve(None, {}) is None


def test_unknown_placeholder_raises():
    with pytest.raises(KeyError):
        resolve("{nope}/x", {"llm_root": "/data"})


def test_leaves_shell_dollar_braces_untouched():
    # llama-swap macros ${PORT}/${host}/${tensor_parallel} must NOT be resolved,
    # and must not raise even though {PORT} is not in the mapping.
    assert resolve("x ${PORT} ${host}", {"repo_path": "/r"}) == "x ${PORT} ${host}"
    assert resolve("-v {repo_path}/m:/m ${PORT}", {"repo_path": "/r"}) == "-v /r/m:/m ${PORT}"
    assert resolve("--tensor-parallel-size ${tensor_parallel}", {}) == "--tensor-parallel-size ${tensor_parallel}"
