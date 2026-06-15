import pytest
from sparkyard.model import load_models
from sparkyard import download


def _models():
    raw = {"defaults": {}, "models": [
        {"name": "HasRepo", "engine": "vllm", "container": "h", "path": "vllm/org/HasRepo",
         "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1, "hf_repo": "org/HasRepo"},
        {"name": "NoRepo", "engine": "vllm", "container": "n", "path": "vllm/org/NoRepo",
         "max_model_len": 1, "max_num_seqs": 1, "kv_dtype_bytes": 1},
    ]}
    return load_models(raw)


class _Settings:
    llm_root = "/llm"


def test_select_all_returns_only_hf_repo_entries():
    got = [m.name for m in download.select(_models())]
    assert got == ["HasRepo"]


def test_select_only_by_name():
    got = [m.name for m in download.select(_models(), only="HasRepo")]
    assert got == ["HasRepo"]


def test_select_unknown_name_raises():
    with pytest.raises(ValueError):
        download.select(_models(), only="Nope")


def test_select_only_without_hf_repo_raises():
    with pytest.raises(ValueError):
        download.select(_models(), only="NoRepo")


def test_run_downloads_to_llm_root_path_and_skips_on_disk(monkeypatch):
    calls = []
    monkeypatch.setattr(download, "snapshot", lambda repo, local, token: calls.append((repo, local, token)))
    # nothing on disk -> HasRepo downloaded to /llm/vllm/org/HasRepo
    rc = download.run(_models(), _Settings(), token="tok", exists=lambda p: False)
    assert rc == 0
    assert calls == [("org/HasRepo", "/llm/vllm/org/HasRepo", "tok")]
    # everything on disk -> skipped, no snapshot calls
    calls.clear()
    rc = download.run(_models(), _Settings(), token=None, exists=lambda p: True)
    assert rc == 0 and calls == []


def test_run_failure_returns_nonzero(monkeypatch):
    def boom(repo, local, token):
        raise RuntimeError("hf down")
    monkeypatch.setattr(download, "snapshot", boom)
    rc = download.run(_models(), _Settings(), token=None, exists=lambda p: False)
    assert rc == 1
