import pytest
from sparkyard.model import load_models
from sparkyard import download


class _SettingsP:
    llm_root = "/llm"
    repo_path = "/repo"
    home = "/home"
    def placeholder_map(self):
        return {"llm_root": self.llm_root, "repo_path": self.repo_path, "home": self.home}


def _gguf_model(mount, gguf):
    raw = {"defaults": {}, "models": [
        {"name": "G", "engine": "llamacpp", "container": "g", "hf_repo": "org/G-GGUF",
         "mount": mount, "gguf": gguf, "ctx_size": 8192},
    ]}
    return load_models(raw)


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


def test_shard_family_single_file():
    assert download.shard_family("model-Q4_K_M.gguf") == ["model-Q4_K_M.gguf"]


def test_shard_family_multipart():
    assert download.shard_family("model-Q4_K_M-00001-of-00003.gguf") == [
        "model-Q4_K_M-00001-of-00003.gguf",
        "model-Q4_K_M-00002-of-00003.gguf",
        "model-Q4_K_M-00003-of-00003.gguf",
    ]


def test_gguf_families_groups_shards():
    files = ["m-Q4_K_M.gguf", "m-Q8_0-00001-of-00002.gguf", "m-Q8_0-00002-of-00002.gguf", "README.md"]
    fams = download.gguf_families(files)
    assert set(fams) == {"m-Q4_K_M", "m-Q8_0"}
    assert fams["m-Q8_0"] == ["m-Q8_0-00001-of-00002.gguf", "m-Q8_0-00002-of-00002.gguf"]
    assert fams["m-Q4_K_M"] == ["m-Q4_K_M.gguf"]


def test_gguf_target_resolves_host_dir_and_basename():
    m = _gguf_model("/llm/gguf:/models/gguf", "gguf/org/G-GGUF/G-Q4_K_M.gguf")[0]
    host_dir, base = download.gguf_target(m, _SettingsP())
    assert host_dir == "/llm/gguf/org/G-GGUF"
    assert base == "G-Q4_K_M.gguf"


def test_gguf_target_handles_unresolved_placeholder_mount():
    # if a caller passes a not-yet-resolved mount, gguf_target still resolves it
    m = _gguf_model("{llm_root}/ollama:/models/ollama", "ollama/org/G-GGUF/G-Q4_K_M.gguf")[0]
    host_dir, base = download.gguf_target(m, _SettingsP())
    assert host_dir == "/llm/ollama/org/G-GGUF"
    assert base == "G-Q4_K_M.gguf"


def test_run_downloads_gguf_with_allow_patterns(monkeypatch):
    calls = []
    monkeypatch.setattr(download, "snapshot",
                        lambda repo, local, token, allow_patterns=None: calls.append((repo, local, allow_patterns)))
    models = _gguf_model("/llm/gguf:/models/gguf", "gguf/org/G-GGUF/G-Q4_K_M.gguf")
    rc = download.run(models, _SettingsP(), token="t", exists=lambda p: False)
    assert rc == 0
    assert calls == [("org/G-GGUF", "/llm/gguf/org/G-GGUF", ["G-Q4_K_M.gguf"])]


def test_gguf_target_rejects_superstring_mount_segment():
    m = _gguf_model("/llm/gguf:/models/gguf", "gguf2/org/x/f.gguf")[0]
    with pytest.raises(ValueError):
        download.gguf_target(m, _SettingsP())


def test_run_skips_gguf_when_all_shards_on_disk(monkeypatch):
    calls = []
    monkeypatch.setattr(download, "snapshot", lambda *a, **k: calls.append(a))
    models = _gguf_model("/llm/gguf:/models/gguf", "gguf/org/G-GGUF/G-00001-of-00002.gguf")
    rc = download.run(models, _SettingsP(), token=None, exists=lambda p: True)
    assert rc == 0 and calls == []
