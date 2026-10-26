import os
from sparkyard.settings import Settings, DEFAULT_VLLM_REF

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_loads_fields():
    s = Settings.load(os.path.join(FIXTURES, "settings.local.yaml"))
    assert s.llm_root == "/data/LLMs"
    assert s.repo_path == "/repo"
    assert s.home == "/home/acme"


def test_placeholder_map_has_all_keys():
    s = Settings.load(os.path.join(FIXTURES, "settings.local.yaml"))
    m = s.placeholder_map()
    assert m["llm_root"] == "/data/LLMs"
    assert m["repo_path"] == "/repo"
    assert m["home"] == "/home/acme"


def test_home_is_optional():
    import tempfile, textwrap
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(textwrap.dedent("""
            llm_root: /x
            repo_path: /y
        """))
        path = f.name
    s = Settings.load(path)
    assert s.home == ""


def test_vllm_defaults_apply_when_block_absent():
    s = Settings.load(os.path.join(FIXTURES, "settings.local.yaml"))
    assert s.vllm.upstream == "https://github.com/eugr/spark-vllm-docker"
    assert s.vllm.clone_path == "/repo/vllm/build/spark-vllm-docker"  # {repo_path} resolved
    assert s.vllm.vllm_ref == DEFAULT_VLLM_REF


def test_vllm_block_overrides_and_resolves_clone_path():
    import tempfile, textwrap
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(textwrap.dedent("""
            llm_root: /x
            repo_path: /custom/repo
            vllm:
              upstream: https://example.com/fork.git
              clone_path: "{repo_path}/build/clone"
              vllm_ref: deadbeef
        """))
        path = f.name
    s = Settings.load(path)
    assert s.vllm.upstream == "https://example.com/fork.git"
    assert s.vllm.clone_path == "/custom/repo/build/clone"
    assert s.vllm.vllm_ref == "deadbeef"


def test_vllm_partial_block_fills_remaining_defaults():
    import tempfile, textwrap
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(textwrap.dedent("""
            llm_root: /x
            repo_path: /repo
            vllm:
              upstream: https://example.com/fork.git
        """))
        path = f.name
    s = Settings.load(path)
    assert s.vllm.upstream == "https://example.com/fork.git"   # overridden
    assert s.vllm.clone_path == "/repo/vllm/build/spark-vllm-docker"  # default, {repo_path} resolved
    assert s.vllm.vllm_ref == DEFAULT_VLLM_REF                 # default


def test_cluster_absent_is_none():
    s = Settings.load(os.path.join(FIXTURES, "settings.local.yaml"))
    assert s.cluster is None


def test_cluster_parsed():
    s = Settings.load(os.path.join(FIXTURES, "settings.cluster.yaml"))
    assert s.cluster.head_cx7_ip == "192.168.100.10"
    assert s.cluster.eth_if == "enp1s0f0np0"
    assert s.cluster.worker.cx7_ip == "192.168.100.11"
    assert s.cluster.worker.name == "spark-2"
    assert s.cluster.worker.ssh_host == "10.0.0.2"
    assert s.cluster.worker.ssh_user == "acme"
    assert s.cluster.master_port == 29501
    assert s.cluster.ib_hca == "rocep1s0f0,roceP2p1s0f0"
    assert s.cluster_dir == "/repo/llama-swap/cluster"


def test_cluster_missing_key_raises(tmp_path):
    import pytest, textwrap
    path = tmp_path / "s.yaml"
    path.write_text(textwrap.dedent("""
        llm_root: /x
        repo_path: /y
        cluster:
          head_cx7_ip: 192.168.100.10
          ib_hca: rocep1s0f0
          workers:
            - {name: spark-2, ssh_host: 10.0.0.2, ssh_user: acme, cx7_ip: 192.168.100.11}
    """))
    with pytest.raises(KeyError):
        Settings.load(str(path))
