import os
from sparkyard.settings import Settings

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
    assert s.vllm.vllm_ref == "7852e50e4"


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
    assert s.vllm.vllm_ref == "7852e50e4"                      # default
