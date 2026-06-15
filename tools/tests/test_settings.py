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
