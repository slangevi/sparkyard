import argparse
import os
import shutil

from sparkyard import cli

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _make_checkout(root):
    # minimal discoverable checkout: the marker + a valid models.yaml + settings
    shutil.copy(os.path.join(FIXTURES, "models.yaml"), os.path.join(root, "models.example.yaml"))
    shutil.copy(os.path.join(FIXTURES, "models.yaml"), os.path.join(root, "models.yaml"))
    shutil.copy(os.path.join(FIXTURES, "settings.local.yaml"), os.path.join(root, "settings.local.yaml"))


def test_find_repo_root_from_subdir(tmp_path):
    (tmp_path / "models.example.yaml").write_text("models: []\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert cli._find_repo_root(str(sub)) == str(tmp_path)


def test_find_repo_root_none_when_no_marker(tmp_path):
    sub = tmp_path / "x"
    sub.mkdir()
    assert cli._find_repo_root(str(sub)) is None


def test_validate_resolves_from_subdir(tmp_path, monkeypatch):
    _make_checkout(str(tmp_path))
    sub = tmp_path / "deep"
    sub.mkdir()
    monkeypatch.chdir(sub)
    assert cli.main(["validate"]) == 0


def test_no_checkout_fails_closed(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # no marker at or above
    rc = cli.main(["validate"])
    assert rc == 2
    assert "could not locate a sparkyard checkout" in capsys.readouterr().err


def test_explicit_models_overrides_autodiscovery(tmp_path, monkeypatch):
    _make_checkout(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert cli.main(["--models", str(tmp_path / "models.yaml"),
                     "--settings", os.path.join(FIXTURES, "settings.local.yaml"),
                     "validate"]) == 0


def test_resolve_paths_fills_all_render_outputs(tmp_path, monkeypatch):
    # A render/add-model namespace carries all five path attrs; unset ones
    # resolve against the discovered root (covers the render-output keys).
    (tmp_path / "models.example.yaml").write_text("models: []\n")
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(**{k: None for k in cli._PATH_DEFAULTS})
    assert cli._resolve_paths(args) is None
    for key, default in cli._PATH_DEFAULTS.items():
        assert getattr(args, key) == os.path.join(str(tmp_path), default)


def test_resolve_paths_skips_absent_attrs(tmp_path, monkeypatch):
    # validate/doctor/download/vllm-node namespaces lack the render-output
    # attrs; the hasattr guard must leave them absent, not create them.
    (tmp_path / "models.example.yaml").write_text("models: []\n")
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(models=None, settings=None)
    assert cli._resolve_paths(args) is None
    assert args.models == os.path.join(str(tmp_path), "models.yaml")
    assert not hasattr(args, "llama_swap_out")
    assert not hasattr(args, "env_out")
