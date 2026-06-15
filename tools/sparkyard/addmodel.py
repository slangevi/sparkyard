"""add-model wizard: introspect a HF repo, propose a models.yaml entry, append it
(text + PyYAML guard — no ruamel), render, and optionally download the weights."""
import os
import sys
import yaml

from .introspect import fetch_repo_metadata, derive_entry, IntrospectError
from .render import load, RenderError, render_all, atomic_write
from .settings import Settings


class AppendError(Exception):
    pass


def _hf_token(settings_path):
    """Read HF_TOKEN from env, else from secrets.env beside settings_path. None if blank/absent."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    secrets = os.path.join(os.path.dirname(os.path.abspath(settings_path)), "secrets.env")
    if os.path.exists(secrets):
        for line in open(secrets):
            if line.startswith("HF_TOKEN="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                return val or None
    return None


def entry_to_yaml(entry):
    """One model entry as a 2-space-indented YAML list item (PyYAML quotes as needed)."""
    block = yaml.safe_dump([entry], sort_keys=False, default_flow_style=False)
    return "".join("  " + line + "\n" for line in block.splitlines())


def append_model(models_path, entry):
    """Append `entry` at EOF of models.yaml (where `models:` is the last block).
    Fail-closed: refuses unless `models` is the last top-level key, and verifies the
    result parses with exactly one more model before writing."""
    with open(models_path) as f:
        text = f.read()
    data = yaml.safe_load(text)
    keys = list(data.keys()) if isinstance(data, dict) else []
    if not keys or keys[-1] != "models" or not isinstance(data.get("models"), list):
        raise AppendError(
            "models.yaml: `models` must be the last top-level key and a list to auto-append")
    new_text = (text if text.endswith("\n") else text + "\n") + entry_to_yaml(entry)
    try:
        new_data = yaml.safe_load(new_text)
    except yaml.YAMLError as e:
        raise AppendError(f"appended entry would make models.yaml unparseable: {e}")
    if not isinstance(new_data, dict) or len(new_data.get("models", [])) != len(data["models"]) + 1:
        raise AppendError("append produced an unexpected model count — aborting")
    atomic_write(models_path, new_text)


def _download(repo, path, settings_path):
    from . import download
    settings = Settings.load(settings_path)
    local = os.path.join(settings.llm_root, path)
    print(f"Downloading {repo} -> {local} ...")
    download.snapshot(repo, local, _hf_token(settings_path))
    print("Download complete. Reload with: docker compose up -d llama-swap litellm")


def run(args, *, input_fn=input):
    try:
        config, files = fetch_repo_metadata(args.repo)
    except IntrospectError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1
    entry, hints, is_gguf = derive_entry(args.repo, config, files, name=args.name)
    if is_gguf:
        print(f"✗ '{args.repo}' is a GGUF repo. The add-model wizard supports vLLM/safetensors "
              f"models only — add GGUF models to models.yaml by hand.", file=sys.stderr)
        return 2

    print("Proposed models.yaml entry:\n")
    print(entry_to_yaml(entry), end="")
    for h in hints:
        print(f"  hint: {h}")
    if args.dry_run:
        print("\n(dry-run: nothing written)")
        return 0
    if not args.yes:
        resp = input_fn("\nAppend this entry to models.yaml? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("cancelled.")
            return 0

    try:
        append_model(args.models, entry)
    except (AppendError, OSError) as e:
        print(f"✗ {e}\n\nAdd this entry to models.yaml by hand:\n\n{entry_to_yaml(entry)}",
              file=sys.stderr)
        return 1

    try:
        settings, models = load(args.models, args.settings)
        render_all(settings, models, args.llama_swap_out, args.litellm_out, args.env_out)
    except RenderError as e:
        print(f"✗ entry appended, but render failed (fix models.yaml then `make render`): {e}",
              file=sys.stderr)
        return 1

    print(f"✓ added '{entry['name']}' and rendered {len(models)} models.")
    if args.download:
        _download(args.repo, entry["path"], args.settings)
    else:
        print("\nNext: fetch weights + reload (or re-run with --download):")
        print(f"  make add-model HF_REPO={args.repo} ADDARGS=--download")
        print("  # then: docker compose up -d llama-swap litellm")
    return 0
