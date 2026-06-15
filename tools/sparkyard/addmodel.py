"""add-model wizard: introspect a HF repo, propose a models.yaml entry, append it
(text + PyYAML guard — no ruamel), render, and optionally download the weights."""
import os
import sys
import yaml

from .introspect import (fetch_repo_metadata, derive_entry, derive_gguf_entry,
                         is_gguf_repo, IntrospectError)
from .render import load, RenderError, render_all, atomic_write
from .settings import Settings
from . import download


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


def _finish(args, entry, hints, *, input_fn):
    """Shared tail: print proposal, confirm, append, render, optional download."""
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
        rc = download.run(models, settings, _hf_token(args.settings), only=entry["name"])
        if rc != 0:
            return rc
        print("Reload with: docker compose up -d llama-swap litellm")
    else:
        print("\nNext: fetch weights + reload (or re-run with --download):")
        print(f"  make add-model HF_REPO={args.repo} ADDARGS=--download")
        print("  # then: docker compose up -d llama-swap litellm")
    return 0


def _select_family(args, families, *, input_fn, isatty):
    """Return the chosen family label, or None on a (printed) error/cancel."""
    labels = sorted(families)

    def _list(dest, rows):
        for label in rows:
            print(f"    {label} ({len(families[label])} file(s))", file=dest)

    if args.gguf_file:
        matches = [label for label in labels if args.gguf_file.lower() in label.lower()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            print(f"✗ no GGUF quant matches '{args.gguf_file}'. Available:", file=sys.stderr)
            _list(sys.stderr, labels)
        else:
            print(f"✗ '{args.gguf_file}' matches {len(matches)} quants — be more specific:",
                  file=sys.stderr)
            _list(sys.stderr, matches)
        return None

    tty = isatty if isatty is not None else sys.stdin.isatty
    if not tty():
        print("✗ multiple GGUF quants available — pass --gguf-file <pattern>:", file=sys.stderr)
        _list(sys.stderr, labels)
        return None
    print("Available GGUF quants:")
    for i, label in enumerate(labels, 1):
        print(f"  {i}. {label} ({len(families[label])} file(s))")
    resp = input_fn("Pick a quant [number]: ").strip()
    if not resp.isdigit() or not (1 <= int(resp) <= len(labels)):
        print("✗ invalid selection.", file=sys.stderr)
        return None
    return labels[int(resp) - 1]


def _run_gguf(args, files, config, *, input_fn, isatty):
    families = download.gguf_families(files)
    chosen = _select_family(args, families, input_fn=input_fn, isatty=isatty)
    if chosen is None:
        return 2
    first_shard = families[chosen][0]
    entry, hints = derive_gguf_entry(args.repo, first_shard, config, name=args.name)
    return _finish(args, entry, hints, input_fn=input_fn)


def run(args, *, input_fn=input, isatty=None):
    try:
        config, files = fetch_repo_metadata(args.repo)
    except IntrospectError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    if is_gguf_repo(files):
        return _run_gguf(args, files, config, input_fn=input_fn, isatty=isatty)

    if config is None:
        print(f"✗ '{args.repo}': no config.json and no .gguf files — can't introspect.",
              file=sys.stderr)
        return 1

    entry, hints, _ = derive_entry(args.repo, config, files, name=args.name)
    return _finish(args, entry, hints, input_fn=input_fn)
