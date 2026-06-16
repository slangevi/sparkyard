"""Sparkyard generator CLI: `validate`, `render`, `doctor`, and `add-model` subcommands."""
import argparse
import os
import sys
import yaml

from .render import load, RenderError, render_all
from . import doctor as doctor_mod
from . import addmodel


MARKER = "models.example.yaml"
_PATH_DEFAULTS = {
    "models": "models.yaml",
    "settings": "settings.local.yaml",
    "llama_swap_out": "llama-swap/config.yaml",
    "litellm_out": "LiteLLM/config.yaml",
    "env_out": ".env",
}


def _find_repo_root(start=None):
    """Walk up from `start` (or cwd) to the dir containing MARKER; None if not found."""
    d = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.isfile(os.path.join(d, MARKER)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _resolve_paths(args):
    """Fill any unset (None) path arg with <repo_root>/<default>. Returns an
    error string if a default is needed but no checkout is found, else None."""
    need = [a for a in _PATH_DEFAULTS if hasattr(args, a) and getattr(args, a) is None]
    if not need:
        return None
    root = _find_repo_root()
    if root is None:
        return (f"could not locate a sparkyard checkout (no {MARKER} at or above "
                f"{os.getcwd()}); pass --models/--settings explicitly")
    for a in need:
        setattr(args, a, os.path.join(root, _PATH_DEFAULTS[a]))
    return None


def _add_render_outputs(p):
    p.add_argument("--llama-swap-out", default=None)
    p.add_argument("--litellm-out", default=None)
    p.add_argument("--env-out", default=None)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkyard")
    parser.add_argument("--models", default=None)
    parser.add_argument("--settings", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate", help="validate models.yaml + settings")
    sub.add_parser("doctor", help="advisory on-disk report (never blocks render)")
    r = sub.add_parser("render", help="render all live config files")
    # P3 cutover: the generated env IS the live compose .env now.
    _add_render_outputs(r)

    am = sub.add_parser("add-model", help="introspect a HF repo and add it to models.yaml")
    am.add_argument("repo")
    am.add_argument("--name", default=None)
    am.add_argument("--dry-run", action="store_true")
    am.add_argument("--yes", action="store_true")
    am.add_argument("--download", action="store_true")
    am.add_argument("--gguf-file", default=None,
                    help="quant pattern to select from a GGUF repo (substring match)")
    _add_render_outputs(am)

    dl = sub.add_parser("download", help="fetch HF weights for SSOT entries with hf_repo")
    dl.add_argument("--model", default=None, help="download only this model (by name); omit for all")

    vn = sub.add_parser("vllm-node", help="clone + build the vllm-node serving image(s)")
    vn.add_argument("--variant", choices=["base", "tf5", "mxfp4"], default=None,
                    help="single variant to build; default builds base + tf5")
    vn.add_argument("--vllm-ref", default=None, help="override the settings vllm_ref pin")
    vn.add_argument("--print", dest="dry_run", action="store_true",
                    help="print the resolved build plan and exit (no side effects)")

    args = parser.parse_args(argv)

    err = _resolve_paths(args)
    if err:
        print(f"✗ {err}", file=sys.stderr)
        return 2

    if args.cmd == "add-model":
        return addmodel.run(args)

    if args.cmd == "vllm-node":
        from . import vllm_node
        from .settings import Settings
        try:
            settings = Settings.load(args.settings)
        except (OSError, KeyError, yaml.YAMLError) as e:
            print(f"✗ {e}", file=sys.stderr)
            return 1
        return vllm_node.run(args, settings)

    try:
        settings, models = load(args.models, args.settings)
    except RenderError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    if args.cmd == "validate":
        print(f"✓ {len(models)} models valid")
        return 0

    if args.cmd == "doctor":
        lines, summary = doctor_mod.check(models, settings)
        for ln in lines:
            print(ln)
        print(summary)
        return 0

    if args.cmd == "render":
        render_all(settings, models, args.llama_swap_out, args.litellm_out, args.env_out)
        print(f"✓ rendered {len(models)} models -> "
              f"{args.llama_swap_out}, {args.litellm_out}, {args.env_out}")
        return 0

    if args.cmd == "download":
        from . import download
        token = addmodel._hf_token(args.settings)
        return download.run(models, settings, token, only=args.model)


if __name__ == "__main__":
    sys.exit(main())
