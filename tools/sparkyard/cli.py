"""Sparkyard generator CLI: `validate`, `render`, `doctor`, and `add-model` subcommands."""
import argparse
import sys

from .render import load, RenderError, render_all
from . import doctor as doctor_mod
from . import addmodel


def _add_render_outputs(p):
    p.add_argument("--llama-swap-out", default="llama-swap/config.yaml")
    p.add_argument("--litellm-out", default="LiteLLM/config.yaml")
    p.add_argument("--env-out", default=".env")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkyard")
    parser.add_argument("--models", default="models.yaml")
    parser.add_argument("--settings", default="settings.local.yaml")
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
    _add_render_outputs(am)

    dl = sub.add_parser("download", help="fetch HF weights for SSOT entries with hf_repo")
    dl.add_argument("--model", default=None, help="download only this model (by name); omit for all")

    args = parser.parse_args(argv)

    if args.cmd == "add-model":
        return addmodel.run(args)

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
