"""Sparkyard generator CLI (Click): validate/render/doctor/add-model/… subcommands.

Thin Click front-end over the existing dispatch logic. The argparse parsing
layer was replaced by Click for nicer help; everything that carries the
exit-code contract (_find_repo_root / _resolve_paths / _PATH_DEFAULTS / the
RenderError + settings error handling / every downstream .run() signature) is
unchanged. `main(argv) -> int` is preserved so the console-script wrapper
(`sys.exit(main())`), make recipes, and the existing tests keep their contract.
"""
import argparse
import os
import sys

import click
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


def _dispatch(args):
    """Per-command logic, preserved verbatim from the argparse implementation.
    Returns an int exit code; surfaces application errors as return values."""
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

    if args.cmd == "update":
        from . import update
        from .settings import Settings
        root = _find_repo_root()
        if root is None:
            print(f"✗ could not locate a sparkyard checkout (no {MARKER})", file=sys.stderr)
            return 2
        try:
            settings = Settings.load(args.settings)
        except (OSError, KeyError, yaml.YAMLError) as e:
            print(f"✗ {e}", file=sys.stderr)
            return 1
        return update.run(root, settings, check=args.check, notes=args.notes,
                          model=args.model, components=args.components)

    if args.cmd in ("init", "secrets", "build", "start", "stop", "bench"):
        from . import ops
        root = _find_repo_root()
        if root is None:
            print(f"✗ could not locate a sparkyard checkout (no {MARKER})", file=sys.stderr)
            return 2
        if args.cmd == "bench":
            return ops.bench(root, args.mode, args.base_url)
        return getattr(ops, args.cmd)(root)

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


def _ns(obj, cmd, **kw):
    """Build the argparse.Namespace that _dispatch + downstream .run() expect.
    `obj` is the group's ctx.obj dict carrying --models/--settings."""
    return argparse.Namespace(cmd=cmd, models=obj["models"], settings=obj["settings"], **kw)


class OrderedGroup(click.Group):
    """A group whose --help lists commands in registration (lifecycle) order
    instead of Click's default alphabetical order."""

    def list_commands(self, ctx):
        return list(self.commands)


@click.group(cls=OrderedGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--models", metavar="PATH", default=None,
              help="Path to models.yaml (default: autodiscovered from the checkout).")
@click.option("--settings", metavar="PATH", default=None,
              help="Path to settings.local.yaml (default: autodiscovered from the checkout).")
@click.pass_context
def cli(ctx, models, settings):
    """SSOT-driven multi-engine LLM stack generator for the NVIDIA DGX Spark."""
    ctx.obj = {"models": models, "settings": settings}


@cli.command()
@click.pass_obj
def init(obj):
    """Seed settings.local.yaml, models.yaml, and secrets for a first run."""
    return _dispatch(_ns(obj, "init"))


@cli.command()
@click.pass_obj
def secrets(obj):
    """Scaffold + generate secrets.env."""
    return _dispatch(_ns(obj, "secrets"))


@cli.command()
@click.option("--llama-swap-out", metavar="PATH", default=None,
              help="Override the llama-swap config output path (default: autodiscovered).")
@click.option("--litellm-out", metavar="PATH", default=None,
              help="Override the LiteLLM config output path (default: autodiscovered).")
@click.option("--env-out", metavar="PATH", default=None,
              help="Override the compose .env output path (default: autodiscovered).")
@click.pass_obj
def render(obj, llama_swap_out, litellm_out, env_out):
    """Render all live config files from the SSOT (fail-closed on invalid models.yaml)."""
    return _dispatch(_ns(obj, "render", llama_swap_out=llama_swap_out,
                         litellm_out=litellm_out, env_out=env_out))


@cli.command()
@click.pass_obj
def validate(obj):
    """Validate models.yaml + settings (fail-closed)."""
    return _dispatch(_ns(obj, "validate"))


@cli.command()
@click.pass_obj
def doctor(obj):
    """Advisory on-disk model report (never blocks render)."""
    return _dispatch(_ns(obj, "doctor"))


@cli.command()
@click.pass_obj
def build(obj):
    """Build the local llama-cpp + llama-swap images (docker compose build)."""
    return _dispatch(_ns(obj, "build"))


@cli.command()
@click.pass_obj
def start(obj):
    """Start the stack (docker compose up -d)."""
    return _dispatch(_ns(obj, "start"))


@cli.command()
@click.pass_obj
def stop(obj):
    """Stop the stack (docker compose down)."""
    return _dispatch(_ns(obj, "stop"))


@cli.command("add-model")
@click.argument("repo")
@click.option("--name", default=None, help="Override the inferred model name.")
@click.option("--dry-run", is_flag=True,
              help="Print the derived entry without writing models.yaml.")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
@click.option("--download", "download_", is_flag=True,
              help="Download the weights after adding the entry.")
@click.option("--gguf-file", default=None,
              help="Quant pattern to select from a GGUF repo (substring match).")
@click.option("--llama-swap-out", metavar="PATH", default=None,
              help="Override the llama-swap config output path (default: autodiscovered).")
@click.option("--litellm-out", metavar="PATH", default=None,
              help="Override the LiteLLM config output path (default: autodiscovered).")
@click.option("--env-out", metavar="PATH", default=None,
              help="Override the compose .env output path (default: autodiscovered).")
@click.pass_obj
def add_model(obj, repo, name, dry_run, yes, download_, gguf_file,
              llama_swap_out, litellm_out, env_out):
    """Introspect a Hugging Face repo and append it to models.yaml."""
    return _dispatch(_ns(obj, "add-model", repo=repo, name=name, dry_run=dry_run,
                         yes=yes, download=download_, gguf_file=gguf_file,
                         llama_swap_out=llama_swap_out, litellm_out=litellm_out,
                         env_out=env_out))


@cli.command()
@click.option("--model", default=None,
              help="Download only this model (by name); omit for all.")
@click.pass_obj
def download(obj, model):
    """Fetch Hugging Face weights for SSOT entries that carry an hf_repo."""
    return _dispatch(_ns(obj, "download", model=model))


@cli.command("vllm-node")
@click.option("--variant", type=click.Choice(["base", "tf5", "mxfp4"]), default=None,
              help="Single variant to build; default builds base + tf5.")
@click.option("--vllm-ref", default=None, help="Override the settings vllm_ref pin.")
@click.option("--print", "dry_run", is_flag=True,
              help="Print the resolved build plan and exit (no side effects).")
@click.pass_obj
def vllm_node(obj, variant, vllm_ref, dry_run):
    """Clone + build the vllm-node serving image(s)."""
    return _dispatch(_ns(obj, "vllm-node", variant=variant, vllm_ref=vllm_ref,
                         dry_run=dry_run))


@cli.command()
@click.argument("components", nargs=-1)
@click.option("--check", is_flag=True,
              help="Dry run: report only; make no changes and no pull/build.")
@click.option("--notes", is_flag=True,
              help="Summarize what the updates provide (via the local gateway; raw notes if it's down).")
@click.option("--model", default=None,
              help="Gateway model for the --notes summary (default: first from /v1/models).")
@click.pass_obj
def update(obj, components, check, notes, model):
    """Check for + apply upstream component updates (never floats pins).

    With no COMPONENT args, checks/updates everything. Name one or more to scope
    the run: ollama, litellm, litellm-db, open-webui, llama-swap, llama-cpp,
    vllm-node. Source-built components (llama-cpp, vllm-node) rebuild only when
    named explicitly; a bare `update` reports them without building.
    """
    return _dispatch(_ns(obj, "update", components=components, check=check,
                         notes=notes, model=model))


@cli.command()
@click.option("--mode", type=click.Choice(["quality", "speed"]), default=None,
              help="Benchmark mode (quality or speed).")
@click.option("--base-url", default=None, help="Override the gateway base URL.")
@click.pass_obj
def bench(obj, mode, base_url):
    """Benchmark the served models."""
    return _dispatch(_ns(obj, "bench", mode=mode, base_url=base_url))


def main(argv=None):
    """Run the Click group and translate its outcome to the argparse-era exit
    codes (0 ok / 1 app error / 2 usage|no-checkout). Preserved so the console
    script wrapper `sys.exit(main())`, make recipes, and existing tests keep
    their contract."""
    try:
        return cli.main(args=argv, prog_name="sparkyard", standalone_mode=False) or 0
    except click.UsageError as e:        # bad/missing option, unknown choice/command
        e.show()                          # argparse did sys.exit(2) on parse errors...
        raise SystemExit(e.exit_code)     # ...so re-raise SystemExit, not return it
    except click.ClickException as e:
        e.show()
        return e.exit_code
    except click.Abort:
        return 1


if __name__ == "__main__":
    sys.exit(main())
