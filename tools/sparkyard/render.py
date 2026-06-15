"""Load settings + models.yaml, validate, resolve placeholders, render templates.

Writes are atomic (temp file in the same dir, then os.replace). Validation
errors raise RenderError before any file is written (fail closed)."""
import os
import tempfile
import yaml
from jinja2 import Environment, FileSystemLoader

from .settings import Settings
from .model import load_models
from .validate import validate
from .placeholders import resolve

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


class RenderError(Exception):
    pass


def _env():
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )


def load(models_path, settings_path):
    """Return (settings, models) with placeholders resolved; raises RenderError on any problem."""
    try:
        settings = Settings.load(settings_path)
    except FileNotFoundError:
        raise RenderError(f"settings file not found: {settings_path}")
    except KeyError as e:
        raise RenderError(f"settings file missing required key: {e}")
    try:
        with open(models_path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise RenderError(f"models file not found: {models_path}")
    except yaml.YAMLError as e:
        raise RenderError(f"models.yaml is not valid YAML: {e}")
    try:
        raw = resolve(raw, settings.placeholder_map())
        models = load_models(raw)
    except KeyError as e:
        raise RenderError(f"models.yaml problem (missing key or unknown placeholder): {e}")
    errors = validate(models)
    if errors:
        raise RenderError("invalid models.yaml:\n  - " + "\n  - ".join(errors))
    return settings, models


def render_llama_swap(models):
    return _env().get_template("llama-swap.config.yaml.j2").render(models=models)


def render_litellm(models):
    return _env().get_template("litellm.config.yaml.j2").render(models=models)


def render_compose_env(settings):
    return _env().get_template("compose-env.j2").render(settings=settings)


def atomic_write(path, content):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def render_all(settings, models, ls_out, ll_out, env_out):
    """Render + atomically write all three live config files from loaded objects."""
    atomic_write(ls_out, render_llama_swap(models))
    atomic_write(ll_out, render_litellm(models))
    atomic_write(env_out, render_compose_env(settings))
