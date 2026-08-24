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
    """Return (settings, models, groups) with placeholders resolved; raises
    RenderError on any problem. `groups` is the optional llama-swap routing
    group map from models.yaml ({} when the key is absent)."""
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
    groups = raw.get("groups") or {}
    errors = validate(models, groups)
    if errors:
        raise RenderError("invalid models.yaml:\n  - " + "\n  - ".join(errors))
    return settings, models, groups


# llama-swap's global healthCheckTimeout is the ONLY ready-wait control it has
# (v251 has no `readyTimeout` key at either level), so it must cover the slowest
# cold load in the set or that model can never start: llama-swap kills it with
# "health check timed out" and the operator sees a model that simply never loads.
HEALTH_CHECK_FLOOR = 120        # llama-swap's own default
REQUEST_TIMEOUT_HEADROOM = 300  # load time + room to actually generate


def ready_ceiling(models):
    """The slowest cold load across the model set, in seconds."""
    return max([m.ready_timeout for m in models], default=HEALTH_CHECK_FLOOR)


def render_llama_swap(models, groups=None):
    return _env().get_template("llama-swap.config.yaml.j2").render(
        models=models, groups=groups or {},
        health_check_timeout=max(ready_ceiling(models), HEALTH_CHECK_FLOOR))


def render_litellm(models):
    return _env().get_template("litellm.config.yaml.j2").render(
        models=models,
        request_timeout=ready_ceiling(models) + REQUEST_TIMEOUT_HEADROOM)


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


def render_all(settings, models, ls_out, ll_out, env_out, groups=None):
    """Render + atomically write all three live config files from loaded objects."""
    atomic_write(ls_out, render_llama_swap(models, groups))
    atomic_write(ll_out, render_litellm(models))
    atomic_write(env_out, render_compose_env(settings))
