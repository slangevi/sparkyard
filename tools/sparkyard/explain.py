"""`sparkyard explain <model>` — show how a model would be launched, without
launching it.

launch.py already computes the adaptive gpu-memory-utilization and prints the
docker argv under `--print`, but reaching it meant extracting the folded `cmd:`
from the generated llama-swap config, substituting ${PORT}/${host} by hand, and
running it inside the llama-swap container. This does that."""
import re
import shlex


class ExplainError(Exception):
    pass


def print_argv(config, name, port="5999", host="0.0.0.0"):
    """Return an argv that dry-runs `name`'s launcher. Pure; no side effects."""
    models = (config or {}).get("models") or {}
    if name not in models:
        raise ExplainError(f"unknown model {name!r}; known: {', '.join(sorted(models))}")
    cmd = (models[name] or {}).get("cmd") or ""
    flat = " ".join(cmd.split())
    if "launch.py" not in flat:
        raise ExplainError(
            f"{name} is not launched via launch.py (llamacpp models exec docker "
            f"directly), so there is no gmem plan to dry-run")
    flat = flat.replace("${PORT}", port).replace("${host}", host)
    return shlex.split(flat) + ["--print"]


_KEEP = re.compile(r"^\[auto-gmem\]|^gpu_memory_utilization=")


def summarise(output):
    """Keep the sizing lines from a launch.py --print run; drop the argv dump."""
    return "\n".join(l for l in output.splitlines() if _KEEP.match(l.strip()))
