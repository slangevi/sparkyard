"""Resolve {token} placeholders in strings/lists/dicts using a mapping.

Shell-style ${VAR} (e.g. llama-swap's ${PORT}/${host}/${tensor_parallel}
macros) are intentionally left untouched via a negative lookbehind for '$'.
"""
import re

_TOKEN = re.compile(r"(?<!\$)\{(\w+)\}")


def resolve(value, mapping):
    """Recursively replace {key} tokens. Unknown keys raise KeyError."""
    if isinstance(value, str):
        def repl(match):
            key = match.group(1)
            if key not in mapping:
                raise KeyError(f"unknown placeholder {{{key}}}")
            return str(mapping[key])
        return _TOKEN.sub(repl, value)
    if isinstance(value, list):
        return [resolve(item, mapping) for item in value]
    if isinstance(value, dict):
        return {k: resolve(v, mapping) for k, v in value.items()}
    return value
