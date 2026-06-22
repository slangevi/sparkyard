"""Persist a freshly-built vLLM ref across the four places sparkyard records it:
settings.local.yaml, settings.py DEFAULT_VLLM_REF, VLLM_NODE_PROVENANCE.md, and
(read-only) the clone's build artifacts. All functions are pure given injected
readers; string-based rewrites are comment/format preserving. Mirrors update.py."""
import os
import re
from collections import namedtuple

from .update import UpdateError

BuiltRefs = namedtuple("BuiltRefs", "vllm flashinfer wheel built_date")

_WHEEL_DATE = re.compile(r"\.d(\d{4})(\d{2})(\d{2})")


def read_built_refs(clone_path, read_text, listdir):
    """Read the refs/wheel a vllm-node build recorded in <clone>/wheels. Pure
    given read_text/listdir. Missing files degrade to empty strings."""
    wheels = os.path.join(clone_path, "wheels")

    def _read(name):
        try:
            return read_text(os.path.join(wheels, name)).strip()
        except (OSError, KeyError):
            return ""

    try:
        entries = listdir(wheels)
    except (OSError, KeyError):
        entries = []
    wheel = next((e for e in sorted(entries) if e.startswith("vllm-") and e.endswith(".whl")), "")
    m = _WHEEL_DATE.search(wheel)
    built_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""
    return BuiltRefs(_read(".vllm-commit"), _read(".flashinfer-commit"), wheel, built_date)


_DEFAULT_REF = re.compile(r'(DEFAULT_VLLM_REF\s*=\s*")[^"]*(")')


def rewrite_default_ref(settings_py_text, sha):
    text, n = _DEFAULT_REF.subn(rf'\g<1>{sha}\g<2>', settings_py_text, count=1)
    if n != 1:
        raise UpdateError('DEFAULT_VLLM_REF = "…" not found in settings.py')
    return text


_VLLM_BLOCK = re.compile(r"^vllm:\s*$", re.M)
_VLLM_REF_LINE = re.compile(r"^(\s*)vllm_ref:\s*\S+(.*)$", re.M)


def upsert_settings_local_ref(yaml_text, sha):
    """Set vllm.vllm_ref in settings.local.yaml text. String-based so comments
    survive. Three cases: key present (replace), vllm: block present but no key
    (insert key), no block (append a block)."""
    if _VLLM_REF_LINE.search(yaml_text):
        return _VLLM_REF_LINE.sub(rf"\g<1>vllm_ref: {sha}\g<2>", yaml_text, count=1)
    if _VLLM_BLOCK.search(yaml_text):
        # insert the key as the first child of the existing vllm: block (2-space indent)
        return _VLLM_BLOCK.sub(f"vllm:\n  vllm_ref: {sha}", yaml_text, count=1)
    sep = "" if yaml_text.endswith("\n") or yaml_text == "" else "\n"
    return f"{yaml_text}{sep}vllm:\n  vllm_ref: {sha}\n"


_PROV_DATE = re.compile(r"\(built \d{4}-\d{2}-\d{2}\)")
_PROV_VLLM = re.compile(r"(\|\s*vLLM\s*\|\s*`)[^`]*(`\s*\|\s*`)[^`]*(`\s*\|)")
_PROV_FLASH = re.compile(r"(\|\s*FlashInfer\s*\|\s*`)[^`]*(`)")
_PROV_VLLM_REF_ARG = re.compile(r"--vllm-ref \S+")
_PROV_DEFAULT_REF = re.compile(r"default `[^`]*`")


def rewrite_provenance(text, built):
    """Update the vLLM/FlashInfer rows + the 'built YYYY-MM-DD' heading from a
    BuiltRefs. Raises if the table anchors are missing. Best-effort: also
    rewrites --vllm-ref and default `ref` tokens in the Reproduce prose."""
    out, nd = _PROV_DATE.subn(f"(built {built.built_date})", text, count=1)
    out, nv = _PROV_VLLM.subn(rf"\g<1>{built.vllm}\g<2>{built.wheel}\g<3>", out, count=1)
    out, nf = _PROV_FLASH.subn(rf"\g<1>{built.flashinfer}\g<2>", out, count=1)
    if not (nd and nv and nf):
        raise UpdateError("VLLM_NODE_PROVENANCE.md anchors not found for rewrite")
    # Best-effort: update reproduce-section refs (do NOT raise if absent).
    out = _PROV_VLLM_REF_ARG.sub(f"--vllm-ref {built.vllm}", out)
    out = _PROV_DEFAULT_REF.sub(f"default `{built.vllm}`", out)
    return out
