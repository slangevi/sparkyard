"""Structural validation of the model list. Returns a list of error strings
(empty = valid). On-disk path/quant checks are deferred to P4."""

import re

import yaml

ENGINES = {"vllm", "llamacpp"}
REQUIRED = {
    "vllm": ["path", "container", "max_model_len", "max_num_seqs", "kv_dtype_bytes"],
    "llamacpp": ["gguf", "mount", "container", "ctx_size"],
}
NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
RESERVED_LITELLM_KEYS = ("model", "api_base", "api_key")


def _is_yaml_special(s):
    """True if a charset-valid string would be re-typed by a YAML parser
    (e.g. 'true'->bool, '1.5'->float, 'null'->None, '0x10'->int). Such a
    name/alias is emitted UNQUOTED into the generated configs and silently
    retyped, breaking model-name routing."""
    try:
        return yaml.safe_load(s) != s
    except yaml.YAMLError:
        return True


def validate(models):
    errors = []
    seen = {}
    for m in models:
        if m.engine not in ENGINES:
            errors.append(f"{m.name}: unknown engine '{m.engine}' (allowed: {sorted(ENGINES)})")
            continue

        for field in REQUIRED.get(m.engine, []):
            if field not in m.raw:
                errors.append(f"{m.name}: missing required field '{field}' for engine {m.engine}")

        for name in m.served_names:
            if name in seen:
                errors.append(
                    f"duplicate served name '{name}' (used by {seen[name]} and {m.name})"
                )
            else:
                seen[name] = m.name

        gmem = m.raw.get("gmem", {})
        if "min" in gmem and "max" in gmem and gmem["min"] > gmem["max"]:
            errors.append(f"{m.name}: gmem.min ({gmem['min']}) > gmem.max ({gmem['max']})")

        # name + aliases charset (a served name becomes a YAML key + a LiteLLM
        # model_name; YAML-special/number-like tokens corrupt or retype it).
        if not isinstance(m.name, str):
            errors.append(f"{m.name!r}: name must be a string matching [A-Za-z0-9._-]+")
        elif not NAME_RE.match(m.name):
            errors.append(f"{m.name}: name must be a string matching [A-Za-z0-9._-]+")
        elif _is_yaml_special(m.name):
            errors.append(f"{m.name}: name is a YAML-special token (parses to "
                          f"{type(yaml.safe_load(m.name)).__name__}) and would be retyped "
                          f"in generated configs — rename it")
        aliases = m.raw.get("aliases")
        if aliases is not None:
            if not isinstance(aliases, list):
                errors.append(f"{m.name}: aliases must be a list (got {type(aliases).__name__})")
            else:
                for a in aliases:
                    if not isinstance(a, str) or not NAME_RE.match(a):
                        errors.append(f"{m.name}: alias {a!r} must be a string matching [A-Za-z0-9._-]+")
                    elif _is_yaml_special(a):
                        errors.append(f"{m.name}: alias {a!r} is a YAML-special token and would be retyped — rename it")

        # list-typed flag fields: a bare string explodes char-by-char downstream.
        for fld in ("vllm_flags", "llamacpp_flags"):
            val = m.raw.get(fld)
            if val is not None and not isinstance(val, list):
                errors.append(f"{m.name}: {fld} must be a list (got {type(val).__name__})")

        # litellm passthrough must not override routing/secret keys.
        lite = m.raw.get("litellm")
        if lite is not None and not isinstance(lite, dict):
            errors.append(f"{m.name}: litellm must be a mapping (got {type(lite).__name__})")
        elif isinstance(lite, dict):
            for key in RESERVED_LITELLM_KEYS:
                if key in lite:
                    errors.append(f"{m.name}: litellm block may not set reserved key '{key}'")

        # gmem.override, if present, must be a fraction in (0, 1).
        ov = m.raw.get("gmem", {}).get("override")
        if ov is not None:
            if not isinstance(ov, (int, float)) or isinstance(ov, bool) or not (0 < ov < 1):
                errors.append(f"{m.name}: gmem.override must be a number in (0, 1) (got {ov!r})")

        repo = m.raw.get("hf_repo")
        if repo is not None:
            if not isinstance(repo, str) or not repo.strip() or repo.count("/") != 1:
                errors.append(f"{m.name}: hf_repo must be a non-empty 'org/model' string (got {repo!r})")

        if m.engine == "vllm":
            if m.gmem_min is None or m.gmem_max is None:
                errors.append(f"{m.name}: vllm model needs gmem.min/max (none set and no engine default)")
            if m.image is None:
                errors.append(f"{m.name}: vllm model needs an image (none set and no engine default)")
            flags = " ".join(m.vllm_flags)
            kvb = m.raw.get("kv_dtype_bytes")
            if kvb not in (1, 2):
                errors.append(f"{m.name}: kv_dtype_bytes must be 1 or 2 (got {kvb!r})")
            if ("--quantization mxfp4" in flags or "--mxfp4-backend" in flags) \
                    and "mxfp4" not in (m.image or ""):
                errors.append(f"{m.name}: uses mxfp4 flags but image '{m.image}' is not an mxfp4 image")
            kd = re.search(r"--kv-cache-dtype[\s=]+(\S+)", flags)
            if kd:
                dtype = kd.group(1)
                is_fp8 = dtype.startswith("fp8")  # fp8, fp8_e5m2, fp8_e4m3 are all 1-byte
                if is_fp8 and kvb != 1:
                    errors.append(f"{m.name}: --kv-cache-dtype {dtype} implies kv_dtype_bytes: 1 (got {kvb})")
                elif not is_fp8 and kvb == 1:
                    errors.append(f"{m.name}: --kv-cache-dtype {dtype} implies kv_dtype_bytes: 2 (got 1)")
        if m.engine == "llamacpp" and m.image is None:
            errors.append(f"{m.name}: llamacpp model needs an image")
        for fld in ("extra_docker_args", "pre_launch_cmd"):
            val = m.raw.get(fld)
            if isinstance(val, str) and "'" in val:
                errors.append(f"{m.name}: {fld} contains a single quote, which breaks shell quoting")

    return errors
