"""Structural validation of the model list. Returns a list of error strings
(empty = valid). On-disk path/quant checks are deferred to P4."""

import ipaddress
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


CLUSTER_RESERVED_FLAGS = (
    "-tp", "--tensor-parallel-size", "-pp", "--pipeline-parallel-size",
    "--distributed-executor-backend", "--nnodes", "--node-rank",
    "--master-addr", "--master-port", "--headless",
)


def _is_ip(v):
    try:
        ipaddress.ip_address(v)
        return True
    except (ValueError, TypeError):
        return False


def _nonempty_str(v):
    return isinstance(v, str) and bool(v.strip())


# eth_if / ib_hca items / ssh_host / ssh_user reach ssh argv, a remote shell
# command and `-e K=V` docker args: no whitespace, quotes, or leading '-'.
CLUSTER_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")
_TOKEN_HINT = "letters, digits, '.', '_', ':' or '-', not starting with '-'"


def _bad_token(v):
    return CLUSTER_TOKEN_RE.fullmatch(v) is None


def validate_cluster(cluster):
    """Check the settings `cluster:` block. Exactly one worker is supported."""
    errors = []
    if not _is_ip(cluster.head_cx7_ip):
        errors.append(f"cluster: head_cx7_ip {cluster.head_cx7_ip!r} is not a valid IP address")
    port = cluster.master_port
    if not isinstance(port, int) or isinstance(port, bool) or not (1 <= port <= 65535):
        errors.append(f"cluster: master_port must be an integer in 1..65535 (got {port!r})")
    for fld in ("eth_if", "ib_hca"):
        val = getattr(cluster, fld)
        if not _nonempty_str(val):
            errors.append(f"cluster: {fld} must be a non-empty string")
        elif any(_bad_token(x) for x in (val.split(",") if fld == "ib_hca" else [val])):
            what = "comma-separated HCA names, each" if fld == "ib_hca" else "it"
            errors.append(f"cluster: {fld} {val!r} is invalid — {what} must be {_TOKEN_HINT}")
    if len(cluster.workers) != 1:
        errors.append(f"cluster: exactly one worker is supported (got {len(cluster.workers)})")
        return errors
    w = cluster.workers[0]
    if not _is_ip(w.cx7_ip):
        errors.append(f"cluster: worker cx7_ip {w.cx7_ip!r} is not a valid IP address")
    for fld in ("ssh_host", "ssh_user"):
        val = getattr(w, fld)
        if not _nonempty_str(val):
            errors.append(f"cluster: worker {fld} must be a non-empty string")
        elif _bad_token(val):
            errors.append(f"cluster: worker {fld} {val!r} is invalid — must be {_TOKEN_HINT}")
    return errors


GROUP_FLAGS = ("swap", "exclusive", "persistent")


def _validate_groups(groups, model_names):
    """Check a llama-swap `groups` mapping against the model list. llama-swap
    silently ignores a group whose member is not a model ID, which presents as
    "my model never stays loaded" — so membership is checked fail-closed here."""
    errors = []
    if not isinstance(groups, dict):
        errors.append(f"groups must be a mapping (got {type(groups).__name__})")
        return errors
    owner = {}
    for gname, g in groups.items():
        if not isinstance(gname, str) or not NAME_RE.match(gname):
            errors.append(f"group name {gname!r} must be a string matching [A-Za-z0-9._-]+")
            continue
        if not isinstance(g, dict):
            errors.append(f"group '{gname}': must be a mapping (got {type(g).__name__})")
            continue
        for flag in GROUP_FLAGS:
            if flag in g and not isinstance(g[flag], bool):
                errors.append(f"group '{gname}': {flag} must be true or false (got {g[flag]!r})")
        members = g.get("members")
        if not isinstance(members, list) or not members:
            errors.append(f"group '{gname}': members must be a non-empty list of model names")
            continue
        for mem in members:
            if mem not in model_names:
                errors.append(f"group '{gname}': member '{mem}' is not a model name in "
                              f"models.yaml (groups key off model names, not aliases)")
            elif mem in owner:
                errors.append(f"{mem}: appears in more than one group "
                              f"('{owner[mem]}' and '{gname}') — llama-swap allows only one")
            else:
                owner[mem] = gname
    return errors


def validate(models, groups=None, cluster=None):
    errors = []
    if cluster is not None:
        errors.extend(validate_cluster(cluster))
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

        nodes = m.nodes
        if isinstance(nodes, bool) or nodes not in (1, 2) or not isinstance(nodes, int):
            errors.append(f"{m.name}: nodes must be 1 or 2 (got {nodes!r})")
        elif nodes == 2:
            if m.engine != "vllm":
                errors.append(f"{m.name}: nodes > 1 is only supported for engine vllm")
            if cluster is None:
                errors.append(f"{m.name}: nodes: 2 needs a cluster: block in settings.local.yaml")
            for item in (m.raw.get("vllm_flags") or []):
                if not isinstance(item, str):
                    continue
                for tok in item.split():
                    if tok.split("=", 1)[0] in CLUSTER_RESERVED_FLAGS:
                        errors.append(f"{m.name}: vllm_flags {tok.split('=', 1)[0]} is set by "
                                      f"the cluster launcher on nodes: 2 models — remove it")

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
        rc = m.raw.get("reclaim_containers")
        if rc is not None:
            if not isinstance(rc, list) or not all(
                    isinstance(n, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", n) for n in rc):
                errors.append(f"{m.name}: reclaim_containers must be a list of container names "
                              f"(got {rc!r})")
        for fld in ("extra_docker_args", "pre_launch_cmd"):
            val = m.raw.get(fld)
            if isinstance(val, str) and "'" in val:
                errors.append(f"{m.name}: {fld} contains a single quote, which breaks shell quoting")

    if groups:
        errors.extend(_validate_groups(groups, {m.name for m in models}))
        two = {m.name for m in models if m.nodes == 2 and not isinstance(m.nodes, bool)}
        if isinstance(groups, dict):
            for gname, g in groups.items():
                if not isinstance(g, dict) or not isinstance(g.get("members"), list):
                    continue
                both = sorted({x for x in g["members"] if x in two})
                if not both:
                    continue
                if g.get("swap") is False and len(both) >= 2:
                    errors.append(f"group '{gname}': nodes: 2 models {both} cannot be loaded "
                                  f"together (swap: false) — they share the master port and "
                                  f"worker memory")
                # A non-exclusive group never unloads other groups (including the
                # implicit, exclusive default group), and a persistent group is never
                # unloaded by them — so any nodes: 2 model outside this group could be
                # resident at the same time as this group's nodes: 2 member.
                if g.get("exclusive") is False or g.get("persistent") is True:
                    outside = sorted(two - set(both))
                    if outside:
                        kind = ("exclusive: false" if g.get("exclusive") is False
                                else "persistent: true")
                        errors.append(f"group '{gname}' ({kind}): nodes: 2 model(s) {both} "
                                      f"and {outside} outside the group cannot be loaded "
                                      f"together — put every nodes: 2 model in '{gname}' "
                                      f"(they share the master port and worker memory)")

    return errors
