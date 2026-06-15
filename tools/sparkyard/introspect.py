"""Best-effort HF repo introspection for the add-model wizard (stdlib only).

derive_entry() produces a CONSERVATIVE, internally-consistent vLLM baseline plus
human-readable hints for special cases — never an auto-tuned entry that could
fail Tier-A validation. The user always confirms/edits the proposal."""
import json
import re
import urllib.request

HF = "https://huggingface.co"


class IntrospectError(Exception):
    pass


def _read(url, opener):
    with opener(url) as resp:
        return resp.read()


def fetch_repo_metadata(repo, opener=urllib.request.urlopen):
    """Return (config: dict|None, files: list[str]).

    Siblings (the file list) are the primary signal — they tell us whether the
    repo is GGUF. config.json is OPTIONAL: pure GGUF repos often lack one.
    Raises IntrospectError only if the repo itself can't be listed."""
    try:
        info = json.loads(_read(f"{HF}/api/models/{repo}", opener))
        files = [s.get("rfilename", "") for s in info.get("siblings", [])]
    except Exception as e:
        raise IntrospectError(f"could not list files for '{repo}': {e}")
    try:
        config = json.loads(_read(f"{HF}/{repo}/resolve/main/config.json", opener))
    except Exception:
        config = None
    return config, files


def is_gguf_repo(files):
    """True if any sibling file is a .gguf weight."""
    return any(f.endswith(".gguf") for f in files)


def infer_ctx_size(config, default=8192):
    """(ctx_size, inferred). Reads max_position_embeddings from config.json (or its
    text_config) when available — the model's trained context. Else the conservative
    default with inferred=False so the caller can warn."""
    if config:
        text_cfg = config.get("text_config") or {}
        mpe = config.get("max_position_embeddings")
        if mpe is None:
            mpe = text_cfg.get("max_position_embeddings")
        if mpe is not None:
            return int(mpe), True
    return default, False


def _slug(name, prefix="vllm-"):
    return prefix + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def derive_gguf_entry(repo, chosen_file, config, name=None):
    """Return (entry, hints) for a llamacpp/GGUF model. `chosen_file` is the
    repo-relative path of the first shard of the chosen quant."""
    name = name or repo.split("/")[-1]
    ctx, inferred = infer_ctx_size(config)
    hints = []
    if inferred:
        hints.append(f"ctx_size {ctx} inferred from config.json; it drives llama.cpp "
                     f"KV allocation — lower it to save memory.")
    else:
        hints.append(f"WARNING: couldn't infer context length (no config.json in repo); "
                     f"defaulted ctx_size to {ctx} — raise toward the model's trained "
                     f"context if needed.")
    if "/" in chosen_file:
        hints.append(f"'{chosen_file}' is in a subdirectory; v1 uses the repo-root GGUF "
                     f"convention — verify the gguf/mount path or place the file manually.")
    entry = {
        "name": name,
        "engine": "llamacpp",
        "container": _slug(name, "llamacpp-"),
        "hf_repo": repo,
        "mount": "{llm_root}/gguf:/models/gguf",
        "gguf": f"gguf/{repo}/{chosen_file}",
        "ctx_size": ctx,
        "parallel": 4,
        "no_mmap": True,
        "unified_memory": True,
        "llamacpp_flags": ["--jinja"],
    }
    return entry, hints


def derive_entry(repo, config, files, name=None):
    """Return (entry: dict|None, hints: list[str], is_gguf: bool)."""
    if is_gguf_repo(files):
        return None, [], True

    if config is None:
        raise IntrospectError(f"'{repo}': no config.json and no .gguf files — can't introspect")

    name = name or repo.split("/")[-1]
    text_cfg = config.get("text_config") or {}
    mml = config.get("max_position_embeddings") or text_cfg.get("max_position_embeddings") or 131072
    blob = " ".join([
        str(config.get("model_type") or ""),
        " ".join(config.get("architectures") or []),
        repo,
    ]).lower()

    hints = []
    image = "vllm-node:latest"
    if "mxfp4" in blob or "gpt-oss" in blob or "gpt_oss" in blob:
        image = "vllm-node-mxfp4"
        hints.append("Looks like an mxfp4/GPT-OSS model: image set to vllm-node-mxfp4. "
                     "Add --quantization mxfp4 and the mxfp4 flags your build needs.")
    if any(k in blob for k in ("mamba", "hybrid", "qwen3_next", "nemotron_h")):
        hints.append("Looks like a mamba/hybrid model: it may need vllm-node-tf5:latest + "
                     "--mamba-ssm-cache-dtype float16. (Some run on vllm-node — try as-is first.)")

    entry = {
        "name": name,
        "engine": "vllm",
        "tier": "M",
        "container": _slug(name),
        "path": f"vllm/{repo}",
        "hf_repo": repo,
        "image": image,
        "max_model_len": int(mml),
        "max_num_seqs": 10,
        "kv_dtype_bytes": 1,
        "gmem": {"min": 0.40, "max": 0.85},
        "safety_gib": 6,
        "vllm_flags": [
            "--kv-cache-dtype fp8",
            "--load-format fastsafetensors",
            "--enable-prefix-caching",
            "--trust-remote-code",
        ],
    }
    return entry, hints, False
