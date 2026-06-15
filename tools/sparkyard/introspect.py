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
    """Return (config: dict, files: list[str]). Raises IntrospectError if config.json is unreachable."""
    try:
        config = json.loads(_read(f"{HF}/{repo}/resolve/main/config.json", opener))
    except Exception as e:
        raise IntrospectError(f"could not fetch config.json for '{repo}': {e}")
    try:
        info = json.loads(_read(f"{HF}/api/models/{repo}", opener))
        files = [s.get("rfilename", "") for s in info.get("siblings", [])]
    except Exception:
        files = []
    return config, files


def _slug(name):
    return "vllm-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def derive_entry(repo, config, files, name=None):
    """Return (entry: dict|None, hints: list[str], is_gguf: bool)."""
    if any(f.endswith(".gguf") for f in files):
        return None, [], True

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
