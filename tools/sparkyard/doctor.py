"""Advisory on-disk report for models.yaml. Never blocks render (caller exits 0).

Resolves each model's HOST path from settings.llm_root (mounted at /models in
containers; `path`/`gguf` in models.yaml are /models-relative)."""
import os


def check(models, settings):
    """Return (lines: list[str], summary: str). Reads the filesystem; no mutation."""
    lines = []
    present = 0
    counted = 0
    for m in models:
        counted += 1
        rel = m.raw["path"] if m.engine == "vllm" else m.raw["gguf"]
        host = os.path.join(settings.llm_root, rel)
        if os.path.exists(host):
            present += 1
            lines.append(f"  [ok]      {m.name}")
        else:
            lines.append(f"  [MISSING] {m.name}: {host}")
        ct = m.chat_template
        if ct and not ct.startswith("/"):
            cth = os.path.join(settings.llm_root, ct)
            if not os.path.exists(cth):
                lines.append(f"            ! chat_template missing: {cth}")
        if m.engine == "vllm":
            fl = " ".join(m.vllm_flags)
            if ("--mamba-ssm-cache-dtype" in fl or "--mamba_ssm_cache_dtype" in fl) \
                    and "tf5" not in (m.image or ""):
                lines.append(f"            ~ mamba flag on non-tf5 image '{m.image}' — "
                             f"works today; if it fails to load, try a vllm-node-tf5 image")
    summary = f"{present}/{counted} models have weights on disk"
    return lines, summary
