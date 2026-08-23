"""Advisory on-disk report for models.yaml. Never blocks render (caller exits 0).

Resolves each model's HOST path from settings.llm_root (mounted at /models in
containers; `path`/`gguf` in models.yaml are /models-relative)."""
import os

# Mirrors launch.py's defaults. The CUDA view of memory is the crash-guard
# ceiling minus a fixed overhead; gpu_memory_utilization is a fraction of THAT,
# and vLLM reserves the whole fraction rather than only what it needs.
SYSTEM_RAM_CEILING_GIB = 117.81
CUDA_OVERHEAD_GIB = 6.3
_GIB = 1024 ** 3


def _weights_gib(host):
    """Sum of *.safetensors under `host`, in GiB. 0.0 if unreadable."""
    total = 0
    for root, _dirs, files in os.walk(host):
        for f in files:
            if f.endswith(".safetensors"):
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return total / _GIB


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
            if "fastsafetensors" in fl:
                lines.append("            ! --load-format fastsafetensors buffers the whole "
                             "weight set while CUDA reserves its budget; measured +32GiB peak "
                             "(86 vs 54) on a 21.8GiB model. Risky on unified memory.")
            gmax = (m.raw.get("gmem") or {}).get("max")
            if gmax and os.path.exists(host):
                budget = gmax * (SYSTEM_RAM_CEILING_GIB - CUDA_OVERHEAD_GIB)
                w = _weights_gib(host)
                if w and w >= budget:
                    lines.append(f"            ! weights {w:.1f}GiB exceed the gmem budget "
                                 f"{budget:.1f}GiB (gmem.max {gmax} x "
                                 f"{SYSTEM_RAM_CEILING_GIB - CUDA_OVERHEAD_GIB:.1f}GiB) — "
                                 f"raise gmem.max or the model cannot load")
    summary = f"{present}/{counted} models have weights on disk"
    return lines, summary
