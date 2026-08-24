"""`sparkyard models` — what is configured, and what of it is actually here.

Answers the questions that otherwise require ad-hoc YAML spelunking: which
models exist, which engine and image each uses, which have weights on disk, and
what aliases callers can address them by. `--json` makes it scriptable."""
import json
import os


def _rows(models, settings):
    out = []
    for m in models:
        rel = m.raw.get("path") if m.engine == "vllm" else m.raw.get("gguf")
        host = os.path.join(settings.llm_root, rel) if rel else None
        g = m.raw.get("gmem") or {}
        out.append({
            "name": m.name,
            "engine": m.engine,
            "image": m.image or "",
            "aliases": list(m.raw.get("aliases") or []),
            "ctx": m.raw.get("max_model_len") or m.raw.get("ctx_size"),
            "gmem_min": g.get("min"),
            "gmem_max": g.get("max"),
            "path": rel or "",
            "on_disk": bool(host and os.path.exists(host)),
        })
    return out


def render(models, settings, *, on_disk=False, engine=None, as_json=False):
    rows = _rows(models, settings)
    if on_disk:
        rows = [r for r in rows if r["on_disk"]]
    if engine:
        rows = [r for r in rows if r["engine"] == engine]
    if as_json:
        return json.dumps(rows, indent=2)
    if not rows:
        return "no models match"
    w = max(len(r["name"]) for r in rows)
    lines = [f"{'NAME':<{w}}  {'ENGINE':<9} {'CTX':>7}  {'GMEM':<11} {'ON-DISK':<7} ALIASES"]
    for r in rows:
        gm = (f"{r['gmem_min']}–{r['gmem_max']}"
              if r["gmem_min"] is not None else "-")
        lines.append(f"{r['name']:<{w}}  {r['engine']:<9} {str(r['ctx'] or '-'):>7}  "
                     f"{gm:<11} {'yes' if r['on_disk'] else 'no':<7} "
                     f"{', '.join(r['aliases'])}")
    present = sum(1 for r in rows if r["on_disk"])
    lines.append(f"\n{present}/{len(rows)} with weights on disk")
    return "\n".join(lines)
