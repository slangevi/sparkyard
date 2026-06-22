"""Release-notes summary for `sparkyard update --check --notes`. Fetches
llama-swap GitHub release bodies between the pinned and latest versions and
summarizes them via the local LiteLLM gateway; images get a best-effort
one-liner. Pure assembly + injected I/O (mirrors update.py); falls back to raw
notes when the gateway is unreachable. No external API; stdlib only."""
import json
import os
import urllib.request
from collections import namedtuple

LLAMA_SWAP_REPO = "mostlygeek/llama-swap"
GATEWAY_BASE_URL = "http://localhost:14000"
_PROMPT_CHAR_BUDGET = 12000   # cap on the joined release bodies (the instruction prefix adds ~260 chars)

Release = namedtuple("Release", "tag version body url")

# Per-image changelog links (best-effort; a digest delta has no notes of its own).
CHANGELOG_URLS = {
    "ollama": "https://github.com/ollama/ollama/releases",
    "litellm": "https://github.com/BerriAI/litellm/releases",
    "litellm-db": "https://www.postgresql.org/docs/release/",
    "open-webui": "https://github.com/open-webui/open-webui/releases",
}


def llamaswap_notes(releases_json, current_version, latest_version):
    """From a parsed GitHub /releases list, return Releases for versions in
    (current, latest], newest-first. Non-vNNN tags are skipped. Pure."""
    out = []
    for rel in releases_json:
        tag = rel.get("tag_name", "")
        try:
            v = int(str(tag).lstrip("v"))
        except ValueError:
            continue
        if current_version < v <= latest_version:
            out.append(Release(tag, v, rel.get("body") or "", rel.get("html_url") or ""))
    return sorted(out, key=lambda r: r.version, reverse=True)


def build_summary_prompt(notes):
    """Assemble the gateway prompt from a list of Release. Pure."""
    body = "\n\n".join(f"## {r.tag}\n{r.body}" for r in notes)[:_PROMPT_CHAR_BUDGET]
    return (
        "You are summarizing release notes for llama-swap (an on-demand model "
        "proxy) for an operator running it on an NVIDIA DGX Spark. In 3-5 concise "
        "bullets, say what these releases add, fix, or break that an operator "
        "would care about. Be specific; skip boilerplate.\n\n" + body
    )


def image_note(service, old_digest, new_digest, version=None):
    """Best-effort one-liner for a digest-pinned image. Pure. `version` is
    reserved for a future image-version label (not extracted today — a digest
    delta has no version on the index manifest we fetch)."""
    def short(d):
        return d.split(":", 1)[-1][:8] if d else "?"
    ver = f" (v{version})" if version else ""
    url = CHANGELOG_URLS.get(service)
    tail = f" — changelog: {url}" if url else ""
    return f"{service}: {short(old_digest)}→{short(new_digest)}{ver}{tail}"


def read_master_key(root):
    """LITELLM_MASTER_KEY from env, else secrets.env at the repo root. None if blank/absent."""
    key = os.environ.get("LITELLM_MASTER_KEY")
    if key:
        return key
    secrets = os.path.join(root, "secrets.env")
    if os.path.exists(secrets):
        with open(secrets) as fh:
            for line in fh:
                if line.startswith("LITELLM_MASTER_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def _http_get_json(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _gateway_chat(prompt, *, base_url, key, model):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"].strip()


def _list_models(base_url, key):
    data = _http_get_json(f"{base_url}/v1/models", headers={"Authorization": f"Bearer {key}"})
    ids = [m["id"] for m in (data.get("data") or [])]
    return ids[0] if ids else None


NotesDeps = namedtuple("NotesDeps", "http_get_json gateway_chat list_models")
REAL_NOTES = NotesDeps(_http_get_json, _gateway_chat, _list_models)


def _print_raw(notes_list, reason):
    print(f"  ({reason} — raw notes)")
    for r in notes_list:
        body = (r.body or "").strip()
        trimmed = body[:1500] + ("…" if len(body) > 1500 else "")
        print(f"  {r.tag}: {r.url}")
        for ln in trimmed.splitlines():
            print(f"    {ln}")


def _render_llamaswap(root, ls_plan, model, base_url, deps):
    cur, latest = ls_plan["current"], ls_plan["latest"]
    print(f"─ llama-swap v{cur} → v{latest}")
    try:
        releases = deps.http_get_json(
            f"https://api.github.com/repos/{LLAMA_SWAP_REPO}/releases",
            headers={"Accept": "application/vnd.github+json"})
        nlist = llamaswap_notes(releases, cur, latest)
    except Exception as e:
        print(f"  (could not fetch release notes: {e})")
        return
    if not nlist:
        print("  (no release notes found)")
        return
    key = read_master_key(root)
    chosen = model
    if key and not chosen:
        try:
            chosen = deps.list_models(base_url, key)
        except Exception:
            chosen = None
    if key and chosen:
        try:
            summary = deps.gateway_chat(build_summary_prompt(nlist),
                                        base_url=base_url, key=key, model=chosen)
            for ln in summary.splitlines():
                print(f"  {ln}")
            print(f"  (summarized from {len(nlist)} release(s) via {chosen} @ {base_url})")
            return
        except Exception as e:
            _print_raw(nlist, f"gateway unavailable: {e}")
            return
    _print_raw(nlist, "no gateway/model — start the stack or pass --model")


def render_notes(root, image_results, ls_plan, *, model=None,
                 base_url=GATEWAY_BASE_URL, deps=REAL_NOTES):
    """Print the '--notes' section after the update report. Never raises."""
    try:
        newer_images = [r for r in image_results if r.status == "newer"]
        ls_newer = ls_plan.get("status") == "newer"
        if not newer_images and not ls_newer:
            print("\nNothing to summarize.")
            return
        print("\nWhat these updates provide  (--notes)")
        if ls_newer:
            _render_llamaswap(root, ls_plan, model, base_url, deps)
        for r in newer_images:
            print("─ " + image_note(r.pin.service, r.pin.digest, r.new_digest))
    except Exception as e:   # belt-and-suspenders: notes never break update
        print(f"  (notes unavailable: {e})")
