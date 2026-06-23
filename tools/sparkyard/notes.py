"""Release-notes summary for `sparkyard update --check --notes`. Fetches
llama-swap GitHub release bodies between the pinned and latest versions and
summarizes them via the local LiteLLM gateway; images get a best-effort
one-liner. Pure assembly + injected I/O (mirrors update.py); falls back to raw
notes when the gateway is unreachable. No external API; stdlib only."""
import json
import os
import subprocess
import urllib.request
from collections import namedtuple

class UpdateNotesError(Exception):
    pass


LLAMA_SWAP_REPO = "mostlygeek/llama-swap"
VLLM_REPO = "vllm-project/vllm"
LLAMACPP_REPO = "ggml-org/llama.cpp"
GATEWAY_BASE_URL = "http://localhost:14000"
_PROMPT_CHAR_BUDGET = 12000   # cap on the joined notes/commit body (the instruction prefix adds ~535 chars)

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


def releases_body(notes):
    """Concatenate Release bodies for the prompt. Pure."""
    return "\n\n".join(f"## {r.tag}\n{r.body}" for r in notes)


def commits_body(subjects):
    """Bullet-list commit subject lines for the prompt. Pure."""
    return "\n".join(f"- {s}" for s in subjects)


def build_summary_prompt(source, body):
    """`source`: human label of what changed (e.g. 'llama-swap releases v224→v226',
    'litellm commits', 'vllm-node: vLLM commits since 7852e50e'). `body`: preassembled
    notes/commit text (capped here). Asks for 3-5 bullets + one Recommendation line."""
    body = body[:_PROMPT_CHAR_BUDGET]
    return (
        f"You are summarizing changes in {source} for an operator running it on an "
        "NVIDIA DGX Spark. In 3-5 concise bullets, say what changed that an operator "
        "would care about (features, fixes, breaking changes); skip boilerplate. "
        "Then end with exactly one final line, choosing ONE of these two forms:\n"
        "  Recommendation: Apply — <one-line reason>   (use for additive / bugfix / routine)\n"
        "  Recommendation: Review first — <one-line reason>   (use for breaking changes, "
        "auth or default-behavior changes, deprecations, or anything needing operator "
        "action)\nThis is advisory, based only on the notes below.\n\n" + body
    )


def image_note(service, old_digest, new_digest, version=None):
    """Best-effort one-liner for a digest-pinned image. Pure. `version` is
    reserved for a future image-version label (not extracted today — a digest
    delta has no version on the index manifest we fetch)."""
    ver = f" (v{version})" if version else ""
    url = CHANGELOG_URLS.get(service)
    tail = f" — changelog: {url}" if url else ""
    return f"{service}: {_short(old_digest)}→{_short(new_digest)}{ver}{tail}"


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


def _image_labels(ref):
    p = subprocess.run(["docker", "buildx", "imagetools", "inspect", ref,
                        "--format", "{{json .Image}}"], capture_output=True, text=True)
    if p.returncode != 0:
        raise UpdateNotesError(p.stderr.strip())
    return json.loads(p.stdout)


NotesDeps = namedtuple("NotesDeps", "http_get_json gateway_chat list_models image_labels")
REAL_NOTES = NotesDeps(_http_get_json, _gateway_chat, _list_models, _image_labels)


def _print_raw(notes_list, reason):
    print(f"  ({reason} — raw notes)")
    for r in notes_list:
        body = (r.body or "").strip()
        trimmed = body[:1500] + ("…" if len(body) > 1500 else "")
        print(f"  {r.tag}: {r.url}")
        for ln in trimmed.splitlines():
            print(f"    {ln}")


def _gateway_summary(root, source, body, model, base_url, deps):
    """Try a gateway summary. Returns ((summary, model_used), None) on success, or
    (None, reason) when there's no key/model or the call fails."""
    key = read_master_key(root)
    chosen = model
    if key and not chosen:
        try:
            chosen = deps.list_models(base_url, key)
        except Exception:
            chosen = None
    if not (key and chosen):
        return None, "no gateway/model — start the stack or pass --model"
    try:
        out = deps.gateway_chat(build_summary_prompt(source, body),
                                base_url=base_url, key=key, model=chosen)
        return (out, chosen), None
    except Exception as e:
        return None, f"gateway unavailable: {e}"


def _print_summary(res, base_url, count_note=""):
    summary, chosen = res
    for ln in summary.splitlines():
        print(f"  {ln}")
    print(f"  (summarized {count_note}via {chosen} @ {base_url})")


def _print_raw_lines(lines, reason):
    print(f"  ({reason} — raw)")
    for ln in lines:
        print(f"  {ln}")


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
    res, reason = _gateway_summary(root, f"llama-swap releases v{cur}→v{latest}",
                                   releases_body(nlist), model, base_url, deps)
    if res:
        _print_summary(res, base_url, f"from {len(nlist)} release(s) ")
    else:
        _print_raw(nlist, reason)


def compare_commits(repo, base, head, http_get_json, *, cap=40):
    """GitHub compare base...head → (subjects, total_commits). subjects = the most-
    recent `cap` non-merge commit first-lines (GitHub lists oldest→newest). Pure
    given http_get_json; raises on fetch error (caller fail-softs)."""
    data = http_get_json(f"https://api.github.com/repos/{repo}/compare/{base}...{head}")
    total = data.get("total_commits", 0)
    subjects = []
    for c in data.get("commits", []):
        msg = (c.get("commit", {}).get("message") or "").splitlines()
        line = msg[0].strip() if msg else ""
        if line and not line.lower().startswith("merge "):
            subjects.append(line)
    return subjects[-cap:], total


def resolve_head(repo, branch, http_get_json):
    """Return the commit SHA at the HEAD of `branch`. Pure given http_get_json;
    raises UpdateNotesError on a malformed response."""
    data = http_get_json(f"https://api.github.com/repos/{repo}/commits/{branch}")
    sha = data.get("sha") if isinstance(data, dict) else None
    if not isinstance(sha, str) or not sha:
        raise UpdateNotesError(f"no commit sha for {repo}@{branch}")
    return sha


def _labels_from(image_json):
    """Recursively collect any OCI Labels/labels dict from an
    `imagetools inspect --format '{{json .Image}}'` blob. Pure."""
    found = {}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("Labels", "labels") and isinstance(v, dict):
                    found.update(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(image_json)
    return found


def _github_repo(url):
    """'https://github.com/o/r(.git)(#frag)(/tree/..)' -> 'o/r', else None."""
    if "github.com/" not in url:
        return None
    tail = url.split("github.com/", 1)[1].split("#", 1)[0].strip("/")
    parts = tail.split("/")
    if len(parts) < 2:
        return None
    repo = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    return f"{parts[0]}/{repo}"


def image_revision(image_json):
    """(git_sha, 'owner/repo') from OCI revision+source labels, or None."""
    labels = _labels_from(image_json)
    rev = labels.get("org.opencontainers.image.revision")
    repo = _github_repo(labels.get("org.opencontainers.image.source") or "")
    return (rev, repo) if rev and repo else None


def _short(d):
    """First 8 hex chars of a digest (after the 'sha256:' prefix), or '?'."""
    return d.split(":", 1)[-1][:8] if d else "?"


def _render_image(root, r, model, base_url, deps):
    """Summarize an image's commit diff via OCI revision labels; fail-soft to the
    one-liner if provenance/compare/gateway is unavailable."""
    def oneliner():
        print("─ " + image_note(r.pin.service, r.pin.digest, r.new_digest))
    try:
        old = image_revision(deps.image_labels(f"{r.pin.repo}@{r.pin.digest}"))
        new = image_revision(deps.image_labels(f"{r.pin.repo}@{r.new_digest}"))
    except Exception:
        return oneliner()
    if not old or not new or old[1] != new[1]:
        return oneliner()
    try:
        subs, total = compare_commits(old[1], old[0], new[0], deps.http_get_json)
    except Exception:
        return oneliner()
    if not subs:
        return oneliner()
    print(f"─ {r.pin.service}: {total} commit(s) "
          f"({_short(r.pin.digest)}→{_short(r.new_digest)})")
    res, reason = _gateway_summary(root, f"{r.pin.service} commits",
                                   commits_body(subs), model, base_url, deps)
    if res:
        _print_summary(res, base_url)
    else:
        _print_raw_lines(subs, reason)


def _render_vllm(root, vllm_ref, model, base_url, deps):
    """Summarize vLLM commits between the pinned ref and main. Fail-soft to the
    report-only note (so vllm-node behaves as before when the compare is unavailable)."""
    note = (f"vllm-node : pinned vLLM ref {vllm_ref}; run "
            f"`sparkyard update vllm-node` to rebuild at {VLLM_REPO}@main HEAD.")
    try:
        subs, total = compare_commits(VLLM_REPO, vllm_ref, "main", deps.http_get_json)
    except Exception:
        print("─ " + note)
        return
    if not subs:
        print("─ " + note)
        return
    print(f"─ vllm-node: {total} commit(s) since {vllm_ref[:9]} on {VLLM_REPO}@main")
    if total > 250:
        print("  (large jump — review carefully; list truncated)")
    res, reason = _gateway_summary(root, f"vllm-node: vLLM commits since {vllm_ref[:9]}",
                                   commits_body(subs), model, base_url, deps)
    if res:
        _print_summary(res, base_url, f"from {len(subs)} of {total} commit(s) ")
    else:
        _print_raw_lines(subs, reason)


def _render_llamacpp(root, llamacpp_ref, model, base_url, deps):
    note = (f"llama-cpp : pinned llama.cpp ref {llamacpp_ref}; run "
            f"`sparkyard update llama-cpp` to rebuild at {LLAMACPP_REPO}@master HEAD.")
    try:
        subs, total = compare_commits(LLAMACPP_REPO, llamacpp_ref, "master", deps.http_get_json)
    except Exception:
        print("─ " + note)
        return
    if not subs:
        print("─ " + note)
        return
    print(f"─ llama-cpp: {total} commit(s) since {llamacpp_ref[:9]} on {LLAMACPP_REPO}@master")
    res, reason = _gateway_summary(root, f"llama-cpp: llama.cpp commits since {llamacpp_ref[:9]}",
                                   commits_body(subs), model, base_url, deps)
    if res:
        _print_summary(res, base_url, f"from {len(subs)} of {total} commit(s) ")
    else:
        _print_raw_lines(subs, reason)


def render_notes(root, image_results, ls_plan, *, vllm_ref=None, llamacpp_ref=None,
                 model=None, base_url=GATEWAY_BASE_URL, deps=REAL_NOTES):
    """Print the '--notes' section after the update report. Never raises."""
    try:
        newer_images = [r for r in image_results if r.status == "newer"]
        ls_newer = ls_plan.get("status") == "newer"
        if not newer_images and not ls_newer and not vllm_ref and not llamacpp_ref:
            print("\nNothing to summarize.")
            return
        print("\nWhat these updates provide  (--notes)")
        if ls_newer:
            _render_llamaswap(root, ls_plan, model, base_url, deps)
        for r in newer_images:
            _render_image(root, r, model, base_url, deps)
        if vllm_ref:
            _render_vllm(root, vllm_ref, model, base_url, deps)
        if llamacpp_ref:
            _render_llamacpp(root, llamacpp_ref, model, base_url, deps)
    except Exception as e:   # belt-and-suspenders: notes never break update
        print(f"  (notes unavailable: {e})")
