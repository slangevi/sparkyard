"""SSOT-driven model download: fetch HuggingFace weights for models.yaml entries
that carry an `hf_repo`, into {llm_root}/{path}. Backs `make download`."""
import os
import sys


def snapshot(repo, local, token):
    """Fetch a HF repo into `local`. huggingface_hub is imported lazily so the rest
    of the CLI works without it installed."""
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=local, token=token)


def select(models, only=None):
    """Models to download. only=<name> → just that one (ValueError if unknown or it
    has no hf_repo). Otherwise every model carrying an hf_repo."""
    if only is not None:
        match = [m for m in models if m.name == only]
        if not match:
            names = ", ".join(m.name for m in models)
            raise ValueError(f"unknown model '{only}' (known: {names})")
        if not match[0].hf_repo:
            raise ValueError(f"'{only}' has no hf_repo in models.yaml — add `hf_repo: <org/model>` "
                             f"or re-add via `make add-model HF_REPO=...`")
        return match
    return [m for m in models if m.hf_repo]


def run(models, settings, token, only=None, exists=os.path.exists):
    """Download selected models. Skips on-disk entries; notes entries lacking hf_repo.
    Returns 0 on success, 1 on any failure / bad selection."""
    try:
        targets = select(models, only)
    except ValueError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1
    if only is None:
        for m in models:
            if not m.hf_repo:
                print(f"[no hf_repo] {m.name} (skipped — add hf_repo to fetch)")
    if not targets:
        print("nothing to download (no entries with hf_repo).")
        return 0
    failures = 0
    for m in targets:
        rel = m.raw.get("path")
        if not rel:
            print(f"[no path] {m.name} (GGUF/llamacpp — fetch by hand)")
            continue
        local = os.path.join(settings.llm_root, rel)
        if exists(local):
            print(f"[skip] {m.name} (on disk: {local})")
            continue
        print(f"[download] {m.name}: {m.hf_repo} -> {local}")
        try:
            snapshot(m.hf_repo, local, token)
        except Exception as e:  # noqa: BLE001 — surface any HF/network error, keep going
            print(f"[FAILED] {m.name}: {e}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0
