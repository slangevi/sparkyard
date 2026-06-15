"""SSOT-driven model download: fetch HuggingFace weights for models.yaml entries
that carry an `hf_repo`, into {llm_root}/{path}. Backs `make download`."""
import os
import re
import sys

from .placeholders import resolve

_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")


def shard_family(basename):
    """All filenames in a GGUF shard family. Non-shard -> [basename].
    Naming is deterministic (`stem-00001-of-NNNNN.gguf`), so no HF call is needed."""
    m = _SHARD_RE.match(basename)
    if not m:
        return [basename]
    stem, total = m.group("stem"), int(m.group("total"))
    return [f"{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]


def gguf_families(files):
    """Group repo .gguf files into quant families keyed by a label (the shard stem,
    or the bare filename sans .gguf for single-file quants). Values are sorted members."""
    fams = {}
    for f in files:
        if not f.endswith(".gguf"):
            continue
        m = _SHARD_RE.match(f)
        label = m.group("stem") if m else f[: -len(".gguf")]
        fams.setdefault(label, []).append(f)
    for label in fams:
        fams[label].sort()
    return fams


def gguf_target(model, settings):
    """(host_dir, basename) for a llamacpp entry — map mount+gguf to a host path.
    `resolve` is a no-op if the mount's {llm_root} was already resolved by load()."""
    host_spec, container_spec = model.raw["mount"].rsplit(":", 1)
    host_spec = resolve(host_spec, settings.placeholder_map())
    container_full = "/models/" + model.raw["gguf"]
    cspec = container_spec.rstrip("/") + "/"
    if not container_full.startswith(cspec):
        raise ValueError(f"{model.name}: gguf '{container_full}' is not under mount '{container_spec}'")
    rel = container_full[len(cspec):]
    host_full = os.path.join(host_spec, rel)
    return os.path.dirname(host_full), os.path.basename(host_full)


def snapshot(repo, local, token, allow_patterns=None):
    """Fetch a HF repo (or just allow_patterns of it) into `local`. huggingface_hub
    is imported lazily so the rest of the CLI works without it installed."""
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=local, token=token, allow_patterns=allow_patterns)


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
        try:
            if m.raw.get("path"):
                local = os.path.join(settings.llm_root, m.raw["path"])
                if exists(local):
                    print(f"[skip] {m.name} (on disk: {local})")
                    continue
                print(f"[download] {m.name}: {m.hf_repo} -> {local}")
                snapshot(m.hf_repo, local, token)
            elif m.raw.get("gguf"):
                host_dir, base = gguf_target(m, settings)
                family = shard_family(base)
                if all(exists(os.path.join(host_dir, f)) for f in family):
                    print(f"[skip] {m.name} (on disk: {os.path.join(host_dir, base)})")
                    continue
                print(f"[download] {m.name}: {m.hf_repo} {family} -> {host_dir}")
                snapshot(m.hf_repo, host_dir, token, allow_patterns=family)
            else:
                print(f"[no path/gguf] {m.name} (can't determine destination — fetch by hand)")
        except Exception as e:  # noqa: BLE001 — surface any HF/network/mapping error, keep going
            print(f"[FAILED] {m.name}: {e}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0
