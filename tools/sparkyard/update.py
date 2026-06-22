"""`update` subcommand: check for newer upstream component versions, bump the
pins in the tracked files (docker-compose.yml digests, llama-swap Dockerfile
ARGs), and pull/build the affected services. Never floats to :latest, never
commits.

Parse/plan/rewrite are pure and unit-tested; side effects (registry/GitHub/
docker) live behind injected deps for testability — mirrors vllm_node.py."""
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
import yaml
from collections import namedtuple


class UpdateError(Exception):
    pass


# A digest-pinned compose image. ref is the exact original substring
# ("repo[:tag]@sha256:digest"); digest is "sha256:...".
ImagePin = namedtuple("ImagePin", "service ref repo tag digest")
# status: "up-to-date" | "newer" | "error" | "no-tag"
ImageResult = namedtuple("ImageResult", "pin new_digest status")


def _split_repo_tag(name_part):
    """'ghcr.io/o/o:main' -> ('ghcr.io/o/o', 'main'); 'ollama/ollama' -> (.., None).
    Splits the tag only in the final path segment (so a registry :port is safe)."""
    slash = name_part.rfind("/")
    head, last = name_part[:slash + 1], name_part[slash + 1:]
    if ":" in last:
        repo_last, tag = last.split(":", 1)
        return head + repo_last, tag
    return name_part, None


def parse_image_pins(compose_text):
    """Return the digest-pinned ImagePins from a docker-compose text (services
    whose image: contains '@sha256:'). Pure."""
    doc = yaml.safe_load(compose_text) or {}
    pins = []
    for service, body in (doc.get("services") or {}).items():
        ref = (body or {}).get("image", "")
        if "@sha256:" not in ref:
            continue
        name_part, digest = ref.split("@", 1)
        repo, tag = _split_repo_tag(name_part)
        pins.append(ImagePin(service, ref, repo, tag, digest))
    return pins


def plan_image_updates(pins, resolve_digest):
    """For each pin, resolve repo:tag -> current digest and classify. Fail-soft:
    a resolver exception marks that image 'error'; a tagless pin is 'no-tag'."""
    results = []
    for pin in pins:
        if pin.tag is None:
            results.append(ImageResult(pin, None, "no-tag"))
            continue
        try:
            latest = resolve_digest(f"{pin.repo}:{pin.tag}")
        except Exception:
            results.append(ImageResult(pin, None, "error"))
            continue
        if not isinstance(latest, str) or not latest.startswith("sha256:"):
            # a malformed resolver result must not slip through as 'newer'
            results.append(ImageResult(pin, None, "error"))
            continue
        status = "up-to-date" if latest == pin.digest else "newer"
        results.append(ImageResult(pin, latest if status == "newer" else None, status))
    return results


def rewrite_compose(compose_text, results):
    """Swap the digest of each 'newer' result's pin in place. Format/comment
    preserving (string replace). Raises if an old ref is not unique."""
    text = compose_text
    for r in results:
        if r.status != "newer":
            continue
        if compose_text.count(r.pin.ref) != 1:
            raise UpdateError(f"image ref not unique in compose: {r.pin.ref}")
        text = text.replace(r.pin.ref, r.pin.ref.replace(r.pin.digest, r.new_digest))
    return text


LlamaSwapPin = namedtuple("LlamaSwapPin", "version sha256")
_LS_VERSION = re.compile(r"^ARG LLAMA_SWAP_VERSION=(\d+)", re.M)
_LS_SHA = re.compile(r"^ARG LLAMA_SWAP_SHA256=([0-9a-f]{64})", re.M)


def parse_llamaswap_pin(dockerfile_text):
    v = _LS_VERSION.search(dockerfile_text)
    s = _LS_SHA.search(dockerfile_text)
    if not v or not s:
        raise UpdateError("could not find LLAMA_SWAP_VERSION/SHA256 ARGs")
    return LlamaSwapPin(int(v.group(1)), s.group(1))


def plan_llamaswap_update(current_version, latest_tag, sha_of):
    """current_version: int. latest_tag: e.g. 'v226'. sha_of(version)->sha256 is
    only called when strictly newer (avoids a needless download)."""
    try:
        latest = int(str(latest_tag).lstrip("v"))
    except ValueError:
        raise UpdateError(f"unrecognised llama-swap tag format: {latest_tag!r}")
    if latest == current_version:
        status, new_sha = "up-to-date", None
    elif latest < current_version:
        status, new_sha = "older", None
    else:
        status, new_sha = "newer", sha_of(latest)
    return {"current": current_version, "latest": latest, "new_sha": new_sha, "status": status}


def rewrite_llamaswap(dockerfile_text, new_version, new_sha):
    text, n1 = _LS_VERSION.subn(f"ARG LLAMA_SWAP_VERSION={new_version}", dockerfile_text, count=1)
    text, n2 = _LS_SHA.subn(f"ARG LLAMA_SWAP_SHA256={new_sha}", text, count=1)
    if n1 != 1 or n2 != 1:
        raise UpdateError("llama-swap ARG lines not found for rewrite")
    return text


LLAMA_SWAP_REPO = "mostlygeek/llama-swap"

Deps = namedtuple("Deps", "resolve_digest latest_release release_sha256 docker_pull docker_build")


def _manifest_digest(imagetools_json):
    """Extract the top-level index/manifest-list digest from the JSON emitted by
    `imagetools inspect --format '{{json .Manifest}}'`. Pure (unit-tested).

    The top-level `.digest` is the digest the `image: repo@sha256:…` pin uses; the
    per-arch entries in `.manifests[]` are NOT what we want."""
    try:
        digest = json.loads(imagetools_json)["digest"]
    except (ValueError, KeyError, TypeError) as e:
        raise UpdateError(f"could not parse manifest digest: {e}")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise UpdateError(f"unexpected manifest digest: {digest!r}")
    return digest


def _resolve_digest(repo_tag):
    # `{{.Manifest.Digest}}` is silently ignored by some buildx versions (prints
    # the default verbose block); `{{json .Manifest}}` reliably yields the index
    # descriptor whose top-level `digest` matches the compose pin.
    p = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", repo_tag, "--format", "{{json .Manifest}}"],
        capture_output=True, text=True)
    if p.returncode != 0:
        raise UpdateError(f"imagetools inspect failed for {repo_tag}: {p.stderr.strip()}")
    return _manifest_digest(p.stdout)


def _latest_release(repo):
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/latest",
                                 headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["tag_name"]


def _release_sha256(repo, version):
    url = (f"https://github.com/{repo}/releases/download/v{version}/"
           f"llama-swap_{version}_linux_arm64.tar.gz")
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as r:
        for chunk in iter(lambda: r.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _docker_pull(root, services):
    return subprocess.run(["docker", "compose", "pull", *services], cwd=root).returncode


def _docker_build(root, services):
    return subprocess.run(["docker", "compose", "build", *services], cwd=root).returncode


REAL = Deps(_resolve_digest, _latest_release, _release_sha256, _docker_pull, _docker_build)


def _short(d):  # "sha256:abcd1234..." -> "abcd1234"
    return d.split(":", 1)[-1][:8] if d else "?"


def format_report(image_results, ls_plan, llamacpp_note, vllm_note):
    lines = ["", "Component        Current     Latest      Status"]
    for r in image_results:
        if r.status == "newer":
            cur, latest, st = _short(r.pin.digest), _short(r.new_digest), "NEWER"
        elif r.status == "up-to-date":
            cur, latest, st = _short(r.pin.digest), _short(r.pin.digest), "up to date"
        else:  # error | no-tag
            cur, latest, st = _short(r.pin.digest), "-", r.status
        lines.append(f"{r.pin.service:<16} {cur:<11} {latest:<11} {st}")
    ls_latest = f"v{ls_plan['latest']}" if ls_plan.get("latest") is not None else "-"
    lines.append(f"{'llama-swap':<16} {'v' + str(ls_plan.get('current', '?')):<11} "
                 f"{ls_latest:<11} {'NEWER' if ls_plan.get('status') == 'newer' else ls_plan.get('status')}")
    lines += ["", llamacpp_note, vllm_note, ""]
    return "\n".join(lines)


def _atomic_write(path, text):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


_LLAMACPP_NOTE = ("llama-cpp : builds llama.cpp HEAD from source for GB10/SM121 + CUDA 13.1 "
                  "(no upstream image targets this yet). Refresh: docker compose build "
                  "--no-cache llama-cpp.")


def run(root, settings, *, check=False, notes=False, model=None, deps=REAL):
    """Check/apply component updates. Returns 0 on success (incl. nothing-to-do),
    1 if an apply-phase rewrite/IO step fails. Per-image registry lookups are
    fail-soft; side effects are injected via `deps` for testability."""
    compose_path = os.path.join(root, "docker-compose.yml")
    ls_path = os.path.join(root, "llama-swap", "llama-swap.Dockerfile")
    with open(compose_path) as f:
        compose_text = f.read()
    with open(ls_path) as f:
        ls_text = f.read()

    image_results = plan_image_updates(parse_image_pins(compose_text), deps.resolve_digest)
    # Under --check the report only needs the version, so skip the (multi-MB) sha
    # download; the real fetch happens only when applying.
    sha_of = (lambda v: None) if check else (lambda v: deps.release_sha256(LLAMA_SWAP_REPO, v))
    # Parse + release-check together: a malformed Dockerfile or a failed GitHub
    # lookup degrades llama-swap to status 'error' rather than crashing the run.
    try:
        ls_pin = parse_llamaswap_pin(ls_text)
        ls_plan = plan_llamaswap_update(
            ls_pin.version, deps.latest_release(LLAMA_SWAP_REPO), sha_of=sha_of)
    except Exception as e:
        ls_plan = {"current": "?", "latest": None, "new_sha": None, "status": "error"}
        print(f"note: llama-swap check failed: {e}", file=sys.stderr)

    vllm_note = (f"vllm-node : pinned vLLM ref {settings.vllm.vllm_ref}; bump settings.local.yaml "
                 f"vllm.vllm_ref + run `make vllm-node` (~30 min) to update.")
    print(format_report(image_results, ls_plan, _LLAMACPP_NOTE, vllm_note))
    if notes:
        from . import notes as notes_mod
        notes_mod.render_notes(root, image_results, ls_plan, model=model)

    newer_images = [r for r in image_results if r.status == "newer"]
    ls_newer = ls_plan.get("status") == "newer"
    if check:
        print("Dry run (--check). Run `make update` to apply." if (newer_images or ls_newer)
              else "Everything up to date.")
        return 0
    if not newer_images and not ls_newer:
        print("Everything up to date.")
        return 0

    docker_rc = 0
    try:
        if newer_images:
            _atomic_write(compose_path, rewrite_compose(compose_text, image_results))
        if ls_newer:
            _atomic_write(ls_path, rewrite_llamaswap(ls_text, ls_plan["latest"], ls_plan["new_sha"]))
            docker_rc = deps.docker_build(root, ["llama-swap"]) or docker_rc
        if newer_images:
            docker_rc = deps.docker_pull(root, [r.pin.service for r in newer_images]) or docker_rc
    except (UpdateError, OSError) as e:
        # A rewrite/IO/missing-binary failure → clean coded exit, not a traceback.
        # Safe: rewrite guards raise before writing, so no partial/corrupt file.
        print(f"✗ {e}", file=sys.stderr)
        return 1

    if docker_rc:
        print("\n• Bumped pins in the tracked files, but a docker pull/build step "
              "reported an error — check the output above.")
    else:
        print("\n✓ Bumped pins in the tracked files and pulled/built the changed services.")
    print("  Review `git diff` and commit when you're happy.")
    return 0
