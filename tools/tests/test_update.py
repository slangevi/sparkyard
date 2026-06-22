import pytest

from sparkyard import update

COMPOSE = """\
services:
  ollama:
    image: ollama/ollama:latest@sha256:aaaa
  litellm:
    image: docker.litellm.ai/berriai/litellm-database:main-stable@sha256:bbbb
  litellm-db:
    image: postgres:15-alpine@sha256:cccc  # a comment
  open-webui:
    image: ghcr.io/open-webui/open-webui:main@sha256:dddd  # pinned 2026-06-14
  llama-swap:
    image: sparkyard/llama-swap-spark:latest
    build:
      context: ./llama-swap
"""


def test_parse_image_pins_extracts_only_digest_pinned():
    pins = update.parse_image_pins(COMPOSE)
    by = {p.service: p for p in pins}
    assert set(by) == {"ollama", "litellm", "litellm-db", "open-webui"}  # not llama-swap
    assert by["ollama"].repo == "ollama/ollama" and by["ollama"].tag == "latest"
    assert by["ollama"].digest == "sha256:aaaa"
    assert by["litellm"].repo == "docker.litellm.ai/berriai/litellm-database"
    assert by["litellm"].tag == "main-stable"
    assert by["open-webui"].repo == "ghcr.io/open-webui/open-webui"
    assert by["open-webui"].tag == "main"


def test_plan_image_updates_classifies():
    pins = update.parse_image_pins(COMPOSE)

    def fake_resolve(repo_tag):
        return {"ollama/ollama:latest": "sha256:NEW",            # changed
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",  # same
                "postgres:15-alpine": "sha256:cccc",             # same
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}[repo_tag]

    res = {r.pin.service: r for r in update.plan_image_updates(pins, fake_resolve)}
    assert res["ollama"].status == "newer" and res["ollama"].new_digest == "sha256:NEW"
    assert res["litellm"].status == "up-to-date"
    assert res["litellm-db"].status == "up-to-date"


def test_plan_image_updates_failsoft_on_resolve_error():
    pins = update.parse_image_pins(COMPOSE)

    def boom(repo_tag):
        if repo_tag == "ollama/ollama:latest":
            raise update.UpdateError("registry down")
        return "sha256:" + {"docker.litellm.ai/berriai/litellm-database:main-stable": "bbbb",
                            "postgres:15-alpine": "cccc",
                            "ghcr.io/open-webui/open-webui:main": "dddd"}[repo_tag]

    res = {r.pin.service: r for r in update.plan_image_updates(pins, boom)}
    assert res["ollama"].status == "error" and res["ollama"].new_digest is None
    assert res["litellm"].status == "up-to-date"


def test_rewrite_compose_swaps_only_newer_digests_and_preserves_comments():
    pins = update.parse_image_pins(COMPOSE)
    results = update.plan_image_updates(pins, lambda rt:
        "sha256:NEW" if rt == "ollama/ollama:latest" else
        {"docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
         "postgres:15-alpine": "sha256:cccc",
         "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}[rt])
    out = update.rewrite_compose(COMPOSE, results)
    assert "ollama/ollama:latest@sha256:NEW" in out
    assert "sha256:aaaa" not in out
    assert "# pinned 2026-06-14" in out          # comments preserved
    assert "sha256:bbbb" in out                  # unchanged pins untouched


def test_rewrite_compose_raises_if_old_ref_not_unique():
    dup = COMPOSE + "  dupe:\n    image: ollama/ollama:latest@sha256:aaaa\n"
    pins = update.parse_image_pins(dup)
    results = update.plan_image_updates(pins, lambda rt: "sha256:NEW")
    try:
        update.rewrite_compose(dup, results)
        assert False, "expected a non-unique-ref error"
    except update.UpdateError:
        pass


DOCKERFILE = """\
FROM ubuntu:22.04@sha256:4f83
ARG LLAMA_SWAP_VERSION=224
ARG LLAMA_SWAP_SHA256=d62c1d140a6ba3482c50b19f254b085f116a1d3d282a9d0f1ff4113b8a56f4cd
RUN wget llama-swap_${LLAMA_SWAP_VERSION}_linux_arm64.tar.gz
"""


def test_parse_llamaswap_pin():
    pin = update.parse_llamaswap_pin(DOCKERFILE)
    assert pin.version == 224
    assert pin.sha256 == "d62c1d140a6ba3482c50b19f254b085f116a1d3d282a9d0f1ff4113b8a56f4cd"


def test_plan_llamaswap_update_newer():
    plan = update.plan_llamaswap_update(224, "v226", sha_of=lambda v: "f" * 64)
    assert plan["status"] == "newer" and plan["latest"] == 226 and plan["new_sha"] == "f" * 64


def test_plan_llamaswap_update_not_newer_does_not_downgrade():
    assert update.plan_llamaswap_update(224, "v224", sha_of=lambda v: "x")["status"] == "up-to-date"
    assert update.plan_llamaswap_update(224, "v223", sha_of=lambda v: "x")["status"] == "older"
    # sha_of must not even be called when not newer:
    update.plan_llamaswap_update(224, "v224", sha_of=lambda v: (_ for _ in ()).throw(AssertionError))


def test_rewrite_llamaswap_bumps_both_args():
    out = update.rewrite_llamaswap(DOCKERFILE, 226, "a" * 64)
    assert "ARG LLAMA_SWAP_VERSION=226" in out
    assert "ARG LLAMA_SWAP_SHA256=" + "a" * 64 in out
    assert "=224" not in out


def test_parse_llamaswap_pin_raises_when_args_missing():
    with pytest.raises(update.UpdateError):
        update.parse_llamaswap_pin("FROM ubuntu:22.04\nRUN echo hi\n")


def test_plan_llamaswap_update_raises_on_malformed_tag():
    with pytest.raises(update.UpdateError):
        update.plan_llamaswap_update(224, "v226-beta", sha_of=lambda v: "x")


def test_plan_image_updates_malformed_digest_is_error_not_newer():
    # a resolver returning a non-sha256 value must be 'error', never 'newer'
    pins = update.parse_image_pins(COMPOSE)
    res = {r.pin.service: r for r in update.plan_image_updates(pins, lambda rt: "not-a-digest")}
    assert res["ollama"].status == "error" and res["ollama"].new_digest is None


def test_manifest_digest_extracts_top_level_index_digest():
    # imagetools `{{json .Manifest}}` output: top-level .digest is the pin digest;
    # the per-arch manifests[].digest must NOT be picked.
    blob = ('{"schemaVersion":2,"mediaType":"application/vnd.oci.image.index.v1+json",'
            '"digest":"sha256:INDEX","size":10301,'
            '"manifests":[{"digest":"sha256:ARCH","platform":{"architecture":"arm64"}}]}')
    assert update._manifest_digest(blob) == "sha256:INDEX"


def test_manifest_digest_raises_on_garbage():
    with pytest.raises(update.UpdateError):
        update._manifest_digest("Name: docker.io/library/postgres\nDigest: sha256:xyz\n")
    with pytest.raises(update.UpdateError):
        update._manifest_digest('{"size": 10}')  # no digest key


import os
import types


def _settings(ref="7852e50e4"):
    return types.SimpleNamespace(vllm=types.SimpleNamespace(vllm_ref=ref))


def _fake_deps(resolved, latest_tag="v224", sha="e" * 64, calls=None):
    calls = calls if calls is not None else {}
    return update.Deps(
        resolve_digest=lambda rt: resolved[rt],
        latest_release=lambda repo: latest_tag,
        release_sha256=lambda repo, v: sha,
        docker_pull=lambda root, services: calls.setdefault("pull", []).append(services),
        docker_build=lambda root, services: calls.setdefault("build", []).append(services),
    )


def _write_repo(tmp_path):
    (tmp_path / "docker-compose.yml").write_text(COMPOSE.replace("sha256:aaaa",
        "sha256:" + "0" * 8))  # ollama starts at sha256:00000000
    (tmp_path / "llama-swap").mkdir()
    (tmp_path / "llama-swap" / "llama-swap.Dockerfile").write_text(DOCKERFILE)
    return tmp_path


def test_run_check_writes_nothing_and_runs_no_docker(tmp_path, capsys):
    _write_repo(tmp_path)
    calls = {}
    resolved = {"ollama/ollama:latest": "sha256:" + "0" * 8,  # up to date
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}
    rc = update.run(str(tmp_path), _settings(), check=True,
                    deps=_fake_deps(resolved, latest_tag="v226", calls=calls))
    assert rc == 0
    assert calls == {}  # no pull/build
    assert "0" * 8 in (tmp_path / "docker-compose.yml").read_text()
    assert "VERSION=224" in (tmp_path / "llama-swap" / "llama-swap.Dockerfile").read_text()
    out = capsys.readouterr().out
    assert "llama-swap" in out and "226" in out  # report shows the available bump


def test_run_apply_bumps_pins_and_pulls_builds(tmp_path):
    _write_repo(tmp_path)
    calls = {}
    resolved = {"ollama/ollama:latest": "sha256:NEWHASH",       # newer
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}
    rc = update.run(str(tmp_path), _settings(), check=False,
                    deps=_fake_deps(resolved, latest_tag="v226", sha="a" * 64, calls=calls))
    assert rc == 0
    compose = (tmp_path / "docker-compose.yml").read_text()
    assert "ollama/ollama:latest@sha256:NEWHASH" in compose
    df = (tmp_path / "llama-swap" / "llama-swap.Dockerfile").read_text()
    assert "VERSION=226" in df and "SHA256=" + "a" * 64 in df
    assert calls["pull"] == [["ollama"]]          # only the changed image service
    assert calls["build"] == [["llama-swap"]]


def test_run_failsoft_image_error_leaves_pin(tmp_path):
    _write_repo(tmp_path)
    def resolve(rt):
        if rt == "ollama/ollama:latest":
            raise update.UpdateError("down")
        return {"docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}[rt]
    deps = update.Deps(resolve, lambda r: "v224", lambda r, v: "x", lambda root, s: None, lambda root, s: None)
    rc = update.run(str(tmp_path), _settings(), check=False, deps=deps)
    assert rc == 0
    assert "0" * 8 in (tmp_path / "docker-compose.yml").read_text()  # ollama pin untouched


def test_run_apply_error_returns_1_not_crash(tmp_path):
    # a duplicated compose pin makes rewrite_compose raise UpdateError; run() must
    # catch it and return 1 (clean coded exit), not propagate a traceback.
    _write_repo(tmp_path)
    compose = tmp_path / "docker-compose.yml"
    compose.write_text(compose.read_text() +
                       "  dupe:\n    image: ollama/ollama:latest@sha256:" + "0" * 8 + "\n")
    calls = {}
    resolved = {"ollama/ollama:latest": "sha256:NEWHASH",
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}
    rc = update.run(str(tmp_path), _settings(), check=False,
                    deps=_fake_deps(resolved, latest_tag="v224", calls=calls))
    assert rc == 1
    assert calls == {}  # failed at rewrite, before any docker call


def test_run_apply_all_uptodate_writes_nothing_and_no_docker(tmp_path, capsys):
    _write_repo(tmp_path)
    calls = {}
    resolved = {"ollama/ollama:latest": "sha256:" + "0" * 8,  # all current
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}
    rc = update.run(str(tmp_path), _settings(), check=False,
                    deps=_fake_deps(resolved, latest_tag="v224", calls=calls))  # llama-swap also current
    assert rc == 0
    assert calls == {}  # idle apply run makes no docker calls
    assert "0" * 8 in (tmp_path / "docker-compose.yml").read_text()
    assert "VERSION=224" in (tmp_path / "llama-swap" / "llama-swap.Dockerfile").read_text()
    assert "Everything up to date." in capsys.readouterr().out


def test_run_notes_calls_render(tmp_path, monkeypatch):
    _write_repo(tmp_path)
    import sparkyard.notes as notes_mod
    seen = {}
    monkeypatch.setattr(notes_mod, "render_notes",
                        lambda root, imgs, ls, **kw: seen.update(root=root, model=kw.get("model")))
    resolved = {"ollama/ollama:latest": "sha256:" + "0" * 8,
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}
    rc = update.run(str(tmp_path), _settings(), check=True, notes=True, model="m1",
                    deps=_fake_deps(resolved, latest_tag="v224"))
    assert rc == 0
    assert seen["root"] == str(tmp_path) and seen["model"] == "m1"


def test_run_without_notes_does_not_render(tmp_path, monkeypatch):
    _write_repo(tmp_path)
    import sparkyard.notes as notes_mod
    called = []
    monkeypatch.setattr(notes_mod, "render_notes", lambda *a, **k: called.append(1))
    resolved = {"ollama/ollama:latest": "sha256:" + "0" * 8,
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}
    update.run(str(tmp_path), _settings(), check=True, deps=_fake_deps(resolved, latest_tag="v224"))
    assert called == []


def test_run_notes_passes_vllm_ref(tmp_path, monkeypatch):
    _write_repo(tmp_path)
    import sparkyard.notes as notes_mod
    seen = {}
    monkeypatch.setattr(notes_mod, "render_notes",
                        lambda root, imgs, ls, **kw: seen.update(vllm_ref=kw.get("vllm_ref")))
    resolved = {"ollama/ollama:latest": "sha256:" + "0" * 8,
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}
    update.run(str(tmp_path), _settings(ref="abc123"), check=True, notes=True,
               deps=_fake_deps(resolved, latest_tag="v224"))
    assert seen["vllm_ref"] == "abc123"
