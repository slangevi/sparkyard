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


def test_deps_has_sourcebuilt_fields():
    assert update.Deps._fields == (
        "resolve_digest", "latest_release", "release_sha256",
        "docker_pull", "docker_build",
        "resolve_head", "commits_behind", "docker_build_arg",
    )


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


def _settings(ref="7852e50e4", clone="/clone"):
    return types.SimpleNamespace(
        vllm=types.SimpleNamespace(vllm_ref=ref, clone_path=clone))


def _fake_deps(resolved, latest_tag="v224", sha="e" * 64, calls=None,
               head="h" * 40, behind=0):
    calls = calls if calls is not None else {}
    return update.Deps(
        resolve_digest=lambda rt: resolved[rt],
        latest_release=lambda repo: latest_tag,
        release_sha256=lambda repo, v: sha,
        docker_pull=lambda root, services: calls.setdefault("pull", []).append(services),
        docker_build=lambda root, services: calls.setdefault("build", []).append(services),
        resolve_head=lambda repo, branch: head,
        commits_behind=lambda repo, base, hd: behind,
        docker_build_arg=lambda root, service, build_args:
            calls.setdefault("build_arg", []).append((service, build_args)) or 0,
    )


LLAMACPP_DF_REPO = """\
FROM nvidia/cuda:13.1.0-devel@sha256:7f32
WORKDIR /app
# llama.cpp pinned 2026-06-01; bump via `sparkyard update llama-cpp`
ARG LLAMA_CPP_REF=oldsha000
RUN git clone https://github.com/ggml-org/llama.cpp src \\
 && git -C src checkout ${LLAMA_CPP_REF}
"""


PROVENANCE = ("## Pinned refs (built 2026-06-11)\n\n"
              "| Component  | Git commit  | Built artifact |\n"
              "|------------|-------------|----------------|\n"
              "| vLLM       | `7852e50e4` | `vllm-old.whl` |\n"
              "| FlashInfer | `28406af5`  | `flashinfer_python-0.6.13` |\n")


def _write_repo(tmp_path):
    (tmp_path / "docker-compose.yml").write_text(COMPOSE.replace("sha256:aaaa",
        "sha256:" + "0" * 8))  # ollama starts at sha256:00000000
    (tmp_path / "llama-swap").mkdir()
    (tmp_path / "llama-swap" / "llama-swap.Dockerfile").write_text(DOCKERFILE)
    (tmp_path / "llama-cpp").mkdir()
    (tmp_path / "llama-cpp" / "llama-cpp.Dockerfile").write_text(LLAMACPP_DF_REPO)
    (tmp_path / "settings.local.yaml").write_text("llm_root: /srv\nrepo_path: /r\n")
    (tmp_path / "tools" / "sparkyard").mkdir(parents=True)
    (tmp_path / "tools" / "sparkyard" / "settings.py").write_text(
        'DEFAULT_VLLM_REF = "7852e50e4"\n')
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm" / "VLLM_NODE_PROVENANCE.md").write_text(PROVENANCE)
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
    deps = update.Deps(resolve, lambda r: "v224", lambda r, v: "x",
                       lambda root, s: None, lambda root, s: None,
                       lambda repo, b: "h" * 40, lambda repo, b, h: 0,
                       lambda root, s, ba: 0)
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


def _recording_deps(resolved, latest_tag="v224", sha="e" * 64, calls=None):
    """Like _fake_deps but records every repo:tag the resolver is asked for, so a
    test can assert that unselected images are never queried."""
    calls = calls if calls is not None else {}
    asked = calls.setdefault("resolved_keys", [])
    def resolve(rt):
        asked.append(rt)
        return resolved[rt]
    return update.Deps(
        resolve_digest=resolve,
        latest_release=lambda repo: calls.setdefault("latest", []).append(repo) or latest_tag,
        release_sha256=lambda repo, v: sha,
        docker_pull=lambda root, services: calls.setdefault("pull", []).append(services),
        docker_build=lambda root, services: calls.setdefault("build", []).append(services),
        resolve_head=lambda repo, branch: "h" * 40,
        commits_behind=lambda repo, base, head: 0,
        docker_build_arg=lambda root, service, build_args: 0,
    )


def test_run_scoped_to_one_image_pulls_only_it(tmp_path):
    _write_repo(tmp_path)
    calls = {}
    resolved = {"ollama/ollama:latest": "sha256:NEWHASH"}  # only ollama is queried
    rc = update.run(str(tmp_path), _settings(), check=False, components=["ollama"],
                    deps=_recording_deps(resolved, calls=calls))
    assert rc == 0
    compose = (tmp_path / "docker-compose.yml").read_text()
    assert "ollama/ollama:latest@sha256:NEWHASH" in compose
    assert "sha256:bbbb" in compose and "sha256:cccc" in compose  # others untouched
    assert calls["resolved_keys"] == ["ollama/ollama:latest"]      # no other registry calls
    assert calls["pull"] == [["ollama"]]
    assert "build" not in calls and "latest" not in calls          # llama-swap untouched


def test_run_scoped_to_llamaswap_builds_only_it(tmp_path):
    _write_repo(tmp_path)
    calls = {}
    rc = update.run(str(tmp_path), _settings(), check=False, components=["llama-swap"],
                    deps=_recording_deps({}, latest_tag="v226", sha="a" * 64, calls=calls))
    assert rc == 0
    df = (tmp_path / "llama-swap" / "llama-swap.Dockerfile").read_text()
    assert "VERSION=226" in df and "SHA256=" + "a" * 64 in df
    assert calls["build"] == [["llama-swap"]]
    assert calls["resolved_keys"] == [] and "pull" not in calls    # no image work


def test_run_scoped_to_vllm_node_check_shows_only_note(tmp_path, capsys):
    _write_repo(tmp_path)
    calls = {}
    rc = update.run(str(tmp_path), _settings(), check=True, components=["vllm-node"],
                    deps=_recording_deps({}, calls=calls))
    assert rc == 0
    out = capsys.readouterr().out
    assert "vllm-node" in out
    assert "Component" not in out                                  # no empty table
    assert calls.get("resolved_keys") == [] and "latest" not in calls


def test_run_unknown_component_fails_closed(tmp_path, capsys):
    _write_repo(tmp_path)
    calls = {}
    rc = update.run(str(tmp_path), _settings(), check=True, components=["bogus"],
                    deps=_recording_deps({}, calls=calls))
    assert rc == 2
    err = capsys.readouterr().err
    assert "bogus" in err and "valid" in err
    assert calls == {} or calls.get("resolved_keys") == []         # no network before abort


LLAMACPP_DF = """\
FROM nvidia/cuda:13.1.0-devel@sha256:7f32
WORKDIR /app
# llama.cpp pinned 2026-06-22; bump via `sparkyard update llama-cpp`
ARG LLAMA_CPP_REF=abc1234def5678
RUN git clone https://github.com/ggml-org/llama.cpp src \\
 && git -C src checkout ${LLAMA_CPP_REF}
"""


def test_parse_llamacpp_pin():
    assert update.parse_llamacpp_pin(LLAMACPP_DF) == "abc1234def5678"


def test_parse_llamacpp_pin_raises_when_arg_missing():
    with pytest.raises(update.UpdateError):
        update.parse_llamacpp_pin("FROM ubuntu\nRUN echo hi\n")


def test_rewrite_llamacpp_ref_bumps_arg_and_comment():
    out = update.rewrite_llamacpp_ref(LLAMACPP_DF, "f" * 40, "2026-07-01")
    assert "ARG LLAMA_CPP_REF=" + "f" * 40 in out
    assert "abc1234def5678" not in out
    assert "# llama.cpp pinned 2026-07-01;" in out
    assert "2026-06-22" not in out


def test_rewrite_llamacpp_ref_raises_when_arg_missing():
    with pytest.raises(update.UpdateError):
        update.rewrite_llamacpp_ref("FROM ubuntu\n", "f" * 40, "2026-07-01")


def _img(service, repo, tag, digest, new_digest, status):
    pin = update.ImagePin(service, f"{repo}:{tag}@{digest}", repo, tag, digest)
    return update.ImageResult(pin, new_digest, status)


def test_format_report_omits_llamaswap_row_when_no_plan():
    r = _img("ollama", "ollama/ollama", "latest", "sha256:aaaa", None, "up-to-date")
    out = update.format_report([r], None, None, None)
    assert "ollama" in out
    assert "llama-swap" not in out


def test_format_report_omits_table_when_only_a_note():
    out = update.format_report([], None, None, "vllm-node : pinned vLLM ref X; ...")
    assert "Component" not in out          # no empty table header
    assert "vllm-node" in out


def test_format_report_full_output_unchanged():
    # Golden string: the no-filter path (all rows + ls row + both notes) must stay
    # byte-for-byte identical — exact equality locks that invariant, not just token
    # presence, so a future spacing/ordering regression fails here.
    r = _img("ollama", "ollama/ollama", "latest", "sha256:aaaa", "sha256:bbbb", "newer")
    ls_plan = {"current": 224, "latest": 226, "new_sha": "e" * 64, "status": "newer"}
    out = update.format_report([r], ls_plan, "llama-cpp : ...", "vllm-node : ...")
    assert out == (
        "\nComponent        Current     Latest      Status\n"
        "ollama           aaaa        bbbb        NEWER\n"
        "llama-swap       v224        v226        NEWER\n"
        "\nllama-cpp : ...\nvllm-node : ...\n"
    )


def test_plan_sourcebuilt_uptodate_when_head_starts_with_current():
    p = update.plan_sourcebuilt("7852e50e4", "7852e50e4abcdef0000", 0)
    assert p["status"] == "up-to-date"


def test_plan_sourcebuilt_newer():
    p = update.plan_sourcebuilt("7852e50e4", "ffffffff0000", 12)
    assert p["status"] == "newer" and p["total"] == 12 and p["head"] == "ffffffff0000"


def test_format_sourcebuilt_note_newer_mentions_count_and_command():
    note = update.format_sourcebuilt_note(
        "vllm-node", "main", update.plan_sourcebuilt("7852e50e4", "ffff0000", 12))
    assert "12 commit" in note and "main" in note
    assert "sparkyard update vllm-node" in note


def test_format_sourcebuilt_note_uptodate():
    note = update.format_sourcebuilt_note(
        "llama-cpp", "master", update.plan_sourcebuilt("abc123", "abc123def", 0))
    assert "up to date" in note.lower()


def test_llamacpp_explicit_apply_builds_and_rewrites_on_success(tmp_path, capsys):
    _write_repo(tmp_path)
    calls = {}
    resolved = {"ollama/ollama:latest": "sha256:" + "0" * 8,
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}
    deps = _fake_deps(resolved, latest_tag="v224", calls=calls,
                      head="newsha111" + "0" * 31, behind=5)
    rc = update.run(str(tmp_path), _settings(), check=False, components=["llama-cpp"], deps=deps)
    assert rc == 0
    df = (tmp_path / "llama-cpp" / "llama-cpp.Dockerfile").read_text()
    assert "ARG LLAMA_CPP_REF=newsha111" in df and "oldsha000" not in df
    assert calls["build_arg"] == [("llama-server", {"LLAMA_CPP_REF": "newsha111" + "0" * 31})]


def test_llamacpp_explicit_apply_no_write_on_build_failure(tmp_path):
    _write_repo(tmp_path)
    resolved = {"ollama/ollama:latest": "sha256:" + "0" * 8,
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}
    deps = update.Deps(
        resolve_digest=lambda rt: resolved[rt], latest_release=lambda r: "v224",
        release_sha256=lambda r, v: "x", docker_pull=lambda root, s: None,
        docker_build=lambda root, s: None,
        resolve_head=lambda repo, b: "newsha111" + "0" * 31,
        commits_behind=lambda repo, b, h: 5,
        docker_build_arg=lambda root, service, ba: 1)  # build FAILS
    rc = update.run(str(tmp_path), _settings(), check=False, components=["llama-cpp"], deps=deps)
    assert rc != 0
    assert "oldsha000" in (tmp_path / "llama-cpp" / "llama-cpp.Dockerfile").read_text()


def test_llamacpp_allmode_reports_but_does_not_build(tmp_path):
    _write_repo(tmp_path)
    calls = {}
    resolved = {"ollama/ollama:latest": "sha256:" + "0" * 8,
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}
    deps = _fake_deps(resolved, latest_tag="v224", calls=calls,
                      head="newsha111" + "0" * 31, behind=5)
    rc = update.run(str(tmp_path), _settings(), check=False, deps=deps)  # no components = all
    assert rc == 0
    assert "build_arg" not in calls  # report-only in all-mode
    assert "oldsha000" in (tmp_path / "llama-cpp" / "llama-cpp.Dockerfile").read_text()


def test_llamacpp_check_reports_no_build(tmp_path):
    _write_repo(tmp_path)
    calls = {}
    resolved = {"ollama/ollama:latest": "sha256:" + "0" * 8,
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}
    deps = _fake_deps(resolved, latest_tag="v224", calls=calls,
                      head="oldsha000abc", behind=0)  # up to date
    rc = update.run(str(tmp_path), _settings(), check=True, components=["llama-cpp"], deps=deps)
    assert rc == 0 and "build_arg" not in calls


def _clone_readers():
    files = {"/clone/wheels/.vllm-commit": "newsha111\n",
             "/clone/wheels/.flashinfer-commit": "ffaa22\n"}
    entries = {"/clone/wheels": [
        "vllm-0.23.0.dev1+gnewsha111.d20260701-cp312-cp312-linux_aarch64.whl"]}
    return (lambda p: files[p]), (lambda p: entries[p])


def test_vllmnode_explicit_apply_syncs_four_refs_on_success(tmp_path):
    _write_repo(tmp_path)
    resolved = {"ollama/ollama:latest": "sha256:" + "0" * 8,
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}
    deps = _fake_deps(resolved, latest_tag="v224", head="newsha111" + "0" * 31, behind=9)
    read_text, listdir = _clone_readers()
    builds = []
    rc = update.run(str(tmp_path), _settings(), check=False, components=["vllm-node"],
                    deps=deps, build_vllm=lambda ref: builds.append(ref) or 0,
                    read_text=read_text, listdir=listdir)
    assert rc == 0 and builds == ["newsha111" + "0" * 31]
    assert "vllm_ref: newsha111" in (tmp_path / "settings.local.yaml").read_text()
    assert 'DEFAULT_VLLM_REF = "newsha111' in (
        tmp_path / "tools" / "sparkyard" / "settings.py").read_text()
    prov = (tmp_path / "vllm" / "VLLM_NODE_PROVENANCE.md").read_text()
    assert "`newsha111`" in prov and "built 2026-07-01" in prov


def test_vllmnode_apply_no_writes_on_build_failure(tmp_path):
    _write_repo(tmp_path)
    resolved = {"ollama/ollama:latest": "sha256:" + "0" * 8,
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}
    deps = _fake_deps(resolved, latest_tag="v224", head="newsha111" + "0" * 31, behind=9)
    read_text, listdir = _clone_readers()
    rc = update.run(str(tmp_path), _settings(), check=False, components=["vllm-node"],
                    deps=deps, build_vllm=lambda ref: 1,  # build FAILS
                    read_text=read_text, listdir=listdir)
    assert rc != 0
    assert "7852e50e4" in (tmp_path / "settings.local.yaml").read_text() or \
           "vllm_ref" not in (tmp_path / "settings.local.yaml").read_text()
    assert 'DEFAULT_VLLM_REF = "7852e50e4"' in (
        tmp_path / "tools" / "sparkyard" / "settings.py").read_text()


def test_llamacpp_missing_dockerfile_failsoft_no_build(tmp_path, capsys):
    """If llama-cpp.Dockerfile is absent, _handle_llamacpp fails soft:
    run() does not raise and does not build."""
    _write_repo(tmp_path)
    # Remove the Dockerfile so the open() call in _handle_llamacpp fails.
    (tmp_path / "llama-cpp" / "llama-cpp.Dockerfile").unlink()
    calls = {}
    resolved = {"ollama/ollama:latest": "sha256:" + "0" * 8,
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}
    deps = _fake_deps(resolved, latest_tag="v224", calls=calls,
                      head="newsha111" + "0" * 31, behind=5)
    rc = update.run(str(tmp_path), _settings(), check=False, components=["llama-cpp"], deps=deps)
    assert rc == 0
    assert "build_arg" not in calls  # no build attempted


def test_vllmnode_allmode_reports_but_does_not_build(tmp_path):
    _write_repo(tmp_path)
    resolved = {"ollama/ollama:latest": "sha256:" + "0" * 8,
                "docker.litellm.ai/berriai/litellm-database:main-stable": "sha256:bbbb",
                "postgres:15-alpine": "sha256:cccc",
                "ghcr.io/open-webui/open-webui:main": "sha256:dddd"}
    deps = _fake_deps(resolved, latest_tag="v224", head="newsha111" + "0" * 31, behind=9)
    builds = []
    rc = update.run(str(tmp_path), _settings(), check=False, deps=deps,
                    build_vllm=lambda ref: builds.append(ref) or 0)
    assert rc == 0 and builds == []  # report-only in all-mode
    assert 'DEFAULT_VLLM_REF = "7852e50e4"' in (
        tmp_path / "tools" / "sparkyard" / "settings.py").read_text()
