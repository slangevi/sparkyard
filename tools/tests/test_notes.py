from sparkyard import notes


RELEASES = [
    {"tag_name": "v226", "body": "Add foo. Fix bar.", "html_url": "https://x/226"},
    {"tag_name": "v225", "body": "Speed up baz.", "html_url": "https://x/225"},
    {"tag_name": "v224", "body": "old", "html_url": "https://x/224"},
    {"tag_name": "nightly", "body": "skip", "html_url": "https://x/n"},
]


def test_llamaswap_notes_filters_range_newest_first():
    out = notes.llamaswap_notes(RELEASES, 224, 226)
    assert [r.version for r in out] == [226, 225]          # (224, 226], newest-first
    assert out[0].tag == "v226" and out[0].url == "https://x/226"
    assert out[0].body == "Add foo. Fix bar."


def test_llamaswap_notes_empty_when_uptodate():
    assert notes.llamaswap_notes(RELEASES, 226, 226) == []


def test_build_summary_prompt_includes_source_body_and_recommendation():
    body = notes.releases_body(notes.llamaswap_notes(RELEASES, 224, 226))
    p = notes.build_summary_prompt("llama-swap releases v224→v226", body)
    assert "llama-swap releases v224→v226" in p           # source label
    assert "Add foo. Fix bar." in p and "Speed up baz." in p  # bodies
    assert "Recommendation: Apply" in p and "Recommendation: Review first" in p
    assert len(notes.build_summary_prompt("x", "y" * 50000)) <= 12000 + 600  # body capped


def test_commits_body_formats_subjects():
    assert notes.commits_body(["add foo", "fix bar"]) == "- add foo\n- fix bar"


def test_image_note_formats_delta_and_link():
    n = notes.image_note("litellm", "sha256:8402d2372a", "sha256:0e8dfd6910")
    assert "litellm: 8402d237→0e8dfd69" in n
    assert "github.com/BerriAI/litellm/releases" in n


def test_image_note_unknown_service_has_no_link():
    assert notes.image_note("mystery", "sha256:aaaaaaaa", "sha256:bbbbbbbb") == "mystery: aaaaaaaa→bbbbbbbb"


def test_image_note_optional_version():
    assert "(v15.18)" in notes.image_note("litellm-db", "sha256:dddddddd", "sha256:eeeeeeee", version="15.18")


def test_read_master_key_from_secrets_env(tmp_path):
    (tmp_path / "secrets.env").write_text('FOO=1\nLITELLM_MASTER_KEY="sk-abc123"\n')
    assert notes.read_master_key(str(tmp_path)) == "sk-abc123"


def test_read_master_key_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    assert notes.read_master_key(str(tmp_path)) is None


import types


def _img(service, old, new, status="newer"):
    pin = types.SimpleNamespace(service=service, digest=old)
    return types.SimpleNamespace(pin=pin, new_digest=new, status=status)


def _deps(*, releases=RELEASES, chat=None, models="m1", calls=None):
    calls = calls if calls is not None else {}
    def http_get_json(url, headers=None):
        calls.setdefault("get", []).append(url)
        return releases
    def gateway_chat(prompt, *, base_url, key, model):
        calls.update(prompt=prompt, model=model, base_url=base_url, key=key)
        if chat is None:
            raise RuntimeError("no gateway")
        return chat
    def list_models(base_url, key):
        return models
    def image_labels(ref):
        return {}
    return notes.NotesDeps(http_get_json, gateway_chat, list_models, image_labels)


def test_render_summarizes_via_gateway(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-x")
    ls_plan = {"current": 224, "latest": 226, "status": "newer"}
    calls = {}
    notes.render_notes(str(tmp_path), [], ls_plan, model="big-model",
                       deps=_deps(chat="• added foo\n• fixed bar", calls=calls))
    out = capsys.readouterr().out
    assert "llama-swap v224 → v226" in out
    assert "added foo" in out and "fixed bar" in out
    assert "via big-model" in out
    assert calls["model"] == "big-model"          # --model honored
    assert "Add foo. Fix bar." in calls["prompt"]  # notes fed to the gateway


def test_render_falls_back_to_raw_when_gateway_errors(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-x")
    ls_plan = {"current": 224, "latest": 226, "status": "newer"}
    notes.render_notes(str(tmp_path), [], ls_plan, model="m", deps=_deps(chat=None))
    out = capsys.readouterr().out
    assert "raw notes" in out.lower()
    assert "v226" in out and "Add foo. Fix bar." in out   # raw body shown


def test_render_no_model_from_list_uses_raw_notes(tmp_path, capsys, monkeypatch):
    # key present but the gateway lists no models and no --model given → raw notes,
    # and the gateway is NOT called with model=None.
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-x")
    ls_plan = {"current": 224, "latest": 226, "status": "newer"}
    calls = {}
    notes.render_notes(str(tmp_path), [], ls_plan, deps=_deps(models=None, chat="x", calls=calls))
    out = capsys.readouterr().out
    assert "raw notes" in out.lower() and "start the stack" in out
    assert "prompt" not in calls          # gateway_chat never invoked


def test_render_no_key_uses_raw_notes(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)   # no secrets.env in tmp
    ls_plan = {"current": 224, "latest": 226, "status": "newer"}
    notes.render_notes(str(tmp_path), [], ls_plan, deps=_deps(chat="x"))
    out = capsys.readouterr().out
    assert "raw notes" in out.lower() and "Speed up baz." in out


def test_render_images_oneliner_and_nothing(tmp_path, capsys):
    imgs = [_img("litellm", "sha256:8402d2372a", "sha256:0e8dfd6910")]
    notes.render_notes(str(tmp_path), imgs, {"status": "up-to-date"}, deps=_deps())
    out = capsys.readouterr().out
    assert "litellm: 8402d237→0e8dfd69" in out
    # nothing newer at all -> "Nothing to summarize."
    capsys.readouterr()
    notes.render_notes(str(tmp_path), [], {"status": "up-to-date"}, deps=_deps())
    assert "Nothing to summarize" in capsys.readouterr().out


_COMPARE = {
    "total_commits": 3,
    "commits": [
        {"commit": {"message": "add A\n\ndetail"}},
        {"commit": {"message": "Merge pull request #9 from x"}},
        {"commit": {"message": "fix B"}},
    ],
}


def test_compare_commits_drops_merges_and_caps():
    subs, total = notes.compare_commits("o/r", "aaa", "bbb", lambda url: _COMPARE, cap=40)
    assert subs == ["add A", "fix B"] and total == 3


def test_compare_commits_cap_keeps_most_recent():
    many = {"total_commits": 100,
            "commits": [{"commit": {"message": f"c{i}"}} for i in range(100)]}
    subs, total = notes.compare_commits("o/r", "a", "b", lambda url: many, cap=5)
    assert subs == ["c95", "c96", "c97", "c98", "c99"] and total == 100


_IMAGE_JSON = {"linux/arm64": {"config": {"Labels": {
    "org.opencontainers.image.revision": "33df5891",
    "org.opencontainers.image.source": "https://github.com/BerriAI/litellm",
}}}}


def test_image_revision_extracts_sha_and_repo():
    assert notes.image_revision(_IMAGE_JSON) == ("33df5891", "BerriAI/litellm")


def test_image_revision_none_when_no_provenance():
    assert notes.image_revision({"linux/amd64": {"config": {"Labels": {
        "org.opencontainers.image.version": "24.04"}}}}) is None


def test_github_repo_parses_variants():
    assert notes._github_repo("https://github.com/o/r.git") == "o/r"
    assert notes._github_repo("https://github.com/o/r#frag") == "o/r"
    assert notes._github_repo("https://github.com/o/r/tree/main") == "o/r"
    assert notes._github_repo("https://gitlab.com/o/r") is None


def _deps2(*, releases=RELEASES, chat=None, models="m1", labels=None, calls=None):
    calls = calls if calls is not None else {}
    def http_get_json(url, headers=None):
        calls.setdefault("get", []).append(url)
        return _COMPARE if "/compare/" in url else releases
    def gateway_chat(prompt, *, base_url, key, model):
        calls.update(prompt=prompt, model=model, base_url=base_url, key=key)
        if chat is None:
            raise RuntimeError("no gateway")
        return chat
    def list_models(base_url, key):
        return models
    def image_labels(ref):
        calls.setdefault("inspect", []).append(ref)
        return (labels or {}).get(ref, {})
    return notes.NotesDeps(http_get_json, gateway_chat, list_models, image_labels)


def _imgres(service, repo, old, new):
    pin = types.SimpleNamespace(service=service, repo=repo, digest=old)
    return types.SimpleNamespace(pin=pin, new_digest=new, status="newer")


def test_render_image_summarizes_commits(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-x")
    r = _imgres("litellm", "docker.litellm.ai/berriai/litellm-database",
                "sha256:OLD", "sha256:NEW")
    labels = {
        "docker.litellm.ai/berriai/litellm-database@sha256:OLD":
            {"c": {"config": {"Labels": {
                "org.opencontainers.image.revision": "aaa",
                "org.opencontainers.image.source": "https://github.com/BerriAI/litellm"}}}},
        "docker.litellm.ai/berriai/litellm-database@sha256:NEW":
            {"c": {"config": {"Labels": {
                "org.opencontainers.image.revision": "bbb",
                "org.opencontainers.image.source": "https://github.com/BerriAI/litellm"}}}},
    }
    calls = {}
    notes.render_notes(str(tmp_path), [r], {"status": "up-to-date"},
                       deps=_deps2(chat="• did things\nRecommendation: Apply — routine",
                                   labels=labels, calls=calls))
    out = capsys.readouterr().out
    assert "litellm" in out and "did things" in out and "Recommendation: Apply" in out
    assert "/compare/aaa...bbb" in calls["get"][-1]
    assert "add A" in calls["prompt"]


def test_render_image_falls_back_to_oneliner_without_provenance(tmp_path, capsys):
    r = _imgres("ollama", "ollama/ollama", "sha256:8402d2372a", "sha256:0e8dfd6910")
    notes.render_notes(str(tmp_path), [r], {"status": "up-to-date"}, deps=_deps2(labels={}))
    out = capsys.readouterr().out
    assert "ollama: 8402d237→0e8dfd69" in out


def test_render_image_falls_back_to_oneliner_on_compare_error(tmp_path, capsys, monkeypatch):
    # labels present + valid, but the GitHub compare fetch errors → one-liner, no crash.
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-x")
    r = _imgres("litellm", "docker.litellm.ai/berriai/litellm-database",
                "sha256:8402d2372a", "sha256:0e8dfd6910")
    labels = {
        "docker.litellm.ai/berriai/litellm-database@sha256:8402d2372a":
            {"c": {"config": {"Labels": {
                "org.opencontainers.image.revision": "aaa",
                "org.opencontainers.image.source": "https://github.com/BerriAI/litellm"}}}},
        "docker.litellm.ai/berriai/litellm-database@sha256:0e8dfd6910":
            {"c": {"config": {"Labels": {
                "org.opencontainers.image.revision": "bbb",
                "org.opencontainers.image.source": "https://github.com/BerriAI/litellm"}}}},
    }
    deps = _deps2(labels=labels)
    def boom(url, headers=None):
        raise RuntimeError("network error")
    deps = deps._replace(http_get_json=boom)
    notes.render_notes(str(tmp_path), [r], {"status": "up-to-date"}, deps=deps)
    out = capsys.readouterr().out
    assert "litellm: 8402d237→0e8dfd69" in out          # one-liner fallback, no crash


def test_render_vllm_summarizes_commits_since_ref(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-x")
    calls = {}
    notes.render_notes(str(tmp_path), [], {"status": "up-to-date"}, vllm_ref="7852e50e4",
                       deps=_deps2(chat="• vllm stuff\nRecommendation: Review first — big jump",
                                   calls=calls))
    out = capsys.readouterr().out
    assert "vllm-node" in out and "vllm stuff" in out and "Recommendation: Review first" in out
    assert "/compare/7852e50e4...main" in calls["get"][-1]
    assert "3 commit(s) since" in out          # _COMPARE total_commits == 3


def test_render_vllm_failsoft_on_compare_error(tmp_path, capsys):
    deps = _deps2()
    def boom(url, headers=None):
        raise RuntimeError("gh down")
    deps = deps._replace(http_get_json=boom)
    notes.render_notes(str(tmp_path), [], {"status": "up-to-date"},
                       vllm_ref="7852e50e4", deps=deps)
    out = capsys.readouterr().out
    assert "vllm-node" in out and "sparkyard update vllm-node" in out  # report-only note, no crash


def test_resolve_head_returns_sha():
    from sparkyard import notes
    calls = {}
    def fake_get(url):
        calls["url"] = url
        return {"sha": "a1b2c3d4e5f6", "commit": {}}
    sha = notes.resolve_head("ggml-org/llama.cpp", "master", fake_get)
    assert sha == "a1b2c3d4e5f6"
    assert calls["url"] == "https://api.github.com/repos/ggml-org/llama.cpp/commits/master"


def test_resolve_head_raises_on_missing_sha():
    from sparkyard import notes
    import pytest
    with pytest.raises(notes.UpdateNotesError):
        notes.resolve_head("o/r", "main", lambda url: {"no_sha": 1})


def test_render_notes_includes_llamacpp_when_ref_given(capsys):
    from sparkyard import notes
    deps = notes.NotesDeps(
        http_get_json=lambda url: {"total_commits": 3, "commits": [
            {"commit": {"message": "fix: a thing"}}]},
        gateway_chat=lambda *a, **k: (_ for _ in ()).throw(Exception("no gw")),
        list_models=lambda *a, **k: [], image_labels=lambda *a, **k: {})
    notes.render_notes("/r", [], {}, llamacpp_ref="oldsha", deps=deps)
    out = capsys.readouterr().out
    assert "llama-cpp" in out
