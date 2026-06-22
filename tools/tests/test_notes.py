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


def test_build_summary_prompt_includes_bodies_and_truncates():
    p = notes.build_summary_prompt(notes.llamaswap_notes(RELEASES, 224, 226))
    assert "Add foo. Fix bar." in p and "Speed up baz." in p
    assert "v226" in p
    big = [notes.Release("v999", 999, "x" * 50000, "u")]
    assert len(notes.build_summary_prompt(big)) <= 12000 + 300  # body capped + ~260-char instruction


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
    return notes.NotesDeps(http_get_json, gateway_chat, list_models)


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
