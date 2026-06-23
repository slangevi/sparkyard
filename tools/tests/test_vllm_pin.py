import pytest
from sparkyard import vllm_pin


def _readers(files, entries):
    return (lambda p: files[p]), (lambda p: entries[p])


def test_read_built_refs_parses_all_fields():
    files = {
        "/c/wheels/.vllm-commit": "7852e50e4\n",
        "/c/wheels/.flashinfer-commit": "28406af5\n",
    }
    entries = {"/c/wheels": ["junk.txt",
        "vllm-0.22.1rc1.dev403+g7852e50e4.d20260611-cp312-cp312-linux_aarch64.whl"]}
    read_text, listdir = _readers(files, entries)
    got = vllm_pin.read_built_refs("/c", read_text, listdir)
    assert got.vllm == "7852e50e4"
    assert got.flashinfer == "28406af5"
    assert got.wheel.startswith("vllm-0.22.1rc1.dev403+g7852e50e4")
    assert got.built_date == "2026-06-11"


def test_read_built_refs_missing_wheel_is_blank():
    files = {"/c/wheels/.vllm-commit": "abc\n", "/c/wheels/.flashinfer-commit": "def\n"}
    entries = {"/c/wheels": ["readme.md"]}
    read_text, listdir = _readers(files, entries)
    got = vllm_pin.read_built_refs("/c", read_text, listdir)
    assert got.vllm == "abc" and got.wheel == "" and got.built_date == ""


def test_upsert_settings_local_ref_inserts_block_when_absent():
    src = "llm_root: /srv/llm\nrepo_path: /home/x/sparkyard\n"
    out = vllm_pin.upsert_settings_local_ref(src, "deadbeef")
    assert "vllm:" in out and "vllm_ref: deadbeef" in out
    assert "llm_root: /srv/llm" in out  # untouched


def test_upsert_settings_local_ref_replaces_existing_value():
    src = "repo_path: /r\nvllm:\n  upstream: u\n  vllm_ref: oldsha  # pin\n"
    out = vllm_pin.upsert_settings_local_ref(src, "newsha")
    assert "vllm_ref: newsha" in out and "oldsha" not in out
    assert "upstream: u" in out and "# pin" in out  # comment + sibling preserved


def test_upsert_settings_local_ref_inserts_key_into_existing_block():
    src = "vllm:\n  upstream: u\n"
    out = vllm_pin.upsert_settings_local_ref(src, "newsha")
    assert "vllm_ref: newsha" in out and "upstream: u" in out


def test_rewrite_default_ref():
    src = 'DEFAULT_VLLM_REF = "7852e50e4"\n'
    assert vllm_pin.rewrite_default_ref(src, "abc999") == 'DEFAULT_VLLM_REF = "abc999"\n'


def test_rewrite_default_ref_raises_when_absent():
    with pytest.raises(vllm_pin.UpdateError):
        vllm_pin.rewrite_default_ref("X = 1\n", "abc")


_PROV_WITH_REPRODUCE = (
    "## Pinned refs (built 2026-06-11)\n\n"
    "| Component  | Git commit  | Built artifact |\n"
    "|------------|-------------|----------------|\n"
    "| vLLM       | `7852e50e4` | `vllm-old.whl` |\n"
    "| FlashInfer | `28406af5`  | `flashinfer_python-0.6.13` |\n"
    "\n## Reproduce\n\n"
    "The pin lives in `settings.local.yaml` (`vllm.vllm_ref`, default `7852e50e4`);\n"
    "this file mirrors it.\n\n"
    "```bash\n"
    "./build-and-copy.sh --vllm-ref 7852e50e4\n"
    "./build-and-copy.sh --tf5 --vllm-ref 7852e50e4\n"
    "```\n"
)


def test_rewrite_provenance_updates_rows_and_date():
    prov = ("## Pinned refs (built 2026-06-11)\n\n"
            "| Component  | Git commit  | Built artifact |\n"
            "|------------|-------------|----------------|\n"
            "| vLLM       | `7852e50e4` | `vllm-old.whl` |\n"
            "| FlashInfer | `28406af5`  | `flashinfer_python-0.6.13` |\n")
    built = vllm_pin.BuiltRefs("aaa111", "bbb222", "vllm-new.whl", "2026-07-01")
    out = vllm_pin.rewrite_provenance(prov, built)
    assert "built 2026-07-01" in out and "built 2026-06-11" not in out
    assert "`aaa111`" in out and "`7852e50e4`" not in out
    assert "`bbb222`" in out and "`28406af5`" not in out
    assert "`vllm-new.whl`" in out


def test_rewrite_provenance_updates_reproduce_refs():
    """The two --vllm-ref args and the default `ref` token are updated; no old ref remains."""
    built = vllm_pin.BuiltRefs("newref999", "flashXX", "vllm-new.whl", "2026-07-01")
    out = vllm_pin.rewrite_provenance(_PROV_WITH_REPRODUCE, built)
    assert "7852e50e4" not in out
    assert out.count("--vllm-ref newref999") == 2
    assert "default `newref999`" in out


def test_rewrite_provenance_reproduce_refs_absent_does_not_raise():
    """Table present but no Reproduce section: must NOT raise (best-effort only)."""
    prov = ("## Pinned refs (built 2026-06-11)\n\n"
            "| Component  | Git commit  | Built artifact |\n"
            "|------------|-------------|----------------|\n"
            "| vLLM       | `oldref` | `vllm-old.whl` |\n"
            "| FlashInfer | `flashold` | `flashinfer_python-0.6.13` |\n")
    built = vllm_pin.BuiltRefs("newref", "flashnew", "vllm-new.whl", "2026-07-01")
    out = vllm_pin.rewrite_provenance(prov, built)   # must not raise
    assert "oldref" not in out and "newref" in out


def test_rewrite_provenance_raises_table_present_but_flash_missing():
    """Table with vLLM row but FlashInfer row absent → still raises (fail-closed)."""
    prov = ("## Pinned refs (built 2026-06-11)\n\n"
            "| Component  | Git commit  | Built artifact |\n"
            "|------------|-------------|----------------|\n"
            "| vLLM       | `oldref` | `vllm-old.whl` |\n")
    with pytest.raises(vllm_pin.UpdateError):
        vllm_pin.rewrite_provenance(prov, vllm_pin.BuiltRefs("a", "b", "w", "2026-07-01"))


def test_rewrite_provenance_raises_when_table_missing():
    with pytest.raises(vllm_pin.UpdateError):
        vllm_pin.rewrite_provenance("no table here\n",
                                    vllm_pin.BuiltRefs("a", "b", "w", "2026-07-01"))
