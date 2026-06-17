import pytest
from click.testing import CliRunner

from sparkyard import cli

ALL_COMMANDS = [
    "init", "secrets", "render", "validate", "doctor", "build",
    "start", "stop", "add-model", "download", "vllm-node", "update", "bench",
]


def test_top_level_help_lists_all_commands():
    result = CliRunner().invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    out = result.output
    # Click renders a clean "Commands:" block (argparse rendered "positional arguments:")
    assert "Commands:" in out
    assert "SSOT-driven multi-engine LLM stack generator" in out
    for cmd in ALL_COMMANDS:
        assert cmd in out, f"{cmd} missing from help"


def test_commands_listed_in_lifecycle_order_not_alphabetical():
    out = CliRunner().invoke(cli.cli, ["--help"]).output
    # registration order is preserved: init before bench, render before validate
    assert out.index("\n  init") < out.index("\n  bench")
    assert out.index("\n  render") < out.index("\n  validate")


def test_vllm_node_help_shows_choices_and_description():
    result = CliRunner().invoke(cli.cli, ["vllm-node", "--help"])
    assert result.exit_code == 0
    assert "[base|tf5|mxfp4]" in result.output
    assert "Clone + build the vllm-node serving image(s)." in result.output


def test_bench_help_shows_mode_choices():
    result = CliRunner().invoke(cli.cli, ["bench", "--help"])
    assert result.exit_code == 0
    assert "[quality|speed]" in result.output


def test_dash_h_alias_works():
    result = CliRunner().invoke(cli.cli, ["-h"])
    assert result.exit_code == 0
    assert "Commands:" in result.output


def test_main_help_returns_zero_via_shim(capsys):
    # the preserved int-return contract: main(["--help"]) prints help and returns 0
    rc = cli.main(["--help"])
    assert rc == 0
    assert "Commands:" in capsys.readouterr().out


def test_main_translates_usage_error_to_systemexit_2():
    # the shim mirrors argparse: parse/usage errors raise SystemExit(2), not return an int
    with pytest.raises(SystemExit) as exc:
        cli.main(["definitely-not-a-command"])
    assert exc.value.code == 2
