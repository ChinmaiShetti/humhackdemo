from typer.testing import CliRunner

from ha.cli import app

runner = CliRunner()


def test_help_lists_all_spec_commands():
    out = runner.invoke(app, ["--help"]).output
    for cmd in ["data", "gen", "neigh", "judge", "calib", "select", "systems", "eval", "run",
                "paper", "dashboard", "doctor", "models", "registry", "kaggle"]:
        assert cmd in out, cmd


def test_unimplemented_command_points_to_plan():
    # `neigh` (P4.1), `judge` and `select` (P5) are real now; `dashboard` is still a stub (P8).
    res = runner.invoke(app, ["dashboard"])
    assert res.exit_code == 2
    assert "P8" in res.output
