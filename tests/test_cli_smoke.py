"""Smoke test for the exp CLI gate."""

from typer.testing import CliRunner

from cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output
