"""Tests for the unified ezpz Click-based CLI."""

from __future__ import annotations

from click.testing import CliRunner

import ezpz.launch
import ezpz.test
from ezpz.cli import main as cli_main
from ezpz.utils import tar_env, yeet_env


def test_test_subcommand_passes_args(monkeypatch):
    recorded: dict[str, object] = {}

    def fake_run(args):
        recorded["args"] = args
        return 0

    monkeypatch.setattr(ezpz.test, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(cli_main, ["test", "--foo", "bar"])

    assert result.exit_code == 0
    assert recorded["args"] == ("--foo", "bar")


def test_launch_subcommand_surfaces_usage(monkeypatch):
    def fake_run(args):
        raise SystemExit("No command provided to ezpz launch")

    monkeypatch.setattr(ezpz.launch, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(cli_main, ["launch"])

    assert result.exit_code != 0
    assert "No command provided to ezpz launch" in result.output


def test_tar_env_subcommand_returns_exit_status(monkeypatch):
    recorded: dict[str, object] = {}

    def fake_run(args):
        recorded["args"] = args
        return 0

    monkeypatch.setattr(tar_env, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(cli_main, ["tar-env"])

    assert result.exit_code == 0
    assert recorded["args"] == ()


def test_yeet_env_nonzero_exit_propagates(monkeypatch):
    def fake_run(args):
        return 3

    monkeypatch.setattr(yeet_env, "run", fake_run)

    runner = CliRunner()
    result = runner.invoke(cli_main, ["yeet-env"])

    assert result.exit_code == 3
