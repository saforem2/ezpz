"""Unit tests for ezpz.doctor diagnostics."""

from __future__ import annotations

import builtins
import json
import sys
from types import ModuleType

import pytest
from click.testing import CliRunner

import ezpz.doctor as doctor
from ezpz.cli import main as cli_main


@pytest.fixture(autouse=True)
def restore_sys_modules():
    """Ensure temporary module injections are cleaned up."""
    snapshot = sys.modules.copy()
    yield
    for name in list(sys.modules.keys()):
        if name not in snapshot:
            del sys.modules[name]
        elif sys.modules[name] is not snapshot[name]:
            sys.modules[name] = snapshot[name]


def test_check_mpi_handles_missing_components(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mpi4py":
            raise ImportError("mocked failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        doctor, "_command_exists", lambda *_args, **_kwargs: False
    )
    result = doctor.check_mpi(which=lambda _cmd: None)
    assert result.status == "error"
    assert "launcher" in result.message.lower()


def test_check_mpi_success(monkeypatch):
    fake_mpi = ModuleType("mpi4py")
    fake_mpi.MPI = object()
    sys.modules["mpi4py"] = fake_mpi
    monkeypatch.setattr(doctor, "_command_exists", lambda *_a, **_k: True)
    result = doctor.check_mpi(which=lambda _cmd: "/usr/bin/mpiexec")
    assert result.status == "ok"


# def test_check_wandb_warns_without_key(monkeypatch):
#     fake_wandb = ModuleType("wandb")
#     fake_wandb.__version__ = "1.0.0"
#     sys.modules["wandb"] = fake_wandb
#     result = doctor.check_wandb(environ={})
#     assert result.status == "warning"
#     assert "WANDB" in result.remedy


def test_run_returns_error_on_failed_check(monkeypatch):
    failing = [
        doctor.CheckResult(
            name="demo", status="error", message="failed", remedy=None
        )
    ]
    monkeypatch.setattr(doctor, "run_checks", lambda: failing)
    exit_code = doctor.run([])
    assert exit_code == 1


def test_cli_doctor_json(monkeypatch):
    outcomes = [
        doctor.CheckResult(name="mpi", status="ok", message="ok"),
        doctor.CheckResult(
            name="torch", status="warning", message="warn", remedy="fix"
        ),
    ]
    monkeypatch.setattr(doctor, "run_checks", lambda: outcomes)
    runner = CliRunner()
    result = runner.invoke(cli_main, ["doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["name"] == "mpi"
    assert payload[1]["remedy"] == "fix"
