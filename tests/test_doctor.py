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


def test_run_returns_error_on_failed_check(monkeypatch, capsys):
    failing = [
        doctor.CheckResult(
            name="demo", status="error", message="failed", remedy=None
        )
    ]
    monkeypatch.setattr(doctor, "run_checks", lambda: failing)
    exit_code = doctor.run([])
    capsys.readouterr()  # discard runtime context output
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


class TestCheckScheduler:
    """Tests for check_scheduler covering PBS, SLURM, and no-scheduler paths."""

    def test_pbs_detected(self, monkeypatch, capsys):
        """PBS_JOBID in environ and get_scheduler returning PBS -> ok."""
        monkeypatch.setattr(doctor.ezpz.configs, "get_scheduler", lambda: "PBS")
        result = doctor.check_scheduler(
            get_scheduler=doctor.ezpz.configs.get_scheduler,
            environ={"PBS_JOBID": "12345.pbs01"},
        )
        capsys.readouterr()
        assert result.status == "ok"
        assert result.name == "scheduler"
        assert "PBS" in result.message

    def test_slurm_detected(self, monkeypatch, capsys):
        """SLURM_JOB_ID in environ and get_scheduler returning SLURM -> ok."""
        monkeypatch.setattr(doctor.ezpz.configs, "get_scheduler", lambda: "SLURM")
        result = doctor.check_scheduler(
            get_scheduler=doctor.ezpz.configs.get_scheduler,
            environ={"SLURM_JOB_ID": "98765"},
        )
        capsys.readouterr()
        assert result.status == "ok"
        assert result.name == "scheduler"
        assert "SLURM" in result.message

    def test_no_scheduler_warns(self, capsys):
        """Empty environ and get_scheduler returning UNKNOWN -> warning."""
        result = doctor.check_scheduler(
            get_scheduler=lambda: "UNKNOWN",
            environ={},
        )
        capsys.readouterr()
        assert result.status == "warning"
        assert result.name == "scheduler"
        assert "No scheduler" in result.message or "local" in result.message.lower()


class TestCheckTorchDevice:
    """Tests for check_torch_device covering env override and no-accelerator."""

    def test_device_env_override(self, monkeypatch, capsys):
        """TORCH_DEVICE=cpu in environ -> ok with device mentioned."""
        monkeypatch.setenv("TORCH_DEVICE", "cpu")
        result = doctor.check_torch_device()
        capsys.readouterr()
        assert result.status == "ok"
        assert result.name == "torch"
        assert "cpu" in result.message.lower() or "TORCH_DEVICE" in result.message

    def test_no_accelerator_warns(self, monkeypatch, capsys):
        """All accelerators unavailable -> warning."""
        monkeypatch.delenv("TORCH_DEVICE", raising=False)
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        if hasattr(torch, "xpu"):
            monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
            monkeypatch.setattr(torch.xpu, "device_count", lambda: 0)
        monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        result = doctor.check_torch_device()
        capsys.readouterr()
        assert result.status == "warning"
        assert result.name == "torch"
        assert "accelerator" in result.message.lower() or "no" in result.message.lower()


class TestCheckHostfile:
    """Tests for check_hostfile covering no-job and PBS-with-nodefile paths."""

    def test_no_job_context_is_ok(self, capsys):
        """No PBS_JOBID or SLURM_JOB_ID in environ -> ok (no hostfile needed)."""
        result = doctor.check_hostfile(environ={})
        capsys.readouterr()
        assert result.status == "ok"
        assert result.name == "hostfile"
        assert "not required" in result.message.lower() or "no scheduler" in result.message.lower()

    def test_pbs_with_valid_nodefile(self, tmp_path, capsys):
        """PBS_JOBID + PBS_NODEFILE pointing to real file -> ok."""
        nodefile = tmp_path / "pbs_nodefile"
        nodefile.write_text("node001\nnode002\n", encoding="utf-8")
        env = {
            "PBS_JOBID": "99999.pbs01",
            "PBS_NODEFILE": str(nodefile),
        }
        result = doctor.check_hostfile(environ=env)
        capsys.readouterr()
        assert result.status == "ok"
        assert result.name == "hostfile"
        assert "HOSTFILE" in result.message or "PBS_NODEFILE" in result.message
