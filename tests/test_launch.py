"""Tests for the ezpz.launch module."""

import os
import sys
import tempfile
from pathlib import Path

from types import SimpleNamespace

import pytest

try:
    import ezpz.launch as launch
    LAUNCH_AVAILABLE = True
except ImportError:
    LAUNCH_AVAILABLE = False


@pytest.mark.skipif(not LAUNCH_AVAILABLE, reason="ezpz.launch not available")
class TestLaunch:
    def test_command_exists(self):
        """Test command_exists function."""
        # Test with a command that should exist
        assert launch.command_exists("python") is True
        
        # Test with a command that should not exist
        assert launch.command_exists("nonexistent_command_xyz") is False

    def test_get_scheduler(self):
        """Test get_scheduler function."""
        scheduler = launch.get_scheduler()
        assert isinstance(scheduler, str)
        # Should be one of the known schedulers or "UNKNOWN"
        assert scheduler in ["PBS", "SLURM", "UNKNOWN"]

    def test_run_bash_command(self):
        """Test run_bash_command function."""
        # Test a simple command
        result = launch.run_bash_command("echo 'test'")
        assert result is not None

    def test_get_scheduler_from_pbs(self, mock_pbs_env):
        """Test get_scheduler function with PBS environment."""
        # Set PBS environment variables
        os.environ["PBS_JOBID"] = "test.job"
        scheduler = launch.get_scheduler()
        assert scheduler == "PBS"

    def test_get_scheduler_from_slurm(self):
        """Test get_scheduler function with SLURM environment."""
        # Save original environment
        original_env = os.environ.copy()
        
        # Set SLURM environment variables
        os.environ["SLURM_JOB_ID"] = "test.job"
        scheduler = launch.get_scheduler()
        assert scheduler == "SLURM"
        
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

    def test_main_fallback_to_mpirun(self, monkeypatch):
        """Ensure the CLI falls back to mpirun when no scheduler is active."""
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setattr(launch, "get_scheduler", lambda: "unknown")
        monkeypatch.setattr(launch, "get_active_jobid", lambda: None)
        monkeypatch.setattr(launch, "configure_warnings", lambda: None)
        cleanup_called: list[bool] = []
        monkeypatch.setattr(
            launch.ezpz.dist,
            "cleanup",
            lambda: cleanup_called.append(True),
        )

        recorded: dict[str, object] = {}

        def fake_run(cmd, check=False):
            recorded["cmd"] = cmd
            recorded["check"] = check
            return SimpleNamespace(returncode=7)

        monkeypatch.setattr(launch.subprocess, "run", fake_run)
        rc = launch.run(["python", "-m", "ezpz.test_dist"])

        assert rc == 7
        assert cleanup_called, "Expected cleanup to be invoked"
        assert recorded["cmd"] == [
            "mpirun",
            "-np",
            "4",
            "python",
            "-m",
            "ezpz.test_dist",
        ]
        assert recorded["check"] is False

    def test_build_executable_respects_hostfile(self, monkeypatch, tmp_path):
        """Ensure build_executable forwards user-supplied hostfiles."""
        hostfile = tmp_path / "hosts.txt"
        hostfile.write_text("node001\n", encoding="utf-8")
        captured: dict[str, object] = {}

        def fake_build_launch_cmd(**kwargs):
            captured.update(kwargs)
            return "mpiexec"

        monkeypatch.setattr(launch.ezpz.pbs, "build_launch_cmd", fake_build_launch_cmd)
        cmd = launch.build_executable(
            cmd_to_launch=["python", "-m", "demo"],
            hostfile=hostfile,
        )
        assert cmd[0] == "mpiexec"
        assert captured["hostfile"] == hostfile

    def test_fallback_includes_hostfile_and_counts(self, monkeypatch, tmp_path):
        """Fallback mpirun honours hostfile and process count overrides."""
        hostfile = tmp_path / "hosts.txt"
        hostfile.write_text("node001\nnode002\n", encoding="utf-8")
        monkeypatch.setattr(launch, "get_scheduler", lambda: "unknown")
        monkeypatch.setattr(launch, "get_active_jobid", lambda: None)
        monkeypatch.setattr(launch, "configure_warnings", lambda: None)
        cleanup_called: list[bool] = []
        monkeypatch.setattr(
            launch.ezpz.dist,
            "cleanup",
            lambda: cleanup_called.append(True),
        )
        recorded: dict[str, object] = {}

        def fake_run(cmd, check=False):
            recorded["cmd"] = cmd
            recorded["check"] = check
            return SimpleNamespace(returncode=3)

        monkeypatch.setattr(launch.subprocess, "run", fake_run)
        rc = launch.run(
            [
                "--hostfile",
                str(hostfile),
                "--nnode",
                "2",
                "--nproc_per_node",
                "4",
                "python",
                "-m",
                "ezpz.test_dist",
            ]
        )

        assert rc == 3
        assert cleanup_called
        assert recorded["check"] is False
        assert recorded["cmd"] == [
            "mpirun",
            "-np",
            "8",
            "--hostfile",
            str(hostfile),
            "--map-by",
            "ppr:4:node",
            "python",
            "-m",
            "ezpz.test_dist",
        ]
