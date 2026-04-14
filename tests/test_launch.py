"""Tests for the ezpz.launch module."""

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import ezpz.launch as launch
# try:
#     import ezpz.launch as launch
#
#     LAUNCH_AVAILABLE = True
# except ImportError:
#     LAUNCH_AVAILABLE = False


class TestLaunch:
    def test_command_exists(self):
        """Test command_exists function."""
        # Test with a command that should exist
        assert launch.command_exists("python3") is True

        # Test with a command that should not exist
        assert launch.command_exists("nonexistent_command_xyz") is False

    def test_get_scheduler(self):
        """Test get_scheduler function."""
        scheduler = launch.get_scheduler("SLURM")
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
        scheduler = launch.get_scheduler("PBS")
        assert scheduler == "PBS"

    def test_get_scheduler_from_slurm(self):
        """Test get_scheduler function with SLURM environment."""
        # Save original environment
        original_env = os.environ.copy()

        # Set SLURM environment variables
        os.environ["SLURM_JOB_ID"] = "test.job"
        scheduler = launch.get_scheduler("SLURM")
        assert scheduler == "SLURM"

        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

    def test_main_fallback_to_mpirun(self, monkeypatch):
        """Ensure the CLI falls back to mpirun when no scheduler is active."""
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.delenv("CPU_BIND", raising=False)
        monkeypatch.setattr(launch, "get_scheduler", lambda: "unknown")
        monkeypatch.setattr(launch, "get_active_jobid", lambda: None)
        monkeypatch.setattr(launch, "configure_warnings", lambda: None)
        monkeypatch.setattr(launch.ezpz, "get_machine", lambda: "localhost")
        cleanup_called: list[bool] = []
        monkeypatch.setattr(
            launch.ezpz.distributed,
            "cleanup",
            lambda: cleanup_called.append(True),
        )

        recorded: dict[str, object] = {}

        def fake_run(cmd, check=False):
            recorded["cmd"] = cmd
            recorded["check"] = check
            return SimpleNamespace(returncode=7)

        monkeypatch.setattr(launch.subprocess, "run", fake_run)
        rc = launch.run(["python", "-m", "ezpz.examples.test"])

        assert rc == 7
        assert cleanup_called, "Expected cleanup to be invoked"
        assert recorded["cmd"] == [
            "mpirun",
            "-np",
            "4",
            "python",
            "-m",
            "ezpz.examples.test",
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
        monkeypatch.delenv("CPU_BIND", raising=False)
        monkeypatch.setattr(launch, "get_scheduler", lambda: "unknown")
        monkeypatch.setattr(launch, "get_active_jobid", lambda: None)
        monkeypatch.setattr(launch, "configure_warnings", lambda: None)
        monkeypatch.setattr(launch.ezpz, "get_machine", lambda: "localhost")
        cleanup_called: list[bool] = []
        monkeypatch.setattr(
            launch.ezpz.distributed,
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
                "ezpz.examples.test",
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
            "ezpz.examples.test",
        ]

    def test_parse_args_supports_separator_passthrough(self):
        """Unknown launch flags before -- are forwarded to the launcher."""
        args = launch.parse_args(
            [
                "-n",
                "2",
                "-x",
                "PYTHONPATH=/tmp/.venv/bin",
                "-x",
                "EZPZ_LOG_LEVEL=DEBUG",
                "--",
                "python3",
                "-m",
                "ezpz.examples.vit",
                "--fsdp",
            ]
        )

        assert args.nproc == 2
        assert args.launcher_args == [
            "-x",
            "PYTHONPATH=/tmp/.venv/bin",
            "-x",
            "EZPZ_LOG_LEVEL=DEBUG",
        ]
        assert args.command == [
            "python3",
            "-m",
            "ezpz.examples.vit",
            "--fsdp",
        ]

    def test_parse_args_without_separator_treats_unknown_as_command(self):
        """Legacy form without -- still captures the command to run."""
        args = launch.parse_args(["--nproc", "2", "python", "-m", "demo", "--flag"])

        assert args.nproc == 2
        assert args.launcher_args == []
        assert args.command == ["python", "-m", "demo", "--flag"]

    def test_fallback_cpu_bind_cli_precedence_over_env(self, monkeypatch):
        """CLI --cpu-bind overrides CPU_BIND and emits precedence warning."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("CPU_BIND", "--cpu-bind=list:0-1")
        monkeypatch.setattr(launch, "get_scheduler", lambda: "unknown")
        monkeypatch.setattr(launch, "get_active_jobid", lambda: None)
        monkeypatch.setattr(launch, "configure_warnings", lambda: None)
        monkeypatch.setattr(launch.ezpz, "get_machine", lambda: "localhost")
        monkeypatch.setattr(launch.ezpz.distributed, "cleanup", lambda: None)
        recorded: dict[str, object] = {}
        warnings: list[str] = []

        def fake_run(cmd, check=False):
            recorded["cmd"] = cmd
            recorded["check"] = check
            return SimpleNamespace(returncode=0)

        def fake_warning(msg, *args):
            warnings.append(msg % args if args else msg)

        monkeypatch.setattr(launch.subprocess, "run", fake_run)
        monkeypatch.setattr(launch.logger, "warning", fake_warning)

        rc = launch.run(
            [
                "--cpu-bind",
                "list:2-3",
                "python",
                "-m",
                "ezpz.examples.test",
            ]
        )

        assert rc == 0
        assert recorded["cmd"] == [
            "mpirun",
            "-np",
            "2",
            "--cpu-bind=list:2-3",
            "python",
            "-m",
            "ezpz.examples.test",
        ]
        assert recorded["check"] is False
        assert any(
            "Both --cpu-bind and CPU_BIND are specified." in message
            and "Precedence order is: --cpu-bind > CPU_BIND." in message
            for message in warnings
        )


class TestRunCommand:
    """Tests for run_command covering exit codes and output filtering."""

    def test_returns_exit_code(self, monkeypatch, capsys):
        """run_command returns the subprocess exit code on success."""
        mock_process = MagicMock()
        mock_process.stdout = iter(["hello\n"])
        mock_process.returncode = 0
        mock_process.__enter__ = lambda self: self
        mock_process.__exit__ = lambda self, *a: None

        monkeypatch.setattr(
            launch.subprocess, "Popen", lambda *a, **kw: mock_process
        )
        rc = launch.run_command(["echo", "hello"])
        capsys.readouterr()
        assert rc == 0

    def test_returns_nonzero_on_failure(self, monkeypatch, capsys):
        """run_command returns non-zero exit code on failure."""
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.returncode = 1
        mock_process.__enter__ = lambda self: self
        mock_process.__exit__ = lambda self, *a: None

        monkeypatch.setattr(
            launch.subprocess, "Popen", lambda *a, **kw: mock_process
        )
        rc = launch.run_command(["false"])
        capsys.readouterr()
        assert rc == 1

    def test_filters_lines(self, monkeypatch, capsys):
        """Lines matching filters are excluded from printed output."""
        mock_process = MagicMock()
        mock_process.stdout = iter([
            "keep this line\n",
            "skip_this line\n",
            "also keep\n",
        ])
        mock_process.returncode = 0
        mock_process.__enter__ = lambda self: self
        mock_process.__exit__ = lambda self, *a: None

        monkeypatch.setattr(
            launch.subprocess, "Popen", lambda *a, **kw: mock_process
        )
        rc = launch.run_command(["echo", "test"], filters=["skip_this"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "keep this line" in captured.out
        assert "also keep" in captured.out
        assert "skip_this" not in captured.out


class TestGetActiveJobid:
    """Tests for get_active_jobid covering PBS, SLURM, and unknown."""

    def test_pbs_returns_pbs_jobid(self, monkeypatch):
        """When scheduler is PBS, delegates to pbs module."""
        monkeypatch.setattr(
            "ezpz.configs.get_scheduler", lambda _scheduler=None: "PBS"
        )
        monkeypatch.setattr(
            "ezpz.pbs.get_pbs_jobid_of_active_job", lambda: "12345"
        )
        result = launch.get_active_jobid()
        assert result == "12345"

    def test_slurm_returns_slurm_jobid(self, monkeypatch):
        """When scheduler is SLURM, delegates to slurm module."""
        monkeypatch.setattr(
            "ezpz.configs.get_scheduler", lambda _scheduler=None: "SLURM"
        )
        monkeypatch.setattr(
            "ezpz.slurm.get_slurm_jobid_of_active_job", lambda: "67890"
        )
        result = launch.get_active_jobid()
        assert result == "67890"

    def test_unknown_returns_none(self, monkeypatch):
        """When scheduler is UNKNOWN, returns None."""
        monkeypatch.setattr(
            "ezpz.configs.get_scheduler", lambda _scheduler=None: "UNKNOWN"
        )
        result = launch.get_active_jobid()
        assert result is None


class TestKillExistingProcesses:
    """Tests for kill_existing_processes covering pkill invocation."""

    def test_calls_pkill(self, monkeypatch, capsys):
        """Verify subprocess receives a pkill command with the filter pattern."""
        monkeypatch.setattr(launch.ezpz, "get_machine", lambda: "localhost")
        recorded_cmds = []

        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.returncode = 0
        mock_process.__enter__ = lambda self: self
        mock_process.__exit__ = lambda self, *a: None

        original_popen = launch.subprocess.Popen

        def tracking_popen(cmd, **kwargs):
            recorded_cmds.append(cmd)
            return mock_process

        monkeypatch.setattr(launch.subprocess, "Popen", tracking_popen)
        rc = launch.kill_existing_processes(filters=["my_process"])
        capsys.readouterr()
        assert rc == 0
        assert len(recorded_cmds) == 1
        assert recorded_cmds[0][0] == "pkill"
        assert "-f" in recorded_cmds[0]

    def test_handles_pkill_failure(self, monkeypatch, capsys):
        """pkill returning non-zero should not crash."""
        monkeypatch.setattr(launch.ezpz, "get_machine", lambda: "localhost")

        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.returncode = 1
        mock_process.__enter__ = lambda self: self
        mock_process.__exit__ = lambda self, *a: None

        monkeypatch.setattr(
            launch.subprocess, "Popen", lambda *a, **kw: mock_process
        )
        rc = launch.kill_existing_processes(filters=["some_filter"])
        capsys.readouterr()
        assert rc == 1  # reflects pkill's non-zero exit, but no exception
