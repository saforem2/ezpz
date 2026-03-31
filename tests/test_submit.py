"""Tests for ezpz.submit — job submission helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from ezpz.submit import (
    detect_env_setup,
    generate_pbs_script,
    generate_slurm_script,
    submit,
    submit_job,
)


# ── detect_env_setup ─────────────────────────────────────────────────────────


class TestDetectEnvSetup:
    def test_picks_up_virtual_env(self, tmp_path: Path):
        venv = tmp_path / "myvenv"
        venv.mkdir()
        with patch.dict(os.environ, {"VIRTUAL_ENV": str(venv)}, clear=False):
            result = detect_env_setup()
        assert "source" in result
        assert "myvenv/bin/activate" in result

    def test_picks_up_conda_prefix(self, tmp_path: Path):
        with patch.dict(
            os.environ,
            {"CONDA_PREFIX": "/opt/conda/envs/myenv"},
            clear=False,
        ):
            # Clear VIRTUAL_ENV so conda path is taken
            env = os.environ.copy()
            env.pop("VIRTUAL_ENV", None)
            with patch.dict(os.environ, env, clear=True):
                result = detect_env_setup()
        assert "conda activate" in result
        assert "myenv" in result

    def test_picks_up_ezpz_setup_env(self, tmp_path: Path):
        setup_file = tmp_path / "setup.sh"
        setup_file.write_text("module load foo")
        with patch.dict(
            os.environ,
            {"EZPZ_SETUP_ENV": str(setup_file)},
            clear=False,
        ):
            env = os.environ.copy()
            env.pop("VIRTUAL_ENV", None)
            env.pop("CONDA_PREFIX", None)
            with patch.dict(os.environ, env, clear=True):
                env2 = os.environ.copy()
                env2["EZPZ_SETUP_ENV"] = str(setup_file)
                with patch.dict(os.environ, env2, clear=True):
                    result = detect_env_setup()
        assert "source" in result
        assert "setup.sh" in result

    def test_returns_empty_when_nothing_set(self):
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        env.pop("CONDA_PREFIX", None)
        env.pop("EZPZ_SETUP_ENV", None)
        with patch.dict(os.environ, env, clear=True):
            result = detect_env_setup()
        assert result == ""


# ── generate_pbs_script ──────────────────────────────────────────────────────


class TestGeneratePBSScript:
    def test_basic_script(self):
        script = generate_pbs_script(
            "python3 -m ezpz.examples.test",
            nodes=2,
            time="01:00:00",
            queue="debug",
            account="myproject",
            env_setup="",
        )
        assert "#!/bin/bash --login" in script
        assert "#PBS -l select=2" in script
        assert "#PBS -l walltime=01:00:00" in script
        assert "#PBS -q debug" in script
        assert "#PBS -A myproject" in script
        assert "ezpz launch python3 -m ezpz.examples.test" in script

    def test_filesystems_colon_separated(self):
        script = generate_pbs_script(
            "echo hello",
            filesystems="home,eagle,grand",
            env_setup="",
        )
        assert "#PBS -l filesystems=home:eagle:grand" in script

    def test_no_account_omits_directive(self):
        with patch.dict(os.environ, {}, clear=True):
            script = generate_pbs_script(
                "echo hello",
                account=None,
                env_setup="",
            )
        # Should not have an empty -A line
        assert "#PBS -A" not in script

    def test_no_launch_flag(self):
        script = generate_pbs_script(
            "mpirun ./my_binary",
            wrap_with_launch=False,
            env_setup="",
        )
        assert "ezpz launch" not in script
        assert "mpirun ./my_binary" in script

    def test_includes_env_setup(self):
        script = generate_pbs_script(
            "echo test",
            env_setup="source /path/to/venv/bin/activate",
        )
        assert "source /path/to/venv/bin/activate" in script

    def test_job_name(self):
        script = generate_pbs_script(
            "echo test",
            job_name="my-job",
            env_setup="",
        )
        assert "#PBS -N my-job" in script


# ── generate_slurm_script ────────────────────────────────────────────────────


class TestGenerateSLURMScript:
    def test_basic_script(self):
        script = generate_slurm_script(
            "python3 -m ezpz.examples.test",
            nodes=4,
            time="02:00:00",
            queue="batch",
            account="proj123",
            env_setup="",
        )
        assert "#!/bin/bash --login" in script
        assert "#SBATCH --nodes=4" in script
        assert "#SBATCH --time=02:00:00" in script
        assert "#SBATCH --partition=batch" in script
        assert "#SBATCH --account=proj123" in script
        assert "ezpz launch python3 -m ezpz.examples.test" in script

    def test_no_account_omits_directive(self):
        with patch.dict(os.environ, {}, clear=True):
            script = generate_slurm_script(
                "echo hello",
                account=None,
                env_setup="",
            )
        assert "#SBATCH --account" not in script

    def test_job_name(self):
        script = generate_slurm_script(
            "echo test",
            job_name="my-slurm-job",
            env_setup="",
        )
        assert "#SBATCH --job-name=my-slurm-job" in script


# ── submit_job ───────────────────────────────────────────────────────────────


class TestSubmitJob:
    def test_pbs_calls_qsub(self, tmp_path: Path):
        script = tmp_path / "job.sh"
        script.write_text("#!/bin/bash\necho hello")
        with patch("ezpz.submit.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="12345.pbs-server\n", returncode=0
            )
            job_id = submit_job(script, "PBS")
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["qsub", str(script)]
        assert job_id == "12345.pbs-server"

    def test_slurm_calls_sbatch(self, tmp_path: Path):
        script = tmp_path / "job.sh"
        script.write_text("#!/bin/bash\necho hello")
        with patch("ezpz.submit.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="Submitted batch job 67890\n", returncode=0
            )
            job_id = submit_job(script, "SLURM")
        assert mock_run.call_args[0][0] == ["sbatch", str(script)]
        assert job_id == "Submitted batch job 67890"

    def test_unknown_scheduler_returns_none(self, tmp_path: Path):
        script = tmp_path / "job.sh"
        script.write_text("#!/bin/bash\necho hello")
        job_id = submit_job(script, "UNKNOWN")
        assert job_id is None

    def test_missing_binary_returns_none(self, tmp_path: Path):
        script = tmp_path / "job.sh"
        script.write_text("#!/bin/bash\necho hello")
        with patch(
            "ezpz.submit.subprocess.run", side_effect=FileNotFoundError
        ):
            job_id = submit_job(script, "PBS")
        assert job_id is None


# ── submit (integration) ────────────────────────────────────────────────────


class TestSubmit:
    def test_dry_run_does_not_call_subprocess(self, capsys):
        with patch("ezpz.submit.subprocess.run") as mock_run:
            result = submit(
                command=["python3", "-m", "ezpz.examples.test"],
                nodes=2,
                queue="debug",
                scheduler="PBS",
                dry_run=True,
            )
        mock_run.assert_not_called()
        assert result is None
        output = capsys.readouterr().out
        assert "#PBS -l select=2" in output
        assert "dry-run" in output

    def test_submit_existing_script(self, tmp_path: Path):
        script = tmp_path / "myjob.sh"
        script.write_text("#!/bin/bash\necho hello")
        with patch("ezpz.submit.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="99999.pbs\n", returncode=0
            )
            job_id = submit(script=script, scheduler="PBS")
        assert job_id == "99999.pbs"

    def test_account_env_fallback(self, capsys):
        with patch.dict(os.environ, {"PBS_ACCOUNT": "fallback_proj"}):
            result = submit(
                command=["echo", "hello"],
                scheduler="PBS",
                dry_run=True,
            )
        output = capsys.readouterr().out
        assert "#PBS -A fallback_proj" in output

    def test_no_scheduler_prints_error(self, capsys):
        with patch("ezpz.configs.get_scheduler", return_value="UNKNOWN"):
            result = submit(
                command=["echo", "hello"],
            )
        assert result is None

    def test_job_name_derived_from_module(self, capsys):
        submit(
            command=["python3", "-m", "ezpz.examples.fsdp", "--model", "small"],
            scheduler="PBS",
            dry_run=True,
        )
        output = capsys.readouterr().out
        assert "#PBS -N ezpz.examples.fsdp" in output
