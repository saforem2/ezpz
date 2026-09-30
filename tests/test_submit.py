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
    def test_prefers_the_installed_utils_sh(self):
        """Default is the local file, NOT a curl.

        Changed deliberately: compute nodes have no outbound route, so a
        generated `source <(curl ...)` hangs ~270 s and then leaves the
        environment unconfigured. This test previously asserted the curl
        form; it now pins the opposite.
        """
        env = os.environ.copy()
        env.pop("EZPZ_SETUP_ENV", None)
        with patch.dict(os.environ, env, clear=True):
            result = detect_env_setup()
        assert "ezpz_setup_env" in result
        assert "curl" not in result
        assert result.startswith("source /")

    def test_falls_back_to_curl_when_no_local_copy(self):
        """The network form survives for installs that lack bin/utils.sh."""
        import ezpz.submit as mod

        env = os.environ.copy()
        env.pop("EZPZ_SETUP_ENV", None)
        real_is_file = Path.is_file

        def _no_utils(self):
            if self.name == "utils.sh":
                return False
            return real_is_file(self)

        with patch.dict(os.environ, env, clear=True):
            with patch.object(Path, "is_file", _no_utils):
                result = mod.detect_env_setup()
        assert "curl" in result
        assert "ezpz_setup_env" in result

    def test_picks_up_ezpz_setup_env_file(self, tmp_path: Path):
        setup_file = tmp_path / "setup.sh"
        setup_file.write_text("module load foo")
        with patch.dict(
            os.environ,
            {"EZPZ_SETUP_ENV": str(setup_file)},
            clear=True,
        ):
            result = detect_env_setup()
        assert "source" in result
        assert "setup.sh" in result

    def test_picks_up_ezpz_setup_env_inline(self):
        inline = "module load frameworks && source .venv/bin/activate"
        with patch.dict(
            os.environ,
            {"EZPZ_SETUP_ENV": inline},
            clear=True,
        ):
            result = detect_env_setup()
        assert result == inline


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
            command=[
                "python3",
                "-m",
                "ezpz.examples.fsdp",
                "--model",
                "small",
            ],
            scheduler="PBS",
            dry_run=True,
        )
        output = capsys.readouterr().out
        assert "#PBS -N ezpz.examples.fsdp" in output


class TestSlurmGpuDirectives:
    """SLURM GPU allocation flags.

    Without `--gpus-per-node` a Perlmutter GPU job is allocated no GPUs,
    so a generated script was unusable there and had to be hand-edited --
    which defeats the point of `ezpz submit`. `--constraint gpu` selects
    the GPU node type, and `--ntasks-per-node` pairs ranks to GPUs.
    """

    def test_gpus_per_node_emitted(self):
        out = generate_slurm_script("echo hi", gpus_per_node=4)
        assert "#SBATCH --gpus-per-node=4" in out

    def test_ntasks_per_node_emitted(self):
        out = generate_slurm_script("echo hi", ntasks_per_node=4)
        assert "#SBATCH --ntasks-per-node=4" in out

    def test_constraint_emitted(self):
        out = generate_slurm_script("echo hi", constraint="gpu")
        assert "#SBATCH --constraint=gpu" in out

    def test_omitted_when_unset(self):
        """A CPU job must not get GPU directives it never asked for."""
        out = generate_slurm_script("echo hi")
        assert "--gpus-per-node" not in out
        assert "--ntasks-per-node" not in out
        assert "--constraint" not in out

    def test_perlmutter_shape(self):
        """The exact combination a Perlmutter GPU job needs."""
        out = generate_slurm_script(
            "python3 -m ezpz.examples.fsdp_tp --tp 2",
            nodes=2,
            constraint="gpu",
            queue="debug",
            account="m4388_g",
            gpus_per_node=4,
            ntasks_per_node=4,
        )
        for want in (
            "#SBATCH --nodes=2",
            "#SBATCH --constraint=gpu",
            "#SBATCH --gpus-per-node=4",
            "#SBATCH --ntasks-per-node=4",
            "#SBATCH --account=m4388_g",
        ):
            assert want in out, f"missing {want}"


class TestStrictToggle:
    """`set -e` is wrong for multi-arm experiment scripts.

    An arm that times out (rc=124) or whose teardown returns non-zero
    would abort the whole job under `set -e`, losing every later arm.
    `set -u` is never emitted at all: SKILL.md records that Lmod is not
    `set -u`-clean and sourcing /etc/profile under it kills the script
    before it prints a line.
    """

    def test_strict_is_default(self):
        assert "set -eo pipefail" in generate_pbs_script("echo hi")
        assert "set -eo pipefail" in generate_slurm_script("echo hi")

    def test_no_strict_drops_errexit_keeps_pipefail(self):
        for gen in (generate_pbs_script, generate_slurm_script):
            out = gen("echo hi", strict=False)
            assert "set -o pipefail" in out
            assert "set -eo pipefail" not in out

    def test_never_emits_nounset(self):
        for gen in (generate_pbs_script, generate_slurm_script):
            for strict in (True, False):
                assert "set -u" not in gen("echo hi", strict=strict)
                assert "-euo" not in gen("echo hi", strict=strict)


class TestSubmitThreadsNewOptions:
    """The new knobs must reach the generated script through `submit()`."""

    def test_slurm_gpu_flags_reach_script(self, capsys):
        submit(
            command=["python3", "-m", "ezpz.examples.test"],
            scheduler="SLURM",
            dry_run=True,
            gpus_per_node=4,
            ntasks_per_node=4,
            constraint="gpu",
        )
        out = capsys.readouterr().out
        assert "#SBATCH --gpus-per-node=4" in out
        assert "#SBATCH --ntasks-per-node=4" in out
        assert "#SBATCH --constraint=gpu" in out

    def test_strict_false_reaches_script(self, capsys):
        submit(
            command=["echo", "hi"],
            scheduler="PBS",
            dry_run=True,
            strict=False,
        )
        out = capsys.readouterr().out
        assert "set -o pipefail" in out
        assert "set -eo pipefail" not in out

    def test_gpu_flags_ignored_for_pbs(self, capsys):
        """PBS has no --gpus-per-node; passing it must not corrupt output."""
        submit(
            command=["echo", "hi"],
            scheduler="PBS",
            dry_run=True,
            gpus_per_node=4,
            constraint="gpu",
        )
        out = capsys.readouterr().out
        assert "gpus-per-node" not in out
        assert "constraint" not in out


class TestLaunchIsDefault:
    """`ezpz launch` computes --cpu-bind; bare mpiexec supplies none.

    A hand-rolled `mpiexec -n 24 -ppn 12` ran an entire measurement
    campaign with no CPU binding, which is a real confound for a
    timing-sensitive hang. The generated script must default to
    `ezpz launch` so that mistake is not reachable by default.
    """

    def test_launch_wraps_by_default(self):
        for gen in (generate_pbs_script, generate_slurm_script):
            assert "ezpz launch echo hi" in gen("echo hi")

    def test_no_launch_is_opt_in(self):
        for gen in (generate_pbs_script, generate_slurm_script):
            out = gen("echo hi", wrap_with_launch=False)
            assert "ezpz launch" not in out
