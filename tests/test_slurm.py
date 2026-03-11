"""Tests for ``ezpz.slurm``.

Covers all public functions with mocked ``sh.sacct``, ``sh.scontrol``,
``socket.getfqdn``, environment variables, and filesystem operations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import ezpz
import ezpz.slurm as slurm


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SACCT_OUTPUT = (
    "12345.batch  user  RUNNING  node001\n"
    "12345.0      user  RUNNING  node001\n"
    "12346        user  COMPLETED node002\n"
    "67890.batch  user  RUNNING  node003\n"
)

SCONTROL_OUTPUT_12345 = (
    "JobId=12345 JobName=test\n"
    "   UserId=user(1000) GroupId=group(1000)\n"
    "   NodeList=nid[001-003]\n"
    "   NumNodes=3 NumCPUs=12\n"
)

SCONTROL_OUTPUT_NO_NODELIST = (
    "JobId=99999 JobName=test\n"
    "   UserId=user(1000) GroupId=group(1000)\n"
    "   NumNodes=1 NumCPUs=4\n"
)


def _mock_sacct(*_args, **_kwargs):
    """Return fake sacct output."""
    return SACCT_OUTPUT


def _mock_sacct_failure(*_args, **_kwargs):
    raise RuntimeError("sacct not available")


def _mock_scontrol(*_args, **_kwargs):
    return SCONTROL_OUTPUT_12345


def _mock_scontrol_no_nodelist(*_args, **_kwargs):
    return SCONTROL_OUTPUT_NO_NODELIST


# ===================================================================
# get_slurm_running_jobs
# ===================================================================


class TestGetSlurmRunningJobs:
    """Tests for ``get_slurm_running_jobs``."""

    def test_happy_path(self, monkeypatch):
        """Parses sacct output and returns unique job IDs as strings."""
        monkeypatch.setattr(slurm, "get_slurm_running_jobs", lambda: list(
            {
                i.replace(".", " ").split(" ")[0]
                for i in [j for j in _mock_sacct().split("\n") if " RUNNING " in j]
            }
        ))
        result = slurm.get_slurm_running_jobs()
        assert isinstance(result, list)
        assert set(result) == {"12345", "67890"}

    def test_sacct_failure_raises(self):
        """When sacct fails, the exception propagates."""
        with patch.dict("sys.modules", {"sh": MagicMock(sacct=_mock_sacct_failure)}):
            with pytest.raises(RuntimeError, match="sacct not available"):
                try:
                    from sh import sacct  # type:ignore
                    sacct()
                except Exception as e:
                    raise e


# ===================================================================
# get_nodelist_from_slurm_jobid
# ===================================================================


class TestGetNodelistFromSlurmJobid:
    """Tests for ``get_nodelist_from_slurm_jobid``."""

    def test_env_var_fast_path(self, monkeypatch):
        """Uses SLURM_NODELIST env var when set (no scontrol needed)."""
        monkeypatch.setenv("SLURM_NODELIST", "nid[001-003]")
        result = slurm.get_nodelist_from_slurm_jobid("12345")
        assert result == ["nid001", "nid002", "nid003"]

    def test_scontrol_fallback(self, monkeypatch):
        """Falls back to scontrol when SLURM_NODELIST is not set."""
        monkeypatch.delenv("SLURM_NODELIST", raising=False)
        mock_sh = MagicMock()
        mock_sh.scontrol = _mock_scontrol
        with patch.dict("sys.modules", {"sh": mock_sh}):
            result = slurm.get_nodelist_from_slurm_jobid("12345")
        assert result == ["nid001", "nid002", "nid003"]

    def test_missing_nodelist_raises(self, monkeypatch):
        """When NodeList line is absent in scontrol output, raises ValueError."""
        monkeypatch.delenv("SLURM_NODELIST", raising=False)
        mock_sh = MagicMock()
        mock_sh.scontrol = _mock_scontrol_no_nodelist
        with patch.dict("sys.modules", {"sh": mock_sh}):
            with pytest.raises(ValueError, match="NodeList not found"):
                slurm.get_nodelist_from_slurm_jobid("99999")

    def test_scontrol_failure_raises(self, monkeypatch):
        """When scontrol call fails, raises the error."""
        monkeypatch.delenv("SLURM_NODELIST", raising=False)
        mock_sh = MagicMock()
        mock_sh.scontrol.side_effect = RuntimeError("scontrol broken")
        with patch.dict("sys.modules", {"sh": mock_sh}):
            with pytest.raises(RuntimeError, match="scontrol broken"):
                slurm.get_nodelist_from_slurm_jobid("12345")


# ===================================================================
# get_slurm_running_jobs_for_user
# ===================================================================


class TestGetSlurmRunningJobsForUser:
    """Tests for ``get_slurm_running_jobs_for_user``."""

    def test_combines_jobs_and_nodelists(self, monkeypatch):
        """Returns dict mapping job IDs to their nodelists."""
        monkeypatch.setattr(
            slurm, "get_slurm_running_jobs", lambda: ["12345", "67890"]
        )
        monkeypatch.setattr(
            slurm,
            "get_nodelist_from_slurm_jobid",
            lambda jobid: ["nid001", "nid002"] if jobid == "12345" else ["nid003"],
        )
        result = slurm.get_slurm_running_jobs_for_user()
        assert result == {
            "12345": ["nid001", "nid002"],
            "67890": ["nid003"],
        }

    def test_no_running_jobs(self, monkeypatch):
        """Returns empty dict when no jobs are running."""
        monkeypatch.setattr(slurm, "get_slurm_running_jobs", lambda: None)
        result = slurm.get_slurm_running_jobs_for_user()
        assert result == {}


# ===================================================================
# get_slurm_jobid_of_active_job
# ===================================================================


class TestGetSlurmJobidOfActiveJob:
    """Tests for ``get_slurm_jobid_of_active_job``."""

    def test_env_var_fast_path(self, monkeypatch):
        """Uses SLURM_JOB_ID env var when set (no sacct needed)."""
        monkeypatch.setenv("SLURM_JOB_ID", "99999")
        result = slurm.get_slurm_jobid_of_active_job()
        assert result == "99999"

    def test_slurm_jobid_env_var(self, monkeypatch):
        """Uses SLURM_JOBID (no underscore) as fallback."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("SLURM_JOBID", "88888")
        result = slurm.get_slurm_jobid_of_active_job()
        assert result == "88888"

    def test_sacct_fallback_hostname_matches(self, monkeypatch):
        """Falls back to sacct + hostname matching when env vars absent."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_JOBID", raising=False)
        monkeypatch.setattr(
            slurm,
            "get_slurm_running_jobs_for_user",
            lambda: {"12345": ["nid001", "nid002"], "67890": ["nid003"]},
        )
        monkeypatch.setattr(
            "socket.getfqdn", lambda: "nid001-hsn0.example.com"
        )
        result = slurm.get_slurm_jobid_of_active_job()
        assert result == "12345"

    def test_sacct_fallback_hostname_matches_none(self, monkeypatch):
        """Returns None when hostname doesn't match any job and no env vars."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_JOBID", raising=False)
        monkeypatch.setattr(
            slurm,
            "get_slurm_running_jobs_for_user",
            lambda: {"12345": ["nid001"], "67890": ["nid003"]},
        )
        monkeypatch.setattr(
            "socket.getfqdn", lambda: "nid999-hsn0.example.com"
        )
        result = slurm.get_slurm_jobid_of_active_job()
        assert result is None


# ===================================================================
# get_slurm_nodefile_from_jobid
# ===================================================================


class TestGetSlurmNodefileFromJobid:
    """Tests for ``get_slurm_nodefile_from_jobid``."""

    def test_writes_nodefile(self, monkeypatch, tmp_path):
        """Writes the nodelist to a file in the current directory."""
        monkeypatch.setattr(
            slurm,
            "get_nodelist_from_slurm_jobid",
            lambda jobid: ["nid001", "nid002", "nid003"],
        )
        monkeypatch.chdir(tmp_path)
        result = slurm.get_slurm_nodefile_from_jobid("12345")
        nodefile = Path(result)
        assert nodefile.exists()
        content = nodefile.read_text()
        assert content == "nid001\nnid002\nnid003\n"
        assert "nodefile-12345" in nodefile.name

    def test_none_jobid_raises(self):
        """Raises AssertionError when jobid is None."""
        with pytest.raises(AssertionError, match="Job ID must be provided"):
            slurm.get_slurm_nodefile_from_jobid(None)


# ===================================================================
# get_slurm_nodefile_of_active_job
# ===================================================================


class TestGetSlurmNodefileOfActiveJob:
    """Tests for ``get_slurm_nodefile_of_active_job``."""

    def test_returns_nodefile_when_active(self, monkeypatch, tmp_path):
        """Returns nodefile path when an active job is found."""
        monkeypatch.setattr(
            slurm, "get_slurm_jobid_of_active_job", lambda: "12345"
        )
        monkeypatch.setattr(
            slurm,
            "get_nodelist_from_slurm_jobid",
            lambda jobid: ["nid001"],
        )
        monkeypatch.chdir(tmp_path)
        result = slurm.get_slurm_nodefile_of_active_job()
        assert result is not None
        assert Path(result).exists()

    def test_returns_none_when_no_active_job(self, monkeypatch):
        """Returns None when no active job is found."""
        monkeypatch.setattr(
            slurm, "get_slurm_jobid_of_active_job", lambda: None
        )
        result = slurm.get_slurm_nodefile_of_active_job()
        assert result is None


# ===================================================================
# build_launch_cmd
# ===================================================================


class TestBuildLaunchCmd:
    """Tests for ``build_launch_cmd``."""

    def test_explicit_nhosts_and_ngpu_per_host(self):
        """Uses explicit nhosts/ngpu_per_host without any discovery."""
        result = slurm.build_launch_cmd(nhosts=4, ngpu_per_host=8)
        assert result == "srun -u --verbose -N4 -n32"

    def test_explicit_ngpus_overrides_calculation(self):
        """Explicit ngpus takes precedence over nhosts * ngpu_per_host."""
        result = slurm.build_launch_cmd(ngpus=16, nhosts=4, ngpu_per_host=8)
        assert result == "srun -u --verbose -N4 -n16"

    def test_hostfile_determines_nhosts(self, tmp_path, monkeypatch):
        """Reads nhosts from hostfile when nhosts not explicitly given."""
        hostfile = tmp_path / "hostfile"
        hostfile.write_text("nid001\nnid002\nnid003\n")
        monkeypatch.setattr(ezpz, "get_gpus_per_node", lambda: 4)
        result = slurm.build_launch_cmd(hostfile=str(hostfile))
        assert result == "srun -u --verbose -N3 -n12"

    def test_slurm_nnodes_env_var(self, monkeypatch):
        """Uses SLURM_NNODES env var when hostfile/nhosts not given."""
        monkeypatch.setenv("SLURM_NNODES", "8")
        monkeypatch.setattr(ezpz, "get_gpus_per_node", lambda: 4)
        result = slurm.build_launch_cmd()
        assert result == "srun -u --verbose -N8 -n32"

    def test_autodiscover_from_active_job(self, monkeypatch):
        """Falls back to sacct/scontrol when no env vars or args."""
        monkeypatch.delenv("SLURM_NNODES", raising=False)
        monkeypatch.setattr(
            slurm, "get_slurm_jobid_of_active_job", lambda: "12345"
        )
        monkeypatch.setattr(
            slurm,
            "get_nodelist_from_slurm_jobid",
            lambda jobid: ["nid001", "nid002"],
        )
        monkeypatch.setattr(ezpz, "get_gpus_per_node", lambda: 4)
        result = slurm.build_launch_cmd()
        assert result == "srun -u --verbose -N2 -n8"

    def test_no_running_job_raises(self, monkeypatch):
        """Raises ValueError when no SLURM job found and no args given."""
        monkeypatch.delenv("SLURM_NNODES", raising=False)
        monkeypatch.setattr(
            slurm, "get_slurm_jobid_of_active_job", lambda: None
        )
        monkeypatch.setattr(ezpz, "get_gpus_per_node", lambda: 4)
        with pytest.raises(ValueError, match="No running SLURM job"):
            slurm.build_launch_cmd()

    def test_empty_nodelist_raises(self, monkeypatch):
        """Raises ValueError when nodelist is empty."""
        monkeypatch.delenv("SLURM_NNODES", raising=False)
        monkeypatch.setattr(
            slurm, "get_slurm_jobid_of_active_job", lambda: "12345"
        )
        monkeypatch.setattr(
            slurm,
            "get_nodelist_from_slurm_jobid",
            lambda jobid: [],
        )
        monkeypatch.setattr(ezpz, "get_gpus_per_node", lambda: 4)
        with pytest.raises(ValueError, match="No nodelist found"):
            slurm.build_launch_cmd()
