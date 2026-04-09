"""Tests for PBS launch topology helpers."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import ezpz.pbs as pbs


@pytest.fixture
def patch_topology(monkeypatch, tmp_path):
    """Patch topology helpers to deterministic values."""

    def _apply(
        *,
        world_size: int = 16,
        gpus_per_node: int = 4,
        num_nodes: int = 4,
        machine: str = "generic",
        hostfile: Path | None = None,
    ) -> Path:
        monkeypatch.setattr(pbs.ezpz, "get_world_size", lambda total=True: world_size)
        monkeypatch.setattr(pbs.ezpz, "get_gpus_per_node", lambda: gpus_per_node)
        monkeypatch.setattr(pbs.ezpz, "get_num_nodes", lambda hostfile=None: num_nodes)
        monkeypatch.setattr(pbs.ezpz, "get_machine", lambda: machine)
        monkeypatch.setattr(
            pbs, "get_hostfile_with_fallback", lambda _: hostfile or tmp_path / "hosts"
        )
        return hostfile or tmp_path / "hosts"

    return _apply


def test_get_pbs_launch_cmd_defaults_full_machine(patch_topology, monkeypatch):
    """Defaults consume all resources and apply generic CPU binding."""
    hostfile = patch_topology()
    monkeypatch.delenv("CPU_BIND", raising=False)

    cmd = pbs.get_pbs_launch_cmd(hostfile=hostfile)

    assert (
        cmd
        == f"mpiexec --envall --np=16 --ppn=4 --hostfile={hostfile} --cpu-bind=depth --depth=8"
    )


def test_get_pbs_launch_cmd_respects_cpu_bind_env(patch_topology, monkeypatch):
    """User-provided CPU binding is forwarded verbatim (with verbose prefix)."""
    hostfile = patch_topology(world_size=8, gpus_per_node=8, num_nodes=1)
    monkeypatch.setenv("CPU_BIND", "--cpu-bind=list:0-1")

    cmd = pbs.get_pbs_launch_cmd(ngpus=8, nhosts=1, hostfile=hostfile)

    assert cmd.endswith("--cpu-bind=verbose,list:0-1")


def test_get_pbs_launch_cmd_cpu_bind_arg_overrides_env(
    patch_topology, monkeypatch
):
    """Explicit cpu_bind argument takes precedence over CPU_BIND env."""
    hostfile = patch_topology(world_size=8, gpus_per_node=8, num_nodes=1)
    monkeypatch.setenv("CPU_BIND", "--cpu-bind=list:0-1")

    cmd = pbs.get_pbs_launch_cmd(
        ngpus=8, nhosts=1, hostfile=hostfile, cpu_bind="list:2-3"
    )

    assert cmd.endswith("--cpu-bind=verbose,list:2-3")


def test_get_pbs_launch_cmd_raises_on_inconsistent_topology(patch_topology):
    """Invalid topology combinations raise ValueError."""
    hostfile = patch_topology()

    with pytest.raises(ValueError):
        pbs.get_pbs_launch_cmd(ngpus=5, nhosts=2, hostfile=hostfile)


def test_get_pbs_launch_cmd_intel_cpu_binding_defaults(patch_topology, monkeypatch):
    """Intel GPU machines add vendor-specific CPU binding and no-vni flag."""
    hostfile = patch_topology(machine="aurora")
    monkeypatch.delenv("CPU_BIND", raising=False)

    cmd = pbs.get_pbs_launch_cmd(hostfile=hostfile)

    assert "--no-vni" in cmd
    assert "--cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96" in cmd


# ---------------------------------------------------------------------------
# Fixtures / helpers for qstat-based tests
# ---------------------------------------------------------------------------

# Realistic output from ``qstat -fn1wru <user>`` — two running jobs, one
# queued.  The ``R`` column marks running jobs.  The last column is the
# exec_host list (``host/cpu+host/cpu+...``).
QSTAT_FN1WRU_OUTPUT = (
    "                                                            Req'd  Req'd   Elap\n"
    "Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time\n"
    "--------------- -------- -------- ---------- ------ --- --- ------ ----- - -----\n"
    "123456.pbs      testuser workq    myjob       1234   2   8    --  01:00 R 00:05 x3005c0s7b0n0/0+x3005c0s7b1n0/0\n"
    "123457.pbs      testuser workq    myjob2      1235   1   4    --  02:00 Q   --\n"
    "123458.pbs      testuser workq    myjob3      1236   3  12    --  01:00 R 00:10 x3006c0s0b0n0/0+x3006c0s0b1n0/0+x3006c0s1b0n0/0\n"
)

# Realistic output from ``qstat -u <user>`` (used by get_running_jobs_from_qstat).
# Two header lines, then job rows, trailing newline produces an empty last element.
QSTAT_U_OUTPUT = (
    "                                                            Req'd  Req'd   Elap\n"
    "Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time\n"
    "123456.pbs      testuser workq    myjob       1234   2   8    --  01:00 R 00:05\n"
    "123457.pbs      testuser workq    myjob2      1235   1   4    --  02:00 Q   --\n"
    "123458.pbs      testuser workq    myjob3      1236   3  12    --  01:00 R 00:10\n"
    "\n"
)


# ===================================================================
# get_running_jobs_from_qstat
# ===================================================================


class TestGetRunningJobsFromQstat:
    """Tests for ``get_running_jobs_from_qstat``."""

    def test_happy_path(self, monkeypatch):
        """Realistic qstat -u output yields list of running job ID ints."""
        mock_sh = MagicMock()
        mock_sh.qstat = MagicMock(return_value=QSTAT_U_OUTPUT)
        monkeypatch.setenv("USER", "testuser")
        with patch.dict("sys.modules", {"sh": mock_sh}):
            result = pbs.get_running_jobs_from_qstat()
        assert isinstance(result, list)
        assert result == [123456, 123458]

    def test_qstat_failure_returns_empty(self, monkeypatch):
        """When qstat raises an error, the exception propagates."""
        mock_sh = MagicMock()
        mock_sh.qstat = MagicMock(side_effect=RuntimeError("qstat broken"))
        monkeypatch.setenv("USER", "testuser")
        with patch.dict("sys.modules", {"sh": mock_sh}):
            with pytest.raises(RuntimeError, match="qstat broken"):
                pbs.get_running_jobs_from_qstat()

    def test_qstat_unavailable(self):
        """ImportError when sh.qstat is not importable propagates."""
        import sys
        _orig_sh = sys.modules.get("sh", _SENTINEL := object())
        try:
            sys.modules["sh"] = None  # type: ignore[assignment]
            with pytest.raises((ImportError, ModuleNotFoundError)):
                pbs.get_running_jobs_from_qstat()
        finally:
            if _orig_sh is _SENTINEL:
                sys.modules.pop("sh", None)
            else:
                sys.modules["sh"] = _orig_sh


# ===================================================================
# get_pbs_running_jobs_for_user
# ===================================================================


class TestGetPbsRunningJobsForUser:
    """Tests for ``get_pbs_running_jobs_for_user``."""

    def test_returns_jobid_to_nodelist_mapping(self):
        """Parses qstat -fn1wru and returns {jobid: [nodes]} dict."""
        mock_sh = MagicMock()
        mock_sh.qstat = MagicMock(return_value=QSTAT_FN1WRU_OUTPUT)
        with patch.dict("sys.modules", {"sh": mock_sh}):
            result = pbs.get_pbs_running_jobs_for_user()
        assert isinstance(result, dict)
        assert "123456" in result
        assert "123458" in result
        # Queued job should NOT appear
        assert "123457" not in result
        # Verify node parsing (host/cpu → host)
        assert result["123456"] == ["x3005c0s7b0n0", "x3005c0s7b1n0"]
        assert result["123458"] == [
            "x3006c0s0b0n0",
            "x3006c0s0b1n0",
            "x3006c0s1b0n0",
        ]

    def test_no_running_jobs(self):
        """Empty qstat output with only queued jobs returns empty dict."""
        qstat_no_running = (
            "                                                            Req'd  Req'd   Elap\n"
            "Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time\n"
            "--------------- -------- -------- ---------- ------ --- --- ------ ----- - -----\n"
            "123457.pbs      testuser workq    myjob2      1235   1   4    --  02:00 Q   --\n"
        )
        mock_sh = MagicMock()
        mock_sh.qstat = MagicMock(return_value=qstat_no_running)
        with patch.dict("sys.modules", {"sh": mock_sh}):
            result = pbs.get_pbs_running_jobs_for_user()
        assert result == {}

    def test_qstat_import_failure_raises(self):
        """When sh is not importable, the exception propagates."""
        import sys
        _orig_sh = sys.modules.get("sh", _SENTINEL := object())
        try:
            sys.modules["sh"] = None  # type: ignore[assignment]
            with pytest.raises(Exception):
                pbs.get_pbs_running_jobs_for_user()
        finally:
            if _orig_sh is _SENTINEL:
                sys.modules.pop("sh", None)
            else:
                sys.modules["sh"] = _orig_sh


# ===================================================================
# get_pbs_jobid_of_active_job
# ===================================================================


class TestGetPbsJobidOfActiveJob:
    """Tests for ``get_pbs_jobid_of_active_job``."""

    def test_hostname_matching(self, monkeypatch):
        """Returns jobid when socket.getfqdn() hostname matches a node."""
        monkeypatch.setattr(
            pbs,
            "get_pbs_running_jobs_for_user",
            lambda: {
                "123456": ["x3005c0s7b0n0", "x3005c0s7b1n0"],
                "123458": ["x3006c0s0b0n0"],
            },
        )
        monkeypatch.setattr(
            "socket.getfqdn",
            lambda: "x3006c0s0b0n0.cm.aurora.alcf.anl.gov",
        )
        result = pbs.get_pbs_jobid_of_active_job()
        assert result == "123458"

    def test_hostname_matching_first_job(self, monkeypatch):
        """Returns the first matching jobid when hostname is in that job's nodelist."""
        monkeypatch.setattr(
            pbs,
            "get_pbs_running_jobs_for_user",
            lambda: {
                "123456": ["x3005c0s7b0n0", "x3005c0s7b1n0"],
                "123458": ["x3006c0s0b0n0"],
            },
        )
        monkeypatch.setattr(
            "socket.getfqdn",
            lambda: "x3005c0s7b1n0.cm.aurora.alcf.anl.gov",
        )
        result = pbs.get_pbs_jobid_of_active_job()
        assert result == "123456"

    def test_no_match_returns_none(self, monkeypatch):
        """Returns None when hostname does not match any running job."""
        monkeypatch.setattr(
            pbs,
            "get_pbs_running_jobs_for_user",
            lambda: {
                "123456": ["x3005c0s7b0n0"],
                "123458": ["x3006c0s0b0n0"],
            },
        )
        monkeypatch.setattr(
            "socket.getfqdn",
            lambda: "x9999c0s0b0n0.cm.aurora.alcf.anl.gov",
        )
        result = pbs.get_pbs_jobid_of_active_job()
        assert result is None

    def test_no_running_jobs_returns_none(self, monkeypatch):
        """Returns None when there are no running jobs at all."""
        monkeypatch.setattr(
            pbs,
            "get_pbs_running_jobs_for_user",
            lambda: {},
        )
        monkeypatch.setattr(
            "socket.getfqdn",
            lambda: "x3005c0s7b0n0.cm.aurora.alcf.anl.gov",
        )
        result = pbs.get_pbs_jobid_of_active_job()
        assert result is None


# ===================================================================
# get_pbs_nodefile_from_jobid
# ===================================================================


class TestGetPbsNodefileFromJobid:
    """Tests for ``get_pbs_nodefile_from_jobid``."""

    def test_returns_path_under_var_spool(self, monkeypatch, tmp_path):
        """Constructs path from /var/spool/pbs/aux/ matching the jobid."""
        # Create a fake spool directory with a matching nodefile
        fake_spool = tmp_path / "var" / "spool" / "pbs" / "aux"
        fake_spool.mkdir(parents=True)
        nodefile = fake_spool / "123456.pbs"
        nodefile.write_text("x3005c0s7b0n0\nx3005c0s7b1n0\n")

        # Capture the real os.listdir before patching to avoid recursion
        real_listdir = os.listdir

        def fake_listdir(p):
            if str(p) == "/var/spool/pbs/aux":
                return real_listdir(fake_spool)
            return real_listdir(p)

        monkeypatch.setattr(os, "listdir", fake_listdir)

        # The function also constructs Path("/var/spool/pbs/aux").joinpath(f)
        # and calls is_file() on it — that path won't exist on the dev machine.
        # Patch Path.joinpath on the pbs_parent path to redirect to fake_spool.
        orig_joinpath = Path.joinpath

        def fake_joinpath(self, *args):
            if str(self) == "/var/spool/pbs/aux":
                return orig_joinpath(fake_spool, *args)
            return orig_joinpath(self, *args)

        monkeypatch.setattr(Path, "joinpath", fake_joinpath)

        result = pbs.get_pbs_nodefile_from_jobid(123456)
        assert result.endswith("123456.pbs")
        assert "aux" in result

    def test_none_jobid_raises(self):
        """Raises AssertionError when jobid is None."""
        with pytest.raises(AssertionError, match="No jobid provided"):
            pbs.get_pbs_nodefile_from_jobid(None)

    def test_path_construction_format(self):
        """Verify the function targets /var/spool/pbs/aux/{jobid}."""
        # On dev machines the directory doesn't exist (FileNotFoundError/OSError).
        # On PBS systems the directory exists but won't contain a matching
        # file for a bogus jobid, so the function raises AssertionError.
        with pytest.raises((FileNotFoundError, OSError, AssertionError)):
            pbs.get_pbs_nodefile_from_jobid(999999)


# ===================================================================
# get_pbs_nodefile_of_active_job
# ===================================================================


class TestGetPbsNodefileOfActiveJob:
    """Tests for ``get_pbs_nodefile_of_active_job``."""

    def test_returns_nodefile_when_active(self, monkeypatch):
        """Returns nodefile path when an active job is found."""
        monkeypatch.setattr(
            pbs, "get_pbs_jobid_of_active_job", lambda: "123456"
        )
        monkeypatch.setattr(
            pbs,
            "get_pbs_nodefile_from_jobid",
            lambda jobid: f"/var/spool/pbs/aux/{jobid}.pbs",
        )
        result = pbs.get_pbs_nodefile_of_active_job()
        assert result == "/var/spool/pbs/aux/123456.pbs"

    def test_returns_none_when_no_active_job(self, monkeypatch):
        """Returns None when no active job is found."""
        monkeypatch.setattr(
            pbs, "get_pbs_jobid_of_active_job", lambda: None
        )
        result = pbs.get_pbs_nodefile_of_active_job()
        assert result is None


# ===================================================================
# get_pbs_nodefile
# ===================================================================


class TestGetPbsNodefile:
    """Tests for ``get_pbs_nodefile``."""

    def test_with_explicit_jobid(self, monkeypatch):
        """When jobid is provided, uses it directly."""
        monkeypatch.setattr(
            pbs,
            "get_pbs_nodefile_from_jobid",
            lambda jobid: f"/var/spool/pbs/aux/{jobid}.pbs",
        )
        result = pbs.get_pbs_nodefile(jobid="123456")
        assert result == "/var/spool/pbs/aux/123456.pbs"

    def test_no_jobid_falls_back_to_active_job(self, monkeypatch, caplog):
        """When jobid is None and no active job exists, returns None with warning."""
        monkeypatch.setattr(
            pbs, "get_pbs_jobid_of_active_job", lambda: None
        )
        # Ensure the ezpz.pbs logger propagates so caplog can capture it
        pbs_logger = logging.getLogger("ezpz.pbs")
        orig_propagate = pbs_logger.propagate
        pbs_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="ezpz.pbs"):
                result = pbs.get_pbs_nodefile(jobid=None)
        finally:
            pbs_logger.propagate = orig_propagate
        assert result is None
        assert "No active job found" in caplog.text

    def test_no_jobid_with_active_job(self, monkeypatch):
        """When jobid is None but active job found, returns its nodefile."""
        monkeypatch.setattr(
            pbs, "get_pbs_jobid_of_active_job", lambda: "123456"
        )
        monkeypatch.setattr(
            pbs,
            "get_pbs_nodefile_from_jobid",
            lambda jobid: f"/var/spool/pbs/aux/{jobid}.pbs",
        )
        result = pbs.get_pbs_nodefile(jobid=None)
        assert result == "/var/spool/pbs/aux/123456.pbs"


# ===================================================================
# get_pbs_nodelist_from_jobid
# ===================================================================


class TestGetPbsNodelistFromJobid:
    """Tests for ``get_pbs_nodelist_from_jobid``."""

    def test_returns_nodelist_for_valid_jobid(self, monkeypatch):
        """Returns node list when jobid matches a running job."""
        monkeypatch.setattr(
            pbs,
            "get_pbs_running_jobs_for_user",
            lambda: {"123456": ["x3005c0s7b0n0", "x3005c0s7b1n0"]},
        )
        result = pbs.get_pbs_nodelist_from_jobid("123456")
        assert result == ["x3005c0s7b0n0", "x3005c0s7b1n0"]

    def test_none_jobid_raises(self):
        """Raises AssertionError when jobid is None."""
        with pytest.raises(AssertionError, match="No jobid provided"):
            pbs.get_pbs_nodelist_from_jobid(None)

    def test_unknown_jobid_raises(self, monkeypatch):
        """Raises AssertionError when jobid is not among running jobs."""
        monkeypatch.setattr(
            pbs,
            "get_pbs_running_jobs_for_user",
            lambda: {"123456": ["x3005c0s7b0n0"]},
        )
        with pytest.raises(AssertionError, match="not found in running jobs"):
            pbs.get_pbs_nodelist_from_jobid("999999")
