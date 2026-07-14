"""Tests for PBS launch topology helpers."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

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
        == f"mpiexec --envall --line-buffer --np=16 --ppn=4 --hostfile={hostfile} --cpu-bind=depth --depth=8"
    )


def test_get_pbs_launch_cmd_respects_cpu_bind_env(patch_topology, monkeypatch):
    """User-provided CPU binding is forwarded verbatim (with verbose prefix)."""
    hostfile = patch_topology(world_size=8, gpus_per_node=8, num_nodes=1)
    monkeypatch.setenv("CPU_BIND", "--cpu-bind=list:0-1")

    cmd = pbs.get_pbs_launch_cmd(ngpus=8, nhosts=1, hostfile=hostfile)

    assert cmd.endswith("--cpu-bind=list:0-1")


def test_get_pbs_launch_cmd_cpu_bind_arg_overrides_env(
    patch_topology, monkeypatch
):
    """Explicit cpu_bind argument takes precedence over CPU_BIND env."""
    hostfile = patch_topology(world_size=8, gpus_per_node=8, num_nodes=1)
    monkeypatch.setenv("CPU_BIND", "--cpu-bind=list:0-1")

    cmd = pbs.get_pbs_launch_cmd(
        ngpus=8, nhosts=1, hostfile=hostfile, cpu_bind="list:2-3"
    )

    assert cmd.endswith("--cpu-bind=list:2-3")


def test_get_pbs_launch_cmd_raises_on_inconsistent_topology(patch_topology):
    """Invalid topology combinations raise ValueError."""
    hostfile = patch_topology()

    with pytest.raises(ValueError):
        pbs.get_pbs_launch_cmd(ngpus=5, nhosts=2, hostfile=hostfile)


def test_get_pbs_launch_cmd_intel_cpu_binding_defaults(patch_topology, monkeypatch):
    """Intel GPU machines add vendor-specific CPU binding (no longer --no-vni).

    Pre-v0.18.x this also auto-added `--no-vni` for a transient
    network-interface workaround; that was dropped. Users who still
    need it can pass it via launcher passthrough or set CPU_BIND.
    """
    hostfile = patch_topology(machine="aurora")
    monkeypatch.delenv("CPU_BIND", raising=False)

    cmd = pbs.get_pbs_launch_cmd(hostfile=hostfile)

    assert "--no-vni" not in cmd, (
        "--no-vni should no longer be auto-added on Aurora/Sunspot"
    )
    assert "--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100" in cmd


# ---------------------------------------------------------------------------
# _run_qstat_with_retry
# ---------------------------------------------------------------------------


class TestRunQstatWithRetry:
    """Tests for ``_run_qstat_with_retry``."""

    def test_succeeds_on_first_try(self):
        """Returns output when qstat succeeds immediately."""
        fn = MagicMock(return_value="output")
        result = pbs._run_qstat_with_retry(fn, "-u", "testuser")
        assert result == "output"
        assert fn.call_count == 1

    def test_retries_on_communication_failure(self, monkeypatch):
        """Retries on transient PBS server errors and succeeds."""
        monkeypatch.setattr(pbs, "_QSTAT_RETRY_DELAY", 0)
        fn = MagicMock(
            side_effect=[
                Exception("Communication failure."),
                Exception("cannot connect to server foo"),
                "output",
            ]
        )
        result = pbs._run_qstat_with_retry(fn, "-u", "testuser")
        assert result == "output"
        assert fn.call_count == 3

    def test_raises_after_max_retries(self, monkeypatch):
        """Raises after exhausting all retries."""
        monkeypatch.setattr(pbs, "_QSTAT_RETRY_DELAY", 0)
        monkeypatch.setattr(pbs, "_QSTAT_MAX_RETRIES", 3)
        fn = MagicMock(side_effect=Exception("Communication failure."))
        with pytest.raises(Exception, match="Communication failure"):
            pbs._run_qstat_with_retry(fn, "-u", "testuser")
        assert fn.call_count == 3

    def test_non_transient_error_raises_immediately(self):
        """Non-transient errors are not retried."""
        fn = MagicMock(side_effect=RuntimeError("something else"))
        with pytest.raises(RuntimeError, match="something else"):
            pbs._run_qstat_with_retry(fn, "-u", "testuser")
        assert fn.call_count == 1


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
        monkeypatch.setenv("USER", "testuser")
        monkeypatch.setattr(pbs.shutil, "which", lambda _c: "/usr/bin/qstat")
        monkeypatch.setattr(
            pbs.subprocess,
            "run",
            lambda *a, **k: MagicMock(returncode=0, stdout=QSTAT_U_OUTPUT, stderr=""),
        )
        result = pbs.get_running_jobs_from_qstat()
        assert isinstance(result, list)
        assert result == [123456, 123458]

    def test_qstat_failure_returns_empty(self, monkeypatch):
        """When qstat exits non-zero, a RuntimeError propagates."""
        monkeypatch.setenv("USER", "testuser")
        monkeypatch.setattr(pbs.shutil, "which", lambda _c: "/usr/bin/qstat")
        monkeypatch.setattr(
            pbs.subprocess,
            "run",
            lambda *a, **k: MagicMock(returncode=1, stdout="", stderr="qstat broken"),
        )
        with pytest.raises(RuntimeError, match="qstat broken"):
            pbs.get_running_jobs_from_qstat()

    def test_qstat_unavailable(self, monkeypatch):
        """FileNotFoundError when the qstat binary isn't on PATH propagates."""
        monkeypatch.setenv("USER", "testuser")
        monkeypatch.setattr(pbs.shutil, "which", lambda _c: None)
        with pytest.raises(FileNotFoundError):
            pbs.get_running_jobs_from_qstat()


# ===================================================================
# get_pbs_running_jobs_for_user
# ===================================================================


class TestGetPbsRunningJobsForUser:
    """Tests for ``get_pbs_running_jobs_for_user``."""

    def setup_method(self):
        pbs._pbs_jobs_cache = None

    def test_returns_jobid_to_nodelist_mapping(self, monkeypatch):
        """Parses qstat -fn1wru and returns {jobid: [nodes]} dict."""
        monkeypatch.setattr(pbs.shutil, "which", lambda _c: "/usr/bin/qstat")
        monkeypatch.setattr(
            pbs.subprocess,
            "run",
            lambda *a, **k: MagicMock(
                returncode=0, stdout=QSTAT_FN1WRU_OUTPUT, stderr=""
            ),
        )
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

    def test_no_running_jobs(self, monkeypatch):
        """Empty qstat output with only queued jobs returns empty dict."""
        qstat_no_running = (
            "                                                            Req'd  Req'd   Elap\n"
            "Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time\n"
            "--------------- -------- -------- ---------- ------ --- --- ------ ----- - -----\n"
            "123457.pbs      testuser workq    myjob2      1235   1   4    --  02:00 Q   --\n"
        )
        monkeypatch.setattr(pbs.shutil, "which", lambda _c: "/usr/bin/qstat")
        monkeypatch.setattr(
            pbs.subprocess,
            "run",
            lambda *a, **k: MagicMock(
                returncode=0, stdout=qstat_no_running, stderr=""
            ),
        )
        result = pbs.get_pbs_running_jobs_for_user()
        assert result == {}

    def test_qstat_import_failure_raises(self, monkeypatch):
        """When the qstat binary isn't on PATH, the exception propagates."""
        monkeypatch.setattr(pbs.shutil, "which", lambda _c: None)
        with pytest.raises(FileNotFoundError):
            pbs.get_pbs_running_jobs_for_user()


# ===================================================================
# get_pbs_jobid_of_active_job
# ===================================================================


class TestGetPbsJobidOfActiveJob:
    """Tests for ``get_pbs_jobid_of_active_job``.

    All tests in this class must clear ``$PBS_JOBID`` first so the
    qstat-based slow path is actually exercised. Without that
    delenv, a CI runner (or login-node) with ``PBS_JOBID`` set
    would short-circuit on the fast path and silently bypass the
    qstat mocks.
    """

    def test_hostname_matching(self, monkeypatch):
        """Returns jobid when socket.getfqdn() hostname matches a node."""
        monkeypatch.delenv("PBS_JOBID", raising=False)
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
        monkeypatch.delenv("PBS_JOBID", raising=False)
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
        monkeypatch.delenv("PBS_JOBID", raising=False)
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
        monkeypatch.delenv("PBS_JOBID", raising=False)
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

    # --- Fast path: $PBS_JOBID short-circuit -----------------------------

    def test_pbs_jobid_env_short_circuits_qstat(self, monkeypatch):
        """Setting $PBS_JOBID must bypass the qstat lookup entirely.

        Regression for the Aurora compute-node bug where the qstat
        fallback failed with ImportError because the qstat binary
        isn't on $PATH on compute nodes. Mocking the slow path to
        explode proves we never reach it.
        """
        monkeypatch.setenv(
            "PBS_JOBID", "12345.aurora-pbs-0001.head.cm.aurora.alcf.anl.gov"
        )
        monkeypatch.delenv("PBS_NODEFILE", raising=False)

        def _explode():
            raise ImportError(
                "qstat not on PATH — fast path failed to short-circuit"
            )

        monkeypatch.setattr(
            pbs, "get_pbs_running_jobs_for_user", _explode
        )
        result = pbs.get_pbs_jobid_of_active_job()
        # Server suffix stripped to match the rest of ezpz's jobid handling.
        assert result == "12345"

    def test_pbs_jobid_env_empty_string_falls_back(self, monkeypatch):
        """Empty $PBS_JOBID must fall through to qstat (truthy check).

        Pins the ``if pbs_jobid:`` truthy semantics so future "code
        cleanup" can't silently change it to ``is not None`` (which
        would short-circuit on an empty string and return ``""``).
        """
        monkeypatch.setenv("PBS_JOBID", "")
        monkeypatch.delenv("PBS_NODEFILE", raising=False)
        called = []
        monkeypatch.setattr(
            pbs,
            "get_pbs_running_jobs_for_user",
            lambda: (called.append(1), {})[1],
        )
        monkeypatch.setattr("socket.getfqdn", lambda: "no-match.local")
        result = pbs.get_pbs_jobid_of_active_job()
        assert called, (
            "Empty PBS_JOBID must NOT short-circuit; qstat fallback "
            "should still run."
        )
        assert result is None

    def test_pbs_jobid_unset_falls_back_to_qstat(self, monkeypatch):
        """When $PBS_JOBID is absent we must walk the qstat path."""
        monkeypatch.delenv("PBS_JOBID", raising=False)
        monkeypatch.delenv("PBS_NODEFILE", raising=False)
        called = []
        monkeypatch.setattr(
            pbs,
            "get_pbs_running_jobs_for_user",
            lambda: (called.append(1), {"99999": ["host-a"]})[1],
        )
        monkeypatch.setattr("socket.getfqdn", lambda: "host-a.local")
        result = pbs.get_pbs_jobid_of_active_job()
        assert called == [1]
        assert result == "99999"

    def test_pbs_jobid_with_no_dot_returned_as_is(self, monkeypatch):
        """SLURM-style bare numeric jobid (no `.suffix`) shouldn't error.

        ``.split(".")[0]`` on a string without ``.`` returns the
        whole string — pin this so a future refactor to
        ``.split(".", 1)[0]`` etc. doesn't quietly break.
        """
        monkeypatch.setenv("PBS_JOBID", "987654")
        monkeypatch.delenv("PBS_NODEFILE", raising=False)
        # Slow path mocked away — fast path should win.
        monkeypatch.setattr(
            pbs,
            "get_pbs_running_jobs_for_user",
            lambda: pytest.fail("must not reach qstat path"),
        )
        assert pbs.get_pbs_jobid_of_active_job() == "987654"

    # --- Fast-path hostname cross-check ----------------------------------

    def test_nodefile_hostname_match_returns_jobid(self, monkeypatch, tmp_path):
        """When PBS_NODEFILE is present AND contains our hostname, trust $PBS_JOBID."""
        nodefile = tmp_path / "nodefile"
        nodefile.write_text("x3005c0s7b0n0\nx3005c0s7b1n0\n")
        monkeypatch.setenv("PBS_JOBID", "55555.foo.bar")
        monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
        monkeypatch.setattr(
            "socket.getfqdn",
            lambda: "x3005c0s7b1n0.cm.aurora.alcf.anl.gov",
        )
        # Slow path mocked to fail — proves we returned via fast path.
        monkeypatch.setattr(
            pbs,
            "get_pbs_running_jobs_for_user",
            lambda: pytest.fail("must not reach qstat path"),
        )
        assert pbs.get_pbs_jobid_of_active_job() == "55555"

    def test_nodefile_fqdn_match_aurora_compute_node(
        self, monkeypatch, tmp_path
    ):
        """Aurora-shaped FQDNs in PBS_NODEFILE + short local hostname.

        Regression for the false-positive caught on Aurora job 8518207:
        PBS_NODEFILE on Aurora contains
        `x4114c1s7b0n0.hostmgmt2.cm.aurora.alcf.anl.gov` (FQDN with
        the hostmgmt2 suffix) while `socket.getfqdn()` on compute
        nodes returns just `x4114c1s7b0n0` (reverse DNS doesn't
        include the hostmgmt suffix inside a job). The check must
        strip both sides to the short hostname before membership.

        Without the fix: every Aurora call logs a misleading
        "hostname not in PBS_NODEFILE" warning and burns a qstat
        round-trip on every rank, even though the job is correct.
        """
        nodefile = tmp_path / "nodefile"
        nodefile.write_text(
            "x4114c1s0b0n0.hostmgmt2.cm.aurora.alcf.anl.gov\n"
            "x4114c1s7b0n0.hostmgmt2.cm.aurora.alcf.anl.gov\n"
        )
        monkeypatch.setenv(
            "PBS_JOBID",
            "8518207.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov",
        )
        monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
        # On compute nodes inside a job, getfqdn returns the SHORT name.
        monkeypatch.setattr("socket.getfqdn", lambda: "x4114c1s7b0n0")
        monkeypatch.setattr(
            pbs,
            "get_pbs_running_jobs_for_user",
            lambda: pytest.fail("must not reach qstat path"),
        )
        assert pbs.get_pbs_jobid_of_active_job() == "8518207"

    def test_nodefile_hostname_mismatch_falls_through_to_qstat(
        self, monkeypatch, tmp_path, caplog
    ):
        """When PBS_NODEFILE doesn't contain our hostname, $PBS_JOBID is
        lying — log a debug message and fall through to qstat.

        Defends against the (rare) case of a stale PBS_JOBID export
        or running ezpz from an ssh-into-compute-node shell of a
        different job.

        Logged at DEBUG (not WARNING) because the function
        self-recovers via qstat — surfacing the message at WARNING
        on 96-rank jobs would produce 96 lines of yellow output for
        a condition that's already handled.
        """
        nodefile = tmp_path / "nodefile"
        nodefile.write_text("x9999c0s0b0n0\n")  # Some other host
        monkeypatch.setenv("PBS_JOBID", "11111.foo.bar")
        monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
        monkeypatch.setattr(
            "socket.getfqdn",
            lambda: "x3005c0s7b1n0.cm.aurora.alcf.anl.gov",
        )
        # Slow path returns the actually-correct jobid for our host.
        monkeypatch.setattr(
            pbs,
            "get_pbs_running_jobs_for_user",
            lambda: {"22222": ["x3005c0s7b1n0"]},
        )
        # Capture at DEBUG so we see the message — it was at WARNING
        # before, demoted because of 96-rank spam.
        with caplog.at_level(logging.DEBUG, logger=pbs.logger.name):
            result = pbs.get_pbs_jobid_of_active_job()
        assert result == "22222"
        assert any(
            "is not in $PBS_NODEFILE" in rec.message
            and rec.levelname == "DEBUG"
            for rec in caplog.records
        ), "Expected a DEBUG message about the hostname mismatch."

    def test_nodefile_unreadable_falls_through_silently(
        self, monkeypatch, tmp_path
    ):
        """If PBS_NODEFILE is set but unreadable (deleted, permissions),
        treat it as 'no cross-check available' and trust $PBS_JOBID.

        Don't bail or fall through to qstat — the file's absence
        doesn't prove $PBS_JOBID is wrong; it usually just means we're
        in a subprocess that lost the file descriptor or the file got
        cleaned up. The original motivation for the fast path was
        compute-node subprocesses; making them THIS sensitive to
        filesystem state defeats the point.
        """
        monkeypatch.setenv("PBS_JOBID", "77777.foo.bar")
        monkeypatch.setenv(
            "PBS_NODEFILE", str(tmp_path / "does-not-exist")
        )
        monkeypatch.setattr(
            pbs,
            "get_pbs_running_jobs_for_user",
            lambda: pytest.fail("must not reach qstat path"),
        )
        assert pbs.get_pbs_jobid_of_active_job() == "77777"


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
