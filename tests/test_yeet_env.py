"""Tests for ``ezpz.utils.yeet_env``.

Covers environment detection, node discovery, rsync execution,
argument parsing, and the main run() flow.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import ezpz.utils.yeet_env as yeet


# ===================================================================
# _detect_env_source
# ===================================================================


class TestDetectEnvSource:
    """Tests for ``_detect_env_source``."""

    def test_returns_path(self):
        """Returns a resolved Path."""
        result = yeet._detect_env_source()
        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_returns_sys_prefix_in_venv(self):
        """When in a venv, returns sys.prefix."""
        import sys

        # In the test venv, sys.prefix != sys.base_prefix
        if sys.prefix != sys.base_prefix:
            assert yeet._detect_env_source() == Path(sys.prefix).resolve()


# ===================================================================
# _get_worker_nodes
# ===================================================================


class TestGetWorkerNodes:
    """Tests for ``_get_worker_nodes``."""

    def test_reads_hostfile(self, tmp_path):
        """Reads unique hostnames from a hostfile."""
        hf = tmp_path / "hostfile"
        hf.write_text("node01\nnode01\nnode02\nnode03\nnode02\n")
        nodes = yeet._get_worker_nodes(hostfile=str(hf))
        assert nodes == ["node01", "node02", "node03"]

    def test_strips_fqdn(self, tmp_path):
        """Strips FQDN suffixes to short hostnames."""
        hf = tmp_path / "hostfile"
        hf.write_text("node01.cluster.local\nnode02.cluster.local\n")
        nodes = yeet._get_worker_nodes(hostfile=str(hf))
        assert nodes == ["node01", "node02"]

    def test_deduplicates(self, tmp_path):
        """Deduplicates hostnames (common in PBS nodefiles)."""
        hf = tmp_path / "hostfile"
        # PBS nodefiles repeat hostnames for each GPU
        hf.write_text("\n".join(["node01"] * 4 + ["node02"] * 4))
        nodes = yeet._get_worker_nodes(hostfile=str(hf))
        assert nodes == ["node01", "node02"]


# ===================================================================
# _rsync_to_node
# ===================================================================


class TestSafeRmtree:
    """Tests for ``_safe_rmtree``."""

    @patch("ezpz.utils.yeet_env.subprocess.run")
    def test_allows_tmp_path(self, mock_run):
        """Paths under /tmp/ are allowed."""
        dst = Path("/tmp/test-venv")
        # Mock resolve to return the same path
        result = yeet._safe_rmtree(dst)
        assert result is True
        mock_run.assert_called_once()

    def test_refuses_non_tmp_path(self):
        """Paths outside /tmp/ are refused."""
        result = yeet._safe_rmtree(Path("/home/user/.venv"))
        assert result is False

    def test_refuses_traversal(self):
        """Refuses paths that traverse out of /tmp/ via '..'."""
        result = yeet._safe_rmtree(Path("/tmp/../home/user"))
        assert result is False

    def test_refuses_tmp_root(self):
        """Refuses /tmp/ itself."""
        result = yeet._safe_rmtree(Path("/tmp"))
        assert result is False


class TestRsyncToNode:
    """Tests for ``_rsync_to_node``."""

    @patch("ezpz.utils.yeet_env.subprocess.run")
    def test_success(self, mock_run, tmp_path):
        """Successful rsync returns (node, elapsed, 0)."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        src = tmp_path / "env"
        src.mkdir()
        node, elapsed, rc = yeet._rsync_to_node(src, Path("/tmp/env"), "node01")
        assert node == "node01"
        assert rc == 0
        assert elapsed >= 0
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "rsync"
        assert "-rlD" in call_args
        assert call_args[-1] == "node01:/tmp/env/"

    @patch("ezpz.utils.yeet_env.subprocess.run")
    def test_failure(self, mock_run, tmp_path):
        """Failed rsync returns non-zero returncode."""
        mock_run.return_value = MagicMock(
            returncode=23, stderr="some files could not be transferred"
        )
        src = tmp_path / "env"
        src.mkdir()
        node, elapsed, rc = yeet._rsync_to_node(src, Path("/tmp/env"), "node01")
        assert node == "node01"
        assert rc == 23

    @patch("ezpz.utils.yeet_env.subprocess.run")
    def test_trailing_slash_on_src(self, mock_run, tmp_path):
        """Source path gets a trailing slash for rsync content sync."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        src = tmp_path / "env"
        src.mkdir()
        yeet._rsync_to_node(src, Path("/tmp/env"), "node01")
        call_args = mock_run.call_args[0][0]
        src_arg = [a for a in call_args if str(tmp_path) in a][0]
        assert src_arg.endswith("/")


# ===================================================================
# rsync retry on transient failure
# ===================================================================


class TestRsyncRetry:
    """Verify the per-target retry loop in the fan-out completion handler.

    The retry block lives inside ``run()`` (an end-to-end orchestrator
    that does tarball detection, ssh-extracts, etc.) so we can't easily
    unit-test it in isolation. Instead, mock ``_rsync_to_node`` at the
    module boundary and drive the full pipeline against a fake hostfile.
    """

    def _make_hostfile(self, tmp_path, nodes):
        hf = tmp_path / "hostfile"
        hf.write_text("\n".join(nodes) + "\n")
        return hf

    def _make_src(self, tmp_path):
        """Build a minimal venv-shaped dir so run() recognizes it."""
        src = tmp_path / "fakenv"
        (src / "bin").mkdir(parents=True)
        (src / "bin" / "activate").write_text("# fake activate\n")
        (src / "pyvenv.cfg").write_text(
            "home = /usr/bin\nversion = 3.12.0\n"
        )
        return src

    def test_retry_succeeds_on_second_attempt(
        self, tmp_path, monkeypatch
    ):
        """One node fails initially then succeeds on retry — final state
        should record rc=0 for that node, not a failure."""
        # Fail-then-succeed sequence per target node.
        # Map: node -> list of returncodes to return on each call.
        call_log = {"node01": [], "node02": []}
        flake_count = {"node01": 0}  # node01 fails its first attempt

        def fake_rsync(src, dst, node, **kwargs):
            call_log.setdefault(node, []).append(node)
            if node == "node01" and flake_count[node] < 1:
                flake_count[node] += 1
                return (node, 0.1, 255)  # transient ssh fail
            return (node, 0.2, 0)

        # Force retries=2 (default) so we have headroom; ensure rsync
        # local-copy step succeeds too.
        monkeypatch.setattr(yeet, "_DEFAULT_RSYNC_RETRIES", 2)
        monkeypatch.setattr(yeet, "_rsync_to_node", fake_rsync)
        # Bypass real hostname detection.
        monkeypatch.setattr(yeet, "_get_current_hostname", lambda: "node00")
        # Force the local-copy path regardless of platform. Without
        # this, pytest's tmp_path under /tmp on Linux runners makes
        # run() skip the local-copy step and fan out from `dst`
        # directly — a different code path than the test claims to
        # model. On macOS tmp_path is under /var/folders so this is
        # already True, but pin it for consistency.
        monkeypatch.setattr(yeet, "_needs_local_copy", lambda src: True)
        # Skip the local venv-paths patcher (only relevant after a real copy).
        monkeypatch.setattr(
            yeet, "_patch_venv_paths_local", lambda dst, src: None
        )

        src = self._make_src(tmp_path)
        dst = tmp_path / "tmp_dst"
        hf = self._make_hostfile(tmp_path, ["node00", "node01", "node02"])

        rc = yeet.run(["--src", str(src), "--dst", str(dst),
                       "--hostfile", str(hf)])

        # The wrapper should have retried node01 and the whole run
        # should succeed.
        assert rc == 0, "run() should return 0 when retries recover"
        # node01 should have been attempted twice (initial + 1 retry);
        # node02 should have been attempted once.
        assert len(call_log["node01"]) == 2
        assert len(call_log["node02"]) == 1

    def test_retry_exhausted_records_failure(
        self, tmp_path, monkeypatch
    ):
        """A persistently failing node should be retried RETRIES extra
        times, then recorded as a final failure (not infinite loop)."""
        call_log = {"badnode": []}

        def fake_rsync(src, dst, node, **kwargs):
            call_log.setdefault(node, []).append(node)
            if node == "badnode":
                return (node, 0.1, 255)  # always fails
            return (node, 0.2, 0)

        monkeypatch.setattr(yeet, "_DEFAULT_RSYNC_RETRIES", 2)
        monkeypatch.setattr(yeet, "_rsync_to_node", fake_rsync)
        monkeypatch.setattr(yeet, "_get_current_hostname", lambda: "node00")
        monkeypatch.setattr(yeet, "_needs_local_copy", lambda src: True)
        monkeypatch.setattr(
            yeet, "_patch_venv_paths_local", lambda dst, src: None
        )

        src = self._make_src(tmp_path)
        dst = tmp_path / "tmp_dst"
        hf = self._make_hostfile(
            tmp_path, ["node00", "badnode", "goodnode"]
        )

        rc = yeet.run(["--src", str(src), "--dst", str(dst),
                       "--hostfile", str(hf)])

        # run() must report failure (non-zero rc) when any node
        # exhausts retries — otherwise a regression that silently
        # returns 0 despite failed nodes would slip through and
        # downstream training would burn its allocation thinking
        # yeet succeeded.
        assert rc != 0, "run() must return non-zero when a node exhausts retries"

        # badnode should have been attempted exactly RETRIES + 1 = 3 times
        # (1 initial + 2 retries), then bounded — not infinite.
        assert len(call_log["badnode"]) == 3, (
            f"expected 3 attempts on badnode "
            f"(1 initial + 2 retries), got {len(call_log['badnode'])}"
        )

    def test_retries_zero_restores_fail_fast(
        self, tmp_path, monkeypatch
    ):
        """Setting RETRIES=0 should disable retries (one attempt only)."""
        call_log = {"flakynode": []}

        def fake_rsync(src, dst, node, **kwargs):
            call_log.setdefault(node, []).append(node)
            if node == "flakynode":
                return (node, 0.1, 255)
            return (node, 0.2, 0)

        monkeypatch.setattr(yeet, "_DEFAULT_RSYNC_RETRIES", 0)
        monkeypatch.setattr(yeet, "_rsync_to_node", fake_rsync)
        monkeypatch.setattr(yeet, "_get_current_hostname", lambda: "node00")
        monkeypatch.setattr(yeet, "_needs_local_copy", lambda src: True)
        monkeypatch.setattr(
            yeet, "_patch_venv_paths_local", lambda dst, src: None
        )

        src = self._make_src(tmp_path)
        dst = tmp_path / "tmp_dst"
        hf = self._make_hostfile(
            tmp_path, ["node00", "flakynode"]
        )

        rc = yeet.run(["--src", str(src), "--dst", str(dst),
                       "--hostfile", str(hf)])

        # With RETRIES=0 and a permanently-failing node, run() must
        # return non-zero — this pins the fail-fast contract that
        # `--retries 0` is meant to restore.
        assert rc != 0, "run() must return non-zero when retries=0 and a node fails"

        # flakynode should be attempted exactly once (no retries)
        assert len(call_log["flakynode"]) == 1


# ===================================================================
# --min-success-nodes / --min-success-fraction (proceed-with-spares)
# ===================================================================


class TestProceedWithSpares:
    """Tests for the --min-success-nodes / --min-success-fraction flags.

    When set, yeet returns rc=0 even with some failures, as long as
    the success count meets the threshold. Writes the failed-nodes
    list to $dst/.ezpz-yeet-failed-nodes.txt for downstream
    consumers (training scripts, ezpz launch wrappers).
    """

    def _make_hostfile(self, tmp_path, nodes):
        hf = tmp_path / "hostfile"
        hf.write_text("\n".join(nodes) + "\n")
        return hf

    def _make_src(self, tmp_path):
        src = tmp_path / "fakenv"
        (src / "bin").mkdir(parents=True)
        (src / "bin" / "activate").write_text("# fake activate\n")
        (src / "pyvenv.cfg").write_text(
            "home = /usr/bin\nversion = 3.12.0\n"
        )
        return src

    def _setup_common(self, monkeypatch, fail_nodes):
        """Stub out the heavy machinery so we can drive run() in unit."""
        def fake_rsync(src, dst, node, **kwargs):
            if node in fail_nodes:
                return (node, 0.1, 255)
            return (node, 0.2, 0)

        # No retries — make the failure permanent so we can test the
        # threshold logic in isolation from PR #160's retry path.
        monkeypatch.setattr(yeet, "_DEFAULT_RSYNC_RETRIES", 0)
        monkeypatch.setattr(yeet, "_rsync_to_node", fake_rsync)
        monkeypatch.setattr(yeet, "_get_current_hostname", lambda: "node00")
        monkeypatch.setattr(yeet, "_needs_local_copy", lambda src: True)
        monkeypatch.setattr(
            yeet, "_patch_venv_paths_local", lambda dst, src: None
        )

    def test_threshold_met_returns_zero_and_writes_sentinel(
        self, tmp_path, monkeypatch
    ):
        """When ok_nodes >= --min-success-nodes, return 0 + write file."""
        self._setup_common(monkeypatch, fail_nodes={"badnode"})
        src = self._make_src(tmp_path)
        dst = tmp_path / "tmp_dst"
        hf = self._make_hostfile(
            tmp_path, ["node00", "node01", "node02", "badnode"]
        )

        # 4 total, 3 succeed, 1 fails. With --min-success-nodes=3,
        # threshold is met → rc=0.
        rc = yeet.run([
            "--src", str(src), "--dst", str(dst),
            "--hostfile", str(hf),
            "--min-success-nodes", "3",
        ])
        assert rc == 0, "threshold met (3/4 ok, need >=3); should return 0"

        # Sentinel file should exist and contain just the failed node.
        sentinel = dst / ".ezpz-yeet-failed-nodes.txt"
        assert sentinel.exists()
        assert sentinel.read_text().strip() == "badnode"

    def test_threshold_missed_returns_one_no_sentinel(
        self, tmp_path, monkeypatch
    ):
        """When ok_nodes < threshold, fail like before (no sentinel)."""
        self._setup_common(monkeypatch, fail_nodes={"bad1", "bad2"})
        src = self._make_src(tmp_path)
        dst = tmp_path / "tmp_dst"
        hf = self._make_hostfile(
            tmp_path, ["node00", "node01", "bad1", "bad2"]
        )

        # 4 total, 2 succeed, 2 fail. With --min-success-nodes=3,
        # threshold is NOT met → rc=1, no sentinel.
        rc = yeet.run([
            "--src", str(src), "--dst", str(dst),
            "--hostfile", str(hf),
            "--min-success-nodes", "3",
        ])
        assert rc != 0, "threshold not met (2/4 ok, need >=3); should fail"
        assert not (dst / ".ezpz-yeet-failed-nodes.txt").exists(), (
            "sentinel must NOT be written when threshold isn't met — "
            "otherwise downstream tooling would silently consume a "
            "stale 'this run partially succeeded' marker."
        )

    def test_no_failures_no_sentinel_even_with_threshold(
        self, tmp_path, monkeypatch
    ):
        """When ALL nodes succeed, no sentinel written, rc=0."""
        self._setup_common(monkeypatch, fail_nodes=set())
        src = self._make_src(tmp_path)
        dst = tmp_path / "tmp_dst"
        hf = self._make_hostfile(
            tmp_path, ["node00", "node01", "node02"]
        )

        rc = yeet.run([
            "--src", str(src), "--dst", str(dst),
            "--hostfile", str(hf),
            "--min-success-nodes", "2",
        ])
        assert rc == 0
        # No failures → no sentinel (avoids stale data confusing
        # downstream readers next run).
        assert not (dst / ".ezpz-yeet-failed-nodes.txt").exists()

    def test_fraction_threshold_met(
        self, tmp_path, monkeypatch
    ):
        """--min-success-fraction 0.75 of 4 nodes = ceil(3) → 3 needed."""
        self._setup_common(monkeypatch, fail_nodes={"badnode"})
        src = self._make_src(tmp_path)
        dst = tmp_path / "tmp_dst"
        hf = self._make_hostfile(
            tmp_path, ["node00", "node01", "node02", "badnode"]
        )

        rc = yeet.run([
            "--src", str(src), "--dst", str(dst),
            "--hostfile", str(hf),
            "--min-success-fraction", "0.75",
        ])
        assert rc == 0, "3/4 = 75% ok, threshold = ceil(0.75*4) = 3, met"
        sentinel = dst / ".ezpz-yeet-failed-nodes.txt"
        assert sentinel.exists()
        assert "badnode" in sentinel.read_text()

    def test_flags_mutex_rejects_both(self):
        """--min-success-nodes and --min-success-fraction are mutex."""
        with pytest.raises(SystemExit):
            yeet.parse_args([
                "--min-success-nodes", "5",
                "--min-success-fraction", "0.9",
            ])

    def test_fraction_out_of_range_rejected(self):
        """--min-success-fraction must be in (0, 1]."""
        for bad in ["0.0", "-0.5", "1.5", "2.0"]:
            with pytest.raises(SystemExit):
                yeet.parse_args(["--min-success-fraction", bad])

    def test_min_success_nodes_zero_rejected(self):
        """--min-success-nodes < 1 is meaningless and rejected."""
        with pytest.raises(SystemExit):
            yeet.parse_args(["--min-success-nodes", "0"])

    def test_default_behavior_unchanged_without_threshold(
        self, tmp_path, monkeypatch
    ):
        """Without --min-success-* flags, any failure → rc=1 + no sentinel."""
        self._setup_common(monkeypatch, fail_nodes={"badnode"})
        src = self._make_src(tmp_path)
        dst = tmp_path / "tmp_dst"
        hf = self._make_hostfile(
            tmp_path, ["node00", "node01", "badnode"]
        )

        rc = yeet.run([
            "--src", str(src), "--dst", str(dst),
            "--hostfile", str(hf),
            # no --min-success-* flag
        ])
        assert rc != 0, "no flag = original fail-on-any behavior"
        assert not (dst / ".ezpz-yeet-failed-nodes.txt").exists()

    # ── Regressions for review findings ─────────────────────────────

    def test_clean_run_under_threshold_fails(
        self, tmp_path, monkeypatch
    ):
        """Regression (codex P2): --min-success-nodes is a HARD lower bound.

        A clean run (zero failures) that synced fewer nodes than the
        threshold MUST fail. Otherwise downstream training silently
        under-provisions: `yeet --min-success-nodes 512` against a
        500-node hostfile with all 500 successful would return 0
        and the user would launch on 500 nodes thinking they had 512.
        """
        self._setup_common(monkeypatch, fail_nodes=set())
        src = self._make_src(tmp_path)
        dst = tmp_path / "tmp_dst"
        # Only 4 nodes available, but caller asks for >=10.
        hf = self._make_hostfile(
            tmp_path, ["node00", "node01", "node02", "node03"]
        )

        rc = yeet.run([
            "--src", str(src), "--dst", str(dst),
            "--hostfile", str(hf),
            "--min-success-nodes", "10",  # > total
        ])
        assert rc != 0, (
            "clean run with too few nodes must fail when threshold "
            "exceeds the available node count — otherwise the flag "
            "isn't a real lower bound"
        )
        # No sentinel — there are no FAILED nodes to record. The
        # situation is "we needed 10 but only had 4." That belongs
        # in the log, not the sentinel file (which is documented as
        # the list of nodes that failed, not the list of nodes the
        # user wished they had).
        assert not (dst / ".ezpz-yeet-failed-nodes.txt").exists()

    def test_stale_sentinel_removed_on_clean_rerun(
        self, tmp_path, monkeypatch
    ):
        """Regression (codex+copilot P2): stale sentinel must not survive.

        Set up: a prior run wrote a sentinel listing badnode. The
        next run is clean (no failures), so the sentinel from the
        prior run must be removed. Otherwise downstream tooling
        would read the stale list and exclude badnode from a launch
        that should be using it.
        """
        self._setup_common(monkeypatch, fail_nodes=set())
        src = self._make_src(tmp_path)
        dst = tmp_path / "tmp_dst"
        dst.mkdir()
        stale_sentinel = dst / ".ezpz-yeet-failed-nodes.txt"
        stale_sentinel.write_text("badnode-from-prior-run\n")
        assert stale_sentinel.exists()  # pre-condition

        hf = self._make_hostfile(
            tmp_path, ["node00", "node01", "node02"]
        )
        rc = yeet.run([
            "--src", str(src), "--dst", str(dst),
            "--hostfile", str(hf),
            "--min-success-nodes", "3",
        ])
        assert rc == 0
        assert not stale_sentinel.exists(), (
            "stale sentinel from prior run must be removed when the "
            "current run has no failures — otherwise downstream "
            "tooling reads wrong data"
        )

    def test_stale_sentinel_removed_when_threshold_missed(
        self, tmp_path, monkeypatch
    ):
        """Regression: stale sentinel removed even on failed run.

        If the threshold isn't met, we return rc=1 (don't write a
        new sentinel) — but we ALSO must remove any prior sentinel,
        so downstream tooling doesn't see a "yeet succeeded with
        these failures" marker from a run that actually failed.
        """
        self._setup_common(monkeypatch, fail_nodes={"bad1", "bad2"})
        src = self._make_src(tmp_path)
        dst = tmp_path / "tmp_dst"
        dst.mkdir()
        stale_sentinel = dst / ".ezpz-yeet-failed-nodes.txt"
        stale_sentinel.write_text("badnode-from-prior-run\n")
        assert stale_sentinel.exists()  # pre-condition

        hf = self._make_hostfile(
            tmp_path, ["node00", "node01", "bad1", "bad2"]
        )
        rc = yeet.run([
            "--src", str(src), "--dst", str(dst),
            "--hostfile", str(hf),
            "--min-success-nodes", "3",  # 2/4 ok, threshold missed
        ])
        assert rc != 0
        assert not stale_sentinel.exists(), (
            "stale sentinel must be cleared even when the new run "
            "fails the threshold — otherwise downstream tooling "
            "consumes a marker from a previous run that no longer "
            "describes reality"
        )

    def test_fraction_uses_hostfile_count_not_rsync_op_count(
        self, tmp_path, monkeypatch
    ):
        """Regression (copilot P2): denominator is len(nodes), not total_nodes.

        When `src` is under /tmp (needs_local_copy = False),
        `total_nodes` = remote_nodes_count (no +1 for the local
        node). But the docs promise the fraction is computed
        against the "full node list including the local node" —
        which is the HOSTFILE count, not the rsync-op count.
        Pin the contract.

        Setup: 5-node hostfile, _needs_local_copy returns False
        (so local rsync skipped, only 4 remote rsyncs run).
        Fraction 0.6 should round-ceil to 3 against hostfile=5
        (NOT to 3 against total_nodes=4 — but in this case the
        threshold value is the same so we have to detect the bug
        via fraction 0.8 → ceil(0.8*5)=4 vs ceil(0.8*4)=4 …).
        Easier: use fraction 0.5 → ceil(0.5*5)=3 vs ceil(0.5*4)=2.
        Make 2 nodes fail; should land below threshold of 3 →
        fail. (If bug were still present: threshold would be 2,
        2 nodes ok would meet it, return 0.)
        """
        # Override _needs_local_copy → False so total_nodes <
        # len(nodes). This is the scenario the bug only manifests in.
        self._setup_common(monkeypatch, fail_nodes={"bad1", "bad2"})
        monkeypatch.setattr(yeet, "_needs_local_copy", lambda src: False)

        src = self._make_src(tmp_path)
        dst = tmp_path / "tmp_dst"
        hf = self._make_hostfile(
            tmp_path,
            ["node00", "node01", "node02", "bad1", "bad2"],  # 5 hosts
        )

        # With the fix: threshold = ceil(0.5 * 5) = 3.
        # 2 succeed (node01, node02 — node00 is local + skipped),
        # 2 fail (bad1, bad2). 2 < 3 → fail.
        #
        # If bug were still present: threshold = ceil(0.5 * 4) = 2.
        # 2 >= 2 → return 0. The assertion below would fail.
        rc = yeet.run([
            "--src", str(src), "--dst", str(dst),
            "--hostfile", str(hf),
            "--min-success-fraction", "0.5",
        ])
        assert rc != 0, (
            "fraction must be computed against the hostfile node "
            "count (5), not the rsync-op count (4). With 0.5 of 5 "
            "→ ceil = 3 needed; only 2 succeeded; should fail."
        )


# ===================================================================
# parse_args
# ===================================================================


class TestParseArgs:
    """Tests for ``parse_args``."""

    def test_defaults(self):
        """All args are optional with sensible defaults."""
        args = yeet.parse_args([])
        assert args.src is None
        assert args.dst is None
        assert args.hostfile is None
        assert args.dry_run is False

    def test_tarball_hint_fires_when_present(self, tmp_path, caplog):
        """A same-named tarball next to the env triggers the hint."""
        env = tmp_path / "myenv"
        env.mkdir()
        (env / "pyvenv.cfg").write_text("home = /usr/bin\n")
        tarball = tmp_path / "myenv.tar.gz"
        tarball.write_bytes(b"x" * 100)
        # Make tarball newer than pyvenv.cfg (otherwise it's stale).
        import os
        os.utime(tarball, (1700000000, 1700000000))
        os.utime(env / "pyvenv.cfg", (1600000000, 1600000000))

        with caplog.at_level("WARNING", logger=yeet.logger.name):
            yeet._suggest_tarball_if_present(env)
        records = [r.getMessage() for r in caplog.records]
        joined = " ".join(records)
        assert "Tip: found" in joined
        assert "myenv.tar.gz" in joined
        assert "ezpz yeet" in joined

    def test_tarball_hint_silent_when_stale(self, tmp_path, caplog):
        """Don't suggest a tarball older than the venv's pyvenv.cfg."""
        env = tmp_path / "myenv"
        env.mkdir()
        cfg = env / "pyvenv.cfg"
        cfg.write_text("home = /usr/bin\n")
        tarball = tmp_path / "myenv.tar.gz"
        tarball.write_bytes(b"x" * 100)
        import os
        # Tarball is older than cfg.
        os.utime(tarball, (1600000000, 1600000000))
        os.utime(cfg, (1700000000, 1700000000))

        with caplog.at_level("WARNING", logger=yeet.logger.name):
            yeet._suggest_tarball_if_present(env)
        assert not any("Tip:" in r.getMessage() for r in caplog.records)

    def test_tarball_hint_silent_when_absent(self, tmp_path, caplog):
        """No tarball nearby, no hint."""
        env = tmp_path / "myenv"
        env.mkdir()
        with caplog.at_level("WARNING", logger=yeet.logger.name):
            yeet._suggest_tarball_if_present(env)
        assert not any("Tip:" in r.getMessage() for r in caplog.records)

    def test_tarball_hint_silent_when_src_is_file(self, tmp_path, caplog):
        """If src is itself a file (not a directory), don't search."""
        f = tmp_path / "something.tar.gz"
        f.write_bytes(b"x")
        with caplog.at_level("WARNING", logger=yeet.logger.name):
            yeet._suggest_tarball_if_present(f)
        assert not any("Tip:" in r.getMessage() for r in caplog.records)

    def test_via_click_no_args_does_not_grab_argv(self):
        """`ezpz yeet` (no args) used to leak `sys.argv[1:]` into argparse.

        Regression: when cli/__init__.py called
        ``yeet_env.run(list(args) if args else None)`` with empty
        ``args``, ``run()`` saw ``argv=None`` and argparse fell back to
        ``sys.argv[1:]`` — which contained ``["yeet"]`` from the
        invoking process, picked up as a positional SRC = "yeet".
        """
        # Simulate the harness invocation when no positional/flag args are present.
        args = yeet.parse_args([])
        # src must be None — NOT "yeet" or any sys.argv leakage.
        assert args.src is None
        assert args.src_positional is None

    def test_all_flags(self):
        """Parses all flags explicitly."""
        args = yeet.parse_args([
            "--src", "/path/to/env",
            "--dst", "/tmp/myenv",
            "--hostfile", "/path/to/hostfile",
            "--dry-run",
        ])
        assert args.src == "/path/to/env"
        assert args.dst == "/tmp/myenv"
        assert args.hostfile == "/path/to/hostfile"
        assert args.dry_run is True

    def test_copy_flag(self):
        """--copy flag is parsed."""
        args = yeet.parse_args(["--copy"])
        assert args.copy is True
        assert args.compress is False

    def test_compress_flag(self):
        """--compress flag is parsed."""
        args = yeet.parse_args(["--compress"])
        assert args.compress is True
        assert args.copy is False

    def test_positional_src(self):
        """Positional SRC populates args.src."""
        args = yeet.parse_args(["/path/to/env"])
        assert args.src == "/path/to/env"

    def test_positional_with_flags(self):
        """Positional SRC works alongside other flags."""
        args = yeet.parse_args(["/path/to/env", "--dry-run", "--copy"])
        assert args.src == "/path/to/env"
        assert args.dry_run is True
        assert args.copy is True

    def test_positional_and_src_flag_conflict(self):
        """Passing both positional SRC and --src exits with usage error."""
        with pytest.raises(SystemExit):
            yeet.parse_args(["/path/a", "--src", "/path/b"])


class TestTarballSrc:
    """Tests for the .tar.gz / .tgz source-detection logic in run()."""

    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node01")
    @patch("ezpz.utils.yeet_env._get_worker_nodes", return_value=["node01"])
    def test_tarball_strips_suffix_for_dst(self, _nodes, _host, tmp_path, caplog):
        """A .tar.gz src derives env_name by stripping the suffix."""
        tarball = tmp_path / "myenv.tar.gz"
        tarball.write_bytes(b"fake")
        with caplog.at_level("INFO", logger=yeet.logger.name):
            rc = yeet.run(["--src", str(tarball), "--dry-run"])
        assert rc == 0
        # Default dst should be /tmp/myenv/ (suffix stripped)
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "/tmp/myenv/" in joined

    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node01")
    @patch("ezpz.utils.yeet_env._get_worker_nodes", return_value=["node01"])
    def test_tgz_also_recognized(self, _nodes, _host, tmp_path, caplog):
        """A .tgz src is also recognized."""
        tarball = tmp_path / "myenv.tgz"
        tarball.write_bytes(b"fake")
        with caplog.at_level("INFO", logger=yeet.logger.name):
            rc = yeet.run(["--src", str(tarball), "--dry-run"])
        assert rc == 0
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "/tmp/myenv/" in joined


# ===================================================================
# run (integration)
# ===================================================================


class TestRun:
    """Tests for the ``run`` entry point."""

    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node01")
    @patch("ezpz.utils.yeet_env._get_worker_nodes", return_value=["node01"])
    def test_single_node_dry_run(self, _nodes, _host, caplog):
        """Single node dry-run shows local copy plan."""
        with caplog.at_level("INFO", logger=yeet.logger.name):
            rc = yeet.run(["--dry-run"])
        assert rc == 0
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "dry-run" in joined
        assert "local:" in joined

    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node01")
    @patch("ezpz.utils.yeet_env._get_worker_nodes", return_value=["node01", "node02", "node03"])
    def test_dry_run_shows_nodes(self, _nodes, _host, caplog):
        """Dry run prints source, target, and node list."""
        with caplog.at_level("INFO", logger=yeet.logger.name):
            rc = yeet.run(["--dry-run"])
        assert rc == 0
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "Source:" in joined
        assert "Target:" in joined
        assert "node02" in joined
        assert "node03" in joined
        assert "dry-run" in joined

    def test_nonexistent_src(self):
        """Returns 1 when --src path doesn't exist."""
        rc = yeet.run(["--src", "/nonexistent/path/to/env"])
        assert rc == 1

    @patch("ezpz.utils.yeet_env._rsync_to_node")
    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node01")
    @patch("ezpz.utils.yeet_env._get_worker_nodes", return_value=["node01", "node02"])
    def test_syncs_local_and_remote(self, _nodes, _host, mock_rsync, capsys):
        """Syncs local and remote nodes in parallel."""
        mock_rsync.return_value = ("node01", 3.0, 0)
        rc = yeet.run([])
        assert rc == 0
        # Should be called for both nodes
        assert mock_rsync.call_count == 2
        call_nodes = [c.args[2] for c in mock_rsync.call_args_list]
        assert "node01" in call_nodes
        assert "node02" in call_nodes

    @patch("ezpz.utils.yeet_env._patch_venv_paths_local")
    @patch("ezpz.utils.yeet_env._rsync_to_node")
    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node01")
    @patch("ezpz.utils.yeet_env._get_worker_nodes", return_value=["node01"])
    def test_generic_source_skips_venv_footer(
        self, _nodes, _host, mock_rsync, mock_patch_venv, tmp_path, capsys,
    ):
        """A non-venv directory source uses the generic 'Synced to ...' footer."""
        # Make a plain dir source (no bin/activate, no conda-meta)
        src = tmp_path / "data"
        src.mkdir()
        (src / "file.bin").write_bytes(b"x")
        # Use a fresh dst that definitely doesn't have venv markers
        dst = tmp_path / "out"
        mock_rsync.return_value = ("node01", 0.1, 0)

        rc = yeet.run(["--src", str(src), "--dst", str(dst)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Synced to" in out
        assert "To use this environment" not in out
        # The venv-patching step should NOT run for a non-venv source.
        mock_patch_venv.assert_not_called()

    @patch("ezpz.utils.yeet_env._rsync_to_node")
    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node01")
    @patch("ezpz.utils.yeet_env._get_worker_nodes", return_value=["node01"])
    def test_venv_source_uses_venv_footer(
        self, _nodes, _host, mock_rsync, tmp_path, capsys,
    ):
        """A directory with bin/activate triggers the venv footer."""
        src = tmp_path / "myenv"
        (src / "bin").mkdir(parents=True)
        (src / "bin" / "activate").write_text("# fake venv")
        dst = tmp_path / "out"
        (dst / "bin").mkdir(parents=True)
        (dst / "bin" / "activate").write_text("# fake venv")
        mock_rsync.return_value = ("node01", 0.1, 0)

        rc = yeet.run(["--src", str(src), "--dst", str(dst)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "To use this environment" in out
        assert f"source {dst}/bin/activate" in out


# ===================================================================
# Per-host HSN suffix detection
# ===================================================================


class TestHsnSuffix:
    """Tests for ``_maybe_apply_hsn_suffix`` (per-node HSN detection)."""

    def test_empty_input(self):
        assert yeet._maybe_apply_hsn_suffix([]) == []

    def test_already_suffixed_left_alone(self):
        nodes = ["node01-hsn0", "node02-hsn0"]
        assert yeet._maybe_apply_hsn_suffix(nodes) == nodes

    @patch("ezpz.utils.yeet_env.socket.gethostbyname")
    def test_all_resolve(self, mock_resolve):
        mock_resolve.return_value = "10.0.0.1"
        result = yeet._maybe_apply_hsn_suffix(["node01", "node02"])
        assert result == ["node01-hsn0", "node02-hsn0"]

    @patch("ezpz.utils.yeet_env.socket.gethostbyname")
    def test_none_resolve(self, mock_resolve):
        import socket as _socket
        mock_resolve.side_effect = _socket.gaierror
        result = yeet._maybe_apply_hsn_suffix(["node01", "node02"])
        # Bare names are kept when HSN is unavailable
        assert result == ["node01", "node02"]

    @patch("ezpz.utils.yeet_env.socket.gethostbyname")
    def test_heterogeneous_allocation(self, mock_resolve):
        """The bug the original code had: only first node was probed."""
        import socket as _socket

        def _fake_resolve(name: str) -> str:
            # node01-hsn0 resolves; node02-hsn0 does not.
            if name == "node01-hsn0":
                return "10.0.0.1"
            raise _socket.gaierror

        mock_resolve.side_effect = _fake_resolve
        result = yeet._maybe_apply_hsn_suffix(["node01", "node02"])
        # Per-node decisions: HSN where it resolves, bare name where not.
        assert result == ["node01-hsn0", "node02"]


# ===================================================================
# Greedy fan-out source selection
# ===================================================================


class TestPickSource:
    """Tests for the module-level ``pick_source`` function.

    Exercises the actual algorithm used by ``run()`` instead of
    reconstructing it, so a future refactor of the production code
    cannot drift away from these expectations.
    """

    def test_least_loaded_wins(self):
        active = {"a": 5, "b": 1, "c": 3}
        # 'b' is uniquely least-loaded — no tie-break needed
        assert yeet.pick_source(active, max_per_source=8) == "b"

    def test_capped_sources_excluded(self):
        active = {"a": 8, "b": 8, "c": 2}  # a, b at cap
        assert yeet.pick_source(active, max_per_source=8) == "c"

    def test_all_at_cap_returns_none(self):
        active = {"a": 8, "b": 8}
        assert yeet.pick_source(active, max_per_source=8) is None

    def test_empty_returns_none(self):
        assert yeet.pick_source({}, max_per_source=8) is None

    def test_random_tie_break_visits_all_candidates(self):
        """All zero-loaded candidates must eventually be picked.

        With the old deterministic tie-break, 'a' was picked every
        iteration until its cap was hit, defeating the whole point of
        the tree distribution.
        """
        import random as _random
        active = {"a": 0, "b": 0, "c": 0}
        rng = _random.Random(42)  # seeded for determinism
        seen = {
            yeet.pick_source(active, max_per_source=8, rng=rng)
            for _ in range(100)
        }
        assert seen == {"a", "b", "c"}

    def test_does_not_disturb_global_random_state(self):
        """A fresh module-internal Random() must not consume the
        process-global random sequence the caller may have seeded."""
        import random as _random
        _random.seed(1234)
        before = _random.random()

        _random.seed(1234)
        yeet.pick_source({"a": 0, "b": 0}, max_per_source=8)
        after = _random.random()

        # The pick_source call should not have advanced the global RNG.
        assert before == after

    def test_uses_supplied_rng_for_determinism(self):
        import random as _random
        active = {"a": 0, "b": 0, "c": 0}
        rng1 = _random.Random(42)
        rng2 = _random.Random(42)
        picks_1 = [
            yeet.pick_source(active, max_per_source=8, rng=rng1)
            for _ in range(20)
        ]
        picks_2 = [
            yeet.pick_source(active, max_per_source=8, rng=rng2)
            for _ in range(20)
        ]
        assert picks_1 == picks_2


class TestSlurmHostfileCleanupRegistration:
    """Tests that the SLURM hostfile atexit registration de-dupes."""

    def test_registers_once_per_path(self, tmp_path, monkeypatch):
        """Repeated _get_worker_nodes calls in a SLURM env should
        register the cleanup handler exactly once per unique path,
        not once per call.
        """
        import os as _os
        import uuid
        from unittest.mock import MagicMock

        # Use a per-process-unique SLURM_JOB_ID so this test doesn't
        # collide with itself under pytest-xdist (which runs tests in
        # parallel worker processes that share /tmp).  Also include
        # the worker pid for extra safety.
        unique_id = f"yeet-test-{_os.getpid()}-{uuid.uuid4().hex[:8]}"

        # Pretend we're in SLURM with no scheduler-discoverable hostfile.
        monkeypatch.delenv("PBS_NODEFILE", raising=False)
        monkeypatch.delenv("HOSTFILE", raising=False)
        monkeypatch.delenv("PBS_JOBID", raising=False)
        monkeypatch.setenv("SLURM_NODELIST", "node[01-02]")
        monkeypatch.setenv("SLURM_JOB_ID", unique_id)

        # Mock scontrol to produce two hostnames
        result = MagicMock(returncode=0, stdout="node01\nnode02\n")
        monkeypatch.setattr(
            yeet.subprocess, "run", lambda *a, **kw: result,
        )
        # Stub atexit.register to count calls
        registered: list[object] = []
        monkeypatch.setattr(
            yeet.atexit, "register",
            lambda fn, *args: registered.append((fn, args)),
        )
        # Stub HSN suffix probe so we don't hit DNS in the test
        monkeypatch.setattr(
            yeet, "_maybe_apply_hsn_suffix", lambda nodes: nodes,
        )

        # The path the production code will try to write
        hf_path = Path(f"/tmp/_ezpz_hostfile_{unique_id}")

        try:
            yeet._get_worker_nodes()
            yeet._get_worker_nodes()
            yeet._get_worker_nodes()
        finally:
            # Best-effort cleanup: unique path means no other test or
            # process could have created this file, so removing it is
            # safe even on shared /tmp.
            if hf_path.exists():
                hf_path.unlink()
            yeet._REGISTERED_CLEANUPS.discard(str(hf_path))

        # Three calls but only one registration
        assert len(registered) == 1


class TestRemovePartialDst:
    """Tests for ``_remove_partial_dst`` — works regardless of /tmp."""

    def test_removes_directory(self, tmp_path):
        dst = tmp_path / "venv-half-written"
        dst.mkdir()
        (dst / "file.txt").write_text("partial")
        yeet._remove_partial_dst(dst)
        assert not dst.exists()

    def test_silent_on_missing(self, tmp_path):
        # Should not raise
        yeet._remove_partial_dst(tmp_path / "does-not-exist")

    def test_works_outside_tmp(self, tmp_path):
        """Unlike _safe_rmtree, this helper accepts non-/tmp paths.

        The failure-cleanup paths only invoke it on dst directories
        the command itself just wrote, so the /tmp safety guard is
        unnecessary and harmful (it left half-written custom dsts
        behind in the prior implementation).
        """
        # tmp_path is typically /var/folders/... on macOS — definitely
        # not under /tmp/ — so this proves the no-prefix-restriction
        # behavior.
        dst = tmp_path / "custom-dst"
        dst.mkdir()
        (dst / "data").write_bytes(b"x")
        yeet._remove_partial_dst(dst)
        assert not dst.exists()


# ===================================================================
# Stderr drain helper
# ===================================================================


class TestDrainStream:
    """Tests for ``_drain_stream_to_list``."""

    def test_collects_lines(self):
        sink: list[str] = []
        yeet._drain_stream_to_list(iter(["a\n", "b\n", "c\n"]), sink)
        assert sink == ["a\n", "b\n", "c\n"]

    def test_swallows_exceptions(self):
        """A read error should not propagate to the caller."""

        def _raise():
            raise IOError("boom")
            yield  # pragma: no cover

        sink: list[str] = []
        # Should not raise
        yeet._drain_stream_to_list(_raise(), sink)
        assert sink == []


# ===================================================================
# Rsync timeout
# ===================================================================


class TestRsyncTimeout:
    """Tests for the timeout enforcement in ``_rsync_to_node``."""

    @patch("ezpz.utils.yeet_env.subprocess.run")
    def test_timeout_returns_124(self, mock_run, tmp_path):
        """A timed-out rsync returns the bash 'timeout' exit code."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="rsync", timeout=1)
        src = tmp_path / "env"
        src.mkdir()
        node, _, rc = yeet._rsync_to_node(
            src, Path("/tmp/env"), "node01", timeout=1,
        )
        assert node == "node01"
        assert rc == 124

    @patch("ezpz.utils.yeet_env.subprocess.run")
    def test_timeout_passed_through(self, mock_run, tmp_path):
        """The timeout argument is forwarded to subprocess.run."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        src = tmp_path / "env"
        src.mkdir()
        yeet._rsync_to_node(
            src, Path("/tmp/env"), "node01", timeout=42,
        )
        kwargs = mock_run.call_args.kwargs
        assert kwargs["timeout"] == 42


# ===================================================================
# Cleanup helper
# ===================================================================


class TestCleanupPath:
    """Tests for ``_cleanup_path``."""

    def test_removes_existing(self, tmp_path):
        f = tmp_path / "tarball.tgz"
        f.write_bytes(b"x")
        yeet._cleanup_path(f)
        assert not f.exists()

    def test_silent_on_missing(self, tmp_path):
        # Should not raise
        yeet._cleanup_path(tmp_path / "does-not-exist")
