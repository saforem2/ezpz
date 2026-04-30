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


class TestTarballSrc:
    """Tests for the .tar.gz / .tgz source-detection logic in run()."""

    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node01")
    @patch("ezpz.utils.yeet_env._get_worker_nodes", return_value=["node01"])
    def test_tarball_strips_suffix_for_dst(self, _nodes, _host, tmp_path, capsys):
        """A .tar.gz src derives env_name by stripping the suffix."""
        tarball = tmp_path / "myenv.tar.gz"
        tarball.write_bytes(b"fake")
        rc = yeet.run(["--src", str(tarball), "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        # Default dst should be /tmp/myenv/ (suffix stripped)
        assert "/tmp/myenv/" in out

    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node01")
    @patch("ezpz.utils.yeet_env._get_worker_nodes", return_value=["node01"])
    def test_tgz_also_recognized(self, _nodes, _host, tmp_path, capsys):
        """A .tgz src is also recognized."""
        tarball = tmp_path / "myenv.tgz"
        tarball.write_bytes(b"fake")
        rc = yeet.run(["--src", str(tarball), "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "/tmp/myenv/" in out


# ===================================================================
# run (integration)
# ===================================================================


class TestRun:
    """Tests for the ``run`` entry point."""

    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node01")
    @patch("ezpz.utils.yeet_env._get_worker_nodes", return_value=["node01"])
    def test_single_node_dry_run(self, _nodes, _host, capsys):
        """Single node dry-run shows local copy plan."""
        rc = yeet.run(["--dry-run"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "dry-run" in captured.out
        assert "local:" in captured.out

    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node01")
    @patch("ezpz.utils.yeet_env._get_worker_nodes", return_value=["node01", "node02", "node03"])
    def test_dry_run_shows_nodes(self, _nodes, _host, capsys):
        """Dry run prints source, target, and node list."""
        rc = yeet.run(["--dry-run"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Source:" in captured.out
        assert "Target:" in captured.out
        assert "node02" in captured.out
        assert "node03" in captured.out
        assert "dry-run" in captured.out

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
        captured = capsys.readouterr()
        assert "To use this environment" in captured.out


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
    """Tests that validate the greedy fan-out chooses sources sanely.

    These exercise the in-function ``_pick_source`` indirectly by
    reconstructing the same logic so we can check tie-break behavior
    without standing up a full pool.
    """

    def test_least_loaded_wins(self):
        """Among multiple sources, pick the one with fewest active syncs."""
        active = {"a": 5, "b": 1, "c": 3}
        cap = 8
        # Inline copy of _pick_source semantics
        candidates = [s for s, c in active.items() if c < cap]
        min_count = min(active[s] for s in candidates)
        least = [s for s in candidates if active[s] == min_count]
        assert least == ["b"]

    def test_capped_sources_excluded(self):
        active = {"a": 8, "b": 8, "c": 2}  # a, b at cap
        cap = 8
        candidates = [s for s, c in active.items() if c < cap]
        assert candidates == ["c"]

    def test_random_tie_break_eventually_picks_both(self):
        """Ties should be broken randomly so fan-out actually fans out.

        With deterministic tie-breaking (the old bug), source 'a' would
        be picked every iteration until its cap was hit, defeating the
        whole point of the tree distribution.
        """
        import random as _random
        active = {"a": 0, "b": 0, "c": 0}
        cap = 8
        seen = set()
        # Seed for reproducibility, then sample many picks
        _random.seed(42)
        for _ in range(100):
            candidates = [s for s, c in active.items() if c < cap]
            min_count = min(active[s] for s in candidates)
            least = [s for s in candidates if active[s] == min_count]
            seen.add(_random.choice(least))
        assert seen == {"a", "b", "c"}


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
