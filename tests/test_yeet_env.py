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
        # Check rsync command
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "rsync"
        assert "-a" in call_args
        assert "--delete" in call_args
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


# ===================================================================
# run (integration)
# ===================================================================


class TestRun:
    """Tests for the ``run`` entry point."""

    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node01")
    @patch("ezpz.utils.yeet_env._get_worker_nodes", return_value=["node01"])
    def test_single_node_no_sync(self, _nodes, _host, capsys):
        """When only the current node exists, nothing to sync."""
        rc = yeet.run(["--dry-run"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Nothing to sync" in captured.out

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

    @patch("ezpz.utils.yeet_env._rsync_parallel")
    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node01")
    @patch("ezpz.utils.yeet_env._get_worker_nodes", return_value=["node01", "node02"])
    def test_syncs_to_remote_nodes(self, _nodes, _host, mock_rsync, capsys):
        """Syncs only to remote nodes (not self)."""
        mock_rsync.return_value = [("node02", 5.0, 0)]
        rc = yeet.run([])
        assert rc == 0
        # Should only sync to node02, not node01 (self)
        call_args = mock_rsync.call_args
        nodes_arg = call_args[0][2]  # third positional arg
        assert "node02" in nodes_arg
        assert "node01" not in nodes_arg
        captured = capsys.readouterr()
        assert "To use this environment" in captured.out
