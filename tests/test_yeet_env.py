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
