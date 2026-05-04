"""Tests for ``ezpz.utils.kill``."""

from __future__ import annotations

import os
import signal
from unittest.mock import MagicMock, patch

import pytest

import ezpz.utils.kill as kill_mod


# ===================================================================
# parse_args
# ===================================================================


class TestParseArgs:
    def test_no_args(self):
        args = kill_mod.parse_args([])
        assert args.pattern is None
        assert args.all_nodes is False
        assert args.signal == "TERM"
        assert args.dry_run is False

    def test_positional_pattern(self):
        args = kill_mod.parse_args(["train.py"])
        assert args.pattern == "train.py"

    def test_all_flags(self):
        args = kill_mod.parse_args([
            "myproc",
            "--all-nodes",
            "--hostfile", "/tmp/hosts",
            "--signal", "KILL",
            "--dry-run",
        ])
        assert args.pattern == "myproc"
        assert args.all_nodes is True
        assert args.hostfile == "/tmp/hosts"
        assert args.signal == "KILL"
        assert args.dry_run is True

    def test_invalid_signal_rejected(self):
        with pytest.raises(SystemExit):
            kill_mod.parse_args(["--signal", "WTF"])


# ===================================================================
# match logic — Linux /proc path
# ===================================================================


class TestFindMatchesLinux:
    """Tests for the Linux /proc-based match path with mocked filesystem."""

    @patch("os.listdir")
    def test_skips_self_and_parent(self, mock_listdir):
        """Self-pid and parent-pid are never returned as matches."""
        self_pid = os.getpid()
        parent_pid = os.getppid()
        mock_listdir.return_value = [str(self_pid), str(parent_pid), "1234"]

        with patch("ezpz.utils.kill._read_proc_file") as mock_read:
            # PID 1234 should be the only one inspected
            mock_read.side_effect = lambda pid, name: (
                b"python\0-c\0print()\0" if name == "cmdline" else b"FOO=bar\0EZPZ_RUN_COMMAND=x\0"
            )
            matches = kill_mod._find_matches_linux(None)
            pids = [m[0] for m in matches]
            assert self_pid not in pids
            assert parent_pid not in pids
            assert 1234 in pids

    @patch("os.listdir")
    def test_no_pattern_requires_ezpz_marker(self, mock_listdir):
        """Default match requires EZPZ_RUN_COMMAND in environ."""
        mock_listdir.return_value = ["1111", "2222"]

        def fake_read(pid: int, name: str):
            if name == "cmdline":
                return b"python\0script.py\0"
            # 1111 has the marker, 2222 does not
            if pid == 1111:
                return b"PATH=/bin\0EZPZ_RUN_COMMAND=foo\0"
            return b"PATH=/bin\0HOME=/root\0"

        with patch("ezpz.utils.kill._read_proc_file", side_effect=fake_read):
            matches = kill_mod._find_matches_linux(None)
            pids = [m[0] for m in matches]
            assert 1111 in pids
            assert 2222 not in pids

    @patch("os.listdir")
    def test_pattern_matches_cmdline_substring(self, mock_listdir):
        """With STR, match any process whose cmdline contains STR."""
        mock_listdir.return_value = ["1111", "2222"]

        def fake_read(pid: int, name: str):
            if name == "cmdline":
                if pid == 1111:
                    # NUL-separated argv: python -m my.train --epochs 10
                    return b"\0".join([
                        b"python", b"-m", b"my.train",
                        b"--epochs", b"10", b"",
                    ])
                return b"\0".join([b"vim", b"/etc/hosts", b""])
            return b""

        with patch("ezpz.utils.kill._read_proc_file", side_effect=fake_read):
            matches = kill_mod._find_matches_linux("train")
            pids = [m[0] for m in matches]
            assert 1111 in pids
            assert 2222 not in pids

    @patch("os.listdir")
    def test_skips_non_numeric_proc_entries(self, mock_listdir):
        """/proc has many non-PID entries (cpuinfo, meminfo, etc.) — skip them."""
        mock_listdir.return_value = ["1234", "cpuinfo", "self", "stat"]
        with patch("ezpz.utils.kill._read_proc_file", return_value=None):
            matches = kill_mod._find_matches_linux(None)
            assert matches == []  # no readable entries; non-numeric never tried

    @patch("os.listdir")
    def test_skips_unreadable_pids(self, mock_listdir):
        """PIDs whose /proc files raise PermissionError are silently skipped."""
        mock_listdir.return_value = ["1234"]
        with patch("ezpz.utils.kill._read_proc_file", return_value=None):
            matches = kill_mod._find_matches_linux(None)
            assert matches == []


# ===================================================================
# kill_local
# ===================================================================


class TestKillLocal:
    @patch("ezpz.utils.kill.find_matches", return_value=[])
    def test_no_matches_returns_zero_zero(self, _):
        killed, total = kill_mod.kill_local(None, signal.SIGTERM)
        assert killed == 0
        assert total == 0

    @patch("ezpz.utils.kill._kill_pid", return_value=True)
    @patch("ezpz.utils.kill.find_matches", return_value=[(1234, "python script.py")])
    def test_kill_calls_kill_pid(self, _matches, mock_kill, capsys):
        killed, total = kill_mod.kill_local(None, signal.SIGTERM)
        assert killed == 1
        assert total == 1
        mock_kill.assert_called_once_with(1234, signal.SIGTERM)
        out = capsys.readouterr().out
        assert "killed 1234" in out
        assert "(TERM)" in out

    @patch("ezpz.utils.kill._kill_pid")
    @patch("ezpz.utils.kill.find_matches", return_value=[(1234, "x"), (5678, "y")])
    def test_dry_run_never_calls_kill_pid(self, _matches, mock_kill, capsys):
        """--dry-run must not invoke _kill_pid for any match."""
        killed, total = kill_mod.kill_local(None, signal.SIGTERM, dry_run=True)
        assert killed == 2
        assert total == 2
        mock_kill.assert_not_called()
        out = capsys.readouterr().out
        assert "would kill 1234" in out
        assert "would kill 5678" in out

    @patch("ezpz.utils.kill._kill_pid", return_value=False)
    @patch("ezpz.utils.kill.find_matches", return_value=[(1234, "x")])
    def test_failure_returns_partial_count(self, _matches, _kill, capsys):
        killed, total = kill_mod.kill_local(None, signal.SIGTERM)
        assert killed == 0
        assert total == 1
        out = capsys.readouterr().out
        assert "failed to kill 1234" in out


# ===================================================================
# _ssh_kill — verify command construction
# ===================================================================


class TestSshKill:
    @patch("subprocess.run")
    def test_default_pattern_omitted(self, mock_run):
        """Without a pattern, the remote `ezpz kill` runs with no positional."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        kill_mod._ssh_kill("node01", None, "TERM", dry_run=False)
        cmd = mock_run.call_args[0][0]
        # ssh ... node01 "ezpz kill --signal TERM"
        assert cmd[0] == "ssh"
        assert "node01" in cmd
        # Last positional is the joined remote command
        remote_cmd = cmd[-1]
        assert "ezpz kill" in remote_cmd
        assert "--signal TERM" in remote_cmd
        assert "--dry-run" not in remote_cmd

    @patch("subprocess.run")
    def test_pattern_appended_after_flags(self, mock_run):
        """Positional pattern is appended after --signal/--dry-run."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        kill_mod._ssh_kill("node01", "train.py", "KILL", dry_run=True)
        remote_cmd = mock_run.call_args[0][0][-1]
        assert "ezpz kill" in remote_cmd
        assert "--signal KILL" in remote_cmd
        assert "--dry-run" in remote_cmd
        assert "train.py" in remote_cmd
        # --all-nodes must NOT appear (avoid recursive fan-out)
        assert "--all-nodes" not in remote_cmd

    @patch("subprocess.run", side_effect=__import__("subprocess").TimeoutExpired(cmd=[], timeout=60))
    def test_timeout_returns_124(self, _):
        node, rc, stderr = kill_mod._ssh_kill("node01", None, "TERM", False)
        assert node == "node01"
        assert rc == 124
        assert "timed out" in stderr


# ===================================================================
# run() — top-level CLI integration
# ===================================================================


class TestRun:
    @patch("ezpz.utils.kill.kill_local", return_value=(0, 0))
    def test_returns_zero_on_no_matches(self, _, capsys):
        rc = kill_mod.run([])
        assert rc == 0

    @patch("ezpz.utils.kill.kill_local", return_value=(2, 2))
    def test_returns_zero_on_full_success(self, _, capsys):
        rc = kill_mod.run(["myproc"])
        assert rc == 0

    @patch("ezpz.utils.kill.kill_local", return_value=(1, 2))
    def test_returns_one_on_partial_failure(self, _, capsys):
        rc = kill_mod.run(["myproc"])
        assert rc == 1

    @patch("ezpz.utils.kill.kill_remote", return_value=(2, 2))
    def test_all_nodes_dispatches_to_kill_remote(self, mock_remote, capsys):
        rc = kill_mod.run(["--all-nodes", "myproc"])
        assert rc == 0
        mock_remote.assert_called_once()
        kwargs = mock_remote.call_args
        # First positional arg = pattern; second = signal
        assert kwargs.args[0] == "myproc"
        assert kwargs.args[1] == signal.SIGTERM


# ===================================================================
# match logic — macOS ps fallback
# ===================================================================


class TestFindMatchesMacos:
    """Tests for the macOS ps-based match path."""

    def test_no_pattern_prints_unsupported_message(self, capsys):
        """Without a pattern, _find_matches_macos returns [] and warns to stderr."""
        matches = kill_mod._find_matches_macos(None)
        assert matches == []
        err = capsys.readouterr().err
        assert "macOS" in err
        # mention the workaround so users know what to do
        assert "ezpz kill" in err

    @patch("ezpz.utils.kill.subprocess.run")
    def test_pattern_filters_ps_output(self, mock_run):
        """With a pattern, only ps lines whose command contains it match."""
        from subprocess import CompletedProcess
        mock_run.return_value = CompletedProcess(
            args=[], returncode=0,
            stdout=(
                "1234 /usr/bin/python -m my.train --epochs 10\n"
                "5678 vim /etc/hosts\n"
                "9999 /usr/bin/python other.py\n"
            ),
            stderr="",
        )
        matches = kill_mod._find_matches_macos("my.train")
        pids = [m[0] for m in matches]
        assert pids == [1234]

    @patch("ezpz.utils.kill.subprocess.run")
    def test_skips_self_and_parent(self, mock_run):
        """Self-pid and parent-pid are filtered out even if they match."""
        from subprocess import CompletedProcess
        self_pid = os.getpid()
        parent_pid = os.getppid()
        mock_run.return_value = CompletedProcess(
            args=[], returncode=0,
            stdout=(
                f"{self_pid} python my.train\n"
                f"{parent_pid} python my.train\n"
                "12345 python my.train\n"
            ),
            stderr="",
        )
        matches = kill_mod._find_matches_macos("my.train")
        pids = [m[0] for m in matches]
        assert self_pid not in pids
        assert parent_pid not in pids
        assert 12345 in pids


# ===================================================================
# kill_remote — multi-node fan-out
# ===================================================================


class TestKillRemote:
    """Tests for kill_remote (hostfile discovery + SSH fan-out)."""

    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node00")
    @patch("ezpz.utils.yeet_env._get_worker_nodes", return_value=[])
    def test_no_nodes_returns_zero_zero(self, _nodes, _host, capsys):
        succeeded, total = kill_mod.kill_remote(None, signal.SIGTERM)
        assert (succeeded, total) == (0, 0)
        assert "no worker nodes" in capsys.readouterr().out

    @patch("ezpz.utils.kill._ssh_kill")
    @patch("ezpz.utils.kill.kill_local", return_value=(0, 0))
    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node00")
    @patch(
        "ezpz.utils.yeet_env._get_worker_nodes",
        return_value=["node00", "node01", "node02"],
    )
    def test_local_only_then_ssh_to_others(
        self, _nodes, _host, mock_local, mock_ssh, capsys,
    ):
        """Local node runs in-process; remote nodes go via SSH."""
        mock_ssh.return_value = ("nodeX", 0, "")
        succeeded, total = kill_mod.kill_remote("foo", signal.SIGTERM)
        # 3 nodes total: 1 local + 2 remote
        assert total == 3
        # kill_local called once for the local node
        mock_local.assert_called_once()
        # _ssh_kill called once per remote node (not for the local one)
        ssh_targets = [c.args[0] for c in mock_ssh.call_args_list]
        assert sorted(ssh_targets) == ["node01", "node02"]
        # all nodes succeeded → succeeded == total
        assert succeeded == total

    @patch("ezpz.utils.kill._ssh_kill")
    @patch("ezpz.utils.kill.kill_local", return_value=(2, 2))
    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node00")
    @patch(
        "ezpz.utils.yeet_env._get_worker_nodes",
        return_value=["node00", "node01", "node02"],
    )
    def test_partial_remote_failure_reflected_in_count(
        self, _nodes, _host, _local, mock_ssh, capsys,
    ):
        """Remote SSH failure decreases succeeded; local success counts."""
        # node01 fails, node02 succeeds
        def fake_ssh(node, *_args, **_kw):
            return (node, 0 if node == "node02" else 255, "boom" if node != "node02" else "")
        mock_ssh.side_effect = fake_ssh
        succeeded, total = kill_mod.kill_remote("foo", signal.SIGTERM)
        # 1 (local OK) + 1 (node02 OK) = 2; total = 3
        assert (succeeded, total) == (2, 3)

    @patch("ezpz.utils.kill.kill_local", return_value=(1, 2))
    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node00")
    @patch("ezpz.utils.yeet_env._get_worker_nodes", return_value=["node00"])
    def test_local_partial_kill_is_failure(self, _nodes, _host, _local, capsys):
        """When local matches > local kills, the local node counts as failed."""
        # No remote nodes: total=1, succeeded=0 because local was partial
        succeeded, total = kill_mod.kill_remote("foo", signal.SIGTERM)
        assert (succeeded, total) == (0, 1)

    @patch("ezpz.utils.kill._ssh_kill", return_value=("node01", 0, ""))
    @patch("ezpz.utils.kill.kill_local", return_value=(0, 0))
    @patch("ezpz.utils.yeet_env._get_current_hostname", return_value="node00")
    @patch("ezpz.utils.yeet_env._get_worker_nodes")
    def test_hostfile_passthrough(
        self, mock_nodes, _host, _local, _ssh, capsys,
    ):
        """The hostfile kwarg is forwarded to _get_worker_nodes."""
        mock_nodes.return_value = ["node00", "node01"]
        kill_mod.kill_remote(None, signal.SIGTERM, hostfile="/tmp/myhosts")
        mock_nodes.assert_called_once_with(hostfile="/tmp/myhosts")
