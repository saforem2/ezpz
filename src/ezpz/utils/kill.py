"""Kill ezpz-launched python processes (or anything matching a pattern).

Usage::

    ezpz kill                # local node, ezpz-launched python procs
    ezpz kill <STR>          # local node, any process whose cmdline contains <STR>
    ezpz kill --all-nodes    # fan out across the job's hostfile
    ezpz kill --dry-run      # list matches, don't signal
    ezpz kill --signal KILL  # default is TERM; KILL/INT also accepted

Default match (no STR) looks for the ``EZPZ_RUN_COMMAND`` environment
variable, which ``ezpz launch`` sets on every process it starts. This
keeps the no-arg case narrow — won't touch stray python processes.

With STR, matches any process whose ``/proc/<pid>/cmdline`` contains
the substring (``pkill -f`` style).
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

EZPZ_MARKER = "EZPZ_RUN_COMMAND"

# Signals we accept on the command line. Mapped to signal numbers via
# the signal module so we don't need to translate by hand.
_SIGNAL_MAP = {
    "TERM": signal.SIGTERM,
    "KILL": signal.SIGKILL,
    "INT": signal.SIGINT,
    "HUP": signal.SIGHUP,
    "QUIT": signal.SIGQUIT,
}


def _read_proc_file(pid: int, name: str) -> Optional[bytes]:
    """Return /proc/<pid>/<name> bytes, or None on permission/missing."""
    try:
        with open(f"/proc/{pid}/{name}", "rb") as f:
            return f.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def _decode_nul_separated(data: bytes) -> str:
    """NUL-separated → space-joined. Used for cmdline and environ display."""
    return " ".join(p.decode("utf-8", errors="replace") for p in data.split(b"\0") if p)


def _find_matches_linux(pattern: Optional[str]) -> list[tuple[int, str]]:
    """Linux: scan /proc for matching PIDs.

    Returns list of (pid, cmdline_for_display).
    """
    matches: list[tuple[int, str]] = []
    self_pid = os.getpid()
    parent_pid = os.getppid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in (self_pid, parent_pid):
            continue

        cmdline_bytes = _read_proc_file(pid, "cmdline")
        if not cmdline_bytes:
            continue
        cmdline = _decode_nul_separated(cmdline_bytes)
        if not cmdline:
            continue

        if pattern is None:
            # Default: require EZPZ_RUN_COMMAND in the process environ.
            environ_bytes = _read_proc_file(pid, "environ")
            if not environ_bytes:
                continue
            if EZPZ_MARKER.encode() + b"=" not in environ_bytes:
                continue
        else:
            if pattern not in cmdline:
                continue
        matches.append((pid, cmdline))
    return matches


def _find_matches_macos(pattern: Optional[str]) -> list[tuple[int, str]]:
    """macOS dev fallback: parse `ps` output.

    macOS doesn't expose other processes' environment without elevated
    privileges, so the default (no-pattern) form is unsupported here —
    use ``ezpz kill <STR>`` with an explicit substring instead.
    """
    if pattern is None:
        print(
            "  ezpz kill (no pattern) is unsupported on macOS — pass a "
            "substring (`ezpz kill <STR>`) instead. The default "
            "EZPZ_RUN_COMMAND-based match requires reading /proc/<pid>/environ "
            "(Linux only).",
            file=sys.stderr,
        )
        return []

    matches: list[tuple[int, str]] = []
    self_pid = os.getpid()
    parent_pid = os.getppid()
    try:
        result = subprocess.run(
            ["ps", "-ax", "-o", "pid=,command="],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return matches
    if result.returncode != 0:
        return matches
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        head, _, cmdline = line.partition(" ")
        try:
            pid = int(head)
        except ValueError:
            continue
        if pid in (self_pid, parent_pid):
            continue
        if not cmdline:
            continue
        if pattern not in cmdline:
            continue
        matches.append((pid, cmdline))
    return matches


def find_matches(pattern: Optional[str]) -> list[tuple[int, str]]:
    """Public match entry point — picks platform implementation."""
    if Path("/proc").is_dir():
        return _find_matches_linux(pattern)
    return _find_matches_macos(pattern)


def _kill_pid(pid: int, sig: int, *, kill_after: float = 3.0) -> bool:
    """Send *sig* to *pid*; if still alive after *kill_after* seconds, SIGKILL.

    Returns True if the process is gone afterwards, False otherwise.
    """
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return True  # already gone
    except PermissionError:
        logger.warning("permission denied killing pid %d", pid)
        return False

    if sig == signal.SIGKILL:
        # Nothing to escalate to.
        time.sleep(0.1)
        return not _pid_alive(pid)

    # Wait a short grace period, then escalate to SIGKILL.
    deadline = time.monotonic() + kill_after
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    time.sleep(0.1)
    return not _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    """Return True if *pid* exists (and we can signal it)."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — count as alive so the
        # caller doesn't mistakenly report success.
        return True


def kill_local(
    pattern: Optional[str],
    sig: int,
    *,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Kill matches on the local node. Returns (killed, total_matches)."""
    matches = find_matches(pattern)
    if not matches:
        return 0, 0

    killed = 0
    sig_name = next((k for k, v in _SIGNAL_MAP.items() if v == sig), str(sig))
    for pid, cmdline in matches:
        # Truncate display to keep output manageable
        display = cmdline if len(cmdline) <= 120 else cmdline[:117] + "..."
        if dry_run:
            print(f"  would kill {pid}: {display}")
            killed += 1
            continue
        if _kill_pid(pid, sig):
            print(f"  killed {pid} ({sig_name}): {display}")
            killed += 1
        else:
            print(f"  failed to kill {pid}: {display}")
    return killed, len(matches)


def _ssh_kill(node: str, pattern: Optional[str], sig_name: str, dry_run: bool) -> tuple[str, int, str]:
    """SSH into *node* and run `ezpz kill`. Returns (node, returncode, stderr)."""
    remote_cmd = ["ezpz", "kill", "--signal", sig_name]
    if dry_run:
        remote_cmd.append("--dry-run")
    if pattern is not None:
        remote_cmd.append(pattern)
    ssh_cmd = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "BatchMode=yes",
        node,
        shlex.join(remote_cmd),
    ]
    try:
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, check=False, timeout=60,
        )
        # Print remote stdout under a node-prefixed header.
        if result.stdout.strip():
            print(f"[{node}]")
            for line in result.stdout.splitlines():
                print(f"  {line}")
        return node, result.returncode, result.stderr
    except subprocess.TimeoutExpired:
        return node, 124, "ssh timed out after 60s"
    except FileNotFoundError:
        return node, 127, "ssh not found on this host"


def kill_remote(
    pattern: Optional[str],
    sig: int,
    *,
    dry_run: bool = False,
    hostfile: Optional[str] = None,
) -> tuple[int, int]:
    """Fan out kill across the hostfile. Returns (succeeded, total_nodes)."""
    # Lazy import — _get_worker_nodes already handles PBS/SLURM/HSN.
    from ezpz.utils.yeet_env import _get_worker_nodes, _get_current_hostname

    nodes = _get_worker_nodes(hostfile=hostfile)
    if not nodes:
        print("  no worker nodes discovered")
        return 0, 0

    current = _get_current_hostname()
    current_variants = {current, current + "-hsn0", current.removesuffix("-hsn0")}
    remote_nodes = [n for n in nodes if n not in current_variants]

    sig_name = next((k for k, v in _SIGNAL_MAP.items() if v == sig), str(sig))

    # Local kill in-process; SSH to everyone else.
    print(f"[{current} (local)]")
    local_killed, local_matched = kill_local(pattern, sig, dry_run=dry_run)
    if local_matched == 0:
        print("  no matches")

    # A node "succeeded" only when every matched process was killed
    # (or there were no matches). Use the same predicate here and in
    # the multi-node branch below — the previous `local_killed > 0`
    # form treated a partial kill as full success and made `run()`
    # report exit 0 with stragglers still alive.
    local_ok = local_matched == 0 or local_killed == local_matched
    if not remote_nodes:
        return (1 if local_ok else 0), 1

    # Bound concurrency — DNS/SSH on a 1000-node alloc shouldn't fork a
    # thread per node.
    succeeded = 1 if local_ok else 0
    with ThreadPoolExecutor(max_workers=min(16, len(remote_nodes))) as pool:
        futures = {
            pool.submit(_ssh_kill, n, pattern, sig_name, dry_run): n
            for n in remote_nodes
        }
        for fut in as_completed(futures):
            node, rc, stderr = fut.result()
            if rc == 0:
                succeeded += 1
            else:
                print(f"  [{node}] FAILED (exit {rc}): {stderr.strip()}")
    return succeeded, 1 + len(remote_nodes)


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse ezpz kill command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="ezpz kill",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Kill ezpz-launched python processes. Without arguments, "
            "kills processes on the local node whose environment "
            "contains EZPZ_RUN_COMMAND. With a STRING, matches any "
            "process whose cmdline contains the substring (pkill -f "
            "style)."
        ),
    )
    parser.add_argument(
        "pattern",
        nargs="?",
        default=None,
        metavar="STRING",
        help="Substring to match against cmdline (default: match EZPZ_RUN_COMMAND env var)",
    )
    parser.add_argument(
        "--all-nodes",
        action="store_true",
        help="SSH into every node in the hostfile and kill there too",
    )
    parser.add_argument(
        "--hostfile",
        type=str,
        default=None,
        help="Hostfile to use for --all-nodes (default: auto-detect from scheduler)",
    )
    parser.add_argument(
        "--signal",
        type=str,
        default="TERM",
        choices=list(_SIGNAL_MAP.keys()),
        help="Signal to send",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matches without sending any signals",
    )
    return parser.parse_args(argv)


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    sig = _SIGNAL_MAP[args.signal]

    if args.all_nodes:
        succeeded, total = kill_remote(
            args.pattern, sig,
            dry_run=args.dry_run,
            hostfile=args.hostfile,
        )
        if total == 0:
            return 1
        if succeeded == total:
            return 0
        print(f"  {total - succeeded}/{total} node(s) failed")
        return 1

    killed, total = kill_local(args.pattern, sig, dry_run=args.dry_run)
    if total == 0:
        # Friendly hint when nothing matched.  On macOS without a
        # pattern, _find_matches_macos has already printed an
        # "unsupported on macOS" message — don't double up.
        if args.pattern is None:
            if Path("/proc").is_dir():
                print(
                    "  no processes with EZPZ_RUN_COMMAND env var found "
                    f"on {socket.gethostname()}"
                )
        else:
            print(f"  no processes matching {args.pattern!r}")
        return 0
    if killed == total:
        return 0
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
