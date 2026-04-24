"""Distribute a Python environment to worker nodes via parallel rsync.

Default behavior: auto-detect the active Python environment and rsync
it to ``/tmp/<env-name>/`` on every node in the current job allocation.

Usage::

    # Rsync the active venv to all worker nodes:
    ezpz yeet-env

    # Rsync a specific path:
    ezpz yeet-env --src /path/to/env

    # Custom destination:
    ezpz yeet-env --dst /local/fast/storage/myenv

    # Preview without syncing:
    ezpz yeet-env --dry-run
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import ezpz

logger = ezpz.get_logger(__name__)


# ── Environment detection ────────────────────────────────────────────────────


def _detect_env_source() -> Path:
    """Return the active Python environment prefix.

    Checks ``sys.prefix`` (works for both venv and conda envs).
    Falls back to the parent of ``sys.executable`` if prefix looks
    like a system Python.
    """
    prefix = Path(sys.prefix).resolve()
    # If sys.prefix == sys.base_prefix, we're not in a venv — use the
    # executable's parent directory as a fallback.
    if prefix == Path(sys.base_prefix).resolve():
        logger.warning(
            "Not running inside a virtualenv or conda env. "
            "Using sys.executable parent: %s",
            Path(sys.executable).parent.parent,
        )
        return Path(sys.executable).parent.parent.resolve()
    return prefix


def _get_env_size(path: Path) -> str:
    """Return a human-readable size string for a directory."""
    try:
        result = subprocess.run(
            ["du", "-sh", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.split()[0]
    except Exception:
        pass
    return "?"


# ── Node discovery ───────────────────────────────────────────────────────────


def _get_worker_nodes(hostfile: str | None = None) -> list[str]:
    """Get unique worker node hostnames from the job allocation.

    Uses ``get_hostfile_with_fallback`` and ``get_nodes_from_hostfile``
    from ``ezpz.distributed`` to discover nodes from PBS, SLURM, or
    a user-provided hostfile.
    """
    from ezpz.distributed import (
        get_hostfile_with_fallback,
        get_nodes_from_hostfile,
    )

    hf = get_hostfile_with_fallback(hostfile=hostfile)
    nodes = get_nodes_from_hostfile(hf)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for node in nodes:
        # Normalize: strip FQDN suffixes
        short = node.split(".")[0]
        if short not in seen:
            seen.add(short)
            unique.append(short)
    return unique


def _get_current_hostname() -> str:
    """Return the short hostname of the current node."""
    return socket.getfqdn().split(".")[0]


# ── Progress indicator ────────────────────────────────────────────────────────


import threading


class _AggregateProgress:
    """Single-line progress display that aggregates status across all nodes.

    Instead of N competing progress lines (one per node), this shows
    one shared line: ``⠹ 3/8 nodes done  72%  45MB/s  [120s]``
    """

    _FRAMES = ["\u280b", "\u2819", "\u2838", "\u2834", "\u2826", "\u2807"]

    def __init__(self, total_nodes: int) -> None:
        self._total = total_nodes
        self._done = 0
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self._idx = 0
        # Track latest progress from any node
        self._pct = ""
        self._speed = ""

    def update(self, pct: str = "", speed: str = "", eta: str = "") -> None:
        """Called by any node's rsync thread with progress info."""
        with self._lock:
            if pct:
                self._pct = pct
            if speed:
                self._speed = speed
            self._redraw()

    def mark_done(self, node: str, elapsed: float, rc: int) -> None:
        """Called when a node finishes. Prints result on its own line."""
        with self._lock:
            sys.stdout.write("\r\033[K")
            self._done += 1
            icon = "\u2713" if rc == 0 else "\u2717"
            print(f"    {icon} {node} \u2014 {elapsed:.1f}s")
            if self._done < self._total:
                self._redraw()

    def clear(self) -> None:
        """Clear the progress line."""
        with self._lock:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def _redraw(self) -> None:
        elapsed = time.perf_counter() - self._t0
        frame = self._FRAMES[self._idx % len(self._FRAMES)]
        self._idx += 1
        remaining = self._total - self._done
        parts = [f"{frame} {remaining} node(s) syncing"]
        if self._pct:
            parts.append(self._pct)
        if self._speed:
            parts.append(self._speed)
        parts.append(f"[{elapsed:.0f}s]")
        msg = "\r    " + "  ".join(parts)
        sys.stdout.write(msg)
        sys.stdout.flush()


# ── Rsync ────────────────────────────────────────────────────────────────────


def _patch_venv_paths(dst: Path, src: Path, node: str) -> None:
    """Rewrite hardcoded paths in a copied venv so it works from *dst*.

    Patches:
    - ``bin/activate`` (and variants): replace VIRTUAL_ENV path
    - ``bin/python*``: re-link to the system python if they're symlinks
      pointing back to the original location
    - ``pyvenv.cfg``: update ``home`` to point to the system python dir
    """
    src_str = str(src)
    dst_str = str(dst)
    # Patch activate scripts and pyvenv.cfg via sed on the remote node
    patch_cmd = (
        f"sed -i 's|{src_str}|{dst_str}|g' "
        f"{dst_str}/bin/activate "
        f"{dst_str}/bin/activate.csh "
        f"{dst_str}/bin/activate.fish "
        f"{dst_str}/pyvenv.cfg "
        f"2>/dev/null; "
        # Re-link python binaries: if python3 is a symlink to the original
        # location, find the real interpreter and re-link to it.
        f"cd {dst_str}/bin && "
        f"for f in python python3 python3.*; do "
        f"  if [ -L \"$f\" ]; then "
        f"    target=$(readlink \"$f\"); "
        f"    if echo \"$target\" | grep -q '{src_str}'; then "
        # Find the base python from pyvenv.cfg's "home" line
        f"      base=$(grep '^home' {dst_str}/pyvenv.cfg 2>/dev/null "
        f"             | cut -d= -f2 | tr -d ' '); "
        f"      if [ -n \"$base\" ] && [ -x \"$base/python3\" ]; then "
        f"        ln -sf \"$base/python3\" \"$f\"; "
        f"      fi; "
        f"    fi; "
        f"  fi; "
        f"done"
    )
    subprocess.run(
        ["ssh", node, patch_cmd],
        capture_output=True,
        check=False,
    )


def _rsync_to_node(
    src: Path,
    dst: Path,
    node: str,
    *,
    patch_paths: bool = True,
    progress_callback: object | None = None,
) -> tuple[str, float, int]:
    """Rsync *src* to *node*:*dst* via SSH, then patch venv paths.

    Args:
        progress_callback: If provided, called with ``(pct, speed, eta)``
            strings parsed from ``rsync --info=progress2`` output.

    Returns ``(node, elapsed_seconds, returncode)``.
    """
    # Trailing slash on src ensures contents are synced, not the dir itself
    src_str = str(src).rstrip("/") + "/"
    dst_str = f"{node}:{dst}/"
    t0 = time.perf_counter()

    cmd = [
        "rsync",
        "-a",               # archive mode
        "--info=progress2",  # single overall progress line
        src_str,
        dst_str,
    ]

    if progress_callback is not None:
        # Stream output line-by-line to parse progress
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                # rsync --info=progress2 output looks like:
                #   1.23G  14%   45.67MB/s    0:03:12
                line = line.strip()
                if "%" in line:
                    parts = line.split()
                    pct = ""
                    speed = ""
                    eta = ""
                    for p in parts:
                        if p.endswith("%"):
                            pct = p
                        elif p.endswith("/s"):
                            speed = p
                        elif ":" in p and p[0].isdigit():
                            eta = p
                    progress_callback(pct, speed, eta)  # type: ignore[operator]
            # Read stderr before the context manager closes it
            stderr = proc.stderr.read() if proc.stderr else ""
        returncode = proc.returncode or 0
    else:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
        )
        returncode = result.returncode
        stderr = result.stderr

    if returncode != 0:
        logger.warning(
            "rsync to %s failed (exit %d): %s",
            node,
            returncode,
            stderr.strip() if stderr else "",
        )
    elif patch_paths:
        _patch_venv_paths(dst, src, node)
    elapsed = time.perf_counter() - t0
    return node, elapsed, returncode



# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse yeet-env command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="ezpz yeet-env",
        description=(
            "Distribute a Python environment to worker nodes via parallel rsync. "
            "By default, rsyncs the active venv/conda env to /tmp/<env-name>/ "
            "on all nodes in the current job allocation."
        ),
    )
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help="Source environment path (default: active venv/conda env).",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="Destination path on worker nodes (default: /tmp/<env-name>/).",
    )
    parser.add_argument(
        "--hostfile",
        type=str,
        default=None,
        help="Hostfile to read node list from (default: auto-detect from scheduler).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without doing it.",
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        # Filter out stray positional args (e.g. "yeet-env" leaked from
        # click's UNPROCESSED dispatch or deprecated entry points).
        real_unknown = [a for a in unknown if a.startswith("-")]
        if real_unknown:
            parser.error(f"unrecognized arguments: {' '.join(real_unknown)}")
    return args


def run(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for yeet-env."""
    args = parse_args(argv)

    # ── Resolve source ──────────────────────────────────────────────
    if args.src is not None:
        src = Path(args.src).resolve()
        if not src.exists():
            logger.error("Source path does not exist: %s", src)
            return 1
    else:
        src = _detect_env_source()

    env_name = src.name

    # ── Resolve destination ─────────────────────────────────────────
    if args.dst is not None:
        dst = Path(args.dst)
    else:
        dst = Path("/tmp") / env_name

    # ── Discover nodes ──────────────────────────────────────────────
    nodes = _get_worker_nodes(hostfile=args.hostfile)
    current = _get_current_hostname()

    if not nodes:
        print(f"  No worker nodes found.")
        return 1

    # Copy locally first (current node), then rsync to remote nodes.
    # The local copy is needed because /tmp is node-local.
    needs_local_copy = not str(src).startswith("/tmp")
    remote_nodes = [n for n in nodes if n != current]

    # ── Print summary ───────────────────────────────────────────────
    env_size = _get_env_size(src)
    total_nodes = (1 if needs_local_copy else 0) + len(remote_nodes)
    print(f"  Source: {src} ({env_size})")
    print(f"  Target: {dst}/ on {total_nodes} node(s)")
    if needs_local_copy:
        print(f"    local:  {current} (rsync to {dst}/)")
    if remote_nodes:
        print(f"    remote: {', '.join(remote_nodes)}")
    if args.dry_run:
        print(f"  [dry-run] No files transferred.")
        return 0

    if total_nodes == 0:
        print(f"  Nothing to sync (source is already in {dst}).")
        return 0

    # ── Sync ────────────────────────────────────────────────────────
    print(f"  Syncing...")
    t0 = time.perf_counter()

    # Sync all nodes in parallel (local + remote together).
    all_nodes = []
    if needs_local_copy:
        all_nodes.append(current)
    all_nodes.extend(remote_nodes)

    # Cap parallelism to avoid overwhelming the source filesystem.
    max_workers = min(len(all_nodes), 32)
    progress = _AggregateProgress(total_nodes=len(all_nodes))

    results: list[tuple[str, float, int]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _rsync_to_node, src, dst, node,
                progress_callback=progress.update,
            ): node
            for node in all_nodes
        }
        for fut in as_completed(futures):
            n, elapsed, rc = fut.result()
            label = f"{n} (local)" if n == current else n
            progress.mark_done(label, elapsed, rc)
            results.append((n, elapsed, rc))
    progress.clear()

    total_elapsed = time.perf_counter() - t0
    failed = sum(1 for _, _, rc in results if rc != 0)
    if failed:
        print(f"  {failed}/{total_nodes} node(s) failed!")
    else:
        print(f"  Done in {total_elapsed:.1f}s")

    # ── Guidance ────────────────────────────────────────────────────
    is_venv = (src / "bin" / "activate").exists()
    is_conda = (src / "conda-meta").is_dir()
    print()
    print(f"  To use this environment:")
    if is_venv:
        print(f"    deactivate 2>/dev/null")
        print(f"    source {dst}/bin/activate")
    elif is_conda:
        print(f"    conda deactivate")
        print(f"    conda activate {dst}")
    else:
        print(f"    export PATH={dst}/bin:$PATH")
    print()
    print(f"  Then launch your training (from a shared filesystem path):")
    print(f"    cd /path/to/your/project")
    print(f"    ezpz launch python3 -m your_app.train")
    print()
    print(f"  Note: /tmp is node-local. Make sure your working directory")
    print(f"  is on a shared filesystem (e.g. Lustre) before launching,")
    print(f"  so all ranks can access data and outputs.")

    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
