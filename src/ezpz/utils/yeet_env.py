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
import os
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


# ── Rsync ────────────────────────────────────────────────────────────────────


def _rsync_to_node(
    src: Path,
    dst: Path,
    node: str,
) -> tuple[str, float, int]:
    """Rsync *src* to *node*:*dst* via SSH.

    Returns ``(node, elapsed_seconds, returncode)``.
    """
    # Trailing slash on src ensures contents are synced, not the dir itself
    src_str = str(src).rstrip("/") + "/"
    dst_str = f"{node}:{dst}/"
    t0 = time.perf_counter()
    result = subprocess.run(
        [
            "rsync",
            "-a",           # archive mode (preserves permissions, symlinks, etc.)
            "--delete",     # remove files on dst that don't exist on src
            "-q",           # quiet — suppress per-file output
            src_str,
            dst_str,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        logger.warning(
            "rsync to %s failed (exit %d): %s",
            node,
            result.returncode,
            result.stderr.strip(),
        )
    return node, elapsed, result.returncode


def _rsync_parallel(
    src: Path,
    dst: Path,
    nodes: list[str],
) -> list[tuple[str, float, int]]:
    """Rsync *src* to *dst* on all *nodes* in parallel.

    Uses a thread pool with one thread per node (each thread manages
    a subprocess).  Returns a list of ``(node, elapsed, returncode)``
    tuples in completion order.
    """
    results: list[tuple[str, float, int]] = []
    with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        futures = {
            pool.submit(_rsync_to_node, src, dst, node): node
            for node in nodes
        }
        for future in as_completed(futures):
            node, elapsed, rc = future.result()
            icon = "\u2713" if rc == 0 else "\u2717"
            print(f"    {icon} {node} \u2014 {elapsed:.1f}s")
            results.append((node, elapsed, rc))
    return results


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
    results: list[tuple[str, float, int]] = []

    # Local copy first (current node's /tmp is node-local)
    if needs_local_copy:
        node, elapsed, rc = _rsync_to_node(src, dst, current)
        icon = "\u2713" if rc == 0 else "\u2717"
        print(f"    {icon} {node} (local) \u2014 {elapsed:.1f}s")
        results.append((node, elapsed, rc))

    # Remote nodes in parallel
    if remote_nodes:
        results.extend(_rsync_parallel(src, dst, remote_nodes))

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
    print(f"  Then launch your training:")
    print(f"    ezpz launch python3 -m your_app.train")

    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
