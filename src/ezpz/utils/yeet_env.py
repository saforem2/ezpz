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


def _patch_venv_paths_local(dst: Path, src: Path) -> None:
    """Rewrite hardcoded paths in a *local* venv copy so it works from *dst*.

    Called once on the current node before tree distribution, so all
    rsynced copies already have correct paths — no per-node SSH needed.

    Patches:
    - ``bin/activate`` (and variants): replace VIRTUAL_ENV path
    - ``bin/python*``: re-link to the system python if they're symlinks
      pointing back to the original location
    - ``pyvenv.cfg``: update ``home`` to point to the system python dir
    """
    src_str = str(src)
    dst_str = str(dst)
    # Patch activate scripts and pyvenv.cfg via sed
    for fname in ("bin/activate", "bin/activate.csh", "bin/activate.fish",
                   "pyvenv.cfg"):
        fpath = dst / fname
        if fpath.is_file():
            try:
                text = fpath.read_text()
                fpath.write_text(text.replace(src_str, dst_str))
            except OSError:
                pass
    # Re-link python binaries
    bin_dir = dst / "bin"
    if bin_dir.is_dir():
        # Read base python dir from pyvenv.cfg
        cfg = dst / "pyvenv.cfg"
        base_python = None
        if cfg.is_file():
            for line in cfg.read_text().splitlines():
                if line.startswith("home"):
                    base_python = line.split("=", 1)[1].strip()
                    break
        for link in bin_dir.iterdir():
            if link.is_symlink() and link.name.startswith("python"):
                target = str(link.resolve())
                if src_str in target and base_python:
                    py3 = Path(base_python) / "python3"
                    if py3.exists():
                        link.unlink()
                        link.symlink_to(py3)


def _rsync_to_node(
    src: Path,
    dst: Path,
    node: str,
    *,
    from_node: str | None = None,
    progress_callback: object | None = None,
) -> tuple[str, float, int]:
    """Rsync *src* to *node*:*dst*, optionally from a remote source.

    Args:
        from_node: If set, SSH into this node and run rsync from there.
            This enables tree distribution where completed nodes become
            sources. If ``None``, rsync runs locally.
        progress_callback: If provided, called with ``(pct, speed, eta)``
            strings parsed from ``rsync --info=progress2`` output.

    Returns ``(node, elapsed_seconds, returncode)``.
    """
    src_str = str(src).rstrip("/") + "/"
    dst_str = f"{node}:{dst}/"
    t0 = time.perf_counter()

    rsync_cmd = [
        "rsync",
        "-a",               # archive mode
        "--info=progress2",  # single overall progress line
        src_str,
        dst_str,
    ]

    # If source is a remote node, wrap the rsync in an SSH call
    if from_node is not None:
        cmd = ["ssh", from_node, " ".join(rsync_cmd)]
    else:
        cmd = rsync_cmd

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
    #
    # Greedy tree distribution: instead of all N nodes pulling from the
    # source (which saturates the source node's NIC), each completed
    # node immediately becomes a source for others.  A single thread
    # pool runs for the entire sync — as soon as any rsync finishes,
    # new rsyncs are submitted using the newly-available source.
    #
    # Each source has a concurrency cap (MAX_PER_SOURCE) so no single
    # node is overwhelmed.  The tree grows organically: the first node
    # seeds a few targets, each of those fans out to more, etc.

    MAX_PER_SOURCE = 8  # max concurrent outbound rsyncs per source node

    all_nodes: list[str] = []
    if needs_local_copy:
        all_nodes.append(current)
    all_nodes.extend(remote_nodes)

    total = len(all_nodes)
    progress = _AggregateProgress(total_nodes=total)
    results: list[tuple[str, float, int]] = []

    print(f"  Syncing ({total} nodes)...")
    t0 = time.perf_counter()

    # Step 1: rsync source to local /tmp/ and patch paths ONCE.
    # All subsequent rsyncs distribute the already-patched copy.
    if needs_local_copy:
        print(f"    Copying to local /tmp/...")
        _, local_elapsed, local_rc = _rsync_to_node(src, dst, current)
        if local_rc == 0:
            _patch_venv_paths_local(dst, src)
            print(f"    \u2713 {current} (local) \u2014 {local_elapsed:.1f}s")
            results.append((current, local_elapsed, local_rc))
        else:
            print(f"    \u2717 {current} (local) \u2014 FAILED")
            results.append((current, local_elapsed, local_rc))

    remaining = [n for n in all_nodes if n != current]

    # Track per-source active rsync count to enforce MAX_PER_SOURCE.
    source_active: dict[str, int] = {current: 0}
    source_lock = threading.Lock()

    def _pick_source() -> str | None:
        """Pick the source with the fewest active rsyncs, if under cap."""
        best: str | None = None
        best_count = MAX_PER_SOURCE + 1
        for s, count in source_active.items():
            if count < MAX_PER_SOURCE and count < best_count:
                best = s
                best_count = count
        return best

    def _submit_work(pool: ThreadPoolExecutor, futures: dict) -> None:  # type: ignore[type-arg]
        """Submit as many rsyncs as sources allow."""
        while remaining:
            with source_lock:
                src_node = _pick_source()
                if src_node is None:
                    break  # all sources at capacity — wait for completions
                target = remaining.pop(0)
                source_active[src_node] += 1

            remote_src = None if src_node == current else src_node
            fut = pool.submit(
                _rsync_to_node,
                dst,  # always rsync from the /tmp/ copy
                dst,
                target,
                from_node=remote_src,
                progress_callback=progress.update,
            )
            futures[fut] = (target, src_node)

    # Step 2: greedy fan-out using a single persistent pool.
    with ThreadPoolExecutor(max_workers=min(total, 128)) as pool:
        futures: dict = {}
        _submit_work(pool, futures)

        while futures:
            # Wait for the next completion
            done_iter = as_completed(futures)
            fut = next(done_iter)
            n, elapsed, rc = fut.result()
            _, src_used = futures.pop(fut)

            with source_lock:
                source_active[src_used] -= 1
                if rc == 0:
                    # This node now has the data — register as a source
                    source_active[n] = 0

            label = f"{n} (local)" if n == current else n
            progress.mark_done(label, elapsed, rc)
            results.append((n, elapsed, rc))

            # Submit more work now that a source slot freed up
            # (and possibly a new source appeared)
            _submit_work(pool, futures)

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
