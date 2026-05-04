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
import atexit
import logging
import os
import random
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

def _get_ezpz_logger() -> logging.Logger:
    """Use ezpz.get_logger (timestamped, colored) when available.

    Falls back to a plain stdlib logger when ezpz.log isn't importable
    yet (avoids a circular import during early module setup or when
    yeet_env is imported standalone for testing).
    """
    try:
        import ezpz
        return ezpz.get_logger(__name__)
    except (ImportError, AttributeError):
        return logging.getLogger(__name__)


logger = _get_ezpz_logger()

# Disable \r / ANSI escape progress when stdout is not a terminal
# (e.g. redirected to a file or pipe).
_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

# Tracks the SLURM-derived /tmp hostfiles we've already registered an
# atexit handler for, so repeat invocations of _get_worker_nodes (in
# tests, embedded use, etc.) don't register a fresh handler each time.
# Guarded by _REGISTERED_CLEANUPS_LOCK because the check-then-add
# sequence is non-atomic — concurrent calls (multi-threaded embeddings,
# parallel pytest workers) could otherwise both pass the membership
# check and double-register.
_REGISTERED_CLEANUPS: set[str] = set()
_REGISTERED_CLEANUPS_LOCK = threading.Lock()


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


def _suggest_tarball_if_present(src: Path) -> None:
    """Print a one-line hint if a same-named tarball exists nearby.

    Tarball broadcast (`ezpz yeet foo.tar.gz`) is ~10× faster at
    scale than per-file rsync of `foo/` because it reduces the Lustre
    side to one sequential read. Users who run plain `ezpz yeet`
    without realizing they have a pre-built tarball pay the per-file
    cost unnecessarily.

    This only fires when the user did NOT pass --src (handled by the
    caller) — if they explicitly chose a source, don't second-guess
    them. Stale tarballs (older than the env's pyvenv.cfg) are
    skipped so the hint doesn't push users toward an outdated copy.
    """
    if not src.is_dir():
        return  # src is already a file (e.g. tarball passed explicitly)
    name = src.name  # ".venv" / "myenv" / etc.
    candidates = []
    # Look next to the env, then in $cwd. Same-named first; .tgz second.
    for parent in (src.parent, Path.cwd()):
        for suffix in (".tar.gz", ".tgz"):
            candidate = parent / f"{name}{suffix}"
            if candidate.is_file():
                candidates.append(candidate)
    if not candidates:
        return
    tarball = candidates[0]
    # Skip if the tarball is older than the venv's pyvenv.cfg (stale).
    cfg = src / "pyvenv.cfg"
    if cfg.is_file():
        try:
            if tarball.stat().st_mtime < cfg.stat().st_mtime:
                return
        except OSError:
            pass
    try:
        size_gb = tarball.stat().st_size / (1024 ** 3)
        size_str = f" ({size_gb:.1f}G)"
    except OSError:
        size_str = ""
    logger.warning(
        "Tip: found %s%s — pass it explicitly for ~10x faster local "
        "copy at scale:  ezpz yeet %s",
        tarball, size_str, tarball,
    )


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

    Reads nodes from PBS_NODEFILE, SLURM_NODELIST, or a user-provided
    hostfile.  Avoids importing heavy ezpz modules (torch, numpy, etc.)
    so the CLI starts fast even on slow filesystems.
    """
    # Resolve hostfile path
    hf: str | None = hostfile
    if hf is None:
        for var in ("PBS_NODEFILE", "HOSTFILE"):
            val = os.environ.get(var)
            if val and Path(val).is_file():
                hf = val
                break

    # PBS: try standard PBS hostfile locations
    if hf is None:
        pbs_jobid = os.environ.get("PBS_JOBID")
        # 1. If PBS_JOBID is set, try the standard aux paths
        if pbs_jobid:
            for path in (
                f"/var/spool/pbs/aux/{pbs_jobid}",
                f"/var/spool/PBS/aux/{pbs_jobid}",
            ):
                if Path(path).is_file():
                    hf = path
                    logger.info("Found PBS hostfile: %s", hf)
                    break
        # 2. Query qstat for active job → look up aux file
        # qstat -fn1wru $USER lists running jobs with their nodelists
        if hf is None:
            try:
                user = os.environ.get("USER", "")
                if user:
                    qstat = subprocess.run(
                        ["qstat", "-fn1wru", user],
                        capture_output=True, text=True, check=False,
                    )
                    if qstat.returncode == 0:
                        my_host = socket.getfqdn().split(".")[0]
                        # Strip -hsn0 suffix in case getfqdn returned the
                        # HSN form but qstat lists bare hostnames.
                        if my_host.endswith("-hsn0"):
                            my_host = my_host[: -len("-hsn0")]
                        # Each running job line ends with the nodelist
                        for line in qstat.stdout.splitlines():
                            if " R " not in line:
                                continue
                            parts = [p for p in line.split(" ") if p]
                            if not parts:
                                continue
                            jobid = parts[0].split(".")[0]
                            nodelist = parts[-1]
                            # nodelist format: host1/cpu+host2/cpu+...
                            hosts = [h.split("/")[0] for h in nodelist.split("+")]
                            if my_host in hosts:
                                # Found our job — look up its aux file.
                                # PBS aux files are named "<jobid>" or
                                # "<jobid>.<server>" — match exact or prefix.
                                for aux_dir in ("/var/spool/pbs/aux",
                                                 "/var/spool/PBS/aux"):
                                    aux_path = Path(aux_dir)
                                    if not aux_path.is_dir():
                                        continue
                                    for entry in aux_path.iterdir():
                                        if (entry.name == jobid
                                                or entry.name.startswith(jobid + ".")):
                                            hf = str(entry)
                                            logger.info(
                                                "Found PBS hostfile via qstat: %s",
                                                hf,
                                            )
                                            break
                                    if hf:
                                        break
                                break
            except Exception as exc:
                logger.debug("qstat lookup failed: %s", exc)

    # SLURM: expand nodelist with scontrol
    if hf is None:
        slurm_nodelist = os.environ.get("SLURM_NODELIST")
        if slurm_nodelist:
            try:
                result = subprocess.run(
                    ["scontrol", "show", "hostnames", slurm_nodelist],
                    capture_output=True, text=True, check=True,
                )
                nodes = result.stdout.strip().splitlines()
                if nodes:
                    job_id = os.environ.get("SLURM_JOB_ID", "unknown")
                    hf_path = Path(f"/tmp/_ezpz_hostfile_{job_id}")
                    hf_path.write_text("\n".join(nodes) + "\n")
                    hf = str(hf_path)
                    # Register cleanup once per unique path so repeated
                    # _get_worker_nodes calls don't accumulate handlers.
                    # Lock around the check-then-add so concurrent
                    # callers can't both pass the membership check.
                    with _REGISTERED_CLEANUPS_LOCK:
                        newly_registered = hf not in _REGISTERED_CLEANUPS
                        if newly_registered:
                            _REGISTERED_CLEANUPS.add(hf)
                    if newly_registered:
                        atexit.register(_cleanup_path, hf_path)
            except Exception:
                pass

    if hf is None:
        logger.warning(
            "No hostfile found — using localhost only. "
            "Set PBS_NODEFILE or pass --hostfile."
        )
        return [_get_current_hostname()]

    # Read nodes from hostfile
    nodes = [
        line.strip() for line in Path(hf).read_text().splitlines()
        if line.strip()
    ]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for node in nodes:
        short = node.split(".")[0]
        if short not in seen:
            seen.add(short)
            unique.append(short)

    # If hostnames don't already have -hsn0, check if the HSN
    # (high-speed network) interface resolves per-node.  On Aurora
    # the hostfile has bare names but ssh works much faster via the
    # HSN interface (Slingshot vs management network).
    #
    # Probe each node individually rather than trusting the first
    # one — a heterogeneous allocation could mix HSN-equipped and
    # non-HSN nodes, and we'd otherwise route everyone through a
    # nonexistent -hsn0 NIC.  Use a small thread pool so DNS
    # resolution doesn't dominate startup at scale.
    return _maybe_apply_hsn_suffix(unique)


def _maybe_apply_hsn_suffix(nodes: list[str]) -> list[str]:
    """Per-node: append ``-hsn0`` to each node whose HSN NIC resolves.

    Nodes that already carry the suffix are left alone.  Nodes that
    don't have an HSN entry in DNS keep their bare name — mixing both
    forms is fine because the rsync/ssh pipeline addresses each node
    independently.
    """
    if not nodes:
        return nodes

    def _probe(name: str) -> str:
        if name.endswith("-hsn0"):
            return name
        try:
            socket.gethostbyname(name + "-hsn0")
            return name + "-hsn0"
        except (socket.gaierror, OSError):
            return name

    # Cap the probe pool so we don't spawn a thread per host on a
    # 1000-node allocation.  DNS is fast; 16 threads are plenty.
    with ThreadPoolExecutor(max_workers=min(16, len(nodes))) as pool:
        resolved = list(pool.map(_probe, nodes))

    upgraded = sum(
        1 for orig, new in zip(nodes, resolved) if new != orig
    )
    if upgraded:
        logger.info(
            "HSN interface available on %d/%d nodes (-hsn0 suffix)",
            upgraded, len(nodes),
        )
    return resolved


def _get_current_hostname() -> str:
    """Return the short hostname of the current node."""
    return socket.getfqdn().split(".")[0]


def _cleanup_path(path: Path) -> None:
    """Best-effort unlink; swallow errors (used from atexit/finally)."""
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def pick_source(
    source_active: dict[str, int],
    max_per_source: int,
    *,
    rng: random.Random | None = None,
) -> str | None:
    """Pick a least-loaded source under the per-source cap.

    Ties are broken with the supplied ``rng`` (defaults to a fresh
    ``random.Random()`` per call) so the greedy fan-out actually fans
    out — without randomization, ``dict.items()`` order would always
    pick the same source first, pinning all early traffic to one node
    and defeating the tree distribution.

    Returns ``None`` if every source is at capacity (the caller should
    wait for an in-flight sync to finish, freeing a slot).

    Pulled out of run() so tests can exercise the algorithm directly
    instead of reconstructing it inline.
    """
    candidates = [
        s for s, count in source_active.items() if count < max_per_source
    ]
    if not candidates:
        return None
    min_count = min(source_active[s] for s in candidates)
    least_loaded = [s for s in candidates if source_active[s] == min_count]
    if rng is None:
        rng = random.Random()
    return rng.choice(least_loaded)


def _remove_partial_dst(dst: Path) -> None:
    """Remove a half-written destination directory we just created.

    ``_safe_rmtree`` deliberately refuses paths outside ``/tmp`` to
    block callers from blowing away pre-existing user data.  But the
    failure-cleanup paths in run() *just* wrote into ``dst`` — the
    pre-extraction guard already verified ``dst`` was nonexistent or
    removable, so the contents we're deleting are exclusively what we
    put there moments ago.  Allow the removal regardless of where
    ``--dst`` points.

    Skips the ``dst.exists()`` pre-check because ``rmtree`` is
    already tolerant of missing paths via the error callback (and a
    pre-check would just open a TOCTOU window).  Uses an error
    callback instead of ``ignore_errors=True`` so failures are
    visible in the log instead of silently dropped.
    """
    def _on_rm_error(fn: object, path: str, exc_info: object) -> None:
        # Unpack tuple form (Python 3.11 onerror) or bare exception
        # form (Python 3.12+ onexc) — both shapes are passed through
        # depending on which kwarg the rmtree supports.
        if isinstance(exc_info, tuple):
            exc = exc_info[1]
        else:
            exc = exc_info
        # FileNotFoundError = dst was already gone; not worth logging.
        if isinstance(exc, FileNotFoundError):
            return
        logger.warning(
            "Failed to remove %s during partial-dst cleanup: %s", path, exc,
        )

    # onexc replaced onerror in 3.12; the older keyword still works
    # via deprecation shim but emits a DeprecationWarning under 3.12.
    if sys.version_info >= (3, 12):
        shutil.rmtree(dst, onexc=_on_rm_error)
    else:
        shutil.rmtree(dst, onerror=_on_rm_error)


# ── Progress indicator ────────────────────────────────────────────────────────


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
        if not _IS_TTY:
            return
        with self._lock:
            if pct:
                self._pct = pct
            if speed:
                self._speed = speed
            self._redraw()

    def mark_done(self, node: str, elapsed: float, rc: int) -> None:
        """Called when a node finishes. Prints result on its own line."""
        with self._lock:
            if _IS_TTY:
                sys.stdout.write("\r\033[K")
            self._done += 1
            icon = "\u2713" if rc == 0 else "\u2717"
            print(f"    {icon} {node} \u2014 {elapsed:.1f}s")
            if _IS_TTY and self._done < self._total:
                self._redraw()

    def clear(self) -> None:
        """Clear the progress line."""
        if not _IS_TTY:
            return
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


def _safe_rmtree(path: Path) -> bool:
    """Remove a directory only if it's under /tmp/.

    Returns True if removed, False if refused.
    """
    resolved = str(path.resolve())
    # Only check the resolved path to prevent traversal attacks
    # (e.g. /tmp/../home/user resolves to /home/user).
    # macOS /tmp → /private/tmp is handled by resolve().
    tmp_prefixes = ("/tmp/", "/private/tmp/")
    is_safe = (
        any(resolved.startswith(p) for p in tmp_prefixes)
        and resolved not in ("/tmp", "/tmp/", "/private/tmp", "/private/tmp/")
    )
    if not is_safe:
        logger.error("Refusing to rm -rf outside /tmp/: %s", path)
        return False
    subprocess.run(["rm", "-rf", str(path)], check=False)
    return True


def _patch_venv_paths_local(dst: Path, src: Path) -> None:
    """Rewrite hardcoded paths in a *local* venv copy so it works from *dst*.

    Called once on the current node before tree distribution, so all
    rsynced copies already have correct paths — no per-node SSH needed.

    Patches:
    - ``bin/activate`` (and variants): replace VIRTUAL_ENV path
    - ``bin/python*``: re-link to the system python if they're symlinks
      pointing back to the original location
    - ``pyvenv.cfg``: update ``home`` to point to the system python dir

    When ``src`` doesn't match the actual hardcoded path inside the
    venv (e.g. ``src`` was a tarball file), the original path is
    discovered by reading ``VIRTUAL_ENV=`` from ``bin/activate`` and
    used for the find/replace.
    """
    src_str = str(src)
    dst_str = str(dst)

    # If src looks like a file (not a directory) or doesn't appear in
    # the activate script, discover the real original path.
    activate = dst / "bin" / "activate"
    if activate.is_file():
        try:
            text = activate.read_text()
            if src_str not in text:
                # Look for VIRTUAL_ENV='...' line
                import re
                m = re.search(r"^VIRTUAL_ENV=['\"]?([^'\"\n]+)", text, re.M)
                if m:
                    discovered = m.group(1).strip()
                    if discovered and discovered != dst_str:
                        logger.info(
                            "Discovered original venv path from activate: %s",
                            discovered,
                        )
                        src_str = discovered
        except OSError:
            pass
    # Patch activate scripts and pyvenv.cfg
    for fname in ("bin/activate", "bin/activate.csh", "bin/activate.fish",
                   "pyvenv.cfg"):
        fpath = dst / fname
        if fpath.is_file():
            try:
                text = fpath.read_text()
                fpath.write_text(text.replace(src_str, dst_str))
            except OSError:
                pass
    bin_dir = dst / "bin"
    if not bin_dir.is_dir():
        return
    # Read base python dir from pyvenv.cfg
    cfg = dst / "pyvenv.cfg"
    base_python = None
    if cfg.is_file():
        for line in cfg.read_text().splitlines():
            if line.startswith("home"):
                base_python = line.split("=", 1)[1].strip()
                break
    # Re-link python binaries
    for link in bin_dir.iterdir():
        if link.is_symlink() and link.name.startswith("python"):
            target = str(link.resolve())
            if src_str in target and base_python:
                py3 = Path(base_python) / "python3"
                if py3.exists():
                    link.unlink()
                    link.symlink_to(py3)
    # Patch shebangs in entry-point scripts (ezpz, pip, torchrun, etc.)
    # These are regular files with a first line like:
    #   #!/path/to/original/.venv/bin/python3
    for script in bin_dir.iterdir():
        if script.is_symlink() or not script.is_file():
            continue
        try:
            with open(script, "rb") as f:
                first_line = f.readline()
            if not first_line.startswith(b"#!"):
                continue
            shebang = first_line.decode("utf-8", errors="replace")
            if src_str not in shebang:
                continue
            # Replace the old path in the shebang
            with open(script, "rb") as f:
                content = f.read()
            content = content.replace(
                src_str.encode(), dst_str.encode(),
            )
            with open(script, "wb") as f:
                f.write(content)
        except (OSError, UnicodeDecodeError):
            pass


# Per-rsync wallclock cap.  A single dead/unreachable node should not
# block the whole pool slot indefinitely — if a sync exceeds this, the
# subprocess is killed and the node reported as failed.  3600s (1h)
# accommodates large envs over slow links; bump via env var if needed.
_DEFAULT_RSYNC_TIMEOUT = float(os.environ.get("EZPZ_YEET_RSYNC_TIMEOUT", "3600"))


def _detect_rsync_progress_flag() -> list[str]:
    """Pick the right rsync progress flag for the local rsync binary.

    GNU rsync >= 3.1 supports ``--info=progress2`` (single aggregated
    progress line — what the streaming progress callback parses).
    macOS ships ``openrsync`` (advertised as ``rsync version 2.6.9
    compatible``) which only supports the older ``--progress`` flag
    (per-file granularity, no aggregation).

    Probe by checking whether ``--info=`` appears in ``rsync --help``;
    fall back to ``--progress`` if not, and to no flag at all if even
    ``rsync --help`` fails.
    """
    try:
        result = subprocess.run(
            ["rsync", "--help"], capture_output=True, text=True,
            check=False, timeout=5,
        )
        if result.returncode != 0:
            return []
        if any(line.lstrip().startswith("--info=") for line in result.stdout.splitlines()):
            return ["--info=progress2"]
        return ["--progress"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


# Cached at module load — rsync version doesn't change at runtime.
_RSYNC_PROGRESS_FLAGS = _detect_rsync_progress_flag()


def _drain_stream_to_list(stream: object, sink: list[str]) -> None:
    """Read *stream* line-by-line into *sink* until EOF.

    Used to drain a child's stderr in a background thread so the
    child can never block on a full pipe buffer (default ~64KB on
    Linux).  Errors during read are swallowed: we'd rather lose
    diagnostic output than crash the parent.
    """
    try:
        for line in stream:  # type: ignore[attr-defined]
            sink.append(line)
    except Exception:
        pass


def _rsync_to_node(
    src: Path,
    dst: Path,
    node: str,
    *,
    from_node: str | None = None,
    local: bool = False,
    progress_callback: object | None = None,
    timeout: float | None = None,
) -> tuple[str, float, int]:
    """Rsync *src* to *node*:*dst*, optionally from a remote source.

    Args:
        from_node: If set, SSH into this node and run rsync from there.
            This enables tree distribution where completed nodes become
            sources. If ``None``, rsync runs locally.
        local: If ``True``, destination is a local path (no SSH).
            Uses optimized flags: skip metadata sync, use whole-file
            transfer, exclude ``__pycache__``.
        progress_callback: If provided, called with ``(pct, speed, eta)``
            strings parsed from ``rsync --info=progress2`` output.
        timeout: Wallclock cap (seconds).  Defaults to
            ``EZPZ_YEET_RSYNC_TIMEOUT`` (3600).  Pass ``0`` to disable.

    Returns ``(node, elapsed_seconds, returncode)``.
    """
    src_str = str(src).rstrip("/") + "/"
    t0 = time.perf_counter()
    if timeout is None:
        timeout = _DEFAULT_RSYNC_TIMEOUT

    if local:
        # Local copy: skip metadata sync (-t, -p, -g, -o are slow),
        # use whole-file (no delta algorithm), exclude __pycache__.
        rsync_cmd = [
            "rsync",
            "-rlD",              # recursive, symlinks, devices (no -tpgo)
            "--whole-file",      # skip delta algorithm for local copies
            *_RSYNC_PROGRESS_FLAGS,
            "--exclude=__pycache__",
            src_str,
            str(dst) + "/",
        ]
    else:
        dst_str = f"{node}:{dst}/"
        rsync_cmd = [
            "rsync",
            "-rlD",              # skip expensive metadata sync
            *_RSYNC_PROGRESS_FLAGS,
            "--exclude=__pycache__",
            src_str,
            dst_str,
        ]

    # If source is a remote node, wrap the rsync in an SSH call.
    # ConnectTimeout gives us a fast fail when a node is unreachable;
    # ServerAliveInterval probes for hung sessions on long transfers.
    if from_node is not None:
        ssh_cmd = [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            from_node,
            shlex.join(rsync_cmd),
        ]
        cmd = ssh_cmd
    else:
        cmd = rsync_cmd

    stderr_lines: list[str] = []
    timed_out = False

    if progress_callback is not None:
        # Stream stdout line-by-line to parse progress.  Critically,
        # stderr must be drained on a separate thread — otherwise a
        # noisy rsync can fill the pipe buffer and deadlock both
        # processes (the child blocks on stderr write, the parent
        # blocks waiting for stdout to close).
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None and proc.stderr is not None
        stderr_thread = threading.Thread(
            target=_drain_stream_to_list,
            args=(proc.stderr, stderr_lines),
            daemon=True,
        )
        stderr_thread.start()
        try:
            for line in proc.stdout:
                # rsync --info=progress2 output looks like:
                #   1.23G  14%   45.67MB/s    0:03:12
                stripped = line.strip()
                if "%" in stripped:
                    parts = stripped.split()
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
                # Bail early if the deadline passed mid-transfer
                if timeout and (time.perf_counter() - t0) > timeout:
                    timed_out = True
                    break
        except Exception as exc:
            logger.debug("rsync stdout reader for %s aborted: %s", node, exc)
        # If we already broke out of the read loop on the deadline,
        # SIGKILL the child and reap it with a *short* grace timeout —
        # the wait should complete in milliseconds, not the full
        # original budget.  Reserving the full timeout would mask any
        # surprise hang in the wait itself.
        if timed_out:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "rsync subprocess for %s did not exit after SIGKILL", node,
                )
        else:
            try:
                proc.wait(timeout=timeout if timeout else None)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "rsync subprocess for %s did not exit after SIGKILL",
                        node,
                    )
                timed_out = True
        stderr_thread.join(timeout=5)
        # Best-effort cleanup of pipes — Popen would normally do this
        # via its context manager, but we abandoned that pattern in
        # favor of explicit timeout handling.
        for stream in (proc.stdout, proc.stderr):
            try:
                stream.close()  # type: ignore[union-attr]
            except Exception:
                pass
        returncode = proc.returncode if proc.returncode is not None else 1
    else:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout if timeout else None,
            )
            returncode = result.returncode
            if result.stderr:
                stderr_lines.append(result.stderr)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124  # bash convention for "timed out"
            if exc.stderr:
                if isinstance(exc.stderr, bytes):
                    stderr_lines.append(exc.stderr.decode(errors="replace"))
                else:
                    stderr_lines.append(exc.stderr)

    stderr = "".join(stderr_lines)

    # rsync exit 24 = "some files vanished before they could be transferred"
    # This is normal when concurrent rsyncs read from the same /tmp/ source
    # while temporary files (e.g. triton plugin builds) come and go.
    if returncode == 24:
        returncode = 0
    if timed_out:
        logger.warning(
            "rsync to %s timed out after %.0fs", node, timeout,
        )
    elif returncode != 0:
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
    """Parse ezpz yeet command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="ezpz yeet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Distribute files (envs, models, datasets, etc.) to worker nodes "
            "via parallel rsync. By default (no args), rsyncs the active "
            "venv/conda env to /tmp/<env-name>/ on all nodes in the current "
            "job allocation. Pass any path to yeet arbitrary content."
        ),
    )
    parser.add_argument(
        "src_positional",
        nargs="?",
        default=None,
        metavar="SRC",
        help=(
            "Source path (positional shorthand for --src). Mutually "
            "exclusive with --src. May be a directory OR a .tar.gz/.tgz "
            "file."
        ),
    )
    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help=(
            "Source path (defaults to the active venv/conda env). May "
            "be a directory OR a .tar.gz/.tgz file — in the latter "
            "case the tarball is copied to /tmp/ and extracted there, "
            "skipping the create step that --compress does."
        ),
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=None,
        help="Destination path on worker nodes (defaults to /tmp/<env-name>/).",
    )
    parser.add_argument(
        "--hostfile",
        type=str,
        default=None,
        help="Hostfile to read node list from (auto-detected from scheduler when omitted).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help=(
            "Use 'cp -a' instead of rsync for the local copy "
            "(Lustre → /tmp/). Faster for initial copies of large "
            "environments with many small files. Remote node "
            "distribution still uses rsync."
        ),
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help=(
            "Create a .tar.gz archive, copy it to /tmp/, then extract. "
            "Reduces Lustre I/O from millions of small-file reads to "
            "one sequential read. Remote distribution still uses rsync."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without doing it.",
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        # Tolerate one stray "yeet-env"/"yeet" token leaked from old
        # entry points; reject anything else.
        leftover = [a for a in unknown if a not in ("yeet", "yeet-env")]
        if leftover:
            parser.error(
                f"unrecognized arguments: {' '.join(leftover)}"
            )
    # Mutex: positional SRC and --src can't both be set.
    if args.src_positional is not None and args.src is not None:
        parser.error(
            "--src and positional SRC are mutually exclusive; "
            "pick one"
        )
    if args.src is None:
        args.src = args.src_positional
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
        # Convenience hint: if a fresher same-named tarball exists
        # next to the env or in cwd, point it out — tarball broadcast
        # is ~10× faster at scale than per-file rsync. Don't auto-pick
        # it; explicit is safer (the tarball might be stale).
        _suggest_tarball_if_present(src)

    # If --src is a .tar.gz / .tgz file, treat it as a pre-built
    # archive: skip the "tar create" step and just copy + extract.
    src_is_tarball = src.is_file() and (
        src.name.endswith(".tar.gz") or src.name.endswith(".tgz")
    )
    if src_is_tarball:
        # Strip .tar.gz / .tgz suffix to derive the destination name
        env_name = src.name
        for suffix in (".tar.gz", ".tgz"):
            if env_name.endswith(suffix):
                env_name = env_name[: -len(suffix)]
                break
    else:
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
        logger.error("No worker nodes found.")
        return 1

    # Copy locally first (current node), then rsync to remote nodes.
    # The local copy is needed because /tmp is node-local.
    needs_local_copy = not str(src).startswith("/tmp")
    # Filter out the current node — also handle the HSN variant
    # (current node may appear as "node01" while nodes contain "node01-hsn0").
    current_variants = {current, current + "-hsn0", current.removesuffix("-hsn0")}
    remote_nodes = [n for n in nodes if n not in current_variants]

    # ── Print summary ───────────────────────────────────────────────
    env_size = _get_env_size(src)
    total_nodes = (1 if needs_local_copy else 0) + len(remote_nodes)
    logger.info("Source: %s (%s)", src, env_size)
    logger.info("Target: %s/ on %d node(s)", dst, total_nodes)
    if needs_local_copy:
        logger.info("  local:  %s (rsync to %s/)", current, dst)
    if remote_nodes:
        if len(remote_nodes) <= 6:
            logger.info("  remote: %s", ", ".join(remote_nodes))
        else:
            shown = ', '.join(remote_nodes[:3])
            logger.info("  remote: %s, ... (%d nodes)", shown, len(remote_nodes))
    if args.dry_run:
        logger.info("[dry-run] No files transferred.")
        return 0

    if total_nodes == 0:
        logger.info("Nothing to sync (source is already in %s).", dst)
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

    logger.info("Syncing (%d nodes)...", total)
    t0 = time.perf_counter()

    # Step 1: copy source to local /tmp/ and patch paths ONCE.
    # All subsequent rsyncs distribute the already-patched copy.
    if needs_local_copy:
        _local_t0 = time.perf_counter()

        def _spinner(label: str) -> None:
            """Reusable spinner that shows label + elapsed time."""
            if not _IS_TTY:
                return
            elapsed = time.perf_counter() - _local_t0
            frames = _AggregateProgress._FRAMES
            idx = int(elapsed * 2) % len(frames)
            sys.stdout.write(f"\r\033[K    {frames[idx]} {label}  [{elapsed:.0f}s]")
            sys.stdout.flush()

        if src_is_tarball:
            # Source is already a .tar.gz/.tgz: copy it to /tmp/ and
            # extract there. Skips the "tar create" step that --compress
            # does. Useful when you already have a pre-built tarball
            # (e.g. from `ezpz tar-env`) on a shared filesystem.
            method = "tar.gz (pre-built)"
            local_tarball = Path("/tmp") / src.name
            print()
            try:
                tb_size_gb = src.stat().st_size / (1024**3)
            except OSError:
                tb_size_gb = 0.0

            # Step 1: copy tarball Lustre → /tmp/
            _spinner(f"copying {src.name} ({tb_size_gb:.1f}G) → /tmp/")
            cp_proc = subprocess.Popen(
                ["cp", str(src), str(local_tarball)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            while cp_proc.poll() is None:
                _spinner(f"copying {src.name} ({tb_size_gb:.1f}G) → /tmp/")
                time.sleep(0.5)
            if cp_proc.returncode != 0:
                stderr = (cp_proc.stderr.read() or b"").decode()
                logger.warning("cp tarball failed: %s", stderr.strip())
                local_elapsed = time.perf_counter() - _local_t0
                local_rc = cp_proc.returncode or 1
                # Partial copy may have left a truncated tarball behind
                _cleanup_path(local_tarball)
            else:
                # Step 2: extract into dst
                if dst.exists() and not _safe_rmtree(dst):
                    local_elapsed = time.perf_counter() - _local_t0
                    local_rc = 1
                    # We never started extracting, but the local
                    # tarball still needs cleanup.
                    _cleanup_path(local_tarball)
                else:
                    dst.mkdir(parents=True, exist_ok=True)
                    _spinner(f"extracting {local_tarball.name} → {dst}/")
                    tar_extract = subprocess.Popen(
                        ["tar", "-xzf", str(local_tarball),
                         "--strip-components=1", "-C", str(dst)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    while tar_extract.poll() is None:
                        _spinner(f"extracting {local_tarball.name} → {dst}/")
                        time.sleep(0.5)
                    local_elapsed = time.perf_counter() - _local_t0
                    local_rc = tar_extract.returncode or 0
                    if local_rc != 0:
                        stderr = (tar_extract.stderr.read() or b"").decode()
                        logger.warning("tar extract failed: %s", stderr.strip())
                        # Half-extracted dst is unusable — drop it.
                        # We just wrote it ourselves, so skip the
                        # /tmp-only safety guard.
                        _remove_partial_dst(dst)

                    # Always clean up the local tarball copy whether
                    # extraction succeeded or failed.
                    _cleanup_path(local_tarball)
            if _IS_TTY:
                sys.stdout.write("\r\033[K")

        elif args.compress:
            # tar.gz: compress on Lustre (sequential write), copy one
            # file to /tmp/ (sequential read), extract locally.
            # Much less Lustre metadata pressure than per-file rsync/cp.
            method = "tar.gz"
            tarball = Path(f"/tmp/{env_name}.tar.gz")
            print()

            # Step 1: create archive from source on Lustre
            _spinner(f"tar -czf {tarball.name} (compressing)")
            tar_create = subprocess.Popen(
                [
                    "tar", "-czf", str(tarball),
                    "--exclude=__pycache__",
                    "-C", str(src.parent), src.name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            while tar_create.poll() is None:
                _spinner(f"tar -czf {tarball.name} (compressing)")
                time.sleep(0.5)
            if tar_create.returncode != 0:
                stderr = (tar_create.stderr.read() or b"").decode()
                logger.warning("tar create failed: %s", stderr.strip())
                local_elapsed = time.perf_counter() - _local_t0
                local_rc = tar_create.returncode or 1
                # Partial archive: don't ship a corrupt tarball.
                _cleanup_path(tarball)
            else:
                # Show tarball size
                try:
                    tb_size = tarball.stat().st_size / (1024**3)
                    _spinner(f"tar.gz: {tb_size:.1f}G")
                except OSError:
                    pass

                # Step 2: extract into /tmp/
                if dst.exists() and not _safe_rmtree(dst):
                    local_elapsed = time.perf_counter() - _local_t0
                    local_rc = 1
                    _cleanup_path(tarball)
                else:
                    dst.mkdir(parents=True, exist_ok=True)
                    _spinner(f"extracting {tarball.name} → {dst}/")
                    tar_extract = subprocess.Popen(
                        ["tar", "-xzf", str(tarball),
                         "--strip-components=1", "-C", str(dst)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    while tar_extract.poll() is None:
                        _spinner(f"extracting {tarball.name} → /tmp/")
                        time.sleep(0.5)
                    local_elapsed = time.perf_counter() - _local_t0
                    local_rc = tar_extract.returncode or 0
                    if local_rc != 0:
                        stderr = (tar_extract.stderr.read() or b"").decode()
                        logger.warning("tar extract failed: %s", stderr.strip())
                        # Half-extracted dst is unusable — drop it
                        _remove_partial_dst(dst)
                    _cleanup_path(tarball)
            if _IS_TTY:
                sys.stdout.write("\r\033[K")

        elif args.copy:
            # cp -a: faster than rsync for large venvs on parallel
            # filesystems (sequential directory walk vs per-file stat).
            method = "cp"
            print()
            _spinner("cp -a → /tmp/")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and not _safe_rmtree(dst):
                local_elapsed = time.perf_counter() - _local_t0
                local_rc = 1
            else:
                cp_proc = subprocess.Popen(
                    ["cp", "-a", str(src), str(dst)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                while cp_proc.poll() is None:
                    _spinner("cp -a → /tmp/")
                    time.sleep(0.5)
                local_elapsed = time.perf_counter() - _local_t0
                local_rc = cp_proc.returncode or 0
                stderr = (cp_proc.stderr.read() or b"").decode()
                if local_rc != 0:
                    logger.warning("cp failed (exit %d): %s", local_rc, stderr.strip())
                    # Half-copied dst is unusable — drop it
                    _remove_partial_dst(dst)
            if _IS_TTY:
                sys.stdout.write("\r\033[K")

        else:
            method = "rsync"
            def _local_progress(pct: str = "", speed: str = "", eta: str = "") -> None:
                if not _IS_TTY:
                    return
                elapsed = time.perf_counter() - _local_t0
                parts = ["Copying to local /tmp/"]
                if pct:
                    parts.append(pct)
                if speed:
                    parts.append(speed)
                parts.append(f"[{elapsed:.0f}s]")
                msg = "    " + "  ".join(parts)
                sys.stdout.write(f"\r\033[K{msg}")
                sys.stdout.flush()
            print()
            _local_progress()
            _, local_elapsed, local_rc = _rsync_to_node(
                src, dst, current, local=True,
                progress_callback=_local_progress,
            )
            if _IS_TTY:
                sys.stdout.write("\r\033[K")
        if local_rc == 0:
            # Patching only applies to venvs \u2014 it's a no-op for other
            # sources, but skip explicitly to avoid noise.
            if (dst / "bin" / "activate").exists():
                _patch_venv_paths_local(dst, src)
            print(f"    \u2713 {current} (local, {method}) \u2014 {local_elapsed:.1f}s")
            results.append((current, local_elapsed, local_rc))
        else:
            print(f"    \u2717 {current} (local, {method}) \u2014 FAILED")
            results.append((current, local_elapsed, local_rc))
            # No valid local copy — abort, don't distribute a broken env
            logger.error("Local copy failed — aborting distribution.")
            return 1

    remaining = [n for n in all_nodes if n != current]

    # Track per-source active rsync count to enforce MAX_PER_SOURCE.
    source_active: dict[str, int] = {current: 0}
    source_lock = threading.Lock()
    # Function-scoped RNG so we don't disturb the global random state
    # of any caller that has seeded it for their own reasons.
    pick_rng = random.Random()

    def _submit_work(pool: ThreadPoolExecutor, futures: dict) -> None:  # type: ignore[type-arg]
        """Submit as many rsyncs as sources allow."""
        while remaining:
            with source_lock:
                src_node = pick_source(
                    source_active, MAX_PER_SOURCE, rng=pick_rng,
                )
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
        logger.warning("%d/%d node(s) failed!", failed, total_nodes)
    else:
        logger.info("Done in %.1fs", total_elapsed)

    # ── Guidance ────────────────────────────────────────────────────
    # Detect from `dst` so tarball sources (where the directory only
    # exists after extraction) are classified correctly.
    is_venv = (dst / "bin" / "activate").exists()
    is_conda = (dst / "conda-meta").is_dir()
    has_bin = (dst / "bin").is_dir()
    print()
    if is_venv:
        print(f"  To use this environment:")
        print(f"    deactivate 2>/dev/null")
        print(f"    source {dst}/bin/activate")
        print()
        print(f"  Then launch your training (from a shared filesystem path):")
        print(f"    cd /path/to/your/project")
        print(f"    ezpz launch python3 -m your_app.train")
        print()
        print(f"  Note: /tmp is node-local. Make sure your working directory")
        print(f"  is on a shared filesystem (e.g. Lustre) before launching,")
        print(f"  so all ranks can access data and outputs.")
    elif is_conda:
        print(f"  To use this environment:")
        print(f"    conda deactivate")
        print(f"    conda activate {dst}")
        print()
        print(f"  Then launch your training (from a shared filesystem path):")
        print(f"    cd /path/to/your/project")
        print(f"    ezpz launch python3 -m your_app.train")
        print()
        print(f"  Note: /tmp is node-local. Make sure your working directory")
        print(f"  is on a shared filesystem (e.g. Lustre) before launching,")
        print(f"  so all ranks can access data and outputs.")
    else:
        successful = len(results) - failed
        print(f"  Synced to {dst}/ on {successful} node(s).")
        if has_bin:
            print(f"    (looks like a tool directory — add to PATH if needed:")
            print(f"     export PATH={dst}/bin:$PATH)")
        print()
        print(f"  Note: /tmp is node-local. Reference the synced path on each")
        print(f"  worker (e.g. {dst}) — the shared-filesystem source path will")
        print(f"  not see writes from worker nodes.")

    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    if Path(sys.argv[0]).name == "ezpz-yeet-env":
        print(
            "ezpz-yeet-env is deprecated; use 'ezpz yeet' as a drop-in "
            "replacement",
            file=sys.stderr,
        )
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
