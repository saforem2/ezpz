"""Distributed (DCP) checkpoint save/load for the FSDP2 examples.

Thin, FSDP2/DTensor-aware wrappers around ``torch.distributed.checkpoint``
(DCP). Each rank writes/reads its own shard in parallel, so this scales to
the same model sizes the example itself does (no gather-to-rank-0 that would
OOM at large scale).

Layout on disk::

    <ckpt_dir>/
      step-10/
        __0_0.distcp  ...        # DCP shard files (one set per rank)
        meta.json                # {step, tokens_seen, epoch, batch_offset}
        .complete                # written LAST — marks a valid checkpoint
      step-20/
        ...

The ``.complete`` marker is the crash-safety mechanism: a checkpoint whose
save was interrupted (the very failure mode this supports measuring!) has no
marker, so :func:`latest_checkpoint` skips it and resume falls back to the
previous good step.

Design notes:
  * We checkpoint the SHARDED state via ``get_state_dict`` /
    ``set_state_dict`` on ``(model, optimizer)`` — DTensor-aware, so it works
    identically under FSDP-only, HSDP, and 2D FSDP+TP.
  * ``meta.json`` is plain JSON (rank 0 only) — the training bookkeeping
    (step/token/epoch/offset) a resume needs to continue, kept out of the
    tensor state dict so it's trivially readable without DCP.
  * Resume at a DIFFERENT parallelism (reshard) is a DCP capability but is
    NOT a goal here; same world size / mesh is assumed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_COMPLETE_MARKER = ".complete"
_META_FILE = "meta.json"
_STEP_PREFIX = "step-"

# Single-worker pool for backgrounding the /tmp -> shared-FS fan-out copy (see
# start_fanout). One worker is enough: only one async checkpoint is in flight
# at a time (DCP constraint), so copies never need to overlap each other, and a
# single thread keeps the copy strictly ordered after its DCP write.
_FANOUT_POOL: Optional[ThreadPoolExecutor] = None


def _fanout_pool() -> ThreadPoolExecutor:
    global _FANOUT_POOL
    if _FANOUT_POOL is None:
        _FANOUT_POOL = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ezpz-ckpt-fanout"
        )
    return _FANOUT_POOL


def _step_dir(ckpt_dir: os.PathLike | str, step: int) -> Path:
    return Path(ckpt_dir) / f"{_STEP_PREFIX}{step}"


def _clear_stale_marker(out: Path) -> None:
    """rank-0: create the step dir and remove any pre-existing .complete.

    A leftover marker from a prior run into the same dir would make a
    mid-overwrite checkpoint look valid; clearing it first means a crash
    during the (re)write is correctly skipped by latest_checkpoint().
    """
    out.mkdir(parents=True, exist_ok=True)
    marker = out / _COMPLETE_MARKER
    if marker.exists():
        marker.unlink()


def _write_meta_and_marker(out: Path, step: int, meta: Optional[dict]) -> None:
    """rank-0: write meta.json, then the .complete marker LAST.

    The marker's presence is the "complete + safe to resume" contract that
    latest_checkpoint() relies on, so it must be the final write.
    """
    payload = dict(meta or {})
    payload["step"] = step
    (out / _META_FILE).write_text(json.dumps(payload, indent=2))
    (out / _COMPLETE_MARKER).write_text("")


def latest_checkpoint(ckpt_dir: os.PathLike | str) -> Optional[Path]:
    """Return the newest COMPLETE ``step-<N>`` checkpoint dir, or None.

    "Newest" is by numeric step (not lexical, so ``step-10`` beats
    ``step-9``). A ``step-<N>`` dir without a ``.complete`` marker is
    ignored — it was interrupted mid-save and is not safe to load.
    """
    root = Path(ckpt_dir)
    if not root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for p in root.iterdir():
        if not p.is_dir() or not p.name.startswith(_STEP_PREFIX):
            continue
        if not (p / _COMPLETE_MARKER).exists():
            continue
        try:
            step = int(p.name[len(_STEP_PREFIX):])
        except ValueError:
            continue
        candidates.append((step, p))
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


def save_checkpoint(
    ckpt_dir: os.PathLike | str,
    step: int,
    model: Any,
    optimizer: Any,
    *,
    meta: Optional[dict[str, Any]] = None,
) -> Path:
    """Save a sharded DCP checkpoint of ``(model, optimizer)`` at ``step``.

    All ranks participate (DCP writes shards in parallel). Rank 0 writes
    ``meta.json`` and, LAST, the ``.complete`` marker. A final barrier keeps
    ranks in lockstep so the marker isn't observed before every shard landed.

    Returns the ``step-<N>`` directory path.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    out = _step_dir(ckpt_dir, step)
    is_rank0 = _is_rank0()
    if is_rank0:
        _clear_stale_marker(out)
    _barrier()  # ensure the dir exists (and marker cleared) before any writes

    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.save(
        {"model": model_sd, "optim": optim_sd},
        checkpoint_id=str(out),
    )

    if is_rank0:
        _write_meta_and_marker(out, step, meta)
    _barrier()
    if is_rank0:  # gate per-rank log spam (AGENTS.md)
        logger.info("saved checkpoint: %s", out)
    return out


@dataclass
class PendingCheckpoint:
    """Handle for an in-flight async checkpoint (see :func:`save_checkpoint_async`).

    The staged shards are being written to ``stage_path`` (node-local) by a
    DCP background thread; ``future`` resolves when that disk write finishes.
    The fan-out to the durable ``final_path`` on shared FS then happens in two
    phases (see :func:`start_fanout` / :func:`finalize_fanout`): a background,
    collective-free per-rank shard copy, then a main-thread barrier + marker.
    Until finalized, the durable checkpoint does NOT exist / is not marked
    complete.

    ``fanout_future`` holds the background copy (None until start_fanout runs).
    """

    future: Any
    step: int
    meta: Optional[dict[str, Any]]
    stage_path: Path
    final_path: Path
    drained: bool = False
    fanout_future: Optional[Future] = None


def save_checkpoint_async(
    ckpt_dir: os.PathLike | str,
    stage_dir: os.PathLike | str,
    step: int,
    model: Any,
    optimizer: Any,
    *,
    meta: Optional[dict[str, Any]] = None,
) -> PendingCheckpoint:
    """Start an async checkpoint: stage to node-local ``stage_dir``, return.

    ``dcp.async_save`` stages the state dict to CPU memory SYNCHRONOUSLY (so
    it is safe for the caller to keep training / mutating params right after
    this returns), then writes the shards to ``stage_dir/step-N`` in a
    background thread. The returned :class:`PendingCheckpoint` must later be
    passed to :func:`drain`, which blocks on the write and fans the shards out
    to the DURABLE ``ckpt_dir`` on shared FS.

    IMPORTANT: node-local ``stage_dir`` (e.g. ``/tmp``) is NOT resumable on
    its own — the shards are scattered per node. Only the fanned-out
    ``ckpt_dir`` copy (written by :func:`drain`) is durable + resumable. Never
    point ``latest_checkpoint`` at ``stage_dir``.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    stage_out = _step_dir(stage_dir, step)
    final_out = _step_dir(ckpt_dir, step)
    is_rank0 = _is_rank0()
    if is_rank0:
        # Clear the DURABLE marker up front (same crash-safety reasoning as
        # sync): if we die before drain() re-stamps it, the half-fanned-out
        # dir must not look complete.
        _clear_stale_marker(final_out)
    # Every rank owns its stage dir (each writes its own shards there).
    stage_out.mkdir(parents=True, exist_ok=True)
    _barrier()

    model_sd, optim_sd = get_state_dict(model, optimizer)
    future = dcp.async_save(
        {"model": model_sd, "optim": optim_sd},
        checkpoint_id=str(stage_out),
    )
    return PendingCheckpoint(
        future=future,
        step=step,
        meta=meta,
        stage_path=stage_out,
        final_path=final_out,
    )


def _copy_my_shards(pending: PendingCheckpoint) -> None:
    """Per-rank, COLLECTIVE-FREE: wait the DCP write, copy this rank's shards.

    This is the expensive half of the fan-out (the full /tmp -> shared-FS copy
    of one rank's shards), and it contains NO collectives — only ``future``
    wait + local file I/O. That is exactly what makes it safe to run on a
    background thread (:func:`start_fanout`) while the main thread keeps issuing
    training collectives on the process group: a barrier here would cross-match
    the main thread's all-reduce and deadlock, so there is none.

    Each rank copies ONLY the shard files IT wrote (DCP names them
    ``__<rank>_<n>.distcp``); rank 0 also copies the shared ``.metadata``.
    Temp names are per-rank-unique so ranks never os.replace each other's
    partials on the shared dir (FileNotFoundError, seen on job 12471711).
    """
    pending.future.result()  # block until node-local staged write finished
    my_rank = _global_rank()
    pending.final_path.mkdir(parents=True, exist_ok=True)
    for src in sorted(pending.stage_path.glob(f"__{my_rank}_*.distcp")):
        dst = pending.final_path / src.name
        tmp = pending.final_path / f"{src.name}.r{my_rank}.partial"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    if my_rank == 0:
        meta_src = pending.stage_path / ".metadata"
        if meta_src.exists():
            dst = pending.final_path / ".metadata"
            tmp = pending.final_path / ".metadata.r0.partial"
            shutil.copy2(meta_src, tmp)
            os.replace(tmp, dst)


def start_fanout(pending: Optional[PendingCheckpoint]) -> None:
    """Kick the /tmp -> shared-FS fan-out onto a background thread (non-blocking).

    Returns immediately: the per-rank shard copy (:func:`_copy_my_shards`) runs
    on the fan-out pool so the expensive write overlaps subsequent training.
    Pair with :func:`finalize_fanout` at the next save boundary, which joins
    the copy and does the collective barrier + completion marker on the MAIN
    thread. Idempotent / None-safe.
    """
    if pending is None or pending.drained or pending.fanout_future is not None:
        return
    pending.fanout_future = _fanout_pool().submit(_copy_my_shards, pending)


def finalize_fanout(pending: Optional[PendingCheckpoint]) -> Optional[Path]:
    """Main-thread completion of a backgrounded fan-out: join, barrier, mark.

    Joins the background shard copy started by :func:`start_fanout` (starting
    it inline first if it never ran), then does the COLLECTIVE part on the main
    thread in lockstep across ranks: a barrier so all shards have landed, rank
    0 writes ``meta.json`` + the ``.complete`` marker LAST, a second barrier,
    then reclaims the node-local staging copy. Must be called by ALL ranks (it
    barriers). Idempotent / None-safe.

    Because the durable marker is only stamped here, the checkpoint becomes
    resumable exactly at this point — one save interval after the save that
    produced it. The PREVIOUS complete checkpoint stays durable throughout, so
    a crash in the window still loses at most one interval (same guarantee as
    the old inline drain).
    """
    if pending is None or pending.drained:
        return None if pending is None else pending.final_path

    # Ensure the copy ran, then join it (surfaces any exception here, on the
    # main thread, rather than swallowing it on the pool).
    if pending.fanout_future is None:
        _copy_my_shards(pending)
    else:
        pending.fanout_future.result()

    _barrier()  # all ranks' shards fanned out before the marker
    if _global_rank() == 0:
        _write_meta_and_marker(pending.final_path, pending.step, pending.meta)
    _barrier()

    # Reclaim the node-local staging copy — it's transient.
    try:
        shutil.rmtree(pending.stage_path, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass

    pending.drained = True
    if _global_rank() == 0:
        logger.info("saved checkpoint (async): %s", pending.final_path)
    return pending.final_path


def drain(pending: Optional[PendingCheckpoint]) -> Optional[Path]:
    """Synchronous fan-out: copy shards, barrier, mark — all inline.

    Equivalent to ``start_fanout`` immediately followed by ``finalize_fanout``,
    but without backgrounding: the full copy blocks the caller. Kept for the
    run-exit path (where there is nothing left to overlap) and as a simple,
    single-call option. Must be called by ALL ranks (it barriers).
    Idempotent / None-safe.
    """
    if pending is None or pending.drained:
        return None if pending is None else pending.final_path
    # If a background copy is already in flight, finalize_fanout joins it;
    # otherwise finalize_fanout runs the copy inline. Either way it does the
    # collective barrier + marker.
    return finalize_fanout(pending)


def load_checkpoint(
    ckpt_dir: os.PathLike | str,
    model: Any,
    optimizer: Any,
) -> Optional[dict[str, Any]]:
    """Load the latest complete checkpoint into ``(model, optimizer)``.

    Returns the ``meta.json`` payload (``{step, tokens_seen, epoch,
    batch_offset, ...}``) on success, or None when no complete checkpoint
    exists (a fresh run — caller starts from scratch).

    Mutates ``model`` and ``optimizer`` in place via ``set_state_dict``.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_state_dict,
        set_state_dict,
    )

    latest = latest_checkpoint(ckpt_dir)
    if latest is None:
        return None

    # get_state_dict first to shape the containers DCP loads INTO (it fills
    # them in place with the right DTensor layouts for this rank).
    model_sd, optim_sd = get_state_dict(model, optimizer)
    state = {"model": model_sd, "optim": optim_sd}
    dcp.load(state, checkpoint_id=str(latest))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optim"],
    )

    meta_path = latest / _META_FILE
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (ValueError, OSError):
            if _is_rank0():
                logger.warning(
                    "could not parse %s; resuming with step only", meta_path
                )
    meta.setdefault("step", int(latest.name[len(_STEP_PREFIX):]))
    if _is_rank0():  # gate per-rank log spam (AGENTS.md)
        logger.info(
            "resumed from checkpoint: %s (step=%s)", latest, meta.get("step")
        )
    return meta


def shutdown_fanout_pool() -> None:
    """Join the fan-out worker thread (call once at run exit).

    Safe to call when no pool was ever created. After the final
    :func:`finalize_fanout`/:func:`drain` there is no in-flight copy, so this
    just tears the idle worker down cleanly.
    """
    global _FANOUT_POOL
    if _FANOUT_POOL is not None:
        _FANOUT_POOL.shutdown(wait=True)
        _FANOUT_POOL = None


def _global_rank() -> int:
    """This process's global rank, or 0 when not running distributed."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:  # noqa: BLE001
        pass
    return 0


def _is_rank0() -> bool:
    """True on global rank 0, or when not running distributed."""
    return _global_rank() == 0


def _barrier() -> None:
    """Best-effort collective barrier; no-op when not distributed."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:  # noqa: BLE001 — barrier is best-effort
        pass
