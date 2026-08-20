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
    state_dict_options: Any = None,
) -> Path:
    """Save a sharded DCP checkpoint of ``(model, optimizer)`` at ``step``.

    All ranks participate (DCP writes shards in parallel). Rank 0 writes
    ``meta.json`` and, LAST, the ``.complete`` marker. A final barrier keeps
    ranks in lockstep so the marker isn't observed before every shard landed.

    Args:
        state_dict_options: optional ``StateDictOptions`` forwarded to
            ``get_state_dict``. Pass
            ``StateDictOptions(ignore_frozen_params=True)`` for an
            adapter-only (LoRA) checkpoint, which is ~``rank/dim`` the
            size of a full one. ``None`` (the default) preserves the
            existing full-state behavior exactly.

            NOTE: a checkpoint written with this set contains no frozen
            base weights, so it is NOT a standalone model. Pass the same
            option to :func:`load_checkpoint`, and keep the base weights
            available separately.

    Returns the ``step-<N>`` directory path.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    out = _step_dir(ckpt_dir, step)
    is_rank0 = _is_rank0()
    if is_rank0:
        _clear_stale_marker(out)
    _barrier()  # ensure the dir exists (and marker cleared) before any writes

    model_sd, optim_sd = (
        get_state_dict(model, optimizer, options=state_dict_options)
        if state_dict_options is not None
        else get_state_dict(model, optimizer)
    )
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
    Pair with :func:`try_finalize_if_ready` (called each step) to finalize as
    soon as every rank's copy is done, or :func:`finalize_fanout` to force it.
    Idempotent / None-safe.
    """
    if pending is None or pending.drained or pending.fanout_future is not None:
        return
    pending.fanout_future = _fanout_pool().submit(_copy_my_shards, pending)


def _abort_if_any_rank_failed(
    my_err: "Optional[BaseException]", pending: PendingCheckpoint
) -> None:
    """Collectively abort the fan-out if ANY rank's shard copy failed.

    Every rank must call this after attempting its copy, passing its own
    exception (or None). We all-reduce the failure count over the torch PG (see
    :func:`_allreduce_sum_int`) so all ranks agree, then raise together BEFORE
    the finalize barrier — otherwise a rank that raised on its own would leave
    the healthy ranks hanging at the barrier. When not distributed (single
    process, no PG) the count is this rank's own, so it re-raises locally.
    """
    failed = 1 if my_err is not None else 0
    n_failed = _allreduce_sum_int(failed)
    if n_failed > 0:
        if my_err is not None:
            logger.error(
                "async checkpoint fan-out failed on this rank: %s", my_err
            )
        raise RuntimeError(
            "async checkpoint fan-out failed on at least one rank; aborting "
            "(the previous complete checkpoint remains durable for resume)"
        )


def _all_ranks_copy_done(pending: PendingCheckpoint) -> bool:
    """Collective, DEADLOCK-SAFE probe: has EVERY rank's background copy finished
    SUCCESSFULLY? Returns True only when all ranks' copies are done and none
    failed; returns False while any rank is still copying. Raises (identically
    on every rank) if any rank's background copy raised.

    Coordinates via a torch all-reduce (:func:`_allreduce_sum_int`) on the MAIN
    thread, in lockstep across ranks — the same footing as the ``_barrier``
    calls already in the finalize path, so there is no cross-thread hazard (the
    cross-thread danger only applies to the background copy, which issues no
    collectives). Callers must invoke this on all ranks in lockstep; the
    training loop's guard (``pending_ckpt`` is rank-uniform) ensures that.

    Failure handling is the subtle part:
      * ``Future.done()`` is True whether the copy finished OR raised, so we must
        check ``.exception()`` — otherwise a FAILED copy would vote "done" and
        finalize would stamp a marker (or, via ``.result()`` re-raising on one
        rank while others sit at the barrier, hang the job).
      * A failure is surfaced by raising on ALL ranks here (``n_failed`` is the
        same everywhere), BEFORE anyone enters ``finalize_fanout``'s barrier —
        a clean coordinated abort, not a one-rank-crashes-rest-hang deadlock.
        The previous complete checkpoint stays durable for resume.
      * If the MPI collective itself is unavailable we CANNOT make a per-step
        decision that's consistent across ranks (divergence → deadlock), so we
        report not-ready and defer to the save-boundary ``finalize_fanout`` (a
        hard all-rank barrier).
    """
    fut = pending.fanout_future
    if fut is None:
        done_ok, failed = 1, 0
    elif not fut.done():
        done_ok, failed = 0, 0
    elif fut.exception() is not None:  # done() True but the copy raised
        done_ok, failed = 0, 1
    else:
        done_ok, failed = 1, 0
    # Coordinate over the torch PG (works in any launch; no mpi4py needed). A
    # failure raises identically on ALL ranks before any barrier — coordinated
    # abort, not a one-rank-crash-rest-hang deadlock.
    n_failed = _allreduce_sum_int(failed)
    if n_failed > 0:
        if failed and fut is not None:
            logger.error(
                "async checkpoint fan-out failed on this rank: %s",
                fut.exception(),
            )
        raise RuntimeError(
            "async checkpoint fan-out failed on at least one rank; aborting "
            "(the previous complete checkpoint remains durable for resume)"
        )
    n_done = _allreduce_sum_int(done_ok)
    return n_done == _world_size()


def try_finalize_if_ready(
    pending: Optional[PendingCheckpoint],
) -> Optional[Path]:
    """Finalize the fan-out IFF every rank's background copy has finished.

    Call this once per step after :func:`start_fanout`. It cheaply probes
    (:func:`_all_ranks_copy_done`) whether all ranks' copies are done; only then
    does it run the collective :func:`finalize_fanout` (barrier + marker). This
    stamps the durable ``.complete`` marker ~as soon as the copy completes
    (~copy-duration after the save) rather than deferring a full save interval,
    which minimizes the window where a saved-but-not-yet-durable checkpoint
    forces resume from the PREVIOUS one. Returns the durable path when it
    finalized this call, else None. None-safe; no-op if already drained.

    MUST be called by ALL ranks every step (the probe is collective).
    """
    if pending is None or pending.drained:
        return None
    if not _all_ranks_copy_done(pending):
        return None
    return finalize_fanout(pending)


def finalize_fanout(pending: Optional[PendingCheckpoint]) -> Optional[Path]:
    """Main-thread completion of a backgrounded fan-out: join, barrier, mark.

    Joins the background shard copy started by :func:`start_fanout` (starting
    it inline first if it never ran), then does the COLLECTIVE part on the main
    thread in lockstep across ranks: a barrier so all shards have landed, rank
    0 writes ``meta.json`` + the ``.complete`` marker LAST, a second barrier,
    then reclaims the node-local staging copy. Must be called by ALL ranks (it
    barriers). Idempotent / None-safe.

    Durability tradeoff (IMPORTANT): the ``.complete`` marker for a checkpoint
    saved at step N is only stamped here, at the NEXT save boundary (step
    N+save_interval). So during the whole window ``[N, N+save_interval)`` the
    newest resumable checkpoint is N-save_interval, and an arbitrarily-timed
    crash (a real hardware failure, not synchronized to the marker write) can
    lose up to ~2 save intervals of work — vs ~1 interval for a synchronous
    save or the old inline drain, whose marker lands ~1 step after the save.
    This is the cost of overlapping the fan-out with training. The previous
    complete checkpoint is always durable, so recovery is never broken — only
    the worst-case amount of recomputed work grows by one interval. Shrink
    ``--save-interval`` to bound it.
    """
    if pending is None or pending.drained:
        return None if pending is None else pending.final_path

    # Ensure the copy ran, then join it. Capture (don't immediately re-raise) a
    # per-rank failure so we can decide COLLECTIVELY: if we let one rank raise
    # here while healthy ranks fall through to _barrier(), the healthy ranks
    # hang forever. Instead every rank votes, and all abort together (before the
    # barrier) if ANY rank's copy failed.
    my_err: Optional[BaseException] = None
    try:
        if pending.fanout_future is None:
            _copy_my_shards(pending)
        else:
            pending.fanout_future.result()
    except BaseException as exc:  # noqa: BLE001 — surfaced collectively below
        my_err = exc
    _abort_if_any_rank_failed(my_err, pending)

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
    state_dict_options: Any = None,
) -> Optional[dict[str, Any]]:
    """Load the latest complete checkpoint into ``(model, optimizer)``.

    Returns the ``meta.json`` payload (``{step, tokens_seen, epoch,
    batch_offset, ...}``) on success, or None when no complete checkpoint
    exists (a fresh run — caller starts from scratch).

    Mutates ``model`` and ``optimizer`` in place via ``set_state_dict``.

    Args:
        state_dict_options: optional ``StateDictOptions``, which MUST
            match the one used to save. An adapter-only checkpoint
            (``ignore_frozen_params=True``) contains no frozen base
            weights, so shaping the load container with the default
            full-state options asks DCP for keys the checkpoint does not
            have. Default ``None`` keeps every existing caller
            byte-identical.
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
    _kw = {} if state_dict_options is None else {"options": state_dict_options}
    model_sd, optim_sd = get_state_dict(model, optimizer, **_kw)
    state = {"model": model_sd, "optim": optim_sd}
    dcp.load(state, checkpoint_id=str(latest))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optim"],
        **_kw,
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


def _world_size() -> int:
    """Total rank count, or 1 when not running distributed."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:  # noqa: BLE001
        pass
    return 1


def _barrier() -> None:
    """Best-effort collective barrier; no-op when not distributed."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:  # noqa: BLE001 — barrier is best-effort
        pass


def _dist_active() -> bool:
    """True iff a torch.distributed process group is initialized."""
    try:
        import torch.distributed as dist

        return bool(dist.is_available() and dist.is_initialized())
    except Exception:  # noqa: BLE001
        return False


def _allreduce_sum_int(value: int) -> int:
    """Sum an int across ranks via the torch PG (SUM all-reduce).

    Uses ``torch.distributed`` — the same primitive as :func:`_barrier` — so it
    works in ANY distributed launch (gloo, xccl+gloo composite), with no mpi4py
    dependency. Runs on the CPU/gloo tensor so it's valid on the composite
    ``xpu:xccl,cpu:gloo`` backend even from the main thread. When not
    distributed (single process, no PG) returns ``value`` unchanged.

    MUST be called by all ranks in lockstep (it is collective). This is safe on
    the MAIN thread at a fixed per-step point — the same footing as the existing
    ``_barrier`` calls — the cross-thread hazard only applies to the background
    copy, which issues no collectives.
    """
    if not _dist_active():
        return value
    import torch
    import torch.distributed as dist

    t = torch.tensor([int(value)], dtype=torch.int64)  # CPU tensor → gloo path
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())
