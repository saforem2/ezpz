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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_COMPLETE_MARKER = ".complete"
_META_FILE = "meta.json"
_STEP_PREFIX = "step-"


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
    :func:`drain` then fans the shards out to the durable ``final_path`` on
    shared FS and stamps the completion marker there. Until drained, the
    durable checkpoint does NOT exist / is not marked complete.
    """

    future: Any
    step: int
    meta: Optional[dict]
    stage_path: Path
    final_path: Path
    drained: bool = False


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
    # dcp.async_save stages/writes from a background CPU thread and needs a PG
    # with a CPU (gloo) backend. On XPU/CUDA the default PG is xccl/nccl-only,
    # so we pass a dedicated gloo PG. new_group() is collective — resolved
    # (cached) on ALL ranks below, before the rank-0-only work.
    cpu_pg = _cpu_capable_group()
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
        process_group=cpu_pg,
    )
    return PendingCheckpoint(
        future=future,
        step=step,
        meta=meta,
        stage_path=stage_out,
        final_path=final_out,
    )


def drain(pending: Optional[PendingCheckpoint]) -> Optional[Path]:
    """Finish an async checkpoint: block on the staged write, fan out, mark.

    Steps (per rank): wait for the background DCP write to ``stage_path`` to
    finish, copy this rank's shard files ``stage_path/* -> final_path/`` on
    shared FS (rank 0 also copies ``.metadata``), barrier, then rank 0 writes
    ``meta.json`` + the ``.complete`` marker LAST. Idempotent / None-safe.

    Returns the durable ``final_path`` (or None if nothing to drain).
    """
    if pending is None or pending.drained:
        return None if pending is None else pending.final_path

    # Block until the node-local staged write completed.
    pending.future.result()

    is_rank0 = _is_rank0()
    # Fan out: each rank copies its own shard files to the durable dir. Shard
    # files are named per-rank (__<rank>_<n>.distcp) so ranks don't collide;
    # the shared .metadata is copied by rank 0 only.
    pending.final_path.mkdir(parents=True, exist_ok=True)
    for src in sorted(pending.stage_path.glob("*.distcp")):
        # A rank only copies files it produced. DCP names them __<rank>_..;
        # copying all *.distcp present on THIS node is safe because each node
        # only holds its own ranks' shards in stage_dir. Use copy2 to a temp
        # name + rename for atomicity within the shared FS.
        dst = pending.final_path / src.name
        tmp = dst.with_suffix(dst.suffix + ".partial")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    if is_rank0:
        meta_src = pending.stage_path / ".metadata"
        if meta_src.exists():
            dst = pending.final_path / ".metadata"
            tmp = dst.with_suffix(".partial")
            shutil.copy2(meta_src, tmp)
            os.replace(tmp, dst)
    _barrier()  # all shards fanned out before the marker

    if is_rank0:
        _write_meta_and_marker(pending.final_path, pending.step, pending.meta)
    _barrier()

    # Reclaim the node-local staging copy — it's transient.
    try:
        shutil.rmtree(pending.stage_path, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass

    pending.drained = True
    if is_rank0:
        logger.info("saved checkpoint (async): %s", pending.final_path)
    return pending.final_path


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


_CPU_PG = None  # cached gloo group for async_save (created once, collectively)
_CPU_PG_RESOLVED = False


def _cpu_capable_group():
    """Return a process group whose backend supports CPU tensors, or None.

    ``dcp.async_save`` requires the PG to include a CPU (gloo) backend for its
    background-thread collective. If the default PG already has one (a
    gloo-only or ``cpu:gloo,<accel>`` init), return None so async_save uses the
    default. Otherwise (XPU/CUDA-only default PG, e.g. xccl/nccl) create a
    dedicated gloo group ONCE and cache it.

    MUST be called on every rank (``new_group`` is collective). Cached so the
    per-step save path doesn't re-collective.
    """
    global _CPU_PG, _CPU_PG_RESOLVED
    if _CPU_PG_RESOLVED:
        return _CPU_PG
    try:
        import torch
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            _CPU_PG_RESOLVED = True
            return None  # single process — no PG needed
        default = dist.group.WORLD
        # If the default PG already speaks CPU, use it (no extra group).
        if torch.device("cpu") in getattr(default, "_device_types", set()):
            _CPU_PG_RESOLVED = True
            return None
        # Accel-only default PG: build a gloo group over all ranks.
        _CPU_PG = dist.new_group(backend="gloo")
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not create gloo PG for async save: %s", exc)
        _CPU_PG = None
    _CPU_PG_RESOLVED = True
    return _CPU_PG


def _is_rank0() -> bool:
    """True on global rank 0, or when not running distributed."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:  # noqa: BLE001
        pass
    return True


def _barrier() -> None:
    """Best-effort collective barrier; no-op when not distributed."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:  # noqa: BLE001 — barrier is best-effort
        pass
