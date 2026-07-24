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
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_COMPLETE_MARKER = ".complete"
_META_FILE = "meta.json"
_STEP_PREFIX = "step-"


def _step_dir(ckpt_dir: os.PathLike | str, step: int) -> Path:
    return Path(ckpt_dir) / f"{_STEP_PREFIX}{step}"


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
        out.mkdir(parents=True, exist_ok=True)
        # Clear any stale completion marker FIRST. If this step dir already
        # exists (e.g. re-running into the same --ckpt-dir), an old .complete
        # would make the checkpoint look valid mid-overwrite — so if the
        # process dies during this save, latest_checkpoint() must NOT trust
        # it. Removing the marker up front means a partial overwrite is
        # correctly skipped and resume falls back to the previous good step.
        marker = out / _COMPLETE_MARKER
        if marker.exists():
            marker.unlink()
    _barrier()  # ensure the dir exists (and marker cleared) before any writes

    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.save(
        {"model": model_sd, "optim": optim_sd},
        checkpoint_id=str(out),
    )

    if is_rank0:
        payload = dict(meta or {})
        payload["step"] = step
        (out / _META_FILE).write_text(json.dumps(payload, indent=2))
        # Marker LAST: its presence is the "this checkpoint is complete and
        # safe to resume from" contract that latest_checkpoint() relies on.
        (out / _COMPLETE_MARKER).write_text("")
    _barrier()
    if is_rank0:  # gate per-rank log spam (AGENTS.md)
        logger.info("saved checkpoint: %s", out)
    return out


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
