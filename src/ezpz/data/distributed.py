from __future__ import absolute_import, division, print_function, annotations
from typing import Any, Dict, Iterable, Iterator, Optional
import torch
import torch.distributed

# import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# --- usage notes -----------------------------------------------------------
# 1) You must create process groups ahead of time, e.g.:
#    - dp_group: the FSDP replica group (often world_size // tp_size)
#    - tp_group: the tensor-parallel group
#
# 2) In your training loop, call (per epoch):
#       if sampler is not None:
#           sampler.set_epoch(epoch)
#
# 3) If you enable `broadcast_within_tp=True`, only TP leader does I/O and
#    the batches are broadcast to TP peers; that reduces file-system pressure.
#    If you prefer simplicity over I/O savings, set it to False and let each
#    TP rank load the same samples (indices are identical within a DP group).

# --- helpers ---------------------------------------------------------------
#
assert hasattr(torch.distributed, "ProcessGroup")
assert hasattr(torch.distributed, "get_global_rank")
assert hasattr(torch.distributed, "broadcast_object_list")


def _rank_ws(
    pg: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[int, int]:
    """Return (rank, world_size) for a process group (or global default)."""
    if pg is None:
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return torch.distributed.get_rank(pg), torch.distributed.get_world_size(pg)


def _is_dist() -> bool:
    return (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )


def _tp_is_leader(tp_group: Optional[torch.distributed.ProcessGroup]) -> bool:
    if not _is_dist() or tp_group is None:
        return True
    tp_rank, _ = _rank_ws(tp_group)
    return tp_rank == 0


# def _broadcast_batch(batch: Any, tp_group: torch.distributed.ProcessGroup) -> Any:
#     """
#     Broadcast an arbitrary (nested) batch from TP leader to other TP ranks.
#     Uses torch.distributed.broadcast_object_list for generality.
#     For maximal perf, replace with a tensor-only path in your codebase.
#     """
#     obj_list = [batch]
#     torch.distributed.broadcast_object_list(obj_list, src=torch.distributed.get_rank(tp_group) - _rank_ws(tp_group)[0] if False else 0, group=tp_group)
#     # The line above is intentionally simple: src is TP leader (rank 0 within tp_group).
#     return obj_list[0]


def _broadcast_batch(
    batch: Any, tp_group: torch.distributed.ProcessGroup
) -> Any:
    """
    Broadcast an arbitrary (nested) batch from TP leader to other TP ranks.
    Uses torch.distributed.broadcast_object_list for generality.
    For maximal perf, replace with a tensor-only path in your codebase.
    """
    # -    obj_list = [batch]
    # -    torch.distributed.broadcast_object_list(obj_list, src=torch.distributed.get_rank(tp_group) - _rank_ws(tp_group)[0] if False else 0, group=tp_group)
    # -    # The line above is intentionally simple: src is TP leader (rank 0 within tp_group).
    # -    return obj_list[0]
    obj_list = [batch]
    # Pick TP-leader as group-rank 0, then map to its GLOBAL rank for src
    tp_leader_global = torch.distributed.get_global_rank(tp_group, 0)
    torch.distributed.broadcast_object_list(
        obj_list, src=tp_leader_global, group=tp_group
    )
    return obj_list[0]


# - def _broadcast_batch(batch: Any, tp_group: torch.distributed.ProcessGroup) -> Any:


class TPBroadcastDataLoader:
    """
    Wrapper that ensures only TP leader samples/loads, then broadcasts
    each batch to other TP ranks.
    """

    def __init__(
        self, dl: DataLoader, tp_group: torch.distributed.ProcessGroup
    ):
        self.dl = dl
        self.tp_group = tp_group
        self.leader = _tp_is_leader(tp_group)

    def __iter__(self) -> Iterator:
        it: Iterable = iter(self.dl) if self.leader else range(len(self.dl))
        # Non-leaders iterate dummy range to keep step counts aligned
        for maybe_batch in it:
            batch = maybe_batch if self.leader else None
            batch = _broadcast_batch(batch, self.tp_group)
            yield batch

    def __len__(self) -> int:
        return len(self.dl)


# --- main factory ----------------------------------------------------------


def get_random_dataset_fsdp_tp(
    batch_size: int,
    vocab_size: int,
    seq_length: int,
    *,
    num_workers: int = 0,
    pin_memory: bool = True,
    dp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    broadcast_within_tp: bool = False,
    drop_last: bool = True,
    seed: int = 1337,
) -> Dict[str, Any]:
    """
    Build dataset/sampler/dataloader for FSDP (DP) + Tensor Parallel (TP).

    Key idea:
      - Shard the dataset ONLY across the **DP group** (FSDP replica group).
      - Optionally broadcast each batch within TP so only TP-leader does I/O.

    Args:
      dp_group: Process group that defines FSDP data-parallel replicas.
      tp_group: Process group that defines tensor parallel group.
      broadcast_within_tp: If True, TP leader loads and broadcasts batches.
      drop_last: Prefer True for static shapes across DP replicas.
      seed: Base seed for shuffling (per-epoch add epoch to this).

    Returns:
      dict with 'dataset', 'sampler', 'dataloader'
    """
    from ezpz.data.text import RandomTokenDataset

    dset = RandomTokenDataset(vocab_size=vocab_size, seq_length=seq_length)

    use_dist = _is_dist()
    sampler = None

    if use_dist:
        # Determine DP rank/world_size; TP is ignored by the sampler.
        dp_rank, dp_world = _rank_ws(dp_group)
        # Important: num_replicas/rank are DP-based, not global.
        sampler = DistributedSampler(
            dset,
            num_replicas=dp_world,
            rank=dp_rank,
            shuffle=True,
            drop_last=drop_last,
            seed=seed,
        )

    dl = DataLoader(
        dset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),  # never shuffle when a sampler is provided
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )

    if use_dist and broadcast_within_tp and tp_group is not None:
        dl = TPBroadcastDataLoader(dl, tp_group)

    return {
        "dataset": dset,
        "sampler": sampler,
        "dataloader": dl,
    }
