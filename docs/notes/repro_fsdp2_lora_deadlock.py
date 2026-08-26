#!/usr/bin/env python3
"""Standalone reproducer for the FSDP2 + frozen-unit backward deadlock.

**Deliberately has no ezpz dependency** -- upstream maintainers will not
run someone else's training framework. torch + torchrun only.

    TORCH_DDP_TIMEOUT=300 torchrun --nproc_per_node=8 repro_fsdp2_lora_deadlock.py --rank 8

Observed on 2x4 A100 (world_size=8), torch 2.13.0+cu130, NCCL:

    --rank 8   hangs in the first backward
    --rank 17  hangs
    --rank 18  completes

The hang is a reduce-scatter that never finishes while the stream runs
ahead to a later all-gather::

    WorkNCCL(SeqNum=18, OpType=_REDUCE_SCATTER_BASE,
             NumelIn=419840, NumelOut=52480)
    PG status: last enqueued 39, last started 19 (_ALLGATHER_BASE),
               last completed 17

Note ``last completed 17`` -> ``last started 19``: work #18 is skipped.

STATUS: written from the failing configuration but **not yet executed
standalone**. Run it before citing it upstream -- if it does not
reproduce, the difference from the real workload is itself the clue.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

# agpt-2b geometry -- the sizes matter, since the boundary is a
# function of the per-bucket element count.
DIM = 2048
N_LAYERS = 12
N_HEADS = 32
N_KV_HEADS = 4
HIDDEN = 11008
VOCAB = 256128


class Block(nn.Module):
    """Attention + MLP with LoRA adapters; base weights frozen."""

    def __init__(self, r: int) -> None:
        super().__init__()
        hd = DIM // N_HEADS
        kv = N_KV_HEADS * hd
        base = {
            "wq": (DIM, DIM),
            "wk": (kv, DIM),
            "wv": (kv, DIM),
            "wo": (DIM, DIM),
            "w1": (HIDDEN, DIM),
            "w2": (DIM, HIDDEN),
            "w3": (HIDDEN, DIM),
        }
        self.base = nn.ModuleDict(
            {k: nn.Linear(i, o, bias=False) for k, (o, i) in base.items()}
        )
        # A/B adapter pair per base weight -- the only trainable tensors.
        self.lora_a = nn.ModuleDict(
            {k: nn.Linear(i, r, bias=False) for k, (o, i) in base.items()}
        )
        self.lora_b = nn.ModuleDict(
            {k: nn.Linear(r, o, bias=False) for k, (o, i) in base.items()}
        )
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for k in ("wq", "wo"):
            h = h + self.lora_b[k](self.lora_a[k](h))
        h = h + self.base["wo"](self.base["wq"](h))
        m = self.base["w2"](self.base["w1"](h))
        m = m + self.lora_b["w2"](self.lora_a["w2"](self.base["w1"](h)))
        return h + m


class Model(nn.Module):
    def __init__(self, r: int) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(VOCAB, DIM)
        self.layers = nn.ModuleList([Block(r) for _ in range(N_LAYERS)])
        self.norm = nn.LayerNorm(DIM)
        self.output = nn.Linear(DIM, VOCAB, bias=False)
        # Everything outside the blocks is frozen, so tok_embeddings and
        # [norm, output] become FSDP units with ZERO trainable params.
        self.tok_embeddings.weight.requires_grad_(False)
        self.output.weight.requires_grad_(False)
        for p in self.norm.parameters():
            p.requires_grad_(False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.tok_embeddings(ids)
        for blk in self.layers:
            h = blk(h)
        return self.output(self.norm(h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank")
    ap.add_argument("--seq-len", type=int, default=2048)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    ws = dist.get_world_size()

    torch.manual_seed(0)
    model = Model(args.rank).to(torch.bfloat16).cuda()

    mesh = init_device_mesh("cuda", (ws,))
    kw = {"mesh": mesh, "reshard_after_forward": True}
    # The grouping that matters: embeddings and [norm, output] are their
    # own units and hold no trainable parameters.
    fully_shard(model.tok_embeddings, **kw)
    for blk in model.layers:
        fully_shard(blk, **kw)
    fully_shard([model.norm, model.output], **kw)
    fully_shard(model, **kw)

    if dist.get_rank() == 0:
        n = sum(p.numel() for p in model.layers[0].parameters() if p.requires_grad)
        print(f"world_size={ws} lora_rank={args.rank} trainable/block={n}", flush=True)
        print("starting backward -- hangs here at rank 8/17", flush=True)

    ids = torch.randint(0, VOCAB, (1, args.seq_len), device="cuda")
    model(ids).float().pow(2).mean().backward()

    dist.barrier()
    if dist.get_rank() == 0:
        print("COMPLETED -- no deadlock", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
