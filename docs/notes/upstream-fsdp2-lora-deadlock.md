# Draft: upstream PyTorch issue for #239

**Status: DRAFT, unsent.** Filing an upstream issue is the user's call.
Do not submit this without asking.

Two things should land before this goes out:

1. **The Polaris control** (A100 + NCCL at `world_size=8`). Without it
   the report cannot say whether this is NCCL-specific, and the first
   question any reviewer asks is "does it reproduce elsewhere?"
2. **A minimal repro that does not import ezpz.** A maintainer will not
   run our example module. The reproducer below is sketched, not yet
   executed.

---

## Title

FSDP2: deterministic backward deadlock on a reduce-scatter when whole
units are frozen (LoRA), at some parameter counts but not others

## Environment

- torch `2.13.0+cu130`, CUDA 13.0
- 2 nodes x 4 A100-40GB, `world_size=8`, NCCL
- FSDP2 (`fully_shard`), `tp=1`, bf16 with fp32 reduce
- Model: 12-layer decoder, `dim=2048`, `n_kv_heads=4`, `hidden_dim=11008`,
  `vocab=256128`

## What happens

Training deadlocks in the **first** backward pass. No step completes.
All eight ranks report byte-identical watchdog state:

```
WorkNCCL(SeqNum=18, OpType=_REDUCE_SCATTER_BASE,
         NumelIn=419840, NumelOut=52480, Timeout(ms)=300000)

Timeout at collective: _ALLGATHER_BASE, #39
  [0,1,2,3,4,5,6,7] joined but didn't finish collective #39

PG status: last enqueued work: 39,
           last started work: 19 (_ALLGATHER_BASE),
           last completed work: 17
```

Stack lands in
`torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:619`
(`foreach_reduce`).

Note `last completed: 17` -> `last started: 19`: **work #18 was
skipped**, and #19 is an all-gather. The stream ran ahead of the
reduce-scatter it is now blocked on.

## Setup

LoRA adapters are applied to attention + MLP inside every transformer
block; everything else is frozen. Because the adapters live only inside
blocks, two FSDP2 units end up with **zero trainable parameters**:

| FSDP unit | trainable params |
|---|---|
| `tok_embeddings` | 0 |
| `layers.N` (x12) | 14 |
| `[norm, output]` | 0 |
| root | 0 |

A fully-frozen unit is asymmetric in backward: `reshard_after_forward`
discarded its parameters, so backward re-gathers them, but
`post_backward` returns before the reduce-scatter when the group has no
gradients. Measured on 2 gloo ranks: **14 backward all-gathers against
12 reduce-scatters**.

## Why that asymmetry is probably NOT the cause

Removing it does not fix the hang. Forcing the frozen units to stay
gathered (`reshard_after_forward=False`) makes the counts 12/12 — the
enqueued-op count drops 39 -> 37, exactly the two removed all-gathers —
and the deadlock is unchanged in kind:

| | default | asymmetry removed |
|---|---|---|
| stuck collective | `_REDUCE_SCATTER_BASE` | same |
| payload | `419840 -> 52480` | **identical** |
| SeqNum | 18 | 17 |
| PG status | enq 39 / start 19 / done 17 | enq 37 / 18 / 16 |

Same skipped-work signature, renumbered by exactly the intervention's
own delta.

## The strongest clue: an adapter-rank boundary

The outcome depends on LoRA rank, **deterministically** (r8 reproduced
6/6), with an exact boundary:

```
hang:   r = 8, 16, 17
trains: r = 18, 19, 20, 24, 28, 32, 64   (and r=0, no adapters)
```

And the stuck bucket is **not linear in r**:

```
r8   SeqNum=18  NumelIn=419840   NumelOut=52480    = 52480 * 8   (linear)
r17  SeqNum=18  NumelIn=1055232  NumelOut=131904   (linear would be 892160)
```

r17's stuck bucket is **18% larger** than one block's LoRA parameters,
while r8's matches exactly. Both keep `NumelIn = NumelOut * world_size`,
so both are genuine world-size reduce-scatters — but at r17 the bucket
is evidently **not** a single unit's gradients. Something groups or pads
differently right at the boundary, and that tracks the hang better than
any other property we found.

## Ruled out

Each tested, not assumed:

1. **LoRA-specific** — no; hand-freezing embeddings/norm/output with no
   adapters gives the identical 14/12 asymmetry.
2. **Payload size threshold** — no; hanging sizes `[419840, 839680]` vs
   working `[212992, 1679360, 3358720]` are not separable, and the
   smallest working case is *smaller* than the smallest hanging one.
3. **Per-rank collective order** — no; the backward sequence is
   byte-identical (`AAARARARARARAR`) at r=8/16/32/64, and all 8 ranks
   agree on the stuck op.
4. **The AG/RS asymmetry itself** — no; see above.
5. **NCCL protocol selection** — no; `NCCL_PROTO=Simple` and `LL128`
   both hang with `r=8` (486 s / 430 s), same payload, no
   `invalid`/`unknown proto` warning.

## Questions for maintainers

1. What determines reduce-scatter bucket membership in `foreach_reduce`
   when some units in the module tree have no gradients? Is there a
   path where a bucket spans more than one unit?
2. Is `post_backward` returning early for a gradient-less group safe
   with respect to the all-gather/reduce-scatter ordering the stream
   depends on?
3. Is there a supported way to log bucket membership per work id? The
   payload arithmetic identifies *sizes* but cannot name the parameters
   in the bucket, which is exactly what is missing here.

## Minimal reproducer

**Not yet written.** Must not depend on ezpz. Shape:

- 12-layer decoder with the geometry above
- freeze everything, add rank-8 A/B adapters inside each block only
- `fully_shard(tok_embeddings)`, each block, `[norm, output]`, root
- one forward/backward at `world_size=8`, `TORCH_DDP_TIMEOUT=300`
