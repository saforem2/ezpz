# LoRA + FSDP2: the frozen-unit collective asymmetry (#239)

!!! warning "This is an open investigation, not a solved bug"

    The mitigation below removes a **precondition** for the hang. It is
    not a demonstrated cause. The asymmetry it fixes is present in both
    the hanging *and* the working configurations, so the honest status
    is: highest-value falsifiable intervention, awaiting confirmation.

    Two plausible-sounding explanations were tested and **refuted**.
    They are documented here so nobody re-derives them.

## What was observed

On Perlmutter (2 nodes x 4 A100, `world_size=8`, torch 2.13.0+cu130),
`agpt-2b` with `tp=1`, `bs=1`, `seq_len=2048`:

| `--lora-rank` | `--lora-target` | result |
|---|---|---|
| 8  | `attn,mlp` | **hang** in backward |
| 16 | `attn,mlp` | **hang** in backward |
| 16 | `attn`     | trains |
| 32 | `attn,mlp` | trains |
| 64 | `attn,mlp` | trains |

The watchdog fingerprint was `NumelIn=419840, NumelOut=52480`.

## Identifying the stuck collective

That fingerprint is not ambiguous. Per-`TransformerBlock` LoRA parameter
count is `coef · r`, where `coef` follows from the `agpt-2b` geometry
(`dim=2048`, `n_kv_heads=4`, `hidden_dim=11008`):

```
coef(attn)     = 13312
coef(mlp)      = 39168
coef(attn,mlp) = 52480
```

Sweeping `r ∈ [1, 2048]` against all three target sets yields **exactly
one** configuration producing 419840: `r=8, attn,mlp`. And
`419840 / 52480 = 8 = world_size`, consistent with a gradient
reduce-scatter. So the trace came from an **r=8** run, not r=16.

## The structural precondition

`apply_lora` freezes every base parameter first, then re-enables only the
adapter `A`/`B` pairs inside the targeted submodules
(`src/ezpz/tinker/lora.py`). Under `--lora-target attn,mlp` every adapter
therefore lands **inside a transformer block**. But `parallelize()` still
creates four kinds of FSDP2 unit:

| FSDP unit | trainable params |
|---|---|
| `tok_embeddings` | **0 — fully frozen** |
| `layers.N` (x12) | 14 |
| `[norm, output]` | **0 — fully frozen** |
| root | 0 |

A fully-frozen unit is asymmetric in backward. `reshard_after_forward`
discarded its parameters after forward, so backward **re-gathers** them —
but `post_backward` returns before the reduce-scatter when a group has no
gradients. The unit emits an all-gather with no matching reduce-scatter.

Measured on 2 real gloo ranks
(`tests/test_fsdp_tp_frozen_unit_reshard.py`):

```
frozen units resharded:      bwd AG=14  bwd RS=12   <- asymmetric
frozen units kept gathered:  bwd AG=12  bwd RS=12   <- symmetric
```

14 = 12 blocks + 2 frozen units. That shape matches the watchdog report:
ranks blocked on a reduce-scatter at one sequence number while others ran
ahead to an all-gather at a much later one.

## The watchdog trace (job 57601590, torch 2.13.0+cu130)

Lowering `TORCH_DDP_TIMEOUT` to 300s finally produced a dump. **All eight
ranks report byte-identical state:**

```
Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=18, OpType=_REDUCE_SCATTER_BASE,
           NumelIn=419840, NumelOut=52480, Timeout(ms)=300000)

Timeout at collective: _ALLGATHER_BASE, #39
  [0,1,2,3,4,5,6,7] joined but didn't finish collective #39

PG status: last enqueued work: 39,
           last started work: 19 (_ALLGATHER_BASE),
           last completed work: 17
```

Three things follow, and they change the diagnosis from hypothesis to
observation:

1. **This is not a rank divergence.** All 8 ranks joined #39 and report
   the same stuck op. Nobody took a different code path — which rules
   out the usual "one rank has a different collective order" story.

2. **Work #18 was skipped.** The stream went `last completed: 17` →
   `last started: 19`, and **#19 is an `_ALLGATHER_BASE`**. The stuck op
   #18 is the reduce-scatter that the all-gather jumped ahead of. That
   is the AG/RS asymmetry, caught in the act.

3. **It dies in the first backward** — `last_iter=NONE`, no training step
   ever completed. 39 ops enqueued against 17 completed.

The stack trace lands in
`torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:619`
(`foreach_reduce`), the FSDP2 gradient reduce-scatter path.

!!! note "What this still does not settle"

    The trace confirms the *shape* of the failure matches the frozen-unit
    asymmetry. It does not explain why r=32 and r=64 — which have the
    **same** asymmetry — complete normally. Until that is explained, the
    mitigation remains a precondition removal.

## Refuted: "the asymmetry is LoRA-specific"

It is not. The asymmetry tracks **fully-frozen FSDP units**, not LoRA —
freezing `tok_embeddings`/`norm`/`output` by hand with no adapters
anywhere produces the identical 14/12.

More decisively: it is **byte-identical at r=8 (hangs) and r=32
(works)**. A feature present in 100% of the *working* configurations
cannot by itself be the trigger.

## Refuted: "small payloads take a different dispatch path"

The idea that a reduce-scatter below some size threshold behaves
differently fails on the data:

```
per-block RS numel   hang: [419840, 839680]
                     ok:   [212992, 1679360, 3358720]
separable by size?   NO
```

`r16 attn` at 212992 elements **is smaller than either hanging config**
and trains fine. No monotone size threshold separates hang from ok at
per-block or total scale. This hypothesis only survived initial scrutiny
because it was never plotted against the working configurations.

## What actually distinguishes hang from ok: unknown

Nothing structural. Composition is identical — 14 trainable tensors per
block at every rank; only widths scale. Collective *order* is identical
across all 8 ranks at every rank on torch 2.12.1, and **the hang does not
reproduce on 2.12.1 at all**.

Given that #237 (FSDP2 tied-embedding rejection) was also 2.13-only, the
remaining live hypothesis is a **torch 2.13 FSDP2 regression in
backward-prefetch scheduling that a frozen unit's unmatched all-gather can
expose non-deterministically**. That is timing-sensitive, which would
explain why rank is a poor predictor — and raises the real possibility
that `r8/r16 vs r32/r64` is not a boundary at all, but a **flaky race
that happened to land twice**.

This is a hypothesis. It has not been tested. Do not cite it as the cause.

## Refuted: "the collective *order* differs by rank"

The trace shows work #18 skipped in favour of #19, so the obvious next
guess is that the frozen-unit all-gather sits at a different position in
the backward stream depending on `r`. It does not.

Recording the exact backward collective sequence on 2 gloo ranks
(`A` = all-gather, `R` = reduce-scatter, 6 layers):

```
r=8    bwd = AAARARARARARAR
r=16   bwd = AAARARARARARAR   identical
r=32   bwd = AAARARARARARAR   identical
r=64   bwd = AAARARARARARAR   identical
```

Byte-identical at every rank, including the ones that work. The leading
`AAA` is the asymmetry — three all-gathers before the first
reduce-scatter — and it is present in *all four*.

So the r-dependence is **not** an ordering difference. Combined with the
size refutation above, no *static* property of the collective stream
separates hanging from working configurations. That is what pushes the
remaining explanation toward timing rather than structure.

## Mitigation

`frozen_unit_kwargs()` in `src/ezpz/examples/fsdp_tp.py` keeps a
fully-frozen unit gathered, so it never emits the unmatched all-gather:

```python
if any(p.requires_grad for m in ms for p in m.parameters()):
    return fsdp_kwargs
return {**fsdp_kwargs, "reshard_after_forward": False}
```

Correctness-neutral: the parameters are never updated, so never
resharding them changes no math, only residency.

**Cost:** about **+1.7 GiB per rank** on `agpt-2b` at `world_size=8` in
bf16 (embedding and output stay gathered). Affordable on A100-40GB, but
it scales with vocabulary size (256128 here) — re-check for
larger-vocab models.

**Opt out** with `EZPZ_FSDP_FROZEN_RESHARD=1`, which restores the old
behaviour so both arms of an experiment come from one build.

### What this does not address

- `--lora-target unembed` puts a trainable adapter in the `[norm,
  output]` unit, so that unit is no longer frozen and takes the
  unchanged path. That configuration was never reported hanging.
- The HuggingFace path uses a different grouping.
- #237, and torch 2.13 FSDP2 more broadly.

## The usability bug underneath

`TORCH_DDP_TIMEOUT` defaults to **3600s** (`src/ezpz/distributed.py`),
so this failed as a silent one-hour hang rather than a watchdog dump.
In a 29-minute debug allocation the watchdog can **never** fire — which
means several early probes were silent for that reason alone, and that
silence was briefly mistaken for evidence. Set it low when hunting a
hang:

```bash
export TORCH_DDP_TIMEOUT=300
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
```

## Reproducing

`experiments/perlmutter/lora_239_repro.sbatch` runs the **same baseline
three times** before testing anything else. That ordering is deliberate:
two prior jobs each hung once, which is equally consistent with a
deterministic bug and with a coin flip. Two samples cannot tell those
apart, and every downstream claim depends on which it is.

The job also asserts the fix is actually present in the checkout before
running, so it cannot silently test old code and report success.
