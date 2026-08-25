# LoRA + FSDP2: the frozen-unit collective asymmetry (#239)

!!! danger "Open bug. There is no fix, and the leading candidate failed."

    #239 is **unresolved**. The frozen-unit asymmetry described below is
    real and measurable, but removing it on Perlmutter **did not stop the
    deadlock** — see [Refuted: removing the asymmetry](#refuted-removing-the-asymmetry-fixes-it).
    That intervention therefore ships **off by default**.

    **Four** plausible-sounding explanations have now been tested and
    refuted. They are documented here so nobody re-derives them.

    Known workaround: use `--lora-rank 32` or higher, which completes
    normally. Why it does is still unexplained.

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

    The trace confirms the *shape* of the failure is consistent with the
    frozen-unit asymmetry. It does not explain why r=32 and r=64 — which
    have the **same** asymmetry — complete normally. That gap was the
    reason to test the intervention rather than assume it, and the test
    came back negative.

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

## Refuted: "removing the asymmetry fixes it"

This was the leading candidate. It was tested directly and **failed**.

`frozen_unit_kwargs()` in `src/ezpz/examples/fsdp_tp.py` keeps a
fully-frozen unit gathered, so it never emits the unmatched all-gather:

```python
if any(p.requires_grad for m in ms for p in m.parameters()):
    return fsdp_kwargs
return {**fsdp_kwargs, "reshard_after_forward": False}
```

It is correctness-neutral (the parameters are never updated, so never
resharding them changes no math, only residency) and it demonstrably
does what it claims — the AG/RS counts go from 14/12 to 12/12, pinned by
`tests/test_fsdp_tp_frozen_unit_reshard.py`.

**The job still hung.** Perlmutter job `57602201`, same 8x A100 /
torch 2.13.0+cu130, r8 and r16, both `last_iter=NONE`:

| | baseline (`57601590`) | asymmetry removed (`57602201`) |
|---|---|---|
| stuck collective | `_REDUCE_SCATTER_BASE` | `_REDUCE_SCATTER_BASE` |
| payload | `NumelIn=419840, NumelOut=52480` | **identical** |
| SeqNum | 18 | 17 |
| PG status | enq 39, started 19, completed 17 | enq 37, started 18, completed 16 |
| r8 | `rc=134 secs=518/433/437` (3/3) | `rc=134 secs=532` |
| r16 | — | `rc=124 secs=600` |

The intervention *landed*: 39 → 37 enqueued ops is exactly the two
removed all-gathers. And the failure is the same one, renumbered by
exactly that amount — the same skipped-work signature, where the stream
jumps over the reduce-scatter and stalls with an all-gather started
ahead of it.

So the asymmetry is not the cause, and — since the deadlock survives
without it — not even a precondition.

### Consequence for the code

`frozen_unit_kwargs()` is retained, but **inverted to opt-in** and OFF by
default: it costs ~+1.7 GiB/rank on `agpt-2b` at `world_size=8` in bf16
(scaling with the 256128 vocab) and buys nothing. Shipping a memory
regression that failed its own experiment would be worse than shipping
nothing.

```bash
export EZPZ_FSDP_KEEP_FROZEN_GATHERED=1   # opt in; reproduces the negative result
```

It stays in the tree only so the experiment is re-runnable from a
released build rather than a patch someone has to reconstruct.

### What was never in scope either way

- `--lora-target unembed` puts a trainable adapter in the `[norm,
  output]` unit, so that unit is not frozen at all. Never reported hanging.
- The HuggingFace path uses a different grouping.
- #237, and torch 2.13 FSDP2 more broadly.

## Where to look next

Every *static* property of the collective stream has now been ruled out:
LoRA-specificity, payload size, per-rank order, and the AG/RS asymmetry
itself. All eight ranks agree exactly on what they are waiting for. The
remaining explanations are dynamic:

1. **Why r>=32 works** is the sharpest unused clue. Same asymmetry, same
   op sequence, different outcome — so the difference is a *quantity*,
   not a structure. Worth bisecting r in 17..31 to find the boundary,
   which would say whether it is a threshold or a coincidence.
2. **torch 2.13 vs 2.12.1.** #237 diverges across the same boundary. A
   2.12.1 run of this exact config is cheap and would either implicate
   the release or clear it.
3. **The skipped work item.** The stream goes `completed 16` →
   `started 18`, so #17 was enqueued and jumped. Instrumenting FSDP2's
   `foreach_reduce` to log which unit owns each work id would name the
   module involved instead of inferring it from payload arithmetic.

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

The job also asserts `frozen_unit_kwargs` is actually present in the
checkout before running, so an arm cannot silently test old code and
report a result that belongs to a different build. That guard is why the
negative result above is trustworthy: the intervention provably ran.
