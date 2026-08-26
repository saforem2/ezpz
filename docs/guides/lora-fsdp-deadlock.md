# LoRA + FSDP2: the frozen-unit collective asymmetry (#239)

!!! danger "Open bug. There is no fix, and the leading candidate failed."

    #239 is **unresolved**. The frozen-unit asymmetry described below is
    real and measurable, but removing it on Perlmutter **did not stop the
    deadlock** — see [Refuted: removing the asymmetry](#refuted-removing-the-asymmetry-fixes-it).
    That intervention therefore ships **off by default**.

    **Six** plausible-sounding explanations have now been tested and
    refuted. They are documented here so nobody re-derives them.

    Workaround: **`--lora-rank 18` or higher** completed normally in
    every rank tested (18, 19, 20, 24, 28, 32, 64), with the boundary at
    exactly **16 → 18** — r=17 hangs, r=18 trains.

    Treat that as *measured*, not *guaranteed*. Every one of those runs
    is the same configuration: `agpt-2b`, `tp=1`, `bs=1`, `seq_len=2048`,
    `--lora-target attn,mlp`, `world_size=8`, torch 2.13.0+cu130 on
    A100. The mechanism is still unknown, so a different model, target
    set, or world size could put the boundary somewhere else entirely.
    If you hit the hang above r=18, that is new information — please add
    it to [#239](https://github.com/saforem2/ezpz/issues/239).

    **It does not reproduce on Sunspot (XPU/xccl)** — but that run
    changed `world_size` (24 vs 8) as well as the backend, and at ws=24
    the shards no longer divide evenly, so FSDP2 buckets differently.
    It is **not** yet evidence the bug is NCCL-specific. See
    [the Sunspot section](#it-does-not-reproduce-on-xpuxccl-sunspot).

    The sharpest open clue on the NVIDIA side is that r17's stuck bucket
    is **18 % larger than linear in r** while r8's is exactly linear —
    so at r17 the stuck reduce-scatter is *not* one block's LoRA
    parameters. See [the boundary section](#where-to-look-next).

## What was observed

On Perlmutter (2 nodes x 4 A100, `world_size=8`, torch 2.13.0+cu130),
`agpt-2b` with `tp=1`, `bs=1`, `seq_len=2048`:

| `--lora-rank` | `--lora-target` | result |
|---|---|---|
| 0 (no LoRA)   | —          | trains |
| 8  | `attn,mlp` | **hang** in backward |
| 16 | `attn,mlp` | **hang** in backward |
| 16 | `attn`     | trains |
| 17 | `attn,mlp` | **hang** in backward |
| 18 | `attn,mlp` | trains |
| 19 | `attn,mlp` | trains |
| 20 | `attn,mlp` | trains |
| 24 | `attn,mlp` | trains |
| 28 | `attn,mlp` | trains |
| 32 | `attn,mlp` | trains |
| 64 | `attn,mlp` | trains |

The r18–r28 rows come from the bisect (jobs `57604409`, `57604619`) and
**put the boundary at exactly 16→18** — far below the r>=32 this guide
originally implied. r=17 hangs and r=18 trains, one step apart, with r=19 also
training; the passing cells each finished cleanly in 96–175 s.

The r8/r16 `attn,mlp` hang has reproduced **5/5** (jobs `57601590` ×3,
`57602201`, and the same-allocation control in `57604574` at
rc=134/501 s). It is deterministic.

The watchdog fingerprint was `NumelIn=419840, NumelOut=52480`.

!!! note "How the working rows were classified"

    The sweep job (`57540698`) exited non-zero on several cells for
    reasons unrelated to training — `r64 attn,mlp` reports `rc=1` but
    its per-cell log contains the full plotting output through
    iteration 20, so it **trained** and failed afterwards in teardown.
    The hanging cells produce no plots at all. Read those rows as
    "reached end of training", not "exited 0"; `rc` alone misclassifies
    them in both directions.

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
    frozen-unit asymmetry. It does not explain why every r >= 18 — which
    has the **same** asymmetry — completes normally. That gap was the
    reason to test the intervention rather than assume it, and the test
    came back negative.

## Refuted: "the asymmetry is LoRA-specific"

It is not. The asymmetry tracks **fully-frozen FSDP units**, not LoRA —
freezing `tok_embeddings`/`norm`/`output` by hand with no adapters
anywhere produces the identical 14/12.

More decisively: it is **byte-identical at r=8 (hangs) and r=32
(works)**. A feature present in 100% of the *working* configurations
cannot by itself be the trigger.

The converse also holds and is worth stating separately, because the two
are easy to conflate: the *asymmetry* is not LoRA-specific, but the
*hang* does require LoRA. A plain `--lora-rank 0` run — no adapters, so
no fully-frozen units and no asymmetry — trains normally. So LoRA is
necessary for the deadlock while the asymmetry is neither necessary nor
sufficient for it.

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
block at every rank; only widths scale, and the collective order is
identical too (see the order refutation below).

!!! warning "Corrected: this section previously overreached"

    An earlier revision claimed "the hang does not reproduce on torch
    2.12.1 at all". That is **not supported**. The 2.12.1 observations
    are from a local 2-rank **gloo/CPU** probe, which cannot reproduce a
    GPU NCCL deadlock under any torch version — absence there is not
    evidence of absence. **No 2.12.1 run on real GPUs has been done.**
    It remains lead #2 below, untried.

    The same revision floated that `r8/r16 vs r32/r64` might be "a flaky
    race that happened to land twice". That is now **disproven**: the r8
    baseline hung 3/3 with `last_iter=NONE` (job `57601590`). The hang is
    deterministic.

A torch 2.13 FSDP2 scheduling regression is still plausible — #237 is
also 2.13-only — but it is untested, and it now has to explain a
*deterministic* r-dependence rather than a race. Do not cite it as the
cause.

## Refuted: "the collective *order* differs by rank"

The trace shows work #18 skipped in favour of #19, so the obvious next
guess is that the frozen-unit all-gather sits at a different position in
the backward stream depending on `r`. It does not.

Recording the exact backward collective sequence on 2 gloo ranks
(`A` = all-gather, `R` = reduce-scatter, 6 layers). This probe runs on
CPU under torch 2.12.1, so it establishes *ordering* only — it does not
and cannot say anything about whether the deadlock reproduces:

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

## It does not reproduce on XPU/xccl (Sunspot)

The first non-Perlmutter data point, and the strongest constraint yet.

Sunspot job `12473856`: PVC (XPU), **xccl** rather than NCCL, torch
`2.13.0a0+gitcf30153` from the `frameworks` module, `world_size=24`,
same `agpt-2b` / `tp=1` / `bs=1` / `seq_len=2048` /
`--lora-target attn,mlp`, same shipped default arm.

| `--lora-rank` | Perlmutter (A100/NCCL) | Sunspot (PVC/xccl) |
|---|---|---|
| 8  | **hang**, 6/6 | **trains**, `rc=0`, 102 s |
| 17 | **hang** | **trains**, `rc=0`, 76 s |
| 18 | trains | **trains**, `rc=0`, 72 s |

Both ranks that deadlock on NVIDIA train clean on XPU, so the r-boundary
itself does not exist there — it is not that the boundary moved, it is
that there is nothing to bound.

Verified it is the intended configuration and not a silent skip: the log
shows `dispatch key: XPU`, the full
`--lora-rank 8 --lora-target attn,mlp` command line, and plotting output
through end of training.

!!! danger "This is NOT yet evidence that #239 is NCCL-specific"

    It is tempting to read this as "the bug is in NCCL". **Four things
    changed at once**, so the experiment does not isolate the backend:

    | | Perlmutter | Sunspot |
    |---|---|---|
    | collectives | NCCL | xccl |
    | `world_size` | 8 | **24** |
    | torch | `2.13.0+cu130` | `2.13.0a0+gitcf30153` |
    | hardware | A100 | PVC |

    The `world_size` change is the damaging one. `coef(attn,mlp)·r`
    does not divide evenly by 24 at the hanging ranks:

    ```
    ws=8   r8  419840 / 8  = 52480      integral
    ws=24  r8  419840 / 24 = 17493.33   NOT integral
    ws=24  r17 892160 / 24 = 37173.33   NOT integral
    ```

    So FSDP2 pads and buckets *differently* on Sunspot. Given that the
    sharpest open clue is precisely that r17's stuck bucket is
    **non-linear in r**, changing the bucket geometry is not a
    controlled test of the collectives backend -- it may simply have
    sidestepped whatever bucket shape triggers the hang.

    **What this run does establish:** the deadlock is not universal
    across backends and configurations, and the r-boundary is not a
    property of the LoRA geometry alone.

    **The controlled test still to run** is Polaris: A100 + **NCCL**
    like Perlmutter, at `world_size=8`, which varies only the site and
    software stack. Until then, "NCCL-localized" is a hypothesis, not a
    finding.

## Refuted: "the NCCL protocol selects the outcome"

The 256 KiB story pointed at NCCL protocol selection (LL / LL128 /
Simple). Rather than keep inferring protocol from payload size, job
`57605154` set it directly with `NCCL_PROTO`, holding `r=8` fixed so the
protocol is the *only* thing that varies:

| `NCCL_PROTO` | result |
|---|---|
| (default) | **hang** — 5/5 |
| `Simple` | **hang**, `rc=134`, 486 s |
| `LL128` | **hang**, `rc=134`, 430 s |

Both forced protocols hang, with the same `NumelIn=419840,
NumelOut=52480` as the default run, and NCCL logged no `invalid` or
`unknown proto` warning — so the setting was accepted rather than
silently ignored.

**The protocol is not the mechanism.** Not the 256 KiB threshold, and
not protocol selection in general.

## Where to look next

Every *static* property of the collective stream has now been ruled out:
LoRA-specificity, payload size, per-rank order, and the AG/RS asymmetry
itself. All eight ranks agree exactly on what they are waiting for. The
remaining explanations are dynamic:

1. **Where exactly does r flip?** Same asymmetry, same op sequence,
   different outcome — so the difference is a *quantity*, not a
   structure. `experiments/perlmutter/lora_239_rank_bisect.sbatch`
   binary-searches it (a hang costs the full 300s watchdog, so only ~3
   probes fit a debug allocation — hence bisect, not sweep).

    **In progress.** r=24 and r=20 both train, so the boundary is
    **17..20**, not r>=32. A second job (`57604619`) probes 18/17/19.

    !!! failure "REFUTED by r=18 — the 256 KiB prediction was wrong"

        Recorded below as written, unedited, because the point of
        pre-registering it was to be able to lose. **r=18 trained**
        (job `57604619`, 130 s, plots written, zero watchdog lines) —
        its shard is 236 160 B, comfortably *below* 256 KiB, where the
        prediction says it must hang.

        So the NCCL protocol edge does not explain the r-dependence.

    !!! success "But the boundary is now exact: r=17 hangs, r=18 trains"

        Same job. **r=17 hangs; r=18 trains.** One step apart, so the
        flip is precisely **16 → 18**, and any explanation has to
        separate two adjacent ranks.

        The r17 watchdog trace carries the most concrete new fact in the
        investigation:

        ```
        r8   SeqNum=18  NumelIn=419840   NumelOut=52480
        r17  SeqNum=18  NumelIn=1055232  NumelOut=131904
        ```

        r8's stuck bucket is exactly linear in r (`52480 · 8`). **r17's
        is not**: linearity predicts `892160`, the trace says `1055232`
        — 18 % larger. Both keep `NumelIn = NumelOut · 8`, so both are
        still world-size reduce-scatters, but at r17 the stuck bucket is
        **not one block's LoRA parameters**. Something is grouping or
        padding differently at r17 than at r8, and *that* difference
        tracks the boundary far better than any byte threshold.

        Chasing this needs the real bucketing, not arithmetic: a first
        attempt to reconstruct it from `agpt-2b` geometry did not even
        reproduce r8's known `419840`, so the padding story stays
        unwritten until FSDP2 is instrumented to report which
        parameters land in each bucket (lead 3).

        The direct `NCCL_PROTO` test settled the wider question — see
        below. It does not matter at all.

    !!! tip "Pre-registered prediction: the 256 KiB NCCL boundary"

        Alignment cannot explain this — `coef·r` is divisible by 8 and
        by 128 at *every* r. But the **per-rank reduce-scatter shard in
        bytes** (`coef·r/ws · 2` for bf16) crosses **262144 B = 256 KiB**
        exactly between the last known hang and the first known pass:

        | r | shard bytes | vs 256 KiB | outcome |
        |---|---|---|---|
        | 16 | 209 920 | below | **hang** |
        | 17 | 223 040 | below | *predict hang* |
        | 18 | 236 160 | below | *predict hang* |
        | 19 | 249 280 | below | *predict hang* |
        | 20 | 262 400 | **above** | trains |

        256 KiB is a real NCCL protocol/buffer boundary (LL / LL128 /
        Simple selection). So the prediction is that **r=17, 18 and 19
        all hang and the flip is exactly at r=20** — recorded here
        *before* job `57604619` reports, so it cannot be retrofitted.

        This also refutes payload size once more, from the other
        direction: r=64's per-shard payload is `419840`, the very number
        in r=8's *hanging* watchdog trace — identical byte counts land on
        both sides of the boundary. What would matter is not the size
        itself but which NCCL protocol it selects.

        If instead r=17/18/19 train, the boundary is 16→17 and this
        threshold story is wrong.

    Classify these cells on **evidence** (watchdog line vs. reaching the
    plotting stage), never on `rc` or an `iter=` marker: the first
    bisect gated on `iter=`, which these runs never emit, and so
    labelled a clean 173s r24 pass INDETERMINATE.
2. **Is it a torch 2.13 regression?** #237 diverges across the same
   boundary. `experiments/perlmutter/lora_239_torch_version.sbatch` runs
   the real config on real GPUs under Perlmutter's older `.venv`
   (**2.8.0+cu129** — there is no 2.12.1 build there, and a wider gap is
   fine, since the question is "does an older torch hang", not "which
   release introduced it"). Cell 1 is a **control** re-running r8 under
   2.13 in the same allocation, so an older-torch pass cannot be
   confounded by node or topology luck; cell 2 means nothing unless
   cell 1 hangs. If 2.8.0 also hangs, this theory dies and the search
   moves to FSDP2 semantics common to both.
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
