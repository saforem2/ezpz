# LoRA + FSDP2 deadlock (#239): an aws-ofi-nccl transport bug

!!! success "Isolated 2026-09-15. It is not an FSDP2 bug."

    A **~4 MiB reduce-scatter deadlocks on Perlmutter's
    `aws-ofi-nccl` / Slingshot (`cxi`) path** and completes normally
    over TCP. The LoRA rank only controls whether a bucket of that size
    is produced; it is the trigger, not the cause.

    Measured at `--lora-rank 17`, `world_size=8`, torch 2.13.0+cu130
    (jobs 58368557, 58369713):

    | transport | runs | result |
    |---|---|---|
    | `aws-ofi-nccl` (default) | **6/6** | **HANG**, always `NumelIn=1055232` |
    | `NCCL_NET=Socket` | **3/3** | trains, 4/4 iters |

    The six hanging runs include the default baseline twice, plus
    `NCCL_PROTO=Simple`, `NCCL_ALGO=Ring`, `TORCH_NCCL_DESYNC_DEBUG=1`,
    and `NCCL_DEBUG=INFO`. Every one stalls on a **byte-identical**
    buffer. Protocol and algorithm selection are therefore both
    excluded: the defect sits below them, in the network plugin.

    **Workarounds**

    - **`--lora-rank 18`** — full speed (0.20 s/step). Preferred.
    - **`NCCL_NET=Socket`** — any rank, but **~7× slower**
      (1.38 s/step vs 0.20). Real TCP fallback, not a placebo: the
      slowdown matches the documented 8.3× inter-node penalty.

    **This is a NERSC ticket, not an upstream PyTorch one.**

!!! danger "REFUTED 2026-09-18: it is not the payload size alone"

    A standalone `dist.reduce_scatter_tensor` sweep -- no FSDP2, no LoRA,
    no ezpz (`experiments/common/reduce_scatter_size_sweep.py`) -- was run
    on Perlmutter at the identical geometry (ws=8, 2x4 A100, torch 2.13,
    default `aws-ofi-nccl`/`cxi` path), sweeping 0.5 MiB to 12.8 MiB and
    including the exact hanging buffers.

    **Every size completed, in under 0.1 s. Zero watchdog timeouts.**

    | numel | MiB | default plugin | `NCCL_NET=Socket` |
    |---|---|---|---|
    | 419840 | 1.602 | OK | OK |
    | 1055232 | **4.025** | **OK** | OK |
    | 1084416 | 4.137 | OK | OK |

    `1055232` is the precise buffer the LoRA job deadlocks on, on the same
    machine and the same transport. **So a reduce-scatter of that size is
    not sufficient to trigger the bug**, and the "bounded payload window"
    framing -- recorded below as a hypothesis -- is refuted as a complete
    explanation.

    What survives: `NCCL_NET=Socket` still fixes the LoRA job 3/3 while the
    default path hangs it 6/6, on the same nodes. The transport is still
    implicated. But the trigger needs something the bare collective does
    not have -- concurrency with in-flight all-gathers, FSDP2's two-stream
    usage of one communicator, the CUDA-graph/allocator state, or the
    surrounding collective sequence.

    Three other machines also pass the standalone sweep (Polaris A100/NCCL,
    Sunspot and Aurora PVC/XCCL), so it is a valid cross-machine control --
    it just is not a reproducer.

    **Consequence for the NERSC ticket:** it must be filed with the
    LoRA-shaped reproducer. The ten-line version does not reproduce.

!!! danger "ALSO REFUTED 2026-09-19: stream concurrency is not the trigger"

    The obvious follow-up to the size refutation was that FSDP2's
    *concurrency* supplies the missing ingredient -- an all-gather
    overlapping the reduce-scatter, on a second CUDA stream sharing one
    communicator. `experiments/common/reduce_scatter_concurrency_probe.py`
    tests exactly that, four arms, 40 iterations each, at the r17 payload:

    | arm | what it adds | Perlmutter default | socket | Polaris |
    |---|---|---|---|---|
    | A sequential | 40x bare reduce-scatter | OK | OK | OK |
    | B interleaved | all-gather, same stream | OK | OK | OK |
    | C two-stream | all-gather on a side stream, overlapping | **OK** | OK | OK |
    | D two-stream + waits | C plus cross-stream `wait_stream` | **OK** | OK | OK |

    Every arm completed in ~0.12 s on the default `aws-ofi-nccl`/`cxi`
    path, zero watchdog timeouts, on the same 2x4 A100 geometry that
    deadlocks the LoRA job. **Overlapped AG/RS on two streams over one
    communicator is not sufficient either.**

    So the trigger is none of: payload size, protocol, algorithm, rank
    participation, collective ordering, or plain stream concurrency. What
    remains untested in isolation is FSDP2's *scale* of communicator state
    -- 14 units cycling distinct buffers, the caching allocator's memory
    pool interacting with registered/pinned regions, and the parameter
    all-gathers being issued from the autograd engine's threads rather
    than the main thread.

!!! info "Three probes, three refutations: the trigger needs real FSDP2"

    Three attempts were made to reproduce #239 without FSDP2, each testing
    one thing FSDP2 does that the previous probe lacked. All ran on
    Perlmutter at the exact failing geometry (ws=8, 2x4 A100, torch 2.13,
    default `aws-ofi-nccl`/`cxi`), at the r17 payload:

    | probe | tests | result |
    |---|---|---|
    | `reduce_scatter_size_sweep.py` | payload size, 0.5-12.8 MiB | all OK |
    | `reduce_scatter_concurrency_probe.py` | AG/RS overlap, 1 vs 2 streams | all OK |
    | `reduce_scatter_units_probe.py` | 14 cycled buffers; non-main thread; both + overlapping AG | all OK |

    Every arm completed in under 0.7 s with zero watchdog timeouts. The
    last probe's arms carry completed-iteration counts and value checks
    (`F-nonmain-thread(30/30)`), so an arm cannot pass by doing nothing --
    an earlier version reported a 0.04 s "pass" that was exactly that.

    **So none of these is the trigger:** payload size, NCCL protocol,
    algorithm selection, rank participation, collective ordering, torch
    version, stream concurrency, buffer-registration cycling, or the
    calling thread.

    Probing stopped here deliberately. Each probe eliminated a named
    mechanism; a fourth would be guessing rather than testing a
    hypothesis. The reproducer that works is the LoRA one, and the
    refutations above are the useful output -- they bound where to look.

## The stack, and two defects in it

Captured with `NCCL_DEBUG=INFO` on a hanging run (job 58369713):

```
NCCL version 2.29.7
NET/OFI Selected Provider is cxi (found 4 nics)
Using network AWS            <- aws-ofi-nccl 1.6.0
```

**1. The NCCL the job loads is not the NCCL the job asked for.**
`experiments/perlmutter/*.sbatch` runs `module load nccl/2.24.3`, but the
NERSC PyTorch 2.13 install bundles its own `libnccl.so.2` at **2.29.7**
under `site-packages/nvidia/nccl/lib/`, which wins via RPATH. The module
load is inert. So **aws-ofi-nccl 1.6.0 is running against an NCCL five
minor versions newer** than the pairing the site presumably validated.
The plugin ABI is version-sensitive; this alone is a sufficient
explanation and is the first thing to put in the ticket.

**2. The ibverbs fallback is unavailable.** Every rank on both nodes
emits, at init:

```
misc/ibvwrap.cc:173 NCCL WARN lib wrapper not initialized.
```

Harmless on its face — `cxi` is the transport, not verbs — but it means
NCCL has no second path. When the OFI operation stalls there is nothing
to fail over to, so the collective **hangs instead of erroring**. The
last `NCCL INFO` lines before the stall are
`[Service thread] Connection closed by localRank N` — teardown, not data
movement.

!!! warning "The Polaris control varied two things, not one"

    `experiments/polaris/lora_239_a100.sh` loads **no NCCL network
    plugin**. So the Polaris control did not only change site — it also
    silently removed `aws-ofi-nccl` from the comparison. That weakens
    Polaris as a clean "different machine" control, but it *strengthens*
    the transport localization: Polaris is another configuration without
    the plugin, and like `NCCL_NET=Socket` it trains.

## All eight ranks enter the collective

`TORCH_NCCL_DESYNC_DEBUG=1` makes PyTorch name the culprits itself:

```
[0, 1, 2, 3, 4, 5, 6, 7] joined but didn't finish collective #39
```

Every rank joined. **This kills the frozen-unit participation-asymmetry
theory outright** — no rank took an early return out of `post_backward`,
so nothing about gradient-less FSDP2 units can be the mechanism. The
asymmetry documented further down this page is real and measurable, but
it is not what deadlocks.

It also retroactively justifies shipping the `reshard_after_forward`
intervention **off by default**: it was addressing a mechanism that is
not the one at fault.

## The buffer size is invariant; the sequence number is not

| run | SeqNum | NumelIn |
|---|---|---|
| baseline | 18 | 1055232 |
| `NCCL_DEBUG=INFO` | 18 | 1055232 |
| `TORCH_NCCL_DESYNC_DEBUG=1` | **20** | **1055232** |

Instrumentation adds collectives, so the position in the stream moves.
The **stuck buffer does not**. The identity of this bug is a
~4 MiB message, not "the 18th collective" — and `NumelIn=1055232`
reproduces bit-for-bit across **torch 2.11 and 2.13**, so the 2.13
rename of the FSDP2 collectives
(`reduce_scatter_tensor` → `reduce_scatter_single`) is irrelevant to it.

!!! success "The 18.3 % anomaly is solved: FSDP2 dim-0 padding"

    FSDP2 pads each gradient's dim 0 up to a multiple of `world_size`
    before the reduce-scatter (`_get_dim0_padded_size` in
    `_fsdp_common.py`, called from `foreach_reduce`). In
    `src/ezpz/tinker/lora.py:197-198`:

    - `A = nn.Linear(in_features, rank)` → `A.weight` is `(r, in_features)`,
      so **dim 0 is `r` and it pads**;
    - `B = nn.Linear(rank, out_features)` → `B.weight` is `(out_features, r)`,
      whose dim 0 is 2048/512/11008 — all divisible by 8, so **B never pads**.

    ```
    NumelIn(r) = ceil(r/8)*8 * S_in  +  r * S_out
      attn,mlp:  S_in = 23296   S_out = 29184     (S_in + S_out = 52480)
      attn:      S_in =  8192   S_out =  5120
      mlp:       S_in = 15104   S_out = 24064
    ```

    Zero free parameters, and it reproduces every recorded payload exactly:

    | r | predicted | observed |
    |---|---|---|
    | 8  | 419840  | 419840  |
    | 16 | 839680  | 839680  |
    | 17 | 1055232 | 1055232 |
    | 32 | 1679360 | 1679360 |
    | 64 | 3358720 | 3358720 |

    The "+18.3 %" at r17 is exactly **7 pad rows**: `7 × 23296 = 163072`,
    or `20384` per shard — the excess measured earlier. And the negative
    intercept in the old `70599·r − 144953` fit was an **artifact of
    fitting a line through a staircase**: the slope is 29184 *within* a
    step, with a 186368 jump at each `r = 8k+1`. There was never a
    constant term to explain.

!!! danger "Byte figures below this point are wrong — the reduce is fp32"

    The tables further down compute wire sizes as `coef · r / ws · 2`
    "for bf16". **The reduce dtype is fp32**, not bf16:
    `src/ezpz/examples/fsdp_tp.py:2994` sets
    `_reduce_dtype = torch.float32`, and `EZPZ_REDUCE_DTYPE` defaults to
    `"fp32"` (line 3004). Every byte figure in those tables is therefore
    **half the true value, and unpadded on top of that**. Corrected:

    | target | r | NumelIn | MiB (fp32) | result |
    |---|---|---|---|---|
    | attn | 16 | 212992 | 0.812 | trains |
    | attn,mlp | 8 | 419840 | **1.602** | **HANG** |
    | attn,mlp | 16 | 839680 | **3.203** | **HANG** |
    | attn,mlp | 17 | 1055232 | **4.025** | **HANG** |
    | attn,mlp | 18 | 1084416 | 4.137 | trains |
    | attn,mlp | 32 | 1679360 | 6.406 | trains |
    | attn,mlp | 64 | 3358720 | 12.812 | trains |

    This reframes the trigger as a **bounded payload window**, roughly
    **[1.60 MiB, 4.03 MiB]**, rather than "small ranks". Note r8 is the
    *smallest* hanging case while `attn`-only at r16 (0.81 MiB) trains —
    so it is not a simple "below threshold" story.

    Treat the window as a **hypothesis, not a finding**. Within
    `--lora-target attn,mlp` the payload is monotone in `r`, so the
    existing data cannot separate "payload window" from "ranks 8–17 are
    bad"; the single `attn`-r16 point is the only thing distinguishing
    them. The upper edge also falls between 4.025 and 4.137 MiB — a 2.8 %
    gap landing on no round number, with 4 MiB sitting *below* r17. Real
    protocol crossovers usually sit on round values, so the framing may
    still be wrong.

## What was observed

!!! warning "Written before the transport was isolated"

    Everything from here down predates the 2026-09-15 isolation. The
    measurements are accurate and the refutations still stand — they are
    what narrowed the search. But the framing ("frozen-unit collective
    asymmetry") describes a real phenomenon that is **not** the cause of
    the deadlock. Read it as the investigation log it is.


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

**Rerun at the matched `world_size=8`** (job `12473880`), removing the
bucketing confound below — r8 `rc=0` 94 s, r17 `rc=0` 72 s, r18 `rc=0`
68 s; XPU dispatch, plots written, zero watchdog lines, 3/3. The ws=24
caveat that follows is therefore no longer load-bearing: **xccl is
clean at Perlmutter's exact geometry**, so the backend really is not
the variable.

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

## It does not reproduce on Polaris either (A100 + NCCL, ws=8)

**This is the controlled test, and it came back negative.**

Polaris job `7563257` varies *only* the site and software stack:

| | Perlmutter | Polaris |
|---|---|---|
| accelerator | A100 | A100 |
| collectives | NCCL | NCCL |
| `world_size` | 8 | 8 |
| torch | 2.13.0+cu130 | 2.13.0+cu129 |
| `--lora-rank 8` | **hang, 6/6** | **trains**, `rc=0`, 142 s |
| `--lora-rank 17` | **hang** | **trains**, `rc=0`, 107 s |
| `--lora-rank 18` | trains | **trains**, `rc=0`, 107 s |

All three cells clean, 3/3. Verified it is the intended configuration:
`device=cuda`, `--lora-rank 8 --lora-target attn,mlp`, plotting output
written, and **zero watchdog lines**. Both ranks that deadlock on
Perlmutter train here, so the boundary does not shift on Polaris -- it
is absent.

### What this kills

The "#239 is an NCCL bug" hypothesis. A second, independent A100+NCCL
site at the same world size, on the same torch minor, runs the exact
configuration clean. If the deadlock were a property of NCCL's
interaction with FSDP2's reduce-scatter scheduling, it should have
reproduced here.

**#239 has not reproduced anywhere except Perlmutter** -- where it is
nonetheless rock-solid deterministic at 6/6. (The XPU/xccl arm is not
yet conclusive; see the Sunspot section's ws=24 caveat.)

### What that leaves

Something in **Perlmutter's stack specifically**: its NCCL build
(2.29.7), the `nccl/2.24.3` AWS-libfabric plugin, the Slingshot
interconnect, its CUDA 13.0 / cu130 wheel pairing, or the
2-nodes-x-4-GPU topology as realised there. Those are now the live
suspects, not FSDP2 or NCCL-in-general.

!!! warning "Do not over-read this either"

    Polaris differs from Perlmutter in more than "site": cu129 vs
    cu130, a different NCCL, a different interconnect, PBS vs Slurm.
    This narrows the search to Perlmutter's stack; it does not identify
    which component. The honest status is that the r-boundary is real
    and deterministic **on one machine**, and no portable mechanism has
    been found.

    It also means an upstream torch issue is **not** currently
    warranted -- see `docs/notes/upstream-fsdp2-lora-deadlock.md`, which
    stays unsent.

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

## Cross-machine summary: it only happens on Perlmutter

All rows at `world_size=8` (Perlmutter's exact geometry, so FSDP2
buckets identically everywhere), `agpt-2b`, `tp=1`, `bs=1`,
`seq_len=2048`, `--lora-target attn,mlp`, shipped default arm:

| `--lora-rank` | Perlmutter<br>A100 / NCCL / 2.13 | Polaris<br>A100 / NCCL / 2.13 | Sunspot<br>PVC / xccl / 2.13 | Aurora<br>PVC / xccl / **2.10** |
|---|---|---|---|---|
| 8  | **hang**, 6/6 | trains 142 s | trains 94 s | trains 60 s |
| 17 | **hang** | trains 107 s | trains 72 s | trains 59 s |
| 18 | trains | trains 107 s | trains 68 s | trains |

**Two backends, four stacks, every run clean except Perlmutter** --
where the hang is deterministic at 6/6.

The load-bearing comparisons are Polaris (matches Perlmutter on
accelerator, collectives, world size and torch minor -- varies only the
site) and Sunspot (matches on torch minor and world size -- varies the
collectives backend). Aurora is weaker evidence because it ships torch
**2.10**, and #239 has only been seen on 2.13; its value is that it did
not hang, which would have meant the bug predates 2.13.

!!! note "Read the exit codes carefully"

    Several clean runs exit non-zero. Aurora's cells report `rc=1` from
    a post-training `plotext` version mismatch, and Perlmutter's r64
    cell did the same -- in both cases plots were written and no
    watchdog fired. Judge these runs on evidence (watchdog line vs.
    reaching the plotting stage), never on `rc`, or a pass gets
    misfiled as INDETERMINATE.

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
2. **Is it a torch 2.13 regression?** ~~Open~~ — **answered: no.** The
   deadlock reproduces identically on **torch 2.11 and 2.13**, down to
   the same `NumelIn=1055232` bit-for-bit (see *The buffer size is
   invariant* above). An older torch hangs the same way, so this is not
   a 2.13 FSDP2 scheduling regression and the lead is closed.
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
