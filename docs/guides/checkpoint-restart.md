# Checkpoint & Restart Under Failure

How `ezpz.examples.fsdp_tp` behaves when training is interrupted: how it
saves distributed checkpoints, resumes automatically, and how much a
restart-from-checkpoint actually costs — with **real measurements from
Sunspot** (Intel PVC / XPU), not modeled estimates.

!!! info "Key API"
    - [`ezpz.examples._checkpoint`](../python/Code-Reference/examples/fsdp_tp.md) — DCP save/load helpers
    - `fsdp_tp` flags: `--ckpt-dir`, `--save-interval`, `--train-iters`, `--no-resume`
    - Metric: `train/restart_seconds`

## How it works

`fsdp_tp` uses PyTorch **Distributed Checkpoint (DCP)** — each rank writes
its own shard in parallel (no gather-to-rank-0 that would OOM at scale), so
it works identically under FSDP-only, HSDP, and 2D FSDP+TP.

- **Save** every `--save-interval` steps into `--ckpt-dir/step-<N>/`. A
  `.complete` marker is written **last**, so a checkpoint interrupted
  mid-save (e.g. by the failure you're recovering from) is skipped.
- **Resume** is automatic on startup: the newest complete checkpoint is
  loaded and training continues from its step — no flag needed (pass
  `--no-resume` to force a fresh run).

Because resume is automatic, it composes directly with
[`ezpz launch --auto-retry`](../cli/launch/index.md): a relaunch simply
resumes.

```bash
ezpz launch --auto-retry --np <N> -- \
  python3 -m ezpz.examples.fsdp_tp --model debug \
    --ckpt-dir ./ckpts --save-interval 100 --train-iters 3000
```

### Asynchronous checkpointing

By default a save is **synchronous** — the training loop blocks while every
rank writes its shards to the durable `--ckpt-dir`. At large model sizes that
stall recurs every `--save-interval` steps. Pass `--async-ckpt` to overlap the
write with training:

```bash
python3 -m ezpz.examples.fsdp_tp ... \
    --ckpt-dir /shared/ckpts --async-ckpt \
    --ckpt-stage-dir /tmp/ezpz-ckpt --save-interval 100
```

`--async-ckpt` uses `dcp.async_save`: the state dict is staged to CPU memory
**synchronously** (so it's safe to keep training the instant the call
returns; tracked as `train/ckpt_stage_seconds`), then written to fast
**node-local** `--ckpt-stage-dir` (default `/tmp/ezpz-ckpt-<jobid>`) by a
background thread, and finally **fanned out** to the durable `--ckpt-dir` on
shared FS. The fan-out copy runs on a background worker (`start_fanout`) and is
finalized — barrier + `.complete` marker — at the *next* save boundary
(`finalize_fanout`), so the expensive shared-FS write overlaps training rather
than blocking it.

!!! warning "Watch `stage + drain`, and mind the marker lag"
    Two things the naive reading gets wrong (both learned the hard way — see
    the [agpt-2b measurements](#at-realistic-scale-agpt-2b-a-23-gb-checkpoint)):
    the true per-save stall is `ckpt_stage_seconds + ckpt_drain_seconds`, not
    the cheap stage alone; and because the fan-out is backgrounded, a
    checkpoint's durable `.complete` marker lands one full `--save-interval`
    later, so an arbitrarily-timed crash can lose up to ~2 intervals of work
    (vs ~1 for a sync save). Recovery is never broken; worst-case recomputed
    work just grows by an interval. Shrink `--save-interval` to bound it.

!!! warning "Node-local staging is not durable"
    `--ckpt-stage-dir` (e.g. `/tmp`) is node-local and **not resumable on its
    own** — its shards are scattered per node and it carries no completion
    marker. Only the fanned-out `--ckpt-dir` copy on shared FS survives a node
    failure, and resume always reads from there. That's why `--async-ckpt`
    *requires* `--ckpt-dir`; `/tmp` is a staging tier, not the checkpoint.

## The experiment

To measure restart cost we ran two 3000-step jobs on **2 Sunspot nodes (24
XPU ranks, `tp=2`)**, checkpointing every 100 steps:

1. **Baseline** — no failures.
2. **Checkpoint Restart** — a background loop `SIGKILL`s the training ranks
   across all nodes every ~90 s (a real `pkill -9`; PALS then tears down the
   training `mpiexec`, each attempt exiting rc=137). A relaunch loop restarts
   on the same nodes and `fsdp_tp` auto-resumes from the last checkpoint.

![Training progress over time — baseline vs checkpoint restart](checkpoint-restart.png)

## Results

| | Baseline | Checkpoint Restart |
|---|---:|---:|
| Steps | 3000 | 3000 |
| Wall-clock | **5.81 min** | **8.04 min** |
| Failures | 0 | 4 (real SIGKILL) |
| Recovery overhead | — | **+2.23 min (≈38%)** |

Per failure (kill → PALS teardown → relaunch → DCP resume):

| # | resume @ step | lost steps | `restart_seconds` |
|---|---:|---:|---:|
| 1 | 801 | 71 | 10.40 |
| 2 | 1301 | 57 | 10.66 |
| 3 | 1801 | 57 | 10.79 |
| 4 | 2301 | 59 | 11.10 |

**What the numbers mean:**

- **Restart cost ≈ 10.4–11.1 s**, strikingly consistent. `train/restart_seconds`
  is timed from process entry (before `setup_torch`), so it captures the full
  cold path: process launch + distributed init + model build + `dcp.load` +
  first productive step.
- **Lost steps ≈ 57–71**, bounded by the 100-step checkpoint interval — you
  only ever redo work since the last checkpoint. Shrink `--save-interval` to
  reduce lost work (at the cost of more frequent save I/O).
- **≈ 33 s total per failure** (~11 s restart + ~22 s recomputing lost steps),
  which is the +2.23 min overhead across 4 failures.

### Async checkpointing under the same failures

Re-running the identical experiment with `--async-ckpt` (stage to `/tmp`, fan
out to shared FS) on the same 2 nodes / 24 XPU ranks:

![Baseline vs async checkpoint restart](checkpoint-restart-async.png)

| # | resume @ step | lost steps | `restart_seconds` |
|---|---:|---:|---:|
| 1 | 801 | 74 | 8.94 |
| 2 | 1301 | 57 | 8.94 |
| 3 | 1801 | 61 | 9.31 |
| 4 | 2301 | 56 | 9.30 |

- **Per-step stage stall: `train/ckpt_stage_seconds` ≈ 28 ms** (median) — the
  CPU stage barely touches the training thread. At this model's tiny checkpoint
  the *drain* (fan-out to shared FS) is also negligible, so async looks like a
  clean win here. **But that's an artifact of scale** — at 23 GB the drain
  becomes the dominant cost and flips the result (see the [agpt-2b
  section](#at-realistic-scale-agpt-2b-a-23-gb-checkpoint)). Don't generalize
  the debug-model stage number to real checkpoints.
- **Restart cost ≈ 8.9–9.3 s** — the resume path is the same as sync (init +
  `dcp.load`), so restart time is comparable (here marginally lower, within
  run-to-run noise).
- Recovery still works identically: every kill resumed from the last durable
  (fanned-out) checkpoint, never from the node-local `/tmp` staging copy.

### At realistic scale: agpt-2b, a 23 GB checkpoint

The debug-model numbers above make the *mechanism* clear, but the async win is
a rounding error there (28 ms stall). The whole point of async checkpointing is
large checkpoints, so we re-ran the experiment with **agpt-2b** (~2B params,
256K vocab) — a **23 GB sharded checkpoint** — on 2 Sunspot nodes / 24 XPU
ranks, `tp=2`. Three phases (baseline, sync restart, async restart), each with
**3 real SIGKILLs injected at steps 60/120/180** (right after a checkpoint).

Getting this right took two iterations, and the first was a cautionary tale.

#### First attempt: async was *slower*, and the obvious metric hid it

The first cut looked like a landslide for async: the logged
`ckpt_stage_seconds` (0.30 s) was ~12× smaller than the sync
`ckpt_save_seconds` (3.5 s). But async's **wall-clock was higher** (5.65 vs
5.42 min) even though *every logged metric* favored it. The paradox was the
tell: the cost was real but **untimed**.

An async save has two halves. The *stage* (copy state to host, kick off the
background write) is cheap. The *drain* — the fan-out of the full 23 GB from
node-local `/tmp` to shared FS — was a **blocking foreground copy** at the next
step start, landing between `train/dt` windows and separate from
`ckpt_stage_seconds`, so **no metric captured it**. Adding
`train/ckpt_drain_seconds` exposed it:

| per-save training-thread stall | sync | async (blocking drain) |
|---|---:|---:|
| stage | — | `ckpt_stage_seconds` 0.30 s |
| drain (/tmp→FS) | — | `ckpt_drain_seconds` **5.18 s** |
| blocking write | `ckpt_save_seconds` 3.54 s | — |
| **true total** | **≈3.5 s** | **≈5.5 s** |

So `/tmp` staging *added* work (write 23 GB to `/tmp`, then copy it to shared
FS) without moving the expensive write off the critical path — **async was
~1.5× slower per save than plain sync.** Lesson: `ckpt_stage_seconds` alone is
not the async cost; compare `stage + drain` against the sync save.

#### Fix: background the fan-out

The drain copy is pure per-rank file I/O with **no collectives**, so it is safe
to run on a background thread while training continues; only the completion
barrier + `.complete` marker must stay on the main thread. The save now calls
`start_fanout()` (submits the copy to a background worker, returns immediately)
and `finalize_fanout()` at the *next* save boundary (joins the already-finished
copy, then the collective barrier + marker). The expensive copy overlaps a full
save interval of training instead of blocking it:

![agpt-2b (23 GB) — sync vs async checkpoint restart](checkpoint-restart-agpt2b.png)

| per-save training-thread stall | sync | async (backgrounded) |
|---|---:|---:|
| stage | — | `ckpt_stage_seconds` 0.32 s |
| drain residual (barrier + marker) | — | `ckpt_drain_seconds` **0.65 s** |
| blocking write | `ckpt_save_seconds` 3.62 s | — |
| **true total** | **≈3.6 s** | **≈0.97 s** |

- **Backgrounding cut the async per-save stall from ~5.5 s to ~0.97 s — now
  ~3.7× *less* than sync**, not 1.5× more. The residual 0.65 s is just the
  cross-rank barrier + marker; the 23 GB copy is fully hidden. Validated on 24
  ranks with **no cross-thread collective deadlock** (the barriers stay on the
  main thread in lockstep).
- **Restart cost ≈ 37–43 s** (both sync and async), dominated by the 23 GB
  `dcp.load` + distributed init.

!!! warning "Durability tradeoff: the marker now lags one save interval"
    Backgrounding is not free. The `.complete` marker for a checkpoint saved at
    step N is stamped at the *next* save boundary (N + `save_interval`), so a
    checkpoint takes one full interval to become resumable. For an
    arbitrarily-timed crash (a real hardware failure, not synchronized to a
    save), the newest resumable checkpoint can therefore be **up to ~2 save
    intervals old**, vs ~1 for a synchronous save. Recovery is never broken —
    the previous complete checkpoint is always durable — but worst-case
    recomputed work grows by one interval. Shrink `--save-interval` to bound it.

!!! note "Reading the sawtooth: async's teeth look deeper — mostly an artifact"
    In the plot the async (blue) kills land at higher steps (≈80/142/200) and
    drop further back than sync's. That is **largely an artifact of this
    experiment's failure injector**, not async discarding more durable work:
    the injector waits for the `.complete` marker before killing, and that
    marker now lags ~1 interval, so it lets async run ~20 extra steps before
    the kill — steps that are then recomputed. Both sync and async in fact
    resume from the **identical** durable checkpoints (steps 60/120/180 →
    resume 61/121/181). The genuine durability lag (above) is real but bounded;
    this marker-gated experiment kills at the best-case moment and so does not
    measure it.

## Measuring it yourself

`fsdp_tp` logs a `RESUMED from step=N` line on resume and a
`train/restart_seconds` metric on the first post-resume step (into the
metrics JSONL and W&B). The driver script + a reusable plotter live in the
repo under `experiments/checkpoint-restart/`:

```bash
qsub experiments/checkpoint-restart/restart_experiment.pbs   # 2 nodes
python3 experiments/checkpoint-restart/plot_restart.py \
    expt_<jobid>/baseline/*/metrics-0.jsonl \
    expt_<jobid>/restart/*/*/metrics-0.jsonl \
    --out restart_plot.png --report restart_report.md
```

The agpt-2b sync-vs-async comparison above is the same driver run at scale with
a step-driven kill injector; its combined plotter takes all three phases:

```bash
python3 experiments/checkpoint-restart/plot_2b_comparison.py \
    --baseline expt_<jobid>/baseline/*/*/metrics-0.jsonl \
    --sync     expt_<jobid>/sync/*/*/metrics-0.jsonl \
    --async    expt_<jobid>/async/*/*/metrics-0.jsonl \
    --out agpt2b_restart.png --report agpt2b_restart_report.md
```

## Scope

This shows the two behaviors ezpz provides natively: a **baseline** and
**checkpoint restart** (fail → lose steps to the last checkpoint → resume).
Frameworks that recover *without* losing steps or *without* a process
restart — e.g. pause/resume or in-place elastic recovery (TorchFT-style) —
are separate systems not integrated into ezpz and are out of scope here.

Numbers above are from Sunspot: debug-model runs (job `12471687`), the
blocking-drain agpt-2b run (job `12471769`), and the backgrounded-fan-out
agpt-2b run (job `12471771`) — all 2 nodes / 24 XPU ranks. Absolute
`restart_seconds` grows with model size and node count (larger `dcp.load`,
longer init) — the debug→agpt-2b jump (≈10 s → ≈40 s) shows exactly that; the
*mechanism* is parallelism-agnostic since DCP is sharded.
