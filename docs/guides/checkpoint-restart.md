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
returns), then written to fast **node-local** `--ckpt-stage-dir` (default
`/tmp/ezpz-ckpt-<jobid>`) by a background thread, and finally **fanned out**
to the durable `--ckpt-dir` on shared FS. The only cost on the training
thread is the short staging copy (logged + tracked as
`train/ckpt_stage_seconds`).

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

- **Per-step stall from checkpointing: `train/ckpt_stage_seconds` ≈ 28 ms**
  (median) — the async stage barely touches the training thread; the shard
  write + fan-out happen off-thread. Compare to a synchronous save, which
  blocks the loop for the full write.
- **Restart cost ≈ 8.9–9.3 s** — the resume path is the same as sync (init +
  `dcp.load`), so restart time is comparable (here marginally lower, within
  run-to-run noise). Async's win is on the *save* side, not the restart side:
  it removes the per-interval training stall, which matters at scale where a
  synchronous shard write is seconds, not milliseconds.
- Recovery still works identically: every kill resumed from the last durable
  (fanned-out) checkpoint, never from the node-local `/tmp` staging copy.

### At realistic scale: agpt-2b, a 23 GB checkpoint

The debug-model numbers above make the *mechanism* clear, but the async win is
a rounding error there (28 ms stall). The whole point of async checkpointing is
large checkpoints, so we re-ran the experiment with **agpt-2b** (~2B params,
256K vocab) — a **23 GB sharded checkpoint** — on 2 Sunspot nodes / 24 XPU
ranks, `tp=2`. Three phases (baseline, sync restart, async restart), each with
**3 real SIGKILLs injected at steps 60/120/180** (right after a checkpoint), so
every tooth loses at most one save interval:

![agpt-2b (23 GB) — sync vs async checkpoint restart](checkpoint-restart-agpt2b.png)

| | Baseline | Sync restart | Async restart |
|---|---:|---:|---:|
| Steps | 240 | 240 | 240 |
| Wall-clock | **2.03 min** | **5.54 min** | **5.79 min** |
| Failures | 0 | 3 | 3 |
| Per-save stall (median) | — | `ckpt_save_seconds` **3.567 s** | `ckpt_stage_seconds` **0.301 s** |

At 23 GB the contrast the debug model couldn't show is now stark:

- **Per-step save stall: 3.567 s (sync) vs 0.301 s (async) — ≈12× less.** A
  synchronous save blocks the training loop for the *full* 23 GB shard write
  every interval; async only pays the short CPU-stage copy on the training
  thread and writes off-thread. This is exactly the regime async is for.
- **Restart cost ≈ 38–43 s** (both sync and async) — dominated by the 23 GB
  `dcp.load` + distributed init, ~4× the debug model's ~10 s, as expected when
  the checkpoint is ~40× larger.
- **Honest nuance:** async's *total wall-clock* here (5.79 min) is marginally
  **higher** than sync's (5.54 min), not lower. At only 240 steps the 3×~40 s
  restart cost dominates, and ~9 saves × ~3.3 s of stall savings (~30 s) don't
  fully offset run-to-run variance in the restart path. Async's guaranteed win
  is on the **per-step stall** (training smoothness / not blocking the loop);
  the wall-clock advantage compounds with more saves between failures, not
  fewer. Pick async when saves are frequent and large, not to speed up
  recovery.

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

Numbers above are from Sunspot: debug-model runs (job `12471687`) and the
agpt-2b / 23 GB run (job `12471756`), both 2 nodes / 24 XPU ranks. Absolute
`restart_seconds` grows with model size and node count (larger `dcp.load`,
longer init) — the debug→agpt-2b jump (≈10 s → ≈40 s) shows exactly that; the
*mechanism* is parallelism-agnostic since DCP is sharded.
