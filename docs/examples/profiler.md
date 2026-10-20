# Profiling with the PyTorch Profiler

Capture a `torch.profiler` trace from a distributed run on Aurora,
Sunspot, Polaris, or Perlmutter — same command on every system.

!!! info "Key API Functions"

    - [`get_torch_profiler()`][ezpz.profile.get_torch_profiler] — device-aware `torch.profiler.profile` wrapper
    - [`profiling_context_from_args()`][ezpz.profile.profiling_context_from_args] — build the context from CLI flags
    - [`add_profiling_args()`][ezpz.cli.flags.add_profiling_args] — the shared profiler flag-set

See: 🐍 [source](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/profiler.py)

```bash
ezpz launch python3 -m ezpz.examples.profiler --profile
```

That writes one Chrome trace per completed profiling *cycle* (not per
step) plus a key-averages table in the log. With the default schedule a
cycle is 6 steps, so `--steps 20` yields 3 traces per profiled rank —
verified on Polaris: rank0 wrote `step6`, `step12`, `step18`. Load the JSON in `chrome://tracing`,
[Perfetto](https://ui.perfetto.dev), or TensorBoard.

## What it does

`ezpz.examples.profiler` is a synthetic MLP training loop with nothing in
it but the profiler wiring:

```python
from ezpz.profile import profiling_context_from_args

with profiling_context_from_args(args, outdir) as prof:
    for step in range(args.steps):
        ...                      # forward / backward / optimizer step
        if prof is not None:
            prof.step()          # advance the profiler schedule
```

Three things are worth knowing.

**`prof.step()` is required.** The schedule
(`wait`/`warmup`/`active`/`repeat`) only advances when you call it. Leave
it out and the profiler stays in `wait` forever and writes no trace —
with no error. That is the most common way profiling appears "broken".

**Without `--profile`, `prof` is `None`.**
`profiling_context_from_args()` returns a
[`nullcontext`](https://docs.python.org/3/library/contextlib.html#contextlib.nullcontext),
so the unprofiled path carries no overhead and needs no separate branch.

**Device selection is automatic.**
[`get_torch_profiler()`][ezpz.profile.get_torch_profiler] picks the
activity set from what is available:

| System | Accelerator | Activities |
|---|---|---|
| Aurora, Sunspot | Intel PVC | `CPU` + `XPU` |
| Polaris, Perlmutter | NVIDIA | `CPU` + `CUDA` |
| Laptop / login node | — | `CPU` |

## Enough steps to produce a trace

The default schedule is `wait=1, warmup=2, active=3, repeat=5`, so the
first trace lands only after **6 steps**. Running fewer produces no
output at all; the example warns when that happens rather than exiting
silently:

```
--profile was passed but no trace was written. The schedule needs
wait+warmup+active (default 1+2+3=6) steps before the first trace;
--steps is 2.
```

## Common modifications

- **Shape the schedule** — `--pytorch-profiler-wait`,
  `--pytorch-profiler-warmup`, `--pytorch-profiler-active`,
  `--pytorch-profiler-repeat`. Profile one window late in the run with
  `--pytorch-profiler-wait 50 --pytorch-profiler-active 5
  --pytorch-profiler-repeat 1`.

- **Restrict to one rank** — **every rank profiles by default**
  (`rank_zero_only=False`). Each writes its own trace: an 8-rank Polaris
  run produced 24 files, and 12 ranks/node on Aurora scales that up
  fast. Pass `--rank-zero-only` to profile rank 0 alone, which is
  usually what you want.

- **Trim trace size** — `--no-with-stack` removes Python stacks (the
  biggest contributor), `--no-record-shapes` drops tensor shapes,
  `--no-profile-memory` drops allocation tracking.

- **Resize the work** — `--steps`, `--batch-size`, `--hidden-size`,
  `--layers` control the synthetic model, so you can make the profiled
  region as cheap or as expensive as you need.

- **Sampling profiler instead** — `--pyinstrument-profiler` swaps in
  [pyinstrument](https://pyinstrument.readthedocs.io/) for a
  wall-clock-sampled Python call tree. Useful when the bottleneck is
  Python-side rather than in kernels.

## Profiling the real examples

Every training example accepts the same flags, so you can profile actual
work instead of a synthetic loop:

```bash
ezpz launch python3 -m ezpz.examples.fsdp      --model small --profile
ezpz launch python3 -m ezpz.examples.fsdp_tp   --model small --tp 2 --profile
ezpz launch python3 -m ezpz.examples.vit       --model small --profile
ezpz launch python3 -m ezpz.examples.diffusion --model small --profile
```

!!! warning "`fsdp_tp` + `--profile` does not work everywhere ([#275](https://github.com/saforem2/ezpz/issues/275))"

    Measured on `fsdp_tp --model small --tp 2`, 2 nodes:

    | system | result |
    |---|---|
    | Polaris | works |
    | **Aurora** | `RecursionError`, **0 traces** — add `--no-with-stack` |
    | **Perlmutter** | **hangs**, killed at timeout, 0 traces — no workaround known |

    **Aurora.** `p.key_averages()` walks recorded stacks recursively in
    torch's `profiler_util.py`, and FSDP2 + TP stacks exceed Python's
    1000-frame limit. `--no-with-stack` fixes it (job `8881403`: 0
    traces → 5). Raising `sys.setrecursionlimit()` does **not** — it
    trades the crash for a hang, which is worse.

    **Perlmutter.** `--profile` alone is enough; a full 2×2 (job
    `59132774`) shows both profiled arms hanging and both unprofiled
    arms passing, on either transport. There is no flag that avoids it
    today.

    The small `ezpz.examples.profiler` loop above profiles fine on every
    system — this is specific to deep FSDP2 + TP stacks.

The flag-set comes from
[`add_profiling_args()`][ezpz.cli.flags.add_profiling_args], and each
module calls `profiling_context_from_args()` the same way — so the
schedule flags above behave identically everywhere.

!!! warning "The flag is `--profile`, not `--pytorch-profiler`"

    `--profile` sets the `pytorch_profiler` attribute. There is no
    `--pytorch-profiler` flag; the `--pytorch-profiler-*` options only
    configure the schedule.

## Output

Traces are written next to the run's other outputs, one per completed
cycle per profiled rank:

```
outputs/ezpz.examples.profiler/<timestamp>/
├── torch-profiler-rank0-step6-<timestamp>.json
├── torch-profiler-rank0-step12-<timestamp>.json
└── torch-profiler-rank0-step18-<timestamp>.json
```

An 8-rank Polaris run at `--steps 20` produced 24 files this way (3
cycles x 8 ranks) — see `--rank-zero-only` above to cut that to 3.

Each file is a Chrome trace. The key-averages table is logged inline,
sorted by device time on accelerators and by CPU time otherwise.

## On Aurora

Aurora also has Intel's `unitrace`, which profiles at the Level Zero /
oneCCL layer — lower-level than `torch.profiler` and better suited to
questions about SYCL kernels or collective behavior. See
[Profiling Deep Learning Applications](https://docs.alcf.anl.gov/aurora/data-science/profiling_dl/)
in the ALCF user guides. The two are complementary: `torch.profiler`
attributes time to PyTorch operators and Python frames, `unitrace` to
device-level activity.
