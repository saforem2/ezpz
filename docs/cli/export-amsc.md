# `ezpz export-amsc`

Turn a finished run directory into one CSV row for the
[AmSC at-scale benchmarks](https://gitlab.com/amsc2/ai-services/at-scale-services/amsc-atscale-benchmarks),
whose [dashboard](https://amsc-atscale-benchmarks-42b223.gitlab.io/)
reads `benchmarks/<category>/<name>/results/runs.csv`.

```bash
ezpz export-amsc outputs/ezpz.examples.fsdp_tp/2026-08-11-171117 \
    --config agpt-2b/bs1/seq2048/tp1
```

```text
timestamp,system,config,nodes,gpus,status,wall_time_sec,throughput_tokens_per_sec,...
2026-08-11T17:12:30Z,SunSpot,agpt-2b/bs1/seq2048/tp1,1,12,pass,12.739,61216.953,...
```

Append straight into a checked-out benchmark repo — the header is
written only when the file is new:

```bash
ezpz export-amsc <run-dir> --config agpt-2b/bs1/seq2048/tp1 \
    --append benchmarks/training/llm-finetuning/results/runs.csv
```

## Where the numbers come from

| Column | ezpz metric | Note |
|---|---|---|
| `throughput_tokens_per_sec` | `train/tps` | **global** across ranks |
| `throughput_tokens_per_sec_per_gpu` | `train/tps_per_gpu` | per-GPU (torchtitan's `tgs`) |
| `mfu` | `train/mfu` | **percent (0–100)**, per-GPU |
| `tflops` | `train/tflops` | **per-GPU**, not aggregate |
| `final_loss` | last `train/loss` | |
| `wall_time_sec` | `sum(train/dt)` | excludes setup — see below |
| `nodes` / `gpus` | `WORLD_SIZE_IN_USE` | what ran, not what was allocated |

## Three defaults that were measured, not chosen

!!! warning "Throughput is a post-warmup median"

    Step 1 is routinely an order of magnitude slower — compile,
    allocator warmup, lazy init. A real `agpt-2b` series from Sunspot
    (which `get_machine()` reports as the literal `SunSpot`, hence the
    `system` value above):

    ```text
    1045, 34601, 34686, 34715, 34833, 34650
    ```

    Averaging all six reports **29,088** against a true **~34,700** —
    a 16% understatement. The default drops one warmup step and takes
    the median, which is also robust to a mid-run straggler. Change
    with `--warmup` / `--reducer {median,mean,max,min,last}`.

!!! warning "`wall_time_sec` undercounts the job"

    It sums instrumented step time, so model construction, dataset
    load and distributed init are excluded. The JSONL timestamp span
    is *worse* — records exist only for logged steps, so it covered
    1.18 s of a 6.76 s run — and `timings/*` never reach the JSONL at
    all (they go to `tracker.log()`). Pass `--wall-time-sec` with the
    scheduler's figure when you need a true wall time.

!!! note "`nodes`/`gpus` describe the run, not the allocation"

    `NUM_NODES`/`NGPUS` come from the scheduler. A 1-node
    configuration run inside a 4-node allocation reports `NUM_NODES=4,
    NGPUS=48` while only 12 ranks participate — publishing that would
    make a 1-node result look like a catastrophically slow 4-node one.
    The exporter derives from `WORLD_SIZE_IN_USE` and falls back to the
    allocation only when that is unavailable.

## Provenance

Facility, node and GPU counts are recovered in this order:

1. `run_info.json` — written by `History.finalize()` (preferred)
2. `config.json` — only present under the non-default `csv` tracker backend
3. the `### Distributed` section of `report-*.md` — covers older runs

If none is available the command **errors naming the flags to pass**
rather than guessing. In particular `gpus` is never inferred from the
number of rank files: those count ranks, which equals GPUs only at one
rank per GPU, and a wrong GPU count corrupts every per-GPU comparison
on the dashboard.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--config` | *required* | Configuration label, e.g. `agpt-2b/bs1/seq2048/tp1` |
| `--system` / `--nodes` / `--gpus` | auto | Override the detected provenance |
| `--status {pass,fail}` | `pass` | **Cannot be inferred** from a run directory |
| `--error` | empty | Note for a failed run |
| `--warmup` | `1` | Leading steps dropped before reducing |
| `--reducer` | `median` | `median`, `mean`, `max`, `min`, `last` |
| `--wall-time-sec` | derived | Override with the scheduler's figure |
| `--append PATH` | stdout | Append to a `runs.csv` |
| `--no-header` | off | Omit the header on stdout |
