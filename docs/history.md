# 📊 Metric Tracking

The `History` class records, aggregates, and visualizes training metrics across
all distributed ranks, with optional Weights & Biases integration.

## Quick Start

```python
import ezpz
from ezpz.history import History

rank = ezpz.setup_torch()
history = History()

for step in range(100):
    loss = train_step(...)
    # update() returns a summary string suitable for logging
    summary = history.update({"step": step, "loss": loss.item()})
    logger.info(summary)  # e.g. "step=42 loss=0.123456"

if rank == 0:
    history.finalize(outdir="./outputs")
```

## Creating a `History`

```python
history = History(
    keys=["loss", "lr"],          # optional: pre-declare metric names
    distributed_history=True,     # aggregate stats across ranks (default: auto)
    report_dir="./outputs",       # where to write the markdown report
    jsonl_path="./metrics.jsonl", # per-step JSONL log
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `keys` | `None` | Metric names to pre-initialize |
| `distributed_history` | auto | Compute min/max/mean/std across ranks via all-reduce |
| `report_dir` | `OUTPUTS_DIR/history` | Directory for the markdown report |
| `report_enabled` | `True` | Generate a markdown report on `finalize()` |
| `jsonl_path` | `report_dir/{run_id}.jsonl` | Path for per-step JSONL log |
| `jsonl_overwrite` | `False` | Truncate existing JSONL file |

### Distributed history auto-detection

By default, distributed aggregation is **enabled** when `world_size <= 384`
and **disabled** above that threshold to avoid all-reduce overhead on very
large jobs (e.g. 32+ Aurora nodes or 96+ Polaris nodes). You can override
this with:

- `History(distributed_history=False)` to force off
- `EZPZ_NO_DISTRIBUTED_HISTORY=1` or `EZPZ_LOCAL_HISTORY=1` env vars

## `update()`: Record Metrics

```python
summary = history.update(
    {"loss": 0.42, "lr": 1e-3},
    precision=6,       # decimal places in summary string
    use_wandb=True,    # log to W&B if available
)
```

Each call to `update()`:

1. Appends values to the internal history dict
2. If `distributed_history=True`, computes **min, max, mean, std** across all
   ranks via `torch.distributed.all_reduce`
3. Logs to Weights & Biases (if configured)
4. Writes a JSONL entry to disk
5. Returns a formatted summary string: `"loss=0.420000 lr=0.001000"`

### Distributed statistics

For each scalar metric `"loss"`, distributed history creates:

| Key | Description |
|-----|-------------|
| `loss/mean` | Mean across all ranks |
| `loss/max` | Maximum across all ranks |
| `loss/min` | Minimum across all ranks |
| `loss/std` | Standard deviation across ranks |

These are computed every `update()` call and included in both the summary
string and W&B logs.

## `finalize()`: Save, Plot, Report

```python
dataset = history.finalize(
    outdir="./outputs",
    run_name="my-experiment",
    warmup=0.05,       # drop first 5% of samples
    num_chains=128,     # chains in ridge plots
)
```

`finalize()` produces:

| Output | Path |
|--------|------|
| xarray Dataset | `{outdir}/dataset.hdf5` (or `.nc` without h5py) |
| Matplotlib plots | `{outdir}/plots/mplot/*.png` |
| Terminal plots | `{outdir}/plots/tplot/*.txt` |
| Markdown report | `{outdir}/report.md` |
| JSONL log | `{outdir}/{run_id}.jsonl` |

Returns the xarray `Dataset` for further analysis.

??? example "Terminal plot output"

    `finalize()` generates text-based plots directly in the terminal:

    ```
                             loss [2025-08-26-075820]
          ┌─────────────────────────────────────────────────────┐
     11008┤████████████████████████████████▌▐███████████▐▐██████│
     10944┤████████████████████████████████▙▟███████████▟▟██████│
          └──┬──┬────┬─────┬───┬─────┬───┬────┬─────┬──────┬────┘
            44 84   185   296 366   471 551  647   750    867
     loss                          iter
    ```

    When distributed history is enabled, summary plots show min/max/mean/std
    overlaid with distinct markers (`·` mean, `-` min, `+` max, `*` std).

## Weights & Biases Integration

Call `ezpz.setup_wandb()` **before** creating `History()`:

```python
ezpz.setup_wandb(project_name="my-project")
history = History()
```

With W&B active:

- Every `update()` call logs metrics via `wandb.log()`
- `finalize()` uploads the full training history as a `wandb.Table`
- Matplotlib plots are logged as `wandb.Image` artifacts

Set `WANDB_DISABLED=1` or pass `use_wandb=False` to `update()` to skip.

## Environment Variables

| Variable | Effect |
|----------|--------|
| `EZPZ_NO_DISTRIBUTED_HISTORY` | Disable distributed aggregation |
| `EZPZ_LOCAL_HISTORY` | Disable distributed aggregation (alias) |
| `EZPZ_TPLOT_MARKER` | Marker style for terminal plots (`braille`, `fhd`, `hd`) |
| `EZPZ_TPLOT_RAW_MARKER` | Marker for raw data series in terminal plots |
| `EZPZ_TPLOT_TYPE` | Default plot type (`line`, `hist`) |
| `EZPZ_TPLOT_MAX_HEIGHT` | Max height for terminal plots |
| `EZPZ_TPLOT_MAX_WIDTH` | Max width for terminal plots |

## `StopWatch`: Timing Context Manager

```python
from ezpz.history import StopWatch

with StopWatch("forward pass", wbtag="timing/forward"):
    output = model(batch)
```

Logs elapsed time to the logger and optionally to W&B. Useful for profiling
individual phases within your training loop.

## See Also

- [Quick Start: Track Metrics](./quickstart.md#track-metrics) for minimal usage
- [Reference: Complete Example](./reference.md#complete-example-with-history)
  for a full runnable script with terminal output
- [Python API: `ezpz.history`](./python/Code-Reference/history.md) for the
  full API reference
