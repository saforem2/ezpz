# 📡 Experiment Tracking

The `Tracker` class fans out metric logging to one or more external backends
(Weights & Biases, CSV, or your own), with rank-aware defaults for distributed
training.

## Tracker vs History

ezpz has two complementary tracking systems:

| | `History` | `Tracker` |
|---|-----------|-----------|
| **Purpose** | In-process metric accumulation | External metric dispatch |
| **What it does** | Stores per-step values, computes distributed stats (min/max/mean/std), generates plots and reports | Sends metrics to external services (wandb, CSV files, custom backends) |
| **Output** | xarray Dataset, matplotlib/terminal plots, markdown report, JSONL | wandb dashboard, `metrics.csv`, `config.json` |
| **Use when** | You want to analyze, aggregate, or visualize metrics locally | You want metrics in an external dashboard or persisted to structured files |

They work well together — see [Using Both](#using-tracker-with-history) below.

## Quick Start

```python
from ezpz.tracker import setup_tracker

tracker = setup_tracker(
    project_name="my-project",
    backends="wandb,csv",
    outdir="outputs/run-001",
)

for step in range(100):
    loss = train_step(...)
    tracker.log({"loss": loss.item(), "step": step})

tracker.finish()
```

## Creating a Tracker

Use `setup_tracker()` to build a `Tracker` from a list of backend names:

```python
tracker = setup_tracker(
    project_name="my-project",    # passed to wandb
    backends="wandb,csv",         # comma-separated or list
    config={"lr": 1e-4},          # run-level hyperparameters
    outdir="./outputs",           # directory for file-based backends
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `project_name` | `None` | Project name for backends that support it (e.g. wandb) |
| `backends` | `"wandb"` | Comma-separated string or list of backend names |
| `config` | `None` | Run-level config dict (hyperparameters, env info) |
| `outdir` | `cwd` | Output directory for file-based backends |
| `rank` | auto | Distributed rank — auto-detected via `ezpz.distributed.get_rank()` |

Pass `backends="none"` to disable all tracking (returns a `NullTracker`).

## Backends

### Weights & Biases

The `WandbBackend` wraps `wandb.init` with rank-aware defaults:

- **Non-rank-0 processes** automatically get `mode="disabled"`
- Project name resolves from: argument > `WB_PROJECT` > `WANDB_PROJECT` > `WB_PROJECT_NAME` env vars
- Auto-logs system info (hostname, torch version, ezpz version)

```python
tracker = setup_tracker(
    project_name="my-project",
    backends="wandb",
    config={"lr": 1e-4, "batch_size": 32},
)
tracker.log({"loss": 0.42})
tracker.finish()
```

Access the underlying `wandb.Run` when needed:

```python
if tracker.wandb_run is not None:
    tracker.wandb_run.summary["best_loss"] = 0.01
```

### CSV

The `CSVBackend` writes metrics to `metrics.csv` and config to `config.json`:

```python
tracker = setup_tracker(backends="csv", outdir="./logs")
tracker.log({"loss": 0.5, "lr": 1e-4})
tracker.log({"loss": 0.3, "lr": 1e-4, "grad_norm": 0.8})  # new columns auto-extend
tracker.finish()
```

Output files:

| File | Content |
|------|---------|
| `metrics.csv` | One row per `log()` call, columns auto-extend as new keys appear |
| `config.json` | Merged config from `log_config()` calls |

The CSV backend also supports `log_table()` for writing separate tables:

```python
tracker.log_table("results", columns=["name", "score"], data=[["test", 0.95]])
# writes results.csv
```

### Custom Backends

Register your own backend by subclassing `TrackerBackend`:

```python
from ezpz.tracker import TrackerBackend, register_backend

class MyBackend(TrackerBackend):
    name = "my_backend"

    def __init__(self, **kwargs):
        self.entries = []

    def log(self, metrics, step=None, commit=True):
        self.entries.append(metrics)

    def log_config(self, config):
        pass

    def finish(self):
        pass

register_backend("my_backend", MyBackend)

# Now usable via setup_tracker
tracker = setup_tracker(backends="my_backend,csv", outdir="./logs")
```

Override the optional methods (`log_table`, `log_image`, `watch`) for richer
functionality.

## Using Tracker with History

`History` has built-in Tracker support — pass `backends` to the `History`
constructor and `update()` will automatically dispatch metrics to all
configured backends:

```python
import ezpz
from ezpz.history import History

rank = ezpz.setup_torch()

history = History(
    project_name="my-project",
    backends="wandb,csv",
    outdir="./outputs",
    config={"lr": 1e-4, "batch_size": 32},
)

for step in range(100):
    loss = train_step(...)
    metrics = {"step": step, "loss": loss.item()}
    summary = history.update(metrics, step=step)  # aggregates + dispatches
    logger.info(summary)

if rank == 0:
    history.finalize(outdir="./outputs")  # also calls tracker.finish()
```

Access backend-specific features via the `tracker` property:

```python
wb = history.tracker.get_backend("wandb")
if wb is not None:
    wb.watch(model, log="all")
```

The standalone `setup_tracker()` API remains available for cases where you
want metric dispatch without History's aggregation and reporting.

## Backend-Specific Features

Access a specific backend for features that don't apply to all backends:

```python
# Attach wandb gradient tracking
wb = tracker.get_backend("wandb")
if wb is not None:
    wb.watch(model, log="all")

# Log an image (wandb only)
tracker.log_image("sample", "outputs/sample.png", caption="Epoch 10")
```

All `Tracker` methods catch backend exceptions and log warnings — a failing
backend won't crash your training run.

## Environment Variables

| Variable | Effect |
|----------|--------|
| `EZPZ_TRACKER_BACKENDS` | Fallback backend list when `backends` arg is `None` |
| `WANDB_MODE` | Controls wandb mode (`disabled`, `offline`, `online`) |
| `WANDB_DISABLED` | Set to `1` to disable wandb entirely |
| `WB_PROJECT` / `WANDB_PROJECT` | Default wandb project name |
| `WB_PROJECT_NAME` | Alias for wandb project name |

## See Also

- [Metric Tracking (History)](./history.md) for in-process metric
  accumulation, distributed stats, and plotting
- [Quick Start](./quickstart.md) for minimal setup
- [Python API: `ezpz.tracker`](./python/Code-Reference/tracker.md) for the
  full API reference
