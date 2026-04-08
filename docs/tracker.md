# 👀 Tracker

The `Tracker` is a multi-backend experiment tracking system that fans
out metric logging to one or more backends (Weights & Biases, MLflow,
CSV, or custom). It is **rank-aware**: non-rank-0 processes automatically
skip external I/O.

You can use the tracker standalone or let `History` manage it for you
(see [Metric Tracking](./history.md)).

## Quick Start

```python
from ezpz.tracker import setup_tracker

tracker = setup_tracker(
    project_name="my-project",
    backends="wandb,mlflow,csv",
    outdir="./outputs",
)

for step in range(100):
    loss = train_step()
    tracker.log({"loss": loss, "lr": 1e-4}, step=step)

tracker.finish()
```

Or let `History` own the tracker:

```python
history = ezpz.History(
    project_name="my-project",
    backends="wandb,mlflow,csv",
)
# history.update() automatically logs to all backends
```

## Backends

### Weights & Biases

Wraps `wandb.init` with rank-aware, env-var-respecting logic.

- **Rank 0** initializes a real run (online/offline per `WANDB_MODE`)
- **Rank != 0** gets `mode="disabled"` — no network calls, no duplicate runs
- Project name resolved from: `project_name` arg > `WB_PROJECT` >
  `WANDB_PROJECT` > `WB_PROJECT_NAME` > script-derived default

### MLflow

Built-in backend that logs to an [MLflow Tracking](https://mlflow.org)
server or local filesystem.

```bash
# Enable MLflow tracking
EZPZ_TRACKER_BACKENDS=mlflow ezpz launch python3 -m ezpz.examples.vit

# Use alongside wandb
EZPZ_TRACKER_BACKENDS=wandb,mlflow ezpz launch python3 -m ezpz.examples.vit
```

**Features:**

- **Automatic time-series**: Step counter auto-increments so MLflow shows
  line charts (not bar charts) by default
- **System metrics**: CPU/GPU/memory usage logged automatically via
  `mlflow.enable_system_metrics_logging()` (when using native Bearer auth)
- **Environment params**: Hostname, device, world size, git branch, torch
  version, etc. logged under `ezpz.*` prefix
- **User config**: Logged under `config.*` prefix (nested dicts flattened
  with dot-separated keys)
- **Metric grouping**: When distributed stats are present (`loss/mean`,
  `/min`, `/max`, `/std`), the raw per-rank value is renamed to
  `loss/local` so MLflow groups them under a collapsible `loss/` section
- **Artifact uploads**: On `finalize()`, JSONL logs, markdown reports,
  plots, and datasets are uploaded as run artifacts
- **Rank-aware**: Only rank 0 creates a run; all other ranks are silent no-ops

**Experiment name resolution** (first match wins):

1. `project_name` argument
2. `MLFLOW_EXPERIMENT_NAME` env var
3. `WB_PROJECT` / `WANDB_PROJECT` / `WB_PROJECT_NAME` env vars
4. Auto-derived from script: `ezpz.{parent}.{stem}` (e.g. `ezpz.examples.vit`)

**Authentication:**

| Env var | Auth method |
|---------|-------------|
| `MLFLOW_TRACKING_TOKEN` | Bearer token (MLflow native) |
| `AMSC_API_KEY` | `X-API-Key` header (for AMSC servers) |

Credentials are loaded automatically from dotenv files:

1. **`~/.amsc.env`** — user-level credentials (loaded first)
2. **Project `.env`** — project-level overrides (loaded second)

Example `~/.amsc.env`:

```bash
AMSC_API_KEY=your-api-key-here
MLFLOW_TRACKING_URI=https://mlflow.american-science-cloud.org
MLFLOW_TRACKING_INSECURE_TLS=true
```

!!! tip "`python-dotenv` not installed?"

    Set the variables directly in your shell or job script.
    Dotenv loading is a convenience, not a requirement.

### CSV

Writes `metrics.csv` (one row per `log()` call) and `config.json` to
the output directory.

- Columns auto-extend as new metric keys appear
- `log_table()` writes separate CSV files (e.g. `training_history.csv`)
- **Rank 0 only** — non-rank-0 processes buffer rows in memory but skip
  all file I/O

### None

Pass `backends="none"` (or set `EZPZ_TRACKER_BACKENDS=none`) to disable
tracking entirely. Returns a `NullTracker` where all methods are no-ops.

## Environment Variable

Set `EZPZ_TRACKER_BACKENDS` to configure backends without changing code:

```bash
EZPZ_TRACKER_BACKENDS=wandb,csv ezpz launch python3 -m my_app.train
EZPZ_TRACKER_BACKENDS=mlflow ezpz launch python3 -m my_app.train
EZPZ_TRACKER_BACKENDS=wandb,mlflow,csv ezpz launch python3 -m my_app.train
```

This works with both `setup_tracker()` and `History()`. When
`backends=` is not passed explicitly, the env var is used as the
fallback (default: `wandb`).

See [Configuration > Experiment Tracking](./configuration.md#experiment-tracking)
for the full env var reference.

## Custom Backends

Subclass `TrackerBackend` and register it:

```python
from ezpz.tracker import TrackerBackend, register_backend

class MyBackend(TrackerBackend):
    name = "my-tracker"

    def __init__(self, project_name=None, config=None, **kwargs):
        # Initialize your tracker here
        ...

    def log(self, metrics, step=None, commit=True):
        # Log metrics to your tracker
        ...

    def log_config(self, config):
        # Log hyperparameters / run config
        ...

    def finish(self):
        # Clean up resources
        ...

register_backend("my-tracker", MyBackend)
```

Then use it like any built-in backend:

```python
tracker = setup_tracker(backends="my-tracker,csv")
```

## Error Isolation

Backend errors are isolated — if one backend fails, the others still
receive the call. A warning is logged but your training run is never
interrupted by a tracking failure.

## API Reference

See [`ezpz.tracker`](./python/Code-Reference/tracker.md) for the full
API documentation.
