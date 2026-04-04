# Tracker

The `Tracker` is a multi-backend experiment tracking system that fans
out metric logging to one or more backends (Weights & Biases, CSV, or
custom). It is **rank-aware**: non-rank-0 processes automatically skip
external I/O.

You can use the tracker standalone or let `History` manage it for you
(see [Metric Tracking](./history.md)).

## Quick Start

```python
from ezpz.tracker import setup_tracker

tracker = setup_tracker(
    project_name="my-project",
    backends="wandb,csv",
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
    backends="wandb,csv",
)
# history.update() automatically logs to all backends
```

## Backends

### Weights & Biases

Wraps `wandb.init` with rank-aware, env-var-respecting logic.

- **Rank 0** initializes a real run (online/offline per `WANDB_MODE`)
- **Rank != 0** gets `mode="disabled"` — no network calls, no duplicate runs
- Project name resolved from: `project_name` arg > `WB_PROJECT` >
  `WANDB_PROJECT` > `WB_PROJECT_NAME` env vars

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
```

This works with both `setup_tracker()` and `History()`. When
`backends=` is not passed explicitly, the env var is used as the
fallback (default: `wandb`).

## Custom Backends

Subclass `TrackerBackend` and register it:

```python
from ezpz.tracker import TrackerBackend, register_backend

class MLflowBackend(TrackerBackend):
    name = "mlflow"

    def __init__(self, **kwargs):
        import mlflow
        self.run = mlflow.start_run()

    def log(self, metrics, step=None, commit=True):
        import mlflow
        mlflow.log_metrics(metrics, step=step)

    def log_config(self, config):
        import mlflow
        mlflow.log_params(config)

    def finish(self):
        import mlflow
        mlflow.end_run()

register_backend("mlflow", MLflowBackend)
```

Then use it like any built-in backend:

```python
tracker = setup_tracker(backends="mlflow,csv")
```

## Error Isolation

Backend errors are isolated — if one backend fails, the others still
receive the call. A warning is logged but your training run is never
interrupted by a tracking failure.

## API Reference

See [`ezpz.tracker`](./python/Code-Reference/tracker.md) for the full
API documentation.
