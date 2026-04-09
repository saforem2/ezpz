# 👀 Tracker

The `Tracker` is the low-level multi-backend dispatch layer that fans
out metric logging to one or more backends (Weights & Biases, MLflow,
CSV, or custom). It is **rank-aware**: non-rank-0 processes automatically
skip external I/O.

Most users should use [`History`](./history.md) instead — it owns a
`Tracker` internally and adds distributed statistics, JSONL logging,
plots, and reports on top. Use `Tracker` directly only when you want
backend dispatch without the rest of the `History` machinery.

## Standalone Usage

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

## Built-in Backends

The built-in backends are: **wandb**, **mlflow**, **csv**, and **none**.

For detailed configuration, authentication setup, and usage examples for
each backend, see the
[Experiment Tracking > Backends](./history.md#backends) guide.

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
