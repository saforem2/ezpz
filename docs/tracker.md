# Tracker API

The `Tracker` class is a multi-backend metric dispatcher used internally
by [`History`](./history.md). For most use cases, create a `History` with
`backends=` and let it manage the tracker automatically — see
[Experiment Tracking](./history.md).

## Standalone Usage

If you need metric dispatch without History's accumulation and reporting,
use `setup_tracker()` directly:

```python
from ezpz.tracker import setup_tracker

tracker = setup_tracker(
    project_name="my-project",
    backends="wandb,csv",
    config={"lr": 1e-4},
    outdir="./outputs",
)

for step in range(100):
    loss = train_step(...)
    tracker.log({"loss": loss.item()}, step=step)

tracker.finish()
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `project_name` | `None` | Project name for backends that support it |
| `backends` | `"wandb"` | Comma-separated string or list of backend names |
| `config` | `None` | Run-level config dict |
| `outdir` | `cwd` | Output directory for file-based backends |
| `rank` | auto | Distributed rank (auto-detected) |

Pass `backends="none"` to disable all tracking (returns a `NullTracker`).

## See Also

- [Experiment Tracking](./history.md) for the full guide (backends, custom
  backends, distributed stats, finalization)
- [Python API: `ezpz.tracker`](./python/Code-Reference/tracker.md) for the
  complete API reference
