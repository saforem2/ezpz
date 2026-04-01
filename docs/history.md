# 📊 Experiment Tracking

The `History` class handles the full experiment tracking lifecycle:
recording metrics, computing distributed statistics, dispatching to
external backends (Weights & Biases, CSV, or custom), and producing
plots and reports at the end of a run.

## Quick Start

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
    summary = history.update({"loss": loss.item()}, step=step)
    logger.info(summary)  # "loss=0.420000"

if rank == 0:
    history.finalize(outdir="./outputs")
```

`History()` with no backend arguments works identically for local-only
tracking — metrics are still accumulated, plotted, and saved to JSONL,
but nothing is dispatched externally.

## Constructor

```python
history = History(
    # --- Local outputs ---
    distributed_history=True,      # aggregate stats across ranks (default: auto)
    report_dir="./outputs",        # markdown report directory
    jsonl_path="./metrics.jsonl",  # per-step JSONL log

    # --- External backends ---
    project_name="my-project",     # passed to wandb
    backends="wandb,csv",          # comma-separated or list
    config={"lr": 1e-4},           # run-level hyperparameters
    outdir="./outputs",            # directory for file-based backends
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `distributed_history` | auto | Compute min/max/mean/std across ranks via all-reduce |
| `report_dir` | `./outputs/history` | Directory for the markdown report |
| `report_enabled` | `True` | Generate a markdown report on `finalize()` |
| `jsonl_path` | auto | Path for per-step JSONL log (defaults to `{report_dir}/{timestamp}.jsonl`) |
| `jsonl_overwrite` | `False` | Truncate existing JSONL file |
| `project_name` | `None` | Project name for backends that support it (e.g. wandb) |
| `backends` | `None` | Comma-separated string or list of backend names |
| `config` | `None` | Run-level config dict logged via the tracker on init |
| `outdir` | `None` | Output directory for file-based backends (e.g. CSV) |
| `tracker` | `None` | Inject a pre-built `Tracker` instance directly |

### Distributed auto-detection

Distributed aggregation is **enabled** when `world_size <= 384`
and **disabled** above that threshold to avoid all-reduce overhead on
very large jobs. Override with:

- `History(distributed_history=False)`
- `EZPZ_NO_DISTRIBUTED_HISTORY=1` or `EZPZ_LOCAL_HISTORY=1` env vars

## Recording Metrics

```python
summary = history.update(
    {"loss": 0.42, "lr": 1e-3},
    step=42,           # forwarded to tracker backends
    precision=6,       # decimal places in summary string
)
```

Each call to `update()`:

1. Appends values to the internal history dict
2. Computes **min, max, mean, std** across all ranks (when distributed)
3. Dispatches metrics to all configured backends
4. Writes a JSONL entry to disk
5. Returns a summary string: `"loss=0.420000 lr=0.001000"`

### Distributed statistics

For each scalar metric `"loss"`, distributed history creates:

| Key | Value |
|-----|-------|
| `loss/mean` | Mean across all ranks |
| `loss/max` | Maximum across all ranks |
| `loss/min` | Minimum across all ranks |
| `loss/std` | Standard deviation across ranks |

## Backends

Backends control where metrics are dispatched when `update()` is called.
Pass one or more backend names via `backends=` in the constructor.

### Weights & Biases

```python
history = History(
    project_name="my-project",
    backends="wandb",
    config={"lr": 1e-4, "batch_size": 32},
)
```

The wandb backend:

- Disables itself on non-rank-0 processes automatically
- Resolves project name from: argument > `WB_PROJECT` > `WANDB_PROJECT` env vars
- Auto-logs system info (hostname, torch version, ezpz version)
- Logs metrics on each `update()`, uploads training history table on `finalize()`
- Logs matplotlib plots as image artifacts

Set `WANDB_DISABLED=1` or `backends="none"` to disable.

### CSV

```python
history = History(backends="csv", outdir="./logs")

history.update({"loss": 0.5, "lr": 1e-4})
history.update({"loss": 0.3, "lr": 1e-4, "grad_norm": 0.8})  # columns auto-extend
```

The CSV backend writes to the `outdir`:

| File | Content |
|------|---------|
| `metrics.csv` | One row per `update()` call, columns auto-extend as new keys appear |
| `config.json` | Merged config from the `config=` constructor argument |

### Multiple Backends

Combine backends to dispatch everywhere at once:

```python
history = History(
    project_name="my-project",
    backends="wandb,csv",
    outdir="./outputs",
    config={"lr": 1e-4, "batch_size": 32},
)
```

Every `update()` call fans out to all backends. A failing backend logs a
warning but never crashes your training run.

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

# Now usable in History
history = History(backends="my_backend,csv", outdir="./logs")
```

Override the optional methods (`log_table`, `log_image`, `watch`) for
richer functionality.

### Accessing Backend-Specific Features

Use the `tracker` property to reach backend-specific APIs:

```python
# Attach wandb gradient tracking
wb = history.tracker.get_backend("wandb")
if wb is not None:
    wb.watch(model, log="all")

# Access the underlying wandb.Run
if history.tracker.wandb_run is not None:
    history.tracker.wandb_run.summary["best_loss"] = 0.01

# Log an image (wandb only)
history.tracker.log_image("sample", "outputs/sample.png", caption="Epoch 10")
```

## Finalization

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
| JSONL log | `{outdir}/{timestamp}.jsonl` |

It also uploads the training history table to any active backends and
calls `tracker.finish()` to flush and close all backend connections.
Returns the xarray `Dataset` for further analysis.

??? example "Terminal plot output"

    `finalize()` generates text-based plots directly in the terminal.
    Individual metric plots show raw values, while summary plots overlay
    min/max/mean/std with distinct markers (`·` mean, `-` min, `+` max,
    `*` std):

    ```
                              loss                                                      loss/min
    ┌──────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
1.73┤⡇                                                     │1.73┤-                                                     │
    │⢱                                                     │1.45┤--                                                    │
1.45┤⢸                                                     │1.18┤ -                                                    │
    │⢸                                                     │0.91┤ -                                                    │
1.18┤⢸⡄                                                    │    │ --                                                   │
    │ ⢇                                                    │0.63┤  -------  -                                          │
    │ ⢸⡆                                                   │0.36┤   - -------------------------------- ----- ------- --│
0.91┤ ⠈⡇                                                   │0.08┤                  - -    --- -------------------------│
    │  ⣇⡄ ⢰                                                │    └┬────────────┬─────────────┬────────────┬────────────┬┘
0.63┤  ⢿⣷⠾⣸⣼⢠⡆        ⢀                                    │    1.0         49.2          97.5         145.8      194.0
    │   ⠈ ⣿⠙⣿⣿⣀⢰⡇ ⡀   ⢸ ⣠  ⣠⢀                            ⢀ │loss/min                      iter
0.36┤     ⠈ ⠙⠈⣿⢼⣧⠺⣷⣴⠷⣠⢸⢠⡏⡆⣇⣿⡸⣀⣠⣦⡆⣀⣤⣠    ⡀⡇  ⣀⡄    ⢀ ⢠⡆ ⡀ ⣿ │                            loss/std
    │           ⠙ ⠈⢻ ⠸⠙⢼⠃⣿⠘⠙⠇⠘⣿⠟⢿⠘⢿⠻⣤⣠⣀⣸⡧⣧⠶⠙⣿⣧⡴⣆⣴⣧⣿⣠⢸⡇⣼⡇⢀⣿⡄│     ┌─────────────────────────────────────────────────────┐
0.08┤                    ⠈    ⠈ ⠈    ⠁⠈⠙ ⠈  ⠙⠘⠇⠘⠟⠉⠘⠋⠋⠹⡟⠣⢾⡏⠋│0.174┤   *                                                 │
    └┬────────────┬─────────────┬────────────┬────────────┬┘0.145┤   * *                                               │
    1.0         49.2          97.5         145.8      194.0 0.116┤   * ** *  *                                         │
loss                          iter                          0.087┤  ** ****  *     *    **    *      *                 │
                            loss/mean                            │ ********  **  *****  **  *** *   ***      *  ***    │
    ┌──────────────────────────────────────────────────────┐0.058┤********** ********** ** *************** *** **** *  │
1.77┤·                                                     │0.029┤** ********************************************** ***│
    │·                                                     │0.000┤    ***  *  ** *   * * *  ** ** ****  ***   * **** **│
1.49┤·                                                     │     └┬────────────┬────────────┬────────────┬────────────┬┘
    │ ·                                                    │     1.0         49.2         97.5         145.8      194.0
    │ ·                                                    │loss/std                      iter
1.22┤ ·                                                    │                            loss/max
    │ ·                                                    │    ┌──────────────────────────────────────────────────────┐
0.94┤ ··                                                   │1.81┤+                                                     │
    │  ·                                                   │1.53┤++                                                    │
0.67┤  ···                                                 │1.25┤ +                                                    │
    │  ·····    ·                                          │0.97┤ ++                                                   │
    │      ········    ·                                   │    │  +  +                                                │
0.39┤       · ················ ····    · ·   ·       ·   · │0.69┤  ++++++++ ++     +                                   │
    │             ······ ·· ·······························│0.41┤       ++++++++++++++++++++++++++++++++++++++++++++ + │
0.12┤                                ···      ··· · ·······│0.13┤                + +    + +++ +++++++++ +++++++++++++++│
    └┬────────────┬─────────────┬────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
    1.0         49.2          97.5         145.8      194.0     1.0         49.2          97.5         145.8      194.0
loss/mean                     iter                          loss/max                      iter
    ```

    Combined summary plot with all statistics overlaid:

    ```
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
1.81┤ ++ loss/max                                                                                                      │
    │ -- loss/min                                                                                                      │
    │ ·· loss/mean                                                                                                     │
    │ ⢕⢕ loss                                                                                                          │
    │⢣·                                                                                                                │
    │⢸·                                                                                                                │
1.52┤⢸·                                                                                                                │
    │⠘⡄                                                                                                                │
    │ ⡇                                                                                                                │
    │ ⡇                                                                                                                │
    │ ⡇                                                                                                                │
    │ ⢸+                                                                                                               │
1.23┤ ⢸·                                                                                                               │
    │ ⢸⣾                                                                                                               │
    │ ⠈⢻                                                                                                               │
    │  ⢸                                                                                                               │
    │  ·⡇+                                                                                                             │
0.95┤  -⣇⡄                                                                                                             │
    │   ⣿⢣+                                                                                                            │
    │   ⠈⢸+                                                                                                            │
    │    ⢸·                                                                                                            │
    │    ⢸·                                                                                                            │
    │    ⠈⡆⣀    ⢠⡆                                                                                                     │
0.66┤     ⣿⢸++  ⢸⡇                                                                                                     │
    │     ⣿⢸⢀·⣠⡆⢸⡇⢠⡆ ⢀      +                                                                                          │
    │     ⣿⢸⡇⠉⠈⡇⡸⠣⣼⡇ ⣿     +·             ⣠                                                                            │
    │     ⠈⠸⡇-·⢸⡇·⣿⠣⢤⣿·⡀   ⢠⡆+            ⣿                                                                            │
    │      -  -⢸⡇-⠙-⢸⡟⣼⢇   ⢸⡇ ++          ⣿   +     ⢀             +                                                    │
    │      -   ⠸⡇ --⢸⡇⣿⠘⡄⣠+⢸⡇ ·⢠⡆+ + ⡀   +⣿  ⢰⠣⡀ ⢀  ⣿ ⢀⡄         ++              +                                 ⢀⡄  │
0.37┤      -        ⠈⠃⠈-⡇⣿·⡜⡇·⢀⣼⡇⡀+·⢸⡇⡀ +·⣿ ⢀⣼·⢇ ⡿⡀ ⣿ ⢸⡇ + ⢀ ⢀⡄  +·              ⣾                        ⣴        ⢸⡇  │
    │               --  ⡇⡇⠣⡇⢇⡔⠁⠙⣿⢣⢰⡇⢸⠙⢣⣠··⣿+⢸⣿ ⢸⢀⠇⢇·⣿+⡜⢇⢄·+⣿ ⢸⡇ ⢀⡀⣾⢠⡆⣴    +   +⡀ ⣿     + ⣴+          ⢀    ⣿   ⢀⡄   ⢸⡇  │
    │               -   ⠈⠁  ⠸⡇  ⠈-⣿⡇⡇--⠙⣄⠴⢹+⢸⢿-⢸⢸ ⢸⡾⡇⢇⡇·-⣷⢹⡏⡆⡜⡇·⡇⡇⡿⣸⡇⡿⡀   · + ⡸⡇+⣿ ++⣠⢠⢺ ⡿⡀   ·+ ⢀⡄++⣿  + ⡿⡀ +⢸⡇   ⢸⡇  │
    │                             ⠈⢣⡇   ⠙-⠘⡄⢸  ⠸⣸  ⠁ ⠘⠇- ⠙⢸⡇⢿-⠣⡆⡇⠙·⣿⠑⠃⡇⣴· ····⡇⡇⢀⣿⢰⠑⠞⠏⠃⢸·⡇⣇⡄⣴·⣴·⢀⢸⡇⡀⣴⣿⢀⠤⡄·⡇⡇ ⣾⢸⢱ ++⢸⡇· │
    │                               ⠁      ⠈⠃   ⣿         ⢸⡇-  ⠘⠇  ⠻ -⠈⠉⣦⠒⠊⢶⠣⢴⡇⠑⠁⠁⣿··--⠸⣴⡇⠈⢇⡇⠉⠉⢆⡟⣼⣿⣧⠃⢹⢸·⡇⣴⡇⡇⡀⡏⣾⢸+·⡎⣿⡇⢰⡇│
    │                                           ⠙         ⠈⠃            ⠈- - ⠈⠃   ⠈--   ⠈⠃ ⢸⡇  ⢸⡇⠙⠙⠙ ⠘⢼-⢇⠏⠃⠙⣿-⠻⢸⢠⡆⡇⣿⠈⠚⠈│
0.08┤                                                                                      ⠘⠇   ⠁           ⢿   ⠁⠸⡇⢿   │
    └┬───────────────────────────┬────────────────────────────┬───────────────────────────┬───────────────────────────┬┘
    1.0                        49.2                         97.5                        145.8                     194.0
    ```

## `StopWatch`: Timing Context Manager

```python
from ezpz.history import StopWatch

with StopWatch("forward pass", wbtag="timing/forward"):
    output = model(batch)
```

Logs elapsed time to the logger and optionally to W&B. Useful for
profiling individual phases within your training loop.

## Environment Variables

| Variable | Effect |
|----------|--------|
| `EZPZ_NO_DISTRIBUTED_HISTORY` | Disable distributed aggregation |
| `EZPZ_LOCAL_HISTORY` | Alias for above |
| `EZPZ_TRACKER_BACKENDS` | Fallback backend list when `backends` arg is `None` |
| `WANDB_MODE` | Controls wandb mode (`disabled`, `offline`, `online`) |
| `WANDB_DISABLED` | Set to `1` to disable wandb entirely |
| `WB_PROJECT` / `WANDB_PROJECT` | Default wandb project name |
| `EZPZ_TPLOT_MARKER` | Marker style for terminal plots (`braille`, `fhd`, `hd`) |
| `EZPZ_TPLOT_TYPE` | Default plot type (`line`, `hist`) |

## See Also

- [Quick Start](./quickstart.md) for minimal setup
- [Python API: `ezpz.history`](./python/Code-Reference/history.md) for the
  full History API reference
- [Python API: `ezpz.tracker`](./python/Code-Reference/tracker.md) for the
  full Tracker API reference
