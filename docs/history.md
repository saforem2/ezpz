# 📊 Experiment Tracking

The `History` class handles the full experiment tracking lifecycle:
recording metrics, computing distributed statistics, dispatching to
external backends (Weights & Biases, CSV, or custom), and producing
plots and reports at the end of a run.

!!! info "Recent changes"

    **Tracker integration** (v0.11): `History` now owns a `Tracker`
    internally. Pass `backends=` directly to `History()` instead of
    managing wandb separately:

    ```python
    # Before (deprecated)
    import ezpz
    ezpz.setup_wandb(project_name="my-project")
    history = History()
    history.update({"loss": 0.42}, use_wandb=True)

    # After
    history = History(project_name="my-project", backends="wandb,csv")
    history.update({"loss": 0.42}, step=step)
    ```

    - `use_wandb` parameter on `update()` is **deprecated** — use
      `backends="wandb"` in the constructor
    - If an active `wandb.run` is detected without `backends=` being set,
      History will use it automatically with a deprecation warning
    - Backend errors are isolated — a failing backend logs a warning but
      never crashes your training run

## Quick Start

```python
import ezpz

rank = ezpz.setup_torch()

history = ezpz.History(
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
    report_dir="./outputs/history", # markdown report directory
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
| `jsonl_path` | `None` | Path for per-step JSONL log. When `None`, defaults to `<report_dir>/<run_id>.jsonl` |
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

- **Rank 0** initializes a real run (online/offline per `WANDB_MODE`)
- **Rank != 0** gets `mode="disabled"` — no network calls, no duplicate runs
- Resolves project name from: argument > `WB_PROJECT` > `WANDB_PROJECT`
  \> `WB_PROJECT_NAME` env vars > script-derived default
- Auto-logs system info (hostname, torch version, ezpz version)
- Logs metrics on each `update()`, uploads training history table on `finalize()`
- Logs matplotlib plots as image artifacts

Set `WANDB_DISABLED=1` or `backends="none"` to disable.

### MLflow

Built-in backend that logs to an [MLflow Tracking](https://mlflow.org)
server or local filesystem.

!!! tip "Standalone setup"

    For one-call MLflow initialization outside of `History`, use
    `ezpz.setup_mlflow()` — it mirrors `ezpz.setup_wandb()`:

    ```python
    run = ezpz.setup_mlflow(project_name="my-project", config={"lr": 1e-4})
    ```

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

| Env var                | Auth method                          |
|------------------------|--------------------------------------|
| `MLFLOW_TRACKING_TOKEN` | Bearer token (MLflow native)         |
| `AMSC_API_KEY`          | `X-API-Key` header (for AMSC servers) |

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

```python
history = History(backends="csv", outdir="./logs")

history.update({"loss": 0.5, "lr": 1e-4})
history.update({"loss": 0.3, "lr": 1e-4, "grad_norm": 0.8})  # columns auto-extend
```

The CSV backend writes to the `outdir`:

| File                   | Content                                                          |
|------------------------|------------------------------------------------------------------|
| `metrics.csv`          | One row per `update()` call, columns auto-extend as new keys appear |
| `config.json`          | Merged config from the `config=` constructor argument            |
| `training_history.csv` | Written by `log_table()` on `finalize()`                         |

- **Rank 0 only** — non-rank-0 processes buffer rows in memory but skip
  all file I/O

### None

Pass `backends="none"` (or set `EZPZ_TRACKER_BACKENDS=none`) to disable
tracking entirely. Returns a `NullTracker` where all methods are no-ops.

### Multiple Backends

Combine backends to dispatch everywhere at once:

```python
history = History(
    project_name="my-project",
    backends="wandb,mlflow,csv",
    outdir="./outputs",
    config={"lr": 1e-4, "batch_size": 32},
)
```

Every `update()` call fans out to all backends.

### Error Isolation

Backend errors are isolated — if one backend fails during `log()`,
`log_config()`, or `finish()`, the others still receive the call. A
warning is logged but your training run is never interrupted by a
tracking failure. This means a flaky network connection to your MLflow
server won't crash a multi-day training job.

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
| Metrics JSONL | `{outdir}/metrics.jsonl` |
| Metrics CSV | `{outdir}/metrics.csv` (when `csv` backend is active) |
| JSON log symlink | `{outdir}/{timestamp}-rank0.jsonl` → `logs/...` |

All output is co-located in `{outdir}`:

- The **CSV backend** is automatically redirected to `{outdir}` so
  `metrics.csv`, `config.json`, and `training_history.csv` land
  alongside plots and reports.
- A **symlink** to the structured JSON log file is created in `{outdir}`
  so you don't have to hunt for it under `logs/`.

At the end, `finalize()` logs a summary of all output paths:

```
Output files:
  Output Directory: ./outputs/my-experiment/2026-04-04-...
  Report: ./outputs/.../report.md
  Plots (matplotlib): ./outputs/.../plots/mplot
  Plots (terminal): ./outputs/.../plots/tplot
  JSON Log: ./logs/ezpz-test/2026-...-rank0.jsonl
  Metrics JSONL: ./outputs/.../metrics.jsonl
  Metrics CSV: ./outputs/.../metrics.csv
  Dataset: ./outputs/.../train.h5
```

It also uploads the training history table to any active backends and
calls `tracker.finish()` to flush and close all backend connections.
Returns the xarray `Dataset` for further analysis.

??? example "Terminal plot output"

    `finalize()` generates text-based plots directly in the terminal.
    Each metric gets individual plots, and distributed runs include
    min/max/mean/std variants:

    ```
                      accuracy                             accuracy/min
         ┌────────────────────────────────┐     ┌────────────────────────────────┐
    0.984┤                   ▟▖  ▖▌▄ ▟▐ ▙▌│0.969┤         -- -- -----------------│
    0.921┤      ▗ ▄▗▙▄▙▙▟█▄▄███▟▟█████▛█▜█│0.902┤ - --------------------- -------│
         │ ▗ ▖ ▟█▟▌█▜███▛█▛▛▝ ▝▛▀▀  ▝▌▌▀ ▀│0.770┤ ------   -                     │
    0.857┤ ▐ ▙█▜▝▘ ▘▐      ▌              │0.703┤---                             │
    0.793┤ ▐▟██                           │0.570┤--                              │
         │▗▐▛█                            │     └┬───────┬───────┬──────┬───────┬┘
    0.729┤▐▟ ▝                            │     1.0    49.2    97.5   145.8 194.0
         │█▜                              │accuracy/min        iter
    0.665┤▜                               │                accuracy/std
    0.602┤▐                               │     ┌────────────────────────────────┐
         └┬───────┬───────┬──────┬───────┬┘0.078┤   *                            │
         1.0    49.2    97.5   145.8 194.0 0.065┤ *****                          │
    accuracy            iter               0.039┤******** **         *      *    │
                    accuracy/mean          0.026┤************************ *******│
         ┌────────────────────────────────┐0.000┤  ******************************│
    0.977┤                   ··   ·   ····│     └┬───────┬───────┬──────┬───────┬┘
    0.916┤         ·······················│     1.0    49.2    97.5   145.8 194.0
         │     ·············· ····  · · ··│accuracy/std        iter
    0.855┤ · ······ · ·    ·              │                accuracy/max
    0.795┤ ···· ·                         │     ┌────────────────────────────────┐
         │ ···                            │0.984┤             + + +++++++++++++++│
    0.734┤ ··                             │0.930┤ + ++++++++++++++++++++++ ++++++│
    0.674┤··                              │0.820┤ ++++ ++  +                     │
         │··                              │0.766┤++                              │
    0.613┤·                               │0.656┤++                              │
         └┬───────┬───────┬──────┬───────┬┘     └┬───────┬───────┬──────┬───────┬┘
         1.0    49.2    97.5   145.8 194.0      1.0    49.2    97.5   145.8 194.0
    accuracy/mean       iter               accuracy/max        iter
    ```

    Combined summary with all statistics overlaid
    (`·` mean, `-` min, `+` max, `▞` raw):

    ```
         ┌───────────────────────────────────────────────────────────────────────┐
    0.984┤ ++ accuracy/max                           ▗▌        ▗▌  +   ▗  ▗ +▟+▗ │
         │ -- accuracy/min                          ▖▐▌▟     ▟ ▐▌ ▗+  ▗▐ ▗█ ·█▐█ │
         │ ·· accuracy/mean            ▖+  ▟▟    + ▐▙▘▌█+▗▌+ ▌▌▐▌▖█▟▌▄▐▐▚▛█▗▞█▐█·│
         │ ▞▞ accuracy       ·+▗▌ ▄+▗▌▐▌▗ ▟██·+·▗·▟▐█·▛▐▟▟▌▗▌▌▙█▌██▌▜█▛▟·▌▐█▌▘▜▌▛│
    0.915┤         +   +▗▌  ▗▜·▟▌▟▐▖▐▌▛▌█▟███▟█▐▐▞·█▜-▘▐-█▙██▌█▝▝▝▌▘·▐▌··▌▐█▌ ▐▌▌│
         │         + ▗▚▗▐▌▞▖▌▐·█▜█▐▙█▚▌▙█▌▘███▛▟▟▌-▝  ·▝ █▜▐▝▘▜·     ▐▌--▌ ▘▘ ▝▌▌│
         │   ▗   ▖ +▗█▐█▐▙▌▙▘-▌▛·█-██ ▘▘▀▌- ▀ ▘ █     -  ▜    ·-     ▝▌  ▘     · │
         │   █  ▐▌▗+▐▜▐▌██▌·--▙▘·█-▝▝         - █     -                          │
         │   █+ ▐▙█▖▌·▝▌▘▝▘   ▝ ·█              ▜                                │
    0.846┤   █+ ▐██▙▌-· · -     -▝                                               │
         │  +█▗▌▞███▌ - · -     -                                                │
         │ ++█▐▐▌█▛▛▌ - -       -                                                │
         │ ++█▟▝▌█--  - -                                                        │
    0.777┤ ++█▛·-█      -                                                        │
         │ ▖·█▘- █                                                               │
         │▐▌·█·  █                                                               │
         │▐▌▗▜-  ▝                                                               │
    0.708┤▐▌▟▐-                                                                  │
         │▟█▌▐-                                                                  │
         │██·▝-                                                                  │
         │██·                                                                    │
         │▝█-                                                                    │
    0.639┤·█-                                                                    │
         │·█                                                                     │
         │-▜                                                                     │
         │-                                                                      │
    0.570┤-                                                                      │
         └┬─────────────────┬────────────────┬─────────────────┬────────────────┬┘
         1.0              49.2             97.5              145.8          194.0
    ```

    Histograms for each statistic:

    ```
                accuracy/mean hist                      accuracy/max hist
        ┌─────────────────────────────────┐    ┌─────────────────────────────────┐
    80.0┤                          ████   │77.0┤                          ████   │
        │                          ████   │    │                          ████   │
    66.7┤                          ████   │64.2┤                          ████   │
    53.3┤                          ████   │51.3┤                          ████   │
        │                          ████   │    │                          ████   │
    40.0┤                       ███████   │38.5┤                       ██████████│
        │                       ██████████│    │                       ██████████│
    26.7┤                       ██████████│25.7┤                       ██████████│
    13.3┤                █████████████████│12.8┤                    █████████████│
        │             ████████████████████│    │             ████████████████████│
     0.0┤█████████████████████████████████│ 0.0┤███████   ███████████████████████│
        └┬───────┬───────┬───────┬───────┬┘    └┬───────┬───────┬───────┬───────┬┘
       0.60    0.70    0.79    0.89   0.99    0.64    0.73    0.82    0.91   1.00
                 accuracy/min hist                      accuracy/std hist
        ┌─────────────────────────────────┐    ┌─────────────────────────────────┐
    83.0┤                          ████   │67.0┤   ████                          │
        │                          ████   │    │   ████                          │
    69.2┤                          ████   │55.8┤███████                          │
    55.3┤                          ████   │44.7┤███████                          │
        │                          ████   │    │███████                          │
    41.5┤                          ███████│33.5┤██████████                       │
        │                       ██████████│    │██████████                       │
    27.7┤                       ██████████│22.3┤█████████████                    │
    13.8┤                       ██████████│11.2┤█████████████                    │
        │                █████████████████│    │████████████████████             │
     0.0┤█████████████████████████████████│ 0.0┤██████████████████████████   ████│
        └┬───────┬───────┬───────┬───────┬┘    └┬───────┬───────┬───────┬───────┬┘
       0.55    0.66    0.77    0.88   0.99   -0.003   0.018   0.039   0.060 0.082
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
