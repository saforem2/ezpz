# ⚙️ Configuration

`ezpz` behavior can be customized through environment variables:

1. The colorized logging output can be toggled via the `NO_COLOR` environment
   var, e.g. to turn off colors:

    ```bash
    NO_COLOR=1 ezpz launch -- python3 -m ezpz.examples.fsdp
    ```

2. Force logging from **all** ranks (not just rank 0):

    ```bash
    LOG_FROM_ALL_RANKS=1 ezpz launch --line-buffer python3 -m ezpz.examples.vit
    ```

3. Forcing a specific torch device (useful on GPU hosts when you want CPU-only):

    ```bash
    TORCH_DEVICE=cpu ezpz test
    ```

4. Text Based Plots:

    1. Changing the plot marker used in the text-based plots:

        ```bash
        # highest resolution, may not be supported in all terminals
        EZPZ_TPLOT_MARKER="braille" ezpz launch python3 -m your_app.train
        # next-best resolution, more widely supported
        EZPZ_TPLOT_MARKER="fhd" ezpz launch python3 -m your_app.train
        ```

    1. Changing the plot size:

        The plots will automatically scale (up to a reasonable limit) with the
        dimensions of the terminal in which they're run.

        If desired, these can be specified explicitly by overriding the `LINES`
        and `COLUMNS` environment variables, e.g.:

        ```bash
        LINES=40 COLUMNS=100 ezpz test
        ```

## Common Configurations

Copy-paste these for common scenarios:

```bash
# Run offline — no external tracking
EZPZ_TRACKER_BACKENDS=none WANDB_DISABLED=1 ezpz launch -- python3 train.py

# Force CPU mode for debugging (no GPU required)
TORCH_DEVICE=cpu TORCH_BACKEND=gloo ezpz launch -np 2 -- python3 train.py

# Log to all tracking backends simultaneously
EZPZ_TRACKER_BACKENDS=wandb,mlflow,csv ezpz launch -- python3 train.py

# Quiet logging (warnings and errors only)
EZPZ_LOG_LEVEL=WARNING ezpz launch -- python3 train.py

# MLflow with AmSC credentials (see MLflow Credential Files below)
EZPZ_TRACKER_BACKENDS=mlflow ezpz launch -- python3 train.py
```

!!! tip "Choosing a parallelism strategy"

    See the [Distributed Training Guide](./guides/distributed-training.md#quick-reference)
    for a decision table on when to use DDP vs FSDP vs FSDP+TP.

## Environment Variables

### Device & Distribution

| Variable | Purpose | Values / Default |
|----------|---------|-----------------|
| `TORCH_DEVICE` | Force device selection instead of auto-detection. | `cpu`, `cuda`, `mps`, `xpu`. Auto-detected. |
| `TORCH_BACKEND` | Override distributed backend. | `nccl`, `gloo`, `mpi`, `xccl`. Auto-detected. |
| `TORCH_DDP_TIMEOUT` | DDP init timeout for slow launches. | Seconds. Default: `3600`. |
| `MASTER_ADDR` | Rendezvous address for distributed init. | Hostname/IP. Auto-detected from scheduler. |
| `MASTER_PORT` | Rendezvous port for distributed init. | Port number. Auto-detected (picks a free port). |
| `HOSTFILE` | Path to hostfile when scheduler defaults are missing. | File path. Auto-created from PBS/SLURM. |

### Logging

| Variable | Purpose | Values / Default |
|----------|---------|-----------------|
| `NO_COLOR` / `NOCOLOR` / `COLOR` / `COLORTERM` | Enable/disable colored output to suit terminals or log sinks. | Set `NO_COLOR=1` to disable. |
| `EZPZ_LOG_LEVEL` | Set ezpz logging verbosity. | `DEBUG`, `INFO`, `WARNING`, `ERROR`. Default: `INFO`. |
| `LOG_LEVEL` | General log level for various modules. | Same as above. |
| `LOG_FROM_ALL_RANKS` | Allow logs from all ranks (not just rank 0). | Set to `1` to enable. Default: rank 0 only. |
| `EZPZ_LOG_FROM_ALL_RANKS` | Alias for `LOG_FROM_ALL_RANKS`. | Same as above. |

### Log Formatting

#### Prefix Style Presets

Set `EZPZ_LOG_PREFIX_STYLE` to apply a preset. Individual `EZPZ_LOG_*` vars
still override after the preset is applied.

| Style | Output | Description |
|-------|--------|-------------|
| `full` | `[2026-04-04 21:04][I][ezpz/launch:601.run] msg` | Default — all components, full brackets |
| `minimal` | `I ezpz/launch:601.run -- msg` | Level + path, no time, no brackets |
| `time` | `21:04:13 I -- msg` | Time + level, no path, no brackets |
| `plain` | `2026-04-04 21:04 I ezpz/launch:601.run -- msg` | All components, no brackets, no color |
| `none` | `msg` | No prefix at all |

#### Individual Options

| Variable | Purpose | Values / Default |
|----------|---------|-----------------|
| `EZPZ_LOG_PREFIX_STYLE` | Apply a preset that configures multiple options at once (see table above). Individual vars override. | `full` (default), `minimal`, `time`, `plain`, `none`. |
| `EZPZ_LOG_USE_BRACKETS` | Wrap each prefix component in brackets: `[time][level][path]`. | `1` (default) / `0`. |
| `EZPZ_LOG_USE_SINGLE_BRACKET` | Wrap all prefix components in one bracket: `[time level path]`. Only applies when `EZPZ_LOG_USE_BRACKETS=0`. | `0` (default) / `1`. |
| `EZPZ_LOG_USE_COLORED_PREFIX` | Colorize the log prefix (time, level, path). Message content is unaffected. | `1` (default) / `0`. |
| `EZPZ_LOG_SHOW_TIME` | Show timestamp in log output. | `1` (default) / `0`. |
| `EZPZ_LOG_SHOW_LEVEL` | Show log level (`I`, `W`, `E`, etc.) in log output. | `1` (default) / `0`. |
| `EZPZ_LOG_SHOW_PATH` | Show module/file path in log output. | `1` (default) / `0`. |
| `EZPZ_LOG_RANK` | Show rank number in log output. | `0` (default) / `1`. |
| `EZPZ_LOG_ENABLE_LINK_PATH` | Make file paths clickable links (terminal support required). | `0` (default) / `1`. |
| `EZPZ_LOG_TIME_FORMAT` | Override the timestamp format string. | `strftime` format. Default: `%Y-%m-%d %H:%M:%S`. |
| `EZPZ_LOG_TIME_DETAILED` | Use microsecond-precision timestamps. | `0` (default) / `1`. |
| `EZPZ_LOG_DAY_SEPARATOR` | Separator between year/month/day in timestamps. | Default: `-`. |
| `EZPZ_LOG_DAY_TIME_SEPARATOR` | Separator between date and time in timestamps. | Default: ` ` (space). |

### Experiment Tracking

| Variable | Purpose | Values / Default |
|----------|---------|-----------------|
| `EZPZ_TRACKER_BACKENDS` / `EZPZ_TRACKERS` | Comma-separated tracker backends. `EZPZ_TRACKERS` is a shorthand alias. | `wandb`, `csv`, `mlflow`. e.g. `wandb,csv,mlflow`. Default: `wandb`. |
| `WANDB_DISABLED` | Disable Weights & Biases logging. | Set to `1` to disable. |
| `WANDB_MODE` | Set W&B mode. | `online`, `offline`, `dryrun`. |
| `WANDB_PROJECT` / `WB_PROJECT` / `WB_PROJECT_NAME` | Set project name for W&B runs. Also used as MLflow experiment name if `MLFLOW_EXPERIMENT_NAME` is not set. | String (aliases for the same setting). |
| `WANDB_API_KEY` | Supply W&B API key for authentication. | API key string. |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URL or local path. | e.g. `https://mlflow.example.com` or `file:///path/to/mlruns`. |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment name. Falls back to `WANDB_PROJECT`, then auto-derived from script name. | String. |
| `MLFLOW_TRACKING_TOKEN` | Bearer token for MLflow server auth. | Token string. |
| `MLFLOW_TRACKING_INSECURE_TLS` | Skip TLS certificate verification for MLflow server. | `true` / `false`. |
| `AMSC_API_KEY` | API key for AMSC MLflow server. Automatically sent as `X-API-Key` header. | API key string. |
| `EZPZ_LOCAL_HISTORY` | Enable local-only history (skip distributed aggregation). | Set to any truthy value (e.g. `1`). |
| `EZPZ_NO_DISTRIBUTED_HISTORY` | Disable distributed history aggregation. | Set to any truthy value. Auto-enabled at 384+ ranks. |

#### MLflow Credential Files

The MLflow backend automatically loads environment variables from dotenv
files in this order:

1. **`~/.amsc.env`** — User-level credentials (loaded first). Put your
   `AMSC_API_KEY` and `MLFLOW_TRACKING_URI` here so they work across all
   projects without committing secrets to version control.

2. **Project `.env`** — Found by walking upward from the working directory
   (loaded second, **overrides** values from `~/.amsc.env`). Use this for
   project-specific tracking URIs or experiment names.

Example `~/.amsc.env`:

```bash
AMSC_API_KEY=your-api-key-here
MLFLOW_TRACKING_URI=https://mlflow.american-science-cloud.org
MLFLOW_TRACKING_INSECURE_TLS=true
```

!!! tip "No `python-dotenv`?"

    If `python-dotenv` is not installed, set the variables directly in your
    shell environment or job script. The dotenv loading is a convenience,
    not a requirement.

### Plotting

| Variable | Purpose | Values / Default |
|----------|---------|-----------------|
| `EZPZ_TPLOT_TYPE` | Select timeline plot type. | Plot type string. |
| `EZPZ_TPLOT_MARKER` | Marker style for terminal plots. | `braille` (highest resolution), `fhd` (more compatible). |
| `EZPZ_TPLOT_MAX_HEIGHT` | Max height (rows) for terminal plots. | Integer. Auto-scales to terminal height. |
| `EZPZ_TPLOT_MAX_WIDTH` | Max width (columns) for terminal plots. | Integer. Auto-scales to terminal width. |
| `EZPZ_TPLOT_RAW_MARKER` | Marker for raw timeline data points. | Same values as `EZPZ_TPLOT_MARKER`. |

### Model & Debug

| Variable | Purpose | Values / Default |
|----------|---------|-----------------|
| `EZPZ_ATTENTION_FP32` | Force FP32 attention in LLaMA models. | Set to `1` to enable. |
| `EZPZ_DEBUG_NAN` | Enable NaN debugging in model forward pass. | Set to `1` to enable. |
| `PYINSTRUMENT_PROFILER` | Enable pyinstrument profiling. | Set to `1` to enable. |

### Launcher & Jobs

| Variable | Purpose | Values / Default |
|----------|---------|-----------------|
| `CPU_BIND` | CPU binding strategy for MPI process pinning. | `depth` (with `--depth=N`), or `list:core_ranges` (e.g. `list:2-4:10-12`). Machine-specific defaults set automatically. |
| `EZPZ_RUN_COMMAND` | The command being executed (set by launcher, read-only). | Auto-set. |
| `EZPZ_JOB_NAME` | Job name for logging. | String. Falls back to scheduler job name. |
| `EZPZ_LOG_TIMESTAMP` | Timestamp for logging context. | Auto-set. |
| `EZPZ_JSON_LOG_JOB_NAME` | Job name for JSON log handler. | String. |
| `HYDRA_JOB_NAME` | Hydra job name (used as fallback for `EZPZ_JOB_NAME`). | String. |

### Auto-set / Read-only

These are set automatically and don't typically need to be configured by users.

| Variable | Purpose |
|----------|---------|
| `EZPZ_VERSION` | Auto-set to package version at import time. |
| `PYTHONHASHSEED` | Fix Python hash seed for reproducibility. |
| `MAKE_TARBALL` | Trigger tarball creation in `ezpz yeet-env`. |
