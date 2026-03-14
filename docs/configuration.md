# ⚙️ Configuration

ezpz behavior can be customized through environment variables.

Additional configuration can be done through environment variables, including:

1. The colorized logging output can be toggled via the `NO_COLOR` environment
   var, e.g. to turn off colors:

    ```bash
    NO_COLOR=1 ezpz launch python3 -m ezpz.examples.fsdp
    ```

1. Force logging from **all** ranks (not just rank 0):

    ```bash
    EZPZ_LOG_ALL_RANKS=1 ezpz launch --line-buffer python3 -m ezpz.examples.vit
    ```

1. Forcing a specific torch device (useful on GPU hosts when you want CPU-only):

    ```bash
    TORCH_DEVICE=cpu ezpz test
    ```

1. Text Based Plots:

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

??? info "Complete List"

    | Environment Variable                         | Purpose / how it's used                                                          |
    | -------------------------------------------- | :------------------------------------------------------------------------------- |
    | TORCH_DEVICE                                 | Force device selection (cpu, cuda, mps, xpu) when picking the torch device.      |
    | TORCH_BACKEND                                | Override distributed backend (nccl, gloo, mpi, xla).                             |
    | TORCH_DDP_TIMEOUT                            | Adjust DDP init timeout (seconds) for slow launches.                             |
    | MASTER_ADDR                                  | Manually set rendezvous address if auto-detection is wrong/unreachable.          |
    | MASTER_PORT                                  | Manually set rendezvous port for distributed init.                               |
    | HOSTFILE                                     | Point ezpz at a specific hostfile when scheduler defaults are missing/incorrect. |
    | NO_COLOR / NOCOLOR / COLOR / COLORTERM       | Enable/disable colored output to suit terminals or log sinks.                    |
    | EZPZ_LOG_LEVEL                               | Set ezpz logging verbosity.                                                      |
    | LOG_LEVEL                                    | General log level for various modules.                                           |
    | LOG_FROM_ALL_RANKS                           | Allow logs from all ranks (not just rank 0).                                     |
    | PYTHONHASHSEED                               | Fix Python hash seed for reproducibility.                                        |
    | WANDB_DISABLED                               | Disable Weights & Biases logging.                                                |
    | WANDB_MODE                                   | Set W&B mode (online, offline, dryrun).                                          |
    | WANDB_PROJECT / WB_PROJECT / WB_PROJECT_NAME | Set project name for W&B runs.                                                   |
    | WANDB_API_KEY                                | Supply W&B API key for authentication.                                           |
    | EZPZ_LOCAL_HISTORY                           | Control local history storage/enablement.                                        |
    | EZPZ_NO_DISTRIBUTED_HISTORY                  | Disable distributed history aggregation.                                         |
    | EZPZ_TPLOT_TYPE                              | Select timeline plot type.                                                       |
    | EZPZ_TPLOT_MARKER                            | Marker style for timeline plots.                                                 |
    | EZPZ_TPLOT_MAX_HEIGHT                        | Max height for timeline plots.                                                   |
    | EZPZ_TPLOT_MAX_WIDTH                         | Max width for timeline plots.                                                    |
    | EZPZ_TPLOT_RAW_MARKER                        | Marker for raw timeline data.                                                    |
    | CPU_BIND                                     | Override default CPU binding for PBS launch commands (advanced).                 |
    | EZPZ_ATTENTION_FP32                          | Force FP32 attention in LLaMA models.                                            |
    | EZPZ_DEBUG_NAN                               | Enable NaN debugging in model forward pass.                                      |
    | EZPZ_RUN_COMMAND                             | Stores the command being executed (set by launcher).                             |
    | EZPZ_VERSION                                 | Auto-set to package version at import time.                                      |
    | EZPZ_JOB_NAME                                | Job name for logging purposes.                                                   |
    | EZPZ_LOG_TIMESTAMP                           | Timestamp for logging context.                                                   |
    | EZPZ_JSON_LOG_JOB_NAME                       | Job name for JSON log handler.                                                   |
    | PYINSTRUMENT_PROFILER                         | Enable pyinstrument profiling (set to "1").                                      |
    | MAKE_TARBALL                                  | Trigger tarball creation in yeet-env.                                            |
    | HYDRA_JOB_NAME                               | Hydra job name (used as fallback).                                               |
