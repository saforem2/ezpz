# 🩺 Doctor

`ezpz doctor` inspects the active environment and reports whether common launch
prerequisites are satisfied before you submit a distributed job. The command is
available from the main CLI group:

```bash
ezpz doctor
```

## What gets checked?

| Check      | Status Criteria | Remedy Hints |
|------------|-----------------|--------------|
| **MPI**    | Confirms both `mpi4py` and a launcher (`mpiexec`/`mpirun`) are accessible. | Suggests installing the missing component. |
| **Scheduler** | Identifies PBS/SLURM based on environment variables. | Points to the scheduler plug-in mechanism if the host is unknown. |
| **wandb**  | Detects the `wandb` Python package plus credentials or offline mode. | Recommends `WANDB_MODE=offline` when credentials are absent. |
| **torch**  | Validates the PyTorch install and accelerator visibility (`cuda`, `xpu`, `mps`, or CPU). | Advises setting `TORCH_DEVICE` for CPU-only runs. |

Each line is reported with a severity (`OK`, `WARNING`, `ERROR`) and, when
relevant, an actionable remediation hint.

!!! note
    Enable optional integrations with pip extras:
    `pip install ezpz[monitoring]` provides Weights & Biases, while
    `pip install ezpz[profiling]` and `pip install ezpz[terminal]` load
    pyinstrument and plotext respectively.
```text
[OK     ] mpi: mpi4py import succeeded and an MPI launcher was found.
[WARNING] torch: PyTorch import succeeded but no accelerators were detected.
          ↳ Confirm drivers are available or set TORCH_DEVICE=cpu for CPU-only execution.
```

## JSON output for automation

Pass `--json` to obtain a structured payload that is easy to parse in CI:

```bash
ezpz doctor --json
```

```json
[
  {
    "name": "mpi",
    "status": "ok",
    "message": "mpi4py import succeeded and an MPI launcher was found.",
    "remedy": null
  },
  {
    "name": "torch",
    "status": "warning",
    "message": "PyTorch import succeeded but no accelerators were detected.",
    "remedy": "Confirm drivers are available or set TORCH_DEVICE=cpu for CPU-only execution."
  }
]
```

The command returns a non-zero exit code when any check fails (`status ==
"error"`), making it safe to gate job submissions in automation workflows.
