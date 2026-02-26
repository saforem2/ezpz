# Supported Systems

ezpz auto-detects the machine, scheduler, accelerator type, and distributed
backend so that the same user code runs everywhere.  This page documents the
supported systems, how detection works, and the environment variables you can
use to override defaults.

## Systems Matrix

| System     | Scheduler | Launcher  | GPU Type    | Backend  | Hostfile Source                     |
|------------|-----------|-----------|-------------|----------|-------------------------------------|
| Aurora     | PBS Pro   | `mpiexec` | Intel XPU   | `ccl`    | `/var/spool/pbs/aux/$PBS_JOBID.*`   |
| Sunspot    | PBS Pro   | `mpiexec` | Intel XPU   | `ccl`    | `/var/spool/pbs/aux/$PBS_JOBID.*`   |
| Polaris    | PBS Pro   | `mpiexec` | NVIDIA GPU  | `nccl`   | `/var/spool/pbs/aux/$PBS_JOBID.*`   |
| Sophia     | PBS Pro   | `mpiexec` | NVIDIA GPU  | `nccl`   | `/var/spool/pbs/aux/$PBS_JOBID.*`   |
| Sirius     | PBS Pro   | `mpiexec` | NVIDIA GPU  | `nccl`   | `/var/spool/pbs/aux/$PBS_JOBID.*`   |
| Frontier   | SLURM     | `srun`    | AMD GPU     | `nccl`   | `$SLURM_NODELIST`                   |
| Perlmutter | SLURM     | `srun`    | NVIDIA GPU  | `nccl`   | `$SLURM_NODELIST`                   |
| Local      | None      | `mpirun`  | CPU or GPU  | `gloo`   | None                                |

## Machine Detection

Both the Python and shell sides detect the machine from the hostname:

| Hostname Prefix | Machine       | Notes                                          |
|-----------------|---------------|-------------------------------------------------|
| `x4*`           | Aurora        | Or `aurora*` on login nodes                     |
| `x1*`           | Sunspot       | Or `uan*` on login nodes                        |
| `x3*`           | Polaris       | Sirius if `"sirius"` appears in `$PBS_O_HOST`   |
| `sophia-*`      | Sophia        |                                                  |
| `frontier*`     | Frontier      |                                                  |
| `login*`/`nid*` | Perlmutter    |                                                  |

**Python:** `get_machine()` in
[`ezpz.distributed`](../python/Code-Reference/distributed.md) and
[`ezpz.dist`](../python/Code-Reference/dist.md).

**Shell:** `ezpz_get_machine_name()` in
[`utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh).

## Scheduler Detection

**Python:** `get_scheduler()` in `ezpz.configs`:

1. `PBS_JOBID` set → `"PBS"`
2. `SLURM_JOB_ID` or `SLURM_JOBID` set → `"SLURM"`
3. Machine-name fallback (ALCF → PBS, Frontier/Perlmutter → SLURM)
4. Otherwise → `"UNKNOWN"`

**Shell:** `ezpz_get_scheduler_type()` applies the same logic.

## Backend Selection

`get_torch_backend()` selects the `torch.distributed` backend:

| Condition                     | Backend        |
|-------------------------------|----------------|
| `TORCH_BACKEND` env var set   | Value of env var |
| Device is `xpu` (Intel)      | `ccl`          |
| Device is `cuda` (NVIDIA/AMD) | `nccl`         |
| Device is `cpu` or `mps`     | `gloo`         |

Override with:

```bash
export TORCH_BACKEND=gloo   # e.g. for debugging
```

## Environment Variables Reference

### ezpz Internal

| Variable                  | Default   | Description                                    |
|---------------------------|-----------|------------------------------------------------|
| `EZPZ_LOG_LEVEL`         | `"INFO"`  | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `EZPZ_LOG_FROM_ALL_RANKS` | _unset_  | If set (any truthy value), all ranks log — not just rank 0 |
| `EZPZ_RUN_COMMAND`       | _unset_   | Set by `ezpz.launch` to track the running command |

### Distributed Topology

These are **read** during `setup_torch()` and **written** after MPI
initialization so that downstream code can access them:

| Variable          | Description                            |
|-------------------|----------------------------------------|
| `RANK`            | Global MPI rank                        |
| `LOCAL_RANK`      | Rank within the current node           |
| `WORLD_SIZE`      | Total number of processes              |
| `LOCAL_WORLD_SIZE`| Processes per node                     |
| `MASTER_ADDR`     | Address for `torch.distributed` rendezvous |
| `MASTER_PORT`     | Port for `torch.distributed` rendezvous    |

#### Local Rank Fallback Chain

If `LOCAL_RANK` is not set, these are checked in order:

1. `PMI_LOCAL_RANK`
2. `OMPI_COMM_WORLD_LOCAL_RANK`
3. `MPI_LOCALRANKID`
4. `MPICH_LOCALRANKID`
5. `SLURM_LOCAL_ID`

#### GPUs-Per-Node Fallback Chain

If `NGPU_PER_HOST` is not set, these are checked in order:

1. `LOCAL_WORLD_SIZE`
2. `PMI_LOCAL_SIZE`
3. `SLURM_NTASKS_PER_NODE`

### Device and Backend Overrides

| Variable          | Default            | Description                          |
|-------------------|--------------------|--------------------------------------|
| `TORCH_DEVICE`    | _auto-detected_    | Force device type: `cuda`, `xpu`, `mps`, `cpu` |
| `TORCH_BACKEND`   | _auto-detected_    | Force distributed backend: `nccl`, `ccl`, `gloo` |
| `TORCH_DDP_TIMEOUT` | `3600`           | Timeout (seconds) for `init_process_group` |

### Scheduler / Job

| Variable          | Set By    | Description                          |
|-------------------|-----------|--------------------------------------|
| `PBS_JOBID`       | PBS Pro   | Job ID (triggers PBS detection)      |
| `PBS_NODEFILE`    | PBS Pro   | Path to allocated-nodes file         |
| `PBS_O_WORKDIR`   | PBS Pro   | Submission directory                 |
| `SLURM_JOB_ID`   | SLURM     | Job ID (triggers SLURM detection)    |
| `SLURM_NODELIST`  | SLURM     | Compact node list                    |
| `SLURM_NNODES`    | SLURM     | Number of allocated nodes            |
| `SLURM_SUBMIT_DIR`| SLURM     | Submission directory                 |
| `HOSTFILE`        | User      | Override hostfile path for any scheduler |

### Weights & Biases

| Variable          | Default     | Description                          |
|-------------------|-------------|--------------------------------------|
| `WANDB_DISABLED`  | _unset_     | Disable wandb entirely               |
| `WANDB_MODE`      | `"offline"` | `online`, `offline`, `disabled`, `shared` |
| `WANDB_API_KEY`   | _unset_     | API authentication key               |
| `WANDB_PROJECT`   | _unset_     | Project name (also checks `WB_PROJECT`, `WB_PROJECT_NAME`) |

## Overrides and Tips

- **Custom hostfile:** pass `--hostfile /path/to/hostfile` to `ezpz launch`
  or `export HOSTFILE=/path/to/hostfile`.
- **Override rank counts:** use `-n` (total ranks) and `-ppn` (ranks per
  node) with `mpiexec`, or `--ntasks` / `--ntasks-per-node` with `srun`.
- **Debugging backend issues:** set `TORCH_BACKEND=gloo` to fall back to a
  CPU-only backend while debugging connectivity.
- **Wandb network issues:** set `WANDB_MODE=offline` and sync later with
  `wandb sync`.

## Known Failure Modes

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Scheduler not detected | `PBS_NODEFILE` / `SLURM_NODELIST` not in env | Set `EZPZ_LOG_LEVEL=DEBUG` and check output; use `--hostfile` |
| `mpiexec` / `srun` not found | Module not loaded | `module load` the appropriate MPI or use full path |
| Backend init fails (`nccl`/`ccl`) | Driver or module mismatch | Verify GPU drivers and modules; fall back to `TORCH_BACKEND=gloo` |
| Wandb hangs on init | No network on compute nodes | `export WANDB_MODE=offline` |

## Example Launch Commands

=== "Aurora (PBS)"

    ```bash
    source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env
    ezpz launch python3 -m ezpz.examples.diffusion
    ```

=== "Polaris (PBS)"

    ```bash
    source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env
    ezpz launch python3 -m ezpz.examples.diffusion
    ```

=== "Frontier (SLURM)"

    ```bash
    # inside an allocation: salloc -N2 -t 00:30:00 ...
    source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env
    ezpz launch python3 -m ezpz.examples.diffusion
    ```

=== "Perlmutter (SLURM)"

    ```bash
    source <(curl -LsSf https://bit.ly/ezpz-utils) && ezpz_setup_env
    ezpz launch python3 -m ezpz.examples.diffusion
    ```

=== "Local (mpirun)"

    ```bash
    WORLD_SIZE=2 uv run ezpz launch python3 -m ezpz.examples.diffusion
    ```
