# `ezpz submit`

Submit jobs to PBS (`qsub`) or SLURM (`sbatch`) schedulers directly from
the command line.

## Two Modes

### 1. Wrap a command

Provide a command after `--` and `ezpz submit` generates a job script
automatically:

```bash
ezpz submit -N 2 -q debug -t 01:00:00 \
    -- python3 -m ezpz.examples.test --model small
```

The generated script includes:

- Scheduler directives (`#PBS` or `#SBATCH`)
- Activation of your current Python environment (venv or conda)
- `ezpz launch` wrapping for distributed execution

### 2. Submit an existing script

Pass a `.sh` file directly:

```bash
ezpz submit job.sh --nodes 4 --time 02:00:00
```

## Options

| Flag | Description |
|------|-------------|
| `-N`, `--nodes` | Number of compute nodes (default: 1) |
| `-t`, `--time` | Walltime in `HH:MM:SS` format (default: `01:00:00`) |
| `-q`, `--queue` | Queue (PBS) or partition (SLURM) (default: `debug`) |
| `-A`, `--account` | Project/account for billing |
| `--filesystems` | PBS filesystems directive (default: `home`) |
| `--job-name` | Job name (auto-derived from command if omitted) |
| `--scheduler` | Force `PBS` or `SLURM` (auto-detected by default) |
| `--dry-run` | Print the script without submitting |
| `--launch` | Wrap the command with `ezpz launch` |

## Examples

### Dry-run to preview the generated script

```bash
ezpz submit --dry-run -N 2 -q debug -A myproject \
    -- python3 -m ezpz.examples.fsdp --model small
```

### Submit with specific filesystems (PBS/Aurora)

```bash
ezpz submit -N 2 -q debug -t 01:00:00 \
    --filesystems home:eagle:grand \
    -A myproject \
    -- python3 -m ezpz.examples.test
```

### Submit with `ezpz launch` wrapping

```bash
ezpz submit --launch -N 1 -q debug \
    -- python3 -m ezpz.examples.test
```

## Environment Detection

The generated script automatically activates your current environment:

- **venv**: If `VIRTUAL_ENV` is set, adds `source $VIRTUAL_ENV/bin/activate`
- **conda**: If `CONDA_PREFIX` is set, adds `conda activate <env_name>`
- **Custom**: If `EZPZ_SETUP_ENV` points to a file, sources it

## Example: `ezpz benchmark` on Aurora

Submit `ezpz benchmark` to run on 2 nodes:

```bash
#[aurora_frameworks-2025.3.1](ezpz-aurora_frameworks-2025.3.1)
#[04/02/26,10:24:48][aurora-uan-0009][/f/A/f/p/s/ezpz][dev][$?]
; ezpz submit -A AuroraGPT -N 2 -q capacity -t 00:30:00 -- ezpz benchmark
Generated job script:
------------------------------------------------------------
#!/bin/bash --login
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -l filesystems=home
#PBS -A AuroraGPT
#PBS -k doe
#PBS -j oe
#PBS -q capacity
#PBS -N ezpz

set -eo pipefail
cd /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/ezpz

# ── Environment setup ──
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env

ezpz benchmark

------------------------------------------------------------
Submitted job 8414055.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
Script saved to /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/ezpz/.ezpz-submit-20260402_102453.sh
took: 5s
```

### Output

??? quote "Job output"

    ```bash
    [2026-04-02-104328][I][/dev/fd/63:2850] Detected PBS scheduler environment.
    [2026-04-02-104328][W][/dev/fd/63:2886] Current working directory does not match PBS_O_WORKDIR! This may cause issues with the job submission.
    [2026-04-02-104328][W][/dev/fd/63:2887] PBS_O_WORKDIR /flare/AuroraGPT/foremans/projects/saforem2/ezpz
    [2026-04-02-104328][W][/dev/fd/63:2888] WORKING_DIR /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/ezpz
    [2026-04-02-104328][W][/dev/fd/63:2889] Exporting PBS_O_WORKDIR=WORKING_DIR=/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/ezpz and continuing...
    [2026-04-02-104328][I][/dev/fd/63:2582] [ezpz_setup_env]...
    [2026-04-02-104328][I][/dev/fd/63:1363] [PYTHON]
    [2026-04-02-104328][I][/dev/fd/63:1370]   - No conda_prefix OR pythonuserbase OR virtual_env found in environment. Setting up conda...
    Lmod Warning: ONEAPI_DEVICE_SELECTOR has been set to
    "opencl:gpu;level_zero:gpu"
    to enable Triton-XPU, vLLM, Ray, and dpctl functionality.

    If you encounter issues, you can revert to:
      export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"

    If you do need to revert, please email support@alcf.anl.gov
    so we can track and address compatibility issues.

    While processing the following module(s):
        Module fullname      Module Filename
        ---------------      ---------------
        frameworks/2025.3.1  /opt/aurora/26.26.0/modulefiles/frameworks/2025.3.1.lua

    Lmod Warning: ONEAPI_DEVICE_SELECTOR has been set to
    "opencl:gpu;level_zero:gpu"
    to enable Triton-XPU, vLLM, Ray, and dpctl functionality.

    If you encounter issues, you can revert to:
      export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"

    If you do need to revert, please email support@alcf.anl.gov
    so we can track and address compatibility issues.

    While processing the following module(s):
        Module fullname      Module Filename
        ---------------      ---------------
        frameworks/2025.3.1  /opt/aurora/26.26.0/modulefiles/frameworks/2025.3.1.lua

    [2026-04-02-104330][I][/dev/fd/63:852] Setting FI_MR_CACHE_MONITOR=userfaultfd
    [2026-04-02-104330][I][/dev/fd/63:934] List of active modules:

    Currently Loaded Modules:
      1) gcc/13.4.0
      2) oneapi/release/2025.3.1
      3) mpich/opt/5.0.0.aurora_test.3c70a61
      4) libfabric/1.22.0
      5) cray-pals/1.8.0
      6) cray-libpals/1.8.0
      7) gcc-runtime/13.4.0-2tg3zy7            (H)
      8) intel-oneapi-runtime/2025.3.1-h4uj4w3 (H)
      9) hdf5/1.14.6
     10) pti-gpu/0.16.0-rc1
     11) miniforge3/25.11.0-1
     12) frameworks/2025.3.1

      Where:
       H:  Hidden Module

    [2026-04-02-104330][I][/dev/fd/63:1383]   - Found Python at /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1
    [2026-04-02-104330][I][/dev/fd/63:1204]   - Found python root at /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1
    [2026-04-02-104330][I][/dev/fd/63:1219]   - No VIRTUAL_ENV found in environment!
    [2026-04-02-104330][I][/dev/fd/63:1222]   - Looking for venv in venvs/aurora/ezpz-aurora_frameworks-2025.3.1...
    [2026-04-02-104330][I][/dev/fd/63:1246]   - Activating existing venv in VENV_DIR=venvs/ezpz-aurora_frameworks-2025.3.1
    [2026-04-02-104330][I][/dev/fd/63:1248]   - Found /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/ezpz/venvs/aurora/ezpz-aurora_frameworks-2025.3.1/bin/activate
    [2026-04-02-104330][I][/dev/fd/63:1418]   - Using python from: /lus/flare/projects/AuroraGPT/foremans/projects/saforem2/ezpz/venvs/aurora/ezpz-aurora_frameworks-2025.3.1/bin/python3
    [2026-04-02-104330][I][/dev/fd/63:2424] [JOB]
    [2026-04-02-104330][I][/dev/fd/63:2425]   - Parsing job env for foremans
    [2026-04-02-104330][I][/dev/fd/63:2426]   - Detected pbs scheduler
    [2026-04-02-104330][I][/dev/fd/63:2427]   - Machine: aurora
    [2026-04-02-104330][I][/dev/fd/63:2428]   - Hostname: x4514c6s0b0n0
    [2026-04-02-104330][I][/dev/fd/63:2338]   - PBS_JOBID=8414055.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
        to calculate:
          - num_hosts: 2
          - num_cores_per_host: 208
          - num_cpus_per_host: 104
          - num_gpus_per_host: 12
          - depth: 8
          - num_gpus: 24
    [2026-04-02-104331][I][/dev/fd/63:1844] [HOSTS]
    [2026-04-02-104331][I][/dev/fd/63:1846]   - Detected PBS Scheduler
    [2026-04-02-104331][I][/dev/fd/63:1864]   - HOSTFILE=/var/spool/pbs/aux/8414055.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    [2026-04-02-104331][I][/dev/fd/63:1865]   - NHOSTS=2
    [2026-04-02-104331][I][/dev/fd/63:1866]   - HOSTS:
    [2026-04-02-104331][I][/dev/fd/63:1869]     - [host:0] - x4514c6s0b0n0.hsn.cm.aurora.alcf.anl.gov
    [2026-04-02-104331][I][/dev/fd/63:1869]     - [host:1] - x4514c6s1b0n0.hsn.cm.aurora.alcf.anl.gov
    [2026-04-02-104331][I][/dev/fd/63:2030] [DIST_INFO]
    [2026-04-02-104331][I][/dev/fd/63:2031]   - HOSTFILE=/var/spool/pbs/aux/8414055.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    [2026-04-02-104331][I][/dev/fd/63:2032]   - NHOSTS=2
    [2026-04-02-104331][I][/dev/fd/63:2033]   - NGPU_PER_HOST=12
    [2026-04-02-104331][I][/dev/fd/63:2034]   - NGPUS=24
    [2026-04-02-104331][I][/dev/fd/63:2606] [✓] Finished [ezpz_setup_env]
    Benchmark output directory: outputs/benchmarks/20260402_104339
    Environment info written to outputs/benchmarks/20260402_104339/env.json
    Running 7 example(s): test, fsdp, vit, fsdp_tp, diffusion, hf, hf_trainer

    ════════════════════════════════════════════════════════════════
      [1/7] Running: test
             cmd: ezpz launch python3 -m ezpz.examples.test --model small
             log: outputs/benchmarks/20260402_104339/test.log
    ════════════════════════════════════════════════════════════════
      ✓ test completed in 2m 36s

    ════════════════════════════════════════════════════════════════
      [2/7] Running: fsdp
             ETA: ~15m 37s remaining
             cmd: ezpz launch python3 -m ezpz.examples.fsdp --model small
             log: outputs/benchmarks/20260402_104339/fsdp.log
    ════════════════════════════════════════════════════════════════
      ✓ fsdp completed in 40s

    ════════════════════════════════════════════════════════════════
      [3/7] Running: vit
             ETA: ~8m 12s remaining
             cmd: ezpz launch python3 -m ezpz.examples.vit --model small --warmup 0 --fsdp
             log: outputs/benchmarks/20260402_104339/vit.log
    ════════════════════════════════════════════════════════════════
      ✓ vit completed in 59s

    ════════════════════════════════════════════════════════════════
      [4/7] Running: fsdp_tp
             ETA: ~5m 41s remaining
             cmd: ezpz launch python3 -m ezpz.examples.fsdp_tp --model small --dataset stanfordnlp/imdb
             log: outputs/benchmarks/20260402_104339/fsdp_tp.log
    ════════════════════════════════════════════════════════════════
      ✓ fsdp_tp completed in 2m 05s

    ════════════════════════════════════════════════════════════════
      [5/7] Running: diffusion
             ETA: ~4m 46s remaining
             cmd: ezpz launch python3 -m ezpz.examples.diffusion --model small --dataset stanfordnlp/imdb
             log: outputs/benchmarks/20260402_104339/diffusion.log
    ════════════════════════════════════════════════════════════════
      ✓ diffusion completed in 45s

    ════════════════════════════════════════════════════════════════
      [6/7] Running: hf
             ETA: ~2m 50s remaining
             cmd: ezpz launch python3 -m ezpz.examples.hf --dataset_name=eliplutchok/fineweb-small-sample --streaming --model_name_or_path meta-llama/Llama-3.2-1B --bf16=true --do_train=true --do_eval=true --report-to=wandb --logging-steps=1 --max-steps=100 --optim=adamw_torch --logging-first-step --include-for-metrics=inputs,loss --max-eval-samples=100 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --block_size=2048 --fsdp=auto_wrap --output_dir=outputs/benchmarks/20260402_104339/outputs/ezpz.hf/20260402_104339
             log: outputs/benchmarks/20260402_104339/hf.log
    ════════════════════════════════════════════════════════════════
      ✓ hf completed in 3m 53s

    ════════════════════════════════════════════════════════════════
      [7/7] Running: hf_trainer
             ETA: ~1m 50s remaining
             cmd: ezpz launch python3 -m ezpz.examples.hf_trainer --dataset_name=eliplutchok/fineweb-small-sample --streaming --model_name_or_path meta-llama/Llama-3.2-1B --bf16=true --do_train=true --do_eval=true --report-to=wandb --logging-steps=1 --max-steps=100 --optim=adamw_torch --logging-first-step --include-for-metrics=inputs,loss --max-eval-samples=100 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --block_size=2048 --fsdp=auto_wrap --output_dir=outputs/benchmarks/20260402_104339/outputs/ezpz.hf_trainer/20260402_104339
             log: outputs/benchmarks/20260402_104339/hf_trainer.log
    ════════════════════════════════════════════════════════════════
      ✓ hf_trainer completed in 2m 53s

    ════════════════════════════════════════════════════════════════
      7/7 passed in 13m 54s
    ────────────────────────────────────────────────────────────────
      ✓ test           2m 36s
      ✓ fsdp              40s
      ✓ vit               59s
      ✓ fsdp_tp        2m 05s
      ✓ diffusion         45s
      ✓ hf             3m 53s
      ✓ hf_trainer     2m 53s
    ════════════════════════════════════════════════════════════════

    Generating report...
    ```

### Benchmark Report

| Key        | Value                                                         |
|------------|---------------------------------------------------------------|
| Date       | 2026-04-02T15:43:39                                           |
| Git Commit | `3f8cb86` (branch: dev)                                       |
| Job ID     | 8414055.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov (PBS) |
| Nodes      | 2 × 12 GPUs = 24 total                                       |
| Python     | 3.12.12                                                       |
| PyTorch    | 2.10.0a0+git449b176                                           |
| ezpz       | 0.11.3                                                        |

| Example    | Status | Wall Time | Steps | Final Loss | Mean dt (s) | Throughput   | W&B                                                                                       |
|------------|--------|-----------|-------|------------|-------------|--------------|-------------------------------------------------------------------------------------------|
| test       | ✅      | 2m36s     | 199   | 0.2045     | 0.0151      | —            | [link](https://wandb.ai/aurora_gpt/ezpz.examples.test/runs/1z5ossno)                      |
| fsdp       | ✅      | 40s       | —     | —          | —           | —            | [link](https://wandb.ai/aurora_gpt/ezpz.examples.fsdp/runs/14qby73f)                      |
| vit        | ✅      | 59s       | 4     | —          | —           | —            | [link](https://wandb.ai/aurora_gpt/ezpz.examples.vit/runs/v8c3zcrg)                       |
| fsdp_tp    | ✅      | 2m05s     | —     | —          | —           | —            | [link](https://wandb.ai/aurora_gpt/ezpz.examples.fsdp_tp/runs/8bwphqxa)                   |
| diffusion  | ✅      | 45s       | —     | —          | —           | —            | [link](https://wandb.ai/aurora_gpt/ezpz.examples.diffusion/runs/yz3v0qcd)                 |
| hf         | ✅      | 3m53s     | 105   | 1.5889     | 0.4126      | 124692 tok/s | [link](https://wandb.ai/aurora_gpt/ezpz-hf-meta-llama-Llama-3.2-1B/runs/att7dkdq)         |
| hf_trainer | ✅      | 2m53s     | —     | —          | —           | —            | [link](https://wandb.ai/aurora_gpt/ezpz-hf_trainer-meta-llama-Llama-3.2-1B/runs/oawqgr4g) |

## Account Fallback

If `--account` is not provided, `ezpz submit` checks these environment
variables in order:

- `PBS_ACCOUNT` (PBS)
- `SLURM_ACCOUNT` (SLURM)
- `PROJECT`
