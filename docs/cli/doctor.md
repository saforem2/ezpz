# 🩺 Doctor

```bash
ezpz doctor
ezpz doctor --json   # machine-friendly output for CI
```

Health-check your environment and ensure that everything is ready to go!

Checks MPI, scheduler detection, Torch import + accelerators, and `wandb`
readiness, returning non-zero on errors.

<!-- Health-check your environment and ensure that `ezpz` is installed correctly -->
<!---->
<!-- `ezpz doctor` inspects the active environment and reports whether common launch -->
<!-- prerequisites are satisfied before you submit a distributed job. -->
<!---->
<!-- The command is available from the main CLI group: -->
<!---->
<!-- ```bash -->
<!-- ezpz doctor -->
<!-- ezpz doctor --json   # machine-friendly output for CI -->
<!-- ``` -->

## What gets checked?

Explicitly, this checks:

- Python Environment (`conda` / `mamba` / `venv`) details
- Torch installed and accelerators visible
- Weights \& Biases (`wandb`) installation and authentication
- MPI installation and `mpi4py` bindings
- Job scheduler (PBS / Slurm) and determines specifics of _active_ job.

  This includes the number of available nodes, how many accelerators per node,
  and the total number of accelerators.

| Check         | Status Criteria                                                                          | Remedy Hints                                                      |
| ------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **MPI**       | Confirms both `mpi4py` and a launcher (`mpiexec`/`mpirun`) are accessible.               | Suggests installing the missing component.                        |
| **Scheduler** | Identifies PBS/SLURM based on environment variables.                                     | Points to the scheduler plug-in mechanism if the host is unknown. |
| `wandb`       | Detects the `wandb` Python package plus credentials or offline mode.                     | Recommends `WANDB_MODE=offline` when credentials are absent.      |
| `torch`       | Validates the PyTorch install and accelerator visibility (`cuda`, `xpu`, `mps`, or CPU). | Advises setting `TORCH_DEVICE` for CPU-only runs.                 |

Each line is reported with a severity (`OK`, `WARNING`, `ERROR`) and, when
relevant, an actionable remediation hint.

## Example Output

<details closed><summary><code>localhost</code>:</summary>

```bash
(ezpz)
#[12/26/25 @ 15:06:12][~/v/s/ezpz][distributed-metrics][$✘»!?]
; ezpz doctor
== Runtime Context ==
User: samforeman
Machine: arm64
Hostname: Sams-MacBook-Pro-2.local
PBS_JOBID: N/A
PBS_NODEFILE: N/A
ezpz: 0.9.0 (/Users/samforeman/vibes/saforem2/ezpz/src/ezpz/__init__.py)

Module list:
  bash: line 1: module: command not found

Active environment:
  Virtual env: /Users/samforeman/vibes/saforem2/ezpz/venvs/sams-macbook-pro-2.local/ezpz

Python: /Users/samforeman/vibes/saforem2/ezpz/venvs/sams-macbook-pro-2.local/ezpz/bin/python3 (3.12.8)
PyTorch: 2.9.1 (/Users/samforeman/vibes/saforem2/ezpz/venvs/sams-macbook-pro-2.local/ezpz/lib/python3.12/site-packages/torch/__init__.py)

[✅ OKAY] [mpi      ]: mpi4py import succeeded and an MPI launcher was found.
[✅ OKAY] [wandb    ]: wandb authentication provided in '~/.netrc' Should be all set.
[✅ OKAY] [torch    ]: PyTorch detected at least one accelerator.
[✅ OKAY] [hostfile ]: No scheduler detected; HOSTFILE not required.
[⚠️ WARN] [scheduler]: No scheduler detected – assuming local launch mode.
          ↳ Set scheduler environment variables or configure a custom adapter if running under a job queue.
[2025-12-23-162222] Execution time: 5s sec
```

</details>

<details closed><summary>Aurora at ALCF:</summary>

```bash
; ezpz doctor
== Runtime Context ==
User: foremans
Machine: x86_64
Hostname: x4712c1s0b0n0
PBS_JOBID: 8227686.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
PBS_NODEFILE: /var/spool/pbs/aux/8227686.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
ezpz: v0.9.0 (/lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/ezpz-distributed-metrics/src/ezpz/__init__.py)

Module list:
  gcc-runtime/13.3.0-ghotoln <H>
  gmp/6.3.0-mtokfaw <H>
  mpfr/4.2.1-gkcdl5w <H>
  mpc/1.3.1-rdrlvsl <H>
  gcc/13.3.0
  oneapi/release/2025.2.0
  libxml2/2.13.5
  hwloc/2.12.1-level-zero
  mpich/opt/develop-git.6037a7a
  libfabric/1.22.0
  cray-pals/1.8.0
  cray-libpals/1.8.0
  hdf5/1.14.6
  frameworks/2025.2.0

Active environment:
  Virtual env: /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/ezpz-distributed-metrics/venvs/aurora/ezpz-distributed-metrics-aurora_frameworks-2025.2.0
  Conda env: /opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0 (/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0)

Python: /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/ezpz-distributed-metrics/venvs/aurora/ezpz-distributed-metrics-aurora_frameworks-2025.2.0/bin/python3 (3.10.14)
PyTorch: 2.8.0a0+gitba56102 (/opt/aurora/25.190.0/frameworks/aurora_frameworks-2025.2.0/lib/python3.10/site-packages/torch/__init__.py)

Scheduler resources:
  NHOSTS: 2
  NGPU_PER_HOST: 12
  NGPUS: 24

[✅ OKAY] [mpi      ]: mpi4py import succeeded and an MPI launcher was found.
[✅ OKAY] [wandb    ]: wandb authentication provided in '~/.netrc' Should be all set.
[✅ OKAY] [torch    ]: PyTorch detected at least one accelerator.
[✅ OKAY] [hostfile ]: HOSTFILE=/var/spool/pbs/aux/8227686.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov (2 hosts).
[✅ OKAY] [scheduler]: Detected active scheduler: PBS.
```

</details>

<details closed><summary>JSON Output from Sunspot at ALCF</summary>

```bash
#[aurora_frameworks-2025.2.0]
#[~][23s]
#[12/26/25 @ 15:24:07][x1921c0s3b0n0]
; uv run --python=$(which python3) --with "git+https://github.com/saforem2/ezpz@distributed-metrics" ezpz doctor --json
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
    "name": "wandb",
    "status": "ok",
    "message": "wandb authentication provided in '~/.netrc' Should be all set.",
    "remedy": null
  },
  {
    "name": "torch",
    "status": "ok",
    "message": "PyTorch detected at least one accelerator.",
    "remedy": null
  },
  {
    "name": "hostfile",
    "status": "ok",
    "message": "No scheduler detected; HOSTFILE not required.",
    "remedy": null
  },
  {
    "name": "scheduler",
    "status": "ok",
    "message": "Detected active scheduler: PBS.",
    "remedy": null
  }
]
```

</details>
