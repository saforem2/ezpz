# Shell Environment

ezpz ships a collection of Bash helper functions (all prefixed `ezpz_`) that
handle Python environment setup, job discovery, and launch-command construction
on HPC systems.  The two main entry points are:

```bash
source <(curl -fsSL https://bit.ly/ezpz-utils)
ezpz_setup_env      # sets up python + job in one shot
```

/// tip | Already have a working environment?

If you already have a Python environment with `torch` and `mpi4py`, you can
skip the shell helpers entirely and run:

```bash
uv run --with "git+https://github.com/saforem2/ezpz" ezpz test
```

///

## Sourcing `utils.sh`

The helpers live in
[`src/ezpz/bin/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh).
Source them into your current shell session:

```bash
source <(curl -fsSL https://bit.ly/ezpz-utils)
```

??? info "Full list of provided functions"

    ```bash
    $ functions | egrep "ezpz.+\(" | awk '{print $1}' | sort
    _ezpz_install_from_git
    _ezpz_setup_conda_polaris
    ezpz_activate_or_create_micromamba_env
    ezpz_build_bdist_wheel_from_github_repo
    ezpz_check_and_kill_if_running
    ezpz_check_if_already_built
    ezpz_check_working_dir
    ezpz_check_working_dir_pbs
    ezpz_check_working_dir_slurm
    ezpz_ensure_micromamba_hook
    ezpz_ensure_uv
    ezpz_generate_cpu_ranges
    ezpz_get_cpu_bind_aurora
    ezpz_get_dist_launch_cmd
    ezpz_get_job_env
    ezpz_get_jobenv_file
    ezpz_get_jobid_from_hostname
    ezpz_get_machine_name
    ezpz_get_num_gpus_nvidia
    ezpz_get_num_gpus_per_host
    ezpz_get_num_gpus_total
    ezpz_get_num_hosts
    ezpz_get_num_xpus
    ezpz_get_pbs_env
    ezpz_get_pbs_jobid
    ezpz_get_pbs_jobid_from_nodefile
    ezpz_get_pbs_nodefile_from_hostname
    ezpz_get_python_prefix_nersc
    ezpz_get_python_root
    ezpz_get_scheduler_type
    ezpz_get_shell_name
    ezpz_get_slurm_env
    ezpz_get_slurm_running_jobid
    ezpz_get_slurm_running_nodelist
    ezpz_get_tstamp
    ezpz_get_venv_dir
    ezpz_get_working_dir
    ezpz_getjobenv_main
    ezpz_has
    ezpz_head_n_from_pbs_nodefile
    ezpz_install
    ezpz_install_and_setup_micromamba
    ezpz_install_micromamba
    ezpz_install_uv
    ezpz_is_sourced
    ezpz_kill_mpi
    ezpz_launch
    ezpz_load_new_pt_modules_aurora
    ezpz_load_python_modules_nersc
    ezpz_make_slurm_nodefile
    ezpz_parse_hostfile
    ezpz_prepare_repo_in_build_dir
    ezpz_print_hosts
    ezpz_print_job_env
    ezpz_qsme_running
    ezpz_realpath
    ezpz_require_file
    ezpz_reset
    ezpz_reset_pbs_vars
    ezpz_save_deepspeed_env
    ezpz_save_dotenv
    ezpz_save_ds_env
    ezpz_save_pbs_env
    ezpz_save_slurm_env
    ezpz_savejobenv_main
    ezpz_set_proxy_alcf
    ezpz_setup_alcf
    ezpz_setup_conda
    ezpz_setup_conda_aurora
    ezpz_setup_conda_frontier
    ezpz_setup_conda_perlmutter
    ezpz_setup_conda_polaris
    ezpz_setup_conda_sirius
    ezpz_setup_conda_sophia
    ezpz_setup_conda_sunspot
    ezpz_setup_env
    ezpz_setup_env_pt28_aurora
    ezpz_setup_env_pt29_aurora
    ezpz_setup_host
    ezpz_setup_host_pbs
    ezpz_setup_host_slurm
    ezpz_setup_install
    ezpz_setup_job
    ezpz_setup_job_alcf
    ezpz_setup_job_slurm
    ezpz_setup_new_uv_venv
    ezpz_setup_python
    ezpz_setup_python_alcf
    ezpz_setup_python_nersc
    ezpz_setup_python_pt28_aurora
    ezpz_setup_python_pt29_aurora
    ezpz_setup_python_pt_new_aurora
    ezpz_setup_srun
    ezpz_setup_uv_venv
    ezpz_setup_venv_from_conda
    ezpz_setup_venv_from_pythonuserbase
    ezpz_show_env
    ezpz_tail_n_from_pbs_nodefile
    ezpz_timeit
    ezpz_write_job_info
    ezpz_write_job_info_slurm
    log_message
    ```

These are especially useful on DOE leadership computing facilities
(ALCF, OLCF, NERSC) where you need to load the right modules, build
virtual environments on top of system conda, and construct MPI launch commands.

## `ezpz_setup_env`

The recommended one-liner.  Internally it calls:

1. **`ezpz_setup_python`** — load modules and activate a virtual environment
2. **`ezpz_setup_job`** — discover job resources and build a `launch` alias

```bash
ezpz_setup_env
```

### Key Functions

| Function                     | Description                                                                               |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| `ezpz_setup_env`             | Wrapper: `ezpz_setup_python && ezpz_setup_job`                                            |
| `ezpz_setup_job`             | Determine `NGPUS`, `NGPU_PER_HOST`, `NHOSTS`; build `launch` alias                        |
| `ezpz_setup_python`          | Wrapper: `ezpz_setup_conda && ezpz_setup_venv_from_conda`                                 |
| `ezpz_setup_conda`           | Find and activate the appropriate conda module[^1]                                        |
| `ezpz_setup_venv_from_conda` | From `${CONDA_NAME}`, build or activate the venv in `venvs/${CONDA_NAME}/`                |

### Shell Variables Exported

After `ezpz_setup_env` completes, the following are available in your shell
**and** visible to the Python side via `os.environ`:

| Variable        | Example          | Description                       |
|-----------------|------------------|-----------------------------------|
| `NHOSTS`        | `2`              | Number of hosts in the job        |
| `NGPU_PER_HOST` | `12`             | Accelerators per host             |
| `NGPUS`         | `24`             | Total accelerators                |
| `HOSTFILE`      | `/var/spool/...` | Path to hostfile                  |
| `DIST_LAUNCH`   | `mpiexec ...`    | Full launch command prefix        |
| `JOBENV_FILE`   | `.jobenv`        | Saved job environment file        |

## Setup Python

```bash
ezpz_setup_python
```

This will:

1. **Load and activate conda** via `ezpz_setup_conda`.

    How this works varies by machine:

    - **ALCF** (Aurora, Polaris, Sophia, Sunspot, Sirius): Load the most
      recent conda module and activate the base environment.
    - **Frontier**: Load AMD modules (ROCm, RCCL, etc.) and activate base conda.
    - **Perlmutter**: Load the appropriate `pytorch` module and activate.
    - **Unknown**: Look for a `conda`, `mamba`, or `micromamba` executable
      and use that to activate base.

    /// tip | Using your own conda

    If you are already in a conda environment when calling
    `ezpz_setup_python`, it will use that environment as the base.
    For example, if `~/conda/envs/custom` is active, the virtual env will
    be created in `venvs/custom/`.

    ///

2. **Build or activate a virtual environment** on top of the active conda
   base, at `venvs/${CONDA_NAME}/`.

    The venv directory is resolved relative to the first non-empty match:

    1. `$PBS_O_WORKDIR`
    2. `$SLURM_SUBMIT_DIR`
    3. `$(pwd)`

    If the venv doesn't exist, it is created with:

    ```bash
    python3 -m venv venvs/${CONDA_NAME} --system-site-packages
    ```

## Setup Job

```bash
ezpz_setup_job
```

Once inside a suitable Python environment, this function constructs the
launch command. It needs:

1. **Which machine** we're on (and which scheduler: PBS or SLURM)
2. **How many nodes** are allocated
3. **How many GPUs per node**
4. **What type of GPUs** they are

With this it builds the `mpiexec` / `mpirun` / `srun` command and exports
it as the `DIST_LAUNCH` variable (and a convenient `launch` alias).

### Machine Detection

The function `ezpz_get_machine_name` inspects `$(hostname)`:

| Hostname prefix | Machine       | Scheduler |
|-----------------|---------------|-----------|
| `x4*`           | Aurora        | PBS       |
| `x1*`           | Sunspot       | PBS       |
| `x3*`           | Polaris[^2]   | PBS       |
| `sophia-*`      | Sophia        | PBS       |
| `frontier*`     | Frontier      | SLURM     |
| `login*`/`nid*` | Perlmutter    | SLURM     |
| _(other)_       | Unknown       | fallback  |

### PBS Hostfile Discovery

On PBS systems, the hostfile is found by:

1. `ezpz_qsme_running` — list all running jobs for `$USER` with their hosts
2. `ezpz_get_pbs_nodefile_from_hostname` — match `$(hostname)` to a job ID
3. Resolve the hostfile at `/var/spool/pbs/aux/${PBS_JOBID}.*`

### SLURM Hostfile Discovery

On SLURM systems, the hostfile is generated from `$SLURM_NODELIST` using
`scontrol show hostnames` and written to a local file.

[^1]:
    System-dependent. See
    [`ezpz_setup_conda`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh)
    for the full implementation.

[^2]:
    For `x3*` hostnames, `$PBS_O_HOST` is checked to distinguish Polaris
    from Sirius.
