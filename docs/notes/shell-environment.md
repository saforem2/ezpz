# 🏖️ Shell Environment

## Using an existing python environment

> [!TIP]
> If you already have a python environment with {`torch`, and `mpi4py`} then
> you're all set!
>
> Try a simple example without installing anything:
>
> ```bash
> uv run --with "git+https://github.com/saforem2/ezpz" ezpz test
> ```

In this case, the following section is _technically_ optional
but may still be useful.

## Using `ezpz/bin/utils.sh`

The
[`ezpz/bin/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh)
script provides a collection of bash functions, all prefixed with `ezpz_`.

To source it in the current shell session (requires network access):

```bash
source <(curl -fsSL https://bit.ly/ezpz-utils)
ezpz_setup_env
```

<details><summary>List of provided functions:</summary>

To see the complete list of provided functions, you can:

```bash
$ functions | egrep "ezpz.+\(" | awk '{print $1}' | pbc
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
log_message
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
```

</details>

These are useful if you're running at one of the DOE leadership computing
facilities (e.g. ALCF, OLCF, NERSC, ...) and can be used to:

1. Load appropriate modules for setting up Python
   (e.g. `module load frameworks` on Aurora or
    `module load pytorch` on Perlmutter)
1. Creating and activating virtual environments. Explicitly, it will:
   - Activate existing environment if found
   - Automagically create a new environment (inheriting system libraries if
     found) if needed

   ```bash
   ezpz_setup_python
   ```


1. Identify the machine we're running on
   - e.g. are we at one of: ALCF, OLCF, NERSC, ...?  
     If so, we:
       - Load appropriate modules which provide a base python environment
       - Build and activate a virtual environment _on top of_ this base
         environment in your working directory

Especially if any of the following applies to you:

1. You use virtual environments (or _conda_ or _pixi_ environments)
2. You work on multiple different projects, each with unique dependencies.
3. You are working on a system behind a job scheduler, such as {Slurm, PBS}.

However, there are some convenient bash functions that can be used to make life
a bit simpler, especially if you're:

1. [src/ezpz/bin/`utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh)

   ```bash
   source <(curl -fsSL https://bit.ly/ezpz-utils)
   ```


## 🐣 Getting Started

/// warning | 🚧 Work in Progress

The documentation below is a work in progress.
_Please_ feel free to provide input / suggest changes !

///

/// note

1. Source the [`src/ezpz/bin/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh) file:

   ```bash
   source <(curl -L https://bit.ly/ezpz-utils)
   ```

2. Use the `ezpz_setup_env` function to set up your environment:

   ```bash
    ezpz_setup_env
   ```

///

This will 🪄 _automagically_:

1. 🐍 **Setup Python**: Load the appropriate module(s)
   and put you inside a suitable python environment
1. 🧰 **Setup Job**: Determine the resources available in the
   current job and build a `launch` alias for launching executables

We provide a variety of helper functions designed to make your life
easier when working with job schedulers (e.g. `PBS Pro` @ ALCF or
`slurm` elsewhere).

**All** of these functions are:

- located in [`utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh)
- prefixed with `ezpz_*` (e.g. `ezpz_setup_python`)[^1]

We would like to write our application in such a way that it is able to
take full advantage of the resources allocated by the job scheduler.

That is to say, we want to have a single script with the ability to
dynamically `launch` python applications across any number of
accelerators on any of the systems under consideration.

In order to do this, there is some basic setup and information gathering
that needs to occur.

In particular, we need mechanisms for:

1. Setting up a python environment
2. Determining what system / machine we’re on
   - \+ what job scheduler we’re using (e.g. `PBS Pro` @ ALCF or
     `slurm` elsewhere)
3. Determining how many nodes have been allocated in the current job
   (`NHOSTS` $=N_{\mathrm{HOST}}$)
   - \+ Determining how many accelerators exist on each of these nodes
     (`NGPU_PER_HOST`)

This allows us to calculate the total number of accelerators (GPUs) as:

$$
N_{\mathrm{GPU}} = N_{\mathrm{HOST}} \times n_{\mathrm{GPU}}; \,\, n_{\mathrm{GPU}} = N_{\mathrm{GPU}} / N_{\mathrm{HOST}}
$$

is the number of GPUs per host.

With this we have everything we need to build the appropriate
{`mpi`{`run`, `exec`}, `slurm`} command for launching our python
application across them.

Now, there are a few functions in particular worth elaborating on.

| Function                     | Description                                                                               |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| `ezpz_setup_env`             | Wrapper around `ezpz_setup_python` `&&` `ezpz_setup_job`                                  |
| `ezpz_setup_job`             | Determine {`NGPUS`, `NGPU_PER_HOST`, `NHOSTS`}, build `launch` command alias              |
| `ezpz_setup_python`          | Wrapper around `ezpz_setup_conda` `&&` `ezpz_setup_venv_from_conda`                       |
| `ezpz_setup_conda`           | Find and activate appropriate `conda` module to load[^2]                                  |
| `ezpz_setup_venv_from_conda` | From `${CONDA_NAME}`, build or activate the virtual env located in `venvs/${CONDA_NAME}/` |

Table 1: Shell Functions

/// warning | Where am I?

_Some_ of the `ezpz_*` functions (e.g. `ezpz_setup_python`), will try
to create / look for certain directories.

In an effort to be explicit, these directories will be defined
**relative to** a `WORKING_DIR` (e.g. `"${WORKING_DIR}/venvs/"`)

This `WORKING_DIR` will be assigned to the first non-zero match found
below:

1. `PBS_O_WORKDIR`: If found in environment, paths will be relative
   to this
2. `SLURM_SUBMIT_DIR`: Next in line. If not @ ALCF, maybe using
   `slurm`…
3. `$(pwd)`: Otherwise, no worries. Use your _actual_ working
   directory.

///

### 🛠️ Setup Python

```bash
ezpz_setup_python
```

This will:

1. Automatically load and activate `conda` using the `ezpz_setup_conda`
   function.

   How this is done, in practice, varies from machine to machine:

   - **ALCF**[^3]: Automatically load the most recent `conda` module
     and activate the base environment.

   - **Frontier**: Load the appropriate AMD modules (e.g. `rocm`,
     `RCCL`, etc.), and activate base `conda`

   - **Perlmutter**: Load the appropriate `pytorch` module and activate
     environment

   - **Unknown**: In this case, we will look for a `conda`, `mamba`, or
     `micromamba` executable, and if found, use that to activate the
     base environment.

<!-- -->

/// tip | Using your own `conda`

If you are already in a conda environment when calling
`ezpz_setup_python` then it will try and use this instead.

For example, if you have a custom `conda` env at
`~/conda/envs/custom`, then this would bootstrap the `custom`
conda environment and create the virtual env in `venvs/custom/`

2. Build (or activate, if found) a virtual environment on top of (the
   active) base `conda` environment.

   By default, it will try looking in:

   - `$PBS_O_WORKDIR`, otherwise
   - `${SLURM_SUBMIT_DIR}`, otherwise
   - `$(pwd)`

   for a nested folder named `"venvs/${CONDA_NAME}"`.

   If this doesn’t exist, it will attempt to create a new virtual
   environment at this location using:

   ```bash
   python3 -m venv venvs/${CONDA_NAME} --system-site-packages
   ```

   (where we’ve pulled in the `--system-site-packages` from conda).

///

### 🧰 Setup Job

```bash
ezpz_setup_job
```

Now that we are in a suitable python environment, we need to construct
the command that we will use to run python on each of our accelerators.

To do this, we need a few things:

1. What machine we’re on (and what scheduler is it using i.e. {PBS,
   SLURM})
2. How many nodes are available in our active job
3. How many GPUs are on each of those nodes
4. What type of GPUs are they

With this information, we can then use `mpi{exec,run}` or `srun` to
launch python across all of our accelerators.

Again, how this is done will vary from machine to machine and will
depend on the job scheduler in use.

To identify where we are, we look at our `$(hostname)` and see if we’re
running on one of the known machines:

- **ALCF**[^4]: Using PBS Pro via `qsub` and `mpiexec` / `mpirun`.
  - `x4*`: **Aurora**
  - **Aurora**: `x4*` (or `aurora*` on login nodes)
  - **Sunspot**: `x1*` (or `uan*`)
  - **Sophia**: `sophia-*`
  - **Polaris** / **Sirius**: `x3*`
    - to determine between the two, we look at `"${PBS_O_HOST}"`

<!-- -->

- **OLCF**: Using Slurm via `sbatch` / `srun`.

  - `frontier*`: **Frontier**, using Slurm
  - `nid*`: Perlmutter, using Slurm

- Unknown machine: If `$(hostname)` does not match one of these patterns
  we assume that we are running on an unknown machine and will try to
  use `mpirun` as our generic launch command

  Once we have this, we can:

  1. Get `PBS_NODEFILE` from `$(hostname)`:

     - `ezpz_qsme_running`: For each (running) job owned by `${USER}`,
       print out both the jobid as well as a list of hosts the job is
       running on, e.g.:

       ```bash
       <jobid0> host00 host01 host02 host03 ...
       <jobid1> host10 host11 host12 host13 ...
       ...
       ```

     - `ezpz_get_pbs_nodefile_from_hostname`: Look for `$(hostname)` in
       the output from the above command to determine our
       `${PBS_JOBID}`.

       Once we’ve identified our `${PBS_JOBID}` we then know the
       location of our `${PBS_NODEFILE}` since they are named according
       to:

       ```bash
       jobid=$(ezpz_qsme_running | grep "$(hostname)" | awk '{print $1}')
       prefix=/var/spool/pbs/aux
       match=$(/bin/ls "${prefix}" | grep "${jobid}")
       hostfile="${prefix}/${match}"
       ```

  2. Identify number of available accelerators:

[^1]: Plus this is useful for tab-completions in your shell, e.g.:

    ```bash
    $ ezpz_<TAB>
    ezpz_check_and_kill_if_running
    ezpz_get_dist_launch_cmd
    ezpz_get_job_env
    --More--
    ```

[^2]:
    This is system dependent. See
    [`ezpz_setup_conda`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh#L50)

[^3]: Any of {Aurora, Polaris, Sophia, Sunspot, Sirius}

[^4]:
    At ALCF, if our `$(hostname)` starts with `x*`, we’re on a compute
    node.

   <!--
   will build and execute the appropriate {`mpi{exec,run}`, `srun`} command, _launching_ [`ezpz/test_dist.py`](src/ezpz/test_dist.py) (as a module, `-m`).
   -->

<!--
- There are two main, distinct components of `ezpz`:

  1. 🐍 [**Python Library**](docs/python-library.md), e.g. `import ezpz`
  1. 🏖️ [**Shell Environment**](/docs/shell-environment.md) (`ezpz_setup`)

   designed to make life easy.
-->

<!--
>
>
> - Install 🍋 `ezpz`:
>
>     ```bash
>     python3 -m pip install "git+https://github.com/saforem2/ezpz"
>     ```
>
-->

<!--
1. [🐣 Getting Started](#-getting-started)
   1. 📝 [Example](#-example)
-->

<!--
1. [🐚 **Shell Utilities**](#-shell-utilities)
   1. [🏖️ Setup Shell Environment](#%EF%B8%8F-setup-shell-environment)
       1. [🛠️ Setup Python](#%EF%B8%8F-setup-python)
       1. [🧰 Setup Job](#-setup-job)
 1. [🐍 **Python Library**](#-python-library)
-->

<!--

- `ezpz` is:

  - A standalone Python library (`import ezpz`)
  - A collection of Shell utilities (`ezpz_*`)

  with the goal of making life easy.

-->

<!--

### 📝 Example

We provide a complete, entirely self-contained example in
[docs/example.md](/docs/example.md) that walks through:

1. Setting up a suitable python environment + installing `ezpz` into it
2. Launching a (simple) distributed training job across all available resources
   in your {slurm, PBS} job allocation.

-->
