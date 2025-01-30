# 🍋 ezpz

> <q align="center" style="text-align:center;"><i>Work smarter, not harder</i></q>

<!--
<h4 align="center">
    <p>
        <a href="#installation">Installation</a> •
        <a href="#quick-start">Quick Start</a> •
        <a href="#features">Features</a> •
        <a href="CONTRIBUTING.md">Contributing</a>
    <p>
</h4>

<h3 align="center">
    <a href="https://huggingface.co/nanotron"><img style="float: middle; padding: 10px 10px 10px 10px;" width="60" height="55" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png" /></a>
</h3>
<h3 align="center">
<p>Pretraining models made easy
</h3>
-->

<!--
# 🍋 `ezpz`
--

<!--- <details closed><summary>Contents:</summary></details>-->

1. [🐣 Getting Started](#-getting-started)
   1. 📝 [Example](#-example)
1. [🐚 Shell Utilities](#-shell-utilities)
   1. [🏖️ Setup Shell Environment](#%EF%B8%8F-setup-shell-environment)
       1. [🛠️ Setup Python](#%EF%B8%8F-setup-python)
       1. [🧰 Setup Job](#-setup-job)
 1. [🐍 Python Library](#-python-library) 

> [!IMPORTANT]
> The documentation below is a work in progress.  
> *Please* feel free to provide input / suggest changes !

## 🐣 Getting Started

<!--

- `ezpz` is:

  - A standalone Python library (`import ezpz`)
  - A collection of Shell utilities (`ezpz_*`)

  with the goal of making life easy.

-->

There are two main, distinct components of `ezpz`:

1.  🐍 [**Python Library**](#python-library) (`import ezpz`)
2.  🐚 [**Shell Utilities**](#shell-utilities) (`ezpz_*`)

designed to make life easy.

### 📝 Example

We provide a complete, entirely self-contained example in
[docs/example.md](/docs/example.md) that walks through:

1. Setting up a suitable python environment + installing `ezpz` into it
2. Launching a (simple) distributed training job across all available resources
   in your {slurm, PBS} job allocation.

## 🐚 Shell Utilities

The Shell Utilities can be roughly broken up further into two main
components:

<!--1. [🏖️ Setup Shell Environment](#%EF%B8%8F-setup-shell-environment)-->
1. [🛠️ Setup Python](#%EF%B8%8F-setup-python)
1. [🧰 Setup Job](#-setup-job)

<!--
        1.  🐍 [Setup Conda](#-setup-conda)
        2.  👻 [Setup Virtual Environment](#-setup-virtual-environment)
-->

We provide a variety of helper functions designed to make your life
easier when working with job schedulers (e.g. `PBS Pro` @ ALCF or
`slurm` elsewhere).

**All** of these functions are:

- located in [`utils.sh`](../../src/ezpz/bin/utils.sh)
- prefixed with `ezpz_*` (e.g. `ezpz_setup_python`)[^1]

To use these, we can `source` the file directly via:

``` bash
export PBS_O_WORKDIR=$(pwd) # if on ALCF
source /dev/stdin <<< $(curl 'https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh')
```

We would like to write our application in such a way that it is able to
take full advantage of the resources allocated by the job scheduler.

That is to say, we want to have a single script with the ability to
dynamically `launch` python applications across any number of
accelerators on any of the systems under consideration.

In order to do this, there is some basic setup and information gathering
that needs to occur.

In particular, we need mechanisms for:

1.  Setting up a python environment
2.  Determining what system / machine we’re on
    - \+ what job scheduler we’re using (e.g. `PBS Pro` @ ALCF or
      `slurm` elsewhere)
3.  Determining how many nodes have been allocated in the current job
    (`NHOSTS`)
    - \+ Determining how many accelerators exist on each of these nodes
      (`NGPU_PER_HOST`)

This allows us to calculate the total number of accelerators (GPUs) as:
$N_{\mathrm{GPU}} = N_{\mathrm{HOST}} \times n_{\mathrm{GPU}}$

where $n_{\mathrm{GPU}} = N_{\mathrm{GPU}} / N_{\mathrm{HOST}}$ is the
number of GPUs per host.

With this we have everything we need to build the appropriate
{`mpi`{`run`, `exec`}, `slurm`} command for launching our python
application across them.

Now, there are a few functions in particular worth elaborating on.

<div id="tbl-shell-fns">

| Function                     | Description                                                                               |
|------------------------------|-------------------------------------------------------------------------------------------|
| `ezpz_setup_env`             | Wrapper around `ezpz_setup_python` `&&` `ezpz_setup_job`                                  |
| `ezpz_setup_job`             | Determine {`NGPUS`, `NGPU_PER_HOST`, `NHOSTS`}, build `launch` command alias              |
| `ezpz_setup_python`          | Wrapper around `ezpz_setup_conda` `&&` `ezpz_setup_venv_from_conda`                       |
| `ezpz_setup_conda`           | Find and activate appropriate `conda` module to load[^2]                                  |
| `ezpz_setup_venv_from_conda` | From `${CONDA_NAME}`, build or activate the virtual env located in `venvs/${CONDA_NAME}/` |

Table 1: Shell Functions

</div>

> [!WARNING]
>
> ### Where am I?
>
> *Some* of the `ezpz_*` functions (e.g. `ezpz_setup_python`), will try
> to create / look for certain directories.
>
> In an effort to be explicit, these directories will be defined
> **relative to** a `WORKING_DIR` (e.g. `"${WORKING_DIR}/venvs/"`)
>
> This `WORKING_DIR` will be assigned to the first non-zero match found
> below:
>
> 1.  `PBS_O_WORKDIR`: If found in environment, paths will be relative
>     to this
> 2.  `SLURM_SUBMIT_DIR`: Next in line. If not @ ALCF, maybe using
>     `slurm`…
> 3.  `$(pwd)`: Otherwise, no worries. Use your *actual* working
>     directory.

#### 🛠️ Setup Python

``` bash
ezpz_setup_python
```

This will:

1.  Automatically load and activate `conda` using the `ezpz_setup_conda`
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

> [!TIP]
>
> **Using your own `conda`**
>
> If you are already in a conda environment when calling
> `ezpz_setup_python` then it will try and use this instead.
>
> For example, if you have a custom `conda` env at
> `~/conda/envs/custom`, then this would bootstrap the `custom`
> conda environment and create the virtual env in `venvs/custom/`


2.  Build (or activate, if found) a virtual environment on top of (the
    active) base `conda` environment.

    By default, it will try looking in:

    - `$PBS_O_WORKDIR`, otherwise
    - `${SLURM_SUBMIT_DIR}`, otherwise
    - `$(pwd)`

    for a nested folder named `"venvs/${CONDA_NAME}"`.

    If this doesn’t exist, it will attempt to create a new virtual
    environment at this location using:

    ``` bash
    python3 -m venv venvs/${CONDA_NAME} --system-site-packages
    ```

    (where we’ve pulled in the `--system-site-packages` from conda).

#### 🧰 Setup Job

``` bash
ezpz_setup_job
```

Now that we are in a suitable python environment, we need to construct
the command that we will use to run python on each of our acceleratorss.

To do this, we need a few things:

1.  What machine we’re on (and what scheduler is it using i.e. {PBS,
    SLURM})
2.  How many nodes are available in our active job
3.  How many GPUs are on each of those nodes
4.  What type of GPUs are they

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

  1.  Get `PBS_NODEFILE` from `$(hostname)`:

      - `ezpz_qsme_running`: For each (running) job owned by `${USER}`,
        print out both the jobid as well as a list of hosts the job is
        running on, e.g.:

        ``` bash
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

        ``` bash
        jobid=$(ezpz_qsme_running | grep "$(hostname)" | awk '{print $1}')
        prefix=/var/spool/pbs/aux
        match=$(/bin/ls "${prefix}" | grep "${jobid}")
        hostfile="${prefix}/${match}"
        ```

  2.  Identify number of available accelerators:

## 🐍 Python Library

### 👀 Overview

Launch and train across all your accelerators, using your favorite framework +
backend combo.

`ezpz` simplifies the process of:

- <details><summary>Setting up + launching distributed training:</summary>

    - <details closed><summary><code>import ezpz as ez</code></summary>

        - `RANK = `
          [`ez.setup_torch(backend=backend)`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L551)
          [for `backend` $\in$ \{`DDP`, `deepspeed`, `horovod`}]{.dim-text}

        - `RANK =`
          [`ez.get_rank()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#396)

        - `LOCAL_RANK =`
          [`ez.get_local_rank()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#448)

        - `WORLD_SIZE =`
          [`ez.get_world_size()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L417)

        [(see [`ezpz/dist.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py) for more details).]{.dim-text}

    </details>

</details>

- <details closed><summary>Using your favorite framework:</summary>

    - `framework=pytorch` + `backend={DDP, deepspeed, horovod}`

    - `framework=tensorflow` + `backend=horovod`

    - [`ez.get_torch_device()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L332): {`cuda`, `xpu`, `mps`, `cpu`}

    - [`ez.get_torch_backend()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L348): {`nccl`, `ccl`, `gloo`}

  _2ez_ 😎. (see [frameworks](#frameworks) for additional details)

</details>

- <details closed><summary>Writing device agnostic code:</summary>

    - <details><summary><a href="https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L332"><code>ezpz.get_torch_device()</code></a></summary>

        ```python
        >>> import ezpz as ez
        >>> DEVICE = ez.get_torch_device()
        >>> model = torch.nn.Linear(10, 10)
        >>> model.to(DEVICE)
        >>> x = torch.randn((10, 10), device=DEVICE)
        >>> y = model(x)
        >>> y.device
        device(type='mps', index=0)
        ```

    </details>

</details>

- <details closed><summary>Using <code>wandb</code>:</summary>

    - `ez.setup_wandb(project_name='ezpz')`

</details>

- **Full support** for any {`device` + `framework` + `backend`}:
    - device: {`GPU`, `XPU`, `MPS`, `CPU`}
    - framework: {`torch`, `deepspeed`, `horovod`, `tensorflow`}
    - backend: {`DDP`, `deepspeed`, `horovod`}
 
### Install

To install[^5]:

``` bash
python3 -m pip install -e "git+https://github.com/saforem2/ezpz#egg=ezpz" --require-virtualenv
```

- 📂 `ezpz` / `src` / [`ezpz/`](https://github.com/saforem2/ezpz)
  - 📂
    [`bin/`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/):
    - [`utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh):
      Shell utilities for `ezpz`
  - 📂
    [`conf/`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/conf/):
    - ⚙️
      [`config.yaml`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/conf/config.yaml):
      Default `TrainConfig` object
    - ⚙️
      [`ds_config.json`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/conf/ds_config.json):
      DeepSpeed configuration
  - 📂
    [`log/`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/log/):
    Logging configuration.
  - 🐍
    [`__about__.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/__about__.py):
    Version information
  - 🐍
    [`__init__.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/__init__.py):
    Main module
  - 🐍
    [`__main__.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/__main__.py):
    Entry point
  - 🐍
    [`configs.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/configs.py):
    Configuration module
  - 🐍[`cria.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/create.py):
    Baby Llama
  - 🐍[**`dist.py`**](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py):
    Distributed training module
  - 🐍[`history.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/history.py):
    History module
  - 🐍[`jobs.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/jobs.py):
    Jobs module
  - 🐍[`model.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/model.py):
    Model module
  - 🐍[`plot.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/plot.py):
    Plot modul
  - 🐍[`profile.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/profile.py):
    Profile module
  - 🐍[`runtime.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/runtime.py):
    Runtime module
  - 🐍[`test.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test.py):
    Test module
  - 🐍[**`test_dist.py`**](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py):
    Distributed training test module
  - 🐍[`train.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/train.py):
    train module
  - 🐍[`trainer.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/trainer.py):
    trainer module
  - 🐍[`utils.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/utils.py):
    utility module

``` bash
📂 /ezpz/src/ezpz/
┣━━ 📂 bin/
┃   ┣━━ 📄 affinity.sh
┃   ┣━━ 📄 getjobenv
┃   ┣━━ 📄 savejobenv
┃   ┣━━ 📄 saveslurmenv
┃   ┣━━ 📄 setup.sh
┃   ┣━━ 📄 train.sh
┃   ┗━━ 📄 utils.sh
┣━━ 📂 conf/
┃   ┣━━ 📂 hydra/
┃   ┃   ┗━━ 📂 job_logging/
┃   ┃       ┣━━ ⚙️ colorlog1.yaml
┃   ┃       ┣━━ ⚙️ custom.yaml
┃   ┃       ┗━━ ⚙️ enrich.yaml
┃   ┣━━ 📂 logdir/
┃   ┃   ┗━━ ⚙️ default.yaml
┃   ┣━━ ⚙️ config.yaml
┃   ┣━━ 📄 ds_config.json
┃   ┗━━ ⚙️ ds_config.yaml
┣━━ 📂 log/
┃   ┣━━ 📂 conf/
┃   ┃   ┗━━ 📂 hydra/
┃   ┃       ┗━━ 📂 job_logging/
┃   ┃           ┗━━ ⚙️ enrich.yaml
┃   ┣━━ 🐍 __init__.py
┃   ┣━━ 🐍 __main__.py
┃   ┣━━ 🐍 config.py
┃   ┣━━ 🐍 console.py
┃   ┣━━ 🐍 handler.py
┃   ┣━━ 🐍 style.py
┃   ┣━━ 🐍 test.py
┃   ┗━━ 🐍 test_log.py
┣━━ 🐍 __about__.py
┣━━ 🐍 __init__.py
┣━━ 🐍 __main__.py
┣━━ 🐍 configs.py
┣━━ 🐍 cria.py
┣━━ 🐍 dist.py
┣━━ 🐍 history.py
┣━━ 🐍 jobs.py
┣━━ 🐍 loadjobenv.py
┣━━ 🐍 model.py
┣━━ 🐍 plot.py
┣━━ 🐍 profile.py
┣━━ 🐍 runtime.py
┣━━ 🐍 savejobenv.py
┣━━ 🐍 test.py
┣━━ 🐍 test_dist.py
┣━━ 🐍 train.py
┣━━ 🐍 trainer.py
┗━━ 🐍 utils.py
```

[^1]: Plus this is useful for tab-completions in your shell, e.g.:

    ``` bash
    $ ezpz_<TAB>
    ezpz_check_and_kill_if_running
    ezpz_get_dist_launch_cmd
    ezpz_get_job_env
    --More--
    ```

[^2]: This is system dependent. See
    [`ezpz_setup_conda`](../../src/ezpz/bin/utils.sh)

[^3]: Any of {Aurora, Polaris, Sophia, Sunspot, Sirius}

[^4]: At ALCF, if our `$(hostname)` starts with `x*`, we’re on a compute
    node.

[^5]: Note the `--require-virtualenv` isn’t *strictly* required, but I
    highly recommend to always try and work within a virtual
    environment, when possible.

