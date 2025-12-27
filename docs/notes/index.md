# 📝 Notes

/// warning | 🚧 Work in Progress

The notes section is being refreshed; some pages may be incomplete or out of
date. When in doubt, refer to the source code at
<https://github.com/saforem2/ezpz> or open an issue with questions or problems.

///

<!--

## 2024-12-30

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

  _2ez_ 😎.

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

## 2024-11-06

- [ ] Save `PBS_*` env to `~/pbs_jobenvs/${PBS_JOBID}.env` when dumping vars in [`utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh)

## Getting Started

There are two main, distinct components of `ezpz`:

1. [Shell interface](#shell-interface)
2. [Python Library](#python-library)

### Shell Interface

- [`bin/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh):
  - Provides various (`bash` / shell) helper functions to make life easy
  - Designed to be `source`-d, e.g.

     ```bash
     source ezpz/src/ezpz/bin/utils.sh
     ```

  - All functions prefixed with `ezpz_`

To use:

```bash
git clone https://github.com/saforem2/ezpz deps/ezpz
# on ALCF:
export PBS_O_WORKDIR=$(pwd)
source deps/ezpz/src/ezpz/bin/utils.sh
ezpz_setup_python
# from a compute node:
ezpz_setup_job
```

### Python Library

WIP

## Old

<details closed><summary>Old:</summary>

## Startup Files

In your `{.bashrc,.zshrc}`, you can:

```bash
ezpz_setup_alcf() {
    file=$(mktemp)
    curl -Ls https://raw.githubusercontent.com/saforem2/ezpz/main/src/ezpz/bin/utils.sh > "${file}"
    echo "Saving 'utils.sh' to ${file} and sourcing..."
    source "${file}" || exit
    hn=$(hostname)
    setup_alcf
}

hn=$(hostname)
if [[ "${hn}" == x1 || "${hn}" == x]]
if [[ $(hostname) == x3* || $(hostname) == polaris* ]]; then

elif [[ $(hostname) == x4* || $(hostname) == aurora* ]]; then
elif [[ $(hostname) == x1* || $(hostname) == uan* ]]; then
elif [[ $(hostname) == bastion* ]]; then
else
fi

MACHINE=$(echo "${machine}" | tr '[:upper:]' '[:lower:]')
export PATH="${HOME}/bin/${MACHINE}:${PATH}"
export HISTFILE="$HOME/.zsh_history-${MACHINE}"
# export CODESTATS_API_KEY="SFMyNTY.YzJGdFptOXlaVzFoYmc9PSMjTWpBNE1UST0.NQ4Oy3FSJcT4nMaMlVnYcnCtPc2mqImViSGiIxyJFrg"
export ZSH_COMPDUMP="${ZSH}/cache/.zcompdump-${MACHINE}"
```

1. Clone `ezpz` + navigate into it:

    ```bash
    git clone https://github.com/saforem2/ezpz
    cd ezpz
    ```

2. Source [`src/ezpz/bin/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh)

    ```bash
    #[🌌][01:16:07 PM][foremans@x4017c4s5b0n0][…/ezpz][🌱 jobs-rewrite]via ⨁ v1.3.450
    $ export PBS_O_WORKDIR=$(pwd) && source src/ezpz/bin/utils.sh
    Using WORKING_DIR: /home/foremans/2024-07-10-131541/ezpz
    ```

3. Setup `python`:

    ```bash
    #[🌌][01:16:17 PM][foremans@x4017c4s5b0n0][…/ezpz][🌱 jobs-rewrite]via ⨁ v1.3.450
    $ setup_python
    No conda_prefix OR virtual_env found in environment...
    Setting up conda...
    machine name: aurora

    The following have been reloaded with a version change:
      1) intel_compute_runtime/release/821.36 => intel_compute_runtime/release/803.29
      2) oneapi/eng-compiler/2024.04.15.002 => oneapi/release/2024.1

    Found conda at: /opt/aurora/24.086.0/frameworks/aurora_nre_models_frameworks-2024.1
    No VIRTUAL_ENV found in environment!
        - Trying to setup from /opt/aurora/24.086.0/frameworks/aurora_nre_models_frameworks-2024.1
        - Using VENV_DIR=/home/foremans/2024-07-10-131541/ezpz/venvs/aurora_nre_models_frameworks-2024.1

        - Creating a new virtual env on top of aurora_nre_models_frameworks-2024.1 in /home/foremans/2024-07-10-131541/ezpz/venvs/aurora_nre_models_frameworks-2024.1
    [python] Using: /home/foremans/2024-07-10-131541/ezpz/venvs/aurora_nre_models_frameworks-2024.1/bin/python3
    ```

4. Setup ALCF:

    1. via `bash` script:

      <details closed><summary><code>.jobenv</code>:<summary>

        ```bash
        #[aurora_nre_models_frameworks-2024.1](👻 aurora_nre_models_frameworks-2024.1)
        #[🌌][01:16:45 PM][foremans@x4017c4s5b0n0][…/ezpz][🌱 jobs-rewrite]via ⨁ v1.3.450 [⏱ 13s]
        $ setup_alcf

        [ezpz/bin/utils.sh]

        [2024-07-10-131719]
            • USER=foremans
            • MACHINE=aurora
            • HOST=x4017c4s5b0n0

        [setupHost]
            • Using hostfile: /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
            • Found in environment:
                • HOSTFILE: /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
                • Writing PBS vars to: /home/foremans/.pbsenv

        [save_pbs_env]
            • Using:
                • hostfile: /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
                • jobenv_file: /home/foremans/.pbsenv
              to calculate:
                • num_hosts: 2
                • num_gpus_per_host: 12
                • num_gpus: 24
                • DIST_LAUNCH: mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 16
            • Setting:
                • HOSTFILE: /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
                • JOBENV_FILE: /home/foremans/.pbsenv

        [HOSTS]
            • [host:0] - x4017c4s5b0n0.hostmgmt2017.cm.aurora.alcf.anl.gov
            • [host:1] - x4017c4s6b0n0.hostmgmt2017.cm.aurora.alcf.anl.gov

        [DIST INFO]
            • HOSTFILE=/var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
            • NHOSTS=2
            • NGPU_PER_HOST=12
            • NGPUS=24
            • DIST_LAUNCH=mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 16

        [LAUNCH]:
            • To launch across all available GPUs, use: launch
              launch = mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 16
        ```

    </details>

    2. via `python`:

      <details closed><summary><code>.jobenv</code>:<summary>

        ```bash
        #[aurora_nre_models_frameworks-2024.1](👻 aurora_nre_models_frameworks-2024.1)
        #[🌌][01:20:20 PM][foremans@x4017c4s5b0n0][…/ezpz][🌱 jobs-rewrite]via ⨁ v1.3.450
        $ python3 -m ezpz.jobs
        2024-07-10 13:21:51,992 - numexpr.utils - INFO - Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
        2024-07-10 13:21:51,992 - numexpr.utils - INFO - Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
        2024-07-10 13:21:51,992 - numexpr.utils - INFO - NumExpr defaulting to 8 threads.
        /home/foremans/2024-07-10-131541/ezpz/venvs/aurora_nre_models_frameworks-2024.1/lib/python3.9/site-packages/pandas/core/computation/expressions.py:21: UserWarning: Pandas requires version '2.8.4' or newer of 'numexpr' (version '2.7.3' currently installed).
          from pandas.core.computation.check import NUMEXPR_INSTALLED
        [2024-07-10 13:21:54.534132][INFO][__init__:156] - Setting logging level to 'INFO' on 'RANK == 0'
        [2024-07-10 13:21:54.537096][INFO][__init__:157] - Setting logging level to 'CRITICAL' on all others 'RANK != 0'
        [2024-07-10 13:21:54.537529][INFO][__init__:160] - To disable this behavior, and log from ALL ranks (not recommended), set: 'export LOG_FROM_ALL_RANKS=1'  in your environment, and re-run.
        [2024-07-10 13:21:54.564493][INFO][dist:95] -

        [dist_info]:
          • FRAMEWORK=pytorch
          • DEVICE=xpu
          • DEVICE_ID=xpu:0
          • DISTRIBUTED_BACKEND=ccl
          • GPUS_PER_NODE=12
          • HOSTS=['x4017c4s5b0n0.hostmgmt2017.cm.aurora.alcf.anl.gov', 'x4017c4s6b0n0.hostmgmt2017.cm.aurora.alcf.anl.gov']
          • HOSTFILE=/var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
          • HOSTNAME=x4017c4s5b0n0.hostmgmt2017.cm.aurora.alcf.anl.gov
          • LOCAL_RANK=0
          • MACHINE=Aurora
          • NUM_NODES=2
          • NGPUS=24
          • NGPUS_AVAILABLE=24
          • NODE_ID=0
          • RANK=0
          • SCHEDULER=PBS
          • WORLD_SIZE_TOTAL=24
          • WORLD_SIZE_IN_USE=1
          • LAUNCH_CMD=mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 16


        [2024-07-10 13:21:54.591833][INFO][jobs:164] - Saving job env to /home/foremans/PBS-jobs/698077/.jobenv
        [2024-07-10 13:21:54.596525][INFO][jobs:164] - Saving job env to /home/foremans/2024-07-10-131541/ezpz/.jobenv
        [2024-07-10 13:21:54.613725][INFO][jobs:354] - Caught pbs_jobid='698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov', pbs_nodefile=PosixPath('/var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov') from env. Saving jobenv!
        [2024-07-10 13:21:54.655237][WARNING][jobs:144] - /home/foremans/PBS-jobs/698077  already in /home/foremans/PBS-jobs.log,  not appending !!
        [2024-07-10 13:21:54.655766][INFO][jobs:369] - Writing PBS env vars to  /home/foremans/PBS-jobs/698077 / jobenv{.sh, .yaml, .json}
        [2024-07-10 13:21:54.666092][INFO][jobs:241] - Saving job env to /home/foremans/PBS-jobs/698077/jobenv.sh
        [2024-07-10 13:21:54.682342][INFO][jobs:258] - Saving job env to /home/foremans/PBS-jobs/698077/jobenv.json
        [2024-07-10 13:21:54.700122][INFO][jobs:271] - Saving job env to /home/foremans/PBS-jobs/698077/jobenv.yaml
        [2024-07-10 13:21:54.707680][CRITICAL][jobs:381] -
        Run:

            source .jobenv

        to set these environment variables.

        6.59s user 8.17s system 16% cpu 1:27.78s total
        ```

    </details>

    <details closed><summary><code>.jobenv</code>:<summary>

    ```bash
    #[aurora_nre_models_frameworks-2024.1](👻 aurora_nre_models_frameworks-2024.1)
    #[🌌][01:21:58 PM][foremans@x4017c4s5b0n0][…/ezpz][🌱 jobs-rewrite]via ⨁ v1.3.450 [⏱ 1m27s]
    $ cat .jobenv
    #!/bin/bash --login
    FRAMEWORK="pytorch"
    DEVICE="xpu"
    DEVICE_ID="xpu:0"
    DISTRIBUTED_BACKEND="ccl"
    GPUS_PER_NODE="12"
    HOSTS="[x4017c4s5b0n0, x4017c4s6b0n0]"
    HOSTFILE="/var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov"
    HOSTNAME="x4017c4s5b0n0.hostmgmt2017.cm.aurora.alcf.anl.gov"
    LOCAL_RANK="0"
    MACHINE="Aurora"
    NUM_NODES="2"
    NGPUS="24"
    NGPUS_AVAILABLE="24"
    NODE_ID="0"
    RANK="0"
    SCHEDULER="PBS"
    WORLD_SIZE_TOTAL="24"
    WORLD_SIZE_IN_USE="1"
    LAUNCH_CMD="mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 16"
    PBS_O_HOME="/home/foremans"
    PBS_O_LANG="en_US.UTF-8"
    PBS_O_LOGNAME="foremans"
    PBS_O_PATH="/home/foremans/micromamba/condabin:/home/foremans/homebrew/bin:/home/foremans/homebrew/sbin:/home/foremans/bin/aurora:/opt/cray/pals/1.3.3/bin:/opt/cray/libfabric/1.15.2.0/bin:/opt/aurora/24.086.0/support/tools/gpu_validation:/opt/aurora/24.086.0/intel-gpu-umd/821.36/bin:/opt/aurora/24.086.0/CNDA/mpich/20231026/mpich-ofi-all-icc-default-pmix-gpu-drop20231026/bin:/opt/aurora/24.086.0/support/tools/mpi_wrapper_utils:/opt/aurora/24.086.0/CNDA/oneapi/dpcpp-ct/eng-20240227/bin:/opt/aurora/24.086.0/oneapi/advisor/latest/bin64:/opt/aurora/24.086.0/oneapi/vtune/latest/bin64:/opt/aurora/24.086.0/oneapi/debugger/latest/opt/debugger/bin:/opt/aurora/24.086.0/CNDA/oneapi/mkl/develop_20240229/bin:/opt/aurora/24.086.0/CNDA/oneapi/compiler/eng-20240227/bin:/opt/aurora/24.086.0/spack/gcc/0.7.0/install/linux-sles15-x86_64/gcc-12.2.0/gcc-12.2.0-jf4ov3v3scg7dvd76qhsuugl3jp42gfn/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/home/foremans/.local/bin:/home/foremans/bin:/usr/local/bin:/usr/bin:/bin:/opt/c3/bin:/usr/lib/mit/bin:/usr/lib/mit/sbin:/opt/pbs/bin:/sbin:/home/foremans/.local/share/kitty-ssh-kitten/kitty/bin:/home/foremans/.cargo/bin:/home/foremans/.fzf/bin:/home/foremans/.luarocks/bin"
    PBS_O_MAIL="/var/spool/mail/foremans"
    PBS_O_SHELL="/bin/zsh"
    PBS_O_TZ="America/Chicago"
    PBS_O_HOST="aurora-uan-0009.hostmgmt1000.cm.aurora.alcf.anl.gov"
    PBS_O_WORKDIR="/home/foremans/2024-07-10-131541/ezpz"
    PBS_O_SYSTEM="Linux"
    PBS_O_QUEUE="lustre_scaling"
    PBS_JOBID_SHORT="698077.aurora"
    PBS_HOOK_RESOURCES="eJydUNFuwyAM/KFNIixtoyHe9gl9tyhxEhYCzECr/P2cpZO67W0SD9h3vjs7I12RYEQ7RxiyblTeO1Nc8EfjarzrYXAe85oLLlnfKU/fw8p4H29grI01FLAT92EwzldC1tnRgKMp7oqwlZa/MWzYzVAPXOIYadVvLlvCDTO03sGyJvwFXCoFoE1DC2UrEbLtg+7zSyeOx/bQHmQn1JTsAm4xI2obl1QLZ6gUyUDBXEAK2YqTkGcpxaFpoD/JoZOtCjbVrNvmqIIDwhwrWdT7pAqxR1u0VJFPtMXhIMkbJmTOUJBUovjOFEjkIrmyMpfi5n0xduZr+q8L+10lzy7BBYOdFkNzZrESi/GPOwl146q4BbXoXoXgpz4qVoR/7rcP/3H+BG1nxmU="
    PBS_JOBNAME="STDIN"
    PBS_JOBID="698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov"
    PBS_QUEUE="lustre_scaling"
    PBS_JOBCOOKIE="5D073B7E1C16CA8D16018CC9224570E3"
    PBS_NODENUM="0"
    PBS_TASKNUM="1"
    PBS_MOMPORT="15003"
    PBS_NODEFILE="/var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov"
    PBS_ACCOUNT="Aurora_Deployment"
    PBS_JOBDIR="/home/foremans"
    PBS_ENVIRONMENT="PBS_INTERACTIVE"
    NHOSTS="2"
    NGPU_PER_HOST="12"
    BACKEND="ccl"
    alias launch="mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 16"
    echo "$(which launch)"
    ```

    </details>


## Custom hostfile

### Bash

```bash
#[aurora_nre_models_frameworks-2024.1](👻 aurora_nre_models_frameworks-2024.1)
#[🌌][01:25:25 PM][foremans@x4017c4s5b0n0][…/ezpz][🌱 jobs-rewrite]via ⨁ v1.3.450
$ head -1 "$PBS_NODEFILE" > nodefile && cat nodefile
x4017c4s5b0n0.hostmgmt2017.cm.aurora.alcf.anl.gov

#[aurora_nre_models_frameworks-2024.1](👻 aurora_nre_models_frameworks-2024.1)
#[🌌][01:25:28 PM][foremans@x4017c4s5b0n0][…/ezpz][🌱 jobs-rewrite]via ⨁ v1.3.450
$ setup_alcf nodefile

[ezpz/bin/utils.sh]

[2024-07-10-132537]
    • USER=foremans
    • MACHINE=aurora
    • HOST=x4017c4s5b0n0

[setupHost]
    • Caught 1 arguments
    • Caught 1 arguments
    • hostfile=nodefile
        • Writing PBS vars to: /home/foremans/.pbsenv

[save_pbs_env]
    • Caught 1 arguments

    • Caught hostfile != PBS_NODEFILE
        • hostfile: nodefile
        • PBS_NODEFILE: /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov

    • Using:
        • hostfile: nodefile
        • jobenv_file: /home/foremans/.pbsenv
      to calculate:
        • num_hosts: 1
        • num_gpus_per_host: 12
        • num_gpus: 12
        • DIST_LAUNCH: mpiexec --verbose --envall -n 12 -ppn 12 --hostfile nodefile --cpu-bind depth -d 16
    • Setting:
        • HOSTFILE: nodefile
        • JOBENV_FILE: /home/foremans/.pbsenv

[HOSTS]
    • [host:0] - x4017c4s5b0n0.hostmgmt2017.cm.aurora.alcf.anl.gov

[DIST INFO]
    • HOSTFILE=nodefile
    • NHOSTS=1
    • NGPU_PER_HOST=12
    • NGPUS=12
    • DIST_LAUNCH=mpiexec --verbose --envall -n 12 -ppn 12 --hostfile nodefile --cpu-bind depth -d 16

[LAUNCH]:
    • To launch across all available GPUs, use: launch
      launch = mpiexec --verbose --envall -n 12 -ppn 12 --hostfile nodefile --cpu-bind depth -d 16
```

### Python

```bash
#[aurora_nre_models_frameworks-2024.1](👻 aurora_nre_models_frameworks-2024.1)
#[🌌][01:26:10 PM][foremans@x4017c4s5b0n0][…/ezpz][🌱 jobs-rewrite]via ⨁ v1.3.450 [⏱ 6s]
$ python3 -m ezpz.jobs --hostfile nodefile
2024-07-10 13:26:41,045 - numexpr.utils - INFO - Note: detected 208 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2024-07-10 13:26:41,045 - numexpr.utils - INFO - Note: NumExpr detected 208 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
2024-07-10 13:26:41,045 - numexpr.utils - INFO - NumExpr defaulting to 8 threads.
/home/foremans/2024-07-10-131541/ezpz/venvs/aurora_nre_models_frameworks-2024.1/lib/python3.9/site-packages/pandas/core/computation/expressions.py:21: UserWarning: Pandas requires version '2.8.4' or newer of 'numexpr' (version '2.7.3' currently installed).
  from pandas.core.computation.check import NUMEXPR_INSTALLED
[2024-07-10 13:26:41.973940][INFO][__init__:156] - Setting logging level to 'INFO' on 'RANK == 0'
[2024-07-10 13:26:41.976941][INFO][__init__:157] - Setting logging level to 'CRITICAL' on all others 'RANK != 0'
[2024-07-10 13:26:41.977373][INFO][__init__:160] - To disable this behavior, and log from ALL ranks (not recommended), set: 'export LOG_FROM_ALL_RANKS=1'  in your environment, and re-run.
[2024-07-10 13:26:41.990751][WARNING][dist:1127] - Mismatch in `ngpus_in_use` and `ngpus_available` ngpus_in_use=12 vs. ngpus_available=24
[2024-07-10 13:26:41.991378][INFO][dist:95] -

[dist_info]:
  • FRAMEWORK=pytorch
  • DEVICE=xpu
  • DEVICE_ID=xpu:0
  • DISTRIBUTED_BACKEND=ccl
  • GPUS_PER_NODE=12
  • HOSTS=['x4017c4s5b0n0.hostmgmt2017.cm.aurora.alcf.anl.gov']
  • HOSTFILE=/home/foremans/2024-07-10-131541/ezpz/nodefile
  • HOSTNAME=x4017c4s5b0n0.hostmgmt2017.cm.aurora.alcf.anl.gov
  • LOCAL_RANK=0
  • MACHINE=Aurora
  • NUM_NODES=1
  • NGPUS=12
  • NGPUS_AVAILABLE=24
  • NODE_ID=0
  • RANK=0
  • SCHEDULER=PBS
  • WORLD_SIZE_TOTAL=24
  • WORLD_SIZE_IN_USE=1
  • LAUNCH_CMD=mpiexec --verbose --envall -n 12 -ppn 12 --hostfile nodefile --cpu-bind depth -d 16


[2024-07-10 13:26:41.999545][WARNING][dist:1127] - Mismatch in `ngpus_in_use` and `ngpus_available` ngpus_in_use=12 vs. ngpus_available=24
[2024-07-10 13:26:42.002941][WARNING][dist:1127] - Mismatch in `ngpus_in_use` and `ngpus_available` ngpus_in_use=12 vs. ngpus_available=24
[2024-07-10 13:26:42.017104][WARNING][dist:1127] - Mismatch in `ngpus_in_use` and `ngpus_available` ngpus_in_use=12 vs. ngpus_available=24
[2024-07-10 13:26:42.017647][INFO][jobs:164] - Saving job env to /home/foremans/PBS-jobs/698077/.jobenv
[2024-07-10 13:26:42.022741][INFO][jobs:164] - Saving job env to /home/foremans/2024-07-10-131541/ezpz/.jobenv
[2024-07-10 13:26:42.027785][CRITICAL][jobs:381] -
Run:

    source .jobenv

to set these environment variables.

6.55s user 7.58s system 218% cpu 6.474s total

```

```bash
#[aurora_nre_models_frameworks-2024.1](👻 aurora_nre_models_frameworks-2024.1)
#[🌌][01:26:54 PM][foremans@x4017c4s5b0n0][…/ezpz][🌱 jobs-rewrite]via ⨁ v1.3.450 [⏱ 6s]
$ cat .jobenv
#!/bin/bash --login
FRAMEWORK="pytorch"
DEVICE="xpu"
DEVICE_ID="xpu:0"
DISTRIBUTED_BACKEND="ccl"
GPUS_PER_NODE="12"
HOSTS="[x4017c4s5b0n0]"
HOSTFILE="nodefile"
HOSTNAME="x4017c4s5b0n0.hostmgmt2017.cm.aurora.alcf.anl.gov"
LOCAL_RANK="0"
MACHINE="Aurora"
NUM_NODES="1"
NGPUS="12"
NGPUS_AVAILABLE="24"
NODE_ID="0"
RANK="0"
SCHEDULER="PBS"
WORLD_SIZE_TOTAL="24"
WORLD_SIZE_IN_USE="1"
LAUNCH_CMD="mpiexec --verbose --envall -n 12 -ppn 12 --hostfile nodefile --cpu-bind depth -d 16"
PBS_O_HOME="/home/foremans"
PBS_O_LANG="en_US.UTF-8"
PBS_O_LOGNAME="foremans"
PBS_O_PATH="/home/foremans/micromamba/condabin:/home/foremans/homebrew/bin:/home/foremans/homebrew/sbin:/home/foremans/bin/aurora:/opt/cray/pals/1.3.3/bin:/opt/cray/libfabric/1.15.2.0/bin:/opt/aurora/24.086.0/support/tools/gpu_validation:/opt/aurora/24.086.0/intel-gpu-umd/821.36/bin:/opt/aurora/24.086.0/CNDA/mpich/20231026/mpich-ofi-all-icc-default-pmix-gpu-drop20231026/bin:/opt/aurora/24.086.0/support/tools/mpi_wrapper_utils:/opt/aurora/24.086.0/CNDA/oneapi/dpcpp-ct/eng-20240227/bin:/opt/aurora/24.086.0/oneapi/advisor/latest/bin64:/opt/aurora/24.086.0/oneapi/vtune/latest/bin64:/opt/aurora/24.086.0/oneapi/debugger/latest/opt/debugger/bin:/opt/aurora/24.086.0/CNDA/oneapi/mkl/develop_20240229/bin:/opt/aurora/24.086.0/CNDA/oneapi/compiler/eng-20240227/bin:/opt/aurora/24.086.0/spack/gcc/0.7.0/install/linux-sles15-x86_64/gcc-12.2.0/gcc-12.2.0-jf4ov3v3scg7dvd76qhsuugl3jp42gfn/bin:/opt/clmgr/sbin:/opt/clmgr/bin:/opt/sgi/sbin:/opt/sgi/bin:/home/foremans/.local/bin:/home/foremans/bin:/usr/local/bin:/usr/bin:/bin:/opt/c3/bin:/usr/lib/mit/bin:/usr/lib/mit/sbin:/opt/pbs/bin:/sbin:/home/foremans/.local/share/kitty-ssh-kitten/kitty/bin:/home/foremans/.cargo/bin:/home/foremans/.fzf/bin:/home/foremans/.luarocks/bin"
PBS_O_MAIL="/var/spool/mail/foremans"
PBS_O_SHELL="/bin/zsh"
PBS_O_TZ="America/Chicago"
PBS_O_HOST="aurora-uan-0009.hostmgmt1000.cm.aurora.alcf.anl.gov"
PBS_O_WORKDIR="/home/foremans/2024-07-10-131541/ezpz"
PBS_O_SYSTEM="Linux"
PBS_O_QUEUE="lustre_scaling"
PBS_JOBID_SHORT="698077.aurora"
PBS_HOOK_RESOURCES="eJydUNFuwyAM/KFNIixtoyHe9gl9tyhxEhYCzECr/P2cpZO67W0SD9h3vjs7I12RYEQ7RxiyblTeO1Nc8EfjarzrYXAe85oLLlnfKU/fw8p4H29grI01FLAT92EwzldC1tnRgKMp7oqwlZa/MWzYzVAPXOIYadVvLlvCDTO03sGyJvwFXCoFoE1DC2UrEbLtg+7zSyeOx/bQHmQn1JTsAm4xI2obl1QLZ6gUyUDBXEAK2YqTkGcpxaFpoD/JoZOtCjbVrNvmqIIDwhwrWdT7pAqxR1u0VJFPtMXhIMkbJmTOUJBUovjOFEjkIrmyMpfi5n0xduZr+q8L+10lzy7BBYOdFkNzZrESi/GPOwl146q4BbXoXoXgpz4qVoR/7rcP/3H+BG1nxmU="
PBS_JOBNAME="STDIN"
PBS_JOBID="698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov"
PBS_QUEUE="lustre_scaling"
PBS_JOBCOOKIE="5D073B7E1C16CA8D16018CC9224570E3"
PBS_NODENUM="0"
PBS_TASKNUM="1"
PBS_MOMPORT="15003"
PBS_NODEFILE="/var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov"
PBS_ACCOUNT="Aurora_Deployment"
PBS_JOBDIR="/home/foremans"
PBS_ENVIRONMENT="PBS_INTERACTIVE"
NHOSTS="1"
NGPU_PER_HOST="12"
BACKEND="ccl"
alias launch="mpiexec --verbose --envall -n 12 -ppn 12 --hostfile nodefile --cpu-bind depth -d 16"
echo "$(which launch)"
```


To reset after using custom hostfile:

```bash
#[aurora_nre_models_frameworks-2024.1](👻 aurora_nre_models_frameworks-2024.1)
#[🌌][01:27:39 PM][foremans@x4017c4s5b0n0][…/ezpz][🌱 jobs-rewrite]via ⨁ v1.3.450
$ unset hostfile HOSTFILE

#[aurora_nre_models_frameworks-2024.1](👻 aurora_nre_models_frameworks-2024.1)
#[🌌][01:27:41 PM][foremans@x4017c4s5b0n0][…/ezpz][🌱 jobs-rewrite]via ⨁ v1.3.450
$ setup_alcf

[ezpz/bin/utils.sh]

[2024-07-10-132744]
    • USER=foremans
    • MACHINE=aurora
    • HOST=x4017c4s5b0n0

[setupHost]
    • Using hostfile: /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    • Found in environment:
        • Writing PBS vars to: /home/foremans/.pbsenv

[save_pbs_env]
    • Using:
        • hostfile: /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
        • jobenv_file: /home/foremans/.pbsenv
      to calculate:
        • num_hosts: 2
        • num_gpus_per_host: 12
        • num_gpus: 24
        • DIST_LAUNCH: mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 16
    • Setting:
        • HOSTFILE: /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
        • JOBENV_FILE: /home/foremans/.pbsenv

[HOSTS]
    • [host:0] - x4017c4s5b0n0.hostmgmt2017.cm.aurora.alcf.anl.gov
    • [host:1] - x4017c4s6b0n0.hostmgmt2017.cm.aurora.alcf.anl.gov

[DIST INFO]
    • HOSTFILE=/var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    • NHOSTS=2
    • NGPU_PER_HOST=12
    • NGPUS=24
    • DIST_LAUNCH=mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 16

[LAUNCH]:
    • To launch across all available GPUs, use: launch
      launch = mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/698077.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 16
```

</details>

-->
