# ✨ `ezpz`

[![Pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#pytorch) [![Tensorflow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?&logo=TensorFlow&logoColor=white)](#tensorflow) [![hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc)

> [!NOTE]
> This library is **very much** still a WIP.  
> Any ideas / issues / suggestions for improving things would be greatly appreciated.

Simplifies the process of setting up distributed training for:

- [`framework=pytorch`](#pytorch) + `backend={DDP, deepspeed, horovod}`

- [`framework=tensorflow`](#tensorflow) + `backend=horovod`

ezpz setup on any of `{thetaGPU, Polaris, Perlmutter}`:

```bash
git clone 'https://github.com/saforem2/ezpz' .
source ./ezpz/src/ezpz/bin/savejobenv
python3 -m pip install -e ezpz --require-virtualenv
# e.g. to launch src/ezpz/__main__.py with pytorch + deepspeed:
launch $(which python3) -m ezpz framework=pytorch backend=deepspeed
```

_2ez_.

## Setup

<details open><summary><h3>ALCF:</h3></summary>


```bash
# Most recent `conda` versions as of 10-17-2023
if [[ $(hostname) == x3* ]]; then
    export MACHINE="polaris"
    export CONDA_DATE="2023-10-04"
elif [[ $(hostname) == theta* ]]; then
    export MACHINE="thetaGPU"
    export CONDA_DATE="2023-01-11"
fi
module load "conda/${CONDA_DATE}" ; conda activate base
# Clone saforem2/ezpz and navigate into it
git clone https://github.com/saforem2/ezpz
cd ezpz
# Make a new venv for this project,
# in the project root: ./venvs/$MACHINE/$CONDA_DATE
VENV_DIR="venvs/${MACHINE}/${CONDA_DATE}"
python3 -m venv "${VENV_DIR}" --system-site-packages
source "venvs/${MACHINE}/${CONDA_DATE}/bin/activate"
# install `ezpz` into this `venv`
python3 -m pip install -e .
# to launch simple training example
# (launches `src/ezpz/__main__.py`)
cd src/ezpz
./bin/train.sh framework=pytorch backend=DDP
```
</details>

<details open><summary><h3>Perlmutter (@ NERSC):</h3></summary>

```bash
# request slurm allocation with `salloc`
NODES=2 ; HRS=2 ; salloc --nodes $NODES --qos preempt --time $HRS:00:00 -C 'gpu&hbm80g' --gpus=$(( 4 * NODES )) -A <proj>_g
# load `pytorch/2.0.1` module
module load libfabric cudatoolkit pytorch/2.0.1
# Clone saforem2/ezpz and navigate into it
git clone https://github.com/saforem2/ezpz
cd ezpz
# update pip and install `ezpz`
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e .
cd src/ezpz
./bin/train.sh framework=pytorch backend=DDP
```

</details>

where `framework` $\in$ `{pytorch, tensorflow}`, and `backend` $\in$ `{DDP,
deepspeed, horovod}`[^tf-hvd]  

[^tf-hvd]: Note `framework=tensorflow` is **only** compatible with `backend=horovod`


<details closed><summary><b>Deprecated:</b></summary>

- Install:
  ```bash
  git clone https://github.com/saforem2/ezpz
  python3 -m pip install -e ezpz
  ```

- Determine available resources:
  ```bash
  [ "$(hostname)==theta*" ] && HOSTFILE="${COBALT_NODEFILE}"  # ThetaGPU @ ALCF
  [ "$(hostname)==x3*" ] && HOSTFILE="${PBS_NODEFILE}"        # Polaris @ ALCF
  [ "$(hostname)==nid*" ] && HOSTFILE="${SLURM_NODELIST}"     # Perlmutter @ NERSC
  NHOSTS=$(wc -l < "${HOSTFILE}")
  NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
  NGPUS="$((${NHOSTS}*${NGPU_PER_HOST}))";
  echo $NHOSTS $NGPU_PER_HOST $NGPUS
  2 4 8
  ```'

- Example `python` script:

  ```python
  """
  ezpz/test.py
  """
  from ezpz import setup_torch, setup_tensorflow


  def test(
      framework: str = 'pytorch',
      backend: str = 'deepspeed',
      port: str = '5432'
  ):
  if framework == 'pytorch':
      _ = setup_torch(
          backend=backend,
          port=port,
      )
  elif framework == 'tensorflow':
      _ = setup_tensorflow()
  else:
      raise ValueError

  if __name__ == '__main__':
      import sys
      try:
          framework = sys.argv[1]
      except IndexError:
              framework = 'pytorch'
      try:
          backend = sys.argv[2]
      except IndexError:
          backend = 'deepspeed'
      try:
          port = sys.argv[3]
      except IndexError:
          port = '5432'
      test(framework=framework, backend=backend, port=port)
  ```
  
</details>


## Examples

> [!IMPORTANT]
> We can `launch` on any of `{ThetaGPU, Polaris, Perlmutter}` (*)
> with a specific `{framework, backend}` combo by
> 1. [`savejobenv`](./src/ezpz/bin/savejobenv):
>     - This will `export launch=<launcher> <launcher-opts>`
>       for `<launcher>` $\in$ `{mpirun,mpiexec,srun}`
>       on (*) respectively.
>     - By default, `launch <exec>` will launch `<exec>` across
>       _all_ the available GPUs in your active `{COBALT,PBS,slurm}` job.
> 2. `launch`
>     - e.g. `launch $(which python3) -m ezpz framework=<framework> backend=<backend>`, will:
>         - `launch` [`__main__.py`](./src/ezpz/__main__.py) (in this case)
>           with framework `<framework>` and backend `<backend>`
>           (e.g. `pytorch` and `deepspeed`)
>
> - Complete example:      
> ```bash
> #!/bin/bash --login
> git clone https://github.com/saforem2/ezpz
> ./ezpz/src/ezpz/bin/savejobenv
> launch $(which python3) -m ezpz framework=<framework> backend=<backend>
> ```
> for `framework` $\in$ `{pytorch, tensorflow}` and `backend` $\in$ `{horovod, deepspeed, DDP}`[^1]

[^1]: `deepspeed`, `DDP` only support `pytorch`

### PyTorch

<details closed><summary><h3>✅ PyTorch + [...]</h3></summary>
  
<details closed><summary><h4><code>DDP</code>:</h4></summary>

```bash
launch framework=pytorch backend=DDP
```

<details closed><summary><b>Output:</b></summary>

```bash
Connected to tcp://x3005c0s31b1n0.hsn.cm.polaris.alcf.anl.gov:7919
Found executable /soft/datascience/conda/2023-10-04/mconda3/bin/python3
Launching application c079ffa9-4732-45ba-995b-e5685330311b
[10/05/23 16:56:26][INFO][dist.py:362] - Using DDP for distributed training
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 0 / 7
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 2 / 7
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 4 / 7
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 3 / 7
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 1 / 7
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 6 / 7
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 5 / 7
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 7 / 7
```

</details>
</details>

<details closed><summary><h4><code>deepspeed</code>:</h4></summary>

```bash
launch framework=pytorch backend=deepspeed
```

<details closed><summary><b>Output:</b></summary>

```bash
Connected to tcp://x3005c0s31b1n0.hsn.cm.polaris.alcf.anl.gov:7919
Found executable /soft/datascience/conda/2023-10-04/mconda3/bin/python3
Launching application c1c5bcd5-c300-4927-82e4-236d4643e31d
[10/05/23 16:56:34][INFO][dist.py:362] - Using deepspeed for distributed training
[2023-10-05 16:56:34,949] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-10-05 16:56:34,949] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-10-05 16:56:34,949] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-10-05 16:56:34,949] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-10-05 16:56:34,953] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-10-05 16:56:34,953] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-10-05 16:56:34,953] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-10-05 16:56:34,953] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-10-05 16:56:40,160] [INFO] [comm.py:637:init_distributed] cdb=None
[2023-10-05 16:56:40,160] [INFO] [comm.py:637:init_distributed] cdb=None
[2023-10-05 16:56:40,160] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2023-10-05 16:56:40,160] [INFO] [comm.py:637:init_distributed] cdb=None
[2023-10-05 16:56:40,160] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2023-10-05 16:56:40,160] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2023-10-05 16:56:40,160] [INFO] [comm.py:637:init_distributed] cdb=None
[2023-10-05 16:56:40,160] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2023-10-05 16:56:40,767] [INFO] [comm.py:637:init_distributed] cdb=None
[2023-10-05 16:56:40,767] [INFO] [comm.py:637:init_distributed] cdb=None
[2023-10-05 16:56:40,767] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2023-10-05 16:56:40,767] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2023-10-05 16:56:40,767] [INFO] [comm.py:637:init_distributed] cdb=None
[2023-10-05 16:56:40,767] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2023-10-05 16:56:40,767] [INFO] [comm.py:637:init_distributed] cdb=None
[2023-10-05 16:56:40,767] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2023-10-05 16:56:41,621] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=4, local_rank=0, world_size=8, master_addr=10.140.57.89, master_port=29500
[2023-10-05 16:56:41,621] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=5, local_rank=1, world_size=8, master_addr=10.140.57.89, master_port=29500
[2023-10-05 16:56:41,621] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=0, local_rank=0, world_size=8, master_addr=10.140.57.89, master_port=29500
[2023-10-05 16:56:41,621] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=6, local_rank=2, world_size=8, master_addr=10.140.57.89, master_port=29500
[2023-10-05 16:56:41,621] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=1, local_rank=1, world_size=8, master_addr=10.140.57.89, master_port=29500
[2023-10-05 16:56:41,621] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=7, local_rank=3, world_size=8, master_addr=10.140.57.89, master_port=29500
[2023-10-05 16:56:41,621] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=2, local_rank=2, world_size=8, master_addr=10.140.57.89, master_port=29500
[2023-10-05 16:56:41,621] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=3, local_rank=3, world_size=8, master_addr=10.140.57.89, master_port=29500
[2023-10-05 16:56:41,621] [INFO] [comm.py:668:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
[10/05/23 16:56:41][INFO][dist.py:413] - RANK: 0 / 7
[10/05/23 16:56:41][INFO][dist.py:413] - RANK: 2 / 7
[10/05/23 16:56:41][INFO][dist.py:413] - RANK: 1 / 7
[10/05/23 16:56:41][INFO][dist.py:413] - RANK: 7 / 7
[10/05/23 16:56:41][INFO][dist.py:413] - RANK: 4 / 7
[10/05/23 16:56:41][INFO][dist.py:413] - RANK: 5 / 7
[10/05/23 16:56:41][INFO][dist.py:413] - RANK: 6 / 7
[10/05/23 16:56:41][INFO][dist.py:413] - RANK: 3 / 7
```

</details>
</details>

<details closed><summary><h4><code>horovod</code></h4></summary>

```bash
launch framework=pytorch backend=horovod
```

<details closed><summary><b>Output:</b></summary>

```bash
Connected to tcp://x3005c0s31b1n0.hsn.cm.polaris.alcf.anl.gov:7919
Found executable /soft/datascience/conda/2023-10-04/mconda3/bin/python3
Launching application c079ffa9-4732-45ba-995b-e5685330311b
[10/05/23 16:56:26][INFO][dist.py:362] - Using DDP for distributed training
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 0 / 7
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 2 / 7
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 4 / 7
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 3 / 7
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 1 / 7
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 6 / 7
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 5 / 7
[10/05/23 16:56:27][INFO][dist.py:413] - RANK: 7 / 7
```

</details>
</details>
</details>

### TensorFlow

<details closed><summary><h3>✅ TensorFlow + <code>horovod</code>:</h3></summary>

```bash
launch framework=tensorflow backend=horovod
```

<details closed><summary><b>Output:</b></summary>

```bash
Connected to tcp://x3005c0s31b1n0.hsn.cm.polaris.alcf.anl.gov:7919
Found executable /soft/datascience/conda/2023-10-04/mconda3/bin/python3
Launching application 2b7b89f3-5f40-42de-aa12-a15876baee09
2023-10-05 16:56:49.870938: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-10-05 16:56:49.870938: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-10-05 16:56:49.870938: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-10-05 16:56:49.870940: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-10-05 16:56:50.038355: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-10-05 16:56:50.038355: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-10-05 16:56:50.038353: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-10-05 16:56:50.038359: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-10-05 16:57:00.277129: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 0, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:07:00.0,compute capability: 8.0
[10/05/23 16:57:00][INFO][dist.py:203] - RANK: 4 / 7
2023-10-05 16:57:00.303774: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 0, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:07:00.0,compute capability: 8.0
[10/05/23 16:57:00][INFO][dist.py:203] - RANK: 0 / 7
2023-10-05 16:57:00.430211: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 1, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:46:00.0,compute capability: 8.0
[10/05/23 16:57:00][INFO][dist.py:203] - RANK: 5 / 7
2023-10-05 16:57:00.445891: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 1, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:46:00.0,compute capability: 8.0
2023-10-05 16:57:00.447921: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 2, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:85:00.0,compute capability: 8.0
[10/05/23 16:57:00][INFO][dist.py:203] - RANK: 1 / 7
[10/05/23 16:57:00][INFO][dist.py:203] - RANK: 2 / 7
2023-10-05 16:57:00.452035: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 2, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:85:00.0,compute capability: 8.0
[10/05/23 16:57:00][INFO][dist.py:203] - RANK: 6 / 7
2023-10-05 16:57:00.458780: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 3, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:c7:00.0,compute capability: 8.0
[10/05/23 16:57:00][INFO][dist.py:203] - RANK: 7 / 7
2023-10-05 16:57:00.472986: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 3, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:c7:00.0,compute capability: 8.0
[10/05/23 16:57:00][INFO][dist.py:203] - RANK: 3 / 7
```

</details>
</details>

## Helper Utilities

- [`src/ezpz/bin/savejobenv`](./src/ezpz/bin/savejobenv): Shell script to save
  relevant job related environment variables to a file which can be `sourced`
  from new login instances.
- [`src/ezpz/bin/getjobenv`](./src/ezpz/bin/getjobenv): Shell script that, when
  sourced, will populate the current environment with the necessary job-related
  variables.


<!--<details open><summary><h3>savejobenv</h3></summary>-->

### `savejobenv`

Launch a job, clone (or navigate into) `ezpz`, and `source` [`src/ezpz/bin/savejobenv`](./src/ezpz/bin/savejobenv):

```bash
(thetalogin4) $ qsub-gpu -A datascience -n 2 -q full-node --attrs="filesystems=home,grand,eagle,theta-fs0:ssds=required" -t 06:00 -I
Job routed to queue "full-node".
Wait for job 10155652 to start...
Opening interactive session to thetagpu04
[...]
```

```bash
(thetagpu04) $ git clone https://github.com/saforem2/ezpz
(thetagpu04) $ source ezpz/src/ezpz/bin/savejobenv
┌───────────────────────────────────────────────────────────────────
│ Writing COBALT vars to /home/foremans/.cobaltenv
│ HOSTFILE: /var/tmp/cobalt.10155652
│ NHOSTS: 2
│ 8 GPUs per host
│ 16 GPUs total
└───────────────────────────────────────────────────────────────────
┌───────────────────────────────────────────────────────────────────
│ [DIST INFO]:
│   • Writing Job info to /home/foremans/.cobaltenv
│     • HOSTFILE: /var/tmp/cobalt.10155652
│     • NHOSTS: 2
│     • NGPU_PER_HOST: 8
│     • NGPUS = (NHOSTS * NGPU_PER_HOST) = 16
│ [Hosts]:
│       • thetagpu04 thetagpu19
│ [Launch]:
│     • Use: 'launch' (=mpirun -n  -N  --hostfile /var/tmp/cobalt.10155652 -x PATH -x LD_LIBRARY_PATH)
│       to launch job
└───────────────────────────────────────────────────────────────────
┌────────────────────────────────────────────────────────────────────────────────
│ YOU ARE HERE: /home/foremans
│ Run 'source ./bin/getjobenv' in a NEW SHELL to automatically set env vars
└────────────────────────────────────────────────────────────────────────────────
```


<!--
<details closed><summary><h3><code>getjobenv</code></h3></summary>
-->


### `getjobenv`

Now, in a **NEW SHELL**

```bash
(localhost)   $ ssh <user>@theta
```

```bash
(thetalogin4) $ ssh thetagpu19
```

```bash
(thetagpu19)  $ module load conda/2023-01-11; conda activate base
(thetagpu19)  $ cd ezpz
(thetagpu19)  $ source ./src/ezpz/bin/getjobenv
┌──────────────────────────────────────────────────────────────────
│ [Hosts]: 
│     • thetagpu04, thetagpu19
└──────────────────────────────────────────────────────────────────
┌──────────────────────────────────────────────────────────────────
│ [DIST INFO]: 
│     • Loading job env from: /home/foremans/.cobaltenv
│     • HOSTFILE: /var/tmp/cobalt.10155652
│     • NHOSTS: 2
│     • NGPU_PER_HOST: 8
│     • NGPUS (NHOSTS x NGPU_PER_HOST): 16
│     • DIST_LAUNCH: mpirun -n 16 -N 8 --hostfile /var/tmp/cobalt.10155652 -x PATH -x LD_LIBRARY_PATH
│     • Defining alias: launch: aliased to mpirun -n 16 -N 8 --hostfile /var/tmp/cobalt.10155652 -x PATH -x LD_LIBRARY_PATH
└──────────────────────────────────────────────────────────────────
(thetagpu19) $ mkdir -p venvs/thetaGPU/2023-01-11
(thetagpu19) $ python3 -m venv venvs/thetaGPU/2023-01-11 --system-site-packages
(thetagpu19) $ source venvs/thetaGPU/2023-01-11/bin/activate
(thetagpu19) $ python3 -m pip install -e . --require-virtualenv
(thetagpu19) $ launch python3 -m ezpz framework=pytorch backend=DDP
[2023-10-26 12:21:26,716][ezpz.dist][INFO] - Using DDP for distributed training
[2023-10-26 12:21:26,787][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 13
[2023-10-26 12:21:26,787][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 14
[2023-10-26 12:21:26,787][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 8
[2023-10-26 12:21:26,787][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 12
[2023-10-26 12:21:26,787][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 6
[2023-10-26 12:21:26,788][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 9
[2023-10-26 12:21:26,787][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 10
[2023-10-26 12:21:26,788][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 15
[2023-10-26 12:21:26,788][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 11
[2023-10-26 12:21:26,789][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 7
[2023-10-26 12:21:26,789][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 3
[2023-10-26 12:21:26,789][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 1
[2023-10-26 12:21:26,789][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 4
[2023-10-26 12:21:26,789][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 5
[2023-10-26 12:21:26,789][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 2
[2023-10-26 12:21:26,798][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 0
[2023-10-26 12:21:26,811][torch.distributed.distributed_c10d][INFO] - Rank 14: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:26,812][torch.distributed.distributed_c10d][INFO] - Rank 6: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:26,814][torch.distributed.distributed_c10d][INFO] - Rank 13: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:26,815][torch.distributed.distributed_c10d][INFO] - Rank 7: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:26,816][torch.distributed.distributed_c10d][INFO] - Rank 8: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:26,817][torch.distributed.distributed_c10d][INFO] - Rank 3: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:26,819][torch.distributed.distributed_c10d][INFO] - Rank 12: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:26,820][torch.distributed.distributed_c10d][INFO] - Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:26,821][torch.distributed.distributed_c10d][INFO] - Rank 10: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:26,823][torch.distributed.distributed_c10d][INFO] - Rank 4: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:26,825][torch.distributed.distributed_c10d][INFO] - Rank 9: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:26,825][torch.distributed.distributed_c10d][INFO] - Rank 5: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:26,827][torch.distributed.distributed_c10d][INFO] - Rank 15: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:26,828][torch.distributed.distributed_c10d][INFO] - Rank 2: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:26,830][torch.distributed.distributed_c10d][INFO] - Rank 11: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:26,831][torch.distributed.distributed_c10d][INFO] - Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
[2023-10-26 12:21:27,035][ezpz.dist][INFO] - RANK: 0 / 15
{
  "framework": "pytorch",
  "backend": "DDP",
  "use_wandb": false,
  "seed": null,
  "port": null,
  "ds_config_path": null,
  "wandb_project_name": null,
  "precision": null,
  "ngpus": null
}
[2023-10-26 12:21:27,038][__main__][INFO] - Output dir: /lus/grand/projects/datascience/foremans/locations/thetaGPU/projects/saforem2/ezpz/outputs/runs/pytorch/DDP/2023-10-26/12-21-25
[2023-10-26 12:21:27,097][ezpz.dist][INFO] - RANK: 8 / 15
[2023-10-26 12:21:27,103][ezpz.dist][INFO] - RANK: 6 / 15
[2023-10-26 12:21:27,104][ezpz.dist][INFO] - RANK: 14 / 15
[2023-10-26 12:21:27,111][ezpz.dist][INFO] - RANK: 13 / 15
[2023-10-26 12:21:27,116][ezpz.dist][INFO] - RANK: 1 / 15
[2023-10-26 12:21:27,126][ezpz.dist][INFO] - RANK: 7 / 15
[2023-10-26 12:21:27,135][ezpz.dist][INFO] - RANK: 10 / 15
[2023-10-26 12:21:27,139][ezpz.dist][INFO] - RANK: 12 / 15
[2023-10-26 12:21:27,141][ezpz.dist][INFO] - RANK: 9 / 15
[2023-10-26 12:21:27,141][ezpz.dist][INFO] - RANK: 15 / 15
[2023-10-26 12:21:27,141][ezpz.dist][INFO] - RANK: 11 / 15
[2023-10-26 12:21:27,141][ezpz.dist][INFO] - RANK: 5 / 15
[2023-10-26 12:21:27,144][ezpz.dist][INFO] - RANK: 2 / 15
[2023-10-26 12:21:27,145][ezpz.dist][INFO] - RANK: 4 / 15
[2023-10-26 12:21:27,145][ezpz.dist][INFO] - RANK: 3 / 15
16.56s user 30.05s system 706% cpu 6.595s total
```

while this example looked at ThetaGPU, the exact same process will work on any
of `{ThetaGPU, Polaris, Perlmutter}`.

2ez
