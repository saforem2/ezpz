# ezpz

[![Pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Tensorflow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org)

<!-- <img alt="pyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> -->
<!-- <a href="https://www.tensorflow.org"><img alt="tensorflow" src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?&logo=TensorFlow&logoColor=white"></a>  -->

Simplifies the process of setting up distributed training for:

- `pytorch` + `{DDP, deepspeed, horovod}`

- `tensorflow` + `horovod`


> [!NOTE]
> This library is **very much** still a WIP.  
> Any ideas / issues / suggestions for improving things would be greatly appreciated.

## Setup

<!--
Test
----
- Line1
  ```bash
  whoami
  ```
  - Line2
    ```bash
    echo "${USER}"
    ```
  - Line3
    ```python
    def f(x):
        return x ** 2
    ```
-->

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


## Tests

> [!IMPORTANT]
> We can launch with a specific `{framework, backend}` combo by:
> ```bash
> export mpi_cmd="mpiexec --verbose --envall -n ${NGPUS} --ppn ${NGPU_PER_HOST}"
> export launch="${mpi_cmd} $(which python3) -m ezpz"
> launch framework=<framework> backend=<backend>
> ```
> for `framework` $\in$ `{pytorch, tensorflow}` and `backend` $\in$ `{horovod, deepspeed, DDP}`[^1]

[^1]: `deepspeed`, `DDP` only support `pytorch`

<details closed><summary><h3>✅ PyTorch + [...]</h3></summary>
  
<details closed><summary><h4><code>DDP</code>:</h4></summary>

```bash
$ launch framework=pytorch backend=DDP
```
<!--mpiexec --verbose --envall -n "${NGPUS}" --ppn "${NGPU_PER_HOST}" python3 test.py framework=pytorch framework=DDP-->

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
$ launch framework=pytorch backend=deepspeed
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
$ launch framework=pytorch backend=horovod
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

<details closed><summary><h3>✅ TensorFlow + <code>horovod</code>:</h3></summary>

```bash
$ launch framework=tensorflow backend=horovod
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


<details closed><summary><h3><code>savejobenv</code></h3></summary>

Launch a job, clone (or navigate into) `ezpz`, and run [`src/ezpz/bin/savejobenv`](./src/ezpz/bin/savejobenv):

```bash
(thetalogin5) $ qsub-gpu -A datascience -n 4 -q full-node --attrs="filesystems=home,grand,eagle,theta-fs0:ssds=required" -t 12:00 -I
(thetagpu13) $ git clone https://github.com/saforem2/ezpz
(thetagpu13) $ cd ezpz/src/ezpz
(thetagpu13) $ ./bin/savejobenv
┌──────────────────────────────────────────────────────────────────┐
│ [DIST INFO]:
│   • Writing Job info to /home/foremans/.cobaltenv
│       • NHOSTS: 4
│       • NGPU_PER_HOST: 8
│       • NGPUS = (NHOSTS * NGPU_PER_HOST) = 32
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│ Saving COBALT env to /home/foremans/.cobaltenv from thetagpu13
│ Writing COBALT vars to /home/foremans/.cobaltenv                 │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│ Copying COBALT_NODEFILE to clipboard...
│ COBALT_NODEFILE: /var/tmp/cobalt.10154591
│ [Hosts]:
│   thetagpu13 thetagpu12 thetagpu19 thetagpu18
└──────────────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────────────────┐
│ Run 'source getjobenv' in a NEW SHELL to automatically set env vars   │
└───────────────────────────────────────────────────────────────────────┘
```
</details>

<details closed><summary><h3><code>getjobenv</code></h3></summary>

Now, in a **NEW SHELL**

```bash
(localhost) $ ssh foremans@theta
(thetalogin5) $ ssh thetagpu18
(thetagpu18) $ module load conda/2023-01-11; cond activate base
(thetagpu18) $ cd ezpz
(thetagpu18) $ mkdir -p venvs/thetaGPU/2023-01-11
(thetagpu18) $ python3 -m venv venvs/thetaGPU/2023-01-11 --system-site-packages
(thetagpu18) $ source venvs/thetaGPU/2023-01-11/bin/activate
(thetagpu18) $ python3 -m pip install -e .
(thetagpu18) $ cd ezpz/src/ezpz
(thetagpu18) $ source bin/getjobenv
RUNNING_JOB_FILE: /var/tmp/cobalt-running-job
JOBID: 10154591
Loading job env from: /home/foremans/.cobaltenv
Defining alias mpilaunch: mpilaunch: aliased to mpirun -n 32 -N 8 --hostfile /var/tmp/cobalt.10154591 -x PATH -x LD_LIBRARY_PATH
HOSTFILE: /var/tmp/cobalt.10154591
NHOSTS: 4
NGPU_PER_HOST: 8
NGPUS (NHOSTS x NGPU_PER_HOST): 32
HOSTS: thetagpu13 thetagpu12 thetagpu19 thetagpu18

(thetagpu18) $ mpilaunch python3 -m ezpz pytorch DDP
Using DDP for distributed training
RANK: 0 / 31
RANK: 25 / 31
RANK: 24 / 31
RANK: 15 / 31
RANK: 26 / 31
RANK: 31 / 31
RANK: 2 / 31
RANK: 12 / 31
RANK: 1 / 31
RANK: 28 / 31
RANK: 3 / 31
RANK: 14 / 31
RANK: 4 / 31
RANK: 10 / 31
RANK: 27 / 31
RANK: 5 / 31
RANK: 30 / 31
RANK: 29 / 31
RANK: 9 / 31
RANK: 7 / 31
RANK: 6 / 31
RANK: 13 / 31
RANK: 8 / 31
RANK: 11 / 31
RANK: 18 / 31
RANK: 16 / 31
RANK: 21 / 31
RANK: 20 / 31
RANK: 22 / 31
RANK: 19 / 31
RANK: 17 / 31
RANK: 23 / 31
```

while this example looked at ThetaGPU, the exact same process will work on any
of `{ThetaGPU, Polaris, Perlmutter}`.

2ez

