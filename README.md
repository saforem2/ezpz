# ezpz

Simplifies the process of setting up distributed training.

Example:


## Setup

- Install:

  ```bash
  python3 -m pip install "git+https://github.com/saforem2/ezpz"
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
  ```

- Example `python` script:

  ```python
  """
  ezpz/test.py
  """
  from ezpz import setup_torch, setup_tensorflow
  
  
  def test(
          framework: str = 'pytorch',
          backend: str = 'deepspeed',
          port: int | str = '5432'
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

- ✅ PyTorch + DDP:
  ```bash
  mpiexec --verbose --envall -n "${NGPUS}" --ppn "${NGPU_PER_HOST}" python3 test.py pytorch DDP
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


- ✅ PyTorch + DeepSpeed:
  ```bash
  mpiexec --verbose --envall -n "${NGPUS}" --ppn "${NGPU_PER_HOST}" python3 test.py pytorch deepspeed
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


- ✅ PyTorch + Horovod:

  ```bash
  mpiexec --verbose --envall -n "${NGPUS}" --ppn "${NGPU_PER_HOST}" python3 test.py pytorch horovod
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

- ✅ TensorFlow + Horovod:

  ```bash
  mpiexec --verbose --envall -n "${NGPUS}" --ppn "${NGPU_PER_HOST}" python3 test.py tensorflow
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

2ez
