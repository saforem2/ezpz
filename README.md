# ezpz
2ez

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
  # test.py
  import os
  from ezpz.dist import setup_torch, setup_tensorflow


  def main():
      framework = os.environ.get('FRAMEWORK', 'pytorch')
      backend = os.environ.get('BACKEND', 'DDP')
      if framework == 'pytorch':
          setup_torch(
              backend=backend,
          )
      elif framework == 'tensorflow':
          setup_tensorflow(backend='horovod')
      else:
          raise ValueError(f'Unrecognized framework: {framework}')


  if __name__ == '__main__':
      main()
  ```

- PyTorch + DDP:
  ```bash
  FRAMEWORK=pytorch BACKEND=DDP mpiexec --verbose --envall -n "${NGPUS}" --ppn "${NGPU_PER_HOST}" python3 test.py
  ```

  <details closed><summary><b>Output:</b></summary>
    
  ```bash
  Connected to tcp://x3005c0s31b0n0.hsn.cm.polaris.alcf.anl.gov:7919
  Found executable /lus/grand/projects/datascience/foremans/locations/polaris/projects/saforem2/l2hmc-qcd/venvs/polaris/2023-09-21/bin/python3
  Launching application cab4e1d7-bc92-4704-a2b8-b600fc621aa8
  Using DDP for distributed training
  Global Rank: 0 / 7
  Global Rank: 7 / 7
  Global Rank: 6 / 7
  Global Rank: 2 / 7
  Global Rank: 4 / 7
  Global Rank: 5 / 7
  Global Rank: 3 / 7
  Global Rank: 1 / 7
  ```
  </details>


- PyTorch + DeepSpeed:
  ```bash
  FRAMEWORK=pytorch BACKEND=deepspeed mpiexec --verbose --envall -n "${NGPUS}" --ppn "${NGPU_PER_HOST}" python3 test.py
  ```

  <details closed><summary><b>Output:</b></summary>
    
  ```bash
  Connected to tcp://x3005c0s31b0n0.hsn.cm.polaris.alcf.anl.gov:7919
  Found executable /lus/grand/projects/datascience/foremans/locations/polaris/projects/saforem2/l2hmc-qcd/venvs/polaris/2023-09-21/bin/python3
  Launching application f677e525-0be5-41ca-8de1-2a39f082d29b
  Using deepspeed for distributed training
  [2023-09-22 11:10:38,144] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
  [2023-09-22 11:10:38,144] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
  [2023-09-22 11:10:38,144] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
  [2023-09-22 11:10:38,144] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
  [2023-09-22 11:10:38,144] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
  [2023-09-22 11:10:38,144] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
  [2023-09-22 11:10:38,144] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
  [2023-09-22 11:10:38,144] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
  [2023-09-22 11:10:59,339] [INFO] [comm.py:637:init_distributed] cdb=None
  [2023-09-22 11:10:59,339] [INFO] [comm.py:637:init_distributed] cdb=None
  [2023-09-22 11:10:59,339] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
  [2023-09-22 11:10:59,339] [INFO] [comm.py:637:init_distributed] cdb=None
  [2023-09-22 11:10:59,339] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
  [2023-09-22 11:10:59,340] [INFO] [comm.py:637:init_distributed] cdb=None
  [2023-09-22 11:10:59,340] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
  [2023-09-22 11:10:59,339] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
  [2023-09-22 11:11:00,036] [INFO] [comm.py:637:init_distributed] cdb=None
  [2023-09-22 11:11:00,036] [INFO] [comm.py:637:init_distributed] cdb=None
  [2023-09-22 11:11:00,036] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
  [2023-09-22 11:11:00,036] [INFO] [comm.py:637:init_distributed] cdb=None
  [2023-09-22 11:11:00,036] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
  [2023-09-22 11:11:00,036] [INFO] [comm.py:637:init_distributed] cdb=None
  [2023-09-22 11:11:00,036] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
  [2023-09-22 11:11:00,036] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
  [2023-09-22 11:11:01,577] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=0, local_rank=0, world_size=8, master_addr=10.140.57.90, master_port=29500
  [2023-09-22 11:11:01,577] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=4, local_rank=0, world_size=8, master_addr=10.140.57.90, master_port=29500
  [2023-09-22 11:11:01,577] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=1, local_rank=1, world_size=8, master_addr=10.140.57.90, master_port=29500
  [2023-09-22 11:11:01,577] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=5, local_rank=1, world_size=8, master_addr=10.140.57.90, master_port=29500
  [2023-09-22 11:11:01,577] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=2, local_rank=2, world_size=8, master_addr=10.140.57.90, master_port=29500
  [2023-09-22 11:11:01,577] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=6, local_rank=2, world_size=8, master_addr=10.140.57.90, master_port=29500
  [2023-09-22 11:11:01,577] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=3, local_rank=3, world_size=8, master_addr=10.140.57.90, master_port=29500
  [2023-09-22 11:11:01,577] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=7, local_rank=3, world_size=8, master_addr=10.140.57.90, master_port=29500
  [2023-09-22 11:11:01,577] [INFO] [comm.py:668:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
  Global Rank: 0 / 7
  Global Rank: 5 / 7
  Global Rank: 7 / 7
  Global Rank: 4 / 7
  Global Rank: 6 / 7
  Global Rank: 3 / 7
  Global Rank: 1 / 7
  Global Rank: 2 / 7
  ```

  </details>


- PyTorch + Horovod:

  ```bash
  FRAMEWORK=pytorch BACKEND=horovod mpiexec --verbose --envall -n "${NGPUS}" --ppn "${NGPU_PER_HOST}" python3 test.py
  ```

  <details closed><summary><b>Output:</b></summary>
    
  ```bash
  Connected to tcp://x3005c0s31b0n0.hsn.cm.polaris.alcf.anl.gov:7919
  Found executable /lus/grand/projects/datascience/foremans/locations/polaris/projects/saforem2/l2hmc-qcd/venvs/polaris/2023-09-21/bin/python3
  Launching application b6100af9-1c67-4aed-8571-057cecb561eb
  Using horovod for distributed training
  Global Rank: 5 / 7
  Global Rank: 7 / 7
  Global Rank: 6 / 7
  Global Rank: 2 / 7
  Global Rank: 1 / 7
  Global Rank: 3 / 7
  Global Rank: 4 / 7
  Global Rank: 0 / 7
  ```

  </details>

- TensorFlow + Horovod:

  ```bash
  $ FRAMEWORK=tensorflow BACKEND=horovod mpiexec --verbose --envall -n "${NGPUS}" --ppn "${NGPU_PER_HOST}" python3 test.py
  ```

  <details closed><summary><b>Output:</b></summary>
    
  ```bash
  Connected to tcp://x3005c0s31b0n0.hsn.cm.polaris.alcf.anl.gov:7919
  Found executable /lus/grand/projects/datascience/foremans/locations/polaris/projects/saforem2/l2hmc-qcd/venvs/polaris/2023-09-21/bin/python3
  Launching application 46136d83-185d-4c2e-9e36-21cf0b570eae
  2023-09-22 11:17:26.259738: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
  To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
  2023-09-22 11:17:26.259735: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
  To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
  2023-09-22 11:17:26.259735: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
  To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
  2023-09-22 11:17:26.261666: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
  To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
  2023-09-22 11:17:26.276882: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
  To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
  2023-09-22 11:17:26.276884: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
  To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
  2023-09-22 11:17:26.276916: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
  To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
  2023-09-22 11:17:26.276914: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
  To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
  2023-09-22 11:17:30.984040: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 0, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:07:00.0, 
  compute capability: 8.0
  2023-09-22 11:17:31.013615: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 0, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:07:00.0, 
  compute capability: 8.0
  4, Physical GPUs and 1 Logical GPUs
  2023-09-22 11:17:31.177448: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 3, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:c7:00.0, 
  compute capability: 8.0
  2023-09-22 11:17:31.178344: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 2, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:85:00.0, 
  compute capability: 8.0
  2023-09-22 11:17:31.178532: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 3, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:c7:00.0, 
  compute capability: 8.0
  2023-09-22 11:17:31.179635: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 2, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:85:00.0, 
  compute capability: 8.0
  2023-09-22 11:17:31.181771: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 1, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:46:00.0, 
  compute capability: 8.0
  2023-09-22 11:17:31.187252: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38341 MB memory:  -> device: 1, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:46:00.0, 
  compute capability: 8.0
  Using: float32 precision
  Using: float32 precision
  Using: float32 precision
  Using: float32 precision
  Using: float32 precision
  Using: float32 precision
  Using: float32 precision
  Using: float32 precision
  RANK: 0, LOCAL_RANK: 0
  RANK: 4, LOCAL_RANK: 0
  RANK: 7, LOCAL_RANK: 3
  RANK: 3, LOCAL_RANK: 3
  RANK: 6, LOCAL_RANK: 2
  RANK: 2, LOCAL_RANK: 2
  RANK: 1, LOCAL_RANK: 1
  RANK: 5, LOCAL_RANK: 1
  ```

  </details>
