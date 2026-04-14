# ⁉️ FAQ

## General

### Does ezpz work on my laptop?

Yes. On Mac it uses the MPS backend, on Linux/Windows it uses CPU or CUDA if
available. `ezpz.setup_torch()` auto-detects everything. Single-process mode
works without MPI.

### Do I need MPI installed?

For multi-GPU training, yes (MPICH or OpenMPI). For single-process
development/debugging on a laptop, no — ezpz falls back gracefully.

### How do I use FSDP instead of DDP?

One flag: `ezpz.wrap_model(model, use_fsdp=True)`. That's it.

### How do I disable W&B logging?

Set `WANDB_DISABLED=1` in your environment before running.

### What happens on a single GPU?

ezpz initializes a process group of size 1. DDP/FSDP wrapping still works (it
becomes a no-op wrapper). Your code runs identically.

### How do I debug distributed issues?

Use `ezpz doctor` to check your environment. For NCCL issues, set
`NCCL_DEBUG=INFO`. For XPU issues, check that the correct Intel modules are
loaded (see Common Issues below).

### When should I use DDP vs FSDP?

**DDP** (Distributed Data Parallel): each rank holds a full copy of the model.
Use this when the model fits comfortably in one GPU's memory.

**FSDP** (Fully Sharded Data Parallel): parameters are sharded across GPUs.
Use this when the model is too large for a single GPU, or you want to reduce
per-GPU memory usage.

FSDP is the default in ezpz — `wrap_model(model)` uses `use_fsdp=True`. On
CPU and MPS devices, `wrap_model()` automatically falls back to DDP since FSDP
isn't supported on those backends.

### When should I use FSDP + Tensor Parallelism?

For very large models where FSDP alone isn't enough. Pass
`tensor_parallel_size` to `setup_torch()` to enable 2D parallelism:

```python
rank = ezpz.setup_torch(tensor_parallel_size=8)
```

See [`ezpz.examples.fsdp_tp`](../examples/fsdp-tp.md) for a working example.

### How do I set up distributed data loading?

Use `torch.utils.data.distributed.DistributedSampler` to shard data across
ranks. Key points:

- Call `sampler.set_epoch(epoch)` before each epoch for correct shuffling
- Use `drop_last=True` for consistent batch sizes across ranks
- See `ezpz.data.distributed` for a ready-made factory
  (`get_random_dataset_fsdp_tp()`) that handles sampler setup and optional
  tensor-parallel broadcasting

### How do I enable debug logging?

Set these environment variables as needed:

- `EZPZ_LOG_LEVEL=DEBUG` — ezpz internal decisions (device selection, backend choice, hostfile resolution)
- `NCCL_DEBUG=INFO` — NCCL connection setup and transport selection (use `TRACE` for full collective-level detail)
- `LOG_FROM_ALL_RANKS=1` — output from every rank, not just rank 0

See [Troubleshooting](../troubleshooting.md) for a step-by-step debugging
workflow.

## ⚠️ Common Issues

1. `ImportError: <path-to-kernel.so>: undefined symbol: [...]`

   This can happen when (for whatever reason) you have the wrong modules
   loaded.

   For example, on Aurora, some of the newer environments have a version of
   PyTorch which was built with a newer version of the Intel OneAPI software
   stack. This relies on a newer set of modules than those provided by e.g. the
   `module load frameworks`.

   Therefore, if you try and `python3 -c 'import torch'` in this environment,
   _without_ having loaded the correct set of (newer) modules, you will
   encounter something like:

   <details closed><summary><code>code</code></summary>

   ```bash
   #[🐍 2025-08-pt29]
   #[~/d/f/p/s/ezpz][🌱 saforem2/yeet-env][📦📝🤷✓]
   #[08/26/25 @ 07:44:31][x4204c4s2b0n0]
   ; ezpz test
   Traceback (most recent call last):
     File "/tmp/2025-08-pt29/bin/ezpz test", line 4, in <module>
       from ezpz.test import main
     File "/lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/src/ezpz/__init__.py", line 19, in <module>
       import torch
     File "/lus/flare/projects/datascience/foremans/micromamba/envs/2025-08-pt29/lib/python3.11/site-packages/torch/__init__.py", line 407, in <module>
       from torch._C import *  # noqa: F403
       ^^^^^^^^^^^^^^^^^^^^^^
   ImportError: /lus/flare/projects/datascience/foremans/micromamba/envs/2025-08-pt29/lib/python3.11/site-packages/torch/lib/libtorch-xpu-ops-sycl-ZetaKernel.so: undefined symbol: _ZN4sycl3_V17handler28extractArgsAndReqsFromLambdaEPcRKSt6vectorINS0_6detail19kernel_param_desc_tESaIS5_EEb
   [1]    24429 exit 1     ezpz test
   took: 0h:00m:16s
   ```

   This can be resolved by loading the correct set of modules.
   - On Aurora, for example, we can:

     ```shell
     ; ezpz test
     # ... (module loading output trimmed) ...
     [2025-08-26 07:57:12,910880][I][launch/ezpz:356:launch] ----[🍋 ezpz.launch][started][2025-08-26-075712]----
     [2025-08-26 07:57:16,124166][I][pbs/ezpz:228:pbs] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
     # ... (cpubind and MPI startup output trimmed) ...
     [2025-08-26 07:57:36,961298][I][dist/ezpz:1389:dist] Using device='xpu' with backend='xccl' + 'xccl' for distributed training.
     [2025-08-26 07:57:37,049349][I][examples.test_dist/ezpz:639:test_dist.__main__] Took: 9.48 seconds to setup torch
     [2025-08-26 07:57:38,429479][I][examples.test_dist/ezpz:274:test_dist.__main__] Model size: 352364544 parameters
     [2025-08-26 07:57:55,508021][I][examples.test_dist/ezpz:247:test_dist.__main__] Warmup complete at step 50
     [2025-08-26 07:57:56,477134][I][examples.test_dist/ezpz:218:test_dist.__main__] iter=100 loss=11008.000000 dtf=0.000675 dtb=0.001717
     [2025-08-26 07:57:58,426199][I][examples.test_dist/ezpz:218:test_dist.__main__] iter=200 loss=10944.000000 dtf=0.000665 dtb=0.001735
     # ... (training iterations continue) ...
     [2025-08-26 07:58:12,005505][I][examples.test_dist/ezpz:218:test_dist.__main__] iter=900 loss=10944.000000 dtf=0.000667 dtb=0.001736
     [2025-08-26 07:58:20,765711][I][examples.test_dist/ezpz:238:test_dist.__main__] dataset=<xarray.Dataset> Size: 34kB
     Dimensions:  (draw: 949)
     Data variables:
         iter     (draw) int64 8kB 51 52 53 54 55 ... 995 996 997 998 999
         loss     (draw) float32 4kB 1.094e+04 1.101e+04 ... 1.094e+04
         dtf      (draw) float64 8kB 0.0007204 ... 0.0007987
         dtb      (draw) float64 8kB 0.0018 ... 0.001675
     [2025-08-26 07:58:20,767159][I][examples.test_dist/ezpz:311:test_dist.__main__] Took: 26.77 seconds to finish training
     took: 0h:01m:18s
     ```

     </details>

2. **NCCL timeout on multi-node jobs**

    Symptom: training hangs or crashes with `NCCL timeout` after scaling to
    multiple nodes.

    This usually means NCCL picked the wrong network interface. Check available
    interfaces with `ip link show` and set:

    ```bash
    NCCL_SOCKET_IFNAME=eth0 ezpz launch python3 -m your_app.train
    ```

    Also verify that `MASTER_ADDR` is reachable from all nodes:

    ```bash
    echo $MASTER_ADDR
    ping -c 1 $MASTER_ADDR
    ```

3. **Training runs on CPU despite GPUs being available**

    Symptom: `setup_torch()` reports `device='cpu'` even though GPUs are
    present.

    This usually means GPU drivers or modules aren't loaded. Run `ezpz doctor`
    to diagnose. As a quick workaround, force the device explicitly:

    ```bash
    TORCH_DEVICE=cuda ezpz launch python3 -m your_app.train
    ```

    On HPC systems, ensure the correct modules are loaded (e.g.
    `module load cuda` for NVIDIA, or the appropriate Intel modules for XPU).
