# 🙋 Frequently Asked Questions

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
