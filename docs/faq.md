# 🙋 Frequently Asked Questions

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

   <details closed><summary><code>code</code></summary>:

   ```bash
   #[🐍 2025-08-pt29]
   #[~/d/f/p/s/ezpz][🌱 saforem2/yeet-env][📦📝🤷✓]
   #[08/26/25 @ 07:44:31][x4204c4s2b0n0]
   ; ezpz-test
   Traceback (most recent call last):
     File "/tmp/2025-08-pt29/bin/ezpz-test", line 4, in <module>
       from ezpz.test import main
     File "/lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/src/ezpz/__init__.py", line 19, in <module>
       import torch
     File "/lus/flare/projects/datascience/foremans/micromamba/envs/2025-08-pt29/lib/python3.11/site-packages/torch/__init__.py", line 407, in <module>
       from torch._C import *  # noqa: F403
       ^^^^^^^^^^^^^^^^^^^^^^
   ImportError: /lus/flare/projects/datascience/foremans/micromamba/envs/2025-08-pt29/lib/python3.11/site-packages/torch/lib/libtorch-xpu-ops-sycl-ZetaKernel.so: undefined symbol: _ZN4sycl3_V17handler28extractArgsAndReqsFromLambdaEPcRKSt6vectorINS0_6detail19kernel_param_desc_tESaIS5_EEb
   [1]    24429 exit 1     ezpz-test
   took: 0h:00m:16s
   ```

   This can be resolved by loading the correct set of modules.
   - On Aurora, for example, we can:

     ```shell
     #[🐍 2025-08-pt29]
     #[~/d/f/p/s/ezpz][🌱 saforem2/yeet-env][📦📝🤷✓] [⏱️ 8m43s]
     #[08/26/25 @ 07:57:03][x4204c4s2b0n0]
     ; ezpz-test
     [W826 07:57:09.062669769 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
       Overriding a previously registered kernel for the same operator and the same dispatch key
       operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
         registered at /lus/flare/projects/datascience/foremans/projects/argonne-lcf/frameworks-standalone/builds/nightly/2025-07-24-095425/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
       dispatch key: XPU
       previous kernel: registered at /lus/flare/projects/datascience/foremans/projects/argonne-lcf/frameworks-standalone/builds/nightly/2025-07-24-095425/pytorch/aten/src/ATen/VmapModeRegistrations.cpp:37
            new kernel: registered at /lus/flare/projects/datascience/foremans/projects/argonne-lcf/frameworks-standalone/builds/nightly/2025-07-24-095425/intel-extension-for-pytorch/build/Release/csrc/gpu/csrc/gpu/xpu/ATen/RegisterXPU_0.cpp:172 (function operator())
     [2025-08-26 07:57:12,902747][I][__init__/ezpz:265:__init__.ezpz] Setting logging level to 'INFO' on 'RANK == 0'
     [2025-08-26 07:57:12,904622][I][__init__/ezpz:266:__init__.ezpz] Setting logging level to 'CRITICAL' on all others 'RANK != 0'


     [2025-08-26 07:57:12,910880][I][launch/ezpz:356:launch] ----[🍋 ezpz.launch][started][2025-08-26-075712]----
     [2025-08-26 07:57:16,122020][I][launch/ezpz:361:launch] Job ID: 7419283
     [2025-08-26 07:57:16,122674][I][launch/ezpz:362:launch] nodelist: ['x4204c4s2b0n0', 'x4204c4s3b0n0']
     [2025-08-26 07:57:16,122983][I][launch/ezpz:363:launch] hostfile: /var/spool/pbs/aux/7419283.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
     [2025-08-26 07:57:16,124166][I][pbs/ezpz:228:pbs] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
     [2025-08-26 07:57:16,124950][I][launch/ezpz:332:launch] Building command to execute by piecing together:
     [2025-08-26 07:57:16,125270][I][launch/ezpz:333:launch] (1.) : mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/7419283.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
     [2025-08-26 07:57:16,125795][I][launch/ezpz:334:launch] (2.) cmd_to_launch: /lus/flare/projects/datascience/foremans/micromamba/envs/2025-08-pt29/bin/python3 -m ezpz.test_dist
     [2025-08-26 07:57:16,126304][I][launch/ezpz:441:launch] Took: 3.22 seconds to build command.
     [2025-08-26 07:57:16,126571][I][launch/ezpz:444:launch] Executing:
     mpiexec
       --verbose
       --envall
       --np=24
       --ppn=12
       --hostfile=/var/spool/pbs/aux/7419283.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
       --no-vni
       --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
       /lus/flare/projects/datascience/foremans/micromamba/envs/2025-08-pt29/bin/python3
       -m
       ezpz.test_dist
     [2025-08-26 07:57:16,127609][I][launch/ezpz:176:launch] Filtering for Aurora-specific messages. To view list of filters, run with EZPZ_LOG_LEVEL=DEBUG
     [2025-08-26 07:57:16,127984][I][launch/ezpz:460:launch] Execution started @ 2025-08-26-075716...
     [2025-08-26 07:57:16,128310][I][launch/ezpz:463:launch] ----[🍋 ezpz.launch][stop][2025-08-26-075716]----
     [2025-08-26 07:57:16,128648][I][launch/ezpz:99:launch] Caught 20 filters
     [2025-08-26 07:57:16,128901][I][launch/ezpz:100:launch] Running command:
      mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/7419283.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 /lus/flare/projects/datascience/foremans/micromamba/envs/2025-08-pt29/bin/python3 -m ezpz.test_dist
     Disabling local launch: multi-node application
     Connected to tcp://x4204c4s2b0n0.hsn.cm.aurora.alcf.anl.gov:7919
     Launching application 1eb3bdcb-f2c8-4bf9-b79e-25a490bda5c7
     cpubind:list x4204c4s3b0n0 pid 203966 rank 12 0: mask 0x1c
     cpubind:list x4204c4s3b0n0 pid 203967 rank 13 1: mask 0x1c00
     cpubind:list x4204c4s3b0n0 pid 203968 rank 14 2: mask 0x1c0000
     cpubind:list x4204c4s3b0n0 pid 203969 rank 15 3: mask 0x1c000000
     cpubind:list x4204c4s3b0n0 pid 203970 rank 16 4: mask 0x1c00000000
     cpubind:list x4204c4s3b0n0 pid 203971 rank 17 5: mask 0x1c0000000000
     cpubind:list x4204c4s3b0n0 pid 203972 rank 18 6: mask 0x1c0000000000000
     cpubind:list x4204c4s3b0n0 pid 203973 rank 19 7: mask 0x1c000000000000000
     cpubind:list x4204c4s3b0n0 pid 203974 rank 20 8: mask 0x1c00000000000000000
     cpubind:list x4204c4s3b0n0 pid 203975 rank 21 9: mask 0x1c0000000000000000000
     cpubind:list x4204c4s3b0n0 pid 203976 rank 22 10: mask 0x1c000000000000000000000
     cpubind:list x4204c4s3b0n0 pid 203977 rank 23 11: mask 0x1c00000000000000000000000
     cpubind:list x4204c4s2b0n0 pid 27818 rank 0 0: mask 0x1c
     cpubind:list x4204c4s2b0n0 pid 27819 rank 1 1: mask 0x1c00
     cpubind:list x4204c4s2b0n0 pid 27820 rank 2 2: mask 0x1c0000
     cpubind:list x4204c4s2b0n0 pid 27821 rank 3 3: mask 0x1c000000
     cpubind:list x4204c4s2b0n0 pid 27822 rank 4 4: mask 0x1c00000000
     cpubind:list x4204c4s2b0n0 pid 27823 rank 5 5: mask 0x1c0000000000
     cpubind:list x4204c4s2b0n0 pid 27824 rank 6 6: mask 0x1c0000000000000
     cpubind:list x4204c4s2b0n0 pid 27825 rank 7 7: mask 0x1c000000000000000
     cpubind:list x4204c4s2b0n0 pid 27826 rank 8 8: mask 0x1c00000000000000000
     cpubind:list x4204c4s2b0n0 pid 27827 rank 9 9: mask 0x1c0000000000000000000
     cpubind:list x4204c4s2b0n0 pid 27828 rank 10 10: mask 0x1c000000000000000000000
     cpubind:list x4204c4s2b0n0 pid 27829 rank 11 11: mask 0x1c00000000000000000000000
       operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
         registered at /lus/flare/projects/datascience/foremans/projects/argonne-lcf/frameworks-standalone/builds/nightly/2025-07-24-095425/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
     [2025-08-26 07:57:27,562570][I][__init__/ezpz:265:__init__.ezpz] Setting logging level to 'INFO' on 'RANK == 0'
     [2025-08-26 07:57:27,564617][I][__init__/ezpz:266:__init__.ezpz] Setting logging level to 'CRITICAL' on all others 'RANK != 0'
     [2025-08-26 07:57:27,572099][I][test_dist/ezpz:111:test_dist.__main__] Outputs will be saved to /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-08-26-075727
     [2025-08-26 07:57:27,572677][I][dist/ezpz:1171:dist] Using fw='ddp' with torch_{device,backend}= {xpu, xccl}
     [2025-08-26 07:57:27,573394][I][dist/ezpz:1035:dist] Caught MASTER_PORT=57707 from environment!
     [2025-08-26 07:57:27,573850][I][dist/ezpz:1051:dist] Using torch.distributed.init_process_group with
     - master_addr='x4204c4s2b0n0.hsn.cm.aurora.alcf.anl.gov'
     - master_port='57707'
     - world_size=24
     - rank=0
     - local_rank=0
     - timeout=datetime.timedelta(seconds=3600)
     - backend='xccl'
     [2025-08-26 07:57:27,574703][I][dist/ezpz:768:dist] Calling torch.distributed.init_process_group_with: rank=0 world_size=24 backend=xccl
     [2025-08-26 07:57:36,394953][I][pbs/ezpz:228:pbs] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
     2025:08:26-07:57:36:(27818) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
     [2025-08-26 07:57:36,961298][I][dist/ezpz:1389:dist] Using device='xpu' with backend='xccl' + 'xccl' for distributed training.
     [2025-08-26 07:57:36,962014][I][dist/ezpz:1434:dist] ['x4204c4s2b0n0'][ 0/23]
     [2025-08-26 07:57:36,960983][I][dist/ezpz:1434:dist] ['x4204c4s2b0n0'][10/23]
     [2025-08-26 07:57:36,961070][I][dist/ezpz:1434:dist] ['x4204c4s2b0n0'][ 1/23]
     [2025-08-26 07:57:36,961062][I][dist/ezpz:1434:dist] ['x4204c4s3b0n0'][13/23]
     [2025-08-26 07:57:36,961117][I][dist/ezpz:1434:dist] ['x4204c4s3b0n0'][14/23]
     [2025-08-26 07:57:36,961042][I][dist/ezpz:1434:dist] ['x4204c4s3b0n0'][16/23]
     [2025-08-26 07:57:36,961237][I][dist/ezpz:1434:dist] ['x4204c4s2b0n0'][ 3/23]
     [2025-08-26 07:57:36,961100][I][dist/ezpz:1434:dist] ['x4204c4s3b0n0'][17/23]
     [2025-08-26 07:57:36,961070][I][dist/ezpz:1434:dist] ['x4204c4s2b0n0'][ 5/23]
     [2025-08-26 07:57:36,961030][I][dist/ezpz:1434:dist] ['x4204c4s2b0n0'][ 6/23]
     [2025-08-26 07:57:36,961081][I][dist/ezpz:1434:dist] ['x4204c4s2b0n0'][ 7/23]
     [2025-08-26 07:57:36,960983][I][dist/ezpz:1434:dist] ['x4204c4s2b0n0'][ 8/23]
     [2025-08-26 07:57:36,961062][I][dist/ezpz:1434:dist] ['x4204c4s2b0n0'][ 9/23]
     [2025-08-26 07:57:36,961014][I][dist/ezpz:1434:dist] ['x4204c4s2b0n0'][11/23]
     [2025-08-26 07:57:36,961069][I][dist/ezpz:1434:dist] ['x4204c4s2b0n0'][ 2/23]
     [2025-08-26 07:57:36,961063][I][dist/ezpz:1434:dist] ['x4204c4s2b0n0'][ 4/23]
     [2025-08-26 07:57:36,961094][I][dist/ezpz:1434:dist] ['x4204c4s3b0n0'][18/23]
     [2025-08-26 07:57:36,961052][I][dist/ezpz:1434:dist] ['x4204c4s3b0n0'][19/23]
     [2025-08-26 07:57:36,961057][I][dist/ezpz:1434:dist] ['x4204c4s3b0n0'][21/23]
     [2025-08-26 07:57:36,961057][I][dist/ezpz:1434:dist] ['x4204c4s3b0n0'][22/23]
     [2025-08-26 07:57:36,961097][I][dist/ezpz:1434:dist] ['x4204c4s3b0n0'][23/23]
     [2025-08-26 07:57:36,961057][I][dist/ezpz:1434:dist] ['x4204c4s3b0n0'][12/23]
     [2025-08-26 07:57:36,961125][I][dist/ezpz:1434:dist] ['x4204c4s3b0n0'][15/23]
     [2025-08-26 07:57:36,961153][I][dist/ezpz:1434:dist] ['x4204c4s3b0n0'][20/23]
     [2025-08-26 07:57:37,049349][I][test_dist/ezpz:639:test_dist.__main__] Took: 9.48 seconds to setup torch
     [2025-08-26 07:57:38,429479][I][test_dist/ezpz:274:test_dist.__main__] Model size: 352364544 parameters
     [2025-08-26 07:57:38,431186][I][test_dist/ezpz:278:test_dist.__main__]
     =================================================================
     Layer (type:depth-idx)                   Param #
     =================================================================
     SequentialLinearNet                      --
     ├─Sequential: 1-1                        352,364,544
     =================================================================
     Total params: 352,364,544
     Trainable params: 352,364,544
     Non-trainable params: 0
     =================================================================
     [2025-08-26 07:57:38,432150][I][test_dist/ezpz:286:test_dist.__main__] Took: 1.37825637194328 seconds to build model
     [2025-08-26 07:57:38,470760][I][test_dist/ezpz:574:test_dist.__main__] model=
     SequentialLinearNet(
       (layers): Sequential(
         (0): Linear(in_features=2048, out_features=4096, bias=True)
         (1): ReLU()
         (2): Linear(in_features=4096, out_features=8192, bias=True)
         (3): ReLU()
         (4): Linear(in_features=8192, out_features=16384, bias=True)
         (5): ReLU()
         (6): Linear(in_features=16384, out_features=8192, bias=True)
         (7): ReLU()
         (8): Linear(in_features=8192, out_features=4096, bias=True)
         (9): ReLU()
         (10): Linear(in_features=4096, out_features=2048, bias=True)
       )
     )
     [2025-08-26 07:57:52,714675][I][test_dist/ezpz:290:test_dist.__main__] Took: 14.28 seconds to build optimizer
     [2025-08-26 07:57:52,747214][I][pbs/ezpz:228:pbs] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
     [2025-08-26 07:57:52,753235][I][dist/ezpz:1664:dist] Setting up wandb from rank=0
     [2025-08-26 07:57:52,753677][I][dist/ezpz:1665:dist] Using WB_PROJECT=ezpz.test_dist
     wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
     wandb: Tracking run with wandb version 0.21.1
     wandb: Run data is saved locally in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20250826_075752-5wgkqmrd
     wandb: Run `wandb offline` to turn off syncing.
     wandb: Syncing run revived-snow-5851
     wandb:  View project at https://wandb.ai/aurora_gpt/ezpz.test_dist
     wandb:  View run at https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/5wgkqmrd
     [2025-08-26 07:57:53,974792][I][dist/ezpz:1694:dist] wandb.run=[revived-snow-5851](https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/5wgkqmrd)
     [2025-08-26 07:57:53,977590][I][pbs/ezpz:228:pbs] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
     [2025-08-26 07:57:53,981659][I][dist/ezpz:1738:dist] Running on machine='Aurora'
     [2025-08-26 07:57:53,998002][I][test_dist/ezpz:293:test_dist.__main__] Took: 1.28 seconds to build trainer
     [2025-08-26 07:57:53,998843][I][test_dist/ezpz:297:test_dist.__main__] config:
     {
       "acc_events": false,
       "backend": "DDP",
       "batch_size": 64,
       "cp": 1,
       "dtype": "bfloat16",
       "input_size": 2048,
       "layer_sizes": [
         4096,
         8192,
         16384,
         8192,
         4096
       ],
       "log_freq": 1,
       "output_size": 2048,
       "pp": 1,
       "print_freq": 100,
       "profile_memory": true,
       "pyinstrument_profiler": false,
       "pytorch_profiler": false,
       "pytorch_profiler_active": 3,
       "pytorch_profiler_repeat": 5,
       "pytorch_profiler_wait": 1,
       "pytorch_profiler_warmup": 2,
       "rank_zero_only": false,
       "record_shapes": true,
       "tp": 1,
       "train_iters": 1000,
       "warmup": 50,
       "with_flops": true,
       "with_modules": true,
       "with_stack": true
     }
     [2025-08-26 07:57:54,000396][I][test_dist/ezpz:299:test_dist.__main__] Took: 26.43 to get here.
     [2025-08-26 07:57:55,508021][I][test_dist/ezpz:247:test_dist.__main__] Warmup complete at step 50
     [2025-08-26 07:57:56,477134][I][test_dist/ezpz:218:test_dist.__main__] iter=100 loss=11008.000000 dtf=0.000675 dtb=0.001717
     [2025-08-26 07:57:58,426199][I][test_dist/ezpz:218:test_dist.__main__] iter=200 loss=10944.000000 dtf=0.000665 dtb=0.001735
     [2025-08-26 07:58:00,363985][I][test_dist/ezpz:218:test_dist.__main__] iter=300 loss=10944.000000 dtf=0.000684 dtb=0.001741
     [2025-08-26 07:58:02,310994][I][test_dist/ezpz:218:test_dist.__main__] iter=400 loss=10944.000000 dtf=0.000674 dtb=0.001729
     [2025-08-26 07:58:04,253050][I][test_dist/ezpz:218:test_dist.__main__] iter=500 loss=10944.000000 dtf=0.000669 dtb=0.001730
     [2025-08-26 07:58:06,195840][I][test_dist/ezpz:218:test_dist.__main__] iter=600 loss=10944.000000 dtf=0.000672 dtb=0.001740
     [2025-08-26 07:58:08,139168][I][test_dist/ezpz:218:test_dist.__main__] iter=700 loss=10944.000000 dtf=0.000664 dtb=0.001736
     [2025-08-26 07:58:10,084974][I][test_dist/ezpz:218:test_dist.__main__] iter=800 loss=10944.000000 dtf=0.000677 dtb=0.001750
     [2025-08-26 07:58:12,005505][I][test_dist/ezpz:218:test_dist.__main__] iter=900 loss=10944.000000 dtf=0.000667 dtb=0.001736
     [2025-08-26 07:58:20,023565][I][history/ezpz:824:history] Saving iter plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-08-26-075727/ezpz.test_dist/plots/mplot
     [2025-08-26 07:58:20,211523][I][history/ezpz:824:history] Saving loss plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-08-26-075727/ezpz.test_dist/plots/mplot
     [2025-08-26 07:58:20,372340][I][history/ezpz:824:history] Saving dtf plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-08-26-075727/ezpz.test_dist/plots/mplot
     [2025-08-26 07:58:20,532649][I][history/ezpz:824:history] Saving dtb plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-08-26-075727/ezpz.test_dist/plots/mplot
     [2025-08-26 07:58:20,685178][I][history/ezpz:720:history] Saving tplots to /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-08-26-075727/ezpz.test_dist/plots/tplot
                         loss [2025-08-26-075820]
          ┌─────────────────────────────────────────────────────┐
     11072┤    ▌    ▌ ▐  ▐ ▐  ▐                                 │
          │    ▌    ▌ ▐  ▐ ▐  ▐                                 │
     11040┤    ▌    ▌ ▐  ▐ ▐  ▐                                 │
          │    ▌    ▌ ▐  ▐ ▐  ▐                                 │
          │    ▌    ▌ ▐  ▐ ▐  ▐                                 │
     11008┤████████████████████████████████▌▐███████████▐▐██████│
          │████████████████████████████████▌▐███████████▐▐██████│
     10976┤████████████████████████████████▌▐███████████▐▐██████│
          │████████████████████████████████▌▐███████████▐▐██████│
     10944┤████████████████████████████████▙▟███████████▟▟██████│
          │▐▌        ▌ ▌         ▌ ▌   ▐▐ ▌ ▌▐▌▐▐█▌▐█▌▌▐█▌ ▐ █  │
          │▐▌        ▌ ▌         ▌ ▌   ▐▐ ▌ ▌▐▌▐▐█▌▐█▌▌▐█▌ ▐ █  │
     10912┤▐▌        ▌ ▌         ▌ ▌   ▐▐ ▌ ▌▐▌▐▐█▌▐█▌▌▐█▌ ▐ █  │
          │▐▌        ▌ ▌         ▌ ▌   ▐▐ ▌ ▌▐▌▐▐█▌▐█▌▌▐█▌ ▐ █  │
     10880┤▝▌        ▌ ▌         ▌ ▌   ▐▐ ▌ ▌▐▌▐▝▛▌▐▛▌▌▐█▌ ▐ █  │
          └──┬──┬────┬─────┬───┬─────┬───┬────┬─────┬──────┬────┘
            44 84   185   296 366   471 551  647   750    867
     loss                          iter
     text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-08-26-075727/ezpz.test_dist/plots/tplot/loss.txt
                            dtf [2025-08-26-075820]
             ┌──────────────────────────────────────────────────┐
     0.001042┤                                                 ▌│
             │                                                 ▌│
     0.000974┤                                                 ▌│
             │                                                 ▌│
             │                                                 ▌│
     0.000905┤   ▌                          ▐                  ▌│
             │  ▐▌        ▗                 ▐                  ▌│
     0.000837┤  ▐▌   ▗    ▐▖         ▖    ▗ ▐  ▗               ▌│
             │  ▐▌   ▐▖   ▐▌    ▖    ▌    ▐ ▐  ▐▖    ▖    ▌    ▙│
     0.000768┤  ▐▌   ▐▌   ▐▌    █    █▗   ▐ ▐  ▐▌   ▐▌▖   █    █│
             │ ▗▐█  ▐▐▌▄ ▗▐▌▌  ▌█▐  ▖█▐  ▗▐▗█ ▄▐▌▌ ▙▐▌▌  ▖█▐   █│
             │▖▐▐█  ▐▐▌█ ▐▐▌▌  ▌█▟ ▖▌█▐  █▐▐█ █▐▌▌ █▐▌▌  ▌█▐ ▗▙▌│
     0.000699┤▌▟▐█▄ ▐▐██▗▐▟▌▙▖▄▙██▖▌▌█▟▖▖█▐▟█▄█▟▙▙▖█▐▌▙▖▄▙█▟▗▐█▌│
             │███████████▜▛███████████████████▛▛▛▛█████████████ │
     0.000631┤   ▘                                   ▐          │
             └─┬──┬─────┬────┬───┬────┬───┬────┬─────┬─────┬────┘
              25 84    185  296 366  471 551  647   750   867
     dtf                             iter
     text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-08-26-075727/ezpz.test_dist/plots/tplot/dtf.txt
                          dtf [2025-08-26-075820]
          ┌─────────────────────────────────────────────────────┐
     567.0┤█████                                                │
          │█████                                                │
     472.5┤█████                                                │
          │█████                                                │
          │█████                                                │
     378.0┤█████                                                │
          │███████████                                          │
     283.5┤███████████                                          │
          │███████████                                          │
     189.0┤███████████                                          │
          │███████████                                          │
          │███████████                                          │
      94.5┤███████████                                          │
          │████████████████                                     │
       0.0┤██████████████████████████ ██████████           █████│
          └┬────────────┬────────────┬────────────┬────────────┬┘
        0.00061      0.00072      0.00084      0.00095   0.00106
     freq                           dtf
     text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-08-26-075727/ezpz.test_dist/plots/tplot/dtf-hist.txt
                           dtb [2025-08-26-075820]
            ┌───────────────────────────────────────────────────┐
     0.00265┤                                                  ▟│
            │                                                  █│
     0.00242┤                                                  █│
            │                                                  █│
            │   ▖                                              █│
     0.00219┤   ▌                           ▐                  █│
            │   ▌                           ▐                  █│
     0.00196┤   ▌▗                          ▐                  █│
            │ ▖ ▌▐▗ ▖ ▗▗▖▗   ▟▗▖ ▖ ▄▖ ▐▙ ▗  ▐▄▗ ▖ ▗ ▐ ▗▗    ▄▗▐█│
     0.00173┤█▙▙███▙▙▄▟█▙▟▄▄▟██▙▙▙▙██▟██▙█▟▟██▟▄▙▟██▟▄▟███▄▙██▟█│
            │████████████▌██▌█▌█████▛███████▜██████████▜████████│
            │▛████▜██████▌██▌█▌▐▐███▘██▜███ ▐████▜█▌███▐███████▌│
     0.00150┤▌▛█▜█▐█▘██▜█▌▜▛▘▛▌▐▝▀▀▛ █▛▐▛▛█ ▐█▛█▝▐█▌▐█▀▝█▛█▛▜▜▛▘│
            │▌ ▐          ▝           ▘   ▝  ▘   ▐  ▐▌    ▐  ▝  │
     0.00128┤▌ ▐                                     ▌    ▝     │
            └─┬──┬─────┬─────┬─────┬──────┬────┬─────┬─────┬────┘
             25 84    185   296   420    551  647   750   867
     dtb                            iter
     text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-08-26-075727/ezpz.test_dist/plots/tplot/dtb.txt
                         dtb [2025-08-26-075820]
        ┌───────────────────────────────────────────────────────┐
     714┤                 █████                                 │
        │                 █████                                 │
     595┤                 █████                                 │
        │                 █████                                 │
        │                 █████                                 │
     476┤                 █████                                 │
        │                 █████                                 │
     357┤                 █████                                 │
        │                 █████                                 │
     238┤                 █████                                 │
        │                 █████                                 │
        │                 █████                                 │
     119┤      █████      █████                                 │
        │      ██████████ ██████████                            │
       0┤█████ ██████████ ██████████      █████            █████│
        └┬─────────────┬────────────┬─────────────┬────────────┬┘
       0.00121      0.00159      0.00196       0.00234   0.00271
     freq                          dtb
     text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-08-26-075727/ezpz.test_dist/plots/tplot/dtb-hist.txt
     [2025-08-26 07:58:20,765711][I][test_dist/ezpz:238:test_dist.__main__] dataset=<xarray.Dataset> Size: 34kB
     Dimensions:  (draw: 949)
     Coordinates:
       * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 942 943 944 945 946 947 948
     Data variables:
         iter     (draw) int64 8kB 51 52 53 54 55 56 57 ... 994 995 996 997 998 999
         loss     (draw) float32 4kB 1.094e+04 1.101e+04 ... 1.101e+04 1.094e+04
         dtf      (draw) float64 8kB 0.0007204 0.0006706 ... 0.000786 0.0007987
         dtb      (draw) float64 8kB 0.0018 0.001277 0.001722 ... 0.00186 0.001675
     [2025-08-26 07:58:20,767159][I][test_dist/ezpz:311:test_dist.__main__] Took: 26.77 seconds to finish training
     [2025-08-26 07:58:20,767684][I][test_dist/ezpz:655:test_dist.__main__] Took: 53.20 seconds
     wandb:
     wandb: 🚀 View run revived-snow-5851 at: https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/5wgkqmrd
     wandb: Find logs at: ../../../../../../../lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20250826_075752-5wgkqmrd/logs
     Application 1eb3bdcb resources: utime=1292s stime=200s maxrss=8414632KB inblock=0 oublock=2200 minflt=11887747 majflt=20605 nvcsw=494913 nivcsw=7778
     [2025-08-26 07:58:23,184273][I][launch/ezpz:467:launch] Execution finished with 0.
     [2025-08-26 07:58:23,185063][I][launch/ezpz:468:launch] Executing finished in 67.06 seconds.
     [2025-08-26 07:58:23,185548][I][launch/ezpz:469:launch] Took 67.06 seconds to run. Exiting.
     took: 0h:01m:18s
     ```

     </details>
