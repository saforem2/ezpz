# Train ViT with FSDP on MNIST

See:

- 📘 [examples/ViT](../python/Code-Reference/examples/vit.md)
- 🐍 [src/ezpz/examples/vit.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/vit.py)

```bash
ezpz launch python3 -m ezpz.examples.vit --compile # --fsdp
```

<details closed><summary><code>--help</code></summary>

```bash
$ python3 -m ezpz.examples.vit --help
usage: ezpz.examples.vit [-h] [--img_size IMG_SIZE] [--batch_size BATCH_SIZE]
                        [--num_heads NUM_HEADS] [--head_dim HEAD_DIM]
                        [--hidden-dim HIDDEN_DIM] [--mlp-dim MLP_DIM]
                        [--dropout DROPOUT]
                        [--attention-dropout ATTENTION_DROPOUT]
                        [--num_classes NUM_CLASSES] [--dataset {fake,mnist}]
                        [--depth DEPTH] [--patch_size PATCH_SIZE]
                        [--dtype DTYPE] [--compile]
                        [--num_workers NUM_WORKERS] [--max_iters MAX_ITERS]
                        [--warmup WARMUP] [--attn_type {native,sdpa}]
                        [--cuda_sdpa_backend {flash_sdp,mem_efficient_sdp,math_sdp,cudnn_sdp,all}]
                        [--fsdp]

Train a simple ViT

options:
    -h, --help            show this help message and exit
    --img_size IMG_SIZE, --img-size IMG_SIZE
                        Image size
    --batch_size BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
    --num_heads NUM_HEADS, --num-heads NUM_HEADS
                        Number of heads
    --head_dim HEAD_DIM, --head-dim HEAD_DIM
                        Hidden Dimension
    --hidden-dim HIDDEN_DIM, --hidden_dim HIDDEN_DIM
                        Hidden Dimension
    --mlp-dim MLP_DIM, --mlp_dim MLP_DIM
                        MLP Dimension
    --dropout DROPOUT     Dropout rate
    --attention-dropout ATTENTION_DROPOUT, --attention_dropout ATTENTION_DROPOUT
                        Attention Dropout rate
    --num_classes NUM_CLASSES, --num-classes NUM_CLASSES
                        Number of classes
    --dataset {fake,mnist}
                        Dataset to use
    --depth DEPTH         Depth
    --patch_size PATCH_SIZE, --patch-size PATCH_SIZE
                        Patch size
    --dtype DTYPE         Data type
    --compile             Compile model
    --num_workers NUM_WORKERS, --num-workers NUM_WORKERS
                        Number of workers
    --max_iters MAX_ITERS, --max-iters MAX_ITERS
                        Maximum iterations
    --warmup WARMUP       Warmup iterations (or fraction) before starting to
                        collect metrics.
    --attn_type {native,sdpa}, --attn-type {native,sdpa}
                        Attention function to use.
    --cuda_sdpa_backend {flash_sdp,mem_efficient_sdp,math_sdp,cudnn_sdp,all}, --cuda-sdpa-backend {flash_sdp,mem_efficient_sdp,math_sdp,cudnn_sdp,all}
                        CUDA SDPA backend to use.
    --fsdp                Use FSDP

```

</details>


<details closed><summary>Output on Sunspot:</summary>

```bash

#[aurora_frameworks-2025.2.0](ezpz-aurora_frameworks-2025.2.0)
#[/t/d/f/p/s/ezpz][dev][?] [󰔛  48s]
#[12/31/25 @ 11:41:01][x1921c0s7b0n0]
; LINES=40 COLUMNS=120 ezpz launch python3 -m ezpz.examples.vit --compile --fsdp | tee ezpz-examples-vit-$(tstamp).log


[2025-12-31 11:41:24,439765][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2025-12-31-114124]----
[2025-12-31 11:41:25,295485][I][ezpz/launch:416:launch] Job ID: 12458339
[2025-12-31 11:41:25,296275][I][ezpz/launch:417:launch] nodelist: ['x1921c0s3b0n0', 'x1921c0s7b0n0']
[2025-12-31 11:41:25,296669][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
[2025-12-31 11:41:25,297331][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
[2025-12-31 11:41:25,298068][I][ezpz/launch:367:build_executable] Building command to execute by piecing together:
[2025-12-31 11:41:25,298451][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
[2025-12-31 11:41:25,299251][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: python3 -m ezpz.examples.vit --compile --fsdp
[2025-12-31 11:41:25,299999][I][ezpz/launch:433:launch] Took: 1.54 seconds to build command.
[2025-12-31 11:41:25,300356][I][ezpz/launch:436:launch] Executing:
mpiexec
  --envall
  --np=24
  --ppn=12
  --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
  --no-vni
  --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
  python3
  -m
  ezpz.examples.vit
  --compile
  --fsdp
[2025-12-31 11:41:25,301637][I][ezpz/launch:443:launch] Execution started @ 2025-12-31-114125...
[2025-12-31 11:41:25,302092][I][ezpz/launch:139:run_command] Running command:
 mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 python3 -m ezpz.examples.vit --compile --fsdp
cpubind:list x1921c0s7b0n0 pid 88478 rank 12 0: mask 0x1c
cpubind:list x1921c0s7b0n0 pid 88479 rank 13 1: mask 0x1c00
cpubind:list x1921c0s7b0n0 pid 88480 rank 14 2: mask 0x1c0000
cpubind:list x1921c0s7b0n0 pid 88481 rank 15 3: mask 0x1c000000
cpubind:list x1921c0s7b0n0 pid 88482 rank 16 4: mask 0x1c00000000
cpubind:list x1921c0s7b0n0 pid 88483 rank 17 5: mask 0x1c0000000000
cpubind:list x1921c0s7b0n0 pid 88484 rank 18 6: mask 0x1c0000000000000
cpubind:list x1921c0s7b0n0 pid 88485 rank 19 7: mask 0x1c000000000000000
cpubind:list x1921c0s7b0n0 pid 88486 rank 20 8: mask 0x1c00000000000000000
cpubind:list x1921c0s7b0n0 pid 88487 rank 21 9: mask 0x1c0000000000000000000
cpubind:list x1921c0s7b0n0 pid 88488 rank 22 10: mask 0x1c000000000000000000000
cpubind:list x1921c0s7b0n0 pid 88489 rank 23 11: mask 0x1c00000000000000000000000
cpubind:list x1921c0s3b0n0 pid 94973 rank 0 0: mask 0x1c
cpubind:list x1921c0s3b0n0 pid 94974 rank 1 1: mask 0x1c00
cpubind:list x1921c0s3b0n0 pid 94975 rank 2 2: mask 0x1c0000
cpubind:list x1921c0s3b0n0 pid 94976 rank 3 3: mask 0x1c000000
cpubind:list x1921c0s3b0n0 pid 94977 rank 4 4: mask 0x1c00000000
cpubind:list x1921c0s3b0n0 pid 94978 rank 5 5: mask 0x1c0000000000
cpubind:list x1921c0s3b0n0 pid 94979 rank 6 6: mask 0x1c0000000000000
cpubind:list x1921c0s3b0n0 pid 94980 rank 7 7: mask 0x1c000000000000000
cpubind:list x1921c0s3b0n0 pid 94981 rank 8 8: mask 0x1c00000000000000000
cpubind:list x1921c0s3b0n0 pid 94982 rank 9 9: mask 0x1c0000000000000000000
cpubind:list x1921c0s3b0n0 pid 94983 rank 10 10: mask 0x1c000000000000000000000
cpubind:list x1921c0s3b0n0 pid 94984 rank 11 11: mask 0x1c00000000000000000000000
[2025-12-31 11:41:42,956411][I][ezpz/dist:1501:setup_torch_distributed] Using torch_{device,backend}= {xpu, xccl}
[2025-12-31 11:41:42,958950][I][ezpz/dist:1366:setup_torch_DDP] Caught MASTER_PORT=50489 from environment!
[2025-12-31 11:41:42,959628][I][ezpz/dist:1382:setup_torch_DDP] Using torch.distributed.init_process_group with
- master_addr='x1921c0s3b0n0'
- master_port='50489'
- world_size=24
- rank=0
- local_rank=0
- timeout=datetime.timedelta(seconds=3600)
- backend='xccl'
[2025-12-31 11:41:42,960531][I][ezpz/dist:1014:init_process_group] Calling torch.distributed.init_process_group_with: rank=0 world_size=24 backend=xccl
[2025-12-31 11:41:43,681927][I][ezpz/dist:1727:setup_torch] Using device='xpu' with backend='xccl' + 'xccl' for distributed training.
[2025-12-31 11:41:43,682743][W][ezpz/dist:544:print_dist_setup] Using [24 / 24] available "xpu" devices !!
[2025-12-31 11:41:43,683196][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=00/23][local_rank=00/11]
[2025-12-31 11:41:43,682378][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=01/23][local_rank=01/11]
[2025-12-31 11:41:43,682420][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=02/23][local_rank=02/11]
[2025-12-31 11:41:43,682434][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=05/23][local_rank=05/11]
[2025-12-31 11:41:43,682427][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=07/23][local_rank=07/11]
[2025-12-31 11:41:43,682423][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=08/23][local_rank=08/11]
[2025-12-31 11:41:43,682392][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=09/23][local_rank=09/11]
[2025-12-31 11:41:43,682369][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=11/23][local_rank=11/11]
[2025-12-31 11:41:43,682437][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=03/23][local_rank=03/11]
[2025-12-31 11:41:43,682437][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=04/23][local_rank=04/11]
[2025-12-31 11:41:43,682416][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=06/23][local_rank=06/11]
[2025-12-31 11:41:43,682416][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=10/23][local_rank=10/11]
[2025-12-31 11:41:43,685753][I][ezpz/dist:2039:setup_wandb] Setting up wandb from rank=0
[2025-12-31 11:41:43,682476][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=14/23][local_rank=02/11]
[2025-12-31 11:41:43,682507][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=17/23][local_rank=05/11]
[2025-12-31 11:41:43,682477][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=12/23][local_rank=00/11]
[2025-12-31 11:41:43,682541][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=13/23][local_rank=01/11]
[2025-12-31 11:41:43,682519][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=15/23][local_rank=03/11]
[2025-12-31 11:41:43,682518][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=16/23][local_rank=04/11]
[2025-12-31 11:41:43,682547][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=18/23][local_rank=06/11]
[2025-12-31 11:41:43,686167][I][ezpz/dist:2040:setup_wandb] Using WB_PROJECT=ezpz.examples.vit
[2025-12-31 11:41:43,682561][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=19/23][local_rank=07/11]
[2025-12-31 11:41:43,682522][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=20/23][local_rank=08/11]
[2025-12-31 11:41:43,682555][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=21/23][local_rank=09/11]
[2025-12-31 11:41:43,682555][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=22/23][local_rank=10/11]
[2025-12-31 11:41:43,682561][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=23/23][local_rank=11/11]
wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.23.1
wandb: Run data is saved locally in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20251231_114143-zxgz3u90
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run dashing-water-238
wandb:  View project at https://wandb.ai/aurora_gpt/ezpz.examples.vit
wandb:  View run at https://wandb.ai/aurora_gpt/ezpz.examples.vit/runs/zxgz3u90
[2025-12-31 11:41:45,150189][I][ezpz/dist:2069:setup_wandb] wandb.run=[dashing-water-238](https://wandb.ai/aurora_gpt/ezpz.examples.vit/runs/zxgz3u90)
[2025-12-31 11:41:45,156302][I][ezpz/dist:2112:setup_wandb] Running on machine='SunSpot'
[2025-12-31 11:41:45,160729][I][examples/vit:509:main] Using native for SDPA backend
[2025-12-31 11:41:45,161376][I][examples/vit:535:main] Using AttentionBlock Attention with args.compile=True
[2025-12-31 11:41:45,161974][I][examples/vit:287:train_fn] asdict(config)={'img_size': 224, 'batch_size': 128, 'num_heads': 16, 'head_dim': 64, 'depth': 24, 'patch_size': 16, 'hidden_dim': 1024, 'mlp_dim': 4096, 'dropout': 0.0, 'attention_dropout': 0.0, 'num_classes': 1000}
[2025-12-31 11:43:08,029207][I][examples/vit:354:train_fn]
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
VisionTransformer                        [128, 1000]               200,704
├─PatchEmbed: 1-1                        [128, 196, 1024]          787,456
├─Dropout: 1-2                           [128, 196, 1024]          --
├─Identity: 1-3                          [128, 196, 1024]          --
├─Identity: 1-4                          [128, 196, 1024]          --
├─Sequential: 1-5                        [128, 196, 1024]          201,547,776
├─Identity: 1-6                          [128, 196, 1024]          --
├─LayerNorm: 1-7                         [128, 1024]               2,048
├─Dropout: 1-8                           [128, 1024]               --
├─Linear: 1-9                            [128, 1000]               1,025,000
==========================================================================================
Total params: 203,562,984
Trainable params: 203,562,984
Non-trainable params: 0
Total mult-adds (G): 45.69
==========================================================================================
Input size (MB): 77.07
Forward/backward pass size (MB): 49532.61
Params size (MB): 813.45
Estimated Total Size (MB): 50423.13
==========================================================================================
[2025-12-31 11:43:08,033481][I][examples/vit:355:train_fn] Model size: nparams=0.91 B
[2025-12-31 11:43:08,037767][I][ezpz/dist:685:wrap_model] Wrapping model with: fsdp
[2025-12-31 11:43:08,038570][I][ezpz/dist:644:wrap_with_fsdp] Wrapping model model with FSDP + bf16
[2025-12-31 11:43:08,091577][I][ezpz/dist:685:wrap_model] Wrapping model with: fsdp
[2025-12-31 11:43:08,092649][I][ezpz/dist:644:wrap_with_fsdp] Wrapping model model with FSDP + bf16
[2025-12-31 11:43:08,100677][I][examples/vit:399:train_fn] Compiling model
[2025-12-31 11:43:09,398983][I][ezpz/history:220:__init__] Using History with distributed_history=True
[2025-12-31 11:43:09,400415][I][examples/vit:408:train_fn] Training with 24 x xpu (s), using torch_dtype=torch.bfloat16
2025:12:31-11:43:09:(94973) |CCL_WARN| value of CCL_OP_SYNC changed to be 1 (default:0)
2025:12:31-11:43:09:(94973) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
[2025-12-31 11:43:46,170981][I][examples/vit:445:train_fn] iter=10 loss=7.123047 dt=0.553471 dtd=0.002507 dtf=0.045768 dto=0.503978 dtb=0.001217 loss/mean=7.023489 loss/max=7.123047 loss/min=6.898193 loss/std=0.049023 dt/mean=0.552877 dt/max=0.555279 dt/min=0.545638 dt/std=0.001835 dtd/mean=0.002496 dtd/max=0.003046dtd/min=0.001631 dtd/std=0.000394 dtf/mean=0.045593 dtf/max=0.051107 dtf/min=0.042029 dtf/std=0.002142 dto/mean=0.503350 dto/max=0.506550 dto/min=0.498504 dto/std=0.002170 dtb/mean=0.001438 dtb/max=0.001672 dtb/min=0.001196 dtb/std=0.000132
[2025-12-31 11:43:46,851395][I][examples/vit:445:train_fn] iter=11 loss=6.976074 dt=0.541243 dtd=0.004021 dtf=0.062206 dto=0.473205 dtb=0.001811 loss/mean=7.008413 loss/max=7.090820 loss/min=6.912598 loss/std=0.040969 dt/mean=0.536942 dt/max=0.546332 dt/min=0.520310 dt/std=0.007877 dtd/mean=0.002226 dtd/max=0.004021dtd/min=0.001744 dtd/std=0.000545 dtf/mean=0.061639 dtf/max=0.070714 dtf/min=0.044218 dtf/std=0.008138 dto/mean=0.471697 dto/max=0.475141 dto/min=0.469537 dto/std=0.001402 dtb/mean=0.001380 dtb/max=0.001963 dtb/min=0.001150 dtb/std=0.000262
[2025-12-31 11:43:47,430196][I][examples/vit:445:train_fn] iter=12 loss=6.982910 dt=0.562127 dtd=0.001694 dtf=0.074675 dto=0.484511 dtb=0.001247 loss/mean=7.006419 loss/max=7.086182 loss/min=6.901367 loss/std=0.040265 dt/mean=0.552204 dt/max=0.568422 dt/min=0.525764 dt/std=0.012748 dtd/mean=0.001947 dtd/max=0.003053dtd/min=0.001616 dtd/std=0.000366 dtf/mean=0.064005 dtf/max=0.077034 dtf/min=0.037640 dtf/std=0.012711 dto/mean=0.485028 dto/max=0.489131 dto/min=0.477219 dto/std=0.003571 dtb/mean=0.001223 dtb/max=0.001366 dtb/min=0.001123 dtb/std=0.000072
[2025-12-31 11:43:48,018899][I][examples/vit:445:train_fn] iter=13 loss=7.123291 dt=0.574730 dtd=0.001661 dtf=0.081235 dto=0.490605 dtb=0.001229 loss/mean=7.035747 loss/max=7.123291 loss/min=6.962402 loss/std=0.037161 dt/mean=0.565507 dt/max=0.577657 dt/min=0.529671 dt/std=0.015646 dtd/mean=0.002368 dtd/max=0.003979dtd/min=0.001570 dtd/std=0.000743 dtf/mean=0.071740 dtf/max=0.086445 dtf/min=0.036011 dtf/std=0.015086 dto/mean=0.490142 dto/max=0.493799 dto/min=0.486441 dto/std=0.001875 dtb/mean=0.001257 dtb/max=0.001372 dtb/min=0.001148 dtb/std=0.000073
[2025-12-31 11:43:48,621641][I][examples/vit:445:train_fn] iter=14 loss=7.078857 dt=0.585398 dtd=0.001848 dtf=0.092174 dto=0.490163 dtb=0.001212 loss/mean=7.039999 loss/max=7.144287 loss/min=6.930176 loss/std=0.045554 dt/mean=0.572720 dt/max=0.596080 dt/min=0.525547 dt/std=0.021488 dtd/mean=0.002000 dtd/max=0.003031dtd/min=0.001540 dtd/std=0.000411 dtf/mean=0.078718 dtf/max=0.099351 dtf/min=0.035763 dtf/std=0.021073 dto/mean=0.490764 dto/max=0.494255 dto/min=0.485794 dto/std=0.002201 dtb/mean=0.001238 dtb/max=0.001388 dtb/min=0.001134 dtb/std=0.000071
[2025-12-31 11:43:49,184725][I][examples/vit:445:train_fn] iter=15 loss=6.999756 dt=0.547280 dtd=0.002261 dtf=0.042797 dto=0.501021 dtb=0.001201 loss/mean=7.025787 loss/max=7.114990 loss/min=6.905762 loss/std=0.045429 dt/mean=0.549587 dt/max=0.556100 dt/min=0.540050 dt/std=0.004111 dtd/mean=0.002679 dtd/max=0.004398dtd/min=0.001534 dtd/std=0.000824 dtf/mean=0.045316 dtf/max=0.051857 dtf/min=0.038373 dtf/std=0.004181 dto/mean=0.500319 dto/max=0.504468 dto/min=0.495079 dto/std=0.002244 dtb/mean=0.001273 dtb/max=0.001589 dtb/min=0.001138 dtb/std=0.000093
[2025-12-31 11:43:49,751177][I][examples/vit:445:train_fn] iter=16 loss=7.035156 dt=0.552947 dtd=0.005752 dtf=0.045131 dto=0.500782 dtb=0.001282 loss/mean=7.036611 loss/max=7.098877 loss/min=6.955322 loss/std=0.033146 dt/mean=0.551258 dt/max=0.560192 dt/min=0.535375 dt/std=0.007355 dtd/mean=0.002670 dtd/max=0.005752dtd/min=0.001668 dtd/std=0.000785 dtf/mean=0.048956 dtf/max=0.057094 dtf/min=0.035704 dtf/std=0.006351 dto/mean=0.498375 dto/max=0.501600 dto/min=0.493599 dto/std=0.002057 dtb/mean=0.001257 dtb/max=0.001549 dtb/min=0.001154 dtb/std=0.000084
[2025-12-31 11:43:50,342453][I][examples/vit:445:train_fn] iter=17 loss=7.061523 dt=0.574637 dtd=0.001679 dtf=0.081328 dto=0.490444 dtb=0.001186 loss/mean=7.021118 loss/max=7.085205 loss/min=6.936279 loss/std=0.039788 dt/mean=0.573129 dt/max=0.585108 dt/min=0.530401 dt/std=0.013818 dtd/mean=0.002402 dtd/max=0.003376dtd/min=0.001572 dtd/std=0.000573 dtf/mean=0.078347 dtf/max=0.090862 dtf/min=0.034962 dtf/std=0.013843 dto/mean=0.491064 dto/max=0.494646 dto/min=0.484829 dto/std=0.002669 dtb/mean=0.001316 dtb/max=0.002719 dtb/min=0.001134 dtb/std=0.000301
[2025-12-31 11:43:50,907138][I][examples/vit:445:train_fn] iter=18 loss=7.003906 dt=0.549318 dtd=0.001677 dtf=0.050848 dto=0.495618 dtb=0.001175 loss/mean=7.013641 loss/max=7.090820 loss/min=6.952637 loss/std=0.040548 dt/mean=0.546475 dt/max=0.557757 dt/min=0.532767 dt/std=0.007566 dtd/mean=0.001922 dtd/max=0.002493dtd/min=0.001535 dtd/std=0.000293 dtf/mean=0.047784 dtf/max=0.058870 dtf/min=0.035033 dtf/std=0.007802 dto/mean=0.495508 dto/max=0.498537 dto/min=0.490491 dto/std=0.002002 dtb/mean=0.001261 dtb/max=0.001378 dtb/min=0.001142 dtb/std=0.000078
[2025-12-31 11:43:51,484854][I][examples/vit:445:train_fn] iter=19 loss=7.041260 dt=0.561579 dtd=0.001742 dtf=0.065111 dto=0.493460 dtb=0.001265 loss/mean=7.010325 loss/max=7.069092 loss/min=6.918701 loss/std=0.034166 dt/mean=0.559986 dt/max=0.571382 dt/min=0.539101 dt/std=0.012104 dtd/mean=0.002376 dtd/max=0.003085dtd/min=0.001572 dtd/std=0.000508 dtf/mean=0.062027 dtf/max=0.074546 dtf/min=0.042795 dtf/std=0.012000 dto/mean=0.494332 dto/max=0.498158 dto/min=0.490743 dto/std=0.001922 dtb/mean=0.001251 dtb/max=0.001353 dtb/min=0.001152 dtb/std=0.000064
[2025-12-31 11:43:52,055573][I][examples/vit:445:train_fn] iter=20 loss=6.997803 dt=0.552940 dtd=0.001672 dtf=0.053787 dto=0.496280 dtb=0.001200 loss/mean=6.989868 loss/max=7.053223 loss/min=6.925049 loss/std=0.036800 dt/mean=0.547769 dt/max=0.559897 dt/min=0.534946 dt/std=0.008613 dtd/mean=0.001785 dtd/max=0.002056dtd/min=0.001528 dtd/std=0.000147 dtf/mean=0.048491 dtf/max=0.061220 dtf/min=0.035675 dtf/std=0.008928 dto/mean=0.496275 dto/max=0.500639 dto/min=0.490855 dto/std=0.002610 dtb/mean=0.001217 dtb/max=0.001327 dtb/min=0.001148 dtb/std=0.000048
[2025-12-31 11:43:52,611680][I][examples/vit:445:train_fn] iter=21 loss=6.998291 dt=0.540779 dtd=0.001687 dtf=0.038476 dto=0.499404 dtb=0.001211 loss/mean=7.003561 loss/max=7.090332 loss/min=6.914307 loss/std=0.037822 dt/mean=0.544520 dt/max=0.550538 dt/min=0.536078 dt/std=0.004683 dtd/mean=0.002323 dtd/max=0.003039dtd/min=0.001560 dtd/std=0.000511 dtf/mean=0.041033 dtf/max=0.047372 dtf/min=0.036879 dtf/std=0.003967 dto/mean=0.499904 dto/max=0.503667 dto/min=0.495127 dto/std=0.002135 dtb/mean=0.001261 dtb/max=0.001416 dtb/min=0.001143 dtb/std=0.000075
[2025-12-31 11:43:53,183222][I][examples/vit:445:train_fn] iter=22 loss=7.034912 dt=0.555447 dtd=0.001667 dtf=0.051319 dto=0.501284 dtb=0.001176 loss/mean=7.010580 loss/max=7.069092 loss/min=6.896729 loss/std=0.035534 dt/mean=0.551292 dt/max=0.563020 dt/min=0.540584 dt/std=0.006897 dtd/mean=0.001881 dtd/max=0.002553dtd/min=0.001538 dtd/std=0.000234 dtf/mean=0.047625 dtf/max=0.057773 dtf/min=0.035589 dtf/std=0.007505 dto/mean=0.500537 dto/max=0.504555 dto/min=0.495521 dto/std=0.002361 dtb/mean=0.001250 dtb/max=0.001384 dtb/min=0.001140 dtb/std=0.000072
[2025-12-31 11:43:53,740381][I][examples/vit:445:train_fn] iter=23 loss=7.053711 dt=0.532156 dtd=0.001700 dtf=0.037040 dto=0.492084 dtb=0.001332 loss/mean=7.012115 loss/max=7.072510 loss/min=6.937012 loss/std=0.037721 dt/mean=0.543828 dt/max=0.552063 dt/min=0.532156 dt/std=0.005287 dtd/mean=0.002414 dtd/max=0.003177dtd/min=0.001526 dtd/std=0.000522 dtf/mean=0.040836 dtf/max=0.047915 dtf/min=0.036022 dtf/std=0.003855 dto/mean=0.499323 dto/max=0.504240 dto/min=0.491794 dto/std=0.003607 dtb/mean=0.001255 dtb/max=0.001941 dtb/min=0.001140 dtb/std=0.000151
[2025-12-31 11:43:54,312724][I][examples/vit:445:train_fn] iter=24 loss=6.995850 dt=0.554028 dtd=0.001683 dtf=0.059569 dto=0.491485 dtb=0.001290 loss/mean=7.014862 loss/max=7.105469 loss/min=6.932373 loss/std=0.042253 dt/mean=0.552388 dt/max=0.563557 dt/min=0.540821 dt/std=0.007662 dtd/mean=0.001921 dtd/max=0.002703dtd/min=0.001534 dtd/std=0.000338 dtf/mean=0.055772 dtf/max=0.067215 dtf/min=0.043597 dtf/std=0.008266 dto/mean=0.493442 dto/max=0.496154 dto/min=0.490350 dto/std=0.001392 dtb/mean=0.001253 dtb/max=0.001360 dtb/min=0.001146 dtb/std=0.000070
[2025-12-31 11:43:54,864834][I][examples/vit:445:train_fn] iter=25 loss=6.989746 dt=0.529958 dtd=0.001715 dtf=0.036859 dto=0.490215 dtb=0.001169 loss/mean=6.991943 loss/max=7.069336 loss/min=6.922119 loss/std=0.033318 dt/mean=0.541667 dt/max=0.548696 dt/min=0.529958 dt/std=0.005039 dtd/mean=0.002466 dtd/max=0.003185dtd/min=0.001575 dtd/std=0.000553 dtf/mean=0.041371 dtf/max=0.048563 dtf/min=0.036859 dtf/std=0.003874 dto/mean=0.496573 dto/max=0.500643 dto/min=0.490215 dto/std=0.002821 dtb/mean=0.001258 dtb/max=0.001384 dtb/min=0.001135 dtb/std=0.000078
[2025-12-31 11:43:55,437119][I][examples/vit:445:train_fn] iter=26 loss=6.968506 dt=0.546641 dtd=0.001670 dtf=0.054910 dto=0.488716 dtb=0.001344 loss/mean=6.991028 loss/max=7.068604 loss/min=6.904297 loss/std=0.037109 dt/mean=0.549688 dt/max=0.563167 dt/min=0.534744 dt/std=0.008585 dtd/mean=0.001888 dtd/max=0.002943dtd/min=0.001548 dtd/std=0.000339 dtf/mean=0.050904 dtf/max=0.062982 dtf/min=0.035026 dtf/std=0.008717 dto/mean=0.495662 dto/max=0.501035 dto/min=0.488716 dto/std=0.003472 dtb/mean=0.001235 dtb/max=0.001401 dtb/min=0.001144 dtb/std=0.000073
[2025-12-31 11:43:55,995820][I][examples/vit:445:train_fn] iter=27 loss=6.978760 dt=0.537507 dtd=0.001727 dtf=0.043980 dto=0.490458 dtb=0.001342 loss/mean=6.979248 loss/max=7.040527 loss/min=6.909180 loss/std=0.033942 dt/mean=0.546848 dt/max=0.553108 dt/min=0.537507 dt/std=0.004505 dtd/mean=0.002372 dtd/max=0.003054dtd/min=0.001576 dtd/std=0.000510 dtf/mean=0.047175 dtf/max=0.054335 dtf/min=0.042687 dtf/std=0.003816 dto/mean=0.496045 dto/max=0.499992 dto/min=0.490458 dto/std=0.002448 dtb/mean=0.001257 dtb/max=0.001398 dtb/min=0.001141 dtb/std=0.000075
[2025-12-31 11:43:56,561654][I][examples/vit:445:train_fn] iter=28 loss=6.957520 dt=0.544225 dtd=0.001738 dtf=0.049179 dto=0.491973 dtb=0.001335 loss/mean=7.001587 loss/max=7.085205 loss/min=6.938965 loss/std=0.041432 dt/mean=0.547454 dt/max=0.556320 dt/min=0.538058 dt/std=0.005739 dtd/mean=0.001845 dtd/max=0.002195dtd/min=0.001537 dtd/std=0.000170 dtf/mean=0.048554 dtf/max=0.057690 dtf/min=0.037920 dtf/std=0.006230 dto/mean=0.495795 dto/max=0.500834 dto/min=0.491973 dto/std=0.002287 dtb/mean=0.001259 dtb/max=0.001345 dtb/min=0.001142 dtb/std=0.000063
[2025-12-31 11:43:57,189071][I][examples/vit:445:train_fn] iter=29 loss=6.988281 dt=0.534699 dtd=0.001799 dtf=0.040657 dto=0.490928 dtb=0.001315 loss/mean=6.994578 loss/max=7.048340 loss/min=6.937988 loss/std=0.027964 dt/mean=0.544900 dt/max=0.551771 dt/min=0.534699 dt/std=0.004580 dtd/mean=0.002247 dtd/max=0.002995dtd/min=0.001549 dtd/std=0.000491 dtf/mean=0.044591 dtf/max=0.051231 dtf/min=0.039750 dtf/std=0.003824 dto/mean=0.496827 dto/max=0.500810 dto/min=0.490928 dto/std=0.002457 dtb/mean=0.001235 dtb/max=0.001358 dtb/min=0.001123 dtb/std=0.000069
[2025-12-31 11:43:57,748218][I][examples/vit:445:train_fn] iter=30 loss=6.985840 dt=0.533758 dtd=0.001650 dtf=0.050599 dto=0.480177 dtb=0.001332 loss/mean=6.989146 loss/max=7.067383 loss/min=6.923584 loss/std=0.037772 dt/mean=0.540073 dt/max=0.553733 dt/min=0.522639 dt/std=0.009107 dtd/mean=0.001880 dtd/max=0.003254dtd/min=0.001542 dtd/std=0.000426 dtf/mean=0.048866 dtf/max=0.060465 dtf/min=0.035745 dtf/std=0.007963 dto/mean=0.488087 dto/max=0.493582 dto/min=0.480177 dto/std=0.003904 dtb/mean=0.001240 dtb/max=0.001400 dtb/min=0.001133 dtb/std=0.000071
[2025-12-31 11:43:58,304455][I][examples/vit:445:train_fn] iter=31 loss=7.036133 dt=0.537426 dtd=0.001847 dtf=0.041248 dto=0.492881 dtb=0.001450 loss/mean=6.983521 loss/max=7.059814 loss/min=6.831787 loss/std=0.051785 dt/mean=0.544735 dt/max=0.551057 dt/min=0.537426 dt/std=0.004584 dtd/mean=0.002383 dtd/max=0.003034dtd/min=0.001607 dtd/std=0.000480 dtf/mean=0.043669 dtf/max=0.049314 dtf/min=0.038986 dtf/std=0.003890 dto/mean=0.497398 dto/max=0.499443 dto/min=0.492881 dto/std=0.001455 dtb/mean=0.001284 dtb/max=0.001450 dtb/min=0.001134 dtb/std=0.000072
[2025-12-31 11:43:58,874302][I][examples/vit:445:train_fn] iter=32 loss=7.063232 dt=0.548507 dtd=0.003385 dtf=0.052093 dto=0.491810 dtb=0.001220 loss/mean=7.014659 loss/max=7.083496 loss/min=6.964844 loss/std=0.028705 dt/mean=0.549208 dt/max=0.561162 dt/min=0.535437 dt/std=0.008543 dtd/mean=0.001971 dtd/max=0.003385dtd/min=0.001532 dtd/std=0.000501 dtf/mean=0.050375 dtf/max=0.062169 dtf/min=0.034519 dtf/std=0.008469 dto/mean=0.495621 dto/max=0.500835 dto/min=0.491810 dto/std=0.002426 dtb/mean=0.001241 dtb/max=0.001386 dtb/min=0.001151 dtb/std=0.000063
[2025-12-31 11:43:59,460448][I][examples/vit:445:train_fn] iter=33 loss=7.014893 dt=0.565532 dtd=0.001647 dtf=0.077921 dto=0.484568 dtb=0.001397 loss/mean=6.989370 loss/max=7.080322 loss/min=6.889648 loss/std=0.038522 dt/mean=0.571665 dt/max=0.579882 dt/min=0.527901 dt/std=0.010159 dtd/mean=0.002393 dtd/max=0.003695dtd/min=0.001533 dtd/std=0.000620 dtf/mean=0.079081 dtf/max=0.086684 dtf/min=0.035872 dtf/std=0.009693 dto/mean=0.488923 dto/max=0.492225 dto/min=0.484558 dto/std=0.001953 dtb/mean=0.001268 dtb/max=0.001397 dtb/min=0.001142 dtb/std=0.000080
[2025-12-31 11:44:00,033386][I][examples/vit:445:train_fn] iter=34 loss=6.924805 dt=0.551365 dtd=0.001685 dtf=0.057195 dto=0.491146 dtb=0.001340 loss/mean=6.987102 loss/max=7.076416 loss/min=6.924805 loss/std=0.040217 dt/mean=0.551095 dt/max=0.565749 dt/min=0.532316 dt/std=0.009500 dtd/mean=0.001880 dtd/max=0.003064dtd/min=0.001539 dtd/std=0.000441 dtf/mean=0.052058 dtf/max=0.065967 dtf/min=0.035650 dtf/std=0.009136 dto/mean=0.495916 dto/max=0.500652 dto/min=0.491146 dto/std=0.002525 dtb/mean=0.001241 dtb/max=0.001362 dtb/min=0.001132 dtb/std=0.000064
[2025-12-31 11:44:00,591921][I][examples/vit:445:train_fn] iter=35 loss=7.016602 dt=0.537373 dtd=0.001779 dtf=0.042157 dto=0.492096 dtb=0.001342 loss/mean=6.998067 loss/max=7.065430 loss/min=6.934570 loss/std=0.032389 dt/mean=0.545997 dt/max=0.553157 dt/min=0.537373 dt/std=0.005182 dtd/mean=0.002441 dtd/max=0.003077dtd/min=0.001527 dtd/std=0.000540 dtf/mean=0.044476 dtf/max=0.050953 dtf/min=0.039099 dtf/std=0.004223 dto/mean=0.497806 dto/max=0.501319 dto/min=0.492096 dto/std=0.002441 dtb/mean=0.001274 dtb/max=0.001416 dtb/min=0.001141 dtb/std=0.000080
[2025-12-31 11:44:01,158348][I][examples/vit:445:train_fn] iter=36 loss=6.978760 dt=0.541763 dtd=0.001664 dtf=0.048450 dto=0.490279 dtb=0.001369 loss/mean=6.995901 loss/max=7.066895 loss/min=6.940430 loss/std=0.036383 dt/mean=0.548592 dt/max=0.558116 dt/min=0.539835 dt/std=0.006324 dtd/mean=0.001809 dtd/max=0.002132dtd/min=0.001542 dtd/std=0.000170 dtf/mean=0.050593 dtf/max=0.059671 dtf/min=0.039017 dtf/std=0.006891 dto/mean=0.494902 dto/max=0.498382 dto/min=0.490279 dto/std=0.002043 dtb/mean=0.001288 dtb/max=0.001652 dtb/min=0.001146 dtb/std=0.000109
[2025-12-31 11:44:01,711696][I][examples/vit:445:train_fn] iter=37 loss=7.022461 dt=0.535284 dtd=0.001657 dtf=0.037517 dto=0.494719 dtb=0.001390 loss/mean=6.992991 loss/max=7.046387 loss/min=6.944336 loss/std=0.029941 dt/mean=0.543202 dt/max=0.549582 dt/min=0.535284 dt/std=0.004462 dtd/mean=0.002494 dtd/max=0.003559dtd/min=0.001580 dtd/std=0.000603 dtf/mean=0.040854 dtf/max=0.046776 dtf/min=0.036170 dtf/std=0.003687 dto/mean=0.498583 dto/max=0.502137 dto/min=0.494058 dto/std=0.002170 dtb/mean=0.001271 dtb/max=0.001390 dtb/min=0.001139 dtb/std=0.000077
[2025-12-31 11:44:02,280079][I][examples/vit:445:train_fn] iter=38 loss=6.986328 dt=0.548107 dtd=0.001649 dtf=0.053165 dto=0.491910 dtb=0.001383 loss/mean=6.996755 loss/max=7.054688 loss/min=6.938477 loss/std=0.031311 dt/mean=0.546894 dt/max=0.558704 dt/min=0.535178 dt/std=0.008020 dtd/mean=0.001793 dtd/max=0.002057dtd/min=0.001551 dtd/std=0.000153 dtf/mean=0.047758 dtf/max=0.059941 dtf/min=0.035057 dtf/std=0.008577 dto/mean=0.496073 dto/max=0.500114 dto/min=0.491910 dto/std=0.002364 dtb/mean=0.001270 dtb/max=0.001640 dtb/min=0.001150 dtb/std=0.000104
[2025-12-31 11:44:02,856722][I][examples/vit:445:train_fn] iter=39 loss=7.020752 dt=0.546577 dtd=0.016042 dtf=0.036540 dto=0.492701 dtb=0.001293 loss/mean=6.978821 loss/max=7.059570 loss/min=6.918213 loss/std=0.034333 dt/mean=0.564803 dt/max=0.572285 dt/min=0.546577 dt/std=0.005915 dtd/mean=0.002931 dtd/max=0.016042dtd/min=0.001555 dtd/std=0.002778 dtf/mean=0.062856 dtf/max=0.070756 dtf/min=0.036540 dtf/std=0.006919 dto/mean=0.497736 dto/max=0.500957 dto/min=0.492701 dto/std=0.001980 dtb/mean=0.001280 dtb/max=0.001372 dtb/min=0.001146 dtb/std=0.000064
[2025-12-31 11:44:03,484847][I][examples/vit:445:train_fn] iter=40 loss=7.038086 dt=0.547441 dtd=0.001716 dtf=0.055994 dto=0.488404 dtb=0.001327 loss/mean=6.982727 loss/max=7.047607 loss/min=6.922119 loss/std=0.027483 dt/mean=0.553565 dt/max=0.566671 dt/min=0.537408 dt/std=0.008552 dtd/mean=0.002041 dtd/max=0.003054dtd/min=0.001531 dtd/std=0.000497 dtf/mean=0.054263 dtf/max=0.064043 dtf/min=0.035097 dtf/std=0.009167 dto/mean=0.495966 dto/max=0.499893 dto/min=0.488404 dto/std=0.002985 dtb/mean=0.001296 dtb/max=0.001680 dtb/min=0.001146 dtb/std=0.000105
[2025-12-31 11:44:04,051667][I][examples/vit:445:train_fn] iter=41 loss=6.915527 dt=0.552904 dtd=0.001650 dtf=0.059114 dto=0.490879 dtb=0.001260 loss/mean=6.969635 loss/max=7.041992 loss/min=6.915527 loss/std=0.034554 dt/mean=0.548587 dt/max=0.563511 dt/min=0.528830 dt/std=0.008541 dtd/mean=0.002088 dtd/max=0.003764dtd/min=0.001639 dtd/std=0.000605 dtf/mean=0.056581 dtf/max=0.070745 dtf/min=0.036007 dtf/std=0.008648 dto/mean=0.488652 dto/max=0.491791 dto/min=0.483873 dto/std=0.002280 dtb/mean=0.001265 dtb/max=0.001627 dtb/min=0.001147 dtb/std=0.000103
[2025-12-31 11:44:04,617773][I][examples/vit:445:train_fn] iter=42 loss=7.012207 dt=0.548069 dtd=0.001654 dtf=0.053101 dto=0.491959 dtb=0.001356 loss/mean=6.975698 loss/max=7.050781 loss/min=6.911133 loss/std=0.039932 dt/mean=0.551597 dt/max=0.562866 dt/min=0.536423 dt/std=0.008136 dtd/mean=0.002157 dtd/max=0.003054dtd/min=0.001590 dtd/std=0.000517 dtf/mean=0.052623 dtf/max=0.063816 dtf/min=0.034826 dtf/std=0.008871 dto/mean=0.495546 dto/max=0.499639 dto/min=0.486973 dto/std=0.003141 dtb/mean=0.001271 dtb/max=0.001630 dtb/min=0.001166 dtb/std=0.000097
[2025-12-31 11:44:05,192372][I][examples/vit:445:train_fn] iter=43 loss=7.033203 dt=0.550102 dtd=0.001678 dtf=0.057883 dto=0.489192 dtb=0.001350 loss/mean=6.985393 loss/max=7.039551 loss/min=6.920654 loss/std=0.031372 dt/mean=0.553980 dt/max=0.566679 dt/min=0.528025 dt/std=0.009980 dtd/mean=0.002231 dtd/max=0.003273dtd/min=0.001563 dtd/std=0.000566 dtf/mean=0.056648 dtf/max=0.067398 dtf/min=0.035996 dtf/std=0.008460 dto/mean=0.493855 dto/max=0.496731 dto/min=0.486409 dto/std=0.002721 dtb/mean=0.001247 dtb/max=0.001350 dtb/min=0.001152 dtb/std=0.000063
[2025-12-31 11:44:05,826063][I][examples/vit:445:train_fn] iter=44 loss=6.973877 dt=0.532274 dtd=0.001971 dtf=0.036101 dto=0.492857 dtb=0.001345 loss/mean=6.990285 loss/max=7.067139 loss/min=6.924072 loss/std=0.035801 dt/mean=0.555672 dt/max=0.568058 dt/min=0.532274 dt/std=0.009855 dtd/mean=0.002150 dtd/max=0.003292dtd/min=0.001585 dtd/std=0.000497 dtf/mean=0.056383 dtf/max=0.068974 dtf/min=0.036101 dtf/std=0.010292 dto/mean=0.495896 dto/max=0.499771 dto/min=0.488599 dto/std=0.002768 dtb/mean=0.001242 dtb/max=0.001371 dtb/min=0.001137 dtb/std=0.000075
[2025-12-31 11:44:06,387633][I][examples/vit:445:train_fn] iter=45 loss=6.961426 dt=0.535653 dtd=0.001709 dtf=0.048612 dto=0.483982 dtb=0.001350 loss/mean=6.996358 loss/max=7.062012 loss/min=6.884521 loss/std=0.040265 dt/mean=0.542404 dt/max=0.559822 dt/min=0.527250 dt/std=0.008836 dtd/mean=0.001885 dtd/max=0.002413dtd/min=0.001640 dtd/std=0.000180 dtf/mean=0.050246 dtf/max=0.065857 dtf/min=0.035893 dtf/std=0.008954 dto/mean=0.488982 dto/max=0.492639 dto/min=0.482805 dto/std=0.002919 dtb/mean=0.001291 dtb/max=0.001448 dtb/min=0.001156 dtb/std=0.000077
[2025-12-31 11:44:06,961704][I][examples/vit:445:train_fn] iter=46 loss=6.964111 dt=0.550985 dtd=0.001907 dtf=0.059332 dto=0.488405 dtb=0.001342 loss/mean=6.973521 loss/max=7.031738 loss/min=6.858154 loss/std=0.038123 dt/mean=0.554314 dt/max=0.567629 dt/min=0.535901 dt/std=0.009881 dtd/mean=0.002210 dtd/max=0.003346dtd/min=0.001561 dtd/std=0.000565 dtf/mean=0.056978 dtf/max=0.070332 dtf/min=0.034959 dtf/std=0.011268 dto/mean=0.493840 dto/max=0.497745 dto/min=0.485991 dto/std=0.003266 dtb/mean=0.001286 dtb/max=0.001635 dtb/min=0.001164 dtb/std=0.000103
[2025-12-31 11:44:07,528125][I][examples/vit:445:train_fn] iter=47 loss=6.907471 dt=0.543190 dtd=0.001644 dtf=0.047197 dto=0.493004 dtb=0.001345 loss/mean=6.980642 loss/max=7.041748 loss/min=6.907471 loss/std=0.032389 dt/mean=0.549512 dt/max=0.559355 dt/min=0.533583 dt/std=0.008415 dtd/mean=0.002627 dtd/max=0.003939dtd/min=0.001644 dtd/std=0.000690 dtf/mean=0.047363 dtf/max=0.054968 dtf/min=0.035022 dtf/std=0.005985 dto/mean=0.498255 dto/max=0.503351 dto/min=0.489778 dto/std=0.003418 dtb/mean=0.001268 dtb/max=0.001630 dtb/min=0.001153 dtb/std=0.000095
[2025-12-31 11:44:08,097434][I][examples/vit:445:train_fn] iter=48 loss=7.002686 dt=0.541488 dtd=0.002684 dtf=0.049631 dto=0.487836 dtb=0.001338 loss/mean=6.982534 loss/max=7.033203 loss/min=6.895020 loss/std=0.036592 dt/mean=0.551410 dt/max=0.563519 dt/min=0.536141 dt/std=0.008927 dtd/mean=0.002193 dtd/max=0.003062dtd/min=0.001587 dtd/std=0.000548 dtf/mean=0.053340 dtf/max=0.065195 dtf/min=0.034974 dtf/std=0.009226 dto/mean=0.494591 dto/max=0.499796 dto/min=0.486923 dto/std=0.003596 dtb/mean=0.001286 dtb/max=0.001655 dtb/min=0.001141 dtb/std=0.000131
[2025-12-31 11:44:08,661279][I][examples/vit:445:train_fn] iter=49 loss=7.002686 dt=0.542613 dtd=0.001766 dtf=0.048552 dto=0.490940 dtb=0.001355 loss/mean=6.976736 loss/max=7.031982 loss/min=6.920410 loss/std=0.029877 dt/mean=0.548064 dt/max=0.558144 dt/min=0.528284 dt/std=0.008295 dtd/mean=0.002401 dtd/max=0.003570dtd/min=0.001610 dtd/std=0.000556 dtf/mean=0.048864 dtf/max=0.057200 dtf/min=0.034755 dtf/std=0.006469 dto/mean=0.495515 dto/max=0.499618 dto/min=0.490152 dto/std=0.002826 dtb/mean=0.001284 dtb/max=0.001587 dtb/min=0.001162 dtb/std=0.000087
[2025-12-31 11:44:09,231208][I][examples/vit:445:train_fn] iter=50 loss=6.931396 dt=0.547319 dtd=0.001740 dtf=0.053359 dto=0.490897 dtb=0.001322 loss/mean=6.974437 loss/max=7.043213 loss/min=6.879150 loss/std=0.036067 dt/mean=0.552140 dt/max=0.563654 dt/min=0.535980 dt/std=0.008599 dtd/mean=0.002151 dtd/max=0.003277dtd/min=0.001553 dtd/std=0.000540 dtf/mean=0.053484 dtf/max=0.064760 dtf/min=0.035122 dtf/std=0.009319 dto/mean=0.495181 dto/max=0.499243 dto/min=0.489069 dto/std=0.002899 dtb/mean=0.001324 dtb/max=0.002486 dtb/min=0.001163 dtb/std=0.000250
[2025-12-31 11:44:09,797610][I][examples/vit:445:train_fn] iter=51 loss=7.023926 dt=0.552592 dtd=0.002844 dtf=0.049213 dto=0.499337 dtb=0.001197 loss/mean=6.980530 loss/max=7.030029 loss/min=6.914795 loss/std=0.032682 dt/mean=0.550045 dt/max=0.560307 dt/min=0.531263 dt/std=0.007693 dtd/mean=0.002682 dtd/max=0.004037dtd/min=0.001598 dtd/std=0.000747 dtf/mean=0.049188 dtf/max=0.058036 dtf/min=0.038372 dtf/std=0.006299 dto/mean=0.496907 dto/max=0.500463 dto/min=0.489458 dto/std=0.003049 dtb/mean=0.001267 dtb/max=0.001602 dtb/min=0.001156 dtb/std=0.000087
[2025-12-31 11:44:10,363707][I][examples/vit:445:train_fn] iter=52 loss=6.960693 dt=0.551561 dtd=0.001690 dtf=0.050783 dto=0.497870 dtb=0.001218 loss/mean=6.966482 loss/max=7.043457 loss/min=6.915283 loss/std=0.036278 dt/mean=0.550334 dt/max=0.560913 dt/min=0.537666 dt/std=0.007433 dtd/mean=0.002181 dtd/max=0.003166dtd/min=0.001575 dtd/std=0.000547 dtf/mean=0.051788 dtf/max=0.062053 dtf/min=0.037312 dtf/std=0.007927 dto/mean=0.495098 dto/max=0.498044 dto/min=0.490341 dto/std=0.002284 dtb/mean=0.001266 dtb/max=0.001358 dtb/min=0.001140 dtb/std=0.000061
[2025-12-31 11:44:10,931612][I][examples/vit:445:train_fn] iter=53 loss=7.017822 dt=0.545592 dtd=0.001675 dtf=0.050715 dto=0.491968 dtb=0.001235 loss/mean=6.985382 loss/max=7.057373 loss/min=6.908936 loss/std=0.042567 dt/mean=0.551792 dt/max=0.562141 dt/min=0.537001 dt/std=0.007543 dtd/mean=0.002780 dtd/max=0.004606dtd/min=0.001675 dtd/std=0.000926 dtf/mean=0.052424 dtf/max=0.060432 dtf/min=0.039165 dtf/std=0.006296 dto/mean=0.495295 dto/max=0.498500 dto/min=0.489057 dto/std=0.002735 dtb/mean=0.001292 dtb/max=0.001522 dtb/min=0.001190 dtb/std=0.000074
[2025-12-31 11:44:11,527923][I][examples/vit:445:train_fn] iter=54 loss=6.985107 dt=0.575351 dtd=0.001764 dtf=0.083875 dto=0.488249 dtb=0.001463 loss/mean=6.992676 loss/max=7.049805 loss/min=6.947266 loss/std=0.030004 dt/mean=0.572358 dt/max=0.590934 dt/min=0.532585 dt/std=0.019908 dtd/mean=0.002081 dtd/max=0.003054dtd/min=0.001576 dtd/std=0.000473 dtf/mean=0.076998 dtf/max=0.094887 dtf/min=0.034315 dtf/std=0.021102 dto/mean=0.491975 dto/max=0.496513 dto/min=0.484865 dto/std=0.003040 dtb/mean=0.001305 dtb/max=0.001565 dtb/min=0.001157 dtb/std=0.000102
[2025-12-31 11:44:12,092608][I][examples/vit:445:train_fn] iter=55 loss=7.013184 dt=0.541360 dtd=0.001652 dtf=0.047927 dto=0.490385 dtb=0.001395 loss/mean=6.976034 loss/max=7.013184 loss/min=6.916748 loss/std=0.022861 dt/mean=0.548009 dt/max=0.558589 dt/min=0.527243 dt/std=0.008395 dtd/mean=0.002544 dtd/max=0.004086dtd/min=0.001652 dtd/std=0.000705 dtf/mean=0.048646 dtf/max=0.056369 dtf/min=0.035417 dtf/std=0.006174 dto/mean=0.495564 dto/max=0.500990 dto/min=0.488435 dto/std=0.003688 dtb/mean=0.001255 dtb/max=0.001589 dtb/min=0.001152 dtb/std=0.000093
[2025-12-31 11:44:12,661426][I][examples/vit:445:train_fn] iter=56 loss=6.979492 dt=0.543044 dtd=0.001651 dtf=0.047903 dto=0.492262 dtb=0.001228 loss/mean=6.967062 loss/max=7.018066 loss/min=6.936035 loss/std=0.019726 dt/mean=0.551182 dt/max=0.563105 dt/min=0.535433 dt/std=0.008675 dtd/mean=0.002081 dtd/max=0.002967dtd/min=0.001551 dtd/std=0.000493 dtf/mean=0.052216 dtf/max=0.063671 dtf/min=0.034369 dtf/std=0.009174 dto/mean=0.495594 dto/max=0.500209 dto/min=0.490665 dto/std=0.002401 dtb/mean=0.001291 dtb/max=0.001646 dtb/min=0.001155 dtb/std=0.000107
[2025-12-31 11:44:13,233383][I][examples/vit:445:train_fn] iter=57 loss=7.020020 dt=0.550351 dtd=0.001707 dtf=0.058013 dto=0.489272 dtb=0.001359 loss/mean=6.987966 loss/max=7.084473 loss/min=6.930420 loss/std=0.033261 dt/mean=0.553659 dt/max=0.563487 dt/min=0.531056 dt/std=0.009301 dtd/mean=0.002432 dtd/max=0.003670dtd/min=0.001586 dtd/std=0.000675 dtf/mean=0.056426 dtf/max=0.066028 dtf/min=0.039267 dtf/std=0.007650 dto/mean=0.493490 dto/max=0.497225 dto/min=0.488415 dto/std=0.002310 dtb/mean=0.001311 dtb/max=0.001698 dtb/min=0.001156 dtb/std=0.000114
[2025-12-31 11:44:13,799446][I][examples/vit:445:train_fn] iter=58 loss=7.002441 dt=0.539602 dtd=0.001681 dtf=0.044228 dto=0.492348 dtb=0.001344 loss/mean=6.970520 loss/max=7.014648 loss/min=6.915039 loss/std=0.027552 dt/mean=0.548941 dt/max=0.562235 dt/min=0.537927 dt/std=0.007564 dtd/mean=0.002117 dtd/max=0.003019dtd/min=0.001555 dtd/std=0.000459 dtf/mean=0.048742 dtf/max=0.060135 dtf/min=0.035138 dtf/std=0.008185 dto/mean=0.496812 dto/max=0.502624 dto/min=0.489014 dto/std=0.003660 dtb/mean=0.001270 dtb/max=0.001597 dtb/min=0.001141 dtb/std=0.000087
[2025-12-31 11:44:14,365745][I][examples/vit:445:train_fn] iter=59 loss=6.976074 dt=0.545786 dtd=0.004498 dtf=0.044176 dto=0.495701 dtb=0.001412 loss/mean=6.980560 loss/max=7.049561 loss/min=6.927490 loss/std=0.035961 dt/mean=0.550514 dt/max=0.560466 dt/min=0.537196 dt/std=0.006847 dtd/mean=0.002418 dtd/max=0.004498dtd/min=0.001566 dtd/std=0.000686 dtf/mean=0.046305 dtf/max=0.053561 dtf/min=0.034752 dtf/std=0.005206 dto/mean=0.500512 dto/max=0.504297 dto/min=0.494702 dto/std=0.002884 dtb/mean=0.001279 dtb/max=0.001634 dtb/min=0.001151 dtb/std=0.000107
[2025-12-31 11:44:14,936011][I][examples/vit:445:train_fn] iter=60 loss=6.976074 dt=0.543320 dtd=0.001655 dtf=0.049496 dto=0.490866 dtb=0.001303 loss/mean=6.970001 loss/max=7.036621 loss/min=6.898926 loss/std=0.035801 dt/mean=0.552024 dt/max=0.564689 dt/min=0.531746 dt/std=0.009553 dtd/mean=0.002156 dtd/max=0.002979dtd/min=0.001593 dtd/std=0.000493 dtf/mean=0.053857 dtf/max=0.066426 dtf/min=0.035270 dtf/std=0.009612 dto/mean=0.494735 dto/max=0.499199 dto/min=0.487998 dto/std=0.003390 dtb/mean=0.001275 dtb/max=0.001639 dtb/min=0.001146 dtb/std=0.000109
[2025-12-31 11:44:15,506217][I][examples/vit:445:train_fn] iter=61 loss=6.945312 dt=0.548354 dtd=0.001752 dtf=0.052330 dto=0.492835 dtb=0.001437 loss/mean=6.979533 loss/max=7.035400 loss/min=6.931152 loss/std=0.032682 dt/mean=0.554836 dt/max=0.564493 dt/min=0.535241 dt/std=0.007730 dtd/mean=0.002565 dtd/max=0.004511dtd/min=0.001650 dtd/std=0.000884 dtf/mean=0.055713 dtf/max=0.062818 dtf/min=0.042708 dtf/std=0.005875 dto/mean=0.495278 dto/max=0.498574 dto/min=0.488966 dto/std=0.002417 dtb/mean=0.001280 dtb/max=0.001465 dtb/min=0.001148 dtb/std=0.000081
[2025-12-31 11:44:16,076049][I][examples/vit:445:train_fn] iter=62 loss=6.972412 dt=0.546730 dtd=0.001677 dtf=0.048429 dto=0.495294 dtb=0.001330 loss/mean=6.971202 loss/max=7.067139 loss/min=6.926025 loss/std=0.035695 dt/mean=0.551392 dt/max=0.563749 dt/min=0.540256 dt/std=0.006845 dtd/mean=0.002239 dtd/max=0.003050dtd/min=0.001548 dtd/std=0.000518 dtf/mean=0.048769 dtf/max=0.058594 dtf/min=0.035327 dtf/std=0.007358 dto/mean=0.499111 dto/max=0.505159 dto/min=0.491680 dto/std=0.003729 dtb/mean=0.001274 dtb/max=0.001676 dtb/min=0.001192 dtb/std=0.000100
[2025-12-31 11:44:16,650334][I][examples/vit:445:train_fn] iter=63 loss=7.005859 dt=0.552379 dtd=0.001657 dtf=0.062421 dto=0.486960 dtb=0.001341 loss/mean=6.973338 loss/max=7.027588 loss/min=6.924316 loss/std=0.028371 dt/mean=0.556298 dt/max=0.568336 dt/min=0.537445 dt/std=0.009247 dtd/mean=0.002435 dtd/max=0.003509dtd/min=0.001638 dtd/std=0.000643 dtf/mean=0.060589 dtf/max=0.069913 dtf/min=0.042611 dtf/std=0.007837 dto/mean=0.491980 dto/max=0.495426 dto/min=0.486960 dto/std=0.002708 dtb/mean=0.001295 dtb/max=0.001423 dtb/min=0.001170 dtb/std=0.000074
[2025-12-31 11:44:17,210434][I][examples/vit:445:train_fn] iter=64 loss=6.880127 dt=0.541831 dtd=0.001892 dtf=0.047023 dto=0.491627 dtb=0.001288 loss/mean=6.960815 loss/max=7.019775 loss/min=6.880127 loss/std=0.031311 dt/mean=0.548143 dt/max=0.557688 dt/min=0.539300 dt/std=0.005644 dtd/mean=0.002390 dtd/max=0.003387dtd/min=0.001551 dtd/std=0.000566 dtf/mean=0.048313 dtf/max=0.056816 dtf/min=0.038611 dtf/std=0.005935 dto/mean=0.496163 dto/max=0.500338 dto/min=0.488939 dto/std=0.002805 dtb/mean=0.001276 dtb/max=0.001454 dtb/min=0.001152 dtb/std=0.000074
[2025-12-31 11:44:17,787924][I][examples/vit:445:train_fn] iter=65 loss=6.976318 dt=0.559084 dtd=0.001655 dtf=0.062767 dto=0.493255 dtb=0.001406 loss/mean=6.976980 loss/max=7.027100 loss/min=6.893311 loss/std=0.027690 dt/mean=0.556598 dt/max=0.569323 dt/min=0.534337 dt/std=0.009635 dtd/mean=0.002188 dtd/max=0.003194dtd/min=0.001556 dtd/std=0.000524 dtf/mean=0.056096 dtf/max=0.067222 dtf/min=0.036337 dtf/std=0.008326 dto/mean=0.497030 dto/max=0.500812 dto/min=0.490245 dto/std=0.002525 dtb/mean=0.001284 dtb/max=0.001407 dtb/min=0.001138 dtb/std=0.000077
[2025-12-31 11:44:18,381406][I][examples/vit:445:train_fn] iter=66 loss=6.983398 dt=0.573647 dtd=0.001842 dtf=0.084582 dto=0.485829 dtb=0.001394 loss/mean=6.970398 loss/max=7.032715 loss/min=6.902100 loss/std=0.030131 dt/mean=0.569103 dt/max=0.587544 dt/min=0.532877 dt/std=0.017357 dtd/mean=0.002384 dtd/max=0.003915dtd/min=0.001578 dtd/std=0.000702 dtf/mean=0.073834 dtf/max=0.092314 dtf/min=0.036181 dtf/std=0.018459 dto/mean=0.491609 dto/max=0.495428 dto/min=0.485829 dto/std=0.002475 dtb/mean=0.001277 dtb/max=0.001401 dtb/min=0.001148 dtb/std=0.000076
[2025-12-31 11:44:18,947506][I][examples/vit:445:train_fn] iter=67 loss=6.940918 dt=0.543547 dtd=0.001657 dtf=0.050418 dto=0.490096 dtb=0.001376 loss/mean=6.968242 loss/max=7.025391 loss/min=6.918701 loss/std=0.026993 dt/mean=0.547320 dt/max=0.557373 dt/min=0.535130 dt/std=0.006803 dtd/mean=0.002079 dtd/max=0.003073dtd/min=0.001624 dtd/std=0.000505 dtf/mean=0.049035 dtf/max=0.057288 dtf/min=0.034316 dtf/std=0.007389 dto/mean=0.494918 dto/max=0.500424 dto/min=0.487285 dto/std=0.003527 dtb/mean=0.001288 dtb/max=0.001435 dtb/min=0.001139 dtb/std=0.000081
[2025-12-31 11:44:19,502071][I][examples/vit:445:train_fn] iter=68 loss=6.991943 dt=0.533134 dtd=0.001785 dtf=0.039080 dto=0.490912 dtb=0.001356 loss/mean=6.969615 loss/max=7.003418 loss/min=6.930420 loss/std=0.022183 dt/mean=0.542188 dt/max=0.549496 dt/min=0.533134 dt/std=0.005293 dtd/mean=0.002421 dtd/max=0.003266dtd/min=0.001572 dtd/std=0.000535 dtf/mean=0.042995 dtf/max=0.050066 dtf/min=0.038533 dtf/std=0.004131 dto/mean=0.495474 dto/max=0.500356 dto/min=0.488510 dto/std=0.003042 dtb/mean=0.001297 dtb/max=0.001410 dtb/min=0.001162 dtb/std=0.000070
[2025-12-31 11:44:20,070372][I][examples/vit:445:train_fn] iter=69 loss=7.012207 dt=0.544062 dtd=0.001666 dtf=0.051993 dto=0.489078 dtb=0.001324 loss/mean=6.973155 loss/max=7.043945 loss/min=6.887695 loss/std=0.033773 dt/mean=0.547593 dt/max=0.559857 dt/min=0.530368 dt/std=0.008733 dtd/mean=0.001796 dtd/max=0.002224dtd/min=0.001536 dtd/std=0.000192 dtf/mean=0.051259 dtf/max=0.063074 dtf/min=0.038031 dtf/std=0.008452 dto/mean=0.493259 dto/max=0.496895 dto/min=0.485137 dto/std=0.003015 dtb/mean=0.001279 dtb/max=0.001380 dtb/min=0.001159 dtb/std=0.000066
[2025-12-31 11:44:20,627564][I][examples/vit:445:train_fn] iter=70 loss=6.985107 dt=0.538448 dtd=0.001651 dtf=0.044355 dto=0.490999 dtb=0.001443 loss/mean=6.974213 loss/max=7.021484 loss/min=6.896484 loss/std=0.029232 dt/mean=0.543371 dt/max=0.551703 dt/min=0.527644 dt/std=0.006657 dtd/mean=0.002375 dtd/max=0.003153dtd/min=0.001574 dtd/std=0.000496 dtf/mean=0.045956 dtf/max=0.054686 dtf/min=0.035504 dtf/std=0.006137 dto/mean=0.493748 dto/max=0.497705 dto/min=0.486225 dto/std=0.002552 dtb/mean=0.001291 dtb/max=0.001443 dtb/min=0.001157 dtb/std=0.000085
[2025-12-31 11:44:21,204799][I][examples/vit:445:train_fn] iter=71 loss=6.967285 dt=0.554443 dtd=0.001661 dtf=0.062375 dto=0.489160 dtb=0.001246 loss/mean=6.981527 loss/max=7.044678 loss/min=6.928955 loss/std=0.027964 dt/mean=0.549297 dt/max=0.567496 dt/min=0.531782 dt/std=0.012098 dtd/mean=0.001862 dtd/max=0.002834dtd/min=0.001556 dtd/std=0.000373 dtf/mean=0.054260 dtf/max=0.073424 dtf/min=0.036083 dtf/std=0.013392 dto/mean=0.491894 dto/max=0.494504 dto/min=0.487670 dto/std=0.001843 dtb/mean=0.001281 dtb/max=0.001418 dtb/min=0.001158 dtb/std=0.000069
[2025-12-31 11:44:21,760781][I][examples/vit:445:train_fn] iter=72 loss=7.029541 dt=0.535045 dtd=0.001645 dtf=0.040870 dto=0.491108 dtb=0.001422 loss/mean=6.974447 loss/max=7.048584 loss/min=6.897217 loss/std=0.039160 dt/mean=0.543644 dt/max=0.550786 dt/min=0.535045 dt/std=0.004667 dtd/mean=0.002461 dtd/max=0.003550dtd/min=0.001526 dtd/std=0.000591 dtf/mean=0.044358 dtf/max=0.051755 dtf/min=0.040474 dtf/std=0.003896 dto/mean=0.495545 dto/max=0.500688 dto/min=0.488670 dto/std=0.002805 dtb/mean=0.001279 dtb/max=0.001422 dtb/min=0.001155 dtb/std=0.000076
[2025-12-31 11:44:22,328875][I][examples/vit:445:train_fn] iter=73 loss=6.993164 dt=0.547316 dtd=0.001647 dtf=0.056473 dto=0.487819 dtb=0.001377 loss/mean=6.980540 loss/max=7.037354 loss/min=6.936035 loss/std=0.022609 dt/mean=0.550646 dt/max=0.560607 dt/min=0.538582 dt/std=0.006682 dtd/mean=0.002002 dtd/max=0.003021dtd/min=0.001545 dtd/std=0.000469 dtf/mean=0.053917 dtf/max=0.063606 dtf/min=0.041277 dtf/std=0.007304 dto/mean=0.493448 dto/max=0.496497 dto/min=0.487819 dto/std=0.002221 dtb/mean=0.001278 dtb/max=0.001377 dtb/min=0.001162 dtb/std=0.000063
[2025-12-31 11:44:22,887782][I][examples/vit:445:train_fn] iter=74 loss=6.941406 dt=0.531679 dtd=0.002066 dtf=0.037595 dto=0.490738 dtb=0.001280 loss/mean=6.966014 loss/max=7.019043 loss/min=6.914551 loss/std=0.026277 dt/mean=0.545510 dt/max=0.552582 dt/min=0.531679 dt/std=0.005882 dtd/mean=0.002382 dtd/max=0.003061dtd/min=0.001572 dtd/std=0.000461 dtf/mean=0.045219 dtf/max=0.052495 dtf/min=0.037595 dtf/std=0.004411 dto/mean=0.496657 dto/max=0.500951 dto/min=0.490738 dto/std=0.002988 dtb/mean=0.001252 dtb/max=0.001401 dtb/min=0.001154 dtb/std=0.000066
[2025-12-31 11:44:23,458502][I][examples/vit:445:train_fn] iter=75 loss=6.978027 dt=0.548765 dtd=0.001628 dtf=0.059501 dto=0.486367 dtb=0.001269 loss/mean=6.970184 loss/max=7.062744 loss/min=6.931396 loss/std=0.029685 dt/mean=0.548443 dt/max=0.560805 dt/min=0.523214 dt/std=0.010115 dtd/mean=0.001942 dtd/max=0.003119dtd/min=0.001538 dtd/std=0.000407 dtf/mean=0.053364 dtf/max=0.066240 dtf/min=0.035507 dtf/std=0.009720 dto/mean=0.491896 dto/max=0.496909 dto/min=0.484341 dto/std=0.003186 dtb/mean=0.001242 dtb/max=0.001451 dtb/min=0.001156 dtb/std=0.000071
[2025-12-31 11:44:24,014596][I][examples/vit:445:train_fn] iter=76 loss=6.985596 dt=0.536120 dtd=0.001693 dtf=0.039975 dto=0.493047 dtb=0.001404 loss/mean=6.964742 loss/max=7.011475 loss/min=6.906006 loss/std=0.028838 dt/mean=0.543380 dt/max=0.550556 dt/min=0.535266 dt/std=0.004281 dtd/mean=0.002444 dtd/max=0.003496dtd/min=0.001526 dtd/std=0.000557 dtf/mean=0.043236 dtf/max=0.050752 dtf/min=0.039072 dtf/std=0.004082 dto/mean=0.496424 dto/max=0.500685 dto/min=0.490631 dto/std=0.002426 dtb/mean=0.001276 dtb/max=0.001404 dtb/min=0.001152 dtb/std=0.000077
[2025-12-31 11:44:24,586864][I][examples/vit:445:train_fn] iter=77 loss=6.954102 dt=0.544373 dtd=0.002432 dtf=0.053808 dto=0.486797 dtb=0.001336 loss/mean=6.960897 loss/max=7.024658 loss/min=6.891602 loss/std=0.023274 dt/mean=0.551542 dt/max=0.565116 dt/min=0.528373 dt/std=0.009726 dtd/mean=0.002183 dtd/max=0.003900dtd/min=0.001531 dtd/std=0.000681 dtf/mean=0.055409 dtf/max=0.066639 dtf/min=0.038593 dtf/std=0.008326 dto/mean=0.492651 dto/max=0.497892 dto/min=0.486389 dto/std=0.003083 dtb/mean=0.001299 dtb/max=0.001462 dtb/min=0.001137 dtb/std=0.000087
[2025-12-31 11:44:25,148194][I][examples/vit:445:train_fn] iter=78 loss=6.989014 dt=0.538471 dtd=0.001659 dtf=0.042484 dto=0.492945 dtb=0.001383 loss/mean=6.963501 loss/max=7.031006 loss/min=6.866943 loss/std=0.033885 dt/mean=0.546347 dt/max=0.555600 dt/min=0.537032 dt/std=0.005993 dtd/mean=0.002213 dtd/max=0.003034dtd/min=0.001528 dtd/std=0.000459 dtf/mean=0.047623 dtf/max=0.056190 dtf/min=0.039640 dtf/std=0.005395 dto/mean=0.495243 dto/max=0.501018 dto/min=0.488581 dto/std=0.002792 dtb/mean=0.001267 dtb/max=0.001398 dtb/min=0.001154 dtb/std=0.000076
[2025-12-31 11:44:25,721842][I][examples/vit:445:train_fn] iter=79 loss=6.931885 dt=0.542540 dtd=0.002252 dtf=0.051432 dto=0.487435 dtb=0.001422 loss/mean=6.969462 loss/max=7.032715 loss/min=6.901123 loss/std=0.028970 dt/mean=0.550206 dt/max=0.565533 dt/min=0.528354 dt/std=0.009935 dtd/mean=0.001829 dtd/max=0.002445dtd/min=0.001561 dtd/std=0.000229 dtf/mean=0.054415 dtf/max=0.067371 dtf/min=0.034810 dtf/std=0.009250 dto/mean=0.492668 dto/max=0.498301 dto/min=0.486465 dto/std=0.002945 dtb/mean=0.001294 dtb/max=0.001487 dtb/min=0.001164 dtb/std=0.000075
[2025-12-31 11:44:26,276271][I][examples/vit:445:train_fn] iter=80 loss=6.936035 dt=0.532950 dtd=0.001721 dtf=0.038723 dto=0.491147 dtb=0.001358 loss/mean=6.963684 loss/max=7.024170 loss/min=6.921875 loss/std=0.023841 dt/mean=0.540915 dt/max=0.548500 dt/min=0.531412 dt/std=0.004877 dtd/mean=0.002421 dtd/max=0.003114dtd/min=0.001551 dtd/std=0.000514 dtf/mean=0.041307 dtf/max=0.047349 dtf/min=0.036128 dtf/std=0.003940 dto/mean=0.495872 dto/max=0.500373 dto/min=0.488285 dto/std=0.003103 dtb/mean=0.001315 dtb/max=0.001717 dtb/min=0.001169 dtb/std=0.000108
[2025-12-31 11:44:26,846997][I][examples/vit:445:train_fn] iter=81 loss=6.985107 dt=0.549708 dtd=0.001654 dtf=0.061090 dto=0.485630 dtb=0.001333 loss/mean=6.972931 loss/max=7.017578 loss/min=6.905029 loss/std=0.027552 dt/mean=0.552880 dt/max=0.562668 dt/min=0.543452 dt/std=0.006471 dtd/mean=0.002273 dtd/max=0.003894dtd/min=0.001556 dtd/std=0.000726 dtf/mean=0.058314 dtf/max=0.067256 dtf/min=0.045681 dtf/std=0.007345 dto/mean=0.490981 dto/max=0.493956 dto/min=0.485628 dto/std=0.002666 dtb/mean=0.001311 dtb/max=0.001704 dtb/min=0.001162 dtb/std=0.000106
[2025-12-31 11:44:27,413965][I][examples/vit:445:train_fn] iter=82 loss=6.937012 dt=0.546381 dtd=0.001862 dtf=0.055170 dto=0.487981 dtb=0.001368 loss/mean=6.966380 loss/max=7.048340 loss/min=6.838867 loss/std=0.039788 dt/mean=0.551038 dt/max=0.561079 dt/min=0.537928 dt/std=0.008294 dtd/mean=0.002390 dtd/max=0.003380dtd/min=0.001555 dtd/std=0.000528 dtf/mean=0.054311 dtf/max=0.063300 dtf/min=0.041959 dtf/std=0.007793 dto/mean=0.493044 dto/max=0.497323 dto/min=0.486872 dto/std=0.002960 dtb/mean=0.001293 dtb/max=0.001395 dtb/min=0.001167 dtb/std=0.000076
[2025-12-31 11:44:27,984653][I][examples/vit:445:train_fn] iter=83 loss=6.985107 dt=0.550229 dtd=0.001647 dtf=0.056393 dto=0.490807 dtb=0.001382 loss/mean=6.963084 loss/max=7.007812 loss/min=6.907959 loss/std=0.031128 dt/mean=0.549396 dt/max=0.562687 dt/min=0.535747 dt/std=0.008718 dtd/mean=0.001755 dtd/max=0.001962dtd/min=0.001559 dtd/std=0.000121 dtf/mean=0.051470 dtf/max=0.063067 dtf/min=0.035457 dtf/std=0.008940 dto/mean=0.494870 dto/max=0.499143 dto/min=0.490429 dto/std=0.002354 dtb/mean=0.001300 dtb/max=0.001728 dtb/min=0.001155 dtb/std=0.000122
[2025-12-31 11:44:28,537182][I][examples/vit:445:train_fn] iter=84 loss=6.973389 dt=0.534974 dtd=0.001673 dtf=0.036269 dto=0.495746 dtb=0.001287 loss/mean=6.965759 loss/max=7.026855 loss/min=6.903564 loss/std=0.032271 dt/mean=0.542610 dt/max=0.549104 dt/min=0.534161 dt/std=0.004655 dtd/mean=0.002365 dtd/max=0.003183dtd/min=0.001539 dtd/std=0.000499 dtf/mean=0.041301 dtf/max=0.047553 dtf/min=0.036269 dtf/std=0.003798 dto/mean=0.497654 dto/max=0.500980 dto/min=0.491551 dto/std=0.002146 dtb/mean=0.001290 dtb/max=0.001446 dtb/min=0.001182 dtb/std=0.000064
[2025-12-31 11:44:29,115826][I][examples/vit:445:train_fn] iter=85 loss=6.950439 dt=0.558459 dtd=0.001671 dtf=0.068260 dto=0.487266 dtb=0.001263 loss/mean=6.957977 loss/max=7.028564 loss/min=6.898193 loss/std=0.033203 dt/mean=0.559883 dt/max=0.574775 dt/min=0.532710 dt/std=0.012395 dtd/mean=0.002208 dtd/max=0.003862dtd/min=0.001546 dtd/std=0.000686 dtf/mean=0.063394 dtf/max=0.076809 dtf/min=0.043333 dtf/std=0.011269 dto/mean=0.492993 dto/max=0.497380 dto/min=0.485169 dto/std=0.003052 dtb/mean=0.001288 dtb/max=0.001563 dtb/min=0.001162 dtb/std=0.000086
[2025-12-31 11:44:29,670026][I][examples/vit:445:train_fn] iter=86 loss=7.009766 dt=0.534606 dtd=0.001993 dtf=0.039826 dto=0.491411 dtb=0.001375 loss/mean=6.958598 loss/max=7.009766 loss/min=6.899414 loss/std=0.031855 dt/mean=0.540097 dt/max=0.547066 dt/min=0.534287 dt/std=0.004305 dtd/mean=0.002514 dtd/max=0.003489dtd/min=0.001577 dtd/std=0.000571 dtf/mean=0.040688 dtf/max=0.046485 dtf/min=0.035639 dtf/std=0.003391 dto/mean=0.495592 dto/max=0.498272 dto/min=0.491132 dto/std=0.002257 dtb/mean=0.001302 dtb/max=0.001635 dtb/min=0.001158 dtb/std=0.000102
[2025-12-31 11:44:30,240187][I][examples/vit:445:train_fn] iter=87 loss=6.973633 dt=0.546146 dtd=0.001685 dtf=0.051094 dto=0.491970 dtb=0.001395 loss/mean=6.971039 loss/max=7.020264 loss/min=6.933350 loss/std=0.022269 dt/mean=0.549631 dt/max=0.560658 dt/min=0.534888 dt/std=0.008498 dtd/mean=0.001789 dtd/max=0.002021dtd/min=0.001546 dtd/std=0.000133 dtf/mean=0.051294 dtf/max=0.063392 dtf/min=0.034089 dtf/std=0.009265 dto/mean=0.495243 dto/max=0.499371 dto/min=0.491971 dto/std=0.001839 dtb/mean=0.001305 dtb/max=0.001736 dtb/min=0.001192 dtb/std=0.000109
[2025-12-31 11:44:30,798004][I][examples/vit:445:train_fn] iter=88 loss=6.987549 dt=0.537142 dtd=0.001752 dtf=0.040246 dto=0.493808 dtb=0.001337 loss/mean=6.963084 loss/max=7.027344 loss/min=6.892090 loss/std=0.031674 dt/mean=0.546340 dt/max=0.553229 dt/min=0.537142 dt/std=0.004898 dtd/mean=0.002376 dtd/max=0.003142dtd/min=0.001525 dtd/std=0.000507 dtf/mean=0.044116 dtf/max=0.050353 dtf/min=0.039742 dtf/std=0.003789 dto/mean=0.498551 dto/max=0.501502 dto/min=0.493724 dto/std=0.002093 dtb/mean=0.001297 dtb/max=0.001641 dtb/min=0.001172 dtb/std=0.000097
[2025-12-31 11:44:31,358036][I][examples/vit:445:train_fn] iter=89 loss=6.999512 dt=0.540019 dtd=0.001784 dtf=0.043270 dto=0.493582 dtb=0.001382 loss/mean=6.959117 loss/max=7.009033 loss/min=6.911621 loss/std=0.030571 dt/mean=0.545878 dt/max=0.553031 dt/min=0.538571 dt/std=0.004671 dtd/mean=0.002110 dtd/max=0.002961dtd/min=0.001550 dtd/std=0.000511 dtf/mean=0.044835 dtf/max=0.051310 dtf/min=0.035622 dtf/std=0.004969 dto/mean=0.497620 dto/max=0.501857 dto/min=0.492999 dto/std=0.002472 dtb/mean=0.001312 dtb/max=0.001624 dtb/min=0.001139 dtb/std=0.000105
[2025-12-31 11:44:31,914119][I][examples/vit:445:train_fn] iter=90 loss=6.974854 dt=0.533850 dtd=0.001765 dtf=0.036614 dto=0.494134 dtb=0.001338 loss/mean=6.958517 loss/max=7.021973 loss/min=6.883301 loss/std=0.034609 dt/mean=0.542916 dt/max=0.550830 dt/min=0.532366 dt/std=0.005862 dtd/mean=0.002452 dtd/max=0.003262dtd/min=0.001581 dtd/std=0.000536 dtf/mean=0.042942 dtf/max=0.050301 dtf/min=0.036614 dtf/std=0.004438 dto/mean=0.496239 dto/max=0.500294 dto/min=0.487712 dto/std=0.003015 dtb/mean=0.001283 dtb/max=0.001410 dtb/min=0.001172 dtb/std=0.000070
[2025-12-31 11:44:32,481910][I][examples/vit:445:train_fn] iter=91 loss=6.937500 dt=0.541830 dtd=0.001671 dtf=0.050767 dto=0.488064 dtb=0.001327 loss/mean=6.972473 loss/max=7.016113 loss/min=6.905762 loss/std=0.029035 dt/mean=0.548734 dt/max=0.560571 dt/min=0.537999 dt/std=0.007399 dtd/mean=0.001965 dtd/max=0.003166dtd/min=0.001547 dtd/std=0.000407 dtf/mean=0.053735 dtf/max=0.064046 dtf/min=0.043528 dtf/std=0.007340 dto/mean=0.491734 dto/max=0.496073 dto/min=0.485957 dto/std=0.002691 dtb/mean=0.001300 dtb/max=0.001597 dtb/min=0.001170 dtb/std=0.000089
[2025-12-31 11:44:33,043662][I][examples/vit:445:train_fn] iter=92 loss=6.998291 dt=0.548472 dtd=0.001909 dtf=0.041036 dto=0.504320 dtb=0.001207 loss/mean=6.966492 loss/max=7.025879 loss/min=6.869385 loss/std=0.035048 dt/mean=0.550566 dt/max=0.557350 dt/min=0.538100 dt/std=0.005511 dtd/mean=0.002456 dtd/max=0.003087dtd/min=0.001583 dtd/std=0.000483 dtf/mean=0.045417 dtf/max=0.053022 dtf/min=0.039740 dtf/std=0.004579 dto/mean=0.501428 dto/max=0.505175 dto/min=0.492384 dto/std=0.003188 dtb/mean=0.001266 dtb/max=0.001568 dtb/min=0.001139 dtb/std=0.000099
[2025-12-31 11:44:33,616062][I][examples/vit:445:train_fn] iter=93 loss=6.958008 dt=0.557801 dtd=0.001683 dtf=0.057402 dto=0.497513 dtb=0.001203 loss/mean=6.956726 loss/max=7.039795 loss/min=6.893555 loss/std=0.031795 dt/mean=0.551173 dt/max=0.563375 dt/min=0.537488 dt/std=0.008381 dtd/mean=0.001777 dtd/max=0.002197dtd/min=0.001544 dtd/std=0.000160 dtf/mean=0.052961 dtf/max=0.064476 dtf/min=0.038342 dtf/std=0.008512 dto/mean=0.495137 dto/max=0.499545 dto/min=0.488660 dto/std=0.002794 dtb/mean=0.001297 dtb/max=0.001390 dtb/min=0.001186 dtb/std=0.000062
[2025-12-31 11:44:34,170184][I][examples/vit:445:train_fn] iter=94 loss=6.932129 dt=0.535490 dtd=0.001663 dtf=0.036295 dto=0.496155 dtb=0.001377 loss/mean=6.957316 loss/max=7.034180 loss/min=6.903564 loss/std=0.034499 dt/mean=0.542536 dt/max=0.550027 dt/min=0.533227 dt/std=0.004821 dtd/mean=0.002261 dtd/max=0.003167dtd/min=0.001545 dtd/std=0.000522 dtf/mean=0.040928 dtf/max=0.048024 dtf/min=0.036295 dtf/std=0.004014 dto/mean=0.498028 dto/max=0.501231 dto/min=0.491412 dto/std=0.002408 dtb/mean=0.001319 dtb/max=0.001770 dtb/min=0.001174 dtb/std=0.000137
[2025-12-31 11:44:34,766947][I][examples/vit:445:train_fn] iter=95 loss=7.007324 dt=0.574808 dtd=0.001640 dtf=0.085170 dto=0.486684 dtb=0.001314 loss/mean=6.961996 loss/max=7.024170 loss/min=6.907715 loss/std=0.031311 dt/mean=0.577398 dt/max=0.589445 dt/min=0.535511 dt/std=0.010870 dtd/mean=0.002120 dtd/max=0.003110dtd/min=0.001541 dtd/std=0.000493 dtf/mean=0.082637 dtf/max=0.091980 dtf/min=0.039676 dtf/std=0.011149 dto/mean=0.491343 dto/max=0.495454 dto/min=0.486382 dto/std=0.002531 dtb/mean=0.001299 dtb/max=0.001484 dtb/min=0.001167 dtb/std=0.000077
[2025-12-31 11:44:35,342445][I][examples/vit:445:train_fn] iter=96 loss=6.925049 dt=0.555994 dtd=0.001698 dtf=0.062406 dto=0.490538 dtb=0.001352 loss/mean=6.953420 loss/max=7.043213 loss/min=6.894775 loss/std=0.029877 dt/mean=0.556934 dt/max=0.570160 dt/min=0.535535 dt/std=0.010346 dtd/mean=0.002087 dtd/max=0.003044dtd/min=0.001580 dtd/std=0.000471 dtf/mean=0.061529 dtf/max=0.074854 dtf/min=0.038407 dtf/std=0.010709 dto/mean=0.492037 dto/max=0.496335 dto/min=0.484400 dto/std=0.002789 dtb/mean=0.001282 dtb/max=0.001380 dtb/min=0.001141 dtb/std=0.000074
[2025-12-31 11:44:35,914652][I][examples/vit:445:train_fn] iter=97 loss=6.983398 dt=0.548493 dtd=0.001690 dtf=0.055728 dto=0.489711 dtb=0.001364 loss/mean=6.968516 loss/max=7.042725 loss/min=6.877930 loss/std=0.033432 dt/mean=0.554486 dt/max=0.566077 dt/min=0.527362 dt/std=0.010731 dtd/mean=0.002252 dtd/max=0.003045dtd/min=0.001639 dtd/std=0.000466 dtf/mean=0.057385 dtf/max=0.067722 dtf/min=0.034953 dtf/std=0.008798 dto/mean=0.493517 dto/max=0.497564 dto/min=0.487921 dto/std=0.002598 dtb/mean=0.001333 dtb/max=0.002560 dtb/min=0.001131 dtb/std=0.000266
[2025-12-31 11:44:36,481155][I][examples/vit:445:train_fn] iter=98 loss=6.971191 dt=0.540851 dtd=0.001772 dtf=0.049648 dto=0.488115 dtb=0.001315 loss/mean=6.949229 loss/max=7.023438 loss/min=6.886230 loss/std=0.038867 dt/mean=0.548532 dt/max=0.560564 dt/min=0.533105 dt/std=0.008585 dtd/mean=0.002162 dtd/max=0.003021dtd/min=0.001553 dtd/std=0.000494 dtf/mean=0.052808 dtf/max=0.064711 dtf/min=0.035826 dtf/std=0.008877 dto/mean=0.492223 dto/max=0.498025 dto/min=0.485677 dto/std=0.002881 dtb/mean=0.001339 dtb/max=0.001943 dtb/min=0.001168 dtb/std=0.000157
[2025-12-31 11:44:37,042311][I][examples/vit:445:train_fn] iter=99 loss=6.990479 dt=0.546523 dtd=0.001735 dtf=0.048142 dto=0.495458 dtb=0.001188 loss/mean=6.955434 loss/max=6.990479 loss/min=6.912842 loss/std=0.020577 dt/mean=0.546202 dt/max=0.555450 dt/min=0.530648 dt/std=0.006208 dtd/mean=0.002618 dtd/max=0.003311dtd/min=0.001735 dtd/std=0.000414 dtf/mean=0.048004 dtf/max=0.054751 dtf/min=0.039186 dtf/std=0.004728 dto/mean=0.494301 dto/max=0.499130 dto/min=0.487077 dto/std=0.002902 dtb/mean=0.001278 dtb/max=0.001364 dtb/min=0.001174 dtb/std=0.000054
[2025-12-31 11:44:37,610112][I][examples/vit:445:train_fn] iter=100 loss=6.989990 dt=0.543729 dtd=0.003237 dtf=0.049453 dto=0.489665 dtb=0.001375 loss/mean=6.959951 loss/max=7.008057 loss/min=6.914062 loss/std=0.023027 dt/mean=0.550001 dt/max=0.561548 dt/min=0.535187 dt/std=0.008194 dtd/mean=0.002467 dtd/max=0.003242 dtd/min=0.001580 dtd/std=0.000563 dtf/mean=0.052846 dtf/max=0.065690 dtf/min=0.035347 dtf/std=0.009017 dto/mean=0.493395 dto/max=0.496908 dto/min=0.487503 dto/std=0.002604 dtb/mean=0.001294 dtb/max=0.001396 dtb/min=0.001162 dtb/std=0.000070
/lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/src/ezpz/history.py:2223: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:835.)
  x = torch.Tensor(x).numpy(force=True)
[2025-12-31 11:44:37,825148][I][ezpz/history:2385:finalize] Saving plots to /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/mplot (matplotlib) and /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot (tplot)
                            train_dt                                                   train_dt/min
     ┌─────────────────────────────────────────────────────┐      ┌────────────────────────────────────────────────────┐
0.585┤  ▟                                                  │0.5466┤-               -                                   │
     │  █                                                  │0.5422┤-               -                       -           │
0.576┤  █ ▖                    ▗                       ▗   │0.5378┤-  - - -- - - ----      -  -----    -   --   ---    │
     │ ▐▐▐▌                    █      ▟                █   │0.5334┤- -- ----- -- ----- ----- ---- --  --- ------- --- -│
0.567┤ ▞▐▐▌                    █      █                █   │      │- - -- - - -- -  ---- -- ----   -- ------  - -  --- │
     │ ▌▐▐▌        ▟           █      █                █   │0.5291┤- -        - -    --- -   -       - ----         -  │
0.558┤▐ ▐▐▌▟       █           ▛▖     ▛▖          ▖   ▗█   │0.5247┤---        -                         -              │
     │▐ ▐▐▌█ ▖▖    █           ▌▌    ▐ ▌ ▗       ▐▌   █▌▌  │0.5203┤ -                                                  │
     │█ ▐▞█ ██▌    ▛▖  ▗▌▗ ▖ ▗▜▌▌▗  ▗█ ▌ █     ▗▗▐▌   █▌▐  │      └┬────────────┬────────────┬───────────┬────────────┬┘
0.548┤█ ▝▌▝ ██▌▗  ▗▘▌ ▞▞▝▜▐▌ ▐▝▌▌█▗▗▜█ ▌ █ ▄▌  ▛█▐▌▖ ▗▜▌ ▌▖│      1.0         23.5         46.0        68.5        91.0
     │█     ██▚█▟ ▐ ▌▖▌  ▐▐▚▄▌  ▚█▌▘ ▜ ▐▟█▐█▙▌▗▌▐▌█▌ ▐▐▌ █▝│train_dt/min                   iter
0.539┤▝     ▝█▐▛█ ▐ ▜▚▌  ▐▐       ▘    ▐▛█▞██▚▜▌▐▌█▙▌▌▐▌ ▝ │                           train_dt/std
     │       █▐▌▝▄▘   ▘  ▐▞            ▝▌▝▌█▝ ▐▌▝▌▜▝▜  ▘   │      ┌────────────────────────────────────────────────────┐
0.530┤       ▝▝▌          ▘                ▝   ▘           │0.0215┤  *                                                 │
     └┬────────────┬────────────┬────────────┬────────────┬┘0.0182┤  *                      *      *                   │
     1.0         23.5         46.0         68.5        91.0 0.0149┤  *                     **     **                   │
train_dt                      iter                          0.0117┤ ** **                  **     **  *       *        │
                           train_dt/mean                          │ *****     * **    ** * ** ** *** ** ***  **    **  │
      ┌────────────────────────────────────────────────────┐0.0084┤ *** ***** ** ***** ***** ******************* *** **│
0.5774┤                                                ·   │0.0051┤*  *  ** **** ***              * * *** ** ***** *   │
      │    ·                                           ·   │0.0018┤*                                                   │
0.5707┤  ···        ·           ·                      ·   │      └┬────────────┬────────────┬───────────┬────────────┬┘
      │  ···       ··          ··      ·               ·   │      1.0         23.5         46.0        68.5        91.0
      │  ···       ··  ·       ··     ··               ·   │train_dt/std                   iter
0.5639┤ ····       ··  ·       ··     ··               ·   │                           train_dt/max
      │ ·····      ··  ·       ··     ··          ·    ·   │      ┌────────────────────────────────────────────────────┐
0.5572┤ ·····      ··  ·       ··    ···         ··    ··  │0.5961┤  +                                                 │
      │······      ··  ·· ··   ·· · ····       · ··    ··  │0.5878┤  +                      +      +               +   │
0.5504┤······ ··   · · ····· ···········   · ······  ···· ·│0.5795┤  + +        +          ++     ++               +   │
      │·· ········ · ··· ·····   ··   ··· ·········· ··· · │0.5712┤  ++++      ++  +       ++     ++          +    +   │
      │··   ········ ···  ··           ······ ······· ·· · │      │ +++++      + + ++ ++   ++  +++++  +  ++  ++    ++  │
0.5437┤··     · · ·   ·    ·            ··· · ·· ·· ·  ·   │0.5629┤ +++ +++++  + +++ + +++++ ++++ ++++++++++++++ +++ ++│
      │··         ·                            ·  ·        │0.5546┤++ +  +++++++ ++                ++++++ +++++++  + + │
0.5369┤ ·                                                  │0.5463┤ +       +     +                 +      + ++        │
      └┬────────────┬────────────┬───────────┬────────────┬┘      └┬────────────┬────────────┬───────────┬────────────┬┘
      1.0         23.5         46.0        68.5        91.0       1.0         23.5         46.0        68.5        91.0
train_dt/mean                  iter                         train_dt/max                   iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_dt.txt
     ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
0.596┤ ++ train_dt/max                                                                                                 │
     │ -- train_dt/min                                                                                                 │
     │ ·· train_dt/mean                                      +                                                         │
     │ ▞▞ train_dt                                          ++                                                  +      │
     │    ++                                                ++              +                                  ++      │
     │    ▗▌   +                                            ++             ++                                  ++      │
0.583┤    ▐▌  ++                                            ++             ++                                  ++      │
     │    ▐▌  ++                   +                        ++             ++                                  ++      │
     │    ▌▌  ++                  ++                        ++             ++                                  ++      │
     │    ▌▌  ++                  ++                        +▖            + +                                  +·      │
     │   ▗▘▌  ▗▌                  ++                        ▐▌            + ▖                      +           ·▟      │
     │   ▐ ▐  ▐▌ +                +·      +                 ▐▌            +▐▌                     ++           ·█      │
0.571┤  +▞·▐ +▐▌++               +··     ++                 ▐▌            +▐▌                     ++           ·█+     │
     │  +▌·▐ +▞▌++               +··     ++     +  +        ▐▌          ++ ▐▌     +               ++           ·█+     │
     │ +▗▘·▐ +▌▌++               +·▖+    +·+   ++ ++        ▐▌         +++ ▞▌    ++      +  +     ++           ·▛▖+    │
     │ +▐· ▐ +▌▌++   + +  +      +▐▌+    ·· +++  +++ +  +   ▐▐ ++   ++++++ ▌▚    ++     ++ ++ +  +++         + ·▌▌ +   │
     │ +▞· ▐ +▌▐+▟  ++++ ++      +▐▌+    ··      +++++ ++ ++▐▐+  +++    ++·▌▐    ++ +  +++ +++ +++++  +    +++ ▗▘▌  + +│
0.558┤ +▌· ·▌▐ ▐+█+ ++++ ++     +·▞▌+    ··       + + ++ +  ▌▐+         ++·▌▐  + ++++ +++ ++++   ++·▖++   ++++ ▐·▌  ++ │
     │ +▌  ·▌▐ ▐▗▘▌ ++++ ++ +   +·▌▐+ + +··            +    ▌▐          ·+▐ ▐+++ +++ ++++ ++++   +·▐▌++   + +▗▌▐ ▌  ++ │
     │++▌  ·▌▐ ▐▐·▌+▗▌+▗++++ +  +·▌▐+++ +··    ··  ·        ▌▐       · ···▐ ▐++ ++▗+ ++++  +++   +·▐▌++   +  ▐▌▐ ▐·  + │
     │▌+▌  ·▌▞ ▐▞·▐+▐▌+█++ +  +++▐ ▐ + ++· ·▗▌· · ··    ·▗  ▌▝▖ ·   · ··▗·▞ ·▌+  +█+  +++   ++·  +·▐▌+ ++ +  ▌▌▐ ▝▖    │
     │▌▐    ▌▌ ▐▌·▐ ▐▌·█++·     +▐  ▌   +·  ▞▚·▗· ▗▌ · ·▗▘▜·▌·▌· ▖ ·   ▗▜·▌ ·▌+   █+·   +·  ·· ▖·▗·▞▌+·  ++ ▗▘▌▐  ▚·  ·│
     │▐▐    █  ·▘ ·▌▐▌·█ ·· ·    ▐  ▌ ·  ▖ ▗▘▝▄▜ ·▐▚· ··▞  ▌▌ ▌ ▐▌·   ▖▞▐·▌ ·▌+· ▗▜· · ▟· ····▐▚▗▜·▌▌··    ·▐·▌▞  ▝▖·· │
0.546┤▐▐    ▝   ·  ▌▌▐·▛▖▗▌· ·  ·▌  ▚· ·▐▝▀▀   ▐ ·▐▐   ▗▘  ▝▌ ▌ ▌▌ ▗ ▞▝▘ █   ▌· ·▐▐·▗▌·█·  ···▐▝▌▐·▌▚·▗·· · ▌ ▚▌   ▌ ▟ │
     │·█           ▐▌▐·▌▌▐▚ ▗ · ·▌  ▐   ▐--    ▐ ·▐ ▌  ▞      ▌▐ ▐ ▌▜    █   ▚· ▖▐ ▌▐▌ █·▗▌ ··▞  ▐·▌▐·█  ··▗▘ ▐▌   ▚▗▘▚│
     │·▜           ▐▌▐▐ ▌▐▐ █ ··▗▘  ▝▖▗ ▐--    ▐  ▌ ▝▄▄▘      ▚▞ ▐▐      ▜   ▐·▐▌▌ ▌▌▌▐ ▌▞▚ ▟·▌   █ ▐▗▀▖  ·▐  ▐▌   ▐▞  │
     │··    -    -  ▘▝▟-▌▐ ▙▘▌ ·▐    ▌▛▖▌--     ▌ ▌              ▝▌    - -   ▐ ▞▐▌ ▌▌▚▐ ▌▌▐▞▐-▌-  █ ▐▐ ▌ ▖ ▌  ▐▌    ▘  │
     │-·   --   --  --█-▚▞ ▜-▚  ▐    ▜ ▚▌- -    ▌ ▌       ---     --  - --    ▌▌ ▘ █-▐▞ █  ▘▝▟  - █ ▐▞ ▚▞▌▐---▐▌       │
     │-    - -  - ----█-▐▌-  ▐-▗▘-  -  ▝▌- -  - ▌ ▌- -  --  -  - - - -   -   -█    ▜-▐▌ ▝ -- █   -▜ ▐▌-  ▐▌   ▝▌--    -│
0.533┤-    -  - -    -█ ▐▌    ▀▘ -  -       --- ▌▞- -- ---   ----  --     --- ▜   -  ▐▌-- -- ▜    ---▘    ▘    - -  -- │
     │-    -   -      ▝ ▐▌    --  --        ----▝ -   -- -   -- -   -          -  -   ▘--- - -                   - - - │
     │-   --             ▘    --   -         - - --    -     --                 --    -- -  -                    --    │
     │- ----                  --                  -           -                  -    --                          -    │
     │--   -                  --                                                      --                               │
     │--                       -                                                       -                               │
0.520┤ -                                                                                                               │
     └┬───────────────────────────┬───────────────────────────┬───────────────────────────┬───────────────────────────┬┘
     1.0                        23.5                        46.0                        68.5                       91.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_dt_summary.txt
                       train_dt/mean hist                                          train_dt/max hist
    ┌──────────────────────────────────────────────────────┐  ┌────────────────────────────────────────────────────────┐
31.0┤                ██████                                │24┤           ██████                                       │
    │                ██████                                │  │           ███████████                                  │
25.8┤                ██████                                │20┤           ███████████                                  │
    │                ██████                                │  │           ███████████                                  │
    │           ███████████                                │  │           ███████████                                  │
20.7┤           ███████████                                │16┤           ███████████                                  │
    │           ███████████                                │  │█████ ████████████████                                  │
15.5┤           ███████████                                │12┤█████ ████████████████                                  │
    │     █████████████████                                │  │█████ ████████████████                                  │
10.3┤     ██████████████████████                           │ 8┤█████ ██████████████████████                            │
    │     ██████████████████████                           │  │█████ ██████████████████████                            │
    │     ██████████████████████                           │  │█████ ██████████████████████                            │
 5.2┤███████████████████████████                ██████     │ 4┤█████ ████████████████████████████           █████      │
    │████████████████████████████████      ███████████     │  │█████ ████████████████████████████████████████████ █████│
 0.0┤██████████████████████████████████████████████████████│ 0┤█████ ████████████████████████████████████████████ █████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘  └┬─────────────┬─────────────┬────────────┬─────────────┬┘
   0.535        0.546         0.557        0.568      0.579  0.544         0.558         0.571        0.585       0.598
                        train_dt/min hist                                           train_dt/std hist
    ┌──────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
23.0┤                           █████                      │28.0┤                ██████                                │
    │                           ███████████                │    │                ██████                                │
19.2┤                           ███████████                │23.3┤                ██████                                │
    │                           ███████████                │    │           ███████████                                │
    │                      ████████████████                │    │     █████████████████                                │
15.3┤                      ████████████████                │18.7┤     █████████████████                                │
    │                      ████████████████                │    │     █████████████████                                │
11.5┤                      ████████████████                │14.0┤     █████████████████                                │
    │                      ████████████████                │    │     █████████████████                                │
    │                ██████████████████████                │    │     █████████████████                                │
 7.7┤           ████████████████████████████████           │ 9.3┤     ██████████████████████                           │
    │           ████████████████████████████████           │    │     ██████████████████████                           │
 3.8┤           ████████████████████████████████           │ 4.7┤     ██████████████████████                           │
    │           ████████████████████████████████           │    │     ███████████████████████████                      │
    │██████████████████████████████████████████████████████│    │███████████████████████████████████████████      █████│
 0.0┤██████████████████████████████████████████████████████│ 0.0┤███████████████████████████████████████████      █████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
  0.5191       0.5263        0.5334       0.5406     0.5477   0.0010       0.0063        0.0117       0.0170     0.0224
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_dt_hist.txt
                             train_dtb                                                  train_dtb/min
       ┌───────────────────────────────────────────────────┐        ┌──────────────────────────────────────────────────┐
0.00181┤▟                                                  │0.001196┤-                           -             -       │
       │█                                                  │0.001184┤-                      -    -           ---  -    │
0.00170┤█                                                  │0.001172┤-                      -    --          --- --- - │
       │█                                                  │0.001159┤-                -  ----    --  - -   --------- --│
0.00160┤█                                                  │        │ - - -      -  - ------------- -------  --------  │
       │█                                                  │0.001147┤ ----------- ------- - -  ------    -      - - -  │
0.00149┤█                                                  │0.001135┤ ----   - -- -     -          -     -          -  │
       │█           ▖           ▗        ▗                 │0.001123┤ -        -                                       │
       │█          ▐▌▖          ▌▚ ▗▟  ▖ █▗▌ ▖▟    ▖       │        └┬───────────┬────────────┬───────────┬───────────┬┘
0.00138┤█          ▌█▌▗▜  ▄▗  ▖ ▌▐▗██ ▐▝▚█▐▐▐▙▘▚▗▌▞▚▟ ▗▌▗▖▞│        1.0        23.5         46.0        68.5       91.0
       │█      ▟▗▀▚▘█▝▘ ▙█ ▘▀▀▚ ▌▐▌▀▌▀▟  ▜▞▐▐▝  ▘▌▌ ▘▜▐▚▘▚▌│train_dtb/min                   iter
0.00128┤█  ▟ ▖ ▌█   █   ▝▜    ▐ ▌▐▌   ▝  ▐▌ ▜    ▝▌  ▐▐  ▐▌│                            train_dtb/std
       │▌▀▖█▐▌▟ █   ▜         ▐▗▘▝▌       ▘          ▝▟  ▐▌│        ┌──────────────────────────────────────────────────┐
0.00117┤  ▝▘▜▝▜ ▜              ▘                      ▝   ▘│0.000301┤    *                                             │
       └┬────────────┬───────────┬────────────┬───────────┬┘0.000259┤ * **                 *                        *  │
       1.0         23.5        46.0         68.5       91.0 0.000216┤** **                **                        *  │
train_dtb                      iter                         0.000174┤** **                **                        *  │
                           train_dtb/mean                           │** **  *             **                        ** │
        ┌──────────────────────────────────────────────────┐0.000132┤** **  *      ****  ***  ***          ******  *** │
0.001438┤·                                                 │0.000090┤ **** ****** ***** ******** *************** ******│
        │·                                                 │0.000048┤     *    * **  * *    *        * *     *    *  * │
0.001402┤·                                                 │        └┬───────────┬────────────┬───────────┬───────────┬┘
        │·                                                 │        1.0        23.5         46.0        68.5       91.0
        │ ·                                                │train_dtb/std                   iter
0.001365┤ ·                                                │                           train_dtb/max
        │ ·                                              · │       ┌───────────────────────────────────────────────────┐
0.001328┤ ·                    ·                        ·· │0.00272┤    +                                              │
        │ ·  ·                ·· · ·           ··  ··  ··· │0.00249┤   ++                 +                         +  │
0.001291┤ · ··           ·  · ······  · ···  · ············│0.00225┤   ++                 +                         +  │
        │ · ··      ·  · ·  ············ ··· ··   ·  ·· ·· │0.00202┤   ++                 +                         +  │
        │ ···· · ···· · · · ·· ·····       ·· ·       ·    │       │ + ++  +              +                         ++ │
0.001254┤ ··  · ·······    ··     ·         ·              │0.00179┤++ ++  +      + +++ +++   ++++         + +++   +++ │
        │ ··  ·   ···                                      │0.00156┤ + ++  +    + +++ ++  ++++ ++++ ++  +++++++ ++++++ │
0.001217┤ ·   ·                                            │0.00133┤ ++ +++++++++++++ ++   +     + ++++++++ +   + + +++│
        └┬───────────┬────────────┬───────────┬───────────┬┘       └┬────────────┬───────────┬────────────┬───────────┬┘
        1.0        23.5         46.0        68.5       91.0        1.0         23.5        46.0         68.5       91.0
train_dtb/mean                  iter                        train_dtb/max                  iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_dtb.txt
       ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
0.00272┤ ++ train_dtb/max                                                                                              │
       │ -- train_dtb/min                                                                                              │
       │ ·· train_dtb/mean                                                                                             │
       │ ▞▞ train_dtb                                                                                             +    │
       │        ++                                                                                               ++    │
       │        ++                                       +                                                       ++    │
0.00245┤        ++                                      ++                                                       ++    │
       │        ++                                      ++                                                       ++    │
       │        ++                                      ++                                                       ++    │
       │        ++                                      ++                                                       ++    │
       │        ++                                      ++                                                       + +   │
       │        ++                                      ++                                                       + +   │
0.00219┤        ++                                      ++                                                       + +   │
       │        ++                                      ++                                                       + +   │
       │       + +                                      ++                                                       + +   │
       │       + +                                      ++                                                       + +   │
       │       + +                                      ++                                                       + +   │
0.00192┤ +     + +      +                               ++                                                       +  +  │
       │++     + +     ++                               ++                                                       +  +  │
       │++     + +     ++                               ++                                                       +  +  │
       │▗▌     + +     ++                               ++                                                       +  +  │
       │▐▌     + +     ++                               ++                                                     + +  +  │
       │▐▌     + +     ++                               ++       +                            ++ +    +       ++ +  +  │
0.00165┤▐▌     + +     ++               +    +        + ++      ++      +                    + +++   ++       ++ +  +  │
       │▐▚     + +     ++              ++ + + ++    ++ ++ +     + + ++ ++                    + +++   + +++    ++ +  +  │
       │▐▐    ++ +     ++              ++++ +  +   +    + +   ++   + + ++                   +  +++  +    + ++ ++ +  +  │
       │▐▐   + + +     ++              ++++ +  +   +      +  +       ++ +                   +  +++ +     ++ ++ + +  +  │
       │▌▐   +   +     ++              +++++   +   +      + +         + +                   +  ++++      ++ ++  ++  +  │
       │▌▝▖  +   +     ++        ▗▌    +++++   +   +      + +▗▚       ▗  ++   +  ▗  +  +  ++   ++ +      ++ ++  ++  +  │
0.00139┤▌·▌  +    +  ++++ ++++  +▞▌+▖  + ▄▄+   +  +       ++ ▐ ▚   ▗▌ ▛▖   ▞▄▖ ++█+▗▚++ +▖ ▗▞▖  +▗    ▗  ▖+  +   +  + +│
       │▌·▌++      ++    + ▗▄▖++▗▘▌▐▝▄▄▄▀ ▝▖   ▗▄▄▄▄▄▄▄▄▖  + ▐ ▐ ▞▄▌▐▗▘▚ ▖ ▌ ▝▚▖▐ ▌▐ ▜  ▐▝▄▘ ▝▖ ▞▀▖  ▞▘▚▞▝▖   ▗▚ ▗▄▚·+▞│
       │▌·▌      ·  +  ▗▚  ▌ ▝▀▀▀ ▐▐    ·  ▚▄▌▗▘   ·· ··▝▌   ▌·▝▖▌   ▜  ▀▝▟·  ·▝▀·▚▞  ▌ ▌ ▝ ··▝▀··▚·▐·····▝▜ ·▐·▀▘  ▌ ▌│
       │▌·▚ ···▟· ·▖ ··▞·▌·▌ ·· ··▐▌···· ····▝▌····  ·   ▐···▌ ·█ ······· ▝ ··  · ▐▌··▝▀▘· ·       ▚▌      ▝▖ ▌  ·  ▐▗▘│
       │▌ ·▀▄▖▞ ▌ ▐▝▖▗▗▘ ▐▐··  ·  ▝▌                      ▌▗▄▘  ▝       -          ▘                  -     ▚▄▌      █ │
       │ -  -▝▘-▝▄▌-▝▘▜  -▜        -    - -  ----  ---  --▝▘ -----  ---- -- -  ----------- ----------- -  -- ----   -▝-│
0.00112┤  -- -- ---  ---- --------- ---- - --    --   --   --     --       - --           -             --  -    ---   │
       └┬───────────────────────────┬──────────────────────────┬───────────────────────────┬──────────────────────────┬┘
       1.0                        23.5                       46.0                        68.5                      91.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_dtb_summary.txt
                       train_dtb/mean hist                                         train_dtb/max hist
    ┌──────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
28.0┤           ███████████                                │50.0┤█████                                                 │
    │           ███████████                                │    │█████                                                 │
23.3┤           ███████████                                │41.7┤█████                                                 │
    │           ███████████                                │    │█████                                                 │
    │     █████████████████                                │    │█████                                                 │
18.7┤     █████████████████                                │33.3┤█████                                                 │
    │     █████████████████                                │    │█████                                                 │
14.0┤     █████████████████                                │25.0┤█████                                                 │
    │     █████████████████                                │    │█████      █████                                      │
 9.3┤     █████████████████                                │16.7┤█████      █████                                      │
    │     ██████████████████████                           │    │████████████████                                      │
    │███████████████████████████                           │    │████████████████                                      │
 4.7┤███████████████████████████                           │ 8.3┤████████████████                                      │
    │████████████████████████████████      █████      █████│    │████████████████      █████                ██████     │
 0.0┤████████████████████████████████      █████      █████│ 0.0┤███████████████████████████                ███████████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
  0.001208    0.001268      0.001328     0.001388  0.001448   0.00126      0.00164       0.00202      0.00240   0.00278
                      train_dtb/min hist                                          train_dtb/std hist
  ┌────────────────────────────────────────────────────────┐  ┌────────────────────────────────────────────────────────┐
18┤           ██████     ██████                            │36┤      █████                                             │
  │           █████████████████                            │  │      █████                                             │
15┤           █████████████████                            │30┤      █████                                             │
  │           █████████████████                            │  │█████ █████                                             │
  │           █████████████████                            │  │█████ █████                                             │
12┤           █████████████████                            │24┤█████ █████                                             │
  │           █████████████████                            │  │█████ █████                                             │
 9┤      █████████████████████████████████                 │18┤█████ ███████████                                       │
  │      █████████████████████████████████                 │  │█████ ███████████                                       │
  │      █████████████████████████████████                 │  │█████ ███████████                                       │
 6┤      █████████████████████████████████                 │12┤█████ ███████████                                       │
  │      █████████████████████████████████                 │  │█████ ███████████                                       │
 3┤      █████████████████████████████████            █████│ 6┤█████ ███████████                                       │
  │█████ █████████████████████████████████      █████ █████│  │█████ ███████████                                       │
  │█████ ████████████████████████████████████████████ █████│  │█████ ██████████████████████                 █████      │
 0┤█████ ████████████████████████████████████████████ █████│ 0┤█████ ██████████████████████                 █████ █████│
  └┬─────────────┬─────────────┬────────────┬─────────────┬┘  └┬─────────────┬─────────────┬────────────┬─────────────┬┘
 0.001120    0.001140      0.001159     0.001179   0.001199  0.000037    0.000106      0.000174     0.000243   0.000312
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_dtb_hist.txt
                             train_dtd                                                  train_dtd/min
      ┌────────────────────────────────────────────────────┐        ┌──────────────────────────────────────────────────┐
0.0160┤                ▟                                   │0.001744┤ -                                              - │
      │                █                                   │0.001708┤--                                              - │
0.0136┤                █                                   │0.001671┤-- -                   -                        - │
      │                █                                   │0.001635┤-- -             - --  - -  --                 -- │
0.0112┤                █                                   │        │-- -       -    -- --- --- --- -               -- │
      │                █                                   │0.001598┤ - --   -- -   --- ------------ -- -     -  -- ---│
0.0088┤                █                                   │0.001561┤  ----- ----  --- - - -  ------ ----  --------- - │
      │                █                                   │0.001525┤  ---- -- - --- -               - ----  - -       │
      │                █                                   │        └┬───────────┬────────────┬───────────┬───────────┬┘
0.0064┤   ▗            █                                   │        1.0        23.5         46.0        68.5       91.0
      │   █            █                                   │train_dtd/min                   iter
0.0040┤▗  █            █          ▗▌                       │                           train_dtd/std
      │█  █        ▟   █    ▗ ▗   ▐▌         ▗            ▞│       ┌───────────────────────────────────────────────────┐
0.0016┤▝▄▄▀▄▄▄▄▄▄▄▄▛▄▄▄▛▄▄▞▞▌▚▛▄▄▄▟▚▄▄▄▄▄▄▄▞▄▛▞▄▄▄▞▄▄▄▚▄▄▄▌│0.00278┤                *                                  │
      └┬────────────┬────────────┬───────────┬────────────┬┘0.00233┤                *                                  │
      1.0         23.5         46.0        68.5        91.0 0.00189┤                *                                  │
train_dtd                      iter                         0.00145┤                *                                  │
                          train_dtd/mean                           │                *                                  │
       ┌───────────────────────────────────────────────────┐0.00101┤  **            *      ***  *  *       *           │
0.00293┤                ·                                  │0.00056┤*** ****** ************** *************************│
       │                ·                                  │0.00012┤    * **  *   * *  *             *    *  * *  *    │
0.00273┤                ·       ·                          │       └┬────────────┬───────────┬────────────┬───────────┬┘
       │   ·            ·      ··                          │       1.0         23.5        46.0         68.5       91.0
       │  ··            ·    ····   ·                    · │train_dtd/std                  iter
0.00254┤· ··           ··   ······  ·             ·      · │                           train_dtd/max
       │· · ·  ··    ····   ··········· · ·  · ·· · · ·  ··│      ┌────────────────────────────────────────────────────┐
0.00234┤· · ······  ·····   ··············· ···········  · │0.0160┤                +                                   │
       │ ·· ······ ······ · ······························ │0.0137┤                +                                   │
0.00215┤ ·· ············· ···········  ········ ·········· │0.0113┤                +                                   │
       │ ·· ·············· ·    · ··    ····· · ·········  │0.0090┤                +                                   │
       │ ·· ·············· ·            ····· · ···· ··    │      │                +                                   │
0.00195┤ ·  ········ ····  ·            ··· · · ····  ·    │0.0067┤   +            +                                   │
       │     ···  ·· ····               ···   · ····  ·    │0.0043┤++++++++ ++++++++++++++++++++++++++++++++++++++ ++++│
0.00176┤      ·         ·                ·       · ·  ·    │0.0020┤     ++++++    ++   +     +      + +   + +  ++ +    │
       └┬────────────┬───────────┬────────────┬───────────┬┘      └┬────────────┬────────────┬───────────┬────────────┬┘
       1.0         23.5        46.0         68.5       91.0       1.0         23.5         46.0        68.5        91.0
train_dtd/mean                 iter                         train_dtd/max                  iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_dtd.txt
      ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
0.0160┤ ++ train_dtd/max                  ▗▌                                                                           │
      │ -- train_dtd/min                  ▐▌                                                                           │
      │ ·· train_dtd/mean                 ▐▌                                                                           │
      │ ▞▞ train_dtd                      ▐▌                                                                           │
      │                                   ▐▌                                                                           │
      │                                   ▐▌                                                                           │
0.0136┤                                   ▐▌                                                                           │
      │                                   ▐▌                                                                           │
      │                                   ▐▌                                                                           │
      │                                   ▐▌                                                                           │
      │                                   ▐▌                                                                           │
      │                                   ▐▌                                                                           │
0.0112┤                                   ▌▌                                                                           │
      │                                   ▌▌                                                                           │
      │                                   ▌▌                                                                           │
      │                                   ▌▌                                                                           │
      │                                   ▌▌                                                                           │
0.0088┤                                   ▌▚                                                                           │
      │                                   ▌▐                                                                           │
      │                                   ▌▐                                                                           │
      │                                   ▌▐                                                                           │
      │                                   ▌▐                                                                           │
      │                                   ▌▐                                                                           │
0.0064┤                                  ▗▘▐                                                                           │
      │                                  ▐+▐                                                                           │
      │       ▟                          ▐+▐                                                                           │
      │      +█                          ▐+▐                                                                           │
      │      +█                          ▐+▐                                                                           │
      │      ▐▐+                         ▐+▐                +      ▟  +                                                │
0.0039┤ ▖  ++▐▐+                         ▐+▐         +    +++  +  +█ ++     +             +    +    +                  │
      │▐▌ +++▐ ▌                   +    +▐+▐ +      ++ + ++++ ++ ++▛▖++ +  ++      +    +++   ++   + +                 │
      │▞▚+  +▌ ▌+ +  + +  + + ++++▟ +++++▐+▐+ ++++ +  + +  + + ++ +▌▌+ + +++ +++ ++ ++++   + +  + +  + +  +++  +++++++▗│
      │▌▐    ▌·▌+++ +++ ++ +++   ▗▜   ++ ▐+▐     ++  ·▖   ▖ ·   + ▐ ▌ ·        ++ +        ++   ++   ++ ++  + +      ·▌│
      │▘▝▖ ··▌ ▌·+·+ ·+·  · ·+·  ▞ ▌  ·+·▐·▐  ····+··▐▚··▞▌··  · ·▐·▐· ······  ·+· ·  · ··▖·▗····+····+·  · ·+ ·  ···▗▘│
      │ -▌· ▞  ▌ · ·· · ·· · · ··▌·▌·· · ▐·▐··   ▖·▗ ▌▝▖▗▘▐  ·· · ▐ ▐     ▖ ▗·· · · ··▞▖ ▞▚▗▀▖  ▖·   ▖· ·· ·▗·· ··   ▐ │
0.0015┤- ▝▀▀---▝▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀--▝▀▀▀▀▀▀--▀▀▀▀▀▝▀▘▀▘-▝▀--▀▀▀▀▀▀▀▀--▀▀▀▀▀▝▀▘▀▀▀▀▀▀▀▀▀-▝▀▘-▘-▝▀▀▝▀▀▀▀▝▀▀▀▀▀▀▘▀▀▀▀▀▀▀▀▀-│
      └┬───────────────────────────┬───────────────────────────┬──────────────────────────┬───────────────────────────┬┘
      1.0                        23.5                        46.0                       68.5                       91.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_dtd_summary.txt
                      train_dtd/mean hist                                          train_dtd/max hist
  ┌────────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
24┤                            ██████                      │63.0┤█████                                                 │
  │                            ██████                      │    │█████                                                 │
20┤                            ██████                      │52.5┤█████                                                 │
  │                            ██████                      │    │█████                                                 │
  │                            ██████                      │    │█████                                                 │
16┤                 █████      ██████                      │42.0┤█████                                                 │
  │                 █████      ██████                      │    │█████                                                 │
12┤                 █████      ██████                      │31.5┤█████                                                 │
  │█████ █████      █████      ██████                      │    │███████████                                           │
 8┤█████ █████████████████████████████████                 │21.0┤███████████                                           │
  │█████ █████████████████████████████████                 │    │███████████                                           │
  │█████ ███████████████████████████████████████           │    │███████████                                           │
 4┤█████ ███████████████████████████████████████           │10.5┤███████████                                           │
  │█████ ████████████████████████████████████████████ █████│    │███████████                                           │
 0┤█████ ████████████████████████████████████████████ █████│ 0.0┤████████████████                                 █████│
  └┬─────────────┬─────────────┬────────────┬─────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
 0.00170      0.00202       0.00234      0.00266    0.00298   0.0013       0.0052        0.0090       0.0128     0.0167
                       train_dtd/min hist                                         train_dtd/std hist
    ┌──────────────────────────────────────────────────────┐  ┌────────────────────────────────────────────────────────┐
29.0┤█████                                                 │60┤      █████                                             │
    │█████                                                 │  │      █████                                             │
24.2┤█████                                                 │50┤      █████                                             │
    │███████████                                           │  │      █████                                             │
    │████████████████                                      │  │      █████                                             │
19.3┤████████████████                                      │40┤      █████                                             │
    │████████████████                                      │  │      █████                                             │
14.5┤████████████████                                      │30┤      █████                                             │
    │████████████████                                      │  │      █████                                             │
    │████████████████                                      │  │      █████                                             │
 9.7┤████████████████                                      │20┤      █████                                             │
    │████████████████           █████                      │  │█████ █████                                             │
 4.8┤████████████████           █████                      │10┤█████ ███████████                                       │
    │████████████████████████████████                      │  │█████ ███████████                                       │
    │██████████████████████████████████████           █████│  │█████ ███████████                                       │
 0.0┤██████████████████████████████████████           █████│ 0┤█████ ████████████████                             █████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘  └┬─────────────┬─────────────┬────────────┬─────────────┬┘
  0.001515    0.001575      0.001635     0.001694  0.001754  0.00000      0.00073       0.00145      0.00217    0.00290
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_dtd_hist.txt
                             train_dtf                                                 train_dtf/min
      ┌────────────────────────────────────────────────────┐      ┌────────────────────────────────────────────────────┐
0.0922┤  ▟                                                 │0.0457┤                                        -           │
      │ ▗▜                                             ▗   │0.0437┤ -      -                               -  -  -     │
0.0828┤ ▐▐ ▖                   ▗▌     ▗▌               █   │0.0418┤--   - -- -                  --         ---- --     │
      │ ▌▐▐▌        ▖          ▐▌     ▐▌               █   │0.0399┤ -   - ----                 ---    --   ---- --     │
0.0735┤▗▘▐▐▌       ▐▌          ▐▌     ▐▌               █   │      │ - - - ------ --       --  --- - - ---------------- │
      │▐ ▐▐▌       ▐▌          ▐▌     ▐▌               █   │0.0380┤ --- --- ---- --       -- ---- --- --- ------- ---- │
0.0641┤▐ ▐▐▌▖      ▐▌          ▐▚     ▐▌         ▗▌    █   │0.0360┤  ------ - -------------- ---- -- -- - -------  ----│
      │▞ ▐▐█▌      ▐▌          ▐▐    ▟▞▌ ▗▌    ▗ ▐▌    ▌▌  │0.0341┤            -     -   -  -- -   -      -    -       │
      │▌ ▐▐█▌ ▗▌   ▐▚   ▟ ▖▟   ▐▐▗▌  █▌▌ ▐▌▄▌  █ ▟▌   ▗▌▐  │      └┬────────────┬────────────┬───────────┬────────────┬┘
0.0548┤▌ ▐▐█▐ ▐▙▌  ▐▐  ▟ ▜▌█  ▖▐▐▐▌ ▖█▌▌ ▟██▌▟ ▌▀█▌   █▌ ▌ │      1.0         23.5         46.0        68.5        91.0
      │▌ ▐▐▝▐▗██▌▗▟▌▝▖▟█  ▌█▞▟▚▀▝▟▌▐▝█▌▐▐███▚█▟▌ █▙▌ ▟█▌ ▝▄│train_dtf/min                  iter
0.0454┤▘ ▐▐  ███▚▜█▌ ███  █ ▘     ▚▌  ▘▐▞▜██▐██▌ ██▌ ██▌   │                           train_dtf/std
      │   ▘  ███ ▝▛▌ ▝██  █            ▐▌ ▜█▝▌▜▌ █▜▚▌▛█▌   │      ┌────────────────────────────────────────────────────┐
0.0361┤      ▝▜▜      ▝▜  ▜             ▘  ▝   ▘ ▜  ▝▌▝▌   │0.0211┤  *                      *                          │
      └┬────────────┬────────────┬───────────┬────────────┬┘0.0179┤  *                     **      *                   │
      1.0         23.5         46.0        68.5        91.0 0.0148┤  *                     **     **                   │
train_dtf                      iter                         0.0116┤ ** **                  **     **  *                │
                          train_dtf/mean                          │ ******      **  * ** ***** *  ** ** * * * **   ** *│
      ┌────────────────────────────────────────────────────┐0.0085┤ *** ******** *** *************************** ***** │
0.0826┤                                                ·   │0.0053┤*  *  ** **** **     *    * ** * ***** ** ***** * * │
      │  · ·        ·                                  ·   │0.0021┤*                                          *        │
0.0756┤  ···       ··           ·                      ·   │      └┬────────────┬────────────┬───────────┬────────────┬┘
      │  ···       ··          ··      ·               ·   │      1.0         23.5         46.0        68.5        91.0
      │  ···       ··          ··     ··               ·   │train_dtf/std                  iter
0.0687┤ ····       ··          ··     ··               ·   │                           train_dtf/max
      │ ····       ··          ··     ··          ·    ·   │      ┌────────────────────────────────────────────────────┐
0.0617┤ ·····      ··  ·       ··    ···         ··    ··  │0.0994┤  +                                                 │
      │· ····      ··  ·       ··   ····       · ··    ··  │0.0905┤  + +                    +      +               +   │
0.0547┤· ····  ·   ··  ·····   ·· · ····  ·  ······    ··  │0.0817┤  +++        +          ++     ++               +   │
      │· ···· ··   · · · ··· ······················· ··· ··│0.0729┤ +++++      ++          ++     ++  +       +    ++  │
      │· ··········· ···   ····  ···· ·················· · │      │ +++++  +   + + + +++   ++ ++ +++ ++ ++++ ++    ++ +│
0.0477┤·  · ········ ···    ·      ·   ······ ·········· · │0.0641┤+ ++ +++++ ++ +++++ + +++++++++++++++++++++++ +++ + │
      │      ···· ·· ··                 · · · ······· ··   │0.0553┤+  +  +++++++ ++     ++   + +  +++++++ ++++++++++ + │
0.0407┤      ·· ·     ·                        · ··    ·   │0.0465┤      ++ +  +  +                 +      + ++    +   │
      └┬────────────┬────────────┬───────────┬────────────┬┘      └┬────────────┬────────────┬───────────┬────────────┬┘
      1.0         23.5         46.0        68.5        91.0       1.0         23.5         46.0        68.5        91.0
train_dtf/mean                 iter                         train_dtf/max                  iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_dtf.txt
     ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
0.099┤ ++ train_dtf/max                                                                                                │
     │ -- train_dtf/min                                                                                                │
     │ ·· train_dtf/mean                                     +                                                         │
     │ ▞▞ train_dtf                                         ++                                                         │
     │    ▗▌                                                ++              +                                   +      │
     │    ▐▌   +                                            ++             ++                                  ++      │
0.088┤    ▐▌  ++                                            ++             ++                                  ++      │
     │    ▞▌  ++                   +                        ++             ++                                  +▗      │
     │   +▌▌  ++                  ++                        +▖             ▗▌                                  +█      │
     │   +▌▌  ++                  ++                        ▐▌             ▐▌                                  +█      │
     │  +▗▘▌  ▗▌                  ++                        ▐▌             ▐▌                                  ·█      │
     │  +▞ ▌  ▐▌                  +▖                        ▐▌            +▐▌                                  ·█      │
0.078┤  ▗▘·▌  ▐▌                  ▐▌                        ▐▌            +▐▌                      +           ·█      │
     │ +▐ ·▐  ▐▌ +                ▐▌                        ▐▌            +▐▌                     ++           ·█+     │
     │ +▌ ·▐  ▐▌++               +▐▌                        ▐▌            +▞▚     +               ++           ·▌▌     │
     │ +▌ ·▐ +▐▌++               +▐▌      +  +              ▐▐            +▌▐    ++               ++           ·▌▌     │
     │+ ▌· ▐ +▐▌++               +▞▌     ++ ++  +  +        ▐▐          + +▌▐    ++               ++           ·▌▌     │
0.067┤+▐ · ▐ +▌▌++     +         +▌▚     ++ ++ + +++        ▐▐         ++ +▌▐    ++         + +   +▗▌          ▐·▌+    │
     │+▐·  ▐ +▌▚+▗    ++         +▌▐+    +++ ++   ++ +      ▌▐  +   +  +++ ▌▐    ++    + + +++ +  +▐▌          ▐·▌ +  +│
     │+▐·  ▐ +▌▐+█    ++         ·▌▐+    + +  +    +++  +   ▌▐ ++  ++  +++·▌▐  + ++ + ++++ +++  +++▐▌ +    + + ▐·▌  ++ │
     │+▞   ▐ ·▌▐+█+   ++  +      ·▌▐+    +·        +++ ++ + ▌▐+  + + + +▟+▗▘▝▖++ +▟++ ++++ +++   +·▐▌++   ++++ ▐ ▐  ++ │
     │·▌   ▐ ·▌▐▗▜+   +▗ ++    ++▐ ▐+ +  ··        ++ ++++ +▌▝▖   ++  ++█+▐ ·▌++ +█++ +▗+ ++++▗▌ +·▐▌++   ++++ ▐ ▝▖ ++ │
     │·▌   ·▌·▌▐▐·▌  ++█+++ + +++▐ ▐+++ +·· ▗▌ ▗  ▗▌+ ++ +  ▌·▌  ▖++   ·█+▐ ·▌+ ++█+ ++█+ ++++▐▚ +·▐▌++   ++++ ▐ ·▚ ++ │
0.056┤▗▘   ·▌▗▘▐▐·▌ +++█+++++ +++▐ ·▌++ +··▗▘▚▗▜· ▐▌+  +    ▌·▌ ▐▌++ · ▐▐+▌ ·▌+ +▗▜+ ▖+█+  ++·▐▐ ▗·▌▌++   +++▗▌▐  ▝▖++ │
     │▐    ·▌▐ ▐▌·▌+++·█+▗▌+ ++++▐ ·▌++ +· ▌·▐▞▐· ▐▚+       ▌·▌·▞▌++·· ▐▝▖▌ ·▌+  ▐▐+▐▌+█+·  ··▐ ▀▜·▌▌++   +·+▐▌▐   ▌ + │
     │▐    ·▌▐ ▐▌·▐+++·█+▐▌  ++++▐  ▌+ +▗▌ ▌  ▘▐ ·▌▐ · ▗▌   ▌·▌·▌▌ ·  ▖▞·▌▌ ·▌+  ▐▐·▐▌+█·▗▌···▞ ·▐·▌▐++   ··+▐▌▐   ▚· ·│
     │▞    ·▌▐ ▝▌·▐+▗▌·▌▌▐▐   +▖+▌  ▐+·+▐▌ ▌   ▐  ▌▝▖ ·▞▐ ▗▄▌·▌▗▘▚ · ▞▚▌·█  ·▚+▗▌▐·▌▞▌▐▐·▐▌·▟·▌  ▐·▌▐+▟+++·▗·▌▌▌   ▐·· │
     │▌    ·▌▐ ·· ▐+▐▌·▌▌▐▐ ▗ ▐▌·▌  ▐·▗+▐▌ ▌   ▐  ▌·▌▞▄▌ ▚▘   ▌▐ ▐··▞ ▝▌ █   ▐·▐▌▌·▌▌▌▐·▌▐▌·█·▌  ▐·▌▐·█   ·█·▌▌▌    ▚·▞│
     │▌    ·▌▐  · ·▌▐▌▐ ▌▐ ▌█·▐▚·▌  ▝▖█ ▐▐ ▌   ▝▖▐  ▜         ▝▀ ▐ ·▌    ▜   ▐·▞▐▌·▌▌▚▐·▌▌▐▐▐·▌  ▐·▌▐▗▜   ·█·▌▌▌     ▀ │
0.045┤▘     ▌▐     ▌▐▐▐ ▌▞ █ ▌▞▐▐    ▙▀▖▞▐▐     ▌▐               ▐ ▐         ▐·▌▐▌ ▙▘▐▐·▌▌▐▞▝▄▘  ·█ ▐▐·▌· ▗▘█ ▐▌       │
     │ -    ▙▘   - ▚▌▐▐-▐▌ ▝ ▚▌ █    █ ▌▌▐▐     ▌▌                ▀▀ -  -     ▌▌ ▘ █ ▐▌ ▙▘▐▌·█ - ·█-▐▞ ▌ ▖▐-█ ▐▌       │
     │--    ▝   -- ▐▌▝▟-▐▌-- ▐▌ ▜    ▝ ▚▌ █     ▌▌                  -- --     █    █-▐▌ █  ▘ █  - █-▐▌ ▌▞▌▐-▜ ▐▌       │
     │ -        -- ▐▌ █ ▐▌--  ▘ -    --▐▌ █     ▙▘          -   -   - ---     █    ▝ ▐▌ ▜  - █  - █ ▝▌ ▝ ▚▌ - ▐▌-    - │
     │  -   -   --  ▘ █ ▐▌- -----   -  ▝▌ █     █        ----  - -  - -- -    ▝-  -  ▝▌- --- ▝  - █ - -- ▐▌  -▐▌ -  -- │
     │   --- -  - ----▝  ▘-    --  --   --▝--- -▜-- -   -   - --  - -  -  ----  ---    -   - -   -▜  -- --▘   ▝▌ -  - -│
0.034┤        --                 --           -    - ---     - -   -         -              -         -           --   │
     └┬───────────────────────────┬───────────────────────────┬───────────────────────────┬───────────────────────────┬┘
     1.0                        23.5                        46.0                        68.5                       91.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_dtf_summary.txt
                       train_dtf/mean hist                                         train_dtf/max hist
    ┌──────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
23.0┤     ██████                                           │27.0┤                ██████                                │
    │     ██████                                           │    │                ██████                                │
19.2┤     ██████     ██████                                │22.5┤                ██████                                │
    │     ██████     ██████                                │    │                ██████                                │
    │██████████████████████                                │    │                ██████                                │
15.3┤██████████████████████                                │18.0┤█████           ██████                                │
    │██████████████████████                                │    │█████      ███████████                                │
11.5┤██████████████████████                                │13.5┤█████      ███████████                                │
    │██████████████████████                                │    │██████████████████████                                │
 7.7┤██████████████████████                                │ 9.0┤██████████████████████                                │
    │██████████████████████                                │    │███████████████████████████                           │
    │██████████████████████                                │    │████████████████████████████████                      │
 3.8┤████████████████████████████████                 █████│ 4.5┤████████████████████████████████           ██████     │
    │████████████████████████████████      ████████████████│    │████████████████████████████████      ████████████████│
 0.0┤████████████████████████████████      ████████████████│ 0.0┤████████████████████████████████      ████████████████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
   0.039        0.050         0.062        0.073      0.085    0.044        0.059         0.073        0.087      0.102
                      train_dtf/min hist                                           train_dtf/std hist
  ┌────────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
30┤      █████                                             │32.0┤                ██████                                │
  │      █████                                             │    │                ██████                                │
25┤      █████                                             │26.7┤                ██████                                │
  │      █████                                             │    │                ██████                                │
  │      █████                                             │    │                ██████                                │
20┤█████ █████                                             │21.3┤                ██████                                │
  │█████ █████                                             │    │           ███████████                                │
15┤█████ █████                                             │16.0┤           ███████████                                │
  │█████ █████                                             │    │█████      ███████████                                │
  │█████ █████      ███████████                            │    │██████████████████████                                │
10┤█████ █████      ███████████                            │10.7┤██████████████████████                                │
  │█████ █████      ███████████                            │    │██████████████████████                                │
 5┤█████ ██████████████████████           ██████           │ 5.3┤██████████████████████                                │
  │█████ ██████████████████████      ████████████████      │    │███████████████████████████                           │
  │█████ ████████████████████████████████████████████ █████│    │██████████████████████████████████████           █████│
 0┤█████ ████████████████████████████████████████████ █████│ 0.0┤██████████████████████████████████████     ███████████│
  └┬─────────────┬─────────────┬────────────┬─────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
 0.0336       0.0367        0.0399       0.0430      0.0462   0.0013       0.0065        0.0116       0.0168     0.0219
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_dtf_hist.txt
                             train_dto                                                 train_dto/min
      ┌────────────────────────────────────────────────────┐      ┌────────────────────────────────────────────────────┐
0.5043┤▌                                             ▗▌    │0.4985┤-                                                   │
      │▌  ▄   ▖                                      ▐▌    │0.4937┤-  -  --    -  --           -               ---     │
0.4991┤▌ ▐▐  ▞▌               ▟                      ▐▌    │0.4888┤- -- -------- ---- - ---- ---- - - --- ----------   │
      │▌ ▐▐ ▐ ▌               ▌▌   ▖             ▗   ▐▝▖   │0.4840┤- - -      - -   ---- -  -  - - ---------- - -- ----│
0.4939┤▌ ▐▐▞▌ ▌       ▟       ▌▌  ▐▌▗▌           █ ▗▄▟ ▌ ▗▌│      │--         -      - -                               │
      │▌ ▐▐▌  ▚▖ ▗▗▚▗▄▛▟ ▗▟▗▌ ▌▚ ▟▞▙▘▌▟   ▗ ▗▙▌▖ █▗▌ █ ▌ ▐▌│0.4792┤--                                                  │
0.4888┤▌▞▀ ▘   ▐▞▘█▐▌ ▘▐▞▀█▐▌▞▘▐▗▀▌▝ █▐▞▚▞█▟▐██▌▗▜▌  █ ▙▚▟▚│0.4744┤--                                                  │
      │▌▌       ▘ █▐▌   ▘ ▐▞▝   ▘    ▜▐▌   ▜▌▜▜▙▘▝▌  ▝ ▜ ▝ │0.4695┤ -                                                  │
      │▌▌         █▝▌     ▐▌           ▘    ▘  ▝           │      └┬────────────┬────────────┬───────────┬────────────┬┘
0.4836┤█          █        ▘                               │      1.0         23.5         46.0        68.5        91.0
      │█          ▜                                        │train_dto/min                  iter
0.4784┤█                                                   │                           train_dto/std
      │█                                                   │       ┌───────────────────────────────────────────────────┐
0.4732┤▜                                                   │0.00390┤           *                 *                     │
      └┬────────────┬────────────┬───────────┬────────────┬┘0.00349┤ *     * * *         *   * ***  *                  │
      1.0         23.5         46.0        68.5        91.0 0.00307┤ *     *** *      * ** ******* ***  ** *  * * *    │
train_dto                      iter                         0.00265┤ *  *  *** *     ***  ********** ** ******* *** ** │
                          train_dto/mean                           │ * ** **** **** **     *  * *  * *** *   ** *  ** *│
      ┌────────────────────────────────────────────────────┐0.00223┤********* ********     *         ***     ***       │
0.5033┤·                                                   │0.00181┤***  * **  **                     *        *       │
      │·  ·   ·                    ·                 ·     │0.00139┤ *      *   *                                      │
0.4981┤· ··  ··      ···    ·     ···              · · ·   │       └┬────────────┬───────────┬────────────┬───────────┬┘
      │· ·· ··· ···· ··········  ···· · · ···  · ·······   │       1.0         23.5        46.0         68.5       91.0
      │· ····  ·  ··· · ···· ···· ··········· ·············│train_dto/std                  iter
0.4928┤· · ·      ···   ····    ·    · ·  · ····· ·  · ··· │                           train_dto/max
      │· ·        · ·   ·· ·                               │      ┌────────────────────────────────────────────────────┐
0.4875┤··         ·      ·                                 │0.5066┤+  +   +                     +                +     │
      │··                                                  │0.5013┤+ ++  ++ ++++ ++++   + +  ++++ +++ +++ ++ + +++ +   │
0.4822┤··                                                  │0.4961┤+ ++ +  +  +++ + ++++ ++++ ++++ ++++++++++ ++ ++++++│
      │··                                                  │0.4908┤+ + +      + +    + +              +    +           │
      │··                                                  │      │++                                                  │
0.4770┤··                                                  │0.4856┤++                                                  │
      │··                                                  │0.4804┤++                                                  │
0.4717┤ ·                                                  │0.4751┤ +                                                  │
      └┬────────────┬────────────┬───────────┬────────────┬┘      └┬────────────┬────────────┬───────────┬────────────┬┘
      1.0         23.5         46.0        68.5        91.0       1.0         23.5         46.0        68.5        91.0
train_dto/mean                 iter                         train_dto/max                  iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_dto.txt
      ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
0.5066┤ ++ train_dto/max                                                                                               │
      │ -- train_dto/min                                               +                                    +          │
      │ ·· train_dto/mean                                          +  ++                                   +▟          │
      │ ▞▞ train_dto                                 +            ++  ++                                   +█          │
      │▌    ++      +  +                +           ++            ++  ++                                +  +▛▖         │
      │▌    +▄▄    + ▗▌+   + ++   +   ++ + +        ++         + +  + ++   +          +    +      +    + + +▌▌ +       │
0.5004┤▌    ▐·▐    +▗▘▌+  + + +  ++  +++  + +    + +  +   +   + ++ ·+ ++  ++ +++   + ++ + ++ +   ++   +   +·▌▚++       │
      │▌    ▐·▐   + ▞ ▌· +    +  ++ + ++    + + ++ +   ++▗▌   + ++·· ++· + ++  +  ++ +++ +++++   + +  +   +·▌▐ +     + │
      │▌    ▐ ▐+ ++ ▌ ▌· +    + + + + ·+·  ·++ +++ + ·   ▐▝▖++  ++··  ·· + ++  + ++ ++++  + ++  + ·+ + ·· +·▌▝▖·  ++++ │
      │▌    ▐ ▐++  ▐  ▚·+ ·   ·++·+ +··· ···++  +++ ··   ▐·▌ +   +· · ··+  ·+   +++ ++·+      + +·· + ·  ·+·▌ ▚· +    +│
      │▌    ▐ ▝▖+▖ ▞  ▐·+· ····+· · +···  · ·+·  ·+ ·· · ▌·▚   ··· ▗· ··+ ··+  · ++· ·· ·    ·++ ·▗  ··   ·▐·· ▌++   ▗ │
      │▌    ▌-·▌▐▚▐  -▐··     ·+· ·+·  ·▗   ·· ···+·  · ·▌ ▐· · ·· █ ··▟+· ·+···  · ·····  ···++ ·█·· ·   ·▐ · ▌     █ │
0.4942┤▌   +▌ -▌▞▝▌ - ▐ ·     · · ·+·   █   ··  ··+·    ▗▘ ▝▖ ·  ·▗▘▌ ▞▐ · ··  · ··  ··· ····· +· █··  ▞▄▄▌▐   ▌  ·  █·│
      │▌  + ▌ -▌▌   - ▐        ·▗▚·+·  ▐ ▌ ▖··  ▗▌· ▗▌  ▐   ▌·  ▖ ▐-▚▗▘▐· ▗▌·   ···   ··▗▌▗▌· · ·▐▝▖· ▗▘- ▌▐-  ▐ · ·▐▝▖│
      │▌  + ▌ -█·  -   ▚▄    ▞▖·▐ ▜· ▗▌▞ ▝▀▌··▟ ▐▌· ▐▌  ▐   ▐· ▐▌ ▌-▐▞ ▐· ▌▌·     ·    ·▐▌▐▌  ·· ▐-▌ ▄▞ - ▐▐-  ▐··  ▐ ▌│
      │▌ +▗▄▌ -▜·---    -▚- ▞ ▐·▐ ▐·▞▘▝▌   ▚·▞ ▌▌▌· ▌▚ ▞▀  -▐  ▌▐▐- -▘--▌▐ ▚  ▗▌ ▟ ▗▌ ▟ ▞▌▐▐ ▟ · ▞-▌▐-  - ▐▐- -▐ ▗  ▞ ▌│
      │▌ +▌·- - -         ▚▞  ▐·▐ ▐·▌      ▐▐· ▝▌▐·▗▘▐▗▘  -- ▌▞-▝▌- - --▌▞-▐ ▞▘▐▞ ▚▘▚▗▜-▌▐▞▐ █  ▗▘-▐▐    - █-- -▌▌▀▌▌ ▝│
0.4880┤▚+ ▌ -  --          ▘  ▐ ▐ ▝▖▌      ▝▌·   ▐·▐-▐▞ --  -▜ - -- - --▌▌-▐ ▌ -▘  -▐▞▝▄▘▐▌-█-▌ ▞  ▐▌    - ▜ - -▙▘ ▝▌  │
      │▐+▗▘ -  --             ▝▖▐ -▙▘       -   -▐ ▌ -▘     - -      - -█- -▙▘--  - -▘-█ ▐▌-▜-▚▗▘  ▝▌     --   -█ -  --│
      │▐+▐ --  --             -▌▞ -█        - ----▌▌  -     - -         ▝  -█  - -    -▜  ▘ - ▐▞-  --      -    ▝- --  │
      │▐+▐- -  --              ▌▌ -█        --   -█         --              ▝   -     --       ▘    -           --  -  │
      │▐·▞-     -              ▌▌  ▝         -   -▜          -                         -                         -     │
      │▐·▌-                    ▐▌                 -                                                                    │
0.4819┤▐·▌-                    ▐▌                                                                                      │
      │▐·▌                     ▐▌                                                                                      │
      │▐▐-                     ▝▌                                                                                      │
      │▐▐-                                                                                                             │
      │▐▐-                                                                                                             │
      │▐▞-                                                                                                             │
0.4757┤▐▌                                                                                                              │
      │▐▌                                                                                                              │
      │▐▌                                                                                                              │
      │·▘                                                                                                              │
      │-·                                                                                                              │
      │--                                                                                                              │
0.4695┤ -                                                                                                              │
      └┬───────────────────────────┬───────────────────────────┬──────────────────────────┬───────────────────────────┬┘
      1.0                        23.5                        46.0                       68.5                       91.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_dto_summary.txt
                       train_dto/mean hist                                         train_dto/max hist
    ┌──────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
41.0┤                                      █████           │31.0┤                                      █████           │
    │                                      █████           │    │                                      ███████████     │
34.2┤                                      █████           │25.8┤                                      ███████████     │
    │                                      █████           │    │                                      ███████████     │
    │                                      █████           │    │                                      ███████████     │
27.3┤                                      █████           │20.7┤                                      ███████████     │
    │                                ███████████           │    │                                      ███████████     │
20.5┤                                ███████████           │15.5┤                                █████████████████     │
    │                                ███████████           │    │                                █████████████████     │
13.7┤                                █████████████████     │10.3┤                                █████████████████     │
    │                                █████████████████     │    │                                ██████████████████████│
    │                                █████████████████     │    │                           ███████████████████████████│
 6.8┤                           ███████████████████████████│ 5.2┤                           ███████████████████████████│
    │                           ███████████████████████████│    │                           ███████████████████████████│
 0.0┤█████                 ████████████████████████████████│ 0.0┤█████                 ████████████████████████████████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
  0.4703       0.4789        0.4875       0.4961     0.5048    0.474        0.482         0.491        0.499      0.508
                       train_dto/min hist                                          train_dto/std hist
    ┌──────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
28.0┤                                ███████████           │17.0┤                      ██████████                      │
    │                                ███████████           │    │                      ██████████                      │
23.3┤                                ███████████           │14.2┤                      ████████████████                │
    │                                ███████████           │    │                ██████████████████████                │
    │                           ████████████████           │    │                ██████████████████████                │
18.7┤                           ████████████████           │11.3┤                ██████████████████████                │
    │                           ████████████████           │    │                ██████████████████████                │
14.0┤                           ████████████████           │ 8.5┤                ██████████████████████                │
    │                           ████████████████           │    │           ███████████████████████████                │
    │                           ████████████████           │    │           ███████████████████████████                │
 9.3┤                           ██████████████████████     │ 5.7┤           ███████████████████████████     ██████     │
    │                           ██████████████████████     │    │           ███████████████████████████████████████████│
 4.7┤                           ██████████████████████     │ 2.8┤██████████████████████████████████████████████████████│
    │                           ██████████████████████     │    │██████████████████████████████████████████████████████│
    │█████      ███████████████████████████████████████████│    │██████████████████████████████████████████████████████│
 0.0┤█████      ███████████████████████████████████████████│ 0.0┤██████████████████████████████████████████████████████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
  0.4682       0.4761        0.4840       0.4919     0.4998   0.00128      0.00196       0.00265      0.00333   0.00402
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_dto_hist.txt
                           train_loss                                                train_loss/min
     ┌─────────────────────────────────────────────────────┐     ┌─────────────────────────────────────────────────────┐
7.123┤▌▗▌                                                  │6.965┤  -          -                                       │
     │▌▐▌                                                  │6.943┤ --- -  - ---- --        -                           │
7.083┤▌▐▚                                                  │6.920┤ ---------- ---  - --  - - ----- ---- - -   -        │
     │▌▐▐ ▖        ▖                                       │6.898┤ - -  -- -- --    - ------- -- -- ----- ----- -- - --│
7.042┤▌▞▐▐▌▗ ▟    ▐▌   ▗                                   │     │-      -    --      ----       -- ---  - --- ---- -- │
     │▌▌▐▞▌█▗▜    ▞▌  ▖█ ▟    ▖  ▗       ▗▌                │6.876┤            -       -- -               - --    -  -  │
7.002┤▌▌▐▌▜▐▐▐   ▗▘▐▟▐█▐▞▐  ▄▟▙▌▟▛▖  ▖  ▟▐▐       ▗    ▗   │6.854┤            -        -                   --          │
     │▌▌ ▘  ▀ ▚▖ ▟ ▐█▌▜▐▌▐ ▐ ██▜▐▌▌ ▐▌  ▛▟ ▌▗▗ ▗▗ █▗▌▗▌█  ▄│6.832┤            -                             -          │
     │▝▘       ▚█  ▐▌▘ ▐▌▝▖▐ ██  ▘▝▌▐▙▜▐ ▜ ▙▜█ █▛▖▛▌▐▐▌█▗▜ │     └┬────────────┬────────────┬────────────┬────────────┬┘
6.961┤          ▝  ▐▌  ▐▌ ▝█ █▝    ▐▌█ █   █▝▌▌█▌▝▌ ▝▟▐█▐  │     1.0         23.5         46.0         68.5        91.0
     │             ▐▌  ▐▌  █ ▜      ▘█ ▝   ▝  ▚▛▌    ▜▝█▌  │train_loss/min                iter
6.921┤              ▘  ▝▌  █         █                  ▘  │                          train_loss/std
     │                     ▝         █                     │      ┌────────────────────────────────────────────────────┐
6.880┤                               ▜                     │0.0518┤            *                                       │
     └┬────────────┬────────────┬────────────┬────────────┬┘0.0464┤* **       **                                       │
     1.0         23.5         46.0         68.5        91.0 0.0411┤ *** *  * ***           *                           │
train_loss                    iter                          0.0358┤ ***************  *** ***   *      *     *        * │
                         train_loss/mean                          │   * * * **** *** ** ****  * * * * *   ************ │
     ┌─────────────────────────────────────────────────────┐0.0304┤           **  * *    *  ***  ****** ***** ****  ** │
7.040┤  ·                                                  │0.0251┤                          *      *  * * *   *     **│
     │  ··                                                 │0.0197┤                          *                       * │
7.025┤ · ·                                                 │      └┬────────────┬────────────┬───────────┬────────────┬┘
     │··  ·                                                │      1.0         23.5         46.0        68.5        91.0
     │··   ·  ·    ·                                       │train_loss/std                 iter
7.010┤ ·   · ··   ··                                       │                         train_loss/max
     │     ·· · · ···                                      │     ┌─────────────────────────────────────────────────────┐
6.995┤     ··  ········   ·    ·                           │7.144┤  +                                                  │
     │      ·  ·· ··· ·  ··    · ·                         │7.119┤+ ++                                                 │
6.979┤          · ·    ·· ··· ·· ···     ··                │7.093┤ + + ++ +                                            │
     │                 ··  · ·· ···· ·· ···    ·     ·     │7.067┤ +  +++++++ +++            +                         │
     │                  ·     ·  ········  ·· · ··· ··  ·  │     │      +  ++++ +++++ +    + ++ +     + +   +          │
6.964┤                               ·      ··· ······ ·· ·│7.042┤          +       ++ +++++ +++ +++++++++++++ + +++++ │
     │                                           ·· · ···· │7.016┤                          +++  +  ++ ++  ++ + ++   ++│
6.949┤                                                   · │6.990┤                                                   + │
     └┬────────────┬────────────┬────────────┬────────────┬┘     └┬────────────┬────────────┬────────────┬────────────┬┘
     1.0         23.5         46.0         68.5        91.0      1.0         23.5         46.0         68.5        91.0
train_loss/mean               iter                          train_loss/max                iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_loss.txt
     ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
7.144┤ ++ train_loss/max                                                                                               │
     │ -- train_loss/min                                                                                               │
     │ ·· train_loss/mean                                                                                              │
     │ ▞▞ train_loss                                                                                                   │
     │▌  ▐▌ +          +                                                                                               │
     │▌ +▐▐  +        ++                                                                                               │
7.092┤▌++▐▐   + +   + + +                                                                                              │
     │▌ +▐▝▖   ++  ++ + +   +    +++                            +                                                      │
     │▐  ▌ ▌     + + ++  ++++   +   +                          ++                                                      │
     │▐  ▌ ▌   ▖ ++       ++ + ++▟   ++   +     +++            ++      +               +                               │
     │▐  ▌ ▚  ▐▌  +   ▗   ++ ++  █     + ++   ++  +         +  ++     ++              ++                               │
     │▐  ▌ ▐  ▌▌ ▗   ▗▜   ++  + ▐ ▌     +  +++++  + +   + ++ + + + + + +       +  ++  ++        +                ++    │
7.040┤▐  ▌·▐ ▐ ▌ █   ▌▐    +    ▐ ▌        ▟   ▗  ++ + +++   + + ++ ++ +    + ++ +  + ++    +  ++            ++++ +    │
     │▝▖▗▘ ▝▖▌·▐▗▜  ▐ ▐         ▌ ▐       ▗▜  ▗▜   +  ++ ▗   + + ++     + ++ ++ ++ ▟ +++ +++ ++ + ++   +    +   +  +   │
     │·▌▐   ▌▌ ▐▐ ▌ ▐  ▌       ▗▘ ▝▖ ▗ ▗▌ ▞▐  ▞▐         █  ▖+ +▗▌+      +   ++  + ▛▖ +++     + ++  + ++  ++        +  │
     │ ▌▐   █  ▐▌·▌ ▌··▌       ▐ · ▌ █ ▐▌▗▘▐ ▗▘▐         █ ▐▌ ▟ ▐▐+     ▗    ++▗▌  ▌▌   +        +   ▖  ++      ▗   + +│
     │ ▌▐   ▜   ▘·▚▄▌  ▌·   ·  ▞· ·▌ █ ▌▐▐ ▐ ▐  ▌    ▞▀▌ █ ▐▚▗▜ ▌ ▌     █     +▌▚  ▌▐               ▐▌   ▖  ▗   █   ++ │
6.988┤ ▌▐         ··   ▝▄▖·· ·▖▌·  ▌·▌▙▘▐▌  ▌▐  ▌··  ▌ ▌▗▀▖▌▐▞ ▙▘ ▐    ▐▐     ▐ ▐ ▐  ▌     ▖        ▐▐  ▞▌  █   █    ▗▄│
     │ ▌▐                ▌ · ▐▝▘·  ▌▗▘█  ▘ ·▌▐ ·▌ · ·▌ ▌▐·▌▌·▘ ▜·  ▌ · ▞▐   ▖ ▌  ▌▐ ·▌  ▟ ▐▌  ▗▌ ▟  ▐▝▖▞ ▐  ▌▌ ▗▜  ▖▗▘ │
     │ ▝▘                ▐ ▟ ▌     ▌▐ ▝   · ▌▐· ▚  · ▌·▌▐·█·  ·· ··▝▜ ·▌▐ ▗▀▌ ▌··▝▟· ▌ ▞▝▖▐▌  ▐▌ ▌▚ ▌ ▜   ▌·▌▚ ▐▐ ▐▝▌  │
     │    -               ▀ █    - ▐▐       ▌▐   ▚ ▖ ▌ ▐▐ █    ·    ▝▖▐ ▐·▐ ▐▗▘   ▝  ▐▗▘·▚▌▐··▌▌▐·▝▖▌· ·  ▚▗▘▐ ▐▝▖▐    │
     │   --  -  -           ▝   -- ▐▐       ▚▌    ▀▌ ▌ ▐▌ ▝          ▚▞ ▐·▐ ▝▟       ▐▐  ▝▌▐  ▌▐▐  ▚▌·  ··▐▐ ·▌▞ ▌▌· ··│
     │   -- - ---               -- ▐▌   -   ▐▌     ▌▐  ▐▌    -       ▝▌ ▐ ▐  █       ▐▌     ▌▐ ▐▌   ▘      █  ▚▌ ▌▌ ·  │
6.936┤   -- -  --     -     --- -- ▐▌--- -  ▐▌     ▐▐  ▐▌   -- -         ▌▌  ▝      - ▘     ▌▐ ▝▌          ▜  ▐▌ █     │
     │  -  --   -    - -   -  - -- ▐▌    -  ▐▌     ▐▐   ▘   --- -  - --- ▌▌   -   ---  -    ▝▘        -        ▘ █     │
     │  -  --    --  -  -- -   -- - ▘     --▐▌ --   █  -    - -  ----   -▌▌  --  --- ---     -       --          ▝     │
     │ --  --      ---   - -   -- --        -▘- -   █ -- ----     ---   -▙▘ - -  ---  --    --   -   -- -       -    --│
     │- -   -       --    -    -- --             -  ▝- --           -   -█  - -  ---    -   - - - -  ----  -   --   -  │
     │-              -         -- --             - - - --               -█--  -  - -     - -  - -  --  - --- --  -  -  │
6.884┤                         --  -              --   --               -█     --        - -  - -         ---    -  -  │
     │                         --                 --    -                ▝                --  - -          --     --   │
     │                         --                 --                                       -   --           -          │
     │                         --                  -                                           --                      │
     │                         --                                                              --                      │
     │                         --                                                               -                      │
6.832┤                          -                                                                                      │
     └┬───────────────────────────┬───────────────────────────┬───────────────────────────┬───────────────────────────┬┘
     1.0                        23.5                        46.0                        68.5                       91.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_loss_summary.txt
                      train_loss/mean hist                                         train_loss/max hist
    ┌──────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
21.0┤           █████                                      │22.0┤           █████                                      │
    │           █████                                      │    │           ███████████                                │
17.5┤     ███████████                                      │18.3┤           ███████████                                │
    │     ███████████                                      │    │           ███████████                                │
    │     █████████████████                                │    │           ███████████                                │
14.0┤     █████████████████                                │14.7┤           ███████████                                │
    │     █████████████████                                │    │     █████████████████                                │
10.5┤     ██████████████████████                           │11.0┤     ██████████████████████                           │
    │     ██████████████████████                           │    │     ██████████████████████                           │
 7.0┤     ██████████████████████                           │ 7.3┤     █████████████████████████████████                │
    │████████████████████████████████                      │    │     █████████████████████████████████                │
    │███████████████████████████████████████████           │    │     █████████████████████████████████                │
 3.5┤███████████████████████████████████████████      █████│ 3.7┤     █████████████████████████████████     ██████     │
    │██████████████████████████████████████████████████████│    │██████████████████████████████████████████████████████│
 0.0┤██████████████████████████████████████████████████████│ 0.0┤██████████████████████████████████████████████████████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
   6.945        6.970         6.995        7.019      7.044    6.984        7.026         7.067        7.109      7.151
                       train_loss/min hist                                        train_loss/std hist
    ┌──────────────────────────────────────────────────────┐  ┌────────────────────────────────────────────────────────┐
23.0┤                                ██████                │18┤                 ███████████                            │
    │                                ██████                │  │                 ███████████                            │
19.2┤                                ██████                │15┤                 █████████████████                      │
    │                           ███████████                │  │           ███████████████████████                      │
    │                           ████████████████           │  │           ███████████████████████                      │
15.3┤                           ████████████████           │12┤           ███████████████████████                      │
    │                      █████████████████████           │  │           ███████████████████████                      │
11.5┤                      █████████████████████           │ 9┤           ████████████████████████████                 │
    │                      █████████████████████           │  │           ████████████████████████████                 │
    │                      █████████████████████           │  │           ████████████████████████████                 │
 7.7┤                      █████████████████████           │ 6┤█████      ████████████████████████████                 │
    │                      █████████████████████           │  │█████      ████████████████████████████                 │
 3.8┤                ██████████████████████████████████████│ 3┤█████ █████████████████████████████████                 │
    │                ██████████████████████████████████████│  │█████ ████████████████████████████████████████████ █████│
    │██████████████████████████████████████████████████████│  │█████ ████████████████████████████████████████████ █████│
 0.0┤██████████████████████████████████████████████████████│ 0┤█████ ████████████████████████████████████████████ █████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘  └┬─────────────┬─────────────┬────────────┬─────────────┬┘
   6.826        6.862         6.898        6.935      6.971  0.0183       0.0270        0.0358       0.0445      0.0532
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/plots/tplot/train_loss_hist.txt
[2025-12-31 11:44:43,437983][W][ezpz/history:2320:save_dataset] Unable to save dataset to W&B, skipping!
[2025-12-31 11:44:43,439857][I][utils/__init__:651:dataset_to_h5pyfile] Saving dataset to: /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/train_dataset.h5
[2025-12-31 11:44:43,456174][I][ezpz/history:2433:finalize] Saving history report to /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-114142/2025-12-31-114437/report.md
[2025-12-31 11:44:43,461857][I][examples/vit:463:train_fn] dataset=<xarray.Dataset> Size: 26kB
Dimensions:          (draw: 91)
Coordinates:
  * draw             (draw) int64 728B 0 1 2 3 4 5 6 7 ... 84 85 86 87 88 89 90
Data variables: (12/35)
    train_iter       (draw) int64 728B 10 11 12 13 14 15 ... 95 96 97 98 99 100
    train_loss       (draw) float32 364B 7.123 6.976 6.983 ... 6.971 6.99 6.99
    train_dt         (draw) float64 728B 0.5535 0.5412 0.5621 ... 0.5465 0.5437
    train_dtd        (draw) float64 728B 0.002507 0.004021 ... 0.001735 0.003237
    train_dtf        (draw) float64 728B 0.04577 0.06221 ... 0.04814 0.04945
    train_dto        (draw) float64 728B 0.504 0.4732 0.4845 ... 0.4955 0.4897
    ...               ...
    train_dto_min    (draw) float64 728B 0.4985 0.4695 0.4772 ... 0.4871 0.4875
    train_dto_std    (draw) float64 728B 0.00217 0.001402 ... 0.002902 0.002604
    train_dtb_mean   (draw) float64 728B 0.001438 0.00138 ... 0.001278 0.001294
    train_dtb_max    (draw) float64 728B 0.001672 0.001963 ... 0.001364 0.001396
    train_dtb_min    (draw) float64 728B 0.001196 0.00115 ... 0.001174 0.001162
    train_dtb_std    (draw) float64 728B 0.0001324 0.0002622 ... 7.029e-05
[2025-12-31 11:44:43,525868][I][examples/vit:544:<module>] Took 180.57 seconds
wandb:
wandb: 🚀 View run dashing-water-238 at:
wandb: Find logs at: ../../../../../../lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20251231_114143-zxgz3u90/logs
[2025-12-31 11:44:46,233909][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2025-12-31-114446]----
[2025-12-31 11:44:46,234588][I][ezpz/launch:448:launch] Execution finished with 0.
[2025-12-31 11:44:46,234983][I][ezpz/launch:449:launch] Executing finished in 200.93 seconds.
[2025-12-31 11:44:46,235331][I][ezpz/launch:450:launch] Took 200.93 seconds to run. Exiting.
took: 3m 27s
```

</details>
