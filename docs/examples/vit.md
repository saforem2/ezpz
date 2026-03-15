# Train ViT with FSDP on MNIST

!!! info "Key API Functions"

    - [`setup_torch()`][ezpz.distributed.setup_torch] — Initialize distributed training
    - [`wrap_model()`][ezpz.distributed.wrap_model] — Wrap model for DDP / FSDP
    - [`ViTConfig`][ezpz.configs.ViTConfig] — Vision Transformer configuration
    - [`History`][ezpz.history.History] — Track training metrics

See:

- 📘 [examples/ViT](../python/Code-Reference/examples/vit.md)
- 🐍 [src/ezpz/examples/vit.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/vit.py)

```bash
ezpz launch python3 -m ezpz.examples.vit --compile # --fsdp
```

## Help

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

## Output

<details closed><summary>Output on Sunspot</summary>

```bash
$ ezpz launch python3 -m ezpz.examples.vit

[2025-12-31 12:13:01,324304][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2025-12-31-121301]----
[2025-12-31 12:13:02,176169][I][ezpz/launch:416:launch] Job ID: 12458339
[2025-12-31 12:13:02,176953][I][ezpz/launch:417:launch] nodelist: ['x1921c0s3b0n0', 'x1921c0s7b0n0']
[2025-12-31 12:13:02,177350][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
[2025-12-31 12:13:02,178010][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
[2025-12-31 12:13:02,178699][I][ezpz/launch:367:build_executable] Building command to execute by piecing together:
[2025-12-31 12:13:02,179082][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
[2025-12-31 12:13:02,179891][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: python3 -m ezpz.examples.vit
[2025-12-31 12:13:02,180622][I][ezpz/launch:433:launch] Took: 1.46 seconds to build command.
[2025-12-31 12:13:02,180965][I][ezpz/launch:436:launch] Executing:
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
[2025-12-31 12:13:02,182157][I][ezpz/launch:443:launch] Execution started @ 2025-12-31-121302...
[2025-12-31 12:13:02,182600][I][ezpz/launch:139:run_command] Running command:
 mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 python3 -m ezpz.examples.vit
cpubind:list x1921c0s7b0n0 pid 108722 rank 12 0: mask 0x1c
cpubind:list x1921c0s7b0n0 pid 108723 rank 13 1: mask 0x1c00
cpubind:list x1921c0s7b0n0 pid 108724 rank 14 2: mask 0x1c0000
cpubind:list x1921c0s7b0n0 pid 108725 rank 15 3: mask 0x1c000000
cpubind:list x1921c0s7b0n0 pid 108726 rank 16 4: mask 0x1c00000000
cpubind:list x1921c0s7b0n0 pid 108727 rank 17 5: mask 0x1c0000000000
cpubind:list x1921c0s7b0n0 pid 108728 rank 18 6: mask 0x1c0000000000000
cpubind:list x1921c0s7b0n0 pid 108729 rank 19 7: mask 0x1c000000000000000
cpubind:list x1921c0s7b0n0 pid 108730 rank 20 8: mask 0x1c00000000000000000
cpubind:list x1921c0s7b0n0 pid 108731 rank 21 9: mask 0x1c0000000000000000000
cpubind:list x1921c0s7b0n0 pid 108732 rank 22 10: mask 0x1c000000000000000000000
cpubind:list x1921c0s7b0n0 pid 108733 rank 23 11: mask 0x1c00000000000000000000000
cpubind:list x1921c0s3b0n0 pid 105486 rank 0 0: mask 0x1c
cpubind:list x1921c0s3b0n0 pid 105487 rank 1 1: mask 0x1c00
cpubind:list x1921c0s3b0n0 pid 105488 rank 2 2: mask 0x1c0000
cpubind:list x1921c0s3b0n0 pid 105489 rank 3 3: mask 0x1c000000
cpubind:list x1921c0s3b0n0 pid 105490 rank 4 4: mask 0x1c00000000
cpubind:list x1921c0s3b0n0 pid 105491 rank 5 5: mask 0x1c0000000000
cpubind:list x1921c0s3b0n0 pid 105492 rank 6 6: mask 0x1c0000000000000
cpubind:list x1921c0s3b0n0 pid 105493 rank 7 7: mask 0x1c000000000000000
cpubind:list x1921c0s3b0n0 pid 105494 rank 8 8: mask 0x1c00000000000000000
cpubind:list x1921c0s3b0n0 pid 105495 rank 9 9: mask 0x1c0000000000000000000
cpubind:list x1921c0s3b0n0 pid 105496 rank 10 10: mask 0x1c000000000000000000000
cpubind:list x1921c0s3b0n0 pid 105497 rank 11 11: mask 0x1c00000000000000000000000
[2025-12-31 12:13:08,706913][I][ezpz/dist:1501:setup_torch_distributed] Using torch_{device,backend}= {xpu, xccl}
[2025-12-31 12:13:08,709436][I][ezpz/dist:1366:setup_torch_DDP] Caught MASTER_PORT=45161 from environment!
[2025-12-31 12:13:08,710117][I][ezpz/dist:1382:setup_torch_DDP] Using torch.distributed.init_process_group with
- master_addr='x1921c0s3b0n0'
- master_port='45161'
- world_size=24
- rank=0
- local_rank=0
- timeout=datetime.timedelta(seconds=3600)
- backend='xccl'
[2025-12-31 12:13:08,711400][I][ezpz/dist:1014:init_process_group] Calling torch.distributed.init_process_group_with: rank=0 world_size=24 backend=xccl
[2025-12-31 12:13:09,470261][I][ezpz/dist:1727:setup_torch] Using device='xpu' with backend='xccl' + 'xccl' for distributed training.
[2025-12-31 12:13:09,471063][W][ezpz/dist:544:print_dist_setup] Using [24 / 24] available "xpu" devices !!
[2025-12-31 12:13:09,471499][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=00/23][local_rank=00/11]
[2025-12-31 12:13:09,470671][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=10/23][local_rank=10/11]
[2025-12-31 12:13:09,470709][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=01/23][local_rank=01/11]
[2025-12-31 12:13:09,470724][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=02/23][local_rank=02/11]
[2025-12-31 12:13:09,470717][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=03/23][local_rank=03/11]
[2025-12-31 12:13:09,470725][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=04/23][local_rank=04/11]
[2025-12-31 12:13:09,470729][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=05/23][local_rank=05/11]
[2025-12-31 12:13:09,470727][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=06/23][local_rank=06/11]
[2025-12-31 12:13:09,470702][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=07/23][local_rank=07/11]
[2025-12-31 12:13:09,470697][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=08/23][local_rank=08/11]
[2025-12-31 12:13:09,470703][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=09/23][local_rank=09/11]
[2025-12-31 12:13:09,470729][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=11/23][local_rank=11/11]
[2025-12-31 12:13:09,474499][I][ezpz/dist:2039:setup_wandb] Setting up wandb from rank=0
[2025-12-31 12:13:09,474926][I][ezpz/dist:2040:setup_wandb] Using WB_PROJECT=ezpz.examples.vit
[2025-12-31 12:13:09,470772][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=12/23][local_rank=00/11]
[2025-12-31 12:13:09,470811][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=13/23][local_rank=01/11]
[2025-12-31 12:13:09,470827][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=14/23][local_rank=02/11]
[2025-12-31 12:13:09,470866][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=15/23][local_rank=03/11]
[2025-12-31 12:13:09,470869][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=16/23][local_rank=04/11]
[2025-12-31 12:13:09,470813][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=17/23][local_rank=05/11]
[2025-12-31 12:13:09,470869][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=18/23][local_rank=06/11]
[2025-12-31 12:13:09,470871][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=19/23][local_rank=07/11]
[2025-12-31 12:13:09,470827][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=20/23][local_rank=08/11]
[2025-12-31 12:13:09,470825][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=21/23][local_rank=09/11]
[2025-12-31 12:13:09,470874][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=22/23][local_rank=10/11]
[2025-12-31 12:13:09,470870][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=23/23][local_rank=11/11]
wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.23.1
wandb: Run data is saved locally in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20251231_121309-g19jy6bl
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run snowy-hill-239
wandb:  View project at https://wandb.ai/aurora_gpt/ezpz.examples.vit
wandb:  View run at https://wandb.ai/aurora_gpt/ezpz.examples.vit/runs/g19jy6bl
[2025-12-31 12:13:10,974322][I][ezpz/dist:2069:setup_wandb] wandb.run=[snowy-hill-239](https://wandb.ai/aurora_gpt/ezpz.examples.vit/runs/g19jy6bl)
[2025-12-31 12:13:10,980450][I][ezpz/dist:2112:setup_wandb] Running on machine='SunSpot'
[2025-12-31 12:13:10,983391][I][examples/vit:509:main] Using native for SDPA backend
[2025-12-31 12:13:10,984013][I][examples/vit:535:main] Using AttentionBlock Attention with args.compile=False
[2025-12-31 12:13:10,984652][I][examples/vit:287:train_fn] asdict(config)={'img_size': 224, 'batch_size': 128, 'num_heads': 16, 'head_dim': 64, 'depth': 24, 'patch_size': 16, 'hidden_dim': 1024, 'mlp_dim': 4096, 'dropout': 0.0, 'attention_dropout': 0.0, 'num_classes': 1000}
[2025-12-31 12:14:34,029080][I][examples/vit:354:train_fn] 
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
[2025-12-31 12:14:34,032988][I][examples/vit:355:train_fn] Model size: nparams=0.91 B
[2025-12-31 12:14:34,038818][I][ezpz/dist:685:wrap_model] Wrapping model with: ddp
2025:12:31-12:14:34:(105486) |CCL_WARN| value of CCL_OP_SYNC changed to be 1 (default:0)
2025:12:31-12:14:34:(105486) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
[2025-12-31 12:14:47,099101][I][ezpz/dist:685:wrap_model] Wrapping model with: ddp
[2025-12-31 12:14:47,296113][I][ezpz/history:220:__init__] Using History with distributed_history=True
[2025-12-31 12:14:47,312293][I][examples/vit:408:train_fn] Training with 24 x xpu (s), using torch_dtype=torch.bfloat16
[2025-12-31 12:15:32,154227][I][examples/vit:445:train_fn] iter=10 loss=7.111572 dt=0.744444 dtd=0.003141 dtf=0.021592 dto=0.698994 dtb=0.020717 loss/mean=7.036184 loss/max=7.148438 loss/min=6.926270 loss/std=0.051675 dt/mean=0.744634 dt/max=0.745104 dt/min=0.744115 dt/std=0.000000 dtd/mean=0.003989 dtd/max=0.005223 dtd/min=0.003121 dtd/std=0.000751 dtf/mean=0.021139 dtf/max=0.021592 dtf/min=0.020827 dtf/std=0.000188 dto/mean=0.698727 dto/max=0.699881 dto/min=0.697217 dto/std=0.000961 dtb/mean=0.020778 dtb/max=0.021557 dtb/min=0.020275 dtb/std=0.000320
[2025-12-31 12:15:33,015905][I][examples/vit:445:train_fn] iter=11 loss=7.032715 dt=0.702058 dtd=0.001719 dtf=0.022411 dto=0.657195 dtb=0.020732 loss/mean=7.011394 loss/max=7.086426 loss/min=6.919678 loss/std=0.041570 dt/mean=0.735905 dt/max=0.771187 dt/min=0.697317 dt/std=0.022530 dtd/mean=0.001865 dtd/max=0.002308 dtd/min=0.001698 dtd/std=0.000162 dtf/mean=0.021999 dtf/max=0.023057 dtf/min=0.021225 dtf/std=0.000596 dto/mean=0.691333 dto/max=0.726180 dto/min=0.652583 dto/std=0.022813 dtb/mean=0.020708 dtb/max=0.021126 dtb/min=0.020190 dtb/std=0.000228
[2025-12-31 12:15:33,747367][I][examples/vit:445:train_fn] iter=12 loss=6.982422 dt=0.717778 dtd=0.001721 dtf=0.023286 dto=0.671980 dtb=0.020791 loss/mean=7.006734 loss/max=7.068848 loss/min=6.895752 loss/std=0.042117 dt/mean=0.724329 dt/max=0.732337 dt/min=0.713690 dt/std=0.004574 dtd/mean=0.003245 dtd/max=0.005390 dtd/min=0.001609 dtd/std=0.001376 dtf/mean=0.023302 dtf/max=0.024685 dtf/min=0.021369 dtf/std=0.000678 dto/mean=0.677032 dto/max=0.684881 dto/min=0.669961 dto/std=0.003860 dtb/mean=0.020750 dtb/max=0.021848 dtb/min=0.020185 dtb/std=0.000450
[2025-12-31 12:15:34,556097][I][examples/vit:445:train_fn] iter=13 loss=7.114746 dt=0.742885 dtd=0.001825 dtf=0.023194 dto=0.697149 dtb=0.020716 loss/mean=7.035777 loss/max=7.121094 loss/min=6.938965 loss/std=0.043673 dt/mean=0.743002 dt/max=0.746045 dt/min=0.739701 dt/std=0.002211 dtd/mean=0.004097 dtd/max=0.005476 dtd/min=0.001825 dtd/std=0.001211 dtf/mean=0.022114 dtf/max=0.023244 dtf/min=0.021464 dtf/std=0.000590 dto/mean=0.696058 dto/max=0.699107 dto/min=0.692656 dto/std=0.002238 dtb/mean=0.020733 dtb/max=0.021334 dtb/min=0.020149 dtb/std=0.000345
[2025-12-31 12:15:35,310688][I][examples/vit:445:train_fn] iter=14 loss=7.011475 dt=0.720220 dtd=0.001705 dtf=0.022270 dto=0.675471 dtb=0.020774 loss/mean=7.039348 loss/max=7.114502 loss/min=6.937744 loss/std=0.041890 dt/mean=0.750122 dt/max=0.777821 dt/min=0.720220 dt/std=0.018705 dtd/mean=0.001850 dtd/max=0.002233 dtd/min=0.001625 dtd/std=0.000194 dtf/mean=0.022314 dtf/max=0.024492 dtf/min=0.021074 dtf/std=0.000813 dto/mean=0.705263 dto/max=0.730777 dto/min=0.675471 dto/std=0.018371 dtb/mean=0.020695 dtb/max=0.021149 dtb/min=0.020171 dtb/std=0.000279
[2025-12-31 12:15:36,066943][I][examples/vit:445:train_fn] iter=15 loss=7.011230 dt=0.735597 dtd=0.001686 dtf=0.022349 dto=0.690830 dtb=0.020732 loss/mean=7.028381 loss/max=7.112061 loss/min=6.949463 loss/std=0.039451 dt/mean=0.751743 dt/max=0.761701 dt/min=0.735432 dt/std=0.008930 dtd/mean=0.002519 dtd/max=0.004533 dtd/min=0.001567 dtd/std=0.000977 dtf/mean=0.022890 dtf/max=0.024537 dtf/min=0.021450 dtf/std=0.000851 dto/mean=0.705617 dto/max=0.715304 dto/min=0.690162 dto/std=0.008198 dtb/mean=0.020716 dtb/max=0.021369 dtb/min=0.020218 dtb/std=0.000304
[2025-12-31 12:15:36,829552][I][examples/vit:445:train_fn] iter=16 loss=7.066895 dt=0.728091 dtd=0.001674 dtf=0.023043 dto=0.682651 dtb=0.020723 loss/mean=7.036306 loss/max=7.118652 loss/min=6.950928 loss/std=0.040736 dt/mean=0.735283 dt/max=0.739360 dt/min=0.728091 dt/std=0.003266 dtd/mean=0.003385 dtd/max=0.005231 dtd/min=0.001674 dtd/std=0.001218 dtf/mean=0.022158 dtf/max=0.023252 dtf/min=0.021710 dtf/std=0.000435 dto/mean=0.688973 dto/max=0.694191 dto/min=0.682651 dto/std=0.003334 dtb/mean=0.020767 dtb/max=0.021937 dtb/min=0.020278 dtb/std=0.000388
[2025-12-31 12:15:37,624977][I][examples/vit:445:train_fn] iter=17 loss=7.056885 dt=0.739866 dtd=0.001641 dtf=0.022526 dto=0.694913 dtb=0.020786 loss/mean=7.031006 loss/max=7.094238 loss/min=6.946289 loss/std=0.042612 dt/mean=0.747859 dt/max=0.765443 dt/min=0.731697 dt/std=0.012958 dtd/mean=0.002078 dtd/max=0.003025 dtd/min=0.001545 dtd/std=0.000425 dtf/mean=0.022239 dtf/max=0.023510 dtf/min=0.021278 dtf/std=0.000711 dto/mean=0.702795 dto/max=0.719565 dto/min=0.686761 dto/std=0.012629 dtb/mean=0.020747 dtb/max=0.021336 dtb/min=0.020204 dtb/std=0.000284
[2025-12-31 12:15:38,414813][I][examples/vit:445:train_fn] iter=18 loss=7.021973 dt=0.730126 dtd=0.001684 dtf=0.022472 dto=0.685149 dtb=0.020821 loss/mean=7.009206 loss/max=7.126221 loss/min=6.934570 loss/std=0.041062 dt/mean=0.757921 dt/max=0.776361 dt/min=0.729369 dt/std=0.015862 dtd/mean=0.002343 dtd/max=0.004546 dtd/min=0.001629 dtd/std=0.000985 dtf/mean=0.022413 dtf/max=0.023448 dtf/min=0.021651 dtf/std=0.000472 dto/mean=0.712472 dto/max=0.730281 dto/min=0.685116 dto/std=0.015107 dtb/mean=0.020693 dtb/max=0.021261 dtb/min=0.020149 dtb/std=0.000297
[2025-12-31 12:15:39,209452][I][examples/vit:445:train_fn] iter=19 loss=7.029053 dt=0.733423 dtd=0.001668 dtf=0.022388 dto=0.688487 dtb=0.020880 loss/mean=7.010173 loss/max=7.071777 loss/min=6.963379 loss/std=0.031614 dt/mean=0.757083 dt/max=0.784014 dt/min=0.728249 dt/std=0.018796 dtd/mean=0.001842 dtd/max=0.002232 dtd/min=0.001557 dtd/std=0.000210 dtf/mean=0.022281 dtf/max=0.023509 dtf/min=0.021485 dtf/std=0.000466 dto/mean=0.712357 dto/max=0.738712 dto/min=0.684110 dto/std=0.018800 dtb/mean=0.020603 dtb/max=0.021067 dtb/min=0.020249 dtb/std=0.000253
[2025-12-31 12:15:39,959071][I][examples/vit:445:train_fn] iter=20 loss=6.989258 dt=0.729999 dtd=0.001656 dtf=0.023220 dto=0.684263 dtb=0.020860 loss/mean=6.997142 loss/max=7.051025 loss/min=6.924805 loss/std=0.037109 dt/mean=0.736245 dt/max=0.744330 dt/min=0.723712 dt/std=0.006234 dtd/mean=0.002480 dtd/max=0.003943 dtd/min=0.001558 dtd/std=0.000659 dtf/mean=0.023104 dtf/max=0.024808 dtf/min=0.021568 dtf/std=0.000875 dto/mean=0.689984 dto/max=0.697561 dto/min=0.679231 dto/std=0.005511 dtb/mean=0.020676 dtb/max=0.021082 dtb/min=0.020291 dtb/std=0.000217
[2025-12-31 12:15:40,714336][I][examples/vit:445:train_fn] iter=21 loss=6.949463 dt=0.738463 dtd=0.001676 dtf=0.022789 dto=0.693238 dtb=0.020760 loss/mean=7.004436 loss/max=7.096680 loss/min=6.928467 loss/std=0.038073 dt/mean=0.744181 dt/max=0.747653 dt/min=0.738463 dt/std=0.002405 dtd/mean=0.003571 dtd/max=0.005418 dtd/min=0.001676 dtd/std=0.001278 dtf/mean=0.022411 dtf/max=0.023831 dtf/min=0.021547 dtf/std=0.000525 dto/mean=0.697513 dto/max=0.700923 dto/min=0.693238 dto/std=0.001938 dtb/mean=0.020686 dtb/max=0.021187 dtb/min=0.020283 dtb/std=0.000298
[2025-12-31 12:15:41,522736][I][examples/vit:445:train_fn] iter=22 loss=7.014893 dt=0.755023 dtd=0.001776 dtf=0.025431 dto=0.706979 dtb=0.020837 loss/mean=7.004486 loss/max=7.110352 loss/min=6.913818 loss/std=0.038965 dt/mean=0.748414 dt/max=0.766116 dt/min=0.732838 dt/std=0.012672 dtd/mean=0.002078 dtd/max=0.003655 dtd/min=0.001651 dtd/std=0.000579 dtf/mean=0.022676 dtf/max=0.025431 dtf/min=0.021047 dtf/std=0.001224 dto/mean=0.702984 dto/max=0.719493 dto/min=0.687241 dto/std=0.012053 dtb/mean=0.020675 dtb/max=0.021195 dtb/min=0.020129 dtb/std=0.000270
[2025-12-31 12:15:42,268381][I][examples/vit:445:train_fn] iter=23 loss=7.009033 dt=0.732504 dtd=0.001736 dtf=0.022508 dto=0.687491 dtb=0.020769 loss/mean=7.007579 loss/max=7.079834 loss/min=6.947510 loss/std=0.039306 dt/mean=0.757627 dt/max=0.778756 dt/min=0.731312 dt/std=0.016388 dtd/mean=0.002094 dtd/max=0.003084 dtd/min=0.001562 dtd/std=0.000526 dtf/mean=0.022488 dtf/max=0.023932 dtf/min=0.021376 dtf/std=0.000707 dto/mean=0.712397 dto/max=0.732798 dto/min=0.685533 dto/std=0.016056 dtb/mean=0.020648 dtb/max=0.021215 dtb/min=0.020275 dtb/std=0.000233
[2025-12-31 12:15:43,132486][I][examples/vit:445:train_fn] iter=24 loss=7.014648 dt=0.791617 dtd=0.001653 dtf=0.025728 dto=0.743375 dtb=0.020861 loss/mean=7.015188 loss/max=7.098389 loss/min=6.912109 loss/std=0.045387 dt/mean=0.763487 dt/max=0.791617 dt/min=0.728050 dt/std=0.020098 dtd/mean=0.001822 dtd/max=0.002207 dtd/min=0.001626 dtd/std=0.000180 dtf/mean=0.022584 dtf/max=0.025728 dtf/min=0.021624 dtf/std=0.000843 dto/mean=0.718427 dto/max=0.746834 dto/min=0.683279 dto/std=0.019673 dtb/mean=0.020654 dtb/max=0.021240 dtb/min=0.020345 dtb/std=0.000230
[2025-12-31 12:15:43,923470][I][examples/vit:445:train_fn] iter=25 loss=6.988525 dt=0.730475 dtd=0.005105 dtf=0.022559 dto=0.682103 dtb=0.020708 loss/mean=6.990112 loss/max=7.058594 loss/min=6.914795 loss/std=0.035373 dt/mean=0.745873 dt/max=0.765539 dt/min=0.722147 dt/std=0.014330 dtd/mean=0.002260 dtd/max=0.005105 dtd/min=0.001588 dtd/std=0.000836 dtf/mean=0.023081 dtf/max=0.025434 dtf/min=0.021319 dtf/std=0.001245 dto/mean=0.699837 dto/max=0.717573 dto/min=0.677176 dto/std=0.013419 dtb/mean=0.020695 dtb/max=0.021258 dtb/min=0.020226 dtb/std=0.000287
[2025-12-31 12:15:44,671377][I][examples/vit:445:train_fn] iter=26 loss=6.978271 dt=0.728732 dtd=0.001722 dtf=0.027302 dto=0.678941 dtb=0.020767 loss/mean=7.000173 loss/max=7.060303 loss/min=6.930664 loss/std=0.038023 dt/mean=0.751072 dt/max=0.780739 dt/min=0.718407 dt/std=0.018831 dtd/mean=0.001792 dtd/max=0.002193 dtd/min=0.001632 dtd/std=0.000166 dtf/mean=0.022540 dtf/max=0.027302 dtf/min=0.021307 dtf/std=0.001100 dto/mean=0.706061 dto/max=0.735676 dto/min=0.673876 dto/std=0.019266 dtb/mean=0.020680 dtb/max=0.021233 dtb/min=0.020343 dtb/std=0.000255
[2025-12-31 12:15:45,426093][I][examples/vit:445:train_fn] iter=27 loss=6.985840 dt=0.735329 dtd=0.001691 dtf=0.022562 dto=0.690344 dtb=0.020733 loss/mean=6.988373 loss/max=7.067383 loss/min=6.901123 loss/std=0.034054 dt/mean=0.742626 dt/max=0.748739 dt/min=0.735329 dt/std=0.003427 dtd/mean=0.003301 dtd/max=0.005561 dtd/min=0.001586 dtd/std=0.001388 dtf/mean=0.022639 dtf/max=0.024640 dtf/min=0.021827 dtf/std=0.000775 dto/mean=0.695893 dto/max=0.701744 dto/min=0.690344 dto/std=0.003605 dtb/mean=0.020792 dtb/max=0.021600 dtb/min=0.020282 dtb/std=0.000347
[2025-12-31 12:15:46,187495][I][examples/vit:445:train_fn] iter=28 loss=6.958496 dt=0.743998 dtd=0.001662 dtf=0.022803 dto=0.698705 dtb=0.020828 loss/mean=7.000173 loss/max=7.078369 loss/min=6.931885 loss/std=0.038867 dt/mean=0.746971 dt/max=0.760301 dt/min=0.736719 dt/std=0.009229 dtd/mean=0.002349 dtd/max=0.003210 dtd/min=0.001617 dtd/std=0.000579 dtf/mean=0.022645 dtf/max=0.024866 dtf/min=0.021274 dtf/std=0.001089 dto/mean=0.701287 dto/max=0.713176 dto/min=0.691328 dto/std=0.008573 dtb/mean=0.020691 dtb/max=0.021143 dtb/min=0.020316 dtb/std=0.000245
[2025-12-31 12:15:46,977725][I][examples/vit:445:train_fn] iter=29 loss=6.984131 dt=0.770773 dtd=0.001760 dtf=0.024091 dto=0.724120 dtb=0.020802 loss/mean=6.992188 loss/max=7.054688 loss/min=6.922852 loss/std=0.032682 dt/mean=0.756948 dt/max=0.776455 dt/min=0.732494 dt/std=0.015026 dtd/mean=0.001839 dtd/max=0.002294 dtd/min=0.001539 dtd/std=0.000199 dtf/mean=0.022533 dtf/max=0.024091 dtf/min=0.021525 dtf/std=0.000700 dto/mean=0.711877 dto/max=0.731058 dto/min=0.687968 dto/std=0.014665 dtb/mean=0.020698 dtb/max=0.021198 dtb/min=0.020312 dtb/std=0.000234
[2025-12-31 12:15:47,731430][I][examples/vit:445:train_fn] iter=30 loss=6.968750 dt=0.728727 dtd=0.001725 dtf=0.022213 dto=0.683936 dtb=0.020853 loss/mean=6.986908 loss/max=7.042236 loss/min=6.934814 loss/std=0.030321 dt/mean=0.738177 dt/max=0.744469 dt/min=0.728727 dt/std=0.003020 dtd/mean=0.003073 dtd/max=0.004729 dtd/min=0.001595 dtd/std=0.000956 dtf/mean=0.022537 dtf/max=0.024266 dtf/min=0.021735 dtf/std=0.000633 dto/mean=0.691848 dto/max=0.697793 dto/min=0.683936 dto/std=0.002686 dtb/mean=0.020719 dtb/max=0.021254 dtb/min=0.020223 dtb/std=0.000282
[2025-12-31 12:15:48,481466][I][examples/vit:445:train_fn] iter=31 loss=7.029053 dt=0.736371 dtd=0.001709 dtf=0.022263 dto=0.691483 dtb=0.020916 loss/mean=6.982574 loss/max=7.067383 loss/min=6.832764 loss/std=0.049333 dt/mean=0.745070 dt/max=0.751480 dt/min=0.736371 dt/std=0.003513 dtd/mean=0.003256 dtd/max=0.005504 dtd/min=0.001579 dtd/std=0.001344 dtf/mean=0.022521 dtf/max=0.023703 dtf/min=0.021750 dtf/std=0.000556 dto/mean=0.698558 dto/max=0.705201 dto/min=0.691483 dto/std=0.003774 dtb/mean=0.020735 dtb/max=0.021304 dtb/min=0.020201 dtb/std=0.000310
[2025-12-31 12:15:49,251607][I][examples/vit:445:train_fn] iter=32 loss=7.045166 dt=0.749095 dtd=0.001659 dtf=0.023011 dto=0.703600 dtb=0.020825 loss/mean=7.009898 loss/max=7.066162 loss/min=6.954834 loss/std=0.026349 dt/mean=0.745355 dt/max=0.756215 dt/min=0.736894 dt/std=0.007296 dtd/mean=0.002465 dtd/max=0.004041 dtd/min=0.001549 dtd/std=0.000795 dtf/mean=0.022334 dtf/max=0.023693 dtf/min=0.021286 dtf/std=0.000763 dto/mean=0.699811 dto/max=0.710788 dto/min=0.691312 dto/std=0.007118 dtb/mean=0.020746 dtb/max=0.021371 dtb/min=0.020353 dtb/std=0.000265
[2025-12-31 12:15:49,997199][I][examples/vit:445:train_fn] iter=33 loss=7.020264 dt=0.727653 dtd=0.001750 dtf=0.022298 dto=0.682872 dtb=0.020734 loss/mean=6.983205 loss/max=7.062500 loss/min=6.895508 loss/std=0.035641 dt/mean=0.738254 dt/max=0.745029 dt/min=0.727653 dt/std=0.003339 dtd/mean=0.003172 dtd/max=0.004648 dtd/min=0.001657 dtd/std=0.000887 dtf/mean=0.022837 dtf/max=0.024299 dtf/min=0.022065 dtf/std=0.000576 dto/mean=0.691531 dto/max=0.697682 dto/min=0.682872 dto/std=0.002794 dtb/mean=0.020714 dtb/max=0.021391 dtb/min=0.020169 dtb/std=0.000318
[2025-12-31 12:15:50,759356][I][examples/vit:445:train_fn] iter=34 loss=6.914062 dt=0.741301 dtd=0.001698 dtf=0.024205 dto=0.694543 dtb=0.020855 loss/mean=6.980601 loss/max=7.046875 loss/min=6.911621 loss/std=0.036172 dt/mean=0.742607 dt/max=0.749077 dt/min=0.736232 dt/std=0.003409 dtd/mean=0.003402 dtd/max=0.005554 dtd/min=0.001676 dtd/std=0.001323 dtf/mean=0.022480 dtf/max=0.024390 dtf/min=0.021743 dtf/std=0.000859 dto/mean=0.695940 dto/max=0.702199 dto/min=0.691508 dto/std=0.003538 dtb/mean=0.020785 dtb/max=0.021545 dtb/min=0.020341 dtb/std=0.000307
[2025-12-31 12:15:51,512264][I][examples/vit:445:train_fn] iter=35 loss=6.983398 dt=0.734306 dtd=0.001671 dtf=0.022545 dto=0.689456 dtb=0.020635 loss/mean=6.997111 loss/max=7.062988 loss/min=6.936035 loss/std=0.028505 dt/mean=0.742718 dt/max=0.749095 dt/min=0.734306 dt/std=0.003453 dtd/mean=0.003206 dtd/max=0.005422 dtd/min=0.001573 dtd/std=0.001368 dtf/mean=0.022523 dtf/max=0.024025 dtf/min=0.021785 dtf/std=0.000639 dto/mean=0.696285 dto/max=0.702540 dto/min=0.689456 dto/std=0.003634 dtb/mean=0.020704 dtb/max=0.021270 dtb/min=0.020327 dtb/std=0.000209
[2025-12-31 12:15:52,257198][I][examples/vit:445:train_fn] iter=36 loss=6.991455 dt=0.735148 dtd=0.001708 dtf=0.022518 dto=0.690147 dtb=0.020774 loss/mean=6.995128 loss/max=7.067383 loss/min=6.942139 loss/std=0.032034 dt/mean=0.741631 dt/max=0.747561 dt/min=0.735148 dt/std=0.003155 dtd/mean=0.003153 dtd/max=0.005476 dtd/min=0.001570 dtd/std=0.001396 dtf/mean=0.022958 dtf/max=0.024346 dtf/min=0.021980 dtf/std=0.000610 dto/mean=0.694814 dto/max=0.700863 dto/min=0.690147 dto/std=0.003289 dtb/mean=0.020706 dtb/max=0.021249 dtb/min=0.020209 dtb/std=0.000304
[2025-12-31 12:15:53,082889][I][examples/vit:445:train_fn] iter=37 loss=7.010742 dt=0.754607 dtd=0.001597 dtf=0.024032 dto=0.708152 dtb=0.020825 loss/mean=6.982076 loss/max=7.057861 loss/min=6.918457 loss/std=0.034110 dt/mean=0.746617 dt/max=0.757576 dt/min=0.737717 dt/std=0.007612 dtd/mean=0.002758 dtd/max=0.005157 dtd/min=0.001536 dtd/std=0.001230 dtf/mean=0.022482 dtf/max=0.024220 dtf/min=0.021333 dtf/std=0.000915 dto/mean=0.700606 dto/max=0.711254 dto/min=0.690800 dto/std=0.007668 dtb/mean=0.020771 dtb/max=0.021395 dtb/min=0.020302 dtb/std=0.000313
[2025-12-31 12:15:53,822050][I][examples/vit:445:train_fn] iter=38 loss=6.975830 dt=0.725656 dtd=0.001658 dtf=0.025023 dto=0.678155 dtb=0.020820 loss/mean=6.994375 loss/max=7.053711 loss/min=6.947266 loss/std=0.024550 dt/mean=0.753547 dt/max=0.784940 dt/min=0.718371 dt/std=0.020758 dtd/mean=0.001801 dtd/max=0.002128 dtd/min=0.001635 dtd/std=0.000166 dtf/mean=0.022542 dtf/max=0.025023 dtf/min=0.021558 dtf/std=0.000725 dto/mean=0.708533 dto/max=0.739468 dto/min=0.674357 dto/std=0.020838 dtb/mean=0.020671 dtb/max=0.021199 dtb/min=0.020265 dtb/std=0.000210
[2025-12-31 12:15:54,657351][I][examples/vit:445:train_fn] iter=39 loss=6.992188 dt=0.764405 dtd=0.001675 dtf=0.025172 dto=0.716769 dtb=0.020789 loss/mean=6.970714 loss/max=7.027100 loss/min=6.914795 loss/std=0.034333 dt/mean=0.754408 dt/max=0.770382 dt/min=0.729280 dt/std=0.013428 dtd/mean=0.002184 dtd/max=0.003573 dtd/min=0.001631 dtd/std=0.000654 dtf/mean=0.022970 dtf/max=0.025172 dtf/min=0.021572 dtf/std=0.001008 dto/mean=0.708547 dto/max=0.723172 dto/min=0.684613 dto/std=0.012350 dtb/mean=0.020707 dtb/max=0.021262 dtb/min=0.020282 dtb/std=0.000285
[2025-12-31 12:15:55,440080][I][examples/vit:445:train_fn] iter=40 loss=7.038330 dt=0.733582 dtd=0.001704 dtf=0.024849 dto=0.686207 dtb=0.020821 loss/mean=6.987508 loss/max=7.041504 loss/min=6.928223 loss/std=0.026493 dt/mean=0.760830 dt/max=0.792707 dt/min=0.725636 dt/std=0.020782 dtd/mean=0.001795 dtd/max=0.002238 dtd/min=0.001635 dtd/std=0.000158 dtf/mean=0.022468 dtf/max=0.024849 dtf/min=0.021375 dtf/std=0.000751 dto/mean=0.715907 dto/max=0.747222 dto/min=0.682017 dto/std=0.020906 dtb/mean=0.020659 dtb/max=0.021324 dtb/min=0.020248 dtb/std=0.000278
[2025-12-31 12:15:56,186584][I][examples/vit:445:train_fn] iter=41 loss=6.945068 dt=0.727525 dtd=0.001735 dtf=0.022900 dto=0.682011 dtb=0.020879 loss/mean=6.969208 loss/max=7.058594 loss/min=6.916992 loss/std=0.032973 dt/mean=0.748452 dt/max=0.770154 dt/min=0.727149 dt/std=0.015412 dtd/mean=0.002192 dtd/max=0.003208 dtd/min=0.001573 dtd/std=0.000546 dtf/mean=0.022726 dtf/max=0.024180 dtf/min=0.021340 dtf/std=0.000864 dto/mean=0.702787 dto/max=0.723392 dto/min=0.681505 dto/std=0.014915 dtb/mean=0.020747 dtb/max=0.021484 dtb/min=0.020331 dtb/std=0.000298
[2025-12-31 12:15:57,002317][I][examples/vit:445:train_fn] iter=42 loss=7.038574 dt=0.776704 dtd=0.001668 dtf=0.022678 dto=0.731496 dtb=0.020862 loss/mean=6.976430 loss/max=7.039307 loss/min=6.919434 loss/std=0.034110 dt/mean=0.756732 dt/max=0.786670 dt/min=0.723782 dt/std=0.018553 dtd/mean=0.001802 dtd/max=0.002211 dtd/min=0.001626 dtd/std=0.000166 dtf/mean=0.022396 dtf/max=0.023093 dtf/min=0.021225 dtf/std=0.000533 dto/mean=0.711839 dto/max=0.741265 dto/min=0.678716 dto/std=0.018654 dtb/mean=0.020695 dtb/max=0.021359 dtb/min=0.020267 dtb/std=0.000279
[2025-12-31 12:15:57,815591][I][examples/vit:445:train_fn] iter=43 loss=7.035645 dt=0.758882 dtd=0.012152 dtf=0.022963 dto=0.702949 dtb=0.020818 loss/mean=6.980540 loss/max=7.037842 loss/min=6.929443 loss/std=0.032973 dt/mean=0.765730 dt/max=0.788675 dt/min=0.732798 dt/std=0.018196 dtd/mean=0.002373 dtd/max=0.012152 dtd/min=0.001639 dtd/std=0.002068 dtf/mean=0.022709 dtf/max=0.024687 dtf/min=0.021031 dtf/std=0.000978 dto/mean=0.719959 dto/max=0.741003 dto/min=0.687917 dto/std=0.017812 dtb/mean=0.020690 dtb/max=0.021201 dtb/min=0.020271 dtb/std=0.000228
[2025-12-31 12:15:58,555728][I][examples/vit:445:train_fn] iter=44 loss=6.953613 dt=0.719717 dtd=0.001672 dtf=0.022583 dto=0.674526 dtb=0.020936 loss/mean=6.987224 loss/max=7.059814 loss/min=6.906738 loss/std=0.037973 dt/mean=0.744013 dt/max=0.769423 dt/min=0.719717 dt/std=0.016983 dtd/mean=0.001891 dtd/max=0.002867 dtd/min=0.001532 dtd/std=0.000326 dtf/mean=0.022596 dtf/max=0.025096 dtf/min=0.021569 dtf/std=0.000788 dto/mean=0.698715 dto/max=0.723603 dto/min=0.674526 dto/std=0.016540 dtb/mean=0.020811 dtb/max=0.021485 dtb/min=0.020209 dtb/std=0.000297
[2025-12-31 12:15:59,320244][I][examples/vit:445:train_fn] iter=45 loss=6.965576 dt=0.736210 dtd=0.001692 dtf=0.024698 dto=0.689147 dtb=0.020674 loss/mean=6.996837 loss/max=7.056885 loss/min=6.898682 loss/std=0.037109 dt/mean=0.744331 dt/max=0.750168 dt/min=0.736210 dt/std=0.003461 dtd/mean=0.003146 dtd/max=0.005443 dtd/min=0.001592 dtd/std=0.001397 dtf/mean=0.022662 dtf/max=0.024698 dtf/min=0.021666 dtf/std=0.000771 dto/mean=0.697826 dto/max=0.704015 dto/min=0.689147 dto/std=0.003852 dtb/mean=0.020697 dtb/max=0.021296 dtb/min=0.020216 dtb/std=0.000292
[2025-12-31 12:16:00,081771][I][examples/vit:445:train_fn] iter=46 loss=6.968750 dt=0.741624 dtd=0.001717 dtf=0.023044 dto=0.696055 dtb=0.020807 loss/mean=6.971436 loss/max=7.033203 loss/min=6.863525 loss/std=0.034499 dt/mean=0.747135 dt/max=0.763320 dt/min=0.732857 dt/std=0.011744 dtd/mean=0.002163 dtd/max=0.003133 dtd/min=0.001573 dtd/std=0.000460 dtf/mean=0.022883 dtf/max=0.025204 dtf/min=0.021264 dtf/std=0.001195 dto/mean=0.701406 dto/max=0.716596 dto/min=0.688054 dto/std=0.011067 dtb/mean=0.020684 dtb/max=0.021180 dtb/min=0.020288 dtb/std=0.000252
[2025-12-31 12:16:00,841569][I][examples/vit:445:train_fn] iter=47 loss=6.917969 dt=0.734437 dtd=0.001652 dtf=0.023252 dto=0.688742 dtb=0.020790 loss/mean=6.979259 loss/max=7.023926 loss/min=6.917969 loss/std=0.027274 dt/mean=0.740831 dt/max=0.746801 dt/min=0.734437 dt/std=0.002858 dtd/mean=0.003643 dtd/max=0.005405 dtd/min=0.001531 dtd/std=0.001477 dtf/mean=0.022557 dtf/max=0.024191 dtf/min=0.021717 dtf/std=0.000603 dto/mean=0.693941 dto/max=0.701646 dto/min=0.688742 dto/std=0.002652 dtb/mean=0.020690 dtb/max=0.021128 dtb/min=0.020172 dtb/std=0.000248
[2025-12-31 12:16:01,595278][I][examples/vit:445:train_fn] iter=48 loss=6.997559 dt=0.732685 dtd=0.001657 dtf=0.022673 dto=0.687710 dtb=0.020645 loss/mean=6.980276 loss/max=7.027344 loss/min=6.907715 loss/std=0.035048 dt/mean=0.745125 dt/max=0.751038 dt/min=0.732685 dt/std=0.004158 dtd/mean=0.002830 dtd/max=0.004684 dtd/min=0.001657 dtd/std=0.001000 dtf/mean=0.022955 dtf/max=0.024705 dtf/min=0.022187 dtf/std=0.000669 dto/mean=0.698627 dto/max=0.705444 dto/min=0.687710 dto/std=0.004085 dtb/mean=0.020713 dtb/max=0.021169 dtb/min=0.020320 dtb/std=0.000278
[2025-12-31 12:16:02,343776][I][examples/vit:445:train_fn] iter=49 loss=6.997070 dt=0.729157 dtd=0.001664 dtf=0.023304 dto=0.683426 dtb=0.020763 loss/mean=6.977102 loss/max=7.046875 loss/min=6.920898 loss/std=0.028033 dt/mean=0.738032 dt/max=0.743651 dt/min=0.729157 dt/std=0.003392 dtd/mean=0.003257 dtd/max=0.005523 dtd/min=0.001553 dtd/std=0.001387 dtf/mean=0.022512 dtf/max=0.023798 dtf/min=0.021764 dtf/std=0.000579 dto/mean=0.691580 dto/max=0.697807 dto/min=0.683426 dto/std=0.003778 dtb/mean=0.020684 dtb/max=0.021177 dtb/min=0.020273 dtb/std=0.000231
[2025-12-31 12:16:03,087624][I][examples/vit:445:train_fn] iter=50 loss=6.925293 dt=0.730377 dtd=0.001676 dtf=0.022686 dto=0.684904 dtb=0.021110 loss/mean=6.973846 loss/max=7.045166 loss/min=6.856689 loss/std=0.040075 dt/mean=0.737219 dt/max=0.743345 dt/min=0.730377 dt/std=0.003285 dtd/mean=0.003267 dtd/max=0.005473 dtd/min=0.001551 dtd/std=0.001356 dtf/mean=0.022508 dtf/max=0.023485 dtf/min=0.021924 dtf/std=0.000446 dto/mean=0.690625 dto/max=0.697362 dto/min=0.684904 dto/std=0.003584 dtb/mean=0.020819 dtb/max=0.021817 dtb/min=0.020345 dtb/std=0.000324
[2025-12-31 12:16:03,908992][I][examples/vit:445:train_fn] iter=51 loss=7.015381 dt=0.749356 dtd=0.001659 dtf=0.024072 dto=0.702847 dtb=0.020778 loss/mean=6.979340 loss/max=7.034912 loss/min=6.907959 loss/std=0.035908 dt/mean=0.744960 dt/max=0.756240 dt/min=0.735700 dt/std=0.007744 dtd/mean=0.002572 dtd/max=0.004506 dtd/min=0.001551 dtd/std=0.000984 dtf/mean=0.022622 dtf/max=0.024409 dtf/min=0.021383 dtf/std=0.001038 dto/mean=0.699012 dto/max=0.709381 dto/min=0.689005 dto/std=0.007465 dtb/mean=0.020755 dtb/max=0.021618 dtb/min=0.020256 dtb/std=0.000345
[2025-12-31 12:16:04,680315][I][examples/vit:445:train_fn] iter=52 loss=6.945557 dt=0.718842 dtd=0.001774 dtf=0.022528 dto=0.673614 dtb=0.020926 loss/mean=6.967998 loss/max=7.070068 loss/min=6.917236 loss/std=0.039451 dt/mean=0.752761 dt/max=0.781505 dt/min=0.718842 dt/std=0.019747 dtd/mean=0.001806 dtd/max=0.002223 dtd/min=0.001580 dtd/std=0.000170 dtf/mean=0.022435 dtf/max=0.023659 dtf/min=0.021713 dtf/std=0.000526 dto/mean=0.707762 dto/max=0.735720 dto/min=0.673614 dto/std=0.019645 dtb/mean=0.020759 dtb/max=0.021476 dtb/min=0.020367 dtb/std=0.000264
[2025-12-31 12:16:05,486745][I][examples/vit:445:train_fn] iter=53 loss=7.022705 dt=0.726803 dtd=0.001651 dtf=0.022481 dto=0.681990 dtb=0.020681 loss/mean=6.981262 loss/max=7.050781 loss/min=6.899170 loss/std=0.040407 dt/mean=0.752933 dt/max=0.773121 dt/min=0.726373 dt/std=0.016582 dtd/mean=0.001963 dtd/max=0.003030 dtd/min=0.001545 dtd/std=0.000437 dtf/mean=0.022516 dtf/max=0.024074 dtf/min=0.021568 dtf/std=0.000693 dto/mean=0.707640 dto/max=0.726559 dto/min=0.680444 dto/std=0.016295 dtb/mean=0.020813 dtb/max=0.021907 dtb/min=0.020303 dtb/std=0.000374
[2025-12-31 12:16:06,253200][I][examples/vit:445:train_fn] iter=54 loss=6.986328 dt=0.714074 dtd=0.001667 dtf=0.022484 dto=0.669063 dtb=0.020860 loss/mean=6.987539 loss/max=7.055664 loss/min=6.936035 loss/std=0.028705 dt/mean=0.751958 dt/max=0.784207 dt/min=0.714074 dt/std=0.020769 dtd/mean=0.001780 dtd/max=0.002099 dtd/min=0.001626 dtd/std=0.000151 dtf/mean=0.022335 dtf/max=0.023637 dtf/min=0.021591 dtf/std=0.000496 dto/mean=0.707091 dto/max=0.738269 dto/min=0.669063 dto/std=0.020846 dtb/mean=0.020752 dtb/max=0.021352 dtb/min=0.020242 dtb/std=0.000278
[2025-12-31 12:16:07,036859][I][examples/vit:445:train_fn] iter=55 loss=7.020752 dt=0.726670 dtd=0.001671 dtf=0.023300 dto=0.680836 dtb=0.020863 loss/mean=6.975759 loss/max=7.020752 loss/min=6.936035 loss/std=0.021573 dt/mean=0.748641 dt/max=0.772063 dt/min=0.725118 dt/std=0.017210 dtd/mean=0.002003 dtd/max=0.003673 dtd/min=0.001546 dtd/std=0.000579 dtf/mean=0.022822 dtf/max=0.024702 dtf/min=0.021458 dtf/std=0.000904 dto/mean=0.703149 dto/max=0.725319 dto/min=0.678733 dto/std=0.017051 dtb/mean=0.020667 dtb/max=0.021340 dtb/min=0.020243 dtb/std=0.000276
[2025-12-31 12:16:07,827772][I][examples/vit:445:train_fn] iter=56 loss=6.957275 dt=0.730915 dtd=0.001995 dtf=0.022926 dto=0.685141 dtb=0.020853 loss/mean=6.961477 loss/max=7.016602 loss/min=6.924561 loss/std=0.020670 dt/mean=0.757991 dt/max=0.778522 dt/min=0.730217 dt/std=0.017030 dtd/mean=0.002200 dtd/max=0.003252 dtd/min=0.001647 dtd/std=0.000499 dtf/mean=0.022664 dtf/max=0.023615 dtf/min=0.021357 dtf/std=0.000580 dto/mean=0.712382 dto/max=0.732616 dto/min=0.684696 dto/std=0.016824 dtb/mean=0.020746 dtb/max=0.021194 dtb/min=0.020332 dtb/std=0.000239
[2025-12-31 12:16:08,579242][I][examples/vit:445:train_fn] iter=57 loss=7.012207 dt=0.737117 dtd=0.001701 dtf=0.023760 dto=0.690935 dtb=0.020722 loss/mean=6.988149 loss/max=7.083252 loss/min=6.920166 loss/std=0.034222 dt/mean=0.753169 dt/max=0.786585 dt/min=0.716214 dt/std=0.021419 dtd/mean=0.001782 dtd/max=0.002007 dtd/min=0.001624 dtd/std=0.000127 dtf/mean=0.022379 dtf/max=0.023760 dtf/min=0.021507 dtf/std=0.000557 dto/mean=0.708152 dto/max=0.740705 dto/min=0.672085 dto/std=0.021671 dtb/mean=0.020857 dtb/max=0.021699 dtb/min=0.020344 dtb/std=0.000374
[2025-12-31 12:16:09,362010][I][examples/vit:445:train_fn] iter=58 loss=6.983643 dt=0.763087 dtd=0.001641 dtf=0.024827 dto=0.715777 dtb=0.020842 loss/mean=6.971110 loss/max=7.022461 loss/min=6.918945 loss/std=0.027204 dt/mean=0.748961 dt/max=0.769213 dt/min=0.724905 dt/std=0.015968 dtd/mean=0.001924 dtd/max=0.002593 dtd/min=0.001557 dtd/std=0.000297 dtf/mean=0.022521 dtf/max=0.024827 dtf/min=0.021563 dtf/std=0.000732 dto/mean=0.703753 dto/max=0.723593 dto/min=0.680019 dto/std=0.015505 dtb/mean=0.020763 dtb/max=0.021321 dtb/min=0.020258 dtb/std=0.000261
[2025-12-31 12:16:10,113690][I][examples/vit:445:train_fn] iter=59 loss=6.993652 dt=0.732712 dtd=0.001718 dtf=0.022345 dto=0.687850 dtb=0.020799 loss/mean=6.973938 loss/max=7.031738 loss/min=6.913330 loss/std=0.033942 dt/mean=0.740580 dt/max=0.746455 dt/min=0.732712 dt/std=0.003285 dtd/mean=0.003164 dtd/max=0.005442 dtd/min=0.001579 dtd/std=0.001364 dtf/mean=0.022843 dtf/max=0.024134 dtf/min=0.021971 dtf/std=0.000731 dto/mean=0.693711 dto/max=0.699074 dto/min=0.687850 dto/std=0.003418 dtb/mean=0.020862 dtb/max=0.021546 dtb/min=0.020358 dtb/std=0.000337
[2025-12-31 12:16:10,896999][I][examples/vit:445:train_fn] iter=60 loss=6.969971 dt=0.747576 dtd=0.001661 dtf=0.023705 dto=0.701328 dtb=0.020880 loss/mean=6.962677 loss/max=7.035400 loss/min=6.872803 loss/std=0.038965 dt/mean=0.746533 dt/max=0.760261 dt/min=0.733879 dt/std=0.009983 dtd/mean=0.002699 dtd/max=0.004842 dtd/min=0.001575 dtd/std=0.001212 dtf/mean=0.022815 dtf/max=0.025205 dtf/min=0.021562 dtf/std=0.001125 dto/mean=0.700239 dto/max=0.713503 dto/min=0.687099 dto/std=0.009887 dtb/mean=0.020781 dtb/max=0.021246 dtb/min=0.020256 dtb/std=0.000274
[2025-12-31 12:16:11,665090][I][examples/vit:445:train_fn] iter=61 loss=6.933838 dt=0.748712 dtd=0.001657 dtf=0.023562 dto=0.702590 dtb=0.020903 loss/mean=6.976868 loss/max=7.038330 loss/min=6.912598 loss/std=0.033942 dt/mean=0.748078 dt/max=0.776145 dt/min=0.718618 dt/std=0.019553 dtd/mean=0.001892 dtd/max=0.002529 dtd/min=0.001623 dtd/std=0.000261 dtf/mean=0.022607 dtf/max=0.024323 dtf/min=0.021660 dtf/std=0.000795 dto/mean=0.702790 dto/max=0.728456 dto/min=0.674124 dto/std=0.019070 dtb/mean=0.020790 dtb/max=0.021661 dtb/min=0.020219 dtb/std=0.000346
[2025-12-31 12:16:12,412241][I][examples/vit:445:train_fn] iter=62 loss=6.958252 dt=0.732601 dtd=0.001710 dtf=0.022547 dto=0.687455 dtb=0.020889 loss/mean=6.965841 loss/max=7.051514 loss/min=6.913818 loss/std=0.035426 dt/mean=0.739734 dt/max=0.745430 dt/min=0.732601 dt/std=0.003257 dtd/mean=0.003280 dtd/max=0.005450 dtd/min=0.001559 dtd/std=0.001292 dtf/mean=0.022775 dtf/max=0.024696 dtf/min=0.021823 dtf/std=0.000722 dto/mean=0.692936 dto/max=0.698621 dto/min=0.687455 dto/std=0.003339 dtb/mean=0.020744 dtb/max=0.021340 dtb/min=0.020305 dtb/std=0.000279
[2025-12-31 12:16:13,174938][I][examples/vit:445:train_fn] iter=63 loss=7.009521 dt=0.740339 dtd=0.001691 dtf=0.023941 dto=0.693831 dtb=0.020876 loss/mean=6.973592 loss/max=7.028320 loss/min=6.924316 loss/std=0.029232 dt/mean=0.739645 dt/max=0.748818 dt/min=0.733537 dt/std=0.005658 dtd/mean=0.003032 dtd/max=0.005469 dtd/min=0.001573 dtd/std=0.001450 dtf/mean=0.022828 dtf/max=0.024976 dtf/min=0.021450 dtf/std=0.001211 dto/mean=0.692919 dto/max=0.702017 dto/min=0.685927 dto/std=0.005692 dtb/mean=0.020867 dtb/max=0.022359 dtb/min=0.020390 dtb/std=0.000440
[2025-12-31 12:16:13,919647][I][examples/vit:445:train_fn] iter=64 loss=6.896729 dt=0.731400 dtd=0.001672 dtf=0.022233 dto=0.686662 dtb=0.020833 loss/mean=6.962646 loss/max=7.011475 loss/min=6.896729 loss/std=0.027964 dt/mean=0.737714 dt/max=0.741256 dt/min=0.731400 dt/std=0.002549 dtd/mean=0.003417 dtd/max=0.005453 dtd/min=0.001575 dtd/std=0.001243 dtf/mean=0.022408 dtf/max=0.023186 dtf/min=0.021953 dtf/std=0.000294 dto/mean=0.691057 dto/max=0.695397 dto/min=0.686662 dto/std=0.002354 dtb/mean=0.020831 dtb/max=0.021627 dtb/min=0.020358 dtb/std=0.000299
[2025-12-31 12:16:14,724718][I][examples/vit:445:train_fn] iter=65 loss=6.983154 dt=0.740475 dtd=0.001657 dtf=0.026268 dto=0.691680 dtb=0.020869 loss/mean=6.979614 loss/max=7.029297 loss/min=6.906494 loss/std=0.025466 dt/mean=0.737195 dt/max=0.748038 dt/min=0.728805 dt/std=0.006987 dtd/mean=0.002650 dtd/max=0.005013 dtd/min=0.001557 dtd/std=0.001223 dtf/mean=0.022836 dtf/max=0.026268 dtf/min=0.021512 dtf/std=0.001202 dto/mean=0.690811 dto/max=0.701612 dto/min=0.682157 dto/std=0.007034 dtb/mean=0.020898 dtb/max=0.022102 dtb/min=0.020411 dtb/std=0.000431
[2025-12-31 12:16:15,460588][I][examples/vit:445:train_fn] iter=66 loss=6.993408 dt=0.717219 dtd=0.001660 dtf=0.022625 dto=0.672191 dtb=0.020743 loss/mean=6.968008 loss/max=7.035645 loss/min=6.893066 loss/std=0.032624 dt/mean=0.751308 dt/max=0.772520 dt/min=0.717219 dt/std=0.019154 dtd/mean=0.001972 dtd/max=0.002954 dtd/min=0.001552 dtd/std=0.000424 dtf/mean=0.022521 dtf/max=0.023821 dtf/min=0.021679 dtf/std=0.000597 dto/mean=0.705970 dto/max=0.727169 dto/min=0.672191 dto/std=0.018948 dtb/mean=0.020846 dtb/max=0.021620 dtb/min=0.020417 dtb/std=0.000325
[2025-12-31 12:16:16,284567][I][examples/vit:445:train_fn] iter=67 loss=6.943604 dt=0.745598 dtd=0.002759 dtf=0.023811 dto=0.698298 dtb=0.020731 loss/mean=6.967296 loss/max=7.018311 loss/min=6.920166 loss/std=0.025391 dt/mean=0.745446 dt/max=0.759088 dt/min=0.734187 dt/std=0.009741 dtd/mean=0.002868 dtd/max=0.005055 dtd/min=0.001623 dtd/std=0.001256 dtf/mean=0.022759 dtf/max=0.024636 dtf/min=0.021525 dtf/std=0.001030 dto/mean=0.699143 dto/max=0.711942 dto/min=0.687225 dto/std=0.009435 dtb/mean=0.020677 dtb/max=0.021094 dtb/min=0.020287 dtb/std=0.000258
[2025-12-31 12:16:17,011642][I][examples/vit:445:train_fn] iter=68 loss=6.976074 dt=0.713172 dtd=0.001813 dtf=0.023177 dto=0.667256 dtb=0.020926 loss/mean=6.972229 loss/max=7.019043 loss/min=6.924316 loss/std=0.023519 dt/mean=0.747308 dt/max=0.778938 dt/min=0.712412 dt/std=0.020759 dtd/mean=0.001799 dtd/max=0.002089 dtd/min=0.001641 dtd/std=0.000141 dtf/mean=0.022417 dtf/max=0.023780 dtf/min=0.021402 dtf/std=0.000554 dto/mean=0.702239 dto/max=0.732663 dto/min=0.667256 dto/std=0.020857 dtb/mean=0.020854 dtb/max=0.021991 dtb/min=0.020358 dtb/std=0.000379
[2025-12-31 12:16:17,757204][I][examples/vit:445:train_fn] iter=69 loss=7.019287 dt=0.736641 dtd=0.001659 dtf=0.023917 dto=0.690050 dtb=0.021014 loss/mean=6.978435 loss/max=7.033447 loss/min=6.893066 loss/std=0.031614 dt/mean=0.736665 dt/max=0.741711 dt/min=0.731619 dt/std=0.003202 dtd/mean=0.003064 dtd/max=0.005396 dtd/min=0.001570 dtd/std=0.001377 dtf/mean=0.022623 dtf/max=0.024116 dtf/min=0.021729 dtf/std=0.000645 dto/mean=0.690252 dto/max=0.696442 dto/min=0.686604 dto/std=0.003588 dtb/mean=0.020725 dtb/max=0.021321 dtb/min=0.020251 dtb/std=0.000325
[2025-12-31 12:16:18,541456][I][examples/vit:445:train_fn] iter=70 loss=6.986084 dt=0.760055 dtd=0.001617 dtf=0.025038 dto=0.712560 dtb=0.020840 loss/mean=6.975403 loss/max=7.017090 loss/min=6.896240 loss/std=0.027344 dt/mean=0.746176 dt/max=0.763127 dt/min=0.731291 dt/std=0.012082 dtd/mean=0.002768 dtd/max=0.005176 dtd/min=0.001572 dtd/std=0.001342 dtf/mean=0.022493 dtf/max=0.025038 dtf/min=0.021381 dtf/std=0.001002 dto/mean=0.700163 dto/max=0.716501 dto/min=0.684296 dto/std=0.012116 dtb/mean=0.020752 dtb/max=0.021390 dtb/min=0.020222 dtb/std=0.000294
[2025-12-31 12:16:19,334196][I][examples/vit:445:train_fn] iter=71 loss=6.953369 dt=0.771233 dtd=0.001641 dtf=0.022572 dto=0.726191 dtb=0.020829 loss/mean=6.974915 loss/max=7.038574 loss/min=6.913574 loss/std=0.029813 dt/mean=0.757112 dt/max=0.789940 dt/min=0.722059 dt/std=0.020186 dtd/mean=0.001796 dtd/max=0.002250 dtd/min=0.001634 dtd/std=0.000169 dtf/mean=0.022474 dtf/max=0.023432 dtf/min=0.021478 dtf/std=0.000466 dto/mean=0.712175 dto/max=0.744055 dto/min=0.677076 dto/std=0.020119 dtb/mean=0.020667 dtb/max=0.021133 dtb/min=0.020371 dtb/std=0.000207
[2025-12-31 12:16:20,077115][I][examples/vit:445:train_fn] iter=72 loss=7.032959 dt=0.734149 dtd=0.001679 dtf=0.022509 dto=0.688994 dtb=0.020967 loss/mean=6.976024 loss/max=7.038330 loss/min=6.899414 loss/std=0.038571 dt/mean=0.740182 dt/max=0.745929 dt/min=0.734149 dt/std=0.003146 dtd/mean=0.003266 dtd/max=0.005491 dtd/min=0.001601 dtd/std=0.001299 dtf/mean=0.022640 dtf/max=0.024450 dtf/min=0.021874 dtf/std=0.000701 dto/mean=0.693552 dto/max=0.699313 dto/min=0.688994 dto/std=0.002868 dtb/mean=0.020724 dtb/max=0.021125 dtb/min=0.020261 dtb/std=0.000253
[2025-12-31 12:16:20,836082][I][examples/vit:445:train_fn] iter=73 loss=7.007568 dt=0.745784 dtd=0.001600 dtf=0.024390 dto=0.699126 dtb=0.020668 loss/mean=6.979442 loss/max=7.045898 loss/min=6.938721 loss/std=0.027204 dt/mean=0.742050 dt/max=0.748410 dt/min=0.735671 dt/std=0.003312 dtd/mean=0.003212 dtd/max=0.005472 dtd/min=0.001600 dtd/std=0.001346 dtf/mean=0.022506 dtf/max=0.024390 dtf/min=0.021779 dtf/std=0.000684 dto/mean=0.695677 dto/max=0.702715 dto/min=0.691317 dto/std=0.003525 dtb/mean=0.020656 dtb/max=0.021041 dtb/min=0.020242 dtb/std=0.000207
[2025-12-31 12:16:21,664389][I][examples/vit:445:train_fn] iter=74 loss=6.933594 dt=0.749712 dtd=0.001732 dtf=0.024171 dto=0.702734 dtb=0.021074 loss/mean=6.961670 loss/max=7.008789 loss/min=6.928467 loss/std=0.021573 dt/mean=0.744975 dt/max=0.755956 dt/min=0.736076 dt/std=0.007805 dtd/mean=0.002906 dtd/max=0.004958 dtd/min=0.001630 dtd/std=0.001174 dtf/mean=0.022639 dtf/max=0.024171 dtf/min=0.021378 dtf/std=0.001003 dto/mean=0.698668 dto/max=0.708864 dto/min=0.689240 dto/std=0.007451 dtb/mean=0.020762 dtb/max=0.021149 dtb/min=0.020382 dtb/std=0.000228
[2025-12-31 12:16:22,453938][I][examples/vit:445:train_fn] iter=75 loss=6.971191 dt=0.717920 dtd=0.001719 dtf=0.022678 dto=0.672664 dtb=0.020860 loss/mean=6.967296 loss/max=7.068115 loss/min=6.936523 loss/std=0.028438 dt/mean=0.752582 dt/max=0.785606 dt/min=0.717920 dt/std=0.021208 dtd/mean=0.001796 dtd/max=0.002113 dtd/min=0.001629 dtd/std=0.000155 dtf/mean=0.022076 dtf/max=0.022856 dtf/min=0.021302 dtf/std=0.000369 dto/mean=0.707902 dto/max=0.740324 dto/min=0.672664 dto/std=0.021404 dtb/mean=0.020809 dtb/max=0.022220 dtb/min=0.020329 dtb/std=0.000479
[2025-12-31 12:16:23,201097][I][examples/vit:445:train_fn] iter=76 loss=6.988281 dt=0.727026 dtd=0.001768 dtf=0.022971 dto=0.681522 dtb=0.020766 loss/mean=6.961833 loss/max=7.000977 loss/min=6.909180 loss/std=0.025466 dt/mean=0.755362 dt/max=0.786134 dt/min=0.719971 dt/std=0.020749 dtd/mean=0.001817 dtd/max=0.002205 dtd/min=0.001637 dtd/std=0.000169 dtf/mean=0.022384 dtf/max=0.024022 dtf/min=0.021393 dtf/std=0.000706 dto/mean=0.710380 dto/max=0.739394 dto/min=0.674898 dto/std=0.020961 dtb/mean=0.020782 dtb/max=0.021310 dtb/min=0.020390 dtb/std=0.000236
[2025-12-31 12:16:23,947330][I][examples/vit:445:train_fn] iter=77 loss=6.943848 dt=0.730181 dtd=0.001688 dtf=0.022086 dto=0.685413 dtb=0.020994 loss/mean=6.956411 loss/max=7.021484 loss/min=6.871338 loss/std=0.025088 dt/mean=0.741112 dt/max=0.746444 dt/min=0.730181 dt/std=0.003829 dtd/mean=0.002945 dtd/max=0.005359 dtd/min=0.001569 dtd/std=0.001373 dtf/mean=0.022700 dtf/max=0.024477 dtf/min=0.021997 dtf/std=0.000720 dto/mean=0.694724 dto/max=0.701460 dto/min=0.685413 dto/std=0.003837 dtb/mean=0.020743 dtb/max=0.021349 dtb/min=0.020327 dtb/std=0.000262
[2025-12-31 12:16:24,773082][I][examples/vit:445:train_fn] iter=78 loss=6.978516 dt=0.761054 dtd=0.001700 dtf=0.024937 dto=0.713578 dtb=0.020839 loss/mean=6.965098 loss/max=7.022461 loss/min=6.865479 loss/std=0.034166 dt/mean=0.748827 dt/max=0.766382 dt/min=0.732463 dt/std=0.012634 dtd/mean=0.002365 dtd/max=0.004309 dtd/min=0.001603 dtd/std=0.000892 dtf/mean=0.022818 dtf/max=0.025502 dtf/min=0.021410 dtf/std=0.001281 dto/mean=0.702965 dto/max=0.718497 dto/min=0.686230 dto/std=0.012092 dtb/mean=0.020679 dtb/max=0.021142 dtb/min=0.020321 dtb/std=0.000249
[2025-12-31 12:16:25,518658][I][examples/vit:445:train_fn] iter=79 loss=6.922363 dt=0.727205 dtd=0.001661 dtf=0.023797 dto=0.680897 dtb=0.020849 loss/mean=6.969279 loss/max=7.024170 loss/min=6.917969 loss/std=0.028303 dt/mean=0.751349 dt/max=0.784285 dt/min=0.716415 dt/std=0.020322 dtd/mean=0.001798 dtd/max=0.002212 dtd/min=0.001578 dtd/std=0.000171 dtf/mean=0.022376 dtf/max=0.024469 dtf/min=0.021327 dtf/std=0.000698 dto/mean=0.706422 dto/max=0.737019 dto/min=0.671566 dto/std=0.020411 dtb/mean=0.020752 dtb/max=0.021284 dtb/min=0.020372 dtb/std=0.000268
[2025-12-31 12:16:26,272580][I][examples/vit:445:train_fn] iter=80 loss=6.928955 dt=0.734315 dtd=0.003959 dtf=0.022724 dto=0.686737 dtb=0.020895 loss/mean=6.962504 loss/max=7.028076 loss/min=6.925293 loss/std=0.027552 dt/mean=0.740758 dt/max=0.746343 dt/min=0.734315 dt/std=0.002784 dtd/mean=0.003410 dtd/max=0.005488 dtd/min=0.001590 dtd/std=0.001289 dtf/mean=0.022813 dtf/max=0.024931 dtf/min=0.021937 dtf/std=0.000724 dto/mean=0.693835 dto/max=0.700043 dto/min=0.686737 dto/std=0.003044 dtb/mean=0.020699 dtb/max=0.021340 dtb/min=0.020316 dtb/std=0.000263
[2025-12-31 12:16:27,084889][I][examples/vit:445:train_fn] iter=81 loss=6.980469 dt=0.746773 dtd=0.001703 dtf=0.023629 dto=0.700639 dtb=0.020802 loss/mean=6.972107 loss/max=7.020996 loss/min=6.912598 loss/std=0.028705 dt/mean=0.748201 dt/max=0.759930 dt/min=0.739375 dt/std=0.007907 dtd/mean=0.002643 dtd/max=0.004569 dtd/min=0.001562 dtd/std=0.000958 dtf/mean=0.022796 dtf/max=0.024607 dtf/min=0.021381 dtf/std=0.001134 dto/mean=0.702011 dto/max=0.712560 dto/min=0.692673 dto/std=0.007507 dtb/mean=0.020751 dtb/max=0.021455 dtb/min=0.020430 dtb/std=0.000278
[2025-12-31 12:16:27,843702][I][examples/vit:445:train_fn] iter=82 loss=6.937988 dt=0.734645 dtd=0.001703 dtf=0.023584 dto=0.688593 dtb=0.020765 loss/mean=6.963796 loss/max=7.047852 loss/min=6.843262 loss/std=0.039979 dt/mean=0.755872 dt/max=0.788405 dt/min=0.721559 dt/std=0.020939 dtd/mean=0.001804 dtd/max=0.002214 dtd/min=0.001632 dtd/std=0.000168 dtf/mean=0.022503 dtf/max=0.023584 dtf/min=0.021390 dtf/std=0.000611 dto/mean=0.710801 dto/max=0.742479 dto/min=0.676907 dto/std=0.021092 dtb/mean=0.020765 dtb/max=0.021568 dtb/min=0.020342 dtb/std=0.000294
[2025-12-31 12:16:28,646170][I][examples/vit:445:train_fn] iter=83 loss=6.970703 dt=0.768635 dtd=0.001662 dtf=0.023451 dto=0.722643 dtb=0.020879 loss/mean=6.966736 loss/max=7.012695 loss/min=6.913574 loss/std=0.026851 dt/mean=0.751228 dt/max=0.781235 dt/min=0.720570 dt/std=0.019909 dtd/mean=0.001883 dtd/max=0.002957 dtd/min=0.001620 dtd/std=0.000329 dtf/mean=0.022392 dtf/max=0.024167 dtf/min=0.021578 dtf/std=0.000560 dto/mean=0.706234 dto/max=0.735096 dto/min=0.675456 dto/std=0.019756 dtb/mean=0.020719 dtb/max=0.021357 dtb/min=0.020337 dtb/std=0.000256
[2025-12-31 12:16:29,459074][I][examples/vit:445:train_fn] iter=84 loss=6.983887 dt=0.747875 dtd=0.001695 dtf=0.024866 dto=0.700442 dtb=0.020872 loss/mean=6.968292 loss/max=7.023682 loss/min=6.903809 loss/std=0.031855 dt/mean=0.752166 dt/max=0.773170 dt/min=0.729233 dt/std=0.015648 dtd/mean=0.002163 dtd/max=0.003639 dtd/min=0.001540 dtd/std=0.000643 dtf/mean=0.022767 dtf/max=0.024866 dtf/min=0.021553 dtf/std=0.000826 dto/mean=0.706544 dto/max=0.726773 dto/min=0.683277 dto/std=0.015370 dtb/mean=0.020692 dtb/max=0.021045 dtb/min=0.020339 dtb/std=0.000232
[2025-12-31 12:16:30,208391][I][examples/vit:445:train_fn] iter=85 loss=6.955322 dt=0.724813 dtd=0.001680 dtf=0.023383 dto=0.678959 dtb=0.020791 loss/mean=6.962280 loss/max=7.042480 loss/min=6.893799 loss/std=0.035695 dt/mean=0.751438 dt/max=0.784257 dt/min=0.715721 dt/std=0.020661 dtd/mean=0.001808 dtd/max=0.002161 dtd/min=0.001590 dtd/std=0.000179 dtf/mean=0.022539 dtf/max=0.023851 dtf/min=0.021704 dtf/std=0.000611 dto/mean=0.706313 dto/max=0.738357 dto/min=0.671076 dto/std=0.020862 dtb/mean=0.020778 dtb/max=0.021399 dtb/min=0.020329 dtb/std=0.000294
[2025-12-31 12:16:30,995940][I][examples/vit:445:train_fn] iter=86 loss=7.006348 dt=0.746630 dtd=0.004681 dtf=0.023454 dto=0.697523 dtb=0.020972 loss/mean=6.955139 loss/max=7.006592 loss/min=6.906006 loss/std=0.033146 dt/mean=0.744196 dt/max=0.762628 dt/min=0.727688 dt/std=0.013330 dtd/mean=0.002605 dtd/max=0.004681 dtd/min=0.001628 dtd/std=0.001020 dtf/mean=0.022811 dtf/max=0.024531 dtf/min=0.021475 dtf/std=0.000946 dto/mean=0.697820 dto/max=0.715555 dto/min=0.680953 dto/std=0.012916 dtb/mean=0.020960 dtb/max=0.022034 dtb/min=0.020438 dtb/std=0.000390
[2025-12-31 12:16:31,768262][I][examples/vit:445:train_fn] iter=87 loss=6.986572 dt=0.753831 dtd=0.001649 dtf=0.023473 dto=0.707854 dtb=0.020854 loss/mean=6.973979 loss/max=7.022217 loss/min=6.937012 loss/std=0.024000 dt/mean=0.752923 dt/max=0.785754 dt/min=0.716823 dt/std=0.020745 dtd/mean=0.001765 dtd/max=0.002053 dtd/min=0.001533 dtd/std=0.000135 dtf/mean=0.022395 dtf/max=0.023473 dtf/min=0.021672 dtf/std=0.000459 dto/mean=0.708042 dto/max=0.740092 dto/min=0.672523 dto/std=0.020517 dtb/mean=0.020721 dtb/max=0.021177 dtb/min=0.020375 dtb/std=0.000236
[2025-12-31 12:16:32,573348][I][examples/vit:445:train_fn] iter=88 loss=6.975830 dt=0.744247 dtd=0.001681 dtf=0.024306 dto=0.697233 dtb=0.021027 loss/mean=6.963277 loss/max=7.026367 loss/min=6.892090 loss/std=0.030571 dt/mean=0.742532 dt/max=0.756320 dt/min=0.730731 dt/std=0.010125 dtd/mean=0.002749 dtd/max=0.004906 dtd/min=0.001625 dtd/std=0.001149 dtf/mean=0.022892 dtf/max=0.025352 dtf/min=0.021483 dtf/std=0.001251 dto/mean=0.696179 dto/max=0.709292 dto/min=0.684075 dto/std=0.009735 dtb/mean=0.020712 dtb/max=0.021224 dtb/min=0.020333 dtb/std=0.000239
[2025-12-31 12:16:33,343035][I][examples/vit:445:train_fn] iter=89 loss=6.995605 dt=0.719805 dtd=0.001665 dtf=0.022857 dto=0.674301 dtb=0.020983 loss/mean=6.958171 loss/max=7.008789 loss/min=6.910400 loss/std=0.030195 dt/mean=0.746791 dt/max=0.771666 dt/min=0.719805 dt/std=0.018258 dtd/mean=0.002107 dtd/max=0.003667 dtd/min=0.001627 dtd/std=0.000692 dtf/mean=0.022595 dtf/max=0.024857 dtf/min=0.021425 dtf/std=0.001067 dto/mean=0.701338 dto/max=0.722970 dto/min=0.674301 dto/std=0.017279 dtb/mean=0.020751 dtb/max=0.021155 dtb/min=0.020439 dtb/std=0.000192
[2025-12-31 12:16:34,121240][I][examples/vit:445:train_fn] iter=90 loss=6.995850 dt=0.726353 dtd=0.001734 dtf=0.022452 dto=0.681219 dtb=0.020947 loss/mean=6.958120 loss/max=7.021484 loss/min=6.881348 loss/std=0.034720 dt/mean=0.749065 dt/max=0.770316 dt/min=0.725386 dt/std=0.016169 dtd/mean=0.001976 dtd/max=0.002963 dtd/min=0.001595 dtd/std=0.000374 dtf/mean=0.022522 dtf/max=0.024208 dtf/min=0.021439 dtf/std=0.000793 dto/mean=0.703768 dto/max=0.723543 dto/min=0.680416 dto/std=0.015765 dtb/mean=0.020798 dtb/max=0.021766 dtb/min=0.020314 dtb/std=0.000326
[2025-12-31 12:16:34,889609][I][examples/vit:445:train_fn] iter=91 loss=6.919434 dt=0.725853 dtd=0.004117 dtf=0.022681 dto=0.678346 dtb=0.020710 loss/mean=6.969594 loss/max=7.016113 loss/min=6.892090 loss/std=0.027690 dt/mean=0.750679 dt/max=0.769203 dt/min=0.725032 dt/std=0.015449 dtd/mean=0.002085 dtd/max=0.004117 dtd/min=0.001563 dtd/std=0.000580 dtf/mean=0.022936 dtf/max=0.024109 dtf/min=0.021977 dtf/std=0.000661 dto/mean=0.704853 dto/max=0.723677 dto/min=0.678346 dto/std=0.015343 dtb/mean=0.020805 dtb/max=0.021383 dtb/min=0.020440 dtb/std=0.000236
[2025-12-31 12:16:35,646065][I][examples/vit:445:train_fn] iter=92 loss=6.972900 dt=0.743182 dtd=0.001629 dtf=0.022283 dto=0.698406 dtb=0.020864 loss/mean=6.962301 loss/max=7.013672 loss/min=6.867188 loss/std=0.034444 dt/mean=0.754227 dt/max=0.777483 dt/min=0.731005 dt/std=0.016673 dtd/mean=0.002124 dtd/max=0.003964 dtd/min=0.001629 dtd/std=0.000662 dtf/mean=0.022493 dtf/max=0.024020 dtf/min=0.021623 dtf/std=0.000811 dto/mean=0.708874 dto/max=0.731137 dto/min=0.686752 dto/std=0.015792 dtb/mean=0.020735 dtb/max=0.021157 dtb/min=0.020330 dtb/std=0.000213
[2025-12-31 12:16:36,405894][I][examples/vit:445:train_fn] iter=93 loss=6.955811 dt=0.737651 dtd=0.001657 dtf=0.027067 dto=0.688207 dtb=0.020721 loss/mean=6.956289 loss/max=7.032227 loss/min=6.893311 loss/std=0.029941 dt/mean=0.739291 dt/max=0.745214 dt/min=0.734239 dt/std=0.003010 dtd/mean=0.003118 dtd/max=0.005423 dtd/min=0.001554 dtd/std=0.001424 dtf/mean=0.022985 dtf/max=0.027067 dtf/min=0.022070 dtf/std=0.001058 dto/mean=0.692402 dto/max=0.699386 dto/min=0.688207 dto/std=0.003538 dtb/mean=0.020786 dtb/max=0.022114 dtb/min=0.020341 dtb/std=0.000411
[2025-12-31 12:16:37,197312][I][examples/vit:445:train_fn] iter=94 loss=6.941406 dt=0.748396 dtd=0.001713 dtf=0.022729 dto=0.703060 dtb=0.020894 loss/mean=6.956451 loss/max=7.019287 loss/min=6.906250 loss/std=0.030571 dt/mean=0.753825 dt/max=0.763172 dt/min=0.746897 dt/std=0.006138 dtd/mean=0.003277 dtd/max=0.005392 dtd/min=0.001634 dtd/std=0.001310 dtf/mean=0.022594 dtf/max=0.024555 dtf/min=0.021348 dtf/std=0.001013 dto/mean=0.707138 dto/max=0.716177 dto/min=0.699779 dto/std=0.006010 dtb/mean=0.020816 dtb/max=0.021603 dtb/min=0.020155 dtb/std=0.000312
[2025-12-31 12:16:37,943003][I][examples/vit:445:train_fn] iter=95 loss=6.995361 dt=0.725441 dtd=0.001654 dtf=0.023189 dto=0.679825 dtb=0.020773 loss/mean=6.960592 loss/max=7.019287 loss/min=6.904785 loss/std=0.028771 dt/mean=0.744709 dt/max=0.761425 dt/min=0.725441 dt/std=0.013316 dtd/mean=0.002854 dtd/max=0.004914 dtd/min=0.001627 dtd/std=0.001285 dtf/mean=0.022630 dtf/max=0.024223 dtf/min=0.021440 dtf/std=0.000929 dto/mean=0.698424 dto/max=0.715335 dto/min=0.679825 dto/std=0.012162 dtb/mean=0.020800 dtb/max=0.021414 dtb/min=0.020413 dtb/std=0.000289
[2025-12-31 12:16:38,685541][I][examples/vit:445:train_fn] iter=96 loss=6.924316 dt=0.730290 dtd=0.001679 dtf=0.022138 dto=0.685656 dtb=0.020817 loss/mean=6.952393 loss/max=7.033936 loss/min=6.894531 loss/std=0.028505 dt/mean=0.738270 dt/max=0.744446 dt/min=0.730290 dt/std=0.003330 dtd/mean=0.003186 dtd/max=0.005505 dtd/min=0.001642 dtd/std=0.001371 dtf/mean=0.022534 dtf/max=0.023826 dtf/min=0.021813 dtf/std=0.000534 dto/mean=0.691715 dto/max=0.697518 dto/min=0.685656 dto/std=0.003466 dtb/mean=0.020836 dtb/max=0.021774 dtb/min=0.020360 dtb/std=0.000338
[2025-12-31 12:16:39,454027][I][examples/vit:445:train_fn] iter=97 loss=6.977295 dt=0.747579 dtd=0.001707 dtf=0.026247 dto=0.698855 dtb=0.020769 loss/mean=6.967041 loss/max=7.055420 loss/min=6.875977 loss/std=0.035210 dt/mean=0.745013 dt/max=0.753496 dt/min=0.739475 dt/std=0.005213 dtd/mean=0.002956 dtd/max=0.004715 dtd/min=0.001561 dtd/std=0.000987 dtf/mean=0.022581 dtf/max=0.026247 dtf/min=0.021254 dtf/std=0.001179 dto/mean=0.698686 dto/max=0.706904 dto/min=0.692804 dto/std=0.004965 dtb/mean=0.020791 dtb/max=0.021900 dtb/min=0.020284 dtb/std=0.000342
[2025-12-31 12:16:40,195874][I][examples/vit:445:train_fn] iter=98 loss=6.967773 dt=0.732756 dtd=0.001660 dtf=0.025091 dto=0.685154 dtb=0.020851 loss/mean=6.942464 loss/max=7.022705 loss/min=6.878174 loss/std=0.035641 dt/mean=0.739319 dt/max=0.745367 dt/min=0.732756 dt/std=0.003418 dtd/mean=0.002662 dtd/max=0.003719 dtd/min=0.001660 dtd/std=0.000606 dtf/mean=0.022934 dtf/max=0.025091 dtf/min=0.021985 dtf/std=0.000785 dto/mean=0.692973 dto/max=0.697970 dto/min=0.685154 dto/std=0.002909 dtb/mean=0.020750 dtb/max=0.021213 dtb/min=0.020436 dtb/std=0.000199
[2025-12-31 12:16:40,950324][I][examples/vit:445:train_fn] iter=99 loss=6.977539 dt=0.745598 dtd=0.002238 dtf=0.023939 dto=0.698594 dtb=0.020827 loss/mean=6.956676 loss/max=7.008301 loss/min=6.899658 loss/std=0.022355 dt/mean=0.741473 dt/max=0.748397 dt/min=0.733044 dt/std=0.004455 dtd/mean=0.003068 dtd/max=0.005405 dtd/min=0.001547 dtd/std=0.001378 dtf/mean=0.023009 dtf/max=0.025263 dtf/min=0.021605 dtf/std=0.000825 dto/mean=0.694567 dto/max=0.701266 dto/min=0.687203 dto/std=0.004004 dtb/mean=0.020828 dtb/max=0.021858 dtb/min=0.020438 dtb/std=0.000346
[2025-12-31 12:16:41,771715][I][examples/vit:445:train_fn] iter=100 loss=7.000244 dt=0.760887 dtd=0.001625 dtf=0.026943 dto=0.711486 dtb=0.020833 loss/mean=6.960460 loss/max=7.000244 loss/min=6.919434 loss/std=0.022609 dt/mean=0.746579 dt/max=0.762490 dt/min=0.731703 dt/std=0.012070 dtd/mean=0.002105 dtd/max=0.003019 dtd/min=0.001625 dtd/std=0.000459 dtf/mean=0.022804 dtf/max=0.026943 dtf/min=0.021464 dtf/std=0.001315 dto/mean=0.700953 dto/max=0.715302 dto/min=0.686720 dto/std=0.011352 dtb/mean=0.020716 dtb/max=0.021132 dtb/min=0.020289 dtb/std=0.000214
/lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/src/ezpz/history.py:2223: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
Consider using tensor.detach() first. (Triggered internally at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:835.)
  x = torch.Tensor(x).numpy(force=True)
[2025-12-31 12:16:42,102625][I][ezpz/history:2385:finalize] Saving plots to /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/mplot (matplotlib) and /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot (tplot)
                  train_dt                               train_dt/min
     ┌─────────────────────────────────┐      ┌────────────────────────────────┐
0.792┤    ▗▌                           │0.7469┤--                      -    -- │
0.777┤    ▐▌     ▗                     │0.7304┤--------------------------------│
     │    ▐▌▗▌  ▗█     ▗   ▗▌   ▟      │0.7138┤--    -   - - --- ----------    │
0.762┤    ▐▌▐▌  █▌▌    █   ▐▌ ▟ █ ▖   ▞│0.6973┤-                               │
0.747┤▖▖  █▌▐▙▌▐█▌▌▖▗▌ █▜ ▗▐▌▌█▗█▐▌▗▟▟▌│      └┬───────┬───────┬──────┬───────┬┘
0.732┤█▌▟▗▜▌▌▛█▟▛▌█▚▐▌▗▜▝▙█▞▜▌█▐█▌▌▐▜▛▌│      1.0    23.5    46.0   68.5   91.0
     │██▘▀ ▝▘▘▝ ▌▘█ ▀█▞   █▌ ▙▘▘▝▌▙▀▐▘ │train_dt/min         iter
0.717┤█▝          ▝  ▀▌   ▀▌ ▝    ▝    │                 train_dt/std
0.702┤▜                                │      ┌────────────────────────────────┐
     └┬───────┬───────┬───────┬───────┬┘0.0225┤*    *    *   ***   ********    │
     1.0    23.5    46.0    68.5   91.0 0.0188┤** ***** **** ***************   │
train_dt            iter                0.0113┤***** ** ** * *  ******** **** *│
                 train_dt/mean          0.0075┤***** ****  ***  ********    ***│
      ┌────────────────────────────────┐0.0000┤**  *              *    *       │
0.7657┤     ·     ·                    │      └┬───────┬───────┬──────┬───────┬┘
0.7588┤    ··    ··                    │      1.0    23.5    46.0   68.5   91.0
      │   ··· ·  ··    ·    · · ·      │train_dt/std         iter
0.7519┤ ·······  ··  ···  · ·········  │                train_dt/max
0.7450┤ ········· ···· ········· ···· ·│     ┌─────────────────────────────────┐
      │····· ····  ···  ········ ······│0.793┤     +     ++    +    ++  ++     │
0.7381┤····   ··    ··   ····       ·· │0.783┤++ + +++  +++  ++++ ++++ +++++   │
0.7312┤···                             │0.763┤+++++++++ + ++ + ++ ++++++ ++++ +│
      │··                              │0.752┤+++ + ++++  +++  ++++++ ++    +++│
0.7243┤ ·                              │0.732┤ ++                              │
      └┬───────┬───────┬──────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
      1.0    23.5    46.0   68.5   91.0      1.0    23.5    46.0    68.5   91.0
train_dt/mean        iter               train_dt/max        iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_dt.txt
     ┌─────────────────────────────────────────────────────────────────────────┐
0.793┤ ++ train_dt/max        +                                                │
     │ -- train_dt/min       ++ +                      +        +              │
     │ ·· train_dt/mean     +++ +        +  +         ++  ++ + ++ + +          │
     │ ▞▞ train_dt          +++++       ++ ++         ++ + +++ ++++++          │
0.777┤   +  ++  +█++ +      ++++▖      +++ ++  +    + ++ + +++ ++++++   +      │
     │  ++  ++  +█++++      +++▐▌      ++ + + ++   ++ +▖ + +++ + ++++  ++      │
     │ +++  ++  +█+++▟      ++ ▐▌+     +    + ++  +++ ▐▌ + +++ + ▖+++++++      │
     │++++  ++  +█+++█      +▗ ▐▌+     +    + ++  +++ ▐▌ + +++ +▐▌+++   +      │
0.761┤+++ ++ + +▗▜ ++█      +█·▐▌+ +   +    ▟ ++  +++ ▐▌ + ++▖ +▐▌ ++   +++   ▗│
     │+++ ++·· +▐▐ ++█      +█·▐▐+++   +   ·█ ++  + ++▌▌ + +▐▌ +▐▌  +   + +  +▌│
     │+++ ++·· ▟▐▐ +▐▐  +  ▗▌█·▐▐+++   +· · █+ +  + ++▌▚ + ·▐▌+ ▐▌  ▖   ··+ ++▌│
     │+++··+·· █▐▐ ·▐▐ ▗+++▐▌█·▞▐ +++ +▖ ·· █+ +  +·++▌▐+▗··▐▌+·▞▌·▐▌ ····+++▐ │
0.745┤++· ·+·· █▐▐··▐▐+█·  ▐▌█ ▌▐  ·· ▐▌  · ▌▌▞▌+ · ▖▐·▐+█ ·▐▌+▖▌▐·▌▐·  ·▟+▗▌▐·│
     │▌+▗ ·· ·+▌█▐  ▞▐·█·▗·▌█▐ ▌▐··▖·+▐▌    ▌▌▌▌ +·▐▌▐ ▐▐▐ ·▌▌▐▌▌▐▐··▌ ▗▌█·▐▌▌ │
     │▌+█ ·▟ ··▌█▐ ▗▘▐·▛▖█ ▌█▐ ▌▝▖▞▚··▞▌    ▌▙▘▐▟·▟▐▌▐ ▐▞▐  ▌▌▐▐▌▐▐  ▌ ▐▝█ ▐▐▌ │
     │▌·█ ▖▛▖ ▐-█▐ ▐-▐▗▘▌▛▄▌█▐ ▌ ▙▘▝▖ ▌▚   ▗▘█-▐▛▄▜▐▌▞ ▝▌▐  ▌▌▐▝▌ █  ▌ ▐--▌▐▐▌ │
     │▌·█▐▚▌▚▟▌ ▝▝▖▌ ▐▞ █   █ ▌▌-█ -▝▖▌▐   ▌ ▝- ▘▜▐▞▌▌-- -▌ ▌▌▌-  █ -▌ ▐--▌▐-▘-│
0.729┤▌▐▐▐▝▌--▘  -▝▘  ▘ ▜   █-▝▌-█   ▝ ▐▗ ▐-- --  ▐▌▌▌-- -▌▞-▜ - -█--▌▗▐ -▙▘   │
     │▚▐▐▌    -   --        ▝ - -█     ▐█ ▌-- --  ▐▌▙▘-- -█--- ---▝--▌▌▀  ▝    │
     │▐▐▝▌         -        -    ▜     ▝▛▄▘--  -  ▐▌█  - -█---  -----▜         │
     │▐▞-                               -█  -      ▘█     ▝  -    - -          │
0.713┤▐▌-                                ▝          ▜                          │
     │▐▌                                                                       │
     │▐▌                                                                       │
     │-▘                                                                       │
0.697┤ -                                                                       │
     └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
     1.0              23.5              46.0              68.5             91.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_dt_summary.txt
             train_dt/mean hist                       train_dt/max hist
    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
22.0┤                 ████             │23.0┤       ███                        │
18.3┤                 ████             │19.2┤       ███                        │
    │             ████████             │    │       ███                        │
14.7┤          ██████████████          │15.3┤       ███                        │
11.0┤          ██████████████          │11.5┤       ███          ████   ████   │
    │          █████████████████       │    │       ███          ████   ████   │
 7.3┤          █████████████████       │ 7.7┤       ████████████████████████   │
 3.7┤       ████████████████████████   │ 3.8┤   ███████████████████████████████│
    │       ███████████████████████████│    │   ███████████████████████████████│
 0.0┤███    ███████████████████████████│ 0.0┤██████████████████████████████████│
    └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
   0.722   0.734    0.745   0.756 0.768    0.730   0.746    0.763   0.779 0.795
              train_dt/min hist                       train_dt/std hist
    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
29.0┤                        ███       │26.0┤   ████                           │
    │                        ███       │    │   ████                           │
24.2┤                    ███████       │21.7┤   ████                           │
19.3┤                    ███████       │17.3┤   ████                           │
    │                    ███████       │    │   ████                           │
14.5┤                    ███████       │13.0┤   ████                    ███████│
    │             ████   ███████       │    │   ████                    ███████│
 9.7┤             ██████████████       │ 8.7┤   ████   ████   ████   ██████████│
 4.8┤          █████████████████████   │ 4.3┤   ███████████████████████████████│
    │          ████████████████████████│    │██████████████████████████████████│
 0.0┤███       ████████████████████████│ 0.0┤██████████████████████████████████│
    └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬────────┘
   0.695   0.709    0.722   0.736 0.749   -0.0010  0.0051  0.0113  0.0174
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_dt_hist.txt
                    train_dtb                             train_dtb/min
        ┌──────────────────────────────┐        ┌──────────────────────────────┐
0.021110┤             ▌       ▖        │0.020440┤                 --  - -------│
0.021031┤             ▌     ▗▐▌   ▗    │0.020336┤   - ------ ------------------│
        │             ▌     █▐▙▌  ▛▖   │0.020233┤-----------------  --      -- │
0.020952┤       ▖   ▌ ▌▖ ▗  ███▌ ▐▌▌   │0.020129┤ - --  -    -              -  │
0.020873┤  ▗▚▗ ▐▌▖ ▙▌ █▙▖▞▜▟▐█▜▙▙▜▌▌▟ ▖│        └┬──────┬───────┬──────┬──────┬┘
0.020793┤▗ ▐▐█▗▌█▙▚▘▌▖█▌█▌▝█▝▜▐▝▜▐ ███▀│        1.0   23.5    46.0   68.5  91.0
        │▐▟▞▐▐█ ▜▛  █▙▀▌█  ▜ ▐▝ ▝  █▌▀ │train_dtb/min         iter
0.020714┤▀▀▘  ▘  ▌  ██ ▌▝  ▝ ▐     ▝▘  │                  train_dtb/std
0.020635┤        ▌  ▝▜       ▝         │        ┌──────────────────────────────┐
        └┬──────┬───────┬──────┬──────┬┘0.000479┤ *                   *        │
        1.0   23.5    46.0   68.5  91.0 0.000431┤***           ** *** *  *  *  │
train_dtb             iter              0.000335┤***  * ***   ******* *  * ****│
                 train_dtb/mean         0.000288┤******************************│
        ┌──────────────────────────────┐0.000192┤   *    **          *    ** **│
0.020960┤                        ·     │        └┬──────┬───────┬──────┬──────┬┘
0.020900┤                        ·     │        1.0   23.5    46.0   68.5  91.0
        │                ···     ·     │train_dtb/std         iter
0.020841┤               ·····    ·   ··│                 train_dtb/max
0.020781┤     ·  ·  · ······· ·  · ····│       ┌───────────────────────────────┐
        │···  · ····· ······· ······ ··│0.02236┤                  +   +        │
0.020722┤···  ···········  ··· ·····  ·│0.02214┤  +           +  +++ ++  +  ++ │
0.020662┤ · ····  ····· ·  · · ···     │0.02170┤+++   + +    ++ ++++ ++ ++ ++++│
        │   ···    ·         ·         │0.02148┤++++++++++++++++++ ++++++++++++│
0.020603┤   ·                          │0.02104┤++ ++ +  + +++ +   +++ + +++  +│
        └┬──────┬───────┬──────┬──────┬┘       └┬───────┬──────┬───────┬──────┬┘
        1.0   23.5    46.0   68.5  91.0        1.0    23.5   46.0    68.5  91.0
train_dtb/mean        iter              train_dtb/max        iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_dtb.txt
       ┌───────────────────────────────────────────────────────────────────────┐
0.02236┤ ++ train_dtb/max                        +                             │
       │ -- train_dtb/min                       ++                             │
       │ ·· train_dtb/mean                      ++         +                   │
       │ ▞▞ train_dtb                           ++ +      ++             +     │
0.02199┤                                        ++++ +    ++       +    ++     │
       │     +                           +      ++++++    ++      ++    ++  +  │
       │  + ++                         + +      ++++++    ++      ++    ++ +++ │
       │ ++ ++                        ++ +   +  ++++++    ++      ++  + ++ +++ │
0.02162┤ ++ ++                        + ++  ++  + + ++    ++      ++ ++ +++ ++ │
       │+++ ++       +     +          + ++  +++++   ++    ++    + ++ ++ +++ ++ │
       │+++ ++      ++    ++    + +   +  +  +++++   ++    ++   ++ ++ ++ +++ ++ │
       │+++ ++      ++   +++ + + ++   +   +++++++   ++ +  +++ +  +++ + ++ + ++ │
0.02124┤++++  +     ++  +  +++ + +++  +    +++ +    + ++  +++ +  +++ + ++   ++ │
       │++++  +  +++++ +      +   + + +     +       +  +  + ++   ++ ++  +   ++ │
       │ + +   ++     +              + ▗            +  ++ ▖  +   ++           +│
       │                               █              ▖  ▐▌ ▗     + ▗▚▖        │
       │                ▗         ▗   ▗▜ ▖      ▖  · ▐▚ ▟▐▐ ▛▖ ▖   ▞▟ ▐   ▖    │
0.02087┤      ▗▀▌▗▗▌  ▖▞▀▖▗▌ ▄▖▗▞▚█ ▖ ▐▐▞▌▞▀▌▗▗▀▝▀▄▌·▌▝▄█▐ █ ▚▞▌▗▀▚▌▝ ▝▄▌▐▌▗▗▚▄│
       │·▞▄▚▄▞▘ ▚▘▘▐▞▟▝▘·▝▌▌▞·▝▘· ·█▝▌▞ ▘▙▘ ▝▌▀··  ▝▟ ···█·▝· ·▝▀··▘·· █▝▌▝▘▘ ·│
       │▀·▝·  · ····▘ ··   ▜· ·· ··▜·▝▌  ▝ ·        ·  · ▜   ··   ·    ▝       │
       │       ·                                                               │
0.02050┤                                                                       │
       │                 -               -    -  ----- -  --  --   --- -  ---- │
       │-    -  ---------- - ------ - ----  --- -   -- - - ---- ---  -- --  - -│
       │ -- -- - -  -   --- -  -  ----  - -- - --     ----               -     │
0.02013┤  --  -  -        -          -                                   -     │
       └┬─────────────────┬────────────────┬─────────────────┬────────────────┬┘
       1.0              23.5             46.0              68.5            91.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_dtb_summary.txt
             train_dtb/mean hist                    train_dtb/max hist
    ┌──────────────────────────────────┐  ┌────────────────────────────────────┐
23.0┤       ███                        │24┤    ███                             │
19.2┤       ███   ████                 │20┤    ███████                         │
    │       ███   ████                 │  │    ███████                         │
15.3┤       ██████████                 │16┤███████████                         │
11.5┤       ██████████████             │12┤███████████                         │
    │       ██████████████             │  │███████████                         │
 7.7┤   ██████████████████             │ 8┤██████████████████                  │
 3.8┤   ████████████████████████       │ 4┤██████████████████   ████           │
    │   ████████████████████████       │  │████████████████████████████████    │
 0.0┤██████████████████████████████████│ 0┤████████████████████████████████████│
    └┬───────┬────────┬───────┬────────┘  └┬────────┬────────┬───────┬─────────┘
  0.02059  0.02068  0.02078 0.02088      0.02098  0.02134  0.02170 0.02206
             train_dtb/min hist                      train_dtb/std hist
    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
21.0┤                    ████          │21.0┤          ████                    │
    │                    ████          │    │       ███████                    │
17.5┤                    ████          │17.5┤   ███████████                    │
14.0┤             ████   ████          │14.0┤   ███████████                    │
    │             ████   ████          │    │   ███████████                    │
10.5┤             ████   ████          │10.5┤   ███████████                    │
    │       ████████████████████   ████│    │█████████████████████             │
 7.0┤       ████████████████████   ████│ 7.0┤█████████████████████             │
 3.5┤███████████████████████████   ████│ 3.5┤████████████████████████          │
    │██████████████████████████████████│    │████████████████████████   ███████│
 0.0┤██████████████████████████████████│ 0.0┤██████████████████████████████████│
    └┬───────┬────────┬───────┬────────┘    └┬───────┬────────┬───────┬────────┘
  0.02011  0.02020  0.02028 0.02037       0.000179 0.000257 0.000335 0.000413
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_dtb_hist.txt
                   train_dtd                             train_dtd/min
      ┌────────────────────────────────┐       ┌───────────────────────────────┐
0.0122┤           ▟                    │0.00312┤-                              │
0.0104┤           █                    │0.00259┤-                              │
      │           █                    │0.00206┤--                             │
0.0086┤           █                    │0.00153┤-------------------------------│
0.0069┤           █                    │       └┬───────┬──────┬───────┬──────┬┘
0.0051┤     ▗     █                    │       1.0    23.5   46.0    68.5  91.0
      │     █     █            ▗ ▐ ▗   │train_dtd/min        iter
0.0034┤▖    █     █        ▖   █ ▐ █   │                 train_dtd/std
0.0016┤▚▄▄▄▄█▄▄▄▄▄▛▄▄▄▄▙▄▄▟▚▄▄▄▛▄▟▄█▄▄▙│       ┌───────────────────────────────┐
      └┬───────┬───────┬──────┬───────┬┘0.00207┤           *                   │
      1.0    23.5    46.0   68.5   91.0 0.00174┤ *    **** ***  * * ***     ***│
train_dtd            iter               0.00110┤********** **** ***************│
                train_dtd/mean          0.00077┤******** **** **** ********* **│
       ┌───────────────────────────────┐0.00013┤** * **  ***  **** ** *****    │
0.00410┤··                             │       └┬───────┬──────┬───────┬──────┬┘
0.00371┤··                             │       1.0    23.5   46.0    68.5  91.0
       │··  ·       ·                  │train_dtd/std        iter
0.00332┤····· ···   ··   ··  · ·    ·  │                 train_dtd/max
0.00293┤··········  ··  ··· ····    ···│      ┌────────────────────────────────┐
       │········ · ···  ········  ·····│0.0122┤           +                    │
0.00254┤········ · ·· · ············ ··│0.0105┤           +                    │
0.00215┤·······  ···· ··············  ·│0.0071┤           +                    │
       │·······  ···  ···· ·········  ·│0.0054┤+++++++++++++++ ++++++++++++++++│
0.00176┤·· · ··  ···  ···· ·· ·····    │0.0020┤++++++++  +++ +++++++++++++++  +│
       └┬───────┬──────┬───────┬──────┬┘      └┬───────┬───────┬──────┬───────┬┘
       1.0    23.5   46.0    68.5  91.0       1.0    23.5    46.0   68.5   91.0
train_dtd/mean       iter               train_dtd/max        iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_dtd.txt
      ┌────────────────────────────────────────────────────────────────────────┐
0.0122┤ ++ train_dtd/max        ▗▌                                             │
      │ -- train_dtd/min        ▐▌                                             │
      │ ·· train_dtd/mean       ▐▌                                             │
      │ ▞▞ train_dtd            ▐▌                                             │
0.0104┤                         ▐▌                                             │
      │                         ▐▌                                             │
      │                         ▐▌                                             │
      │                         ▐▌                                             │
0.0086┤                         ▐▌                                             │
      │                         ▐▌                                             │
      │                         ▐▌                                             │
      │                         ▐▌                                             │
0.0068┤                         ▐▚                                             │
      │                         ▐▐                                             │
      │                         ▐▐                                             │
      │                         ▐▐                                             │
      │+ +  +   +   +   + +++   ▐▐ ++ ++      + +++   + ++  + +         ++ + + │
0.0051┤+++ ++  ++ ▗▌+  +++  +   ▐▐++++ +     +++  + ++++ + ++++      +  + ++++ │
      │+++ +++ ++ ▐▌+ + ++  +   ▐▐++ + +     +++  ++++++ + + + +   ▟++  +   ++ │
      │·+·+ ++ ++ ▐▌+ + +   +   ▐▐++   +     +++  ++++++ + + +▗+  +█++ ▗+   ++ │
      │·+·+ +++ · ▐▌+ +     + + ▐▐++·  +   + +++  ++++++ + + +█+ ++█+++█    ++ │
0.0033┤▖+·+ ·++··+▞▌·++······+++▐▐+·· ·· +++ +·+···+++·+·· + +█+ ++█+++█·· · ·+│
      │▌··+··++··+▌▌·+· ·   ·+++▐▐·· · ·++++ ···  ·+▗··· · +··▛▖ +▐▐+·▗▜· · ·· │
      │▚··+···+·· ▌▌··· ·   ·+··▐▐··   ·+++· ···  ··█··· · · ·▌▌++▐▐··▐▐·    · │
      │▐·-·  ·· ··▌▌· ·      · ·▐▐··    ···▗·· ·   ▐▐· ·  ·· ·▌▌··▐▐··▐▐·    ▟·│
0.0015┤ ▀▀▀▀▀▀▀▀▀▀▘▝▀▀▀▀▀▀▀▀▄▀▀▀▀-▀▀▀▀▀▀▀▀▀▘▀▀▀▀▀▀▀▀-▀▚▞▀▄▀▀▀▀▘▝▀▀▀-▀▀▀▝▄▀▀▀▀▘▚│
      └┬─────────────────┬─────────────────┬────────────────┬─────────────────┬┘
      1.0              23.5              46.0             68.5             91.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_dtd_summary.txt
             train_dtd/mean hist                     train_dtd/max hist
    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
27.0┤████                              │31.0┤          ████                    │
22.5┤████                              │25.8┤████      ████                    │
    │████                              │    │████      ████                    │
18.0┤████                              │20.7┤████      ████                    │
13.5┤████                              │15.5┤████   ███████                    │
    │███████             ████          │    │██████████████                    │
 9.0┤███████   ██████████████          │10.3┤██████████████                    │
 4.5┤████████████████████████          │ 5.2┤██████████████                    │
    │███████████████████████████   ████│    │██████████████                    │
 0.0┤██████████████████████████████████│ 0.0┤█████████████                 ████│
    └┬───────┬────────┬───────┬────────┘    └┬───────┬────────┬───────┬────────┘
  0.00166  0.00230  0.00293 0.00357       0.0016  0.0043   0.0071  0.0098
             train_dtd/min hist                      train_dtd/std hist
    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
88.0┤████                              │22.0┤████                ████          │
    │████                              │    │████                ████          │
73.3┤████                              │18.3┤████                ████          │
58.7┤████                              │14.7┤████                ████          │
    │████                              │    │████             ███████          │
44.0┤████                              │11.0┤████   ███       ███████          │
    │████                              │    │██████████   ███████████          │
29.3┤████                              │ 7.3┤██████████   ███████████          │
14.7┤████                              │ 3.7┤████████████████████████          │
    │████                              │    │████████████████████████          │
 0.0┤███████                       ████│ 0.0┤████████████████████████      ████│
    └┬───────┬────────┬───────┬────────┘    └┬───────┬────────┬───────┬────────┘
  0.00146  0.00189  0.00233 0.00276       0.00004  0.00057  0.00110 0.00163
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_dtd_hist.txt
                   train_dtf                             train_dtf/min
       ┌───────────────────────────────┐       ┌───────────────────────────────┐
0.02730┤     ▟                      ▖ ▗│0.02219┤        --   -  -     -    --- │
0.02635┤     █            ▗         ▌▗▌│0.02173┤  -----------------------------│
       │    ▗█            ▐         ▌█▌│0.02128┤-------- ---- --- -------------│
0.02540┤   ▗██   ▗▌     ▗ ▐ ▗  ▖ ▖  ▌▛▌│0.02083┤-   -      -                   │
0.02445┤   ▐██  ▖▌▌▗▌   █ ▐ █▗▐▌▐▌▗ ▌▌▌│       └┬───────┬──────┬───────┬──────┬┘
0.02350┤   ▐██▟ ▌▌▌▐▌▗▌ █▄█▟▜█▐▐▟▌█ ▌▌▘│       1.0    23.5   46.0    68.5  91.0
       │▐▌▖▟███▗▌▌▚▟██▌▟▐▐█▛▐█▐▐ ▝▝▖▙▌ │train_dtf/min        iter
0.02254┤▐▜▝▌▝▘▀▛▛▘ ▀ ▀▀▘▐▝▜▘▝▘▜▝   ▀▛▌ │                 train_dtf/std
0.02159┤▌                     ▝        │       ┌───────────────────────────────┐
       └┬───────┬──────┬───────┬──────┬┘0.00132┤    **      *     *    *  *  **│
       1.0    23.5   46.0    68.5  91.0 0.00113┤    ***  **** ** ********** ***│
train_dtf            iter               0.00075┤ ******************************│
                train_dtf/mean          0.00056┤*****  *** ****** *** * * *  * │
       ┌───────────────────────────────┐0.00019┤*                 *            │
0.02330┤ ·                             │       └┬───────┬──────┬───────┬──────┬┘
0.02294┤·· · ·   ··  ·              · ·│       1.0    23.5   46.0    68.5  91.0
       │···· ·  ······ ·····   ········│train_dtf/std        iter
0.02258┤······························ │                 train_dtf/max
0.02222┤·····  ·   ·  ··· ·· ···· ·    │       ┌───────────────────────────────┐
       │···                   ·        │0.02730┤     +                      + +│
0.02186┤·                              │0.02635┤    ++    + +    ++    +  +++++│
0.02150┤·                              │0.02445┤ ++++ ++++++++++++++++++++++++ │
       │·                              │0.02350┤++++   +   + +++  + + + + +    │
0.02114┤·                              │0.02159┤+                              │
       └┬───────┬──────┬───────┬──────┬┘       └┬───────┬──────┬───────┬──────┬┘
       1.0    23.5   46.0    68.5  91.0        1.0    23.5   46.0    68.5  91.0
train_dtf/mean       iter               train_dtf/max        iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_dtf.txt
      ┌────────────────────────────────────────────────────────────────────────┐
0.0273┤ ++ train_dtf/max                                                       │
      │ -- train_dtf/min                                                ▗▌    ▗│
      │ ·· train_dtf/mean                                               ▐▌   +▌│
      │ ▞▞ train_dtf                                                    ▐▌   +▌│
0.0262┤            █+                             ▟                     ▐▌ ▗▌+▌│
      │            █+                             █                     ▐▌ ▐▌+▌│
      │          ▗▌█+                             █                     ▐▌ ▐▚+▌│
      │         ▗▐▌█+                             █          +          ▐▌ ▐▐▗▘│
0.0251┤         █▐▌█+         ▖    +          +   █         ++       +  ▐▌ ▐▝▟ │
      │         █▐▌▛▖+      ▗▀▝▖  ++         ▖+  +█   ▟     +▖+  +▖ ++  ▐▌ ▐+█ │
      │  +     +█▐▌▌▌+      ▐  ▌ + ▖ +     +▐▌+ ++█ + █     ▐▌ + ▐▌ ++  ▐▌ ▐ █ │
      │ ++++  ++█▐▌▌▌+    + ▐  ▌+ ▐▌++ +  ++▐▌++ +▛▖+ █ +▖  ▐▌ + ▐▌+++  ▌▌ ▐ █ │
0.0241┤ ++ +  ++█▐▚▌▌ ▖+ ▗▌+▐  ▌+ ▐▌++ ▗ +++▐▌++ +▌▌+ █+▐▝▖+▐▐ + ▞▌+▗▌++▌▌+▐ █ │
      │ ++ +  +▐▝▟▐▌▌▐▌++▐▌+▞  ▌+ ▐▌ + █++++▐▌  ▗▌▌▌▗+█+▐+▌+▐▐ ++▌▌+▐▌  ▌▌ ▐ ▝ │
      │ ++ +  +▐ █▐▌▌▐▌ +▐▌ ▌  ▌+ ▐▌  +█+ ++▞▚▞▄▐▌▌▌█▐+▌▐+▌ ▐ ▌▗▖▌▌+▌▌  ▌▌ ▐   │
      │ +· + ++▐ █▐▌▌▐▌  ▐▚ ▌  ▌+ ▌▌  ▖█    ▌▐▌▐▐▌▌▌█▌ ▌▐+▌ ▞ ▌▌▝▘▚▞▘▐  ▌▌ ▐   │
      │ ▞▜  + ▗█ █▐▌▌▐▚  ▐▐▗▘  ▌+ ▌▌▟▐▌▌▌ ▗█ ▐▌▝▟▚▌▙▘▘ ▌▐+▌ ▌ ▙▘     ▐  ▌▌▟▐   │
0.0230┤+▌▐ ▗▌ ▐█ █▐▌▌▐▐ ▟▐▐▐· ·▚▗▌▌▝▐▌▐▌▌ ▐▝ ▐▌ █▐▌█   ▌▐ ▌▟▌·█·   · ▝▖·▌▙▜▐···│
      │+▌▐·▞▌ ▌▝·█▝▌▚▘▐·▛▟▝▟····▀▝▌··▘·▘▌·▌· ▐▌·▜▐▌▜· ·▚▐·▝▐▌·▝· ·····▌▟▌▝▝▟·  │
      │+▌▐·▌▝▀▌ ·▝    ▐▐·▜·▝·  ··      ·▝▀▘ ·▝▌  ▐▌  ···▀· ▐▌·  ··  · ▝▐▌  █   │
      │▐··▀ ··         ▀ -           -            ▘       ·▝▌           ▘  ▜   │
0.0219┤▞            -  -- ---       - --      - - -     -- -- -        --  --  │
      │▌    ----- - - - -   ---   ---  ---- --------- -- - ----  -- --- - ---- │
      │·-- - -  -- ---  -   -  --- -   -   -     -   --  --- - --  - --  -- - -│
      │·  -     -                -                                             │
0.0208┤-                                                                       │
      └┬─────────────────┬─────────────────┬────────────────┬─────────────────┬┘
      1.0              23.5              46.0             68.5             91.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_dtf_summary.txt
             train_dtf/mean hist                    train_dtf/max hist
    ┌──────────────────────────────────┐  ┌────────────────────────────────────┐
37.0┤                    ████          │24┤              ████████              │
30.8┤                    ████          │20┤              ████████              │
    │                    ████          │  │           ███████████              │
24.7┤                    ████          │16┤           ███████████              │
18.5┤                    ███████       │12┤           ██████████████           │
    │                 ██████████       │  │           ██████████████           │
12.3┤                 ██████████████   │ 8┤           ██████████████           │
 6.2┤                 ██████████████   │ 4┤       ██████████████████           │
    │             ██████████████████   │  │       ██████████████████    ███████│
 0.0┤███       ████████████████████████│ 0┤████   █████████████████████████████│
    └┬───────┬────────┬───────┬────────┘  └┬────────┬────────┬───────┬────────┬┘
  0.02104  0.02163  0.02222 0.02281      0.0213  0.0229   0.0244  0.0260 0.0276
             train_dtf/min hist                      train_dtf/std hist
    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
23.0┤             ████                 │19.0┤             ████                 │
    │             ████                 │    │          ███████                 │
19.2┤             ████                 │15.8┤          ███████████             │
15.3┤             ███████████          │12.7┤          ███████████             │
    │          ██████████████          │    │          ███████████             │
11.5┤          ██████████████          │ 9.5┤       ██████████████   ███       │
    │          ██████████████          │    │       ██████████████   ███       │
 7.7┤          ██████████████   ████   │ 6.3┤       ███████████████████████████│
 3.8┤          █████████████████████   │ 3.2┤       ███████████████████████████│
    │   ███████████████████████████████│    │██████████████████████████████████│
 0.0┤██████████████████████████████████│ 0.0┤██████████████████████████████████│
    └┬───────┬────────┬───────┬────────┘    └┬───────┬────────┬───────┬────────┘
  0.02077  0.02114  0.02151 0.02188       0.00014  0.00044  0.00075 0.00106
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_dtf_hist.txt
                  train_dto                              train_dto/min
     ┌─────────────────────────────────┐      ┌────────────────────────────────┐
0.743┤    ▗▌                           │0.6998┤--  -                   -    -- │
0.729┤    ▐▌     ▟          ▖          │0.6840┤--------------------------------│
     │    ▐▌▗▌  ▗█     ▗   ▐▌   ▟      │0.6683┤--    -   - - --- --- ------    │
0.715┤    ▐▌▐▌  ██     █   ▐▌ ▟ █ ▖   ▞│0.6526┤-                               │
0.700┤▖▖  █▌▐▙▌▐█▌▌▖▗▌ █▜ ▗▐▌▌█▐█▐▌▗▟▗▌│      └┬───────┬───────┬──────┬───────┬┘
0.686┤█▙▛▄▜▌▌▛█▞▛▌█▚▐▌▗▜▝▙█▐▜▌█▐█▌▌▐█▛▌│      1.0    23.5    46.0   68.5   91.0
     │█▜▘▝ ▝▌▘▝ ▌▘█ ▀█▞   █▌ ▙▘▘▝▌▙▜▐  │train_dto/min        iter
0.672┤█           ▝  ▀▌   ▀▌ ▝    ▝    │                 train_dto/std
0.657┤▜                                │      ┌────────────────────────────────┐
     └┬───────┬───────┬───────┬───────┬┘0.0228┤*         *    **   ********    │
     1.0    23.5    46.0    68.5   91.0 0.0192┤** ***** **** ***************   │
train_dto           iter                0.0119┤******** ** * *  ******** **** *│
                train_dto/mean          0.0082┤***** ****  ***  ********   ****│
      ┌────────────────────────────────┐0.0010┤*** * ****   **  ****** *    ** │
0.7200┤     ·     ·                    │      └┬───────┬───────┬──────┬───────┬┘
0.7128┤    ··    ··                    │      1.0    23.5    46.0   68.5   91.0
      │   ··· ·  ··    ·    · · · ··   │train_dto/std        iter
0.7056┤ ······· ···  ···· · ·········  │                train_dto/max
0.6985┤·········· ···· ········· ······│     ┌─────────────────────────────────┐
      │····· ····  ···  ········ ······│0.747┤     +    +++    +    ++  ++     │
0.6913┤····   ··    ··  ·····       ·· │0.737┤++ + +++  +++  ++++ ++++ +++ +   │
0.6842┤··                              │0.716┤+++++++++ ++++ + ++ ++++++ ++++ +│
      │··                              │0.706┤+++ + ++++  +++  ++++++ ++    +++│
0.6770┤ ·                              │0.685┤ +                               │
      └┬───────┬───────┬──────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
      1.0    23.5    46.0   68.5   91.0      1.0    23.5    46.0    68.5   91.0
train_dto/mean       iter               train_dto/max       iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_dto.txt
     ┌─────────────────────────────────────────────────────────────────────────┐
0.747┤ ++ train_dto/max       +                                                │
     │ -- train_dto/min      ++                        +        +              │
     │ ·· train_dto/mean    +++ +        +  +         ++  ++   ++ + +          │
     │ ▞▞ train_dto         +++++       ++ ++         ++ + + + ++++++          │
0.731┤   +  ++  +█++ +      +++▗▌      +++ ++       + ++ + +++ ++++++   +      │
     │  ++  ++  +█++++      +++▐▌      ++++ +  +   ++ +▖ + +++ + ++++  ++      │
     │ +++  ++  +█+++▟      ++ ▐▌+     +  + + ++  +++ ▐▌ + +++ + ▖+++++++      │
     │++++  ++  +█+++█      +  ▐▌+     +    + ++  +++ ▐▌ + +++ +▐▌+++   +      │
0.716┤+++ ++ + + █+++█      +▟·▐▌+ +   +    ▗ ++  +++ ▐▌ + +++ +▐▌ ++   +++   +│
     │+++ ++·· +▐▐ ++█      +█·▐▌+++   +   ·█ ++  + ++▐▌ + +▗▌ +▐▌  +   + +  +▗│
     │+++ ++·· +▐▐ +▐▐  +   ▖█·▐▌+++   +· · █+ +  + ++▌▚ +··▐▌+ ▐▌  ▖   · +  +▌│
     │+++··+·· ▟▐▐ ·▐▐ ++  ▐▌█·▌▌ +++ +· ·· █+ +  +·++▌▐+· ·▐▌+·▞▌·▐▌ ····+ +▗▘│
0.700┤++· ·+·· █▐▐··▐▐+▟+++▐▌█ ▌▐  ·+ ▗▌  · ▛▖▗▌+ ···+▌▐+▟ ·▐▌+▖▌▚·▞▐·  ·▟+++▐·│
     │▌+▗ ·· ·+▛▟▐··▞▐+█· ·▞▙▜ ▌▐····+▐▌    ▌▌▌▌++·▗▌▐ ▐▗▜ ·▌▌▐▌▌▐ ▌▝▖ ▗▌█·▗▌▞ │
     │▌+█ ·▟ ··▌█▐ ▗▘▐·█·▟ ▌█▐ ▌▐ ▗▌· ▐▌    ▌▙▘▐▗+·▐▌▐ ▐▞▐  ▌▌▐▚▌▐▐  ▌ ▐▚█·▐▌▌ │
     │▌·█▗▌▛▖ ▐-█▐ ▐-▐▗▘▌▛▄▌█▐ ▌▐ ▌▐-·▌▌   ▗▘█ ▐▛▖▟▐▌▐ ▝▌▐  ▌▌▐▐▌▝▟  ▌ ▐▐▌▌▐▐▌ │
     │▌·█▐▚▌▚▟▌ ▝▐ ▌ ▐▞ █   █▝▖▌-█ -▀▖▌▐   ▌ ▝- ▘▜▐▐▌▌ - -▌ ▌▌▐-▘ █  ▌ ▐-▘▌▐▝▌-│
0.684┤▌▗▜▞▝▌ -▘  ▝▖▌  ▘ ▜   █-▝▌-█   ▝▘▐▗ ▐-- --  ▐▌▌▌-- -▌▞-▙▘- -█ -▌ ▐ -▙▘   │
     │▌▐▐▌    -  -▝▌        ▜   -█     ▐█ ▞-- --  ▐▌▌▌-- -█--▝ ---▜--▌▞▟  ▝    │
     │▐▐▝▌        --        -    ▜     ▐▛▖▌--  -  ▐▌█  - -█---  -----▜         │
     │▐▐-                               ▘█  -      ▘█     ▝  -    - -          │
0.668┤▐▌                                 ▝          ▜                          │
     │▐▌                                                                       │
     │▐▌                                                                       │
     │▝▌                                                                       │
0.653┤ -                                                                       │
     └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
     1.0              23.5              46.0              68.5             91.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_dto_summary.txt
             train_dto/mean hist                     train_dto/max hist
    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
20.0┤                 ████             │25.0┤       ███                        │
16.7┤          ████   ████             │20.8┤       ███                        │
    │          ████   ███████          │    │       ███                        │
13.3┤          ██████████████          │16.7┤       ███                        │
10.0┤          █████████████████       │12.5┤       ███          ████          │
    │          █████████████████       │    │       ███   ████   ████   ████   │
 6.7┤          █████████████████████   │ 8.3┤       ████████████████████████   │
 3.3┤          █████████████████████   │ 4.2┤       ███████████████████████████│
    │          ████████████████████████│    │   ███████████████████████████████│
 0.0┤███    ███████████████████████████│ 0.0┤██████████████████████████████████│
    └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
   0.675   0.687    0.698   0.710 0.722    0.682   0.699    0.716   0.733 0.750
             train_dto/min hist                      train_dto/std hist
    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
28.0┤                        ███       │20.0┤   ████                           │
    │                        ███       │    │   ████                           │
23.3┤                        ███       │16.7┤   ████                           │
18.7┤                    ███████       │13.3┤   ████                           │
    │                    ███████       │    │   ████                    ████   │
14.0┤             ████   ███████       │10.0┤███████             ████   ███████│
    │             ██████████████████   │    │██████████       █████████████████│
 9.3┤             ██████████████████   │ 6.7┤██████████       █████████████████│
 4.7┤             ██████████████████   │ 3.3┤██████████████████████████████████│
    │          ████████████████████████│    │██████████████████████████████████│
 0.0┤███       ████████████████████████│ 0.0┤██████████████████████████████████│
    └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬────────┘
   0.650   0.663    0.676   0.689 0.702   -0.0000  0.0059  0.0119  0.0178
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_dto_hist.txt
                 train_loss                            train_loss/min
     ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
7.115┤▙▌                               │6.963┤  -- -  ---                      │
7.078┤█▌                               │6.920┤-- --------------------- ---- - -│
     │█▌▙     ▖                        │6.876┤ -     --    --   ----  -------- │
7.042┤▐▌▛▄   ▐▚ ▗▛▌  ▗▗    ▗▟          │6.833┤       -                  -      │
7.006┤▐▝▘▐▞▌ ▌▐▗█▌▌▗▄██▙ ▌▖█▌▌   ▟▗▖▗ ▗│     └┬───────┬───────┬───────┬───────┬┘
6.969┤▝  ▐▌▝▙▌▐▀▛▌▌▟█▌▜▝▖▙▌█▌▙▙▐▐▌▌▙▐▗▌│     1.0    23.5    46.0    68.5   91.0
     │   ▝▌ ▝ ▐  ▌▜██▌▝ ▙█▜▝▌▌█▐▌▘ ██▌ │train_loss/min      iter
6.933┤        ▐    ▜▜   ▝█   ▘▝▞▘  ▜▝▌ │                train_loss/std
6.897┤        ▝          ▜             │      ┌────────────────────────────────┐
     └┬───────┬───────┬───────┬───────┬┘0.0517┤*      *                        │
     1.0    23.5    46.0    68.5   91.0 0.0465┤**** * *      **                │
train_loss          iter                0.0362┤  ****************** * * ** * * │
               train_loss/mean          0.0310┤   *   ****  * **************** │
     ┌─────────────────────────────────┐0.0207┤                *   * *    *   *│
7.039┤···                              │      └┬───────┬───────┬──────┬───────┬┘
7.023┤···                              │      1.0    23.5    46.0   68.5   91.0
     │···  ·                           │train_loss/std       iter
7.007┤·· ···  ·                        │               train_loss/max
6.991┤    ······· ·                    │     ┌─────────────────────────────────┐
     │     ········ ····               │7.148┤+                                │
6.975┤          ·· ··········  · · ·   │7.124┤++++++                           │
6.959┤           ·   ·· ···  ··········│7.074┤++ ++++++++++  + +     +         │
     │                        ·  ·· ···│7.050┤    +  + ++++++++++++++++++++ ++ │
6.942┤                               · │7.000┤                +  +++ +  ++++  +│
     └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
     1.0    23.5    46.0    68.5   91.0      1.0    23.5    46.0    68.5   91.0
train_loss/mean     iter                train_loss/max      iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_loss.txt
     ┌─────────────────────────────────────────────────────────────────────────┐
7.148┤ ++ train_loss/max                                                       │
     │ -- train_loss/min                                                       │
     │ ·· train_loss/mean                                                      │
     │ ▞▞ train_loss                                                           │
7.096┤▌ █   +  +++                                                             │
     │▌+█   + + ++                          +                                  │
     │▚ █  ▖ ++  +  +  +   +            +  ++             +                    │
     │▐ █ ▐▚ ++   ++ ++ + + +  + ++    +++ ++            ++                 +  │
7.043┤▐▗▜ ▐▐  +       +▗ +  + +++ +  +++++ ++   +       +++     + +        ++  │
     │▝▟▐·▌·▌▗         ▛▖   +▗▌▗▜  ++  + + ++++++ ++ + +▖++    ++++     +  ++  │
     │·█▐ ▌ ▝▜        ▐ ▚    ▐▌▐▐   +  ▖▟ ▗++   ++  +▗+▐▚++ +++++++ + +++++ +  │
     │ █·▀▘ ·▐ ▞▄▟    ▐ ▐  ▗▌▐▌▐▐     ▐▌▛▖█▗▌   ▟+   █ ▐▝▖++    +  ▖ +  +    + │
6.991┤ █     ▐·▌ ▐ ···▞·▐ ·▌▌▐▌▐▝▖· ▞▜▐▌▌█▐▐▌▗  █  ▖▗▘▌▐ ▌ +      ▐▚ ▗▄   ▗   ▞│
     │ ▜      ▌▌  ▚▟·▗▌·▐·▞ ▙▘▐▌·▌· ▌▐▐▚▌▝▐▞▚▀▖ █ ▐▌▐·▐▌·▌▗▌ ▖ ▖ ▗▐ ▚▌▐   █  ▗▘│
     │        ▚▌   ▝▖▛▌ ▐▐  ▝·▐▌·▌ ▖▌▐▞▐▌ ▐▌··▚·█·▌▚▞ ▐▌·▌▐▌▐▌▐▌·▛▟ ·▘▐▗▌ █▗▚▞ │
     │       -▐▌    ▜   -█    ▐▌ ▚▀▌▌▐▌▐▌ ▝▌  ▐ █·▌▐▌ ▐▌ ▌▌▐▞▌▐▚▐ ▜···▝▟▝▟▝▟···│
     │    ---- ▘-      --█ -- ▝▌   ▚▌▐▌▝▌     ▐▐ ▌▌▝▌  ▘ ▙▘▝▌▌▐▐▌      █ ▜ █·  │
6.938┤  --  -- --  -- ---█--- - -  ▐▌▐▌  --   ▝▌ ▙▘     -▜-  ▌▐ ▘   -  █   █   │
     │---     --- -------█  -- --  ▝▌-▘ -  --   -█  -   - -  ▚▘    --  ▜   ▝  -│
     │ --       ----- ---▝   -   - --- --    - --█---  --  --  --- ---   --  - │
     │  -           - ---         -- -- -    --  ▜ - -- -  --  -- - -- --  - - │
6.885┤                --          -- --      --            --  --     ---  --  │
     │                --          -- --       -             -  --       -   -  │
     │                --           -  -                        --              │
     │                --                                        -              │
6.833┤                 -                                                       │
     └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
     1.0              23.5              46.0              68.5             91.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_loss_summary.txt
            train_loss/mean hist                    train_loss/max hist
    ┌──────────────────────────────────┐  ┌────────────────────────────────────┐
22.0┤       ███████                    │24┤    ███                             │
18.3┤       ███████                    │20┤    ███                             │
    │       ███████                    │  │    ███                             │
14.7┤       ███████                    │16┤    ███████████                     │
11.0┤   ██████████████                 │12┤    ██████████████                  │
    │   ██████████████                 │  │    ██████████████                  │
 7.3┤   █████████████████████          │ 8┤██████████████████                  │
 3.7┤   █████████████████████      ████│ 4┤██████████████████████   ████       │
    │   ████████████████████████   ████│  │████████████████████████████████    │
 0.0┤██████████████████████████████████│ 0┤████████████████████████████████████│
    └┬───────┬────────┬───────┬───────┬┘  └┬────────┬────────┬───────┬────────┬┘
   6.938   6.965    6.991   7.017 7.044  6.994    7.034    7.074   7.115  7.155
            train_loss/min hist                      train_loss/std hist
  ┌────────────────────────────────────┐    ┌──────────────────────────────────┐
24┤                     ████           │20.0┤             ████                 │
  │                     ████           │    │       ███   ████                 │
20┤                     ████████       │16.7┤       ███   ████                 │
16┤                     ████████       │13.3┤       ███   ████                 │
  │                  ███████████       │    │       ██████████   ████          │
12┤              ███████████████       │10.0┤   █████████████████████          │
  │              ███████████████       │    │   █████████████████████          │
 8┤              ██████████████████    │ 6.7┤████████████████████████          │
 4┤       █████████████████████████    │ 3.3┤███████████████████████████       │
  │████   █████████████████████████████│    │███████████████████████████   ████│
 0┤████████████████████████████████████│ 0.0┤███████████████████████████   ████│
  └┬────────┬────────┬───────┬────────┬┘    └┬───────┬────────┬───────┬────────┘
 6.827    6.863    6.898   6.934  6.969   0.0193  0.0277   0.0362  0.0446
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/plots/tplot/train_loss_hist.txt
[2025-12-31 12:16:47,448186][W][ezpz/history:2320:save_dataset] Unable to save dataset to W&B, skipping!
[2025-12-31 12:16:47,449805][I][utils/__init__:651:dataset_to_h5pyfile] Saving dataset to: /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/train_dataset.h5
[2025-12-31 12:16:47,467124][I][ezpz/history:2433:finalize] Saving history report to /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/2025-12-31-121308/2025-12-31-121642/report.md
[2025-12-31 12:16:47,473135][I][examples/vit:463:train_fn] dataset=<xarray.Dataset> Size: 26kB
Dimensions:          (draw: 91)
Coordinates:
  * draw             (draw) int64 728B 0 1 2 3 4 5 6 7 ... 84 85 86 87 88 89 90
Data variables: (12/35)
    train_iter       (draw) int64 728B 10 11 12 13 14 15 ... 95 96 97 98 99 100
    train_loss       (draw) float32 364B 7.112 7.033 6.982 ... 6.968 6.978 7.0
    train_dt         (draw) float64 728B 0.7444 0.7021 0.7178 ... 0.7456 0.7609
    train_dtd        (draw) float64 728B 0.003141 0.001719 ... 0.002238 0.001625
    train_dtf        (draw) float64 728B 0.02159 0.02241 ... 0.02394 0.02694
    train_dto        (draw) float64 728B 0.699 0.6572 0.672 ... 0.6986 0.7115
    ...               ...
    train_dto_min    (draw) float64 728B 0.6972 0.6526 0.67 ... 0.6872 0.6867
    train_dto_std    (draw) float64 728B 0.0009612 0.02281 ... 0.004004 0.01135
    train_dtb_mean   (draw) float64 728B 0.02078 0.02071 ... 0.02083 0.02072
    train_dtb_max    (draw) float64 728B 0.02156 0.02113 ... 0.02186 0.02113
    train_dtb_min    (draw) float64 728B 0.02027 0.02019 ... 0.02044 0.02029
    train_dtb_std    (draw) float64 728B 0.0003202 0.0002283 ... 0.0002136
[2025-12-31 12:16:47,618825][I][examples/vit:544:<module>] Took 218.91 seconds
wandb:
wandb: 🚀 View run snowy-hill-239 at: 
wandb: Find logs at: ../../../../../../lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20251231_121309-g19jy6bl/logs
[2025-12-31 12:16:49,364101][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2025-12-31-121649]----
[2025-12-31 12:16:49,364806][I][ezpz/launch:448:launch] Execution finished with 0.
[2025-12-31 12:16:49,365202][I][ezpz/launch:449:launch] Executing finished in 227.18 seconds.
[2025-12-31 12:16:49,365551][I][ezpz/launch:450:launch] Took 227.18 seconds to run. Exiting.
```

</details>


