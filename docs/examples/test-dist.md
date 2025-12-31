# Train MLP with DDP on MNIST

Train a simple fully connected (`torch.nn.Linear`) network using DDP
on the MNIST dataset.

See: \[📘 [docs](../python/Code-Reference/test_dist.md)\],
\[🐍 [source](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py)\]

```bash
# or, equivalently: ezpz test
ezpz launch python3 -m ezpz.test_dist
```

<details closed><summary><code>--help</code></summary>

```bash
$ python3 -m ezpz.test_dist --help
usage: test_dist.py [-h] [--warmup WARMUP] [--tp TP] [--pp PP] [--deepspeed_config DEEPSPEED_CONFIG] [--cp CP] [--backend BACKEND]
                    [--pyinstrument-profiler] [-p] [--rank-zero-only] [--pytorch-profiler-wait PYTORCH_PROFILER_WAIT]
                    [--pytorch-profiler-warmup PYTORCH_PROFILER_WARMUP] [--pytorch-profiler-active PYTORCH_PROFILER_ACTIVE]
                    [--pytorch-profiler-repeat PYTORCH_PROFILER_REPEAT] [--profile-memory] [--record-shapes] [--with-stack]
                    [--with-flops] [--with-modules] [--acc-events] [--train-iters TRAIN_ITERS] [--log-freq LOG_FREQ]
                    [--print-freq PRINT_FREQ] [--batch-size BATCH_SIZE] [--input-size INPUT_SIZE] [--output-size OUTPUT_SIZE]
                    [--layer-sizes LAYER_SIZES] [--dtype DTYPE] [--dataset DATASET] [--dataset-root DATASET_ROOT]
                    [--num-workers NUM_WORKERS] [--no-distributed-history]

ezpz test: A simple PyTorch distributed smoke test Trains a simple MLP on MNIST dataset using DDP. NOTE: `ezpz test` is a lightweight
wrapper around: `ezpz launch python3 -m ezpz.test_dist`

options:
    -h, --help            show this help message and exit
    --warmup WARMUP       Warmup iterations
    --tp TP               Tensor parallel size
    --pp PP               Pipeline length
    --deepspeed_config DEEPSPEED_CONFIG
                        Deepspeed config file
    --cp CP               Context parallel size
    --backend BACKEND     Backend (DDP, DeepSpeed, etc.)
    --pyinstrument-profiler
                        Profile the training loop
    -p, --profile         Use PyTorch profiler
    --rank-zero-only      Run profiler only on rank 0
    --pytorch-profiler-wait PYTORCH_PROFILER_WAIT
                        Wait time before starting the PyTorch profiler
    --pytorch-profiler-warmup PYTORCH_PROFILER_WARMUP
                        Warmup iterations for the PyTorch profiler
    --pytorch-profiler-active PYTORCH_PROFILER_ACTIVE
                        Active iterations for the PyTorch profiler
    --pytorch-profiler-repeat PYTORCH_PROFILER_REPEAT
                        Repeat iterations for the PyTorch profiler
    --profile-memory      Profile memory usage
    --record-shapes       Record shapes in the profiler
    --with-stack          Include stack traces in the profiler
    --with-flops          Include FLOPs in the profiler
    --with-modules        Include module information in the profiler
    --acc-events          Accumulate events in the profiler
    --train-iters TRAIN_ITERS, --train_iters TRAIN_ITERS
                        Number of training iterations
    --log-freq LOG_FREQ, --log_freq LOG_FREQ
                        Logging frequency
    --print-freq PRINT_FREQ, --print_freq PRINT_FREQ
                        Printing frequency
    --batch-size BATCH_SIZE
                        Batch size
    --input-size INPUT_SIZE
                        Input size
    --output-size OUTPUT_SIZE
                        Output size
    --layer-sizes LAYER_SIZES
                        Comma-separated list of layer sizes
    --dtype DTYPE         Data type (fp16, float16, bfloat16, bf16, float32, etc.)
    --dataset DATASET     Dataset to use for training (e.g., mnist).
    --dataset-root DATASET_ROOT
                        Directory to cache dataset downloads.
    --num-workers NUM_WORKERS
                        Number of dataloader workers to use.
    --no-distributed-history
                        Disable distributed history aggregation
```

</details>


<details closed><summary>Output on Sunspot:</summary>

```bash
#[aurora_frameworks-2025.2.0](ezpz-aurora_frameworks-2025.2.0)
#[/t/d/f/p/s/ezpz][dev][2s]
#[12/31/25 @ 11:30:29][x1921c0s7b0n0]
; ezpz test


[2025-12-31 11:30:42,775692][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2025-12-31-113042]----
[2025-12-31 11:30:43,644334][I][ezpz/launch:416:launch] Job ID: 12458339
[2025-12-31 11:30:43,645146][I][ezpz/launch:417:launch] nodelist: ['x1921c0s3b0n0', 'x1921c0s7b0n0']
[2025-12-31 11:30:43,645554][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
[2025-12-31 11:30:43,646228][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
[2025-12-31 11:30:43,647063][I][ezpz/launch:367:build_executable] Building command to execute by piecing together:
[2025-12-31 11:30:43,647473][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
[2025-12-31 11:30:43,648172][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/venvs/sunspot/ezpz-aurora_frameworks-2025.2.0/bin/python3 -m ezpz.test_dist
[2025-12-31 11:30:43,648999][I][ezpz/launch:433:launch] Took: 0.87 seconds to build command.
[2025-12-31 11:30:43,649375][I][ezpz/launch:436:launch] Executing:
mpiexec
  --envall
  --np=24
  --ppn=12
  --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
  --no-vni
  --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
  /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/venvs/sunspot/ezpz-aurora_frameworks-2025.2.0/bin/python3
  -m
  ezpz.test_dist
[2025-12-31 11:30:43,650731][I][ezpz/launch:443:launch] Execution started @ 2025-12-31-113043...
[2025-12-31 11:30:43,651220][I][ezpz/launch:139:run_command] Running command:
 mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/venvs/sunspot/ezpz-aurora_frameworks-2025.2.0/bin/python3 -m ezpz.test_dist
cpubind:list x1921c0s7b0n0 pid 82824 rank 12 0: mask 0x1c
cpubind:list x1921c0s7b0n0 pid 82825 rank 13 1: mask 0x1c00
cpubind:list x1921c0s7b0n0 pid 82826 rank 14 2: mask 0x1c0000
cpubind:list x1921c0s7b0n0 pid 82827 rank 15 3: mask 0x1c000000
cpubind:list x1921c0s7b0n0 pid 82828 rank 16 4: mask 0x1c00000000
cpubind:list x1921c0s7b0n0 pid 82829 rank 17 5: mask 0x1c0000000000
cpubind:list x1921c0s7b0n0 pid 82830 rank 18 6: mask 0x1c0000000000000
cpubind:list x1921c0s7b0n0 pid 82831 rank 19 7: mask 0x1c000000000000000
cpubind:list x1921c0s7b0n0 pid 82832 rank 20 8: mask 0x1c00000000000000000
cpubind:list x1921c0s7b0n0 pid 82833 rank 21 9: mask 0x1c0000000000000000000
cpubind:list x1921c0s7b0n0 pid 82834 rank 22 10: mask 0x1c000000000000000000000
cpubind:list x1921c0s7b0n0 pid 82835 rank 23 11: mask 0x1c00000000000000000000000
cpubind:list x1921c0s3b0n0 pid 92001 rank 0 0: mask 0x1c
cpubind:list x1921c0s3b0n0 pid 92002 rank 1 1: mask 0x1c00
cpubind:list x1921c0s3b0n0 pid 92003 rank 2 2: mask 0x1c0000
cpubind:list x1921c0s3b0n0 pid 92004 rank 3 3: mask 0x1c000000
cpubind:list x1921c0s3b0n0 pid 92005 rank 4 4: mask 0x1c00000000
cpubind:list x1921c0s3b0n0 pid 92006 rank 5 5: mask 0x1c0000000000
cpubind:list x1921c0s3b0n0 pid 92007 rank 6 6: mask 0x1c0000000000000
cpubind:list x1921c0s3b0n0 pid 92008 rank 7 7: mask 0x1c000000000000000
cpubind:list x1921c0s3b0n0 pid 92009 rank 8 8: mask 0x1c00000000000000000
cpubind:list x1921c0s3b0n0 pid 92010 rank 9 9: mask 0x1c0000000000000000000
cpubind:list x1921c0s3b0n0 pid 92011 rank 10 10: mask 0x1c000000000000000000000
cpubind:list x1921c0s3b0n0 pid 92012 rank 11 11: mask 0x1c00000000000000000000000
[2025-12-31 11:30:49,869292][I][ezpz/test_dist:132:__post_init__] Outputs will be saved to /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049
[2025-12-31 11:30:49,871638][I][ezpz/dist:1501:setup_torch_distributed] Using torch_{device,backend}= {xpu, xccl}
[2025-12-31 11:30:49,872308][I][ezpz/dist:1366:setup_torch_DDP] Caught MASTER_PORT=32935 from environment!
[2025-12-31 11:30:49,872846][I][ezpz/dist:1382:setup_torch_DDP] Using torch.distributed.init_process_group with
- master_addr='x1921c0s3b0n0'
- master_port='32935'
- world_size=24
- rank=0
- local_rank=0
- timeout=datetime.timedelta(seconds=3600)
- backend='xccl'
[2025-12-31 11:30:49,873745][I][ezpz/dist:1014:init_process_group] Calling torch.distributed.init_process_group_with: rank=0 world_size=24 backend=xccl
[2025-12-31 11:30:50,606235][I][ezpz/dist:1727:setup_torch] Using device='xpu' with backend='xccl' + 'xccl' for distributed training.
[2025-12-31 11:30:50,607054][W][ezpz/dist:544:print_dist_setup] Using [24 / 24] available "xpu" devices !!
[2025-12-31 11:30:50,607503][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=00/23][local_rank=00/11]
[2025-12-31 11:30:50,606736][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=10/23][local_rank=10/11]
[2025-12-31 11:30:50,606796][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=01/23][local_rank=01/11]
[2025-12-31 11:30:50,606796][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=02/23][local_rank=02/11]
[2025-12-31 11:30:50,606807][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=03/23][local_rank=03/11]
[2025-12-31 11:30:50,606817][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=04/23][local_rank=04/11]
[2025-12-31 11:30:50,606806][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=05/23][local_rank=05/11]
[2025-12-31 11:30:50,606747][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=06/23][local_rank=06/11]
[2025-12-31 11:30:50,606787][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=07/23][local_rank=07/11]
[2025-12-31 11:30:50,606737][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=08/23][local_rank=08/11]
[2025-12-31 11:30:50,606783][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=09/23][local_rank=09/11]
[2025-12-31 11:30:50,606793][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=11/23][local_rank=11/11]
[2025-12-31 11:30:50,610395][I][ezpz/test_dist:678:main] Took: 0.76 seconds to setup torch
[2025-12-31 11:30:50,606862][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=12/23][local_rank=00/11]
[2025-12-31 11:30:50,606858][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=16/23][local_rank=04/11]
[2025-12-31 11:30:50,606933][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=13/23][local_rank=01/11]
[2025-12-31 11:30:50,606960][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=14/23][local_rank=02/11]
[2025-12-31 11:30:50,606980][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=15/23][local_rank=03/11]
[2025-12-31 11:30:50,606972][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=17/23][local_rank=05/11]
[2025-12-31 11:30:50,606946][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=18/23][local_rank=06/11]
[2025-12-31 11:30:50,606949][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=19/23][local_rank=07/11]
[2025-12-31 11:30:50,606925][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=20/23][local_rank=08/11]
[2025-12-31 11:30:50,606960][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=21/23][local_rank=09/11]
[2025-12-31 11:30:50,606970][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=22/23][local_rank=10/11]
[2025-12-31 11:30:50,606949][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=23/23][local_rank=11/11]
[2025-12-31 11:30:50,617739][I][ezpz/test_dist:461:train] Model size: 567434 parameters
[2025-12-31 11:30:50,619069][I][ezpz/test_dist:465:train]
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
SequentialLinearNet                      --
├─Sequential: 1-1                        567,434
=================================================================
Total params: 567,434
Trainable params: 567,434
Non-trainable params: 0
=================================================================
[2025-12-31 11:30:50,620391][I][ezpz/test_dist:473:train] Took: 0.006357558071613312 seconds to build model
[2025-12-31 11:30:50,622656][I][ezpz/test_dist:601:build_model_and_optimizer] model=
SequentialLinearNet(
  (layers): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): ReLU()
    (4): Linear(in_features=256, out_features=128, bias=True)
    (5): ReLU()
    (6): Linear(in_features=128, out_features=10, bias=True)
  )
)
[2025-12-31 11:30:50,624614][I][ezpz/dist:685:wrap_model] Wrapping model with: ddp
2025:12:31-11:30:50:(92001) |CCL_WARN| value of CCL_OP_SYNC changed to be 1 (default:0)
2025:12:31-11:30:50:(92001) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
[2025-12-31 11:31:03,431773][I][ezpz/test_dist:479:train] Took: 12.81 seconds to build optimizer
[2025-12-31 11:31:03,484655][I][ezpz/history:220:__init__] Using History with distributed_history=True
[2025-12-31 11:31:03,487406][I][ezpz/dist:2039:setup_wandb] Setting up wandb from rank=0
[2025-12-31 11:31:03,487916][I][ezpz/dist:2040:setup_wandb] Using WB_PROJECT=ezpz.test_dist
wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.23.1
wandb: Run data is saved locally in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20251231_113103-cppqal9m
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run polar-surf-6861
wandb:  View project at https://wandb.ai/aurora_gpt/ezpz.test_dist
wandb:  View run at https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/cppqal9m
[2025-12-31 11:31:05,498721][I][ezpz/dist:2069:setup_wandb] wandb.run=[polar-surf-6861](https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/cppqal9m)
[2025-12-31 11:31:05,504592][I][ezpz/dist:2112:setup_wandb] Running on machine='SunSpot'
[2025-12-31 11:31:05,852704][I][ezpz/test_dist:482:train] Took: 2.42 seconds to build trainer
[2025-12-31 11:31:05,853930][I][ezpz/test_dist:486:train] config:
{
  "acc_events": false,
  "backend": "DDP",
  "batch_size": 128,
  "cp": 1,
  "dataset": "mnist",
  "dataset_root": "/lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/datasets/mnist",
  "dtype": "bf16",
  "input_size": 784,
  "layer_sizes": [
    512,
    256,
    128
  ],
  "log_freq": 1,
  "no_distributed_history": false,
  "num_workers": 0,
  "output_size": 10,
  "pp": 1,
  "print_freq": 10,
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
  "train_iters": 200,
  "warmup": 5,
  "with_flops": true,
  "with_modules": true,
  "with_stack": true
}
[2025-12-31 11:31:05,856304][I][ezpz/test_dist:488:train] Took: 19.73 to get here.
[2025-12-31 11:31:20,012644][I][ezpz/test_dist:369:train] Warmup complete at step 5
[2025-12-31 11:31:20,317949][I][ezpz/test_dist:325:train_step] iter=10 loss=1.046174 accuracy=0.726562 dtf=0.010201 dtb=0.001751 loss/mean=1.094917 loss/max=1.271076 loss/min=0.961706 loss/std=0.078834 accuracy/mean=0.704427 accuracy/max=0.765625 accuracy/min=0.593750 accuracy/std=0.042472 dtf/mean=0.010742 dtf/max=0.012226 dtf/min=0.010026 dtf/std=0.000653 dtb/mean=0.001594 dtb/max=0.001852 dtb/min=0.001306 dtb/std=0.000177
[2025-12-31 11:31:21,122273][I][ezpz/test_dist:325:train_step] iter=20 loss=0.931834 accuracy=0.779412 dtf=0.005889 dtb=0.178909 loss/mean=0.592798 loss/max=0.931834 loss/min=0.390783 loss/std=0.136802 accuracy/mean=0.817402 accuracy/max=0.897059 accuracy/min=0.691176 accuracy/std=0.050930 dtf/mean=0.006413 dtf/max=0.006798 dtf/min=0.005839 dtf/std=0.000321 dtb/mean=0.204767 dtb/max=0.234498 dtb/min=0.178215 dtb/std=0.020612
[2025-12-31 11:31:21,659906][I][ezpz/test_dist:325:train_step] iter=30 loss=0.500784 accuracy=0.851562 dtf=0.009988 dtb=0.001570 loss/mean=0.459434 loss/max=0.755573 loss/min=0.280539 loss/std=0.115654 accuracy/mean=0.861003 accuracy/max=0.937500 accuracy/min=0.773438 accuracy/std=0.038405 dtf/mean=0.010492 dtf/max=0.011835 dtf/min=0.009957 dtf/std=0.000546 dtb/mean=0.001607 dtb/max=0.001853 dtb/min=0.001314 dtb/std=0.000157
[2025-12-31 11:31:22,283836][I][ezpz/test_dist:325:train_step] iter=40 loss=0.478971 accuracy=0.867647 dtf=0.005750 dtb=0.001340 loss/mean=0.319425 loss/max=0.549011 loss/min=0.172847 loss/std=0.095734 accuracy/mean=0.903799 accuracy/max=0.970588 accuracy/min=0.823529 accuracy/std=0.040494 dtf/mean=0.006246 dtf/max=0.007002 dtf/min=0.005576 dtf/std=0.000431 dtb/mean=0.001377 dtb/max=0.001588 dtb/min=0.001080 dtb/std=0.000155
[2025-12-31 11:31:22,834823][I][ezpz/test_dist:325:train_step] iter=50 loss=0.349907 accuracy=0.875000 dtf=0.010030 dtb=0.001433 loss/mean=0.298854 loss/max=0.401245 loss/min=0.211164 loss/std=0.060323 accuracy/mean=0.910807 accuracy/max=0.953125 accuracy/min=0.867188 accuracy/std=0.024594 dtf/mean=0.010629 dtf/max=0.011482 dtf/min=0.009914 dtf/std=0.000466 dtb/mean=0.001609 dtb/max=0.001820 dtb/min=0.001307 dtb/std=0.000167
[2025-12-31 11:31:23,512006][I][ezpz/test_dist:325:train_step] iter=60 loss=0.313367 accuracy=0.926471 dtf=0.005733 dtb=0.001325 loss/mean=0.182718 loss/max=0.313367 loss/min=0.097909 loss/std=0.047081 accuracy/mean=0.952206 accuracy/max=0.985294 accuracy/min=0.911765 accuracy/std=0.018628 dtf/mean=0.006096 dtf/max=0.006441 dtf/min=0.005576 dtf/std=0.000266 dtb/mean=0.001393 dtb/max=0.001545 dtb/min=0.001141 dtb/std=0.000121
[2025-12-31 11:31:24,034381][I][ezpz/test_dist:325:train_step] iter=70 loss=0.262147 accuracy=0.929688 dtf=0.009596 dtb=0.001584 loss/mean=0.216635 loss/max=0.327601 loss/min=0.138058 loss/std=0.060615 accuracy/mean=0.938477 accuracy/max=0.968750 accuracy/min=0.890625 accuracy/std=0.025447 dtf/mean=0.009859 dtf/max=0.011898 dtf/min=0.009397 dtf/std=0.000751 dtb/mean=0.001568 dtb/max=0.001724 dtb/min=0.001312 dtb/std=0.000131
[2025-12-31 11:31:24,636021][I][ezpz/test_dist:325:train_step] iter=80 loss=0.156417 accuracy=0.955882 dtf=0.005665 dtb=0.001253 loss/mean=0.111892 loss/max=0.192739 loss/min=0.060052 loss/std=0.033597 accuracy/mean=0.971814 accuracy/max=1.000000 accuracy/min=0.941176 accuracy/std=0.012681 dtf/mean=0.006088 dtf/max=0.006498 dtf/min=0.005522 dtf/std=0.000325 dtb/mean=0.001360 dtb/max=0.001627 dtb/min=0.001064 dtb/std=0.000163
[2025-12-31 11:31:25,102420][I][ezpz/test_dist:325:train_step] iter=90 loss=0.195402 accuracy=0.937500 dtf=0.009428 dtb=0.001697 loss/mean=0.154431 loss/max=0.232959 loss/min=0.095484 loss/std=0.043264 accuracy/mean=0.954102 accuracy/max=0.976562 accuracy/min=0.921875 accuracy/std=0.017222 dtf/mean=0.010213 dtf/max=0.012565 dtf/min=0.009396 dtf/std=0.000995 dtb/mean=0.001599 dtb/max=0.001942 dtb/min=0.001136 dtb/std=0.000188
[2025-12-31 11:31:25,595109][I][ezpz/test_dist:325:train_step] iter=100 loss=0.079676 accuracy=1.000000 dtf=0.005687 dtb=0.001371 loss/mean=0.076284 loss/max=0.154268 loss/min=0.046152 loss/std=0.028206 accuracy/mean=0.987132 accuracy/max=1.000000 accuracy/min=0.955882 accuracy/std=0.012246 dtf/mean=0.006011 dtf/max=0.007985 dtf/min=0.005620 dtf/std=0.000624 dtb/mean=0.001366 dtb/max=0.001765 dtb/min=0.001072 dtb/std=0.000164
[2025-12-31 11:31:26,171868][I][ezpz/test_dist:325:train_step] iter=110 loss=0.154822 accuracy=0.960938 dtf=0.009538 dtb=0.001698 loss/mean=0.115471 loss/max=0.210122 loss/min=0.066416 loss/std=0.037737 accuracy/mean=0.969401 accuracy/max=0.984375 accuracy/min=0.937500 accuracy/std=0.013327 dtf/mean=0.009814 dtf/max=0.012341 dtf/min=0.009225 dtf/std=0.000823 dtb/mean=0.001554 dtb/max=0.001722 dtb/min=0.001096 dtb/std=0.000190
[2025-12-31 11:31:26,733522][I][ezpz/test_dist:325:train_step] iter=120 loss=0.053370 accuracy=0.985294 dtf=0.005611 dtb=0.001238 loss/mean=0.056127 loss/max=0.126032 loss/min=0.025829 loss/std=0.023853 accuracy/mean=0.990196 accuracy/max=1.000000 accuracy/min=0.955882 accuracy/std=0.013197 dtf/mean=0.006132 dtf/max=0.006825 dtf/min=0.005512 dtf/std=0.000383 dtb/mean=0.001366 dtb/max=0.001595 dtb/min=0.001062 dtb/std=0.000149
[2025-12-31 11:31:27,304428][I][ezpz/test_dist:325:train_step] iter=130 loss=0.106341 accuracy=0.976562 dtf=0.009662 dtb=0.001578 loss/mean=0.087535 loss/max=0.152122 loss/min=0.041659 loss/std=0.031179 accuracy/mean=0.979818 accuracy/max=1.000000 accuracy/min=0.953125 accuracy/std=0.010792 dtf/mean=0.010319 dtf/max=0.010932 dtf/min=0.009599 dtf/std=0.000411 dtb/mean=0.001625 dtb/max=0.001806 dtb/min=0.001306 dtb/std=0.000155
[2025-12-31 11:31:27,837406][I][ezpz/test_dist:325:train_step] iter=140 loss=0.038179 accuracy=1.000000 dtf=0.005624 dtb=0.001283 loss/mean=0.031276 loss/max=0.058039 loss/min=0.018561 loss/std=0.009414 accuracy/mean=1.000000 accuracy/max=1.000000 accuracy/min=1.000000 accuracy/std=0.000000 dtf/mean=0.006149 dtf/max=0.006750 dtf/min=0.005484 dtf/std=0.000363 dtb/mean=0.001352 dtb/max=0.001624 dtb/min=0.001071 dtb/std=0.000156
[2025-12-31 11:31:28,429546][I][ezpz/test_dist:325:train_step] iter=150 loss=0.075107 accuracy=0.976562 dtf=0.009395 dtb=0.001523 loss/mean=0.078655 loss/max=0.160285 loss/min=0.043498 loss/std=0.028539 accuracy/mean=0.978516 accuracy/max=1.000000 accuracy/min=0.945312 accuracy/std=0.014131 dtf/mean=0.010171 dtf/max=0.010841 dtf/min=0.009276 dtf/std=0.000518 dtb/mean=0.001609 dtb/max=0.001818 dtb/min=0.001068 dtb/std=0.000180
[2025-12-31 11:31:28,948358][I][ezpz/test_dist:325:train_step] iter=160 loss=0.039511 accuracy=0.985294 dtf=0.005642 dtb=0.001399 loss/mean=0.038084 loss/max=0.072799 loss/min=0.011212 loss/std=0.016665 accuracy/mean=0.993873 accuracy/max=1.000000 accuracy/min=0.970588 accuracy/std=0.008404 dtf/mean=0.006147 dtf/max=0.006805 dtf/min=0.005509 dtf/std=0.000402 dtb/mean=0.001383 dtb/max=0.001682 dtb/min=0.001094 dtb/std=0.000159
[2025-12-31 11:31:29,524806][I][ezpz/test_dist:325:train_step] iter=170 loss=0.090782 accuracy=0.968750 dtf=0.009549 dtb=0.001483 loss/mean=0.063093 loss/max=0.135812 loss/min=0.029736 loss/std=0.026772 accuracy/mean=0.984375 accuracy/max=1.000000 accuracy/min=0.960938 accuracy/std=0.011935 dtf/mean=0.010131 dtf/max=0.010809 dtf/min=0.009328 dtf/std=0.000468 dtb/mean=0.001589 dtb/max=0.001801 dtb/min=0.001083 dtb/std=0.000189
[2025-12-31 11:31:30,100256][I][ezpz/test_dist:325:train_step] iter=180 loss=0.028730 accuracy=1.000000 dtf=0.005630 dtb=0.001255 loss/mean=0.031807 loss/max=0.089583 loss/min=0.009972 loss/std=0.017174 accuracy/mean=0.995098 accuracy/max=1.000000 accuracy/min=0.970588 accuracy/std=0.008130 dtf/mean=0.006067 dtf/max=0.006434 dtf/min=0.005511 dtf/std=0.000292 dtb/mean=0.001388 dtb/max=0.001594 dtb/min=0.001066 dtb/std=0.000147
[2025-12-31 11:31:30,617119][I][ezpz/test_dist:325:train_step] iter=190 loss=0.044844 accuracy=0.984375 dtf=0.009522 dtb=0.001602 loss/mean=0.051969 loss/max=0.151458 loss/min=0.025844 loss/std=0.027686 accuracy/mean=0.985677 accuracy/max=1.000000 accuracy/min=0.953125 accuracy/std=0.011423 dtf/mean=0.010067 dtf/max=0.012187 dtf/min=0.009460 dtf/std=0.000770 dtb/mean=0.001618 dtb/max=0.001843 dtb/min=0.001297 dtb/std=0.000141
[2025-12-31 11:31:32,515303][I][ezpz/history:2385:finalize] Saving plots to /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049/plots/mplot (matplotlib) and /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049/plots/tplot (tplot)
                  accuracy                              accuracy/min
     ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
1.000┤              ▟▗▙▟▄▄▙▟▟▟▄▙▄▟▄███▛│1.000┤            ---------------------│
0.931┤       ▖▄▄▟▟██▛▛▛▀▝▐▀▘▀▀ ▝ ▝ ▘   │0.826┤   -------------      -  -       │
     │   ▖▗▄██▜▀▀▀ ▘                   │0.651┤----                             │
0.862┤   ▌█▘▝▘                         │0.477┤-                                │
0.793┤ ▌▗█▝                            │     └┬───────┬───────┬───────┬───────┬┘
0.724┤▗█▌                              │     1.0    49.2    97.5    145.8 194.0
     │▐█▌                              │accuracy/min        iter
0.655┤█▌▘                              │                accuracy/std
0.586┤▀▌                               │     ┌─────────────────────────────────┐
     └┬───────┬───────┬───────┬───────┬┘0.067┤**                               │
     1.0    49.2    97.5    145.8 194.0 0.056┤****                             │
accuracy            iter                0.034┤ ************         *          │
                accuracy/mean           0.022┤       **************************│
     ┌─────────────────────────────────┐0.000┤                      *   *  *   │
1.000┤                ·  ··············│     └┬───────┬───────┬───────┬───────┬┘
0.928┤         ············ ····       │     1.0    49.2    97.5    145.8 194.0
     │      ·····                      │accuracy/std        iter
0.856┤   ···                           │                accuracy/max
0.784┤  ··                             │     ┌─────────────────────────────────┐
     │ ··                              │1.000┤      + +++++++++++++++++++++++++│
0.712┤ ·                               │0.952┤  +++++++++                      │
0.640┤·                                │0.855┤ +++                             │
     │·                                │0.807┤ +                               │
0.568┤·                                │0.711┤+                                │
     └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
     1.0    49.2    97.5    145.8 194.0      1.0    49.2    97.5    145.8 194.0
accuracy/mean       iter                accuracy/max        iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049/plots/tplot/accuracy.txt
     ┌─────────────────────────────────────────────────────────────────────────┐
1.000┤ ++ accuracy/max        +  ++  +▗++▟++▗++++▖▗++▐++▟+▗++▖+·▖+▗▌++▗▟+▟▗▗▌▗▀│
     │ -- accuracy/min  ++++++++++·++▌▟+▙█▞▖▟▗▀▖▟▙▜▗▜▞▟▌█▗▛▞▀▀▜▞▚▀▀▙▞▀▌▜▞·▀▀▀▘·│
     │ ·· accuracy/mean ++·++▗ ▌ ▄▙▌▄▌█·▛█▌▚▌▜-▜-█▝▟▝▌-▚▀▌▌----▘---▌--▘--- --- │
     │ ▞▞ accuracy      +▖▗·▙▛▄▌▐██▜ ▝-▀▘▐▌------▜------ - ----- ---- -  --  --│
0.913┤        +++ +  ▗▌▐▖▙▀▖█▌▜▝▞▝▝------ ▘-   - -  -  -       -      -        │
     │    ++++  ▖  ▙▟▟▌▞▚▀-▜-▘--- --    -              -       -               │
     │    + ▗▌ ▐▙▟▄▌▐▝▛-------  -                                              │
     │  +++ ▐▌▗▀▌▘  ▝-▘- -      -                                              │
0.826┤  +++ ▐▌▐ ▌- ---                                                         │
     │ + + ·▐▌▞-▌-----                                                         │
     │ +▟ ·▗▟▚▌-- -  -                                                         │
     │ +█··▌ - --                                                              │
0.738┤ +█▗▌▌--                                                                 │
     │+▗▌▘█ -                                                                  │
     │+█▌ ▌-                                                                   │
     │ █▌ ▌-                                                                   │
     │▐█▌ ▌-                                                                   │
0.651┤▐█▌-                                                                     │
     │▐▐▌-                                                                     │
     │▘▐▌                                                                      │
     │·-▘                                                                      │
0.564┤·-                                                                       │
     │ -                                                                       │
     │ -                                                                       │
     │ -                                                                       │
0.477┤--                                                                       │
     └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
     1.0              49.2              97.5              145.8           194.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049/plots/tplot/accuracy_summary.txt
            accuracy/mean hist                       accuracy/max hist
   ┌───────────────────────────────────┐   ┌───────────────────────────────────┐
108┤                               ████│132┤                               ████│
 90┤                               ████│110┤                               ████│
   │                               ████│   │                               ████│
 72┤                               ████│ 88┤                               ████│
 54┤                               ████│ 66┤                               ████│
   │                               ████│   │                               ████│
 36┤                            ███████│ 44┤                               ████│
 18┤                        ███████████│ 22┤                            ███████│
   │              █████████████████████│   │                        ███████████│
  0┤███████████████████████████████████│  0┤███████   █████████████████████████│
   └┬────────┬───────┬────────┬───────┬┘   └┬────────┬───────┬────────┬───────┬┘
  0.55     0.67    0.78     0.90   1.02   0.698    0.777   0.855    0.934 1.013
              accuracy/min hist                       accuracy/std hist
    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
85.0┤                           ████   │73.0┤       ███                        │
    │                           ████   │    │       ███                        │
70.8┤                           ████   │60.8┤       ███                        │
56.7┤                           ████   │48.7┤   ███████                        │
    │                           ████   │    │   ███████                        │
42.5┤                           ███████│36.5┤   ███████                        │
    │                           ███████│    │   ███████████                    │
28.3┤                        ██████████│24.3┤   ██████████████                 │
14.2┤                    ██████████████│12.2┤   ██████████████                 │
    │          ████   █████████████████│    │   ████████████████████████       │
 0.0┤██████████████████████████████████│ 0.0┤██████████████████████████████████│
    └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
   0.45    0.60     0.74    0.88   1.02   -0.003   0.015    0.034   0.052 0.070
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049/plots/tplot/accuracy_hist.txt
                     dtb                                   dtb/min
     ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
0.179┤  ▟                              │0.178┤  -                              │
0.149┤  █                              │0.119┤  -                              │
     │  █                              │0.060┤  -                              │
0.120┤  █                              │0.001┤---------------------------------│
0.090┤  █                              │     └┬───────┬───────┬───────┬───────┬┘
0.060┤  █                              │     1.0    49.2    97.5    145.8 194.0
     │  █                              │dtb/min             iter
0.031┤  █                              │                    dtb/std
0.001┤▄▄█▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│      ┌────────────────────────────────┐
     └┬───────┬───────┬───────┬───────┬┘0.0206┤  *                             │
     1.0    49.2    97.5    145.8 194.0 0.0172┤  *                             │
dtb                 iter                0.0103┤  *                             │
                  dtb/mean              0.0069┤  *                             │
     ┌─────────────────────────────────┐0.0001┤********************************│
0.205┤  ·                              │      └┬───────┬───────┬──────┬───────┬┘
0.171┤  ·                              │      1.0    49.2    97.5   145.8 194.0
     │  ·                              │dtb/std              iter
0.137┤  ·                              │                   dtb/max
0.103┤  ·                              │     ┌─────────────────────────────────┐
     │  ·                              │0.234┤  +                              │
0.069┤  ·                              │0.196┤  +                              │
0.035┤  ·                              │0.118┤  +                              │
     │  ·                              │0.079┤  +                              │
0.001┤·································│0.002┤+++++++++++++++++++++++++++++++++│
     └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
     1.0    49.2    97.5    145.8 194.0      1.0    49.2    97.5    145.8 194.0
dtb/mean            iter                dtb/max             iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049/plots/tplot/dtb.txt
     ┌─────────────────────────────────────────────────────────────────────────┐
0.234┤ ++ dtb/max                                                              │
     │ -- dtb/min                                                              │
     │ ·· dtb/mean                                                             │
     │ ▞▞ dtb                                                                  │
0.196┤     ·                                                                   │
     │     ·                                                                   │
     │     ▟                                                                   │
     │     █                                                                   │
0.157┤     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
0.118┤     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
0.079┤     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
0.040┤     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
     │     █                                                                   │
0.001┤▄▄▄▄▄█▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
     └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
     1.0              49.2              97.5              145.8           194.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049/plots/tplot/dtb_summary.txt
                dtb/mean hist                           dtb/max hist
     ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
193.0┤████                             │193.0┤████                             │
160.8┤████                             │160.8┤████                             │
     │████                             │     │████                             │
128.7┤████                             │128.7┤████                             │
 96.5┤████                             │ 96.5┤████                             │
     │████                             │     │████                             │
 64.3┤████                             │ 64.3┤████                             │
 32.2┤████                             │ 32.2┤████                             │
     │████                             │     │████                             │
  0.0┤███                          ████│  0.0┤███                          ████│
     └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
   -0.008   0.048   0.103   0.158 0.214    -0.009   0.055   0.118   0.181 0.245
                dtb/min hist                            dtb/std hist
     ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
193.0┤████                             │193.0┤████                             │
     │████                             │     │████                             │
160.8┤████                             │160.8┤████                             │
128.7┤████                             │128.7┤████                             │
     │████                             │     │████                             │
 96.5┤████                             │ 96.5┤████                             │
     │████                             │     │████                             │
 64.3┤████                             │ 64.3┤████                             │
 32.2┤████                             │ 32.2┤████                             │
     │████                             │     │████                             │
  0.0┤███                          ████│  0.0┤███                          ████│
     └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬────────┘
   -0.007   0.041   0.090   0.138 0.186    -0.0008  0.0048  0.0103  0.0159
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049/plots/tplot/dtb_hist.txt
                      dtf                                   dtf/min
      ┌────────────────────────────────┐      ┌────────────────────────────────┐
0.0122┤  ▐                             │0.0115┤  -   -  -  -  -  -   -  -  -   │
0.0111┤  ▐  ▐         ▐  ▗▌  ▖  ▟  ▗   │0.0095┤--------------------------------│
      │▙ ▐ ▖▐   ▌  ▌▖ █  ▐▌  ▌  █▖▖▐ ▗▖│0.0075┤  -  -   -  -  -  -  --  -  -   │
0.0100┤▛▙█▙▙▟▟▙▄▙█▟▙▙▟█▄▙▟▙▄▟▙▄▙█▙▙▟▄█▙│0.0055┤  -  -   -  -  -  -   -  -  -   │
0.0089┤  █▝ ▜ ▝▐▌  ▌  █  ▐  ▐▌▘▝▌  █  ▀│      └┬───────┬───────┬──────┬───────┬┘
0.0078┤  █  ▐  ▐▌  ▌  █  ▐  ▐▌  ▌  █   │      1.0    49.2    97.5   145.8 194.0
      │  █  ▐  ▐▌  ▌  █  ▐  ▐▌  ▌  █   │dtf/min              iter
0.0067┤  █  ▐  ▐▌  ▌  █  ▐  ▐▌  ▌  █   │                    dtf/std
0.0056┤  ▝  ▐  ▝▌  ▌  ▜  ▐  ▝▌  ▌  ▜   │       ┌───────────────────────────────┐
      └┬───────┬───────┬──────┬───────┬┘0.00104┤     * **  * * *       **** * *│
      1.0    49.2    97.5   145.8 194.0 0.00091┤ ** ****** ********************│
dtf                  iter               0.00065┤*******************************│
                   dtf/mean             0.00052┤*******************************│
      ┌────────────────────────────────┐0.00027┤ ** *   **  *   *    * *** *** │
0.0123┤  ·                      ·      │       └┬───────┬──────┬───────┬───────┘
0.0113┤  ·   ·  ·  ·  ·  ·   ·  ·  ·   │       1.0    49.2   97.5    145.8
      │···· ··· · ·····  ·  ··  ·· · · │dtf/std              iter
0.0102┤································│                    dtf/max
0.0092┤······  ·····  · ·· ···· ·  ··· │      ┌────────────────────────────────┐
      │  ·  ·   ·  ·  ·  ·  ··  ·  ·   │0.0144┤  +   +  +        +   +     +   │
0.0081┤  ·  ·   ·  ·  ·  ·  ··  ·  ·   │0.0131┤++++++++++++++++++++++++++++++++│
0.0071┤  ·  ·   ·  ·  ·  ·  ··  ·  ·   │0.0104┤ +++++++++++++++++++++++++++++++│
      │  ·  ·   ·  ·  ·  ·  ··  ·  ·   │0.0091┤  +  +   +  +  +  +  ++  +  +   │
0.0060┤     ·   ·  ·  ·  ·   ·  ·  ·   │0.0064┤  +  +   +  +     +   +  +  +   │
      └┬───────┬───────┬──────┬───────┬┘      └┬───────┬───────┬──────┬───────┬┘
      1.0    49.2    97.5   145.8 194.0       1.0    49.2    97.5   145.8 194.0
dtf/mean             iter               dtf/max              iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049/plots/tplot/dtf.txt
      ┌────────────────────────────────────────────────────────────────────────┐
0.0144┤ ++ dtf/max  +                            +                             │
      │ -- dtf/min  +      +                     +       +                     │
      │ ·· dtf/mean +      +                     +      ++             +       │
      │ ▞▞ dtf      +      +                     +      ++      +      +       │
0.0129┤     ++      + +    +       +   ++ +   ++ +  +   ++     ++   +  +       │
      │     ++    + +++ + +++    ++++ +++ ++  ++ + ++   ++    +++ +++ ++  +  + │
      │ ++++▗·+++++++++ +++++ +++++++++++ +++++++++++++++++++++++++++++++++++++│
      │ ++++█·+++++++++++++++++++++++++++ ++++++++++++++++++++++·++++++++++++++│
0.0115┤+ +++█·++++++▖++++++++++++++++++++ ·++++++▗+++++++·++++++▟++++++++++++++│
      │  +++█·+++++▐▌ +++++·+++++++·++++++▌++++++█+ ++++▗▌++++++█++++++·▖++++++│
      │▖·+++█··+ + ▐▌  ++++▗++ ++·▗▌+·+ ·+▌+ ++ +█+  + +▐▌ +++++█+·++++▐▌+++++ │
      │▌·   █·· ▗  ▐▌· ·  +█ · ···▐▌··  ·▟▌   + ·█+     ▐▌   ·++█·▗  + ▐▌++ · ▖│
0.0100┤█▜···█···█··▐▌▗·▗···█······▐▌▗▌··▗█▌···▖·▗█······▐▌······█·█·▌··▐▌···▐▐▌│
      │█▐▖▖▐▌▙▌▐▐··▟▌█▖█···█▗▟▗▌▗▌▐▌▐▌▗ ███· ▐▌▗██···▄·▗█▌··▗▌▖·█▖█▖▌  ▐▌··▗▐▐▌│
      │▝-▀▚▞▌▝▝█▝▟▟█▐▝▙▜▞▞▜▛▛▛▟▚█▐█▚▜▐▛▞▌▜▛▞▀▞▝▛▌▌▜▄█▝▀▞▐▝▚▚▟▜▛▚▛▛▀▚▀▄▜▟▚▟▄▛▛▟▙│
      │     ▌  -  -█·     ▐▌  -   █      ▐▌  --  ▌   ---▐-- -  ▐▌  -   █  -   -│
      │     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
0.0085┤     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
      │     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
      │     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
      │     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
0.0070┤     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
      │     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
      │     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
      │     ▌      ▜-     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
0.0055┤             -      ▘      ▝       ▘      ▘      ▝       ▘      ▝       │
      └┬─────────────────┬─────────────────┬────────────────┬─────────────────┬┘
      1.0              49.2              97.5             145.8           194.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049/plots/tplot/dtf_summary.txt
                dtf/mean hist                           dtf/max hist
     ┌─────────────────────────────────┐    ┌──────────────────────────────────┐
152.0┤                    ███          │75.0┤                        ███       │
126.7┤                    ███          │62.5┤                 ████   ███       │
     │                    ███          │    │                 ████   ███       │
101.3┤                    ███          │50.0┤                 ████   ███       │
 76.0┤                    ███          │37.5┤                 ████   ███       │
     │                    ███          │    │                 ████   ███       │
 50.7┤                    ███          │25.0┤                 ██████████       │
 25.3┤                    ███          │12.5┤                 ██████████████   │
     │████                ██████       │    │████             █████████████████│
  0.0┤███             █████████████████│ 0.0┤███████          █████████████████│
     └┬───────┬───────┬───────┬────────┘    └┬───────┬────────┬───────┬────────┘
   0.0057  0.0074  0.0092  0.0109         0.0061  0.0083   0.0104  0.0126
               dtf/min hist                             dtf/std hist
   ┌───────────────────────────────────┐    ┌──────────────────────────────────┐
156┤                     ████          │31.0┤       ███                        │
   │                     ████          │    │       ███                        │
130┤                     ████          │25.8┤       ███              ███       │
104┤                     ████          │20.7┤   ███████              ███████   │
   │                     ████          │    │   ██████████████       ███████   │
 78┤                     ████          │15.5┤   ███████████████████████████████│
   │                     ████          │    │██████████████████████████████████│
 52┤                     ████          │10.3┤██████████████████████████████████│
 26┤                     ████          │ 5.2┤██████████████████████████████████│
   │████                 ███████       │    │██████████████████████████████████│
  0┤███                  ██████████████│ 0.0┤██████████████████████████████████│
   └┬────────┬───────┬────────┬────────┘    └┬───────┬────────┬───────┬────────┘
  0.0052  0.0069  0.0085   0.0101         0.00023  0.00044  0.00065 0.00086
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049/plots/tplot/dtf_hist.txt
                    loss                                  loss/min
    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
1.75┤▌                                 │1.66┤-                                 │
1.46┤▌                                 │1.11┤--                                │
    │▐                                 │0.56┤ -----                            │
1.17┤▐▌                                │0.01┤    ------------------------------│
0.89┤ ▙█                               │    └┬───────┬────────┬───────┬───────┬┘
0.60┤ ▝▜▄                              │    1.0    49.2     97.5    145.8 194.0
    │   ▀█▄▖▖▖                         │loss/min            iter
0.31┤    ▝▘▝▀▛▀▙▟▄▙▄▄▗   ▖  ▗          │                  loss/std
0.02┤           ▝ ▀▀▀▀▀▀▀▛▜▛▜▜█▜▜▟█▟▄▄▄│     ┌─────────────────────────────────┐
    └┬───────┬────────┬───────┬───────┬┘0.137┤  *                              │
    1.0    49.2     97.5    145.8 194.0 0.116┤ ******                          │
loss                iter                0.073┤** **********                    │
                  loss/mean             0.052┤*      **************************│
    ┌──────────────────────────────────┐0.009┤                  *** * ******** │
1.74┤·                                 │     └┬───────┬───────┬───────┬───────┬┘
1.46┤·                                 │     1.0    49.2    97.5    145.8 194.0
    │·                                 │loss/std            iter
1.17┤ ·                                │                  loss/max
0.89┤ ·                                │    ┌──────────────────────────────────┐
    │ ··                               │1.83┤+                                 │
0.60┤  ···                             │1.53┤++                                │
0.32┤    ····                          │0.94┤ ++++                             │
    │       ···········                │0.65┤    +++++++++++                   │
0.03┤             ·  ··················│0.06┤           +++++++++++++++++++++++│
    └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
    1.0    49.2     97.5    145.8 194.0     1.0    49.2     97.5    145.8 194.0
loss/mean           iter                loss/max            iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049/plots/tplot/loss.txt
    ┌──────────────────────────────────────────────────────────────────────────┐
1.83┤ ++ loss/max                                                              │
    │ -- loss/min                                                              │
    │ ·· loss/mean                                                             │
    │ ▞▞ loss                                                                  │
1.53┤▐+                                                                        │
    │▐+                                                                        │
    │▐·                                                                        │
    │ ▌                                                                        │
1.22┤ ▌+                                                                       │
    │ ▌+                                                                       │
    │ ▙▌                                                                       │
    │ ▝▌+                                                                      │
0.92┤  ▌+ ▗                                                                    │
    │  ▌ ▌█ +                                                                  │
    │  ▐▌▌█+++                                                                 │
    │  -▝██+ ++                                                                │
    │   -▝█▖  ++                                                               │
0.62┤   --·▚▗▌ +++   +                                                         │
    │    --▐▞▚▟ ++++++                                                         │
    │    ---▘▝█▗▗▙▚+++▌     +                                                  │
    │     ----▝▌▜▝▝▄▄ ▌+++▖++++ +                                              │
0.31┤        ---- ··▀▞▚▞▚▛▐▖++▖++▗ ++  +                                       │
    │         -  -----▝▌··▝▚▙▞▙▚▞▌▌++▗+++▄+++++++▖ +   +     + + +             │
    │             - -----·---▘█··▌▜▀▙▜▞▚▖█▗▄▗▖ ▗▗▌+▖▗▗++▗▌++++++▗+++++▖  ++++++│
    │                    - ---▝--·----▌-▝▌▀·▌▚▀▀█▝▟▚▛▟▀▟▌▚▟▗▙▗▄▄▛▄▄▚▖▟▌▄▌·▄▗···│
0.01┤                                    -     -▝---▘---▘--▘-▘-▝--▝-▝▝▝▝▝▀▝▘▀▀▞│
    └┬─────────────────┬──────────────────┬─────────────────┬─────────────────┬┘
    1.0              49.2               97.5              145.8           194.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049/plots/tplot/loss_summary.txt
               loss/mean hist                           loss/max hist
     ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
127.0┤████                             │107.0┤████                             │
105.8┤████                             │ 89.2┤████                             │
     │████                             │     │████                             │
 84.7┤████                             │ 71.3┤████                             │
 63.5┤████                             │ 53.5┤████                             │
     │████                             │     │████                             │
 42.3┤████                             │ 35.7┤███████                          │
 21.2┤███████                          │ 17.8┤███████                          │
     │█████████████                    │     │█████████████████                │
  0.0┤█████████████████████████████████│  0.0┤█████████████████████████████████│
     └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
    -0.04   0.42    0.89    1.35   1.82     -0.02   0.46    0.94    1.43   1.91
                loss/min hist                           loss/std hist
     ┌─────────────────────────────────┐    ┌──────────────────────────────────┐
146.0┤████                             │73.0┤   ████                           │
     │████                             │    │   ████                           │
121.7┤████                             │60.8┤   ████                           │
 97.3┤████                             │48.7┤   ████                           │
     │████                             │    │   ███████                        │
 73.0┤████                             │36.5┤   ███████                        │
     │████                             │    │   ███████                        │
 48.7┤████                             │24.3┤   ███████████                    │
 24.3┤███████                          │12.2┤   █████████████████████          │
     │██████████                       │    │███████████████████████████       │
  0.0┤█████████████████████████████████│ 0.0┤██████████████████████████████████│
     └┬───────┬───────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
    -0.06   0.39    0.83    1.28   1.73    0.004   0.038    0.073   0.108 0.142
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049/plots/tplot/loss_hist.txt
[2025-12-31 11:31:37,843701][I][ezpz/history:2433:finalize] Saving history report to /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-31-113049/report.md
[2025-12-31 11:31:37,850659][I][ezpz/test_dist:348:finalize] dataset=<xarray.Dataset> Size: 39kB
Dimensions:        (draw: 194)
Coordinates:
  * draw           (draw) int64 2kB 0 1 2 3 4 5 6 ... 188 189 190 191 192 193
Data variables: (12/25)
    iter           (draw) int64 2kB 6 7 8 9 10 11 12 ... 194 195 196 197 198 199
    loss           (draw) float32 776B 1.749 1.581 1.343 ... 0.02264 0.03851
    accuracy       (draw) float32 776B 0.6094 0.625 0.6797 ... 0.9922 1.0 1.0
    dtf            (draw) float64 2kB 0.01076 0.009483 ... 0.009295 0.009377
    dtb            (draw) float64 2kB 0.001477 0.001752 ... 0.00169 0.001521
    iter_mean      (draw) float64 2kB 6.0 7.0 8.0 9.0 ... 197.0 198.0 199.0
    ...             ...
    dtf_min        (draw) float64 2kB 0.00996 0.009416 ... 0.009295 0.009235
    dtf_std        (draw) float64 2kB 0.0004828 0.0005322 ... 0.0005547
    dtb_mean       (draw) float64 2kB 0.001618 0.001605 ... 0.001609 0.001615
    dtb_max        (draw) float64 2kB 0.001847 0.001759 ... 0.001745 0.001847
    dtb_min        (draw) float64 2kB 0.001336 0.00131 ... 0.001343 0.00108
    dtb_std        (draw) float64 2kB 0.0001653 0.0001423 ... 0.0001303 0.000167
[2025-12-31 11:31:38,752472][I][ezpz/test_dist:500:train] Took: 32.90 seconds to finish training
[2025-12-31 11:31:38,753362][I][ezpz/test_dist:695:main] Took: 52.63 seconds
wandb:
wandb: 🚀 View run polar-surf-6861 at:
wandb: Find logs at: ../../../../../../lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20251231_113103-cppqal9m/logs
[2025-12-31 11:31:40,781540][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2025-12-31-113140]----
[2025-12-31 11:31:40,782309][I][ezpz/launch:448:launch] Execution finished with 0.
[2025-12-31 11:31:40,782738][I][ezpz/launch:449:launch] Executing finished in 57.13 seconds.
[2025-12-31 11:31:40,783099][I][ezpz/launch:450:launch] Took 57.13 seconds to run. Exiting.
took: 1m 5s
```
</details>
