# Train CNN with FSDP on MNIST

!!! info "Key API Functions"

    - [`setup_torch()`][ezpz.distributed.setup_torch] — Initialize distributed training
    - [`wrap_model()`][ezpz.distributed.wrap_model] — Wrap model for FSDP (with `strategy="fsdp"`)
    - [`TrainConfig`][ezpz.configs.TrainConfig] — Training configuration

See:

- 📘 [examples/FSDP](../python/Code-Reference/examples/fsdp.md)
- 🐍 [src/ezpz/examples/fsdp.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py)

```bash
ezpz launch python3 -m ezpz.examples.fsdp
```

## Source

<details closed><summary><code>src/ezpz/examples/fsdp.py</code></summary>

```python title="src/ezpz/examples/fsdp.py"
--8<-- "src/ezpz/examples/fsdp.py"
```

</details>

## Code Walkthrough

### FSDP Wrapping

The model is wrapped with `FullyShardedDataParallel` instead of DDP.
Parameters are sharded across GPUs and gathered on-demand for each
forward/backward pass, reducing per-GPU memory:

```python title="src/ezpz/examples/fsdp.py" linenums="275"
    model = FSDP(
        model,
        device_id=device,
        mixed_precision=MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        ),
    )
```

### Training Loop

The training loop is identical to the DDP version — only the wrapping
changes:

```python title="src/ezpz/examples/fsdp.py" linenums="153"
@ezpz.timeitlogit(rank=ezpz.get_rank())
def train(
    model: nn.Module | DistributedDataParallel | FSDP,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    sampler: DistributedSampler | None = None,
) -> dict:
    ...
    model.train()
    ddp_loss = torch.zeros(2).to(device)
    ...
    for _, (batch, target) in enumerate(train_loader):
        batch, target = batch.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = F.nll_loss(output, target, reduction="sum")
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)
    ...
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
```

### When to Use FSDP vs DDP

- **DDP** replicates the full model on every GPU. Simpler and lower
  communication overhead when the model fits in memory.
- **FSDP** shards model parameters across GPUs. Use it when the model is
  too large to fit in a single GPU's memory.

## Help

<details closed><summary><code>--help</code></summary>

```bash
$ python3 -m ezpz.examples.fsdp --help
usage: fsdp.py [-h] [--num-workers N]
            [--dataset {MNIST,OpenImages,ImageNet,ImageNet1k}]
            [--batch-size N] [--dtype D] [--test-batch-size N] [--epochs N]
            [--lr LR] [--gamma M] [--seed S] [--save-model]
            [--data-prefix DATA_PREFIX]

PyTorch MNIST Example using FSDP

options:
-h, --help            show this help message and exit
--num-workers N       number of data loading workers (default: 4)
--dataset {MNIST,OpenImages,ImageNet,ImageNet1k}
                        Dataset to use (default: MNIST)
--batch-size N        input batch size for training (default: 64)
--dtype D             Datatype for training (default=bf16).
--test-batch-size N   input batch size for testing (default: 1000)
--epochs N            number of epochs to train (default: 10)
--lr LR               learning rate (default: 1e-3)
--gamma M             Learning rate step gamma (default: 0.7)
--seed S              random seed (default: 1)
--save-model          For Saving the current Model
--data-prefix DATA_PREFIX
                        data directory prefix
```

</details>

## Output

<details closed><summary>Output on Sunspot</summary>

```bash
$ ezpz launch python3 -m ezpz.examples.fsdp

[2025-12-31 12:21:21,523041][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2025-12-31-122121]----
[2025-12-31 12:21:22,375537][I][ezpz/launch:416:launch] Job ID: 12458339
[2025-12-31 12:21:22,376302][I][ezpz/launch:417:launch] nodelist: ['x1921c0s3b0n0', 'x1921c0s7b0n0']
[2025-12-31 12:21:22,376691][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
[2025-12-31 12:21:22,377360][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
[2025-12-31 12:21:22,378079][I][ezpz/launch:367:build_executable] Building command to execute by piecing together:
[2025-12-31 12:21:22,378474][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
[2025-12-31 12:21:22,379293][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: python3 -m ezpz.examples.fsdp
[2025-12-31 12:21:22,380037][I][ezpz/launch:433:launch] Took: 1.45 seconds to build command.
[2025-12-31 12:21:22,380393][I][ezpz/launch:436:launch] Executing:
mpiexec
  --envall
  --np=24
  --ppn=12
  --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
  --no-vni
  --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
  python3
  -m
  ezpz.examples.fsdp
[2025-12-31 12:21:22,381628][I][ezpz/launch:443:launch] Execution started @ 2025-12-31-122122...
[2025-12-31 12:21:22,382071][I][ezpz/launch:139:run_command] Running command:
 mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 python3 -m ezpz.examples.fsdp
cpubind:list x1921c0s7b0n0 pid 111174 rank 12 0: mask 0x1c
cpubind:list x1921c0s7b0n0 pid 111175 rank 13 1: mask 0x1c00
cpubind:list x1921c0s7b0n0 pid 111176 rank 14 2: mask 0x1c0000
cpubind:list x1921c0s7b0n0 pid 111177 rank 15 3: mask 0x1c000000
cpubind:list x1921c0s7b0n0 pid 111178 rank 16 4: mask 0x1c00000000
cpubind:list x1921c0s7b0n0 pid 111179 rank 17 5: mask 0x1c0000000000
cpubind:list x1921c0s7b0n0 pid 111180 rank 18 6: mask 0x1c0000000000000
cpubind:list x1921c0s7b0n0 pid 111181 rank 19 7: mask 0x1c000000000000000
cpubind:list x1921c0s7b0n0 pid 111182 rank 20 8: mask 0x1c00000000000000000
cpubind:list x1921c0s7b0n0 pid 111183 rank 21 9: mask 0x1c0000000000000000000
cpubind:list x1921c0s7b0n0 pid 111184 rank 22 10: mask 0x1c000000000000000000000
cpubind:list x1921c0s7b0n0 pid 111185 rank 23 11: mask 0x1c00000000000000000000000
cpubind:list x1921c0s3b0n0 pid 107043 rank 0 0: mask 0x1c
cpubind:list x1921c0s3b0n0 pid 107044 rank 1 1: mask 0x1c00
cpubind:list x1921c0s3b0n0 pid 107045 rank 2 2: mask 0x1c0000
cpubind:list x1921c0s3b0n0 pid 107046 rank 3 3: mask 0x1c000000
cpubind:list x1921c0s3b0n0 pid 107047 rank 4 4: mask 0x1c00000000
cpubind:list x1921c0s3b0n0 pid 107048 rank 5 5: mask 0x1c0000000000
cpubind:list x1921c0s3b0n0 pid 107049 rank 6 6: mask 0x1c0000000000000
cpubind:list x1921c0s3b0n0 pid 107050 rank 7 7: mask 0x1c000000000000000
cpubind:list x1921c0s3b0n0 pid 107051 rank 8 8: mask 0x1c00000000000000000
cpubind:list x1921c0s3b0n0 pid 107052 rank 9 9: mask 0x1c0000000000000000000
cpubind:list x1921c0s3b0n0 pid 107053 rank 10 10: mask 0x1c000000000000000000000
cpubind:list x1921c0s3b0n0 pid 107054 rank 11 11: mask 0x1c00000000000000000000000
[2025-12-31 12:21:26,964250][I][ezpz/dist:1501:setup_torch_distributed] Using torch_{device,backend}= {xpu, xccl}
[2025-12-31 12:21:26,967037][I][ezpz/dist:1366:setup_torch_DDP] Caught MASTER_PORT=41625 from environment!
[2025-12-31 12:21:26,967795][I][ezpz/dist:1382:setup_torch_DDP] Using torch.distributed.init_process_group with
- master_addr='x1921c0s3b0n0'
- master_port='41625'
- world_size=24
- rank=0
- local_rank=0
- timeout=datetime.timedelta(seconds=3600)
- backend='xccl'
[2025-12-31 12:21:26,968707][I][ezpz/dist:1014:init_process_group] Calling torch.distributed.init_process_group_with: rank=0 world_size=24 backend=xccl
[2025-12-31 12:21:27,619965][I][ezpz/dist:1727:setup_torch] Using device='xpu' with backend='xccl' + 'xccl' for distributed training.
[2025-12-31 12:21:27,620787][W][ezpz/dist:544:print_dist_setup] Using [24 / 24] available "xpu" devices !!
[2025-12-31 12:21:27,621230][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=00/23][local_rank=00/11]
[2025-12-31 12:21:27,620421][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=01/23][local_rank=01/11]
[2025-12-31 12:21:27,620452][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=02/23][local_rank=02/11]
[2025-12-31 12:21:27,620445][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=04/23][local_rank=04/11]
[2025-12-31 12:21:27,620450][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=05/23][local_rank=05/11]
[2025-12-31 12:21:27,620418][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=06/23][local_rank=06/11]
[2025-12-31 12:21:27,620439][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=07/23][local_rank=07/11]
[2025-12-31 12:21:27,620431][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=08/23][local_rank=08/11]
[2025-12-31 12:21:27,620400][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=09/23][local_rank=09/11]
[2025-12-31 12:21:27,620398][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=10/23][local_rank=10/11]
[2025-12-31 12:21:27,620433][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=11/23][local_rank=11/11]
[2025-12-31 12:21:27,620451][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=03/23][local_rank=03/11]
[2025-12-31 12:21:27,620523][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=12/23][local_rank=00/11]
[2025-12-31 12:21:27,620546][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=13/23][local_rank=01/11]
[2025-12-31 12:21:27,620556][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=14/23][local_rank=02/11]
[2025-12-31 12:21:27,620557][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=16/23][local_rank=04/11]
[2025-12-31 12:21:27,620568][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=15/23][local_rank=03/11]
[2025-12-31 12:21:27,620557][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=17/23][local_rank=05/11]
[2025-12-31 12:21:27,620575][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=19/23][local_rank=07/11]
[2025-12-31 12:21:27,620556][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=20/23][local_rank=08/11]
[2025-12-31 12:21:27,620560][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=21/23][local_rank=09/11]
[2025-12-31 12:21:27,620578][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=22/23][local_rank=10/11]
[2025-12-31 12:21:27,620579][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=23/23][local_rank=11/11]
[2025-12-31 12:21:27,620579][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=18/23][local_rank=06/11]
[2025-12-31 12:21:28,206982][I][ezpz/dist:2039:setup_wandb] Setting up wandb from rank=0
[2025-12-31 12:21:28,207580][I][ezpz/dist:2040:setup_wandb] Using WB_PROJECT=ezpz.examples.fsdp
wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.23.1
wandb: Run data is saved locally in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20251231_122128-11cqdt05
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run vivid-glade-86
wandb:  View project at https://wandb.ai/aurora_gpt/ezpz.examples.fsdp
wandb:  View run at https://wandb.ai/aurora_gpt/ezpz.examples.fsdp/runs/11cqdt05
[2025-12-31 12:21:29,790902][I][ezpz/dist:2069:setup_wandb] wandb.run=[vivid-glade-86](https://wandb.ai/aurora_gpt/ezpz.examples.fsdp/runs/11cqdt05)
[2025-12-31 12:21:29,796125][I][ezpz/dist:2112:setup_wandb] Running on machine='SunSpot'
[2025-12-31 12:21:30,092593][I][examples/fsdp:196:prepare_model_optimizer_and_scheduler] 
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
Net                                      --
├─Conv2d: 1-1                            320
├─Conv2d: 1-2                            18,496
├─Dropout: 1-3                           --
├─Dropout: 1-4                           --
├─Linear: 1-5                            1,179,776
├─Linear: 1-6                            1,290
=================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
=================================================================
[2025-12-31 12:21:30,134352][I][examples/fsdp:212:prepare_model_optimizer_and_scheduler] model=FullyShardedDataParallel(
  (_fsdp_wrapped_module): Net(
    (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
    (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (dropout1): Dropout(p=0.25, inplace=False)
    (dropout2): Dropout(p=0.5, inplace=False)
    (fc1): Linear(in_features=9216, out_features=128, bias=True)
    (fc2): Linear(in_features=128, out_features=10, bias=True)
  )
)
[2025-12-31 12:21:30,173375][I][ezpz/history:220:__init__] Using History with distributed_history=True
2025:12:31-12:21:30:(107043) |CCL_WARN| value of CCL_OP_SYNC changed to be 1 (default:0)
2025:12:31-12:21:30:(107043) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
[2025-12-31 12:21:55,502783][I][examples/fsdp:340:fsdp_main] epoch=1 dt=12.487221 train_loss=0.596659 test_loss=0.143485 test_acc=95.563553 dt/mean=11.990577 dt/max=12.487222 dt/min=11.897395 dt/std=0.119125 train_loss/mean=0.596659 train_loss/max=0.596659 train_loss/min=0.596659 train_loss/std=0.000173 test_loss/mean=0.143485 test_loss/max=0.143485 test_loss/min=0.143485 test_loss/std=0.000000 test_acc/mean=95.563560 test_acc/max=95.563553 test_acc/min=95.563553 test_acc/std=0.000000
[2025-12-31 12:21:55,911549][I][examples/fsdp:340:fsdp_main] epoch=2 dt=0.361235 train_loss=0.174450 test_loss=0.080361 test_acc=97.511993 dt/mean=0.365279 dt/max=0.373996 dt/min=0.355496 dt/std=0.005433 train_loss/mean=0.174450 train_loss/max=0.174450 train_loss/min=0.174450 train_loss/std=0.000000 test_loss/mean=0.080361 test_loss/max=0.080361 test_loss/min=0.080361 test_loss/std=0.000022 test_acc/mean=97.511993 test_acc/max=97.511993 test_acc/min=97.511993 test_acc/std=0.000000
[2025-12-31 12:21:56,308947][I][examples/fsdp:340:fsdp_main] epoch=3 dt=0.359641 train_loss=0.120487 test_loss=0.060764 test_acc=98.021584 dt/mean=0.358203 dt/max=0.361614 dt/min=0.353194 dt/std=0.002922 train_loss/mean=0.120487 train_loss/max=0.120487 train_loss/min=0.120487 train_loss/std=0.000000 test_loss/mean=0.060764 test_loss/max=0.060764 test_loss/min=0.060764 test_loss/std=0.000015 test_acc/mean=98.021591 test_acc/max=98.021584 test_acc/min=98.021584 test_acc/std=0.000000
[2025-12-31 12:21:56,703145][I][examples/fsdp:340:fsdp_main] epoch=4 dt=0.356608 train_loss=0.098917 test_loss=0.052346 test_acc=98.301361 dt/mean=0.356618 dt/max=0.359070 dt/min=0.353434 dt/std=0.001995 train_loss/mean=0.098917 train_loss/max=0.098917 train_loss/min=0.098917 train_loss/std=0.000000 test_loss/mean=0.052346 test_loss/max=0.052346 test_loss/min=0.052346 test_loss/std=0.000000 test_acc/mean=98.301361 test_acc/max=98.301361 test_acc/min=98.301361 test_acc/std=0.031250
[2025-12-31 12:21:57,100230][I][examples/fsdp:340:fsdp_main] epoch=5 dt=0.357687 train_loss=0.085740 test_loss=0.047243 test_acc=98.441246 dt/mean=0.356900 dt/max=0.360295 dt/min=0.352879 dt/std=0.002699 train_loss/mean=0.085740 train_loss/max=0.085740 train_loss/min=0.085740 train_loss/std=0.000000 test_loss/mean=0.047243 test_loss/max=0.047243 test_loss/min=0.047243 test_loss/std=0.000000 test_acc/mean=98.441246 test_acc/max=98.441246 test_acc/min=98.441246 test_acc/std=0.000000
[2025-12-31 12:21:57,497234][I][examples/fsdp:340:fsdp_main] epoch=6 dt=0.357410 train_loss=0.080569 test_loss=0.044845 test_acc=98.471222 dt/mean=0.356574 dt/max=0.359746 dt/min=0.353584 dt/std=0.002156 train_loss/mean=0.080569 train_loss/max=0.080569 train_loss/min=0.080569 train_loss/std=0.000000 test_loss/mean=0.044845 test_loss/max=0.044845 test_loss/min=0.044845 test_loss/std=0.000015 test_acc/mean=98.471222 test_acc/max=98.471222 test_acc/min=98.471222 test_acc/std=0.000000
[2025-12-31 12:21:57,893327][I][examples/fsdp:340:fsdp_main] epoch=7 dt=0.355675 train_loss=0.075174 test_loss=0.043703 test_acc=98.481216 dt/mean=0.356044 dt/max=0.358311 dt/min=0.353675 dt/std=0.001370 train_loss/mean=0.075174 train_loss/max=0.075174 train_loss/min=0.075174 train_loss/std=0.000022 test_loss/mean=0.043703 test_loss/max=0.043703 test_loss/min=0.043703 test_loss/std=0.000011 test_acc/mean=98.481224 test_acc/max=98.481216 test_acc/min=98.481216 test_acc/std=0.000000
[2025-12-31 12:21:58,292161][I][examples/fsdp:340:fsdp_main] epoch=8 dt=0.358490 train_loss=0.073104 test_loss=0.041848 test_acc=98.551163 dt/mean=0.359055 dt/max=0.362143 dt/min=0.355792 dt/std=0.001879 train_loss/mean=0.073104 train_loss/max=0.073104 train_loss/min=0.073104 train_loss/std=0.000022 test_loss/mean=0.041848 test_loss/max=0.041848 test_loss/min=0.041848 test_loss/std=0.000000 test_acc/mean=98.551170 test_acc/max=98.551163 test_acc/min=98.551163 test_acc/std=0.000000
[2025-12-31 12:21:58,692175][I][examples/fsdp:340:fsdp_main] epoch=9 dt=0.359963 train_loss=0.069403 test_loss=0.041198 test_acc=98.571144 dt/mean=0.360091 dt/max=0.363091 dt/min=0.356911 dt/std=0.001945 train_loss/mean=0.069403 train_loss/max=0.069403 train_loss/min=0.069403 train_loss/std=0.000022 test_loss/mean=0.041198 test_loss/max=0.041198 test_loss/min=0.041198 test_loss/std=0.000011 test_acc/mean=98.571152 test_acc/max=98.571144 test_acc/min=98.571144 test_acc/std=0.000000
[2025-12-31 12:21:59,091674][I][examples/fsdp:340:fsdp_main] epoch=10 dt=0.358637 train_loss=0.068348 test_loss=0.041941 test_acc=98.571144 dt/mean=0.358994 dt/max=0.361870 dt/min=0.356423 dt/std=0.001696 train_loss/mean=0.068348 train_loss/max=0.068348 train_loss/min=0.068348 train_loss/std=0.000000 test_loss/mean=0.041941 test_loss/max=0.041941 test_loss/min=0.041941 test_loss/std=0.000000 test_acc/mean=98.571152 test_acc/max=98.571144 test_acc/min=98.571144 test_acc/std=0.000000
[2025-12-31 12:21:59,093446][I][examples/fsdp:342:fsdp_main] 11 epochs took 28.9s
[2025-12-31 12:21:59,124624][I][ezpz/history:2385:finalize] Saving plots to /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/plots/mplot (matplotlib) and /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/plots/tplot (tplot)
                     dt                                    dt/min
    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
12.5┤▌                                 │11.9┤-                                 │
10.5┤▚                                 │ 8.0┤ -                                │
    │▝▖                                │ 4.2┤  -                               │
 8.4┤ ▌                                │ 0.4┤   -------------------------------│
 6.4┤ ▐                                │    └┬───────┬────────┬───────┬───────┬┘
 4.4┤  ▌                               │    1.0     3.2      5.5     7.8   10.0
    │  ▚                               │dt/min              iter
 2.4┤  ▝▖                              │                   dt/std
 0.4┤   ▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│     ┌─────────────────────────────────┐
    └┬───────┬────────┬───────┬───────┬┘0.119┤*                                │
    1.0     3.2      5.5     7.8   10.0 0.099┤ *                               │
dt                  iter                0.060┤  *                              │
                   dt/mean              0.041┤   *                             │
    ┌──────────────────────────────────┐0.001┤    *****************************│
12.0┤·                                 │     └┬───────┬───────┬───────┬───────┬┘
10.1┤·                                 │     1.0     3.2     5.5     7.8   10.0
    │·                                 │dt/std              iter
 8.1┤ ·                                │                   dt/max
 6.2┤ ·                                │    ┌──────────────────────────────────┐
    │  ·                               │12.5┤+                                 │
 4.2┤  ·                               │10.5┤ +                                │
 2.3┤   ·                              │ 6.4┤  +                               │
    │   ·                              │ 4.4┤   +                              │
 0.4┤    ······························│ 0.4┤    ++++++++++++++++++++++++++++++│
    └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
    1.0     3.2      5.5     7.8   10.0     1.0     3.2      5.5     7.8   10.0
dt/mean             iter                dt/max              iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/plots/tplot/dt.txt
    ┌──────────────────────────────────────────────────────────────────────────┐
12.5┤ ++ dt/max                                                                │
    │ -- dt/min                                                                │
    │ ·· dt/mean                                                               │
    │ ▞▞ dt                                                                    │
10.5┤ ▌                                                                        │
    │ ▐                                                                        │
    │ ▝▖                                                                       │
    │  ▌                                                                       │
 8.4┤  ▐                                                                       │
    │  ▐                                                                       │
    │   ▌                                                                      │
    │   ▚                                                                      │
 6.4┤   ▐                                                                      │
    │    ▌                                                                     │
    │    ▚                                                                     │
    │    ▐                                                                     │
    │     ▌                                                                    │
 4.4┤     ▌                                                                    │
    │     ▐                                                                    │
    │     ▝▖                                                                   │
    │      ▌                                                                   │
 2.4┤      ▐                                                                   │
    │      ▝▖                                                                  │
    │       ▌                                                                  │
    │       ▐                                                                  │
 0.4┤       ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬─────────────────┬──────────────────┬─────────────────┬─────────────────┬┘
    1.0               3.2                5.5               7.8             10.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/plots/tplot/dt_summary.txt
               dt/mean hist                             dt/max hist
   ┌───────────────────────────────────┐   ┌───────────────────────────────────┐
9.0┤████                               │9.0┤████                               │
7.5┤████                               │7.5┤████                               │
   │████                               │   │████                               │
6.0┤████                               │6.0┤████                               │
4.5┤████                               │4.5┤████                               │
   │████                               │   │████                               │
3.0┤████                               │3.0┤████                               │
1.5┤████                               │1.5┤████                               │
   │████                           ████│   │████                           ████│
0.0┤███                            ████│0.0┤███                            ████│
   └┬────────┬───────┬────────┬───────┬┘   └┬────────┬───────┬────────┬───────┬┘
  -0.2      3.0     6.2      9.3   12.5   -0.2      3.1     6.4      9.7   13.0
                dt/min hist                             dt/std hist
   ┌───────────────────────────────────┐   ┌───────────────────────────────────┐
9.0┤████                               │9.0┤████                               │
   │████                               │   │████                               │
7.5┤████                               │7.5┤████                               │
6.0┤████                               │6.0┤████                               │
   │████                               │   │████                               │
4.5┤████                               │4.5┤████                               │
   │████                               │   │████                               │
3.0┤████                               │3.0┤████                               │
1.5┤████                               │1.5┤████                               │
   │████                           ████│   │████                           ████│
0.0┤███                            ████│0.0┤███                            ████│
   └┬────────┬───────┬────────┬───────┬┘   └┬────────┬───────┬────────┬───────┬┘
  -0.2      3.0     6.1      9.3   12.4   -0.004   0.028   0.060    0.092 0.124
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/plots/tplot/dt_hist.txt
                  test_acc                              test_acc/min
     ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
98.57┤              ▗▄▄▄▄▄▄▄▄▄▄▞▀▀▀▀▀▀▀│98.57┤           ----------------------│
98.07┤       ▄▄▄▄▀▀▀▘                  │97.57┤    -------                      │
     │     ▄▀                          │96.57┤  --                             │
97.57┤   ▞▀                            │95.56┤--                               │
97.07┤  ▐                              │     └┬───────┬───────┬───────┬───────┬┘
96.57┤ ▗▘                              │     1.0     3.2     5.5     7.8   10.0
     │ ▞                               │test_acc/min        iter
96.06┤▐                                │                 test_acc/std
95.56┤▌                                │      ┌────────────────────────────────┐
     └┬───────┬───────┬───────┬───────┬┘0.0312┤          *                     │
     1.0     3.2     5.5     7.8   10.0 0.0260┤         * *                    │
test_acc            iter                0.0156┤        *   *                   │
                test_acc/mean           0.0104┤       *     *                  │
     ┌─────────────────────────────────┐0.0000┤********      ******************│
98.57┤              ···················│      └┬───────┬───────┬──────┬───────┬┘
98.07┤           ···                   │      1.0     3.2     5.5    7.8   10.0
     │       ····                      │test_acc/std         iter
97.57┤    ···                          │                test_acc/max
97.07┤   ·                             │     ┌─────────────────────────────────┐
     │  ·                              │98.57┤           ++++++++++++++++++++++│
96.57┤  ·                              │98.07┤    +++++++                      │
96.06┤ ·                               │97.07┤   +                             │
     │·                                │96.57┤  +                              │
95.56┤·                                │95.56┤++                               │
     └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
     1.0     3.2     5.5     7.8   10.0      1.0     3.2     5.5     7.8   10.0
test_acc/mean       iter                test_acc/max        iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/plots/tplot/test_acc.txt
     ┌─────────────────────────────────────────────────────────────────────────┐
98.57┤ ++ test_acc/max                                    ▗▄▄▄▞▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀│
     │ -- test_acc/min             ▗▄▄▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▘···                 │
     │ ·· test_acc/mean       ▄▄▄▀▀▘··                                         │
     │ ▞▞ test_acc        ▄▄▀▀                                                 │
98.07┤                ▄▄▀▀··                                                   │
     │              ▗▞···                                                      │
     │            ▗▞▘·                                                         │
     │           ▄▘·                                                           │
97.57┤         ▄▀·                                                             │
     │       ▗▀·                                                               │
     │       ▞                                                                 │
     │      ▗▘                                                                 │
97.07┤      ▞                                                                  │
     │     ▗▘                                                                  │
     │     ▞                                                                   │
     │    ▗▘                                                                   │
     │    ▞                                                                    │
96.57┤   ▗▘                                                                    │
     │   ▞                                                                     │
     │  ▗▘                                                                     │
     │  ▞                                                                      │
96.06┤ ▗▘                                                                      │
     │ ▞                                                                       │
     │▗▘                                                                       │
     │▞                                                                        │
95.56┤▌                                                                        │
     └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
     1.0               3.2               5.5               7.8             10.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/plots/tplot/test_acc_summary.txt
            test_acc/mean hist                       test_acc/max hist
   ┌───────────────────────────────────┐   ┌───────────────────────────────────┐
7.0┤                               ████│7.0┤                               ████│
5.8┤                               ████│5.8┤                               ████│
   │                               ████│   │                               ████│
4.7┤                               ████│4.7┤                               ████│
3.5┤                               ████│3.5┤                               ████│
   │                               ████│   │                               ████│
2.3┤                               ████│2.3┤                               ████│
1.2┤                               ████│1.2┤                               ████│
   │████                 ████   ███████│   │████                 ████   ███████│
0.0┤███                  ███    ███████│0.0┤███                  ███    ███████│
   └┬────────┬───────┬────────┬───────┬┘   └┬────────┬───────┬────────┬───────┬┘
  95.4     96.2    97.1     97.9   98.7   95.4     96.2    97.1     97.9   98.7
             test_acc/min hist                       test_acc/std hist
   ┌───────────────────────────────────┐   ┌───────────────────────────────────┐
7.0┤                               ████│9.0┤████                               │
   │                               ████│   │████                               │
5.8┤                               ████│7.5┤████                               │
4.7┤                               ████│6.0┤████                               │
   │                               ████│   │████                               │
3.5┤                               ████│4.5┤████                               │
   │                               ████│   │████                               │
2.3┤                               ████│3.0┤████                               │
1.2┤                               ████│1.5┤████                               │
   │████                 ████   ███████│   │████                           ████│
0.0┤███                  ███    ███████│0.0┤███                            ████│
   └┬────────┬───────┬────────┬───────┬┘   └┬────────┬───────┬────────┬────────┘
  95.4     96.2    97.1     97.9   98.7   -0.0014  0.0071  0.0156  0.0241
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/plots/tplot/test_acc_hist.txt
                  test_loss                             test_loss/min
     ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
0.143┤▌                                │0.143┤-                                │
0.126┤▝▖                               │0.109┤ --                              │
     │ ▚                               │0.075┤   -----                         │
0.109┤  ▌                              │0.041┤        -------------------------│
0.092┤  ▝▖                             │     └┬───────┬───────┬───────┬───────┬┘
0.075┤   ▝▖                            │     1.0     3.2     5.5     7.8   10.0
     │    ▝▚▖                          │test_loss/min       iter
0.058┤      ▝▚▄▄▄▖                     │                  test_loss/std
0.041┤           ▝▀▀▀▀▀▀▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄│         ┌─────────────────────────────┐
     └┬───────┬───────┬───────┬───────┬┘0.0000216┤   *                         │
     1.0     3.2     5.5     7.8   10.0 0.0000180┤  * ***         *            │
test_loss           iter                0.0000108┤ *     *       * ***     *   │
               test_loss/mean           0.0000072┤*       *     *     *   * *  │
     ┌─────────────────────────────────┐0.0000000┤*        *****       ***   **│
0.143┤·                                │         └┬──────┬──────┬──────┬──────┬┘
0.126┤·                                │         1.0    3.2    5.5    7.8  10.0
     │ ·                               │test_loss/std         iter
0.109┤  ·                              │                test_loss/max
0.092┤  ·                              │     ┌─────────────────────────────────┐
     │   ·                             │0.143┤+                                │
0.075┤    ·                            │0.126┤ ++                              │
0.058┤     ···                         │0.092┤   ++                            │
     │        ·······                  │0.075┤     +++                         │
0.041┤               ··················│0.041┤        +++++++++++++++++++++++++│
     └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
     1.0     3.2     5.5     7.8   10.0      1.0     3.2     5.5     7.8   10.0
test_loss/mean      iter                test_loss/max       iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/plots/tplot/test_loss.txt
     ┌─────────────────────────────────────────────────────────────────────────┐
0.143┤ ++ test_loss/max                                                        │
     │ -- test_loss/min                                                        │
     │ ·· test_loss/mean                                                       │
     │ ▞▞ test_loss                                                            │
0.126┤  ▌                                                                      │
     │  ▐                                                                      │
     │   ▌                                                                     │
     │   ▐                                                                     │
0.109┤    ▌                                                                    │
     │    ▐                                                                    │
     │     ▌                                                                   │
     │     ▐                                                                   │
0.092┤      ▌                                                                  │
     │      ▐                                                                  │
     │       ▌                                                                 │
     │       ▝▖                                                                │
     │        ▝▄                                                               │
0.075┤          ▚▖                                                             │
     │           ▝▚                                                            │
     │             ▀▖                                                          │
     │              ▝▚▖                                                        │
0.058┤                ▝▀▚▄▖                                                    │
     │                    ▝▀▚▄▖                                                │
     │                        ▝▀▀▄▄▖                                           │
     │                             ▝▀▀▚▄▄▄▄▄▄▄▄········                        │
0.041┤                                         ▀▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
     └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
     1.0               3.2               5.5               7.8             10.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/plots/tplot/test_loss_summary.txt
           test_loss/mean hist                     test_loss/max hist
 ┌─────────────────────────────────────┐ ┌─────────────────────────────────────┐
6┤████                                 │6┤████                                 │
5┤████                                 │5┤████                                 │
 │████                                 │ │████                                 │
4┤████                                 │4┤████                                 │
3┤████                                 │3┤████                                 │
 │████                                 │ │████                                 │
2┤████████                             │2┤████████                             │
1┤████████   ████                  ████│1┤████████   ████                  ████│
 │████████   ████                  ████│ │████████   ████                  ████│
0┤███████    ████                  ████│0┤███████    ████                  ████│
 └┬────────┬────────┬────────┬────────┬┘ └┬────────┬────────┬────────┬────────┬┘
 0.037   0.064    0.092    0.120  0.148  0.037   0.064    0.092    0.120  0.148
           test_loss/min hist                        test_loss/std hist
 ┌─────────────────────────────────────┐    ┌──────────────────────────────────┐
6┤████                                 │5.00┤████                              │
 │████                                 │    │████                              │
5┤████                                 │4.17┤████                              │
4┤████                                 │3.33┤████                              │
 │████                                 │    │████                              │
3┤████                                 │2.50┤████                              │
 │████                                 │    │████             ████   ███       │
2┤████████                             │1.67┤████             ████   ███       │
1┤████████   ████                  ████│0.83┤████             ████   ███   ████│
 │████████   ████                  ████│    │████             ████   ███   ████│
0┤███████    ████                  ████│0.00┤███              ███    ███   ████│
 └┬────────┬────────┬────────┬────────┬┘    └┬────────────────┬───────┬────────┘
 0.037   0.064    0.092    0.120  0.148   -0.0000010      0.0000108 0.0000167
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/plots/tplot/test_loss_hist.txt
                 train_loss                            train_loss/min
     ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
0.597┤▌                                │0.597┤-                                │
0.509┤▐                                │0.421┤ --                              │
     │ ▌                               │0.244┤   --                            │
0.421┤ ▐                               │0.068┤     ----------------------------│
0.333┤  ▌                              │     └┬───────┬───────┬───────┬───────┬┘
0.244┤  ▐                              │     1.0     3.2     5.5     7.8   10.0
     │   ▌                             │train_loss/min      iter
0.156┤   ▝▄▄▄▖                         │                 train_loss/std
0.068┤       ▝▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│        ┌──────────────────────────────┐
     └┬───────┬───────┬───────┬───────┬┘0.000173┤*                             │
     1.0     3.2     5.5     7.8   10.0 0.000144┤*                             │
train_loss          iter                0.000086┤ *                            │
               train_loss/mean          0.000058┤  *                ********   │
     ┌─────────────────────────────────┐0.000000┤   ****************        ***│
0.597┤·                                │        └┬──────┬───────┬──────┬──────┬┘
0.509┤·                                │        1.0    3.2     5.5    7.8  10.0
     │ ·                               │train_loss/std        iter
0.421┤ ·                               │               train_loss/max
0.333┤  ·                              │     ┌─────────────────────────────────┐
     │  ·                              │0.597┤+                                │
0.244┤   ·                             │0.509┤ +                               │
0.156┤    ·                            │0.333┤  +                              │
     │     ·······                     │0.244┤   ++                            │
0.068┤            ·····················│0.068┤     ++++++++++++++++++++++++++++│
     └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
     1.0     3.2     5.5     7.8   10.0      1.0     3.2     5.5     7.8   10.0
train_loss/mean     iter                train_loss/max      iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/plots/tplot/train_loss.txt
     ┌─────────────────────────────────────────────────────────────────────────┐
0.597┤ ++ train_loss/max                                                       │
     │ -- train_loss/min                                                       │
     │ ·· train_loss/mean                                                      │
     │ ▞▞ train_loss                                                           │
0.509┤ ▐                                                                       │
     │ ▝▖                                                                      │
     │  ▚                                                                      │
     │  ▐                                                                      │
0.421┤   ▌                                                                     │
     │   ▐                                                                     │
     │   ▝▖                                                                    │
     │    ▌                                                                    │
0.333┤    ▐                                                                    │
     │     ▌                                                                   │
     │     ▚                                                                   │
     │     ▝▖                                                                  │
     │      ▌                                                                  │
0.244┤      ▐                                                                  │
     │       ▌                                                                 │
     │       ▚                                                                 │
     │       ▝▖                                                                │
0.156┤        ▝▀▄▖                                                             │
     │           ▝▀▄▖                                                          │
     │              ▝▀▚▄▄▄▖                                                    │
     │                 ···▝▀▀▀▚▄▄▄▄▄▄▄▖········                                │
0.068┤                                ▝▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
     └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
     1.0               3.2               5.5               7.8             10.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/plots/tplot/train_loss_summary.txt
           train_loss/mean hist                     train_loss/max hist
   ┌───────────────────────────────────┐   ┌───────────────────────────────────┐
8.0┤████                               │8.0┤████                               │
6.7┤████                               │6.7┤████                               │
   │████                               │   │████                               │
5.3┤████                               │5.3┤████                               │
4.0┤████                               │4.0┤████                               │
   │████                               │   │████                               │
2.7┤████                               │2.7┤████                               │
1.3┤████                               │1.3┤████                               │
   │████   ████                    ████│   │████   ████                    ████│
0.0┤███    ███                     ████│0.0┤███    ███                     ████│
   └┬────────┬───────┬────────┬───────┬┘   └┬────────┬───────┬────────┬───────┬┘
  0.04     0.19    0.33     0.48   0.62   0.04     0.19    0.33     0.48   0.62
            train_loss/min hist                    train_loss/std hist
   ┌───────────────────────────────────┐ ┌─────────────────────────────────────┐
8.0┤████                               │6┤████                                 │
   │████                               │ │████                                 │
6.7┤████                               │5┤████                                 │
5.3┤████                               │4┤████                                 │
   │████                               │ │████                                 │
4.0┤████                               │3┤████████                             │
   │████                               │ │████████                             │
2.7┤████                               │2┤████████                             │
1.3┤████                               │1┤████████                         ████│
   │████   ████                    ████│ │████████                         ████│
0.0┤███    ███                     ████│0┤███████                          ████│
   └┬────────┬───────┬────────┬───────┬┘ └┬─────────────────┬────────┬─────────┘
  0.04     0.19    0.33     0.48   0.62  -0.000008      0.000086  0.000133
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/plots/tplot/train_loss_hist.txt
[2025-12-31 12:22:03,182749][W][ezpz/history:2320:save_dataset] Unable to save dataset to W&B, skipping!
[2025-12-31 12:22:03,184704][I][utils/__init__:651:dataset_to_h5pyfile] Saving dataset to: /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/train_dataset.h5
[2025-12-31 12:22:03,196685][I][ezpz/history:2433:finalize] Saving history report to /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-122159/report.md
[2025-12-31 12:22:03,202017][I][examples/fsdp:360:fsdp_main] dataset=<xarray.Dataset> Size: 2kB
Dimensions:          (draw: 10)
Coordinates:
  * draw             (draw) int64 80B 0 1 2 3 4 5 6 7 8 9
Data variables: (12/25)
    epoch            (draw) int64 80B 1 2 3 4 5 6 7 8 9 10
    dt               (draw) float64 80B 12.49 0.3612 0.3596 ... 0.36 0.3586
    train_loss       (draw) float32 40B 0.5967 0.1744 0.1205 ... 0.0694 0.06835
    test_loss        (draw) float32 40B 0.1435 0.08036 ... 0.0412 0.04194
    test_acc         (draw) float32 40B 95.56 97.51 98.02 ... 98.55 98.57 98.57
    epoch_mean       (draw) float64 80B 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0
    ...               ...
    test_loss_min    (draw) float64 80B 0.1435 0.08036 ... 0.0412 0.04194
    test_loss_std    (draw) float64 80B 0.0 2.158e-05 ... 1.079e-05 0.0
    test_acc_mean    (draw) float64 80B 95.56 97.51 98.02 ... 98.55 98.57 98.57
    test_acc_max     (draw) float64 80B 95.56 97.51 98.02 ... 98.55 98.57 98.57
    test_acc_min     (draw) float64 80B 95.56 97.51 98.02 ... 98.55 98.57 98.57
    test_acc_std     (draw) float64 80B 0.0 0.0 0.0 0.03125 ... 0.0 0.0 0.0 0.0
[2025-12-31 12:22:03,205311][I][examples/fsdp:452:<module>] Took 36.24 seconds
wandb:
wandb: 🚀 View run vivid-glade-86 at: 
wandb: Find logs at: ../../../../../../lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20251231_122128-11cqdt05/logs
[2025-12-31 12:22:04,704632][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2025-12-31-122204]----
[2025-12-31 12:22:04,705324][I][ezpz/launch:448:launch] Execution finished with 0.
[2025-12-31 12:22:04,705724][I][ezpz/launch:449:launch] Executing finished in 42.32 seconds.
[2025-12-31 12:22:04,706075][I][ezpz/launch:450:launch] Took 42.32 seconds to run. Exiting.
```


</details>


