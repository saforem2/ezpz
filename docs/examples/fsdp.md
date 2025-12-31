# Train CNN with FSDP on MNIST

See:

- 📘 [examples/FSDP](../python/Code-Reference/examples/fsdp.md)
- 🐍 [src/ezpz/examples/fsdp.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp.py)

```bash
ezpz launch python3 -m ezpz.examples.fsdp
```

<details closed><summary><code>--help</code></summary>

```bash

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


<details closed><summary>Output on Sunspot:</summary>

```bash
[2025-12-31 11:47:46,719945][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2025-12-31-114746]----
[2025-12-31 11:47:47,574805][I][ezpz/launch:416:launch] Job ID: 12458339
[2025-12-31 11:47:47,575627][I][ezpz/launch:417:launch] nodelist: ['x1921c0s3b0n0', 'x1921c0s7b0n0']
[2025-12-31 11:47:47,576025][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov
[2025-12-31 11:47:47,576705][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
[2025-12-31 11:47:47,577412][I][ezpz/launch:367:build_executable] Building command to execute by piecing together:
[2025-12-31 11:47:47,577798][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
[2025-12-31 11:47:47,578603][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: python3 -m ezpz.examples.fsdp
[2025-12-31 11:47:47,579347][I][ezpz/launch:433:launch] Took: 1.52 seconds to build command.
[2025-12-31 11:47:47,579705][I][ezpz/launch:436:launch] Executing:
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
[2025-12-31 11:47:47,581014][I][ezpz/launch:443:launch] Execution started @ 2025-12-31-114747...
[2025-12-31 11:47:47,581479][I][ezpz/launch:139:run_command] Running command:
 mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/12458339.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 python3 -m ezpz.examples.fsdp
cpubind:list x1921c0s7b0n0 pid 96130 rank 12 0: mask 0x1c
cpubind:list x1921c0s7b0n0 pid 96131 rank 13 1: mask 0x1c00
cpubind:list x1921c0s7b0n0 pid 96132 rank 14 2: mask 0x1c0000
cpubind:list x1921c0s7b0n0 pid 96133 rank 15 3: mask 0x1c000000
cpubind:list x1921c0s7b0n0 pid 96134 rank 16 4: mask 0x1c00000000
cpubind:list x1921c0s7b0n0 pid 96135 rank 17 5: mask 0x1c0000000000
cpubind:list x1921c0s7b0n0 pid 96136 rank 18 6: mask 0x1c0000000000000
cpubind:list x1921c0s7b0n0 pid 96137 rank 19 7: mask 0x1c000000000000000
cpubind:list x1921c0s7b0n0 pid 96138 rank 20 8: mask 0x1c00000000000000000
cpubind:list x1921c0s7b0n0 pid 96139 rank 21 9: mask 0x1c0000000000000000000
cpubind:list x1921c0s7b0n0 pid 96140 rank 22 10: mask 0x1c000000000000000000000
cpubind:list x1921c0s7b0n0 pid 96141 rank 23 11: mask 0x1c00000000000000000000000
cpubind:list x1921c0s3b0n0 pid 98682 rank 0 0: mask 0x1c
cpubind:list x1921c0s3b0n0 pid 98683 rank 1 1: mask 0x1c00
cpubind:list x1921c0s3b0n0 pid 98684 rank 2 2: mask 0x1c0000
cpubind:list x1921c0s3b0n0 pid 98685 rank 3 3: mask 0x1c000000
cpubind:list x1921c0s3b0n0 pid 98686 rank 4 4: mask 0x1c00000000
cpubind:list x1921c0s3b0n0 pid 98687 rank 5 5: mask 0x1c0000000000
cpubind:list x1921c0s3b0n0 pid 98688 rank 6 6: mask 0x1c0000000000000
cpubind:list x1921c0s3b0n0 pid 98689 rank 7 7: mask 0x1c000000000000000
cpubind:list x1921c0s3b0n0 pid 98690 rank 8 8: mask 0x1c00000000000000000
cpubind:list x1921c0s3b0n0 pid 98691 rank 9 9: mask 0x1c0000000000000000000
cpubind:list x1921c0s3b0n0 pid 98692 rank 10 10: mask 0x1c000000000000000000000
cpubind:list x1921c0s3b0n0 pid 98693 rank 11 11: mask 0x1c00000000000000000000000
[2025-12-31 11:47:54,098451][I][ezpz/dist:1501:setup_torch_distributed] Using torch_{device,backend}= {xpu, xccl}
[2025-12-31 11:47:54,102280][I][ezpz/dist:1366:setup_torch_DDP] Caught MASTER_PORT=45233 from environment!
[2025-12-31 11:47:54,103268][I][ezpz/dist:1382:setup_torch_DDP] Using torch.distributed.init_process_group with
- master_addr='x1921c0s3b0n0'
- master_port='45233'
- world_size=24
- rank=0
- local_rank=0
- timeout=datetime.timedelta(seconds=3600)
- backend='xccl'
[2025-12-31 11:47:54,104659][I][ezpz/dist:1014:init_process_group] Calling torch.distributed.init_process_group_with: rank=0 world_size=24 backend=xccl
[2025-12-31 11:47:54,152113][I][ezpz/dist:1727:setup_torch] Using device='xpu' with backend='xccl' + 'xccl' for distributed training.
[2025-12-31 11:47:54,152911][W][ezpz/dist:544:print_dist_setup] Using [24 / 24] available "xpu" devices !!
[2025-12-31 11:47:54,153351][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=00/23][local_rank=00/11]
[2025-12-31 11:47:54,152519][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=08/23][local_rank=08/11]
[2025-12-31 11:47:54,152577][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=01/23][local_rank=01/11]
[2025-12-31 11:47:54,152572][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=02/23][local_rank=02/11]
[2025-12-31 11:47:54,152570][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=03/23][local_rank=03/11]
[2025-12-31 11:47:54,152590][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=04/23][local_rank=04/11]
[2025-12-31 11:47:54,152589][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=05/23][local_rank=05/11]
[2025-12-31 11:47:54,152557][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=06/23][local_rank=06/11]
[2025-12-31 11:47:54,152542][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=07/23][local_rank=07/11]
[2025-12-31 11:47:54,152547][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=09/23][local_rank=09/11]
[2025-12-31 11:47:54,152538][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=0/1][rank=10/23][local_rank=10/11]
[2025-12-31 11:47:54,152551][I][ezpz/dist:1774:setup_torch] ['x1921c0s3b0n0'][device='xpu'][node=1/1][rank=11/23][local_rank=11/11]
[2025-12-31 11:47:54,152773][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=14/23][local_rank=02/11]
[2025-12-31 11:47:54,152773][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=12/23][local_rank=00/11]
[2025-12-31 11:47:54,152885][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=13/23][local_rank=01/11]
[2025-12-31 11:47:54,152889][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=15/23][local_rank=03/11]
[2025-12-31 11:47:54,152898][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=16/23][local_rank=04/11]
[2025-12-31 11:47:54,152827][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=17/23][local_rank=05/11]
[2025-12-31 11:47:54,152861][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=18/23][local_rank=06/11]
[2025-12-31 11:47:54,152891][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=19/23][local_rank=07/11]
[2025-12-31 11:47:54,152819][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=20/23][local_rank=08/11]
[2025-12-31 11:47:54,152874][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=21/23][local_rank=09/11]
[2025-12-31 11:47:54,152894][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=0/1][rank=22/23][local_rank=10/11]
[2025-12-31 11:47:54,152828][I][ezpz/dist:1774:setup_torch] ['x1921c0s7b0n0'][device='xpu'][node=1/1][rank=23/23][local_rank=11/11]
[2025-12-31 11:47:55,067065][I][ezpz/dist:2039:setup_wandb] Setting up wandb from rank=0
[2025-12-31 11:47:55,067658][I][ezpz/dist:2040:setup_wandb] Using WB_PROJECT=ezpz.examples.fsdp
wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.23.1
wandb: Run data is saved locally in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20251231_114755-z78eje8p
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run ethereal-dream-85
wandb:  View project at https://wandb.ai/aurora_gpt/ezpz.examples.fsdp
wandb:  View run at https://wandb.ai/aurora_gpt/ezpz.examples.fsdp/runs/z78eje8p
[2025-12-31 11:47:57,053806][I][ezpz/dist:2069:setup_wandb] wandb.run=[ethereal-dream-85](https://wandb.ai/aurora_gpt/ezpz.examples.fsdp/runs/z78eje8p)
[2025-12-31 11:47:57,058771][I][ezpz/dist:2112:setup_wandb] Running on machine='SunSpot'
[2025-12-31 11:47:57,370298][I][examples/fsdp:196:prepare_model_optimizer_and_scheduler] 
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
[2025-12-31 11:47:57,410040][I][examples/fsdp:212:prepare_model_optimizer_and_scheduler] model=FullyShardedDataParallel(
  (_fsdp_wrapped_module): Net(
    (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
    (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (dropout1): Dropout(p=0.25, inplace=False)
    (dropout2): Dropout(p=0.5, inplace=False)
    (fc1): Linear(in_features=9216, out_features=128, bias=True)
    (fc2): Linear(in_features=128, out_features=10, bias=True)
  )
)
[2025-12-31 11:47:57,453207][I][ezpz/history:220:__init__] Using History with distributed_history=True
2025:12:31-11:47:57:(98682) |CCL_WARN| value of CCL_OP_SYNC changed to be 1 (default:0)
2025:12:31-11:47:57:(98682) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
[2025-12-31 11:48:22,658303][I][examples/fsdp:340:fsdp_main] epoch=1 dt=12.461527 train_loss=0.572582 test_loss=0.135816 test_acc=95.833336 dt/mean=11.956943 dt/max=12.461528 dt/min=11.818278 dt/std=0.146939 train_loss/mean=0.572582 train_loss/max=0.572582 train_loss/min=0.572582 train_loss/std=0.000173 test_loss/mean=0.135816 test_loss/max=0.135816 test_loss/min=0.135816 test_loss/std=0.000000 test_acc/mean=95.833336 test_acc/max=95.833336 test_acc/min=95.833336 test_acc/std=0.031250
[2025-12-31 11:48:23,064689][I][examples/fsdp:340:fsdp_main] epoch=2 dt=0.367236 train_loss=0.163549 test_loss=0.074778 test_acc=97.731812 dt/mean=0.364841 dt/max=0.372286 dt/min=0.358430 dt/std=0.004049 train_loss/mean=0.163549 train_loss/max=0.163549 train_loss/min=0.163549 train_loss/std=0.000000 test_loss/mean=0.074778 test_loss/max=0.074778 test_loss/min=0.074778 test_loss/std=0.000000 test_acc/mean=97.731812 test_acc/max=97.731812 test_acc/min=97.731812 test_acc/std=0.000000
[2025-12-31 11:48:23,459625][I][examples/fsdp:340:fsdp_main] epoch=3 dt=0.356768 train_loss=0.113108 test_loss=0.059584 test_acc=97.981613 dt/mean=0.357609 dt/max=0.359823 dt/min=0.354217 dt/std=0.001847 train_loss/mean=0.113108 train_loss/max=0.113108 train_loss/min=0.113108 train_loss/std=0.000000 test_loss/mean=0.059584 test_loss/max=0.059584 test_loss/min=0.059584 test_loss/std=0.000000 test_acc/mean=97.981613 test_acc/max=97.981613 test_acc/min=97.981613 test_acc/std=0.000000
[2025-12-31 11:48:23,857437][I][examples/fsdp:340:fsdp_main] epoch=4 dt=0.359818 train_loss=0.092698 test_loss=0.050447 test_acc=98.311348 dt/mean=0.359508 dt/max=0.362709 dt/min=0.355736 dt/std=0.002326 train_loss/mean=0.092698 train_loss/max=0.092698 train_loss/min=0.092698 train_loss/std=0.000000 test_loss/mean=0.050447 test_loss/max=0.050447 test_loss/min=0.050447 test_loss/std=0.000000 test_acc/mean=98.311356 test_acc/max=98.311348 test_acc/min=98.311348 test_acc/std=0.000000
[2025-12-31 11:48:24,257505][I][examples/fsdp:340:fsdp_main] epoch=5 dt=0.359638 train_loss=0.082438 test_loss=0.047144 test_acc=98.381294 dt/mean=0.360006 dt/max=0.362248 dt/min=0.357063 dt/std=0.001731 train_loss/mean=0.082438 train_loss/max=0.082438 train_loss/min=0.082438 train_loss/std=0.000000 test_loss/mean=0.047144 test_loss/max=0.047144 test_loss/min=0.047144 test_loss/std=0.000000 test_acc/mean=98.381302 test_acc/max=98.381294 test_acc/min=98.381294 test_acc/std=0.000000
[2025-12-31 11:48:24,656790][I][examples/fsdp:340:fsdp_main] epoch=6 dt=0.359433 train_loss=0.076101 test_loss=0.045158 test_acc=98.421265 dt/mean=0.360550 dt/max=0.363060 dt/min=0.358042 dt/std=0.001495 train_loss/mean=0.076101 train_loss/max=0.076101 train_loss/min=0.076101 train_loss/std=0.000022 test_loss/mean=0.045158 test_loss/max=0.045158 test_loss/min=0.045158 test_loss/std=0.000000 test_acc/mean=98.421265 test_acc/max=98.421265 test_acc/min=98.421265 test_acc/std=0.000000
[2025-12-31 11:48:25,063983][I][examples/fsdp:340:fsdp_main] epoch=7 dt=0.366541 train_loss=0.070617 test_loss=0.043559 test_acc=98.481216 dt/mean=0.366717 dt/max=0.369678 dt/min=0.363789 dt/std=0.002093 train_loss/mean=0.070617 train_loss/max=0.070617 train_loss/min=0.070617 train_loss/std=0.000000 test_loss/mean=0.043559 test_loss/max=0.043559 test_loss/min=0.043559 test_loss/std=0.000000 test_acc/mean=98.481224 test_acc/max=98.481216 test_acc/min=98.481216 test_acc/std=0.000000
[2025-12-31 11:48:25,463280][I][examples/fsdp:340:fsdp_main] epoch=8 dt=0.359435 train_loss=0.069483 test_loss=0.042672 test_acc=98.511192 dt/mean=0.360584 dt/max=0.363287 dt/min=0.357675 dt/std=0.001700 train_loss/mean=0.069483 train_loss/max=0.069483 train_loss/min=0.069483 train_loss/std=0.000022 test_loss/mean=0.042672 test_loss/max=0.042672 test_loss/min=0.042672 test_loss/std=0.000000 test_acc/mean=98.511192 test_acc/max=98.511192 test_acc/min=98.511192 test_acc/std=0.000000
[2025-12-31 11:48:25,864237][I][examples/fsdp:340:fsdp_main] epoch=9 dt=0.360543 train_loss=0.066796 test_loss=0.041592 test_acc=98.511192 dt/mean=0.360639 dt/max=0.363370 dt/min=0.357561 dt/std=0.001879 train_loss/mean=0.066796 train_loss/max=0.066796 train_loss/min=0.066796 train_loss/std=0.000000 test_loss/mean=0.041592 test_loss/max=0.041592 test_loss/min=0.041592 test_loss/std=0.000011 test_acc/mean=98.511192 test_acc/max=98.511192 test_acc/min=98.511192 test_acc/std=0.000000
[2025-12-31 11:48:26,265706][I][examples/fsdp:340:fsdp_main] epoch=10 dt=0.360412 train_loss=0.065580 test_loss=0.041717 test_acc=98.491203 dt/mean=0.361177 dt/max=0.364643 dt/min=0.357440 dt/std=0.002297 train_loss/mean=0.065580 train_loss/max=0.065580 train_loss/min=0.065580 train_loss/std=0.000000 test_loss/mean=0.041717 test_loss/max=0.041717 test_loss/min=0.041717 test_loss/std=0.000011 test_acc/mean=98.491203 test_acc/max=98.491203 test_acc/min=98.491203 test_acc/std=0.000000
[2025-12-31 11:48:26,267509][I][examples/fsdp:342:fsdp_main] 11 epochs took 28.8s
[2025-12-31 11:48:26,301229][I][ezpz/history:2385:finalize] Saving plots to /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/plots/mplot (matplotlib) and /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/plots/tplot (tplot)
                               dt                                                        dt/min
    ┌──────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
12.5┤▌                                                     │11.8┤-                                                     │
    │▚                                                     │ 9.9┤-                                                     │
10.4┤▝▖                                                    │ 8.0┤ -                                                    │
    │ ▚                                                    │ 6.1┤  -                                                   │
 8.4┤ ▝▖                                                   │    │   -                                                  │
    │  ▌                                                   │ 4.2┤    -                                                 │
 6.4┤  ▐                                                   │ 2.3┤     -                                                │
    │   ▌                                                  │ 0.4┤      ------------------------------------------------│
    │   ▐                                                  │    └┬────────────┬─────────────┬────────────┬────────────┬┘
 4.4┤    ▌                                                 │    1.0          3.2           5.5          7.8        10.0
    │    ▚                                                 │dt/min                        iter
 2.4┤    ▝▖                                                │                             dt/std
    │     ▚                                                │     ┌─────────────────────────────────────────────────────┐
 0.4┤     ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│0.147┤*                                                    │
    └┬────────────┬─────────────┬────────────┬────────────┬┘0.123┤*                                                    │
    1.0          3.2           5.5          7.8        10.0 0.098┤ *                                                   │
dt                            iter                          0.074┤  *                                                  │
                             dt/mean                             │   *                                                 │
    ┌──────────────────────────────────────────────────────┐0.050┤    *                                                │
12.0┤·                                                     │0.026┤     *                                               │
    │·                                                     │0.001┤      ***********************************************│
10.0┤·                                                     │     └┬────────────┬────────────┬────────────┬────────────┬┘
    │ ·                                                    │     1.0          3.2          5.5          7.8        10.0
    │ ·                                                    │dt/std                        iter
 8.1┤  ·                                                   │                             dt/max
    │  ·                                                   │    ┌──────────────────────────────────────────────────────┐
 6.2┤   ·                                                  │12.5┤+                                                     │
    │   ·                                                  │10.4┤+                                                     │
 4.2┤   ·                                                  │ 8.4┤ +                                                    │
    │    ·                                                 │ 6.4┤  +                                                   │
    │    ·                                                 │    │   +                                                  │
 2.3┤     ·                                                │ 4.4┤    +                                                 │
    │     ·                                                │ 2.4┤     +                                                │
 0.4┤      ················································│ 0.4┤      ++++++++++++++++++++++++++++++++++++++++++++++++│
    └┬────────────┬─────────────┬────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
    1.0          3.2           5.5          7.8        10.0     1.0          3.2           5.5          7.8        10.0
dt/mean                       iter                          dt/max                        iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/plots/tplot/dt.txt
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
12.5┤ ++ dt/max                                                                                                        │
    │ -- dt/min                                                                                                        │
    │ ·· dt/mean                                                                                                       │
    │ ▞▞ dt                                                                                                            │
    │-▚                                                                                                                │
    │ ▐                                                                                                                │
10.4┤ ·▌                                                                                                               │
    │ -▚                                                                                                               │
    │  ▐                                                                                                               │
    │  -▌                                                                                                              │
    │   ▐                                                                                                              │
    │   ▝▖                                                                                                             │
 8.4┤   -▌                                                                                                             │
    │    ▐                                                                                                             │
    │    ▝▖                                                                                                            │
    │     ▌                                                                                                            │
    │     ▐                                                                                                            │
 6.4┤     ▝▖                                                                                                           │
    │      ▚                                                                                                           │
    │      ▐+                                                                                                          │
    │       ▌                                                                                                          │
    │       ▚                                                                                                          │
    │       ▐·                                                                                                         │
 4.4┤        ▌                                                                                                         │
    │        ▚                                                                                                         │
    │        ▐·                                                                                                        │
    │         ▌                                                                                                        │
    │         ▐+                                                                                                       │
    │         ▝▖                                                                                                       │
 2.4┤          ▌                                                                                                       │
    │          ▐·                                                                                                      │
    │          ▝▖                                                                                                      │
    │           ▌                                                                                                      │
    │           ▐·                                                                                                     │
    │           ▝▖                                                                                                     │
 0.4┤            ▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
    └┬───────────────────────────┬────────────────────────────┬───────────────────────────┬───────────────────────────┬┘
    1.0                         3.2                          5.5                         7.8                       10.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/plots/tplot/dt_summary.txt
                         dt/mean hist                                                 dt/max hist
   ┌───────────────────────────────────────────────────────┐   ┌───────────────────────────────────────────────────────┐
9.0┤█████                                                  │9.0┤█████                                                  │
   │█████                                                  │   │█████                                                  │
7.5┤█████                                                  │7.5┤█████                                                  │
   │█████                                                  │   │█████                                                  │
   │█████                                                  │   │█████                                                  │
6.0┤█████                                                  │6.0┤█████                                                  │
   │█████                                                  │   │█████                                                  │
4.5┤█████                                                  │4.5┤█████                                                  │
   │█████                                                  │   │█████                                                  │
3.0┤█████                                                  │3.0┤█████                                                  │
   │█████                                                  │   │█████                                                  │
   │█████                                                  │   │█████                                                  │
1.5┤█████                                             █████│1.5┤█████                                             █████│
   │█████                                             █████│   │█████                                             █████│
0.0┤█████                                             █████│0.0┤█████                                             █████│
   └┬─────────────┬────────────┬─────────────┬────────────┬┘   └┬─────────────┬────────────┬─────────────┬────────────┬┘
  -0.2           3.0          6.2           9.3        12.5   -0.2           3.1          6.4           9.7        13.0
                          dt/min hist                                                 dt/std hist
   ┌───────────────────────────────────────────────────────┐   ┌───────────────────────────────────────────────────────┐
9.0┤█████                                                  │9.0┤█████                                                  │
   │█████                                                  │   │█████                                                  │
7.5┤█████                                                  │7.5┤█████                                                  │
   │█████                                                  │   │█████                                                  │
   │█████                                                  │   │█████                                                  │
6.0┤█████                                                  │6.0┤█████                                                  │
   │█████                                                  │   │█████                                                  │
4.5┤█████                                                  │4.5┤█████                                                  │
   │█████                                                  │   │█████                                                  │
   │█████                                                  │   │█████                                                  │
3.0┤█████                                                  │3.0┤█████                                                  │
   │█████                                                  │   │█████                                                  │
1.5┤█████                                                  │1.5┤█████                                                  │
   │█████                                             █████│   │█████                                             █████│
   │█████                                             █████│   │█████                                             █████│
0.0┤█████                                             █████│0.0┤█████                                             █████│
   └┬─────────────┬────────────┬─────────────┬────────────┬┘   └┬─────────────┬────────────┬─────────────┬────────────┬┘
  -0.2           3.0          6.1           9.2        12.3   -0.005        0.035        0.074         0.114      0.153
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/plots/tplot/dt_hist.txt
                            test_acc                                                  test_acc/min
     ┌─────────────────────────────────────────────────────┐     ┌─────────────────────────────────────────────────────┐
98.51┤                       ▗▄▄▄▄▄▄▄▄▄▄▄▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀│98.51┤                       ------------------------------│
     │               ▗▄▞▀▀▀▀▀▘                             │98.06┤            -----------                              │
98.06┤           ▗▄▞▀▘                                     │97.62┤      ------                                         │
     │        ▄▄▀▘                                         │97.17┤     -                                               │
97.62┤     ▗▀▀                                             │     │    -                                                │
     │     ▌                                               │96.73┤   -                                                 │
97.17┤    ▞                                                │96.28┤  -                                                  │
     │   ▗▘                                                │95.83┤--                                                   │
     │   ▌                                                 │     └┬────────────┬────────────┬────────────┬────────────┬┘
96.73┤  ▐                                                  │     1.0          3.2          5.5          7.8        10.0
     │ ▗▘                                                  │test_acc/min                  iter
96.28┤ ▞                                                   │                           test_acc/std
     │▐                                                    │      ┌────────────────────────────────────────────────────┐
95.83┤▌                                                    │0.0312┤*                                                   │
     └┬────────────┬────────────┬────────────┬────────────┬┘0.0260┤*                                                   │
     1.0          3.2          5.5          7.8        10.0 0.0208┤ *                                                  │
test_acc                      iter                          0.0156┤  *                                                 │
                          test_acc/mean                           │   *                                                │
     ┌─────────────────────────────────────────────────────┐0.0104┤    *                                               │
98.51┤                             ························│0.0052┤     *                                              │
     │                 ············                        │0.0000┤      **********************************************│
98.06┤               ··                                    │      └┬────────────┬────────────┬───────────┬────────────┬┘
     │            ···                                      │      1.0          3.2          5.5         7.8        10.0
     │      ······                                         │test_acc/std                   iter
97.62┤     ·                                               │                          test_acc/max
     │    ·                                                │     ┌─────────────────────────────────────────────────────┐
97.17┤    ·                                                │98.51┤                       ++++++++++++++++++++++++++++++│
     │   ·                                                 │98.06┤            +++++++++++                              │
96.73┤   ·                                                 │97.62┤      ++++++                                         │
     │  ·                                                  │97.17┤     +                                               │
     │ ·                                                   │     │    +                                                │
96.28┤ ·                                                   │96.73┤   +                                                 │
     │·                                                    │96.28┤  +                                                  │
95.83┤·                                                    │95.83┤++                                                   │
     └┬────────────┬────────────┬────────────┬────────────┬┘     └┬────────────┬────────────┬────────────┬────────────┬┘
     1.0          3.2          5.5          7.8        10.0      1.0          3.2          5.5          7.8        10.0
test_acc/mean                 iter                          test_acc/max                  iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/plots/tplot/test_acc.txt
     ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
98.51┤ ++ test_acc/max                                                           ▄▄▄▄▄▄▄▄▄▄▄▄▞▀▀▀▀▀▀▀▀▀▀▀▀▚▄▄▄▄▄▄▄▄▄▄▄▄│
     │ -- test_acc/min                                  ▄▄▄▄▄▄▄▄▄▄▄▄▞▀▀▀▀▀▀▀▀▀▀▀▀                                      │
     │ ·· test_acc/mean                    ▗▄▄▄▄▄▄▀▀▀▀▀▀············                                                   │
     │ ▞▞ test_acc                       ▄▞▘············                                                               │
     │                                ▄▞▀··                                                                            │
     │                             ▗▄▀··                                                                               │
98.06┤                          ▗▄▀▘·                                                                                  │
     │                       ▗▄▀▘·                                                                                     │
     │                    ▄▞▀▘·                                                                                        │
     │                ▗▄▀▀·                                                                                            │
     │            ▗▄▞▀▘                                                                                                │
     │           ·▌                                                                                                    │
97.62┤           ▐                                                                                                     │
     │          ·▌                                                                                                     │
     │          ▐                                                                                                      │
     │         ·▌                                                                                                      │
     │         ▐                                                                                                       │
97.17┤        ·▌                                                                                                       │
     │        ▐                                                                                                        │
     │       ·▌                                                                                                        │
     │       ▐                                                                                                         │
     │      ·▌                                                                                                         │
     │      ▐                                                                                                          │
96.73┤     ·▌                                                                                                          │
     │     ▐                                                                                                           │
     │    ·▌                                                                                                           │
     │    ▐                                                                                                            │
     │   ·▌                                                                                                            │
     │   ▐                                                                                                             │
96.28┤  ·▌                                                                                                             │
     │  ▐                                                                                                              │
     │ ·▌                                                                                                              │
     │ ▐                                                                                                               │
     │·▌                                                                                                               │
     │▐                                                                                                                │
95.83┤▌                                                                                                                │
     └┬───────────────────────────┬───────────────────────────┬───────────────────────────┬───────────────────────────┬┘
     1.0                         3.2                         5.5                         7.8                       10.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/plots/tplot/test_acc_summary.txt
                      test_acc/mean hist                                           test_acc/max hist
   ┌───────────────────────────────────────────────────────┐   ┌───────────────────────────────────────────────────────┐
7.0┤                                                  █████│7.0┤                                                  █████│
   │                                                  █████│   │                                                  █████│
5.8┤                                                  █████│5.8┤                                                  █████│
   │                                                  █████│   │                                                  █████│
   │                                                  █████│   │                                                  █████│
4.7┤                                                  █████│4.7┤                                                  █████│
   │                                                  █████│   │                                                  █████│
3.5┤                                                  █████│3.5┤                                                  █████│
   │                                                  █████│   │                                                  █████│
2.3┤                                                  █████│2.3┤                                                  █████│
   │                                                  █████│   │                                                  █████│
   │                                                  █████│   │                                                  █████│
1.2┤█████                                  ██████████ █████│1.2┤█████                                  ██████████ █████│
   │█████                                  ██████████ █████│   │█████                                  ██████████ █████│
0.0┤█████                                  ██████████ █████│0.0┤█████                                  ██████████ █████│
   └┬─────────────┬────────────┬─────────────┬────────────┬┘   └┬─────────────┬────────────┬─────────────┬────────────┬┘
  95.71         96.44        97.17         97.90      98.63   95.71         96.44        97.17         97.90      98.63
                       test_acc/min hist                                           test_acc/std hist
   ┌───────────────────────────────────────────────────────┐   ┌───────────────────────────────────────────────────────┐
7.0┤                                                  █████│9.0┤█████                                                  │
   │                                                  █████│   │█████                                                  │
5.8┤                                                  █████│7.5┤█████                                                  │
   │                                                  █████│   │█████                                                  │
   │                                                  █████│   │█████                                                  │
4.7┤                                                  █████│6.0┤█████                                                  │
   │                                                  █████│   │█████                                                  │
3.5┤                                                  █████│4.5┤█████                                                  │
   │                                                  █████│   │█████                                                  │
   │                                                  █████│   │█████                                                  │
2.3┤                                                  █████│3.0┤█████                                                  │
   │                                                  █████│   │█████                                                  │
1.2┤                                                  █████│1.5┤█████                                                  │
   │█████                                  ██████████ █████│   │█████                                             █████│
   │█████                                  ██████████ █████│   │█████                                             █████│
0.0┤█████                                  ██████████ █████│0.0┤█████                                             █████│
   └┬─────────────┬────────────┬─────────────┬────────────┬┘   └┬─────────────┬────────────┬─────────────┬────────────┬┘
  95.71         96.44        97.17         97.90      98.63   -0.0014      0.0071       0.0156        0.0241     0.0326
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/plots/tplot/test_acc_hist.txt
                            test_loss                                                 test_loss/min
     ┌─────────────────────────────────────────────────────┐     ┌─────────────────────────────────────────────────────┐
0.136┤▌                                                    │0.136┤-                                                    │
     │▝▖                                                   │0.120┤ -                                                   │
0.120┤ ▚                                                   │0.104┤  -                                                  │
     │  ▌                                                  │0.089┤   -                                                 │
0.104┤  ▝▖                                                 │     │    -                                                │
     │   ▐                                                 │0.073┤     --                                              │
0.089┤    ▚                                                │0.057┤       -----------                                   │
     │    ▝▖                                               │0.042┤                  -----------------------------------│
     │     ▝▖                                              │     └┬────────────┬────────────┬────────────┬────────────┬┘
0.073┤      ▝▚▖                                            │     1.0          3.2          5.5          7.8        10.0
     │        ▝▚▖                                          │test_loss/min                 iter
0.057┤          ▝▀▄▄▄                                      │                            test_loss/std
     │               ▀▀▀▄▄▄▄▄▄                             │         ┌─────────────────────────────────────────────────┐
0.042┤                        ▀▀▀▀▀▀▀▀▀▀▀▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│0.0000108┤                                           ******│
     └┬────────────┬────────────┬────────────┬────────────┬┘0.0000090┤                                          *      │
     1.0          3.2          5.5          7.8        10.0 0.0000072┤                                         *       │
test_loss                     iter                          0.0000054┤                                        *        │
                         test_loss/mean                              │                                       *         │
     ┌─────────────────────────────────────────────────────┐0.0000036┤                                      *          │
0.136┤·                                                    │0.0000018┤                                     *           │
     │·                                                    │0.0000000┤**************************************           │
0.120┤ ·                                                   │         └┬───────────┬───────────┬───────────┬───────────┬┘
     │  ·                                                  │         1.0         3.2         5.5         7.8       10.0
     │  ·                                                  │test_loss/std                   iter
0.104┤   ·                                                 │                          test_loss/max
     │    ·                                                │     ┌─────────────────────────────────────────────────────┐
0.089┤    ·                                                │0.136┤+                                                    │
     │     ·                                               │0.120┤ +                                                   │
0.073┤      ·                                              │0.104┤  +                                                  │
     │       ···                                           │0.089┤   +                                                 │
     │          ···                                        │     │    +                                                │
0.057┤             ··                                      │0.073┤     ++                                              │
     │               ···············                       │0.057┤       +++++++++++                                   │
0.042┤                              ·······················│0.042┤                  +++++++++++++++++++++++++++++++++++│
     └┬────────────┬────────────┬────────────┬────────────┬┘     └┬────────────┬────────────┬────────────┬────────────┬┘
     1.0          3.2          5.5          7.8        10.0      1.0          3.2          5.5          7.8        10.0
test_loss/mean                iter                          test_loss/max                 iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/plots/tplot/test_loss.txt
     ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
0.136┤ ++ test_loss/max                                                                                                │
     │ -- test_loss/min                                                                                                │
     │ ·· test_loss/mean                                                                                               │
     │ ▞▞ test_loss                                                                                                    │
     │  ▌                                                                                                              │
     │  ▐                                                                                                              │
0.120┤   ▚                                                                                                             │
     │   ▝▖                                                                                                            │
     │    ▚                                                                                                            │
     │    ▝▖                                                                                                           │
     │     ▚                                                                                                           │
     │     ▝▖                                                                                                          │
0.104┤      ▐                                                                                                          │
     │      ·▌                                                                                                         │
     │       ▐                                                                                                         │
     │       ·▌                                                                                                        │
     │        ▐                                                                                                        │
0.089┤        ·▚                                                                                                       │
     │         ▝▖                                                                                                      │
     │         ·▚                                                                                                      │
     │          ▝▖                                                                                                     │
     │          ·▚                                                                                                     │
     │           ▝▖                                                                                                    │
0.073┤            ▝▄                                                                                                   │
     │             ·▀▄                                                                                                 │
     │               ·▀▚▖                                                                                              │
     │                  ▝▚▖                                                                                            │
     │                    ▝▀▄                                                                                          │
     │                       ▀▄▖                                                                                       │
0.057┤                         ▝▀▄▄                                                                                    │
     │                             ▀▚▄▖                                                                                │
     │                                ▝▀▄▄                                                                             │
     │                                   ·▀▀▄▄▄▄                                                                       │
     │                                      ····▀▀▀▀▄▄▄▄▖                                                              │
     │                                                  ▝▀▀▀▀▀▀▀▀▀▀▀▀▄▄▄▄▄▄·······                                     │
0.042┤                                                                     ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
     └┬───────────────────────────┬───────────────────────────┬───────────────────────────┬───────────────────────────┬┘
     1.0                         3.2                         5.5                         7.8                       10.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/plots/tplot/test_loss_summary.txt
                      test_loss/mean hist                                         test_loss/max hist
   ┌───────────────────────────────────────────────────────┐   ┌───────────────────────────────────────────────────────┐
7.0┤█████                                                  │7.0┤█████                                                  │
   │█████                                                  │   │█████                                                  │
5.8┤█████                                                  │5.8┤█████                                                  │
   │█████                                                  │   │█████                                                  │
   │█████                                                  │   │█████                                                  │
4.7┤█████                                                  │4.7┤█████                                                  │
   │█████                                                  │   │█████                                                  │
3.5┤█████                                                  │3.5┤█████                                                  │
   │█████                                                  │   │█████                                                  │
2.3┤█████                                                  │2.3┤█████                                                  │
   │█████                                                  │   │█████                                                  │
   │█████                                                  │   │█████                                                  │
1.2┤█████ █████      █████                            █████│1.2┤█████ █████      █████                            █████│
   │█████ █████      █████                            █████│   │█████ █████      █████                            █████│
0.0┤█████ █████      █████                            █████│0.0┤█████ █████      █████                            █████│
   └┬─────────────┬────────────┬─────────────┬────────────┬┘   └┬─────────────┬────────────┬─────────────┬────────────┬┘
  0.037         0.063        0.089         0.114      0.140   0.037         0.063        0.089         0.114      0.140
                      test_loss/min hist                                          test_loss/std hist
   ┌───────────────────────────────────────────────────────┐   ┌───────────────────────────────────────────────────────┐
7.0┤█████                                                  │8.0┤█████                                                  │
   │█████                                                  │   │█████                                                  │
5.8┤█████                                                  │6.7┤█████                                                  │
   │█████                                                  │   │█████                                                  │
   │█████                                                  │   │█████                                                  │
4.7┤█████                                                  │5.3┤█████                                                  │
   │█████                                                  │   │█████                                                  │
3.5┤█████                                                  │4.0┤█████                                                  │
   │█████                                                  │   │█████                                                  │
   │█████                                                  │   │█████                                                  │
2.3┤█████                                                  │2.7┤█████                                                  │
   │█████                                                  │   │█████                                             █████│
1.2┤█████                                                  │1.3┤█████                                             █████│
   │█████ █████      █████                            █████│   │█████                                             █████│
   │█████ █████      █████                            █████│   │█████                                             █████│
0.0┤█████ █████      █████                            █████│0.0┤█████                                             █████│
   └┬─────────────┬────────────┬─────────────┬────────────┬┘   └┬─────────────┬────────────┬─────────────┬─────────────┘
  0.037         0.063        0.089         0.114      0.140   -0.0000005   0.0000025   0.0000054     0.0000083
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/plots/tplot/test_loss_hist.txt
                           train_loss                                                train_loss/min
     ┌─────────────────────────────────────────────────────┐     ┌─────────────────────────────────────────────────────┐
0.573┤▌                                                    │0.573┤-                                                    │
     │▐                                                    │0.488┤ -                                                   │
0.488┤ ▌                                                   │0.404┤  -                                                  │
     │ ▐                                                   │0.319┤   -                                                 │
0.404┤  ▌                                                  │     │    -                                                │
     │  ▝▖                                                 │0.235┤     -                                               │
0.319┤   ▚                                                 │0.150┤      -------                                        │
     │   ▝▖                                                │0.066┤             ----------------------------------------│
     │    ▚                                                │     └┬────────────┬────────────┬────────────┬────────────┬┘
0.235┤    ▝▖                                               │     1.0          3.2          5.5          7.8        10.0
     │     ▚                                               │train_loss/min                iter
0.150┤      ▚▄▄                                            │                           train_loss/std
     │         ▀▀▀▄▄▄                                      │        ┌──────────────────────────────────────────────────┐
0.066┤               ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│0.000173┤*                                                 │
     └┬────────────┬────────────┬────────────┬────────────┬┘0.000144┤*                                                 │
     1.0          3.2          5.5          7.8        10.0 0.000115┤ *                                                │
train_loss                    iter                          0.000086┤  *                                               │
                         train_loss/mean                            │  *                                               │
     ┌─────────────────────────────────────────────────────┐0.000058┤   *                                              │
0.573┤·                                                    │0.000029┤    *                      *          *           │
     │·                                                    │0.000000┤     ********************** ********** ***********│
0.488┤ ·                                                   │        └┬───────────┬────────────┬───────────┬───────────┬┘
     │ ·                                                   │        1.0         3.2          5.5         7.8       10.0
     │  ·                                                  │train_loss/std                  iter
0.404┤  ·                                                  │                         train_loss/max
     │   ·                                                 │     ┌─────────────────────────────────────────────────────┐
0.319┤   ·                                                 │0.573┤+                                                    │
     │    ·                                                │0.488┤ +                                                   │
0.235┤    ·                                                │0.404┤  +                                                  │
     │     ·                                               │0.319┤   +                                                 │
     │      ·                                              │     │    +                                                │
0.150┤       ···                                           │0.235┤     +                                               │
     │          ········                                   │0.150┤      +++++++                                        │
0.066┤                  ···································│0.066┤             ++++++++++++++++++++++++++++++++++++++++│
     └┬────────────┬────────────┬────────────┬────────────┬┘     └┬────────────┬────────────┬────────────┬────────────┬┘
     1.0          3.2          5.5          7.8        10.0      1.0          3.2          5.5          7.8        10.0
train_loss/mean               iter                          train_loss/max                iter
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/plots/tplot/train_loss.txt
     ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
0.573┤ ++ train_loss/max                                                                                               │
     │ -- train_loss/min                                                                                               │
     │ ·· train_loss/mean                                                                                              │
     │ ▞▞ train_loss                                                                                                   │
     │ ▐                                                                                                               │
     │  ▌                                                                                                              │
0.488┤  ▐                                                                                                              │
     │   ▌                                                                                                             │
     │   ▐                                                                                                             │
     │   ▝▖                                                                                                            │
     │    ▚                                                                                                            │
     │    ▝▖                                                                                                           │
0.404┤     ▌                                                                                                           │
     │     ▐                                                                                                           │
     │      ▌                                                                                                          │
     │      ▐                                                                                                          │
     │      ·▌                                                                                                         │
0.319┤       ▚                                                                                                         │
     │       ▝▖                                                                                                        │
     │        ▚                                                                                                        │
     │        ▐                                                                                                        │
     │         ▌                                                                                                       │
     │         ▐                                                                                                       │
0.235┤         ·▌                                                                                                      │
     │          ▐                                                                                                      │
     │          ▝▖                                                                                                     │
     │           ▚                                                                                                     │
     │           ▝▖                                                                                                    │
     │            ▚                                                                                                    │
0.150┤             ▀▚▄▖                                                                                                │
     │                ▝▀▄▄                                                                                             │
     │                   ·▀▚▄▖                                                                                         │
     │                      ·▝▀▚▄▄▄▖                                                                                   │
     │                          ···▝▀▀▀▚▄▄▄▄                                                                           │
     │                                      ▀▀▀▀▀▀▄▄▄▄▄▄▖············                                                  │
0.066┤                                                  ▝▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
     └┬───────────────────────────┬───────────────────────────┬───────────────────────────┬───────────────────────────┬┘
     1.0                         3.2                         5.5                         7.8                       10.0
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/plots/tplot/train_loss_summary.txt
                     train_loss/mean hist                                         train_loss/max hist
   ┌───────────────────────────────────────────────────────┐   ┌───────────────────────────────────────────────────────┐
8.0┤█████                                                  │8.0┤█████                                                  │
   │█████                                                  │   │█████                                                  │
6.7┤█████                                                  │6.7┤█████                                                  │
   │█████                                                  │   │█████                                                  │
   │█████                                                  │   │█████                                                  │
5.3┤█████                                                  │5.3┤█████                                                  │
   │█████                                                  │   │█████                                                  │
4.0┤█████                                                  │4.0┤█████                                                  │
   │█████                                                  │   │█████                                                  │
2.7┤█████                                                  │2.7┤█████                                                  │
   │█████                                                  │   │█████                                                  │
   │█████                                                  │   │█████                                                  │
1.3┤█████ █████                                       █████│1.3┤█████ █████                                       █████│
   │█████ █████                                       █████│   │█████ █████                                       █████│
0.0┤█████ █████                                       █████│0.0┤█████ █████                                       █████│
   └┬─────────────┬────────────┬─────────────┬────────────┬┘   └┬─────────────┬────────────┬─────────────┬────────────┬┘
  0.04          0.18         0.32          0.46        0.60   0.04          0.18         0.32          0.46        0.60
                      train_loss/min hist                                         train_loss/std hist
   ┌───────────────────────────────────────────────────────┐   ┌───────────────────────────────────────────────────────┐
8.0┤█████                                                  │7.0┤█████                                                  │
   │█████                                                  │   │█████                                                  │
6.7┤█████                                                  │5.8┤█████                                                  │
   │█████                                                  │   │█████                                                  │
   │█████                                                  │   │█████                                                  │
5.3┤█████                                                  │4.7┤█████                                                  │
   │█████                                                  │   │█████                                                  │
4.0┤█████                                                  │3.5┤█████                                                  │
   │█████                                                  │   │█████                                                  │
   │█████                                                  │   │█████                                                  │
2.7┤█████                                                  │2.3┤█████                                                  │
   │█████                                                  │   │█████ █████                                            │
1.3┤█████                                                  │1.2┤█████ █████                                            │
   │█████ █████                                       █████│   │█████ █████                                       █████│
   │█████ █████                                       █████│   │█████ █████                                       █████│
0.0┤█████ █████                                       █████│0.0┤█████ █████                                       █████│
   └┬─────────────┬────────────┬─────────────┬────────────┬┘   └┬─────────────┬────────────┬─────────────┬────────────┬┘
  0.04          0.18         0.32          0.46        0.60   -0.000008   0.000039     0.000086      0.000133  0.000180
text saved in /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/plots/tplot/train_loss_hist.txt
[2025-12-31 11:48:30,915286][W][ezpz/history:2320:save_dataset] Unable to save dataset to W&B, skipping!
[2025-12-31 11:48:30,916960][I][utils/__init__:651:dataset_to_h5pyfile] Saving dataset to: /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/train_dataset.h5
[2025-12-31 11:48:30,929437][I][ezpz/history:2433:finalize] Saving history report to /lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz-fsdp/2025-12-31-114826/report.md
[2025-12-31 11:48:30,935764][I][examples/fsdp:360:fsdp_main] dataset=<xarray.Dataset> Size: 2kB
Dimensions:          (draw: 10)
Coordinates:
  * draw             (draw) int64 80B 0 1 2 3 4 5 6 7 8 9
Data variables: (12/25)
    epoch            (draw) int64 80B 1 2 3 4 5 6 7 8 9 10
    dt               (draw) float64 80B 12.46 0.3672 0.3568 ... 0.3605 0.3604
    train_loss       (draw) float32 40B 0.5726 0.1635 0.1131 ... 0.0668 0.06558
    test_loss        (draw) float32 40B 0.1358 0.07478 ... 0.04159 0.04172
    test_acc         (draw) float32 40B 95.83 97.73 97.98 ... 98.51 98.51 98.49
    epoch_mean       (draw) float64 80B 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0
    ...               ...
    test_loss_min    (draw) float64 80B 0.1358 0.07478 ... 0.04159 0.04172
    test_loss_std    (draw) float64 80B 0.0 0.0 0.0 ... 0.0 1.079e-05 1.079e-05
    test_acc_mean    (draw) float64 80B 95.83 97.73 97.98 ... 98.51 98.51 98.49
    test_acc_max     (draw) float64 80B 95.83 97.73 97.98 ... 98.51 98.51 98.49
    test_acc_min     (draw) float64 80B 95.83 97.73 97.98 ... 98.51 98.51 98.49
    test_acc_std     (draw) float64 80B 0.03125 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
[2025-12-31 11:48:30,939991][I][examples/fsdp:452:<module>] Took 36.84 seconds
wandb:
wandb: 🚀 View run ethereal-dream-85 at: 
wandb: Find logs at: ../../../../../../lus/tegu/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20251231_114755-z78eje8p/logs
[2025-12-31 11:48:32,494248][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2025-12-31-114832]----
[2025-12-31 11:48:32,495029][I][ezpz/launch:448:launch] Execution finished with 0.
[2025-12-31 11:48:32,495446][I][ezpz/launch:449:launch] Executing finished in 44.91 seconds.
[2025-12-31 11:48:32,495793][I][ezpz/launch:450:launch] Took 44.92 seconds to run. Exiting.

```


</details>


