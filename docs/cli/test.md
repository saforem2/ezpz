# ✅ `ezpz test`

Run the bundled test suite (great for first-time validation):

```bash
ezpz test
```

(should take ~ 1 min)

- Or, if you already have `torch` + `mpi4py`, try without installing:

    ```bash
    TMPDIR=$(pwd) uv run \
        --python=$(which python3) \
        --with "git+https://github.com/saforem2/ezpz" \
        ezpz test
    ```

    - <details closed><summary>localhost (MacBook Pro):</summary>

        ```bash
        (ezpz)
        #[12/26/25 @ 14:59:27][~/v/s/ezpz][distributed-metrics][$✘»!?] [20s]
        ; ezpz test
        [2025-12-26 14:59:36,627513][I][ezpz/test_dist:132:__post_init__] Outputs will be saved to /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936
        [2025-12-26 14:59:36,629251][I][ezpz/dist:1506:setup_torch_distributed] Using fw='ddp' with torch_{device,backend}= {mps, gloo}
        [2025-12-26 14:59:36,635161][I][ezpz/dist:1371:setup_torch_DDP] Caught MASTER_PORT=58309 from environment!
        [2025-12-26 14:59:36,635780][I][ezpz/dist:1387:setup_torch_DDP] Using torch.distributed.init_process_group with
        - master_addr='Sams-MacBook-Pro-2.local'
        - master_port='58309'
        - world_size=2
        - rank=0
        - local_rank=0
        - timeout=datetime.timedelta(seconds=3600)
        - backend='gloo'
        [2025-12-26 14:59:36,636493][I][ezpz/dist:1019:init_process_group] Calling torch.distributed.init_process_group_with: rank=0 world_size=2 backend=gloo
        [2025-12-26 14:59:36,753207][I][ezpz/dist:1732:setup_torch] Using device='mps' with backend='gloo' + 'gloo' for distributed training.
        [2025-12-26 14:59:36,781940][I][ezpz/dist:1779:setup_torch] ['Sams-MacBook-Pro-2.local'][device='mps'][node=0/0][rank=1/1][local_rank=1/1]
        [2025-12-26 14:59:36,806242][W][ezpz/dist:544:print_dist_setup] Using [2 / 2] available "mps" devices !!
        [2025-12-26 14:59:36,806669][I][ezpz/dist:1779:setup_torch] ['Sams-MacBook-Pro-2.local'][device='mps'][node=0/0][rank=0/1][local_rank=0/1]
        [2025-12-26 14:59:36,807024][I][ezpz/test_dist:678:main] Took: 0.18 seconds to setup torch
        [2025-12-26 14:59:36,816995][I][ezpz/test_dist:461:train] Model size: 567434 parameters
        [2025-12-26 14:59:36,817813][I][ezpz/test_dist:465:train]
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
        [2025-12-26 14:59:36,818532][I][ezpz/test_dist:473:train] Took: 0.00975050003034994 seconds to build model
        [2025-12-26 14:59:36,818765][W][ezpz/test_dist:590:build_model_and_optimizer] MPS does not support torch.distributed collectives; falling back to CPU
        [2025-12-26 14:59:36,819313][I][ezpz/test_dist:601:build_model_and_optimizer] model=
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
        [2025-12-26 14:59:37,383487][I][ezpz/dist:685:wrap_model] Wrapping model with: ddp
        [2025-12-26 14:59:37,402510][I][ezpz/test_dist:479:train] Took: 0.58 seconds to build optimizer
        [2025-12-26 14:59:37,586325][I][ezpz/history:220:__init__] Using History with distributed_history=True
        [2025-12-26 14:59:37,668674][I][ezpz/dist:2044:setup_wandb] Setting up wandb from rank=0
        [2025-12-26 14:59:37,669043][I][ezpz/dist:2045:setup_wandb] Using WB_PROJECT=ezpz.test_dist
        wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
        wandb: Tracking run with wandb version 0.23.1
        wandb: Run data is saved locally in /Users/samforeman/vibes/saforem2/ezpz/wandb/run-20251226_145937-rj4d7rus
        wandb: Run `wandb offline` to turn off syncing.
        wandb: Syncing run soft-grass-6851
        wandb:  View project at https://wandb.ai/aurora_gpt/ezpz.test_dist
        wandb:  View run at https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/rj4d7rus
        [2025-12-26 14:59:39,090331][I][ezpz/dist:2074:setup_wandb] wandb.run=[soft-grass-6851](https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/rj4d7rus)
        [2025-12-26 14:59:39,218933][I][ezpz/dist:2117:setup_wandb] Running on machine='localhost'
        [2025-12-26 14:59:39,479361][I][ezpz/test_dist:482:train] Took: 2.08 seconds to build trainer
        [2025-12-26 14:59:39,480013][I][ezpz/test_dist:486:train] config:
        {
          "acc_events": false,
          "backend": "DDP",
          "batch_size": 128,
          "cp": 1,
          "dataset": "mnist",
          "dataset_root": "/Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/datasets/mnist",
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
        [2025-12-26 14:59:39,481620][I][ezpz/test_dist:488:train] Took: 3.65 to get here.
        [2025-12-26 14:59:39,984314][I][ezpz/test_dist:369:train] Warmup complete at step 5
        [2025-12-26 14:59:40,063715][I][ezpz/test_dist:325:train_step] iter=10 loss=1.188046 accuracy=0.593750 dtf=0.008557 dtb=0.001808 loss/mean=1.197080 loss/max=1.206113 loss/min=1.188046 loss/std=0.009037 accuracy/mean=0.625000 accuracy/max=0.656250 accuracy/min=0.593750 accuracy/std=0.031250 dtf/mean=0.008275 dtf/max=0.008557 dtf/min=0.007993 dtf/std=0.000282 dtb/mean=0.002003 dtb/max=0.002198 dtb/min=0.001808 dtb/std=0.000195
        [2025-12-26 14:59:40,274480][I][ezpz/test_dist:325:train_step] iter=20 loss=0.650923 accuracy=0.742188 dtf=0.010504 dtb=0.008142 loss/mean=0.728713 loss/max=0.806504 loss/min=0.650923 loss/std=0.077790 accuracy/mean=0.769531 accuracy/max=0.796875 accuracy/min=0.742188 accuracy/std=0.027344 dtf/mean=0.010317 dtf/max=0.010504 dtf/min=0.010130 dtf/std=0.000187 dtb/mean=0.008175 dtb/max=0.008207 dtb/min=0.008142 dtb/std=0.000032
        [2025-12-26 14:59:40,534115][I][ezpz/test_dist:325:train_step] iter=30 loss=0.642461 accuracy=0.804688 dtf=0.009502 dtb=0.001911 loss/mean=0.528537 loss/max=0.642461 loss/min=0.414612 loss/std=0.113924 accuracy/mean=0.824219 accuracy/max=0.843750 accuracy/min=0.804688 accuracy/std=0.019531 dtf/mean=0.010211 dtf/max=0.010919 dtf/min=0.009502 dtf/std=0.000708 dtb/mean=0.001896 dtb/max=0.001911 dtb/min=0.001881 dtb/std=0.000015
        [2025-12-26 14:59:40,729254][I][ezpz/test_dist:325:train_step] iter=40 loss=0.349402 accuracy=0.898438 dtf=0.007339 dtb=0.004863 loss/mean=0.359106 loss/max=0.368810 loss/min=0.349402 loss/std=0.009704 accuracy/mean=0.890625 accuracy/max=0.898438 accuracy/min=0.882812 accuracy/std=0.007812 dtf/mean=0.007400 dtf/max=0.007461 dtf/min=0.007339 dtf/std=0.000061 dtb/mean=0.004861 dtb/max=0.004863 dtb/min=0.004860 dtb/std=0.000000
        [2025-12-26 14:59:40,904186][I][ezpz/test_dist:325:train_step] iter=50 loss=0.345590 accuracy=0.867188 dtf=0.006774 dtb=0.001858 loss/mean=0.350946 loss/max=0.356301 loss/min=0.345590 loss/std=0.005355 accuracy/mean=0.878906 accuracy/max=0.890625 accuracy/min=0.867188 accuracy/std=0.011719 dtf/mean=0.006920 dtf/max=0.007066 dtf/min=0.006774 dtf/std=0.000146 dtb/mean=0.001857 dtb/max=0.001858 dtb/min=0.001856 dtb/std=0.000001
        [2025-12-26 14:59:41,069650][I][ezpz/test_dist:325:train_step] iter=60 loss=0.376659 accuracy=0.890625 dtf=0.007758 dtb=0.001745 loss/mean=0.320235 loss/max=0.376659 loss/min=0.263812 loss/std=0.056424 accuracy/mean=0.914062 accuracy/max=0.937500 accuracy/min=0.890625 accuracy/std=0.023438 dtf/mean=0.007664 dtf/max=0.007758 dtf/min=0.007569 dtf/std=0.000095 dtb/mean=0.001749 dtb/max=0.001753 dtb/min=0.001745 dtb/std=0.000004
        [2025-12-26 14:59:41,242790][I][ezpz/test_dist:325:train_step] iter=70 loss=0.575540 accuracy=0.828125 dtf=0.007760 dtb=0.001824 loss/mean=0.494479 loss/max=0.575540 loss/min=0.413418 loss/std=0.081061 accuracy/mean=0.855469 accuracy/max=0.882812 accuracy/min=0.828125 accuracy/std=0.027344 dtf/mean=0.007917 dtf/max=0.008074 dtf/min=0.007760 dtf/std=0.000157 dtb/mean=0.001858 dtb/max=0.001892 dtb/min=0.001824 dtb/std=0.000034
        [2025-12-26 14:59:41,415724][I][ezpz/test_dist:325:train_step] iter=80 loss=0.196338 accuracy=0.953125 dtf=0.007632 dtb=0.003868 loss/mean=0.225939 loss/max=0.255540 loss/min=0.196338 loss/std=0.029601 accuracy/mean=0.933594 accuracy/max=0.953125 accuracy/min=0.914062 accuracy/std=0.019531 dtf/mean=0.007239 dtf/max=0.007632 dtf/min=0.006847 dtf/std=0.000393 dtb/mean=0.004381 dtb/max=0.004893 dtb/min=0.003868 dtb/std=0.000513
        [2025-12-26 14:59:41,579460][I][ezpz/test_dist:325:train_step] iter=90 loss=0.331747 accuracy=0.906250 dtf=0.008618 dtb=0.004053 loss/mean=0.344878 loss/max=0.358009 loss/min=0.331747 loss/std=0.013131 accuracy/mean=0.906250 accuracy/max=0.906250 accuracy/min=0.906250 accuracy/std=0.000000 dtf/mean=0.008693 dtf/max=0.008768 dtf/min=0.008618 dtf/std=0.000075 dtb/mean=0.004049 dtb/max=0.004053 dtb/min=0.004045 dtb/std=0.000004
        [2025-12-26 14:59:41,729606][I][ezpz/test_dist:325:train_step] iter=100 loss=0.188108 accuracy=0.937500 dtf=0.007073 dtb=0.001962 loss/mean=0.180938 loss/max=0.188108 loss/min=0.173769 loss/std=0.007169 accuracy/mean=0.945312 accuracy/max=0.953125 accuracy/min=0.937500 accuracy/std=0.007812 dtf/mean=0.006854 dtf/max=0.007073 dtf/min=0.006634 dtf/std=0.000219 dtb/mean=0.001962 dtb/max=0.001962 dtb/min=0.001962 dtb/std=0.000000
        [2025-12-26 14:59:41,884339][I][ezpz/test_dist:325:train_step] iter=110 loss=0.267521 accuracy=0.890625 dtf=0.007719 dtb=0.002057 loss/mean=0.383564 loss/max=0.499606 loss/min=0.267521 loss/std=0.116043 accuracy/mean=0.871094 accuracy/max=0.890625 accuracy/min=0.851562 accuracy/std=0.019531 dtf/mean=0.007575 dtf/max=0.007719 dtf/min=0.007431 dtf/std=0.000144 dtb/mean=0.002060 dtb/max=0.002063 dtb/min=0.002057 dtb/std=0.000003
        [2025-12-26 14:59:42,050014][I][ezpz/test_dist:325:train_step] iter=120 loss=0.210285 accuracy=0.937500 dtf=0.011066 dtb=0.001822 loss/mean=0.241504 loss/max=0.272723 loss/min=0.210285 loss/std=0.031219 accuracy/mean=0.937500 accuracy/max=0.937500 accuracy/min=0.937500 accuracy/std=0.000000 dtf/mean=0.010052 dtf/max=0.011066 dtf/min=0.009037 dtf/std=0.001015 dtb/mean=0.001869 dtb/max=0.001915 dtb/min=0.001822 dtb/std=0.000047
        [2025-12-26 14:59:42,230004][I][ezpz/test_dist:325:train_step] iter=130 loss=0.139174 accuracy=0.968750 dtf=0.010818 dtb=0.001807 loss/mean=0.133106 loss/max=0.139174 loss/min=0.127037 loss/std=0.006068 accuracy/mean=0.964844 accuracy/max=0.968750 accuracy/min=0.960938 accuracy/std=0.003906 dtf/mean=0.010070 dtf/max=0.010818 dtf/min=0.009322 dtf/std=0.000748 dtb/mean=0.004232 dtb/max=0.006658 dtb/min=0.001807 dtb/std=0.002425
        [2025-12-26 14:59:42,401759][I][ezpz/test_dist:325:train_step] iter=140 loss=0.217151 accuracy=0.921875 dtf=0.007524 dtb=0.001881 loss/mean=0.205181 loss/max=0.217151 loss/min=0.193212 loss/std=0.011969 accuracy/mean=0.929688 accuracy/max=0.937500 accuracy/min=0.921875 accuracy/std=0.007812 dtf/mean=0.007589 dtf/max=0.007655 dtf/min=0.007524 dtf/std=0.000065 dtb/mean=0.001849 dtb/max=0.001881 dtb/min=0.001817 dtb/std=0.000032
        [2025-12-26 14:59:42,562758][I][ezpz/test_dist:325:train_step] iter=150 loss=0.388715 accuracy=0.882812 dtf=0.006638 dtb=0.001826 loss/mean=0.378151 loss/max=0.388715 loss/min=0.367587 loss/std=0.010564 accuracy/mean=0.886719 accuracy/max=0.890625 accuracy/min=0.882812 accuracy/std=0.003906 dtf/mean=0.006729 dtf/max=0.006820 dtf/min=0.006638 dtf/std=0.000091 dtb/mean=0.001828 dtb/max=0.001829 dtb/min=0.001826 dtb/std=0.000002
        [2025-12-26 14:59:42,732920][I][ezpz/test_dist:325:train_step] iter=160 loss=0.197628 accuracy=0.921875 dtf=0.010449 dtb=0.002640 loss/mean=0.255450 loss/max=0.313271 loss/min=0.197628 loss/std=0.057821 accuracy/mean=0.917969 accuracy/max=0.921875 accuracy/min=0.914062 accuracy/std=0.003906 dtf/mean=0.010021 dtf/max=0.010449 dtf/min=0.009594 dtf/std=0.000428 dtb/mean=0.002552 dtb/max=0.002640 dtb/min=0.002463 dtb/std=0.000089
        [2025-12-26 14:59:42,889920][I][ezpz/test_dist:325:train_step] iter=170 loss=0.325840 accuracy=0.867188 dtf=0.007486 dtb=0.002018 loss/mean=0.304081 loss/max=0.325840 loss/min=0.282321 loss/std=0.021760 accuracy/mean=0.882812 accuracy/max=0.898438 accuracy/min=0.867188 accuracy/std=0.015625 dtf/mean=0.007106 dtf/max=0.007486 dtf/min=0.006727 dtf/std=0.000380 dtb/mean=0.002002 dtb/max=0.002018 dtb/min=0.001986 dtb/std=0.000016
        [2025-12-26 14:59:43,052496][I][ezpz/test_dist:325:train_step] iter=180 loss=0.146518 accuracy=0.945312 dtf=0.007811 dtb=0.001911 loss/mean=0.152537 loss/max=0.158556 loss/min=0.146518 loss/std=0.006019 accuracy/mean=0.945312 accuracy/max=0.945312 accuracy/min=0.945312 accuracy/std=0.000000 dtf/mean=0.007945 dtf/max=0.008078 dtf/min=0.007811 dtf/std=0.000133 dtb/mean=0.001863 dtb/max=0.001911 dtb/min=0.001816 dtb/std=0.000048
        [2025-12-26 14:59:43,202332][I][ezpz/test_dist:325:train_step] iter=190 loss=0.141739 accuracy=0.953125 dtf=0.009768 dtb=0.002052 loss/mean=0.185415 loss/max=0.229091 loss/min=0.141739 loss/std=0.043676 accuracy/mean=0.953125 accuracy/max=0.953125 accuracy/min=0.953125 accuracy/std=0.000000 dtf/mean=0.009895 dtf/max=0.010022 dtf/min=0.009768 dtf/std=0.000127 dtb/mean=0.002053 dtb/max=0.002054 dtb/min=0.002052 dtb/std=0.000001
        [2025-12-26 14:59:43,943497][I][ezpz/history:2385:finalize] Saving plots to /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936/plots/mplot (matplotlib) and /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936/plots/tplot (tplot)
                          accuracy                              accuracy/min
             ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
        0.984┤            ▗      ▗▙▖▖▗ ▌▖▖▄▟▗▙▄│0.961┤        -------------------------│
        0.919┤     ▗▄▙▄▌▙█▟▄▄██▄██▀█▙▟████▛▛█▜█│0.836┤ ------------ ----- --- -- -   - │
             │   ▖▗▞██▀▜▜▛█▘▝▜▛▛▘  ▘▘▘▘  ▐ ▘▘▝▌│0.711┤----                             │
        0.854┤ ▐▗▙█▘   ▝▝▌                     │0.586┤--                               │
        0.789┤ ▐██▘                            │     └┬───────┬───────┬───────┬───────┬┘
        0.724┤▗██▜                             │     1.0    49.2    97.5    145.8 194.0
             │██▌                              │accuracy/min        iter
        0.659┤█▀                               │                accuracy/std
        0.594┤▐                                │     ┌─────────────────────────────────┐
             └┬───────┬───────┬───────┬───────┬┘0.105┤  *                              │
             1.0    49.2    97.5    145.8 194.0 0.088┤ ***                             │
        accuracy            iter                0.053┤**** * * *                       │
                        accuracy/mean           0.035┤************* ******* ********** │
             ┌─────────────────────────────────┐0.000┤  *******************************│
        0.969┤           ·        ···  ·· ·····│     └┬───────┬───────┬───────┬───────┬┘
        0.910┤        ·························│     1.0    49.2    97.5    145.8 194.0
             │     ··············  ····  ·  ·· │accuracy/std        iter
        0.852┤   ······  ·                     │                accuracy/max
        0.793┤ ····                            │     ┌─────────────────────────────────┐
             │ ···                             │0.984┤         ++++ +++ +++++++++ +++++│
        0.734┤ ··                              │0.928┤  +++++++++++++++++++++++++++ +++│
        0.676┤··                               │0.816┤ +++++                           │
             │··                               │0.760┤+++                              │
        0.617┤··                               │0.648┤++                               │
             └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
             1.0    49.2    97.5    145.8 194.0      1.0    49.2    97.5    145.8 194.0
        accuracy/mean       iter                accuracy/max        iter
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936/plots/tplot/accuracy.txt
             ┌─────────────────────────────────────────────────────────────────────────┐
        0.984┤ ++ accuracy/max                            ▗▌         ▟  +      ▗▌ ▗▟+  │
             │ -- accuracy/min        +                  ▖▐▌▟  ▗   ▖ █  ▖    ▟ ▐▌ ▐█▗++│
             │ ·· accuracy/mean    +· ·  ▗▌  +  ▗·    +▗▖▌▐██  █+ ▐▙ █▗▖▌▐  ▟▌█▐▚ ▟▜█▟▞│
             │ ▞▞ accuracy       ▗▌+▟+▗▖ ▐▌  · ▗█▟▗+  ▗▐▙▌▐▜▜▗▟█·+▐█▗█▐▌▌▟+▄▜▘▜█▐█·▐██▌│
        0.918┤            ▗▗▌█▗▌▗█▌+▛▖▌▌ ▞▌▟▗▖▐████▟▗ ▌▀▜▙▘·▐▞▐▛▟▟▐▜▛█▞▝▀▝▟█▐-▐▌▐█·-▜▛▌│
             │            ▛▞▌█▐▙█▘▌▗▌▐▌▌▐▌█▝▘▐▌▀██▀▌█▞▌  ▝--▐▌▐▌▝▛▛▐▌ ▘- -▐█ -▝▌▝▛- ▐▌ │
             │    +     +▐·▌▌█▐█▜·▐▌·▐▌▚█▌▌  ▝▌-▐█ ·█▘▌     -▘-▘   ▝▌  -  -█        ▐▌ │
             │    +  ▖ ++▐·▌▚▘██·-▐▌-▐▌-█·▘     ▝▛ -▜·                     ▜        ▝▌ │
        0.852┤    +  ▌++▌▞·▘▝··▝  ▐▌ ▐▌ ▝-         - -                                 │
             │  +▟+ ▟▌▗·▛ - ··--   ▘ ▐▌                                                │
             │ ++█+ █▙▜·▌ - -·        ▘                                                │
             │ ++█+▟██▝▞     -                                                         │
        0.785┤ ++█▐█▌█--                                                               │
             │ ++█▞█▘█ -                                                               │
             │ +▗█▌█ █                                                                 │
             │ ▖██▌▜ ▜                                                                 │
             │ ▌██▌- -                                                                 │
        0.719┤ ▌██▌- -                                                                 │
             │▌▌█▌▌  -                                                                 │
             │▙▌█▌▌                                                                    │
             │█▙▘▘                                                                     │
        0.652┤▜█-                                                                      │
             │·█                                                                       │
             │·█                                                                       │
             │-█                                                                       │
        0.586┤-▝                                                                       │
             └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
             1.0              49.2              97.5              145.8           194.0
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936/plots/tplot/accuracy_summary.txt
                     accuracy/mean hist                       accuracy/max hist
            ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
        74.0┤                           ████   │71.0┤                           ████   │
        61.7┤                           ████   │59.2┤                           ████   │
            │                           ████   │    │                           ████   │
        49.3┤                           ███████│47.3┤                        ███████   │
        37.0┤                           ███████│35.5┤                        ██████████│
            │                        ██████████│    │                        ██████████│
        24.7┤                        ██████████│23.7┤                        ██████████│
        12.3┤                        ██████████│11.8┤                    ██████████████│
            │             █████████████████████│    │          ████████████████████████│
         0.0┤██████████████████████████████████│ 0.0┤██████████████████████████████████│
            └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
           0.60    0.70     0.79    0.89   0.98    0.63    0.72     0.82    0.91   1.00
                     accuracy/min hist                        accuracy/std hist
          ┌────────────────────────────────────┐    ┌──────────────────────────────────┐
        84┤                             ███    │91.0┤████                              │
          │                             ███    │    │████                              │
        70┤                             ███    │75.8┤████                              │
        56┤                             ███    │60.7┤███████                           │
          │                             ███    │    │███████                           │
        42┤                             ███████│45.5┤███████                           │
          │                         ███████████│    │███████                           │
        28┤                         ███████████│30.3┤██████████                        │
        14┤                         ███████████│15.2┤██████████                        │
          │                  ██████████████████│    │██████████   ████                 │
         0┤████████████████████████████████████│ 0.0┤████████████████████    ██████████│
          └┬────────┬────────┬───────┬────────┬┘    └┬───────┬────────┬───────┬───────┬┘
         0.57     0.67     0.77    0.88    0.98   -0.005   0.024    0.053   0.081 0.110
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936/plots/tplot/accuracy_hist.txt
                              dtb                                   dtb/min
              ┌────────────────────────────────┐      ┌────────────────────────────────┐
        0.0083┤  ▟  ▗                         ▐│0.0081┤  -  -                         -│
        0.0072┤  █  █                         ▐│0.0060┤- -- --       -             -- -│
              │ ▗█  █                        ▖▐│0.0038┤- -----  ------ ---   --- ------│
        0.0061┤▗▐█  █▖   ▖  ▗▖           ▐ ▗█▌█│0.0016┤--------------------------------│
        0.0050┤▟▐▐▖▐█▌  ▐▙▖ █▌ ▗▌      ▖ ▐ ▐█▌█│      └┬───────┬───────┬──────┬───────┬┘
        0.0038┤██▐▌▐██  ▐█▌▄█▌ ▟▌▌    ▄█ ▐ ▐█▌█│      1.0    49.2    97.5   145.8 194.0
              │██▐▌▐██  ▐█▌██▌ █▌▌    ██ ▐ ▐█▙█│dtb/min              iter
        0.0027┤██▝█▟█▜▖▖▟████▌▖█▛▚  ▖▙██▙▐▟▐███│                    dtb/std
        0.0016┤▜▝  ▀  ▜▀█▛█▛▘█▜▛▌▝█▟▀▀▜█▜▀▜▛█▜▝│       ┌───────────────────────────────┐
              └┬───────┬───────┬──────┬───────┬┘0.00326┤   *                           │
              1.0    49.2    97.5   145.8 194.0 0.00272┤   *               *           │
        dtb                  iter               0.00163┤******   * * * **  * *   *     │
                           dtb/mean             0.00109┤*******  * *** ** ** *****   **│
              ┌────────────────────────────────┐0.00000┤*******************************│
        0.0082┤  ·  ·                         ·│       └┬───────┬──────┬───────┬───────┘
        0.0071┤  ·  ·                         ·│       1.0    49.2   97.5    145.8
              │  ·  ·                       · ·│dtb/std              iter
        0.0060┤  ·· ·                      ·· ·│                    dtb/max
        0.0049┤· ·· ··       ·           · ·· ·│      ┌────────────────────────────────┐
              │· ·····  ······ · · · ··· · ·· ·│0.0090┤  ++                           +│
        0.0038┤· ·····  ············ ··· · ····│0.0078┤  ++++              + +      + +│
        0.0027┤·······  ············ ····· ····│0.0053┤+++++++  ++++++++++++ +++++ ++++│
              │································│0.0041┤+++++++  ++++++++++++ ++++++++++│
        0.0016┤ ·  ·  ························ │0.0016┤++++++++++++++++++++++++++++++++│
              └┬───────┬───────┬──────┬───────┬┘      └┬───────┬───────┬──────┬───────┬┘
              1.0    49.2    97.5   145.8 194.0       1.0    49.2    97.5   145.8 194.0
        dtb/mean             iter               dtb/max              iter
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936/plots/tplot/dtb.txt
              ┌────────────────────────────────────────────────────────────────────────┐
        0.0090┤ ++ dtb/max                                                             │
              │ -- dtb/min                                                             │
              │ ·· dtb/mean                                                           ▗│
              │ ▞▞ dtb                                                                ▌│
        0.0078┤     ▌ +    ▌                                                          ▌│
              │     ▌ +    ▌                                                          ▌│
              │     ▌ +    ▌                                                          ▌│
              │     ▌ +    ▌                                                          ▌│
        0.0065┤   ▗▌▌ ++   ▌                                 +   +                +▖  ▌│
              │   ▐▌▌ ++ + ▌                                ++  ++                ▐▌  ▌│
              │   ▐▌▙·++ + ▌                                ++  ++         ▗      ▐▌ +▌│
              │ · ▐▐▌▌·+ + ▌                  +   +        +++  ++         █    ▟ ▐▌ ▐▌│
        0.0053┤▗▌ ▐·▘▌·+ +▖▌▗        ▌       ▟▟   +        +++  ++         █    █ ▐▌ ▐▌│
              │▐▌ ▐··▌·+ ▐▌▌█        ▌       ██   +   ▖    +++  ++         █    █ ▐▌ ▐▌│
              │▐▌ ▐··█·· ▐▌██+      +▌▗▚  +  ██   + ▗▐▌    +++  +·   ▗▌  + █    █ ▐▌ ▐▌│
              │█▌▖▟··█·· ▐▌██+      +▌▟▐  + ▗██   + █▐▌ +  +++  ··   ▐▌  + █    █ ▐▌ ▐▌│
              │██▌█··█▖· ▐▌██▗▌     ·▌█▐ +· ███   ++█▐▌+▌  ++·  ··  ▗█▌  +▗▜    █ ▐▌ ▐▌│
        0.0041┤██▌█··█▌· ▐▌██▐▌     ·▌█▐++▗ █▜▜   ·▐▐▐▌·▌  ···  ·· +▐█▌  +▐▐    █ ▐▌ ▐▌│
              │██▌█·-█▌· ▐▌██▐▌     ▗▌█▐++█ █▐▐   ·▐▐▐▌·▌  ···  ·· ▟▐█▌  +▐▐    █ ▐▌ ▐▌│
              │██▌█·-█▌· ▐▌██▐▌     ▐▌█▐++█ █▐▐   ·▐▐▐▌·▌  ···  ·· █▐█▌  ·▐▐    █ ▐█ ▟▌│
              │██▌█·-█▌· ▟▌██▌▌     ▐▌█▐+·█ █▐▐   ·▐▐▐▌▟▌  ···  ·· █▐█▌  ·▐▐    █ ▐█▐█▌│
        0.0028┤██▌█·-█▌▖+█▌█▛▌▌     ▐▌█▐··█ █▐▐   ·▐▐▐▜▛▌  ···  ·▖ █▐█▌ +·▐▐·▖  █ ▞█▐█▌│
              │██▌█·-▜▜▌▐▐▌█▌ ▙▖ + ▗▐▌█▐▗▌█▞█▐▐ ▗▌·▐▐▐--▌  ···+ ▐▌·█▐█▌▗▌·▐▐▐▌  █ ▌█▐█▌│
              │██▐█--  ▚▐▐▝▘▘ ▘▌▗▌▖▐▐▌█▐▐▚█▌▐▝▐▗▐▌·▐▝▟- ▚▗+···▞▌▟▙▛█▐█▌▐▐·▐▐▟▙▌+█ ▌▛▟▝▌│
              │▝▜ ▘    ▝▟·-   -▚▞▜▐█▐█▐▐▟·▀▌▝-▐▛▟▚▙█ ▛- ▝█▗▟▖▄▌▜▜▜·█▐▜▙▜▝▄▟·▀█▚▄▜+▌▘█ ▌│
        0.0016┤                 ▘  ▘▘▝  ▀   -  ▘ - ▝ ▘-   ▀▘▜▝- -  ▝▌ ▝  ▘   ▝▝  ▀  ▝  │
              └┬─────────────────┬─────────────────┬────────────────┬─────────────────┬┘
              1.0              49.2              97.5             145.8           194.0
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936/plots/tplot/dtb_summary.txt
                        dtb/mean hist                           dtb/max hist
             ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
        107.0┤████                             │105.0┤████                             │
         89.2┤████                             │ 87.5┤████                             │
             │████                             │     │████                             │
         71.3┤████                             │ 70.0┤████                             │
         53.5┤████                             │ 52.5┤████                             │
             │████                             │     │████                             │
         35.7┤████                             │ 35.0┤████                             │
         17.8┤███████      ████                │ 17.5┤███████      ████                │
             │█████████████████   ███          │     │███████████████████████          │
          0.0┤██████████████████████████   ████│  0.0┤███████████████████████   ███████│
             └┬───────┬───────┬───────┬────────┘     └┬───────┬───────┬───────┬────────┘
           0.0013  0.0031  0.0049  0.0067          0.0013  0.0033  0.0053  0.0073
                       dtb/min hist                             dtb/std hist
           ┌───────────────────────────────────┐     ┌─────────────────────────────────┐
        126┤████                               │160.0┤████                             │
           │████                               │     │████                             │
        105┤████                               │133.3┤████                             │
         84┤████                               │106.7┤████                             │
           │████                               │     │████                             │
         63┤████                               │ 80.0┤████                             │
           │████                               │     │████                             │
         42┤████                               │ 53.3┤████                             │
         21┤███████                            │ 26.7┤████                             │
           │███████   ████████   ████          │     │███████                          │
          0┤████████████████████████████   ████│  0.0┤██████████████████████████   ████│
           └┬────────┬───────┬────────┬────────┘     └┬───────┬───────┬───────┬────────┘
          0.0013  0.0031  0.0049   0.0066          -0.00015 0.00074 0.00163 0.00252
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936/plots/tplot/dtb_hist.txt
                              dtf                                   dtf/min
              ┌────────────────────────────────┐      ┌────────────────────────────────┐
        0.0155┤  ▌ ▐                     ▐     │0.0153┤    -                     -     │
        0.0139┤  ▌ ▐                     ▐     │0.0122┤ --------   -   -  -  -   --    │
              │  ▌▖▐ ▌     ▌   ▐         ▐▗    │0.0091┤--------------------------------│
        0.0124┤ ▌█▌█ ▌     ▌   ▐         ▐▐▌   │0.0060┤- -- ---  ----------------------│
        0.0108┤ ██▙█▗▙█▟ ▖ ▌ ▗ ▐ ▐▐▌ █  ▖▟▐▌   │      └┬───────┬───────┬──────┬───────┬┘
        0.0092┤▗▜████▜██▐▌▗▌ ▐▌▟▗█▐▙██▐▌▌██▙▖▌ │      1.0    49.2    97.5   145.8 194.0
              │▐ █▜▝▜▐██▟▌█▙▟▐▌█▐█▟███▟█▙███▌▌▖│dtf/min              iter
        0.0076┤▌ █  ▐▐█▘▜███▛▛▜███▛▌▜▘█▐███▜█▙█│                    dtf/std
        0.0060┤▘      ▘  ▀▛▛▘▘▐▝▛▀▘   ▝▐▀▀ ▐▝▀▘│      ┌────────────────────────────────┐
              └┬───────┬───────┬──────┬───────┬┘0.0382┤   *                            │
              1.0    49.2    97.5   145.8 194.0 0.0318┤   *                            │
        dtf                  iter               0.0191┤   *                            │
                           dtf/mean             0.0127┤   *                            │
              ┌────────────────────────────────┐0.0000┤********************************│
        0.0510┤   ·                            │      └┬───────┬───────┬──────┬───────┬┘
        0.0435┤   ·                            │      1.0    49.2    97.5   145.8 194.0
              │   ·                            │dtf/std              iter
        0.0360┤   ·                            │                   dtf/max
        0.0285┤   ·                            │     ┌─────────────────────────────────┐
              │   ·                            │0.089┤   +                             │
        0.0210┤   ·                            │0.075┤   +                             │
        0.0135┤  ··· ·     ·   ·         ·     │0.048┤   +                             │
              │ ······························ │0.034┤   +                             │
        0.0060┤·· ·····························│0.006┤+++++++++++++++++++++++++++++++++│
              └┬───────┬───────┬──────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
              1.0    49.2    97.5   145.8 194.0      1.0    49.2    97.5    145.8 194.0
        dtf/mean             iter               dtf/max             iter
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936/plots/tplot/dtf.txt
             ┌─────────────────────────────────────────────────────────────────────────┐
        0.089┤ ++ dtf/max                                                              │
             │ -- dtf/min                                                              │
             │ ·· dtf/mean                                                             │
             │ ▞▞ dtf                                                                  │
        0.075┤      +                                                                  │
             │      +                                                                  │
             │      +                                                                  │
             │      +                                                                  │
        0.061┤      +                                                                  │
             │      +                                                                  │
             │      +                                                                  │
             │      ·                                                                  │
        0.048┤      ·                                                                  │
             │      ·                                                                  │
             │      ·                                                                  │
             │      ·                                                                  │
             │      ·                                                                  │
        0.034┤      ·                                                                  │
             │      ·                                                                  │
             │      ·                                                                  │
             │      ·                                                                  │
        0.020┤      ·                                                                  │
             │   +▗ ·   ▗                                                 ·▖           │
             │  ▗·█+▗  ▗█· ·▟ + ·         ▖         ▟    + ·     +·       ▐▌ ▖▖        │
             │ ▗▀▀█▟▛▖▟▌▌▚▄▛█▗▌▐▖▙▖▖▖▟·▗▗▖▌·+▄+▟▗·+▖█+▖▄▗▌·▟▞▄▗▞▟▚▌·▗▄▗▗▙▗█▌▟▚▌▄·▖·▖ · │
        0.006┤▚▘   ▀▘▝▝    ▘ ▀▚▀▝ ▝▝▜▐▀▌▀▚▀▜▜ ▚▘▘▜▀▀▀▚▜▐▜▛▀▘▘▝▘▘▝ ▝▀▌ ▜▀▀▟▜▝▝·▀▝▞▀▟▝▙▟▀│
             └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
             1.0              49.2              97.5              145.8           194.0
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936/plots/tplot/dtf_summary.txt
                       dtf/mean hist                            dtf/max hist
           ┌───────────────────────────────────┐     ┌─────────────────────────────────┐
        168┤████                               │189.0┤████                             │
        140┤████                               │157.5┤████                             │
           │████                               │     │████                             │
        112┤████                               │126.0┤████                             │
         84┤████                               │ 94.5┤████                             │
           │████                               │     │████                             │
         56┤████                               │ 63.0┤████                             │
         28┤████                               │ 31.5┤████                             │
           │███████                            │     │████                             │
          0┤██████████                     ████│  0.0┤███████                      ████│
           └┬────────┬───────┬────────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
          0.004    0.016   0.028    0.041 0.053     0.002   0.025   0.048   0.070 0.093
                        dtf/min hist                            dtf/std hist
            ┌──────────────────────────────────┐     ┌─────────────────────────────────┐
        45.0┤███████                           │193.0┤████                             │
            │██████████                        │     │████                             │
        37.5┤██████████                        │160.8┤████                             │
        30.0┤██████████████                    │128.7┤████                             │
            │██████████████                    │     │████                             │
        22.5┤██████████████                    │ 96.5┤████                             │
            │██████████████                    │     │████                             │
        15.0┤█████████████████                 │ 64.3┤████                             │
         7.5┤█████████████████████             │ 32.2┤████                             │
            │███████████████████████████       │     │████                             │
         0.0┤██████████████████████████████████│  0.0┤███                          ████│
            └┬───────┬────────┬───────┬────────┘     └┬───────┬───────┬───────┬───────┬┘
          0.0056  0.0081   0.0106  0.0132          -0.002   0.009   0.019   0.030 0.040
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936/plots/tplot/dtf_hist.txt
                            loss                                  loss/min
            ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
        1.75┤▌                                 │1.75┤-                                 │
        1.47┤▚                                 │1.19┤--                                │
            │▐                                 │0.64┤ --------- - -           -        │
        1.19┤▝▖                                │0.08┤      ----------------------------│
        0.92┤ ▙                                │    └┬───────┬────────┬───────┬───────┬┘
        0.64┤ ▐▙▟▖                             │    1.0    49.2     97.5    145.8 194.0
            │ ▝▀██▟▖▖▖▗ ▌▖                    ▖│loss/min            iter
        0.36┤   ▝▝▐█▛█▛▙▙█▙█▙▙█▟▙ ▗▟▄▄▙▄▄▄▖▌▙▐▌│                  loss/std
        0.08┤         ▘▝▝ ▘  ▀▝▝▝▀▀▀▀▀▜▀▀▀█▜▀█▀│     ┌─────────────────────────────────┐
            └┬───────┬────────┬───────┬───────┬┘0.207┤   *                             │
            1.0    49.2     97.5    145.8 194.0 0.173┤  **                             │
        loss                iter                0.104┤ ****  * * *  ****          * *  │
                          loss/mean             0.069┤******************************** │
            ┌──────────────────────────────────┐0.000┤*** ******* *** *****************│
        1.76┤·                                 │     └┬───────┬───────┬───────┬───────┬┘
        1.48┤·                                 │     1.0    49.2    97.5    145.8 194.0
            │·                                 │loss/std            iter
        1.21┤ ·                                │                  loss/max
        0.94┤ ·                                │    ┌──────────────────────────────────┐
            │ ·                                │1.76┤+                                 │
        0.66┤  ···                             │1.49┤++                                │
        0.39┤   ··· ··· ·                    · │0.94┤ +++                              │
            │     ·····························│0.67┤  +++++++++++++++++++++  +    +++ │
        0.12┤                ·   ··············│0.12┤        ++++++++++++++++++++++++++│
            └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
            1.0    49.2     97.5    145.8 194.0     1.0    49.2     97.5    145.8 194.0
        loss/mean           iter                loss/max            iter
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936/plots/tplot/loss.txt
            ┌──────────────────────────────────────────────────────────────────────────┐
        1.76┤ ++ loss/max                                                              │
            │ -- loss/min                                                              │
            │ ·· loss/mean                                                             │
            │ ▞▞ loss                                                                  │
        1.48┤▐                                                                         │
            │ ▌                                                                        │
            │ ▌                                                                        │
            │ ▌                                                                        │
        1.20┤ ▙·                                                                       │
            │ █·                                                                       │
            │ ▐·                                                                       │
            │ ▝▖                                                                       │
        0.92┤  ▙+                                                                      │
            │  ▝▖                                                                      │
            │  -▌ + ▗+                                                                 │
            │  -▌▌·+█+                                                                 │
            │   █▌·▖█·+                                                                │
        0.64┤   █▚▟▌▌▌▌▖                                                               │
            │   ▝--▝▌▐█▌ ▖  · +      ▟                                                 │
            │    - -▌▝▌▙▟▌ ▗▌··   ▗  █ ▗            +                                  │
            │    -  ▌ -█▝▌▟▐▌▗▟+· █ +█ ▐+ ▖ ▖▗+ ▗ ▖ +              +                ▗▌ │
        0.36┤          ▝ ▌█▞▌▛▛▄▟▞▛▄·█▗█▚▞▌▐▌█·▐█▐▌▗▖▗ ▖  + · ▟   +▗▌      +  +▖ ▗  ▐▌ │
            │            ▝▝▌▜-▘▐█▌-▐▞▀▟▜▐▌▐▐█-▚▟█▟▚▌▙█▐▌++·+▞▌█▗▗▖▟▐▌▐··+▖▗▟ +▐▌·▛▖ ▐▌ │
            │                   ▀▌ ▝▌ █  ▌ ▘▝ -▜█▌▐▌▘▜▀▙▚▖▄▖▌▙▛▟▌▚▜▟▜▐▄▚▟▙██▗▌▟▌▐▌▌·█▌▖│
            │                            ▘      ▘▘ ▘   ▝▝▛▐▚▌▝ ▝--▝▝▝▟ ▝▌▀▝▝▌█▜▜▞▌▚▟█▜▚│
        0.08┤                                                        ▜  -   ▘▝ ▝▌ ▝▜▜  │
            └┬─────────────────┬──────────────────┬─────────────────┬─────────────────┬┘
            1.0              49.2               97.5              145.8           194.0
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936/plots/tplot/loss_summary.txt
                       loss/mean hist                           loss/max hist
            ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
        98.0┤████                              │76.0┤███████                           │
        81.7┤████                              │63.3┤███████                           │
            │████                              │    │███████                           │
        65.3┤███████                           │50.7┤███████                           │
        49.0┤███████                           │38.0┤███████                           │
            │███████                           │    │███████                           │
        32.7┤███████                           │25.3┤███████                           │
        16.3┤███████                           │12.7┤██████████                        │
            │██████████████                    │    │█████████████████                 │
         0.0┤█████████████████   ███████   ████│ 0.0┤███████████████████████████   ████│
            └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
           0.04    0.49     0.94    1.38   1.83    0.05    0.49     0.94    1.39   1.83
                        loss/min hist                           loss/std hist
             ┌─────────────────────────────────┐    ┌──────────────────────────────────┐
        101.0┤████                             │76.0┤████                              │
             │████                             │    │████                              │
         84.2┤████                             │63.3┤████                              │
         67.3┤████                             │50.7┤████                              │
             │███████                          │    │███████                           │
         50.5┤███████                          │38.0┤███████                           │
             │███████                          │    │██████████                        │
         33.7┤███████                          │25.3┤██████████████                    │
         16.8┤██████████                       │12.7┤██████████████                    │
             │█████████████                    │    │█████████████████████             │
          0.0┤████████████████    ███   ███████│ 0.0┤████████████████████████   ███████│
             └┬───────┬───────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
            0.01    0.46    0.92    1.37   1.82   -0.009   0.047    0.104   0.160 0.216
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936/plots/tplot/loss_hist.txt
        [2025-12-26 14:59:47,081689][I][ezpz/history:2433:finalize] Saving history report to /Users/samforeman/vibes/saforem2/ezpz/outputs/ezpz.test_dist/2025-12-26-145936/report.md
        [2025-12-26 14:59:47,085092][I][ezpz/test_dist:348:finalize] dataset=<xarray.Dataset> Size: 39kB
        Dimensions:        (draw: 194)
        Coordinates:
          * draw           (draw) int64 2kB 0 1 2 3 4 5 6 ... 188 189 190 191 192 193
        Data variables: (12/25)
            iter           (draw) int64 2kB 6 7 8 9 10 11 12 ... 194 195 196 197 198 199
            loss           (draw) float32 776B 1.751 1.595 1.422 ... 0.2113 0.1499
            accuracy       (draw) float32 776B 0.7031 0.6484 0.6875 ... 0.9141 0.9531
            dtf            (draw) float64 2kB 0.007276 0.006534 ... 0.007024 0.007607
            dtb            (draw) float64 2kB 0.004568 0.002039 ... 0.001932 0.008309
            iter_mean      (draw) float64 2kB 6.0 7.0 8.0 9.0 ... 197.0 198.0 199.0
            ...             ...
            dtf_min        (draw) float64 2kB 0.006776 0.006534 ... 0.006853 0.00668
            dtf_std        (draw) float64 2kB 0.00025 0.0001769 ... 8.545e-05 0.0004633
            dtb_mean       (draw) float64 2kB 0.003325 0.002154 ... 0.002945 0.00807
            dtb_max        (draw) float64 2kB 0.004568 0.002269 ... 0.003958 0.008309
            dtb_min        (draw) float64 2kB 0.002083 0.00204 ... 0.001932 0.007832
            dtb_std        (draw) float64 2kB 0.001242 0.0001147 ... 0.001013 0.0002383
        [2025-12-26 14:59:47,608766][I][ezpz/test_dist:500:train] Took: 8.13 seconds to finish training
        [2025-12-26 14:59:47,609602][I][ezpz/test_dist:695:main] Took: 11.78 seconds
        wandb:
        wandb: 🚀 View run soft-grass-6851 at:
        wandb: Find logs at: wandb/run-20251226_145937-rj4d7rus/logs
        [2025-12-23-162222] Execution time: 19s sec

        ```

    </details>

    - <details closed><summary>{Aurora, Sunspot} @ ALCF</summary>

        ```bash
        module load frameworks
        TMPDIR=$(pwd) uv run \
            --python=$(which python3) \
            --with "git+https://github.com/saforem2/ezpz" \
            ezpz test
        ```

        <details closed><summary>Output:</summary>

        ```bash
        #[12/26/25,12:56:24][x4310c1s0b0n0][/f/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007]
        $ module load frameworks \
            && TMPDIR=$(pwd) uv run \
                --python=$(which python3) \
                --with "git+https://github.com/saforem2/ezpz" \
                ezpz test
        [2025-12-26 12:56:59,991844][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2025-12-26-125659]----
        [2025-12-26 12:57:00,950846][I][ezpz/launch:416:launch] Job ID: 8234998
        [2025-12-26 12:57:00,951634][I][ezpz/launch:417:launch] nodelist: ['x4310c1s0b0n0', 'x4310c1s1b0n0']
        [2025-12-26 12:57:00,952019][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/8234998.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
        [2025-12-26 12:57:01,231960][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
        [2025-12-26 12:57:01,233271][I][ezpz/launch:367:build_executable] Building command to execute by piecing together:
        [2025-12-26 12:57:01,233694][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8234998.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
        [2025-12-26 12:57:01,234378][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: /home/foremans/datascience/foremans/.cache/builds-v0/.tmpCcpdMz/bin/python -m ezpz.test_dist
        [2025-12-26 12:57:01,235161][I][ezpz/launch:433:launch] Took: 1.24 seconds to build command.
        [2025-12-26 12:57:01,235513][I][ezpz/launch:436:launch] Executing:
        mpiexec
          --envall
          --np=24
          --ppn=12
          --hostfile=/var/spool/pbs/aux/8234998.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
          --no-vni
          --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
          /home/foremans/datascience/foremans/.cache/builds-v0/.tmpCcpdMz/bin/python
          -m
          ezpz.test_dist
        [2025-12-26 12:57:01,236843][I][ezpz/launch:220:get_aurora_filters] Filtering for Aurora-specific messages. To view list of filters, run with EZPZ_LOG_LEVEL=DEBUG
        [2025-12-26 12:57:01,237331][I][ezpz/launch:443:launch] Execution started @ 2025-12-26-125701...
        [2025-12-26 12:57:01,237728][I][ezpz/launch:138:run_command] Caught 24 filters
        [2025-12-26 12:57:01,238051][I][ezpz/launch:139:run_command] Running command:
         mpiexec --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/8234998.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 /home/foremans/datascience/foremans/.cache/builds-v0/.tmpCcpdMz/bin/python -m ezpz.test_dist
        cpubind:list x4310c1s1b0n0 pid 147083 rank 12 0: mask 0x1c
        cpubind:list x4310c1s1b0n0 pid 147084 rank 13 1: mask 0x1c00
        cpubind:list x4310c1s1b0n0 pid 147085 rank 14 2: mask 0x1c0000
        cpubind:list x4310c1s1b0n0 pid 147086 rank 15 3: mask 0x1c000000
        cpubind:list x4310c1s1b0n0 pid 147087 rank 16 4: mask 0x1c00000000
        cpubind:list x4310c1s1b0n0 pid 147088 rank 17 5: mask 0x1c0000000000
        cpubind:list x4310c1s1b0n0 pid 147089 rank 18 6: mask 0x1c0000000000000
        cpubind:list x4310c1s1b0n0 pid 147090 rank 19 7: mask 0x1c000000000000000
        cpubind:list x4310c1s1b0n0 pid 147091 rank 20 8: mask 0x1c00000000000000000
        cpubind:list x4310c1s1b0n0 pid 147092 rank 21 9: mask 0x1c0000000000000000000
        cpubind:list x4310c1s1b0n0 pid 147093 rank 22 10: mask 0x1c000000000000000000000
        cpubind:list x4310c1s1b0n0 pid 147094 rank 23 11: mask 0x1c00000000000000000000000
        cpubind:list x4310c1s0b0n0 pid 114692 rank 0 0: mask 0x1c
        cpubind:list x4310c1s0b0n0 pid 114693 rank 1 1: mask 0x1c00
        cpubind:list x4310c1s0b0n0 pid 114694 rank 2 2: mask 0x1c0000
        cpubind:list x4310c1s0b0n0 pid 114695 rank 3 3: mask 0x1c000000
        cpubind:list x4310c1s0b0n0 pid 114696 rank 4 4: mask 0x1c00000000
        cpubind:list x4310c1s0b0n0 pid 114697 rank 5 5: mask 0x1c0000000000
        cpubind:list x4310c1s0b0n0 pid 114698 rank 6 6: mask 0x1c0000000000000
        cpubind:list x4310c1s0b0n0 pid 114699 rank 7 7: mask 0x1c000000000000000
        cpubind:list x4310c1s0b0n0 pid 114700 rank 8 8: mask 0x1c00000000000000000
        cpubind:list x4310c1s0b0n0 pid 114701 rank 9 9: mask 0x1c0000000000000000000
        cpubind:list x4310c1s0b0n0 pid 114702 rank 10 10: mask 0x1c000000000000000000000
        cpubind:list x4310c1s0b0n0 pid 114703 rank 11 11: mask 0x1c00000000000000000000000
        [2025-12-26 12:57:09,319444][I][ezpz/test_dist:132:__post_init__] Outputs will be saved to /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709
        [2025-12-26 12:57:09,322179][I][ezpz/dist:1506:setup_torch_distributed] Using fw='ddp' with torch_{device,backend}= {xpu, xccl}
        [2025-12-26 12:57:09,323025][I][ezpz/dist:1371:setup_torch_DDP] Caught MASTER_PORT=57733 from environment!
        [2025-12-26 12:57:09,323626][I][ezpz/dist:1387:setup_torch_DDP] Using torch.distributed.init_process_group with
        - master_addr='x4310c1s0b0n0.hsn.cm.aurora.alcf.anl.gov'
        - master_port='57733'
        - world_size=24
        - rank=0
        - local_rank=0
        - timeout=datetime.timedelta(seconds=3600)
        - backend='xccl'
        [2025-12-26 12:57:09,324720][I][ezpz/dist:1019:init_process_group] Calling torch.distributed.init_process_group_with: rank=0 world_size=24 backend=xccl
        [2025-12-26 12:57:11,367607][I][ezpz/dist:1732:setup_torch] Using device='xpu' with backend='xccl' + 'xccl' for distributed training.
        [2025-12-26 12:57:11,369584][W][ezpz/dist:544:print_dist_setup] Using [24 / 24] available "xpu" devices !!
        [2025-12-26 12:57:11,370083][I][ezpz/dist:1779:setup_torch] ['x4310c1s0b0n0'][device='xpu'][node=0/1][rank=00/23][local_rank=00/11]
        [2025-12-26 12:57:11,369426][I][ezpz/dist:1779:setup_torch] ['x4310c1s0b0n0'][device='xpu'][node=0/1][rank=02/23][local_rank=02/11]
        [2025-12-26 12:57:11,369554][I][ezpz/dist:1779:setup_torch] ['x4310c1s0b0n0'][device='xpu'][node=1/1][rank=05/23][local_rank=05/11]
        [2025-12-26 12:57:11,369558][I][ezpz/dist:1779:setup_torch] ['x4310c1s0b0n0'][device='xpu'][node=0/1][rank=06/23][local_rank=06/11]
        [2025-12-26 12:57:11,369660][I][ezpz/dist:1779:setup_torch] ['x4310c1s0b0n0'][device='xpu'][node=1/1][rank=07/23][local_rank=07/11]
        [2025-12-26 12:57:11,369585][I][ezpz/dist:1779:setup_torch] ['x4310c1s0b0n0'][device='xpu'][node=0/1][rank=08/23][local_rank=08/11]
        [2025-12-26 12:57:11,369597][I][ezpz/dist:1779:setup_torch] ['x4310c1s0b0n0'][device='xpu'][node=1/1][rank=09/23][local_rank=09/11]
        [2025-12-26 12:57:11,369637][I][ezpz/dist:1779:setup_torch] ['x4310c1s0b0n0'][device='xpu'][node=0/1][rank=10/23][local_rank=10/11]
        [2025-12-26 12:57:11,369590][I][ezpz/dist:1779:setup_torch] ['x4310c1s0b0n0'][device='xpu'][node=1/1][rank=11/23][local_rank=11/11]
        [2025-12-26 12:57:11,369710][I][ezpz/dist:1779:setup_torch] ['x4310c1s0b0n0'][device='xpu'][node=1/1][rank=01/23][local_rank=01/11]
        [2025-12-26 12:57:11,369715][I][ezpz/dist:1779:setup_torch] ['x4310c1s0b0n0'][device='xpu'][node=1/1][rank=03/23][local_rank=03/11]
        [2025-12-26 12:57:11,369129][I][ezpz/dist:1779:setup_torch] ['x4310c1s1b0n0'][device='xpu'][node=0/1][rank=14/23][local_rank=02/11]
        [2025-12-26 12:57:11,369276][I][ezpz/dist:1779:setup_torch] ['x4310c1s1b0n0'][device='xpu'][node=0/1][rank=12/23][local_rank=00/11]
        [2025-12-26 12:57:11,369686][I][ezpz/dist:1779:setup_torch] ['x4310c1s0b0n0'][device='xpu'][node=0/1][rank=04/23][local_rank=04/11]
        [2025-12-26 12:57:11,369570][I][ezpz/dist:1779:setup_torch] ['x4310c1s1b0n0'][device='xpu'][node=1/1][rank=13/23][local_rank=01/11]
        [2025-12-26 12:57:11,369439][I][ezpz/dist:1779:setup_torch] ['x4310c1s1b0n0'][device='xpu'][node=1/1][rank=15/23][local_rank=03/11]
        [2025-12-26 12:57:11,372392][I][ezpz/test_dist:678:main] Took: 2.07 seconds to setup torch
        [2025-12-26 12:57:11,369272][I][ezpz/dist:1779:setup_torch] ['x4310c1s1b0n0'][device='xpu'][node=0/1][rank=16/23][local_rank=04/11]
        [2025-12-26 12:57:11,369296][I][ezpz/dist:1779:setup_torch] ['x4310c1s1b0n0'][device='xpu'][node=1/1][rank=17/23][local_rank=05/11]
        [2025-12-26 12:57:11,369515][I][ezpz/dist:1779:setup_torch] ['x4310c1s1b0n0'][device='xpu'][node=0/1][rank=18/23][local_rank=06/11]
        [2025-12-26 12:57:11,369551][I][ezpz/dist:1779:setup_torch] ['x4310c1s1b0n0'][device='xpu'][node=1/1][rank=19/23][local_rank=07/11]
        [2025-12-26 12:57:11,369556][I][ezpz/dist:1779:setup_torch] ['x4310c1s1b0n0'][device='xpu'][node=0/1][rank=20/23][local_rank=08/11]
        [2025-12-26 12:57:11,369524][I][ezpz/dist:1779:setup_torch] ['x4310c1s1b0n0'][device='xpu'][node=1/1][rank=21/23][local_rank=09/11]
        [2025-12-26 12:57:11,369569][I][ezpz/dist:1779:setup_torch] ['x4310c1s1b0n0'][device='xpu'][node=0/1][rank=22/23][local_rank=10/11]
        [2025-12-26 12:57:11,369353][I][ezpz/dist:1779:setup_torch] ['x4310c1s1b0n0'][device='xpu'][node=1/1][rank=23/23][local_rank=11/11]
        [2025-12-26 12:57:11,386631][I][ezpz/test_dist:461:train] Model size: 567434 parameters
        [2025-12-26 12:57:11,388753][I][ezpz/test_dist:465:train]
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
        [2025-12-26 12:57:11,390055][I][ezpz/test_dist:473:train] Took: 0.007092675659805536 seconds to build model
        [2025-12-26 12:57:11,392504][I][ezpz/test_dist:601:build_model_and_optimizer] model=
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
        [2025-12-26 12:57:11,394462][I][ezpz/dist:685:wrap_model] Wrapping model with: ddp
        2025:12:26-12:57:11:(114692) |CCL_WARN| value of CCL_OP_SYNC changed to be 1 (default:0)
        2025:12:26-12:57:11:(114692) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
        [2025-12-26 12:57:24,214420][I][ezpz/test_dist:479:train] Took: 12.82 seconds to build optimizer
        [2025-12-26 12:57:24,257102][I][ezpz/history:220:__init__] Using History with distributed_history=True
        [2025-12-26 12:57:24,262059][I][ezpz/dist:2044:setup_wandb] Setting up wandb from rank=0
        [2025-12-26 12:57:24,262600][I][ezpz/dist:2045:setup_wandb] Using WB_PROJECT=ezpz.test_dist
        wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
        wandb: Tracking run with wandb version 0.21.3
        wandb: Run data is saved locally in /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/wandb/run-20251226_125724-adhgoy9j
        wandb: Run `wandb offline` to turn off syncing.
        wandb: Syncing run winter-salad-6843
        wandb:  View project at https://wandb.ai/aurora_gpt/ezpz.test_dist
        wandb:  View run at https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/adhgoy9j
        [2025-12-26 12:57:30,839972][I][ezpz/dist:2074:setup_wandb] wandb.run=[winter-salad-6843](https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/adhgoy9j)
        [2025-12-26 12:57:30,846065][I][ezpz/dist:2117:setup_wandb] Running on machine='Aurora'
        [2025-12-26 12:57:32,361320][I][ezpz/test_dist:482:train] Took: 8.15 seconds to build trainer
        [2025-12-26 12:57:32,362820][I][ezpz/test_dist:486:train] config:
        {
          "acc_events": false,
          "backend": "DDP",
          "batch_size": 128,
          "cp": 1,
          "dataset": "mnist",
          "dataset_root": "/lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/datasets/mnist",
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
        [2025-12-26 12:57:32,364988][I][ezpz/test_dist:488:train] Took: 28.84 to get here.
        [2025-12-26 12:57:46,725491][I][ezpz/test_dist:369:train] Warmup complete at step 5
        [2025-12-26 12:57:46,963482][I][ezpz/test_dist:325:train_step] iter=10 loss=0.994967 accuracy=0.750000 dtf=0.011305 dtb=0.001640 loss/mean=1.035069 loss/max=1.218811 loss/min=0.923871 loss/std=0.067301 accuracy/mean=0.714844 accuracy/max=0.804688 accuracy/min=0.609375 accuracy/std=0.046054 dtf/mean=0.010381 dtf/max=0.011685 dtf/min=0.009660 dtf/std=0.000662 dtb/mean=0.001692 dtb/max=0.002077 dtb/min=0.001408 dtb/std=0.000237
        [2025-12-26 12:57:47,784965][I][ezpz/test_dist:325:train_step] iter=20 loss=0.843957 accuracy=0.779412 dtf=0.007382 dtb=0.232720 loss/mean=0.587017 loss/max=0.843957 loss/min=0.312610 loss/std=0.137216 accuracy/mean=0.806373 accuracy/max=0.911765 accuracy/min=0.705882 accuracy/std=0.054310 dtf/mean=0.006949 dtf/max=0.007548 dtf/min=0.006570 dtf/std=0.000303 dtb/mean=0.211198 dtb/max=0.238684 dtb/min=0.176031 dtb/std=0.020564
        [2025-12-26 12:57:48,288727][I][ezpz/test_dist:325:train_step] iter=30 loss=0.465919 accuracy=0.867188 dtf=0.009977 dtb=0.001979 loss/mean=0.438402 loss/max=0.722735 loss/min=0.278721 loss/std=0.110631 accuracy/mean=0.866536 accuracy/max=0.921875 accuracy/min=0.750000 accuracy/std=0.035937 dtf/mean=0.010105 dtf/max=0.010829 dtf/min=0.009644 dtf/std=0.000391 dtb/mean=0.001774 dtb/max=0.002093 dtb/min=0.001422 dtb/std=0.000223
        [2025-12-26 12:57:49,034654][I][ezpz/test_dist:325:train_step] iter=40 loss=0.458118 accuracy=0.882353 dtf=0.007307 dtb=0.002033 loss/mean=0.297673 loss/max=0.516792 loss/min=0.184366 loss/std=0.080388 accuracy/mean=0.912990 accuracy/max=0.955882 accuracy/min=0.838235 accuracy/std=0.031458 dtf/mean=0.006865 dtf/max=0.007475 dtf/min=0.006140 dtf/std=0.000433 dtb/mean=0.001488 dtb/max=0.002033 dtb/min=0.001172 dtb/std=0.000251
        [2025-12-26 12:57:49,656664][I][ezpz/test_dist:325:train_step] iter=50 loss=0.364185 accuracy=0.882812 dtf=0.010035 dtb=0.002136 loss/mean=0.296386 loss/max=0.433208 loss/min=0.205008 loss/std=0.066657 accuracy/mean=0.912109 accuracy/max=0.953125 accuracy/min=0.851562 accuracy/std=0.027274 dtf/mean=0.009980 dtf/max=0.010566 dtf/min=0.009565 dtf/std=0.000270 dtb/mean=0.001785 dtb/max=0.002197 dtb/min=0.001444 dtb/std=0.000243
        [2025-12-26 12:57:50,516216][I][ezpz/test_dist:325:train_step] iter=60 loss=0.303229 accuracy=0.926471 dtf=0.006841 dtb=0.001837 loss/mean=0.181245 loss/max=0.303229 loss/min=0.074041 loss/std=0.051771 accuracy/mean=0.952206 accuracy/max=1.000000 accuracy/min=0.911765 accuracy/std=0.024108 dtf/mean=0.006655 dtf/max=0.006969 dtf/min=0.006220 dtf/std=0.000242 dtb/mean=0.001543 dtb/max=0.001904 dtb/min=0.001178 dtb/std=0.000215
        [2025-12-26 12:57:51,748835][I][ezpz/test_dist:325:train_step] iter=70 loss=0.287316 accuracy=0.906250 dtf=0.010923 dtb=0.002028 loss/mean=0.213261 loss/max=0.345638 loss/min=0.130070 loss/std=0.065777 accuracy/mean=0.937174 accuracy/max=0.968750 accuracy/min=0.867188 accuracy/std=0.025958 dtf/mean=0.010181 dtf/max=0.011084 dtf/min=0.009712 dtf/std=0.000379 dtb/mean=0.001803 dtb/max=0.002258 dtb/min=0.001430 dtb/std=0.000229
        [2025-12-26 12:57:54,740809][I][ezpz/test_dist:325:train_step] iter=80 loss=0.206866 accuracy=0.926471 dtf=0.006063 dtb=0.001766 loss/mean=0.113710 loss/max=0.206866 loss/min=0.068099 loss/std=0.038122 accuracy/mean=0.974265 accuracy/max=1.000000 accuracy/min=0.926471 accuracy/std=0.019102 dtf/mean=0.005980 dtf/max=0.006408 dtf/min=0.005786 dtf/std=0.000135 dtb/mean=0.001514 dtb/max=0.001766 dtb/min=0.001132 dtb/std=0.000189
        [2025-12-26 12:57:55,375104][I][ezpz/test_dist:325:train_step] iter=90 loss=0.220868 accuracy=0.914062 dtf=0.010806 dtb=0.001936 loss/mean=0.166121 loss/max=0.261424 loss/min=0.083375 loss/std=0.047467 accuracy/mean=0.951172 accuracy/max=0.984375 accuracy/min=0.914062 accuracy/std=0.017065 dtf/mean=0.010863 dtf/max=0.011598 dtf/min=0.010269 dtf/std=0.000426 dtb/mean=0.001793 dtb/max=0.002010 dtb/min=0.001455 dtb/std=0.000182
        [2025-12-26 12:57:55,916235][I][ezpz/test_dist:325:train_step] iter=100 loss=0.101629 accuracy=0.970588 dtf=0.007392 dtb=0.001704 loss/mean=0.077895 loss/max=0.216991 loss/min=0.044901 loss/std=0.037287 accuracy/mean=0.988358 accuracy/max=1.000000 accuracy/min=0.955882 accuracy/std=0.013408 dtf/mean=0.006932 dtf/max=0.007560 dtf/min=0.006267 dtf/std=0.000455 dtb/mean=0.001566 dtb/max=0.002013 dtb/min=0.001161 dtb/std=0.000249
        [2025-12-26 12:57:56,422680][I][ezpz/test_dist:325:train_step] iter=110 loss=0.174663 accuracy=0.953125 dtf=0.011343 dtb=0.001621 loss/mean=0.119567 loss/max=0.200464 loss/min=0.068889 loss/std=0.039575 accuracy/mean=0.970052 accuracy/max=0.992188 accuracy/min=0.937500 accuracy/std=0.014901 dtf/mean=0.010806 dtf/max=0.012639 dtf/min=0.010221 dtf/std=0.000509 dtb/mean=0.001810 dtb/max=0.002037 dtb/min=0.001430 dtb/std=0.000182
        [2025-12-26 12:57:56,786762][I][ezpz/test_dist:325:train_step] iter=120 loss=0.074708 accuracy=0.985294 dtf=0.006787 dtb=0.001536 loss/mean=0.049546 loss/max=0.090880 loss/min=0.026799 loss/std=0.018310 accuracy/mean=0.991422 accuracy/max=1.000000 accuracy/min=0.985294 accuracy/std=0.007246 dtf/mean=0.006472 dtf/max=0.006828 dtf/min=0.005932 dtf/std=0.000261 dtb/mean=0.001562 dtb/max=0.001867 dtb/min=0.001090 dtb/std=0.000205
        [2025-12-26 12:57:57,246460][I][ezpz/test_dist:325:train_step] iter=130 loss=0.137289 accuracy=0.953125 dtf=0.010142 dtb=0.001862 loss/mean=0.095899 loss/max=0.145525 loss/min=0.054574 loss/std=0.030761 accuracy/mean=0.974935 accuracy/max=1.000000 accuracy/min=0.945312 accuracy/std=0.016102 dtf/mean=0.010148 dtf/max=0.012131 dtf/min=0.009641 dtf/std=0.000639 dtb/mean=0.001848 dtb/max=0.002093 dtb/min=0.001321 dtb/std=0.000210
        [2025-12-26 12:57:57,832532][I][ezpz/test_dist:325:train_step] iter=140 loss=0.038551 accuracy=0.985294 dtf=0.006596 dtb=0.001460 loss/mean=0.037799 loss/max=0.061152 loss/min=0.015614 loss/std=0.011380 accuracy/mean=0.995098 accuracy/max=1.000000 accuracy/min=0.985294 accuracy/std=0.006944 dtf/mean=0.006719 dtf/max=0.007528 dtf/min=0.006087 dtf/std=0.000449 dtb/mean=0.001491 dtb/max=0.001719 dtb/min=0.001157 dtb/std=0.000206
        [2025-12-26 12:57:58,329794][I][ezpz/test_dist:325:train_step] iter=150 loss=0.084032 accuracy=0.968750 dtf=0.010424 dtb=0.001986 loss/mean=0.076138 loss/max=0.141387 loss/min=0.033583 loss/std=0.027965 accuracy/mean=0.979818 accuracy/max=1.000000 accuracy/min=0.945312 accuracy/std=0.013514 dtf/mean=0.010651 dtf/max=0.011385 dtf/min=0.009915 dtf/std=0.000520 dtb/mean=0.001795 dtb/max=0.002165 dtb/min=0.001298 dtb/std=0.000235
        [2025-12-26 12:57:58,871216][I][ezpz/test_dist:325:train_step] iter=160 loss=0.030340 accuracy=1.000000 dtf=0.006370 dtb=0.001434 loss/mean=0.036724 loss/max=0.116999 loss/min=0.011584 loss/std=0.026702 accuracy/mean=0.992647 accuracy/max=1.000000 accuracy/min=0.941176 accuracy/std=0.014082 dtf/mean=0.006482 dtf/max=0.006820 dtf/min=0.005905 dtf/std=0.000327 dtb/mean=0.001546 dtb/max=0.001796 dtb/min=0.001153 dtb/std=0.000192
        [2025-12-26 12:57:59,277568][I][ezpz/test_dist:325:train_step] iter=170 loss=0.060540 accuracy=0.984375 dtf=0.010029 dtb=0.001871 loss/mean=0.067327 loss/max=0.170805 loss/min=0.035560 loss/std=0.030100 accuracy/mean=0.982096 accuracy/max=1.000000 accuracy/min=0.937500 accuracy/std=0.013047 dtf/mean=0.010218 dtf/max=0.012835 dtf/min=0.009561 dtf/std=0.000796 dtb/mean=0.001831 dtb/max=0.002365 dtb/min=0.001390 dtb/std=0.000244
        [2025-12-26 12:57:59,752142][I][ezpz/test_dist:325:train_step] iter=180 loss=0.039758 accuracy=0.985294 dtf=0.006253 dtb=0.001701 loss/mean=0.034456 loss/max=0.081928 loss/min=0.009000 loss/std=0.020232 accuracy/mean=0.990809 accuracy/max=1.000000 accuracy/min=0.955882 accuracy/std=0.012603 dtf/mean=0.006565 dtf/max=0.007686 dtf/min=0.005779 dtf/std=0.000649 dtb/mean=0.001519 dtb/max=0.002028 dtb/min=0.001091 dtb/std=0.000251
        [2025-12-26 12:58:00,304971][I][ezpz/test_dist:325:train_step] iter=190 loss=0.086260 accuracy=0.953125 dtf=0.011277 dtb=0.001865 loss/mean=0.054108 loss/max=0.114451 loss/min=0.015817 loss/std=0.026246 accuracy/mean=0.985026 accuracy/max=1.000000 accuracy/min=0.953125 accuracy/std=0.013514 dtf/mean=0.010987 dtf/max=0.011464 dtf/min=0.010086 dtf/std=0.000501 dtb/mean=0.001754 dtb/max=0.002030 dtb/min=0.001315 dtb/std=0.000212
        [2025-12-26 12:58:02,269674][I][ezpz/history:2385:finalize] Saving plots to /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709/plots/mplot (matplotlib) and /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709/plots/tplot (tplot)
                          accuracy                              accuracy/min
             ┌─────────────────────────────────┐    ┌──────────────────────────────────┐
        1.000┤              ▟ ▄▖▄▄▄▟▄▄█▙▙██▙█▞▛│0.99┤         -------------------------│
        0.934┤       ▗ ▌▄██▟█▛▛█▝▜▘▘▀ ▝▝▝▘ ▝▝▘ │0.80┤  -------------                   │
             │    ▐▟█▟▜▝▀▀▀▝▀                  │0.62┤ ---                              │
        0.867┤   ▙▛▌▝▘                         │0.44┤-                                 │
        0.801┤ ▟▟▐▘                            │    └┬───────┬────────┬───────┬───────┬┘
        0.734┤▗█▌▝                             │    1.0    49.2     97.5    145.8 194.0
             │▐▛▌                              │accuracy/min        iter
        0.668┤▐                                │                accuracy/std
        0.602┤▛                                │     ┌─────────────────────────────────┐
             └┬───────┬───────┬───────┬───────┬┘0.068┤*                                │
             1.0    49.2    97.5    145.8 194.0 0.058┤****                             │
        accuracy            iter                0.038┤ ****** * *                      │
                        accuracy/mean           0.027┤   ************************** ** │
             ┌─────────────────────────────────┐0.007┤                *****************│
        0.995┤            ·   · ···············│     └┬───────┬───────┬───────┬───────┬┘
        0.922┤        ············             │     1.0    49.2    97.5    145.8 194.0
             │    ·······                      │accuracy/std        iter
        0.849┤   ··                            │                accuracy/max
        0.776┤  ··                             │     ┌─────────────────────────────────┐
             │ ··                              │1.000┤       ++++++++++++++++++++++++++│
        0.703┤··                               │0.951┤  ++++++++                       │
        0.630┤·                                │0.852┤ ++                              │
             │·                                │0.802┤++                               │
        0.557┤·                                │0.703┤+                                │
             └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
             1.0    49.2    97.5    145.8 194.0      1.0    49.2    97.5    145.8 194.0
        accuracy/mean       iter                accuracy/max        iter
        text saved in /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709/plots/tplot/accuracy.txt
            ┌──────────────────────────────────────────────────────────────────────────┐
        1.00┤ ++ accuracy/max    +       + ++ ▖+++++++++▗+++▖▟+▗▖▖▄▚▗▄+▟+▞▙▚▟▗▗▗+▙▚·▙▗▄│
            │ -- accuracy/min  ++++++++++·++▙+▌+ ▖▄▗▌▞▖▗█▗▟▀▙▀▟▌▌▛▝▝▌▝▟▝▄▌█·▘▘▘▜▞█▐▐·▘·│
            │ ·· accuracy/mean ++▖ +  █ ▗▖▗▌█▐▐▐▞█▝█▜▌▝▜▘▌▀ ▜--▘▝---▘-▝ ▝▘▝-----▘▝-▘---│
            │ ▞▞ accuracy       ▗▚·▗▛▌▛▄▘▌▞▐█▐-▀-▝-▝-▘---▘-- -  --- - ---- - -   -  -  │
        0.91┤     ++ + ▗ ▌▗▖▌·▌▜▐·▀▀▘▝▘▝-▝--▘▀-----    ---        -                    │
            │    ++++  █▗▙█▌▙▀▌▐▀-------- ---                                          │
            │   +++▗▌ █▐▛▌ ▜▝- --------    -                                           │
            │  ++ +▐▌▐█ ▘---  --    -                                                  │
        0.81┤  +▖ ▗▘▐▌▝ ----   -    -                                                  │
            │ +▟▌▗█ ▐▌----                                                             │
            │ +█▌▐▝ ▝▌--                                                               │
            │+▗▛▌▌ -  -                                                                │
        0.72┤+█▌▚▌--                                                                   │
            │+█▌ ▌--                                                                   │
            │ ▌▘-▘-                                                                    │
            │ ▌ ---                                                                    │
            │ ▌---                                                                     │
        0.62┤█▌--                                                                      │
            │▝- -                                                                      │
            │·-                                                                        │
            │·-                                                                        │
        0.53┤ -                                                                        │
            │ -                                                                        │
            │-                                                                         │
            │-                                                                         │
        0.44┤-                                                                         │
            └┬─────────────────┬──────────────────┬─────────────────┬─────────────────┬┘
            1.0              49.2               97.5              145.8           194.0
        text saved in /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709/plots/tplot/accuracy_summary.txt
                    accuracy/mean hist                        accuracy/max hist
           ┌───────────────────────────────────┐     ┌─────────────────────────────────┐
        114┤                               ████│134.0┤                             ████│
         95┤                               ████│111.7┤                             ████│
           │                               ████│     │                             ████│
         76┤                               ████│ 89.3┤                             ████│
         57┤                               ████│ 67.0┤                             ████│
           │                               ████│     │                             ████│
         38┤                            ███████│ 44.7┤                             ████│
         19┤                        ███████████│ 22.3┤                          ███████│
           │                     ██████████████│     │                    █████████████│
          0┤███████████████████████████████████│  0.0┤███████   ███████████████████████│
           └┬────────┬───────┬────────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
          0.54     0.66    0.78     0.90   1.01     0.690   0.771   0.852   0.932 1.013
                      accuracy/min hist                       accuracy/std hist
            ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
        75.0┤                              ████│89.0┤   ████                           │
            │                              ████│    │   ████                           │
        62.5┤                           ███████│74.2┤   ████                           │
        50.0┤                           ███████│59.3┤   ████                           │
            │                           ███████│    │   ████                           │
        37.5┤                           ███████│44.5┤   ████                           │
            │                        ██████████│    │███████                           │
        25.0┤                        ██████████│29.7┤██████████                        │
        12.5┤                        ██████████│14.8┤██████████████                    │
            │          ████████████████████████│    │█████████████████   ███████       │
         0.0┤██████████████████████████████████│ 0.0┤██████████████████████████████████│
            └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
           0.41    0.56     0.71    0.86   1.01    0.004   0.021    0.038   0.054 0.071
        text saved in /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709/plots/tplot/accuracy_hist.txt
                             dtb                                   dtb/min
             ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
        0.233┤  ▟                              │0.176┤  -                              │
        0.194┤  █                              │0.118┤  -                              │
             │  █                              │0.059┤  -                              │
        0.156┤  █                              │0.001┤---------------------------------│
        0.117┤  █                              │     └┬───────┬───────┬───────┬───────┬┘
        0.079┤  █                              │     1.0    49.2    97.5    145.8 194.0
             │  █                              │dtb/min             iter
        0.040┤  █                              │                    dtb/std
        0.001┤▄▄█▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│      ┌────────────────────────────────┐
             └┬───────┬───────┬───────┬───────┬┘0.0206┤  *                             │
             1.0    49.2    97.5    145.8 194.0 0.0172┤  *                             │
        dtb                 iter                0.0103┤  *                             │
                          dtb/mean              0.0069┤  *                             │
             ┌─────────────────────────────────┐0.0001┤********************************│
        0.211┤  ·                              │      └┬───────┬───────┬──────┬───────┬┘
        0.176┤  ·                              │      1.0    49.2    97.5   145.8 194.0
             │  ·                              │dtb/std              iter
        0.141┤  ·                              │                   dtb/max
        0.106┤  ·                              │     ┌─────────────────────────────────┐
             │  ·                              │0.239┤  +                              │
        0.071┤  ·                              │0.199┤  +                              │
        0.036┤  ·                              │0.120┤  +                              │
             │  ·                              │0.081┤  +                              │
        0.001┤·································│0.002┤+++++++++++++++++++++++++++++++++│
             └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
             1.0    49.2    97.5    145.8 194.0      1.0    49.2    97.5    145.8 194.0
        dtb/mean            iter                dtb/max             iter
        text saved in /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709/plots/tplot/dtb.txt
             ┌─────────────────────────────────────────────────────────────────────────┐
        0.239┤ ++ dtb/max                                                              │
             │ -- dtb/min                                                              │
             │ ·· dtb/mean                                                             │
             │ ▞▞ dtb                                                                  │
        0.199┤     █                                                                   │
             │     █                                                                   │
             │     █                                                                   │
             │     █                                                                   │
        0.159┤     █                                                                   │
             │     █                                                                   │
             │     █                                                                   │
             │     █                                                                   │
        0.120┤     █                                                                   │
             │     █                                                                   │
             │     █                                                                   │
             │     █                                                                   │
             │     █                                                                   │
        0.080┤     █                                                                   │
             │     █                                                                   │
             │     █                                                                   │
             │     █                                                                   │
        0.041┤     █                                                                   │
             │     █                                                                   │
             │     █                                                                   │
             │     █                                                                   │
        0.001┤▄▄▄▄▄█▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
             └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
             1.0              49.2              97.5              145.8           194.0
        text saved in /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709/plots/tplot/dtb_summary.txt
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
           -0.008   0.049   0.106   0.163 0.221    -0.009   0.056   0.120   0.185 0.249
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
           -0.007   0.041   0.089   0.136 0.184    -0.0008  0.0048  0.0103  0.0159
        text saved in /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709/plots/tplot/dtb_hist.txt
                              dtf                                   dtf/min
              ┌────────────────────────────────┐      ┌────────────────────────────────┐
        0.0188┤            ▖         ▖     ▐   │0.0129┤  -            -  -   -  -  -   │
        0.0167┤            ▌      ▖  ▌     ▐   │0.0106┤--------------------------------│
              │  ▗         ▌     ▐▌  ▌     ▐   │0.0082┤  -  -   -  -  -  -  --  -  -   │
        0.0146┤  ▐      ▖  ▌  ▗  ▐▌  ▌  ▗  ▐   │0.0058┤  -  -   -  -  -  -   -  -  -   │
        0.0124┤▌ ▐ ▖▐▗▐▗▌  ▌▗▖▟▖ ▐▌▗ █▖ █ ▌▟  ▌│      └┬───────┬───────┬──────┬───────┬┘
        0.0103┤██▟████▟█▙▟ ▙████▛███▙██▟████▄█▙│      1.0    49.2    97.5   145.8 194.0
              │▀▀█ ▀▜▘▘▜▛▀▀▛▀▘█▀ ▐▝▀▜▛▘▘▌▘▀█▘  │dtf/min              iter
        0.0082┤  ▜  ▐  ▐▌  ▌  ▜  ▐  ▐▌  ▌  █   │                    dtf/std
        0.0061┤         ▘  ▌     ▝   ▘  ▌  ▜   │       ┌───────────────────────────────┐
              └┬───────┬───────┬──────┬───────┬┘0.00142┤            *   *          *   │
              1.0    49.2    97.5   145.8 194.0 0.00120┤       *    **  *   ***    *  *│
        dtf                  iter               0.00076┤**  * ***   *******************│
                           dtf/mean             0.00054┤*******************************│
              ┌────────────────────────────────┐0.00011┤ *** ********   * *    * ** ** │
        0.0136┤  ·                   ·     ·   │       └┬───────┬──────┬───────┬───────┘
        0.0124┤  ·            ·  ·   ·  ·  ·   │       1.0    49.2   97.5    145.8
              │  ·   ·  ·  ·  ·· ·   ·  ·  ·   │dtf/std              iter
        0.0111┤··········  ········· ··········│                    dtf/max
        0.0098┤································│      ┌────────────────────────────────┐
              │  ·  ·   ····  ·  ·  ··  ·  ·   │0.0188┤            +         +     +   │
        0.0085┤  ·  ·   ·  ·  ·  ·  ··  ·  ·   │0.0167┤  +         +     +   +     +   │
        0.0073┤  ·  ·   ·  ·  ·  ·  ··  ·  ·   │0.0126┤++++++++++++++++++++++++++++++++│
              │  ·  ·   ·  ·  ·  ·   ·  ·  ·   │0.0105┤  + ++++++++++ ++++  ++  +  + + │
        0.0060┤            ·                   │0.0064┤  +  +   +  +  +  +   +  +  +   │
              └┬───────┬───────┬──────┬───────┬┘      └┬───────┬───────┬──────┬───────┬┘
              1.0    49.2    97.5   145.8 194.0       1.0    49.2    97.5   145.8 194.0
        dtf/mean             iter               dtf/max              iter
        text saved in /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709/plots/tplot/dtf.txt
              ┌────────────────────────────────────────────────────────────────────────┐
        0.0188┤ ++ dtf/max                                                     ▗▌      │
              │ -- dtf/min                 ▖                     ▖             ▐▌      │
              │ ·· dtf/mean               ▐▌                    ▐▌             ▐▌      │
              │ ▞▞ dtf                    ▐▌                    ▐▌             ▐▌      │
        0.0166┤                           ▐▌             ▟      ▐▌             ▐▌      │
              │                           ▐▌             █      ▐▌             ▐▌      │
              │                           ▐▌             █      ▐▌             ▐▌      │
              │     ▗+                    ▐▌             █      ▐▌             ▐▌      │
        0.0145┤     █+                    ▐▌             █      ▐▌             ▐▌      │
              │     █+                    ▐▌        +    █      ▐▌             ▐▌      │
              │     █·      +             ▐▌      ▌ +    █    + ▐▌      ▗      ▐▌      │
              │     █·      ▖   ▗  ▟      ▐▌  +   ▌ +  + █    + ▐▌+  +  █  +++ ▐▌      │
        0.0123┤▌    █·     ▐▌   █  █      ▐▌ ++   ▌▗++ ++█    + ▐▌▖+ +  █+ +▖+▗▐▌    +▖│
              │▌    █·  ▗++▐▌▗▌ █ +█      ▐▌ ▟▟+++▌█+++++█+  ▟+ ▐▌▌▟ +  █+++▌+█▐▌++ +▐▌│
              │█▗▖+ █·++█++▐▌▐▌▗▜▗▌█+  ▖  ▐▌▗██▖+▟▌█▟▗+▖+█++▖█+▖▐▌▌█▗++ █▌++█▖█▐▌++▖▖▐▌│
              │█▌▛▟+█▗▌▖█▟▟█▐▐▌▐▐█▌█++▐▌ +▐▙▟██▙▟▐▌█▜▌▀▙▚▌▚▞██▟▌▐█▙▌█+▌▐█▌▟▌█▌█▐▙▗▗██▐▌│
              │█▌▌▐▄█▐▝▐████▐▌▛█▐█▐█+▟▐▌++▐▌██▐██▐▌█▐▌·▜-▌▐▘██▜▙▜▜█▌▜·▙▜▌█-▐██▐▐█▐▞▜█▀▌│
        0.0101┤▝▌▌▐·▌▀-▝█▝▛█▝▌-▝·▜▐▛▞▛▟▝▄▚▟▘▜▀▝▌▜▐▙▀▝▌---▌▝--▀▝█▐-▝▘▝▄▘▐▌▝--▀█▝▟▘▀-·▝-▝│
              │--- -▌ - ---█·-----▐▌------█ -----▐▌- - - ▌-  ---▐ -- --▐▌- --- █-- ----│
              │     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
              │     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
        0.0080┤     ▌      █·     ▐▌      █      ▐▌      ▌      ▐      ▐▌      █       │
              │     ▌      ▜·     ▐▌      █      ▝▌      ▌      ▐      ▐▌      █       │
              │     ·      -·     ▝▌      █      -·      ▌      ▐      ▐▌      █       │
              │             -      -      █       -      ·      ▝      ▝▌      ▜       │
        0.0058┤                           ▝              -              -      -       │
              └┬─────────────────┬─────────────────┬────────────────┬─────────────────┬┘
              1.0              49.2              97.5             145.8           194.0
        text saved in /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709/plots/tplot/dtf_summary.txt
                        dtf/mean hist                           dtf/max hist
            ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
        87.0┤                 ████             │89.0┤             ████                 │
        72.5┤                 ███████          │74.2┤             ████                 │
            │                 ███████          │    │          ███████                 │
        58.0┤                 ███████          │59.3┤          ███████                 │
        43.5┤                 ███████          │44.5┤          ███████                 │
            │                 ███████          │    │          ███████                 │
        29.0┤                 ███████          │29.7┤          ███████                 │
        14.5┤                 ███████          │14.8┤          ███████████             │
            │████         ███████████          │    │████   ██████████████             │
         0.0┤███████      █████████████████████│ 0.0┤████████████████████████   ███████│
            └┬───────┬────────┬───────┬────────┘    └┬───────┬────────┬───────┬────────┘
          0.0056  0.0077   0.0098  0.0119         0.0059  0.0092   0.0126  0.0160
                        dtf/min hist                            dtf/std hist
             ┌─────────────────────────────────┐    ┌──────────────────────────────────┐
        119.0┤                ████             │55.0┤       ███                        │
             │                ████             │    │       ███                        │
         99.2┤                ████             │45.8┤       ███                        │
         79.3┤                ████             │36.7┤   ███████████                    │
             │                ████             │    │   ███████████                    │
         59.5┤                ███████          │27.5┤   ███████████                    │
             │                ███████          │    │██████████████                    │
         39.7┤                ███████          │18.3┤█████████████████                 │
         19.8┤                ███████          │ 9.2┤█████████████████████             │
             │████            ███████          │    │████████████████████████      ████│
          0.0┤███████         █████████████████│ 0.0┤██████████████████████████████████│
             └┬───────┬───────┬───────┬────────┘    └┬───────┬────────┬───────┬────────┘
           0.0055  0.0074  0.0094  0.0113         0.00005  0.00040  0.00076 0.00112
        text saved in /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709/plots/tplot/dtf_hist.txt
                            loss                                  loss/min
            ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
        1.75┤▌                                 │1.62┤-                                 │
        1.46┤▌                                 │1.08┤--                                │
            │▚                                 │0.54┤ -----                            │
        1.17┤▐▖                                │0.01┤    ------------------------------│
        0.89┤ ▙▙                               │    └┬───────┬────────┬───────┬───────┬┘
        0.60┤ ▐█▄▗                             │    1.0    49.2     97.5    145.8 194.0
            │  ▝██▟▄ ▖                         │loss/min            iter
        0.31┤    ▝▝▝▀▛▛▄█▙▙▄▙▄▗▖ ▖             │                  loss/std
        0.03┤           ▝ ▝▀▀▀▀▀▀█▀▛▟█▙▚▜▙▟▟▙▙▄│     ┌─────────────────────────────────┐
            └┬───────┬────────┬───────┬───────┬┘0.137┤  *                              │
            1.0    49.2     97.5    145.8 194.0 0.116┤ *****                           │
        loss                iter                0.074┤** ***********                   │
                          loss/mean             0.053┤*      **************************│
            ┌──────────────────────────────────┐0.011┤                   * **  ********│
        1.70┤·                                 │     └┬───────┬───────┬───────┬───────┬┘
        1.42┤·                                 │     1.0    49.2    97.5    145.8 194.0
            │·                                 │loss/std            iter
        1.15┤ ·                                │                  loss/max
        0.87┤ ·                                │    ┌──────────────────────────────────┐
            │ ··                               │1.76┤+                                 │
        0.59┤  ···                             │1.48┤++                                │
        0.31┤    ····                          │0.91┤ ++++                             │
            │      ············                │0.63┤    +++++++++++++                 │
        0.03┤             ·  ··················│0.06┤            ++++++++++++++++++++++│
            └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
            1.0    49.2     97.5    145.8 194.0     1.0    49.2     97.5    145.8 194.0
        loss/mean           iter                loss/max            iter
        text saved in /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709/plots/tplot/loss.txt
            ┌──────────────────────────────────────────────────────────────────────────┐
        1.76┤ ++ loss/max                                                              │
            │ -- loss/min                                                              │
            │ ·· loss/mean                                                             │
            │ ▞▞ loss                                                                  │
        1.47┤▐+                                                                        │
            │▐+                                                                        │
            │▐·                                                                        │
            │ ▌                                                                        │
        1.18┤ ▌+                                                                       │
            │ ▌+                                                                       │
            │ ▌▖+                                                                      │
            │ ▝▌+                                                                      │
        0.89┤  ▌+▌                                                                     │
            │  ▌+▌▗                                                                    │
            │  ▝▖▌█+ +                                                                 │
            │  -▚██++++                                                                │
            │   -▜█▌▗▌▗ +                                                              │
        0.59┤   -▝▌▌▐▚█+++   +                                                         │
            │    --▐▌▐█▗ ▖++++++                                                       │
            │    ---▘ ▜▐▟▙▙ ++++++  +                                                  │
            │     ---- █·▀▝▞▄·▌▗▗▗▌+++▖ +  ++                                          │
        0.30┤     -  --▝- · ▘▜▌▞▜▞▚+▖▐▌▄+▟+++▗ + ▄+                                    │
            │        --- -----▝ -▌▐▚▚▌█▝▚▌▙▗▖█+▖+█+++▖+++▖++ + +  +              +  +  │
            │             - -----·---▘▀· ·▝▘█▐▞▝▚█▗▜▞▌▖▗▄▌++▟++▖▗▖++▖+▗+++++++++++++++ │
            │                    -  -----·--▝-▘--▘▀-▘▝▝▘▜▝▜▚▀▄▀▚▌▙▐·▙▐▜▗▀▌▟·▖·▖▗▖▄·▄·▗·│
        0.01┤                                          ---- ----▘-▘▀-▀-▀-▝▀▀▀▀▝▀▝▘▀·▀▀▀│
            └┬─────────────────┬──────────────────┬─────────────────┬─────────────────┬┘
            1.0              49.2               97.5              145.8           194.0
        text saved in /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709/plots/tplot/loss_summary.txt
                       loss/mean hist                          loss/max hist
             ┌─────────────────────────────────┐   ┌───────────────────────────────────┐
        127.0┤████                             │102┤████                               │
        105.8┤████                             │ 85┤████                               │
             │████                             │   │████                               │
         84.7┤████                             │ 68┤████                               │
         63.5┤████                             │ 51┤████                               │
             │████                             │   │███████                            │
         42.3┤████                             │ 34┤███████                            │
         21.2┤███████                          │ 17┤███████████                        │
             │█████████████                    │   │██████████████████                 │
          0.0┤██████████████████████████   ████│  0┤███████████████████████████████████│
             └┬───────┬───────┬───────┬───────┬┘   └┬────────┬───────┬────────┬───────┬┘
            -0.04   0.41    0.87    1.32   1.78   -0.01    0.45    0.91     1.38   1.84
                        loss/min hist                          loss/std hist
             ┌─────────────────────────────────┐  ┌────────────────────────────────────┐
        145.0┤████                             │72┤    ███                             │
             │████                             │  │    ███                             │
        120.8┤████                             │60┤    ███                             │
         96.7┤████                             │48┤    ███                             │
             │████                             │  │    ███                             │
         72.5┤████                             │36┤    ███████                         │
             │████                             │  │    ███████                         │
         48.3┤████                             │24┤    ██████████████                  │
         24.2┤███████                          │12┤█████████████████████████           │
             │██████████                       │  │█████████████████████████████       │
          0.0┤█████████████████████████████████│ 0┤████████████████████████████████████│
             └┬───────┬───────┬───────┬───────┬┘  └┬────────┬────────┬───────┬────────┬┘
            -0.06   0.38    0.81    1.25   1.69  0.006    0.040    0.074   0.109  0.143
        text saved in /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709/plots/tplot/loss_hist.txt
        [2025-12-26 12:58:07,565854][I][ezpz/history:2433:finalize] Saving history report to /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/outputs/ezpz.test_dist/2025-12-26-125709/report.md
        [2025-12-26 12:58:07,571828][I][ezpz/test_dist:348:finalize] dataset=<xarray.Dataset> Size: 39kB
        Dimensions:        (draw: 194)
        Coordinates:
          * draw           (draw) int64 2kB 0 1 2 3 4 5 6 ... 188 189 190 191 192 193
        Data variables: (12/25)
            iter           (draw) int64 2kB 6 7 8 9 10 11 12 ... 194 195 196 197 198 199
            loss           (draw) float32 776B 1.746 1.533 1.29 ... 0.03311 0.02764
            accuracy       (draw) float32 776B 0.625 0.6016 0.6328 ... 0.9922 0.9922
            dtf            (draw) float64 2kB 0.0127 0.01003 0.01162 ... 0.01053 0.01025
            dtb            (draw) float64 2kB 0.001811 0.001627 ... 0.001683 0.00256
            iter_mean      (draw) float64 2kB 6.0 7.0 8.0 9.0 ... 197.0 198.0 199.0
            ...             ...
            dtf_min        (draw) float64 2kB 0.01021 0.009599 ... 0.01024 0.009542
            dtf_std        (draw) float64 2kB 0.0007831 0.0006131 ... 0.0004008
            dtb_mean       (draw) float64 2kB 0.001742 0.001728 ... 0.001774 0.001822
            dtb_max        (draw) float64 2kB 0.002061 0.002182 ... 0.002031 0.00256
            dtb_min        (draw) float64 2kB 0.001459 0.00144 ... 0.001345 0.001372
            dtb_std        (draw) float64 2kB 0.0002062 0.0002116 ... 0.0002654
        [2025-12-26 12:58:08,256424][I][ezpz/test_dist:500:train] Took: 35.89 seconds to finish training
        [2025-12-26 12:58:08,257557][I][ezpz/test_dist:695:main] Took: 64.73 seconds
        wandb:
        wandb: 🚀 View run winter-salad-6843 at: https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/adhgoy9j
        wandb: Find logs at: ../../../../../../../../../lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/saforem2/tmp/2025-12-26-124007/wandb/run-20251226_125724-adhgoy9j/logs
        [2025-12-26 12:58:10,167355][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2025-12-26-125810]----
        [2025-12-26 12:58:10,168735][I][ezpz/launch:448:launch] Execution finished with 0.
        [2025-12-26 12:58:10,169220][I][ezpz/launch:449:launch] Executing finished in 68.93 seconds.
        [2025-12-26 12:58:10,169583][I][ezpz/launch:450:launch] Took 68.93 seconds to run. Exiting.
        took: 1m 16s
        ```

      </details>

    </details>

    - <details closed><summary>{Polaris} @ ALCF</summary>

        ```bash
        module load conda
        TMPDIR=$(pwd) uv run \
            --python=$(which python3) \
            --with "git+https://github.com/saforem2/ezpz" \
            ezpz test
        ```

        <details closed><summary>Output:</summary>

        ```bash
        (2025-09-25/base)
        #[/eagle/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131][⏱️ 1m56s]
        #[12/26/25 @ 13:20:57][x3102c0s13b0n0]
        ; TMPDIR=$(pwd) uv run --python=$(which python3) --with "git+https://github.com/saforem2/ezpz@distributed-metrics" ezpz test
            Updated https://github.com/saforem2/ezpz (e21d0a9cdc19557ad4f4be88fc2315af0fbfa2db)
            Updated https://github.com/saforem2/ambivalent (b8de07d9daad215d3db0d18b4aa99cb73107ef77)
              Built ezpz @ git+https://github.com/saforem2/ezpz@e21d0a9cdc19557ad4f4be88fc2315af0fbfa2db
              Built ambivalent @ git+https://github.com/saforem2/ambivalent@b8de07d9daad215d3db0d18b4aa99cb73107ef77
              Built antlr4-python3-runtime==4.9.3
        Installed 87 packages in 1.40s
        warning: `propcache==0.4.0` is yanked (reason: "ref leak https://github.com/aio-libs/propcache/issues/159")


        [2025-12-26 13:21:31,922789][I][ezpz/launch:396:launch] ----[🍋 ezpz.launch][started][2025-12-26-132131]----
        [2025-12-26 13:21:32,593377][I][ezpz/launch:416:launch] Job ID: 6826897
        [2025-12-26 13:21:32,594224][I][ezpz/launch:417:launch] nodelist: ['x3102c0s13b0n0', 'x3102c0s13b1n0']
        [2025-12-26 13:21:32,594624][I][ezpz/launch:418:launch] hostfile: /var/spool/pbs/aux/6826897.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
        [2025-12-26 13:21:32,595323][I][ezpz/pbs:264:get_pbs_launch_cmd] ✅ Using [8/8] GPUs [2 hosts] x [4 GPU/host]
        [2025-12-26 13:21:32,596845][I][ezpz/launch:367:build_executable] Building command to execute by piecing together:
        [2025-12-26 13:21:32,597234][I][ezpz/launch:368:build_executable] (1.) launch_cmd: mpiexec --envall --np=8 --ppn=4 --hostfile=/var/spool/pbs/aux/6826897.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind=depth --depth=8
        [2025-12-26 13:21:32,597798][I][ezpz/launch:369:build_executable] (2.) cmd_to_launch: /home/foremans/.cache/uv/builds-v0/.tmpwG7Oyq/bin/python -m ezpz.test_dist
        [2025-12-26 13:21:32,598339][I][ezpz/launch:433:launch] Took: 0.68 seconds to build command.
        [2025-12-26 13:21:32,598684][I][ezpz/launch:436:launch] Executing:
        mpiexec
          --envall
          --np=8
          --ppn=4
          --hostfile=/var/spool/pbs/aux/6826897.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
          --cpu-bind=depth
          --depth=8
          /home/foremans/.cache/uv/builds-v0/.tmpwG7Oyq/bin/python
          -m
          ezpz.test_dist
        [2025-12-26 13:21:32,600442][I][ezpz/launch:443:launch] Execution started @ 2025-12-26-132132...
        [2025-12-26 13:21:32,600884][I][ezpz/launch:139:run_command] Running command:
         mpiexec --envall --np=8 --ppn=4 --hostfile=/var/spool/pbs/aux/6826897.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind=depth --depth=8 /home/foremans/.cache/uv/builds-v0/.tmpwG7Oyq/bin/python -m ezpz.test_dist
        [2025-12-26 13:21:41,009597][I][ezpz/test_dist:132:__post_init__] Outputs will be saved to /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141
        [2025-12-26 13:21:41,011757][I][ezpz/dist:1506:setup_torch_distributed] Using fw='ddp' with torch_{device,backend}= {cuda, nccl}
        [2025-12-26 13:21:41,013713][I][ezpz/dist:1371:setup_torch_DDP] Caught MASTER_PORT=49717 from environment!
        [2025-12-26 13:21:41,014243][I][ezpz/dist:1387:setup_torch_DDP] Using torch.distributed.init_process_group with
        - master_addr='x3102c0s13b0n0.hsn.cm.polaris.alcf.anl.gov'
        - master_port='49717'
        - world_size=8
        - rank=0
        - local_rank=0
        - timeout=datetime.timedelta(seconds=3600)
        - backend='nccl'
        [2025-12-26 13:21:41,015130][I][ezpz/dist:1019:init_process_group] Calling torch.distributed.init_process_group_with: rank=0 world_size=8 backend=nccl
        [2025-12-26 13:21:48,305115][I][ezpz/dist:1732:setup_torch] Using device='cuda' with backend='nccl' + 'nccl' for distributed training.
        [2025-12-26 13:21:48,306060][W][ezpz/dist:544:print_dist_setup] Using [8 / 8] available "cuda" devices !!
        [2025-12-26 13:21:48,306511][I][ezpz/dist:1779:setup_torch] ['x3102c0s13b0n0'][device='cuda'][node=0/1][rank=0/7][local_rank=0/3]
        [2025-12-26 13:21:48,305536][I][ezpz/dist:1779:setup_torch] ['x3102c0s13b0n0'][device='cuda'][node=1/1][rank=3/7][local_rank=3/3]
        [2025-12-26 13:21:48,305529][I][ezpz/dist:1779:setup_torch] ['x3102c0s13b0n0'][device='cuda'][node=1/1][rank=1/7][local_rank=1/3]
        [2025-12-26 13:21:48,305423][I][ezpz/dist:1779:setup_torch] ['x3102c0s13b1n0'][device='cuda'][node=0/1][rank=4/7][local_rank=0/3]
        [2025-12-26 13:21:48,305537][I][ezpz/dist:1779:setup_torch] ['x3102c0s13b0n0'][device='cuda'][node=0/1][rank=2/7][local_rank=2/3]
        [2025-12-26 13:21:48,305414][I][ezpz/dist:1779:setup_torch] ['x3102c0s13b1n0'][device='cuda'][node=1/1][rank=5/7][local_rank=1/3]
        [2025-12-26 13:21:48,305415][I][ezpz/dist:1779:setup_torch] ['x3102c0s13b1n0'][device='cuda'][node=0/1][rank=6/7][local_rank=2/3]
        [2025-12-26 13:21:48,305412][I][ezpz/dist:1779:setup_torch] ['x3102c0s13b1n0'][device='cuda'][node=1/1][rank=7/7][local_rank=3/3]
        [2025-12-26 13:21:48,308064][I][ezpz/test_dist:678:main] Took: 7.31 seconds to setup torch
        [2025-12-26 13:21:48,321964][I][ezpz/test_dist:461:train] Model size: 567434 parameters
        [2025-12-26 13:21:48,323195][I][ezpz/test_dist:465:train]
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
        [2025-12-26 13:21:48,324424][I][ezpz/test_dist:473:train] Took: 0.005884354992303997 seconds to build model
        [2025-12-26 13:21:48,326217][I][ezpz/test_dist:601:build_model_and_optimizer] model=
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
        [2025-12-26 13:21:48,327959][I][ezpz/dist:685:wrap_model] Wrapping model with: ddp
        [2025-12-26 13:21:48,691473][I][ezpz/test_dist:479:train] Took: 0.37 seconds to build optimizer
        [2025-12-26 13:21:48,734475][I][ezpz/history:220:__init__] Using History with distributed_history=True
        [2025-12-26 13:21:48,738296][I][ezpz/dist:2044:setup_wandb] Setting up wandb from rank=0
        [2025-12-26 13:21:48,738722][I][ezpz/dist:2045:setup_wandb] Using WB_PROJECT=ezpz.test_dist
        wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
        wandb: setting up run 01zkj7vc
        wandb: Tracking run with wandb version 0.22.1
        wandb: Run data is saved locally in /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/wandb/run-20251226_132148-01zkj7vc
        wandb: Run `wandb offline` to turn off syncing.
        wandb: Syncing run smart-breeze-6848
        wandb:  View project at https://wandb.ai/aurora_gpt/ezpz.test_dist
        wandb:  View run at https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/01zkj7vc
        [2025-12-26 13:21:55,570075][I][ezpz/dist:2074:setup_wandb] wandb.run=[smart-breeze-6848](https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/01zkj7vc)
        [2025-12-26 13:21:55,577966][I][ezpz/dist:2117:setup_wandb] Running on machine='Polaris'
        [2025-12-26 13:21:56,263208][I][ezpz/test_dist:482:train] Took: 7.57 seconds to build trainer
        [2025-12-26 13:21:56,264200][I][ezpz/test_dist:486:train] config:
        {
          "acc_events": false,
          "backend": "DDP",
          "batch_size": 128,
          "cp": 1,
          "dataset": "mnist",
          "dataset_root": "/lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/datasets/mnist",
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
        [2025-12-26 13:21:56,266230][I][ezpz/test_dist:488:train] Took: 18.32 to get here.
        [2025-12-26 13:21:56,692071][I][ezpz/test_dist:369:train] Warmup complete at step 5
        [2025-12-26 13:21:56,803374][I][ezpz/test_dist:325:train_step] iter=10 loss=1.009584 accuracy=0.765625 dtf=0.016586 dtb=0.000765 loss/mean=1.138943 loss/max=1.389118 loss/min=0.988708 loss/std=0.116546 accuracy/mean=0.690430 accuracy/max=0.796875 accuracy/min=0.578125 accuracy/std=0.067085 dtf/mean=0.016839 dtf/max=0.017218 dtf/min=0.016586 dtf/std=0.000178 dtb/mean=0.000758 dtb/max=0.000768 dtb/min=0.000744 dtb/std=0.000008
        [2025-12-26 13:21:57,036584][I][ezpz/test_dist:325:train_step] iter=20 loss=0.516474 accuracy=0.812500 dtf=0.016623 dtb=0.000767 loss/mean=0.621663 loss/max=0.751371 loss/min=0.513288 loss/std=0.093839 accuracy/mean=0.791992 accuracy/max=0.859375 accuracy/min=0.718750 accuracy/std=0.046209 dtf/mean=0.016998 dtf/max=0.017245 dtf/min=0.016623 dtf/std=0.000208 dtb/mean=0.000759 dtb/max=0.000767 dtb/min=0.000751 dtb/std=0.000005
        [2025-12-26 13:21:57,265033][I][ezpz/test_dist:325:train_step] iter=30 loss=0.482071 accuracy=0.828125 dtf=0.016847 dtb=0.000769 loss/mean=0.436843 loss/max=0.533845 loss/min=0.284811 loss/std=0.069080 accuracy/mean=0.870117 accuracy/max=0.914062 accuracy/min=0.828125 accuracy/std=0.023089 dtf/mean=0.017028 dtf/max=0.017492 dtf/min=0.016678 dtf/std=0.000223 dtb/mean=0.000757 dtb/max=0.000769 dtb/min=0.000743 dtb/std=0.000008
        [2025-12-26 13:21:57,485773][I][ezpz/test_dist:325:train_step] iter=40 loss=0.411392 accuracy=0.843750 dtf=0.016916 dtb=0.000771 loss/mean=0.455263 loss/max=0.584419 loss/min=0.397925 loss/std=0.055186 accuracy/mean=0.859375 accuracy/max=0.875000 accuracy/min=0.843750 accuracy/std=0.012956 dtf/mean=0.017048 dtf/max=0.017304 dtf/min=0.016830 dtf/std=0.000140 dtb/mean=0.000759 dtb/max=0.000771 dtb/min=0.000751 dtb/std=0.000006
        [2025-12-26 13:21:57,720448][I][ezpz/test_dist:325:train_step] iter=50 loss=0.340432 accuracy=0.859375 dtf=0.017033 dtb=0.000771 loss/mean=0.400236 loss/max=0.587103 loss/min=0.278782 loss/std=0.088603 accuracy/mean=0.871094 accuracy/max=0.906250 accuracy/min=0.843750 accuracy/std=0.024080 dtf/mean=0.017107 dtf/max=0.017321 dtf/min=0.016968 dtf/std=0.000112 dtb/mean=0.000767 dtb/max=0.000785 dtb/min=0.000748 dtb/std=0.000011
        [2025-12-26 13:21:57,968693][I][ezpz/test_dist:325:train_step] iter=60 loss=0.325704 accuracy=0.906250 dtf=0.018421 dtb=0.000773 loss/mean=0.347035 loss/max=0.470769 loss/min=0.274286 loss/std=0.057969 accuracy/mean=0.888672 accuracy/max=0.906250 accuracy/min=0.828125 accuracy/std=0.024316 dtf/mean=0.018716 dtf/max=0.018999 dtf/min=0.018345 dtf/std=0.000219 dtb/mean=0.000764 dtb/max=0.000776 dtb/min=0.000751 dtb/std=0.000008
        [2025-12-26 13:21:58,215199][I][ezpz/test_dist:325:train_step] iter=70 loss=0.242337 accuracy=0.914062 dtf=0.016899 dtb=0.000785 loss/mean=0.260672 loss/max=0.361649 loss/min=0.186009 loss/std=0.053688 accuracy/mean=0.916016 accuracy/max=0.945312 accuracy/min=0.882812 accuracy/std=0.017794 dtf/mean=0.017151 dtf/max=0.017322 dtf/min=0.016899 dtf/std=0.000136 dtb/mean=0.000774 dtb/max=0.000789 dtb/min=0.000758 dtb/std=0.000012
        [2025-12-26 13:21:58,472737][I][ezpz/test_dist:325:train_step] iter=80 loss=0.344910 accuracy=0.882812 dtf=0.016888 dtb=0.000774 loss/mean=0.274805 loss/max=0.344910 loss/min=0.163093 loss/std=0.059792 accuracy/mean=0.918945 accuracy/max=0.960938 accuracy/min=0.882812 accuracy/std=0.027046 dtf/mean=0.017064 dtf/max=0.017452 dtf/min=0.016775 dtf/std=0.000201 dtb/mean=0.000762 dtb/max=0.000774 dtb/min=0.000756 dtb/std=0.000005
        [2025-12-26 13:21:58,701404][I][ezpz/test_dist:325:train_step] iter=90 loss=0.260920 accuracy=0.914062 dtf=0.016934 dtb=0.000776 loss/mean=0.221058 loss/max=0.312963 loss/min=0.097677 loss/std=0.066769 accuracy/mean=0.930664 accuracy/max=0.992188 accuracy/min=0.898438 accuracy/std=0.027466 dtf/mean=0.017072 dtf/max=0.017282 dtf/min=0.016857 dtf/std=0.000142 dtb/mean=0.000762 dtb/max=0.000776 dtb/min=0.000755 dtb/std=0.000006
        [2025-12-26 13:21:58,925449][I][ezpz/test_dist:325:train_step] iter=100 loss=0.290902 accuracy=0.914062 dtf=0.017022 dtb=0.000771 loss/mean=0.219431 loss/max=0.290902 loss/min=0.158593 loss/std=0.038115 accuracy/mean=0.937500 accuracy/max=0.953125 accuracy/min=0.914062 accuracy/std=0.012353 dtf/mean=0.017146 dtf/max=0.017407 dtf/min=0.016838 dtf/std=0.000171 dtb/mean=0.000763 dtb/max=0.000771 dtb/min=0.000756 dtb/std=0.000004
        [2025-12-26 13:21:59,183043][I][ezpz/test_dist:325:train_step] iter=110 loss=0.270826 accuracy=0.914062 dtf=0.016910 dtb=0.000785 loss/mean=0.220031 loss/max=0.311172 loss/min=0.142488 loss/std=0.060282 accuracy/mean=0.934570 accuracy/max=0.960938 accuracy/min=0.914062 accuracy/std=0.016544 dtf/mean=0.017096 dtf/max=0.017434 dtf/min=0.016804 dtf/std=0.000188 dtb/mean=0.000762 dtb/max=0.000785 dtb/min=0.000753 dtb/std=0.000009
        [2025-12-26 13:21:59,396895][I][ezpz/test_dist:325:train_step] iter=120 loss=0.304672 accuracy=0.921875 dtf=0.017031 dtb=0.000768 loss/mean=0.231112 loss/max=0.329426 loss/min=0.110154 loss/std=0.073585 accuracy/mean=0.928711 accuracy/max=0.953125 accuracy/min=0.882812 accuracy/std=0.024531 dtf/mean=0.017054 dtf/max=0.017213 dtf/min=0.016711 dtf/std=0.000159 dtb/mean=0.000760 dtb/max=0.000769 dtb/min=0.000743 dtb/std=0.000008
        [2025-12-26 13:21:59,631761][I][ezpz/test_dist:325:train_step] iter=130 loss=0.232980 accuracy=0.945312 dtf=0.017138 dtb=0.000771 loss/mean=0.235195 loss/max=0.355287 loss/min=0.102751 loss/std=0.074560 accuracy/mean=0.927734 accuracy/max=0.976562 accuracy/min=0.898438 accuracy/std=0.022693 dtf/mean=0.017109 dtf/max=0.017356 dtf/min=0.016762 dtf/std=0.000199 dtb/mean=0.000760 dtb/max=0.000777 dtb/min=0.000750 dtb/std=0.000009
        [2025-12-26 13:21:59,862446][I][ezpz/test_dist:325:train_step] iter=140 loss=0.168414 accuracy=0.968750 dtf=0.016910 dtb=0.000771 loss/mean=0.210054 loss/max=0.340699 loss/min=0.129359 loss/std=0.068940 accuracy/mean=0.940430 accuracy/max=0.968750 accuracy/min=0.890625 accuracy/std=0.024686 dtf/mean=0.017123 dtf/max=0.017356 dtf/min=0.016893 dtf/std=0.000170 dtb/mean=0.000759 dtb/max=0.000771 dtb/min=0.000751 dtb/std=0.000006
        [2025-12-26 13:22:00,085098][I][ezpz/test_dist:325:train_step] iter=150 loss=0.237147 accuracy=0.929688 dtf=0.016932 dtb=0.000775 loss/mean=0.167624 loss/max=0.237147 loss/min=0.122940 loss/std=0.040060 accuracy/mean=0.941406 accuracy/max=0.953125 accuracy/min=0.921875 accuracy/std=0.012353 dtf/mean=0.017041 dtf/max=0.017280 dtf/min=0.016753 dtf/std=0.000176 dtb/mean=0.000757 dtb/max=0.000775 dtb/min=0.000740 dtb/std=0.000009
        [2025-12-26 13:22:00,305868][I][ezpz/test_dist:325:train_step] iter=160 loss=0.208926 accuracy=0.945312 dtf=0.016980 dtb=0.000771 loss/mean=0.186015 loss/max=0.215280 loss/min=0.128407 loss/std=0.027561 accuracy/mean=0.941406 accuracy/max=0.960938 accuracy/min=0.929688 accuracy/std=0.008735 dtf/mean=0.017058 dtf/max=0.017327 dtf/min=0.016779 dtf/std=0.000193 dtb/mean=0.000756 dtb/max=0.000771 dtb/min=0.000737 dtb/std=0.000009
        [2025-12-26 13:22:00,525172][I][ezpz/test_dist:325:train_step] iter=170 loss=0.232940 accuracy=0.921875 dtf=0.017109 dtb=0.000773 loss/mean=0.198723 loss/max=0.269332 loss/min=0.122802 loss/std=0.053061 accuracy/mean=0.940430 accuracy/max=0.968750 accuracy/min=0.906250 accuracy/std=0.020647 dtf/mean=0.017133 dtf/max=0.017396 dtf/min=0.016898 dtf/std=0.000146 dtb/mean=0.000757 dtb/max=0.000773 dtb/min=0.000743 dtb/std=0.000008
        [2025-12-26 13:22:00,741349][I][ezpz/test_dist:325:train_step] iter=180 loss=0.051174 accuracy=0.992188 dtf=0.016878 dtb=0.000779 loss/mean=0.142097 loss/max=0.257418 loss/min=0.051174 loss/std=0.076244 accuracy/mean=0.966797 accuracy/max=0.992188 accuracy/min=0.929688 accuracy/std=0.022011 dtf/mean=0.017102 dtf/max=0.017473 dtf/min=0.016812 dtf/std=0.000194 dtb/mean=0.000762 dtb/max=0.000779 dtb/min=0.000750 dtb/std=0.000008
        [2025-12-26 13:22:00,962154][I][ezpz/test_dist:325:train_step] iter=190 loss=0.105810 accuracy=0.945312 dtf=0.016914 dtb=0.000775 loss/mean=0.152862 loss/max=0.230180 loss/min=0.094466 loss/std=0.049649 accuracy/mean=0.951172 accuracy/max=0.976562 accuracy/min=0.937500 accuracy/std=0.012807 dtf/mean=0.017123 dtf/max=0.017377 dtf/min=0.016858 dtf/std=0.000202 dtb/mean=0.000761 dtb/max=0.000775 dtb/min=0.000752 dtb/std=0.000007
        [2025-12-26 13:22:04,963504][I][ezpz/history:2385:finalize] Saving plots to /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141/plots/mplot (matplotlib) and /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141/plots/tplot (tplot)
                                    accuracy                                                  accuracy/min
             ┌─────────────────────────────────────────────────────┐     ┌─────────────────────────────────────────────────────┐
        0.992┤               ▖       ▖▄▖ ▗▖▗▗▗▗▄▗▟▄▄▄ ▞▄▟▄▖▄▄▟▗▛▄▄▄│0.953┤   --------------------------------------------------│
        0.930┤        ▄▞▄▟▄▙▟▙▄█▛▙█▀▀▀ ▝▛█▐▜▛▜█▀▀▘▛ ▝▜   ▝▛▜  ▜  ▘▘│0.641┤----                                                 │
        0.867┤    ▞█▜▚▀▘▀█▘▝▘ ▜               ▘                    │     └┬────────────┬────────────┬────────────┬────────────┬┘
        0.742┤▗█▟▟▌▝ ▝   ▘                                         │     1.0         49.2         97.5         145.8      194.0
        0.680┤▐ ▝▌                                                 │accuracy/min                  iter
        0.617┤▜                                                    │                          accuracy/std
             └┬────────────┬────────────┬────────────┬────────────┬┘     ┌─────────────────────────────────────────────────────┐
             1.0         49.2         97.5         145.8      194.0 0.069┤***                                                  │
        accuracy                      iter                          0.049┤******************* *****  *******  **    **  * **   │
                                  accuracy/mean                     0.017┤      * ***  ** ******** ****** *********************│
             ┌─────────────────────────────────────────────────────┐     └┬────────────┬────────────┬────────────┬────────────┬┘
        0.969┤                         · ··························│     1.0         49.2         97.5         145.8      194.0
        0.900┤      ·····························                  │accuracy/std                  iter
        0.832┤    ········                                         │                          accuracy/max
        0.764┤  ···                                                │     ┌─────────────────────────────────────────────────────┐
        0.695┤ ·                                                   │1.000┤      +++++++++++++++++++++++++++++++++++++++++++++++│
        0.627┤··                                                   │0.885┤ +++++++++++++ +                                     │
        0.559┤·                                                    │0.714┤++                                                   │
             └┬────────────┬────────────┬────────────┬────────────┬┘     └┬────────────┬────────────┬────────────┬────────────┬┘
             1.0         49.2         97.5         145.8      194.0      1.0         49.2         97.5         145.8      194.0
        accuracy/mean                 iter                          accuracy/max                  iter
        text saved in /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141/plots/tplot/accuracy.txt
             ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        1.000┤ ++ accuracy/max                                 +                               +   +       + +   + ▗ + ++ +    │
             │ -- accuracy/min                ▟ +    +  ++  + ++++▟▗  ++ ++++ +++▖++ ++++▗▀▙▌+▖+▖++▗▞▚▌▖+▟+·+··+▟▞▄█++▗▙▚+▖▟·▟▞│
             │ ·· accuracy/mean          ++ ++█ ++▖++▗+▖+▗+▗++▖▟·▞▘▜+·+▄▗▙▌ ▄▞▖▗█▌ ▄▜▗▞▞▄▘·█▌▞▝▀▝▄▖▌··▝▝▀▘▚▟▞▖▄▙▘▘·▘▚▖▌▝·▀▝▌▀▐▌│
        0.914┤ ▞▞ accuracy       +▟+++▟++▟+▖▗▚█+ ▐▚▟▗▜▐▚▟▛▄▘▀▀▝▘▀···▀▀▞▝▟█▝▞▐▌▌▐▝▌▐█-▘··▝--▜▝ ----█--------▝▌▝-▜-----▝▌---  --▘│
             │         +  ++ + ▗▗·▌▌▞▄█▗▀▀▟▐▐▝▌▚▟▐·▝▞▝▌-▝▌▜-- ------- --▝▝ - ▘▚▘-▚▌▝  -     --    ▝                  -         │
             │      + +▗▄▄▚▟▗··▌▀▖▌▌▌▐█▐· ▝·▜-- ▐▌----- - -- -    --          -  -▘                                            │
        0.828┤      ++ ▌·█·▝▌▚▐-·▝▘▝▘▐▌▘---  - -▝▌  --                                                                         │
             │  ++ ++▗▙▘·▜-- ▝▟--  - ▐▌  -                                                                                     │
             │ ▗+▗+▖▗██ - --  ▝--     ▘                                                                                        │
        0.742┤ ▌▀▘▀▚▐▝▝--  -    -                                                                                              │
             │ ▌ ···█--                                                                                                        │
             │▗▌·  -█                                                                                                          │
             │█▌· --▜                                                                                                          │
        0.656┤▜▌ ---                                                                                                           │
             │▝▌-- -                                                                                                           │
             │ ·-                                                                                                              │
        0.570┤·--                                                                                                              │
             │ -                                                                                                               │
             │ -                                                                                                               │
        0.484┤-                                                                                                                │
             └┬───────────────────────────┬───────────────────────────┬───────────────────────────┬───────────────────────────┬┘
             1.0                        49.2                        97.5                        145.8                     194.0
        text saved in /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141/plots/tplot/accuracy_summary.txt
                              accuracy/mean hist                                            accuracy/max hist
          ┌────────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
        96┤                                                   █████│79.0┤                                                 █████│
        80┤                                                   █████│65.8┤                                           ███████████│
        64┤                                                   █████│52.7┤                                           ███████████│
        48┤                                             █████ █████│39.5┤                                           ███████████│
        32┤                                       ███████████ █████│26.3┤                                      ████████████████│
        16┤                                       ███████████ █████│13.2┤                                ██████████████████████│
         0┤█████ ████████████████████████████████████████████ █████│ 0.0┤███████████     ██████████████████████████████████████│
          └┬─────────────┬─────────────┬────────────┬─────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
         0.54          0.65          0.76         0.88         0.99    0.64         0.73          0.83         0.92        1.02
                                accuracy/min hist                                           accuracy/std hist
            ┌──────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
        85.0┤                                                 █████│57.0┤     ███████████                                      │
        70.8┤                                                 █████│47.5┤     ███████████                                      │
        56.7┤                                                 █████│38.0┤     ███████████                                      │
        42.5┤                                           ███████████│28.5┤     ███████████                                      │
            │                                           ███████████│    │██████████████████████                                │
        28.3┤                                      ████████████████│19.0┤███████████████████████████                           │
        14.2┤                           ███████████████████████████│ 9.5┤██████████████████████████████████████                │
         0.0┤██████████████████████████████████████████████████████│ 0.0┤███████████████████████████████████████████      █████│
            └┬────────────┬─────────────┬────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
           0.46         0.59          0.72         0.85        0.97    0.004        0.021         0.038        0.055      0.072
        text saved in /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141/plots/tplot/accuracy_hist.txt
                                         dtb                                                       dtb/min
                ┌──────────────────────────────────────────────────┐        ┌──────────────────────────────────────────────────┐
        0.000923┤             ▐                                    │0.000880┤             -                                    │
        0.000896┤             ▐                                    │0.000769┤--------------------------------------------------│
        0.000868┤             ▐                                    │        └┬───────────┬────────────┬───────────┬───────────┬┘
        0.000814┤             ▐                                ▟   │        1.0        49.2         97.5        145.8     194.0
        0.000786┤        ▖    ▟  ▙ ▖  ▗ ▄▖▖▄ ▗ ▟ ▐    ▖  ▖ ▗▄▌▖█▗  │dtb/min                         iter
        0.000759┤▄▄▙▛▛▛▀▀▀▜▀▛▀▀▀▀▀▀▛▀▀▀▀▀▀▀▀▀▀▀▀▀▘▀▀▀▀▀▀▀▀▀▘▀▝▀▀▀▀▀│                               dtb/std
                └┬───────────┬────────────┬───────────┬───────────┬┘         ┌─────────────────────────────────────────────────┐
                1.0        49.2         97.5        145.8     194.0 0.0000341┤              *                            * **  │
        dtb                             iter                        0.0000241┤* * *        ****    **     ** **    *** *** ** *│
                                      dtb/mean                      0.0000091┤*************************************************│
                ┌──────────────────────────────────────────────────┐         └┬───────────┬───────────┬───────────┬───────────┬┘
        0.000893┤             ·                                    │         1.0        49.2        97.5        145.8     194.0
        0.000869┤             ·                                    │dtb/std                         iter
        0.000845┤             ·                                    │                               dtb/max
        0.000820┤             ·                                    │        ┌──────────────────────────────────────────────────┐
        0.000796┤             ·                                    │0.000923┤             +                                    │
        0.000772┤    · ·································   ········│0.000868┤+            ++ +     ++    + + +          ++ +   │
        0.000748┤·············    ··  ·   ·· ················ ·  · │0.000786┤++++++++++++++++++++++++++++++++++++++++++++++++++│
                └┬───────────┬────────────┬───────────┬───────────┬┘        └┬───────────┬────────────┬───────────┬───────────┬┘
                1.0        49.2         97.5        145.8     194.0         1.0        49.2         97.5        145.8     194.0
        dtb/mean                        iter                        dtb/max                         iter
        text saved in /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141/plots/tplot/dtb.txt
                ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        0.000923┤ ++ dtb/max                  ▗▌                                                                               │
                │ -- dtb/min                  ▐▌                                                                               │
                │ ·· dtb/mean                 ▐▌                                                                               │
        0.000888┤ ▞▞ dtb                      ▐▌                                                                               │
                │                             ▐▌                                                                               │
                │                             ▐▌                                                                               │
        0.000853┤                             ▐▌                                                                  +            │
                │                             ▐▌ +                  +                                             +            │
                │                             ▐▌ +                 ++           +                                 +    ▗+      │
        0.000819┤                             ▐▌ +                 ++          ++                                 +    █+      │
                │+                            ▐▌ +                 ++          ++                                 +    █+      │
                │+                            ▐▌ +  ▟+           + ++          ++  ▗▌    ▖           +           ▗▌    ▛▖      │
                │+         +                 ▗▐▌ + +▌▌+ ▗+  +   ++ ++▖         +▗  ▐▌   ▐▌          ++         ▟ ▐▌    ▌▌   ++ │
        0.000784┤+   +   ▟▗+   ▖  ▟+  ▖   + +▛▟▌ ▄▄+▌▐+▖▛▖▄ + +▗▌+ ▟▐▝▌▄▌▗▟▗▚ ▗+█ ▖▐▌+ +▌▚+▖ +  ▖+ ▟+++ ▗▌+ + ▞▘▚▞▝▄▟ +▌▌▗▚▗++ │
                │+ ▗▄▌▗+▄█▛▄▄▚▟▝▞▀▌▀▀▜▙▀▄▞▀▞▚▌▝▝▀··▀··▀▝·▜ ▀▀▀▀▀▝▀▀·▜·▝▝▚▘·▀▝▞▘▀▀▟▝▀▝▀▀▀··▀▝▀▀▀▀▝▞▞+▀▀▀▀▀▝▚▀▚▀▘  ·· +▚▀▘▚▀ ▘▀▀▀│
                │▀▚▘▝▚▘▀··▘·▝·········▝··▘·····-· ·- -··-········ ··· ····-···-·-·································-····· ······│
        0.000749┤ ·······- ----------------------- -- -  ------- --------- --- ----  ------------- ---     -  - ------------ --│
                │--------  -                                                       -- - ---      --  -----------               │
                │-                                                                                                             │
        0.000714┤-                                                                                                             │
                └┬──────────────────────────┬───────────────────────────┬──────────────────────────┬──────────────────────────┬┘
                1.0                       49.2                        97.5                       145.8                    194.0
        text saved in /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141/plots/tplot/dtb_summary.txt
                                  dtb/mean hist                                               dtb/max hist
             ┌─────────────────────────────────────────────────────┐     ┌─────────────────────────────────────────────────────┐
        130.0┤█████                                                │104.0┤█████                                                │
        108.3┤█████                                                │ 86.7┤█████                                                │
         86.7┤█████                                                │ 69.3┤███████████                                          │
         65.0┤███████████                                          │ 52.0┤███████████                                          │
         43.3┤███████████                                          │ 34.7┤███████████                                          │
         21.7┤███████████                                          │ 17.3┤███████████                                          │
          0.0┤████████████████                                █████│  0.0┤██████████████████████████ █████                █████│
             └┬────────────┬────────────┬────────────┬────────────┬┘     └┬────────────┬────────────┬────────────┬────────────┬┘
           0.000741    0.000781     0.000820     0.000860  0.000900    0.000752    0.000796     0.000841     0.000886  0.000930
                                  dtb/min hist                                               dtb/std hist
             ┌─────────────────────────────────────────────────────┐  ┌────────────────────────────────────────────────────────┐
        111.0┤           █████                                     │90┤      █████                                             │
         92.5┤           █████                                     │75┤      █████                                             │
         74.0┤     ███████████                                     │60┤█████ █████                                             │
         55.5┤     ███████████                                     │45┤█████ █████                                             │
             │     ███████████                                     │  │█████ █████                                             │
         37.0┤     ███████████                                     │30┤█████ ███████████                                       │
         18.5┤     ███████████                                     │15┤█████ ███████████                                       │
          0.0┤█████████████████████                           █████│ 0┤█████ ████████████████████████████████████████████ █████│
             └┬────────────┬────────────┬────────────┬────────────┬┘  └┬─────────────┬─────────────┬────────────┬─────────────┬┘
           0.000707    0.000752     0.000797     0.000842  0.000887  0.000003    0.000011      0.000019     0.000027   0.000035
        text saved in /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141/plots/tplot/dtb_hist.txt
                                        dtf                                                       dtf/min
              ┌────────────────────────────────────────────────────┐      ┌────────────────────────────────────────────────────┐
        0.0188┤              ▟          ▗    ▖              ▗▌     │0.0183┤----------------------------------------------------│
        0.0174┤▄▄▄▙▜▟▄▄▞▙█▜▜▛▛█▜▛▛▚▟▞▛█▟▀▀▄▙▜▛▞▙▀█▀▜▛█▙▜██▛▀▜▜▛▞█▜▙│0.0131┤              -               -              -      │
        0.0160┤              ▌              ▐▌              ▐      │      └┬────────────┬────────────┬───────────┬────────────┬┘
        0.0132┤              ▌              ▐▌              ▐      │      1.0         49.2         97.5        145.8      194.0
        0.0118┤              ▌              ▐▌              ▐      │dtf/min                        iter
        0.0104┤              ▌              ▝▌              ▐      │                               dtf/std
              └┬────────────┬────────────┬───────────┬────────────┬┘        ┌──────────────────────────────────────────────────┐
              1.0         49.2         97.5        145.8      194.0 0.000452┤     * *              *               *           │
        dtf                            iter                         0.000331┤* **********  *  *********** * *** ******** ******│
                                     dtf/mean                       0.000150┤************************************************* │
              ┌────────────────────────────────────────────────────┐        └┬───────────┬────────────┬───────────┬───────────┬┘
        0.0187┤              ·               ·              ·      │        1.0        49.2         97.5        145.8     194.0
        0.0174┤····················································│dtf/std                         iter
        0.0160┤              ·              ··              ·      │                              dtf/max
        0.0146┤              ·              ··              ·      │      ┌────────────────────────────────────────────────────┐
        0.0133┤              ·              ··              ·      │0.0190┤++++++++++++++++++++++++++++++++++++++++++++++++++++│
        0.0119┤              ·              ··              ·      │0.0162┤              +              ++              +      │
        0.0106┤              ·               ·              ·      │0.0121┤              +               +              +      │
              └┬────────────┬────────────┬───────────┬────────────┬┘      └┬────────────┬────────────┬───────────┬────────────┬┘
              1.0         49.2         97.5        145.8      194.0       1.0         49.2         97.5        145.8      194.0
        dtf/mean                       iter                         dtf/max                        iter
        text saved in /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141/plots/tplot/dtf.txt
              ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        0.0190┤ ++ dtf/max                    +                                                                   ▗            │
              │ -- dtf/min                   +▖                 +               ·                                +█            │
              │ ·· dtf/mean                  ▐▌                ++              ·▟                   +            ·█ +    +     │
        0.0176┤ ▞▞ dtf      ++ +    +   ++   ▐▐+  + +  +  +    ++ ++ ▗▌+   ++++·█▗+ +  ▖   +▖▖++    +   ▗+  + +  ·█++ +  + + ++│
              │+++++++·▞▖·▗+······▗··▖▄▟▗▜▗▗▚▐▝▟▗▗··▞▖▗▄▄····▄▖▗▄▗▌▗▗▟▚▄·▖·▖·▗▗·▛█·▖▗·▞▝▖▖▄▐▜▌·▗▄·▖▗·▄▖·█·▗▖▄▚▄▀▄▌▛▄▌▗▖·▖▗▖▄·▖·│
              │▚▀▄▖▄▄▀▚▌▝▀▌▀▀▀▀▀▀▀▘▀▀▝--▘-▘▀-█ -▘▘▀▀▘▝▘--▀▀▀▀-▝▘-▀▝▘▘---▀▝▀▝▀▘▘▌▌-▀▝▘▀--▝▝-▀-▝▀▘-▀▝▘▀-▝▀-▀▘▝▝----▌▌▝▝▘▝▀▝▘▝-▀▝▀│
        0.0161┤   ▝                          █                                 ▌▌                                ▌▌            │
              │                              █                                 ▌▌                                ▌▌            │
              │                              █                                 ▌▌                                ▌▌            │
        0.0147┤                              █                                 ▌▌                                ▌▌            │
              │                              █                                 ▌▌                                ▌▌            │
              │                              █                                 █                                 ▚▌            │
              │                              █                                 █                                 ▐▌            │
        0.0133┤                              █                                 █                                 ▐▌            │
              │                              █                                 █                                 ▐▌            │
              │                              █                                 █                                 ▐▌            │
        0.0118┤                              █                                 █                                 ▐▌            │
              │                              █                                 █                                 ▐▌            │
              │                              █                                 █                                 ▐▌            │
        0.0104┤                              ▜                                 ▜                                 ·▘            │
              └┬───────────────────────────┬───────────────────────────┬──────────────────────────┬───────────────────────────┬┘
              1.0                        49.2                        97.5                       145.8                     194.0
        text saved in /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141/plots/tplot/dtf_summary.txt
                                  dtf/mean hist                                              dtf/max hist
             ┌─────────────────────────────────────────────────────┐  ┌────────────────────────────────────────────────────────┐
        100.0┤                                          ██████     │96┤                                       ███████████      │
         83.3┤                                     ███████████     │80┤                                       ███████████      │
         66.7┤                                     ███████████     │64┤                                       ███████████      │
         50.0┤                                     ███████████     │48┤                                       ███████████      │
         33.3┤                                     ███████████     │32┤                                       ███████████      │
         16.7┤                                     ███████████     │16┤                                       ███████████      │
          0.0┤█████                                ████████████████│ 0┤█████                                  ███████████ █████│
             └┬────────────┬────────────┬────────────┬────────────┬┘  └┬─────────────┬─────────────┬────────────┬─────────────┬┘
           0.0102       0.0124       0.0146       0.0169     0.0191  0.0104       0.0126        0.0149       0.0171      0.0194
                                 dtf/min hist                                                 dtf/std hist
           ┌───────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
        138┤                                            █████      │73.0┤     ███████████                                      │
        115┤                                            █████      │60.8┤     ███████████                                      │
         92┤                                            █████      │48.7┤     ███████████                                      │
         69┤                                            █████      │36.5┤     ███████████                                      │
           │                                       ██████████      │    │     ███████████                                      │
         46┤                                       ██████████      │24.3┤██████████████████████                                │
         23┤                                       ██████████      │12.2┤██████████████████████                                │
          0┤█████                                  ██████████ █████│ 0.0┤███████████████████████████     ██████     ███████████│
           └┬─────────────┬────────────┬─────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
          0.0101       0.0122       0.0144        0.0165     0.0187   0.00007      0.00017       0.00027      0.00037   0.00047
        text saved in /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141/plots/tplot/dtf_hist.txt
                                      loss                                                      loss/min
            ┌──────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
        1.76┤▚                                                     │1.76┤--                                                    │
        1.48┤▐                                                     │0.62┤ -----------------------------------------------------│
        1.19┤ ▌                                                    │    └┬────────────┬─────────────┬────────────┬────────────┬┘
        0.91┤ ▝▚▚▄   ▖                                             │    1.0         49.2          97.5         145.8      194.0
        0.34┤    ▀▜▀▀▙▄▜▟▙▟▟▌▟▄▖▄▗▄▖ ▖ ▄▄▗▖▖▄▄▖   ▄                │loss/min                      iter
        0.05┤           ▝  ▝▝ ▀▝▀▘▀▀▀▝▀▀▀▀▀▀▀▛▐▀▀▀▀▀▛▀▄▛▀▜▀▀█▛▀█▞█▙│                            loss/std
            └┬────────────┬─────────────┬────────────┬────────────┬┘     ┌─────────────────────────────────────────────────────┐
            1.0         49.2          97.5         145.8      194.0 0.127┤ *  **  *      **                                    │
        loss                          iter                          0.092┤*********************************** *************** *│
                                    loss/mean                       0.038┤*               *  * * * **   * **** ****************│
            ┌──────────────────────────────────────────────────────┐     └┬────────────┬────────────┬────────────┬────────────┬┘
        1.84┤·                                                     │     1.0         49.2         97.5         145.8      194.0
        1.55┤··                                                    │loss/std                      iter
        1.26┤ ·                                                    │                            loss/max
        0.98┤ ··                                                   │    ┌──────────────────────────────────────────────────────┐
        0.69┤  ····                                                │1.92┤++                                                    │
        0.40┤     ··················· ··  ·                        │1.32┤ +++++++++ ++                                         │
        0.12┤                   · ·································│0.44┤       +++++++++++++++++++++++++++++++++++++++++++++++│
            └┬────────────┬─────────────┬────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
            1.0         49.2          97.5         145.8      194.0     1.0         49.2          97.5         145.8      194.0
        loss/mean                     iter                          loss/max                      iter
        text saved in /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141/plots/tplot/loss.txt
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        1.92┤ ++ loss/max                                                                                                      │
            │ -- loss/min                                                                                                      │
            │ ·· loss/mean                                                                                                     │
        1.61┤ ▞▞ loss                                                                                                          │
            │▐·                                                                                                                │
            │ ▌+                                                                                                               │
        1.29┤ ▚·                                                                                                               │
            │ ▐·                                                                                                               │
            │  ▌++                                                                                                             │
        0.98┤  ▚·+                                                                                                             │
            │  ▝▖·+  +                                                                                                         │
            │   ▝▖·▗++                                                                                                         │
            │    ▝▄▜▗++ ++                                                                                                     │
        0.67┤     ▝·▜▗▌▖+++ +▟+ +   +                                                                                          │
            │      -▝▌▝▌·▄·+▗█ ++++++▖++++ ++ + +                                                                              │
            │         -▐▞·▀▀▘▐···▄▀▚▐▌▟··▗▌▟ ▗ +▟++++   ++   +    +      +                                                     │
        0.36┤           ▘-- ·-▚▀▀·--▀▙▀▄▚▘▌█·█▗▞▝▖▗▌·++·▗▞▖++ +▗▌+·+ ++ ▖▖++▗+▟+++++  +    ++   +    +     ++      +           │
            │              -- - -  - ▜-- -▝▝▞▐▌- ▙▜▝▞▜▗▀▛▌▐▟▗▄▚▘▐···▞▀▚▞█▙▀▖█▗▜▗▙▀▚▌▗+▖▗▄·▗▌▞▖▖++▗▄▖+·++++▗▗+▗+▖▗▌+++▗+++++  +▖│
            │                            -   ▝▌ -▝----▘--  ▘▘ ---▀▀▞----▜▝-▝-▜▝▌█ -▝▌▜▚▌▝▞▟▝▌▜▚▄▞▌·▚·▖▗▀▀▄█▌▚▀▟▝▟▙▜·▞▘▚▗▗·▞▖▗▐▚│
        0.05┤                                                 -                 ▝--    -      --▘- -▀▝▘----▘- --▝▝-▜---▘▘▀-▝▘▀-│
            └┬───────────────────────────┬────────────────────────────┬───────────────────────────┬───────────────────────────┬┘
            1.0                        49.2                         97.5                        145.8                     194.0
        text saved in /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141/plots/tplot/loss_summary.txt
                                loss/mean hist                                                loss/max hist
           ┌───────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
        126┤█████                                                  │85.0┤█████                                                 │
        105┤█████                                                  │70.8┤█████                                                 │
         84┤█████                                                  │56.7┤███████████                                           │
         63┤█████                                                  │42.5┤███████████                                           │
         42┤█████ █████                                            │28.3┤████████████████                                      │
         21┤█████ ██████████                                       │14.2┤██████████████████████                                │
          0┤█████ ██████████ ██████████ ██████████ ██████████ █████│ 0.0┤██████████████████████████████████████████████████████│
           └┬─────────────┬────────────┬─────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
          0.04          0.51         0.98          1.45        1.91    0.06         0.55          1.03         1.51        2.00
                                  loss/min hist                                               loss/std hist
             ┌─────────────────────────────────────────────────────┐    ┌──────────────────────────────────────────────────────┐
        139.0┤█████                                                │44.0┤           █████                                      │
        115.8┤█████                                                │36.7┤           █████                                      │
         92.7┤█████                                                │29.3┤           ████████████████                           │
         69.5┤█████                                                │22.0┤     ███████████████████████████                      │
             │█████                                                │    │     █████████████████████████████████                │
         46.3┤███████████                                          │14.7┤     █████████████████████████████████                │
         23.2┤███████████                                          │ 7.3┤███████████████████████████████████████████           │
          0.0┤██████████████████████████ ██████████████████████████│ 0.0┤██████████████████████████████████████████████████████│
             └┬────────────┬────────────┬────────────┬────────────┬┘    └┬────────────┬─────────────┬────────────┬────────────┬┘
            -0.03        0.44         0.91         1.37        1.84    0.015        0.045         0.074        0.103      0.132
        text saved in /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141/plots/tplot/loss_hist.txt
        [2025-12-26 13:22:10,673046][I][ezpz/history:2433:finalize] Saving history report to /lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/outputs/ezpz.test_dist/2025-12-26-132141/report.md
        [2025-12-26 13:22:10,684947][I][ezpz/test_dist:348:finalize] dataset=<xarray.Dataset> Size: 39kB
        Dimensions:        (draw: 194)
        Coordinates:
          * draw           (draw) int64 2kB 0 1 2 3 4 5 6 ... 188 189 190 191 192 193
        Data variables: (12/25)
            iter           (draw) int64 2kB 6 7 8 9 10 11 12 ... 194 195 196 197 198 199
            loss           (draw) float32 776B 1.761 1.571 1.454 ... 0.2359 0.1281
            accuracy       (draw) float32 776B 0.6562 0.6953 0.6172 ... 0.9141 0.9688
            dtf            (draw) float64 2kB 0.01671 0.01655 ... 0.01678 0.01688
            dtb            (draw) float64 2kB 0.0007633 0.0007603 ... 0.0007723
            iter_mean      (draw) float64 2kB 6.0 7.0 8.0 9.0 ... 197.0 198.0 199.0
            ...             ...
            dtf_min        (draw) float64 2kB 0.0166 0.01655 0.01673 ... 0.01675 0.01678
            dtf_std        (draw) float64 2kB 0.0001951 0.0001735 ... 0.0002166
            dtb_mean       (draw) float64 2kB 0.0007558 0.0007502 ... 0.0007614
            dtb_max        (draw) float64 2kB 0.0008161 0.0007603 ... 0.0007723
            dtb_min        (draw) float64 2kB 0.0007143 0.0007425 ... 0.0007473
            dtb_std        (draw) float64 2kB 2.653e-05 5.994e-06 ... 7.243e-06
        [2025-12-26 13:22:11,411451][I][ezpz/test_dist:500:train] Took: 15.14 seconds to finish training
        [2025-12-26 13:22:11,412326][I][ezpz/test_dist:695:main] Took: 33.47 seconds
        wandb:
        wandb: 🚀 View run smart-breeze-6848 at:
        wandb: Find logs at: ../../../../../../../lus/eagle/projects/AuroraGPT/foremans/projects/saforem2/tmp/2025-12-26-130131/wandb/run-20251226_132148-01zkj7vc/logs
        [2025-12-26 13:22:14,556135][I][ezpz/launch:447:launch] ----[🍋 ezpz.launch][stop][2025-12-26-132214]----
        [2025-12-26 13:22:14,556823][I][ezpz/launch:448:launch] Execution finished with 0.
        [2025-12-26 13:22:14,557231][I][ezpz/launch:449:launch] Executing finished in 41.96 seconds.
        [2025-12-26 13:22:14,557601][I][ezpz/launch:450:launch] Took 41.96 seconds to run. Exiting.
        took: 1m 15s
        ```

        </details>

    </details>
