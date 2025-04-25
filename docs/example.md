# 📑 Simple Example

1. 🏖️ Setup environment[^magic] (see [Shell Environment](docs/shell-environment.md)):

    ```bash
    source <(curl https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh) && ezpz_setup_env
    ```

   [^magic]:
       This will 🪄 _automagically_ source
       [`ezpz/bin/utils.sh`](src/ezpz/bin/utils.sh)
       and (`&&`) call `ezpz_setup_env` to setup your
       python environment.

1. 🐍 Install `ezpz` (see [Python Library](docs/python-library.md)):

    ```bash
    python3 -m pip install "git+https://github.com/saforem2/ezpz"
    ```

1. 🚀 Launch _any_ `*.py`[^module] **_from_** python (see [Launch](docs/launch.md)):

    ```bash
    python3 -m ezpz.launch -m ezpz.test_dist
    ```

    - <details closed><summary>Output:</summary>

        ```bash
        (👻 2024-04-29)
        #[10:46:55 AM][x3013c0s31b1n0][/e/d/f/p/s/ezpz][🌱 dev][✓] [⏱️ 40s]
        ; python3 -m ezpz.launch -m ezpz.test_dist --profile | tee ezpz-test-dist-${NGPUS}-$(tstamp).log
        ```

        ```python
        [2025-04-24 10:47:00][I][ezpz/launch:88:__main__] Job ID: 4294227
        [2025-04-24 10:47:01][I][ezpz/launch:94:__main__] Node file: /var/spool/pbs/aux/4294227.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
        [2025-04-24 10:47:01][I][ezpz/launch:108:__main__] Building command to execute by piecing together:
                (1) ['launch_cmd'] + (2) ['python'] + (3) ['cmd_to_launch']

        1. ['launch_cmd']:
                mpiexec --verbose --envall -n 8 -ppn 4 --hostfile /var/spool/pbs/aux/4294227.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind depth -d 16

        2. ['python']:
                /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/venvs/2024-04-29/bin/python3

        3. ['cmd_to_launch']:
                -m ezpz.test_dist --profile

        [2025-04-24 10:47:01][I][ezpz/launch:125:__main__] Took: 0.34 seconds to build command.
        [2025-04-24 10:47:01][I][ezpz/launch:128:__main__] Evaluating:
                mpiexec --verbose --envall -n 8 -ppn 4 --hostfile /var/spool/pbs/aux/4294227.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind depth -d 16 /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/venvs/2024-04-29/bin/python3 -m ezpz.test_dist --profile
        Connected to tcp://x3013c0s31b1n0.hsn.cm.polaris.alcf.anl.gov:7919
        Launching application ad217fc7-7f70-4fa3-845e-7d2e5da426bc
        Using PMI port 35717,35718
        [2025-04-24 10:47:07][I][ezpz/dist:549] Using get_torch_device_type()='cuda' with backend='nccl'
        [2025-04-24 10:47:07][I][ezpz/dist:925] ['x3013c0s31b1n0'][2/7]
        [2025-04-24 10:47:08][I][ezpz/dist:925] ['x3013c0s37b0n0'][4/7]
        [2025-04-24 10:47:08][I][ezpz/dist:875] Using device='cuda' with backend='ddp' + 'nccl' for distributed training.
        [2025-04-24 10:47:08][I][ezpz/dist:925] ['x3013c0s31b1n0'][0/7]
        [2025-04-24 10:47:08][I][ezpz/dist:925] ['x3013c0s37b0n0'][5/7]
        [2025-04-24 10:47:08][I][ezpz/dist:925] ['x3013c0s37b0n0'][7/7]
        [2025-04-24 10:47:08][I][ezpz/dist:925] ['x3013c0s37b0n0'][6/7]
        [2025-04-24 10:47:08][I][ezpz/dist:925] ['x3013c0s31b1n0'][3/7]
        [2025-04-24 10:47:08][I][ezpz/dist:925] ['x3013c0s31b1n0'][1/7]
        [2025-04-24 10:47:09][I][ezpz/test_dist:397:__main__] model=
        Network(
          (layers): Sequential(
            (0): Linear(in_features=128, out_features=1024, bias=True)
            (1): Linear(in_features=1024, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=256, bias=True)
            (3): Linear(in_features=256, out_features=128, bias=True)
            (4): Linear(in_features=128, out_features=128, bias=True)
          )
        )
        [2025-04-24 10:47:11][I][ezpz/dist:1122] Setting up wandb from rank=0
        [2025-04-24 10:47:11][I][ezpz/dist:1123] Using=WB PROJECT=ezpz.test_dist
        wandb: Currently logged in as: foremans (aurora_gpt). Use `wandb login --relogin` to force relogin
        wandb: wandb version 0.19.10 is available!  To upgrade, please run:
        wandb:  $ pip install wandb --upgrade
        wandb: Tracking run with wandb version 0.16.6
        wandb: Run data is saved locally in /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20250424_104713-ad407z8x
        wandb: Run `wandb offline` to turn off syncing.
        wandb: Syncing run glamorous-shadow-1442
        wandb: ⭐️ View project at https://wandb.ai/aurora_gpt/ezpz.test_dist
        wandb: 🚀 View run at https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/ad407z8x
        [2025-04-24 10:47:15][I][ezpz/dist:1151] W&B RUN=[glamorous-shadow-1442](https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/ad407z8x)
        [2025-04-24 10:47:15][I][ezpz/dist:1191] Running on machine='Polaris'
        [2025-04-24 10:47:15][I][ezpz/test_dist:221:__main__] config:
        {
          "backend": "DDP",
          "batch_size": 64,
          "cp": 1,
          "dtype": "bfloat16",
          "input_size": 128,
          "layer_sizes": [
            1024,
            512,
            256,
            128
          ],
          "log_freq": 1,
          "output_size": 128,
          "pp": 1,
          "print_freq": 10,
          "pyinstrument_profiler": true,
          "tp": 1,
          "train_iters": 100,
          "warmup": 2
        }
        [2025-04-24 10:47:15][I][ezpz/test_dist:194:__main__] Warmup complete at step 2
        [2025-04-24 10:47:15][I][ezpz/test_dist:172:__main__] iter=10 loss=736.000000 dtf=0.000431 dtb=0.000855
        [2025-04-24 10:47:15][I][ezpz/test_dist:172:__main__] iter=20 loss=656.000000 dtf=0.000369 dtb=0.000895
        [2025-04-24 10:47:15][I][ezpz/test_dist:172:__main__] iter=30 loss=616.000000 dtf=0.000369 dtb=0.000842
        [2025-04-24 10:47:15][I][ezpz/test_dist:172:__main__] iter=40 loss=560.000000 dtf=0.000372 dtb=0.000825
        [2025-04-24 10:47:15][I][ezpz/test_dist:172:__main__] iter=50 loss=516.000000 dtf=0.000390 dtb=0.000839
        [2025-04-24 10:47:15][I][ezpz/test_dist:172:__main__] iter=60 loss=504.000000 dtf=0.000370 dtb=0.000873
        [2025-04-24 10:47:15][I][ezpz/test_dist:172:__main__] iter=70 loss=464.000000 dtf=0.000418 dtb=0.000880
        [2025-04-24 10:47:15][I][ezpz/test_dist:172:__main__] iter=80 loss=430.000000 dtf=0.000389 dtb=0.000825
        [2025-04-24 10:47:15][I][ezpz/test_dist:172:__main__] iter=90 loss=406.000000 dtf=0.000392 dtb=0.000845
        [2025-04-24 10:47:17][I][ezpz/history:721] Saving iter plot to: /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
        [2025-04-24 10:47:17][I][ezpz/history:721] Saving loss plot to: /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
        [2025-04-24 10:47:18][I][ezpz/history:721] Saving dtf plot to: /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
        [2025-04-24 10:47:18][I][ezpz/history:721] Saving dtb plot to: /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
        [2025-04-24 10:47:18][I][ezpz/history:618] Saving tplots to /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot
                            loss [2025-04-24-104718]
            ┌──────────────────────────────────────────────────────┐
        1560┤▌                                                     │
            │▌                                                     │
        1363┤▌                                                     │
            │▌                                                     │
            │▐                                                     │
        1166┤▐                                                     │
            │▝▖                                                    │
         969┤ ▌                                                    │
            │ ▐                                                    │
         772┤  ▚▖                                                  │
            │   ▝▀▄▄                                               │
            │       ▀▀▀▄▄▄▄▗▖▗                                     │
         575┤              ▘▝▘▀▀▀▄▞▄▄▄▄ ▄▗                         │
            │                          ▀ ▘▀▀▀▀▀▀▄▄▄▄▄▄ ▗           │
         378┤                                         ▀▘▀▀▀▀▀▀▄▄▄▄▄│
            └─┬─┬────┬───┬───┬──┬──┬──┬──┬──┬────┬──┬──┬──┬───┬──┬─┘
            0 2 6   15  22  30 35 41 46 52 57   67 72 77 82  90 96
        loss                          iter
        text saved in /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/loss.txt
                              dtf [2025-04-24-104718]
                ┌──────────────────────────────────────────────────┐
        0.000437┤                                 ▞▌               │
                │   ▟                             ▌▌               │
        0.000425┤  ▟█                             ▌▌               │
                │  ▛█                          ▖ ▟▌▌               │
                │▖▐ ▐                         ▐▙▌█▌▐               │
        0.000413┤▚▐ ▝▖                        ▐█▌▛▌▐               │
                │ █  ▌                        ▐▜▌▌ ▐              ▖│
        0.000401┤ █  ▐                        ▐ ▚▌ ▐   ▗▖        ▐▌│
                │ ▜   ▌        ▗       ▗▌▗▌   ▐ ▐▌ ▐▗▌ ▌▌        ▐▌│
        0.000389┤     ▌▖  ▗    █ ▟     ▐▌▌▙▙▌▗█ ▐▌ ▐▐▌▄▘▌ ▟  ▗▌ ▖▐▌│
                │     █▙▌ █    █▗█    ▖▐█ █▜▌▟█ ▐▌ ▐▐██ ▐ █  ▐▌▐▌▐▝│
                │     ██▌ ▌▌   █▌▐   ▐▌▐▜ ▝ ███ ▐▌ ▝▟██ ▝▟ ▀▀█▐▐▐▐ │
        0.000377┤     ██▌ ▌▚▖  ▛▌▐▗ ▗▜▚▐    █▝█ ▐▌  ▝▝█  ▜   █ █▐▐ │
                │     ▜█▚▞▌ ▚▖▗▌ ▝▌▚▜  ▘    ▝ ▜ ▐▌    █      ▝ █ ▀ │
        0.000365┤      ▝     ▚▘                  ▘    ▜        ▝   │
                └─┬─┬───┬──┬────┬────┬──┬──┬──┬──┬───┬────┬───┬──┬─┘
                0 2 6  15 20   30   41 46 52 57 63  72   82  90 96
        dtf                             iter
        text saved in /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtf.txt
                            dtf [2025-04-24-104718]
            ┌──────────────────────────────────────────────────────┐
        21.0┤█████                                                 │
            │█████                                                 │
        17.5┤███████████                                           │
            │████████████████                                      │
            │██████████████████████                                │
        14.0┤██████████████████████                                │
            │██████████████████████                                │
        10.5┤██████████████████████                                │
            │███████████████████████████                           │
         7.0┤███████████████████████████                           │
            │███████████████████████████                           │
            │███████████████████████████████████████████           │
         3.5┤███████████████████████████████████████████      █████│
            │██████████████████████████████████████████████████████│
         0.0┤██████████████████████████████████████████████████████│
            └┬────────────┬─────────────┬────────────┬────────────┬┘
          0.000362    0.000381      0.000401     0.000421  0.000440
        freq                           dtf
        text saved in /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtf-hist.txt
                              dtb [2025-04-24-104718]
                ┌──────────────────────────────────────────────────┐
        0.000941┤                                            ▟     │
                │▖                                           █     │
        0.000918┤▌               ▖   ▟                       █     │
                │▌    ▗   ▗    ▗▐▌   █                       █     │
                │▌ ▗▌ █ ▗ █    █▐▌   █            ▖    ▗     █     │
        0.000894┤▌ ▟▌ █▟█▐▐    █▐▌   █           ▐▌    █     █   ▗ │
                │▌▟█▙▌███▐▐    ▛▟▚  ▟█           ▐▌▟   █     █   █ │
        0.000870┤▜███▌███▐▐    ▌█▐  █▛▖  ▗▌ ▖ ▟  ▐▐█   █     █▗  █ │
                │ ▝▝█▌██▛▟▐    ▌█▐▞▜█▌▌  ▐▌▐▌▟▐  ▐▐█  ▞█    ▟██  █ │
        0.000846┤   ▝▌██▌█▐    ▌█ ▘▐█▌▌▖ ▌▌▐▝█▐  ▐▐▌▌▗▘▐▗▌▗▚▌▜█▟ ▛▄│
                │    ▌██▌█▐   ▗▘▝  ▐█▌█▐ ▌▚▞ █▐  ▐▐▌▚▞ ▐▐▚▐▐▌ ▀▛▄▌ │
                │    ▌██▌█▐  ▟▐    ▐▌▘▝ █ ▐▌ █▐▗▌▞▐▌▐▌ ▐▐▝▟ ▘    ▘ │
        0.000823┤    ▚██▌▝ ▌▞▜▐     ▘   ▝  ▘ ▜▝▟█  ▘ ▘ ▝▞ ▝        │
                │    ▐██▌  ▝  ▘                █▝                  │
        0.000799┤     ▀▌▘                      ▜                   │
                └─┬─┬───┬──┬────┬────┬──┬──┬──┬──┬───┬────┬───┬──┬─┘
                0 2 6  15 20   30   41 46 52 57 63  72   82  90 96
        dtb                             iter
        text saved in /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtb.txt
                            dtb [2025-04-24-104718]
            ┌──────────────────────────────────────────────────────┐
        20.0┤     ██████                                           │
            │     ██████     ██████                                │
        16.7┤     ██████     ██████                                │
            │     █████████████████                                │
            │     ██████████████████████                           │
        13.3┤     ██████████████████████                           │
            │     ██████████████████████                           │
        10.0┤     ██████████████████████                           │
            │     ███████████████████████████                      │
         6.7┤     ███████████████████████████                      │
            │     ██████████████████████████████████████           │
            │███████████████████████████████████████████           │
         3.3┤███████████████████████████████████████████           │
            │██████████████████████████████████████████████████████│
         0.0┤██████████████████████████████████████████████████████│
            └┬────────────┬─────────────┬────────────┬────────────┬┘
          0.000793    0.000831      0.000870     0.000909  0.000948
        freq                           dtb
        text saved in /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtb-hist.txt
        [2025-04-24 10:47:19][I][ezpz/utils:198] Saving dataset to: /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/train_dataset.h5
        [2025-04-24 10:47:19][I][ezpz/test_dist:188:__main__] dataset=<xarray.Dataset> Size: 3kB
        Dimensions:  (draw: 97)
        Coordinates:
          * draw     (draw) int64 776B 0 1 2 3 4 5 6 7 8 ... 88 89 90 91 92 93 94 95 96
        Data variables:
            iter     (draw) int64 776B 3 4 5 6 7 8 9 10 11 ... 92 93 94 95 96 97 98 99
            loss     (draw) float32 388B 1.56e+03 1.216e+03 1.04e+03 ... 384.0 378.0
            dtf      (draw) float64 776B 0.0004138 0.0004107 ... 0.0004055 0.0003885
            dtb      (draw) float64 776B 0.0009289 0.0008678 ... 0.0008463 0.0008494

          _     ._   __/__   _ _  _  _ _/_   Recorded: 10:47:07  Samples:  5977
        /_//_/// /_\ / //_// / //_'/ //     Duration: 11.733    CPU time: 8.387
        /   _/                      v5.0.0

        Profile at /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/src/ezpz/profile.py:107

        11.732 main  ezpz/test_dist.py:447
        └─ 11.732 train  ezpz/test_dist.py:204
          ├─ 3.969 Trainer.train  ezpz/test_dist.py:191
          │  ├─ 3.529 Trainer.finalize  ezpz/test_dist.py:175
          │  │  ├─ 2.431 History.finalize  ezpz/history.py:835
          │  │  │  └─ 2.305 History.plot_all  ezpz/history.py:640
          │  │  │     ├─ 1.370 savefig  matplotlib/pyplot.py:1129
          │  │  │     │     [37 frames hidden]  matplotlib, PIL, <built-in>
          │  │  │     └─ 0.859 <module>  seaborn/__init__.py:1
          │  │  │           [11 frames hidden]  seaborn, scipy, importlib, wandb
          │  │  ├─ 0.575 <module>  ambivalent/__init__.py:1
          │  │  │     [6 frames hidden]  ambivalent, IPython, prompt_toolkit
          │  │  │        0.220 <module>  IPython/core/completer.py:1
          │  │  │        └─ 0.208 <module>  jedi/__init__.py:1
          │  │  │           └─ 0.196 <module>  jedi/api/__init__.py:1
          │  │  └─ 0.342 <module>  matplotlib/pyplot.py:1
          │  │        [3 frames hidden]  matplotlib
          │  └─ 0.439 Trainer.train_step  ezpz/test_dist.py:159
          │     ├─ 0.174 Trainer._forward_step  ezpz/test_dist.py:138
          │     │  └─ 0.136 DistributedDataParallel._wrapped_call_impl  torch/nn/modules/module.py:1528
          │     │     └─ 0.136 DistributedDataParallel._call_impl  torch/nn/modules/module.py:1534
          │     └─ 0.158 Trainer._backward_step  ezpz/test_dist.py:149
          ├─ 3.375 Trainer.__init__  <string>:2
          │  └─ 3.375 Trainer.__post_init__  ezpz/test_dist.py:108
          │     └─ 3.347 setup_wandb  ezpz/dist.py:1061
          │        └─ 3.284 init  wandb/sdk/wandb_init.py:935
          │              [50 frames hidden]  wandb, <built-in>, queue, threading, ...
          ├─ 2.243 setup_torch  ezpz/dist.py:783
          │  ├─ 1.189 wrapper  torch/distributed/c10d_logger.py:72
          │  │     [2 frames hidden]  torch, <built-in>
          │  └─ 1.041 setup_torch_distributed  ezpz/dist.py:699
          │     └─ 1.010 setup_torch_DDP  ezpz/dist.py:648
          │        └─ 1.010 init_process_group  ezpz/dist.py:542
          │           └─ 1.004 wrapper  torch/distributed/c10d_logger.py:72
          │                 [4 frames hidden]  torch
          └─ 2.135 build_model_and_optimizer  ezpz/test_dist.py:387
              └─ 2.126 Adam.__init__  torch/optim/adam.py:15
                    [45 frames hidden]  torch, transformers, huggingface_hub,...


        [2025-04-24 10:47:19][I][ezpz/profile:121] Saving pyinstrument profile output to: /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz/pyinstrument_profiles/ezpz_pyinstrument_profiles
        [2025-04-24 10:47:19][I][ezpz/profile:129] PyInstrument profile saved (as html) to:  /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz/pyinstrument_profiles/ezpz_pyinstrument_profiles/pyinstrument-profile-2025-04-24-104719.html
        [2025-04-24 10:47:19][I][ezpz/profile:137] PyInstrument profile saved (as text) to:  /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz/pyinstrument_profiles/ezpz_pyinstrument_profiles/pyinstrument-profile-2025-04-24-104719.txt
        [2025-04-24 10:47:21][I][ezpz/profile:149] Finished with pyinstrument profiler. Took: 11.73259s
        [2025-04-24 10:47:21][I][ezpz/test_dist:461:__main__] Took: 13.77 seconds
        wandb: \ 0.008 MB of 0.093 MB uploaded
        wandb: Run history:
        wandb:  dtb █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
        wandb:  dtf █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
        wandb: iter ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
        wandb: loss █▄▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
        wandb:
        wandb: Run summary:
        wandb:  dtb 0.00085
        wandb:  dtf 0.00039
        wandb: iter 99
        wandb: loss 378.0
        wandb:
        wandb: 🚀 View run glamorous-shadow-1442 at: https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/ad407z8x
        wandb: ⭐️ View project at: https://wandb.ai/aurora_gpt/ezpz.test_dist
        wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
        wandb: Find logs at: ./wandb/run-20250424_104713-ad407z8x/logs
        Application ad217fc7 resources: utime=84s stime=107s maxrss=1769176KB inblock=832 oublock=17256 minflt=1530712 majflt=872 nvcsw=294227 nivcsw=737430
        [2025-04-24 10:47:28][I][ezpz/launch:132:__main__] Command took 27.68 seconds to run.

        real	34.46s
        user	2.80s
        sys	7.60s

        real	33.89s
        user	0.00s
        sys	0.01s
        Time: 0h:00m:35s
        ```

  </details>

   [^module]:
       Technically, we're _launching_ (`-m ezpz.launch`) the
       [`ezpz/test_dist.py`](src/ezpz/test_dist.py) as a module (`-m`),
       in this example.

😎 2 ez.
