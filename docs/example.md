# 📑 Simple Example

1. 🏖️ Setup environment[^magic] (see [Shell Environment](shell-environment.md)):

     ```bash
     source <(curl https://bit.ly/ezpz-utils) && ezpz_setup_env
     ```

1. 🐍 Install `ezpz` (see [Python Library](python-library.md)):

     ```bash
     python3 -m pip install "git+https://github.com/saforem2/ezpz"
     ```

1. 🚀 Launch _any_ `*.py`[^module] **_from_** python (see [Launch](launch.md)):

     ```bash
     python3 -m ezpz.test
     ```

    - <details closed><summary>Output:</summary>

        ```bash
        #[🐍 aurora_nre_models_frameworks-2025.0.0](👻 aurora_nre_models_frameworks-2025.0.0)
        #[05/01/25 @ 10:07:09][x4206c4s1b0n0][/f/d/f/p/s/ezpz][🌱 main][📦📝🤷✓]
        ; python3 -m ezpz.test
        [W501 10:07:15.372342214 OperatorEntry.cpp:155] Warning: Warning only once for all operators,  other operators may also be overridden.
        Overriding a previously registered kernel for the same operator and the same dispatch key
        operator: aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
            registered at /build/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
        dispatch key: XPU
        previous kernel: registered at /build/pytorch/build/aten/src/ATen/RegisterCPU.cpp:30476
                new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:2971 (function operator())
        [2025-05-01 10:07:20,655] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to xpu (auto detect)
        [2025-05-01 10:07:23][I][ezpz/launch:95] Job ID: 4575165
        [2025-05-01 10:07:23][I][ezpz/launch:101] Node file: /var/spool/pbs/aux/4575165.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
        [2025-05-01 10:07:23][I][ezpz/launch:116] Building command to execute by piecing together:
                (1) ['launch_cmd'] + (2) ['python'] + (3) ['cmd_to_launch']

        1. ['launch_cmd']:
                mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/4575165.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind=depth --depth=8

        2. ['python']:
                /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/bin/python3

        3. ['cmd_to_launch']:
                -m ezpz.test_dist

        [2025-05-01 10:07:23][I][ezpz/launch:134] Took: 0.62 seconds to build command.
        [2025-05-01 10:07:23][I][ezpz/launch:137] Evaluating:
                mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/4575165.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind=depth --depth=8 /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/bin/python3 -m ezpz.test_dist
        [2025-05-01 10:07:23][I][ezpz/launch:159] Filtering for Aurora-specific messages. To view list of filters, run with `EZPZ_LOG_LEVEL=DEBUG`
        Disabling local launch: multi-node application
        Connected to tcp://x4206c4s2b0n0.hostmgmt2206.cm.aurora.alcf.anl.gov:7919
        Launching application 0010057d-0cb6-455d-94ae-505529c389cd
        [2025-05-01 10:07:36][I][ezpz/dist:554] Using get_torch_device_type()='xpu' with backend='ccl'
        [2025-05-01 10:07:36][I][ezpz/dist:987] ['x4206c4s2b0n0'][10/23]
        [2025-05-01 10:07:36][I][ezpz/dist:987] ['x4206c4s2b0n0'][11/23]
        [2025-05-01 10:07:36][I][ezpz/dist:987] ['x4206c4s2b0n0'][ 6/23]
        [2025-05-01 10:07:36][I][ezpz/dist:987] ['x4206c4s2b0n0'][ 7/23]
        [2025-05-01 10:07:36][I][ezpz/dist:987] ['x4206c4s2b0n0'][ 3/23]
        [2025-05-01 10:07:36][I][ezpz/dist:987] ['x4206c4s2b0n0'][ 8/23]
        [2025-05-01 10:07:36][I][ezpz/dist:987] ['x4206c4s2b0n0'][ 5/23]
        [2025-05-01 10:07:37][I][ezpz/dist:987] ['x4206c4s2b0n0'][ 9/23]
        [2025-05-01 10:07:37][I][ezpz/dist:987] ['x4206c4s2b0n0'][ 1/23]
        [2025-05-01 10:07:37][I][ezpz/dist:987] ['x4206c4s2b0n0'][ 2/23]
        [2025-05-01 10:07:37][I][ezpz/dist:987] ['x4206c4s2b0n0'][ 4/23]
        [2025-05-01 10:07:38][I][ezpz/dist:987] ['x4206c4s1b0n0'][12/23]
        [2025-05-01 10:07:38][I][ezpz/dist:987] ['x4206c4s1b0n0'][16/23]
        [2025-05-01 10:07:38][I][ezpz/dist:987] ['x4206c4s1b0n0'][15/23]
        [2025-05-01 10:07:38][I][ezpz/dist:987] ['x4206c4s1b0n0'][13/23]
        [2025-05-01 10:07:38][I][ezpz/dist:987] ['x4206c4s1b0n0'][14/23]
        [2025-05-01 10:07:38][I][ezpz/dist:987] ['x4206c4s1b0n0'][20/23]
        [2025-05-01 10:07:38][I][ezpz/dist:987] ['x4206c4s1b0n0'][21/23]
        [2025-05-01 10:07:38][I][ezpz/dist:987] ['x4206c4s1b0n0'][23/23]
        [2025-05-01 10:07:38][I][ezpz/dist:987] ['x4206c4s1b0n0'][22/23]
        [2025-05-01 10:07:38][I][ezpz/dist:987] ['x4206c4s1b0n0'][17/23]
        [2025-05-01 10:07:38][I][ezpz/dist:987] ['x4206c4s1b0n0'][18/23]
        [2025-05-01 10:07:38][I][ezpz/dist:987] ['x4206c4s1b0n0'][19/23]
        [2025-05-01 10:07:38][I][ezpz/dist:936] Using device='xpu' with backend='ddp' + 'ccl' for distributed training.
        [2025-05-01 10:07:38][I][ezpz/dist:987] ['x4206c4s2b0n0'][ 0/23]
        2025:05:01-10:07:38:(49751) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
        [2025-05-01 10:07:39][I][ezpz/test_dist:398:__main__] model=
        Network(
        (layers): Sequential(
            (0): Linear(in_features=128, out_features=1024, bias=True)
            (1): Linear(in_features=1024, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=256, bias=True)
            (3): Linear(in_features=256, out_features=128, bias=True)
            (4): Linear(in_features=128, out_features=128, bias=True)
        )
        )
        [2025-05-01 10:07:50][I][ezpz/dist:1185] Setting up wandb from rank=0
        [2025-05-01 10:07:50][I][ezpz/dist:1186] Using=WB PROJECT=ezpz.test_dist
        wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
        wandb: Tracking run with wandb version 0.19.10
        wandb: Run data is saved locally in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20250501_100750-53eys83m
        wandb: Run `wandb offline` to turn off syncing.
        wandb: Syncing run quiet-frog-1566
        wandb: ⭐️ View project at https://wandb.ai/aurora_gpt/ezpz.test_dist
        wandb: 🚀 View run at https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/53eys83m
        [2025-05-01 10:07:51][I][ezpz/dist:1214] W&B RUN=[quiet-frog-1566](https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/53eys83m)
        [2025-05-01 10:07:51][I][ezpz/dist:1254] Running on machine='Aurora'
        [2025-05-01 10:07:51][I][ezpz/test_dist:221:__main__] config:
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
        "pyinstrument_profiler": false,
        "tp": 1,
        "train_iters": 100,
        "warmup": 2
        }
        [2025-05-01 10:07:51][I][ezpz/test_dist:194:__main__] Warmup complete at step 2
        [2025-05-01 10:07:51][I][ezpz/test_dist:172:__main__] iter=10 loss=736.000000 dtf=0.000657 dtb=0.001384
        [2025-05-01 10:07:51][I][ezpz/test_dist:172:__main__] iter=20 loss=676.000000 dtf=0.000563 dtb=0.001285
        [2025-05-01 10:07:51][I][ezpz/test_dist:172:__main__] iter=30 loss=604.000000 dtf=0.000551 dtb=0.001301
        [2025-05-01 10:07:51][I][ezpz/test_dist:172:__main__] iter=40 loss=564.000000 dtf=0.000564 dtb=0.001276
        [2025-05-01 10:07:51][I][ezpz/test_dist:172:__main__] iter=50 loss=520.000000 dtf=0.000564 dtb=0.001240
        [2025-05-01 10:07:51][I][ezpz/test_dist:172:__main__] iter=60 loss=496.000000 dtf=0.000557 dtb=0.001272
        [2025-05-01 10:07:52][I][ezpz/test_dist:172:__main__] iter=70 loss=466.000000 dtf=0.000548 dtb=0.001269
        [2025-05-01 10:07:52][I][ezpz/test_dist:172:__main__] iter=80 loss=432.000000 dtf=0.000550 dtb=0.001254
        [2025-05-01 10:07:52][I][ezpz/test_dist:172:__main__] iter=90 loss=410.000000 dtf=0.000523 dtb=0.001193
        [2025-05-01 10:07:53][I][ezpz/history:721] Saving iter plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
        [2025-05-01 10:07:53][I][ezpz/history:721] Saving loss plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
        [2025-05-01 10:07:54][I][ezpz/history:721] Saving dtf plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
        [2025-05-01 10:07:54][I][ezpz/history:721] Saving dtb plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
        [2025-05-01 10:07:54][I][ezpz/history:618] Saving tplots to /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot
                            loss [2025-05-01-100754]
            ┌──────────────────────────────────────────────────────┐
        1528┤▌                                                     │
            │▌                                                     │
        1337┤▌                                                     │
            │▚                                                     │
            │▐                                                     │
        1146┤▐                                                     │
            │ ▌                                                    │
        955┤ ▌                                                    │
            │ ▐                                                    │
        764┤  ▚▖                                                  │
            │   ▝▀▄▄▖▗▖                                            │
            │       ▝▘▝▚▚▄▄▄▄                                      │
        573┤                ▀▀▀▀▀▀▄▀▄▄ ▄                          │
            │                          ▀ ▀▀▀▀▀▀▚▄▄▄▄▄▄▖▗           │
        382┤                                         ▝▘▀▀▀▀▀▀▄▄▄▚▄│
            └┬──┬──┬──┬──┬────┬──┬──┬──┬───┬───┬───┬──┬────┬──┬──┬─┘
            1  7 12 17 22   31 38 42 48  55  62  69 76   85 89 96
        loss                          iter
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/loss.txt
                                dtf [2025-05-01-100754]
                ┌──────────────────────────────────────────────────┐
        0.000754┤   ▗▌                                             │
                │   ▐▌                                             │
        0.000716┤   ▐▌                                             │
                │▌  ▐▌                                             │
                │▌  ▐▌              ▟                              │
        0.000677┤▐  ▐▚              █                              │
                │▐▟ ▞▐         ▗    █          ▖                   │
        0.000639┤ ▘█ ▐▞▌  ▞▌ ▗▚█    █    ▗▜ ▖ ▐▌ ▖ ▗▌   ▗▌         │
                │  █ ▐▌▌  ▌▐ ▐▐█    ▛▞▜  ▐ █▙▀█▌▐▌ ▐▙▚ ▖▐▝▀▖  ▗    │
        0.000600┤  █  ▘▌  ▌▝▄▐▐█    ▌ ▐  ▐ ▝▜ █▌▐▌ ▐█▐▐▌▐  ▌  █    │
                │  █   ▚  ▌ ▐▐▐█    ▌ ▝▖ ▌    █▌▐▌ ▐█ ▀▌▐  ▌  █    │
                │  ▜   ▐ ▗▌ ▐▐▐▛▄▖  ▌  ▌ ▌    █▌▟▌▄▐█  ▌▐  ▌  █    │
        0.000562┤       ▀▘▘ ▝▌▝▌ ▐▞▀▘  ▚▞▘    ▜▝█▝▝▟▜  ▚▐  ▌  ▌▚▖▗▖│
                │                ▝▌    ▝▌       ▝  ▝    ▀  ▚▖▗▌ ▐▌▌│
        0.000523┤                                           ▝▀▌  ▘▝│
                └┬──┬────┬──┬───┬──┬──┬──┬───┬──┬───┬──┬────┬────┬─┘
                1  7   17 22  31 36 42 48  55 62  69 76   85   96
        dtf                             iter
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtf.txt
                            dtf [2025-05-01-100754]
            ┌──────────────────────────────────────────────────────┐
        27.0┤     ██████                                           │
            │     ██████                                           │
        22.5┤     ██████                                           │
            │     ██████                                           │
            │     ██████           █████                           │
        18.0┤     ██████           █████                           │
            │     ██████           ██████████                      │
        13.5┤     ██████           ██████████                      │
            │████████████████      ██████████                      │
        9.0┤████████████████      ██████████                      │
            │████████████████████████████████                      │
            │████████████████████████████████                      │
        4.5┤████████████████████████████████                      │
            │██████████████████████████████████████████████████████│
        0.0┤██████████████████████████████████████████████████████│
            └┬────────────┬─────────────┬────────────┬────────────┬┘
        0.000513    0.000576      0.000639     0.000702  0.000765
        freq                           dtf
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtf-hist.txt
                            dtb [2025-05-01-100754]
                ┌───────────────────────────────────────────────────┐
        0.00193┤   ▗▌                                              │
                │   ▐▌                                              │
        0.00181┤   ▐▌                                              │
                │   ▐▌                                              │
                │   ▐▌                                              │
        0.00168┤   ▐▌                                              │
                │   ▐▌                       ▖                      │
        0.00156┤   ▐▌                      ▐▌                      │
                │   ▐▝▖      ▟       ▖ ▖    ▐▌        ▗             │
        0.00143┤▟  ▐ ▝▄  ▗▗ █   ▖  ▐▌▐▌    ▐▌        ▛▖▗▌          │
                │ ▜▗▟  ▐  ▌▘▌▛▄ ▐▚  ▐▝▞▌  ▗▚▟▝▜       ▌▚▐▌▗▌        │
                │ ▐▌   ▐  ▌ ▚▌▐ ▐▐ ▗▐  ▌  ▐   ▐  ▗▚  ▐  █▌▌▐        │
        0.00131┤  ▘   ▝▄▗▌ ▐▌ ▚▟ ▌█▐  ▌  ▞   ▐▗ ▟▐  ▞  ▜▌▌▐        │
                │        ▘   ▘    ▝▌▘  ▝▀▜     ▀▞  ▀▀    ▝ ▝▖  ▞▄▄▞▚│
        0.00119┤                                           ▚▄▄▌ ▝▌ │
                └┬──┬────┬──┬────┬────┬──┬───┬───┬──┬──┬──┬──┬────┬─┘
                1  7   17 22   31   42 48  55  62 69 74 80 85   96
        dtb                            iter
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtb.txt
                            dtb [2025-05-01-100754]
            ┌──────────────────────────────────────────────────────┐
        31.0┤     ██████                                           │
            │     ██████                                           │
        25.8┤     ██████                                           │
            │     ██████                                           │
            │     ██████                                           │
        20.7┤████████████████                                      │
            │██████████████████████                                │
        15.5┤██████████████████████                                │
            │██████████████████████                                │
        10.3┤██████████████████████                                │
            │██████████████████████                                │
            │██████████████████████                                │
        5.2┤███████████████████████████                           │
            │███████████████████████████                           │
        0.0┤████████████████████████████████                 █████│
            └┬────────────┬─────────────┬────────────┬────────────┬┘
        0.00115      0.00135       0.00156      0.00176   0.00196
        freq                           dtb
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtb-hist.txt
        [2025-05-01 10:07:54][I][ezpz/test_dist:188:__main__] dataset=<xarray.Dataset> Size: 3kB
        Dimensions:  (draw: 97)
        Coordinates:
        * draw     (draw) int64 776B 0 1 2 3 4 5 6 7 8 ... 88 89 90 91 92 93 94 95 96
        Data variables:
            iter     (draw) int64 776B 3 4 5 6 7 8 9 10 11 ... 92 93 94 95 96 97 98 99
            loss     (draw) float32 388B 1.528e+03 1.248e+03 1.072e+03 ... 382.0 392.0
            dtf      (draw) float64 776B 0.0007091 0.0006719 ... 0.0005526 0.0005336
            dtb      (draw) float64 776B 0.001446 0.00146 0.001422 ... 0.001251 0.001238
        [2025-05-01 10:07:54][I][ezpz/test_dist:467:__main__] Took: 18.05 seconds
        wandb:
        wandb: 🚀 View run quiet-frog-1566 at: https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/53eys83m
        wandb: Find logs at: ../../../../../../lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20250501_100750-53eys83m/logs
        Application 0010057d resources: utime=874s stime=172s maxrss=3840744KB inblock=378318 oublock=1080 minflt=10297842 majflt=32240 nvcsw=292681 nivcsw=1232922
        [2025-05-01 10:07:57][I][ezpz/launch:170] Command took 34.03 seconds to run.
        took: 0h:00m:48s
        ```

    </details>

[^module]:
    Technically, we're _launching_ (`-m ezpz.launch`) the
    [`ezpz/test_dist.py`](https://github.com/saforem2/ezpz/blob/main/ezpz/test_dist.py) a module (`-m`),
    in this example.

[^magic]:
    This will 🪄 _automagically_ source
    [`ezpz/bin/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh),
    and (`&&`) call `ezpz_setup_env` to setup your
    python environment.

😎 2 ez.
