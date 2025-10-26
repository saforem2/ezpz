# 🍋 `ezpz`

> Write _once_, run _anywhere_

Train across **all** your {NVIDIA, AMD, Intel, MPS, ...} accelerators, `ezpz` 🍋.

See [🍋 `ezpz` docs](https://saforem2.github.io/ezpz) for additional information.

Refer to the [Repository Guidelines](AGENTS.md) before contributing.

## 🐣 Getting Started

1. 🏖️ **Setup** environment[^magic] (see [**Shell Environment**](https://saforem2.github.io/ezpz/shell-environment/)):

    ```bash
    source <(curl -L https://bit.ly/ezpz-utils) && ezpz_setup_env
    ```

    > Prefer not to execute remote shell scripts? Create a local virtual
    > environment and install dependencies directly:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e .[dev]
    ```

   [^magic]:
       This will 🪄 _automagically_ source
       [`ezpz/bin/utils.sh`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/bin/utils.sh)
       and (`&&`) call `ezpz_setup_env` to setup your
       python environment.

1. 🐍 **Install** `ezpz` (see [**Python API**](https://saforem2.github.io/ezpz/Code-Reference/)):

    ```bash
    python3 -m pip install "git+https://github.com/saforem2/ezpz"
    ```

    > Optional extras: `pip install ezpz[monitoring]` enables Weights & Biases
    > logging, `pip install ezpz[profiling]` adds pyinstrument, and
    > `pip install ezpz[terminal]` pulls in plotext for CLI charts.

1. 🩺 **Diagnose** your environment (see [**Doctor**](https://saforem2.github.io/ezpz/doctor/)):

    ```bash
    ezpz doctor
    ```

    > Need machine-readable output? Append `--json` to integrate with your CI.

1. 🚀 **Launch** python  **_from_** python using `ezpz launch` (see [**Launch**](https://saforem2.github.io/ezpz/launch/)).

    ```bash
    # arbitrary python string, for example
    ezpz launch -c "'import ezpz; ezpz.setup_torch()'"
    ```

    <details closed><summary>Examples, launching:</summary>

    - _Any_ `*.py` module ([`ezpz/test_dist.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py), in this example):

        ```bash
        ezpz launch -m ezpz.test_dist
        ```

        <details closed><summary>Output:</summary>

        ```bash
        #[🐍 aurora_nre_models_frameworks-2025.0.0](👻 aurora_nre_models_frameworks-2025.0.0)
        #[/f/d/f/p/s/ezpz][🌱 saforem2/dev][📦🤷✓] [⏱️ 49s]
        #[06/02/25 @ 08:34:27][x4404c4s4b0n0]
        ; WANDB_MODE=offline ezpz launch -m ezpz.test_dist --warmup=10 --layer-sizes='256,512,1024,2048,4096,2048,1024,512,256' --dtype=bf16 --train-iters=5000 --print-freq=100 --log-freq=10
        [W602 08:39:04.786863061 OperatorEntry.cpp:155] Warning: Warning only once for all operators,  other operators may also be overridden.
        Overriding a previously registered kernel for the same operator and the same dispatch key
        operator: aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
            registered at /build/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
        dispatch key: XPU
        previous kernel: registered at /build/pytorch/build/aten/src/ATen/RegisterCPU.cpp:30476
            new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:2971 (function operator())
        [2025-06-02 08:39:11,507270][I][ezpz/__init__:278:ezpz] Setting logging level to 'INFO' on 'RANK == 0'
        [2025-06-02 08:39:11,510558][I][ezpz/__init__:279:ezpz] Setting logging level to 'CRITICAL' on all others 'RANK != 0'
        [2025-06-02 08:39:11,646885][I][ezpz/launch:157] Job ID: 5414072
        [2025-06-02 08:39:11,956377][I][ezpz/launch:163] Node file: /var/spool/pbs/aux/5414072.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
        [2025-06-02 08:39:11,961307][I][ezpz/launch:178] Building command to execute by piecing together:(1.) ['launch_cmd'] + (2.) ['python'] + (3.) ['cmd_to_launch']
        [2025-06-02 08:39:11,962039][I][ezpz/launch:182] (1.) ['launch_cmd']: mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/5414072.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind=depth --depth=8
        [2025-06-02 08:39:11,962616][I][ezpz/launch:183] (2.) ['python']: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/bin/python3
        [2025-06-02 08:39:11,963015][I][ezpz/launch:184] (3.) ['cmd_to_launch']:  -m ezpz.test_dist
        [2025-06-02 08:39:11,963622][I][ezpz/launch:189] Took: 0.45 seconds to build command.
        [2025-06-02 08:39:11,963985][I][ezpz/launch:192] Executing: mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/5414072.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind=depth --depth=8 /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/bin/python3 -m ezpz.test_dist
        [2025-06-02 08:39:11,964786][I][ezpz/launch:119] Filtering for Aurora-specific messages. To view list of filters, run with `EZPZ_LOG_LEVEL=DEBUG`
        [2025-06-02 08:39:11,965257][I][ezpz/launch:199] Execution started @ 2025-06-02-083911...

        Disabling local launch: multi-node application
        Connected to tcp://x4404c4s4b0n0.hostmgmt2404.cm.aurora.alcf.anl.gov:7919
        Launching application 09a72a12-de4b-461f-bd7d-d7990dbee665
        [2025-06-02 08:39:25,068320][I][ezpz/__init__:278:ezpz] Setting logging level to 'INFO' on 'RANK == 0'
        [2025-06-02 08:39:25,070671][I][ezpz/__init__:279:ezpz] Setting logging level to 'CRITICAL' on all others 'RANK != 0'
        [2025-06-02 08:39:25,075236][I][ezpz/dist:760] Using get_torch_device_type()='xpu' with be='ddp'
        [2025-06-02 08:39:25,076000][I][ezpz/dist:573] Initializing process group with rank=0, world_size=24, torch_backend=ccl
        2025:06:02-08:39:26:(23179) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
        [2025-06-02 08:39:26,728835][I][ezpz/dist:964] Using device='xpu' with backend='ddp' + 'ccl' for distributed training.
        [2025-06-02 08:39:26,729616][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 0/23]
        [2025-06-02 08:39:26,728822][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 3/23]
        [2025-06-02 08:39:26,728839][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 1/23]
        [2025-06-02 08:39:26,728828][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 2/23]
        [2025-06-02 08:39:26,728834][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 4/23]
        [2025-06-02 08:39:26,728826][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 5/23]
        [2025-06-02 08:39:26,728821][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 7/23]
        [2025-06-02 08:39:26,728814][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 8/23]
        [2025-06-02 08:39:26,728819][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 9/23]
        [2025-06-02 08:39:26,728816][I][ezpz/dist:1011] ['x4404c4s4b0n0'][10/23]
        [2025-06-02 08:39:26,728815][I][ezpz/dist:1011] ['x4404c4s4b0n0'][11/23]
        [2025-06-02 08:39:26,728883][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 6/23]
        [2025-06-02 08:39:26,728812][I][ezpz/dist:1011] ['x4404c4s6b0n0'][18/23]
        [2025-06-02 08:39:26,728815][I][ezpz/dist:1011] ['x4404c4s6b0n0'][22/23]
        [2025-06-02 08:39:26,728829][I][ezpz/dist:1011] ['x4404c4s6b0n0'][12/23]
        [2025-06-02 08:39:26,728827][I][ezpz/dist:1011] ['x4404c4s6b0n0'][13/23]
        [2025-06-02 08:39:26,728827][I][ezpz/dist:1011] ['x4404c4s6b0n0'][14/23]
        [2025-06-02 08:39:26,728833][I][ezpz/dist:1011] ['x4404c4s6b0n0'][15/23]
        [2025-06-02 08:39:26,728831][I][ezpz/dist:1011] ['x4404c4s6b0n0'][16/23]
        [2025-06-02 08:39:26,728827][I][ezpz/dist:1011] ['x4404c4s6b0n0'][17/23]
        [2025-06-02 08:39:26,728812][I][ezpz/dist:1011] ['x4404c4s6b0n0'][19/23]
        [2025-06-02 08:39:26,728811][I][ezpz/dist:1011] ['x4404c4s6b0n0'][20/23]
        [2025-06-02 08:39:26,731907][I][ezpz/test_dist:468:__main__] Took: 1.66 seconds to setup torch
        [2025-06-02 08:39:26,728812][I][ezpz/dist:1011] ['x4404c4s6b0n0'][21/23]
        [2025-06-02 08:39:26,728813][I][ezpz/dist:1011] ['x4404c4s6b0n0'][23/23]
        [2025-06-02 08:39:26,748088][I][ezpz/test_dist:218:__main__] Model size: 837632 parameters
        [2025-06-02 08:39:26,750571][I][ezpz/test_dist:220:__main__]
        =================================================================
        Layer (type:depth-idx)                   Param #
        =================================================================
        SequentialLinearNet                      --
        ├─Sequential: 1-1                        837,632
        =================================================================
        Total params: 837,632
        Trainable params: 837,632
        Non-trainable params: 0
        =================================================================
        [2025-06-02 08:39:26,751974][I][ezpz/test_dist:226:__main__] Took: 0.011442308983532712 seconds to build model
        [2025-06-02 08:39:26,756362][I][ezpz/test_dist:406:__main__] model=
        SequentialLinearNet(
        (layers): Sequential(
            (0): Linear(in_features=128, out_features=1024, bias=True)
            (1): ReLU()
            (2): Linear(in_features=1024, out_features=512, bias=True)
            (3): ReLU()
            (4): Linear(in_features=512, out_features=256, bias=True)
            (5): ReLU()
            (6): Linear(in_features=256, out_features=128, bias=True)
            (7): ReLU()
            (8): Linear(in_features=128, out_features=128, bias=True)
        )
        )
        [2025-06-02 08:39:37,687236][I][ezpz/test_dist:230:__main__] Took: 10.94 seconds to build optimizer
        [2025-06-02 08:39:37,700439][I][ezpz/dist:1222] Setting up wandb from rank=0
        [2025-06-02 08:39:37,701214][I][ezpz/dist:1223] Using WB_PROJECT=ezpz.test_dist
        wandb: Tracking run with wandb version 0.19.10
        wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
        wandb: WARNING URL not available in offline run
        [2025-06-02 08:39:38,357037][I][ezpz/dist:1249] wandb.run=[None](None)
        [2025-06-02 08:39:38,363539][I][ezpz/dist:1285] Running on machine='Aurora'
        [2025-06-02 08:39:38,368294][I][ezpz/test_dist:233:__main__] Took: 0.68 seconds to build trainer
        [2025-06-02 08:39:38,368985][I][ezpz/test_dist:235:__main__] config:
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
        [2025-06-02 08:39:38,370322][I][ezpz/test_dist:237:__main__] Took: 13.30 to get here.
        [2025-06-02 08:39:38,794611][I][ezpz/test_dist:196:__main__] Warmup complete at step 2
        [2025-06-02 08:39:38,813169][I][ezpz/test_dist:174:__main__] iter=10 loss=904.000000 dtf=0.000644 dtb=0.001260
        [2025-06-02 08:39:38,835905][I][ezpz/test_dist:174:__main__] iter=20 loss=712.000000 dtf=0.000610 dtb=0.001283
        [2025-06-02 08:39:38,858533][I][ezpz/test_dist:174:__main__] iter=30 loss=704.000000 dtf=0.000608 dtb=0.001252
        [2025-06-02 08:39:38,880929][I][ezpz/test_dist:174:__main__] iter=40 loss=684.000000 dtf=0.000607 dtb=0.001315
        [2025-06-02 08:39:38,903701][I][ezpz/test_dist:174:__main__] iter=50 loss=684.000000 dtf=0.000579 dtb=0.001247
        [2025-06-02 08:39:38,926119][I][ezpz/test_dist:174:__main__] iter=60 loss=676.000000 dtf=0.000597 dtb=0.001234
        [2025-06-02 08:39:38,948978][I][ezpz/test_dist:174:__main__] iter=70 loss=664.000000 dtf=0.000603 dtb=0.001242
        [2025-06-02 08:39:38,971256][I][ezpz/test_dist:174:__main__] iter=80 loss=672.000000 dtf=0.000599 dtb=0.001240
        [2025-06-02 08:39:38,993829][I][ezpz/test_dist:174:__main__] iter=90 loss=672.000000 dtf=0.000615 dtb=0.001249
        [2025-06-02 08:39:40,390558][I][ezpz/history:721] Saving iter plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
        [2025-06-02 08:39:40,653794][I][ezpz/history:721] Saving loss plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
        [2025-06-02 08:39:40,894262][I][ezpz/history:721] Saving dtf plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
        [2025-06-02 08:39:41,191474][I][ezpz/history:721] Saving dtb plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
        [2025-06-02 08:39:41,377999][I][ezpz/history:618] Saving tplots to /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot
                            loss [2025-06-02-083941]
            ┌──────────────────────────────────────────────────────┐
        2448┤▌                                                     │
            │▌                                                     │
        2150┤▚                                                     │
            │▐                                                     │
            │▐                                                     │
        1852┤▐                                                     │
            │▝▖                                                    │
        1554┤ ▚                                                    │
            │ ▝▖                                                   │
        1256┤  ▌                                                   │
            │  ▐                                                   │
            │   ▌                                                  │
         958┤   ▝▖                                                 │
            │    ▝▄▄                                               │
         660┤       ▀▀▀▀▀▀▀▀▀▚▄▀▞▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▞▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
            └─┬──┬───┬───┬───┬───┬──┬──┬──┬──┬──┬───┬───┬──┬──┬──┬─┘
            0 2  9  15  22  30  37 42 48 53 59 65  71  79 84 90 96
        loss                          iter
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/loss.txt
                            dtf [2025-06-02-083941]
                ┌──────────────────────────────────────────────────┐
        0.000805┤                                 ▗▌               │
                │                                 ▐▌               │
        0.000766┤                                 ▐▌               │
                │▌                                ▐▌               │
                │▌                                ▐▌               │
        0.000727┤▌   ▖              ▟          ▖  ▐▌▖              │
                │▌  ▐▌              █     ▖   ▐▌  ▐█▌              │
        0.000688┤▌  ▐▌    ▗    ▟    █    ▐▌   ▐▌  ▐█▌   ▗▌         │
                │▌  ▐▌    █    █    █    ▐▌   ▐▌  ▐█▌   ▐▌    ▟    │
        0.000649┤▌  ▐▌    █    █    █    ▐▌   ▐▌  ▐█▌   ▐▌    █    │
                │▌▗▌▞▌    █    █    ▌▀▌  ▞▝▌  ▐▌ ▖▐█▌   ▐▌    █    │
                │▚▀▐▌▌ ▖ ▗█   ▖█▗   ▌ ▌  ▌ ▐  ▐▚▀▌▐█▚   ▟▌▞▚  █▟▗  │
        0.000610┤   ▘▐▐▚▚▘▘▙▜▐▝▀▌▌▞▚▌ ▚▞▖▌ ▝▟▟▐  ▚▐▜ ▚ ▟█▝ ▝▖▐ ▘▜▞▀│
                │    ▝▌    ▝▝▌   ▝      ▐▌  ▐█▞   ▘  ▐▞▜▝   ▚▞  ▐▌ │
        0.000571┤                        ▘  ▝▌▘       ▘         ▝▌ │
                └─┬──┬──┬───┬───┬──┬──┬──┬──┬──┬──┬──┬───┬──┬────┬─┘
                0 2  9 15  22  30 37 42 48 53 60 65 71  79 85   96
        dtf                             iter
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtf.txt
                            dtf [2025-06-02-083941]
            ┌──────────────────────────────────────────────────────┐
        52.0┤     ██████                                           │
            │     ██████                                           │
        43.3┤     ██████                                           │
            │     ██████                                           │
            │     ██████                                           │
        34.7┤     ██████                                           │
            │     ██████                                           │
        26.0┤     ██████                                           │
            │     ██████                                           │
        17.3┤     ██████                                           │
            │     ███████████                                      │
            │████████████████                                      │
         8.7┤██████████████████████                                │
            │██████████████████████████████████████                │
         0.0┤███████████████████████████████████████████      █████│
            └┬────────────┬─────────────┬────────────┬────────────┬┘
        0.000560    0.000624      0.000688     0.000752  0.000815
        freq                           dtf
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtf-hist.txt
                            dtb [2025-06-02-083941]
                ┌──────────────────────────────────────────────────┐
        0.001447┤             ▟         ▟                          │
                │             █         █                          │
        0.001409┤             █         █                    ▖     │
                │             █         █                   ▐▌     │
                │             █         █                   ▐▌     │
        0.001371┤             █         █                   ▐▌     │
                │             █     ▟   █                   ▐▌     │
        0.001333┤   ▗▌▗    ▗▌ █     █   █ ▗                 ▐▌     │
                │▖▗ ▐▌█  ▗ ▐▌ █     ▌▌  █ █   ▗▌▗  ▗▚     ▗▌▐▌     │
        0.001294┤▌█ ▐▌█ ▗▜ ▐▌ █    ▐ ▐  █▐▐   ▐▝▜  ▐▐     ▐▐▐▌     │
                │▐█▟▐▌▛▄▞▝▄▐▚ █▗▗▌ ▐ ▐  █▐▐   ▐ ▐  ▐▐ ▖   ▌▐▐▌▗    │
                │▐██▐▝▘▐▌ ▝▌▐▟█▌█▙▌▐  ▌ █▌ ▌ ▗▐ ▝▄ ▐▐▐▌  ▗▘▐▐▌█    │
        0.001256┤ ▀▌▀   ▘    ▘▀▌▝▝▌▞  ▐ ▛▌ ▚▗█▐  ▝▟▐▝▟▌  ▟  █▌█    │
                │                 █   ▐▗▘   ▘ ▜    ▀ ▝▝▚▀   ▜▜ ▀▀▞▄│
        0.001218┤                 ▝   ▝▌                           │
                └─┬──┬──┬───┬───┬──┬──┬──┬──┬──┬──┬──┬───┬──┬────┬─┘
                0 2  9 15  22  30 37 42 48 53 60 65 71  79 85   96
        dtb                             iter
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtb.txt
                            dtb [2025-06-02-083941]
            ┌──────────────────────────────────────────────────────┐
        38.0┤     ██████                                           │
            │     ██████                                           │
        31.7┤     ██████                                           │
            │     ██████                                           │
            │     ██████                                           │
        25.3┤     ██████                                           │
            │     ██████                                           │
        19.0┤     ███████████                                      │
            │     ███████████                                      │
        12.7┤████████████████      █████                           │
            │████████████████      █████                           │
            │███████████████████████████                           │
         6.3┤███████████████████████████                           │
            │████████████████████████████████                 █████│
         0.0┤████████████████████████████████           ███████████│
            └┬────────────┬─────────────┬────────────┬────────────┬┘
        0.001208    0.001270      0.001333     0.001395  0.001457
        freq                           dtb
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtb-hist.txt
        [2025-06-02 08:39:41,427412][I][ezpz/test_dist:190:__main__] dataset=<xarray.Dataset> Size: 3kB
        Dimensions:  (draw: 97)
        Coordinates:
        * draw     (draw) int64 776B 0 1 2 3 4 5 6 7 8 ... 88 89 90 91 92 93 94 95 96
        Data variables:
            iter     (draw) int64 776B 3 4 5 6 7 8 9 10 11 ... 92 93 94 95 96 97 98 99
            loss     (draw) float32 388B 2.448e+03 2.112e+03 1.664e+03 ... 672.0 688.0
            dtf      (draw) float64 776B 0.0007564 0.0006201 ... 0.0006089 0.0006102
            dtb      (draw) float64 776B 0.001315 0.001286 ... 0.001238 0.001236
        [2025-06-02 08:39:41,429616][I][ezpz/test_dist:241:__main__] Took: 3.06 seconds to finish training
        [2025-06-02 08:39:41,430364][I][ezpz/test_dist:476:__main__] Took: 16.36 seconds
        wandb:
        wandb: You can sync this run to the cloud by running:
        wandb: wandb sync /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/wandb/offline-run-20250602_083937-57itor57
        wandb: Find logs at: ../../../../../../lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/wandb/offline-run-20250602_083937-57itor57/logs
        Application 09a72a12 resources: utime=853s stime=186s maxrss=3932628KB inblock=749276 oublock=904 minflt=11280849 majflt=42365 nvcsw=380342 nivcsw=3251786
        [2025-06-02 08:39:44,095734][I][ezpz/launch:201] Execution finished @ 2025-06-02-083944
        [2025-06-02 08:39:44,096767][I][ezpz/launch:202] Command took 32.13 seconds to run. Exiting.
        took: 0h:00m:43s
        ```

        </details>

    - Arbitrary python string:

        ```bash
        ezpz launch -c "'import ezpz; ezpz.setup_torch()'"
        ```

        <details closed><summary>Output:</summary>

        ```bash
        #[🐍 aurora_nre_models_frameworks-2025.0.0](👻 aurora_nre_models_frameworks-2025.0.0)
        #[/f/d/f/p/s/ezpz][🌱 saforem2/dev][📦🤷✓]
        #[06/02/25 @ 08:06:17][x4404c4s4b0n0]
        ; ezpz launch -c "'import ezpz; ezpz.setup_torch()'"

        [W602 08:06:24.384316779 OperatorEntry.cpp:155] Warning: Warning only once for all operators,  other operators may also be overridden.
        Overriding a previously registered kernel for the same operator and the same dispatch key
        operator: aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
            registered at /build/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
        dispatch key: XPU
        previous kernel: registered at /build/pytorch/build/aten/src/ATen/RegisterCPU.cpp:30476
            new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:2971 (function operator())
        [2025-06-02 08:06:31,007494][I][ezpz/__init__:278:ezpz] Setting logging level to 'INFO' on 'RANK == 0'
        [2025-06-02 08:06:31,009869][I][ezpz/__init__:279:ezpz] Setting logging level to 'CRITICAL' on all others 'RANK != 0'
        [2025-06-02 08:06:31,153935][I][ezpz/launch:157] Job ID: 5414072
        [2025-06-02 08:06:31,463973][I][ezpz/launch:163] Node file: /var/spool/pbs/aux/5414072.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
        [2025-06-02 08:06:31,469362][I][ezpz/launch:178] Building command to execute by piecing together:(1.) ['launch_cmd'] + (2.) ['python'] + (3.) ['cmd_to_launch']
        [2025-06-02 08:06:31,470095][I][ezpz/launch:182] (1.) ['launch_cmd']: mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/5414072.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind=depth --depth=8
        [2025-06-02 08:06:31,470676][I][ezpz/launch:183] (2.) ['python']: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/bin/python3
        [2025-06-02 08:06:31,471081][I][ezpz/launch:184] (3.) ['cmd_to_launch']:  -c 'import ezpz; ezpz.setup_torch()'
        [2025-06-02 08:06:31,471734][I][ezpz/launch:189] Took: 0.46 seconds to build command.
        [2025-06-02 08:06:31,472111][I][ezpz/launch:192] Executing: mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/5414072.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind=depth --depth=8 /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/bin/python3 -c 'import ezpz; ezpz.setup_torch()'
        [2025-06-02 08:06:31,472988][I][ezpz/launch:119] Filtering for Aurora-specific messages. To view list of filters, run with `EZPZ_LOG_LEVEL=DEBUG`
        [2025-06-02 08:06:31,473468][I][ezpz/launch:199] Execution started @ 2025-06-02-080631...

        Disabling local launch: multi-node application
        Connected to tcp://x4404c4s4b0n0.hostmgmt2404.cm.aurora.alcf.anl.gov:7919
        Launching application a166c768-dd6f-4d44-bcd7-d6f0ddd3da16
        [2025-06-02 08:06:48,763446][I][ezpz/__init__:278:ezpz] Setting logging level to 'INFO' on 'RANK == 0'
        [2025-06-02 08:06:48,765755][I][ezpz/__init__:279:ezpz] Setting logging level to 'CRITICAL' on all others 'RANK != 0'
        [2025-06-02 08:06:48,766509][I][ezpz/dist:760] Using get_torch_device_type()='xpu' with be='ddp'
        [2025-06-02 08:06:48,767183][I][ezpz/dist:573] Initializing process group with rank=0, world_size=24, torch_backend=ccl
        2025:06:02-08:06:52:(202581) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
        [2025-06-02 08:06:52,740330][I][ezpz/dist:964] Using device='xpu' with backend='ddp' + 'ccl' for distributed training.
        [2025-06-02 08:06:52,741117][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 0/23]
        [2025-06-02 08:06:52,740305][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 1/23]
        [2025-06-02 08:06:52,740308][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 3/23]
        [2025-06-02 08:06:52,740313][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 4/23]
        [2025-06-02 08:06:52,740304][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 5/23]
        [2025-06-02 08:06:52,740339][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 2/23]
        [2025-06-02 08:06:52,740272][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 7/23]
        [2025-06-02 08:06:52,740283][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 8/23]
        [2025-06-02 08:06:52,740275][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 9/23]
        [2025-06-02 08:06:52,740302][I][ezpz/dist:1011] ['x4404c4s4b0n0'][10/23]
        [2025-06-02 08:06:52,740275][I][ezpz/dist:1011] ['x4404c4s4b0n0'][11/23]
        [2025-06-02 08:06:52,740349][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 6/23]
        [2025-06-02 08:06:52,740225][I][ezpz/dist:1011] ['x4404c4s6b0n0'][21/23]
        [2025-06-02 08:06:52,740227][I][ezpz/dist:1011] ['x4404c4s6b0n0'][22/23]
        [2025-06-02 08:06:52,740224][I][ezpz/dist:1011] ['x4404c4s6b0n0'][23/23]
        [2025-06-02 08:06:52,740253][I][ezpz/dist:1011] ['x4404c4s6b0n0'][12/23]
        [2025-06-02 08:06:52,740240][I][ezpz/dist:1011] ['x4404c4s6b0n0'][13/23]
        [2025-06-02 08:06:52,740250][I][ezpz/dist:1011] ['x4404c4s6b0n0'][14/23]
        [2025-06-02 08:06:52,740247][I][ezpz/dist:1011] ['x4404c4s6b0n0'][15/23]
        [2025-06-02 08:06:52,740258][I][ezpz/dist:1011] ['x4404c4s6b0n0'][16/23]
        [2025-06-02 08:06:52,740240][I][ezpz/dist:1011] ['x4404c4s6b0n0'][17/23]
        [2025-06-02 08:06:52,740287][I][ezpz/dist:1011] ['x4404c4s6b0n0'][18/23]
        [2025-06-02 08:06:52,740226][I][ezpz/dist:1011] ['x4404c4s6b0n0'][19/23]
        [2025-06-02 08:06:52,740235][I][ezpz/dist:1011] ['x4404c4s6b0n0'][20/23]
        Application a166c768 resources: utime=247s stime=157s maxrss=3066848KB inblock=855410 oublock=0 minflt=6675290 majflt=22830 nvcsw=346921 nivcsw=1219341
        [2025-06-02 08:06:55,051587][I][ezpz/launch:201] Execution finished @ 2025-06-02-080655
        [2025-06-02 08:06:55,052786][I][ezpz/launch:202] Command took 23.58 seconds to run. Exiting.
        took: 0h:00m:35s
        ```

        </details>

    - Minimal example
      \[[ezpz / examples / `minimal.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/minimal.py)\]:

        ```bash
        ezpz launch -m ezpz.examples.minimal
        ```

        <details closed><summary>Output:</summary>

        ```bash
        #[🐍 aurora_nre_models_frameworks-2025.0.0](👻 aurora_nre_models_frameworks-2025.0.0)
        #[/f/d/f/p/s/ezpz][🌱 saforem2/dev][📦🤷✓] [⏱️ 58s]
        #[06/02/25 @ 08:24:30][x4404c4s4b0n0]
        ; WANDB_MODE=offline PRINT_ITERS=100 TRAIN_ITERS=1000 ezpz launch -m ezpz.examples.minimal
        [W602 08:24:33.632744487 OperatorEntry.cpp:155] Warning: Warning only once for all operators,  other operators may also be overridden.
        Overriding a previously registered kernel for the same operator and the same dispatch key
        operator: aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
            registered at /build/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
        dispatch key: XPU
        previous kernel: registered at /build/pytorch/build/aten/src/ATen/RegisterCPU.cpp:30476
            new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:2971 (function operator())
        [2025-06-02 08:24:40,394556][I][ezpz/__init__:278:ezpz] Setting logging level to 'INFO' on 'RANK == 0'
        [2025-06-02 08:24:40,397025][I][ezpz/__init__:279:ezpz] Setting logging level to 'CRITICAL' on all others 'RANK != 0'
        [2025-06-02 08:24:40,546683][I][ezpz/launch:157] Job ID: 5414072
        [2025-06-02 08:24:40,862126][I][ezpz/launch:163] Node file: /var/spool/pbs/aux/5414072.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
        [2025-06-02 08:24:40,867464][I][ezpz/launch:178] Building command to execute by piecing together:(1.) ['launch_cmd'] + (2.) ['python'] + (3.) ['cmd_to_launch']
        [2025-06-02 08:24:40,868229][I][ezpz/launch:182] (1.) ['launch_cmd']: mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/5414072.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind=depth --depth=8
        [2025-06-02 08:24:40,868796][I][ezpz/launch:183] (2.) ['python']: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/bin/python3
        [2025-06-02 08:24:40,869195][I][ezpz/launch:184] (3.) ['cmd_to_launch']:  -m ezpz.examples.minimal
        [2025-06-02 08:24:40,869807][I][ezpz/launch:189] Took: 0.47 seconds to build command.
        [2025-06-02 08:24:40,870158][I][ezpz/launch:192] Executing: mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/5414072.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind=depth --depth=8 /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/bin/python3 -m ezpz.examples.minimal
        [2025-06-02 08:24:40,871013][I][ezpz/launch:119] Filtering for Aurora-specific messages. To view list of filters, run with `EZPZ_LOG_LEVEL=DEBUG`
        [2025-06-02 08:24:40,871479][I][ezpz/launch:199] Execution started @ 2025-06-02-082440...

        Disabling local launch: multi-node application
        Connected to tcp://x4404c4s4b0n0.hostmgmt2404.cm.aurora.alcf.anl.gov:7919
        Launching application 51803e72-8555-4056-b49e-4aa9ffb3b099
        [2025-06-02 08:24:54,200723][I][ezpz/__init__:278:ezpz] Setting logging level to 'INFO' on 'RANK == 0'
        [2025-06-02 08:24:54,203301][I][ezpz/__init__:279:ezpz] Setting logging level to 'CRITICAL' on all others 'RANK != 0'
        [2025-06-02 08:24:54,206944][I][ezpz/dist:760] Using get_torch_device_type()='xpu' with be='ddp'
        [2025-06-02 08:24:54,207778][I][ezpz/dist:573] Initializing process group with rank=0, world_size=24, torch_backend=ccl
        2025:06:02-08:24:55:(17665) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
        [2025-06-02 08:24:55,942022][I][ezpz/dist:964] Using device='xpu' with backend='ddp' + 'ccl' for distributed training.
        [2025-06-02 08:24:55,942738][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 0/23]
        [2025-06-02 08:24:55,941993][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 3/23]
        [2025-06-02 08:24:55,942007][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 1/23]
        [2025-06-02 08:24:55,942013][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 2/23]
        [2025-06-02 08:24:55,942019][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 4/23]
        [2025-06-02 08:24:55,942013][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 5/23]
        [2025-06-02 08:24:55,941989][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 8/23]
        [2025-06-02 08:24:55,942001][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 6/23]
        [2025-06-02 08:24:55,941994][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 7/23]
        [2025-06-02 08:24:55,941995][I][ezpz/dist:1011] ['x4404c4s4b0n0'][10/23]
        [2025-06-02 08:24:55,941990][I][ezpz/dist:1011] ['x4404c4s4b0n0'][11/23]
        [2025-06-02 08:24:55,942003][I][ezpz/dist:1011] ['x4404c4s4b0n0'][ 9/23]
        [2025-06-02 08:24:55,942096][I][ezpz/dist:1011] ['x4404c4s6b0n0'][12/23]
        [2025-06-02 08:24:55,942095][I][ezpz/dist:1011] ['x4404c4s6b0n0'][13/23]
        [2025-06-02 08:24:55,942101][I][ezpz/dist:1011] ['x4404c4s6b0n0'][14/23]
        [2025-06-02 08:24:55,942096][I][ezpz/dist:1011] ['x4404c4s6b0n0'][15/23]
        [2025-06-02 08:24:55,942092][I][ezpz/dist:1011] ['x4404c4s6b0n0'][16/23]
        [2025-06-02 08:24:55,942097][I][ezpz/dist:1011] ['x4404c4s6b0n0'][17/23]
        [2025-06-02 08:24:55,942091][I][ezpz/dist:1011] ['x4404c4s6b0n0'][18/23]
        [2025-06-02 08:24:55,942073][I][ezpz/dist:1011] ['x4404c4s6b0n0'][19/23]
        [2025-06-02 08:24:55,942076][I][ezpz/dist:1011] ['x4404c4s6b0n0'][20/23]
        [2025-06-02 08:24:55,942080][I][ezpz/dist:1011] ['x4404c4s6b0n0'][21/23]
        [2025-06-02 08:24:55,945053][I][ezpz/dist:1222] Setting up wandb from rank=0
        [2025-06-02 08:24:55,942081][I][ezpz/dist:1011] ['x4404c4s6b0n0'][22/23]
        [2025-06-02 08:24:55,942072][I][ezpz/dist:1011] ['x4404c4s6b0n0'][23/23]
        [2025-06-02 08:24:55,945440][I][ezpz/dist:1223] Using WB_PROJECT=ezpz.examples.minimal
        wandb: Tracking run with wandb version 0.19.10
        wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
        wandb: WARNING URL not available in offline run
        [2025-06-02 08:24:56,605530][I][ezpz/dist:1249] wandb.run=[None](None)
        [2025-06-02 08:24:56,611884][I][ezpz/dist:1285] Running on machine='Aurora'
        [2025-06-02 08:24:56,655910][I][examples/minimal:88:__main__] model=SequentialLinearNet(
        (layers): Sequential(
            (0): Linear(in_features=128, out_features=256, bias=True)
            (1): ReLU()
            (2): Linear(in_features=256, out_features=512, bias=True)
            (3): ReLU()
            (4): Linear(in_features=512, out_features=1024, bias=True)
            (5): ReLU()
            (6): Linear(in_features=1024, out_features=2048, bias=True)
            (7): ReLU()
            (8): Linear(in_features=2048, out_features=1024, bias=True)
            (9): ReLU()
            (10): Linear(in_features=1024, out_features=512, bias=True)
            (11): ReLU()
            (12): Linear(in_features=512, out_features=256, bias=True)
            (13): ReLU()
            (14): Linear(in_features=256, out_features=128, bias=True)
            (15): ReLU()
            (16): Linear(in_features=128, out_features=128, bias=True)
        )
        )
        [2025-06-02 08:25:07,566410][I][ezpz/dist:144] `setup` took: dt=13.3595s
        [2025-06-02 08:25:08,196630][I][examples/minimal:51:__main__] iter=20 loss=713.134399 dt=0.005150 dtf=0.001118 dtb=0.004031
        [2025-06-02 08:25:08,254359][I][examples/minimal:51:__main__] iter=30 loss=698.142334 dt=0.005140 dtf=0.001098 dtb=0.004042
        [2025-06-02 08:25:08,311676][I][examples/minimal:51:__main__] iter=40 loss=688.149536 dt=0.005088 dtf=0.001100 dtb=0.003988
        [2025-06-02 08:25:08,369744][I][examples/minimal:51:__main__] iter=50 loss=685.806091 dt=0.005097 dtf=0.001088 dtb=0.004009
        [2025-06-02 08:25:08,427011][I][examples/minimal:51:__main__] iter=60 loss=689.389771 dt=0.005140 dtf=0.001099 dtb=0.004041
        [2025-06-02 08:25:08,484186][I][examples/minimal:51:__main__] iter=70 loss=695.363220 dt=0.005125 dtf=0.001111 dtb=0.004014
        [2025-06-02 08:25:08,541436][I][examples/minimal:51:__main__] iter=80 loss=667.858032 dt=0.005074 dtf=0.001092 dtb=0.003982
        [2025-06-02 08:25:08,598606][I][examples/minimal:51:__main__] iter=90 loss=676.533142 dt=0.005130 dtf=0.001084 dtb=0.004046
        [2025-06-02 08:25:08,656182][I][examples/minimal:51:__main__] iter=100 loss=676.170593 dt=0.005510 dtf=0.001399 dtb=0.004111
        [2025-06-02 08:25:08,713804][I][examples/minimal:51:__main__] iter=110 loss=676.684814 dt=0.005106 dtf=0.001093 dtb=0.004013
        [2025-06-02 08:25:08,773811][I][examples/minimal:51:__main__] iter=120 loss=682.333984 dt=0.005353 dtf=0.001093 dtb=0.004260
        [2025-06-02 08:25:08,832594][I][examples/minimal:51:__main__] iter=130 loss=691.218079 dt=0.005333 dtf=0.001119 dtb=0.004214
        [2025-06-02 08:25:08,891644][I][examples/minimal:51:__main__] iter=140 loss=686.254883 dt=0.005318 dtf=0.001096 dtb=0.004223
        [2025-06-02 08:25:08,950476][I][examples/minimal:51:__main__] iter=150 loss=671.173218 dt=0.005462 dtf=0.001090 dtb=0.004372
        [2025-06-02 08:25:09,009324][I][examples/minimal:51:__main__] iter=160 loss=675.119751 dt=0.005372 dtf=0.001095 dtb=0.004277
        [2025-06-02 08:25:09,068117][I][examples/minimal:51:__main__] iter=170 loss=681.518127 dt=0.005401 dtf=0.001101 dtb=0.004299
        [2025-06-02 08:25:09,129145][I][examples/minimal:51:__main__] iter=180 loss=681.293335 dt=0.005290 dtf=0.001100 dtb=0.004189
        [2025-06-02 08:25:09,188790][I][examples/minimal:51:__main__] iter=190 loss=673.555298 dt=0.006316 dtf=0.001088 dtb=0.005228
        [2025-06-02 08:25:09,248623][I][examples/minimal:51:__main__] iter=200 loss=686.017700 dt=0.005552 dtf=0.001355 dtb=0.004196
        [2025-06-02 08:25:09,307659][I][examples/minimal:51:__main__] iter=210 loss=693.399170 dt=0.005361 dtf=0.001096 dtb=0.004265
        [2025-06-02 08:25:09,366454][I][examples/minimal:51:__main__] iter=220 loss=687.048462 dt=0.005304 dtf=0.001083 dtb=0.004222
        [2025-06-02 08:25:09,425278][I][examples/minimal:51:__main__] iter=230 loss=683.272217 dt=0.005334 dtf=0.001091 dtb=0.004242
        [2025-06-02 08:25:09,484085][I][examples/minimal:51:__main__] iter=240 loss=686.674561 dt=0.005240 dtf=0.001100 dtb=0.004140
        [2025-06-02 08:25:09,542500][I][examples/minimal:51:__main__] iter=250 loss=686.590210 dt=0.005419 dtf=0.001090 dtb=0.004330
        [2025-06-02 08:25:09,601444][I][examples/minimal:51:__main__] iter=260 loss=685.613770 dt=0.005404 dtf=0.000970 dtb=0.004434
        [2025-06-02 08:25:09,660262][I][examples/minimal:51:__main__] iter=270 loss=678.604309 dt=0.005277 dtf=0.000975 dtb=0.004302
        [2025-06-02 08:25:09,718685][I][examples/minimal:51:__main__] iter=280 loss=687.360474 dt=0.005371 dtf=0.000978 dtb=0.004393
        [2025-06-02 08:25:09,777952][I][examples/minimal:51:__main__] iter=290 loss=672.192383 dt=0.005500 dtf=0.000973 dtb=0.004527
        [2025-06-02 08:25:09,836219][I][examples/minimal:51:__main__] iter=300 loss=670.950562 dt=0.005342 dtf=0.001353 dtb=0.003989
        [2025-06-02 08:25:09,894611][I][examples/minimal:51:__main__] iter=310 loss=681.033447 dt=0.005213 dtf=0.001068 dtb=0.004145
        [2025-06-02 08:25:09,952968][I][examples/minimal:51:__main__] iter=320 loss=678.913208 dt=0.005336 dtf=0.000975 dtb=0.004361
        [2025-06-02 08:25:10,011736][I][examples/minimal:51:__main__] iter=330 loss=678.553772 dt=0.005430 dtf=0.001081 dtb=0.004349
        [2025-06-02 08:25:10,070662][I][examples/minimal:51:__main__] iter=340 loss=688.489014 dt=0.005390 dtf=0.001087 dtb=0.004303
        [2025-06-02 08:25:10,129419][I][examples/minimal:51:__main__] iter=350 loss=680.676147 dt=0.005368 dtf=0.000978 dtb=0.004390
        [2025-06-02 08:25:10,187801][I][examples/minimal:51:__main__] iter=360 loss=696.601196 dt=0.005339 dtf=0.001079 dtb=0.004261
        [2025-06-02 08:25:10,246699][I][examples/minimal:51:__main__] iter=370 loss=685.925903 dt=0.005347 dtf=0.001099 dtb=0.004248
        [2025-06-02 08:25:10,305350][I][examples/minimal:51:__main__] iter=380 loss=681.857178 dt=0.005277 dtf=0.001088 dtb=0.004188
        [2025-06-02 08:25:10,364235][I][examples/minimal:51:__main__] iter=390 loss=677.403076 dt=0.005545 dtf=0.001099 dtb=0.004445
        [2025-06-02 08:25:10,423312][I][examples/minimal:51:__main__] iter=400 loss=680.605286 dt=0.005513 dtf=0.001338 dtb=0.004175
        [2025-06-02 08:25:10,482306][I][examples/minimal:51:__main__] iter=410 loss=688.305176 dt=0.005358 dtf=0.001094 dtb=0.004264
        [2025-06-02 08:25:10,541514][I][examples/minimal:51:__main__] iter=420 loss=676.714600 dt=0.005456 dtf=0.001107 dtb=0.004349
        [2025-06-02 08:25:10,600146][I][examples/minimal:51:__main__] iter=430 loss=674.251648 dt=0.005348 dtf=0.001116 dtb=0.004232
        [2025-06-02 08:25:10,659099][I][examples/minimal:51:__main__] iter=440 loss=692.857361 dt=0.005285 dtf=0.001091 dtb=0.004194
        [2025-06-02 08:25:10,718127][I][examples/minimal:51:__main__] iter=450 loss=683.334229 dt=0.005442 dtf=0.001094 dtb=0.004348
        [2025-06-02 08:25:10,776750][I][examples/minimal:51:__main__] iter=460 loss=1509.692139 dt=0.005363 dtf=0.001114 dtb=0.004248
        [2025-06-02 08:25:10,836261][I][examples/minimal:51:__main__] iter=470 loss=943.557617 dt=0.005265 dtf=0.001108 dtb=0.004157
        [2025-06-02 08:25:10,895405][I][examples/minimal:51:__main__] iter=480 loss=704.171509 dt=0.005319 dtf=0.001079 dtb=0.004240
        [2025-06-02 08:25:10,954483][I][examples/minimal:51:__main__] iter=490 loss=683.428223 dt=0.005526 dtf=0.001086 dtb=0.004440
        [2025-06-02 08:25:11,013286][I][examples/minimal:51:__main__] iter=500 loss=687.314941 dt=0.005473 dtf=0.001332 dtb=0.004141
        [2025-06-02 08:25:11,080691][I][examples/minimal:51:__main__] iter=510 loss=688.060669 dt=0.005363 dtf=0.001113 dtb=0.004250
        [2025-06-02 08:25:11,139480][I][examples/minimal:51:__main__] iter=520 loss=686.497314 dt=0.005267 dtf=0.001083 dtb=0.004184
        [2025-06-02 08:25:11,198098][I][examples/minimal:51:__main__] iter=530 loss=691.718445 dt=0.005295 dtf=0.001086 dtb=0.004208
        [2025-06-02 08:25:11,256868][I][examples/minimal:51:__main__] iter=540 loss=681.122681 dt=0.005295 dtf=0.001104 dtb=0.004191
        [2025-06-02 08:25:11,315729][I][examples/minimal:51:__main__] iter=550 loss=683.272705 dt=0.005441 dtf=0.001081 dtb=0.004360
        [2025-06-02 08:25:11,374406][I][examples/minimal:51:__main__] iter=560 loss=688.077271 dt=0.005318 dtf=0.001093 dtb=0.004225
        [2025-06-02 08:25:11,433181][I][examples/minimal:51:__main__] iter=570 loss=683.032715 dt=0.005285 dtf=0.001099 dtb=0.004186
        [2025-06-02 08:25:11,491905][I][examples/minimal:51:__main__] iter=580 loss=686.191040 dt=0.005301 dtf=0.001089 dtb=0.004212
        [2025-06-02 08:25:11,550809][I][examples/minimal:51:__main__] iter=590 loss=691.924744 dt=0.005503 dtf=0.001088 dtb=0.004415
        [2025-06-02 08:25:11,609581][I][examples/minimal:51:__main__] iter=600 loss=681.312744 dt=0.005478 dtf=0.001338 dtb=0.004140
        [2025-06-02 08:25:11,668293][I][examples/minimal:51:__main__] iter=610 loss=680.253540 dt=0.005360 dtf=0.001120 dtb=0.004240
        [2025-06-02 08:25:11,726991][I][examples/minimal:51:__main__] iter=620 loss=683.039673 dt=0.005297 dtf=0.001090 dtb=0.004207
        [2025-06-02 08:25:11,785960][I][examples/minimal:51:__main__] iter=630 loss=679.695679 dt=0.005319 dtf=0.001080 dtb=0.004239
        [2025-06-02 08:25:11,845069][I][examples/minimal:51:__main__] iter=640 loss=686.198608 dt=0.005340 dtf=0.001108 dtb=0.004233
        [2025-06-02 08:25:11,903999][I][examples/minimal:51:__main__] iter=650 loss=683.652954 dt=0.005456 dtf=0.001089 dtb=0.004367
        [2025-06-02 08:25:11,962543][I][examples/minimal:51:__main__] iter=660 loss=686.860229 dt=0.005316 dtf=0.001086 dtb=0.004229
        [2025-06-02 08:25:12,021274][I][examples/minimal:51:__main__] iter=670 loss=680.933960 dt=0.005314 dtf=0.001097 dtb=0.004217
        [2025-06-02 08:25:12,079889][I][examples/minimal:51:__main__] iter=680 loss=679.905151 dt=0.005319 dtf=0.001089 dtb=0.004230
        [2025-06-02 08:25:12,138620][I][examples/minimal:51:__main__] iter=690 loss=682.389832 dt=0.005544 dtf=0.000994 dtb=0.004550
        [2025-06-02 08:25:12,196877][I][examples/minimal:51:__main__] iter=700 loss=686.506714 dt=0.005393 dtf=0.001366 dtb=0.004027
        [2025-06-02 08:25:12,255083][I][examples/minimal:51:__main__] iter=710 loss=690.196533 dt=0.005322 dtf=0.001087 dtb=0.004235
        [2025-06-02 08:25:12,313749][I][examples/minimal:51:__main__] iter=720 loss=678.437134 dt=0.005271 dtf=0.001083 dtb=0.004188
        [2025-06-02 08:25:12,372685][I][examples/minimal:51:__main__] iter=730 loss=682.770264 dt=0.005329 dtf=0.001116 dtb=0.004212
        [2025-06-02 08:25:12,431392][I][examples/minimal:51:__main__] iter=740 loss=688.560852 dt=0.005218 dtf=0.001016 dtb=0.004203
        [2025-06-02 08:25:12,489897][I][examples/minimal:51:__main__] iter=750 loss=687.129883 dt=0.005418 dtf=0.001091 dtb=0.004327
        [2025-06-02 08:25:12,548527][I][examples/minimal:51:__main__] iter=760 loss=684.507507 dt=0.005340 dtf=0.001128 dtb=0.004211
        [2025-06-02 08:25:12,607235][I][examples/minimal:51:__main__] iter=770 loss=674.559021 dt=0.005275 dtf=0.001087 dtb=0.004188
        [2025-06-02 08:25:12,666059][I][examples/minimal:51:__main__] iter=780 loss=690.597290 dt=0.005311 dtf=0.001068 dtb=0.004243
        [2025-06-02 08:25:12,724778][I][examples/minimal:51:__main__] iter=790 loss=675.396240 dt=0.005521 dtf=0.001100 dtb=0.004422
        [2025-06-02 08:25:12,783613][I][examples/minimal:51:__main__] iter=800 loss=673.097961 dt=0.005453 dtf=0.001320 dtb=0.004134
        [2025-06-02 08:25:12,842443][I][examples/minimal:51:__main__] iter=810 loss=679.685730 dt=0.005444 dtf=0.001118 dtb=0.004326
        [2025-06-02 08:25:12,901496][I][examples/minimal:51:__main__] iter=820 loss=673.053711 dt=0.005300 dtf=0.001088 dtb=0.004212
        [2025-06-02 08:25:12,960154][I][examples/minimal:51:__main__] iter=830 loss=680.830994 dt=0.005351 dtf=0.001112 dtb=0.004239
        [2025-06-02 08:25:13,018906][I][examples/minimal:51:__main__] iter=840 loss=691.692932 dt=0.005299 dtf=0.001091 dtb=0.004208
        [2025-06-02 08:25:13,077564][I][examples/minimal:51:__main__] iter=850 loss=674.963257 dt=0.005420 dtf=0.001105 dtb=0.004315
        [2025-06-02 08:25:13,136279][I][examples/minimal:51:__main__] iter=860 loss=684.604980 dt=0.005302 dtf=0.001107 dtb=0.004195
        [2025-06-02 08:25:13,194978][I][examples/minimal:51:__main__] iter=870 loss=696.048218 dt=0.005365 dtf=0.001101 dtb=0.004264
        [2025-06-02 08:25:13,253730][I][examples/minimal:51:__main__] iter=880 loss=679.293457 dt=0.005284 dtf=0.001077 dtb=0.004207
        [2025-06-02 08:25:13,312501][I][examples/minimal:51:__main__] iter=890 loss=679.364197 dt=0.005558 dtf=0.001110 dtb=0.004448
        [2025-06-02 08:25:13,371428][I][examples/minimal:51:__main__] iter=900 loss=675.571289 dt=0.005417 dtf=0.001344 dtb=0.004074
        [2025-06-02 08:25:13,430037][I][examples/minimal:51:__main__] iter=910 loss=683.194458 dt=0.005323 dtf=0.001077 dtb=0.004246
        [2025-06-02 08:25:13,488662][I][examples/minimal:51:__main__] iter=920 loss=689.960022 dt=0.005316 dtf=0.001103 dtb=0.004213
        [2025-06-02 08:25:13,547197][I][examples/minimal:51:__main__] iter=930 loss=693.487732 dt=0.005348 dtf=0.001097 dtb=0.004251
        [2025-06-02 08:25:13,606009][I][examples/minimal:51:__main__] iter=940 loss=686.816406 dt=0.005356 dtf=0.001087 dtb=0.004269
        [2025-06-02 08:25:13,664743][I][examples/minimal:51:__main__] iter=950 loss=670.237244 dt=0.005430 dtf=0.001109 dtb=0.004322
        [2025-06-02 08:25:13,723404][I][examples/minimal:51:__main__] iter=960 loss=700.734741 dt=0.005330 dtf=0.001073 dtb=0.004257
        [2025-06-02 08:25:13,782161][I][examples/minimal:51:__main__] iter=970 loss=676.606628 dt=0.005324 dtf=0.001075 dtb=0.004249
        [2025-06-02 08:25:13,840797][I][examples/minimal:51:__main__] iter=980 loss=687.955688 dt=0.005335 dtf=0.001105 dtb=0.004230
        [2025-06-02 08:25:13,900017][I][examples/minimal:51:__main__] iter=990 loss=689.839966 dt=0.005527 dtf=0.001089 dtb=0.004438
        [2025-06-02 08:25:13,953099][I][ezpz/dist:144] `train`((DistributedDataParallel(
        (module): SequentialLinearNet(
            (layers): Sequential(
            (0): Linear(in_features=128, out_features=256, bias=True)
            (1): ReLU()
            (2): Linear(in_features=256, out_features=512, bias=True)
            (3): ReLU()
            (4): Linear(in_features=512, out_features=1024, bias=True)
            (5): ReLU()
            (6): Linear(in_features=1024, out_features=2048, bias=True)
            (7): ReLU()
            (8): Linear(in_features=2048, out_features=1024, bias=True)
            (9): ReLU()
            (10): Linear(in_features=1024, out_features=512, bias=True)
            (11): ReLU()
            (12): Linear(in_features=512, out_features=256, bias=True)
            (13): ReLU()
            (14): Linear(in_features=256, out_features=128, bias=True)
            (15): ReLU()
            (16): Linear(in_features=128, out_features=128, bias=True)
            )
        )
        ), Adam (
        Parameter Group 0
            amsgrad: False
            betas: (0.9, 0.999)
            capturable: False
            differentiable: False
            eps: 1e-08
            foreach: None
            fused: None
            lr: 0.001
            maximize: False
            weight_decay: 0
        ))) took: dt=6.3856s
        [2025-06-02 08:25:15,312954][I][ezpz/history:721] Saving iter plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-06-02-082513/2025-06-02-082513/History-2025-06-02-082513/plots/mplot
        [2025-06-02 08:25:15,581086][I][ezpz/history:721] Saving loss plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-06-02-082513/2025-06-02-082513/History-2025-06-02-082513/plots/mplot
        [2025-06-02 08:25:15,860783][I][ezpz/history:721] Saving dt plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-06-02-082513/2025-06-02-082513/History-2025-06-02-082513/plots/mplot
        [2025-06-02 08:25:16,124027][I][ezpz/history:721] Saving dtf plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-06-02-082513/2025-06-02-082513/History-2025-06-02-082513/plots/mplot
        [2025-06-02 08:25:16,380159][I][ezpz/history:721] Saving dtb plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-06-02-082513/2025-06-02-082513/History-2025-06-02-082513/plots/mplot
        [2025-06-02 08:25:16,627648][I][ezpz/history:618] Saving tplots to /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-06-02-082513/2025-06-02-082513/History-2025-06-02-082513/plots/tplot
                            loss [2025-06-02-082516]
              ┌────────────────────────────────────────────────────┐
        2326.0┤                       ▐                            │
              │                       ▟                            │
        2048.7┤                       █                            │
              │                       █                            │
              │                       █                            │
        1771.5┤                       █                            │
              │                       █                            │
        1494.2┤                       █                            │
              │                       █▌                           │
        1216.9┤                       █▌                           │
              │                       █▌                           │
              │▖                     ▐█▌                           │
         939.7┤▌                     ▐█▌                           │
              │▙                     ▐▛█                           │
         662.4┤▝█▙▙▙█▙▟▟██▙▟▙▄▄▄▙█▄▟▙▟▌▀████▟█▟███▙█▟█▄▄█▙██▙▄█▙█▙▙│
              └┬──┬────┬──────┬───┬───┬───┬───────┬──┬───┬─────┬───┘
              10 61   152    301 374 443 516    682 746 805   937
        loss                           iter
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-06-02-082513/2025-06-02-082513/History-2025-06-02-082513/plots/tplot/loss.txt
                            dt [2025-06-02-082516]
               ┌───────────────────────────────────────────────────┐
        0.00665┤        ▌                                          │
               │        ▌                                          │
        0.00631┤        ▌▖                                         │
               │        ▌▌                                         │
               │        ▌▌                                         │
        0.00597┤        ▌▌                                         │
               │     ▖  ▌▌              ▖                          │
        0.00563┤  ▖  ▙▄ ▌▙▄  ▗  ▗  ▗▖▙ ▙▌▗  ▗ ▄▄   ▗ ▖ ▄▄ ▖  ▗▄  ▗ │
               │▖▖▌ ▟██▄████▟▟█▙▟█▟█▙█▙████▙▟███▟▗█▟▄█▄██▙█▖▟███▟██│
        0.00529┤██▌▟███████████████████████████████████████████████│
               │████████▐▐█████████████▜████▜█████████████████████▌│
               │███▀█▜█▌▐▐████████▐█▜██▐████▐████▐████▌████▌████▌█▌│
        0.00495┤     ▝ ▘  ▀▜██▜█▀▜ ▀▝▘▀▝█▘▘▀ ▀▝▘▝▝▜█▘▜▘ ▘▝▀▘ ▀▝▝▘▘▘│
               │           ▐██▐█ ▐                ▐█ ▐             │
        0.00461┤           ▝▌▝▐▛ ▐                 ▘ ▐             │
               └┬──┬────┬──────┬──────┬───┬───┬───┬───┬────┬───┬───┘
               10 61   152    301    443 516 601 682 746  844 937
        dt                             iter
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-06-02-082513/2025-06-02-082513/History-2025-06-02-082513/plots/tplot/dt.txt
                            dt [2025-06-02-082516]
           ┌───────────────────────────────────────────────────────┐
        648┤                 █████                                 │
           │                 █████                                 │
        540┤                 █████                                 │
           │                 █████                                 │
           │                 █████                                 │
        432┤                 █████                                 │
           │                 █████                                 │
        324┤                 █████                                 │
           │                 █████                                 │
        216┤                 █████                                 │
           │                 ██████████                            │
           │                 ██████████                            │
        108┤           █████ ██████████                            │
           │      ██████████ ██████████                            │
          0┤█████ ██████████ ██████████ █████      ██████████ █████│
           └┬─────────────┬────────────┬─────────────┬────────────┬┘
           0.00452      0.00507      0.00563       0.00618   0.00674
        freq                          dt
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-06-02-082513/2025-06-02-082513/History-2025-06-02-082513/plots/tplot/dt-hist.txt
                            dtf [2025-06-02-082516]
                ┌──────────────────────────────────────────────────┐
        0.001399┤    ▐                                             │
                │    ▐▗   ▗    ▗     ▗    ▐        ▐▗    ▗   ▗▗    │
        0.001321┤ ▌  ▐▐   ▐▐   ▐▗   ▐▐   ▐▐   ▐▐   ▐▐   ▗▐   ▐▐    │
                │ ▌  ▐▐   ▐▐   ▐▐   ▐▐   ▐▐   ▐▐   ▐▐   ▐▐   ▐▐    │
                │ ▌  ▐▐   ▐▐   ▐▐   ▐▐   ▐▐   ▐▐   ▐▐   ▐▐   ▐▐    │
        0.001243┤ ▌  ▐▐   ▐▐   ▐▐   ▐▐   ▐▐   ▐▐   ▐▐   ▐▐   ▐▐    │
                │ ▌▖▗▐▟   ▐▐   ▐▐   ▐▐   ▐▐▖  ▐▐▖  ▐▐  ▗▐▐  ▗▐▐  ▖ │
        0.001164┤▗▙▌▟▐█▟▐▖▟▟ ▐▗▐▐ ▗█▟▟▙▟▗▐▟▌▟▟▐█▙  ▐▟ ▄█▟█▐▙▟▐█▙▟▙▌│
                │▐██████████▌█▐▟▐█▐████████████████▐█▌████████████▌│
        0.001086┤▐████████████▟████████████████████▐███████████████│
                │▐▝▘▀▘▝▝▛▀▀▐██████▛▐▀▀▀▝▀▀▛▀▘▝▝▀▘▀▐█▜█▜▀▝▝▀▜▀▀▀ ▀▜▝│
                │▐         ▐██████▌       ▌       ▐█▐█             │
        0.001008┤▟         ▐██████▌       ▌       ▐█▐█             │
                │█          █████▜▘       ▌       ▐▛▐▜             │
        0.000930┤▝          ▝  ▘          ▌       ▝                │
                └┬──┬───┬───┬───┬──────┬───┬───────┬──┬────┬───┬───┘
                10 61  152 222 301    443 516    682 746  844 937
        dtf                             iter
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-06-02-082513/2025-06-02-082513/History-2025-06-02-082513/plots/tplot/dtf.txt
                            dtf [2025-06-02-082516]
             ┌─────────────────────────────────────────────────────┐
        724.0┤                █████                                │
             │                █████                                │
        603.3┤                █████                                │
             │                █████                                │
             │                █████                                │
        482.7┤                █████                                │
             │                █████                                │
        362.0┤                █████                                │
             │                █████                                │
        241.3┤                █████                                │
             │                █████                                │
             │                █████                                │
        120.7┤                ██████████                           │
             │███████████     ██████████ █████                     │
          0.0┤██████████████████████████ █████          ███████████│
             └┬────────────┬────────────┬────────────┬────────────┬┘
            0.00091      0.00104      0.00116      0.00129   0.00142
        freq                           dtf
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-06-02-082513/2025-06-02-082513/History-2025-06-02-082513/plots/tplot/dtf-hist.txt
                            dtb [2025-06-02-082516]
               ┌───────────────────────────────────────────────────┐
        0.00555┤        ▌                                          │
               │        ▌                                          │
        0.00522┤        ▌▖                                         │
               │        ▌▌                                         │
               │        ▌▌                                         │
        0.00489┤        ▌▌                                         │
               │        ▌▌               ▗                         │
        0.00456┤     ▌  ▌▙   ▗▄▖▗▖  ▖  ▖▌▐        ▗▗         ▗   ▗ │
               │▙ ▌  ▙█▄██▌▙███▌█▙▄█▙█▌███▐▖▐▙█▐▄ ▟█▟█▖██▖▙▖▟█▌▄▗██│
        0.00424┤██▌▟▌█████████████████████▟████████████████▙███████│
               │████████▜▜████████████████████████████████████████▛│
               │▀██████▛▐▐████████▜████▜████▜████▜████▛████▛████▛█▌│
        0.00391┤ ▛▀ ▜▐█▌▐▝████████▐█▜██▐████▐█▜█▜▐████▌████▌████▌█▌│
               │     ▝ ▘  ▘▐██▜█▀▜ ▀ ▘▀▝▀▘▘▘ ▘▝  ▝▜▛▀▜▘▝▘ ▀▘ ▀  ▘▘ │
        0.00358┤           ▝▛▝▐▛ ▐                 ▘ ▐             │
               └┬──┬────┬──────┬──────┬───┬───┬───┬───┬────┬───┬───┘
               10 61   152    301    443 516 601 682 746  844 937
        dtb                            iter
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-06-02-082513/2025-06-02-082513/History-2025-06-02-082513/plots/tplot/dtb.txt
                            dtb [2025-06-02-082516]
             ┌─────────────────────────────────────────────────────┐
        664.0┤                █████                                │
             │                █████                                │
        553.3┤                █████                                │
             │                █████                                │
             │                █████                                │
        442.7┤                █████                                │
             │                █████                                │
        332.0┤                █████                                │
             │                █████                                │
        221.3┤                █████                                │
             │                █████                                │
             │                ██████████                           │
        110.7┤     █████████████████████                           │
             │     █████████████████████                           │
          0.0┤██████████████████████████ ██████████     ███████████│
             └┬────────────┬────────────┬────────────┬────────────┬┘
            0.00350      0.00403      0.00456      0.00510   0.00563
        freq                           dtb
        text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-06-02-082513/2025-06-02-082513/History-2025-06-02-082513/plots/tplot/dtb-hist.txt
        [2025-06-02 08:25:16,757339][I][ezpz/utils:224] Saving dataset to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-06-02-082513/2025-06-02-082513/History-2025-06-02-082513/dataset_dataset.h5
        [2025-06-02 08:25:16,769431][I][examples/minimal:103:__main__] dataset=<xarray.Dataset> Size: 47kB
        Dimensions:  (draw: 989)
        Coordinates:
        * draw     (draw) int64 8kB 0 1 2 3 4 5 6 7 ... 982 983 984 985 986 987 988
        Data variables:
            iter     (draw) int64 8kB 11 12 13 14 15 16 17 ... 994 995 996 997 998 999
            loss     (draw) float64 8kB 1.031e+03 898.9 861.3 ... 673.5 680.4 678.1
            dt       (draw) float64 8kB 0.005432 0.005025 0.005267 ... 0.005351 0.005353
            dtf      (draw) float64 8kB 0.000955 0.000986 0.000986 ... 0.001077 0.001111
            dtb      (draw) float64 8kB 0.004477 0.004039 0.004281 ... 0.004274 0.004242
        wandb:
        wandb: You can sync this run to the cloud by running:
        wandb: wandb sync /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/wandb/offline-run-20250602_082455-err2dwwn
        wandb: Find logs at: ../../../../../../lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/wandb/offline-run-20250602_082455-err2dwwn/logs
        Application 51803e72 resources: utime=1016s stime=189s maxrss=3923136KB inblock=509002 oublock=2760 minflt=10027248 majflt=27746 nvcsw=558010 nivcsw=1523810
        [2025-06-02 08:25:19,307273][I][ezpz/launch:201] Execution finished @ 2025-06-02-082519
        [2025-06-02 08:25:19,308393][I][ezpz/launch:202] Command took 38.44 seconds to run. Exiting.
        took: 0h:00m:50s
        ```

    </details>

    😎 2 ez.

## 🧑‍💻 Hands On

- See my recent talk on:
  [**_LLMs on Aurora_: Hands On with `ezpz`**](https://saforem2.github.io/ezpz/slides-2025-05-07/)
  for a detailed walk-through containing examples and use cases.

  - [🎥 YouTube](https://www.youtube.com/watch?v=15ZK9REQiBo)
  - [Slides (html)](https://samforeman.me/talks/incite-hackathon-2025/ezpz/)
  - [Slides (reveal.js)](https://samforeman.me/talks/incite-hackathon-2025/ezpz/slides)
