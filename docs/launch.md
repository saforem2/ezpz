# 🍋 `ezpz.launch`: Launching Distributed Training with Python

## 🐣 Getting Started

This PR adds a mechanism for `launch`-ing distributed training using directly from `python`.

- In particular, it will use the default "launcher" depending on availability:

  - ALCF (PBS Pro): `mpiexec`
  - Slurm: `srun`
  - Unknown: `mpirun`

  and automatically pull in the specifics about the currently active job when
  building the appropriate.

For example, on any of the ALCF systems, it will automatically:

- Identify `$"{PBS_NODEFILE}"` (by looking at `hostname` of currently active node)
- Use this to calculate:
	- `NHOSTS`
	- `NGPUS_PER_HOST`
	- `WORLD_SIZE` `= NGPUS = NHOSTS * NGPUS_PER_HOST`
- With this information, we can construct the full `mpiexec ...` command needed to launch our distributed application:


	```bash
	; python3 -c 'import ezpz.pbs; print(ezpz.pbs.build_launch_cmd())'
	mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/3773945.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 16 /lus/flare/projects/datascience/foremans/projects/saforem2/mmm/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/python3
	```

## 📓 Example\[s\]

### 🧰 Setup

- Setup environment (load modules, activate base (`conda`) environment):

	```bash
	source <(curl -s https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh) && ezpz_setup_env
	```

- Install 🍋 `ezpz`:

	```bash
	python3 -m pip install "git+https://github.com/saforem2/ezpz"
	```

### 🌌 Aurora

```bash
python3 -m ezpz.launch -m ezpz.test_dist
```

<details closed><summary>Output:</summary>

```bash
[2025-03-31 11:54:24][I][ezpz/launch:42:__main__] Job ID: 3773945
[2025-03-31 11:54:24][I][ezpz/launch:46:__main__] Node file: /var/spool/pbs/aux/3773945.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
[2025-03-31 11:54:24][I][ezpz/launch:54:__main__] Evaluating:
'mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/3773945.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 16 /lus/flare/projects/datascience/foremans/projects/saforem2/mmm/venvs/aurora_nre_models_frameworks-2024.2.1_u1/bin/python3 -m ezpz.test_dist'
Disabling local launch: multi-node application
Connected to tcp://x4407c6s2b0n0.hostmgmt2407.cm.aurora.alcf.anl.gov:7919
Launching application cfb76f19-848e-4243-acb4-0bb7f4a3a768
2025-03-31 11:54:28.421437: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2025-03-31 11:54:34.188901: I itex/core/wrapper/itex_gpu_wrapper.cc:38] Intel Extension for Tensorflow* GPU backend is loaded.
2025-03-31 11:54:34.505707: I itex/core/devices/gpu/itex_gpu_runtime.cc:155] number of sub-devices is zero, expose root device.
2025-03-31 11:54:34.681180: W external/local_tsl/tsl/lib/monitoring/collection_registry.cc:81] Trying to register 2 metrics with the same name: /tensorflow/core/bfc_allocator_delay. The old value will be erased in order to register a new one. Please check if you link the metric more than once, or if the name is already used by other metrics.
2025-03-31 11:54:35.011103: I itex/core/wrapper/itex_gpu_wrapper.cc:38] Intel Extension for Tensorflow* GPU backend is loaded.
2025-03-31 11:54:35.176434: I itex/core/devices/gpu/itex_gpu_runtime.cc:130] Selected platform: Intel(R) Level-Zero
[2025-03-31 11:54:38,990] [INFO] [real_accelerator.py:222:get_accelerator] Setting ds_accelerator to xpu (auto detect)
[2025-03-31 11:54:48][I][ezpz/dist:558] Using get_torch_device_type()='xpu' with backend='ccl'
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s7b0n0'][22/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s7b0n0'][20/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s7b0n0'][17/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s7b0n0'][16/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s7b0n0'][15/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s7b0n0'][13/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s7b0n0'][19/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s7b0n0'][12/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s7b0n0'][21/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s7b0n0'][23/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s7b0n0'][14/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s7b0n0'][18/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s2b0n0'][ 2/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s2b0n0'][10/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s2b0n0'][ 3/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s2b0n0'][ 8/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s2b0n0'][ 4/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s2b0n0'][ 5/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s2b0n0'][ 1/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s2b0n0'][11/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s2b0n0'][ 9/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s2b0n0'][ 6/23]
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s2b0n0'][ 7/23]
[2025-03-31 11:54:49][I][ezpz/dist:869] Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
[2025-03-31 11:54:49][I][ezpz/dist:919] ['x4407c6s2b0n0'][ 0/23]
2025:03:31-11:54:49:(168900) |CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
[2025-03-31 11:54:49][I][ezpz/test_dist:395:__main__] model=
Network(
  (layers): Sequential(
    (0): Linear(in_features=128, out_features=1024, bias=True)
    (1): Linear(in_features=1024, out_features=512, bias=True)
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): Linear(in_features=256, out_features=128, bias=True)
    (4): Linear(in_features=128, out_features=128, bias=True)
  )
)
[2025-03-31 11:55:01][I][ezpz/dist:1096] Setting up wandb from rank=0
[2025-03-31 11:55:01][I][ezpz/dist:1097] Using=WB PROJECT=ezpz.test_dist
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.19.8
wandb: Run data is saved locally in /lus/flare/projects/datascience/foremans/projects/saforem2/mmm/wandb/run-20250331_115502-2lwks867
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run clean-frog-1202
wandb: ⭐️ View project at https://wandb.ai/aurora_gpt/ezpz.test_dist
wandb: 🚀 View run at https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/2lwks867
[2025-03-31 11:55:03][I][ezpz/dist:1125] W&B RUN=[clean-frog-1202](https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/2lwks867)
[2025-03-31 11:55:03][I][ezpz/dist:300] Updating wandb.run: clean-frog-1202 config with "DIST_INFO"
[2025-03-31 11:55:03][I][ezpz/dist:1169] Running on machine='Aurora'
[2025-03-31 11:55:03][I][ezpz/test_dist:219:__main__] config:
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
[rank4]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank10]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank1]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank6]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank8]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank2]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank9]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank0]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank7]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[2025-03-31 11:55:03][I][ezpz/test_dist:192:__main__] Warmup complete at step 2
[rank3]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank17]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank22]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank23]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank15]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank16]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank20]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank13]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank14]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank12]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[2025-03-31 11:55:03][I][ezpz/test_dist:170:__main__] iter=10 loss=736.000000 dtf=0.000485 dtb=0.001316
[rank21]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank19]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank18]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[2025-03-31 11:55:03][I][ezpz/test_dist:170:__main__] iter=20 loss=636.000000 dtf=0.000481 dtb=0.001207
[2025-03-31 11:55:03][I][ezpz/test_dist:170:__main__] iter=30 loss=596.000000 dtf=0.000488 dtb=0.001180
[2025-03-31 11:55:03][I][ezpz/test_dist:170:__main__] iter=40 loss=564.000000 dtf=0.000466 dtb=0.001311
[2025-03-31 11:55:03][I][ezpz/test_dist:170:__main__] iter=50 loss=520.000000 dtf=0.000443 dtb=0.001071
[2025-03-31 11:55:03][I][ezpz/test_dist:170:__main__] iter=60 loss=486.000000 dtf=0.000460 dtb=0.001325
[2025-03-31 11:55:03][I][ezpz/test_dist:170:__main__] iter=70 loss=456.000000 dtf=0.000499 dtb=0.001180
[rank5]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[rank11]:[W reducer.cpp:69] Warning: measureDifference between two events is not supported on XPU backend! (function operator())
[2025-03-31 11:55:03][I][ezpz/test_dist:170:__main__] iter=80 loss=422.000000 dtf=0.000480 dtb=0.001297
[2025-03-31 11:55:03][I][ezpz/test_dist:170:__main__] iter=90 loss=388.000000 dtf=0.000460 dtb=0.000952
[2025-03-31 11:55:04][I][ezpz/history:704] Saving iter plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/mmm/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
[2025-03-31 11:55:04][I][ezpz/history:704] Saving loss plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/mmm/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
[2025-03-31 11:55:04][I][ezpz/history:704] Saving dtf plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/mmm/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
[2025-03-31 11:55:04][I][ezpz/history:704] Saving dtb plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/mmm/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
[2025-03-31 11:55:05][I][ezpz/history:602] Saving tplots to /lus/flare/projects/datascience/foremans/projects/saforem2/mmm/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot
                    loss [2025-03-31-115505]                
    ┌──────────────────────────────────────────────────────┐
1504┤▌                                                     │
    │▌                                                     │
1313┤▌                                                     │
    │▌                                                     │
    │▐                                                     │
1122┤▐                                                     │
    │▝▖                                                    │
 931┤ ▚                                                    │
    │ ▝▖                                                   │
 740┤  ▝▄▖                                                 │
    │    ▝▀▞▚▖▖                                            │
    │        ▝▝▀▀▄▞▄▄▄▄▖                                   │
 549┤                  ▝▀▀▀▀▀▚▞▄▄▄▄▖ ▗                     │
    │                              ▝▀▘▀▀▀▀▚▄▄▄▖▗▖▖         │
 358┤                                         ▝▘▝▝▀▀▀▀▀▞▄▄▄│
    └─┬─┬──┬───┬──┬───┬───┬──┬───┬──┬──┬──┬────┬──┬──┬──┬──┘
    0 2 7 12  20 24  32  39 44  51 57 63 68   77 82 88 94   
loss                          iter                          
text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/mmm/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/loss.txt
                       dtf [2025-03-31-115505]              
        ┌──────────────────────────────────────────────────┐
0.000653┤                                 ▗▌               │
        │▖                                ▐▌               │
0.000617┤▌                                ▐▌               │
        │▌                                ▐▌               │
        │▌                                ▐▙▌              │
0.000581┤▌   ▖    ▗                       ▐█▌         ▗    │
        │▌  ▐▌    █    ▟    ▗             ▐█▌         █    │
0.000545┤▌  ▐▌    █    █    █     ▖   ▗▌  ▐█▌         █    │
        │▌▖ ▐▌    █    █    █    ▐▌   ▐▌  ▐█▌    ▖    █    │
0.000509┤▜▐▗▟▌    █    █    █    ▐▌   ▐▌  ▐█▚   ▐▌    █    │
        │ ▝▌█▌▗▖▖ ▛▄▖▄ █    █▗   ▐▌   ▐▌  ▐▜▝▖  ▐▌    ▛▖   │
        │   ▜▙▘▜▌ ▌ ▜▐▗▘▚▗▄▌██▞▖▖▌▌▄▟▗█▌▄▗▟  ▝▙▚▐▌    ▌▐▞▌ │
0.000473┤    ▝  ▌▞    ▀ ▝▌▜▝▛▌▘▜▐▌▝▐█▌█▜▐▌▜   ▜ ▘▌    ▌ ▘▌ │
        │       ▝               ▐▌  ▘▘▝  ▘       ▝▄▞▞▄▘  ▙▀│
0.000437┤                        ▘                ▝▌ ▜   ▜ │
        └─┬─┬──┬───┬──┬──┬──┬──┬───┬──┬──┬────┬───┬──┬──┬──┘
        0 2 7 12  20 26 32 39 44  51 57 63   74  82 87 94   
dtf                             iter                        
text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/mmm/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtf.txt
                     dtf [2025-03-31-115505]                
    ┌──────────────────────────────────────────────────────┐
38.0┤           █████                                      │
    │           █████                                      │
31.7┤     ███████████                                      │
    │     ███████████                                      │
    │     ███████████                                      │
25.3┤     ███████████                                      │
    │     ███████████                                      │
19.0┤     ███████████                                      │
    │     ███████████                                      │
12.7┤     ███████████                                      │
    │████████████████                                      │
    │████████████████                                      │
 6.3┤██████████████████████                                │
    │██████████████████████████████████████                │
 0.0┤██████████████████████████████████████████████████████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘
  0.000428    0.000486      0.000545     0.000604  0.000662 
freq                           dtf                          
text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/mmm/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtf-hist.txt
                      dtb [2025-03-31-115505]               
       ┌───────────────────────────────────────────────────┐
0.00174┤                  ▗▌                               │
       │                  ▐▌                               │
0.00161┤       ▟          ▐▌                ▖              │
       │▖      █  ▟       ▐▌             ▗ ▐▌  ▗           │
       │▌      █  ▌▌      ▐▌ ▟           █ ▐▌  █           │
0.00148┤▌      █ ▗▘▌      ▐▌▐▐           █ ▐▌  ▛▖ ▖        │
       │▌     ▐▝▄▞ ▌  ▟   ▞▌▐▐          ▐ ▌▐▌ ▐ ▌▐▚   ▗    │
0.00135┤▌     ▐ ▐▌ ▌  ▛▖  ▌▌▐▝▖       ▖ ▐ ▌▌▌ ▐ ▌▐▐   █    │
       │▌  ▟  ▌ ▐▌ ▚  ▌▌  ▌▐▌ ▌  ▟   ▐▌ ▐ ▌▌▚ ▐ ▐▐▐▗  █    │
0.00122┤▐  █  ▌ ▐▌ ▐  ▌▐ ▗▘▝▌ ▚  █   ▐▚ ▌ ▌▌▝▖▐ ▐▌▐█  █    │
       │ ▚▚▘▚▄▘  ▘ ▝▖▞▌▝▄▀    ▝▖▗▀▖ ▖▌▝▖▌ ▌▌ ▐▌ ▐▌ ▜  █  ▗▌│
       │            █          ▝█ ▝▀▝▘ ▝  ▝   ▘  ▘ ▐  █▗▚▐▌│
0.00108┤            ▝           ▜                   ▜ ▛▌ █▌│
       │                                            ▐ ▌  ▜▌│
0.00095┤                                             ▚▌   ▝│
       └─┬─┬──┬───┬──┬──┬───┬──┬──┬──┬──┬──┬──┬───┬──┬──┬──┘
       0 2 7 12  20 26 32  39 46 51 57 63 68 74  82 87 94   
dtb                            iter                         
text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/mmm/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtb.txt
                     dtb [2025-03-31-115505]                
    ┌──────────────────────────────────────────────────────┐
29.0┤           █████                                      │
    │           █████                                      │
24.2┤           █████                                      │
    │           █████                                      │
    │           ███████████                                │
19.3┤           ███████████                                │
    │           ███████████                                │
14.5┤           ███████████                                │
    │           ████████████████                           │
 9.7┤           ████████████████                           │
    │           ████████████████                           │
    │     ██████████████████████████████████████           │
 4.8┤███████████████████████████████████████████           │
    │█████████████████████████████████████████████████     │
 0.0┤██████████████████████████████████████████████████████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘
  0.00092      0.00113       0.00135      0.00156   0.00178 
freq                           dtb                          
text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/mmm/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtb-hist.txt
[2025-03-31 11:55:05][I][ezpz/utils:192] Saving dataset to: /lus/flare/projects/datascience/foremans/projects/saforem2/mmm/outputs/ezpz.test_dist/ezpz.test_dist/train_dataset.h5
[2025-03-31 11:55:05][I][ezpz/test_dist:186:__main__] dataset=<xarray.Dataset> Size: 3kB
Dimensions:  (draw: 97)
Coordinates:
  * draw     (draw) int64 776B 0 1 2 3 4 5 6 7 8 ... 88 89 90 91 92 93 94 95 96
Data variables:
    iter     (draw) int64 776B 3 4 5 6 7 8 9 10 11 ... 92 93 94 95 96 97 98 99
    loss     (draw) float32 388B 1.504e+03 1.184e+03 996.0 ... 374.0 364.0 358.0
    dtf      (draw) float64 776B 0.000628 0.0005136 ... 0.0004577 0.0004578
    dtb      (draw) float64 776B 0.00156 0.001243 ... 0.001199 0.0009906
[2025-03-31 11:55:05][I][ezpz/test_dist:459:__main__] Took: 16.20 seconds
wandb:
wandb: 🚀 View run clean-frog-1202 at: https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/2lwks867
wandb: Find logs at: ../../../../../../lus/flare/projects/datascience/foremans/projects/saforem2/mmm/wandb/run-20250331_115502-2lwks867/logs
Application cfb76f19 resources: utime=960s stime=158s maxrss=3506048KB inblock=149990 oublock=1112 minflt=8373858 majflt=24577 nvcsw=272433 nivcsw=8376029
```

</details>

### ⭐ Polaris


```bash
python3 -m ezpz.launch -m ezpz.test_dist
```

<details closed><summary>Output:</summary>

```bash
[2025-03-31 11:54:04][I][ezpz/launch:42:__main__] Job ID: 4093576
[2025-03-31 11:54:04][I][ezpz/launch:46:__main__] Node file: /var/spool/pbs/aux/4093576.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov
[2025-03-31 11:54:04][I][ezpz/launch:54:__main__] Evaluating:
'mpiexec --verbose --envall -n 8 -ppn 4 --hostfile /var/spool/pbs/aux/4093576.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov --cpu-bind depth -d 16 /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/venvs/2024-04-29/bin/python3 -m ezpz.test_dist'
Connected to tcp://x3006c0s31b0n0.hsn.cm.polaris.alcf.anl.gov:7919
Launching application 2d56c0c9-62cf-4b97-888b-23830c5d836e
Using PMI port 59675,59676
[2025-03-31 11:54:09,471] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-03-31 11:54:09,505] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-03-31 11:54:09,513] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-03-31 11:54:09,518] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-03-31 11:54:11,622] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-03-31 11:54:11,640] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-03-31 11:54:11,650] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-03-31 11:54:11,660] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.3
 [WARNING]  using untested triton version (2.3.0), only 1.0.0 is known to be compatible
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.3
 [WARNING]  using untested triton version (2.3.0), only 1.0.0 is known to be compatible
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.3
 [WARNING]  using untested triton version (2.3.0), only 1.0.0 is known to be compatible
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.3
 [WARNING]  using untested triton version (2.3.0), only 1.0.0 is known to be compatible
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.3
 [WARNING]  using untested triton version (2.3.0), only 1.0.0 is known to be compatible
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.3
 [WARNING]  using untested triton version (2.3.0), only 1.0.0 is known to be compatible
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.3
 [WARNING]  using untested triton version (2.3.0), only 1.0.0 is known to be compatible
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.3
 [WARNING]  using untested triton version (2.3.0), only 1.0.0 is known to be compatible
[2025-03-31 11:54:16][I][ezpz/dist:558] Using get_torch_device_type()='cuda' with backend='nccl'
[2025-03-31 11:54:16][I][ezpz/dist:919] ['x3006c0s13b0n0'][4/7]
[2025-03-31 11:54:16][I][ezpz/dist:869] Using device='cuda' with backend='DDP' + 'nccl' for distributed training.
[2025-03-31 11:54:16][I][ezpz/dist:919] ['x3006c0s31b0n0'][0/7]
[2025-03-31 11:54:17][I][ezpz/dist:919] ['x3006c0s31b0n0'][1/7]
[2025-03-31 11:54:17][I][ezpz/dist:919] ['x3006c0s31b0n0'][3/7]
[2025-03-31 11:54:17][I][ezpz/dist:919] ['x3006c0s31b0n0'][2/7]
[2025-03-31 11:54:17][I][ezpz/dist:919] ['x3006c0s13b0n0'][5/7]
[2025-03-31 11:54:17][I][ezpz/dist:919] ['x3006c0s13b0n0'][6/7]
[2025-03-31 11:54:17][I][ezpz/dist:919] ['x3006c0s13b0n0'][7/7]
[2025-03-31 11:54:18][I][ezpz/test_dist:395:__main__] model=
Network(
  (layers): Sequential(
    (0): Linear(in_features=128, out_features=1024, bias=True)
    (1): Linear(in_features=1024, out_features=512, bias=True)
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): Linear(in_features=256, out_features=128, bias=True)
    (4): Linear(in_features=128, out_features=128, bias=True)
  )
)
[2025-03-31 11:54:18][I][ezpz/dist:1096] Setting up wandb from rank=0
[2025-03-31 11:54:18][I][ezpz/dist:1097] Using=WB PROJECT=ezpz.test_dist
wandb: Currently logged in as: foremans (aurora_gpt). Use `wandb login --relogin` to force relogin
wandb: wandb version 0.19.8 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.16.6
wandb: Run data is saved locally in /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20250331_115420-jtg00dii
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run silver-forest-1201
wandb: ⭐️ View project at https://wandb.ai/aurora_gpt/ezpz.test_dist
wandb: 🚀 View run at https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/jtg00dii
[2025-03-31 11:54:21][I][ezpz/dist:1125] W&B RUN=[silver-forest-1201](https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/jtg00dii)
[2025-03-31 11:54:21][I][ezpz/dist:300] Updating wandb.run: silver-forest-1201 config with "DIST_INFO"
[2025-03-31 11:54:21][I][ezpz/dist:1169] Running on machine='Polaris'
[2025-03-31 11:54:21][I][ezpz/test_dist:219:__main__] config:
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
[2025-03-31 11:54:21][I][ezpz/test_dist:192:__main__] Warmup complete at step 2
[2025-03-31 11:54:21][I][ezpz/test_dist:170:__main__] iter=10 loss=736.000000 dtf=0.000359 dtb=0.000730
[2025-03-31 11:54:21][I][ezpz/test_dist:170:__main__] iter=20 loss=660.000000 dtf=0.000330 dtb=0.000712
[2025-03-31 11:54:21][I][ezpz/test_dist:170:__main__] iter=30 loss=604.000000 dtf=0.000319 dtb=0.000730
[2025-03-31 11:54:21][I][ezpz/test_dist:170:__main__] iter=40 loss=572.000000 dtf=0.000341 dtb=0.000726
[2025-03-31 11:54:21][I][ezpz/test_dist:170:__main__] iter=50 loss=532.000000 dtf=0.000321 dtb=0.000714
[2025-03-31 11:54:21][I][ezpz/test_dist:170:__main__] iter=60 loss=504.000000 dtf=0.000327 dtb=0.000713
[2025-03-31 11:54:21][I][ezpz/test_dist:170:__main__] iter=70 loss=474.000000 dtf=0.000326 dtb=0.000751
[2025-03-31 11:54:21][I][ezpz/test_dist:170:__main__] iter=80 loss=426.000000 dtf=0.000332 dtb=0.000746
[2025-03-31 11:54:21][I][ezpz/test_dist:170:__main__] iter=90 loss=408.000000 dtf=0.000328 dtb=0.000785
[2025-03-31 11:54:21][I][ezpz/history:704] Saving iter plot to: /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
[2025-03-31 11:54:22][I][ezpz/history:704] Saving loss plot to: /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
[2025-03-31 11:54:22][I][ezpz/history:704] Saving dtf plot to: /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
[2025-03-31 11:54:22][I][ezpz/history:704] Saving dtb plot to: /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/mplot
[2025-03-31 11:54:22][I][ezpz/history:602] Saving tplots to /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot
                    loss [2025-03-31-115422]
    ┌──────────────────────────────────────────────────────┐
1496┤▌                                                     │
    │▌                                                     │
1310┤▌                                                     │
    │▚                                                     │
    │▐                                                     │
1124┤▐                                                     │
    │ ▌                                                    │
 938┤ ▚                                                    │
    │ ▐                                                    │
 752┤  ▚▄                                                  │
    │    ▀▀▄▄▖                                             │
    │        ▝▀▀▀▄▚▄▄▄▄                                    │
 566┤                  ▀▀▀▀▄▚▄▄▄▄▗▖                        │
    │                            ▘▝▀▀▀▀▚▞▚▄▄▄▄▄▖▗          │
 380┤                                          ▝▘▀▀▀▀▀▚▄▞▞▄│
    └──┬─┬───┬────┬──┬──┬──┬───┬───┬──┬──┬───┬──┬───┬────┬─┘
    0  4 9  15   25 30 36 40  48  55 60 66  73 78  86   96
loss                          iter
text saved in /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/loss.txt
                       dtf [2025-03-31-115422]
        ┌──────────────────────────────────────────────────┐
0.000467┤▌                                                 │
        │▌                                                 │
0.000442┤▌                                                 │
        │▌                                                 │
        │▌  ▗▌                                             │
0.000417┤▌  ▐▌    ▗    ▗    ▗          ▖    ▖              │
        │▌  ▐▌    █    █    █    ▗▌   ▐▌   ▐▌   ▗▌         │
0.000392┤▌  ▐▌    █    █    █    ▐▌   ▐▌   ▐▌   ▐▌    ▟    │
        │▌  ▐▌    █    █    █    ▐▌   ▐▌   ▐▌   ▐▌    █    │
0.000367┤▌ ▟▐▌    █    █    █    ▐▌   ▐▌   ▐▌   ▐▌    █    │
        │▝▀█▐▌    █    █    █    ▞▌   ▐▌   ▐▌   ▐▌ ▗  █    │
        │   ▘▌    █    ▛▄▌▖▖█    ▌▌   ▐▌   ▐▌   ▐▌ █  █    │
0.000342┤    ▌    █    ▌▝▝▝▚▜▗   ▌▌   ▐▌   ▐▌▖  ▐▚ █  ▛▖   │
        │    ▝▖ ▗ ▌▌▄ ▗▌    ▝█▗▚ ▌▌▗▞▟▐▌▗ ▗▟▜▐▗▌▞ ▚▛▖▖▌▌ ▄▖│
0.000317┤     ▝▞▘▀ ▝ ▀▀▌      ▘▝▄▘▝▘  ▘▝▘▀▘   ▜▝   ▘▝▝ ▚▞ ▝│
        └──┬─┬──┬────┬──┬──┬──┬──┬──┬──┬────┬───┬───┬────┬─┘
        0  4 9 15   25 31 36 43 48 54 60   69  78  86   96
dtf                             iter
text saved in /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtf.txt
                     dtf [2025-03-31-115422]
    ┌──────────────────────────────────────────────────────┐
63.0┤█████                                                 │
    │█████                                                 │
52.5┤█████                                                 │
    │█████                                                 │
    │█████                                                 │
42.0┤█████                                                 │
    │█████                                                 │
31.5┤█████                                                 │
    │█████                                                 │
21.0┤█████                                                 │
    │█████                                                 │
    │███████████                                           │
10.5┤████████████████                                      │
    │██████████████████████     ███████████                │
 0.0┤██████████████████████     ████████████████      █████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘
  0.000311    0.000351      0.000392     0.000433  0.000474
freq                           dtf
text saved in /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtf-hist.txt
                       dtb [2025-03-31-115422]
        ┌──────────────────────────────────────────────────┐
0.000807┤                                           ▗▌     │
        │                                           ▐▌     │
0.000790┤         ▗                              ▞▄ ▐▌  ▖  │
        │         █                              ▌▐▗▐▙▌▐▌  │
        │▌        █                              ▌▝█▐█▌▐▌  │
0.000773┤▐        █                          ▖   ▌ ▐▐█▚▌▌  │
        │▐ ▗▌     █     ▖                   ▞▌ ▗▌▌ ▐▐█▐▌▌  │
0.000756┤▐ ▐▌▖    █    ▐▌ ▖▖               ▗▘▌ ▐▌▌ ▐▐█▝▌▌  │
        │▐ ▐█▌    █    ▐▌▐█▌   ▞▖▗▌ ▗  ▖   ▞ ▌▟▐▚▘ ▐▐▝  ▌  │
0.000739┤▐ ▞█▌    █    ▌▌▟█▌   ▌▌▐▌ █ ▐▌   ▌ ▙▘▀   ▐▌   ▌  │
        │▝▖▌█▌ ▖  █    ▌▝▝▝▐  ▟▌▌▐▌▖█ ▐▌▟ ▐  █     ▝▌   ▌▗ │
        │ ▜ ▝▚▐▌▖ ▌▌  ▗▘   ▝▄▟█▌▌▌▜▌█▟▐▌█ ▐  ▝          ▌█ │
0.000722┤    ▐▌█▌ ▌▌▗ ▐      ▜█▌▌▌ █▐█▐▜ ▚▀             ▚▀▖│
        │    ▝▌█▌▗▌▌▌▜▐       ▘▘▝▌ █▝▛▟                   ▝│
0.000705┤      ▜▝▘ ▚▘ ▀            ▜                       │
        └──┬─┬──┬────┬──┬──┬──┬──┬──┬──┬────┬───┬───┬────┬─┘
        0  4 9 15   25 31 36 43 48 54 60   69  78  86   96
dtb                             iter
text saved in /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtb.txt
                     dtb [2025-03-31-115422]
    ┌──────────────────────────────────────────────────────┐
20.0┤           █████                                      │
    │█████      █████                                      │
16.7┤█████      █████                                      │
    │█████      █████                                      │
    │█████      ████████████████                           │
13.3┤███████████████████████████                           │
    │███████████████████████████                           │
10.0┤███████████████████████████                           │
    │███████████████████████████                           │
 6.7┤███████████████████████████                           │
    │██████████████████████████████████████                │
    │██████████████████████████████████████     ██████     │
 3.3┤█████████████████████████████████████████████████     │
    │██████████████████████████████████████████████████████│
 0.0┤██████████████████████████████████████████████████████│
    └┬────────────┬─────────────┬────────────┬────────────┬┘
  0.000701    0.000728      0.000756     0.000784  0.000811
freq                           dtb
text saved in /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/plots/tplot/dtb-hist.txt
[2025-03-31 11:54:22][I][ezpz/utils:192] Saving dataset to: /lus/eagle/projects/datascience/foremans/projects/saforem2/ezpz/outputs/ezpz.test_dist/ezpz.test_dist/train_dataset.h5
[2025-03-31 11:54:23][I][ezpz/test_dist:186:__main__] dataset=<xarray.Dataset> Size: 3kB
Dimensions:  (draw: 97)
Coordinates:
  * draw     (draw) int64 776B 0 1 2 3 4 5 6 7 8 ... 88 89 90 91 92 93 94 95 96
Data variables:
    iter     (draw) int64 776B 3 4 5 6 7 8 9 10 11 ... 92 93 94 95 96 97 98 99
    loss     (draw) float32 388B 1.496e+03 1.208e+03 1.04e+03 ... 380.0 390.0
    dtf      (draw) float64 776B 0.0004672 0.0003655 ... 0.000327 0.0003201
    dtb      (draw) float64 776B 0.0007786 0.0007711 ... 0.00072 0.0007158
[2025-03-31 11:54:23][I][ezpz/test_dist:459:__main__] Took: 6.70 seconds
wandb: - 0.008 MB of 0.008 MB uploaded^Mwandb:
wandb: Run history:
wandb:  dtb █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:  dtf █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: iter ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb: loss █▄▃▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:
wandb: Run summary:
wandb:  dtb 0.00072
wandb:  dtf 0.00032
wandb: iter 99
wandb: loss 390.0
wandb:
wandb: 🚀 View run silver-forest-1201 at: https://wandb.ai/aurora_gpt/ezpz.test_dist/runs/jtg00dii
wandb: ⭐️ View project at: https://wandb.ai/aurora_gpt/ezpz.test_dist
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20250331_115420-jtg00dii/logs
Application 2d56c0c9 resources: utime=92s stime=79s maxrss=1821516KB inblock=400 oublock=2256 minflt=1873822 majflt=1004 nvcsw=377844 nivcsw=926336
```

</details>
