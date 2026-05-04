# 👣 End-to-End Walkthrough

What a full `ezpz` run actually looks like — from script to terminal
output, plots, and saved artifacts. For setup instructions, see the
[Quick Start](./quickstart.md). For launcher usage and flags, see the
[`ezpz launch` CLI reference](./cli/launch/index.md).

## ✅ Full Example with History

Capture metrics across all ranks, persist JSONL, generate text/PNG plots, and
(when configured) log to Weights & Biases—no extra code on worker ranks.
The `History` class aggregates distributed statistics (min/max/mean/std) and
produces terminal-friendly plots automatically via `finalize()`.


```python title="example.py"
import ezpz
import torch

from ezpz.models.minimal import SequentialLinearNet  # multi-layer Linear+ReLU network

import time

logger = ezpz.get_logger(__name__)

rank = ezpz.setup_torch()
device = ezpz.get_torch_device()
model = SequentialLinearNet(
    input_dim=16,
    output_dim=32,
    sizes=[4, 8, 12]
)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters())

history = ezpz.History()

# Fixed input and target so loss converges over training
batch = torch.randn(1, 16, device=device)
target = torch.randn(1, 32, device=device)

for i in range(10):
    t0 = time.perf_counter()
    output = model(batch)
    loss = ((output - target) ** 2).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    logger.info(
        history.update(
            {
                "iter": i,
                "loss": loss.item(),
                "dt": time.perf_counter() - t0,
            },
            step=i,
        )
    )

if rank == 0:
    history.finalize()

ezpz.cleanup()
```

!!! note "Swap in your own model"

    [`SequentialLinearNet`](./python/Code-Reference/models/minimal.md#ezpz.models.minimal.SequentialLinearNet)
    is a small multi-layer Linear+ReLU network included
    for demonstration. Replace it with any `torch.nn.Module` — the rest of
    the script (setup, wrapping, training loop, history) stays the same.

??? info "🪵 Logs"

    ??? success "Single Process"

        Launching in a single process via `python`:

        ```text
        > python3 example.py
        [2026-01-15 16:29:59,463919][I][ezpz/distributed:569:setup_torch] Using device=mps with backend=gloo
        [2026-01-15 16:29:59,475974][I][ezpz/distributed:1525:_setup_ddp] Caught MASTER_PORT=61496 from environment!
        [2026-01-15 16:29:59,477538][I][ezpz/distributed:1530:_setup_ddp] Using torch.distributed.init_process_group with
        - master_addr='Sams-MacBook-Pro-2.local'
        - master_port='61496'
        - world_size=1
        - rank=0
        - local_rank=0
        - timeout=datetime.timedelta(seconds=3600)
        - backend='gloo'
        [2026-01-15 16:29:59,478263][I][ezpz/distributed:1536:_setup_ddp] init_process_group: rank=0 world_size=1 backend=gloo
        [2026-01-15 16:29:59,789459][I][ezpz/distributed:569:setup_torch] Using device='mps' with backend='gloo' + 'gloo' for distributed training.
        [2026-01-15 16:29:59,872685][W][ezpz/distributed:1043:print_dist_setup] Using [1 / 1] available "mps" devices !!
        [2026-01-15 16:29:59,873382][I][ezpz/distributed:578:setup_torch] ['Sams-MacBook-Pro-2.local'][device='mps'][node=0/0][rank=0/0][local_rank=0/0]
        [2026-01-15 16:30:01,875023][I][ezpz/history:214:__init__] Not using distributed metrics! Will only be tracked from a single rank...
        [2026-01-15 16:30:01,875595][I][ezpz/history:220:__init__] Using History with distributed_history=False
        [2026-01-15 16:30:02,316946][I][ezpz/example:30:<module>] iter=0 loss=31.003010 dt=0.435792
        [2026-01-15 16:30:02,330593][I][ezpz/example:30:<module>] iter=1 loss=57.543598 dt=0.008874
        [2026-01-15 16:30:02,337684][I][ezpz/example:30:<module>] iter=2 loss=28.547897 dt=0.003079
        [2026-01-15 16:30:02,346325][I][ezpz/example:30:<module>] iter=3 loss=22.243866 dt=0.002852
        [2026-01-15 16:30:02,353276][I][ezpz/example:30:<module>] iter=4 loss=25.085716 dt=0.003102
        [2026-01-15 16:30:02,359662][I][ezpz/example:30:<module>] iter=5 loss=27.327484 dt=0.002849
        [2026-01-15 16:30:02,364890][I][ezpz/example:30:<module>] iter=6 loss=19.950121 dt=0.003308
        [2026-01-15 16:30:02,371596][I][ezpz/example:30:<module>] iter=7 loss=36.892731 dt=0.005253
        [2026-01-15 16:30:02,378344][I][ezpz/example:30:<module>] iter=8 loss=28.500504 dt=0.002372
        [2026-01-15 16:30:02,384270][I][ezpz/example:30:<module>] iter=9 loss=33.020760 dt=0.002239
        /Users/samforeman/vibes/saforem2/ezpz/src/ezpz/history.py:2223: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
        Consider using tensor.detach() first. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:837.)
        x = torch.Tensor(x).numpy(force=True)
        [2026-01-15 16:30:02,458225][I][ezpz/history:2385:finalize] Saving plots to /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-163002/2026-01-15-163002/plots/mplot (matplotlib) and /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-163002/2026-01-15-163002/plots/tplot (tplot)
        [2026-01-15 16:30:03,822720][I][ezpz/tplot:321:tplot] Using plot type: line
        [2026-01-15 16:30:03,823148][I][ezpz/tplot:323:tplot] Using plot marker: hd
                                 dt vs iter                       
             ┌─────────────────────────────────────────────────────┐
        0.436┤▌                                                    │
             │▚                                                    │
        0.364┤▝▖                                                   │
             │ ▌                                                   │
             │ ▐                                                   │
        0.291┤  ▌                                                  │
             │  ▚                                                  │
        0.219┤  ▝▖                                                 │
             │   ▚                                                 │
        0.147┤   ▐                                                 │
             │    ▌                                                │
             │    ▐                                                │
        0.074┤    ▝▖                                               │
             │     ▚                                               │
        0.002┤     ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
             └┬─────┬─────┬────┬─────┬─────┬─────┬────┬─────┬──────┘
        0     1     2     3    4     5     6     7    8     9       
        dt                            iter                          
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-163002/2026-01-15-163002/plots/tplot/dt.txt
        [2026-01-15 16:30:03,827907][I][ezpz/tplot:321:tplot] Using plot type: hist
        [2026-01-15 16:30:03,828187][I][ezpz/tplot:323:tplot] Using plot marker: hd
                                freq vs dt                        
           ┌───────────────────────────────────────────────────────┐
        9.0┤█████                                                  │
           │█████                                                  │
        7.5┤█████                                                  │
           │█████                                                  │
           │█████                                                  │
        6.0┤█████                                                  │
           │█████                                                  │
        4.5┤█████                                                  │
           │█████                                                  │
        3.0┤█████                                                  │
           │█████                                                  │
           │█████                                                  │
        1.5┤█████                                             █████│
           │█████                                             █████│
        0.0┤█████                                             █████│
           └┬─────────────┬────────────┬─────────────┬────────────┬┘
           -0.02         0.10         0.22          0.34        0.46 
        freq                          dt                            
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-163002/2026-01-15-163002/plots/tplot/dt-hist.txt
        [2026-01-15 16:30:03,833010][I][ezpz/tplot:321:tplot] Using plot type: line
        [2026-01-15 16:30:03,833296][I][ezpz/tplot:323:tplot] Using plot marker: hd
                                loss vs iter                      
            ┌──────────────────────────────────────────────────────┐
        57.5┤     ▗▌                                               │
            │     ▌▐                                               │
        51.3┤    ▐  ▌                                              │
            │   ▗▘  ▐                                              │
            │   ▞    ▌                                             │
        45.0┤  ▗▘    ▝▖                                            │
            │  ▌      ▚                                            │
        38.7┤ ▐       ▝▖                                           │
            │▗▘        ▚                              ▞▄           │
        32.5┤▞         ▝▖                            ▞  ▀▄        ▗│
            │▘          ▚                           ▞     ▀▄  ▗▄▞▀▘│
            │            ▚▖               ▗        ▞        ▀▀▘    │
        26.2┤             ▝▚▄        ▄▄▄▀▀▘▀▄     ▞                │
            │                ▀▄▄▄▄▀▀▀        ▀▄  ▞                 │
        20.0┤                                  ▀▟                  │
            └┬─────┬─────┬─────┬─────┬────┬─────┬─────┬─────┬──────┘
            1     2     3     4     5    6     7     8     9       
        loss                          iter                          
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-163002/2026-01-15-163002/plots/tplot/loss.txt
        [2026-01-15 16:30:03,837141][W][ezpz/history:2420:finalize] h5py not found! Saving dataset as netCDF instead.
        [2026-01-15 16:30:03,837503][I][utils/__init__:636:save_dataset] Saving dataset to: /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-163002/2026-01-15-163002/dataset_dataset.nc
        [2026-01-15 16:30:03,885343][I][ezpz/history:2433:finalize] Saving history report to /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-163002/2026-01-15-163002/report.md
        >
        ```


    ??? success "`ezpz launch`"

        Launching via `ezpz launch` (fallback with 2 processes on MacBookPro):

        ```text
        > ezpz launch python3 /tmp/test.py
        [2026-01-15 16:25:45,611138][I][ezpz/launch:515:run] No active scheduler detected; falling back to local mpirun: mpirun -np 2 python3 /tmp/test.py
        [2026-01-15 16:25:47,138854][I][ezpz/distributed:569:setup_torch] Using device=mps with backend=gloo
        [2026-01-15 16:25:47,149140][I][ezpz/distributed:1525:_setup_ddp] Caught MASTER_PORT=60839 from environment!
        [2026-01-15 16:25:47,150476][I][ezpz/distributed:1530:_setup_ddp] Using torch.distributed.init_process_group with
        - master_addr='Sams-MacBook-Pro-2.local'
        - master_port='60839'
        - world_size=2
        - rank=0
        - local_rank=0
        - timeout=datetime.timedelta(seconds=3600)
        - backend='gloo'
        [2026-01-15 16:25:47,151050][I][ezpz/distributed:1536:_setup_ddp] init_process_group: rank=0 world_size=2 backend=gloo
        [2026-01-15 16:25:47,242104][I][ezpz/distributed:569:setup_torch] Using device='mps' with backend='gloo' + 'gloo' for distributed training.
        [2026-01-15 16:25:47,261869][I][ezpz/distributed:578:setup_torch] ['Sams-MacBook-Pro-2.local'][device='mps'][node=0/0][rank=1/1][local_rank=1/1]
        [2026-01-15 16:25:47,289930][W][ezpz/distributed:1043:print_dist_setup] Using [2 / 2] available "mps" devices !!
        [2026-01-15 16:25:47,290348][I][ezpz/distributed:578:setup_torch] ['Sams-MacBook-Pro-2.local'][device='mps'][node=0/0][rank=0/1][local_rank=0/1]
        [2026-01-15 16:25:48,882995][I][ezpz/history:220:__init__] Using History with distributed_history=True
        [2026-01-15 16:25:49,293872][I][tmp/test:30:<module>] iter=0 loss=14.438349 dt=0.383613 loss/mean=18.930481 loss/max=23.422613 loss/min=14.438349 loss/std=4.492133 dt/mean=0.383651 dt/max=0.383690 dt/min=0.383613 dt/std=0.000000
        [2026-01-15 16:25:49,310545][I][tmp/test:30:<module>] iter=1 loss=38.289841 dt=0.006327 loss/mean=37.768768 loss/max=38.289841 loss/min=37.247700 loss/std=0.521159 dt/mean=0.006445 dt/max=0.006563 dt/min=0.006327 dt/std=0.000118
        [2026-01-15 16:25:49,323389][I][tmp/test:30:<module>] iter=2 loss=15.649942 dt=0.003752 loss/mean=26.894470 loss/max=38.138996 loss/min=15.649942 loss/std=11.244525 dt/mean=0.003934 dt/max=0.004116 dt/min=0.003752 dt/std=0.000182
        [2026-01-15 16:25:49,335400][I][tmp/test:30:<module>] iter=3 loss=21.518583 dt=0.006340 loss/mean=38.892834 loss/max=56.267082 loss/min=21.518583 loss/std=17.374252 dt/mean=0.006604 dt/max=0.006869 dt/min=0.006340 dt/std=0.000264
        [2026-01-15 16:25:49,343467][I][tmp/test:30:<module>] iter=4 loss=43.398060 dt=0.003205 loss/mean=41.371902 loss/max=43.398060 loss/min=39.345749 loss/std=2.026196 dt/mean=0.002617 dt/max=0.003205 dt/min=0.002029 dt/std=0.000588
        [2026-01-15 16:25:49,351912][I][tmp/test:30:<module>] iter=5 loss=43.348061 dt=0.002345 loss/mean=39.714069 loss/max=43.348061 loss/min=36.080078 loss/std=3.633997 dt/mean=0.002180 dt/max=0.002345 dt/min=0.002014 dt/std=0.000166
        [2026-01-15 16:25:49,360378][I][tmp/test:30:<module>] iter=6 loss=40.937546 dt=0.003073 loss/mean=36.756641 loss/max=40.937546 loss/min=32.575737 loss/std=4.180907 dt/mean=0.002433 dt/max=0.003073 dt/min=0.001794 dt/std=0.000640
        [2026-01-15 16:25:49,368605][I][tmp/test:30:<module>] iter=7 loss=30.643730 dt=0.002785 loss/mean=32.207088 loss/max=33.770447 loss/min=30.643730 loss/std=1.563398 dt/mean=0.002315 dt/max=0.002785 dt/min=0.001844 dt/std=0.000470
        [2026-01-15 16:25:49,377235][I][tmp/test:30:<module>] iter=8 loss=26.110786 dt=0.003046 loss/mean=33.217815 loss/max=40.324844 loss/min=26.110786 loss/std=7.107031 dt/mean=0.002361 dt/max=0.003046 dt/min=0.001676 dt/std=0.000685
        [2026-01-15 16:25:49,384409][I][tmp/test:30:<module>] iter=9 loss=22.861826 dt=0.001886 loss/mean=25.471987 loss/max=28.082148 loss/min=22.861826 loss/std=2.610158 dt/mean=0.002179 dt/max=0.002472 dt/min=0.001886 dt/std=0.000293
        /Users/samforeman/vibes/saforem2/ezpz/src/ezpz/history.py:2223: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
        Consider using tensor.detach() first. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:837.)
        x = torch.Tensor(x).numpy(force=True)
        [2026-01-15 16:25:49,455888][I][ezpz/history:2385:finalize] Saving plots to /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/mplot (matplotlib) and /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot (tplot)
                            dt                                    dt/min
             ┌─────────────────────────────────┐     ┌─────────────────────────────────┐
        0.384┤▌                                │0.384┤-                                │
        0.320┤▐                                │0.129┤ --------------------------------│
        0.256┤ ▚                               │     └┬───────┬───────┬───────┬───────┬┘
        0.129┤ ▝▖                              │     1.0     3.2     5.5     7.8   10.0 
        0.066┤  ▐                              │dt/min              iter
        0.002┤   ▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│                    dt/std
             └┬───────┬───────┬───────┬───────┬┘       ┌───────────────────────────────┐
             1.0     3.2     5.5     7.8   10.0 0.00068┤             \*      \*      \*   │
        dt                  iter                0.00046┤       \*\*\*\*\*\* \*\*   \* \*\*\*\*\*\* \*\*\*│
                        dt/mean                 0.00011┤\*\*\*\*\*\*\*         \*\*\*            │
             ┌─────────────────────────────────┐       └┬───────┬──────┬───────┬──────┬┘
        0.384┤·                                │       1.0     3.2    5.5     7.8  10.0 
        0.320┤·                                │dt/std               iter
        0.256┤ ·                               │                   dt/max
        0.193┤  ·                              │     ┌─────────────────────────────────┐
        0.129┤  ·                              │0.384┤+                                │
        0.066┤   ·                             │0.257┤ ++                              │
        0.002┤    ·····························│0.066┤   ++++++++++++++++++++++++++++++│
             └┬───────┬───────┬───────┬───────┬┘     └┬───────┬───────┬───────┬───────┬┘
            1.0     3.2     5.5     7.8   10.0      1.0     3.2     5.5     7.8   10.0 
        dt/mean             iter                dt/max              iter                
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/dt.txt
             ┌─────────────────────────────────────────────────────────────────────────┐
        0.384┤ ++ dt/max                                                               │
             │ -- dt/min                                                               │
             │ ·· dt/mean                                                              │
        0.320┤ ▞▞ dt                                                                   │
             │ ▐                                                                       │
             │  ▌                                                                      │
        0.256┤  ▚                                                                      │
             │  ▝▖                                                                     │
             │   ▌                                                                     │
        0.193┤   ▐                                                                     │
             │    ▌                                                                    │
             │    ▐                                                                    │
             │    ▝▖                                                                   │
        0.129┤     ▚                                                                   │
             │     ▐                                                                   │
             │      ▌                                                                  │
        0.065┤      ▐                                                                  │
             │      ▝▖                                                                 │
             │       ▚                                                                 │
        0.002┤       ▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
             └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
             1.0               3.2               5.5               7.8             10.0 
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/dt_summary.txt
                       dt/mean hist                             dt/max hist             
           ┌───────────────────────────────────┐   ┌───────────────────────────────────┐
        9.0┤████                               │9.0┤████                               │
        7.5┤████                               │7.5┤████                               │
        6.0┤████                               │6.0┤████                               │
        4.5┤████                               │4.5┤████                               │
        3.0┤████                               │3.0┤████                               │
        1.5┤████                           ████│1.5┤████                           ████│
        0.0┤███                            ████│0.0┤███                            ████│
           └┬────────┬───────┬────────┬───────┬┘   └┬────────┬───────┬────────┬───────┬┘
           -0.01    0.09    0.19     0.30   0.40   -0.01    0.09    0.19     0.30   0.40 
                        dt/min hist                              dt/std hist
           ┌───────────────────────────────────┐    ┌──────────────────────────────────┐
        9.0┤████                               │2.00┤       ███                    ████│
        7.5┤████                               │1.67┤       ███                    ████│
        6.0┤████                               │1.33┤       ███                    ████│
        4.5┤████                               │1.00┤█████████████████   ████   ███████│
           │████                               │    │█████████████████   ████   ███████│
        3.0┤████                               │0.67┤█████████████████   ████   ███████│
        1.5┤████                           ████│0.33┤█████████████████   ████   ███████│
        0.0┤███                            ████│0.00┤█████████████████   ████   ███████│
           └┬────────┬───────┬────────┬───────┬┘    └┬───────┬────────┬───────┬────────┘
           -0.02    0.09    0.19     0.30   0.40   -0.00003 0.00016  0.00034 0.00053     
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/dt_hist.txt
                            loss                                  loss/min              
            ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
        43.4┤              ▗▀▀▀▀▄▄▄▄           │39.3┤    -          ------------       │
        38.6┤   ▟         ▗▘        ▚▖         │22.7┤---- ----------            -------│
        33.7┤  ▞ ▚       ▗▘          ▝▚▖       │    └┬───────┬────────┬───────┬───────┬┘
        24.1┤ ▐   ▚     ▗▘             ▝▀▚▄▖   │    1.0     3.2      5.5     7.8   10.0 
        19.3┤▗▘    ▚   ▄▘                  ▝▀▀▀│loss/min            iter
        14.4┤▌      ▚▄▀                        │                  loss/std
            └┬───────┬────────┬───────┬───────┬┘    ┌──────────────────────────────────┐
            1.0     3.2      5.5     7.8   10.0 17.4┤           \*                      │
        loss                iter                11.8┤       \*\*\*\* \*\*               \*    │
                          loss/mean              3.3┤\*\*\*\*\*\*\*       \*\*\*\*\*\*\*\*\*\*\*\*\*\*\* \*\*\*\*│
            ┌──────────────────────────────────┐    └┬───────┬────────┬───────┬───────┬┘
        41.4┤               ····               │    1.0     3.2      5.5     7.8   10.0 
        37.6┤    ·      ····    ····           │loss/std            iter                
        33.9┤   · ·    ·            ·······    │                  loss/max              
        30.2┤  ·   ·  ·                    ··  │    ┌──────────────────────────────────┐
        26.4┤ ·     ··                       ··│56.3┤           +                      │
        22.7┤·                                 │45.3┤    +++++++ ++++++++++++++++++    │
        18.9┤·                                 │28.9┤++++                          ++++│
            └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
            1.0     3.2      5.5     7.8   10.0     1.0     3.2      5.5     7.8   10.0 
        loss/mean           iter                loss/max            iter
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/loss.txt
            ┌──────────────────────────────────────────────────────────────────────────┐
        56.3┤ ++ loss/max            +                                                 │
            │ -- loss/min           + +                                                │
            │ ·· loss/mean         +   +                                               │
        49.3┤ ▞▞ loss             +     ++                                             │
            │                    +        +                                            │
            │                   +          +                                           │
        42.3┤                  +            +▞▀▀▀▀▀▀▀▀▚▄▄▄▖                            │
            │                 +             ▞·         +++▝▀▀▀▚               +        │
            │        ▖++++++++       ······▐·-·········        ▀▖           ++ +       │
        35.4┤       ▞▚·             ·     ▗▘- ---------········ ▝▚▖+     +++    +      │
            │      ▐--▚··         ··     ▗▘-           ----    ···▝▄+++++     ·  ++    │
            │     ▗▘  -▌ ·       ·       ▞-                ----    ·▚▖········ ··  +   │
            │    ·▌    ▝▖ ··   ··       ▞-                     ------▝▚▄▖        ·· +  │
        28.4┤   ·▞      ▝▖  ···        ▐-                              -▝▀▚▄▖      ··++│
            │  ·▗▘       ▚            ▗▘                                   -▝▀▀▄▄▖   ··│
            │+·▗▘         ▚          ▗▘                                        --▝▀▀▄▄▄│
        21.4┤· ▞           ▌        ▗▞                                                 │
            │·▞            ▝▖    ▗▄▀▘                                                  │
            │▗▘             ▝▖-▄▞▘                                                     │
        14.4┤▌               ▝▀                                                        │
            └┬─────────────────┬──────────────────┬─────────────────┬─────────────────┬┘
            1.0               3.2                5.5               7.8             10.0 
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/loss_summary.txt
                    loss/mean hist                           loss/max hist           
            ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
        2.00┤                           ███████│2.00┤             ███████████          │
        1.67┤                           ███████│1.67┤             ███████████          │
        1.33┤                           ███████│1.33┤             ███████████          │
        1.00┤████   ███████   █████████████████│1.00┤███████   ██████████████      ████│
        0.67┤████   ███████   █████████████████│0.67┤███████   ██████████████      ████│
        0.33┤████   ███████   █████████████████│0.33┤███████   ██████████████      ████│
        0.00┤███    ██████    █████████████████│0.00┤███████   ██████████████      ████│
            └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
        17.9    24.0     30.2    36.3   42.4    22.0    30.9     39.8    48.8   57.7 
                        loss/min hist                           loss/std hist           
            ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
        2.00┤████                          ████│3.00┤████                              │
        1.67┤████                          ████│2.50┤████                              │
        1.33┤████                          ████│2.00┤██████████                        │
        1.00┤████   ██████████   ██████████████│1.50┤██████████                        │
            │████   ██████████   ██████████████│    │██████████                        │
        0.67┤████   ██████████   ██████████████│1.00┤██████████████      ████      ████│
        0.33┤████   ██████████   ██████████████│0.50┤██████████████      ████      ████│
        0.00┤███    ██████████   ██████████████│0.00┤█████████████       ████      ████│
            └┬───────┬────────┬───────┬───────┬┘    └┬───────┬────────┬───────┬───────┬┘
        13.3    20.1     26.9    33.7   40.5    -0.2     4.4      8.9    13.5   18.1 
        text saved in /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/plots/tplot/loss_hist.txt
        [2026-01-15 16:25:50,768264][W][ezpz/history:2420:finalize] h5py not found! Saving dataset as netCDF instead.
        [2026-01-15 16:25:50,768640][I][utils/__init__:636:save_dataset] Saving dataset to: /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/dataset_dataset.nc
        [2026-01-15 16:25:50,817704][I][ezpz/history:2433:finalize] Saving history report to /Users/samforeman/vibes/saforem2/ezpz/outputs/History-2026-01-15-162549/2026-01-15-162549/report.md
        >
        ```
