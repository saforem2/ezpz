# `ezpz` 🍋
Sam Foreman
2024-05-13

\| \[Sam Foreman
[<span class="orcid-green"></span>](https://orcid.org/0000-0002-9981-0876)\]()  
2024-05-13

## 👀 Overview

> **<code>ezpz</code> 🍋**
>
> Launch and train across all your accelerators, using your favorite
> framework + backend combo.
>
> `ezpz` simplifies the process of:
>
> - <details>
>   <summary>
>   Setting up + launching distributed training:
>   </summary>
>
>   - <details closed>
>     <summary>
>     <code>import ezpz as ez</code>
>     </summary>
>
>     - `RANK =`
>       [`ez.setup_torch(backend=backend)`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L551)
>       <span class="dim-text">for `backend` $\in$ {`DDP`, `deepspeed`,
>       `horovod`}</span>
>
>     - `RANK =`
>       [`ez.get_rank()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#396)
>
>     - `LOCAL_RANK =`
>       [`ez.get_local_rank()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#448)
>
>     - `WORLD_SIZE =`
>       [`ez.get_world_size()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L417)
>
>     <span class="dim-text">(see
>     [`ezpz/dist.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py)
>     for more details).</span>
>
>   </details>
>
> </details>
>
> - <details closed>
>   <summary>
>   Using your favorite framework:
>   </summary>
>
>   - `framework=pytorch` + `backend={DDP, deepspeed, horovod}`
>
>   - `framework=tensorflow` + `backend=horovod`
>
>   - [`ez.get_torch_device()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L332):
>     {`cuda`, `xpu`, `mps`, `cpu`}
>
>   - [`ez.get_torch_backend()`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L348):
>     {`nccl`, `ccl`, `gloo`}
>
>   *2ez* 😎. (see [frameworks](#frameworks) for additional details)
>
> </details>
>
> - <details closed>
>   <summary>
>   Writing device agnostic code:
>   </summary>
>
>   - <details>
>     <summary>
>     <a href="https://github.com/saforem2/ezpz/blob/main/src/ezpz/dist.py#L332"><code>ezpz.get_torch_device()</code></a>
>     </summary>
>
>     ``` python
>     >>> import ezpz as ez
>     >>> DEVICE = ez.get_torch_device()
>     >>> model = torch.nn.Linear(10, 10)
>     >>> model.to(DEVICE)
>     >>> x = torch.randn((10, 10), device=DEVICE)
>     >>> y = model(x)
>     >>> y.device
>     device(type='mps', index=0)
>     ```
>
>   </details>
>
> </details>
>
> - <details closed>
>   <summary>
>   Using <code>wandb</code>:
>   </summary>
>
>   - `ez.setup_wandb(project_name='ezpz')`
>
> </details>
>
> - **Full support** for any {`device` + `framework` + `backend`}:
>   - device: {`GPU`, `XPU`, `MPS`, `CPU`}
>   - framework: {`torch`, `deepspeed`, `horovod`, `tensorflow`}
>   - backend: {`DDP`, `deepspeed`, `horovod`}

## 📝 Example

We provide below a complete example that will launch
[`test_dist.py`](./src/ezpz/test_dist.py) (included below) across all
GPUs in your current {`PBS`, `slurm`} job and train a simple model using
either `DDP` or `deepspeed`

<details closed>
<summary>

<code>test_dist.py</code>

</summary>
<!-- <a href="https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py"><code>test_dist.py</code></a>:</summary> -->

``` python
"""
ezpz_ddp.py

- to launch:

$ source ezpz/src/ezpz/bin/savejobenv
$ BACKEND=DDP launch python3 ezpz_ddp.py
"""
import os
import logging
import time
from typing import Optional
import torch
import ezpz as ez

# backend can be any of DDP, deespepeed, horovod
RANK = ez.setup_torch(
  backend=(
      backend := os.environ.get('BACKEND', 'DDP')
  ),
  port=(
      port := os.environ.get("MASTER_PORT", "29500")
  )
)
# RANK = DIST_INIT['rank']
# WORLD_SIZE = DIST_INIT['world_size']
# LOCAL_RANK = DIST_INIT['local_rank']
# if DEVICE == "cuda" and torch.cuda.is_available():
#     torch.cuda.set_device(LOCAL_RANK)
DEVICE = ez.get_torch_device()
WORLD_SIZE = ez.get_world_size()
LOCAL_RANK = ez.get_local_rank()
DEVICE_ID = f"{DEVICE}:{LOCAL_RANK}"


# log only from RANK == 0
logger = logging.getLogger(__name__)
logger.setLevel("INFO") if RANK == 0 else logger.setLevel("CRITICAL")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))  # 64
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", 128))  # 128
OUTPUT_SIZE = int(os.environ.get("OUTPUT_SIZE", 128))  # 128
DTYPE = os.environ.get("DTYPE", torch.get_default_dtype())
TRAIN_ITERS = int(os.environ.get("TRAIN_ITERS", 50))

# logger.info(f"{DIST_INIT=}")


class Network(torch.nn.Module):
  def __init__(
          self,
          input_dim: int = 128,
          output_dim: int = 128,
          sizes: Optional[list[int]] = None,
  ):
      super(Network, self).__init__()
      if sizes is None:
          self.layers = torch.nn.Linear(input_dim, output_dim)
      elif len(sizes) > 0:
          layers = [torch.nn.Linear(input_dim, sizes[0])]
          for idx, size in enumerate(sizes[1:]):
              layers.append(
                  torch.nn.Linear(sizes[idx], size)
              )
          layers.append(torch.nn.Linear(sizes[-1], output_dim))
          self.layers = torch.nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.layers(x)


def calc_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  return (y - x).pow(2).sum()


def plot_losses(losses: dict) -> None:
  import plotext as pltx
  # y = list(losses.values())
  pltx.theme('clear')
  pltx.scatter(list(losses.values()))
  pltx.show()
  pltx.save_fig("test_dist_losses.txt")
  pltx.ylabel("loss")
  pltx.xlabel("iteration")


def main():
  model = Network(
      input_dim=INPUT_SIZE,
      output_dim=OUTPUT_SIZE,
      sizes=[1024, 512, 256, 128]
  )
  model.to(DEVICE)
  model.to(DEVICE_ID)
  logger.info(f'{model=}')
  optimizer = torch.optim.Adam(model.parameters())
  if backend.lower() == 'ddp':
      if WORLD_SIZE > 1:
          from torch.nn.parallel import DistributedDataParallel as DDP
          model = DDP(
              model,
              device_ids=[]
          )
  elif backend.lower() in ('ds', 'deepspeed'):
      import deepspeed
      # config = ez.load_ds_config().update(
      #     {"train_micro_batch_size_per_gpu": BATCH_SIZE}
      # )
      import argparse
      parser = argparse.ArgumentParser(
          description='My training script.'
      )
      parser.add_argument(
          '--local_rank',
          required=False,
          type=int,
          default=-1,
          # default=ez.get_local_rank()),
          help='local rank passed from distributed launcher',
      )
      # Include DeepSpeed configuration arguments
      parser = deepspeed.add_config_arguments(parser)
      cmd_args = parser.parse_args()
      logger.info(f'{cmd_args=}')
      model, optimizer, *_ = deepspeed.initialize(
          args=cmd_args,
          model=model,
          optimizer=optimizer,
      )

  losses = {}
  for iter in range(TRAIN_ITERS):
      t0 = time.perf_counter()
      x = torch.rand((BATCH_SIZE, INPUT_SIZE), dtype=DTYPE).to(DEVICE)
      y = model(x)
      loss = calc_loss(x, y)
      losses[iter] = loss
      dtf = ((t1 := time.perf_counter()) - t0)
      if backend == 'deepspeed':
          model.backward(loss)
          model.step(loss)
      else:
          loss.backward()
          optimizer.step()
      optimizer.zero_grad()
      dtb = time.perf_counter() - t1
      logger.info(
          ', '.join([
              f'{iter=}',
              f'loss={loss.item():.5f}',
              f'dt={dtf+dtb:.3f}',
              f'{dtf=:.3f}',
              f'{dtb=:.3f}'
          ])
      )
  if RANK == 0:
      plot_losses(losses)


if __name__ == '__main__':
  main()
```

</details>

### 🏃🏻‍♂️ Running

1.  `git clone` + `pip install ezpz`:

    ``` bash
    $ git clone https://github.com/saforem2/ezpz
    $ python3 -m pip install -e ezpz
    ```

2.  <span class="dim-text">\[optional\]</span> If using `PBS` or
    `slurm`:

    - <details closed>
      <summary>
      Save Job info:
      </summary>

      - [`savejobenv`](./src/ezpz/bin/savejobenv):

        ``` bash
        $ source ezpz/src/ezpz/bin/savejobenv
        ┌───────────────────────────────────────────────────────────────────
        │ Writing PBS vars to /home/foremans/.pbsenv
        │ HOSTFILE: /var/spool/pbs/aux/8992614.amn-0001
        │ NHOSTS: 2
        │ NGPU_PER_HOST: 12 GPUs per host
        │ NGPUS: 24 GPUs total
        └───────────────────────────────────────────────────────────────────
        ┌───────────────────────────────────────────────────────────────────
        │ [DIST INFO]:
        │   • Writing Job info to /home/foremans/.pbsenv
        │     • HOSTFILE: /var/spool/pbs/aux/8992614.amn-0001
        │     • NHOSTS: 2
        │     • NGPU_PER_HOST: 12
        │     • NGPUS = (NHOSTS * NGPU_PER_HOST) = 24
        └──────────────────────────────────────────────────────────────────
        ┌──────────────────────────────────────────────────────────────────
        │ [Hosts]:
        │       • x1921c0s0b0n0.hostmgmt2000.cm.americas.sgi.com, x1921c0s2b0n0.hostmgmt2000.cm.americas.sgi.com
        │     • [host:0] - x1921c0s0b0n0.hostmgmt2000.cm.americas.sgi.com
        │     • [host:1] - x1921c0s2b0n0.hostmgmt2000.cm.americas.sgi.com
        └──────────────────────────────────────────────────────────────────
        ┌────────────────────────────────────────────────────────────────────────────────
        │ YOU ARE HERE: /home/foremans
        │ Run 'source ./bin/getjobenv' in a NEW SHELL to automatically set env vars
        └────────────────────────────────────────────────────────────────────────────────
        ┌──────────────────────────────────────────────────────────────────
        │ [Launch]:
        │     • Use: 'launch' (=mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/8992614.amn-0001)
        │       to launch job
        └───────────────────────────────────────────────────────────────────
        ```

        this will automatically define a `launch` alias:

        ``` bash
        ┌──────────────────────────────────────────────────────────────────
        │ [Launch]:
        │     • Use: 'launch' (=mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/8992614.amn-0001)
        │       to launch job
        └───────────────────────────────────────────────────────────────────
        ```

      </details>

3.  Launch [`test_dist.py`](./src/ezpz/test_dist.py):

    - DDP:

      ``` bash
      $ launch python3 -m ezpz.test_dist
      ```

    - DeepSpeed:

      ``` bash
      $ BACKEND=deepspeed launch python3 -m ezpz.test_dist --deepspeed --deepspeed_config ezpz/src/ezpz/conf/ds_config.json
      ```

    - Output:

      - <details closed>
        <summary>
        <code>GPU</code>
        </summary>

        ``` bash
        $ launch python3 -m ezpz.test_dist |& tee ezpz-test-dist.log

        Connected to tcp://x3005c0s13b0n0.hsn.cm.polaris.alcf.anl.gov:7919
        Found executable /lus/eagle/projects/datascience/foremans/miniconda3/envs/2024-04-20/bin/python3
        Launching application 9e4c8311-1729-4385-b1d2-d4cd6006ac1d
        [2024-04-20 19:26:22][INFO][dist:290] - [device='cuda'][rank=1/7][local_rank=1/3][node=1/1]
        [2024-04-20 19:26:22][INFO][dist:290] - [device='cuda'][rank=5/7][local_rank=1/3][node=1/1]
        [2024-04-20 19:26:22][INFO][dist:290] - [device='cuda'][rank=3/7][local_rank=3/3][node=1/1]
        [2024-04-20 19:26:22][INFO][dist:290] - [device='cuda'][rank=7/7][local_rank=3/3][node=1/1]
        [2024-04-20 19:26:22][INFO][dist:290] - [device='cuda'][rank=4/7][local_rank=0/3][node=0/1]
        [2024-04-20 19:26:22][INFO][dist:290] - [device='cuda'][rank=6/7][local_rank=2/3][node=0/1]
        [2024-04-20 19:26:22][INFO][dist:290] - [device='cuda'][rank=2/7][local_rank=2/3][node=0/1]
        [2024-04-20 19:26:22][INFO][dist:290] - [device='cuda'][rank=0/7][local_rank=0/3][node=0/1]
        [2024-04-20 19:26:22][WARNING][dist:296] - Using [8 / 8] available "cuda" devices !!
        [2024-04-20 19:26:22][INFO][test_dist:46] - DIST_INIT={'world_size': 8, 'rank': 0, 'local_rank': 0}
        [2024-04-20 19:26:24][INFO][test_dist:84] - model=Network(
          (layers): Sequential(
            (0): Linear(in_features=128, out_features=1024, bias=True)
            (1): Linear(in_features=1024, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=256, bias=True)
            (3): Linear(in_features=256, out_features=128, bias=True)
            (4): Linear(in_features=128, out_features=128, bias=True)
          )
        )
        [2024-04-20 19:26:28][INFO][test_dist:126] - iter=0, loss=2789.99072, dt=0.664, dtf=0.659, dtb=0.005
        [2024-04-20 19:26:28][INFO][test_dist:126] - iter=1, loss=1961.33459, dt=0.002, dtf=0.001, dtb=0.002
        [2024-04-20 19:26:28][INFO][test_dist:126] - iter=2, loss=1450.47461, dt=0.002, dtf=0.000, dtb=0.002
        [2024-04-20 19:26:28][INFO][test_dist:126] - iter=3, loss=1088.81958, dt=0.002, dtf=0.000, dtb=0.002
        [2024-04-20 19:26:28][INFO][test_dist:126] - iter=4, loss=945.28839, dt=0.002, dtf=0.000, dtb=0.002
        [2024-04-20 19:26:28][INFO][test_dist:126] - iter=5, loss=906.78857, dt=0.002, dtf=0.000, dtb=0.001
        [2024-04-20 19:26:28][INFO][test_dist:126] - iter=6, loss=789.18243, dt=0.002, dtf=0.000, dtb=0.002
        [2024-04-20 19:26:28][INFO][test_dist:126] - iter=7, loss=751.63477, dt=0.002, dtf=0.000, dtb=0.002
        [2024-04-20 19:26:28][INFO][test_dist:126] - iter=8, loss=735.62915, dt=0.002, dtf=0.000, dtb=0.002
        [2024-04-20 19:26:28][INFO][test_dist:126] - iter=9, loss=732.12775, dt=0.002, dtf=0.000, dtb=0.001
        ```

        </details>

      - <details closed>
        <summary>
        <code>XPU</code>
        </summary>

        ``` bash
        # [04:50:57 PM] [foremans@x1921c0s0b0n0] ~/q/llm.devkit/Megatron-DeepSpeed/dep/ezpz/s/ezpz  main q4-drop 32s
        $ launch python3 -Wignore test_dist.py
        Connected to tcp://x1921c0s0b0n0.hostmgmt2000.cm.americas.sgi.com:7919
        Found executable /home/foremans/miniconda3/envs/q4-drop/bin/python3
        Launching application 5bf3e9e8-89fb-412a-a49e-3c81601436b7
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=9/23][local_rank=9/11][node=1/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=14/23][local_rank=2/11][node=0/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=3/23][local_rank=3/11][node=1/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=17/23][local_rank=5/11][node=1/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=6/23][local_rank=6/11][node=0/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=13/23][local_rank=1/11][node=1/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=7/23][local_rank=7/11][node=1/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=19/23][local_rank=7/11][node=1/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=8/23][local_rank=8/11][node=0/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=21/23][local_rank=9/11][node=1/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=10/23][local_rank=10/11][node=0/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=22/23][local_rank=10/11][node=0/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=11/23][local_rank=11/11][node=1/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=23/23][local_rank=11/11][node=1/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=2/23][local_rank=2/11][node=0/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=20/23][local_rank=8/11][node=0/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=4/23][local_rank=4/11][node=0/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=15/23][local_rank=3/11][node=1/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=18/23][local_rank=6/11][node=0/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=12/23][local_rank=0/11][node=0/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=1/23][local_rank=1/11][node=1/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=16/23][local_rank=4/11][node=0/1]
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=5/23][local_rank=5/11][node=1/1]
        [2024-04-19 16:51:06][INFO][dist:239] - DistInfo={
            "DEVICE": "xpu",
            "DEVICE_ID": "xpu:0",
            "DISTRIBUTED_BACKEND": "ccl",
            "GPUS_PER_NODE": 12,
            "HOSTFILE": "/var/spool/pbs/aux/8992337.amn-0001",
            "HOSTNAME": "x1921c0s0b0n0.hostmgmt2000.cm.americas.sgi.com",
            "HOSTS": "['x1921c0s0b0n0', 'x1921c0s5b0n0']",
            "LOCAL_RANK": 0,
            "MACHINE": "SunSpot",
            "NGPUS": 24,
            "NODE_ID": 0,
            "NUM_NODES": 2,
            "RANK": 0,
            "SCHEDULER": "PBS",
            "WORLD_SIZE_IN_USE": 24,
            "WORLD_SIZE_TOTAL": 24
        }
        [2024-04-19 16:51:06][INFO][dist:602] - Using oneccl_bindings from: /lus/gila/projects/Aurora_deployment/foremans/q4-drop_sunspot/llm.devkit/torch-ccl/oneccl_bindings_for_pytorch/__init__.py
        [2024-04-19 16:51:06][INFO][dist:604] - Using ipex from: /home/foremans/miniconda3/envs/q4-drop/lib/python3.9/site-packages/intel_extension_for_pytorch/__init__.py
        [2024-04-19 16:51:06][INFO][dist:605] - [0/24] Using device='xpu' with backend='DDP' + 'ccl' for distributed training.
        [2024-04-19 16:51:06][INFO][dist:290] - [device='xpu'][rank=0/23][local_rank=0/11][node=0/1]
        [2024-04-19 16:51:06][WARNING][dist:296] - Using [24 / 24] available "xpu" devices !!
        2024:04:19-16:51:06:(16909) |CCL_WARN| MPI was initialized externally, CCL-MPI specific environment is ignored
        [2024-04-19 16:51:06][INFO][test_dist:71] - model=Network(
          (layers): Sequential(
            (0): Linear(in_features=128, out_features=1024, bias=True)
            (1): Linear(in_features=1024, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=256, bias=True)
            (3): Linear(in_features=256, out_features=128, bias=True)
            (4): Linear(in_features=128, out_features=128, bias=True)
          )
        )
        [2024-04-19 16:51:18][INFO][test_dist:101] - iter=0, loss=2709.53418, dt=1.380, dtf=0.950, dtb=0.430
        [2024-04-19 16:51:18][INFO][test_dist:101] - iter=1, loss=2058.49805, dt=0.133, dtf=0.002, dtb=0.131
        [2024-04-19 16:51:18][INFO][test_dist:101] - iter=2, loss=1507.91187, dt=0.004, dtf=0.001, dtb=0.004
        [2024-04-19 16:51:18][INFO][test_dist:101] - iter=3, loss=1181.78577, dt=0.004, dtf=0.001, dtb=0.003
        [2024-04-19 16:51:18][INFO][test_dist:101] - iter=4, loss=949.43561, dt=0.004, dtf=0.001, dtb=0.003
        [2024-04-19 16:51:18][INFO][test_dist:101] - iter=5, loss=848.14905, dt=0.004, dtf=0.001, dtb=0.003
        [2024-04-19 16:51:18][INFO][test_dist:101] - iter=6, loss=788.76123, dt=0.004, dtf=0.001, dtb=0.003
        [2024-04-19 16:51:18][INFO][test_dist:101] - iter=7, loss=753.59509, dt=0.004, dtf=0.001, dtb=0.003
        [2024-04-19 16:51:18][INFO][test_dist:101] - iter=8, loss=750.62225, dt=0.004, dtf=0.001, dtb=0.003
        [2024-04-19 16:51:18][INFO][test_dist:101] - iter=9, loss=740.23474, dt=0.004, dtf=0.001, dtb=0.003
        Application 5bf3e9e8 resources: utime=621s stime=111s maxrss=1746816KB inblock=192 oublock=16 minflt=10719359 majflt=7493 nvcsw=169332 nivcsw=77546
        ```

      </details>

      - <details closed>
        <summary>
        <code>CPU</code>
        </summary>

        ``` bash
        $ TORCH_DEVICE=cpu mpirun -np 12 python3 test_dist.py
        [2024-04-19 14:44:12][INFO][dist:290] - [device='cpu'][rank=1/11][local_rank=1/11][node=0/0]
        [2024-04-19 14:44:12][INFO][dist:290] - [device='cpu'][rank=3/11][local_rank=3/11][node=0/0]
        [2024-04-19 14:44:12][INFO][dist:290] - [device='cpu'][rank=6/11][local_rank=6/11][node=0/0]
        [2024-04-19 14:44:12][INFO][dist:290] - [device='cpu'][rank=5/11][local_rank=5/11][node=0/0]
        [2024-04-19 14:44:12][INFO][dist:290] - [device='cpu'][rank=2/11][local_rank=2/11][node=0/0]
        [2024-04-19 14:44:12][INFO][dist:290] - [device='cpu'][rank=10/11][local_rank=10/11][node=0/0]
        [2024-04-19 14:44:12][INFO][dist:290] - [device='cpu'][rank=4/11][local_rank=4/11][node=0/0]
        [2024-04-19 14:44:12][INFO][dist:290] - [device='cpu'][rank=7/11][local_rank=7/11][node=0/0]
        [2024-04-19 14:44:12][INFO][dist:290] - [device='cpu'][rank=9/11][local_rank=9/11][node=0/0]
        [2024-04-19 14:44:13][INFO][dist:290] - [device='cpu'][rank=11/11][local_rank=11/11][node=0/0]
        [2024-04-19 14:44:13][INFO][dist:290] - [device='cpu'][rank=8/11][local_rank=8/11][node=0/0]
        [2024-04-19 14:44:13][INFO][dist:239] - DistInfo={
            "DEVICE": "cpu",
            "DEVICE_ID": "cpu:0",
            "DISTRIBUTED_BACKEND": "gloo",
            "GPUS_PER_NODE": 12,
            "HOSTFILE": "/Users/samforeman/projects/saforem2/ezpz/src/ezpz/hostfile",
            "HOSTNAME": "Sams-MacBook-Pro.local",
            "HOSTS": "['Sams-MacBook-Pro']",
            "LOCAL_RANK": 0,
            "MACHINE": "Sams-MacBook-Pro.local",
            "NGPUS": 12,
            "NODE_ID": 0,
            "NUM_NODES": 1,
            "RANK": 0,
            "SCHEDULER": "LOCAL",
            "WORLD_SIZE_IN_USE": 12,
            "WORLD_SIZE_TOTAL": 12
        }
        [2024-04-19 14:44:13][INFO][dist:605] - [0/12] Using device='cpu' with backend='DDP' + 'gloo' for distributed training.
        [2024-04-19 14:44:13][INFO][dist:290] - [device='cpu'][rank=0/11][local_rank=0/11][node=0/0]
        [2024-04-19 14:44:13][WARNING][dist:296] - Using [12 / 12] available "cpu" devices !!
        [2024-04-19 14:44:13][INFO][test_dist:72] - model=Network(
          (layers): Sequential(
            (0): Linear(in_features=128, out_features=1024, bias=True)
            (1): Linear(in_features=1024, out_features=512, bias=True)
            (2): Linear(in_features=512, out_features=256, bias=True)
            (3): Linear(in_features=256, out_features=128, bias=True)
            (4): Linear(in_features=128, out_features=128, bias=True)
          )
        )
        [2024-04-19 14:44:14][INFO][test_dist:102] - iter=0, loss=2801.62549, dt=0.389, dtf=0.042, dtb=0.348
        [2024-04-19 14:44:14][INFO][test_dist:102] - iter=1, loss=2092.84692, dt=0.051, dtf=0.010, dtb=0.041
        [2024-04-19 14:44:14][INFO][test_dist:102] - iter=2, loss=1482.45520, dt=0.037, dtf=0.004, dtb=0.033
        [2024-04-19 14:44:14][INFO][test_dist:102] - iter=3, loss=1174.38037, dt=0.033, dtf=0.002, dtb=0.031
        [2024-04-19 14:44:14][INFO][test_dist:102] - iter=4, loss=938.39917, dt=0.032, dtf=0.003, dtb=0.030
        [2024-04-19 14:44:14][INFO][test_dist:102] - iter=5, loss=888.37390, dt=0.035, dtf=0.001, dtb=0.033
        [2024-04-19 14:44:14][INFO][test_dist:102] - iter=6, loss=784.63470, dt=0.036, dtf=0.003, dtb=0.032
        [2024-04-19 14:44:14][INFO][test_dist:102] - iter=7, loss=749.53839, dt=0.033, dtf=0.002, dtb=0.031
        [2024-04-19 14:44:14][INFO][test_dist:102] - iter=8, loss=732.22656, dt=0.036, dtf=0.003, dtb=0.034
        [2024-04-19 14:44:15][INFO][test_dist:102] - iter=9, loss=730.63776, dt=0.034, dtf=0.001, dtb=0.033
        35.68s user 17.20s system 546% cpu 9.681s total
        ```

    </details>

## 🧰 Helper Utilities

We provide some shell scripts that are useful when working with a job
scheduler (e.g. `PBS Pro` @ ALCF or `slurm` elsewhere).

- [`src/ezpz/bin/savejobenv`](./src/ezpz/bin/savejobenv):

  Shell script to save relevant job related environment variables to a
  file which can be `sourced` from new login instances.

  - <details closed>
    <summary>
    <b><code>savejobenv</code></b>
    </summary>

    - Launch a job, clone (or navigate into) `ezpz`, and `source`
      [`src/ezpz/bin/savejobenv`](./src/ezpz/bin/savejobenv):

    ``` bash
    (thetalogin4) $ qsub-gpu -A datascience -n 2 -q full-node --attrs="filesystems=home,grand,eagle,theta-fs0:ssds=required" -t 06:00 -I
    Job routed to queue "full-node".
    Wait for job 10155652 to start...
    Opening interactive session to thetagpu04
    [...]
    ```

    ``` bash
    (thetagpu04) $ git clone https://github.com/saforem2/ezpz
    (thetagpu04) $ source ezpz/src/ezpz/bin/savejobenv
    ┌───────────────────────────────────────────────────────────────────
    │ Writing COBALT vars to /home/foremans/.cobaltenv
    │ HOSTFILE: /var/tmp/cobalt.10155652
    │ NHOSTS: 2
    │ 8 GPUs per host
    │ 16 GPUs total
    └───────────────────────────────────────────────────────────────────
    ┌───────────────────────────────────────────────────────────────────
    │ [DIST INFO]:
    │   • Writing Job info to /home/foremans/.cobaltenv
    │     • HOSTFILE: /var/tmp/cobalt.10155652
    │     • NHOSTS: 2
    │     • NGPU_PER_HOST: 8
    │     • NGPUS = (NHOSTS * NGPU_PER_HOST) = 16
    │ [Hosts]:
    │       • thetagpu04 thetagpu19
    │ [Launch]:
    │     • Use: 'launch' (=mpirun -n  -N  --hostfile /var/tmp/cobalt.10155652 -x PATH -x LD_LIBRARY_PATH)
    │       to launch job
    └───────────────────────────────────────────────────────────────────
    ┌────────────────────────────────────────────────────────────────────────────────
    │ YOU ARE HERE: /home/foremans
    │ Run 'source ./bin/getjobenv' in a NEW SHELL to automatically set env vars
    └────────────────────────────────────────────────────────────────────────────────
    ```

    </details>

- [`src/ezpz/bin/getjobenv`](./src/ezpz/bin/getjobenv):

  Shell script that, when sourced, will populate the current environment
  with the necessary job-related variables.

  - <details closed>
    <summary>
    <b><code>getjobenv</code></b>
    </summary>

    - Now, in a **NEW SHELL**

      ``` bash
      (localhost)   $ ssh <user>@theta
      ```

      ``` bash
      (thetalogin4) $ ssh thetagpu19
      ```

      ``` bash
      (thetagpu19)  $ module load conda/2023-01-11; conda activate base
      (thetagpu19)  $ cd ezpz
      (thetagpu19)  $ source ./src/ezpz/bin/getjobenv
      ┌──────────────────────────────────────────────────────────────────
      │ [Hosts]: 
      │     • thetagpu04, thetagpu19
      └──────────────────────────────────────────────────────────────────
      ┌──────────────────────────────────────────────────────────────────
      │ [DIST INFO]: 
      │     • Loading job env from: /home/foremans/.cobaltenv
      │     • HOSTFILE: /var/tmp/cobalt.10155652
      │     • NHOSTS: 2
      │     • NGPU_PER_HOST: 8
      │     • NGPUS (NHOSTS x NGPU_PER_HOST): 16
      │     • DIST_LAUNCH: mpirun -n 16 -N 8 --hostfile /var/tmp/cobalt.10155652 -x PATH -x LD_LIBRARY_PATH
      │     • Defining alias: launch: aliased to mpirun -n 16 -N 8 --hostfile /var/tmp/cobalt.10155652 -x PATH -x LD_LIBRARY_PATH
      └──────────────────────────────────────────────────────────────────
      (thetagpu19) $ mkdir -p venvs/thetaGPU/2023-01-11
      (thetagpu19) $ python3 -m venv venvs/thetaGPU/2023-01-11 --system-site-packages
      (thetagpu19) $ source venvs/thetaGPU/2023-01-11/bin/activate
      (thetagpu19) $ python3 -m pip install -e . --require-virtualenv
      (thetagpu19) $ launch python3 -m ezpz framework=pytorch backend=DDP
      [2023-10-26 12:21:26,716][ezpz.dist][INFO] - Using DDP for distributed training
      [2023-10-26 12:21:26,787][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 13
      [2023-10-26 12:21:26,787][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 14
      [2023-10-26 12:21:26,787][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 8
      [2023-10-26 12:21:26,787][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 12
      [2023-10-26 12:21:26,787][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 6
      [2023-10-26 12:21:26,788][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 9
      [2023-10-26 12:21:26,787][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 10
      [2023-10-26 12:21:26,788][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 15
      [2023-10-26 12:21:26,788][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 11
      [2023-10-26 12:21:26,789][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 7
      [2023-10-26 12:21:26,789][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 3
      [2023-10-26 12:21:26,789][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 1
      [2023-10-26 12:21:26,789][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 4
      [2023-10-26 12:21:26,789][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 5
      [2023-10-26 12:21:26,789][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 2
      [2023-10-26 12:21:26,798][torch.distributed.distributed_c10d][INFO] - Added key: store_based_barrier_key:1 to store for rank: 0
      [2023-10-26 12:21:26,811][torch.distributed.distributed_c10d][INFO] - Rank 14: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:26,812][torch.distributed.distributed_c10d][INFO] - Rank 6: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:26,814][torch.distributed.distributed_c10d][INFO] - Rank 13: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:26,815][torch.distributed.distributed_c10d][INFO] - Rank 7: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:26,816][torch.distributed.distributed_c10d][INFO] - Rank 8: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:26,817][torch.distributed.distributed_c10d][INFO] - Rank 3: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:26,819][torch.distributed.distributed_c10d][INFO] - Rank 12: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:26,820][torch.distributed.distributed_c10d][INFO] - Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:26,821][torch.distributed.distributed_c10d][INFO] - Rank 10: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:26,823][torch.distributed.distributed_c10d][INFO] - Rank 4: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:26,825][torch.distributed.distributed_c10d][INFO] - Rank 9: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:26,825][torch.distributed.distributed_c10d][INFO] - Rank 5: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:26,827][torch.distributed.distributed_c10d][INFO] - Rank 15: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:26,828][torch.distributed.distributed_c10d][INFO] - Rank 2: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:26,830][torch.distributed.distributed_c10d][INFO] - Rank 11: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:26,831][torch.distributed.distributed_c10d][INFO] - Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 16 nodes.
      [2023-10-26 12:21:27,035][ezpz.dist][INFO] - RANK: 0 / 15
      {
        "framework": "pytorch",
        "backend": "DDP",
        "use_wandb": false,
        "seed": null,
        "port": null,
        "ds_config_path": null,
        "wandb_project_name": null,
        "precision": null,
        "ngpus": null
      }
      [2023-10-26 12:21:27,038][__main__][INFO] - Output dir: /lus/grand/projects/datascience/foremans/locations/thetaGPU/projects/saforem2/ezpz/outputs/runs/pytorch/DDP/2023-10-26/12-21-25
      [2023-10-26 12:21:27,097][ezpz.dist][INFO] - RANK: 8 / 15
      [2023-10-26 12:21:27,103][ezpz.dist][INFO] - RANK: 6 / 15
      [2023-10-26 12:21:27,104][ezpz.dist][INFO] - RANK: 14 / 15
      [2023-10-26 12:21:27,111][ezpz.dist][INFO] - RANK: 13 / 15
      [2023-10-26 12:21:27,116][ezpz.dist][INFO] - RANK: 1 / 15
      [2023-10-26 12:21:27,126][ezpz.dist][INFO] - RANK: 7 / 15
      [2023-10-26 12:21:27,135][ezpz.dist][INFO] - RANK: 10 / 15
      [2023-10-26 12:21:27,139][ezpz.dist][INFO] - RANK: 12 / 15
      [2023-10-26 12:21:27,141][ezpz.dist][INFO] - RANK: 9 / 15
      [2023-10-26 12:21:27,141][ezpz.dist][INFO] - RANK: 15 / 15
      [2023-10-26 12:21:27,141][ezpz.dist][INFO] - RANK: 11 / 15
      [2023-10-26 12:21:27,141][ezpz.dist][INFO] - RANK: 5 / 15
      [2023-10-26 12:21:27,144][ezpz.dist][INFO] - RANK: 2 / 15
      [2023-10-26 12:21:27,145][ezpz.dist][INFO] - RANK: 4 / 15
      [2023-10-26 12:21:27,145][ezpz.dist][INFO] - RANK: 3 / 15
      16.56s user 30.05s system 706% cpu 6.595s total
      ```

      while this example looked at ThetaGPU, the exact same process will
      work on any of `{ThetaGPU, Polaris, Perlmutter}`.

  </details>

> **<span style="color: var(--ansi-red);">❤️‍🩹 Status</span>**
>
> <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-style: italic">Last Updated</span>: <span style="color: #f06292; text-decoration-color: #f06292; font-weight: bold">05</span><span style="color: #f06292; text-decoration-color: #f06292">/</span><span style="color: #f06292; text-decoration-color: #f06292; font-weight: bold">13</span><span style="color: #f06292; text-decoration-color: #f06292">/</span><span style="color: #f06292; text-decoration-color: #f06292; font-weight: bold">2024</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">@</span> <span style="color: #1a8fff; text-decoration-color: #1a8fff; font-weight: bold">22:04:56</span>
> </pre>
>
> <span class="center"
> style="text-align:center;">[![](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fsaforem2.github.io%2Fezpz&count_bg=%2300CCFF&title_bg=%23303030&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)</span>
