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
    )
)
WORLD_SIZE = ez.get_world_size()
DEVICE = ez.get_torch_device()

# log only from RANK == 0
logger = logging.getLogger(__name__)
logger.setLevel("INFO") if RANK == 0 else logger.setLevel("CRITICAL")

BATCH_SIZE = 64
INPUT_SIZE = 128
OUTPUT_SIZE = 128
DTYPE = torch.get_default_dtype()


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


def main():
    model =  Network(
        input_dim=INPUT_SIZE,
        output_dim=OUTPUT_SIZE,
        sizes=[1024, 512, 256, 128]
    )
    devid = f"{DEVICE}:{ez.get_local_rank()}"
    model.to(devid)
    model.to(DTYPE)
    logger.info(f'{model=}')
    optimizer = torch.optim.Adam(model.parameters())
    if WORLD_SIZE > 1:
        if backend.lower() == 'ddp':
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(
                model,
                device_ids=[]
            )
        elif backend.lower() in ('ds', 'deepspeed'):
            import deepspeed
            model, optimizer, *_ = deepspeed.initialize(
                model=model,
                optimizer=optimizer
            )

    for iter in range(10):
        t0 = time.perf_counter()
        x = torch.rand((BATCH_SIZE, INPUT_SIZE), dtype=DTYPE).to(DEVICE)
        y = model(x)
        loss = calc_loss(x, y)
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


if __name__ == '__main__':
    main()
