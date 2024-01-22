"""
model.py

Simple model implementation
"""
import logging
import os

import hydra
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist
try:
    import intel_extension_for_pytorch
    import oneccl_bindings_for_pytorch
except (ImportError, ModuleNotFoundError):
    pass

from ezpz import (
    TrainConfig,
    setup,
    setup_wandb,
    timeitlogit,
    get_rank,
    get_torch_device,
    get_torch_backend,
    get_world_size,
    get_local_rank,
    get_gpus_per_node,
)

try:
    import wandb
except (ImportError, ModuleNotFoundError):
    wandb = None


log = logging.getLogger(__name__)

RANK = get_rank()
DEVICE = get_torch_device()
BACKEND = get_torch_backend()
WORLD_SIZE = get_world_size()
LOCAL_RANK = get_local_rank()
GPUS_PER_NODE = get_gpus_per_node()
NUM_NODES = WORLD_SIZE // GPUS_PER_NODE


class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # in, out, kernel_size, stride
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)  # in_features, out_features
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.fc1(torch.flatten(x, 1)))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        output = self.pool(F.relu(self.conv(inputs)))
        output = output.view(1)
        return output


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(4, 5)

    def forward(self, input):
        return self.linear(input)


def setup_training(cfg: DictConfig) -> TrainConfig:
    config: TrainConfig = instantiate(cfg)
    assert isinstance(config, TrainConfig)
    rank = setup(
        framework=config.framework,
        backend=config.backend,
        seed=config.seed
    )
    run = None
    # if rank != 0:
    #     # log.setLevel("CRITICAL")
    #     pass
    # log.info(f'[GLOBAL]: {RANK=} / {WORLD_SIZE-1=}, {NUM_NODES=}')
    # log.info(f'[LOCAL]: {LOCAL_RANK=} / {GPUS_PER_NODE=}')
    # if rank == 0:
    if rank == 0:
        if config.use_wandb and wandb is not None:
            run = setup_wandb(
                config=cfg,
                project_name=config.wandb_project_name,
            )
        log.info(f"{config=}")
        log.info(f'Output dir: {os.getcwd()}')
    if (
            wandb is not None
            and run is not None
            and run is wandb.run
            and config.use_wandb
    ):
        assert run is not None
        from rich.emoji import Emoji
        from rich.text import Text
        log.warning(
            Text(f'{Emoji("rocket")} [{run.name}]({run.url})')
        )
    return config


def train(device: str):
    model = Model().to(device)
    if WORLD_SIZE > 1:
        model = DDP(model, device_ids=[device])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss().to(device)
    for i in range(5):
        log.info("Runing Iteration: {} on device {}".format(i, device))
        input = torch.randn(2, 4).to(device)
        labels = torch.randn(2, 5).to(device)
        # forward
        log.info("Runing forward: {} on device {}".format(i, device))
        res = model(input)
        # loss
        log.info("Runing loss: {} on device {}".format(i, device))
        L = loss_fn(res, labels)
        # backward
        log.info("Runing backward: {} on device {}".format(i, device))
        L.backward()
        # update
        log.info("Runing optim: {} on device {}".format(i, device))
        optimizer.step()


@timeitlogit(verbose=True)
@hydra.main(version_base=None, config_path='./conf', config_name='config')
def test_ddp(cfg: DictConfig) -> None:
    torch.xpu.manual_seed(123)
    config = setup_training(cfg)
    device = f'xpu:{LOCAL_RANK}'
    train(device=device)

    # os.environ['RANK'] = RANK
    # os.environ['WORLD_SIZE'] = WORLD_SIZE
    # device = f"xpu:{LOCAL_RANK}"


if __name__ == '__main__':
    test_ddp()
