"""
ezpz/__main__.py

Contains main entry point for training.
"""
# from __future__ import (
#     absolute_import,
#     annotations,
#     division,
#     print_function,
# )
import logging
import os
import time

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

import ezpz as ez
from ezpz import (  # get_gpus_per_node,; get_rank,; get_torch_backend,; get_world_size,
    TrainConfig, get_local_rank, get_torch_device, setup, setup_wandb,
    timeitlogit)

try:
    import wandb
except (ImportError, ModuleNotFoundError):
    wandb = None


RANK = ez.get_rank()
DEVICE = ez.get_torch_device()
BACKEND = ez.get_torch_backend()
WORLD_SIZE = ez.get_world_size()
LOCAL_RANK = ez.get_local_rank()
GPUS_PER_NODE = ez.get_gpus_per_node()
NUM_NODES = WORLD_SIZE // GPUS_PER_NODE

# log.setLevel('INFO')
log = logging.getLogger(__name__)
log.setLevel("INFO") if RANK == 0 else log.setLevel("CRITICAL")


def test_torch_tensor():
    import torch

    torch_device = get_torch_device()
    local_rank = get_local_rank()
    x = torch.tensor([1.0]).to(f"{torch_device}:{local_rank}")
    log.info(f"{torch_device=}, {local_rank=}, {x=}, {x.device=}")


@timeitlogit(verbose=True)
@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> int:
    rank = setup(
        framework=cfg.get("framework"),
        backend=cfg.get("backend"),
        seed=(
            cfg.get("seed")
            if cfg.get("seed") is not None
            else np.random.randint(0, 2**16)
        ),
    )
    if rank == 0 and cfg.get("use_wandb") and wandb is not None:
        run = setup_wandb(
            config=cfg,
            project_name=cfg.get("wandb_project_name"),
        )
    config: TrainConfig = instantiate(cfg)
    assert isinstance(config, TrainConfig)
    t1 = time.perf_counter()
    t0 = os.environ.get("START_TIME", None)
    # if t0 is not None:
    assert t0 is not None
    startup_time = t1 - float(t0)
    run = None
    if wandb is not None and run is not None and run is wandb.run and config.use_wandb:
        assert run is not None
        from rich.emoji import Emoji
        from rich.text import Text

        log.info(f"{config=}")
        log.info(f"Output dir: {os.getcwd()}")
        log.warning(f"Startup time: {startup_time}")
        wandb.log({"startup_time": startup_time})
        log.warning(Text(f'{Emoji("rocket")} [{run.name}]({run.url})'))
    return rank


if __name__ == "__main__":
    t0 = time.perf_counter()
    os.environ["START_TIME"] = f"{t0}"
    rank = main()
