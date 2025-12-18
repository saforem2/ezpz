import os
import time

import torch

import ezpz

logger = ezpz.get_logger(__name__)


@ezpz.timeitlogit(rank=ezpz.get_rank())
def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> ezpz.History:
    unwrapped_model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    history = ezpz.History()
    device_type = ezpz.get_torch_device_type()
    dtype = unwrapped_model.layers[0].weight.dtype
    bsize = int(os.environ.get("BATCH_SIZE", 64))
    isize = unwrapped_model.layers[0].in_features
    warmup = int(os.environ.get("WARMUP_ITERS", 10))
    log_freq = int(os.environ.get("LOG_FREQ", 1))
    print_freq = int(os.environ.get("PRINT_FREQ", 10))
    model.train()
    summary = ""
    for step in range(int(os.environ.get("TRAIN_ITERS", 500))):
        with torch.autocast(
            device_type=device_type,
            dtype=dtype,
        ):
            t0 = time.perf_counter()
            x = torch.rand((bsize, isize), dtype=dtype).to(device_type)
            y = model(x)
            loss = ((y - x) ** 2).sum()
            dtf = (t1 := time.perf_counter()) - t0
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            dtb = time.perf_counter() - t1
            if step % log_freq == 0 and step > warmup:
                summary = history.update(
                    {
                        "iter": step,
                        "loss": loss.item(),
                        "dt": dtf + dtb,
                        "dtf": dtf,
                        "dtb": dtb,
                    }
                )
            if step % print_freq == 0 and step > warmup:
                logger.info(summary)
    return history


@ezpz.timeitlogit(rank=ezpz.get_rank())
def setup():
    rank = ezpz.setup_torch(seed=int(os.environ.get("SEED", 0)))
    if os.environ.get("WANDB_DISABLED", False):
        logger.info("WANDB_DISABLED is set, not initializing wandb")
    elif rank == 0:
        try:
            _ = ezpz.setup_wandb(
                project_name=os.environ.get("PROJECT_NAME", "ezpz.examples.minimal")
            )
        except Exception:
            logger.exception("Failed to initialize wandb, continuing without it")
    device_type = ezpz.get_torch_device_type()
    from ezpz.models.minimal import SequentialLinearNet

    model = SequentialLinearNet(
        input_dim=int((os.environ.get("INPUT_SIZE", 128))),
        output_dim=int(os.environ.get("OUTPUT_SIZE", 128)),
        sizes=[
            int(x)
            for x in os.environ.get(
                "LAYER_SIZES", "256,512,1024,2048,1024,512,256,128"
            ).split(",")
        ],
    )
    model.to(device_type)
    model.to((os.environ.get("DTYPE", torch.bfloat16)))
    try:
        from ezpz.utils import model_summary

        model_summary(model)
    except Exception:
        logger.exception("Failed to summarize model")
    logger.info(f"{model=}")
    optimizer = torch.optim.Adam(model.parameters())
    if ezpz.get_world_size() > 1:
        model = ezpz.dist.wrap_model_for_ddp(model)
        # from torch.nn.parallel import DistributedDataParallel as DDP
        #
        # model = DDP(model, device_ids=[ezpz.get_local_rank()])

    return model, optimizer


def main():
    model, optimizer = setup()
    history = train(model, optimizer)
    if ezpz.get_rank() == 0:
        dataset = history.finalize()
        logger.info(f"{dataset=}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print(
            "\n".join(
                [
                    "Usage: ",
                    " ".join(
                        [
                            "PRINT_ITERS=100",
                            "TRAIN_ITERS=1000",
                            "INPUT_SIZE=128",
                            "OUTPUT_SIZE=128",
                            "LAYER_SIZES=\"'128,256,128'\"",
                            "ezpz-launch",
                            "-m ezpz.examples.minimal",
                        ]
                    ),
                ]
            )
        )
        exit(0)
    else:
        main()
