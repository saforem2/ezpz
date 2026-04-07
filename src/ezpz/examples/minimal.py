"""Minimal synthetic training loop for testing distributed setup and logging.

This example builds a tiny MLP that learns to reconstruct random inputs.
Launch it with:

    ezpz launch -m ezpz.examples.minimal

Running ``python3 -m ezpz.examples.minimal --help`` prints:

    usage: ezpz.examples.minimal --help
    (Set env vars such as PRINT_ITERS=100 TRAIN_ITERS=1000 INPUT_SIZE=128 OUTPUT_SIZE=128 LAYER_SIZES=\"128,256,128\" before calling ezpz launch)

"""

import os
import time
from pathlib import Path

import torch

import ezpz
from ezpz.examples import get_example_outdir

logger = ezpz.get_logger(__name__)


@ezpz.timeitlogit(rank=ezpz.get_rank())
def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    outdir: os.PathLike | str,
) -> ezpz.History:
    """Run a synthetic training loop on random data.

    Args:
        model: Model to train (wrapped or unwrapped).
        optimizer: Optimizer configured for the model.

    Returns:
        Training history with timing and loss metrics.
    """
    unwrapped_model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    metrics_path = Path(outdir).joinpath("metrics.jsonl")
    history = ezpz.History(
        project_name=os.environ.get("PROJECT_NAME", "ezpz.examples.minimal"),
        config={"model": str(unwrapped_model), "outdir": str(outdir)},
        outdir=outdir,
        report_dir=outdir,
        report_enabled=True,
        jsonl_path=metrics_path,
        jsonl_overwrite=True,
        distributed_history=(1 < ezpz.get_world_size() <= 384),
    )
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
    """Initialize distributed runtime, model, and optimizer."""
    ezpz.setup_torch(seed=int(os.environ.get("SEED", 0)))
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
        model = ezpz.distributed.wrap_model_for_ddp(model)
        # from torch.nn.parallel import DistributedDataParallel as DDP
        #
        # model = DDP(model, device_ids=[ezpz.get_local_rank()])

    return model, optimizer


@ezpz.timeitlogit(rank=ezpz.get_rank())
def main():
    """Entrypoint for launching the minimal synthetic training example."""
    t0 = time.perf_counter()
    model, optimizer = setup()
    t_setup = time.perf_counter()
    module_name = "ezpz.examples.minimal"
    outdir = get_example_outdir(module_name)
    logger.info("Outputs will be saved to %s", outdir)
    train_start = time.perf_counter()
    history = train(model, optimizer, outdir)
    train_end = time.perf_counter()
    if ezpz.get_rank() == 0:
        dataset = history.finalize(
            outdir=outdir,
            run_name=module_name,
            dataset_fname="train",
            verbose=False,
        )
        logger.info(f"{dataset=}")
    timings = {
        "main/setup": t_setup - t0,
        "main/train": train_end - train_start,
        "main/total": train_end - t0,
        "timings/training_start": train_start - t0,
        "timings/train_duration": train_end - train_start,
        "timings/end-to-end": train_end - t0,
    }
    logger.info("Timings: %s", timings)
    history.tracker.log(
        {
            (f"timings/{k}" if not k.startswith("timings/") else k): v
            for k, v in timings.items()
        }
    )


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
