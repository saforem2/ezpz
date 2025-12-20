"""
ezpz/examples/fsdp.py
"""

# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import os
from pathlib import Path
import time

import ezpz

# from ezpz.history import WANDB_DISABLED
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from ezpz.models import summarize_model

logger = ezpz.get_logger(__name__)

try:
    import wandb
except Exception:
    wandb = None  # type: ignore


fp = Path(__file__)
fname = f"{fp.parent.stem}.{fp.stem}"
WBPROJ_NAME = f"ezpz.{fp.parent.stem}.{fp.stem}"
OUTPUT_DIR = Path(os.getcwd()).joinpath("outputs", fname)


class Net(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, train_loader, optimizer, epoch, sampler=None):
    DEVICE = ezpz.get_torch_device()
    DEVICE_ID = f"{DEVICE}:{ezpz.get_local_rank()}"
    model.train()
    ddp_loss = torch.zeros(2).to(DEVICE_ID)
    if sampler:
        sampler.set_epoch(epoch)
    ezpz.dist.synchronize()
    t0 = time.perf_counter()
    batch, target = next(iter(train_loader))
    for _, (batch, target) in enumerate(train_loader):
        batch, target = batch.to(DEVICE_ID), target.to(DEVICE_ID)
        optimizer.zero_grad()
        output = model(batch)
        loss = F.nll_loss(output, target, reduction="sum")
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)
    ezpz.dist.synchronize()
    t1 = time.perf_counter()
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)  # type:ignore
    return {
        "epoch": epoch,
        "dt": t1 - t0,
        "train_loss": ddp_loss[0] / ddp_loss[1],
    }


def test(model, test_loader):
    DEVICE = ezpz.get_torch_device()
    DEVICE_ID = f"{DEVICE}:{ezpz.get_local_rank()}"
    model.eval()
    # correct = 0
    ddp_loss = torch.zeros(3).to(DEVICE_ID)
    with torch.no_grad():
        for batch, target in test_loader:
            batch, target = batch.to(DEVICE_ID), target.to(DEVICE_ID)
            output = model(batch)
            ddp_loss[0] += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)  # type:ignore

    test_loss = ddp_loss[0] / ddp_loss[2]

    return {
        "test_loss": test_loss,
        "test_acc": 100.0 * ddp_loss[1] / ddp_loss[2],
    }


def prepare_model_optimizer_and_scheduler(args: argparse.Namespace) -> dict:
    DEVICE = ezpz.get_torch_device()
    DEVICE_ID = f"{DEVICE}:{ezpz.get_local_rank()}"
    if args.dataset == "MNIST":
        num_classes = 10
    elif args.dataset == "OpenImages":
        num_classes = 600
    elif args.dataset == "ImageNet":
        num_classes = 1000
    elif args.dataset == "ImageNet1k":
        num_classes = 1000
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    model = Net(num_classes=num_classes).to(DEVICE_ID)
    logger.info(f"\n{summarize_model(model, verbose=False, depth=2)}")
    dtypes = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtypes[args.dtype]
    model = FSDP(
        model,
        mixed_precision=MixedPrecision(
            param_dtype=dtype,
            cast_forward_inputs=True,
        ),
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    logger.info(f"{model=}")
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def get_data(args: argparse.Namespace) -> dict:
    # data_prefix_fallback = Path(os.getcwd()).joinpath(
    #     ".cache", "ezpz", "data", f"{args.dataset.lower()}"
    # )
    # data_prefix = args.data_prefix or data_prefix_fallback
    # if args.dataset == "MNIST":
    #     from ezpz.data.vision import get_mnist
    #
    #     data = get_mnist(
    #         outdir=Path(data_prefix),
    #         train_batch_size=args.batch_size,
    #         test_batch_size=args.test_batch_size,
    #         pin_memory=True,
    #         num_workers=args.num_workers,
    #     )
    # else:
    #     raise ValueError(f"Unsupported dataset: {args.dataset}")
    data_prefix_fallback = Path(os.getcwd()).joinpath(
        ".cache", "ezpz", "data", f"{args.dataset.lower()}"
    )
    data_prefix = args.data_prefix or data_prefix_fallback
    if args.dataset == "MNIST":
        from ezpz.data.vision import get_mnist

        data = get_mnist(
            outdir=Path(data_prefix),
            train_batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
        )

    elif args.dataset == "ImageNet1k":
        from ezpz.data.vision import get_imagenet1k

        data = get_imagenet1k(
            outdir=Path(data_prefix),
            train_batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
        )

    elif args.dataset == "OpenImages":
        from ezpz.data.vision import get_openimages

        data = get_openimages(
            outdir=Path(data_prefix),
            train_batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
    elif args.dataset == "ImageNet":
        from ezpz.data.vision import get_imagenet

        data = get_imagenet(
            outdir=Path(data_prefix),
            train_batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return data


def fsdp_main(args: argparse.Namespace) -> None:
    rank = ezpz.setup_torch()
    START_TIME = ezpz.get_timestamp() if ezpz.get_rank() == 0 else None
    START_TIME = ezpz.dist.broadcast(START_TIME, root=0)
    if rank == 0:
        # try:
        fp = Path(__file__)
        run = ezpz.setup_wandb(project_name=f"ezpz.{fp.parent.stem}.{fp.stem}")
        if run is not None and wandb is not None and run is wandb.run:
            run.config.update({**vars(args)})
            run.config.update({"ezpz.dist": {**ezpz.get_dist_info()}})

    data = get_data(args)
    ezpz.dist.barrier()
    train_loader = data["train"]["loader"]
    test_loader = data["test"]["loader"]

    tmp = prepare_model_optimizer_and_scheduler(args)
    model = tmp["model"]
    optimizer = tmp["optimizer"]
    scheduler = tmp["scheduler"]

    # if rank == 0:
    outdir = Path(os.getcwd()).joinpath("outputs", fname, START_TIME)
    metrics_path = outdir.joinpath(f"metrics-{rank}.jsonl")
    outdir.mkdir(parents=True, exist_ok=True)
    history = ezpz.history.History(
        report_dir=outdir,
        report_enabled=(rank == 0),
        jsonl_path=metrics_path,
        # jsonl_overwrite=True,
        distributed_history=(
            1 < ezpz.get_world_size() <= 384  # and not config.pytorch_profiler
        ),
    )
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            sampler=data["train"]["sampler"],
        )
        test_metrics = test(model, test_loader)
        scheduler.step()
        logger.info(history.update({**train_metrics, **test_metrics}))

    logger.info(
        " ".join(
            [
                f"{args.epochs + 1} epochs took",
                f"{time.perf_counter() - start:.1f}s",
            ]
        )
    )
    ezpz.dist.barrier()

    if args.save_model:
        ezpz.dist.barrier()  # wait for slowpokes
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    if rank == 0:
        dataset = history.finalize(run_name="ezpz-fsdp", dataset_fname="train")
        logger.info(f"{dataset=}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch MNIST Example using FSDP"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=["MNIST", "OpenImages", "ImageNet", "ImageNet1k"],
        help="Dataset to use (default: MNIST)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        metavar="D",
        help="Datatype for training (default=bf16).",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        required=False,
        default=None,
        help="data directory prefix",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    fsdp_main(args=args)
