"""
ezpz/examples/fsdp_tp.py

Sam Foreman
2025-09-08

Modified from:
<https://pytorch.org/tutorials/intermediate/TP_tutorial.html>


This is the script to test 2D Parallel which combines Tensor/Sequence
parallel with Fully Sharded Data Parallel (TP/SP + FSDP) on a example
Llama2 model. We show an E2E working flow from forward, backward
and optimization.

We enabled Fully Sharded Data Parallel + Tensor Parallel in
separate parallel dimensions:
    Data Parallel ("dp") across hosts
    Tensor Parallel ("tp") within each host

 We use a simple diagram to illustrate below:

┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐
│ Host 1   │       │ Host 2   │       │          │       │ Host N   │
│ 8 GPUs   │       │ 8 GPUs   │       │          │       │ 8 GPUs   │
│          │       │          │       │    ...   │       │          │
│ (TP)     │       │ (TP)     │       │          │       │ (TP)     │
│[0,1,..,7]│       │[8,9..,15]│       │          │       │[8N-8,8N-7│
│          │       │          │       │          │       │ .., 8N-1]│
│          │       │          │       │          │       │          │
└──────────┘       └──────────┘       └──────────┘       └──────────┘

FSDP:
[0, 8, ..., 8N-8], [1, 9, ..., 8N-7], ..., [7, 15, ..., 8N-1]
"""

import os
import argparse
import logging
from pathlib import Path
from time import perf_counter

import ezpz

import torch

import torch.nn as nn
import torch.nn.functional as F


from ezpz.models import summarize_model

from ezpz.models.llama2 import Transformer, ModelArgs
# from ezpz.models.llama import Transformer, ModelArgs

from ezpz.data.text import get_random_dataset, get_hf_data

# from ezpz.data.llama import LlamaDataLoader
#
# import torch.distributed

import torch.distributed
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed._tensor import Shard, Replicate

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
)

logging.getLogger("datasets").setLevel(logging.ERROR)

logger = ezpz.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="2D Parallel Training")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--multiple-of", type=int, default=360)
    parser.add_argument("--ffn-dim-multiplier", type=float, default=None)
    parser.add_argument("--norm-eps", type=float, default=1e-5)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tp", type=int, default=2)
    # parser.add_argument('--dataset', type=str, default='random')
    parser.add_argument(
        "--dataset", type=str, default="eliplutchok/fineweb-small-sample"
    )
    # parser.add_argument('--max_batch_size', type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--depth-init", type=bool, default=True)
    # max_batch_size: int = 32
    # max_seq_len: int = 32768
    # depth_init: bool = True
    return parser.parse_args()


def parallelize(model: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]

    model.init_weights()
    model = parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
            ),
        },
    )

    for _, transformer_block in enumerate(model.layers):
        layer_tp_plan = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), None),  # type:ignore
                desired_input_layouts=(Replicate(), None),  # type:ignore
            ),
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),
        }

        attn_layer = transformer_block.attention
        attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )
    sharded_model = FSDP(
        model,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            cast_forward_inputs=True,
            reduce_dtype=torch.float32,
        ),
        device_mesh=dp_mesh,
    )
    logger.info(f"Model after parallelization:\n{sharded_model=}\n")
    return sharded_model


def train(args: argparse.Namespace):
    _ = ezpz.setup_torch("DDP", tensor_parallel_size=args.tp, seed=args.seed)
    world_size = ezpz.get_world_size()
    assert world_size % args.tp == 0, "WORLD_SIZE must be divisible by TP"
    dpsize = world_size // args.tp
    device_mesh = init_device_mesh(
        str(ezpz.get_torch_device()),
        (dpsize, args.tp),
        mesh_dim_names=("dp", "tp"),
    )
    logger.info(f"Device mesh created:\n{device_mesh=}")

    config = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        multiple_of=args.multiple_of,
    )
    logger.info(f"config:\n{config}")
    if ezpz.get_rank() == 0 and not os.environ.get("WANDB_DISABLED", False):
        try:
            import wandb
        except Exception as e:
            logger.exception("Failed to import wandb")
            raise e
        fp = Path(__file__)
        run = ezpz.setup_wandb(project_name=f"ezpz.{fp.parent.stem}.{fp.stem}")
        assert run is not None and run is wandb.run
        from dataclasses import asdict

        wandb.run.config.update(asdict(config))  # type:ignore

    device_type = str(ezpz.get_torch_device(as_torch_device=False))
    device_id = f"{device_type}:{ezpz.get_local_rank()}"
    model = Transformer.from_model_args(config)
    mstr = summarize_model(
        model,
        verbose=False,
        depth=2,
        # input_size=(
        #     torch.tensor((int(args.batch_size), int(args.seq_length))).to(
        #         torch.long
        #     )
        # ).shape,
    )
    logger.info(f"\n{mstr}")
    model.to(device_id)
    model = parallelize(model, device_mesh)
    logger.info(f"Creating optimizer=AdamW with lr={args.lr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=True)

    device = ezpz.get_torch_device(as_torch_device=False)

    if args.dataset == "random":
        data = get_random_dataset(
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
            seq_length=args.seq_length,
        )
        dataset = data["dataset"]
        dataloader = data["dataloader"]
    else:
        data = get_hf_data(args.dataset)
        dataloader = data.get_data_loader()

    # else:
    #
    #     # # Load a subset of FineWeb (replace with the actual dataset name if different)
    #     dataset_name = 'eliplutchok/fineweb-small-sample'  # Replace with the correct dataset name
    #     # split = "train[:1000]"  # Load the first 1000 samples for testing
    #     tokenizer_name = 'meta-llama/llama-2-7b-hf'
    #     # Get the PyTorch-compatible dataset
    #     dataset = get_torch_dataset(
    #         dataset_name,
    #         tokenizer_name=tokenizer_name,
    #         max_length=args.seq_length,
    #     )
    #     # # Create a DataLoader for batching
    #     dataloader = DataLoader(
    #         hfdset, batch_size=args.batch_size, shuffle=True
    #     )
    # Iterate through the DataLoader and inspect a batch
    # for batch in dataloader:
    #     print("Batch input_ids shape:", batch["input_ids"].shape)
    #     print("Batch attention_mask shape:", batch["attention_mask"].shape)
    #     print("Sample input_ids:", batch["input_ids"][0])
    #     break  # Stop after the first batch for testing
    # # model.init_weights()
    # tdist.barrier()
    # import torch.distributed as tdist
    # from ezpz.utils import breakpoint
    # breakpoint(0)

    logger.info("Starting 2D training...")
    model.train()
    history = ezpz.History()
    # For TP, input needs to be the same across all TP ranks.
    # while for SP, input can be different across all ranks
    # We will use dp_rank for setting the random seed
    # to mimic the behavior of the dataloader
    for epoch in range(args.epochs):
        for idx, batch in enumerate(dataloader):
            t0 = perf_counter()
            if isinstance(batch, dict) and "input_ids" in batch:
                x = batch["input_ids"]
            else:
                x = batch
            assert isinstance(x, torch.Tensor)
            x.to(device)
            # try:
            #     batch = batch['input_ids'].to(device)
            # except Exception:
            #     breakpoint(0)
            # tdist.barrier()
            x.to(torch.long)
            inp = x[:, :-1].to(device)
            labels = x[:, 1:].to(device)
            # try:
            output = model(inp)
            # except Exception:
            #     breakpoint(0)
            # tdist.barrier()
            t1 = perf_counter()
            loss = F.cross_entropy(
                output.reshape(-1, output.size(-1)), labels.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            t2 = perf_counter()
            logger.info(
                history.update(
                    {
                        "train/epoch": epoch,
                        "train/iter": idx,
                        "train/loss": loss.item(),
                        "train/dt": t2 - t0,
                        "train/dtf": t1 - t0,
                        "train/dtb": t2 - t1,
                    }
                ).replace("train/", "")
            )
            if epoch == 0 and idx == 0:
                logger.info(f"{inp.shape=}")
    logger.info("Finished 2D training")
    if ezpz.get_rank() == 0:
        dataset = history.finalize(
            run_name="ezpz-fsdp-tp",
            dataset_fname="train",
            warmup=0.1,
        )
        logger.info(f"{dataset=}")


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"args:\n{args}")
    train(args)
