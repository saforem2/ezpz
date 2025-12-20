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

+-----.-----+-----+-----+
|  0  |  1  |  2  |  3  |
|     |     |     |     |
+-----+-----+-----+-----+
|  4  |  5  |  6  |  7  |
|     |     |     |     |
+-----+-----+-----+-----+
|  8  |  9  | 10  | 11  |
|     |     |     |     |
+-----+-----+-----+-----+


+----------+        +------------+       +----------+       +------------+
| Host 1   |        | Host 2     |       |          |       |  Host N    |
| 8 GPUs   |        | 8 GPUs     |       |          |       |  8 GPUs    |
|          |        |            |       |    ...   |       |            |
| (TP)     |        | (TP)       |       |          |       |  (TP)      |
|[0,1,..,7]|        | [8,9..,15] |       |          |       | [8N-8,8N-7 |
|          |        |            |       |          |       |  .., 8N-1] |
|          |        |            |       |          |       |            |
+----------+        +------------+       +----------+       +------------+

- FSDP:

  [0, 8, ..., 8N-8],
  [1, 9, ..., 8N-7],
  ...,
  [7, 15, ..., 8N-1]

"""

import os
import argparse
import logging
from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional

from torch.utils.data import DataLoader, DistributedSampler

import ezpz

import torch

import torch.nn as nn
import torch.nn.functional as F


from ezpz.models import summarize_model

from ezpz.models.llama import Transformer, ModelArgs
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed._tensor import Shard, Replicate  # type: ignore

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
)

logging.getLogger("datasets").setLevel(logging.ERROR)

logger = ezpz.get_logger(__name__)

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore

fp = Path(__file__)
WBPROJ_NAME = f"ezpz.{fp.parent.stem}.{fp.stem}"
WBRUN_NAME = f"{ezpz.get_timestamp()}"

def _slice_for_sequence_parallel(
    labels: torch.Tensor, local_seq_len: int
) -> torch.Tensor:
    """
    Align the label tensor with the local sequence shard used by tensor/sequence parallelism.

    When SequenceParallel is enabled we only own a slice of the time dimension on each
    tensor-parallel rank. The logits coming from the model already reflect that slice, so
    we narrow the label tensor to the same range before computing the loss.
    """
    if local_seq_len <= 0 or labels.shape[1] == local_seq_len:
        return labels

    try:
        from ezpz import tp as tp_utils  # type: ignore
    except Exception:
        return labels[:, :local_seq_len].contiguous()

    if (
        not hasattr(tp_utils, "tensor_parallel_is_initialized")
        or not tp_utils.tensor_parallel_is_initialized()
    ):
        return labels[:, :local_seq_len].contiguous()

    tp_world = tp_utils.get_tensor_parallel_world_size()
    if tp_world <= 1:
        return labels[:, :local_seq_len].contiguous()

    tp_rank = tp_utils.get_tensor_parallel_rank()
    total_seq = labels.shape[1]
    base = total_seq // tp_world
    remainder = total_seq % tp_world
    start = base * tp_rank + min(tp_rank, remainder)

    # SequenceParallel hands out an extra token to the first `remainder` ranks.
    expected_local = base + (1 if tp_rank < remainder else 0)
    if expected_local != local_seq_len:
        logger.debug(
            "SequenceParallel shard mismatch: expected %s tokens but received %s. Adjusting to local output.",
            expected_local,
            local_seq_len,
        )

    end = min(start + local_seq_len, total_seq)
    shard = labels.new_full(
        (labels.shape[0], local_seq_len),
        fill_value=-100,
        device=labels.device,
        dtype=labels.dtype,
    )

    copy_len = end - start
    copy_len = max(0, min(copy_len, local_seq_len))
    if copy_len > 0:
        shard[:, :copy_len] = labels.narrow(1, start, copy_len)
    return shard


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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default="outputs/fsdp_tp")
    # parser.add_argument('--dataset', type=str, default='random')
    parser.add_argument(
        "--dataset", type=str, default="eliplutchok/fineweb-small-sample"
    )
    parser.add_argument(
        "--tokenizer_name", type=str, default="meta-llama/llama-2-7b-hf"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--hf-split",
        "--hf_split",
        type=str,
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--hf-text-column",
        "--hf_text_column",
        type=str,
        default="text",
        help="Column containing raw text in the dataset.",
    )
    parser.add_argument(
        "--hf-limit",
        "--hf_limit",
        type=int,
        default=512,
        help="Number of rows to sample from the HF dataset for quick experiments.",
    )
    # parser.add_argument('--max_batch_size', type=int, default=None)
    parser.add_argument(
        "--seq-len", type=int, default=int(os.environ.get("SEQ_LEN", 1024))
    )
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--depth-init", type=bool, default=True)
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable mixed precision (use fp32) for debugging NaNs.",
    )
    # max_batch_size: int = 32
    # max_seq_len: int = 32768
    # depth_init: bool = True
    return parser.parse_args()


def parallelize(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mixed_precision: Optional[MixedPrecision],
) -> nn.Module:
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]

    model.init_weights()  # type: ignore
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
                # use DTensor as the output
                # use_local_output=False,
            ),
        },
    )

    assert isinstance(model.layers, Iterable)
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

        attn_layer = transformer_block.attention  # type: ignore
        attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()
        parallelize_module(
            module=transformer_block,  # type: ignore
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )

    # from torch.distributed.fsdp import fully_shard

    sharded_model = FSDP(
        model,
        mixed_precision=mixed_precision,
        device_mesh=dp_mesh,
    )
    logger.info(f"Model after parallelization:\n{sharded_model=}\n")
    return sharded_model


def train(args: argparse.Namespace):
    rank = ezpz.dist.setup_torch(tensor_parallel_size=args.tp, seed=args.seed)
    world_size = ezpz.dist.get_world_size()
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
        fp = Path(__file__)
        run = ezpz.dist.setup_wandb(project_name=WBPROJ_NAME)
        if wandb is not None:
            assert run is not None and run is wandb.run
            from dataclasses import asdict

            wandb.config.update(ezpz.get_dist_info())
            wandb.config.update(asdict(config))  # type:ignore

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
    mp_config: Optional[MixedPrecision] = None
    if not args.fp32:
        mp_config = MixedPrecision(
            param_dtype=torch.bfloat16,
            cast_forward_inputs=True,
            reduce_dtype=torch.float32,
        )
    model = parallelize(model, device_mesh, mp_config)
    logger.info(f"Creating optimizer=AdamW with lr={args.lr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=True)

    device = ezpz.get_torch_device(as_torch_device=False)

    tp_group = device_mesh.get_group("tp")
    if args.dataset.lower() == "mnist":
        data_prefix = Path(os.getcwd()).joinpath(
            ".cache", "ezpz", "data", f"{args.dataset.lower()}"
        )
        from ezpz.data.vision import get_mnist
        from ezpz.data.distributed import TPBroadcastDataLoader

        data = get_mnist(
            outdir=Path(data_prefix),
            train_batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            num_replicas=dpsize,
            rank=device_mesh.get_local_rank("dp"),
            pin_memory=True,
            num_workers=args.num_workers,
        )
        dataset = data["dataset"]
        sampler = data["sampler"]
        dataloader = data["dataloader"]
        if args.tp > 1:
            dataloader = TPBroadcastDataLoader(dataloader, tp_group)
    elif args.dataset.lower() == "random":
        from ezpz.data.distributed import get_random_dataset_fsdp_tp

        data = get_random_dataset_fsdp_tp(
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
            seq_length=args.seq_length,
            dp_group=device_mesh.get_group("dp"),
            tp_group=tp_group,
            broadcast_within_tp=True,
            drop_last=True,
        )
        dataset = data["dataset"]
        sampler = data["sampler"]
        dataloader = data["dataloader"]
    # if args.dataset.lower() != "random":
    else:
        from ezpz.data.hf import load_hf_texts
        from ezpz.data.distributed import TPBroadcastDataLoader

        base_texts = load_hf_texts(
            dataset_name=args.dataset,
            split=args.hf_split,
            text_column=args.hf_text_column,
            limit=args.hf_limit,
        )
        vocab, _ = ezpz.data.hf.build_vocab(base_texts)
        if len(vocab) > args.vocab_size:
            specials = ["<pad>", "<unk>"]
            words = sorted(
                {word for text in base_texts for word in text.lower().split()}
            )
            keep = max(0, args.vocab_size - len(specials))
            vocab = {tok: idx for idx, tok in enumerate(specials + words[:keep])}
            logger.warning(
                "Truncated vocab to %s tokens for model size.",
                args.vocab_size,
            )
        dataset = ezpz.data.hf.ToyTextDataset(
            base_texts, vocab, seq_len=args.seq_len
        )
        sampler = (
            DistributedSampler(
                dataset=dataset,
                num_replicas=dpsize,
                rank=device_mesh.get_local_rank("dp"),
            )
            if ezpz.get_world_size() > 1
            else None
        )
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            drop_last=False,
        )
        if args.tp > 1:
            dataloader = TPBroadcastDataLoader(dataloader, tp_group)

    # ezpz.breakpoint(0)
    logger.info("Starting 2D training...")
    model.train()

    outdir = Path(args.outdir).joinpath(ezpz.utils.get_timestamp())
    metrics_path = outdir.joinpath(f"metrics-{rank}.jsonl")
    outdir.mkdir(parents=True, exist_ok=True)
    history = ezpz.history.History(
        report_dir=args.outdir,
        report_enabled=True,
        jsonl_path=metrics_path,
        jsonl_overwrite=True,
        distributed_history=(
            1 < world_size <= 384  # and not config.pytorch_profiler
        ),
    )

    # For TP, input needs to be the same across all TP ranks.
    # while for SP, input can be different across all ranks
    # We will use dp_rank for setting the random seed
    # to mimic the behavior of the dataloader
    # x = torch.tensor((args.batch_size, args.seq_len))
    x = torch.tensor(0)
    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for idx, batch in enumerate(dataloader):
            ezpz.dist.synchronize()
            t0 = perf_counter()
            if isinstance(batch, dict) and "input_ids" in batch:
                x = batch["input_ids"]
            else:
                x = batch
            assert isinstance(x, torch.Tensor)
            x = x.to(device_id)
            x = x.to(torch.long)
            if args.dataset == "random":
                inp = x[:, :-1]
                labels = x[:, 1:]
            else:
                assert isinstance(batch, torch.Tensor)
                inp = x[:, :-1]
                labels = x[:, 1:]
            inp = inp.to(device_id)
            labels = labels.to(device_id)
            pred = model(inp)
            local_seq_len = pred.shape[1]
            if labels.shape[1] != local_seq_len:
                labels = _slice_for_sequence_parallel(labels, local_seq_len)
            ezpz.dist.synchronize()
            t1 = perf_counter()
            tp_mod = getattr(ezpz, "tp", None)
            tp_rank = (
                getattr(tp_mod, "get_tensor_parallel_rank", lambda: 0)()
                if tp_mod is not None
                else 0
            )
            if epoch == 0 and idx == 0:
                pred_finite = torch.isfinite(pred)
                pred_nonfinite = int((~pred_finite).sum().item())
                pred_max = float(pred.abs().max().item())
                logger.info(
                    "pred_stats rank=%s tp=%s shape=%s nonfinite=%s max_abs=%s",
                    ezpz.get_rank(),
                    tp_rank,
                    tuple(pred.shape),
                    pred_nonfinite,
                    f"{pred_max:.6f}",
                )
            loss = F.cross_entropy(
                pred.reshape(-1, pred.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            if epoch == 0 and idx == 0:
                valid_labels = int((labels != -100).sum().item())
                logger.info(
                    "loss_inputs rank=%s tp=%s local_seq_len=%s labels=%s valid_labels=%s",
                    ezpz.get_rank(),
                    tp_rank,
                    local_seq_len,
                    tuple(labels.shape),
                    valid_labels,
                )
                # loss = F.cross_entropy(
                #     pred.flatten(0, 1),
                #     labels.flatten(0, 1),
                # )
                # loss = output.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
            optimizer.step()
            ezpz.dist.synchronize()
            t2 = perf_counter()
            global_step += 1
            logger.info(
                history.update(
                    {
                        "train/iter": global_step,
                        "train/epoch": epoch,
                        "train/bidx": idx,
                        "train/loss": loss.item(),
                        "train/dt": t2 - t0,
                        "train/dtf": t1 - t0,
                        "train/dtb": t2 - t1,
                    }
                ).replace("train/", "")
            )
            if epoch == 0 and idx == 0:
                logger.info(f"{x.shape}")
    ezpz.dist.barrier()
    logger.info("Finished 2D training")
    if ezpz.get_rank() == 0:
        dataset = history.finalize(
            run_name=WBRUN_NAME,
            dataset_fname="train",
            warmup=0.1,
        )
        logger.info(f"{dataset=}")


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"args:\n{args}")
    train(args)
