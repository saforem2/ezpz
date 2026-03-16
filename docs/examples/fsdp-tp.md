# Train Transformer with FSDP and TP on HF Datasets

FSDP Example with Tensor Parallelism

!!! info "Key API Functions"

    - [`setup_torch()`][ezpz.distributed.setup_torch] — Initialize distributed training
    - [`wrap_model()`][ezpz.distributed.wrap_model] — Wrap model for FSDP
    - [`ezpz.tp`](../python/Code-Reference/tp/index.md) — Tensor parallelism utilities

See:

- 📘 [examples/FSDP TP](../python/Code-Reference/examples/fsdp_tp.md)
- 🐍 [src/ezpz/examples/fsdp_tp.py](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/fsdp_tp.py)

```bash
ezpz launch python3 -m ezpz.examples.fsdp_tp \
    --tp=2 \
    --epochs=5 \
    --batch-size=2 \
    --dataset=eliplutchok/fineweb-small-sample \
```

## Source

<details closed><summary><code>src/ezpz/examples/fsdp_tp.py</code></summary>

```python title="src/ezpz/examples/fsdp_tp.py"
--8<-- "src/ezpz/examples/fsdp_tp.py"
```

</details>

## Code Walkthrough

### Module Docstring: Parallel Layout

The file opens with a docstring describing the 2D parallelism layout --
TP (tensor parallel) within each host, FSDP (data parallel) across hosts.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="1"
"""
ezpz/examples/fsdp_tp.py

2D tensor/sequence parallel + FSDP training demo on a Llama-style model.

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
```

### Imports

Standard library, PyTorch, and `ezpz` imports. The key distributed
primitives -- `DeviceMesh`, `FSDP`, and `parallelize_module` -- are
all pulled in here.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="118"
import os
import sys
import argparse
import logging
import time
from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional

from torch.utils.data import DataLoader, DistributedSampler

import ezpz
import ezpz.distributed
import ezpz.history

import torch

import torch.nn as nn
import torch.nn.functional as F


from ezpz.models import summarize_model
from ezpz.examples import get_example_outdir

from ezpz.models.llama import Transformer, ModelArgs
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed._tensor import Shard, Replicate  # type: ignore

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
)
```

A logger is set up and W&B is optionally imported.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="159"
logging.getLogger("datasets").setLevel(logging.ERROR)

logger = ezpz.get_logger(__name__)

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore

fp = Path(__file__)
WBPROJ_NAME = f"ezpz.{fp.parent.stem}.{fp.stem}"
```

### Model Presets

`MODEL_PRESETS` defines canned configurations (`debug`, `small`,
`medium`, `large`) that override default CLI values for quick
experimentation.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="171"
MODEL_PRESETS = {
    "debug": {
        "dim": 128,
        "n_layers": 4,
        "n_heads": 4,
        "n_kv_heads": 2,
        "multiple_of": 128,
        "seq_length": 256,
        "seq_len": 256,
        "batch_size": 1,
    },
    "small": {
        "dim": 256,
        "n_layers": 8,
        "n_heads": 8,
        "n_kv_heads": 4,
        "multiple_of": 128,
        "seq_length": 512,
        "seq_len": 512,
        "batch_size": 2,
    },
    "medium": {
        "dim": 512,
        "n_layers": 16,
        "n_heads": 8,
        "n_kv_heads": 4,
        "multiple_of": 256,
        "seq_length": 1024,
        "seq_len": 1024,
        "batch_size": 2,
    },
    "large": {
        "dim": 1024,
        "n_layers": 24,
        "n_heads": 16,
        "n_kv_heads": 8,
        "multiple_of": 256,
        "seq_length": 2048,
        "seq_len": 2048,
        "batch_size": 1,
    },
}
```

### Sharding Strategies

Maps user-facing string names to PyTorch `ShardingStrategy` enum values.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="225"
SHARDING_STRATEGIES = {
    "no_shard": ShardingStrategy.NO_SHARD,
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    "hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
}
```

### Sequence Parallel Label Slicing

When sequence parallelism is active, each TP rank only sees a slice of
the sequence dimension. This helper narrows the label tensor to match
the local shard so `cross_entropy` computes the correct loss.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="234"
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
```

### Argument Parsing

`parse_args` defines every CLI flag. The `--model` flag selects a preset
from `MODEL_PRESETS`; any flag the user provides explicitly takes
precedence over the preset via `apply_model_preset`.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="443"
def parse_args(argv: Optional[list[str]] = None):
    """CLI parser for 2D parallel (TP/SP + FSDP) training."""
    if argv is None:
        argv = sys.argv[1:]
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=sorted(MODEL_PRESETS.keys()),
        help="Model size preset (overrides dim/layer defaults)",
    )
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--sharding-strategy", type=str, default="full_shard")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default=None)
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
    args = parser.parse_args(argv)
    apply_model_preset(args, argv)
    return args
```

### `parallelize`: TP Parallelization + FSDP Wrapping

This is the core of the 2D parallelism setup. It takes the model and
device mesh, applies tensor/sequence parallelism along the `"tp"` mesh
dimension, then wraps the result with FSDP along the `"dp"` dimension.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="526"
def parallelize(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mixed_precision: Optional[MixedPrecision],
    sharding_strategy: Optional[ShardingStrategy | str] = None,
    device_id: Optional[torch.device] = None,
) -> nn.Module:
    """Wrap the model with tensor-parallel and FSDP sharding strategies."""
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]

    if isinstance(sharding_strategy, str):
        sharding_strategy = SHARDING_STRATEGIES.get(sharding_strategy, None)

    model.init_weights()  # type: ignore
```

**Top-level TP plan.** The embedding is row-sharded, the final output
projection is column-sharded, and the RMS norm between them uses
`SequenceParallel`.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="541"
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
```

**Per-layer TP plan.** Each transformer block's attention and FFN
sub-modules are parallelized: Q/K/V projections are column-sharded,
output projections are row-sharded, and norms use `SequenceParallel`.
Attention head counts are divided by the TP mesh size.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="559"
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
```

**FSDP wrapping.** After TP is applied, the entire model is wrapped with
FSDP on the `"dp"` sub-mesh.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="597"
    sharded_model = FSDP(
        model,
        mixed_precision=mixed_precision,
        device_mesh=dp_mesh,
        sharding_strategy=sharding_strategy,
        device_id=device_id,
    )
    logger.info(f"Model after parallelization:\n{sharded_model=}\n")
    return sharded_model
```

### `train`: Device Mesh, Data Loading, and Training Loop

`train` orchestrates the full run. It first creates the 2D device mesh,
loads data, then runs the epoch loop.

**Device mesh creation.** World size is split into `dp` x `tp`
dimensions.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="680"
@ezpz.timeitlogit(rank=ezpz.get_rank())
def train(
    args: argparse.Namespace,
    outdir: Path | str | os.PathLike,
) -> int:
    """Run TP/SP + FSDP training and optionally log metrics."""
    world_size = ezpz.distributed.get_world_size()
    assert world_size % args.tp == 0, "WORLD_SIZE must be divisible by TP"
    dpsize = world_size // args.tp
    device_mesh = init_device_mesh(
        str(ezpz.get_torch_device()),
        (dpsize, args.tp),
        mesh_dim_names=("dp", "tp"),
    )
    logger.info(f"Device mesh created:\n{device_mesh=}")
```

**HuggingFace dataset loading.** If `--dataset` is not `"mnist"` or
`"random"`, a tokenized HF text dataset is loaded and the vocab size is
synced to the tokenizer.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="696"
    hf_dataset = None
    hf_tokenizer = None
    if args.dataset.lower() not in {"mnist", "random"}:
        from ezpz.data.hf import get_hf_text_dataset

        seed = int(os.environ.get("EZPZ_HF_SAMPLE_SEED", "1337"))
        hf_dataset, hf_tokenizer = get_hf_text_dataset(
            dataset_name=args.dataset,
            split=args.hf_split,
            text_column=args.hf_text_column,
            tokenizer_name=args.tokenizer_name,
            seq_len=args.seq_len,
            limit=args.hf_limit,
            seed=seed,
        )
        if hf_tokenizer.vocab_size != args.vocab_size:
            logger.warning(
                "Overriding vocab_size from %s to tokenizer vocab_size=%s",
                args.vocab_size,
                hf_tokenizer.vocab_size,
            )
            args.vocab_size = hf_tokenizer.vocab_size
```

**Model construction and parallelization.** A `Transformer` is built
from `ModelArgs`, moved to the device, optionally given a
`MixedPrecision` config, and then handed to `parallelize`.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="719"
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
```

```python title="src/ezpz/examples/fsdp_tp.py" linenums="751"
    model = Transformer.from_model_args(config)
    mstr = summarize_model(
        model,
        verbose=False,
        depth=2,
    )
    logger.info(f"\n{mstr}")
    model.to(device)
    mp_config: Optional[MixedPrecision] = None
    if not args.fp32:
        mp_config = MixedPrecision(
            param_dtype=torch.bfloat16,
            cast_forward_inputs=True,
            reduce_dtype=torch.float32,
        )
    model = parallelize(
        model,
        device_mesh,
        mp_config,
        sharding_strategy=args.sharding_strategy,
        device_id=device,
    )
```

**DataLoader setup.** Three branches: MNIST, random synthetic data, or
HuggingFace datasets. For HF data, a `DistributedSampler` partitions
across the DP dimension, and `TPBroadcastDataLoader` replicates batches
within each TP group.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="835"
    else:
        from ezpz.data.distributed import TPBroadcastDataLoader

        assert hf_dataset is not None
        dataset = hf_dataset
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
```

**Metric tracking.** An `ezpz.history.History` object is created for
JSONL metric logging and optional distributed aggregation.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="864"
    metrics_path = Path(outdir).joinpath(
        f"metrics-{ezpz.distributed.get_rank()}.jsonl"
    )
    Path(outdir).mkdir(parents=True, exist_ok=True)
    history = ezpz.history.History(
        report_dir=outdir,
        report_enabled=True,
        jsonl_path=metrics_path,
        jsonl_overwrite=True,
        distributed_history=(
            1 < world_size <= 384
        ),
    )
```

**Training loop.** Each batch is moved to device, split into
`inp`/`labels`, run through the model, and the loss is computed with
`cross_entropy`. Labels are narrowed for sequence parallelism when
needed. Gradient clipping is applied before `optimizer.step()`.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="885"
    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for idx, batch in enumerate(dataloader):
            ezpz.distributed.synchronize()
            t0 = perf_counter()
            attn_mask = None
            if isinstance(batch, dict) and "input_ids" in batch:
                x = batch["input_ids"]
                attn_mask = batch.get("attention_mask")
            else:
                x = batch
            assert isinstance(x, torch.Tensor)
            x = x.to(device)
            x = x.to(torch.long)
            if args.dataset == "random":
                inp = x[:, :-1]
                labels = x[:, 1:]
            else:
                inp = x[:, :-1]
                labels = x[:, 1:]
            inp = inp.to(device)
            labels = labels.to(device)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)
            pred = model(inp)
            local_seq_len = pred.shape[1]
            if labels.shape[1] != local_seq_len:
                labels = _slice_for_sequence_parallel(labels, local_seq_len)
```

```python title="src/ezpz/examples/fsdp_tp.py" linenums="949"
            loss = F.cross_entropy(
                pred.reshape(-1, pred.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
```

```python title="src/ezpz/examples/fsdp_tp.py" linenums="969"
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm_preclip = None
            if args.max_grad_norm > 0:
                grad_norm_preclip = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
            optimizer.step()
```

After each step, timing and loss metrics are collected into a dict and
passed to `history.update` and `history.log_metrics`.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="980"
            metrics: dict[str, object] = {
                "train/iter": global_step,
                "train/epoch": epoch,
                "train/bidx": idx,
                "train/loss": loss.item(),
                "train/dt": t2 - t0,
                "train/dtf": t1 - t0,
                "train/dtb": t2 - t1,
            }
```

At the end of training, activation hooks are removed, a barrier syncs
all ranks, and `history.finalize` writes the summary dataset on rank 0.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="1043"
    if act_handles:
        for handle in act_handles:
            handle.remove()
    ezpz.distributed.barrier()
    logger.info("Finished 2D training")
    if ezpz.get_rank() == 0:
        dataset = history.finalize(
            run_name=WBPROJ_NAME,
            dataset_fname="train",
            warmup=0.1,
        )
        logger.info(f"{dataset=}")

    return 0
```

### `main` and Entrypoint

`main` calls `ezpz.distributed.setup_torch` to initialize the
distributed backend (including TP groups), determines the output
directory, and dispatches to `train`.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="1059"
@ezpz.timeitlogit(rank=ezpz.get_rank())
def main(args: argparse.Namespace) -> int:
    """Entrypoint to set up distributed context and dispatch training."""
    t0 = time.perf_counter()
    rank = ezpz.distributed.setup_torch(tensor_parallel_size=args.tp, seed=args.seed)
    t_setup = time.perf_counter()
    base_dir = args.outdir if args.outdir else None
    outdir = get_example_outdir(WBPROJ_NAME, base_dir=base_dir)
    logger.info("Outputs will be saved to %s", outdir)
    train_start = time.perf_counter()
    train(args=args, outdir=outdir)
    train_end = time.perf_counter()
    timings = {
        "main/setup_torch": t_setup - t0,
        "main/train": train_end - train_start,
        "main/total": train_end - t0,
        "timings/training_start": train_start - t0,
        "timings/train_duration": train_end - train_start,
        "timings/end-to-end": train_end - t0,
    }
    logger.info("Timings: %s", timings)
```

The `if __name__ == "__main__"` block parses args, runs `main`, cleans
up distributed state, and exits.

```python title="src/ezpz/examples/fsdp_tp.py" linenums="1093"
if __name__ == "__main__":
    args = parse_args()
    main(args)
    ezpz.distributed.cleanup()
    sys.exit(0)
```

## Help

<details closed><summary><code>--help</code></summary>

```bash
usage: fsdp_tp.py [-h] [--dim DIM] [--n-layers N_LAYERS] [--n-heads N_HEADS]
                [--n-kv-heads N_KV_HEADS] [--multiple-of MULTIPLE_OF]
                [--ffn-dim-multiplier FFN_DIM_MULTIPLIER]
                [--norm-eps NORM_EPS] [--vocab-size VOCAB_SIZE]
                [--seq-length SEQ_LENGTH] [--lr LR] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE]
                [--test-batch-size TEST_BATCH_SIZE]
                [--num-workers NUM_WORKERS] [--seed SEED] [--tp TP]
                [--sharding-strategy SHARDING_STRATEGY]
                [--max-grad-norm MAX_GRAD_NORM] [--outdir OUTDIR]
                [--dataset DATASET] [--tokenizer_name TOKENIZER_NAME]
                [--model_name_or_path MODEL_NAME_OR_PATH]
                [--hf-split HF_SPLIT] [--hf-text-column HF_TEXT_COLUMN]
                [--hf-limit HF_LIMIT] [--seq-len SEQ_LEN]
                [--max-seq-len MAX_SEQ_LEN] [--depth-init DEPTH_INIT]
                [--fp32]

2D Parallel Training

options:
-h, --help            show this help message and exit
--dim DIM
--n-layers N_LAYERS
--n-heads N_HEADS
--n-kv-heads N_KV_HEADS
--multiple-of MULTIPLE_OF
--ffn-dim-multiplier FFN_DIM_MULTIPLIER
--norm-eps NORM_EPS
--vocab-size VOCAB_SIZE
--seq-length SEQ_LENGTH
--lr LR
--epochs EPOCHS
--batch-size BATCH_SIZE
--test-batch-size TEST_BATCH_SIZE
--num-workers NUM_WORKERS
--seed SEED
--tp TP
--sharding-strategy SHARDING_STRATEGY
--max-grad-norm MAX_GRAD_NORM
--outdir OUTDIR
--dataset DATASET
--tokenizer_name TOKENIZER_NAME
--model_name_or_path MODEL_NAME_OR_PATH
--hf-split HF_SPLIT, --hf_split HF_SPLIT
                        Dataset split to load.
--hf-text-column HF_TEXT_COLUMN, --hf_text_column HF_TEXT_COLUMN
                        Column containing raw text in the dataset.
--hf-limit HF_LIMIT, --hf_limit HF_LIMIT
                        Number of rows to sample from the HF dataset for
                        quick experiments.
--seq-len SEQ_LEN
--max-seq-len MAX_SEQ_LEN
--depth-init DEPTH_INIT
--fp32                Disable mixed precision (use fp32) for debugging NaNs.
```

</details>

## Output

<details closed><summary>Output on Sunspot</summary>

```bash
$ ezpz launch python3 -m ezpz.examples.fsdp_tp
```

/embed `ezpz-fsdp-tp.html`


</details>
