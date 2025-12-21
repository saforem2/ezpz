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
import time
from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional

from torch.utils.data import DataLoader, DistributedSampler

import ezpz
import ezpz.dist

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


def _sample_tensor_values(
    tensor: Optional[torch.Tensor], max_samples: int
) -> Optional[torch.Tensor]:
    if tensor is None or tensor.numel() == 0 or max_samples <= 0:
        return None
    flat = tensor.detach().flatten()
    if flat.numel() > max_samples:
        idx = torch.randperm(flat.numel(), device=flat.device)[:max_samples]
        flat = flat.index_select(0, idx)
    return flat.float()


def _histogram_dict(
    tensor: Optional[torch.Tensor], bins: int
) -> Optional[dict[str, object]]:
    if tensor is None or tensor.numel() == 0 or bins <= 0:
        return None
    t = tensor.float()
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        return None
    tmin = float(finite.min().item())
    tmax = float(finite.max().item())
    if tmin == tmax:
        tmax = tmin + 1e-6
    counts = torch.histc(finite, bins=bins, min=tmin, max=tmax)
    bin_edges = torch.linspace(tmin, tmax, bins + 1)
    return {
        "bins": int(bins),
        "min": float(tmin),
        "max": float(tmax),
        "counts": counts.cpu().tolist(),
        "bin_edges": bin_edges.cpu().tolist(),
    }


def _parse_hist_layers(spec: str, max_layers: int) -> list[int]:
    if spec.strip().lower() in {"all", "*"}:
        return list(range(max_layers))
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            try:
                lo = int(lo_str)
                hi = int(hi_str)
            except ValueError:
                logger.warning(
                    "Ignoring invalid EZPZ_HIST_LAYERS range entry: %s",
                    part,
                )
                continue
            layers.extend(range(lo, hi + 1))
        else:
            try:
                layers.append(int(part))
            except ValueError:
                logger.warning(
                    "Ignoring invalid EZPZ_HIST_LAYERS entry: %s",
                    part,
                )
                continue
    return [i for i in layers if 0 <= i < max_layers]


def _register_activation_hooks(
    model: nn.Module, layer_ids: list[int]
) -> tuple[dict[str, torch.Tensor], list[torch.utils.hooks.RemovableHandle]]:
    activations: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_id in layer_ids:
        try:
            block = model.layers[layer_id]  # type: ignore[index]
        except Exception:
            continue

        def _make_hook(tag: str):
            def _hook(_module, _inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                if torch.is_tensor(out):
                    activations[tag] = out.detach()
            return _hook

        handles.append(
            block.attention.register_forward_hook(
                _make_hook(f"layer{layer_id}/attn_out")
            )
        )
        handles.append(
            block.feed_forward.register_forward_hook(
                _make_hook(f"layer{layer_id}/ffn_out")
            )
        )
        handles.append(
            block.register_forward_hook(
                _make_hook(f"layer{layer_id}/block_out")
            )
        )

    return activations, handles


def _wandb_log_histograms(
    metrics: dict[str, object],
    *,
    step: int,
    enabled: bool,
) -> None:
    if not enabled or wandb is None or getattr(wandb, "run", None) is None:
        return
    hist_payload: dict[str, object] = {}
    for key, value in metrics.items():
        if key.startswith("hist/") and isinstance(value, dict):
            counts = value.get("counts")
            bin_edges = value.get("bin_edges")
            if isinstance(counts, list) and isinstance(bin_edges, list):
                hist_payload[key] = wandb.Histogram(
                    np_histogram=(counts, bin_edges)
                )
    if hist_payload:
        wandb.log(hist_payload, step=step)



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


def _accumulate_stats(
    tensor: Optional[torch.Tensor],
    sumsq: torch.Tensor,
    max_abs: torch.Tensor,
    nonfinite: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tensor is None or tensor.numel() == 0:
        return sumsq, max_abs, nonfinite
    t = tensor.float()
    nonfinite = nonfinite + (~torch.isfinite(t)).sum()
    max_abs = torch.maximum(max_abs, t.abs().max())
    sumsq = sumsq + (t * t).sum()
    return sumsq, max_abs, nonfinite


def _collect_param_grad_stats(
    model: nn.Module, device: torch.device | str
) -> dict[str, float]:
    param_sumsq = torch.zeros((), device=device)
    param_max = torch.zeros((), device=device)
    param_nonfinite = torch.zeros((), device=device, dtype=torch.int64)

    grad_sumsq = torch.zeros((), device=device)
    grad_max = torch.zeros((), device=device)
    grad_nonfinite = torch.zeros((), device=device, dtype=torch.int64)

    with torch.no_grad():
        for param in model.parameters():
            param_sumsq, param_max, param_nonfinite = _accumulate_stats(
                param, param_sumsq, param_max, param_nonfinite
            )
            if param.grad is not None:
                grad_sumsq, grad_max, grad_nonfinite = _accumulate_stats(
                    param.grad, grad_sumsq, grad_max, grad_nonfinite
                )

    return {
        "param/norm": float(torch.sqrt(param_sumsq).item()),
        "param/max_abs": float(param_max.item()),
        "param/nonfinite": float(param_nonfinite.item()),
        "grad/norm": float(torch.sqrt(grad_sumsq).item()),
        "grad/max_abs": float(grad_max.item()),
        "grad/nonfinite": float(grad_nonfinite.item()),
    }


def _collect_layer_grad_norms(model: nn.Module) -> list[float]:
    layer_sumsq: dict[int, float] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if ".layers." not in name:
                continue
            try:
                layer_str = name.split(".layers.", 1)[1].split(".", 1)[0]
                layer_id = int(layer_str)
            except Exception:
                continue
            grad = param.grad.float()
            layer_sumsq[layer_id] = layer_sumsq.get(layer_id, 0.0) + float(
                (grad * grad).sum().item()
            )
    if not layer_sumsq:
        return []
    max_layer = max(layer_sumsq)
    return [
        (layer_sumsq.get(i, 0.0) ** 0.5) for i in range(max_layer + 1)
    ]


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
    metrics_every = int(os.environ.get("EZPZ_METRICS_EVERY", "1"))
    track_logits = os.environ.get("EZPZ_TRACK_LOGITS", "0") == "1"
    track_hist = os.environ.get("EZPZ_TRACK_HIST", "0") == "1"
    track_act_hist = os.environ.get("EZPZ_TRACK_ACT_HIST", "1") == "1"
    hist_bins = int(os.environ.get("EZPZ_HIST_BINS", "64"))
    hist_samples = int(os.environ.get("EZPZ_HIST_SAMPLES", "20000"))
    dataset_tag = args.dataset.lower().replace("/", "_")
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
    base_model = model
    if not hasattr(base_model, "layers"):
        base_model = getattr(model, "_fsdp_wrapped_module", model)
    act_activations: dict[str, torch.Tensor] = {}
    act_handles: list[torch.utils.hooks.RemovableHandle] = []
    if track_hist and track_act_hist and ezpz.get_rank() == 0:
        hist_layers_spec = os.environ.get(
            "EZPZ_HIST_LAYERS", f"0,{config.n_layers - 1}"
        )
        layer_ids = _parse_hist_layers(hist_layers_spec, config.n_layers)
        act_activations, act_handles = _register_activation_hooks(
            base_model, layer_ids
        )
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
            attn_mask = None
            if isinstance(batch, dict) and "input_ids" in batch:
                x = batch["input_ids"]
                attn_mask = batch.get("attention_mask")
            else:
                x = batch
            assert isinstance(x, torch.Tensor)
            x = x.to(device_id)
            x = x.to(torch.long)
            if args.dataset == "random":
                inp = x[:, :-1]
                labels = x[:, 1:]
            else:
                inp = x[:, :-1]
                labels = x[:, 1:]
            inp = inp.to(device_id)
            labels = labels.to(device_id)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device_id)
            pred = model(inp)
            local_seq_len = pred.shape[1]
            if labels.shape[1] != local_seq_len:
                labels = _slice_for_sequence_parallel(labels, local_seq_len)
            if attn_mask is not None:
                if attn_mask.shape[1] > 1:
                    attn_labels = attn_mask[:, 1:]
                else:
                    attn_labels = attn_mask
                if attn_labels.shape[1] != local_seq_len:
                    attn_labels = _slice_for_sequence_parallel(
                        attn_labels, local_seq_len
                    )
                labels = labels.clone()
                labels[attn_labels == 0] = -100
            pad_id = getattr(dataset, "pad_id", None)
            if pad_id is not None:
                labels = labels.clone()
                labels[labels == int(pad_id)] = -100
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
            grad_norm_preclip = None
            if args.max_grad_norm > 0:
                grad_norm_preclip = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
            optimizer.step()
            ezpz.dist.synchronize()
            t2 = perf_counter()
            global_step += 1
            metrics: dict[str, object] = {
                "train/iter": global_step,
                "train/epoch": epoch,
                "train/bidx": idx,
                "train/loss": loss.item(),
                "train/dt": t2 - t0,
                "train/dtf": t1 - t0,
                "train/dtb": t2 - t1,
            }
            if grad_norm_preclip is not None:
                metrics["grad/norm_preclip"] = float(grad_norm_preclip)
            if global_step % max(metrics_every, 1) == 0:
                metrics.update(_collect_param_grad_stats(model, device_id))
                metrics["opt/lr"] = float(optimizer.param_groups[0]["lr"])
                metrics["input/max"] = float(x.max().item())
                metrics["input/min"] = float(x.min().item())
                metrics["labels/valid"] = float(
                    (labels != -100).sum().item()
                )
                if track_logits:
                    pred_finite = torch.isfinite(pred)
                    metrics["logits/nonfinite"] = float(
                        (~pred_finite).sum().item()
                    )
                    metrics["logits/max_abs"] = float(pred.abs().max().item())
                if track_hist and ezpz.get_rank() == 0:
                    logits_sample = _sample_tensor_values(pred, hist_samples)
                    if logits_sample is not None:
                        logits_hist = _histogram_dict(
                            logits_sample, hist_bins
                        )
                        if logits_hist is not None:
                            metrics[
                                f"hist/{dataset_tag}/logits"
                            ] = logits_hist
                    layer_grad_norms = _collect_layer_grad_norms(base_model)
                    if layer_grad_norms:
                        layer_grad_hist = _histogram_dict(
                            torch.tensor(layer_grad_norms), hist_bins
                        )
                        if layer_grad_hist is not None:
                            metrics[
                                f"hist/{dataset_tag}/grad_norm_per_layer"
                            ] = layer_grad_hist
                    if track_act_hist and act_activations:
                        for act_key, act_tensor in act_activations.items():
                            act_sample = _sample_tensor_values(
                                act_tensor, hist_samples
                            )
                            act_hist = _histogram_dict(
                                act_sample, hist_bins
                            )
                            if act_hist is not None:
                                metrics[
                                    f"hist/{dataset_tag}/activations/{act_key}"
                                ] = act_hist
                    _wandb_log_histograms(
                        metrics, step=global_step, enabled=track_hist
                    )
            history.update(metrics, summarize=False)
            history.log_metrics(
                metrics,
                logger=logger,
                debug_prefixes=("hist/",),
                include_summary=True,
                rank0_only_summary=True,
            )
            if epoch == 0 and idx == 0:
                logger.info(f"{x.shape}")
    if act_handles:
        for handle in act_handles:
            handle.remove()
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
    t0 = time.perf_counter()
    train(args)
    ezpz.dist.cleanup()
    logger.info(f"Took {time.perf_counter() - t0:.2f} seconds")
