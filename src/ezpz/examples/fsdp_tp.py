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

Launch with:

    ezpz launch -m ezpz.examples.fsdp_tp --tp 2 --batch-size 8

Help output (``python3 -m ezpz.examples.fsdp_tp --help``):

    usage: fsdp_tp.py [-h] [--dim DIM] [--n-layers N_LAYERS] [--n-heads N_HEADS]
                      [--n-kv-heads N_KV_HEADS] [--multiple-of MULTIPLE_OF]
                      [--ffn-dim-multiplier FFN_DIM_MULTIPLIER]
                      [--norm-eps NORM_EPS] [--vocab-size VOCAB_SIZE]
                      [--lr LR] [--epochs EPOCHS]
                      [--batch-size BATCH_SIZE]
                      [--test-batch-size TEST_BATCH_SIZE]
                      [--num-workers NUM_WORKERS] [--seed SEED] [--tp TP]
                      [--sharding-strategy SHARDING_STRATEGY]
                      [--max-grad-norm MAX_GRAD_NORM] [--outdir OUTDIR]
                      [--dataset DATASET] [--tokenizer_name TOKENIZER_NAME]
                      [--hf-split HF_SPLIT] [--hf-text-column HF_TEXT_COLUMN]
                      [--hf-limit HF_LIMIT] [--seq-len SEQ_LEN]
                      [--max-seq-len MAX_SEQ_LEN]
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
      --hf-split HF_SPLIT, --hf_split HF_SPLIT
                            Dataset split to load.
      --hf-text-column HF_TEXT_COLUMN, --hf_text_column HF_TEXT_COLUMN
                            Column containing raw text in the dataset.
      --hf-limit HF_LIMIT, --hf_limit HF_LIMIT
                            Max rows from the HF dataset. 0 (default) = no
                            limit. Pass e.g. `--hf-limit 512` to subsample
                            for smoke tests.
      --seq-len SEQ_LEN
      --max-seq-len MAX_SEQ_LEN
      --fp32                Disable mixed precision (use fp32) for debugging NaNs.

The remaining comments outline the parallel layout used to combine TP/SP with FSDP.
"""

import os
import sys
import argparse
import json
import logging
import time
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Optional

from torch.utils.data import DataLoader, DistributedSampler

import ezpz
import ezpz.distributed
import ezpz.history

import torch

import torch.nn as nn
import torch.nn.functional as F


from ezpz.flops import compute_mfu, try_estimate, try_estimate_fake
from ezpz.models import summarize_model
from ezpz.cli.flags import add_profiling_args
from ezpz.profile import profiling_context_from_args
from ezpz.examples import get_example_outdir
from ezpz.examples._presets import arg_provided as _arg_provided

from ezpz.models.llama import Transformer, ModelArgs
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
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

MODEL_PRESETS = {
    "debug": {
        "dim": 128,
        "n_layers": 4,
        "n_heads": 4,
        "n_kv_heads": 2,
        "multiple_of": 128,
        "seq_len": 256,
        "batch_size": 1,
    },
    # ---- size ladder: s / m / l / xl / xxl / xxxl ----
    # Targets ~125M / ~250M / ~500M / ~1B / ~5B / ~10B params (Llama-arch,
    # vocab=32k). This is a BREAKING semantic change: the old
    # `small/medium/large` were toy-scale (~6M / ~50M / ~170M). Those
    # long-form names still parse (via MODEL_ALIASES) but now map to the
    # new s/m/l presets. Use `debug` for the laptop-friendly tiny model.
    "s": {
        "dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "n_kv_heads": 4,
        "multiple_of": 256,
        "seq_len": 2048,
        "batch_size": 4,
    },  # ~125M
    "m": {
        "dim": 1024,
        "n_layers": 16,
        "n_heads": 16,
        "n_kv_heads": 4,
        "multiple_of": 256,
        "seq_len": 2048,
        "batch_size": 4,
    },  # ~246M
    "l": {
        "dim": 1536,
        "n_layers": 16,
        "n_heads": 16,
        "n_kv_heads": 4,
        "multiple_of": 256,
        "seq_len": 2048,
        "batch_size": 2,
    },  # ~495M
    # xl/xxl/xxxl map roughly to Llama-1.5B / Llama-7B / Llama-13B
    # architectures (dim × layers chosen to hit those parameter
    # counts). Batch size stays at 1 — at these scales the user is
    # already past the point where batch tuning matters; FSDP/TP
    # configuration dominates.
    "xl": {
        "dim": 2048,
        "n_layers": 24,
        "n_heads": 32,
        "n_kv_heads": 8,
        "multiple_of": 256,
        "seq_len": 2048,
        "batch_size": 1,
    },
    "xxl": {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "multiple_of": 256,
        "seq_len": 4096,
        "batch_size": 1,
    },
    "xxxl": {
        "dim": 5120,
        "n_layers": 40,
        "n_heads": 40,
        "n_kv_heads": 8,
        "multiple_of": 256,
        "seq_len": 4096,
        "batch_size": 1,
    },
    # ---- torchtitan AuroraGPT (agpt) flavors ----
    # Verbatim from torchtitan/experiments/ezpz/agpt/__init__.py at
    # 0aa404543cd5707d21678f26d1d0dc6a13c9c750 — kept exact (hidden_dim,
    # rope_theta, vocab_size all match) so this example can be A/B'd against
    # a real torchtitan agpt run on the same arch.
    "agpt-2b": {
        "dim": 2048,
        "n_layers": 12,
        "n_heads": 16,
        "n_kv_heads": 4,
        "multiple_of": 256,  # rounds the FFN width up; 11008 % 256 == 0 already
        "vocab_size": 256128,
        "hidden_dim": 11008,
        "rope_theta": 50000.0,
        "seq_len": 8192,  # matches torchtitan's agpt-2b production seq_len
        "batch_size": 1,
    },
    "agpt-20b": {
        "dim": 5120,
        "n_layers": 64,
        "n_heads": 40,
        "n_kv_heads": 8,
        "multiple_of": 1024,  # agpt-20b uses compute_ffn_hidden_dim(5120, multiple_of=1024)
        "vocab_size": 256128,
        "hidden_dim": 14336,
        "rope_theta": 500000.0,
        "seq_len": 2048,
        "batch_size": 1,
    },
}
# Long-form size aliases (--model xl|xlarge|extra-large all resolve to xl).
# NOTE: small/medium/large now map to the new ~125M/~250M/~500M presets
# (s/m/l), not the previous toy-scale architectures. Use `debug` for the
# laptop-friendly tiny model.
MODEL_ALIASES = {
    "small": "s",
    "medium": "m",
    "large": "l",
    "xlarge": "xl",
    "extra-large": "xl",
    "xxlarge": "xxl",
    "extra-extra-large": "xxl",
    "xxxlarge": "xxxl",
    "extra-extra-extra-large": "xxxl",
    # agpt aliases — accept the case- and separator-tolerant forms users
    # naturally write (agpt2b, agpt_2b, AGPT-2B).
    "agpt2b": "agpt-2b",
    "agpt_2b": "agpt-2b",
    "AGPT-2B": "agpt-2b",
    "agpt20b": "agpt-20b",
    "agpt_20b": "agpt-20b",
    "AGPT-20B": "agpt-20b",
}
MODEL_PRESET_FLAGS = {
    "dim": ["--dim"],
    "n_layers": ["--n-layers"],
    "n_heads": ["--n-heads"],
    "n_kv_heads": ["--n-kv-heads"],
    "multiple_of": ["--multiple-of"],
    "vocab_size": ["--vocab-size"],
    "hidden_dim": ["--hidden-dim"],
    "rope_theta": ["--rope-theta"],
    "seq_len": ["--seq-len"],
    "batch_size": ["--batch-size"],
    "activation_checkpoint": ["--activation-checkpoint", "--ac"],
}


# FSDP2 (`fully_shard`) has no `sharding_strategy` enum — sharding behavior
# is controlled by `reshard_after_forward` (bool). We keep the legacy
# FSDP1 strategy NAMES as the CLI surface (so existing scripts don't break)
# and map each to the closest FSDP2 reshard policy:
#   - full_shard  (ZeRO-3): reshard params after forward -> lowest memory  -> True
#   - shard_grad_op (ZeRO-2): keep params unsharded after forward          -> False
#   - no_shard    (ZeRO-0 / replicate): also keep unsharded (closest FSDP2 -> False;
#       true no-shard = don't shard at all, not expressible per-module here)
#   - hybrid_shard (ZeRO-3-like): FSDP1 inter/intra-node hybrid has no
#       direct one-flag FSDP2 equivalent (needs an explicit 2D mesh); map
#       to reshard-after-forward                                     -> True
#   - hybrid_shard_zero2 (ZeRO-2-like): same caveat, but keep params
#       unsharded after forward to mirror the zero2 variant          -> False
# This is the set of valid `--sharding-strategy` values.
SHARDING_STRATEGIES = {
    "no_shard": False,
    "full_shard": True,
    "shard_grad_op": False,
    "hybrid_shard": True,
    "hybrid_shard_zero2": False,
}


def _reshard_after_forward(sharding_strategy: Optional[str]) -> bool:
    """Map a legacy --sharding-strategy name to FSDP2 reshard_after_forward."""
    if sharding_strategy is None:
        return True
    return SHARDING_STRATEGIES.get(sharding_strategy, True)


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


# ---------------------------------------------------------------------------
# Cross-entropy loss implementations.
#
# At large vocab (e.g. agpt's 256K) and long sequence (8192), the default
# eager `F.cross_entropy(logits.reshape(-1, V), ...)` materializes a
# (B*T, V) logits tensor AND an equal-size gradient in fp32 during
# `loss.backward()`. For agpt-2b @ seq=8192 that transient is ~25GB, which
# OOMs a PVC tile (UR_RESULT_ERROR_OUT_OF_RESOURCES) even though the model
# itself uses <20% of memory. Two alternatives shrink that transient:
#
#   - "chunked": split the (B*T) rows into chunks and accumulate the
#     summed loss chunk-by-chunk, so only one chunk's logits/grad exist at
#     a time. Pure eager, no torch.compile dependency.
#   - "compiled": wrap the standard CE in torch.compile so inductor fuses
#     log_softmax+NLL+backward and never materializes the full fp32 logits
#     and gradient at once (this is what torchtitan does via
#     `compile.components=["loss"]`).
#
# Selected via `--loss-impl {eager,chunked,compiled}`.
# ---------------------------------------------------------------------------


def _cross_entropy_eager(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Standard mean-reduced cross-entropy (the original behavior)."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def _cross_entropy_chunked(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Mean-reduced cross-entropy computed over row chunks.

    Mathematically identical to ``_cross_entropy_eager`` (mean over the
    non-ignored tokens), but only one ``chunk_size``-row block of logits
    (and its gradient) is materialized at a time, bounding the backward
    transient by ``chunk_size * vocab`` instead of ``B*T * vocab``.

    We accumulate the SUM of per-token losses across chunks and divide by
    the total valid-token count once, so the result matches mean reduction
    exactly (autograd handles the constant scale through the division).
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    n_rows = flat_labels.shape[0]

    valid = (flat_labels != ignore_index).sum()
    # Accumulate the loss and divide by the valid-token count in fp32.
    # `F.cross_entropy` computes log_softmax/NLL in fp32 internally, so a
    # fp32 accumulator + fp32 denominator track eager's numerics; casting
    # either to a reduced logits dtype (bf16/fp16) would drift from eager
    # and force a dtype promotion on every chunk add.
    denom = valid.clamp(min=1).float()

    total = torch.zeros((), dtype=torch.float32, device=logits.device)
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk_loss = F.cross_entropy(
            flat_logits[start:end],
            flat_labels[start:end],
            ignore_index=ignore_index,
            reduction="sum",
        )
        total = total + chunk_loss.float()
    return total / denom


# Lazily-built torch.compile wrapper around the eager CE. Cached so we only
# compile once (the first call triggers a trace). Module-level so the
# compiled artifact persists across training steps.
_COMPILED_CE = None


def _cross_entropy_compiled(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """torch.compile-fused cross-entropy (torchtitan-style)."""
    global _COMPILED_CE
    if _COMPILED_CE is None:
        _COMPILED_CE = torch.compile(_cross_entropy_eager)
    return _COMPILED_CE(logits, labels, ignore_index=ignore_index)


def _compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    impl: str = "eager",
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Dispatch to the selected cross-entropy implementation."""
    if impl == "chunked":
        return _cross_entropy_chunked(
            logits, labels, ignore_index=ignore_index, chunk_size=chunk_size
        )
    if impl == "compiled":
        return _cross_entropy_compiled(
            logits, labels, ignore_index=ignore_index
        )
    return _cross_entropy_eager(logits, labels, ignore_index=ignore_index)


def _sample_tensor_values(
    tensor: Optional[torch.Tensor], max_samples: int
) -> Optional[torch.Tensor]:
    """Downsample a tensor to at most ``max_samples`` elements for logging."""
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
    """Return histogram metadata for tensor values for logging/visualization."""
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
    """Parse layer id/ranges (e.g., '0-3,7') into a bounded list of indices."""
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
    """Attach forward hooks to capture activations for selected layers."""
    activations: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_id in layer_ids:
        try:
            block = model.layers[layer_id]  # type: ignore[index]
        except Exception:
            continue

        def _make_hook(tag: str):
            """Factory to capture activations under a given tag."""

            def _hook(_module, _inp, out):
                """Store detached activation outputs for histogram logging."""
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
    """Convert histogram dict entries into wandb.Histogram logs."""
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


# `_arg_provided` is imported from `ezpz.examples._presets` — single
# source of truth for the preset-override helper. Originally local
# here (b2c0b67); extracted to a shared module so the same fix
# automatically applies to fsdp.py / vit.py / diffusion.py / test.py.


def apply_model_preset(args: argparse.Namespace, argv: list[str]) -> None:
    if args.model is None:
        return
    # HF repo ID (contains `/`) — leave args.model alone; the model-construction
    # path branches on this. Default --tokenizer_name to the same repo if the
    # user didn't override it, since 99% of HF model repos publish a matching
    # tokenizer at the same path.
    if "/" in args.model:
        if not _arg_provided(argv, ["--tokenizer_name", "--tokenizer-name"]):
            args.tokenizer_name = args.model
        return
    # Resolve aliases (e.g. "xlarge" → "xl") before looking up the
    # preset. Direct preset keys fall through unchanged.
    model_key = MODEL_ALIASES.get(args.model, args.model)
    if model_key not in MODEL_PRESETS:
        valid = sorted({*MODEL_PRESETS.keys(), *MODEL_ALIASES.keys()})
        raise SystemExit(
            f"unknown --model {args.model!r}: not a preset name "
            f"(choices: {', '.join(valid)}) and not a HuggingFace "
            f"repo id (would need a '/' in the name)"
        )
    preset = MODEL_PRESETS[model_key]
    for field_name, value in preset.items():
        flags = MODEL_PRESET_FLAGS.get(field_name, [])
        if not _arg_provided(argv, flags):
            setattr(args, field_name, value)


def parse_args(argv: Optional[list[str]] = None):
    """CLI parser for 2D parallel (TP/SP + FSDP) training."""
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="2D Parallel Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=256,
        help=(
            "Model hidden / embedding dimension (a.k.a. d_model). Overridden "
            "when --model selects a preset."
        ),
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=32,
        help=(
            "Number of TransformerBlocks stacked in the model. Overridden "
            "when --model selects a preset."
        ),
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=32,
        help=(
            "Number of attention heads per layer. Must divide --dim. "
            "Overridden when --model selects a preset."
        ),
    )
    parser.add_argument(
        "--n-kv-heads",
        type=int,
        default=4,
        help=(
            "Number of key/value heads for grouped-query attention (GQA). "
            "Must divide --n-heads. Set equal to --n-heads for standard MHA. "
            "Overridden when --model selects a preset."
        ),
    )
    parser.add_argument(
        "--multiple-of",
        type=int,
        default=360,
        help=(
            "Round the SwiGLU FFN hidden dim up to a multiple of this value "
            "(for hardware-friendly shapes). Ignored when --hidden-dim is "
            "set explicitly."
        ),
    )
    parser.add_argument(
        "--ffn-dim-multiplier",
        type=float,
        default=None,
        help=(
            "Scale factor applied to the SwiGLU FFN hidden dim before the "
            "--multiple-of rounding step. None (default) means no extra "
            "scaling; Llama2-style models use 1.3. Ignored when "
            "--hidden-dim is set explicitly."
        ),
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help=(
            "Override SwiGLU FFN hidden dim. When None (default), TransformerBlock "
            "derives it as `4 * dim` and FeedForward applies the 2/3 + "
            "ffn_dim_multiplier + multiple_of pipeline. Set this to a concrete "
            "value (e.g. 11008 for agpt-2b, 14336 for agpt-20b) to bypass the "
            "formula and hit a published architecture exactly."
        ),
    )
    parser.add_argument(
        "--rope-theta",
        type=float,
        default=10000.0,
        help=(
            "Base frequency for RoPE positional embeddings. Llama1/2 used "
            "10000 (the default); Llama3 uses 500000; agpt-2b uses 50000."
        ),
    )
    parser.add_argument(
        "--norm-eps",
        type=float,
        default=1e-5,
        help="Epsilon added to RMSNorm denominators for numerical stability.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32_000,
        help=(
            "Tokenizer vocabulary size. Sets the embedding table and output "
            "projection sizes; must match the tokenizer used for the dataset."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="Peak learning rate for the AdamW optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of passes over the training dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Per-DP-rank training batch size (a.k.a. micro-batch). "
            "Global batch = --batch-size * (world_size / --tp)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        # No `choices=` — accepts both preset names (validated in
        # apply_model_preset) AND free-form HF repo IDs like
        # `meta-llama/Llama-3.2-1B`. Disambiguation is by the `/` character:
        # presence of `/` => HF repo ID; absence => preset/alias lookup.
        help=(
            "Model size preset (overrides dim/layer defaults). "
            "Presets: debug/small/medium/large/xl/xxl/xxxl/agpt-2b/agpt-20b. "
            "xl/xxl/xxxl accept long-form aliases (`xlarge`/`extra-large`, etc). "
            "agpt presets accept `agpt2b`/`agpt_2b` etc. "
            "Pass a HuggingFace repo id with a `/` (e.g. "
            "`meta-llama/Llama-3.2-1B`) to load HF weights instead — that "
            "path forces --tp 1 (FSDP-only)."
        ),
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        help=(
            "Per-DP-rank batch size for the eval/test loader. Only "
            "consumed by the MNIST data path; ignored for random and HF "
            "datasets."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Subprocess workers for the DataLoader. 0 (default) loads "
            "in-process — fine for tokenized HF datasets; bump for "
            "image pipelines or heavy on-the-fly preprocessing."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed for torch/numpy/python RNGs (forwarded to "
            "ezpz.setup_torch). None (default) leaves the RNGs unseeded "
            "for non-deterministic runs."
        ),
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=2,
        help=(
            "Tensor-parallel degree (a.k.a. TP / Megatron-style sharding). "
            "Must divide WORLD_SIZE. The remaining dimension "
            "(WORLD_SIZE / --tp) is used for FSDP data parallelism. "
            "Set to 1 for FSDP-only. Forced to 1 when --model is a HF "
            "repo id."
        ),
    )
    parser.add_argument(
        "--sharding-strategy",
        type=str,
        default="full_shard",
        choices=list(SHARDING_STRATEGIES.keys()),
        help=(
            "FSDP sharding behavior. The model uses FSDP2 (`fully_shard`), "
            "which controls sharding via `reshard_after_forward` rather than "
            "a strategy enum; the legacy FSDP1 names below are kept as the "
            "CLI surface and mapped to the nearest FSDP2 policy. "
            "`full_shard` (default, ZeRO-3): reshard params after forward — "
            "lowest memory, params re-all-gathered in backward. "
            "`shard_grad_op` (ZeRO-2): keep params unsharded after forward — "
            "more memory, avoids the backward all-gather. "
            "`no_shard`: mapped to keep-unsharded (FSDP2 has no true "
            "per-module no-shard). `hybrid_shard`/`hybrid_shard_zero2`: "
            "FSDP1 hybrid has no one-flag FSDP2 equivalent — `hybrid_shard` "
            "maps to reshard-after-forward, `hybrid_shard_zero2` to "
            "keep-unsharded (mirroring their ZeRO-3/ZeRO-2 variants)."
        ),
    )
    parser.add_argument(
        "--activation-checkpoint",
        "--ac",
        type=str,
        default="none",
        # `full` is an alias for `block` for compatibility with
        # torchtitan's CLI surface (their `activation_checkpoint_mode`
        # uses `full` for what we call `block` — every transformer
        # block wrapped). Resolved in _apply_activation_checkpointing.
        choices=["none", "block", "full", "selective"],
        help=(
            "Activation checkpointing strategy. "
            "`none` (default) keeps all forward activations in memory. "
            "`block` (alias: `full`) wraps each TransformerBlock — typical "
            "30-40 pct activation memory reduction, ~20 pct throughput hit "
            "(matches torchtitan's default for agpt-2b/agpt-20b). "
            "`selective` checkpoints only the attention computation inside "
            "each block — ~15-20 pct memory reduction, ~10 pct throughput "
            "hit. Trade activation memory for recomputation cost — useful "
            "when OOM-ing during training (NOT during init; for init-time "
            "OOM consider increasing --tp or reducing --seq-len). "
            "NOTE: cannot be combined with --compile (upstream AOTAutograd "
            "DeviceMesh-in-saved-tensors bug — see the --compile warning). "
            "With FSDP2 you usually don't need --ac anyway; it was a "
            "workaround for the FSDP1 backward-memory OOM that FSDP2 fixes."
        ),
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help=(
            "Clip gradients to this L2 norm before the optimizer step. "
            "Set to 0 (or negative) to disable gradient clipping."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help=(
            "Base directory for checkpoints and logs. None (default) "
            "writes under the current working directory."
        ),
    )
    # parser.add_argument('--dataset', type=str, default='random')
    parser.add_argument(
        "--dataset",
        type=str,
        default="eliplutchok/fineweb-small-sample",
        help=(
            "Training dataset. Special values: `mnist` (image debug "
            "dataset) and `random` (synthetic tokens, no IO). Anything "
            "else is treated as a HuggingFace dataset repo id."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="meta-llama/llama-2-7b-hf",
        help=(
            "HuggingFace tokenizer repo id used to tokenize the HF "
            "dataset. Auto-overridden to --model when --model is a HF "
            "repo id and --tokenizer_name wasn't passed explicitly."
        ),
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
        default=0,
        help=(
            "Maximum number of rows to sample from the HF dataset. "
            "0 (default) = no limit (use the full dataset). Pass a "
            "positive value (e.g. `--hf-limit 512`) to subsample for "
            "smoke tests. Subsampling is deterministic given "
            "$EZPZ_HF_SAMPLE_SEED."
        ),
    )
    # parser.add_argument('--max_batch_size', type=int, default=None)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=int(os.environ.get("SEQ_LEN", 1024)),
        help=(
            "Training sequence length (tokens per sample). Defaults to "
            "$SEQ_LEN if set, otherwise 1024. Must be <= --max-seq-len."
        ),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=32768,
        help=(
            "Maximum sequence length the model is built to support — "
            "sets the RoPE frequency table size and the attention "
            "scratch budget. Increase if you raise --seq-len."
        ),
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable mixed precision (use fp32) for debugging NaNs.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Compile each TransformerBlock with torch.compile after "
            "FSDP/TP wrap (matches torchtitan's apply_compile pattern). "
            "Per-block compile dodges the Dynamo + DTensor _MaskPartial "
            "graph break that whole-model compile hits on TP-wrapped "
            "tok_embeddings, and amortizes compile cost across N layers."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help=(
            "torch.compile mode (only used when --compile is set). "
            "`default` is safest. `reduce-overhead` enables cudagraphs "
            "for small models / large batches. `max-autotune` does "
            "extensive kernel search — slow startup, fastest steady state."
        ),
    )
    parser.add_argument(
        "--act-mem-budget",
        type=float,
        default=1.0,
        help=(
            "Activation-memory budget for the inductor min-cut partitioner "
            "(sets torch._functorch.config.activation_memory_budget). Only "
            "takes effect with --compile. 1.0 (default) saves ALL "
            "activations (no recompute); lower values let the compiler "
            "recompute activations in backward to cut peak memory — e.g. "
            "0.5 keeps ~half. This is how torchtitan fits larger batches "
            "for the same model (its MemoryBudgetAC sets 0.5). Try 0.5 if "
            "you OOM in backward at a batch size that should fit."
        ),
    )
    parser.add_argument(
        "--loss-impl",
        type=str,
        default="eager",
        choices=["eager", "chunked", "compiled"],
        help=(
            "Cross-entropy implementation. `eager` (default) is the plain "
            "F.cross_entropy over the full (B*T, vocab) logits — simplest, "
            "but at large vocab (e.g. agpt's 256K) and long seq it "
            "materializes a multi-GB fp32 logits+grad transient in "
            "loss.backward() that can OOM a GPU tile "
            "(UR_RESULT_ERROR_OUT_OF_RESOURCES) even when the model itself "
            "fits. `chunked` computes CE over row-chunks (see "
            "--loss-chunk-size) so only one chunk's logits/grad exist at "
            "once — pure eager, no torch.compile needed. `compiled` wraps "
            "CE in torch.compile so inductor fuses log_softmax+NLL+backward "
            "and never materializes the full transient (torchtitan's "
            "approach). NOTE: `--compile` only compiles the transformer "
            "blocks, NOT the loss, so it does NOT fix this — use "
            "--loss-impl for the loss transient."
        ),
    )
    parser.add_argument(
        "--loss-chunk-size",
        type=int,
        default=1024,
        help=(
            "Row-chunk size for --loss-impl=chunked (number of (B*T) "
            "token rows per cross-entropy chunk). Smaller = lower peak "
            "memory, more kernel launches. Ignored for other --loss-impl."
        ),
    )
    # max_batch_size: int = 32
    # max_seq_len: int = 32768
    # Shared profiler flags (--profile / --pyinstrument-profiler / etc.),
    # consumed by profiling_context_from_args around the training loop.
    add_profiling_args(parser)
    args = parser.parse_args(argv)
    apply_model_preset(args, argv)
    return args


def _configure_fsdp_gradient_division(model: nn.Module) -> None:
    """Set FSDP2 gradient divide factor to 1.0 and (on CCL/XPU) force SUM
    reduction for cross-rank gradient comms.

    Mirrors torchtitan's ``disable_fsdp_gradient_division``. FSDP2's default
    reduce-scatter does a MEAN (divide by world size) inside the collective.
    On NCCL that's fine; on CCL (XPU) splitting the divide out of the
    collective and forcing a plain SUM avoids a per-reduce precision loss
    and matches the comm path torchtitan runs on Aurora/Sunspot. The single
    post-hoc divide is folded into FSDP's gradient pipeline via the
    divide-factor=1.0 setting, so numerics are unchanged vs the mean path.

    Safe no-op on modules that aren't FSDP2-wrapped or torch builds without
    these setters.

    Set ``EZPZ_FSDP_GRAD_DIV=0`` to skip this entirely (debug escape hatch:
    leaves FSDP2's default mean reduce-scatter in place).
    """
    if os.environ.get("EZPZ_FSDP_GRAD_DIV", "1") == "0":
        logger.info("Skipping FSDP gradient-division config (EZPZ_FSDP_GRAD_DIV=0)")
        return
    force_sum_reduction = False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        backend = str(torch.distributed.get_backend() or "")
        if backend and "nccl" not in backend.lower():
            force_sum_reduction = True

    n_updated = 0
    for module in model.modules():
        set_divide_factor = getattr(module, "set_gradient_divide_factor", None)
        if callable(set_divide_factor):
            set_divide_factor(1.0)
            n_updated += 1
            if force_sum_reduction:
                set_force_sum = getattr(
                    module, "set_force_sum_reduction_for_comms", None
                )
                if callable(set_force_sum):
                    set_force_sum(True)
    logger.info(
        "Configured FSDP gradient division for %d modules "
        "(force_sum_reduction=%s)",
        n_updated,
        force_sum_reduction,
    )


def parallelize(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mixed_precision: Optional[MixedPrecisionPolicy],
    sharding_strategy: Optional[str] = None,
    activation_checkpoint: str = "none",
) -> nn.Module:
    """Apply tensor parallelism + FSDP2 (``fully_shard``) to the model.

    FSDP2 shards each module group independently (embedding, every
    TransformerBlock, then [norm, output], then the root). This per-module
    sharding keeps the backward-pass gradient/activation memory bounded —
    in particular for the large 256K-vocab embedding and output projection
    — where FSDP1's single flat-parameter wrap would OOM at long sequence
    length. Activation checkpointing (when requested) is applied to each
    block BEFORE ``fully_shard`` so the checkpoint envelope sits inside the
    sharded unit (torchtitan's ordering).
    """
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]

    reshard = _reshard_after_forward(sharding_strategy)

    # `no_shard` is NOT a true ZeRO-0 / DDP replicate under FSDP2: every
    # module group below still gets `fully_shard`, so params, grads, and
    # optimizer state are sharded across the dp mesh regardless. The
    # strategy name only controls `reshard_after_forward` (False here =
    # keep params gathered after forward). There is no per-module "don't
    # shard at all" knob in FSDP2 — that would require skipping fully_shard
    # entirely (plain DDP). Warn so a multi-rank run that asked for
    # replicated params/checkpoints isn't silently sharded.
    if sharding_strategy == "no_shard" and dp_mesh.size() > 1:
        logger.warning(
            "--sharding-strategy=no_shard does NOT give replicated "
            "(ZeRO-0/DDP) params under FSDP2: parameters, gradients, and "
            "optimizer state are still sharded across the %d-rank dp mesh. "
            "Only post-forward resharding is disabled. Use plain DDP if you "
            "need truly replicated parameters.",
            dp_mesh.size(),
        )

    model.init_weights()  # type: ignore

    # Only apply tensor/sequence parallelism when the tp mesh dim is > 1.
    # At tp=1 (FSDP-only) the TP plan is pure overhead: SequenceParallel
    # still wraps norms as DTensors sharded over a size-1 tp dim, which
    # produces a `_NormPartial` placement that must be all-reduced — and
    # combined with FSDP2's dp sharding triggers the "2 sequential
    # all_reduce ... suboptimal" warning every step, for zero benefit
    # (there's nothing to shard across a 1-rank tp group). torchtitan
    # guards the same way (`if parallel_dims.tp_enabled`).
    if tp_mesh.size() > 1:
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

    # Activation checkpointing must wrap each block BEFORE fully_shard so the
    # checkpoint envelope lives inside the FSDP2 unit (torchtitan ordering;
    # the reverse — AC after sharding — is the FSDP1 order and is wrong for
    # FSDP2). _apply_activation_checkpointing replaces each block in
    # `model.layers` in-place with a compile-aware CheckpointWrapper.
    if activation_checkpoint != "none":
        _apply_activation_checkpointing(model, activation_checkpoint)

    # FSDP2: shard each module group on the dp sub-mesh. Per-module sharding
    # (vs FSDP1's one flat param) is what keeps backward memory bounded.
    fsdp_kwargs = {"mesh": dp_mesh, "reshard_after_forward": reshard}
    if mixed_precision is not None:
        fsdp_kwargs["mp_policy"] = mixed_precision

    # Embedding first (largest single param: vocab*dim).
    if getattr(model, "tok_embeddings", None) is not None:
        fully_shard(model.tok_embeddings, **fsdp_kwargs)
    # Each transformer block (or its CheckpointWrapper) as its own unit.
    assert isinstance(model.layers, Iterable)
    for block in model.layers:
        fully_shard(block, **fsdp_kwargs)
    # norm + output together (output is the other vocab*dim-sized param).
    if (
        getattr(model, "norm", None) is not None
        and getattr(model, "output", None) is not None
    ):
        fully_shard([model.norm, model.output], **fsdp_kwargs)
    # Root last.
    fully_shard(model, **fsdp_kwargs)

    _configure_fsdp_gradient_division(model)

    logger.info(f"Model after parallelization (FSDP2):\n{model=}\n")
    return model


def _apply_activation_checkpointing(
    model: nn.Module, mode: str
) -> nn.Module:
    """Wrap transformer blocks with activation checkpointing in-place.

    `mode`:
      - "none": no-op, returns the model unchanged.
      - "block": wrap each transformer block's forward with
        ``torch.utils.checkpoint.checkpoint``. The block re-runs its
        forward during backward instead of caching all intermediate
        activations — saves ~30-40% activation memory for ~20%
        throughput overhead.
      - "selective": wrap only the inner attention call. Smaller memory
        savings (~15-20%), smaller throughput hit (~10%). Less general
        than "block" — only applies to ezpz's Transformer arch where
        each block has an `.attention` submodule.

    Works for both ezpz's Transformer (blocks live at ``model.layers``)
    and HF causal-LM arches (blocks live at the deepest ``ModuleList``).
    For HF + "selective", we fall back to "block" if no `.attention`-
    shaped submodule is found, since HF uses `.self_attn` and the
    selective path expects a specific name.
    """
    if mode == "none":
        return model
    # `full` is a torchtitan-style alias for `block` (both mean "wrap
    # every transformer block"). Normalize here so downstream branches
    # only have to think about {block, selective}.
    if mode == "full":
        mode = "block"

    # HF causal-LM models have their own gradient-checkpointing path that
    # KNOWS about `use_cache`, RNG state, attention-mask plumbing, and
    # the cache-vs-checkpoint interaction. Use that instead of our
    # generic per-block wrap — otherwise we hit a hard
    # `CheckpointError: A different number of tensors was saved during
    # the original forward and recomputation` because HF's DynamicCache
    # gets created on the first forward but skipped on the recompute,
    # producing different saved-tensor counts. Set use_cache=False so
    # the cache code-path is identical on both passes.
    base_model = getattr(model, "_fsdp_wrapped_module", model)
    if hasattr(base_model, "gradient_checkpointing_enable") and hasattr(
        base_model, "config"
    ):
        if getattr(base_model.config, "use_cache", False):
            base_model.config.use_cache = False  # type: ignore[attr-defined]
            logger.info(
                "Disabled use_cache on HF model %s for AC compatibility "
                "(cache and gradient checkpointing are mutually exclusive).",
                type(base_model).__name__,
            )
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info(
            "Applied activation_checkpoint=%s via HF "
            "gradient_checkpointing_enable on %s.",
            mode,
            type(base_model).__name__,
        )
        return model

    # Use the ptd checkpoint_wrapper (torchtitan-style) rather than a
    # monkey-patched closure around torch.utils.checkpoint.checkpoint.
    # The closure approach hits a hard Dynamo graph break under
    # `torch.compile(fullgraph=True)`: Dynamo can't trace through the
    # Python-level `checkpoint(...)` call, breaks the graph, and
    # fullgraph mode turns the break into an error. The ptd wrapper
    # registers as a proper nn.Module wrapper with compile-aware hooks,
    # so the checkpoint boundary stays inside the traced graph.
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper as ptd_checkpoint_wrapper,
    )

    def _find_block_list(m: nn.Module) -> Optional[nn.ModuleList]:
        # ezpz.Transformer wraps layers as `.layers`; FSDP wrappers
        # expose the underlying module via `_fsdp_wrapped_module`.
        candidate = getattr(m, "layers", None) or getattr(
            getattr(m, "_fsdp_wrapped_module", m), "layers", None
        )
        if isinstance(candidate, nn.ModuleList):
            return candidate
        # HF fallback: take the deepest non-empty ModuleList in the graph.
        # Most HF causal-LMs put decoder blocks at e.g.
        # `model.model.layers`.
        deepest: Optional[nn.ModuleList] = None
        deepest_depth = -1
        for name, sub in m.named_modules():
            if isinstance(sub, nn.ModuleList) and len(sub) > 0:
                depth = name.count(".")
                if depth > deepest_depth:
                    deepest_depth = depth
                    deepest = sub
        return deepest

    blocks = _find_block_list(model)
    if blocks is None:
        logger.warning(
            "activation_checkpoint=%s requested but no ModuleList of "
            "transformer blocks was found in the model graph; AC has "
            "NOT been applied.",
            mode,
        )
        return model

    # preserve_rng_state=True matches the semantics of the previous
    # implementation (torch.utils.checkpoint.checkpoint defaults to True):
    # the recomputed forward replays the same RNG state as the original,
    # so dropout masks (and any other RNG-dependent ops) are identical on
    # both passes. Setting it False would desync them and silently corrupt
    # training for models that use dropout.
    if mode == "block":
        # Replace each block with a CheckpointWrapper(block) in-place on
        # the ModuleList. Subsequent per-block torch.compile sees the
        # wrapped modules and can trace through them.
        for i in range(len(blocks)):
            blocks[i] = ptd_checkpoint_wrapper(
                blocks[i], preserve_rng_state=True
            )
    else:
        # selective: only checkpoint .attention. If absent, no-op for
        # this block — caller already logged the missing-attention case.
        for block in blocks:
            attn = getattr(block, "attention", None)
            if attn is None:
                continue
            block.attention = ptd_checkpoint_wrapper(
                attn, preserve_rng_state=True
            )
    logger.info(
        "Applied activation_checkpoint=%s to %d transformer blocks.",
        mode,
        len(blocks),
    )
    # selective AC silently no-ops on blocks lacking a `.attention`
    # submodule (HF arches use `.self_attn`, etc.). Warn the user once
    # if any blocks were skipped so they know to switch to `--ac block`
    # for full coverage on non-ezpz architectures.
    if mode == "selective":
        skipped = sum(
            1 for b in blocks if getattr(b, "attention", None) is None
        )
        if skipped > 0:
            logger.warning(
                "activation_checkpoint=selective: %d/%d blocks had no "
                "`.attention` attribute and were left unwrapped. For "
                "non-ezpz architectures, use --ac block instead.",
                skipped,
                len(blocks),
            )
    return model


def _accumulate_stats(
    tensor: Optional[torch.Tensor],
    sumsq: torch.Tensor,
    max_abs: torch.Tensor,
    nonfinite: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accumulate norm and non-finite counts into running stats."""
    if tensor is None:
        return sumsq, max_abs, nonfinite
    # FSDP2 parameters/grads are DTensors: `.numel()` reports the GLOBAL
    # size (nonzero) but the math below runs on this rank's LOCAL shard,
    # which can be empty (0 elements) on some ranks. Unwrap to the local
    # shard first so the empty-guard actually catches it — otherwise
    # `t.abs().max()` raises "Expected reduction dim ... numel() == 0".
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    if tensor.numel() == 0:
        return sumsq, max_abs, nonfinite
    t = tensor.float()
    nonfinite = nonfinite + (~torch.isfinite(t)).sum()
    max_abs = torch.maximum(max_abs, t.abs().max())
    sumsq = sumsq + (t * t).sum()
    return sumsq, max_abs, nonfinite


def _collect_param_grad_stats(
    model: nn.Module, device: torch.device | str
) -> dict[str, float]:
    """Aggregate parameter/gradient norms and non-finite counts."""
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
    """Return per-layer gradient L2 norms for logging/debugging."""
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
            grad = param.grad
            # FSDP2 grads are DTensors; sum the LOCAL shard (each rank
            # contributes its piece — adequate for a logging-only norm).
            if hasattr(grad, "to_local"):
                grad = grad.to_local()
            grad = grad.float()
            if grad.numel() == 0:
                continue
            layer_sumsq[layer_id] = layer_sumsq.get(layer_id, 0.0) + float(
                (grad * grad).sum().item()
            )
    if not layer_sumsq:
        return []
    max_layer = max(layer_sumsq)
    return [(layer_sumsq.get(i, 0.0) ** 0.5) for i in range(max_layer + 1)]


@ezpz.timeitlogit(rank=ezpz.get_rank())
def train(
    args: argparse.Namespace,
    outdir: Path | str | os.PathLike,
    profiler: Optional[Any] = None,
) -> int:
    """Run TP/SP + FSDP training and optionally log metrics.

    Args:
        args: Parsed CLI namespace.
        outdir: Output directory for metrics / reports.
        profiler: Optional active profiler (``torch.profiler.profile`` or
            ``None``) from :func:`ezpz.profile.profiling_context_from_args`.
            When non-None, ``profiler.step()`` is called once per training
            step so the schedule (wait/warmup/active/repeat) advances.
    """
    world_size = ezpz.distributed.get_world_size()
    assert world_size % args.tp == 0, "WORLD_SIZE must be divisible by TP"
    dpsize = world_size // args.tp
    device_mesh = ezpz.init_device_mesh_safe(
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

    # HF repo IDs are forced to the HF code path; the ezpz Transformer
    # construction below would silently produce a randomly-initialized model
    # with the wrong architecture for the requested repo.
    is_hf_model = bool(args.model and "/" in args.model)

    config = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        multiple_of=args.multiple_of,
        hidden_dim=args.hidden_dim,
        rope_theta=args.rope_theta,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        norm_eps=args.norm_eps,
        max_seq_len=args.max_seq_len,
    )
    logger.info(f"config:\n{config}")
    metrics_every = int(os.environ.get("EZPZ_METRICS_EVERY", "1"))
    track_logits = os.environ.get("EZPZ_TRACK_LOGITS", "0") == "1"
    track_hist = os.environ.get("EZPZ_TRACK_HIST", "0") == "1"
    track_act_hist = os.environ.get("EZPZ_TRACK_ACT_HIST", "1") == "1"
    hist_bins = int(os.environ.get("EZPZ_HIST_BINS", "64"))
    hist_samples = int(os.environ.get("EZPZ_HIST_SAMPLES", "20000"))
    dataset_tag = args.dataset.lower().replace("/", "_")
    # Update wandb config with model args (run already initialised in main).
    if (
        ezpz.get_rank() == 0
        and wandb is not None
        and getattr(wandb, "run", None) is not None
    ):
        from dataclasses import asdict

        wandb.config.update(asdict(config))  # type:ignore

    device_type = ezpz.distributed.get_torch_device_type()
    device = (
        torch.device("cpu")
        if device_type == "cpu"
        else torch.device(f"{device_type}:{ezpz.get_local_rank()}")
    )
    if is_hf_model:
        # HF path: pull arch + weights from the hub. The ezpz Transformer
        # above is skipped entirely. Note we still built `config` above so
        # downstream logging / wandb.config.update(asdict(config)) doesn't
        # crash, but it does NOT reflect the real HF architecture — that's
        # in `model.config` after the load below.
        from transformers import AutoModelForCausalLM

        if args.tp > 1:
            logger.warning(
                "HF model %s requested with --tp=%d; ezpz's TP plan is "
                "hardcoded to its own Transformer module names and won't "
                "match HF's LlamaDecoderLayer / GemmaDecoderLayer / ... "
                "Forcing --tp 1 (FSDP-only).",
                args.model,
                args.tp,
            )
            args.tp = 1
        hf_dtype = torch.float32 if args.fp32 else torch.bfloat16
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        logger.info(
            "Loading HF model %s (dtype=%s)%s",
            args.model,
            hf_dtype,
            " with HF_TOKEN" if hf_token else "",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=hf_dtype,
            token=hf_token,
        )
    else:
        model = Transformer.from_model_args(config)
    mstr = summarize_model(
        model,
        verbose=False,
        depth=2,
    )
    logger.info(f"\n{mstr}")
    model.to(device)

    # FLOPs estimation: try the exact fake-tensor path first, fall back
    # to the linear-scaling probe if it fails.
    #
    # FAKE-TENSOR PATH (preferred): runs the forward+backward at the
    # real (batch, seq) shape under FakeTensorMode (shape-only tensors,
    # no allocations → no OOM) with sdpa_kernel(MATH) forced so SDPA
    # decomposes into bmms that FlopCounterMode can see. Exact count,
    # attention included.
    #
    # LINEAR-SCALING PROBE (fallback): runs at (1, 128) with real
    # tensors and scales by token ratio. Exact for O(seq·dim) MLP/proj
    # ops, but UNDER-COUNTS attention because the O(seq²·dim) Q·Kᵀ and
    # attn·V matmuls don't scale linearly. Worse, on CPU and on fused
    # SDPA backends (flash / efficient / cuDNN), FlopCounterMode often
    # reports zero for the SDPA op entirely — so both probe and actual
    # silently drop attention from the count. Reported MFU is then a
    # lower bound: real utilization is at least the printed number,
    # often significantly higher on long-seq runs.
    _model_flops = try_estimate_fake(
        model, (args.batch_size, args.seq_len)
    )
    if _model_flops > 0:
        logger.info(
            "FLOPs counted exactly via FakeTensorMode at shape "
            "(batch=%d, seq=%d): %.3e (includes attention).",
            args.batch_size,
            args.seq_len,
            _model_flops,
        )
    else:
        _flops_probe_batch = 1
        _flops_probe_seq = min(128, args.seq_len)
        _flops_probe = try_estimate(
            model, (_flops_probe_batch, _flops_probe_seq)
        )
        _actual_tokens = args.batch_size * args.seq_len
        _probe_tokens = _flops_probe_batch * _flops_probe_seq
        _model_flops = int(
            _flops_probe * _actual_tokens / max(_probe_tokens, 1)
        )
        if args.seq_len > _flops_probe_seq:
            logger.warning(
                "Fake-tensor FLOP estimate failed; falling back to "
                "linear-scaling probe (probe seq=%d -> actual seq=%d). "
                "This under-counts O(seq^2) attention by ~%dx; reported "
                "MFU is a lower bound (real utilization is at least this "
                "high).",
                _flops_probe_seq,
                args.seq_len,
                args.seq_len // _flops_probe_seq,
            )

    # FSDP2 mixed-precision policy (param in bf16, reduce in fp32). None when
    # --fp32 is set (pure fp32 for NaN debugging).
    mp_config: Optional[MixedPrecisionPolicy] = None
    if not args.fp32:
        # reduce_dtype: fp32 gradient reduce-scatter is more accurate, but for
        # a large-vocab output projection (e.g. agpt's 256K) the single
        # reduce-scatter tensor can exceed CCL's ~2GB-per-message MPI limit
        # (256K*2048*4B = 2.1GB) → `atl_mpi !req.is_completed`. Set
        # EZPZ_REDUCE_DTYPE=bf16 to halve the collective size (1.05GB) and
        # stay under the limit. Validated against an explicit set: a typo
        # silently falling back to fp32 would re-trigger the very CCL
        # failure this escape hatch exists to avoid, so raise instead.
        _reduce_dtype_env = os.environ.get("EZPZ_REDUCE_DTYPE", "fp32")
        _reduce_dtype_key = _reduce_dtype_env.lower()
        if _reduce_dtype_key == "fp32":
            _reduce_dtype = torch.float32
        elif _reduce_dtype_key in ("bf16", "bfloat16"):
            _reduce_dtype = torch.bfloat16
        else:
            raise ValueError(
                f"Invalid EZPZ_REDUCE_DTYPE={_reduce_dtype_env!r}. Expected "
                "one of: 'fp32', 'bf16', 'bfloat16' (case-insensitive)."
            )
        mp_config = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=_reduce_dtype,
        )
    if is_hf_model:
        # HF path: FSDP2-only wrap (no TP — the TP plan is ezpz-specific).
        # Apply activation checkpointing first (HF models use their own
        # gradient_checkpointing_enable inside _apply_activation_checkpointing),
        # then fully_shard each decoder block + the root.
        if args.activation_checkpoint != "none":
            _apply_activation_checkpointing(model, args.activation_checkpoint)

        # Find the decoder block stack: the SINGLE deepest non-empty
        # ModuleList (e.g. `model.model.layers`). Collecting every ModuleList
        # would over-shard MoE/multimodal models; the deepest one is reliably
        # the decoder stack (matches _find_block_list / HF's _no_split_modules).
        deepest_modlist: Optional[torch.nn.ModuleList] = None
        deepest_depth = -1
        deepest_len = -1
        for name, module in model.named_modules():
            if (
                isinstance(module, torch.nn.ModuleList)
                and len(module) > 0
            ):
                depth = name.count(".")
                if depth > deepest_depth or (
                    depth == deepest_depth and len(module) > deepest_len
                ):
                    deepest_depth = depth
                    deepest_len = len(module)
                    deepest_modlist = module

        hf_fsdp_kwargs = {
            "mesh": device_mesh["dp"],
            "reshard_after_forward": _reshard_after_forward(
                args.sharding_strategy
            ),
        }
        if mp_config is not None:
            hf_fsdp_kwargs["mp_policy"] = mp_config
        if deepest_modlist is not None:
            for block in deepest_modlist:
                fully_shard(block, **hf_fsdp_kwargs)
        else:
            logger.warning(
                "HF model: no decoder ModuleList found; sharding only the "
                "root module (per-layer memory savings will be reduced)."
            )
        fully_shard(model, **hf_fsdp_kwargs)
        _configure_fsdp_gradient_division(model)
    else:
        # TP + FSDP2. parallelize() applies activation checkpointing per
        # block BEFORE fully_shard (correct FSDP2 ordering), so we do NOT
        # re-apply it afterwards.
        model = parallelize(
            model,
            device_mesh,
            mp_config,
            sharding_strategy=args.sharding_strategy,
            activation_checkpoint=args.activation_checkpoint,
        )
    if args.compile:
        # Activation-memory budget for the inductor min-cut partitioner.
        # Default 1.0 = save every activation (no recompute); < 1.0 lets the
        # compiler recompute a fraction of activations in backward to cut
        # peak memory. This is the knob torchtitan's MemoryBudgetAC sets
        # (0.5) — it's why TT fits a larger batch than this example for the
        # same model. Global config, applies to every compiled block below.
        if args.act_mem_budget != 1.0:
            import torch._functorch.config as _functorch_config

            _functorch_config.activation_memory_budget = args.act_mem_budget
            logger.info(
                "Set activation_memory_budget=%.3f (inductor will recompute "
                "activations in backward to cut peak memory).",
                args.act_mem_budget,
            )
        if args.activation_checkpoint != "none":
            # --ac + --compile together trip an upstream AOTAutograd bug:
            #   AssertionError: expected all tensors_saved_with_vc_check to
            #   be Tensors, got [... DeviceMesh]
            # The non-reentrant checkpoint_wrapper saves a DeviceMesh into
            # the autograd graph, which the compiled-backward saved-tensors
            # check rejects. Under FSDP2 every sharded module carries a
            # DeviceMesh, so this fires even at --tp 1 (with FSDP1 it
            # required --tp > 1). Repro + triage:
            # torchtitan/.../docs/upstream-issues/repro_devicemesh_in_saved_tensors.py
            # Not fixable here — drop one of --ac / --compile. (With FSDP2
            # you typically no longer need --ac for memory; it was a
            # workaround for the FSDP1 OOM that FSDP2 already resolves.)
            logger.warning(
                "--compile + --activation-checkpoint=%s will likely crash "
                "with an AOTAutograd 'tensors_saved_with_vc_check ... "
                "DeviceMesh' assertion (upstream bug; fires under FSDP2 even "
                "at --tp 1). Drop one of --ac / --compile. Note FSDP2 usually "
                "removes the need for --ac (it fixed the FSDP1 OOM).",
                args.activation_checkpoint,
            )
        # Compile each TransformerBlock individually rather than the whole
        # model. This is what torchtitan does (apply_compile in
        # torchtitan/models/.../infra/parallelize.py) and it dodges the
        # Dynamo + DTensor _MaskPartial graph break that whole-model
        # compile hits on TP-wrapped tok_embeddings:
        #
        #   RuntimeError when making fake tensor call: call_method
        #   redistribute(...) on DTensor(_MaskPartial(...))
        #
        # The embedding's RowwiseParallel output_fn does a redistribute
        # from _MaskPartial → Shard(1), which Dynamo can't trace under
        # fake tensors. Excluding the embedding from compile (and only
        # compiling the blocks) keeps the speedup where it matters
        # (attention + MLP, the repeated structure) without exposing
        # Dynamo to the TP output transform. Bonus: compile cost is paid
        # once for one block and reused across N layers, not N times.
        #
        # Find the block list: ezpz Transformer has `.layers`, HF
        # decoder-only models nest it as `.model.layers`.
        block_container = None
        if hasattr(model, "layers"):
            block_container = model.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            block_container = model.model.layers
        if block_container is None:
            logger.warning(
                "Could not find a TransformerBlock list (model.layers or "
                "model.model.layers) — falling back to whole-model "
                "torch.compile, which may hit DTensor graph breaks."
            )
            model = torch.compile(model, mode=args.compile_mode)
        else:
            logger.info(
                "Compiling each TransformerBlock with torch.compile"
                "(mode=%s, fullgraph=True) — %d blocks.",
                args.compile_mode,
                len(block_container),
            )
            for layer_id, block in block_container.named_children():
                compiled = torch.compile(
                    block, mode=args.compile_mode, fullgraph=True
                )
                block_container.register_module(layer_id, compiled)
    base_model = model
    if not hasattr(base_model, "layers"):
        base_model = getattr(model, "_fsdp_wrapped_module", model)
    act_activations: dict[str, torch.Tensor] = {}
    act_handles: list[torch.utils.hooks.RemovableHandle] = []
    if track_hist and track_act_hist and ezpz.get_rank() == 0 and not is_hf_model:
        # `_register_activation_hooks` indexes into `model.layers[i]`, which
        # is ezpz-Transformer specific. HF models nest blocks under
        # `model.model.layers` (Llama/Mistral) or `model.gpt_neox.layers`
        # (GPT-NeoX) etc., so the hook registration would key-error. Skip
        # the hooks for HF runs; the rest of the metrics still work.
        hist_layers_spec = os.environ.get(
            "EZPZ_HIST_LAYERS", f"0,{config.n_layers - 1}"
        )
        layer_ids = _parse_hist_layers(hist_layers_spec, config.n_layers)
        act_activations, act_handles = _register_activation_hooks(
            base_model, layer_ids
        )
    logger.info(f"Creating optimizer=AdamW with lr={args.lr}")

    # Prefer the fused AdamW kernel (single kernel for the whole param
    # update) — it's what torchtitan uses and it's measurably faster than
    # `foreach` on XPU. Fall back to foreach if fused isn't supported for
    # this build/device (older torch, CPU, etc.).
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            fused=True,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "Fused AdamW unavailable (%s); falling back to foreach=True.", exc
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            foreach=True,
        )

    # reuse device for input placement

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
            seq_length=args.seq_len,
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

    # outdir = Path(args.outdir).joinpath(ezpz.utils.get_timestamp())
    metrics_path = Path(outdir).joinpath(
        f"metrics-{ezpz.distributed.get_rank()}.jsonl"
    )
    Path(outdir).mkdir(parents=True, exist_ok=True)
    history = ezpz.history.History(
        project_name=WBPROJ_NAME,
        config={"args": vars(args), **ezpz.get_dist_info()},
        outdir=outdir,
        report_dir=outdir,
        report_enabled=True,
        jsonl_path=metrics_path,
        jsonl_overwrite=True,
        # Disable cross-rank history aggregation while profiling (either
        # profiler) — the all-gather of per-rank metrics perturbs the very
        # step times the profiler is measuring.
        distributed_history=(
            1 < world_size <= 384
            and not getattr(args, "pytorch_profiler", False)
            and not getattr(args, "pyinstrument_profiler", False)
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
            # HF causal-LM models return a CausalLMOutput dataclass with a
            # `.logits` tensor; ezpz's Transformer returns logits directly.
            if hasattr(pred, "logits"):
                pred = pred.logits
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
            ezpz.distributed.synchronize()
            t1 = perf_counter()
            tp_mod = getattr(ezpz, "tp", None)
            tp_rank = (
                getattr(tp_mod, "get_tensor_parallel_rank", lambda: 0)()
                if tp_mod is not None
                else 0
            )
            # First-step finite/max debug stats. Gated behind
            # EZPZ_TRACK_LOGITS because `torch.isfinite(pred)` allocates a
            # full `(B, T, vocab)`-shaped bool tensor on the un-reduced
            # logits — at agpt's 256K vocab and long seq that's multiple GB
            # materialized *before* the loss, which can OOM a run that would
            # otherwise fit (the loss itself may be chunked/compiled to stay
            # bounded, but this debug probe is not). Off by default.
            if track_logits and epoch == 0 and idx == 0:
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
            loss = _compute_loss(
                pred,
                labels,
                impl=args.loss_impl,
                ignore_index=-100,
                chunk_size=args.loss_chunk_size,
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
            ezpz.distributed.synchronize()
            t2 = perf_counter()
            global_step += 1
            # Advance the torch.profiler schedule once per optimizer step.
            # No-op when not profiling (profiler is None).
            if profiler is not None:
                profiler.step()
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
                metrics.update(_collect_param_grad_stats(model, device))
                metrics["opt/iter"] = (global_step,)
                metrics["opt/lr"] = float(optimizer.param_groups[0]["lr"])
                metrics["input/iter"] = (global_step,)
                metrics["input/max"] = float(x.max().item())
                metrics["input/min"] = float(x.min().item())
                metrics["labels/valid"] = float((labels != -100).sum().item())
                if track_logits:
                    pred_finite = torch.isfinite(pred)
                    metrics["logits/nonfinite"] = float(
                        (~pred_finite).sum().item()
                    )
                    metrics["logits/max_abs"] = float(pred.abs().max().item())
                if track_hist and ezpz.get_rank() == 0:
                    logits_sample = _sample_tensor_values(pred, hist_samples)
                    if logits_sample is not None:
                        logits_hist = _histogram_dict(logits_sample, hist_bins)
                        if logits_hist is not None:
                            metrics[f"hist/{dataset_tag}/logits"] = logits_hist
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
                            act_hist = _histogram_dict(act_sample, hist_bins)
                            if act_hist is not None:
                                metrics[
                                    f"hist/{dataset_tag}/activations/{act_key}"
                                ] = act_hist
                    if history.tracker.get_backend("wandb") is not None:
                        _wandb_log_histograms(
                            metrics, step=global_step, enabled=track_hist
                        )
            # Reuse the train/dt we already computed above so the MFU
            # denominator can never silently drift from the reported
            # step time.
            dt_step = float(metrics["train/dt"])  # type: ignore[arg-type]
            if _model_flops > 0 and dt_step > 0:
                metrics["train/tflops"] = _model_flops / dt_step / 1e12
                metrics["train/mfu"] = compute_mfu(_model_flops, dt_step)
            # Throughput. `--batch-size` is the PER-DP-RANK micro-batch and
            # `local_seq_len` is this rank's (post-SP-slice) sequence length,
            # so tokens processed per rank per step is batch_size *
            # local_seq_len.
            #   - train/tps_per_gpu : per-rank tokens/sec (torchtitan's `tgs`)
            #   - train/tps         : global tokens/sec across ALL ranks
            # Multiply by world_size, not dp_size: under sequence parallelism
            # each TP rank holds a DISTINCT seq shard (labels are sliced via
            # _slice_for_sequence_parallel), so the TP ranks process distinct
            # tokens too. Using dp_size would undercount global tps by ~tp.
            # At tp=1, world_size == dp_size so this is unchanged.
            if dt_step > 0:
                tokens_per_rank = args.batch_size * local_seq_len
                metrics["train/tps_per_gpu"] = tokens_per_rank / dt_step
                metrics["train/tps"] = tokens_per_rank * world_size / dt_step
            # Device memory: empty on CPU/MPS, 4 keys on CUDA/XPU.
            metrics |= ezpz.get_memory_metrics(prefix="train/")
            history.update(metrics, summarize=False)
            history.log_metrics(
                metrics,
                logger=logger,
                debug_prefixes=("hist/", "grad/", "input/", "labels/", "param/"),
                include_summary=True,
                rank0_only_summary=True,
            )
            if epoch == 0 and idx == 0:
                logger.info(f"{x.shape}")
    if act_handles:
        for handle in act_handles:
            handle.remove()
    ezpz.distributed.barrier()
    logger.info("Finished 2D training")
    return history


@ezpz.timeitlogit(rank=ezpz.get_rank())
def main(args: argparse.Namespace) -> int:
    """Entrypoint to set up distributed context and dispatch training."""
    ezpz.silence_noisy_loggers()
    t0 = time.perf_counter()
    rank = ezpz.distributed.setup_torch(tensor_parallel_size=args.tp, seed=args.seed)
    t_setup = time.perf_counter()
    if rank == 0:
        jstr = json.dumps(vars(args), indent=2, sort_keys=True, default=str)
        logger.info(f"config:\n{jstr}")
    base_dir = args.outdir if args.outdir else None
    outdir = get_example_outdir(WBPROJ_NAME, base_dir=base_dir)
    logger.info("Outputs will be saved to %s", outdir)
    # Tracker setup is handled by History constructor (inside train())
    train_start = time.perf_counter()
    # nullcontext (prof=None) unless --profile / --pyinstrument-profiler set.
    with profiling_context_from_args(args, outdir) as prof:
        history = train(args=args, outdir=outdir, profiler=prof)
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
    history.tracker.log(
        {
            (f"timings/{k}" if not k.startswith("timings/") else k): v
            for k, v in timings.items()
        }
    )
    if ezpz.get_rank() == 0:
        dataset = history.finalize(
            outdir=outdir,
            run_name=WBPROJ_NAME,
            dataset_fname="train",
        )
        del dataset  # logged by finalize()
    return 0


if __name__ == "__main__":
    args = parse_args()
    main(args)
    ezpz.distributed.cleanup()
    sys.exit(0)
