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

The data-parallel dim can itself be split for HSDP via --dp-replicate /
--dp-shard: weights are replicated across `dp_replicate` groups and
sharded within each `dp_shard` group (dp = dp_replicate * dp_shard). The
defaults (dp_replicate=1, dp_shard=-1) give a single flat sharded dp dim.

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
from ezpz.cli.help_format import DefaultsFormatter
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
from ezpz.tinker import lora as _lora
from ezpz.tinker import step as _tinker_step

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
# is a single knob, `reshard_after_forward` (bool). The CLI surface is
# `--reshard-after-forward {always,never}` (+ `--no-reshard-after-forward`),
# which names exactly that. Two behaviors exist:
#   - always (ZeRO-3, default): reshard params after forward  -> True
#       lowest memory, re-all-gathers params in backward.
#   - never  (ZeRO-2):          keep params gathered after fwd -> False
#       more memory, skips the backward all-gather.
RESHARD_POLICIES = ("always", "never")


def _reshard_arg(policy: str) -> bool:
    """Map a --reshard-after-forward policy to the FSDP2 bool.

    Only two behaviors exist under FSDP2: ``always`` -> True, ``never`` ->
    False. Validates against RESHARD_POLICIES so a bad programmatic value
    (the CLI already restricts via ``choices``) raises instead of silently
    resolving to True.
    """
    if policy not in RESHARD_POLICIES:
        raise ValueError(
            f"invalid reshard_after_forward policy {policy!r}; "
            f"expected one of {RESHARD_POLICIES}"
        )
    return policy != "never"


# Legacy `--sharding-strategy` values -> the new reshard policy. Kept only
# for the hidden, deprecated `--sharding-strategy` alias (see parse_args).
# `hybrid_shard*` are intentionally absent: they never did real HSDP under
# FSDP2 (use --dp-replicate / --dp-shard), so they hard-error instead of
# silently mapping to a reshard bool.
_LEGACY_SHARDING_TO_RESHARD = {
    "full_shard": "always",
    "shard_grad_op": "never",
    "no_shard": "never",
}
_LEGACY_HYBRID_SHARDING = ("hybrid_shard", "hybrid_shard_zero2")


def _resolve_reshard_after_forward(args: argparse.Namespace) -> None:
    """Fold the deprecated `--sharding-strategy` alias into `reshard_after_forward`.

    Mutates *args* in place. Called from ``parse_args`` so the module
    entrypoint and tests both see the normalized value. No-op when
    `--sharding-strategy` wasn't passed (the common path).

    - `hybrid_shard` / `hybrid_shard_zero2` -> hard error (SystemExit),
      pointing at --dp-replicate / --dp-shard for real HSDP.
    - `full_shard` -> always; `shard_grad_op` / `no_shard` -> never, with a
      deprecation warning (emitted on rank 0 only — parse_args runs on every
      rank, and AGENTS.md gates per-rank log lines to local_rank 0).
    - If both `--sharding-strategy` and the new flag are passed, the
      deprecated value is applied last (documented precedence).
    """
    ss = getattr(args, "sharding_strategy", None)
    if ss is None:
        return
    if ss in _LEGACY_HYBRID_SHARDING:
        raise SystemExit(
            f"--sharding-strategy={ss} is removed: it never performed real "
            "HSDP under FSDP2. Use --dp-replicate / --dp-shard for hybrid "
            "sharding, and --reshard-after-forward {always,never} for the "
            "ZeRO-3/ZeRO-2 reshard policy."
        )
    if ss not in _LEGACY_SHARDING_TO_RESHARD:
        raise SystemExit(f"unknown --sharding-strategy={ss!r}")
    mapped = _LEGACY_SHARDING_TO_RESHARD[ss]
    # parse_args() runs on every rank before setup_torch; gate the warnings
    # to rank 0 so a large launch doesn't emit N duplicate lines (AGENTS.md
    # "Logging at scale"). get_rank() reads the launcher's env vars, which
    # are already set at parse time.
    if ezpz.get_rank() == 0:
        logger.warning(
            "--sharding-strategy is deprecated; use --reshard-after-forward. "
            "Mapping --sharding-strategy=%s -> --reshard-after-forward=%s.",
            ss,
            mapped,
        )
        if ss == "no_shard":
            logger.warning(
                "--sharding-strategy=no_shard does NOT give replicated "
                "(ZeRO-0/DDP) params under FSDP2: parameters, gradients, and "
                "optimizer state are still sharded across the dp mesh. Only "
                "post-forward resharding is disabled (== "
                "--reshard-after-forward never). Use plain DDP if you need "
                "truly replicated parameters."
            )
    args.reshard_after_forward = mapped


def _resolve_dp_degrees(
    *, world_size: int, tp: int, dp_replicate: int, dp_shard: int
) -> tuple[int, int]:
    """Resolve the (dp_replicate, dp_shard) data-parallel degrees.

    Mirrors torchtitan's semantics:

    - ``dp_shard == -1`` means "use all remaining ranks", i.e.
      ``world_size // (dp_replicate * tp)``.
    - The product ``dp_replicate * dp_shard * tp`` must equal
      ``world_size``.

    With the defaults (``dp_replicate=1``, ``dp_shard=-1``) this yields
    ``dp_shard = world_size // tp`` and ``dp_replicate = 1`` — i.e. a single
    flat data-parallel dim (pure FSDP sharding), identical to the behavior
    before HSDP support was added.

    Returns the resolved ``(dp_replicate, dp_shard)`` tuple. Raises
    ``AssertionError`` with an explicit message on an invalid configuration.
    """
    assert dp_replicate >= 1, (
        f"--dp-replicate must be >= 1 (got {dp_replicate})"
    )
    assert dp_shard == -1 or dp_shard >= 1, (
        f"--dp-shard must be >= 1, or -1 for auto (got {dp_shard})"
    )
    if dp_shard < 0:
        denom = dp_replicate * tp
        assert world_size % denom == 0, (
            f"cannot auto-resolve --dp-shard: WORLD_SIZE({world_size}) is not "
            f"divisible by dp_replicate({dp_replicate}) * tp({tp}) = {denom}"
        )
        dp_shard = world_size // denom
    assert dp_replicate * dp_shard * tp == world_size, (
        f"dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * tp({tp}) "
        f"= {dp_replicate * dp_shard * tp} != WORLD_SIZE({world_size})"
    )
    return dp_replicate, dp_shard


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
    non-ignored tokens). We accumulate the SUM of per-token losses across
    chunks and divide by the total valid-token count once, so the result
    matches mean reduction exactly (autograd handles the constant scale
    through the division).

    .. warning::
        This chunks only the **forward** pass. Because every chunk feeds a
        single autograd graph over the full ``flat_logits``, ``backward()``
        still materializes the entire ``(B*T, vocab)`` logits gradient at
        once — it does NOT bound the backward transient. Measured on
        agpt-2b (bs=2, seq=8192, vocab=256K) this OOMs in ``loss.backward()``
        (~15.6 GiB logits grad) where ``--loss-impl=compiled`` fits. Prefer
        ``compiled`` for large-vocab models; ``chunked`` only meaningfully
        helps the forward transient and is not a backward-memory lever.
        (A true backward bound would need a custom ``autograd.Function`` or
        activation checkpointing around the per-chunk CE.)
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


class _CrossEntropyChunkedBackward(torch.autograd.Function):
    """Row-chunked cross-entropy that also bounds the *backward* transient.

    Numerically identical to :func:`_cross_entropy_eager` (mean over the
    non-ignored tokens), but a custom autograd Function whose backward
    recomputes each row-chunk's gradient on the fly instead of letting
    autograd hold a full ``(B*T, vocab)`` softmax/logsumexp graph. So the
    peak *transient beyond the returned grad buffer* is ``chunk_size*vocab``
    rather than a second full-size graph (unlike :func:`_cross_entropy_chunked`,
    which chunks only the forward and still OOMs in backward at large vocab).

    Scope / when to use this:
      - It is a **general, model-agnostic** method: it operates on any already
        materialized ``(B*T, vocab)`` logits, so it works for **HF models**
        (where ``fused-linear`` cannot go — that needs the ezpz Transformer's
        hidden-state path) and needs **no torch.compile** (handy when inductor
        is unavailable/flaky, or for debugging).
      - It helps at **moderate** vocab/seq where eager's extra softmax buffer
        is what pushes you over: it saves ~one full ``(B*T, vocab)`` buffer
        vs eager.

    What it does NOT do: the returned ``grad_logits`` is still
    ``(B*T, vocab)``, and the *input* logits are already materialized, so two
    logit-sized buffers remain. At very large vocab × long seq (e.g. agpt-2b
    256K vocab, seq=8192: two ~16.8 GiB fp32 buffers) that still OOMs — there,
    use ``fused-linear`` (never materializes logits) or ``compiled``. This
    bounds the *graph*; ``fused-linear`` bounds the *buffers*.

    Returns mean-reduced loss; gradient w.r.t. logits is
    ``(softmax(logits) - onehot(labels)) / valid`` with ignored rows zeroed,
    matching ``F.cross_entropy(reduction="mean", ignore_index=...)``.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
        chunk_size: int,
    ) -> torch.Tensor:
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels.reshape(-1)
        n_rows = flat_labels.shape[0]

        valid = (flat_labels != ignore_index).sum()
        denom = valid.clamp(min=1).to(torch.float32)

        total = torch.zeros((), dtype=torch.float32, device=logits.device)
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            # reduction="sum" so chunk contributions add; fp32 to match eager.
            total = total + F.cross_entropy(
                flat_logits[start:end],
                flat_labels[start:end],
                ignore_index=ignore_index,
                reduction="sum",
            ).to(torch.float32)

        # Save the (already-materialized) logits + labels — NOT a second graph.
        ctx.save_for_backward(flat_logits, flat_labels, denom)
        ctx.ignore_index = ignore_index
        ctx.chunk_size = chunk_size
        ctx.logits_shape = logits.shape
        ctx.logits_dtype = logits.dtype
        return total / denom

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        flat_logits, flat_labels, denom = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        chunk_size = ctx.chunk_size

        # The returned grad buffer is (B*T, vocab); fill it chunk-by-chunk so
        # the per-chunk softmax (chunk*vocab) is the only transient beyond it.
        grad_logits = torch.empty_like(flat_logits)
        scale = (grad_output / denom).to(torch.float32)

        n_rows = flat_labels.shape[0]
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            logits_chunk = flat_logits[start:end].to(torch.float32)
            labels_chunk = flat_labels[start:end]
            # softmax(logits) - onehot(labels), in fp32 (matches eager grad).
            probs = torch.softmax(logits_chunk, dim=-1)
            ignored = labels_chunk == ignore_index
            safe = torch.where(ignored, torch.zeros_like(labels_chunk), labels_chunk)
            probs.scatter_add_(
                -1,
                safe.unsqueeze(-1),
                torch.full(
                    (safe.shape[0], 1), -1.0, dtype=probs.dtype, device=probs.device
                ),
            )
            grad_chunk = probs * scale
            grad_chunk[ignored] = 0.0  # ignored rows contribute zero gradient
            grad_logits[start:end] = grad_chunk.to(grad_logits.dtype)

        grad_logits = grad_logits.reshape(ctx.logits_shape)
        # grads for (logits, labels, ignore_index, chunk_size)
        return grad_logits, None, None, None


def _cross_entropy_chunked_backward(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Functional wrapper around :class:`_CrossEntropyChunkedBackward`."""
    return _CrossEntropyChunkedBackward.apply(
        logits, labels, ignore_index, chunk_size
    )


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Vocab-parallel cross-entropy on plain (non-DTensor) local tensors.

    Mirrors torchtitan's ``_LossParallelCrossEntropy``
    (torchtitan/components/loss.py) and the semantics of
    ``torch.distributed.tensor.parallel.loss_parallel`` — but operates on
    this rank's local ``[N, vocab/tp]`` vocab shard + a process group,
    instead of requiring DTensor inputs. Forward uses three TP all-reduces
    (max for stable log-softmax, sumexp for the denominator, gather for the
    target logprob); backward is fused (softmax − onehot) with zero
    collectives. Handles ``IGNORE_INDEX`` and uneven final shards.

    This bounds the **vocab** dimension across TP ranks (each holds
    ``vocab/tp``), so it only reduces memory when ``tp > 1``. It does NOT
    bound the row (``B*T``) dimension — combine with row-chunking
    (``fused-linear``) for both. Returns mean-reduced loss to match
    :func:`_cross_entropy_eager`.

    Attribution: algorithm adapted from torchtitan (BSD-licensed); ezpz does
    not import torchtitan (it is a sibling checkout, not a dependency).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        local_logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
        global_vocab_size: int,
        tp_group,
    ) -> torch.Tensor:
        import torch.distributed as dist
        import torch.distributed._functional_collectives as funcol

        logits_2d = local_logits.reshape(-1, local_logits.size(-1)).float()
        labels_1d = labels.reshape(-1)

        tp_world = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)
        # Fail fast on a degenerate sharding: if the TP degree exceeds the
        # global vocab, the highest ranks get an empty (local_vocab==0) shard,
        # which would otherwise blow up later in gather/indexing with a
        # confusing error. This config makes no sense for vocab-parallel CE.
        if tp_world > global_vocab_size:
            raise ValueError(
                f"vocab-parallel CE requires global_vocab_size "
                f"({global_vocab_size}) >= TP world size ({tp_world}); "
                "some ranks would hold an empty vocab shard. Use a smaller "
                "--tp or a different --loss-impl."
            )
        chunk = (global_vocab_size + tp_world - 1) // tp_world
        vocab_start = min(global_vocab_size, chunk * tp_rank)
        vocab_end = min(global_vocab_size, vocab_start + chunk)
        local_vocab = max(0, vocab_end - vocab_start)
        if logits_2d.shape[-1] != local_vocab:
            raise ValueError(
                "_VocabParallelCrossEntropy expected local vocab "
                f"{local_vocab} for global {global_vocab_size}, got "
                f"{logits_2d.shape[-1]}."
            )

        # 1) all-reduce MAX for numerically stable distributed log-softmax.
        local_max = torch.amax(logits_2d, dim=-1, keepdim=True)
        local_max = funcol.all_reduce(
            local_max, reduceOp=dist.ReduceOp.MAX.name, group=tp_group
        )
        shifted = logits_2d - local_max
        # 2) all-reduce SUM of exp for the global softmax denominator.
        sumexp = torch.sum(torch.exp(shifted), dim=-1, keepdim=True)
        sumexp = funcol.all_reduce(
            sumexp, reduceOp=dist.ReduceOp.SUM.name, group=tp_group
        )
        log_probs = shifted - torch.log(sumexp)

        # 3) gather the target token's logprob from its owner rank.
        ignored = labels_1d == ignore_index
        safe = torch.where(ignored, torch.zeros_like(labels_1d), labels_1d)
        out_of_range = (safe < vocab_start) | (safe >= vocab_end)
        local_labels = (safe - vocab_start).clamp_(min=0)
        local_labels[out_of_range] = 0
        picked = torch.gather(log_probs, -1, local_labels.unsqueeze(-1))
        picked[out_of_range.unsqueeze(-1)] = 0.0
        picked = funcol.all_reduce(
            picked, reduceOp=dist.ReduceOp.SUM.name, group=tp_group
        )
        nll = -picked.squeeze(-1)
        nll = torch.where(ignored, torch.zeros_like(nll), nll)

        valid = (~ignored).sum().clamp(min=1).to(torch.float32)

        ctx.save_for_backward(log_probs, labels_1d, valid)
        ctx.vocab_start = vocab_start
        ctx.local_vocab = local_vocab
        ctx.ignore_index = ignore_index
        ctx.logits_shape = local_logits.shape
        ctx.logits_dtype = local_logits.dtype
        return nll.sum() / valid

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        log_probs, labels_1d, valid = ctx.saved_tensors
        ignored = labels_1d == ctx.ignore_index
        safe = torch.where(ignored, torch.zeros_like(labels_1d), labels_1d)
        out_of_range = (safe < ctx.vocab_start) | (
            safe >= ctx.vocab_start + ctx.local_vocab
        )
        local_labels = (safe - ctx.vocab_start).clamp_(min=0)
        local_labels[out_of_range] = 0

        # grad = (softmax − onehot)/valid * grad_out, onehot only on owner rank.
        grad = torch.exp(log_probs)
        row = torch.arange(local_labels.shape[0], device=local_labels.device)
        # subtract 1 at the target col on the owning rank (0 if out-of-range).
        grad[row, local_labels] -= (~out_of_range).to(grad.dtype)
        scale = (grad_output / valid).to(grad.dtype)
        grad = grad * scale
        grad[ignored] = 0.0
        grad = grad.reshape(ctx.logits_shape).to(ctx.logits_dtype)
        return grad, None, None, None, None


def _cross_entropy_vocab_parallel(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    global_vocab_size: int,
    tp_group,
) -> torch.Tensor:
    """Functional wrapper around :class:`_VocabParallelCrossEntropy`."""
    return _VocabParallelCrossEntropy.apply(
        local_logits, labels, ignore_index, global_vocab_size, tp_group
    )


def _cross_entropy_fused_linear(
    hidden: torch.Tensor,
    output_module: nn.Module,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Fused output-projection + cross-entropy (Liger / Cut-CE style).

    Computes mean-reduced CE over ``output_module(hidden)`` **without ever
    materializing the full ``(N, vocab)`` logits tensor**. The hidden states
    are split into row-chunks; for each chunk the output projection + CE are
    run under :func:`torch.utils.checkpoint.checkpoint`, so the chunk's
    ``(chunk, vocab)`` logits are freed after the forward and recomputed one
    chunk at a time in backward. Peak logits transient is ``chunk*vocab``
    instead of ``N*vocab`` — bounding BOTH the row and vocab dimensions of
    the loss (chunked-backward bounds the backward graph but still holds two
    logit-sized buffers; loss-parallel bounds only the vocab dim across TP
    ranks; this bounds both, and at tp=1).

    Critically this calls the output projection as a **module**
    (``output_module(h_chunk)``), NOT ``h @ weight.T`` on a raw weight. Under
    FSDP2 the projection weight is a sharded DTensor unsharded only inside the
    module's forward (via FSDP hooks); going through the module lets FSDP
    all-gather the weight and reduce-scatter its gradient correctly. A
    hand-rolled ``h @ weight.T`` on the sharded weight raises
    "mixed Tensor and DTensor" (and bypassing it with ``.full_tensor()`` would
    silently break gradient flow to the sharded param).

    Numerically matches ``F.cross_entropy(output_module(h), labels,
    reduction="mean", ignore_index=...)``. Attribution: the chunked
    fused-linear-CE technique is from Liger-Kernel / Apple Cut-CE /
    torchtune / torchtitan's ChunkedLossWrapper; this is an independent
    pure-PyTorch implementation.
    """
    from torch.utils.checkpoint import checkpoint

    h2d = hidden.reshape(-1, hidden.size(-1))
    labels_1d = labels.reshape(-1)
    n_rows = labels_1d.shape[0]
    # Denominator for mean reduction over non-ignored tokens (fp32, matches
    # eager). Sum per-chunk CE then divide once so autograd scales correctly.
    valid = (labels_1d != ignore_index).sum().clamp(min=1).to(torch.float32)

    def _chunk_loss(h_chunk: torch.Tensor, lbl_chunk: torch.Tensor) -> torch.Tensor:
        logits_c = output_module(h_chunk)  # FSDP unshards weight here
        if hasattr(logits_c, "logits"):
            logits_c = logits_c.logits
        return F.cross_entropy(
            logits_c.float(),
            lbl_chunk,
            ignore_index=ignore_index,
            reduction="sum",
        ).to(torch.float32)

    total = torch.zeros((), dtype=torch.float32, device=hidden.device)
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        h_chunk = h2d[start:end]
        lbl_chunk = labels_1d[start:end]
        if h_chunk.requires_grad:
            # checkpoint frees the chunk's logits after forward; recomputed
            # one chunk at a time in backward -> peak logits = chunk*vocab.
            total = total + checkpoint(
                _chunk_loss, h_chunk, lbl_chunk, use_reentrant=False
            )
        else:
            # eval / no-grad: no checkpoint needed.
            total = total + _chunk_loss(h_chunk, lbl_chunk)
    return total / valid


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


def _localize_logits_for_loss(logits: "torch.Tensor") -> "torch.Tensor":
    """Return a plain tensor of logits for non-loss-parallel CE.

    At tp>1 the output ColwiseParallel uses ``output_layouts=Replicate()``,
    ``use_local_output=False``, so ``logits`` is a REPLICATED ``DTensor``
    while ``labels`` is a plain tensor. Plain ``F.cross_entropy(DTensor,
    Tensor)`` raises "got mixed torch.Tensor and DTensor". Localizing to
    the per-rank tensor (which, being Replicate, holds the full vocab)
    fixes it with no numeric change; ``to_local()`` is differentiable so
    grads still reach the sharded output weight. Mirrors torchtitan
    ``components/loss.py``.

    No-op for a plain tensor (tp=1 / HF path), so it is always safe to call.

    Raises:
        RuntimeError: if ``logits`` is a vocab-SHARDED ``DTensor``
            (``Shard(-1)``). That is vocab-parallel logits and must go
            through ``--loss-impl=loss-parallel``; localizing here would
            silently compute CE on a partial vocab. Explicit raise (not
            ``assert``) so it survives ``python -O``.
    """
    if not hasattr(logits, "to_local"):
        return logits
    from torch.distributed._tensor import Replicate as _Replicate

    if not all(isinstance(p, _Replicate) for p in logits.placements):
        raise RuntimeError(
            "non-loss-parallel loss expects Replicate logits; got "
            f"placements={logits.placements}. Vocab-sharded logits must "
            "use --loss-impl=loss-parallel."
        )
    return logits.to_local()


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
    if impl == "chunked-backward":
        return _cross_entropy_chunked_backward(
            logits, labels, ignore_index=ignore_index, chunk_size=chunk_size
        )
    if impl == "compiled":
        return _cross_entropy_compiled(
            logits, labels, ignore_index=ignore_index
        )
    if impl == "eager":
        return _cross_entropy_eager(logits, labels, ignore_index=ignore_index)
    # `fused-linear` and `loss-parallel` are NOT logits-based CE variants — they
    # are dispatched at the call site (they need the output module / a vocab
    # shard + TP group), and when their specialized path is disabled the caller
    # is expected to have normalized `loss_impl` to a supported value (e.g.
    # `compiled`). Reaching here with one of those (or any unknown) means that
    # normalization was missed; fail loudly instead of silently running eager CE
    # — which would reintroduce the full-(B*T, vocab) logits/grad OOM these modes
    # exist to avoid.
    raise ValueError(
        f"_compute_loss got unhandled impl={impl!r}. Expected one of "
        "{'eager', 'chunked', 'chunked-backward', 'compiled'}. "
        "('fused-linear'/'loss-parallel' are dispatched separately and must be "
        "normalized to a supported impl when their specialized path is disabled.)"
    )


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
        formatter_class=DefaultsFormatter,
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
        "--dp-replicate",
        type=int,
        default=1,
        help=(
            "Data-parallel REPLICATE degree (HSDP outer dim). Weights are "
            "replicated across this many groups; within each group they are "
            "sharded across --dp-shard ranks. Default 1 = no replication "
            "(pure FSDP sharding, i.e. today's behavior). Set >1 for HSDP "
            "(e.g. shard within a node, replicate across nodes). Mirrors "
            "torchtitan's data_parallel_replicate_degree. Constraint: "
            "dp_replicate * dp_shard * tp == WORLD_SIZE."
        ),
    )
    parser.add_argument(
        "--dp-shard",
        type=int,
        default=-1,
        help=(
            "Data-parallel SHARD degree (FSDP inner dim). Weights are "
            "sharded across this many ranks within each replicate group. "
            "Default -1 = 'use all remaining ranks' = "
            "WORLD_SIZE / (dp_replicate * tp), which reproduces today's "
            "flat data-parallel behavior. Mirrors torchtitan's "
            "data_parallel_shard_degree."
        ),
    )
    parser.add_argument(
        "--reshard-after-forward",
        dest="reshard_after_forward",
        nargs="?",
        const="always",
        default="always",
        choices=list(RESHARD_POLICIES),
        help=(
            "FSDP2 reshard_after_forward policy (memory vs. comm tradeoff). "
            "`always` (default, ZeRO-3): reshard params after forward — "
            "lowest memory, re-all-gathers params in backward. `never` "
            "(ZeRO-2): keep params gathered after forward — more memory, "
            "skips the backward all-gather. Bare `--reshard-after-forward` "
            "== `always`; `--no-reshard-after-forward` == `never`. For HSDP "
            "(replicate + shard) use --dp-replicate / --dp-shard."
        ),
    )
    parser.add_argument(
        "--no-reshard-after-forward",
        dest="reshard_after_forward",
        action="store_const",
        const="never",
        help="Alias for --reshard-after-forward never (ZeRO-2).",
    )
    # Deprecated legacy alias, hidden from --help. Resolved post-parse by
    # _resolve_reshard_after_forward: full_shard->always, shard_grad_op/
    # no_shard->never (with a deprecation warning), hybrid_shard*->hard error.
    parser.add_argument(
        "--sharding-strategy",
        dest="sharding_strategy",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
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
        "--meta-init",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help=(
            "Build the native Transformer on the `meta` device, then "
            "materialize only each rank's shard after FSDP2 sharding "
            "(torchtitan pattern). Avoids the OOM from moving the full dense "
            "model onto one device before sharding, which otherwise caps model "
            "size at what fits whole on a single GPU (~2-8B) regardless of "
            "node count. `auto` (default) enables it for large native models "
            "(>= ~6B params) and keeps small models on the exact dense init "
            "path (bit-for-bit reproducible); `on` forces it for any native "
            "model; `off` forces the legacy dense path. Ignored for HF "
            "`from_pretrained` models (they load real pretrained weights). "
            "Override the auto threshold with EZPZ_META_INIT_MIN_PARAMS."
        ),
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=0,
        help=(
            "Enable LoRA fine-tuning with this rank (0 = off, full "
            "fine-tuning). Freezes the base weights and trains a low-rank "
            "update per targeted projection, so only ~rank/dim of the "
            "parameters are trainable and checkpoints shrink by roughly the "
            "same factor. Composes with FSDP2, TP and meta-init. Native "
            "models only -- the HF path is unaffected."
        ),
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help=(
            "LoRA scaling numerator; the update is scaled by alpha/rank. "
            "Defaults to --lora-rank (scale 1.0)."
        ),
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="Dropout on the LoRA adapter branch input (0 = off).",
    )
    parser.add_argument(
        "--lora-target",
        type=str,
        default="attn,mlp",
        help=(
            "Which projections to adapt, comma-separated: `attn` "
            "(wq/wk/wv/wo), `mlp` (w1/w2/w3), `unembed` (the output "
            "projection -- the largest single matrix, off by default). "
            "Ignored unless --lora-rank > 0."
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
            "Base directory for metrics logs + the History report. None "
            "(default) writes under the current working directory. (Model "
            "checkpoints go to --ckpt-dir, not here.)"
        ),
    )
    parser.add_argument(
        "--ckpt-dir",
        "--ckpt_dir",
        type=str,
        default=None,
        dest="ckpt_dir",
        help=(
            "Directory for DCP (sharded) checkpoints. When set, enables "
            "checkpoint save (see --save-interval) AND auto-resume: on "
            "startup the latest complete checkpoint here is loaded and "
            "training continues from it (unless --no-resume). This is what "
            "makes `ezpz launch --auto-retry` resume across attempts."
        ),
    )
    parser.add_argument(
        "--save-interval",
        "--save_interval",
        type=int,
        default=0,
        dest="save_interval",
        help=(
            "Save a checkpoint every N optimizer steps (requires "
            "--ckpt-dir). 0 (default) disables saving."
        ),
    )
    parser.add_argument(
        "--train-iters",
        "--train_iters",
        type=int,
        default=0,
        dest="train_iters",
        help=(
            "Stop after N optimizer steps, regardless of --epochs. 0 "
            "(default) runs the full --epochs pass. Step-based cap for "
            "fixed-length runs / restart-time experiments."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Ignore any existing checkpoint in --ckpt-dir and start fresh "
            "(step 0). Default behavior auto-resumes from the latest."
        ),
    )
    parser.add_argument(
        "--async-ckpt",
        "--async_ckpt",
        action="store_true",
        dest="async_ckpt",
        help=(
            "Save checkpoints asynchronously: stage to fast node-local "
            "--ckpt-stage-dir (background thread, overlaps training), then "
            "fan out to the durable --ckpt-dir on shared FS. Requires "
            "--ckpt-dir. Resume is unchanged (always from --ckpt-dir)."
        ),
    )
    parser.add_argument(
        "--ckpt-stage-dir",
        "--ckpt_stage_dir",
        type=str,
        default=None,
        dest="ckpt_stage_dir",
        help=(
            "Node-local staging dir for --async-ckpt (default "
            "/tmp/ezpz-ckpt-<jobid>). Transient — NOT resumable on its own; "
            "only the fanned-out --ckpt-dir copy is durable."
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
        choices=[
            "eager",
            "chunked",
            "chunked-backward",
            "compiled",
            "loss-parallel",
            "fused-linear",
        ],
        help=(
            "Cross-entropy implementation. The large-vocab output path is the "
            "memory bottleneck: a full (B*T, vocab) fp32 logits tensor + its "
            "grad (agpt-2b 256K vocab, seq=8192, bs=2: ~16.8 GiB EACH) can OOM "
            "a GPU tile (UR_RESULT_ERROR_OUT_OF_RESOURCES) even when the model "
            "fits. Pick by what you need (numbers = measured agpt-2b tp=1):\n"
            "  • `eager` (default): plain F.cross_entropy on full logits. "
            "Simplest; OOMs at agpt-2b bs2/seq8192. Use for small vocab/seq.\n"
            "  • `chunked`: chunks only the FORWARD (--loss-chunk-size). Does "
            "NOT bound backward; still OOMs at large vocab. Rarely useful.\n"
            "  • `chunked-backward`: custom autograd Function that also bounds "
            "the backward graph (recomputes each chunk's grad), saving ~one "
            "full logits buffer vs eager. General + model-agnostic (works for "
            "HF models, no torch.compile needed) — good at MODERATE vocab/seq "
            "or when compile is unavailable. Still holds two logit-sized "
            "buffers, so it does NOT fix the very-large-vocab OOM (use "
            "fused-linear/compiled there).\n"
            "  • `compiled`: torch.compile fuses log_softmax+NLL+backward so "
            "the full transient is never materialized (torchtitan's approach). "
            "Fits (~45 GB) and is the FASTEST that fits (~28%% MFU). Needs "
            "working torch.compile. Best default when it fits.\n"
            "  • `fused-linear` (Liger/Cut-CE): runs the output projection "
            "per row-chunk so the full (B*T,vocab) logits/grad are NEVER built "
            "— bounds BOTH row and vocab dims. LOWEST memory (~32 GB, below "
            "compiled) at ~24%% MFU; trades a little speed for headroom (bigger "
            "batch/seq). ezpz Transformer + tp=1 only (HF / tp>1 fall back to "
            "compiled).\n"
            "  • `loss-parallel`: vocab-parallel CE sharding the vocab across "
            "TP ranks (each holds vocab/tp) via TP all-reduces. Bounds the "
            "VOCAB dim; only helps at tp>1 (at tp=1 falls back to eager). At "
            "tp>1 it is also the only correct path (plain CE hits a "
            "Tensor/DTensor mismatch on Replicate logits). ~23 GB/rank, "
            "~34%% MFU at tp=2.\n"
            "NOTE: `--compile` only compiles the transformer blocks, NOT the "
            "loss, so it does NOT by itself fix the loss transient — use "
            "--loss-impl for that."
        ),
    )
    parser.add_argument(
        "--loss-chunk-size",
        type=int,
        default=1024,
        help=(
            "Row-chunk size (number of (B*T) token rows per cross-entropy "
            "chunk) for --loss-impl=chunked, chunked-backward, and "
            "fused-linear. Smaller = lower peak memory, more kernel launches. "
            "Ignored for eager/compiled/loss-parallel."
        ),
    )
    # max_batch_size: int = 32
    # max_seq_len: int = 32768
    # Shared profiler flags (--profile / --pyinstrument-profiler / etc.),
    # consumed by profiling_context_from_args around the training loop.
    add_profiling_args(parser)
    args = parser.parse_args(argv)
    apply_model_preset(args, argv)
    # Fold the deprecated --sharding-strategy alias into reshard_after_forward
    # (and hard-error the removed hybrid_shard* values).
    _resolve_reshard_after_forward(args)
    _maybe_enable_cpu_backend_for_async_ckpt(args)
    return args


def _maybe_enable_cpu_backend_for_async_ckpt(args: argparse.Namespace) -> None:
    """Register a CPU (gloo) backend when --async-ckpt is set.

    ``dcp.async_save`` asserts the process group includes a CPU backend (it
    stages/writes from a background CPU thread). On an accelerator the default
    PG is xccl/nccl-only, so we must init with a COMPOSITE backend string,
    accelerator-first: ``xpu:xccl,cpu:gloo`` (matching torchtitan). This runs
    in parse_args — BEFORE
    ``setup_torch``/``init_process_group`` — and sets ``TORCH_BACKEND``, which
    ``ezpz.distributed.get_torch_backend`` honors as an override.

    Respects a user-set ``TORCH_BACKEND`` (only fills it when unset) and is a
    no-op on CPU-only runs (backend is already gloo).
    """
    if not getattr(args, "async_ckpt", False):
        return
    # parse_args runs on EVERY rank before distributed init, so gate the
    # informational log to one process per host (env local rank; 0 when unset).
    _lr = (
        os.environ.get("PALS_LOCAL_RANKID")
        or os.environ.get("PMIX_LOCAL_RANK")
        or os.environ.get("LOCAL_RANK")
        or "0"
    )
    is_local0 = _lr == "0"
    override = os.environ.get("TORCH_BACKEND")
    if override:
        # A user override wins, but --async-ckpt needs a CPU backend in the PG
        # or dcp.async_save asserts at the first save. If the override has no
        # cpu:/gloo entry, augment it (append cpu:gloo) rather than fail late.
        if "gloo" in override or "cpu:" in override:
            return  # already has a CPU backend
        os.environ["TORCH_BACKEND"] = f"{override},cpu:gloo"
        if is_local0:
            logger.warning(
                "async-ckpt: TORCH_BACKEND=%s lacks a CPU backend; augmented "
                "to %s (dcp.async_save requires one)",
                override,
                os.environ["TORCH_BACKEND"],
            )
        return
    try:
        import torch

        # hasattr guard: torch.xpu is absent on CPU-only / CUDA-only / ROCm
        # builds, where torch.xpu.is_available() would raise AttributeError.
        if torch.cuda.is_available():
            accel = "cuda:nccl"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            accel = "xpu:xccl"
        else:
            return  # CPU-only: default gloo already has a CPU backend
        # Order matches torchtitan (distributed/utils.py): accelerator FIRST,
        # then cpu:gloo. The accel backend stays primary; cpu:gloo is added so
        # dcp.async_save's background CPU-thread collective has a backend.
        os.environ["TORCH_BACKEND"] = f"{accel},cpu:gloo"
        if is_local0:
            logger.info(
                "async-ckpt: set TORCH_BACKEND=%s so async_save has a CPU "
                "backend",
                os.environ["TORCH_BACKEND"],
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not set composite backend for --async-ckpt: %s", exc)


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


def _estimate_param_count(config: "ModelArgs") -> int:
    """Closed-form lower bound on parameter count from a ModelArgs (no alloc).

    Counts the two vocab-sized matrices (embedding + output = ``2*vocab*dim``)
    plus a per-layer ``~12*dim^2`` for attention (qkvo) + SwiGLU FFN (w1/w2/w3
    at ~4*dim each ⇒ ~12*dim^2 combined for the common hidden≈~4*dim). This is
    an approximation used only to pick the meta-init tier, so an exact FFN width
    isn't needed — it separates agpt-2b (~2B, below) from xxl/xxxl/agpt-20b
    (>=~7B, above) with wide margin.
    """
    dim = int(config.dim)
    n_layers = int(config.n_layers)
    vocab = int(config.vocab_size)
    return 2 * vocab * dim + n_layers * 12 * dim * dim


def _resolve_meta_init(
    args: argparse.Namespace,
    config: "ModelArgs",
    is_hf_model: bool,
) -> bool:
    """Decide whether to use meta-device init for this run.

    Meta-init (build on meta → shard → to_empty → init_weights) avoids OOM-ing
    when the full dense model would not fit on one device before sharding. It is
    NATIVE-Transformer only — HF `from_pretrained` models load real pretrained
    weights and must materialize them, so meta-init never applies there.

    Modes (``--meta-init``): ``off`` → dense (legacy); ``on`` → meta (native);
    ``auto`` (default) → meta iff native and the estimated param count is at or
    above EZPZ_META_INIT_MIN_PARAMS (default 6e9). ``auto`` keeps small models
    (debug/s/m/l/xl/agpt-2b) on the exact dense path (bit-for-bit init) and
    auto-enables meta-init for large ones (xxl/xxxl/agpt-20b).
    """
    mode = getattr(args, "meta_init", "auto")
    if is_hf_model:
        if mode == "on" and ezpz.get_rank() == 0:
            logger.warning(
                "--meta-init on ignored for HF model %s (from_pretrained "
                "loads real weights; meta-init is native-Transformer only).",
                args.model,
            )
        return False
    if mode == "off":
        return False
    if mode == "on":
        return True
    # auto: enable for large native models.
    threshold = float(os.environ.get("EZPZ_META_INIT_MIN_PARAMS", "6e9"))
    est = _estimate_param_count(config)
    use = est >= threshold
    if ezpz.get_rank() == 0:
        logger.info(
            "meta-init=auto: estimated ~%.1fB params (threshold %.1fB) -> %s",
            est / 1e9,
            threshold / 1e9,
            "meta-init ON" if use else "dense init",
        )
    return use


def _lora_is_applied(model: nn.Module) -> bool:
    """True when any submodule is a LoRA wrapper.

    Used to decide whether the TP plans need retargeting. Checked on the
    live module tree rather than on `args.lora_rank` so `parallelize()`
    stays correct for callers that apply LoRA themselves.
    """
    return any(True for _ in _lora.iter_lora_modules(model))


def parallelize(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mixed_precision: Optional[MixedPrecisionPolicy],
    reshard_after_forward: str = "always",
    activation_checkpoint: str = "none",
    loss_parallel: bool = False,
    meta_init: bool = False,
    device: Optional["torch.device"] = None,
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

    ``meta_init``: when True the model was built on the ``meta`` device (no
    storage). Pre-shard ``init_weights`` is skipped; after ``fully_shard`` the
    sharded params are materialized on ``device`` via ``to_empty`` and then
    ``init_weights(buffer_device=device)`` fills them — so the full dense model
    is never placed on one device (avoids the large-model build OOM). Requires
    ``device``.
    """
    tp_mesh = device_mesh["tp"]

    # Choose the mesh fully_shard shards over:
    #   dp_replicate > 1 -> 2D (dp_replicate, dp_shard) submesh; FSDP2 reads a
    #                       2D DP mesh as HSDP (replicate across the outer dim,
    #                       shard within the inner) automatically.
    #   dp_replicate == 1 -> 1D dp_shard mesh; plain FSDP sharding, identical
    #                        to the pre-HSDP behavior (avoids a needless
    #                        size-1 replicate wrap).
    if device_mesh["dp_replicate"].size() > 1:
        fsdp_dp_mesh = device_mesh["dp_replicate", "dp_shard"]
    else:
        fsdp_dp_mesh = device_mesh["dp_shard"]

    reshard = _reshard_arg(reshard_after_forward)

    # Dense path: init the real params now. Meta path: skip — the params are on
    # `meta` (no storage), so init happens after fully_shard via to_empty +
    # init_weights(buffer_device=device) below.
    if not meta_init:
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
        # `parallelize_module` dispatches on the module CLASS, so any plan
        # key naming a LoRA-wrapped Linear raises `NotImplementedError:
        # ColwiseParallel currently only support nn.Linear and
        # nn.Embedding!`. _lora.lora_tp_plan rewrites those keys to the
        # inner `.base` / `.A` / `.B` Linears; it is a no-op when nothing
        # is wrapped, so the non-LoRA path is untouched.
        #
        # Pass the module: the plan must retarget only keys that really
        # are wrapped. `--lora-target` is selective (the default leaves
        # `output` alone), and parallelize_module DROPS keys matching no
        # module without raising -- so retargeting an unwrapped `output`
        # would leave it an unsharded nn.Linear that dies on its first
        # DTensor input.
        root_tp_plan = {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            # With loss_parallel, keep logits vocab-sharded (Shard(-1))
            # and return the LOCAL [N, vocab/tp] tensor so the loss can run
            # vocab-parallel CE (no full-vocab all-gather). Otherwise gather
            # to Replicate() so the loss sees full-vocab logits (default).
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=bool(loss_parallel),
            ),
        }
        if _lora_is_applied(model):
            root_tp_plan = _lora.lora_tp_plan(root_tp_plan, model)
        model = parallelize_module(model, tp_mesh, root_tp_plan)

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
            # Bind a per-block plan rather than reassigning `layer_tp_plan`:
            # that would feed layer 2 the plan already retargeted for layer 1.
            block_tp_plan = layer_tp_plan
            if _lora_is_applied(model):
                block_tp_plan = _lora.lora_tp_plan(
                    layer_tp_plan, transformer_block
                )
            parallelize_module(
                module=transformer_block,  # type: ignore
                device_mesh=tp_mesh,
                parallelize_plan=block_tp_plan,
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
    fsdp_kwargs = {"mesh": fsdp_dp_mesh, "reshard_after_forward": reshard}
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

    # Meta path: params are now sharded DTensors still on `meta`. Materialize
    # ONLY this rank's shard on the real device (to_empty — no full-model copy),
    # then init_weights fills the sharded params and recomputes the freqs_cis
    # buffer on `device` (to_empty leaves buffer data uninitialized). On resume,
    # dcp.load later overwrites the params + persistent buffers.
    if meta_init:
        assert device is not None, "meta_init=True requires a device"
        model.to_empty(device=device)
        model.init_weights(buffer_device=device)  # type: ignore

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


def _build_hf_dataloader(
    dataset,
    *,
    batch_size: int,
    dpsize: int,
    dp_rank: int,
    world_size: int,
):
    """Build the (sampler, DataLoader) for the HF-text training path.

    Both the ``DistributedSampler`` and the ``DataLoader`` use
    ``drop_last=True`` so every batch has a static batch dimension. A
    ragged final batch (e.g. batch_size 2 -> 1) makes torch.compile mark
    the batch dim dynamic and recompile at the epoch boundary; under
    symbolic shapes inductor can no longer fuse log_softmax+NLL+backward,
    so the ``compiled`` CE materializes the full (B*T, vocab) logits grad
    (agpt-2b 256K vocab, seq 8192: ~15.6 GiB) and OOMs at the first step
    of epoch 1. Static shapes avoid the recompile. Matches the ``random``
    branch and torchtitan; the dropped tail is at most (batch_size - 1)
    samples per dp rank per epoch (negligible for LM pretraining).

    Extracted from ``train`` so the drop_last invariant + the too-small
    guard below are unit-testable without launching a run.

    Raises:
        ValueError: if ``drop_last=True`` would leave zero batches for this
            rank (dataset too small for ``batch_size`` * dp degree). Without
            this, training would run zero optimizer steps yet exit 0 — a
            smoke test would falsely "pass". Callers should reduce
            ``--batch-size`` / ``--dp-*`` or raise ``--hf-limit``.
    """
    sampler = (
        DistributedSampler(
            dataset=dataset,
            num_replicas=dpsize,
            rank=dp_rank,
            drop_last=True,
        )
        if world_size > 1
        else None
    )
    # Per-rank sample count after DistributedSampler shards + drops the tail.
    per_rank = (len(dataset) // dpsize) if sampler is not None else len(dataset)
    if per_rank // batch_size < 1:
        raise ValueError(
            "HF dataloader with drop_last=True yields 0 full batches for this "
            f"rank: {per_rank} samples/rank < batch_size={batch_size} "
            f"(dataset={len(dataset)}, dp={dpsize}). Training would run zero "
            "steps. Lower --batch-size / --dp-shard / --dp-replicate or raise "
            "--hf-limit."
        )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=(sampler is None),
        drop_last=True,
    )
    return sampler, dataloader


def _local_dtype(t: "torch.Tensor | None") -> "str":
    """dtype of a (possibly DTensor) tensor's LOCAL shard, as a short string."""
    if t is None:
        return "n/a"
    inner = getattr(t, "to_local", None)
    local = inner() if callable(inner) else t
    return str(local.dtype).replace("torch.", "")


def _push_precision_to_wandb(summary: "dict[str, str]") -> None:
    """Push the precision summary to the W&B run config under `precision/*`
    (rank 0, best-effort). Config (not per-step metric) since dtypes are static
    for the run; `allow_val_change` so the post-step-1 optimizer-state update
    can amend it."""
    if ezpz.get_rank() != 0 or wandb is None or getattr(wandb, "run", None) is None:
        return
    try:
        wandb.config.update(  # type: ignore[union-attr]
            {f"precision/{k}": v for k, v in summary.items()},
            allow_val_change=True,
        )
    except Exception:  # noqa: BLE001 — logging must never break training
        # Debug-level (not silent): a W&B sync failure shouldn't kill training,
        # but swallowing it entirely hides why precision/* never showed up.
        logger.debug("could not push precision summary to W&B", exc_info=True)


def _log_precision_summary(
    model: nn.Module,
    mp_config: "Optional[MixedPrecisionPolicy]",
    reduce_dtype: "torch.dtype",
    args: argparse.Namespace,
) -> "dict[str, str]":
    """Build + log the ACTUAL precision of each component by introspecting the
    live objects (not the intended config), and push it to the W&B run config.
    Returns the summary dict so the caller can (a) fold it into the metrics
    JSONL and (b) amend it with the optimizer-state dtype after the first step.
    Rank-0 logs; the returned dict is identical on every rank.
    """
    # A real parameter (master/sharded copy FSDP owns + the optimizer updates).
    sample = next(iter(model.parameters()), None)
    master = _local_dtype(sample)
    if mp_config is None:
        compute = master  # --fp32: no MP policy, compute == stored dtype
        reduce_s = master
        mode = "off (--fp32)"
    else:
        compute = str(mp_config.param_dtype).replace("torch.", "")
        reduce_s = str(reduce_dtype).replace("torch.", "")
        mode = "on"
    summary = {
        "mixed_precision": mode,
        "master_weights": master,       # stored param / optimizer target
        "compute_weights": compute,     # fwd/bwd param copy (param_dtype)
        "activations": compute,
        "grad_reduce": reduce_s,        # reduce_dtype
        "loss_accum": "float32",        # CE always accumulates in fp32
        "optimizer_states": "pending",  # amended post-step-1
        "reduce_dtype_env": os.environ.get("EZPZ_REDUCE_DTYPE", "fp32"),
        "fp32_flag": str(bool(args.fp32)),
    }
    if ezpz.get_rank() == 0:
        logger.info(
            "precision summary [mixed-precision %s]:\n"
            "  - master weights (stored param / optimizer target) : %s\n"
            "  - compute weights (fwd/bwd param copy)              : %s\n"
            "  - activations / matmuls                             : %s\n"
            "  - gradient reduce-scatter (reduce_dtype)            : %s\n"
            "  - loss / cross-entropy accumulation                : %s (always)\n"
            "  - optimizer states (AdamW)                          : logged at first step\n"
            "  (EZPZ_REDUCE_DTYPE=%s, --fp32=%s)",
            mode, master, compute, compute, reduce_s, summary["loss_accum"],
            summary["reduce_dtype_env"], summary["fp32_flag"],
        )
    _push_precision_to_wandb(summary)
    return summary


def _log_optimizer_state_dtype(
    optimizer: "torch.optim.Optimizer",
    summary: "Optional[dict[str, str]]" = None,
) -> None:
    """Log the dtype of AdamW's exp_avg / exp_avg_sq once, after step 1, and
    amend `summary` + W&B config in place. Called post-step-1 because on a FRESH
    run the state dict is empty until the first step allocates it (a RESUMED run
    already has it from load_checkpoint, so post-step-1 works either way).
    Rank 0 logs; reads the first populated state slot."""
    dtype = "n/a"
    for state in optimizer.state.values():
        ea = state.get("exp_avg")
        eas = state.get("exp_avg_sq")
        if ea is not None or eas is not None:
            dtype = _local_dtype(ea if ea is not None else eas)
            if ezpz.get_rank() == 0:
                logger.info(
                    "precision (optimizer states, post-step-1): "
                    "exp_avg=%s exp_avg_sq=%s",
                    _local_dtype(ea), _local_dtype(eas),
                )
            break
    else:
        if ezpz.get_rank() == 0:
            logger.info("precision (optimizer states): no populated state found")
    if summary is not None:
        summary["optimizer_states"] = dtype
        _push_precision_to_wandb(summary)


@ezpz.timeitlogit(rank=ezpz.get_rank())
def train(
    args: argparse.Namespace,
    outdir: Path | str | os.PathLike,
    profiler: Optional[Any] = None,
    process_start: Optional[float] = None,
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
    # Timestamp for the restart-time metric. Prefer the caller's
    # process-start (main() captures it BEFORE setup_torch, so
    # train/restart_seconds includes distributed init — the dominant cost of a
    # real cold failover). Fall back to now() when train() is called directly.
    _train_t0 = process_start if process_start is not None else perf_counter()
    _restart_logged = False
    _optim_dtype_logged = False  # log AdamW state dtype once, after step 1
    _precision_in_jsonl = False  # fold precision summary into JSONL once
    world_size = ezpz.distributed.get_world_size()
    assert world_size % args.tp == 0, "WORLD_SIZE must be divisible by TP"
    # Resolve the data-parallel topology: dp_replicate (HSDP outer) x
    # dp_shard (FSDP inner). Defaults (replicate=1, shard=-1) reproduce the
    # flat FSDP dp dim exactly.
    dp_replicate, dp_shard = _resolve_dp_degrees(
        world_size=world_size,
        tp=args.tp,
        dp_replicate=args.dp_replicate,
        dp_shard=args.dp_shard,
    )
    dpsize = dp_replicate * dp_shard  # == flattened "dp" size
    # Global batch size = per-DP-rank micro-batch x data-parallel degree.
    # Data parallelism is dp_replicate * dp_shard (== dpsize): BOTH the HSDP
    # replicate dim and the FSDP shard dim get a distinct data slice (the
    # DistributedSampler shards over num_replicas=dpsize). TP is NOT data
    # parallel — tp ranks process the SAME batch (sharded across the hidden
    # dim), so tp does not enter here. Backfilled onto args so it lands in the
    # wandb run config (History logs vars(args); we also push it to
    # wandb.config below). Computed here, not in parse_args, because the
    # default --dp-shard -1 ("use all remaining ranks") only resolves to a
    # concrete degree once WORLD_SIZE is known.
    #
    # This is the general Megatron/NeMo formula
    #   gbs = micro_batch * world_size * grad_accum
    #         / (tensor_parallel * pipeline_parallel)
    # specialized to this example: there is no pipeline parallelism
    # (pipeline_parallel = 1) and no gradient accumulation (grad_accum = 1 —
    # zero_grad/backward/step run every iteration), and world_size / tp ==
    # dp_replicate * dp_shard == dpsize. If either is ever added, this must
    # gain the corresponding factor (* grad_accum, / pipeline_parallel).
    args.global_batch_size = args.batch_size * dpsize
    if ezpz.get_rank() == 0:
        logger.info(
            "global_batch_size=%d (batch_size=%d x dp_replicate=%d x "
            "dp_shard=%d; tp=%d does not scale the batch)",
            args.global_batch_size,
            args.batch_size,
            dp_replicate,
            dp_shard,
            args.tp,
        )
    # fused-linear (Liger/Cut-CE): needs the model's hidden states + output
    # weight, so only the ezpz Transformer (not HF models) and — for now —
    # only tp=1 (tp>1 needs vocab-shard composition, gated to a follow-up).
    # Resolved against the actual model after it's built (see below).
    want_fused_linear = getattr(args, "loss_impl", "eager") == "fused-linear"
    # loss-parallel (vocab-sharded CE) only does anything at tp>1; at tp=1 the
    # local vocab IS the full vocab. NOTE: `use_loss_parallel` is resolved
    # LATER (after the HF branch may force args.tp=1), so an HF model launched
    # with --tp>1 --loss-impl=loss-parallel doesn't enter vocab-parallel CE
    # with a stale TP group. See the resolution just before parallelize().
    # 3D mesh: (dp_replicate, dp_shard, tp). We then flatten the two DP dims
    # into a single named "dp" dim so all existing dp consumers
    # (DistributedSampler num_replicas/rank, loss dp_group) keep working with
    # correct rank arithmetic. fully_shard picks the DP submesh explicitly
    # (2D -> HSDP when dp_replicate > 1, else 1D dp_shard == today's FSDP).
    device_mesh = ezpz.init_device_mesh_safe(
        str(ezpz.get_torch_device()),
        (dp_replicate, dp_shard, args.tp),
        mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
    )
    # Flatten the two DP dims into a single "dp" dim. Must go through
    # flatten_device_mesh_safe: `_flatten` builds the flattened PG via
    # split_group when the default PG is device-bound, which xccl (XPU)
    # doesn't support — the same workaround init_device_mesh_safe applies.
    ezpz.distributed.flatten_device_mesh_safe(
        device_mesh[("dp_replicate", "dp_shard")], "dp"
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
    # NOTE: `config` (ModelArgs) is not mutated later, so it's safe to push
    # here. The resolved CLI `args`, however, ARE still mutated below (the HF
    # branch forces args.tp=1; args.loss_impl can be normalized to
    # "compiled"), so vars(args) is pushed just before the training loop
    # instead (see below) — logging it here would record the requested
    # settings rather than the effective ones actually used.
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
    # Decide meta-device init (native large models only) before building, so the
    # native build below can go on `meta` and never place the full dense model
    # on one device. HF models always resolve to False (they load real weights).
    meta_init = _resolve_meta_init(args, config, is_hf_model)
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
        if meta_init:
            # Build on `meta`: no storage allocated, so even a 20B model costs
            # nothing here. parallelize() shards it, then to_empty materializes
            # only this rank's shard on the real device.
            with torch.device("meta"):
                model = Transformer.from_model_args(config)
        else:
            model = Transformer.from_model_args(config)
        # LoRA must be applied BEFORE parallelize(): the TP plan and
        # fully_shard both need to see the final module tree. Under
        # meta-init the adapters are built on `meta` too and materialized
        # by the same to_empty()/init_weights() pass.
        if getattr(args, "lora_rank", 0) > 0:
            _targets = {
                t.strip() for t in str(args.lora_target).split(",") if t.strip()
            }
            _unknown = _targets - {"attn", "mlp", "unembed"}
            if _unknown:
                raise SystemExit(
                    f"--lora-target: unknown {sorted(_unknown)}; "
                    "expected any of attn, mlp, unembed"
                )
            model = _lora.apply_lora(
                model,
                _lora.LoraConfig(
                    rank=args.lora_rank,
                    alpha=args.lora_alpha,
                    dropout=args.lora_dropout,
                    train_attn="attn" in _targets,
                    train_mlp="mlp" in _targets,
                    train_unembed="unembed" in _targets,
                    seed=args.seed,
                ),
            )
    mstr = summarize_model(
        model,
        verbose=False,
        depth=2,
    )
    logger.info(f"\n{mstr}")
    # Meta models are materialized (sharded) inside parallelize() via to_empty;
    # moving a meta model with .to(device) would NOT allocate real storage, so
    # skip it here. Dense + HF paths place the real model on-device now.
    if not meta_init:
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
    elif meta_init:
        # The real-tensor probe below would run an actual forward, which a
        # meta model (no storage) cannot do. Skip it — MFU/TFLOPS just stay 0
        # for meta-init runs when the fake-tensor count is unavailable.
        if ezpz.get_rank() == 0:
            logger.warning(
                "Fake-tensor FLOP estimate returned 0 and model is on `meta` "
                "(--meta-init); skipping the real-tensor probe. train/tflops "
                "and train/mfu will be 0 for this run."
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
    _reduce_dtype = torch.float32  # default (also the effective dtype under --fp32)
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
    # Resolve loss-parallel NOW that args.tp is final (the HF branch above may
    # have forced tp=1). loss-parallel only does anything at tp>1; at tp=1 the
    # local vocab IS the full vocab, so normalize to compiled CE rather than
    # leaving loss_impl='loss-parallel' to hit an unhandled impl at the call
    # site. This also covers the HF + --tp>1 --loss-impl=loss-parallel case:
    # tp is now 1, so we fall back instead of entering vocab-parallel CE with a
    # stale TP group.
    use_loss_parallel = (
        getattr(args, "loss_impl", "eager") == "loss-parallel" and args.tp > 1
    )
    if getattr(args, "loss_impl", "eager") == "loss-parallel" and not use_loss_parallel:
        logger.warning(
            "--loss-impl=loss-parallel requires tp>1 (got tp=%d); falling back "
            "to compiled CE.",
            args.tp,
        )
        args.loss_impl = "compiled"
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

        # Same HSDP mesh selection as parallelize(): 2D DP submesh when
        # replicating, else the 1D shard mesh (identical to pre-HSDP).
        if device_mesh["dp_replicate"].size() > 1:
            hf_dp_mesh = device_mesh["dp_replicate", "dp_shard"]
        else:
            hf_dp_mesh = device_mesh["dp_shard"]
        hf_fsdp_kwargs = {
            "mesh": hf_dp_mesh,
            "reshard_after_forward": _reshard_arg(args.reshard_after_forward),
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
        # loss-parallel needs vocab-sharded (local) logits out of the output
        # projection; only meaningful at tp>1 (at tp=1 there's nothing to
        # shard, so it falls back to eager — see the loss call site).
        model = parallelize(
            model,
            device_mesh,
            mp_config,
            reshard_after_forward=args.reshard_after_forward,
            activation_checkpoint=args.activation_checkpoint,
            loss_parallel=use_loss_parallel,
            meta_init=meta_init,
            device=device,
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
    # Resolve fused-linear eligibility now that the model exists. Requires the
    # ezpz Transformer (has `.output` weight + return_hidden) and tp=1 for now.
    use_fused_linear = False
    if want_fused_linear:
        has_output = hasattr(base_model, "output") and hasattr(
            base_model.output, "weight"
        )
        if is_hf_model or not has_output:
            logger.warning(
                "--loss-impl=fused-linear needs the ezpz Transformer "
                "(hidden states + output weight); falling back to compiled "
                "for this model."
            )
            # Normalize so the `_compute_loss` call site actually runs compiled
            # CE — leaving loss_impl='fused-linear' would fall through to an
            # unhandled impl (now an error; previously silent eager + OOM).
            args.loss_impl = "compiled"
        elif args.tp > 1:
            logger.warning(
                "--loss-impl=fused-linear with tp>1 is not yet supported "
                "(needs vocab-shard composition); falling back to compiled."
            )
            args.loss_impl = "compiled"
        else:
            use_fused_linear = True
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

    # Log the ACTUAL precision of each component (introspects the live model +
    # MP policy, not the intended config), push it to the W&B run config, and
    # keep the dict to (a) fold into the first metrics row and (b) amend with
    # the optimizer-state dtype after the first step (on a FRESH run the state
    # dict is empty until then; a RESUMED run has it populated by
    # load_checkpoint, so reading it post-step-1 is correct either way).
    _precision_summary = _log_precision_summary(model, mp_config, _reduce_dtype, args)

    # --- Resume from checkpoint (auto-detect latest) --------------------------
    # Both model AND optimizer exist and are fully sharded here, so this is the
    # correct point to restore sharded DCP state into them. Auto-resume (no
    # flag) is what lets `ezpz launch --auto-retry` — which relaunches the
    # IDENTICAL command each attempt — pick up where a failed attempt left off.
    # Async checkpointing needs a durable target to fan out to; a node-local
    # stage dir alone is not resumable after a failure (see _checkpoint.py).
    if args.async_ckpt and not args.ckpt_dir:
        raise ValueError("--async-ckpt requires --ckpt-dir (the durable target)")
    ckpt_stage_dir = args.ckpt_stage_dir
    if args.async_ckpt and not ckpt_stage_dir:
        jobid = (
            os.environ.get("PBS_JOBID")
            or os.environ.get("SLURM_JOB_ID")
            or str(os.getpid())
        ).split(".")[0]
        ckpt_stage_dir = f"/tmp/ezpz-ckpt-{jobid}"
    # In-flight async checkpoint handle (drained before the next save + at exit).
    pending_ckpt = None

    resume_meta: "dict[str, object] | None" = None
    if args.ckpt_dir and not args.no_resume:
        from ezpz.examples._checkpoint import load_checkpoint

        resume_meta = load_checkpoint(args.ckpt_dir, model, optimizer)
        if resume_meta is not None and ezpz.get_rank() == 0:
            logger.info(
                "RESUMED from step=%s (epoch=%s, batch_offset=%s, "
                "tokens_seen=%s)",
                resume_meta.get("step"),
                resume_meta.get("epoch"),
                resume_meta.get("batch_offset"),
                resume_meta.get("tokens_seen"),
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
        # drop_last=True (both sampler + loader) so every batch has a static
        # batch dim — a ragged tail triggers a torch.compile recompile at the
        # epoch boundary that OOMs the compiled CE. See _build_hf_dataloader.
        sampler, dataloader = _build_hf_dataloader(
            dataset,
            batch_size=args.batch_size,
            dpsize=dpsize,
            dp_rank=device_mesh.get_local_rank("dp"),
            world_size=ezpz.get_world_size(),
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

    # Re-push the precision summary now that History has created the W&B run.
    # The initial push (at optimizer-build time, above) runs BEFORE this and so
    # no-ops on `wandb.run is None`; without this second push a run that dies
    # before its first optimizer step would leave W&B with no precision
    # diagnostics at all. Cheap + idempotent (config update, allow_val_change).
    _push_precision_to_wandb(_precision_summary)

    # For TP, input needs to be the same across all TP ranks.
    # while for SP, input can be different across all ranks
    # We will use dp_rank for setting the random seed
    # to mimic the behavior of the dataloader
    # x = torch.tensor((args.batch_size, args.seq_len))
    x = torch.tensor(0)
    global_step = 0
    # Bundle the step's dependencies once. This is a VIEW over the objects
    # built above -- it constructs nothing and owns nothing -- so the loop
    # below and ezpz.tinker's client run the exact same code path. The
    # loss impls are injected rather than imported by the step module,
    # which would otherwise import this module back (circular).
    train_state = _tinker_step.TrainState(
        model=model,
        optimizer=optimizer,
        device=device,
        args=args,
        base_model=base_model,
        dataset=dataset,
        tp_group=tp_group,
        use_fused_linear=use_fused_linear,
        use_loss_parallel=use_loss_parallel,
        profiler=profiler,
        global_step=0,
        compute_loss=_compute_loss,
        localize_logits=_localize_logits_for_loss,
        fused_linear_loss=_cross_entropy_fused_linear,
        vocab_parallel_loss=_cross_entropy_vocab_parallel,
        slice_for_sequence_parallel=_slice_for_sequence_parallel,
    )
    # Cumulative count of training tokens consumed across the whole run
    # (summed global tokens/step). Logged as train/tokens_seen — the standard
    # x-axis for loss-vs-tokens curves. See the metrics block: global
    # tokens/step = batch * full_seq_len * dpsize (full pre-shard seq length,
    # not the SP-local shard, so it's exact and rank-invariant).
    tokens_seen = 0
    # Checkpoint timings from the PREVIOUS step, folded into the next step's
    # metrics dict so they reach JSONL + W&B:
    #   - train/ckpt_save_seconds  : sync save's blocking write (all 23 GB to
    #     the durable dir on the training thread).
    #   - train/ckpt_stage_seconds : async save's CPU-stage stall only (the
    #     cheap part — copy state to host, kick off the background write).
    #   - train/ckpt_drain_seconds : async fan-out (/tmp -> shared FS) blocking
    #     time at the START of the next step. This is the EXPENSIVE half of an
    #     async save and was previously untimed — it lands between steps, so it
    #     is captured by neither ckpt_stage_seconds nor train/dt. The honest
    #     per-save stall for async is stage + drain, NOT stage alone.
    pending_stage_seconds: "float | None" = None
    pending_save_seconds: "float | None" = None
    pending_drain_seconds: "float | None" = None
    # Resume bookkeeping. When resuming, seed the counters from the checkpoint
    # and reconstruct (start_epoch, resume_offset) so the loop skips
    # already-consumed batches. drop_last=True on both sampler and loader makes
    # batches-per-epoch deterministic, so global_step -> (epoch, offset) is
    # exact. batches_per_epoch may be 0 for iterable/unsized loaders — guard it.
    start_epoch = 0
    resume_offset = 0
    if resume_meta is not None:
        global_step = int(resume_meta.get("step", 0) or 0)
        # Keep the step's counter in sync on resume; optim_step increments
        # from here, and metrics read train_state.global_step back out.
        train_state.global_step = global_step
        tokens_seen = int(resume_meta.get("tokens_seen", 0) or 0)
        try:
            batches_per_epoch = len(dataloader)
        except TypeError:
            batches_per_epoch = 0
        if batches_per_epoch > 0:
            start_epoch = global_step // batches_per_epoch
            resume_offset = global_step % batches_per_epoch
        else:
            # Unsized loader: fall back to the saved epoch/offset if present.
            start_epoch = int(resume_meta.get("epoch", 0) or 0)
            resume_offset = int(resume_meta.get("batch_offset", 0) or 0)
        # Time-to-first-post-resume-step: measured against the training entry
        # timestamp captured at the top of train() (see _train_t0), logged once
        # on the first completed step below as train/restart_seconds.
        _resumed = True
    else:
        _resumed = False
    # Push the RESOLVED CLI args to the wandb run config now — after every
    # args mutation (HF tp-force, loss_impl normalization) and after the
    # global_batch_size backfill — so the logged config reflects the settings
    # actually used for the rest of training, not the requested ones.
    if (
        ezpz.get_rank() == 0
        and wandb is not None
        and getattr(wandb, "run", None) is not None
    ):
        # allow_val_change since some keys may already be present from main().
        wandb.config.update(vars(args), allow_val_change=True)  # type:ignore
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for idx, batch in enumerate(dataloader):
            # Skip already-consumed batches in the resumed epoch only.
            if _resumed and epoch == start_epoch and idx < resume_offset:
                continue
            # Step-based stop (independent of --epochs).
            if args.train_iters and global_step >= args.train_iters:
                break
            # Finalize the backgrounded async fan-out AS SOON AS every rank's
            # copy is done — not deferred to the next save boundary. Each step
            # this cheaply votes (a torch all-reduce on the main thread, in
            # lockstep across ranks — the same footing as the barriers already
            # here, no cross-thread hazard) whether all ranks finished copying;
            # when they have, it stamps the durable .complete marker. This keeps
            # the saved-but-not-yet-durable window to ~copy-duration (~seconds)
            # instead of a full save interval, so a crash falls back at most ~1
            # interval like a sync save. The guard is rank-uniform (pending_ckpt
            # is set/cleared identically on every rank), so all ranks enter the
            # collective together. None-safe / no-op until ready.
            if args.async_ckpt and pending_ckpt is not None:
                from ezpz.examples._checkpoint import try_finalize_if_ready

                _fin_t0 = perf_counter()
                if try_finalize_if_ready(pending_ckpt) is not None:
                    pending_drain_seconds = perf_counter() - _fin_t0
                    pending_ckpt = None
            ezpz.distributed.synchronize()
            t0 = perf_counter()
            # The step body now lives in ezpz.tinker.step so the same
            # implementation backs both this loop and the Tinker-style
            # client (one forward_backward + one optim_step here == the
            # old fused body; see test_split_matches_fused_loop).
            #
            # `_t1_holder` carries the mid-step barrier time out of the
            # callback below. The callback fires exactly where the old
            # `t1 = perf_counter()` line sat -- after forward + label
            # masking, before the loss -- so train/dtf and train/dtb keep
            # their original meanings.
            _t1_holder: list[float] = []
            # Per-step stats derived from pred/labels. Computed in the
            # callback (the only point where the step still holds those
            # tensors) and merged into `metrics` below.
            _step_probe: dict[str, object] = {}
            _seq_holder: list[int] = []

            def _at_forward_done(
                pred: "torch.Tensor",
                masked_labels: "torch.Tensor",
                local_seq_len: int,
                _epoch: int = epoch,
                _idx: int = idx,
            ) -> None:
                ezpz.distributed.synchronize()
                _t1_holder.append(perf_counter())
                _seq_holder.append(int(local_seq_len))
                if global_step % max(metrics_every, 1) == 0:
                    _step_probe["labels/valid"] = float(
                        (masked_labels != -100).sum().item()
                    )
                    if track_logits:
                        _finite = torch.isfinite(pred)
                        _step_probe["logits/nonfinite"] = float(
                            (~_finite).sum().item()
                        )
                        _step_probe["logits/max_abs"] = float(
                            pred.abs().max().item()
                        )
                    if track_hist and ezpz.get_rank() == 0:
                        _sample = _sample_tensor_values(pred, hist_samples)
                        if _sample is not None:
                            _hist = _histogram_dict(_sample, hist_bins)
                            if _hist is not None:
                                _step_probe[f"hist/{dataset_tag}/logits"] = _hist
                tp_mod = getattr(ezpz, "tp", None)
                tp_rank = (
                    getattr(tp_mod, "get_tensor_parallel_rank", lambda: 0)()
                    if tp_mod is not None
                    else 0
                )
                # First-step finite/max debug stats. Gated behind
                # EZPZ_TRACK_LOGITS because `torch.isfinite(pred)` allocates
                # a full `(B, T, vocab)`-shaped bool tensor on the un-reduced
                # logits — at agpt's 256K vocab and long seq that's multiple
                # GB materialized *before* the loss, which can OOM a run that
                # would otherwise fit. Off by default. Skipped under
                # fused-linear: `pred` is hidden states there, so finite/
                # max-abs of it is meaningless.
                if (
                    track_logits
                    and not use_fused_linear
                    and _epoch == 0
                    and _idx == 0
                ):
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
                if _epoch == 0 and _idx == 0:
                    valid_labels = int((masked_labels != -100).sum().item())
                    logger.info(
                        "loss_inputs rank=%s tp=%s local_seq_len=%s labels=%s valid_labels=%s",
                        ezpz.get_rank(),
                        tp_rank,
                        local_seq_len,
                        tuple(masked_labels.shape),
                        valid_labels,
                    )

            _fb = _tinker_step.forward_backward(
                train_state, batch, on_forward_done=_at_forward_done
            )
            t1 = _t1_holder[0] if _t1_holder else perf_counter()
            loss = torch.as_tensor(_fb.loss)
            _opt_resp = _tinker_step.optim_step(train_state)
            grad_norm_preclip = _opt_resp.grad_norm
            # AdamW state (exp_avg/exp_avg_sq) is allocated by the first step on
            # a fresh run (a resumed run already has it from load_checkpoint);
            # log its dtype once, now that it exists (also amends the summary +
            # W&B config with the optimizer-state dtype).
            if not _optim_dtype_logged:
                _log_optimizer_state_dtype(optimizer, _precision_summary)
                _optim_dtype_logged = True
            ezpz.distributed.synchronize()
            t2 = perf_counter()
            # global_step and profiler.step() are advanced inside
            # _tinker_step.optim_step (once per OPTIMIZER step, which is
            # what the profiler schedule wants). Mirror the counter back
            # into the loop's local so the metrics below are unchanged.
            global_step = train_state.global_step
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
                # `batch` rather than the old `x`: the step now owns the
                # unpack, so the raw batch is what's still in scope here.
                # Same tensor, pre-shift (input/max|min are token-id
                # sanity bounds, unaffected by the causal shift).
                _ids = (
                    batch["input_ids"]
                    if isinstance(batch, dict) and "input_ids" in batch
                    else batch
                )
                metrics["input/max"] = float(_ids.max().item())
                metrics["input/min"] = float(_ids.min().item())
                # pred/labels-derived stats are computed inside the
                # forward-done callback (that is the only point where the
                # step still holds them) and stashed in _step_probe.
                metrics.update(_step_probe)
                if track_hist and ezpz.get_rank() == 0:
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
                # Per-DEVICE TFLOPS / MFU. `_model_flops` is counted on the
                # FULL, un-sharded model (estimated before parallelize() applies
                # TP), so it is the work of ONE data-parallel group — done
                # COLLECTIVELY by the group's `tp` GPUs. Divide by args.tp so
                # each metric reflects a single GPU's share; otherwise per-GPU
                # TFLOPS/MFU over-count by exactly `tp` (2x at tp=2, 4x at tp=4;
                # at tp=1 this is a no-op). Divide by tp, NOT world_size: unlike
                # the GLOBAL token count (tps_per_gpu ÷ world_size), _model_flops
                # is already per-DP-group, so only the tp ranks sharing that one
                # model must be divided out — the dpsize groups each do this
                # full-model work independently.
                flops_per_gpu = _model_flops / args.tp
                metrics["train/tflops"] = flops_per_gpu / dt_step / 1e12
                metrics["train/mfu"] = compute_mfu(flops_per_gpu, dt_step)
            # Throughput.
            #   - train/tps         : global tokens/sec across all GPUs
            #   - train/tps_per_gpu : per-GPU tokens/sec (torchtitan's `tgs`)
            #
            # Global tokens/step = batch * FULL pre-shard seq len (inp.shape[1])
            # * dpsize. `inp` is Replicate() across the tp group (the TP plan
            # shards only the embedding OUTPUT, never the input), so inp.shape[1]
            # is the full sequence, identical on every rank — exact and
            # rank-invariant even though only rank 0 logs. tp does NOT enter the
            # GLOBAL count: the tp ranks hold the SAME sequence (full-length
            # logits under the default Replicate() output, or Shard(1) slices
            # summing to inp.shape[1] under fused-linear / loss-parallel), never
            # distinct sequences. On the HF path (tp forced to 1, no SP) the
            # tp-dim ranks see DUPLICATE samples, so multiplying by dpsize (not
            # world_size) counts each distinct token once.
            # Global tokens processed THIS step across all distinct-data ranks.
            tokens_this_step = (
                args.batch_size * int(_fb.metrics["input_seq_len"]) * dpsize
            )
            tokens_seen += tokens_this_step
            # Cumulative consumed training tokens — the standard x-axis for
            # loss curves. Accumulated every step (this block runs each step),
            # so it's exact regardless of the metrics-logging interval.
            metrics["train/tokens"] = tokens_this_step
            metrics["train/tokens_seen"] = tokens_seen
            if dt_step > 0:
                # Per-GPU throughput = global tokens / (actual GPU count) / dt.
                # Divide by world_size, NOT dpsize: under TP the `tp` GPUs in a
                # data-parallel group process that group's tokens TOGETHER, so
                # the per-GPU rate is tp× lower. (tokens_per_rank / dt would be
                # per-DP-group, over-counting per-GPU by `tp`; at tp=1
                # world_size==dpsize so this is unchanged.)
                metrics["train/tps_per_gpu"] = tokens_this_step / world_size / dt_step
                metrics["train/tps"] = tokens_this_step / dt_step
            # Async-ckpt staging time from the previous step's save, carried
            # here so it lands in the JSONL + W&B alongside the other train/*
            # metrics (set once, then cleared).
            if pending_stage_seconds is not None:
                metrics["train/ckpt_stage_seconds"] = pending_stage_seconds
                pending_stage_seconds = None
            if pending_save_seconds is not None:
                metrics["train/ckpt_save_seconds"] = pending_save_seconds
                pending_save_seconds = None
            if pending_drain_seconds is not None:
                metrics["train/ckpt_drain_seconds"] = pending_drain_seconds
                pending_drain_seconds = None
            # Restart time: on the FIRST completed step after a resume, log how
            # long from train() entry (process init + dist setup + model build
            # + dcp.load + this step) to a productive step. This is the full
            # cold path a real --auto-retry failover pays. Logged once.
            if _resumed and not _restart_logged:
                metrics["train/restart_seconds"] = perf_counter() - _train_t0
                _restart_logged = True
            # Device memory: empty on CPU/MPS, 4 keys on CUDA/XPU.
            metrics |= ezpz.get_memory_metrics(prefix="train/")
            history.update(metrics, summarize=False)
            # Write the precision summary as its OWN JSONL record, once.
            # Deliberately NOT merged into `metrics`: History is a NUMERIC
            # store (it coerces values with float() and builds an xarray
            # Dataset), so string dtype names there break get_dataset() with
            # "ValueError: too many dimensions 'str'" — taking the end-of-run
            # report/plots down with it. _write_jsonl_entry is string-safe and
            # independent of the numeric store. Emitted after step 1 so
            # optimizer_states is populated.
            if not _precision_in_jsonl:
                try:
                    history._write_jsonl_entry(
                        {f"precision/{k}": v for k, v in _precision_summary.items()}
                    )
                except Exception:  # noqa: BLE001 — logging must not break training
                    logger.debug("could not write precision JSONL entry", exc_info=True)
                _precision_in_jsonl = True
            history.log_metrics(
                metrics,
                logger=logger,
                debug_prefixes=("hist/", "grad/", "input/", "labels/", "param/"),
                include_summary=True,
                rank0_only_summary=True,
            )
            if epoch == 0 and idx == 0:
                logger.info(f"{x.shape}")
            # Save a checkpoint every --save-interval optimizer steps.
            if (
                args.ckpt_dir
                and args.save_interval
                and global_step % args.save_interval == 0
            ):
                _ckpt_meta = {
                    "tokens_seen": tokens_seen,
                    "epoch": epoch,
                    # Next batch to run on resume within `epoch`.
                    "batch_offset": idx + 1,
                }
                if args.async_ckpt:
                    from ezpz.examples._checkpoint import (
                        finalize_fanout,
                        save_checkpoint_async,
                        start_fanout,
                    )

                    # Only one async save may be in flight (DCP constraint), so
                    # ensure the previous one is finalized before starting a new
                    # one. Normally the per-step try_finalize_if_ready already
                    # finalized it (pending_ckpt is None here); this is the
                    # fallback for when the save interval is SHORTER than the
                    # copy — then finalize_fanout blocks on the still-running
                    # copy. Only record the drain time when we actually finalized
                    # here (else keep the per-step poll's measurement). None-safe.
                    if pending_ckpt is not None:
                        _drain_t0 = perf_counter()
                        finalize_fanout(pending_ckpt)
                        pending_drain_seconds = perf_counter() - _drain_t0
                        pending_ckpt = None
                    _t_stage = perf_counter()
                    pending_ckpt = save_checkpoint_async(
                        args.ckpt_dir,
                        ckpt_stage_dir,
                        global_step,
                        model,
                        optimizer,
                        meta=_ckpt_meta,
                    )
                    # Caller-thread stall = staging time only. Stash it so the
                    # NEXT step's metrics dict carries train/ckpt_stage_seconds
                    # through history.update -> JSONL + W&B (this save block runs
                    # after the current step's metrics were already written).
                    _stage_s = perf_counter() - _t_stage
                    pending_stage_seconds = _stage_s
                    # Kick the /tmp -> shared-FS fan-out onto a background
                    # thread; it overlaps the next save interval of training and
                    # is finalized at the next save boundary (above).
                    start_fanout(pending_ckpt)
                    if ezpz.get_rank() == 0:
                        logger.info(
                            "train/ckpt_stage_seconds=%.4f (async stage @ "
                            "step %d)",
                            _stage_s,
                            global_step,
                        )
                else:
                    from ezpz.examples._checkpoint import save_checkpoint

                    # Synchronous save BLOCKS the training loop for the full
                    # write. Time it (train/ckpt_save_seconds) so the sync-vs-
                    # async trade-off is measurable — this is the stall async
                    # removes. Folded into the next step's metrics dict.
                    _t_save = perf_counter()
                    save_checkpoint(
                        args.ckpt_dir,
                        global_step,
                        model,
                        optimizer,
                        meta=_ckpt_meta,
                    )
                    pending_stage_seconds = None  # (async-only; keep clear)
                    _save_s = perf_counter() - _t_save
                    pending_save_seconds = _save_s
                    if ezpz.get_rank() == 0:
                        logger.info(
                            "train/ckpt_save_seconds=%.4f (sync save @ step %d)",
                            _save_s,
                            global_step,
                        )
        # Step-based stop breaks the inner loop; also break the epoch loop.
        if args.train_iters and global_step >= args.train_iters:
            break
    # Finish any in-flight async checkpoint so a run that ends right after a
    # save doesn't lose it (the background fan-out may still be running), then
    # tear down the fan-out worker. drain() joins the background copy if one is
    # in flight, else runs it inline; either way it does the barrier + marker.
    if args.async_ckpt:
        from ezpz.examples._checkpoint import drain, shutdown_fanout_pool

        drain(pending_ckpt)
        shutdown_fanout_pool()
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
    base_dir = args.outdir if args.outdir else None
    # Collective (broadcasts the shared timestamp) — every rank must call it.
    outdir = get_example_outdir(WBPROJ_NAME, base_dir=base_dir)
    # Create the W&B run HERE, so it exists before the slow startup path.
    # History (inside train()) is constructed only after tokenization,
    # model build, FLOP counting, FSDP wrapping and torch.compile — on a
    # 4-node agpt-2b run that is ~60s, and at 20b it spans the OOM-prone
    # build/compile phase. Creating the run at History time meant a job
    # that died before the first step uploaded nothing at all.
    #
    # This does NOT move ownership away from History: History still builds
    # the tracker, and its WandbBackend still calls setup_wandb. It simply
    # *adopts* this run rather than creating one — setup_wandb passes
    # reinit=None and wandb.init() returns the existing run object when one
    # is live (verified on wandb 0.24.0 and 0.28.1), so config updates from
    # both sites merge onto a single run.
    #
    # Gated on the same backend resolution History uses, so
    # EZPZ_TRACKER_BACKENDS=none (or a csv/mlflow-only selection) still
    # keeps W&B entirely out of the run. WANDB_MODE=disabled/offline
    # remains the other opt-out, and setup_wandb additionally no-ops when
    # verify_wandb() fails or rank != 0 — it performs no collectives, so
    # calling it early cannot deadlock or diverge ranks.
    if rank == 0 and "wandb" in ezpz.tracker.resolve_backend_names():
        ezpz.setup_wandb(project_name=WBPROJ_NAME, dir=outdir)
        # Dumped *after* wandb.init so the resolved config lands in the
        # run's captured console log too (wandb does not capture stdout
        # retroactively).
        jstr = json.dumps(vars(args), indent=2, sort_keys=True, default=str)
        logger.info(f"config:\n{jstr}")
    logger.info("Outputs will be saved to %s", outdir)
    # W&B run created above; History (inside train()) adopts it and adds
    # the CSV/JSONL backends.
    train_start = time.perf_counter()
    # nullcontext (prof=None) unless --profile / --pyinstrument-profiler set.
    with profiling_context_from_args(args, outdir) as prof:
        # Pass main()'s t0 (captured before setup_torch) so
        # train/restart_seconds covers the full cold path incl. distributed
        # init — the dominant cost of a real --auto-retry failover.
        history = train(
            args=args, outdir=outdir, profiler=prof, process_start=t0
        )
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
