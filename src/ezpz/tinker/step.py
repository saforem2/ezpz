"""The training step, split into ``forward_backward`` and ``optim_step``.

``ezpz.examples.fsdp_tp`` grew its step inline: batch prep, forward, loss,
``zero_grad``, ``backward``, clip and ``optimizer.step()`` all sit in one
~400-line loop body over ~25 locals. That shape can express exactly one
thing -- "one batch, one update". This module carves out the two halves so
callers can interleave them, which buys three things the fused loop could
not do:

* **gradient accumulation** -- N microbatches, then one update;
* **RL / preference losses** -- generate, score, then update, with the
  extra per-loss inputs riding along in :class:`~ezpz.tinker.types.Datum`;
* **a client API** -- the same two calls a hosted service would expose.

The one semantic change: ``zero_grad`` moves from the *top* of the update
block (``fsdp_tp.py:3416``, immediately before ``backward()``) to the *end*
of :func:`optim_step`. In the fused loop those are equivalent; once the
halves are separable they are not -- the old placement silently discarded
every gradient but the last, so accumulation was impossible. Same number
of ``zero_grad`` calls in the same order for a 1:1 caller, so an
unchanged loop behaves identically.

Everything else is moved verbatim. The state each half needs is gathered
into :class:`TrainState`, which is a plain view over objects ``train()``
already builds -- it owns nothing and constructs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

import ezpz
from ezpz.tinker.types import (
    AdamParams,
    ForwardBackwardOutput,
    OptimStepResponse,
    validate_loss_fn,
)

logger = ezpz.get_logger(__name__)


@dataclass
class TrainState:
    """Everything one step touches.

    A *view* over objects ``fsdp_tp.train()`` already has in scope, not a
    new owner: pass the same ``model`` / ``optimizer`` and behavior is
    unchanged. ``global_step`` is the only field this module mutates.
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device
    args: Any
    """The argparse Namespace. Read live rather than snapshotted: `train()`
    mutates it during setup (`fsdp_tp.py:2547`, `:2610`, `:2679`, `:2822`),
    so a copy taken earlier would be stale."""

    base_model: Optional[torch.nn.Module] = None
    """FSDP-unwrapped view, for ``base_model.output`` under fused-linear."""

    dataset: Any = None
    """Only for ``dataset.pad_id``."""

    tp_group: Any = None
    use_fused_linear: bool = False
    use_loss_parallel: bool = False
    profiler: Any = None
    global_step: int = 0

    # Loss implementations, injected by fsdp_tp so this module does not
    # import it back (that would be circular). Signatures match the
    # originals exactly.
    compute_loss: Optional[Callable[..., torch.Tensor]] = None
    localize_logits: Optional[Callable[..., torch.Tensor]] = None
    fused_linear_loss: Optional[Callable[..., torch.Tensor]] = None
    vocab_parallel_loss: Optional[Callable[..., torch.Tensor]] = None
    slice_for_sequence_parallel: Optional[Callable[..., torch.Tensor]] = None

    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def output_module(self) -> torch.nn.Module:
        """The output projection the fused-linear loss calls per chunk."""
        src = self.base_model if self.base_model is not None else self.model
        out = getattr(src, "output", None)
        if out is None:
            raise AttributeError(
                "fused-linear loss needs `.output` on the (unwrapped) model; "
                f"{type(src).__name__} has none."
            )
        return out


@dataclass
class PreparedBatch:
    """A batch turned into model inputs + masked labels."""

    inp: torch.Tensor
    labels: torch.Tensor
    attn_mask: Optional[torch.Tensor] = None
    num_tokens: int = 0


def prepare_batch(state: TrainState, batch: Any) -> PreparedBatch:
    """Unpack, move to device, shift for causal LM, build the ignore mask.

    Verbatim from ``fsdp_tp.py:3272-3290`` (unpack/shift) and ``:3316-3332``
    (ignore mask). The SP-slicing that sat in between needs the model's
    output shape, so it happens in :func:`forward_backward` instead.
    """
    attn_mask = None
    if isinstance(batch, dict) and "input_ids" in batch:
        x = batch["input_ids"]
        attn_mask = batch.get("attention_mask")
    else:
        x = batch
    assert isinstance(x, torch.Tensor)
    x = x.to(state.device)
    x = x.to(torch.long)
    # Both branches of the original `if args.dataset == "random"` did the
    # same shift; collapsed.
    inp = x[:, :-1]
    labels = x[:, 1:]
    inp = inp.to(state.device)
    labels = labels.to(state.device)
    if attn_mask is not None:
        attn_mask = attn_mask.to(state.device)
    return PreparedBatch(
        inp=inp,
        labels=labels,
        attn_mask=attn_mask,
        num_tokens=int(labels.numel()),
    )


def _apply_ignore_mask(
    state: TrainState,
    labels: torch.Tensor,
    attn_labels: Optional[torch.Tensor],
) -> torch.Tensor:
    """One ``masked_fill`` for attention-pad OR tokenizer-pad.

    Verbatim from ``fsdp_tp.py:3316-3332``. The single-fill form is a
    deliberate perf choice: two ``.clone()`` + boolean index-assigns were
    ~8.8% of an agpt-2b step in ``aten::copy_``.
    """
    pad_id = getattr(state.dataset, "pad_id", None)
    ignore_mask = None
    if attn_labels is not None:
        ignore_mask = attn_labels == 0
    if pad_id is not None:
        pad_mask = labels == int(pad_id)
        ignore_mask = pad_mask if ignore_mask is None else (ignore_mask | pad_mask)
    if ignore_mask is not None:
        labels = labels.masked_fill(ignore_mask, -100)
    return labels


def compute_loss(
    state: TrainState,
    pred: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Three-way dispatch, verbatim from ``fsdp_tp.py:3363-3400``.

    The mode is fixed at setup (``use_fused_linear`` / ``use_loss_parallel``
    are resolved in ``train()``), which is why this reads booleans rather
    than a string.
    """
    args = state.args
    if state.use_fused_linear:
        # `pred` is hidden states (B,T,dim); the fused loss runs the output
        # projection MODULE per row-chunk so FSDP unshards the weight and
        # routes its grad, never materializing (B,T,vocab).
        assert state.fused_linear_loss is not None
        return state.fused_linear_loss(
            pred,
            state.output_module,
            labels,
            ignore_index=-100,
            chunk_size=args.loss_chunk_size,
        )
    if state.use_loss_parallel:
        # `pred` is this rank's local [B, T, vocab/tp] shard; vocab-parallel
        # CE reduces across the TP group and needs the global vocab size to
        # compute shard bounds.
        assert state.vocab_parallel_loss is not None
        return state.vocab_parallel_loss(
            pred,
            labels,
            ignore_index=-100,
            global_vocab_size=args.vocab_size,
            tp_group=state.tp_group,
        )
    # tp>1 non-loss-parallel: `pred` is a REPLICATED DTensor but `labels` is
    # plain, so plain CE would raise "mixed torch.Tensor and DTensor".
    # Localize first (no-op at tp=1 / HF).
    assert state.compute_loss is not None
    assert state.localize_logits is not None
    return state.compute_loss(
        state.localize_logits(pred),
        labels,
        impl=args.loss_impl,
        ignore_index=-100,
        chunk_size=args.loss_chunk_size,
    )


def forward_backward(
    state: TrainState,
    batch: Any,
    loss_fn: str = "cross_entropy",
    *,
    loss_scale: float = 1.0,
    on_forward_done: Optional[Callable[[], None]] = None,
) -> ForwardBackwardOutput:
    """Forward, loss, backward. Accumulates grads; does NOT update.

    Call repeatedly before one :func:`optim_step` to accumulate over
    microbatches. Note ``zero_grad`` is deliberately absent -- it belongs
    to :func:`optim_step`, which is what makes accumulation work.

    Args:
        loss_scale: multiplies the loss before ``backward()``. Pass
            ``1/N`` when accumulating N microbatches so the summed
            gradient matches a single batch of the same total size.
        on_forward_done: invoked after the forward pass and label
            masking, before the loss. Exists so ``fsdp_tp`` can keep
            drawing its ``t1`` barrier exactly where it did
            (``fsdp_tp.py:3333-3334``) and its ``train/dtf`` /
            ``train/dtb`` split stays comparable across the refactor.
            Without this hook the boundary would be swallowed here and
            those two metrics would silently shift meaning.

    Returns:
        The unscaled loss (so the number a caller logs is comparable
        regardless of accumulation) plus the token count.
    """
    validate_loss_fn(loss_fn)
    prepared = prepare_batch(state, batch)
    inp, labels = prepared.inp, prepared.labels

    # fused-linear wants hidden states, everything else wants logits.
    if state.use_fused_linear:
        pred = state.model(inp, return_hidden=True)
    else:
        pred = state.model(inp)
    # HF causal-LM models return a dataclass; ezpz's Transformer returns
    # logits directly.
    if hasattr(pred, "logits"):
        pred = pred.logits

    # dim-1 is the (SP-local) sequence length either way.
    local_seq_len = pred.shape[1]
    attn_labels = None
    if labels.shape[1] != local_seq_len:
        assert state.slice_for_sequence_parallel is not None
        labels = state.slice_for_sequence_parallel(labels, local_seq_len)
    if prepared.attn_mask is not None:
        attn_mask = prepared.attn_mask
        attn_labels = attn_mask[:, 1:] if attn_mask.shape[1] > 1 else attn_mask
        if attn_labels.shape[1] != local_seq_len:
            assert state.slice_for_sequence_parallel is not None
            attn_labels = state.slice_for_sequence_parallel(
                attn_labels, local_seq_len
            )
    labels = _apply_ignore_mask(state, labels, attn_labels)

    # The forward/backward split point. fsdp_tp draws its `t1` barrier
    # here so `train/dtf` keeps meaning "data prep + forward" and
    # `train/dtb` keeps meaning "loss + backward".
    if on_forward_done is not None:
        on_forward_done()

    loss = compute_loss(state, pred, labels)
    (loss * loss_scale if loss_scale != 1.0 else loss).backward()

    return ForwardBackwardOutput(
        loss=float(loss.detach()),
        num_tokens=int((labels != -100).sum().item()),
        metrics={"local_seq_len": float(local_seq_len)},
    )


def optim_step(
    state: TrainState,
    adam_params: Optional[AdamParams] = None,
) -> OptimStepResponse:
    """Clip, update, advance the profiler, then zero the grads.

    Verbatim from ``fsdp_tp.py:3418-3423`` (clip + step) and ``:3433-3437``
    (step counter + profiler), with ``zero_grad`` moved here from
    ``:3416``.

    Args:
        adam_params: when given, ``learning_rate`` and ``grad_clip_norm``
            are applied to this step -- which is how an RL loop varies
            them without rebuilding the optimizer. When ``None``, the
            optimizer's existing LR and ``args.max_grad_norm`` are used,
            reproducing the current behavior exactly.
    """
    args = state.args
    clip = args.max_grad_norm
    if adam_params is not None:
        clip = adam_params.grad_clip_norm
        for group in state.optimizer.param_groups:
            group["lr"] = adam_params.learning_rate

    grad_norm_preclip = None
    if clip and clip > 0:
        grad_norm_preclip = torch.nn.utils.clip_grad_norm_(
            state.model.parameters(), clip
        )
    state.optimizer.step()
    state.global_step += 1
    # Advance the torch.profiler schedule once per OPTIMIZER step (not per
    # microbatch). No-op when not profiling.
    if state.profiler is not None:
        state.profiler.step()
    # Moved from the top of the old update block: with the halves split,
    # zeroing before backward would discard every accumulated microbatch
    # but the last.
    state.optimizer.zero_grad(set_to_none=True)

    return OptimStepResponse(
        step=state.global_step,
        grad_norm=(
            float(grad_norm_preclip) if grad_norm_preclip is not None else None
        ),
        learning_rate=float(state.optimizer.param_groups[0]["lr"]),
    )
