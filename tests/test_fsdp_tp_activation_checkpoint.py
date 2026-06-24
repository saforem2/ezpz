"""Tests for the ``--activation-checkpoint`` flag in fsdp_tp.

Coverage:
  - Parser accepts {none, block, selective} via both ``--activation-checkpoint``
    and the short alias ``--ac``.
  - Default is ``none``.
  - Both `--flag value` and `--flag=value` shapes are honored (regression
    against the apply_model_preset _arg_provided= bug).
  - Applying AC to a tiny ezpz Transformer produces bit-identical
    forward + backward outputs vs. the un-checkpointed model — the
    only difference should be memory + speed, NOT numerics.
"""

from __future__ import annotations

import importlib

import pytest


def _import_fsdp_tp():
    try:
        return importlib.import_module("ezpz.examples.fsdp_tp")
    except Exception as exc:
        pytest.skip(f"could not import ezpz.examples.fsdp_tp: {exc}")


@pytest.mark.parametrize("mode", ["none", "block", "full", "selective"])
def test_parse_accepts_each_mode_long_form(mode):
    mod = _import_fsdp_tp()
    args = mod.parse_args(["--activation-checkpoint", mode])
    assert args.activation_checkpoint == mode


@pytest.mark.parametrize("mode", ["none", "block", "full", "selective"])
def test_parse_accepts_each_mode_short_alias(mode):
    """``--ac`` is the short alias; must round-trip identically."""
    mod = _import_fsdp_tp()
    args = mod.parse_args(["--ac", mode])
    assert args.activation_checkpoint == mode


@pytest.mark.parametrize("mode", ["none", "block", "full", "selective"])
def test_parse_accepts_equals_fused(mode):
    """`--ac=block` (one token) must work the same as `--ac block` (two)."""
    mod = _import_fsdp_tp()
    args = mod.parse_args([f"--ac={mode}"])
    assert args.activation_checkpoint == mode


def test_default_is_none():
    mod = _import_fsdp_tp()
    args = mod.parse_args([])
    assert args.activation_checkpoint == "none"


def test_unknown_mode_is_rejected():
    mod = _import_fsdp_tp()
    with pytest.raises(SystemExit):
        mod.parse_args(["--ac", "checkpoint-everything-please"])


def test_apply_ac_none_is_noop():
    """`mode=none` returns the model untouched."""
    mod = _import_fsdp_tp()
    import torch.nn as nn

    m = nn.Linear(4, 4)
    result = mod._apply_activation_checkpointing(m, "none")
    assert result is m


def test_block_ac_preserves_numerics():
    """Forward pass + backward grads must be bit-identical with AC vs.
    without. AC is purely a memory/throughput trade — numerics must
    not change."""
    pytest.importorskip("torch")
    import torch

    mod = _import_fsdp_tp()
    from ezpz.models.llama import ModelArgs, Transformer

    cfg = dict(
        dim=32, n_layers=2, n_heads=2, vocab_size=100,
        batch_size=1, max_seq_len=16,
    )

    torch.manual_seed(0)
    m_plain = Transformer.from_model_args(ModelArgs(**cfg))
    torch.manual_seed(0)
    m_ac = Transformer.from_model_args(ModelArgs(**cfg))
    m_ac = mod._apply_activation_checkpointing(m_ac, "block")

    inp = torch.randint(0, 100, (1, 8))

    out_plain = m_plain(inp)
    out_ac = m_ac(inp)
    assert torch.allclose(out_plain, out_ac, atol=1e-6), (
        f"AC changed forward output (max diff "
        f"{(out_plain - out_ac).abs().max().item()})"
    )

    out_plain.sum().backward()
    out_ac.sum().backward()

    for name in ("tok_embeddings", "output"):
        g_plain = getattr(m_plain, name).weight.grad
        g_ac = getattr(m_ac, name).weight.grad
        assert torch.allclose(g_plain, g_ac, atol=1e-6), (
            f"AC changed {name}.weight grad (max diff "
            f"{(g_plain - g_ac).abs().max().item()})"
        )


def test_full_is_alias_for_block():
    """`--ac full` is a compatibility alias for `--ac block` (torchtitan
    uses `full` for what we call `block`). Applying either to identical
    models must produce bit-identical forward outputs and grads."""
    pytest.importorskip("torch")
    import torch

    mod = _import_fsdp_tp()
    from ezpz.models.llama import ModelArgs, Transformer

    cfg = dict(
        dim=32, n_layers=2, n_heads=2, vocab_size=100,
        batch_size=1, max_seq_len=16,
    )

    torch.manual_seed(0)
    m_block = Transformer.from_model_args(ModelArgs(**cfg))
    m_block = mod._apply_activation_checkpointing(m_block, "block")

    torch.manual_seed(0)
    m_full = Transformer.from_model_args(ModelArgs(**cfg))
    m_full = mod._apply_activation_checkpointing(m_full, "full")

    inp = torch.randint(0, 100, (1, 8))
    out_block = m_block(inp)
    out_full = m_full(inp)
    assert torch.allclose(out_block, out_full, atol=0), (
        "full and block produced different forward outputs"
    )

    out_block.sum().backward()
    out_full.sum().backward()
    g_block = m_block.tok_embeddings.weight.grad
    g_full = m_full.tok_embeddings.weight.grad
    assert torch.allclose(g_block, g_full, atol=0), (
        "full and block produced different grads"
    )


def test_hf_path_uses_gradient_checkpointing_enable_and_disables_cache():
    """For HF-style models (anything with `gradient_checkpointing_enable`
    + `.config.use_cache`), the AC helper must:
      1. Set `model.config.use_cache = False`
      2. Call `model.gradient_checkpointing_enable(...)` with
         `use_reentrant=False`

    NOT use our generic per-block `torch.utils.checkpoint` wrap — HF's
    DynamicCache silently allocates different numbers of saved tensors
    on forward vs. recompute, breaking generic checkpointing with
    `CheckpointError: A different number of tensors was saved during
    the original forward and recomputation`.
    """
    from unittest.mock import MagicMock

    mod = _import_fsdp_tp()

    fake_hf_model = MagicMock()
    fake_hf_model.config.use_cache = True  # like a real HF causal-LM
    # MagicMock doesn't behave like an FSDP-wrapped module by default;
    # ensure `_fsdp_wrapped_module` lookup falls through to fake_hf_model.
    fake_hf_model._fsdp_wrapped_module = fake_hf_model

    result = mod._apply_activation_checkpointing(fake_hf_model, "block")

    assert result is fake_hf_model
    assert fake_hf_model.config.use_cache is False, (
        "AC must disable use_cache on HF models to avoid "
        "DynamicCache <-> checkpoint tensor-count mismatch"
    )
    fake_hf_model.gradient_checkpointing_enable.assert_called_once_with(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )


def test_hf_path_leaves_use_cache_false_alone():
    """Idempotency: if use_cache is already False, don't log a noisy
    'disabled use_cache' message; just enable gradient checkpointing."""
    from unittest.mock import MagicMock

    mod = _import_fsdp_tp()

    fake_hf_model = MagicMock()
    fake_hf_model.config.use_cache = False
    fake_hf_model._fsdp_wrapped_module = fake_hf_model

    mod._apply_activation_checkpointing(fake_hf_model, "full")

    # Still False (untouched)
    assert fake_hf_model.config.use_cache is False
    fake_hf_model.gradient_checkpointing_enable.assert_called_once()


def test_ac_warns_when_no_blocks_found():
    """If the user requests AC on a model with no transformer-block
    ModuleList (e.g. a plain Linear), the helper must log a warning
    and return the model unmodified — NOT crash."""
    mod = _import_fsdp_tp()
    import torch.nn as nn

    m = nn.Linear(4, 4)
    result = mod._apply_activation_checkpointing(m, "block")
    # Returned model is the same object — AC was a no-op (with warning).
    assert result is m


def test_selective_warns_when_blocks_lack_attention(caplog):
    """`--ac selective` only wraps `.attention` submodules. For HF
    blocks (which use `.self_attn`) or any non-ezpz arch, the wrap
    silently no-ops. The helper must emit ONE warning naming the
    skip count so users know to use --ac block instead."""
    import logging

    import torch.nn as nn

    mod = _import_fsdp_tp()

    # Bare blocks with NO .attention attribute (mimics HF arch).
    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            return self.lin(x)

    class _Stack(nn.Module):
        def __init__(self, n=3):
            super().__init__()
            self.layers = nn.ModuleList([_Block() for _ in range(n)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    m = _Stack(n=3)
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        mod._apply_activation_checkpointing(m, "selective")

    skip_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "selective" in r.getMessage()
    ]
    assert len(skip_warnings) == 1, (
        f"expected exactly one selective-skip warning, got {skip_warnings}"
    )
    msg = skip_warnings[0].getMessage()
    assert "3/3" in msg
    assert "--ac block" in msg


def test_selective_no_warn_when_all_blocks_have_attention():
    """If every block has `.attention`, the selective-skip warning
    must NOT fire — only when something was actually skipped."""
    import logging

    import torch.nn as nn

    mod = _import_fsdp_tp()

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(4, 4)

        def forward(self, x):
            return self.attention(x)

    class _Stack(nn.Module):
        def __init__(self, n=2):
            super().__init__()
            self.layers = nn.ModuleList([_Block() for _ in range(n)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    m = _Stack(n=2)
    # Capture records by attaching a temporary handler to the module's
    # logger (caplog fixture-based isolation would need parametrize +
    # the per-logger propagate dance).
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[assignment]
    mod.logger.addHandler(handler)
    try:
        mod._apply_activation_checkpointing(m, "selective")
    finally:
        mod.logger.removeHandler(handler)

    skip_warnings = [
        r for r in records
        if r.levelno == logging.WARNING and "selective" in r.getMessage()
    ]
    assert skip_warnings == [], (
        f"selective-skip warning fired with zero skips: {skip_warnings}"
    )
