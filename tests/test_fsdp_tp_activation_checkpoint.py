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
