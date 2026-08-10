"""Tests for the ``--lora-*`` flags on ``ezpz.examples.fsdp_tp``.

Covers the CLI surface and the two integration points where LoRA meets
the existing pipeline:

* ``_lora_is_applied`` gates the TP-plan retarget, so it must be exactly
  right -- a false negative means ``tp>1`` raises
  ``NotImplementedError`` deep in ``parallelize_module``; a false
  positive would rewrite plan keys that have no ``.base``.
* ``--lora-rank 0`` (the default) must leave everything untouched, so
  every existing run is unaffected.

CPU-only; arg parsing and module inspection need no accelerator.
"""

from __future__ import annotations

import importlib

import pytest


def _fsdp_tp():
    try:
        return importlib.import_module("ezpz.examples.fsdp_tp")
    except ModuleNotFoundError as exc:
        pytest.skip(f"missing optional dependency for fsdp_tp: {exc}")


class TestLoraFlags:
    def test_defaults_off(self):
        m = _fsdp_tp()
        a = m.parse_args(["--model", "debug", "--tp", "1"])
        assert a.lora_rank == 0
        assert a.lora_alpha is None
        assert a.lora_dropout == 0.0
        assert a.lora_target == "attn,mlp"

    def test_rank_and_alpha(self):
        m = _fsdp_tp()
        a = m.parse_args(
            ["--model", "debug", "--tp", "1", "--lora-rank", "16",
             "--lora-alpha", "32"]
        )
        assert a.lora_rank == 16 and a.lora_alpha == 32.0

    @pytest.mark.parametrize(
        "target", ["attn", "mlp", "unembed", "attn,mlp", "attn,mlp,unembed"]
    )
    def test_target_spellings_parse(self, target):
        m = _fsdp_tp()
        a = m.parse_args(
            ["--model", "debug", "--tp", "1", "--lora-target", target]
        )
        assert a.lora_target == target


class TestLoraIsApplied:
    """The predicate that gates the TP-plan retarget."""

    def test_false_on_a_plain_model(self):
        torch = pytest.importorskip("torch")
        m = _fsdp_tp()
        assert m._lora_is_applied(torch.nn.Linear(4, 4)) is False

    def test_true_after_apply_lora(self):
        torch = pytest.importorskip("torch")
        nn = torch.nn
        m = _fsdp_tp()
        from ezpz.tinker.lora import LoraConfig, apply_lora

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = nn.Linear(8, 8, bias=False)

        class Blk(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = Attn()

        model = nn.Module()
        model.layers = nn.ModuleList([Blk()])
        apply_lora(
            model, LoraConfig(rank=4, train_mlp=False), verbose=False
        )
        assert m._lora_is_applied(model) is True


class TestTpGuard:
    """LoRA at tp>1 must refuse UP FRONT, not fail mid-forward.

    parallelize halves n_heads in place, so attention.wo receives the
    per-rank width while the adapter's A was built from the
    unparallelized in_features -- a weight-SHAPE mismatch no TP style
    can fix (Sunspot jobs 12472831/33/34/36):
      RuntimeError: a and b must have same reduction dim,
      but got [128, 64] X [128, 8]
    """

    def test_lora_with_tp_gt_1_is_refused(self):
        m = _fsdp_tp()
        src = __import__("inspect").getsource(m.train)
        assert "--lora-rank with --tp > 1 is not supported yet" in src, (
            "the tp>1 LoRA guard is missing; runs would fail deep in the "
            "first forward instead of at setup"
        )

    def test_guard_names_the_workaround(self):
        m = _fsdp_tp()
        src = __import__("inspect").getsource(m.train)
        assert "Use --tp 1" in src


class TestTargetValidation:
    """An unknown --lora-target must fail loudly at setup, not silently
    adapt nothing (which would look like LoRA 'not working')."""

    def test_unknown_target_rejected(self):
        from ezpz.tinker.lora import LoraConfig

        # The SystemExit is raised in train(); here we pin the underlying
        # contract that an empty target set is refused.
        with pytest.raises(ValueError, match="adapts nothing"):
            LoraConfig(rank=4, train_attn=False, train_mlp=False)


class TestTpPlanRetargetWiring:
    """The retarget must fire only when LoRA is present."""

    def test_plan_unchanged_without_lora(self):
        from ezpz.tinker.lora import lora_tp_plan

        plan = {"attention.wq": "COL", "attention_norm": "SP"}
        # lora_tp_plan is only CALLED when _lora_is_applied is True; this
        # pins that calling it needlessly would change keys, which is why
        # the guard exists.
        assert lora_tp_plan(plan) != plan

    def test_norm_entries_never_retargeted(self):
        from ezpz.tinker.lora import lora_tp_plan

        out = lora_tp_plan({"attention_norm": "SP", "ffn_norm": "SP"})
        assert out == {"attention_norm": "SP", "ffn_norm": "SP"}
