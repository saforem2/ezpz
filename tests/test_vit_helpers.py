"""Tests for pure-Python helpers in ``ezpz.examples.vit``.

The full module pulls in torch/torchvision/data layers; helpers are
imported lazily so we can skip cleanly if those aren't available.
"""

from __future__ import annotations

import math

import pytest


@pytest.fixture(scope="module")
def vit():
    try:
        from ezpz.examples import vit as _vit
    except Exception as exc:
        pytest.skip(f"ezpz.examples.vit unavailable: {exc}")
    return _vit


class TestComputeLrWarmupIters:
    """Tests for ``_compute_lr_warmup_iters``."""

    def test_zero_disables_warmup(self, vit):
        assert vit._compute_lr_warmup_iters(1000, 0) == 0

    def test_negative_disables_warmup(self, vit):
        assert vit._compute_lr_warmup_iters(1000, -5) == 0

    def test_fraction(self, vit):
        # 0.05 of 2000 → 100
        assert vit._compute_lr_warmup_iters(2000, 0.05) == 100

    def test_tiny_fraction_clamped_to_one(self, vit):
        # 0.0001 of 100 → int(0.01) = 0; min floor of 1 since fraction > 0
        assert vit._compute_lr_warmup_iters(100, 0.0001) == 1

    def test_absolute_step_count(self, vit):
        assert vit._compute_lr_warmup_iters(1000, 50) == 50

    def test_warmup_capped_at_total_minus_one(self, vit):
        """A warmup longer than the run must leave a step for decay."""
        # Old code: 200 → 200, overshoots max_iters=100, decay_steps=max(1,...)
        # so progress goes negative → broken cosine.  Now clamped.
        assert vit._compute_lr_warmup_iters(100, 200) == 99

    def test_fraction_one_capped(self, vit):
        # 1.0 means "100% of iters", but we still need one step for decay
        # (note: 1.0 hits the >= 1 branch, becomes 1 step).  This is
        # mostly a sanity check that we don't crash.
        result = vit._compute_lr_warmup_iters(100, 1.0)
        assert 0 <= result < 100

    def test_zero_max_iters(self, vit):
        """No iters → no warmup."""
        assert vit._compute_lr_warmup_iters(0, 50) == 0


class TestBuildLrScheduler:
    """Tests for ``build_lr_scheduler``."""

    def _make_optimizer(self, lr: float = 1e-3):
        import torch

        m = torch.nn.Linear(2, 2)
        return torch.optim.AdamW(m.parameters(), lr=lr)

    def test_none_max_iters_returns_none(self, vit):
        opt = self._make_optimizer()
        assert vit.build_lr_scheduler(
            opt, max_iters=None, lr_warmup_iters=0.05, min_lr_ratio=0.1,
        ) is None

    def test_zero_max_iters_returns_none(self, vit):
        opt = self._make_optimizer()
        assert vit.build_lr_scheduler(
            opt, max_iters=0, lr_warmup_iters=0.05, min_lr_ratio=0.1,
        ) is None

    def test_warmup_reaches_peak(self, vit):
        """After warmup_iters scheduler.step()s, LR should equal the
        peak (within float tolerance) — this is the main thing the
        original off-by-one bug obscured.
        """
        peak = 1.0
        opt = self._make_optimizer(lr=peak)
        sch = vit.build_lr_scheduler(
            opt, max_iters=100, lr_warmup_iters=10, min_lr_ratio=0.0,
        )
        assert sch is not None
        # Step 0 (initial state, before any sch.step): lambda(0) = 1/10 = 0.1
        assert opt.param_groups[0]["lr"] == pytest.approx(peak * 0.1)
        # After 9 sch.step calls we're at lambda(9) = 10/10 = 1.0
        for _ in range(9):
            sch.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(peak * 1.0)

    def test_disabled_warmup_starts_at_peak(self, vit):
        """warmup=0 → first step already at peak LR."""
        peak = 1.0
        opt = self._make_optimizer(lr=peak)
        sch = vit.build_lr_scheduler(
            opt, max_iters=100, lr_warmup_iters=0, min_lr_ratio=0.0,
        )
        assert sch is not None
        # First step's lambda(0): warmup_iters=0 so we go to decay
        # branch immediately; cosine(0) = 1.0
        assert opt.param_groups[0]["lr"] == pytest.approx(peak * 1.0)

    def test_cosine_decays_to_min_ratio(self, vit):
        """LR should approach peak * min_lr_ratio by the end of training."""
        peak = 1.0
        opt = self._make_optimizer(lr=peak)
        sch = vit.build_lr_scheduler(
            opt, max_iters=100, lr_warmup_iters=10, min_lr_ratio=0.1,
        )
        assert sch is not None
        # Step well past total_iters so progress saturates at 1.0
        # → cosine(pi) = 0 → lr = min_lr_ratio + (1 - 0.1) * 0 = 0.1
        for _ in range(200):
            sch.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(peak * 0.1, rel=1e-6)
        # And at step total_iters - 1, LR should already be close to floor
        opt2 = self._make_optimizer(lr=peak)
        sch2 = vit.build_lr_scheduler(
            opt2, max_iters=100, lr_warmup_iters=10, min_lr_ratio=0.1,
        )
        assert sch2 is not None
        for _ in range(99):
            sch2.step()
        # ~0.1003 by step 99 (progress=89/90 ≈ 0.989, cosine ≈ 0.0003)
        assert 0.1 <= opt2.param_groups[0]["lr"] < 0.11

    def test_warmup_longer_than_run_is_clamped(self, vit):
        """Warmup > max_iters used to make decay_steps = max(1, neg) → 1
        and progress went negative → cosine returned values > 1.
        """
        peak = 1.0
        opt = self._make_optimizer(lr=peak)
        # warmup=200, max_iters=100 → clamped to 99
        sch = vit.build_lr_scheduler(
            opt, max_iters=100, lr_warmup_iters=200, min_lr_ratio=0.0,
        )
        assert sch is not None
        # Walk through every step; LR must always stay in [0, peak]
        seen_lrs = []
        for _ in range(100):
            seen_lrs.append(opt.param_groups[0]["lr"])
            sch.step()
        for lr in seen_lrs:
            assert 0.0 <= lr <= peak + 1e-6, f"lr {lr} out of range"

    def test_min_lr_ratio_clamped_to_unit_interval(self, vit):
        """min_lr_ratio outside [0, 1] is silently clamped."""
        peak = 1.0
        opt = self._make_optimizer(lr=peak)
        # 5.0 should clamp to 1.0 → constant peak LR
        sch = vit.build_lr_scheduler(
            opt, max_iters=100, lr_warmup_iters=0, min_lr_ratio=5.0,
        )
        assert sch is not None
        for _ in range(50):
            sch.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(peak * 1.0)


class TestApplyDatasetOverrides:
    """Tests for the model-preset / MNIST-default precedence."""

    def test_mnist_default_applies_when_no_model(self, vit):
        import argparse

        args = argparse.Namespace(
            dataset="mnist", model=None,
            num_heads=16, head_dim=64, depth=24,
            img_size=224, num_classes=1000, patch_size=16,
            max_iters=100,
        )
        vit.apply_dataset_overrides(args, argv=[])
        # MNIST defaults applied
        assert args.depth == vit.MNIST_DEFAULTS["depth"]
        assert args.num_heads == vit.MNIST_DEFAULTS["num_heads"]
        assert args.max_iters == vit.MNIST_DEFAULTS["max_iters"]

    def test_model_preset_wins_over_mnist_default(self, vit):
        """The original bug: --model small on MNIST silently became the
        tiny MNIST default because dataset overrides ran second.
        """
        import argparse

        args = argparse.Namespace(
            dataset="mnist", model="small",
            # Pretend apply_model_preset already ran:
            num_heads=16, head_dim=64, depth=24, batch_size=128,
            img_size=224, num_classes=1000, patch_size=16,
            max_iters=100,
        )
        vit.apply_dataset_overrides(args, argv=["--model", "small"])
        # depth/num_heads/head_dim are in the small preset; MUST stay.
        assert args.depth == 24, "MNIST default clobbered the model preset!"
        assert args.num_heads == 16
        assert args.head_dim == 64
        # img_size/num_classes/patch_size are NOT in any model preset,
        # so MNIST defaults still apply.
        assert args.img_size == vit.MNIST_DEFAULTS["img_size"]
        assert args.num_classes == vit.MNIST_DEFAULTS["num_classes"]
        assert args.patch_size == vit.MNIST_DEFAULTS["patch_size"]

    def test_explicit_flag_wins_over_mnist_default(self, vit):
        import argparse

        args = argparse.Namespace(
            dataset="mnist", model=None,
            num_heads=16, head_dim=64, depth=8,  # user passed --depth 8
            img_size=224, num_classes=1000, patch_size=16,
            max_iters=100,
        )
        vit.apply_dataset_overrides(args, argv=["--depth", "8"])
        assert args.depth == 8

    def test_non_mnist_dataset_no_op(self, vit):
        import argparse

        args = argparse.Namespace(
            dataset="fake", model=None,
            num_heads=16, head_dim=64, depth=24,
            img_size=224, num_classes=1000, patch_size=16,
            max_iters=100,
        )
        vit.apply_dataset_overrides(args, argv=[])
        # Nothing should change
        assert args.depth == 24
        assert args.num_classes == 1000


class TestAttentionBlockSignature:
    """Regression test: qkv_bias must not break positional callers."""

    def test_format_remains_third_positional(self):
        """AttentionBlock(attn_fn, dim, num_heads, format) must still work
        — the original commit broke this by inserting qkv_bias before
        format.
        """
        import torch
        from ezpz.models.vit.attention import AttentionBlock

        # Identity attention: fine for a sanity instantiate-and-forward test
        def _identity(q, k, v):
            return v

        block = AttentionBlock(_identity, 64, 4, "bshd")
        # If the regression were present, "bshd" would have been
        # bound to qkv_bias as a truthy str.  Verify format took it.
        assert block.permute_qkv.func is torch.permute
        # Confirm qkv_bias defaulted to True (its default)
        assert block.qkv.bias is not None

    def test_qkv_bias_keyword_only(self):
        """Passing qkv_bias positionally should fail."""
        from ezpz.models.vit.attention import AttentionBlock

        def _identity(q, k, v):
            return v

        with pytest.raises(TypeError):
            AttentionBlock(_identity, 64, 4, None, False)  # type: ignore[misc]

    def test_qkv_bias_false(self):
        from ezpz.models.vit.attention import AttentionBlock

        def _identity(q, k, v):
            return v

        block = AttentionBlock(_identity, 64, 4, qkv_bias=False)
        assert block.qkv.bias is None
