"""Tests for ``ezpz.flops``."""

from __future__ import annotations

import pytest
import torch

from ezpz.flops import (
    compute_mfu,
    estimate_model_flops,
    get_peak_flops,
    try_estimate,
)
from ezpz.flops import _extract_loss


class TestGetPeakFlops:
    """Tests for ``get_peak_flops``."""

    def test_a100(self):
        assert get_peak_flops("NVIDIA A100-SXM4-80GB") == 312e12

    def test_h100_sxm(self):
        assert get_peak_flops("NVIDIA H100 80GB HBM3") == 989e12

    def test_h100_nvl(self):
        assert get_peak_flops("NVIDIA H100 NVL") == 835e12

    def test_h100_pcie(self):
        assert get_peak_flops("NVIDIA H100 PCIe") == 756e12

    def test_h200(self):
        assert get_peak_flops("NVIDIA H200") == 989e12

    def test_b200(self):
        assert get_peak_flops("NVIDIA B200") == 2.25e15

    def test_mi300x(self):
        assert get_peak_flops("AMD MI300X") == 1300e12

    def test_mi250x(self):
        assert get_peak_flops("AMD Instinct MI250X") == 191.5e12

    def test_mi355x(self):
        # BF16 dense (5.0 PF), per AMD MI355X brochure
        assert get_peak_flops("AMD MI355X") == 5000e12

    def test_l40s(self):
        # BF16 dense (183 TFLOPS); 366 with sparsity, 362 was FP8
        assert get_peak_flops("NVIDIA L40S") == 183e12

    def test_unknown_returns_none(self):
        """Unknown devices return None — and emit one warning."""
        # Use a unique name so the warn-once cache doesn't suppress
        # the warning we want to assert on.
        with pytest.warns(UserWarning, match="MFU tracking disabled"):
            assert get_peak_flops("Unknown GPU 9000-suite-test") is None

    def test_cpu_returns_none(self):
        """CPU returns None (no meaningful peak FLOPS)."""
        assert get_peak_flops("cpu") is None

    def test_auto_detect(self):
        """Auto-detect returns a positive number or None on CPU."""
        result = get_peak_flops()
        assert result is None or result > 0

    def test_unknown_warns_only_once_per_device(self):
        """Repeated unknown-device lookups should not flood warnings."""
        import warnings as _warnings

        from ezpz import flops as _flops

        # Use a unique sentinel name so a previous test run hasn't
        # already added it to the cache.
        device = "FakeGPU-warn-once-test-zz12345"
        _flops._WARNED_UNKNOWN_DEVICES.discard(device)

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            get_peak_flops(device)
            get_peak_flops(device)
            get_peak_flops(device)
        unknown_warnings = [w for w in caught if device in str(w.message)]
        assert len(unknown_warnings) == 1


class TestExtractLoss:
    """Tests for ``_extract_loss`` — handles HF / dict / tuple outputs."""

    def test_tensor_passthrough(self):
        x = torch.randn(2, 3)
        loss = _extract_loss(x)
        assert loss is not None and loss.shape == ()

    def test_hf_dataclass_with_loss(self):
        class FakeOutput:
            loss = torch.tensor(0.5, requires_grad=True)
            logits = torch.randn(2, 3, requires_grad=True)
        loss = _extract_loss(FakeOutput())
        assert loss is FakeOutput.loss

    def test_hf_dataclass_logits_only(self):
        class FakeOutput:
            loss = None
            logits = torch.randn(2, 3, requires_grad=True)
        loss = _extract_loss(FakeOutput())
        assert loss is not None and loss.shape == ()

    def test_dict_with_logits(self):
        out = {"logits": torch.randn(2, 3, requires_grad=True)}
        loss = _extract_loss(out)
        assert loss is not None and loss.shape == ()

    def test_dict_with_arbitrary_key(self):
        out = {"prediction": torch.randn(2, 3, requires_grad=True)}
        loss = _extract_loss(out)
        assert loss is not None

    def test_tuple_first_tensor(self):
        loss = _extract_loss((torch.randn(2, 3, requires_grad=True), "ignored"))
        assert loss is not None

    def test_no_tensor_returns_none(self):
        assert _extract_loss({"meta": "no tensors here"}) is None
        assert _extract_loss(None) is None
        assert _extract_loss("string") is None


class TestEstimateModelFlops:
    """Tests for ``estimate_model_flops``."""

    def test_linear_model(self):
        """A simple linear model should have countable FLOPS."""
        model = torch.nn.Linear(128, 64)
        flops = estimate_model_flops(model, input_shape=(1, 128), device="cpu")
        assert flops > 0

    def test_with_backward(self):
        """fwd+bwd should be measurably larger than fwd-only.

        The textbook ratio is ~3x (backward ≈ 2x forward) but pure
        Linear with FlopCounterMode counts a different mix of matmuls,
        so the practical floor is ~2x.  We just assert backward
        contributes meaningful additional FLOPS.
        """
        model = torch.nn.Linear(128, 64)
        fwd_only = estimate_model_flops(
            model, input_shape=(1, 128), device="cpu", backward=False
        )
        fwd_bwd = estimate_model_flops(
            model, input_shape=(1, 128), device="cpu", backward=True
        )
        ratio = fwd_bwd / fwd_only
        assert 1.8 <= ratio <= 3.5, f"fwd+bwd / fwd ratio = {ratio:.2f}"

    def test_cnn(self):
        """CNN model should have countable FLOPS."""
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(8 * 26 * 26, 10),
        )
        flops = estimate_model_flops(
            model, input_shape=(1, 1, 28, 28), device="cpu"
        )
        assert flops > 0

    def test_embedding_model(self):
        """Models with embeddings should use randint (not randn)."""
        model = torch.nn.Sequential(
            torch.nn.Embedding(1000, 64),
            torch.nn.Linear(64, 10),
        )
        flops = estimate_model_flops(
            model, input_shape=(4, 16), device="cpu"
        )
        assert flops > 0

    def test_training_mode_restored(self):
        """estimate_model_flops must not flip the caller's train/eval mode."""
        model = torch.nn.Linear(8, 4)
        model.train()
        estimate_model_flops(model, input_shape=(1, 8), device="cpu")
        assert model.training

        model.eval()
        estimate_model_flops(model, input_shape=(1, 8), device="cpu")
        assert not model.training

    def test_parameter_fallback_when_counter_returns_zero(self, monkeypatch):
        """When FlopCounterMode returns 0 (XPU path), fall back to 6*P*B*T."""
        from ezpz import flops as _flops

        class _ZeroCounter:
            def __init__(self, *_a, **_kw): pass
            def __enter__(self): return self
            def __exit__(self, *_a): pass
            def get_total_flops(self): return 0

        monkeypatch.setattr(_flops, "FlopCounterMode", _ZeroCounter)
        model = torch.nn.Linear(16, 8)  # 16*8 + 8 = 136 params
        flops = _flops.estimate_model_flops(
            model, input_shape=(2, 16), device="cpu", backward=True,
        )
        # 6 * params * batch * tokens = 6 * 136 * 2 * 16
        assert flops == 6 * 136 * 2 * 16

    def test_no_zero_grad_when_forward_fails(self, monkeypatch):
        """If forward raises before backward, caller grads must survive.

        The original code unconditionally called ``model.zero_grad()`` in
        the cleanup block, which clobbered caller state even when no
        gradient was ever produced by ``estimate_model_flops``.  The fix
        only zeros grads when backward actually ran.
        """
        from ezpz import flops as _flops

        class _RaisingCounter:
            def __init__(self, *_a, **_kw): pass
            def __enter__(self): raise RuntimeError("simulated forward fail")
            def __exit__(self, *_a): pass
            def get_total_flops(self): return 0

        monkeypatch.setattr(_flops, "FlopCounterMode", _RaisingCounter)
        model = torch.nn.Linear(8, 4)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        # The exception is caught internally; estimation falls back to
        # parameter-based estimate without ever running our backward.
        _flops.estimate_model_flops(
            model, input_shape=(1, 8), device="cpu", backward=True,
        )
        for p in model.parameters():
            assert p.grad is not None
            assert torch.all(p.grad == 1)


class TestTryEstimate:
    """Tests for the ``try_estimate`` wrapper."""

    def test_returns_int_on_success(self):
        model = torch.nn.Linear(8, 4)
        flops = try_estimate(model, (1, 8), device="cpu")
        assert isinstance(flops, int)
        assert flops > 0

    def test_returns_zero_on_exception(self, monkeypatch):
        from ezpz import flops as _flops

        def _raise(*_a, **_kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(_flops, "estimate_model_flops", _raise)
        assert _flops.try_estimate(torch.nn.Linear(2, 2), (1, 2)) == 0

    def test_passes_through_kwargs(self):
        model = torch.nn.Linear(8, 4)
        # backward=False should change the answer
        fwd_only = try_estimate(model, (1, 8), device="cpu", backward=False)
        fwd_bwd = try_estimate(model, (1, 8), device="cpu", backward=True)
        assert fwd_bwd > fwd_only


class TestComputeMfu:
    """Tests for ``compute_mfu``."""

    def test_basic_calculation(self):
        """MFU = model_flops / (peak × dt) × 100."""
        mfu = compute_mfu(
            model_flops=100e12,
            step_duration=1.0,
            peak_flops=200e12,
        )
        assert abs(mfu - 50.0) < 0.01

    def test_per_device(self):
        """MFU is per-device — world_size is irrelevant."""
        mfu = compute_mfu(
            model_flops=100e12,
            step_duration=1.0,
            peak_flops=100e12,
        )
        assert abs(mfu - 100.0) < 0.01

    def test_zero_duration(self):
        """Zero duration returns 0 (not infinity)."""
        assert compute_mfu(100e12, 0.0, peak_flops=200e12) == 0.0

    def test_zero_flops(self):
        """Zero model FLOPS returns 0."""
        assert compute_mfu(0, 1.0, peak_flops=200e12) == 0.0

    def test_negative_inputs_return_zero(self):
        assert compute_mfu(-1, 1.0, peak_flops=200e12) == 0.0
        assert compute_mfu(1.0, -1.0, peak_flops=200e12) == 0.0
        assert compute_mfu(1.0, 1.0, peak_flops=-1.0) == 0.0

    def test_unknown_kwargs_rejected(self):
        """Misspelled kwargs (e.g. peek_flops, world_size) raise TypeError."""
        with pytest.raises(TypeError):
            compute_mfu(  # type: ignore[call-arg]
                100e12, 1.0, peak_flops=200e12, world_size=4,
            )
        with pytest.raises(TypeError):
            compute_mfu(  # type: ignore[call-arg]
                100e12, 1.0, peek_flops=200e12,
            )
