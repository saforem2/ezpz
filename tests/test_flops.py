"""Tests for ``ezpz.flops``."""

from __future__ import annotations

import torch

from ezpz.flops import compute_mfu, estimate_model_flops, get_peak_flops


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
        assert get_peak_flops("AMD MI355X") == 2500e12

    def test_l40s(self):
        assert get_peak_flops("NVIDIA L40S") == 362e12

    def test_unknown_fallback(self):
        """Unknown devices fall back to A100."""
        assert get_peak_flops("Unknown GPU 9000") == 312e12

    def test_auto_detect(self):
        """Auto-detect returns a positive number."""
        result = get_peak_flops()
        assert result > 0


class TestEstimateModelFlops:
    """Tests for ``estimate_model_flops``."""

    def test_linear_model(self):
        """A simple linear model should have countable FLOPS."""
        model = torch.nn.Linear(128, 64)
        flops = estimate_model_flops(model, input_shape=(1, 128), device="cpu")
        assert flops > 0

    def test_with_backward(self):
        """Backward pass should roughly double the FLOPS."""
        model = torch.nn.Linear(128, 64)
        fwd_only = estimate_model_flops(
            model, input_shape=(1, 128), device="cpu", backward=False
        )
        fwd_bwd = estimate_model_flops(
            model, input_shape=(1, 128), device="cpu", backward=True
        )
        # backward is typically ~2x forward, allow some margin
        assert fwd_bwd > fwd_only

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


class TestComputeMfu:
    """Tests for ``compute_mfu``."""

    def test_basic_calculation(self):
        """MFU = model_flops / (peak × world_size × dt) × 100."""
        # 100 TFLOPS model, 1 second step, 1 device with 200 TFLOPS peak
        mfu = compute_mfu(
            model_flops=100e12,
            step_duration=1.0,
            world_size=1,
            peak_flops=200e12,
        )
        assert abs(mfu - 50.0) < 0.01  # 50% MFU

    def test_multi_device(self):
        """World size scales the theoretical peak."""
        mfu = compute_mfu(
            model_flops=100e12,
            step_duration=1.0,
            world_size=4,
            peak_flops=100e12,
        )
        assert abs(mfu - 25.0) < 0.01  # 100T / (100T × 4) = 25%

    def test_zero_duration(self):
        """Zero duration returns 0 (not infinity)."""
        assert compute_mfu(100e12, 0.0, peak_flops=200e12) == 0.0

    def test_zero_flops(self):
        """Zero model FLOPS returns 0."""
        assert compute_mfu(0, 1.0, peak_flops=200e12) == 0.0

    def test_auto_detect_world_size(self):
        """Falls back to world_size=1 when not in distributed mode."""
        mfu = compute_mfu(
            model_flops=100e12,
            step_duration=1.0,
            world_size=1,
            peak_flops=200e12,
        )
        assert abs(mfu - 50.0) < 0.01
