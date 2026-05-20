"""Tests for ``ezpz.utils.get_memory_metrics`` + helpers."""

from __future__ import annotations

import os
from unittest.mock import patch

import torch

import ezpz
from ezpz.utils import (
    get_current_memory_allocated,
    get_current_memory_reserved,
    get_max_memory_allocated,
    get_max_memory_reserved,
    get_memory_metrics,
    reset_peak_memory_stats,
)


class TestUnsupportedDevices:
    """CPU / MPS / unsupported devices should return 0.0 (helpers) or {} (top-level)."""

    def test_cpu_helpers_return_zero(self):
        """Each helper returns 0.0 on CPU instead of raising."""
        cpu = torch.device("cpu")
        # Pre-fix, these raised RuntimeError on CPU.
        assert get_max_memory_allocated(cpu) == 0.0
        assert get_max_memory_reserved(cpu) == 0.0
        assert get_current_memory_allocated(cpu) == 0.0
        assert get_current_memory_reserved(cpu) == 0.0

    def test_reset_peak_is_noop_on_cpu(self):
        """reset_peak_memory_stats() should not raise on CPU."""
        reset_peak_memory_stats(torch.device("cpu"))  # no exception = pass

    def test_get_memory_metrics_returns_empty_on_cpu(self):
        """The top-level helper returns {} on CPU — no keys to mislead trackers."""
        assert get_memory_metrics(torch.device("cpu")) == {}

    def test_get_memory_metrics_returns_empty_on_mps(self):
        """Same for MPS (Apple Silicon)."""
        assert get_memory_metrics("mps") == {}

    def test_get_memory_metrics_accepts_string_device(self):
        """String form like 'cpu' should also short-circuit cleanly."""
        assert get_memory_metrics("cpu") == {}


class TestEnvVarOptOut:
    """EZPZ_TRACK_MEMORY=0 disables the feature even on supported devices."""

    def test_env_var_zero_returns_empty(self):
        with patch.dict(os.environ, {"EZPZ_TRACK_MEMORY": "0"}):
            # Even with a device that would normally work, no keys.
            assert get_memory_metrics("cuda:0") == {}

    def test_env_var_unset_or_one_returns_metrics(self):
        """Default (no env var) or '1' should NOT short-circuit."""
        with patch.dict(os.environ, {"EZPZ_TRACK_MEMORY": "1"}):
            # CPU still returns {} for a different reason (unsupported device);
            # the point is that the env-var check didn't fire.
            result = get_memory_metrics("cpu")
            assert result == {}  # empty for device-reason, not env-reason


class TestKeyShape:
    """When supported, get_memory_metrics returns the documented 4 keys."""

    def test_mock_cuda_returns_four_keys(self, monkeypatch):
        """Patch CUDA availability + memory calls; verify dict shape & units."""
        # Make is_available return True so the helpers take the cuda branch.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        # Return specific byte counts so we can verify unit conversion.
        _GIB = 1024 ** 3
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: 1 * _GIB)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d: 2 * _GIB)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 3 * _GIB)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d: 4 * _GIB)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d: None)

        m = get_memory_metrics("cuda:0", reset_peak=True)
        # Documented schema: 4 keys, all in GiB, no `cuda:0`-specific noise.
        assert set(m.keys()) == {
            "mem_alloc", "mem_peak_alloc", "mem_reserved", "mem_peak_reserved",
        }
        assert m["mem_alloc"] == 1.0
        assert m["mem_peak_alloc"] == 2.0
        assert m["mem_reserved"] == 3.0
        assert m["mem_peak_reserved"] == 4.0

    def test_prefix_is_applied_to_every_key(self, monkeypatch):
        """prefix='train/' should namespace all four keys."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: 0)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d: 0)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 0)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d: 0)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d: None)

        m = get_memory_metrics("cuda:0", prefix="train/")
        assert set(m.keys()) == {
            "train/mem_alloc", "train/mem_peak_alloc",
            "train/mem_reserved", "train/mem_peak_reserved",
        }

    def test_reset_peak_default_true_calls_reset(self, monkeypatch):
        """reset_peak=True (default) calls reset_peak_memory_stats after read."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: 0)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d: 0)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 0)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d: 0)
        called = []
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d: called.append(d))

        get_memory_metrics("cuda:0")  # default reset_peak=True
        assert len(called) == 1, "reset_peak_memory_stats should be called once"

    def test_reset_peak_false_skips_reset(self, monkeypatch):
        """reset_peak=False leaves the counters alone."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: 0)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d: 0)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 0)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d: 0)
        called = []
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d: called.append(d))

        get_memory_metrics("cuda:0", reset_peak=False)
        assert called == []


class TestTopLevelReexport:
    """ezpz.get_memory_metrics should resolve via the lazy __getattr__."""

    def test_attribute_accessible(self):
        assert callable(ezpz.get_memory_metrics)

    def test_attribute_is_same_function(self):
        from ezpz.utils import get_memory_metrics as direct
        assert ezpz.get_memory_metrics is direct
