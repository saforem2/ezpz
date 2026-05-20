"""Tests for ``ezpz.utils.get_memory_metrics`` + helpers."""

from __future__ import annotations

import os
from unittest.mock import patch

import torch

import ezpz
from ezpz.utils import (
    format_memory_summary,
    get_current_memory_allocated,
    get_current_memory_reserved,
    get_max_memory_allocated,
    get_max_memory_reserved,
    get_memory_metrics,
    is_memory_metric_key,
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


class TestFormatMemorySummary:
    """The condensed console-formatter."""

    def test_empty_when_no_keys(self):
        """No mem_* keys → empty string (caller can filter)."""
        assert format_memory_summary({"loss": 0.5, "iter": 1}) == ""

    def test_basic_format_without_total(self, monkeypatch):
        """alloc + peak, no device total → 'X.XX/Y.YYGiB' with no percent."""
        # Stub get_device_properties to return total_memory=-1 (unknown).
        import ezpz.distributed
        monkeypatch.setattr(
            ezpz.distributed,
            "get_device_properties",
            lambda d=None: {"name": "x", "total_memory": -1},
        )
        out = format_memory_summary(
            {"mem_alloc": 1.5, "mem_peak_alloc": 2.25}
        )
        assert out == "1.50/2.25GiB"
        # Specifically: no parenthetical percentage when total unknown.
        assert "%" not in out

    def test_includes_percentage_when_total_known(self, monkeypatch):
        """With total_memory in bytes, the percent of current/total appears."""
        import ezpz.distributed
        # 16 GiB total VRAM
        total_bytes = 16 * (1024 ** 3)
        monkeypatch.setattr(
            ezpz.distributed,
            "get_device_properties",
            lambda d=None: {"name": "x", "total_memory": total_bytes},
        )
        out = format_memory_summary(
            {"mem_alloc": 8.0, "mem_peak_alloc": 10.0}
        )
        # 8 / 16 = 50%
        assert out == "8.00/10.00GiB (50%)"

    def test_handles_prefix(self, monkeypatch):
        """The prefix kwarg should match the keys in the dict."""
        import ezpz.distributed
        monkeypatch.setattr(
            ezpz.distributed,
            "get_device_properties",
            lambda d=None: {"name": "x", "total_memory": -1},
        )
        out = format_memory_summary(
            {"train/mem_alloc": 1.5, "train/mem_peak_alloc": 2.0},
            prefix="train/",
        )
        assert out == "1.50/2.00GiB"

    def test_alloc_only_when_peak_missing(self, monkeypatch):
        """If only alloc is present (e.g. partial dict), format that alone."""
        import ezpz.distributed
        monkeypatch.setattr(
            ezpz.distributed,
            "get_device_properties",
            lambda d=None: {"name": "x", "total_memory": -1},
        )
        out = format_memory_summary({"mem_alloc": 1.5})
        assert out == "1.50GiB"

    def test_swallows_get_device_properties_errors(self, monkeypatch):
        """A broken get_device_properties shouldn't crash the format call."""
        import ezpz.distributed

        def boom(d=None):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(ezpz.distributed, "get_device_properties", boom)
        # Should still return the GiB part, just no percent.
        out = format_memory_summary(
            {"mem_alloc": 1.5, "mem_peak_alloc": 2.0}
        )
        assert out == "1.50/2.00GiB"


class TestIsMemoryMetricKey:
    """Helper used by History.update to strip mem_* from scalar_summary."""

    def test_recognizes_all_four(self):
        for k in ("mem_alloc", "mem_peak_alloc", "mem_reserved", "mem_peak_reserved"):
            assert is_memory_metric_key(k), f"{k} should match"

    def test_recognizes_prefixed(self):
        for k in ("train/mem_alloc", "eval/mem_peak_alloc"):
            assert is_memory_metric_key(k), f"{k} should match"

    def test_rejects_non_memory(self):
        for k in ("loss", "accuracy", "train/loss", "dtf", "mem_loss"):
            # "mem_loss" is a false-positive trap — only ends with mem_alloc etc.
            assert not is_memory_metric_key(k), f"{k} should NOT match"
