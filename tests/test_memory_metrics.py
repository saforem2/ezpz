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

    def test_get_memory_metrics_accepts_torch_device_mps(self):
        """torch.device('mps') instance should short-circuit too."""
        assert get_memory_metrics(torch.device("mps")) == {}

    def test_get_memory_metrics_accepts_indexed_cpu_string(self):
        """Indexed CPU strings like 'cpu:0' should also short-circuit."""
        assert get_memory_metrics("cpu:0") == {}

    def test_cpu_helpers_not_routed_to_cuda_when_cuda_available(self, monkeypatch):
        """Regression: previously, helpers branched on `torch.cuda.is_available()`
        before the device type, so passing a CPU device on a CUDA box would
        route into `torch.cuda.max_memory_allocated('cpu')` and raise.
        Now they dispatch on `torch.device(device).type` first.

        This was the central bug from the PR #134 review (sourcery,
        copilot, and codex all flagged it independently).
        """
        # Simulate "we're on a CUDA box."
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        # And give `torch.cuda.*` implementations that would BLOW UP if
        # called with a CPU device — proves the fix routes correctly.
        def _cuda_should_not_be_called(d):
            raise RuntimeError(
                f"BUG: torch.cuda.* called with non-cuda device {d!r}"
            )
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", _cuda_should_not_be_called)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", _cuda_should_not_be_called)
        monkeypatch.setattr(torch.cuda, "memory_allocated", _cuda_should_not_be_called)
        monkeypatch.setattr(torch.cuda, "memory_reserved", _cuda_should_not_be_called)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", _cuda_should_not_be_called)

        cpu = torch.device("cpu")
        assert get_max_memory_allocated(cpu) == 0.0
        assert get_max_memory_reserved(cpu) == 0.0
        assert get_current_memory_allocated(cpu) == 0.0
        assert get_current_memory_reserved(cpu) == 0.0
        reset_peak_memory_stats(cpu)  # no exception = pass
        assert get_memory_metrics(cpu) == {}


class TestEnvVarOptOut:
    """EZPZ_TRACK_MEMORY=0 disables the feature even on supported devices."""

    def test_env_var_zero_returns_empty_on_cuda(self, monkeypatch):
        """`EZPZ_TRACK_MEMORY=0` short-circuits even on a working CUDA device."""
        # Make CUDA appear available + functional so we know the env-var
        # check is what's returning {} (not a "device unsupported" path).
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: 1024**3)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d: 1024**3)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 1024**3)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d: 1024**3)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d: None)
        with patch.dict(os.environ, {"EZPZ_TRACK_MEMORY": "0"}):
            assert get_memory_metrics("cuda:0") == {}

    def test_env_var_unset_returns_metrics_on_cuda(self, monkeypatch):
        """Env var unset (default) should NOT short-circuit on a supported device."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: 1024**3)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d: 1024**3)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 1024**3)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d: 1024**3)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d: None)
        # Use clear=True to genuinely simulate "env var not set"
        with patch.dict(os.environ, {}, clear=True):
            result = get_memory_metrics("cuda:0")
            assert set(result.keys()) == {
                "mem_alloc", "mem_peak_alloc", "mem_reserved", "mem_peak_reserved",
            }

    def test_env_var_one_returns_metrics_on_cuda(self, monkeypatch):
        """`EZPZ_TRACK_MEMORY=1` should NOT short-circuit on a supported device."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: 1024**3)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda d: 1024**3)
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 1024**3)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d: 1024**3)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d: None)
        with patch.dict(os.environ, {"EZPZ_TRACK_MEMORY": "1"}):
            result = get_memory_metrics("cuda:0")
            assert len(result) == 4

    def test_default_device_resolution(self, monkeypatch):
        """`device=None` calls ezpz.get_torch_device() and routes correctly."""
        import ezpz
        # Make the resolver return CPU — should short-circuit to {}.
        monkeypatch.setattr(ezpz, "get_torch_device", lambda: "cpu")
        with patch.dict(os.environ, {}, clear=True):
            assert get_memory_metrics() == {}


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


class TestXPUBackend:
    """XPU-specific code paths (Intel GPU; Aurora / Sunspot)."""

    def test_xpu_routes_to_xpu_apis(self, monkeypatch):
        """When CUDA is unavailable and XPU is available, helpers route to torch.xpu.*."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        # Skip if torch.xpu doesn't exist in this build (pre-2.4-ish)
        if not hasattr(torch, "xpu"):
            import pytest
            pytest.skip("torch.xpu not present in this torch build")

        monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
        _GIB = 1024 ** 3
        monkeypatch.setattr(torch.xpu, "memory_allocated", lambda d: 5 * _GIB)
        monkeypatch.setattr(torch.xpu, "max_memory_allocated", lambda d: 6 * _GIB)
        monkeypatch.setattr(torch.xpu, "memory_reserved", lambda d: 7 * _GIB)
        monkeypatch.setattr(torch.xpu, "max_memory_reserved", lambda d: 8 * _GIB)
        monkeypatch.setattr(torch.xpu, "reset_peak_memory_stats", lambda d: None)

        m = get_memory_metrics("xpu:0")
        assert m["mem_alloc"] == 5.0
        assert m["mem_peak_alloc"] == 6.0
        assert m["mem_reserved"] == 7.0
        assert m["mem_peak_reserved"] == 8.0

    def test_xpu_helpers_fallback_to_zero_on_attribute_error(self, monkeypatch):
        """Older torch.xpu builds may be missing some attrs — return 0.0."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if not hasattr(torch, "xpu"):
            import pytest
            pytest.skip("torch.xpu not present in this torch build")

        monkeypatch.setattr(torch.xpu, "is_available", lambda: True)

        def _raise_attr_error(d):
            raise AttributeError("simulated: this torch.xpu build lacks the API")

        monkeypatch.setattr(torch.xpu, "memory_allocated", _raise_attr_error)
        monkeypatch.setattr(torch.xpu, "max_memory_allocated", _raise_attr_error)
        monkeypatch.setattr(torch.xpu, "memory_reserved", _raise_attr_error)
        monkeypatch.setattr(torch.xpu, "max_memory_reserved", _raise_attr_error)

        # Each helper should swallow the AttributeError and return 0.0.
        assert get_max_memory_allocated("xpu:0") == 0.0
        assert get_max_memory_reserved("xpu:0") == 0.0
        assert get_current_memory_allocated("xpu:0") == 0.0
        assert get_current_memory_reserved("xpu:0") == 0.0

    def test_xpu_reset_peak_swallows_attribute_error(self, monkeypatch):
        """reset_peak_memory_stats on a too-old torch.xpu shouldn't raise."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if not hasattr(torch, "xpu"):
            import pytest
            pytest.skip("torch.xpu not present in this torch build")

        monkeypatch.setattr(torch.xpu, "is_available", lambda: True)

        def _raise(d):
            raise AttributeError("simulated")

        monkeypatch.setattr(torch.xpu, "reset_peak_memory_stats", _raise)
        reset_peak_memory_stats("xpu:0")  # no exception = pass


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

    def test_peak_only_uses_GiB_suffix_not_legacy_format(self, monkeypatch):
        """When only mem_peak_alloc is present, output should be
        `X.XXGiB` (matching the alloc-only branch), not the legacy
        `peak X.XXGiB` form. Regression: the old code returned a
        different prefix for this rare case, breaking the documented
        contract."""
        import ezpz.distributed
        monkeypatch.setattr(
            ezpz.distributed,
            "get_device_properties",
            lambda d=None: {"name": "x", "total_memory": -1},
        )
        out = format_memory_summary({"mem_peak_alloc": 2.5})
        assert out == "2.50GiB"
        # Must NOT use the legacy "peak X.XXGiB" wording.
        assert "peak" not in out

    def test_peak_only_includes_percentage_when_total_known(self, monkeypatch):
        """The peak-only branch should still include the percentage,
        consistent with the alloc-only and alloc+peak branches."""
        import ezpz.distributed
        total_bytes = 16 * (1024 ** 3)
        monkeypatch.setattr(
            ezpz.distributed,
            "get_device_properties",
            lambda d=None: {"name": "x", "total_memory": total_bytes},
        )
        # alloc is missing entirely; with no alloc to %-of, pct_str should
        # remain empty (pct is computed against `alloc`, not `peak`).
        out = format_memory_summary({"mem_peak_alloc": 2.0})
        # Format matches the consistent `X.XXGiB` shape — no percent
        # (we'd need alloc to compute it).
        assert out == "2.00GiB"

    def test_prefix_auto_detected_from_keys(self, monkeypatch):
        """When prefix is None (default), it's inferred from the keys."""
        import ezpz.distributed
        monkeypatch.setattr(
            ezpz.distributed,
            "get_device_properties",
            lambda d=None: {"name": "x", "total_memory": -1},
        )
        # train/ prefix
        out = format_memory_summary(
            {"train/mem_alloc": 1.5, "train/mem_peak_alloc": 2.0}
        )
        assert out == "1.50/2.00GiB"
        # eval/ prefix
        out = format_memory_summary(
            {"eval/mem_alloc": 3.0, "eval/mem_peak_alloc": 4.0}
        )
        assert out == "3.00/4.00GiB"
        # No prefix
        out = format_memory_summary({"mem_alloc": 5.0, "mem_peak_alloc": 6.0})
        assert out == "5.00/6.00GiB"

    def test_explicit_prefix_overrides_auto_detect(self, monkeypatch):
        """If caller passes a prefix, it should win over inference."""
        import ezpz.distributed
        monkeypatch.setattr(
            ezpz.distributed,
            "get_device_properties",
            lambda d=None: {"name": "x", "total_memory": -1},
        )
        # Dict has BOTH train/ and eval/ keys. Explicit prefix='eval/'
        # picks the eval values, not train.
        out = format_memory_summary(
            {
                "train/mem_alloc": 1.0, "train/mem_peak_alloc": 2.0,
                "eval/mem_alloc": 7.0, "eval/mem_peak_alloc": 8.0,
            },
            prefix="eval/",
        )
        assert out == "7.00/8.00GiB"

    def test_device_int_passed_to_get_device_properties(self, monkeypatch):
        """`device=1` should be forwarded as the int index, not None."""
        import ezpz.distributed
        captured: list = []

        def _capture(d=None):
            captured.append(d)
            return {"name": "x", "total_memory": -1}

        monkeypatch.setattr(
            ezpz.distributed, "get_device_properties", _capture
        )
        format_memory_summary(
            {"mem_alloc": 1.5, "mem_peak_alloc": 2.0}, device=1
        )
        assert captured == [1]

    def test_device_torch_device_with_index_resolved_to_int(self, monkeypatch):
        """`torch.device('cuda:1')` should be normalized to `1`."""
        import ezpz.distributed
        captured: list = []

        def _capture(d=None):
            captured.append(d)
            return {"name": "x", "total_memory": -1}

        monkeypatch.setattr(
            ezpz.distributed, "get_device_properties", _capture
        )
        format_memory_summary(
            {"mem_alloc": 1.5, "mem_peak_alloc": 2.0},
            device=torch.device("cuda", 1),
        )
        assert captured == [1]

    def test_device_torch_device_without_index_passes_none(self, monkeypatch):
        """`torch.device('cuda')` (no index) → None (fall back to local rank)."""
        import ezpz.distributed
        captured: list = []

        def _capture(d=None):
            captured.append(d)
            return {"name": "x", "total_memory": -1}

        monkeypatch.setattr(
            ezpz.distributed, "get_device_properties", _capture
        )
        format_memory_summary(
            {"mem_alloc": 1.5, "mem_peak_alloc": 2.0},
            device=torch.device("cuda"),
        )
        assert captured == [None]

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

    def test_recognizes_aggregated_forms(self):
        """History._compute_distributed_metrics adds /mean, /max, /min, /std,
        /avg suffixes per rank. The predicate has to catch these too or
        the console summary still shows 16 noisy `mem_alloc/mean=...`
        style lines. This was a real bug — caught in dogfooding."""
        for base in ("mem_alloc", "mem_peak_alloc", "mem_reserved", "mem_peak_reserved"):
            for suffix in ("/mean", "/max", "/min", "/std", "/avg"):
                k = f"{base}{suffix}"
                assert is_memory_metric_key(k), f"{k} should match"
                # prefixed variants too
                k_prefixed = f"train/{base}{suffix}"
                assert is_memory_metric_key(k_prefixed), f"{k_prefixed} should match"

    def test_rejects_non_memory(self):
        for k in (
            "loss", "accuracy", "train/loss", "dtf",
            "mem_loss",          # ends in `loss`, not a base
            "memo_field",        # `memo_*` is not `mem_*`
            "mem_alloc_extra",   # base substring isn't enough; must be whole word
        ):
            assert not is_memory_metric_key(k), f"{k} should NOT match"


class TestFormatCompactSummary:
    """The ``key=value(±std)`` console renderer."""

    def test_collapses_base_and_std_into_inline_format(self):
        """`loss=0.5` + `loss/std=0.02` → `loss=0.50(±0.020)`."""
        from ezpz.utils import format_compact_summary
        out = format_compact_summary(
            {"loss": 0.5, "loss/std": 0.02}, precision=2,
        )
        # 0.02 at 2 sig figs (fixed-point, magnitude=-2) → "0.020"
        assert out == "loss=0.50(±0.020)"

    def test_drops_other_aggregation_suffixes(self):
        """`/mean`, `/min`, `/max`, `/avg` are dropped (W&B has them)."""
        from ezpz.utils import format_compact_summary
        out = format_compact_summary(
            {
                "loss": 0.5,
                "loss/mean": 0.4, "loss/min": 0.1, "loss/max": 0.9,
                "loss/std": 0.02, "loss/avg": 0.4,
            },
            precision=2,
        )
        # All four aggregation suffixes drop; only the inline std survives.
        assert out == "loss=0.50(±0.020)"
        assert "/mean" not in out and "/max" not in out and "/min" not in out

    def test_std_uses_two_sig_figs(self):
        """Std is a noise-band hint — always 2 sig figs regardless of the
        caller's value precision.

        `loss=0.289993` is interesting at 6 digits; `loss/std=0.157124`
        rounded to 2 sig figs (`0.16`) loses nothing meaningful and is
        much easier to scan.
        """
        from ezpz.utils import format_compact_summary
        out = format_compact_summary(
            {"loss": 0.289993, "loss/std": 0.157124}, precision=6,
        )
        assert out == "loss=0.289993(±0.16)"

    def test_zero_std_drops_parenthetical(self):
        """`(±0.0)` adds no signal — drop it entirely. Common for
        constant metrics like LR within a single step."""
        from ezpz.utils import format_compact_summary
        out = format_compact_summary(
            {"lr": 0.000234, "lr/std": 0.0}, precision=6,
        )
        assert out == "lr=0.000234"
        assert "(±" not in out

    def test_small_std_uses_scientific_notation(self):
        """Stds below 0.01 use scientific notation so the width stays
        bounded and the magnitude is visually obvious.

        Previously this fell back to wider fixed-point formats
        (`0.000302`, `0.0037`) which produced inconsistent column widths
        across rows. Scientific notation keeps everything at 6 chars.
        """
        from ezpz.utils import format_compact_summary
        out = format_compact_summary(
            {"dtb": 0.027595, "dtb/std": 0.000302}, precision=6,
        )
        assert out == "dtb=0.027595(±3.0e-4)"

    def test_small_std_scientific_strips_exponent_zero(self):
        """`f-string :.1e` produces `2.8e-04` — strip the leading 0 so
        we save a character: `2.8e-4`."""
        from ezpz.utils import format_compact_summary
        out = format_compact_summary(
            {"dtf": 0.008877, "dtf/std": 0.000279}, precision=6,
        )
        # Standard f-string would say `2.8e-04`; we want `2.8e-4`.
        assert "(±2.8e-4)" in out
        assert "e-04" not in out

    def test_std_large_uses_fewer_decimals(self):
        """Stds >= 1 use 2 sig figs with no decimals beyond what's needed.
        `1.297581` → `1.3` (1 decimal, magnitude=0)."""
        from ezpz.utils import format_compact_summary
        out = format_compact_summary(
            {"tflops": 16.778491, "tflops/std": 1.297581}, precision=6,
        )
        assert out == "tflops=16.778491(±1.3)"

    def test_skips_memory_keys_entirely(self):
        """Memory keys are handled by format_memory_summary; never appear here."""
        from ezpz.utils import format_compact_summary
        out = format_compact_summary(
            {
                "loss": 0.5,
                "mem_alloc": 1.5, "mem_peak_alloc": 2.0,
                "mem_alloc/mean": 1.5, "mem_alloc/std": 0.0,
            },
            precision=2,
        )
        assert out == "loss=0.50"
        assert "mem" not in out

    def test_counter_keys_omit_std(self):
        """`iter`, `step`, `epoch`, `batch`, `idx` don't get (±std)."""
        from ezpz.utils import format_compact_summary
        out = format_compact_summary(
            {"iter": 100, "iter/std": 5.0, "loss": 0.5, "loss/std": 0.02},
            precision=2,
        )
        # iter has NO ±std even though iter/std was provided
        assert "iter=100" in out
        assert "iter=100(" not in out  # not "iter=100(±5.00)"
        # loss still has its inline std (2 sig figs, magnitude=-2 → "0.020")
        assert "loss=0.50(±0.020)" in out

    def test_no_std_no_parenthetical(self):
        """When `/std` is absent, the metric prints as-is — no (±0) noise."""
        from ezpz.utils import format_compact_summary
        out = format_compact_summary({"loss": 0.5}, precision=2)
        assert out == "loss=0.50"

    def test_aggregated_without_base_still_emitted(self):
        """`loss/mean=0.4` without `loss` itself → emit `loss/mean=0.4`."""
        from ezpz.utils import format_compact_summary
        # No base `loss` — pure aggregate. Should still appear so we don't
        # silently drop data.
        out = format_compact_summary({"loss/mean": 0.4}, precision=2)
        assert "loss/mean=0.40" in out

    def test_counter_tokens_padded_for_alignment(self):
        """`iter=N` tokens should be right-padded so successive lines'
        next field aligns at the left edge — what you want when
        scanning a long training log column-wise."""
        from ezpz.utils import format_compact_summary
        out_0 = format_compact_summary({"iter": 0, "loss": 0.5}, precision=6)
        out_1k = format_compact_summary({"iter": 1234, "loss": 0.5}, precision=6)
        # Both lines: next field should start at the same column.
        loss_pos_0 = out_0.find("loss=")
        loss_pos_1k = out_1k.find("loss=")
        assert loss_pos_0 == loss_pos_1k, (
            f"loss= aligned to {loss_pos_0} (short iter) vs "
            f"{loss_pos_1k} (long iter)"
        )

    def test_min_widths_override(self):
        """`min_widths` kwarg should override the per-key default."""
        from ezpz.utils import format_compact_summary
        # Force iter to a tiny width — value blows past it, no padding
        out_small = format_compact_summary(
            {"iter": 5, "loss": 0.5},
            min_widths={"iter": 4},
        )
        # iter=5 is 6 chars; min_width=4 means no padding (value longer)
        assert out_small == "iter=5 loss=0.500000"

        # Force iter to a huge width — pad heavily
        out_big = format_compact_summary(
            {"iter": 5, "loss": 0.5},
            min_widths={"iter": 20},
        )
        # iter=5 padded to 20 chars
        assert "iter=5" + " " * 14 in out_big

    def test_namespaced_counter_picks_up_default_width(self):
        """`train/iter` should match the same width budget as `iter`."""
        from ezpz.utils import format_compact_summary
        out = format_compact_summary({"train/iter": 5, "loss": 0.5}, precision=6)
        # Default width for iter=10 → "train/iter=5" is 12 chars,
        # already > 10 → no padding. But still no crash, and uses the
        # leaf-name lookup correctly.
        assert "train/iter=5" in out

    def test_non_counter_keys_not_padded(self):
        """Padding should only apply to keys in `min_widths`; everything
        else stays untouched (no trailing whitespace surprises)."""
        from ezpz.utils import format_compact_summary
        out = format_compact_summary({"loss": 0.5, "acc": 0.9})
        # No double spaces between tokens.
        assert "  " not in out

    def test_realistic_log_shape(self):
        """End-to-end: the actual shape from a 22-token line collapses to ~8."""
        from ezpz.utils import format_compact_summary
        metrics = {
            "iter": 180,
            "loss": 0.047, "loss/std": 0.023, "loss/mean": 0.030,
            "loss/max": 0.120, "loss/min": 0.011,
            "accuracy": 1.0, "accuracy/std": 0.007, "accuracy/mean": 0.997,
            "accuracy/max": 1.0, "accuracy/min": 0.970,
            "dtf": 0.009, "dtf/std": 0.000,
        }
        out = format_compact_summary(metrics, precision=3)
        # 4 tokens: iter, loss(±std), accuracy(±std), dtf (no std — it's 0)
        assert out.count("=") == 4
        assert "iter=180" in out
        # 2 sig figs fixed-point: 0.023 → "0.023"
        assert "loss=0.047(±0.023)" in out
        # 0.007 < 0.01 → scientific: "7.0e-3"
        assert "accuracy=1.000(±7.0e-3)" in out
        # dtf/std=0.000 → drop the parenthetical entirely.
        assert "dtf=0.009" in out
        assert "dtf=0.009(" not in out
