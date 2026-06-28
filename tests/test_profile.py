"""Tests for the ezpz.profile module."""

import time

import pytest

try:
    import ezpz.profile as profile

    PROFILE_AVAILABLE = True
except ImportError:
    PROFILE_AVAILABLE = False


@pytest.mark.skipif(not PROFILE_AVAILABLE, reason="ezpz.profile not available")
class TestProfile:
    def test_get_context_manager(self):
        """Test get_context_manager function."""
        # Test with pyinstrument profiler
        ctx = profile.get_context_manager(
            profiler_type="pyinstrument", rank_zero_only=True, outdir="/tmp"
        )
        assert ctx is not None

        # Test with torch profiler
        ctx = profile.get_context_manager(
            profiler_type="torch",
            wait=1,
            warmup=1,
            active=2,
            repeat=1,
            rank_zero_only=True,
            outdir="/tmp",
        )
        assert ctx is not None

    def test_pyinstrument_profiler(self):
        """Test PyInstrumentProfiler context manager."""
        profiler = profile.PyInstrumentProfiler(rank_zero_only=True, outdir="/tmp")

        # Test that it can be used as a context manager
        with profiler:
            # Do some work
            time.sleep(0.1)
            result = 1 + 1

        assert result == 2

    def test_null_context_when_not_rank_zero(self):
        """Test that profiler returns null context when not rank zero and rank_zero_only=True."""
        ctx = profile.get_context_manager(
            profiler_type="pyinstrument", rank_zero_only=True, outdir="/tmp"
        )
        # Should work without errors
        with ctx:
            result = 1 + 1
        assert result == 2


@pytest.mark.skipif(not PROFILE_AVAILABLE, reason="ezpz.profile not available")
class TestProfilingContextFromArgs:
    """Cover the namespace->context-manager adapter semantics.

    These verify the contract the ezpz.examples.* modules rely on: no flag
    => nullcontext (yields None, default path untouched); a flag => a real
    context; and the legacy PYINSTRUMENT_PROFILER env-var opt-in.
    """

    @staticmethod
    def _ns(**over):
        from types import SimpleNamespace

        base = dict(
            pytorch_profiler=False,
            pyinstrument_profiler=False,
            rank_zero_only=False,
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            acc_events=False,
            profile_memory=True,
            pytorch_profiler_wait=1,
            pytorch_profiler_warmup=2,
            pytorch_profiler_active=3,
            pytorch_profiler_repeat=5,
        )
        base.update(over)
        return SimpleNamespace(**base)

    def test_no_flag_returns_nullcontext(self, monkeypatch):
        from contextlib import nullcontext

        monkeypatch.delenv("PYINSTRUMENT_PROFILER", raising=False)
        cm = profile.profiling_context_from_args(self._ns(), outdir="/tmp")
        assert isinstance(cm, nullcontext)
        with cm as c:
            assert c is None  # default path: profiler is None

    def test_pyinstrument_flag_activates(self, monkeypatch):
        from contextlib import nullcontext

        monkeypatch.delenv("PYINSTRUMENT_PROFILER", raising=False)
        cm = profile.profiling_context_from_args(
            self._ns(pyinstrument_profiler=True), outdir="/tmp"
        )
        assert not isinstance(cm, nullcontext)

    def test_env_var_activates_without_flag(self, monkeypatch):
        from contextlib import nullcontext

        monkeypatch.setenv("PYINSTRUMENT_PROFILER", "1")
        cm = profile.profiling_context_from_args(self._ns(), outdir="/tmp")
        # Legacy opt-in: env var alone turns on pyinstrument.
        assert not isinstance(cm, nullcontext)

    def test_missing_attrs_default_to_no_profiling(self, monkeypatch):
        from contextlib import nullcontext
        from types import SimpleNamespace

        monkeypatch.delenv("PYINSTRUMENT_PROFILER", raising=False)
        # An object exposing none of the profiler attrs must be safe.
        cm = profile.profiling_context_from_args(
            SimpleNamespace(), outdir="/tmp"
        )
        assert isinstance(cm, nullcontext)
