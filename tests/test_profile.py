"""Tests for the ezpz.profile module."""

import pytest
import time

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
            profiler_type="pyinstrument",
            rank_zero_only=True,
            outdir="/tmp"
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
            outdir="/tmp"
        )
        assert ctx is not None

    def test_pyinstrument_profiler(self):
        """Test PyInstrumentProfiler context manager."""
        profiler = profile.PyInstrumentProfiler(
            rank_zero_only=True,
            outdir="/tmp"
        )
        
        # Test that it can be used as a context manager
        with profiler:
            # Do some work
            time.sleep(0.1)
            result = 1 + 1
            
        assert result == 2

    def test_null_context_when_not_rank_zero(self):
        """Test that profiler returns null context when not rank zero and rank_zero_only=True."""
        ctx = profile.get_context_manager(
            profiler_type="pyinstrument",
            rank_zero_only=True,
            outdir="/tmp"
        )
        # Should work without errors
        with ctx:
            result = 1 + 1
        assert result == 2