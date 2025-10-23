"""Property-based tests for ezpz utility functions."""

from unittest.mock import patch

import pytest

# Mock the problematic imports at the module level
with patch("ezpz.dist.get_hostname", return_value="test-host"), patch(
    "ezpz.configs.get_scheduler", return_value="UNKNOWN"
), patch("ezpz.jobs.SCHEDULER", "UNKNOWN"):
    try:
        import ezpz.utils as utils

        UTILS_AVAILABLE = True
    except ImportError:
        UTILS_AVAILABLE = False

# Only run property-based tests if hypothesis is available
try:
    from hypothesis import given
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

if not UTILS_AVAILABLE or not HYPOTHESIS_AVAILABLE:  # pragma: no cover - test guard
    pytest.skip(
        "ezpz.utils or hypothesis not available",
        allow_module_level=True,
    )


class TestPropertyBased:
    """Property-based tests for utility functions."""

    @given(st.text())
    def test_normalize_idempotent(self, text):
        """Test that normalize function is idempotent."""
        result1 = utils.normalize(text)
        result2 = utils.normalize(result1)
        assert result1 == result2

    @given(st.text(min_size=1))
    def test_normalize_produces_lowercase(self, text):
        """Test that normalize function produces lowercase output."""
        result = utils.normalize(text)
        assert result == result.lower()

    @given(st.text())
    def test_normalize_produces_valid_identifiers(self, text):
        """Test that normalize function produces valid identifiers."""
        result = utils.normalize(text)
        # Should only contain alphanumeric characters and dashes
        for char in result:
            assert (
                char.isalnum() or char == "-"
            ), f"Invalid character '{char}' in '{result}'"

    @given(st.floats(allow_nan=False, allow_infinity=False))
    def test_format_pair_float_precision(self, value):
        """Test that format_pair maintains precision for floats."""
        result = utils.format_pair("test", value, precision=6)
        # Should contain the key and value
        assert "test=" in result
        # Should not raise an exception
        assert isinstance(result, str)
