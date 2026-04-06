"""Tests for the ezpz.utils module."""

import re
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _mock_env():
    """Prevent scheduler/hostname detection from hitting real system."""
    with (
        patch("ezpz.distributed.get_hostname", return_value="test-host"),
        patch("ezpz.configs.get_scheduler", return_value="UNKNOWN"),
        patch("ezpz.jobs.SCHEDULER", "UNKNOWN"),
    ):
        yield


class TestGetTimestamp:
    def test_default_format(self):
        import ezpz.utils as utils

        ts = utils.get_timestamp()
        # Default format: YYYY-MM-DD-HHMMSS
        assert re.match(r"\d{4}-\d{2}-\d{2}-\d{6}", ts), (
            f"Expected YYYY-MM-DD-HHMMSS, got: {ts}"
        )

    def test_custom_format(self):
        import ezpz.utils as utils

        ts = utils.get_timestamp("%Y/%m/%d")
        assert re.match(r"\d{4}/\d{2}/\d{2}$", ts), (
            f"Expected YYYY/MM/DD, got: {ts}"
        )

    def test_consecutive_calls_are_consistent(self):
        import ezpz.utils as utils

        ts1 = utils.get_timestamp("%Y-%m-%d")
        ts2 = utils.get_timestamp("%Y-%m-%d")
        assert ts1 == ts2, "Same-second calls should produce same date"


class TestNormalize:
    def test_underscores(self):
        import ezpz.utils as utils

        assert utils.normalize("test_name") == "test-name"

    def test_dots(self):
        import ezpz.utils as utils

        assert utils.normalize("test.name") == "test-name"

    def test_mixed(self):
        import ezpz.utils as utils

        assert utils.normalize("test_name.sub-name") == "test-name-sub-name"

    def test_uppercase(self):
        import ezpz.utils as utils

        assert utils.normalize("TEST_NAME") == "test-name"

    def test_already_normalized(self):
        import ezpz.utils as utils

        assert utils.normalize("test-name") == "test-name"


class TestFormatPair:
    def test_integer(self):
        import ezpz.utils as utils

        assert utils.format_pair("x", 5) == "x=5"

    def test_float_default_precision(self):
        import ezpz.utils as utils

        result = utils.format_pair("x", 5.123456)
        assert result == "x=5.123456"

    def test_float_custom_precision(self):
        import ezpz.utils as utils

        assert utils.format_pair("x", 5.123456, precision=2) == "x=5.12"

    def test_boolean(self):
        import ezpz.utils as utils

        assert utils.format_pair("flag", True) == "flag=True"
