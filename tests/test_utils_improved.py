"""Tests for the ezpz.utils module."""

from unittest.mock import patch

import pytest


class TestUtils:
    """Test the utils module functions."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Setup mocks for all tests in this class."""
        # Mock the problematic functions at the module level
        with patch("ezpz.dist.get_hostname", return_value="test-host"), patch(
            "ezpz.configs.get_scheduler", return_value="UNKNOWN"
        ), patch("ezpz.jobs.SCHEDULER", "UNKNOWN"):
            yield

    def test_get_timestamp(self):
        """Test get_timestamp function."""
        # Import inside the test to ensure mocks are in place
        with patch("ezpz.dist.get_hostname", return_value="test-host"), patch(
            "ezpz.configs.get_scheduler", return_value="UNKNOWN"
        ), patch("ezpz.jobs.SCHEDULER", "UNKNOWN"):
            import ezpz.utils as utils

            timestamp = utils.get_timestamp()
            assert isinstance(timestamp, str)
            assert len(timestamp) > 0

            # Test with custom format
            timestamp_custom = utils.get_timestamp("%Y-%m-%d")
            assert isinstance(timestamp_custom, str)
            assert len(timestamp_custom) > 0

    def test_normalize(self):
        """Test normalize function."""
        # Import inside the test to ensure mocks are in place
        with patch("ezpz.dist.get_hostname", return_value="test-host"), patch(
            "ezpz.configs.get_scheduler", return_value="UNKNOWN"
        ), patch("ezpz.jobs.SCHEDULER", "UNKNOWN"):
            import ezpz.utils as utils

            # Test with dashes
            result = utils.normalize("test-name")
            assert result == "test-name"

            # Test with underscores
            result = utils.normalize("test_name")
            assert result == "test-name"

            # Test with dots
            result = utils.normalize("test.name")
            assert result == "test-name"

            # Test with mixed
            result = utils.normalize("test_name.sub-name")
            assert result == "test-name-sub-name"

            # Test with uppercase
            result = utils.normalize("TEST_NAME")
            assert result == "test-name"

    def test_format_pair(self):
        """Test format_pair function."""
        # Import inside the test to ensure mocks are in place
        with patch("ezpz.dist.get_hostname", return_value="test-host"), patch(
            "ezpz.configs.get_scheduler", return_value="UNKNOWN"
        ), patch("ezpz.jobs.SCHEDULER", "UNKNOWN"):
            import ezpz.utils as utils

            # Test with integer
            result = utils.format_pair("test", 5)
            assert result == "test=5"

            # Test with float
            result = utils.format_pair("test", 5.123456)
            assert result == "test=5.123456"

            # Test with custom precision
            result = utils.format_pair("test", 5.123456, precision=2)
            assert result == "test=5.12"

            # Test with boolean
            result = utils.format_pair("test", True)
            assert result == "test=True"
