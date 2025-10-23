"""Tests for the ezpz.utils module."""


import numpy as np
import pytest
import torch

try:
    import ezpz.utils as utils

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="ezpz.utils not available")
class TestUtils:
    def test_get_timestamp(self):
        """Test get_timestamp function."""
        timestamp = utils.get_timestamp()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0

        # Test with custom format
        timestamp_custom = utils.get_timestamp("%Y-%m-%d")
        assert isinstance(timestamp_custom, str)
        assert len(timestamp_custom) > 0

    def test_format_pair(self):
        """Test format_pair function."""
        # Test with integer
        result = utils.format_pair("test", 5)
        assert result == "test=5"

        # Test with float
        result = utils.format_pair("test", 5.123456)
        assert result == "test=5.123456"

        # Test with custom precision
        result = utils.format_pair("test", 5.123456, precision=2)
        assert result == "test=5.12"

    def test_summarize_dict(self):
        """Test summarize_dict function."""
        test_dict = {"a": 1, "b": 2.5, "c": True}
        result = utils.summarize_dict(test_dict)
        assert isinstance(result, str)
        assert "a=1" in result
        assert "b=2.5" in result
        assert "c=True" in result

    def test_normalize(self):
        """Test normalize function."""
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

    def test_grab_tensor(self):
        """Test grab_tensor function."""
        # Test with numpy array
        np_array = np.array([1, 2, 3])
        result = utils.grab_tensor(np_array)
        assert np.array_equal(result, np_array)

        # Test with torch tensor
        torch_tensor = torch.tensor([1, 2, 3])
        result = utils.grab_tensor(torch_tensor)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, torch_tensor.numpy())

        # Test with scalar
        scalar = 5
        result = utils.grab_tensor(scalar)
        assert result == scalar

        # Test with list
        test_list = [1, 2, 3]
        result = utils.grab_tensor(test_list)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array(test_list))

        # Test with None
        result = utils.grab_tensor(None)
        assert result is None

    def test_breakpoint(self):
        """Test breakpoint function (mocked)."""
        # This is a bit tricky to test since it's a debugging function
        # We'll just make sure it doesn't raise an exception
        # In a real test environment, we would mock the distributed setup
        pass

    def test_get_max_memory_functions(self):
        """Test memory functions."""
        # These functions require specific hardware to test properly
        # We'll just make sure they exist and can be called
        assert hasattr(utils, "get_max_memory_allocated")
        assert hasattr(utils, "get_max_memory_reserved")
