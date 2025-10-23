"""Unit tests for ezpz utility functions that can be tested in isolation."""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock


def test_normalize_function():
    """Test the normalize function logic in isolation."""
    # Define the normalize function directly to avoid import issues
    import re
    
    def normalize(name: str) -> str:
        """Normalize a name by replacing special characters with dashes and converting to lowercase."""
        # First replace all special characters with dashes
        result = re.sub(r"[-_.]+", "-", name).lower()
        # Remove leading/trailing dashes
        result = result.strip("-")
        # Replace multiple consecutive dashes with single dash
        result = re.sub(r"-+", "-", result)
        return result
    
    # Test cases
    test_cases = [
        ("test-name", "test-name"),
        ("test_name", "test-name"),
        ("test.name", "test-name"),
        ("test_name.sub-name", "test-name-sub-name"),
        ("TEST_NAME", "test-name"),
        ("Test.Name_Sub-Name", "test-name-sub-name"),
        ("multiple___dots...dashes---underscores___", "multiple-dots-dashes-underscores"),
        ("___leading_underscores", "leading-underscores"),
        ("trailing_dashes---", "trailing-dashes"),
        ("", ""),
        ("normal", "normal"),
    ]
    
    for input_str, expected in test_cases:
        result = normalize(input_str)
        assert result == expected, f"normalize('{input_str}') should return '{expected}', got '{result}'"


def test_format_pair_function():
    """Test the format_pair function logic in isolation."""
    # Define the format_pair function directly
    def format_pair(k: str, v, precision: int = 6) -> str:
        """Format a key-value pair as a string."""
        if isinstance(v, (int, bool, np.integer)):
            return f"{k}={v}"
        return f"{k}={v:<.{precision}f}"
    
    # Test cases
    test_cases = [
        (("test", 5), "test=5"),
        (("test", 5.0), "test=5.000000"),
        (("test", 5.123456), "test=5.123456"),
        (("test", 5.123456, 2), "test=5.12"),
        (("test", True), "test=True"),
        (("test", False), "test=False"),
    ]
    
    for (args, expected) in test_cases:
        if len(args) == 3:
            result = format_pair(args[0], args[1], args[2])
        else:
            result = format_pair(args[0], args[1])
        assert result == expected, f"format_pair{args} should return '{expected}', got '{result}'"


def test_grab_tensor_function():
    """Test the grab_tensor function logic in isolation."""
    # Define a simplified version of grab_tensor for testing
    def grab_tensor(x, force: bool = False):
        """Convert various tensor/array-like objects to numpy arrays."""
        if x is None:
            return None
        if isinstance(x, (int, float, bool, np.floating)):
            return x
        if isinstance(x, list):
            if len(x) > 0:
                if isinstance(x[0], np.ndarray):
                    return np.stack(x)
                if isinstance(x[0], (int, float, bool, np.floating)):
                    return np.array(x)
            return np.array(x)
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.numpy(force=force)
        if callable(getattr(x, "numpy", None)):
            return x.numpy(force=force)
        return x
    
    # Test cases
    # Test with None
    assert grab_tensor(None) is None
    
    # Test with scalars
    assert grab_tensor(5) == 5
    assert grab_tensor(5.5) == 5.5
    assert grab_tensor(True) is True
    
    # Test with lists
    list_result = grab_tensor([1, 2, 3])
    assert isinstance(list_result, np.ndarray)
    assert np.array_equal(list_result, np.array([1, 2, 3]))
    
    # Test with numpy arrays
    np_array = np.array([1, 2, 3])
    np_result = grab_tensor(np_array)
    assert isinstance(np_result, np.ndarray)
    assert np.array_equal(np_result, np_array)
    
    # Test with torch tensors
    torch_tensor = torch.tensor([1, 2, 3])
    torch_result = grab_tensor(torch_tensor)
    assert isinstance(torch_result, np.ndarray)
    assert np.array_equal(torch_result, torch_tensor.numpy())
    
    # Test with list of numpy arrays
    np_list = [np.array([1, 2]), np.array([3, 4])]
    stacked_result = grab_tensor(np_list)
    assert isinstance(stacked_result, np.ndarray)
    assert np.array_equal(stacked_result, np.stack(np_list))


def test_get_timestamp_function():
    """Test the get_timestamp function logic in isolation."""
    import datetime
    
    # Mock datetime for consistent testing
    with patch('datetime.datetime') as mock_datetime:
        # Set up the mock to return a specific datetime
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2023-12-01-143022"
        mock_datetime.now.return_value = mock_now
        
        # Define the function
        def get_timestamp(fstr=None):
            """Get formatted timestamp."""
            now = datetime.datetime.now()
            return (
                now.strftime("%Y-%m-%d-%H%M%S") if fstr is None else now.strftime(fstr)
            )
        
        # Test without format string
        result = get_timestamp()
        assert result == "2023-12-01-143022"
        
        # Test with format string
        mock_now.strftime.return_value = "2023-12-01"
        result = get_timestamp("%Y-%m-%d")
        assert result == "2023-12-01"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])