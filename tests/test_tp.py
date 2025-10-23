"""Tests for the ezpz.tp (tensor parallel) module."""

import pytest

try:
    import ezpz.tp as tp

    TP_AVAILABLE = True
except ImportError:
    TP_AVAILABLE = False


@pytest.mark.skipif(not TP_AVAILABLE, reason="ezpz.tp not available")
class TestTP:
    def test_tensor_parallel_functions_exist(self):
        """Test that tensor parallel functions exist."""
        # Check that key functions are available
        assert hasattr(tp, "initialize_tensor_parallel")
        assert hasattr(tp, "tensor_parallel_is_initialized")
        assert hasattr(tp, "get_tensor_parallel_group")
        assert hasattr(tp, "get_data_parallel_group")
        assert hasattr(tp, "get_pipeline_parallel_group")
        assert hasattr(tp, "destroy_tensor_parallel")

    def test_utility_functions(self):
        """Test utility functions."""
        # Check that utility functions are available
        assert hasattr(tp, "ensure_divisibility")
        assert hasattr(tp, "divide_and_check_no_remainder")
        assert hasattr(tp, "split_tensor_along_last_dim")

    def test_context_parallel_functions(self):
        """Test context parallel functions."""
        assert hasattr(tp, "get_context_parallel_group")
        assert hasattr(tp, "get_context_parallel_ranks")
        assert hasattr(tp, "get_context_parallel_world_size")
        assert hasattr(tp, "get_context_parallel_rank")

    def test_pipeline_parallel_functions(self):
        """Test pipeline parallel functions."""
        assert hasattr(tp, "get_pipeline_parallel_ranks")

    def test_ensure_divisibility(self):
        """Test ensure_divisibility function."""
        # This should not raise an exception
        tp.ensure_divisibility(10, 2)

        # This should raise an exception
        with pytest.raises(AssertionError):
            tp.ensure_divisibility(10, 3)
