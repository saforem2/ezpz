"""Tests for the ezpz.tp (tensor parallel) module."""

from unittest.mock import MagicMock, patch

import pytest
import torch

try:
    import ezpz.tp as tp
    import ezpz.tp as tp_module  # alias for module-level global access

    TP_AVAILABLE = True
except ImportError:
    TP_AVAILABLE = False


@pytest.mark.skipif(not TP_AVAILABLE, reason="ezpz.tp not available")
class TestTP:
    def test_ensure_divisibility_valid(self):
        """ensure_divisibility should pass for evenly divisible values."""
        tp.ensure_divisibility(10, 2)
        tp.ensure_divisibility(12, 4)
        tp.ensure_divisibility(8, 1)

    def test_ensure_divisibility_invalid(self):
        """ensure_divisibility should raise for non-divisible values."""
        with pytest.raises(AssertionError):
            tp.ensure_divisibility(10, 3)
        with pytest.raises(AssertionError):
            tp.ensure_divisibility(7, 2)

    def test_divide_and_check_no_remainder(self):
        """divide_and_check_no_remainder should return the quotient."""
        assert tp.divide_and_check_no_remainder(10, 2) == 5
        assert tp.divide_and_check_no_remainder(12, 4) == 3
        with pytest.raises(AssertionError):
            tp.divide_and_check_no_remainder(10, 3)

    def test_tensor_parallel_not_initialized_by_default(self):
        """TP should not be initialized without explicit setup."""
        assert tp.tensor_parallel_is_initialized() is False

    def test_get_groups_before_init_raises(self):
        """Group accessors should raise AssertionError before initialization."""
        with pytest.raises(AssertionError, match="not initialized"):
            tp.get_tensor_parallel_group()
        with pytest.raises(AssertionError, match="not initialized"):
            tp.get_data_parallel_group()

    def test_context_parallel_before_init_raises(self):
        """CP accessors should raise before initialization."""
        with pytest.raises(AssertionError, match="not initialized"):
            tp.get_context_parallel_group()


# ---------------------------------------------------------------------------
# Pure function tests (no distributed mocking needed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TP_AVAILABLE, reason="ezpz.tp not available")
class TestSplitTensorAlongLastDim:
    def test_split_tensor_along_last_dim(self):
        """Splitting a (4, 8) tensor into 2 partitions yields two (4, 4) tensors."""
        t = torch.arange(32).reshape(4, 8)
        parts = tp.split_tensor_along_last_dim(t, num_partitions=2)
        assert len(parts) == 2
        assert parts[0].shape == (4, 4)
        assert parts[1].shape == (4, 4)
        # First half should be columns 0-3, second half columns 4-7
        assert torch.equal(parts[0], t[:, :4])
        assert torch.equal(parts[1], t[:, 4:])

    def test_split_tensor_contiguous(self):
        """With contiguous_split_chunks=True, each chunk must be contiguous."""
        t = torch.arange(32).reshape(4, 8)
        parts = tp.split_tensor_along_last_dim(
            t, num_partitions=2, contiguous_split_chunks=True
        )
        assert len(parts) == 2
        for chunk in parts:
            assert chunk.is_contiguous()
        assert torch.equal(parts[0], t[:, :4])
        assert torch.equal(parts[1], t[:, 4:])

    def test_split_tensor_single_partition(self):
        """Splitting into 1 partition returns the original tensor."""
        t = torch.arange(32).reshape(4, 8)
        parts = tp.split_tensor_along_last_dim(t, num_partitions=1)
        assert len(parts) == 1
        assert torch.equal(parts[0], t)


# ---------------------------------------------------------------------------
# State management tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TP_AVAILABLE, reason="ezpz.tp not available")
class TestDestroyResetsAllGlobals:
    def test_destroy_resets_all_globals(self):
        """Manually set module globals to mock values, then verify destroy clears them."""
        sentinel = MagicMock()
        # Set every module-level global to a non-None sentinel
        tp_module._TENSOR_PARALLEL_GROUP = sentinel
        tp_module._TENSOR_PARALLEL_RANKS = [0, 1]
        tp_module._DATA_PARALLEL_GROUP = sentinel
        tp_module._DATA_PARALLEL_RANKS = [0]
        tp_module._PIPELINE_PARALLEL_GROUP = sentinel
        tp_module._PIPELINE_PARALLEL_RANKS = [0]
        tp_module._CONTEXT_PARALLEL_GROUP = sentinel
        tp_module._CONTEXT_PARALLEL_GROUP_RANKS = [0]

        # Precondition: looks initialized
        assert tp.tensor_parallel_is_initialized() is True

        tp.destroy_tensor_parallel()

        # All globals should be None
        assert tp_module._TENSOR_PARALLEL_GROUP is None
        assert tp_module._TENSOR_PARALLEL_RANKS is None
        assert tp_module._DATA_PARALLEL_GROUP is None
        assert tp_module._DATA_PARALLEL_RANKS is None
        assert tp_module._PIPELINE_PARALLEL_GROUP is None
        assert tp_module._PIPELINE_PARALLEL_RANKS is None
        assert tp_module._CONTEXT_PARALLEL_GROUP is None
        assert tp_module._CONTEXT_PARALLEL_GROUP_RANKS is None

        # And the initialized check should now return False
        assert tp.tensor_parallel_is_initialized() is False


# ---------------------------------------------------------------------------
# Mocked distributed tests — initialize_tensor_parallel()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TP_AVAILABLE, reason="ezpz.tp not available")
class TestInitializeTensorParallel:
    """Tests for initialize_tensor_parallel() using mocked torch.distributed."""

    @pytest.fixture(autouse=True)
    def _mock_distributed(self):
        """Patch torch.distributed primitives and clean up globals after each test."""
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=8),
            patch("torch.distributed.get_rank", return_value=0),
            patch(
                "torch.distributed.new_group",
                side_effect=lambda *a, **kw: MagicMock(),
            ),
        ):
            yield
        # Teardown: always reset module globals regardless of test outcome
        tp.destroy_tensor_parallel()

    def test_basic_init_tp2(self):
        """TP=2 on 8 ranks should succeed and report initialized."""
        tp.initialize_tensor_parallel(tensor_parallel_size=2)
        assert tp.tensor_parallel_is_initialized() is True

    def test_init_creates_groups(self):
        """After init with TP=2, all group globals should be non-None."""
        tp.initialize_tensor_parallel(tensor_parallel_size=2)
        assert tp_module._TENSOR_PARALLEL_GROUP is not None
        assert tp_module._DATA_PARALLEL_GROUP is not None
        assert tp_module._PIPELINE_PARALLEL_GROUP is not None
        assert tp_module._CONTEXT_PARALLEL_GROUP is not None

    def test_double_init_raises(self):
        """Calling initialize_tensor_parallel twice should raise AssertionError."""
        tp.initialize_tensor_parallel(tensor_parallel_size=2)
        with pytest.raises(AssertionError, match="already initialized"):
            tp.initialize_tensor_parallel(tensor_parallel_size=2)

    def test_tp1_is_valid(self):
        """TP=1 (the default) should still succeed."""
        tp.initialize_tensor_parallel(tensor_parallel_size=1)
        assert tp.tensor_parallel_is_initialized() is True


@pytest.mark.skipif(not TP_AVAILABLE, reason="ezpz.tp not available")
class TestDestroyTensorParallel:
    """Tests for destroy_tensor_parallel() after a mocked initialization."""

    @pytest.fixture(autouse=True)
    def _mock_distributed(self):
        """Patch torch.distributed primitives and clean up globals after each test."""
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=8),
            patch("torch.distributed.get_rank", return_value=0),
            patch(
                "torch.distributed.new_group",
                side_effect=lambda *a, **kw: MagicMock(),
            ),
        ):
            yield
        tp.destroy_tensor_parallel()

    def test_destroy_after_init(self):
        """After init then destroy, tensor_parallel_is_initialized() is False."""
        tp.initialize_tensor_parallel(tensor_parallel_size=2)
        assert tp.tensor_parallel_is_initialized() is True
        tp.destroy_tensor_parallel()
        assert tp.tensor_parallel_is_initialized() is False

    def test_getters_raise_after_destroy(self):
        """After destroy, group accessors should raise AssertionError."""
        tp.initialize_tensor_parallel(tensor_parallel_size=2)
        tp.destroy_tensor_parallel()
        with pytest.raises(AssertionError):
            tp.get_tensor_parallel_group()
        with pytest.raises(AssertionError):
            tp.get_data_parallel_group()
        with pytest.raises(AssertionError):
            tp.get_pipeline_parallel_group()
        with pytest.raises(AssertionError):
            tp.get_context_parallel_group()
