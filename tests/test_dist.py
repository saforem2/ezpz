"""Tests for the ezpz.dist module."""

import pytest
import os

try:
    import ezpz.dist as dist
    DIST_AVAILABLE = True
except ImportError:
    DIST_AVAILABLE = False


@pytest.mark.skipif(not DIST_AVAILABLE, reason="ezpz.dist not available")
class TestDist:
    def test_get_rank(self):
        """Test get_rank function."""
        rank = dist.get_rank()
        assert isinstance(rank, int)
        assert rank >= 0

    def test_get_world_size(self):
        """Test get_world_size function."""
        world_size = dist.get_world_size()
        assert isinstance(world_size, int)
        assert world_size >= 1

    def test_get_local_rank(self):
        """Test get_local_rank function."""
        local_rank = dist.get_local_rank()
        assert isinstance(local_rank, int)
        assert local_rank >= 0

    def test_get_torch_device_type(self):
        """Test get_torch_device_type function."""
        device_type = dist.get_torch_device_type()
        assert isinstance(device_type, str)
        # Should be one of the common device types
        assert device_type in ["cpu", "cuda", "xpu", "mps"]

    def test_get_torch_device(self):
        """Test get_torch_device function."""
        device = dist.get_torch_device()
        assert device is not None
        # Should contain the device type
        assert dist.get_torch_device_type() in str(device)

    def test_seed_everything(self):
        """Test seed_everything function."""
        # This should not raise an exception
        dist.seed_everything(42)

    def test_query_environment(self):
        """Test query_environment function."""
        env_info = dist.query_environment()
        assert isinstance(env_info, dict)
        # Should contain some basic information
        assert len(env_info) > 0

    def test_get_dist_info(self):
        """Test get_dist_info function."""
        dist_info = dist.get_dist_info()
        assert isinstance(dist_info, dict)
        # Should contain rank and world_size
        assert "rank" in dist_info
        assert "world_size" in dist_info