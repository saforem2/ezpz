"""Comprehensive test suite for ezpz with proper mocking."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Set up environment for testing
os.environ["WANDB_MODE"] = "disabled"
os.environ["EZPZ_LOG_LEVEL"] = "ERROR"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_basic_imports():
    """Test that we can import the basic modules without environment issues."""
    # Mock the problematic imports
    with patch("ezpz.dist.get_hostname", return_value="test-host"), patch(
        "ezpz.configs.get_scheduler", return_value="UNKNOWN"
    ), patch("ezpz.jobs.SCHEDULER", "UNKNOWN"):
        import ezpz

        assert ezpz is not None


@patch("ezpz.dist.get_hostname", return_value="test-host")
@patch("ezpz.configs.get_scheduler", return_value="UNKNOWN")
@patch("ezpz.jobs.SCHEDULER", "UNKNOWN")
def test_version(mock_get_scheduler, mock_get_hostname):
    """Test that ezpz has a version."""
    import ezpz

    assert hasattr(ezpz, "__version__")
    assert isinstance(ezpz.__version__, str)
    assert len(ezpz.__version__) > 0


@patch("ezpz.dist.get_hostname", return_value="test-host")
@patch("ezpz.configs.get_scheduler", return_value="UNKNOWN")
@patch("ezpz.jobs.SCHEDULER", "UNKNOWN")
def test_logger(mock_get_scheduler, mock_get_hostname):
    """Test that get_logger function works."""
    import ezpz

    logger = ezpz.get_logger("test")
    assert logger is not None
    assert hasattr(logger, "info")
    assert hasattr(logger, "error")
    assert hasattr(logger, "debug")


@patch("ezpz.dist.get_hostname", return_value="test-host")
@patch("ezpz.configs.get_scheduler", return_value="UNKNOWN")
@patch("ezpz.jobs.SCHEDULER", "UNKNOWN")
def test_configs_module(mock_get_scheduler, mock_get_hostname):
    """Test configs module functionality."""
    import ezpz.configs as configs

    # Test that paths exist (at least as Path objects)
    assert configs.HERE is not None
    assert configs.PROJECT_DIR is not None
    assert configs.CONF_DIR is not None

    # Test command_exists function
    assert configs.command_exists("python") is True
    assert configs.command_exists("nonexistent_command_xyz") is False

    # Test logging config
    config = configs.get_logging_config()
    assert isinstance(config, dict)
    assert "version" in config


@patch("ezpz.dist.get_hostname", return_value="test-host")
@patch("ezpz.configs.get_scheduler", return_value="UNKNOWN")
@patch("ezpz.jobs.SCHEDULER", "UNKNOWN")
def test_dist_module(mock_get_scheduler, mock_get_hostname):
    """Test dist module functionality."""
    import ezpz.dist as dist

    # Test basic functions
    rank = dist.get_rank()
    assert isinstance(rank, int)
    assert rank >= 0

    world_size = dist.get_world_size()
    assert isinstance(world_size, int)
    assert world_size >= 1

    local_rank = dist.get_local_rank()
    assert isinstance(local_rank, int)
    assert local_rank >= 0

    # Test device functions
    device_type = dist.get_torch_device_type()
    assert isinstance(device_type, str)
    assert device_type in ["cpu", "cuda", "xpu", "mps"]

    device = dist.get_torch_device()
    assert device is not None


@patch("ezpz.dist.get_hostname", return_value="test-host")
@patch("ezpz.configs.get_scheduler", return_value="UNKNOWN")
@patch("ezpz.jobs.SCHEDULER", "UNKNOWN")
def test_utils_module(mock_get_scheduler, mock_get_hostname):
    """Test utils module functionality."""
    import numpy as np

    import ezpz.utils as utils

    # Test timestamp
    timestamp = utils.get_timestamp()
    assert isinstance(timestamp, str)
    assert len(timestamp) > 0

    # Test format_pair
    result = utils.format_pair("test", 5)
    assert result == "test=5"

    # Test normalize
    result = utils.normalize("test_name.sub-name")
    assert result == "test-name-sub-name"

    # Test grab_tensor with numpy array
    np_array = np.array([1, 2, 3])
    result = utils.grab_tensor(np_array)
    assert np.array_equal(result, np_array)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
