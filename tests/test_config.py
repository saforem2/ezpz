"""Test configuration and fixtures for ezpz."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing."""
    original_env = os.environ.copy()

    # Set common environment variables
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["EZPZ_LOG_LEVEL"] = "ERROR"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_pbs_environment():
    """Mock PBS environment variables for testing."""
    original_env = os.environ.copy()

    # Set PBS environment variables
    os.environ["PBS_JOBID"] = "12345.test"
    os.environ["PBS_NODEFILE"] = "/tmp/test_nodefile"
    os.environ["PBS_O_WORKDIR"] = "/tmp"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_slurm_environment():
    """Mock SLURM environment variables for testing."""
    original_env = os.environ.copy()

    # Set SLURM environment variables
    os.environ["SLURM_JOB_ID"] = "67890"
    os.environ["SLURM_NODELIST"] = "node001,node002"
    os.environ["SLURM_SUBMIT_DIR"] = "/tmp"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_torch_device():
    """Mock torch device functions."""
    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.xpu.is_available", return_value=False
    ), patch("torch.backends.mps.is_available", return_value=False):
        yield
