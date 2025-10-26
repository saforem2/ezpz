"""Configuration file for pytest."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path so we can import ezpz modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set environment variables for testing
os.environ["WANDB_MODE"] = "disabled"
os.environ["EZPZ_LOG_LEVEL"] = "ERROR"


@pytest.fixture
def mock_dist_env():
    """Mock distributed environment variables."""
    original_env = os.environ.copy()
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_pbs_env():
    """Mock PBS environment variables."""
    original_env = os.environ.copy()
    temp_home = tempfile.mkdtemp(prefix="ezpz-home-")
    os.environ["HOME"] = temp_home
    os.environ["PBS_JOBID"] = "12345.test"
    os.environ["PBS_NODEFILE"] = "/tmp/test_nodefile"
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_env)
        shutil.rmtree(temp_home, ignore_errors=True)
