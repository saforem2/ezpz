"""Integration tests for ezpz core functionality."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_import_structure():
    """Test that the basic import structure works."""
    # This test just verifies that we can import the main modules
    # without immediate errors, even if they don't work fully

    # Test basic imports that should always work
    import ezpz

    assert hasattr(ezpz, "__version__")

    # Test that we can import submodules
    import ezpz.configs
    import ezpz.lazy
    import ezpz.utils

    # Verify the imports worked
    assert ezpz.configs is not None
    assert ezpz.utils is not None
    assert ezpz.lazy is not None


def test_version_access():
    """Test that version information is accessible."""
    import ezpz

    assert isinstance(ezpz.__version__, str)
    assert len(ezpz.__version__) > 0


def test_logger_creation():
    """Test that logger creation works."""
    import ezpz

    logger = ezpz.get_logger("test")
    assert logger is not None
    assert hasattr(logger, "info")
    assert hasattr(logger, "error")
    assert hasattr(logger, "debug")


class TestEnvironmentVariables:
    """Test environment variable handling."""

    def test_wandb_disabled(self):
        """Test that WANDB_MODE is set to disabled."""
        assert os.environ.get("WANDB_MODE") == "disabled"

    def test_log_level_set(self):
        """Test that EZPZ_LOG_LEVEL is set."""
        assert os.environ.get("EZPZ_LOG_LEVEL") is not None
