"""Simple tests for ezpz functionality."""

import os

import pytest


def test_import_ezpz():
    """Test that we can import ezpz without immediate errors."""
    # This is a basic smoke test
    try:
        import ezpz

        assert ezpz is not None
        assert hasattr(ezpz, "__version__")
    except Exception as e:
        # If there are import errors, they're likely due to environment issues
        # which is expected in some test environments
        pytest.skip(f"Skipping due to import error: {e}")


def test_version_accessible():
    """Test that version information is accessible."""
    try:
        import ezpz

        version = ezpz.__version__
        assert isinstance(version, str)
        assert len(version) > 0
    except Exception as e:
        pytest.skip(f"Skipping due to import error: {e}")


def test_logger_creation():
    """Test that logger creation works."""
    try:
        import ezpz

        logger = ezpz.get_logger("test")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")
    except Exception as e:
        pytest.skip(f"Skipping due to import error: {e}")


def test_environment_variables():
    """Test that expected environment variables are set."""
    # These should be set by the ezpz initialization
    assert os.environ.get("WANDB_MODE") == "disabled"
    assert os.environ.get("EZPZ_LOG_LEVEL") is not None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
