"""Tests for the ezpz.log module."""

import logging
import tempfile
from pathlib import Path

import pytest

try:
    import ezpz.log as log

    LOG_AVAILABLE = True
except ImportError:
    LOG_AVAILABLE = False


@pytest.mark.skipif(not LOG_AVAILABLE, reason="ezpz.log not available")
class TestLog:
    def test_get_logger(self):
        """Test get_logger function."""
        logger = log.get_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_use_colored_logs(self):
        """Test use_colored_logs function."""
        use_colors = log.use_colored_logs()
        assert isinstance(use_colors, bool)

    def test_get_file_logger(self):
        """Test get_file_logger function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = log.get_file_logger(
                name="test_file_logger", fname=str(log_file.with_suffix(""))
            )
            assert isinstance(logger, logging.Logger)
            assert logger.name == "test_file_logger"

            # Test that we can log something
            logger.info("Test message")

            # Check that the file was created
            assert log_file.exists()
