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


@pytest.mark.skipif(not LOG_AVAILABLE, reason="ezpz.log not available")
class TestSilenceNoisyLoggers:
    """The opt-in helper that raises HF Hub / urllib3 / httpx / filelock
    loggers to WARNING — used by every example that touches the Hub."""

    DEFAULT_NAMES = ("httpx", "huggingface_hub", "filelock", "urllib3")

    def _reset(self, names):
        """Restore each logger's level to NOTSET so tests don't leak state."""
        for n in names:
            logging.getLogger(n).setLevel(logging.NOTSET)

    def test_default_targets_raised_to_warning(self):
        try:
            log.silence_noisy_loggers(silence_transformers=False)
            for name in self.DEFAULT_NAMES:
                assert (
                    logging.getLogger(name).level == logging.WARNING
                ), f"{name} not raised to WARNING"
        finally:
            self._reset(self.DEFAULT_NAMES)

    def test_extra_logger_names_also_quieted(self):
        custom = ("matplotlib.font_manager", "_custom_noisy_test")
        try:
            log.silence_noisy_loggers(
                extra=custom, silence_transformers=False
            )
            for name in custom:
                assert (
                    logging.getLogger(name).level == logging.WARNING
                ), f"{name} (extra) not raised to WARNING"
        finally:
            self._reset(self.DEFAULT_NAMES + custom)

    def test_idempotent(self):
        """Calling twice shouldn't error or compound."""
        try:
            log.silence_noisy_loggers(silence_transformers=False)
            log.silence_noisy_loggers(silence_transformers=False)
            assert logging.getLogger("httpx").level == logging.WARNING
        finally:
            self._reset(self.DEFAULT_NAMES)

    def test_custom_level(self):
        """Level param overrides the default WARNING."""
        try:
            log.silence_noisy_loggers(
                level=logging.ERROR, silence_transformers=False
            )
            assert logging.getLogger("httpx").level == logging.ERROR
        finally:
            self._reset(self.DEFAULT_NAMES)

    def test_silence_transformers_swallows_missing_package(self):
        """If transformers isn't installed, the helper should not raise."""
        # We can't easily uninstall transformers in a test, but we can
        # at least verify the call doesn't raise when the flag is True
        # (covers the happy path on machines that have it, and the
        # except-clause is the same shape either way).
        try:
            log.silence_noisy_loggers(silence_transformers=True)
        finally:
            self._reset(self.DEFAULT_NAMES)

    def test_top_level_reexport(self):
        """Resolvable as `ezpz.silence_noisy_loggers` via lazy __getattr__."""
        import ezpz
        assert callable(ezpz.silence_noisy_loggers)
        assert ezpz.silence_noisy_loggers is log.silence_noisy_loggers
