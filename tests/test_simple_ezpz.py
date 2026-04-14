"""Smoke tests for ezpz core imports and basic functionality."""

import logging
import re

import ezpz


def test_import_ezpz():
    """ezpz should import and expose a version string."""
    assert ezpz is not None
    assert hasattr(ezpz, "__version__")


def test_version_is_semver():
    """Version should look like a semver string (e.g. 0.11.3)."""
    assert re.match(r"\d+\.\d+\.\d+", ezpz.__version__), (
        f"Version {ezpz.__version__!r} doesn't match semver pattern"
    )


def test_logger_is_functional():
    """get_logger should return a logger that can actually log."""
    logger = ezpz.get_logger("test_simple")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_simple"
    # Actually invoke it — don't just check hasattr
    logger.info("smoke test message")
