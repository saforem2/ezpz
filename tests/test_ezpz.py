"""Test the main ezpz module imports and basic functionality."""

import logging

import ezpz


def test_public_api_accessible():
    """Key public API symbols should be importable from ezpz."""
    # These are the most important user-facing functions
    assert callable(ezpz.setup_torch)
    assert callable(ezpz.get_torch_device)
    assert callable(ezpz.wrap_model)
    assert callable(ezpz.cleanup)
    assert callable(ezpz.get_logger)
    assert callable(ezpz.synchronize)
    assert callable(ezpz.get_rank)
    assert callable(ezpz.get_world_size)
    assert callable(ezpz.get_local_rank)


def test_get_logger_returns_named_logger():
    """get_logger should return a stdlib Logger with the requested name."""
    logger = ezpz.get_logger("test_ezpz")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_ezpz"


def test_get_logger_rank_zero_suppresses_non_zero():
    """Non-rank-0 loggers should be set to CRITICAL to suppress output."""
    logger = ezpz.get_logger("test_rank1", rank=1, rank_zero_only=True)
    assert logger.level == logging.CRITICAL


def test_get_logger_rank_zero_allows_zero():
    """Rank-0 logger should be set to the requested level, not CRITICAL."""
    logger = ezpz.get_logger(
        "test_rank0", rank=0, rank_zero_only=True, level="INFO"
    )
    assert logger.level == logging.INFO
