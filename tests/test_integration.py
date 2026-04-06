"""Integration tests for ezpz core functionality."""

import os


def test_submodule_imports():
    """Core submodules should be importable."""
    import ezpz.configs
    import ezpz.lazy
    import ezpz.utils

    # Verify they're real modules, not None stubs
    assert hasattr(ezpz.configs, "get_logging_config")
    assert hasattr(ezpz.utils, "get_timestamp")
    assert hasattr(ezpz.lazy, "lazy_import")


def test_version_is_consistent():
    """Version from __about__ should match the top-level __version__."""
    import ezpz
    from ezpz.__about__ import __version__ as about_version

    assert ezpz.__version__ == about_version


def test_conftest_disables_wandb():
    """conftest should set WANDB_MODE=disabled for test isolation."""
    assert os.environ.get("WANDB_MODE") == "disabled"


def test_logging_config_structure():
    """get_logging_config should return a valid dictConfig dict."""
    from ezpz.configs import get_logging_config

    config = get_logging_config(rank=0)
    assert isinstance(config, dict)
    assert "handlers" in config
    assert "root" in config or "loggers" in config


def test_rank_functions_return_ints():
    """Rank helpers should return non-negative integers."""
    import ezpz

    rank = ezpz.get_rank()
    local_rank = ezpz.get_local_rank()
    world_size = ezpz.get_world_size()

    assert isinstance(rank, int) and rank >= 0
    assert isinstance(local_rank, int) and local_rank >= 0
    assert isinstance(world_size, int) and world_size >= 1
