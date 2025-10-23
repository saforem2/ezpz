"""Test the main ezpz module imports and basic functionality."""


import ezpz


def test_ezpz_imports():
    """Test that ezpz can be imported without errors."""
    assert ezpz is not None


def test_ezpz_version():
    """Test that ezpz has a version."""
    assert hasattr(ezpz, "__version__")
    assert isinstance(ezpz.__version__, str)
    assert len(ezpz.__version__) > 0


def test_ezpz_logger():
    """Test that get_logger function works."""
    logger = ezpz.get_logger("test")
    assert logger is not None
    assert hasattr(logger, "info")
    assert hasattr(logger, "error")
    assert hasattr(logger, "debug")
