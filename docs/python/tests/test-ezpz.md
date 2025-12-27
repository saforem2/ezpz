# Main Module Tests

## Overview

The main module tests (`test_ezpz.py`) verify the core functionality of the ezpz package, including initialization, version checking, and basic imports.

## Test Cases

### test_ezpz_imports

Verifies that the ezpz package can be imported without errors.

```python
def test_ezpz_imports():
    """Test that ezpz can be imported without errors."""
    assert ezpz is not None
```

### test_ezpz_version

Ensures that the ezpz package has a valid version string.

```python
def test_ezpz_version():
    """Test that ezpz has a version."""
    assert hasattr(ezpz, "__version__")
    assert isinstance(ezpz.__version__, str)
    assert len(ezpz.__version__) > 0
```

### test_ezpz_logger

Tests the logger creation functionality.

```python
def test_ezpz_logger():
    """Test that get_logger function works."""
    logger = ezpz.get_logger("test")
    assert logger is not None
    assert hasattr(logger, "info")
    assert hasattr(logger, "error")
    assert hasattr(logger, "debug")
```

## Running Tests

```bash
python -m pytest tests/test_ezpz.py
```