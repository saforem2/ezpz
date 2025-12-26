# Logging Module Tests

## Overview

The logging module tests (`test_log.py`) verify the logging functionality, including logger creation, colored logging, and file logging.

## Test Cases

### test_get_logger

Tests the logger creation function.

```python
def test_get_logger(self):
    """Test get_logger function."""
    logger = log.get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
```

### test_use_colored_logs

Verifies the colored logging detection.

```python
def test_use_colored_logs(self):
    """Test use_colored_logs function."""
    use_colors = log.use_colored_logs()
    assert isinstance(use_colors, bool)
```

### test_get_file_logger

Tests the file logger creation functionality.

```python
def test_get_file_logger(self):
    """Test get_file_logger function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = log.get_file_logger(
            name="test_file_logger",
            fname=str(log_file.with_suffix(""))
        )
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_file_logger"

        # Test that we can log something
        logger.info("Test message")
        
        # Check that the file was created
        assert log_file.exists()
```

## Running Tests

```bash
python -m pytest tests/test_log.py
```