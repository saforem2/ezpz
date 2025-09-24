# `ezpz.__init__`

Main entry point for the ezpz package.

- See [ezpz/`__init__.py`](https://github.com/ezpz/ezpz/blob/main/ezpz/__init__.py)

## Overview

The `ezpz.__init__` module serves as the main entry point for the ezpz package,
importing and exposing the most commonly used functions and classes.

## Imported Modules

The module imports and re-exports functionality from several submodules:

- `ezpz.configs`: Configuration management
- `ezpz.dist`: Distributed computing utilities
- `ezpz.history`: Training history and metrics tracking
- `ezpz.jobs`: Job management utilities
- `ezpz.log`: Logging utilities
- `ezpz.tp`: Tensor parallel computing
- `ezpz.utils`: General utility functions

## Key Exports

::: ezpz

## Usage Examples

### Basic Setup

```python
import ezpz

# Initialize distributed training
ezpz.setup_torch()

# Get logger
logger = ezpz.get_logger(__name__)
logger.info("ezpz initialized successfully")
```

### Distributed Computing

```python
import ezpz

# Get process information
rank = ezpz.get_rank()
world_size = ezpz.get_world_size()

# Get device information
device = ezpz.get_torch_device()
device_type = ezpz.get_torch_device_type()
```

### Logging

```python
import ezpz

# Get a logger with proper configuration
logger = ezpz.get_logger("my_module")

# Log messages with different levels
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### History Tracking

```python
import ezpz

# Create a history tracker
history = ezpz.History()

# Add metrics
metrics = {"loss": 0.5, "accuracy": 0.8}
summary = history.update(metrics)
print(summary)  # Formatted summary string
```
