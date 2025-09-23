# `ezpz.utils`

Utility functions for the ezpz package.

- See [ezpz/`utils`](https://github.com/ezpz/ezpz/blob/main/ezpz/utils)

## Overview

The `ezpz.utils` module provides various utility functions for common tasks such as:

- Debugging and breakpoint management
- Timestamp generation and formatting
- String normalization and formatting
- Tensor/array conversion utilities
- Memory monitoring
- Model summary generation
- Data serialization and deserialization

## Key Functions

::: ezpz.utils

## Examples

### Debugging with Distributed Breakpoints

```python
import ezpz.utils as utils

# Set a breakpoint only on rank 0
utils.breakpoint(rank=0)
```

### Timestamp Generation

```python
import ezpz.utils as utils

# Get current timestamp
timestamp = utils.get_timestamp()
print(timestamp)  # Output: "2023-12-01-143022"

# Get timestamp with custom format
date_only = utils.get_timestamp("%Y-%m-%d")
print(date_only)  # Output: "2023-12-01"
```

### String Normalization

```python
import ezpz.utils as utils

# Normalize strings for consistent naming
name = utils.normalize("Test_Name.Sub-Name")
print(name)  # Output: "test-name-sub-name"
```

### Tensor Conversion

```python
import ezpz.utils as utils
import torch
import numpy as np

# Convert various tensor types to numpy arrays
torch_tensor = torch.tensor([1, 2, 3])
numpy_array = utils.grab_tensor(torch_tensor)
print(numpy_array)  # Output: [1 2 3]

# Convert lists to arrays
list_data = [1, 2, 3]
array_data = utils.grab_tensor(list_data)
print(array_data)  # Output: [1 2 3]
```
