# Tensor Parallel Module Tests

## Overview

The tensor parallel module tests (`test_tp.py`) verify the tensor parallel computing functionality, including process group initialization and parallel group management.

## Test Cases

### test_tensor_parallel_functions_exist

Tests that all required tensor parallel functions are available.

```python
def test_tensor_parallel_functions_exist(self):
    """Test that tensor parallel functions exist."""
    # Check that key functions are available
    assert hasattr(tp, "initialize_tensor_parallel")
    assert hasattr(tp, "tensor_parallel_is_initialized")
    assert hasattr(tp, "get_tensor_parallel_group")
    assert hasattr(tp, "get_data_parallel_group")
    assert hasattr(tp, "get_pipeline_parallel_group")
    assert hasattr(tp, "destroy_tensor_parallel")
```

### test_utility_functions

Verifies the utility functions for tensor parallel computing.

```python
def test_utility_functions(self):
    """Test utility functions."""
    # Check that utility functions are available
    assert hasattr(tp, "ensure_divisibility")
    assert hasattr(tp, "divide_and_check_no_remainder")
    assert hasattr(tp, "split_tensor_along_last_dim")
```

### test_context_parallel_functions

Tests the context parallel functionality.

```python
def test_context_parallel_functions(self):
    """Test context parallel functions."""
    assert hasattr(tp, "get_context_parallel_group")
    assert hasattr(tp, "get_context_parallel_ranks")
    assert hasattr(tp, "get_context_parallel_world_size")
    assert hasattr(tp, "get_context_parallel_rank")
```

### test_pipeline_parallel_functions

Verifies the pipeline parallel functionality.

```python
def test_pipeline_parallel_functions(self):
    """Test pipeline parallel functions."""
    assert hasattr(tp, "get_pipeline_parallel_ranks")
```

### test_ensure_divisibility

Tests the divisibility checking function.

```python
def test_ensure_divisibility(self):
    """Test ensure_divisibility function."""
    # This should not raise an exception
    tp.ensure_divisibility(10, 2)
    
    # This should raise an exception
    with pytest.raises(AssertionError):
        tp.ensure_divisibility(10, 3)
```

## Running Tests

```bash
python -m pytest tests/test_tp.py
```