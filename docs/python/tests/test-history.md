# History Module Tests

## Overview

The history module tests (`test_history.py`) verify the metrics tracking and history management functionality, including the StopWatch context manager and History class.

## Test Cases

### test_stopwatch_context_manager

Tests the StopWatch context manager for timing code blocks.

```python
def test_stopwatch_context_manager(self):
    """Test StopWatch context manager."""
    with history.StopWatch("test timer") as sw:
        # Do some work
        result = 1 + 1
    assert result == 2
    # The stopwatch should have recorded some data
    assert hasattr(sw, "data")
    assert isinstance(sw.data, dict)
```

### test_history_initialization

Verifies the History class initialization.

```python
def test_history_initialization(self):
    """Test History class initialization."""
    hist = history.History()
    assert hist is not None
    assert hasattr(hist, "data")
    assert isinstance(hist.data, dict)
```

### test_history_update

Tests the History update method for adding metrics.

```python
def test_history_update(self):
    """Test History update method."""
    hist = history.History()
    # Add some test data
    metrics = {"loss": 0.5, "accuracy": 0.8}
    summary = hist.update(metrics)
    assert isinstance(summary, str)
    assert "loss" in summary
    assert "accuracy" in summary
```

### test_grab_tensor

Verifies the tensor conversion utility function.

```python
def test_grab_tensor(self):
    """Test grab_tensor function."""
    # Test with numpy array
    np_array = np.array([1, 2, 3])
    result = history.grab_tensor(np_array)
    assert np.array_equal(result, np_array)

    # Test with torch tensor
    torch_tensor = torch.tensor([1, 2, 3])
    result = history.grab_tensor(torch_tensor)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, torch_tensor.numpy())

    # Test with scalar
    scalar = 5
    result = history.grab_tensor(scalar)
    assert result == scalar

    # Test with list
    test_list = [1, 2, 3]
    result = history.grab_tensor(test_list)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array(test_list))
```

## Running Tests

```bash
python -m pytest tests/test_history.py
```