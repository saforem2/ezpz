# Lazy Import Tests

## Overview

The lazy import tests (`test_lazy.py`) verify the lazy loading functionality, which allows for deferred module imports to improve startup time.

## Test Cases

### test_lazy_import

Tests the lazy_import function for standard modules.

```python
def test_lazy_import(self):
    """Test lazy_import function."""
    # Test importing a standard library module
    os_module = lazy.lazy_import("os")
    assert os_module is not None
    assert hasattr(os_module, "path")

    # Test importing a non-existent module (should not raise immediately)
    nonexistent = lazy.lazy_import("nonexistent_module_xyz")
    assert nonexistent is not None  # Should return a lazy object

    # But accessing attributes should raise ImportError
    with pytest.raises(ImportError):
        _ = nonexistent.some_attribute
```

### test_lazy_import_with_submodule

Verifies lazy import functionality with submodules.

```python
def test_lazy_import_with_submodule(self):
    """Test lazy_import with submodule."""
    # Test importing a submodule
    path_module = lazy.lazy_import("os.path")
    assert path_module is not None
    assert hasattr(path_module, "join")
```

## Running Tests

```bash
python -m pytest tests/test_lazy.py
```