"""Tests for the ezpz.lazy module."""

import pytest

try:
    import ezpz.lazy as lazy
    LAZY_AVAILABLE = True
except ImportError:
    LAZY_AVAILABLE = False


@pytest.mark.skipif(not LAZY_AVAILABLE, reason="ezpz.lazy not available")
class TestLazy:
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

    def test_lazy_import_with_submodule(self):
        """Test lazy_import with submodule."""
        # Test importing a submodule
        path_module = lazy.lazy_import("os.path")
        assert path_module is not None
        assert hasattr(path_module, "join")