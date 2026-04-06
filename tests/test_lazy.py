"""Tests for the ezpz.lazy module."""

import pytest

try:
    import ezpz.lazy as lazy

    LAZY_AVAILABLE = True
except ImportError:
    LAZY_AVAILABLE = False


@pytest.mark.skipif(not LAZY_AVAILABLE, reason="ezpz.lazy not available")
class TestLazy:
    def test_lazy_import_resolves_on_access(self):
        """Lazy import should return a proxy that resolves to the real module."""
        os_module = lazy.lazy_import("os")
        # Actually USE the module — not just hasattr
        result = os_module.path.join("a", "b")
        assert result == "a/b" or result == "a\\b"  # platform-independent

    def test_lazy_import_nonexistent_raises_on_access(self):
        """Lazy importing a missing module should raise on attribute access."""
        nonexistent = lazy.lazy_import("nonexistent_module_xyz_42")
        with pytest.raises(ImportError):
            _ = nonexistent.some_attribute

    def test_lazy_import_submodule(self):
        """Lazy importing a submodule should resolve correctly."""
        path_module = lazy.lazy_import("os.path")
        # Actually call a function on it
        assert path_module.isabs("/foo") is True
        assert path_module.isabs("relative") is False
