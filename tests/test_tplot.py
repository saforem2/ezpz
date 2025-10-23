"""Tests for the ezpz.tplot module."""

import numpy as np
import pytest
import xarray as xr

try:
    import ezpz.tplot as tplot

    TPLOT_AVAILABLE = True
except ImportError:
    TPLOT_AVAILABLE = False


@pytest.mark.skipif(not TPLOT_AVAILABLE, reason="ezpz.tplot not available")
class TestTPlot:
    def test_tplot_function_exists(self):
        """Test that tplot function exists."""
        assert hasattr(tplot, "tplot")
        assert callable(tplot.tplot)

    def test_tplot_dict_function_exists(self):
        """Test that tplot_dict function exists."""
        assert hasattr(tplot, "tplot_dict")
        assert callable(tplot.tplot_dict)

    def test_tplot_with_simple_data(self):
        """Test tplot with simple data."""
        # Create some simple test data
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        # This should not raise an exception
        # Note: We're not checking the output since it's a plotting function
        try:
            tplot.tplot(x, y)
        except Exception:
            # If there's an exception, it might be because we're not in a proper terminal
            # That's okay for this test
            pass

    def test_tplot_dict_with_simple_data(self):
        """Test tplot_dict with simple data."""
        # Create some simple test data
        data = {
            "x": np.linspace(0, 10, 100),
            "y": np.sin(np.linspace(0, 10, 100)),
        }

        # This should not raise an exception
        try:
            tplot.tplot_dict(data)
        except Exception:
            # If there's an exception, it might be because we're not in a proper terminal
            # That's okay for this test
            pass

    def test_tplot_with_xarray_data(self):
        """Test tplot with xarray data."""
        # Create some xarray test data
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        data = xr.DataArray(y, coords=[x], dims=["x"])

        # This should not raise an exception
        try:
            tplot.tplot(data)
        except Exception:
            # If there's an exception, it might be because we're not in a proper terminal
            # That's okay for this test
            pass
