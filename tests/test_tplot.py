"""Tests for the ezpz.tplot module."""

import numpy as np
import pytest
import xarray as xr

try:
    import ezpz.tplot as tplot
    import plotext  # noqa: F401

    TPLOT_AVAILABLE = True
except ImportError:
    TPLOT_AVAILABLE = False


@pytest.mark.skipif(not TPLOT_AVAILABLE, reason="ezpz.tplot not available")
class TestTPlot:
    def test_tplot_function_exists(self):
        assert hasattr(tplot, "tplot")
        assert callable(tplot.tplot)

    def test_tplot_dict_function_exists(self):
        assert hasattr(tplot, "tplot_dict")
        assert callable(tplot.tplot_dict)

    def test_tplot_with_numpy_arrays(self, tmp_path, capsys):
        """tplot should produce output when given numpy arrays."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        outfile = tmp_path / "plot.txt"
        tplot.tplot(y, x=x, outfile=outfile, figsize=(60, 15))
        capsys.readouterr()  # discard terminal plot output
        assert outfile.exists(), "tplot should write output file"
        content = outfile.read_text()
        assert len(content) > 0, "Plot output should not be empty"

    def test_tplot_dict_writes_output(self, tmp_path, capsys):
        """tplot_dict should produce output when given a dict of arrays."""
        data = {
            "sin": np.sin(np.linspace(0, 10, 50)),
            "cos": np.cos(np.linspace(0, 10, 50)),
        }
        outfile = tmp_path / "dict_plot.txt"
        tplot.tplot_dict(data, outfile=outfile, figsize=(60, 15))
        capsys.readouterr()  # discard terminal plot output
        assert outfile.exists(), "tplot_dict should write output file"
        content = outfile.read_text()
        assert len(content) > 0, "Plot output should not be empty"

    def test_tplot_with_xarray(self, tmp_path, capsys):
        """tplot should handle xarray DataArrays."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        data = xr.DataArray(y, coords=[x], dims=["x"])
        outfile = tmp_path / "xarray_plot.txt"
        tplot.tplot(data, outfile=outfile, figsize=(60, 15))
        capsys.readouterr()  # discard terminal plot output
        assert outfile.exists(), "tplot should write xarray output"
        content = outfile.read_text()
        assert len(content) > 0, "Plot output should not be empty"

    def test_tplot_with_title_and_labels(self, tmp_path, capsys):
        """Verify title/labels are passed through without error."""
        y = np.arange(20, dtype=float)
        outfile = tmp_path / "labeled.txt"
        tplot.tplot(
            y,
            title="Test Title",
            xlabel="X",
            ylabel="Y",
            outfile=outfile,
            figsize=(60, 15),
        )
        capsys.readouterr()  # discard terminal plot output
        assert outfile.exists()

    def test_tplot_clamp_helpers(self):
        """_clamp_width and _clamp_height respect MAX bounds."""
        assert tplot._clamp_width(9999) <= int(
            tplot.os.environ.get("EZPZ_TPLOT_MAX_WIDTH", tplot.MAX_PLOT_WIDTH)
        )
        assert tplot._clamp_height(9999) <= int(
            tplot.os.environ.get(
                "EZPZ_TPLOT_MAX_HEIGHT", tplot.MAX_PLOT_HEIGHT
            )
        )
        assert tplot._clamp_width(None) is None
        assert tplot._clamp_height(None) is None

    def test_resolve_marker_defaults(self):
        """_resolve_marker returns DEFAULT_MARKER when nothing is set."""
        marker = tplot._resolve_marker(None, plot_type="line")
        assert marker == tplot.DEFAULT_MARKER

    def test_resolve_marker_explicit(self):
        """Explicit marker takes priority over env."""
        marker = tplot._resolve_marker("braille", plot_type="line")
        assert marker == "braille"

    def test_resolve_plot_type_default(self):
        """_resolve_plot_type returns default when nothing is set."""
        assert tplot._resolve_plot_type(None, default="line") == "line"
        assert tplot._resolve_plot_type("hist", default="line") == "hist"

    def test_resolve_marker_env_override(self, monkeypatch):
        """EZPZ_TPLOT_MARKER env var overrides default marker."""
        monkeypatch.setenv("EZPZ_TPLOT_MARKER", "braille")
        result = tplot._resolve_marker(None)
        assert result == "braille"

    def test_resolve_plot_type_env_override(self, monkeypatch):
        """EZPZ_TPLOT_TYPE env var overrides default plot type."""
        monkeypatch.setenv("EZPZ_TPLOT_TYPE", "hist")
        result = tplot._resolve_plot_type(None)
        assert result == "hist"

    def test_tplot_histogram_mode(self, tmp_path, capsys):
        """tplot with plot_type='hist' creates output file."""
        y = np.random.randn(200)
        outfile = tmp_path / "hist_plot.txt"
        tplot.tplot(y, plot_type="hist", outfile=outfile, figsize=(60, 15))
        capsys.readouterr()
        assert outfile.exists(), "tplot hist should write output file"
        content = outfile.read_text()
        assert len(content) > 0, "Histogram output should not be empty"

    def test_tplot_scatter_mode(self, tmp_path, capsys):
        """tplot with plot_type='scatter' creates output file."""
        y = np.sin(np.linspace(0, 6, 50))
        outfile = tmp_path / "scatter_plot.txt"
        tplot.tplot(y, plot_type="scatter", outfile=outfile, figsize=(60, 15))
        capsys.readouterr()
        assert outfile.exists(), "tplot scatter should write output file"
        content = outfile.read_text()
        assert len(content) > 0, "Scatter output should not be empty"
