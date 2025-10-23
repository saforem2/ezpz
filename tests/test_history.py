"""Tests for the ezpz.history module."""

import json

import pytest
import numpy as np
import torch

try:
    import ezpz.history as history
    HISTORY_AVAILABLE = True
except ImportError:
    HISTORY_AVAILABLE = False


@pytest.mark.skipif(not HISTORY_AVAILABLE, reason="ezpz.history not available")
class TestHistory:
    def test_stopwatch_context_manager(self):
        """Test StopWatch context manager."""
        with history.StopWatch("test timer") as sw:
            # Do some work
            result = 1 + 1
        assert result == 2
        # The stopwatch should have recorded some data
        assert hasattr(sw, "data")
        assert isinstance(sw.data, dict)

    def test_history_initialization(self):
        """Test History class initialization."""
        hist = history.History()
        assert hist is not None
        assert hasattr(hist, "data")
        assert isinstance(hist.data, dict)

    def test_history_update(self):
        """Test History update method."""
        hist = history.History()
        # Add some test data
        metrics = {"loss": 0.5, "accuracy": 0.8}
        summary = hist.update(metrics)
        assert isinstance(summary, str)
        assert "loss" in summary
        assert "accuracy" in summary

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

    def test_history_jsonl_logging(self, tmp_path):
        """History writes JSONL entries when configured."""
        hist = history.History(
            report_enabled=False,
            jsonl_path=tmp_path / "metrics.jsonl",
            jsonl_overwrite=True,
        )
        hist._rank = 0
        metrics = {"loss": 1.23}
        hist.update(metrics)
        payload = []
        jsonl_file = tmp_path / "metrics.jsonl"
        assert jsonl_file.exists()
        for line in jsonl_file.read_text(encoding="utf-8").splitlines():
            payload.append(json.loads(line))
        assert payload[0]["metrics"]["loss"] == pytest.approx(1.23)

    def test_history_serializes_tensor_metrics(self, tmp_path):
        """Tensor metrics are converted to JSON-safe values on write."""
        hist = history.History(
            report_enabled=False,
            jsonl_path=tmp_path / "tensor_metrics.jsonl",
            jsonl_overwrite=True,
        )
        hist._rank = 0
        metrics = {
            "loss": torch.tensor(0.42, dtype=torch.float64),
            "weights": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        }
        summary = hist.update(metrics)
        assert "loss" in summary
        jsonl_file = tmp_path / "tensor_metrics.jsonl"
        lines = jsonl_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["metrics"]["loss"] == pytest.approx(0.42)
        np.testing.assert_allclose(
            payload["metrics"]["weights"],
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        )

    def test_history_markdown_report_on_plot(self, tmp_path):
        """Plotting produces a markdown report with embedded asset paths."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
        hist = history.History(report_dir=tmp_path, report_enabled=True)
        hist._rank = 0
        values = np.linspace(0.0, 1.0, 8)
        hist.plot(values, key="loss", outdir=tmp_path / "figures")
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close("all")
        report_files = list(tmp_path.rglob("report.md"))
        assert report_files, "Expected a generated report.md"
        contents = report_files[0].read_text(encoding="utf-8")
        assert "loss" in contents
        assert "![loss]" in contents

    def test_history_distributed_metrics_logged(self, tmp_path, monkeypatch):
        """Aggregated metrics from distributed reductions are recorded."""
        hist = history.History(
            report_enabled=False,
            jsonl_path=tmp_path / "metrics.jsonl",
            jsonl_overwrite=True,
        )
        hist._rank = 0
        monkeypatch.setattr(
            hist,
            "_compute_distributed_metrics",
            lambda metrics: {"loss/mean": 0.5, "loss/max": 0.75},
        )
        hist.update({"loss": 1.0})
        assert "loss/mean" in hist.history
        assert hist.history["loss/mean"][-1] == pytest.approx(0.5)

    def test_history_finalize_report_contains_environment(self, tmp_path):
        """Finalized report lives alongside outputs with environment context."""

        hist = history.History(report_dir=tmp_path, report_enabled=True)
        hist._rank = 0
        hist.update({"loss": 0.9, "accuracy": 0.8})
        env_info = {
            "Paths": {"Working directory": "/tmp/work"},
            "Python": {"Version": "3.12"},
        }
        _ = hist.finalize(
            outdir=tmp_path,
            run_name="test",
            save=False,
            plot=False,
            env_info=env_info,
        )
        report_file = tmp_path / "report.md"
        assert report_file.exists()
        contents = report_file.read_text(encoding="utf-8")
        assert "## Environment" in contents
        assert "Working directory" in contents
        assert "## Metric Overview" in contents
