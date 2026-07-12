"""Tests for the ezpz.history module."""

import json
import os
import warnings
from unittest.mock import patch

import numpy as np
import pytest
import torch
import xarray as xr

try:
    import ezpz.history as history

    HISTORY_AVAILABLE = True
except ImportError:
    HISTORY_AVAILABLE = False

from ezpz.tracker import NullTracker, Tracker, TrackerBackend


class _FakeBackend(TrackerBackend):
    """Test double that records all calls."""

    name = "fake"

    def __init__(self):
        self.logs: list[dict] = []
        self.configs: list[dict] = []
        self.tables: list[dict] = []
        self.images: list[dict] = []
        self.finished = False

    def log(self, metrics, step=None, commit=True):
        self.logs.append({"metrics": metrics, "step": step, "commit": commit})

    def log_config(self, config):
        self.configs.append(config)

    def log_table(self, key, columns, data):
        self.tables.append({"key": key, "columns": columns, "data": data})

    def log_image(self, key, image_path, caption=None):
        self.images.append({"key": key, "path": image_path, "caption": caption})

    def finish(self):
        self.finished = True


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

    def test_history_to_dataarray_handles_tuples_and_empty(self):
        """Tuple lists and empty lists are handled safely."""
        hist = history.History()
        tuple_values = [(1,), (2,), (3,)]
        arr = hist.to_DataArray(tuple_values)
        assert arr.shape == (1, 3)
        np.testing.assert_allclose(arr.values, np.array([[1, 2, 3]]))

        empty_arr = hist.to_DataArray([])
        assert empty_arr.shape == (0,)

        warmed = hist.to_DataArray([1, 2, 3], warmup=1.0)
        assert warmed.shape == (0,)

    def test_distributed_std_survives_fp32_cancellation(
        self, monkeypatch
    ):
        """Distributed std for large-magnitude metrics survives the
        ``E[X^2] - E[X]^2`` variance formula.

        Regression: when per-rank values are ~1e5 (e.g. ``tokens_per_sec``)
        and their across-rank variance is small (~hundreds), the
        variance subtraction loses its entire signal to fp32
        catastrophic cancellation — the squared values are ~1e10 and
        the variance is 8 orders of magnitude smaller than mantissa
        precision allows. Pre-fix, ``std`` came back as exactly 0.0
        and the console summary dropped the ``(±std)`` parenthetical
        entirely, making it look like every rank measured the same
        value.

        Fix: promote to fp64 before squaring (commit message has the
        full numerical analysis).
        """
        # Per-rank values mirroring a real training-log signature:
        # tokens_per_sec ≈ 124k across 6 ranks with ~20 std.
        per_rank_values = [
            123880.0, 123920.0, 123870.0, 123890.0, 123900.0, 123860.0,
        ]
        world_size = len(per_rank_values)

        # `History.__init__` short-circuits `distributed_history` to
        # False unless `get_world_size() > 1`. Patch BEFORE constructing
        # so the flag survives, and again later for the actual call.
        monkeypatch.setattr(
            "ezpz.distributed.get_world_size", lambda: world_size
        )
        monkeypatch.setattr(
            torch.distributed,
            "is_initialized",
            lambda: True,
            raising=False,
        )
        # Also patch is_available — checked inside _compute_distributed_metrics.
        monkeypatch.setattr(
            torch.distributed, "is_available", lambda: True, raising=False
        )

        hist = history.History(distributed_history=True)
        hist._rank = 0
        # Force _select_metric_device to return CPU so we don't try to
        # allocate on a GPU that doesn't exist in CI.
        monkeypatch.setattr(
            hist, "_select_metric_device", lambda: torch.device("cpu")
        )
        # Each rank only sees its own value; the all-reduce here
        # simulates the global SUM/MAX/MIN by replacing the local
        # tensor's contents in-place with the across-rank aggregate.
        # Dispatch by call order, not tensor contents — the source
        # always issues reduces in the same order (SUM sum, SUM sq,
        # MAX, MIN), and content-comparison would break for inputs
        # where x ≈ x² (e.g. small floats or values near 0 or 1).
        sum_call_count = [0]  # nonlocal-effective via list

        def _fake_all_reduce(tensor, op=None, **_kw):
            ops = torch.distributed.ReduceOp
            arr = torch.tensor(
                per_rank_values, dtype=tensor.dtype, device=tensor.device
            )
            if op == ops.SUM:
                sum_call_count[0] += 1
                # First SUM is for `sum_vals` (Σx), second is for
                # `sq_vals` (Σx²). Source code never reorders these.
                if sum_call_count[0] == 1:
                    tensor.copy_(arr.sum().to(tensor.dtype).view_as(tensor))
                else:
                    tensor.copy_(
                        arr.square().sum().to(tensor.dtype).view_as(tensor)
                    )
            elif op == ops.MAX:
                tensor.copy_(arr.max().to(tensor.dtype).view_as(tensor))
            elif op == ops.MIN:
                tensor.copy_(arr.min().to(tensor.dtype).view_as(tensor))

        monkeypatch.setattr(
            torch.distributed, "all_reduce", _fake_all_reduce
        )

        # The History helper enters with this rank's local value.
        stats = hist._compute_distributed_metrics(
            {"tokens_per_sec": per_rank_values[0]}
        )

        # Sanity: mean is approximately correct. With fp64 promotion
        # we get full precision; pre-fix the fp32 mean also worked but
        # only to ~7 sig digits for ~1e5 values. The bug is in std,
        # not mean.
        true_mean = sum(per_rank_values) / world_size
        assert stats["tokens_per_sec/mean"] == pytest.approx(true_mean, abs=0.01)
        # The point of the regression: std MUST be non-zero. Pre-fix
        # this was exactly 0.0 due to fp32 cancellation.
        assert stats["tokens_per_sec/std"] > 0.0, (
            "Distributed std collapsed to 0.0 — fp32 cancellation regressed "
            "(see history.py:_compute_distributed_metrics)"
        )
        # Verify the std is close to the actual population std (~20)
        # — not just non-zero by accident.
        import statistics
        expected_std = statistics.pstdev(per_rank_values)
        assert stats["tokens_per_sec/std"] == pytest.approx(
            expected_std, rel=1e-3
        )

    def test_distributed_std_falls_back_when_fp64_unsupported(
        self, monkeypatch, caplog
    ):
        """If the fp64 promotion or probe raises (e.g. some Intel XPU
        devices don't support fp64), the helper should fall back to
        fp32 with a logged warning rather than crashing the run.

        Losing precision on ``/std`` for large-magnitude metrics is
        bad, but losing the entire training run because metric
        collection threw is much worse.
        """
        # Smaller-magnitude values so fp32 cancellation isn't an
        # issue here — this test is about the fallback path firing,
        # not about std accuracy on the fallback.
        per_rank_values = [0.5, 0.6, 0.4, 0.55, 0.45, 0.5]
        world_size = len(per_rank_values)

        monkeypatch.setattr(
            "ezpz.distributed.get_world_size", lambda: world_size
        )
        monkeypatch.setattr(
            torch.distributed, "is_initialized", lambda: True, raising=False
        )
        monkeypatch.setattr(
            torch.distributed, "is_available", lambda: True, raising=False
        )

        hist = history.History(distributed_history=True)
        hist._rank = 0
        monkeypatch.setattr(
            hist, "_select_metric_device", lambda: torch.device("cpu")
        )

        # Patch the Tensor.to method so any fp64 promotion raises.
        # Simulates an XPU device that doesn't support fp64.
        orig_to = torch.Tensor.to

        def _no_fp64(self, *args, **kwargs):
            dtype = kwargs.get("dtype")
            if not dtype and args and isinstance(args[0], torch.dtype):
                dtype = args[0]
            if dtype == torch.float64:
                raise RuntimeError(
                    "Simulated: device does not support fp64"
                )
            return orig_to(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", _no_fp64)

        sum_call_count = [0]

        def _fake_all_reduce(tensor, op=None, **_kw):
            ops = torch.distributed.ReduceOp
            arr = torch.tensor(
                per_rank_values, dtype=tensor.dtype, device=tensor.device
            )
            if op == ops.SUM:
                sum_call_count[0] += 1
                if sum_call_count[0] == 1:
                    tensor.copy_(arr.sum().to(tensor.dtype).view_as(tensor))
                else:
                    tensor.copy_(
                        arr.square().sum().to(tensor.dtype).view_as(tensor)
                    )
            elif op == ops.MAX:
                tensor.copy_(arr.max().to(tensor.dtype).view_as(tensor))
            elif op == ops.MIN:
                tensor.copy_(arr.min().to(tensor.dtype).view_as(tensor))

        monkeypatch.setattr(
            torch.distributed, "all_reduce", _fake_all_reduce
        )

        # Should not raise — the source's try/except catches the
        # fp64 RuntimeError and falls back to fp32.
        stats = hist._compute_distributed_metrics(
            {"loss": per_rank_values[0]}
        )

        # Stats should still be computed (in fp32). Mean is exact for
        # these small values; std is non-zero because the magnitudes
        # are small enough that fp32 cancellation doesn't bite.
        true_mean = sum(per_rank_values) / world_size
        assert stats["loss/mean"] == pytest.approx(true_mean, abs=1e-5)
        assert stats["loss/std"] > 0.0

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


@pytest.mark.skipif(not HISTORY_AVAILABLE, reason="ezpz.history not available")
class TestHistoryTrackerIntegration:
    """Tests for the History-Tracker integration."""

    def test_no_backends_uses_null_tracker(self):
        """History() with no backend args uses NullTracker."""
        hist = history.History()
        assert isinstance(hist.tracker, NullTracker)

    def test_tracker_injection(self):
        """A pre-built Tracker can be injected via the tracker kwarg."""
        backend = _FakeBackend()
        tracker = Tracker([backend])
        hist = history.History(tracker=tracker)
        assert hist.tracker is tracker

    def test_tracker_receives_metrics(self):
        """update() dispatches metrics to the injected tracker."""
        backend = _FakeBackend()
        tracker = Tracker([backend])
        hist = history.History(tracker=tracker)
        hist.update({"loss": 0.5, "accuracy": 0.9})
        assert len(backend.logs) == 1
        assert backend.logs[0]["metrics"]["loss"] == pytest.approx(0.5)

    def test_step_parameter_forwarded(self):
        """update(step=N) forwards the step to tracker.log()."""
        backend = _FakeBackend()
        tracker = Tracker([backend])
        hist = history.History(tracker=tracker)
        hist.update({"loss": 0.3}, step=42)
        assert backend.logs[0]["step"] == 42

    def test_finalize_calls_tracker_finish(self):
        """finalize() calls tracker.finish()."""
        backend = _FakeBackend()
        tracker = Tracker([backend])
        hist = history.History(tracker=tracker, report_enabled=False)
        hist._rank = 0
        hist.update({"loss": 0.5})
        hist.finalize(save=False, plot=False)
        assert backend.finished

    def test_finalize_logs_training_history_table(self):
        """finalize() logs a training_history table via tracker."""
        backend = _FakeBackend()
        tracker = Tracker([backend])
        hist = history.History(tracker=tracker, report_enabled=False)
        hist._rank = 0
        hist.update({"loss": 0.9})
        hist.update({"loss": 0.5})
        hist.finalize(save=False, plot=False)
        table_logs = [t for t in backend.tables if t["key"] == "training_history"]
        assert len(table_logs) == 1
        assert "loss" in table_logs[0]["columns"]
        assert len(table_logs[0]["data"]) == 2

    def test_config_logged_on_init(self):
        """Config dict is forwarded to tracker.log_config()."""
        backend = _FakeBackend()
        tracker = Tracker([backend])
        config = {"lr": 1e-4, "batch_size": 32}
        hist = history.History(tracker=tracker, config=config)
        assert len(backend.configs) == 1
        assert backend.configs[0]["lr"] == pytest.approx(1e-4)

    def test_with_csv_backend(self, tmp_path):
        """History with backends='csv' writes metrics.csv via update()."""
        hist = history.History(
            backends="csv",
            outdir=str(tmp_path),
            report_enabled=False,
        )
        hist.update({"loss": 0.9, "lr": 1e-3})
        hist.update({"loss": 0.5, "lr": 1e-3})
        hist.tracker.finish()
        csv_file = tmp_path / "metrics.csv"
        assert csv_file.exists()
        contents = csv_file.read_text()
        assert "loss" in contents
        assert "lr" in contents

    def test_use_wandb_deprecation_warning(self):
        """Passing use_wandb= emits a DeprecationWarning."""
        hist = history.History()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hist.update({"loss": 0.5}, use_wandb=True)
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "use_wandb" in str(deprecation_warnings[0].message)

    def test_tracker_property(self):
        """The tracker property exposes the internal tracker."""
        backend = _FakeBackend()
        tracker = Tracker([backend])
        hist = history.History(tracker=tracker)
        assert hist.tracker.get_backend("fake") is backend

    @patch.dict(os.environ, {"EZPZ_TRACKER_BACKENDS": "csv"})
    def test_env_var_activates_tracker(self, tmp_path):
        """EZPZ_TRACKER_BACKENDS env var activates backends in History()."""
        hist = history.History(outdir=str(tmp_path), report_enabled=False)
        assert not isinstance(hist.tracker, NullTracker)
        assert hist.tracker.get_backend("csv") is not None

    def test_finalize_redirects_csv_to_outdir(self, tmp_path):
        """finalize() moves CSV backend output to the output directory."""
        hist = history.History(
            backends="csv",
            outdir=str(tmp_path / "init_dir"),
            report_enabled=False,
        )
        hist.update({"loss": 0.9})
        hist.update({"loss": 0.5})
        final_outdir = tmp_path / "final_dir"
        hist.finalize(outdir=final_outdir, save=False, plot=False)
        assert (final_outdir / "metrics.csv").exists()
        # Old location should be cleaned up
        assert not (tmp_path / "init_dir" / "metrics.csv").exists()


class TestHistoryGroups:
    """Tests for prefix-based metric grouping."""

    def test_prefixed_keys_create_separate_groups(self):
        """Metrics with different prefixes land in different groups."""
        hist = history.History()
        hist.update({"train/loss": 0.5, "train/acc": 0.8})
        hist.update({"eval/loss": 0.3, "eval/acc": 0.9})
        assert "train" in hist.groups
        assert "eval" in hist.groups
        assert "loss" in hist.groups["train"]
        assert "acc" in hist.groups["train"]
        assert "loss" in hist.groups["eval"]

    def test_unprefixed_keys_go_to_default_group(self):
        """Metrics without a prefix go to the '' (default) group."""
        hist = history.History()
        hist.update({"loss": 0.5, "dt": 0.01})
        assert "" in hist.groups
        assert "loss" in hist.groups[""]
        assert "dt" in hist.groups[""]

    def test_flattened_history_property(self):
        """The .history property returns a flat dict with full keys."""
        hist = history.History()
        hist.update({"train/loss": 0.5})
        hist.update({"eval/loss": 0.3})
        flat = hist.history
        assert "train/loss" in flat
        assert "eval/loss" in flat
        assert flat["train/loss"] == [0.5]
        assert flat["eval/loss"] == [0.3]

    def test_data_alias_matches_history(self):
        """The .data property is an alias for .history."""
        hist = history.History()
        hist.update({"loss": 0.42})
        assert hist.data == hist.history

    def test_grouped_datasets_independent_dimensions(self):
        """Each group gets its own dataset with independent draw dimension."""
        hist = history.History()
        # Train: 5 entries
        for i in range(5):
            hist.update({"train/loss": float(i), "train/iter": i})
        # Eval: 20 entries
        for i in range(20):
            hist.update({"eval/loss": float(i) * 0.1, "eval/iter": i})
        datasets = hist.get_grouped_datasets()
        assert "train" in datasets
        assert "eval" in datasets
        train_ds = datasets["train"]
        eval_ds = datasets["eval"]
        # Each dataset has its own draw dimension length
        assert train_ds["loss"].shape[0] == 5
        assert eval_ds["loss"].shape[0] == 20

    def test_split_prefix_extracts_correctly(self):
        """_split_prefix extracts the prefix and strips keys."""
        prefix, stripped = history.History._split_prefix(
            {"train/loss": 0.5, "train/acc": 0.8}
        )
        assert prefix == "train"
        assert stripped == {"loss": 0.5, "acc": 0.8}

    def test_split_prefix_no_prefix(self):
        """_split_prefix returns '' for unprefixed keys."""
        prefix, stripped = history.History._split_prefix(
            {"loss": 0.5, "acc": 0.8}
        )
        assert prefix == ""
        assert stripped == {"loss": 0.5, "acc": 0.8}

    def test_flat_dataset_still_works(self):
        """get_dataset() still returns a single flat Dataset."""
        hist = history.History()
        for i in range(5):
            hist.update({"train/loss": float(i)})
            hist.update({"eval/loss": float(i) * 0.1})
        ds = hist.get_dataset()
        assert isinstance(ds, xr.Dataset)
        # Flat dataset has both metrics (padded to max length)
        assert "train_loss" in ds.data_vars or "train/loss" in str(ds)


class TestLogMetricsCondensation:
    """log_metrics should produce a single condensed line (matching the
    History.update path), not the legacy two-line dump."""

    def _capture(self, hist, metrics, **kw):
        """Run log_metrics with a logger that captures messages."""
        import logging
        records: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        log = logging.getLogger(f"_test_capture_{id(records)}")
        log.setLevel(logging.DEBUG)
        log.handlers = [_Capture()]
        log.propagate = False
        hist.log_metrics(metrics, logger=log, **kw)
        return records

    def test_single_info_line_when_summary_included(self, monkeypatch):
        """The two-line layout (base + distributed stats) should collapse
        into ONE info line with inline ±std markers and no mean/min/max
        spam."""
        hist = history.History(backends=[])
        # Stub the distributed summary so we can verify the merge works
        # without needing a real torch distributed group.
        monkeypatch.setattr(
            hist, "_rank", 0, raising=False
        )
        monkeypatch.setattr(
            hist,
            "summarize_distributed_min_max_std",
            lambda d: {
                f"{k}/{suf}": v * mult
                for k, v in d.items()
                if isinstance(v, (int, float))
                for suf, mult in (
                    ("mean", 1.01), ("min", 0.95),
                    ("max", 1.05), ("std", 0.02),
                )
            },
        )
        metrics = {
            "iter": 1, "epoch": 0,
            "loss": 10.0, "dt": 1.0,
            "mem_alloc": 0.18, "mem_peak_alloc": 0.68,
            "mem_reserved": 1.1, "mem_peak_reserved": 1.1,
        }
        records = self._capture(hist, metrics)
        # Exactly one info line (was two before — base + dist-stats).
        assert len(records) == 1
        msg = records[0]
        # Counters bare; non-counter metrics have inline (±std).
        assert "iter=1" in msg
        assert "epoch=0" in msg
        assert "loss=10" in msg and "(±" in msg
        # Memory keys do NOT appear as `mem_alloc=0.18` in the base flow;
        # they're summarized as `memory=0.18/0.68GiB (cur/peak, …)`.
        assert "mem_alloc=" not in msg
        assert "mem_peak_alloc=" not in msg
        assert "memory=" in msg
        # No more `/mean`, `/min`, `/max` spam.
        assert "/mean" not in msg
        assert "/min" not in msg
        assert "/max" not in msg

    def test_no_distributed_stats_still_one_line(self):
        """include_summary=False: still one line, just the bare metrics."""
        hist = history.History(backends=[])
        records = self._capture(
            hist,
            {"iter": 1, "loss": 0.5},
            include_summary=False,
        )
        assert len(records) == 1
        assert "iter=1" in records[0]
        assert "loss=0.5" in records[0]
        # No std markers when summary disabled.
        assert "(±" not in records[0]

    def test_memory_suffix_omitted_when_no_mem_keys(self):
        """CPU/MPS runs: metrics have no mem_* keys → no `memory=` tail."""
        hist = history.History(backends=[])
        records = self._capture(
            hist,
            {"iter": 1, "loss": 0.5},
            include_summary=False,
        )
        assert "memory=" not in records[0]


@pytest.mark.skipif(not HISTORY_AVAILABLE, reason="ezpz.history not available")
class TestUpdateSummaryMemoryToken:
    """`History.update` must emit the `memory=…GiB` token in its summary
    string regardless of whether metric keys use a namespace prefix."""

    def test_memory_token_present_with_prefix(self):
        """Regression: when metrics are namespaced (``train/loss``,
        ``train/mem_alloc``, etc.), the summary string returned by
        ``History.update`` must still contain the collapsed
        ``memory=…GiB`` token.

        Pre-fix, ``History.update`` passed ``prefix="train"`` (no
        trailing slash) to ``format_memory_summary``, which then looked
        up ``trainmem_alloc`` and never found the key, so the memory
        token was silently dropped from the line. Every example with a
        ``"train/"`` prefix (vit, fsdp_tp, diffusion, hf) lost its
        memory readout in the console summary as a result.

        Fix: pass ``prefix=None`` and let the helper auto-detect from
        the keys it scans.
        """
        hist = history.History(backends=[])
        hist._rank = 0
        metrics = {
            "train/loss": 2.94,
            "train/mem_alloc": 1.5,
            "train/mem_peak_alloc": 2.0,
            "train/mem_reserved": 3.0,
            "train/mem_peak_reserved": 4.0,
        }
        summary = hist.update(metrics)
        # The point of the regression: `memory=` MUST appear in the
        # summary string.
        assert "memory=" in summary, (
            f"memory=...GiB token dropped from summary for prefixed metrics — "
            f"got: {summary!r}"
        )
        # And the values should be the actual alloc/peak (1.5/2.0),
        # not zeros — proves the prefix-aware lookup found the keys.
        assert "1.50/2.00GiB" in summary

    def test_memory_token_present_without_prefix(self):
        """Bare (no-prefix) metrics also produce the ``memory=`` token —
        verifies the fix didn't regress the non-prefixed case that was
        accidentally working before."""
        hist = history.History(backends=[])
        hist._rank = 0
        metrics = {
            "loss": 2.94,
            "mem_alloc": 1.5,
            "mem_peak_alloc": 2.0,
            "mem_reserved": 3.0,
            "mem_peak_reserved": 4.0,
        }
        summary = hist.update(metrics)
        assert "memory=" in summary
        assert "1.50/2.00GiB" in summary
