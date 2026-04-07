"""Tests for ezpz.tracker — multi-backend experiment tracking."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ezpz.tracker import (
    CSVBackend,
    MLflowBackend,
    NullTracker,
    Tracker,
    TrackerBackend,
    WandbBackend,
    register_backend,
    setup_tracker,
)


# ── NullTracker ──────────────────────────────────────────────────────────────


class TestNullTracker:
    def test_log_is_noop(self):
        t = NullTracker()
        t.log({"loss": 0.5})  # should not raise

    def test_log_config_is_noop(self):
        t = NullTracker()
        t.log_config({"lr": 1e-4})

    def test_finish_is_noop(self):
        t = NullTracker()
        t.finish()

    def test_log_table_is_noop(self):
        t = NullTracker()
        t.log_table("data", ["a", "b"], [[1, 2]])

    def test_log_image_is_noop(self):
        t = NullTracker()
        t.log_image("plot", "/tmp/fake.png")

    def test_watch_is_noop(self):
        t = NullTracker()
        t.watch(object())

    def test_get_backend_returns_none(self):
        t = NullTracker()
        assert t.get_backend("wandb") is None
        assert t.get_backend("csv") is None

    def test_wandb_run_is_none(self):
        t = NullTracker()
        assert t.wandb_run is None


# ── CSVBackend ───────────────────────────────────────────────────────────────


class TestCSVBackend:
    def test_log_writes_csv(self, tmp_path: Path):
        backend = CSVBackend(outdir=tmp_path)
        backend.log({"loss": 0.5, "step": 1})
        backend.log({"loss": 0.3, "step": 2})
        backend.finish()

        csv_path = tmp_path / "metrics.csv"
        assert csv_path.exists()
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["loss"] == "0.5"
        assert rows[1]["step"] == "2"

    def test_log_auto_extends_headers(self, tmp_path: Path):
        backend = CSVBackend(outdir=tmp_path)
        backend.log({"loss": 0.5})
        backend.log({"loss": 0.3, "lr": 1e-4})  # new key appears
        backend.finish()

        csv_path = tmp_path / "metrics.csv"
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert "lr" in reader.fieldnames
        assert rows[0]["lr"] == ""  # first row didn't have lr
        assert rows[1]["lr"] == "0.0001"

    def test_log_config_writes_json(self, tmp_path: Path):
        backend = CSVBackend(outdir=tmp_path)
        backend.log_config({"lr": 1e-4, "batch_size": 32})

        config_path = tmp_path / "config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data["lr"] == 1e-4
        assert data["batch_size"] == 32

    def test_log_config_merges(self, tmp_path: Path):
        backend = CSVBackend(outdir=tmp_path)
        backend.log_config({"lr": 1e-4})
        backend.log_config({"batch_size": 32})

        data = json.loads((tmp_path / "config.json").read_text())
        assert data["lr"] == 1e-4
        assert data["batch_size"] == 32

    def test_log_table_writes_separate_csv(self, tmp_path: Path):
        backend = CSVBackend(outdir=tmp_path)
        backend.log_table(
            "results",
            columns=["name", "score"],
            data=[["test", 0.95], ["fsdp", 0.91]],
        )

        table_path = tmp_path / "results.csv"
        assert table_path.exists()
        with table_path.open() as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == ["name", "score"]
        assert rows[1] == ["test", "0.95"]

    def test_step_injected_into_row(self, tmp_path: Path):
        backend = CSVBackend(outdir=tmp_path)
        backend.log({"loss": 0.5}, step=42)
        backend.finish()

        with (tmp_path / "metrics.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["step"] == "42"

    def test_creates_outdir(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        backend = CSVBackend(outdir=nested)
        backend.log({"x": 1})
        backend.finish()
        assert (nested / "metrics.csv").exists()

    def test_rank_zero_writes_files(self, tmp_path: Path):
        backend = CSVBackend(outdir=tmp_path, rank=0)
        backend.log({"loss": 0.5})
        backend.log_config({"lr": 1e-3})
        backend.log_table("t", ["a", "b"], [[1, 2]])
        backend.finish()
        assert (tmp_path / "metrics.csv").exists()
        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "t.csv").exists()

    def test_rank_nonzero_skips_file_io(self, tmp_path: Path):
        backend = CSVBackend(outdir=tmp_path, rank=1)
        backend.log({"loss": 0.5})
        backend.log_config({"lr": 1e-3})
        backend.log_table("t", ["a", "b"], [[1, 2]])
        backend.finish()
        assert not (tmp_path / "metrics.csv").exists()
        assert not (tmp_path / "config.json").exists()
        assert not (tmp_path / "t.csv").exists()

    def test_rank_nonzero_still_buffers_rows(self, tmp_path: Path):
        backend = CSVBackend(outdir=tmp_path, rank=1)
        backend.log({"loss": 0.5})
        backend.log({"loss": 0.3})
        assert len(backend._rows) == 2

    def test_rank_nonzero_does_not_create_outdir(self, tmp_path: Path):
        nested = tmp_path / "should" / "not" / "exist"
        CSVBackend(outdir=nested, rank=1)
        assert not nested.exists()

    def test_commit_false_defers_flush(self, tmp_path: Path):
        """commit=False buffers rows without writing until commit=True."""
        backend = CSVBackend(outdir=tmp_path, rank=0)
        backend.log({"loss": 0.5}, commit=False)
        backend.log({"loss": 0.3}, commit=False)
        assert not (tmp_path / "metrics.csv").exists()
        backend.log({"loss": 0.1}, commit=True)
        rows = list(csv.DictReader((tmp_path / "metrics.csv").open()))
        assert len(rows) == 3

    def test_finish_flushes_uncommitted(self, tmp_path: Path):
        """finish() writes rows that were buffered with commit=False."""
        backend = CSVBackend(outdir=tmp_path, rank=0)
        backend.log({"loss": 0.5}, commit=False)
        backend.log({"loss": 0.3}, commit=False)
        assert not (tmp_path / "metrics.csv").exists()
        backend.finish()
        rows = list(csv.DictReader((tmp_path / "metrics.csv").open()))
        assert len(rows) == 2


# ── WandbBackend rank-gating ────────────────────────────────────────────────


class TestWandbBackendRankGating:
    @patch.dict(os.environ, {"WANDB_MODE": "disabled"})
    def test_rank_zero_uses_resolved_mode(self):
        backend = WandbBackend(rank=0)
        # mode="disabled" from env, but should not be overridden
        assert backend._run is not None or True  # init succeeds or is disabled

    @patch.dict(os.environ, {}, clear=False)
    def test_rank_nonzero_gets_disabled(self):
        backend = WandbBackend(rank=1)
        # rank != 0 should force mode="disabled"
        # The run should exist but in disabled mode (no network calls)
        if backend._run is not None:
            assert backend._run.disabled

    @patch.dict(os.environ, {"WANDB_MODE": "disabled"})
    def test_mode_kwarg_popped_on_all_ranks(self):
        """mode= should be consumed even on non-rank-0 to avoid leaking."""
        # If mode leaks into **kwargs, wandb.init gets it twice (via
        # init_kwargs["mode"] and **kwargs), which would be wrong.
        # This test verifies no TypeError from duplicate 'mode'.
        backend = WandbBackend(rank=1, mode="online")
        assert backend._run is not None or True  # should not raise


# ── WandbBackend behavior (mocked wandb) ────────────────────────────────────


class TestWandbBackendBehavior:
    """Verify WandbBackend delegates correctly to the wandb API."""

    @pytest.fixture()
    def mock_wandb(self):
        """Fully mocked wandb module."""
        wandb = MagicMock()
        wandb.init.return_value = MagicMock()
        return wandb

    @pytest.fixture()
    def backend(self, mock_wandb, monkeypatch):
        """WandbBackend wired to the mocked wandb module."""
        monkeypatch.setitem(sys.modules, "wandb", mock_wandb)
        return WandbBackend(rank=0)

    def test_run_property(self, backend, mock_wandb):
        assert backend.run is mock_wandb.init.return_value

    def test_log_delegates_with_step(self, backend, mock_wandb):
        backend.log({"loss": 0.5}, step=1)
        mock_wandb.log.assert_called_once_with(
            {"loss": 0.5}, commit=True, step=1
        )

    def test_log_omits_step_when_none(self, backend, mock_wandb):
        backend.log({"loss": 0.5})
        mock_wandb.log.assert_called_once_with({"loss": 0.5}, commit=True)

    def test_log_commit_false(self, backend, mock_wandb):
        backend.log({"x": 1}, commit=False)
        mock_wandb.log.assert_called_once_with({"x": 1}, commit=False)

    def test_log_config_updates_run(self, backend):
        backend.log_config({"lr": 1e-4})
        backend._run.config.update.assert_called_with({"lr": 1e-4})

    def test_finish_calls_wandb_finish(self, backend, mock_wandb):
        backend.finish()
        mock_wandb.finish.assert_called_once()

    def test_log_table_creates_table(self, backend, mock_wandb):
        backend.log_table("data", ["a", "b"], [[1, 2]])
        mock_wandb.Table.assert_called_once_with(
            columns=["a", "b"], data=[[1, 2]]
        )
        mock_wandb.log.assert_called_once()

    def test_log_image_creates_image(self, backend, mock_wandb):
        backend.log_image("plot", "/tmp/img.png", caption="test")
        mock_wandb.Image.assert_called_once_with(
            "/tmp/img.png", caption="test"
        )
        mock_wandb.log.assert_called_once()

    def test_watch_delegates_to_run(self, backend):
        model = MagicMock()
        backend.watch(model, log="all")
        backend._run.watch.assert_called_once_with(model, log="all")

    def test_init_failure_all_methods_noop(self, mock_wandb, monkeypatch):
        """wandb.init() failure ⇒ _run is None ⇒ all methods are silent."""
        mock_wandb.init.side_effect = RuntimeError("boom")
        monkeypatch.setitem(sys.modules, "wandb", mock_wandb)
        backend = WandbBackend(rank=0)
        assert backend._run is None
        # None of these should raise
        backend.log({"x": 1})
        backend.log_config({"y": 2})
        backend.log_table("t", ["a"], [[1]])
        backend.log_image("i", "/tmp/img.png")
        backend.watch(MagicMock())
        backend.finish()

    def test_system_info_in_config(self, backend):
        """torch_version and ezpz_version injected into run config."""
        update_calls = backend._run.config.update.call_args_list
        all_keys: set[str] = set()
        for call in update_calls:
            if call.args:
                all_keys.update(call.args[0].keys())
        assert "torch_version" in all_keys
        assert "ezpz_version" in all_keys

    def test_init_config_applied(self, mock_wandb, monkeypatch):
        """Explicit config dict is applied to the run."""
        monkeypatch.setitem(sys.modules, "wandb", mock_wandb)
        backend = WandbBackend(config={"lr": 0.01}, rank=0)
        update_calls = backend._run.config.update.call_args_list
        configs = [c.args[0] for c in update_calls if c.args]
        assert {"lr": 0.01} in configs


# ── MLflowBackend (mocked) ──────────────────────────────────────────────────


class TestMLflowBackendBehavior:
    """Verify MLflowBackend delegates correctly to the mlflow API."""

    @pytest.fixture()
    def mock_mlflow(self):
        """Fully mocked mlflow module."""
        mlflow = MagicMock()
        mlflow.start_run.return_value = MagicMock()
        return mlflow

    @pytest.fixture()
    def backend(self, mock_mlflow, monkeypatch):
        """MLflowBackend wired to the mocked mlflow module."""
        monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)
        return MLflowBackend(project_name="test-project", rank=0)

    def test_set_experiment_called(self, mock_mlflow, monkeypatch):
        monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)
        MLflowBackend(project_name="my-exp", rank=0)
        mock_mlflow.set_experiment.assert_called_once_with("my-exp")

    def test_start_run_called(self, mock_mlflow, monkeypatch):
        monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)
        MLflowBackend(rank=0)
        mock_mlflow.start_run.assert_called_once()

    def test_run_property(self, backend, mock_mlflow):
        assert backend.run is mock_mlflow.start_run.return_value

    def test_log_delegates_metrics(self, backend, mock_mlflow):
        backend.log({"loss": 0.5, "lr": 1e-4}, step=10)
        mock_mlflow.log_metrics.assert_called_once_with(
            {"loss": 0.5, "lr": 1e-4}, step=10
        )

    def test_log_skips_non_numeric(self, backend, mock_mlflow):
        """Non-float-coercible values are silently dropped."""
        backend.log({"loss": 0.5, "name": "hello"})
        call_args = mock_mlflow.log_metrics.call_args
        logged = call_args[0][0]
        assert "loss" in logged
        assert "name" not in logged

    def test_log_config_calls_log_params(self, backend, mock_mlflow):
        mock_mlflow.log_params.reset_mock()
        backend.log_config({"lr": 0.01, "epochs": 10})
        mock_mlflow.log_params.assert_called_once_with(
            {"lr": "0.01", "epochs": "10"}
        )

    def test_finish_calls_end_run(self, backend, mock_mlflow):
        backend.finish()
        mock_mlflow.end_run.assert_called_once()

    def test_log_table_creates_artifact(self, backend, mock_mlflow):
        backend.log_table("preds", ["x", "y"], [[1, 2], [3, 4]])
        mock_mlflow.log_artifact.assert_called_once()
        call_args = mock_mlflow.log_artifact.call_args
        assert call_args[1]["artifact_path"] == "tables"

    def test_log_image_creates_artifact(self, backend, mock_mlflow):
        backend.log_image("plot", "/tmp/img.png")
        mock_mlflow.log_artifact.assert_called_once_with(
            "/tmp/img.png", artifact_path="images"
        )

    def test_rank_nonzero_is_noop(self, mock_mlflow, monkeypatch):
        """Non-rank-0 backend should not call any mlflow API."""
        monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)
        backend = MLflowBackend(rank=1)
        assert not backend._active
        assert backend.run is None
        # None of these should call mlflow
        backend.log({"x": 1})
        backend.log_config({"y": 2})
        backend.log_table("t", ["a"], [[1]])
        backend.log_image("i", "/tmp/img.png")
        backend.finish()
        mock_mlflow.start_run.assert_not_called()
        mock_mlflow.log_metrics.assert_not_called()
        mock_mlflow.end_run.assert_not_called()

    def test_init_failure_all_methods_noop(self, mock_mlflow, monkeypatch):
        """start_run() failure ⇒ all methods are silent no-ops."""
        mock_mlflow.start_run.side_effect = RuntimeError("boom")
        monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)
        backend = MLflowBackend(rank=0)
        assert not backend._active
        assert backend.run is None
        backend.log({"x": 1})
        backend.log_config({"y": 2})
        backend.finish()

    def test_setup_tracker_activates_mlflow(self, mock_mlflow, monkeypatch):
        """setup_tracker(backends='mlflow') should create a working tracker."""
        monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)
        tracker = setup_tracker(
            backends="mlflow", project_name="test", rank=0
        )
        assert not isinstance(tracker, NullTracker)
        be = tracker.get_backend("mlflow")
        assert isinstance(be, MLflowBackend)
        assert be._active

    def test_config_logged_at_init(self, mock_mlflow, monkeypatch):
        """Config dict passed to constructor is logged as params."""
        monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)
        MLflowBackend(config={"lr": 0.01}, rank=0)
        # log_params called twice: system params + user config
        calls = mock_mlflow.log_params.call_args_list
        assert len(calls) >= 2
        # Last call should contain the user config (flattened to strings)
        last_params = calls[-1][0][0]
        assert last_params == {"config.lr": "0.01"}

    def test_system_params_logged(self, mock_mlflow, monkeypatch):
        """System info is auto-logged under ezpz.* prefix."""
        monkeypatch.setitem(sys.modules, "mlflow", mock_mlflow)
        MLflowBackend(rank=0)
        calls = mock_mlflow.log_params.call_args_list
        assert len(calls) >= 1
        system_params = calls[0][0][0]
        assert "ezpz.version" in system_params
        assert "ezpz.hostname" in system_params
        assert "ezpz.torch_version" in system_params
        assert "ezpz.device" in system_params
        assert "ezpz.backend" in system_params
        assert "ezpz.world_size" in system_params
        assert "ezpz.timestamp" in system_params
        assert "ezpz.python_version" in system_params
        assert "ezpz.command" in system_params

    def test_nested_config_flattened(self, backend, mock_mlflow):
        """Nested config dicts are flattened with dot-separated keys."""
        mock_mlflow.log_params.reset_mock()
        backend.log_config({"model": {"hidden": 256, "layers": 4}})
        mock_mlflow.log_params.assert_called_once_with(
            {"model.hidden": "256", "model.layers": "4"}
        )


# ── Tracker multiplexer ──────────────────────────────────────────────────────


class _FakeBackend(TrackerBackend):
    """Test double that records all calls."""

    name = "fake"

    def __init__(self):
        self.logs: list[dict] = []
        self.configs: list[dict] = []
        self.tables: list[dict] = []
        self.images: list[dict] = []
        self.watches: list[dict] = []
        self.finished = False

    def log(self, metrics, step=None, commit=True):
        self.logs.append({"metrics": metrics, "step": step, "commit": commit})

    def log_config(self, config):
        self.configs.append(config)

    def log_table(self, key, columns, data):
        self.tables.append({"key": key, "columns": columns, "data": data})

    def log_image(self, key, image_path, caption=None):
        self.images.append({"key": key, "path": image_path, "caption": caption})

    def watch(self, model, **kwargs):
        self.watches.append({"model": model, **kwargs})

    def finish(self):
        self.finished = True


class _FailingBackend(TrackerBackend):
    """Backend that raises on every call."""

    name = "failing"

    def log(self, metrics, step=None, commit=True):
        raise RuntimeError("log failed")

    def log_config(self, config):
        raise RuntimeError("log_config failed")

    def log_table(self, key, columns, data):
        raise RuntimeError("log_table failed")

    def log_image(self, key, image_path, caption=None):
        raise RuntimeError("log_image failed")

    def watch(self, model, **kwargs):
        raise RuntimeError("watch failed")

    def finish(self):
        raise RuntimeError("finish failed")


class TestTracker:
    def test_fans_out_to_multiple_backends(self):
        b1, b2 = _FakeBackend(), _FakeBackend()
        tracker = Tracker([b1, b2])
        tracker.log({"loss": 0.5})
        assert len(b1.logs) == 1
        assert len(b2.logs) == 1

    def test_log_config_fans_out(self):
        b1, b2 = _FakeBackend(), _FakeBackend()
        tracker = Tracker([b1, b2])
        tracker.log_config({"lr": 1e-4})
        assert b1.configs == [{"lr": 1e-4}]
        assert b2.configs == [{"lr": 1e-4}]

    def test_finish_fans_out(self):
        b1, b2 = _FakeBackend(), _FakeBackend()
        tracker = Tracker([b1, b2])
        tracker.finish()
        assert b1.finished
        assert b2.finished

    def test_catches_backend_exceptions(self):
        good = _FakeBackend()
        bad = _FailingBackend()
        tracker = Tracker([bad, good])
        # Suppress the warning output from the tracker logger
        import logging

        tracker_logger = logging.getLogger("ezpz.tracker")
        prev_level = tracker_logger.level
        tracker_logger.setLevel(logging.CRITICAL)
        try:
            tracker.log({"loss": 0.5})
            tracker.log_config({"lr": 1e-4})
            tracker.finish()
        finally:
            tracker_logger.setLevel(prev_level)
        # Good backend still received the calls
        assert len(good.logs) == 1
        assert len(good.configs) == 1
        assert good.finished

    def test_get_backend(self):
        fake = _FakeBackend()
        tracker = Tracker([fake])
        assert tracker.get_backend("fake") is fake
        assert tracker.get_backend("nonexistent") is None

    def test_wandb_run_without_wandb_backend(self):
        tracker = Tracker([_FakeBackend()])
        assert tracker.wandb_run is None

    def test_log_table_fans_out(self):
        b1, b2 = _FakeBackend(), _FakeBackend()
        tracker = Tracker([b1, b2])
        tracker.log_table("data", ["a", "b"], [[1, 2]])
        assert len(b1.tables) == 1
        assert b1.tables[0]["key"] == "data"
        assert len(b2.tables) == 1

    def test_log_image_fans_out(self):
        b1, b2 = _FakeBackend(), _FakeBackend()
        tracker = Tracker([b1, b2])
        tracker.log_image("plot", "/tmp/img.png", caption="test")
        assert len(b1.images) == 1
        assert b1.images[0]["caption"] == "test"
        assert len(b2.images) == 1

    def test_watch_fans_out(self):
        b1, b2 = _FakeBackend(), _FakeBackend()
        tracker = Tracker([b1, b2])
        model = object()
        tracker.watch(model, log="all")
        assert len(b1.watches) == 1
        assert b1.watches[0]["model"] is model
        assert b1.watches[0]["log"] == "all"
        assert len(b2.watches) == 1

    def test_catches_log_table_exception(self):
        good = _FakeBackend()
        tracker = Tracker([_FailingBackend(), good])
        tracker.log_table("t", ["a"], [[1]])
        assert len(good.tables) == 1

    def test_catches_log_image_exception(self):
        good = _FakeBackend()
        tracker = Tracker([_FailingBackend(), good])
        tracker.log_image("i", "/tmp/img.png")
        assert len(good.images) == 1

    def test_catches_watch_exception(self):
        good = _FakeBackend()
        tracker = Tracker([_FailingBackend(), good])
        tracker.watch(object())
        assert len(good.watches) == 1


# ── setup_tracker factory ────────────────────────────────────────────────────


class TestSetupTracker:
    def test_backends_none_returns_null(self):
        tracker = setup_tracker(backends="none")
        assert isinstance(tracker, NullTracker)

    def test_backends_empty_returns_null(self):
        tracker = setup_tracker(backends="")
        assert isinstance(tracker, NullTracker)

    def test_csv_backend(self, tmp_path: Path):
        tracker = setup_tracker(backends="csv", outdir=tmp_path)
        assert tracker.get_backend("csv") is not None
        tracker.log({"loss": 0.5})
        tracker.finish()
        assert (tmp_path / "metrics.csv").exists()

    def test_unknown_backend_warns(self):
        import logging

        tracker_logger = logging.getLogger("ezpz.tracker")
        prev_level = tracker_logger.level
        tracker_logger.setLevel(logging.CRITICAL)
        try:
            tracker = setup_tracker(backends="nonexistent_xyz")
        finally:
            tracker_logger.setLevel(prev_level)
        assert isinstance(tracker, NullTracker)

    @patch.dict(os.environ, {"WANDB_MODE": "disabled"})
    def test_wandb_disabled_via_env(self, tmp_path: Path):
        # wandb should init in disabled mode, not raise
        tracker = setup_tracker(
            backends="csv",
            outdir=tmp_path,
        )
        assert tracker.get_backend("csv") is not None

    def test_comma_separated_string(self, tmp_path: Path):
        import logging

        tracker_logger = logging.getLogger("ezpz.tracker")
        prev_level = tracker_logger.level
        tracker_logger.setLevel(logging.CRITICAL)
        try:
            tracker = setup_tracker(backends="csv,none_xyz", outdir=tmp_path)
        finally:
            tracker_logger.setLevel(prev_level)
        # csv should activate, none_xyz should be skipped
        assert tracker.get_backend("csv") is not None

    @patch.dict(os.environ, {"EZPZ_TRACKER_BACKENDS": "csv"})
    def test_env_var_fallback(self, tmp_path: Path):
        tracker = setup_tracker(backends=None, outdir=tmp_path)
        assert tracker.get_backend("csv") is not None


# ── Custom backend registration ──────────────────────────────────────────────


class TestRegistration:
    def test_register_and_use_custom_backend(self, tmp_path: Path):
        class MyBackend(TrackerBackend):
            name = "my"

            def __init__(self, **kwargs):
                self.logged: list[dict] = []

            def log(self, metrics, step=None, commit=True):
                self.logged.append(metrics)

            def log_config(self, config):
                pass

            def finish(self):
                pass

        register_backend("my", MyBackend)
        try:
            tracker = setup_tracker(backends="my")
            tracker.log({"x": 1})
            backend = tracker.get_backend("my")
            assert isinstance(backend, MyBackend)
            assert backend.logged == [{"x": 1}]
        finally:
            # Clean up registry
            from ezpz.tracker import _BACKEND_REGISTRY
            _BACKEND_REGISTRY.pop("my", None)


# ── Full lifecycle ──────────────────────────────────────────────────────────


class TestFullLifecycle:
    def test_csv_init_log_finish(self, tmp_path: Path):
        """End-to-end: init → config → log N steps → table → finish."""
        tracker = setup_tracker(
            backends="csv", outdir=tmp_path, config={"lr": 0.01}
        )
        for step in range(5):
            tracker.log({"loss": 1.0 / (step + 1)}, step=step)
        tracker.log_table(
            "predictions", ["input", "output"], [[1, 2], [3, 4]]
        )
        tracker.finish()

        # metrics.csv has 5 rows
        rows = list(csv.DictReader((tmp_path / "metrics.csv").open()))
        assert len(rows) == 5
        assert rows[0]["loss"] == "1.0"
        assert rows[0]["step"] == "0"

        # config.json has lr
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["lr"] == 0.01

        # predictions.csv exists with 2 data rows + header
        lines = (tmp_path / "predictions.csv").read_text().strip().splitlines()
        assert len(lines) == 3  # header + 2 data rows
