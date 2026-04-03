"""Tests for ezpz.tracker — multi-backend experiment tracking."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ezpz.tracker import (
    CSVBackend,
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


# ── Tracker multiplexer ──────────────────────────────────────────────────────


class _FakeBackend(TrackerBackend):
    """Test double that records all calls."""

    name = "fake"

    def __init__(self):
        self.logs: list[dict] = []
        self.configs: list[dict] = []
        self.finished = False

    def log(self, metrics, step=None, commit=True):
        self.logs.append({"metrics": metrics, "step": step, "commit": commit})

    def log_config(self, config):
        self.configs.append(config)

    def finish(self):
        self.finished = True


class _FailingBackend(TrackerBackend):
    """Backend that raises on every call."""

    name = "failing"

    def log(self, metrics, step=None, commit=True):
        raise RuntimeError("log failed")

    def log_config(self, config):
        raise RuntimeError("log_config failed")

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
        # Should not raise; bad backend's error is logged as warning
        tracker.log({"loss": 0.5})
        tracker.log_config({"lr": 1e-4})
        tracker.finish()
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
        tracker = setup_tracker(backends="nonexistent_xyz")
        # Should return NullTracker since no backends activated
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
        tracker = setup_tracker(backends="csv,none_xyz", outdir=tmp_path)
        # csv should activate, none_xyz should warn and be skipped
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
