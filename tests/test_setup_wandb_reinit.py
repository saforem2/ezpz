"""Tests that ``setup_wandb`` survives being called twice in one job.

Regression guard for a bug shipped in v0.24.2: ``fsdp_tp`` began
creating its W&B run early (in ``main()``), so ``setup_wandb`` ran a
second time when ``History``'s ``WandbBackend`` later adopted the run.
The post-init ``run.config.update({...})`` writes wall-clock keys
(``tstamp``, ``year``/``month``/``day``) and passed no
``allow_val_change``, so the second call raised::

    ConfigError: Attempted to change value of key "tstamp"
                 from 2026-08-05T16:56:09 to 2026-08-05T16:56:57

The blanket ``except Exception`` then swallowed it and returned
``None``, which made ``WandbBackend._run = None`` -- and
``WandbBackend.log()`` early-returns on that. Net effect: a run visible
in the W&B UI that silently received **zero metrics** for the entire
job. Strictly worse than the startup lag the early init was meant to
fix.

Two independent guards are pinned here:
  1. the config update tolerates value changes, so no raise; and
  2. even if post-init bookkeeping *does* raise, an already-live run is
     returned rather than ``None``.

Runs fully offline (``WANDB_MODE=offline``); no account needed.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def wandb_offline(monkeypatch, tmp_path):
    """Force wandb fully offline and hand back a clean module."""
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    # Never inherit a run leaked by another test.
    if getattr(wandb, "run", None) is not None:
        wandb.finish()
    yield wandb
    if getattr(wandb, "run", None) is not None:
        wandb.finish()


def _setup_wandb():
    mod = importlib.import_module("ezpz.distributed")
    return mod.setup_wandb


class TestDoubleInit:
    """setup_wandb() called twice must yield one live, logging run."""

    def test_second_call_returns_live_run(self, wandb_offline, tmp_path):
        setup_wandb = _setup_wandb()
        first = setup_wandb(project_name="p", dir=str(tmp_path), mode="offline")
        if first is None:
            pytest.skip("wandb unavailable in this environment")
        second = setup_wandb(
            project_name="p", dir=str(tmp_path), mode="offline"
        )
        assert second is not None, (
            "setup_wandb returned None on re-entry. WandbBackend sets "
            "_run=None from that, and WandbBackend.log() early-returns "
            "on _run=None -- the job would log NO metrics at all."
        )
        assert second is first, "expected the same run object, not a new one"

    def test_second_call_does_not_raise_on_wall_clock_keys(
        self, wandb_offline, tmp_path
    ):
        """`tstamp` necessarily differs between calls; that must be fine."""
        setup_wandb = _setup_wandb()
        first = setup_wandb(project_name="p", dir=str(tmp_path), mode="offline")
        if first is None:
            pytest.skip("wandb unavailable in this environment")
        before = dict(first.config).get("tstamp")
        # No sleep needed: the guard is allow_val_change, and datetime
        # resolution makes a differing value overwhelmingly likely. The
        # contract under test is "does not raise" either way.
        second = setup_wandb(
            project_name="p", dir=str(tmp_path), mode="offline"
        )
        assert second is not None
        after = dict(second.config).get("tstamp")
        assert after is not None
        if before != after:
            # Re-stamping is the intended behaviour when it does change.
            assert isinstance(after, str)

    def test_logging_works_after_second_call(self, wandb_offline, tmp_path):
        """The end-to-end symptom: metrics must actually reach the run."""
        setup_wandb = _setup_wandb()
        if setup_wandb(project_name="p", dir=str(tmp_path), mode="offline") is None:
            pytest.skip("wandb unavailable in this environment")
        run = setup_wandb(project_name="p", dir=str(tmp_path), mode="offline")
        assert run is not None
        run.log({"loss": 1.23})  # must not raise

    def test_config_kwarg_survives_second_call(self, wandb_offline, tmp_path):
        """The `config=` branch updates the same key twice across calls."""
        setup_wandb = _setup_wandb()
        first = setup_wandb(
            project_name="p", dir=str(tmp_path), mode="offline",
            config={"lr": 0.001},
        )
        if first is None:
            pytest.skip("wandb unavailable in this environment")
        second = setup_wandb(
            project_name="p", dir=str(tmp_path), mode="offline",
            config={"lr": 0.003},  # different value, same key
        )
        assert second is not None, (
            "re-supplying config= with a changed value must not kill the run"
        )


class TestLiveRunSurvivesPostInitFailure:
    """Defence in depth, independent of the allow_val_change fix.

    Scope matters: a failure of ``wandb.init()`` *itself* leaves no
    usable run and must still return ``None`` (pinned by
    ``test_tracker.py::test_init_failure_all_methods_noop``). Only
    *post-init* bookkeeping on an already-created run is recoverable.
    """

    def test_post_init_exception_returns_existing_run(
        self, wandb_offline, tmp_path, monkeypatch
    ):
        setup_wandb = _setup_wandb()
        first = setup_wandb(project_name="p", dir=str(tmp_path), mode="offline")
        if first is None:
            pytest.skip("wandb unavailable in this environment")

        # Force ANY post-init bookkeeping to blow up.
        import ezpz.distributed as dmod

        def _boom():
            raise RuntimeError("simulated post-init failure")

        monkeypatch.setattr(dmod, "get_hostname", _boom)

        second = setup_wandb(
            project_name="p", dir=str(tmp_path), mode="offline"
        )
        assert second is not None, (
            "a live run must be returned even when post-init bookkeeping "
            "fails; returning None silently disables all metric logging"
        )
        second.log({"loss": 0.5})  # still usable

    def test_init_failure_itself_still_returns_none(
        self, wandb_offline, tmp_path, monkeypatch
    ):
        """The recovery must NOT extend to wandb.init() failing.

        With no usable run there is nothing to salvage, and callers
        (WandbBackend) rely on None to fall back to a silent no-op.
        Guards against 'fixing' the post-init case by broadening the
        outer handler.
        """
        setup_wandb = _setup_wandb()
        import ezpz.distributed as dmod

        monkeypatch.setattr(dmod, "verify_wandb", lambda: True)
        monkeypatch.setattr(
            wandb_offline,
            "init",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert (
            setup_wandb(project_name="p", dir=str(tmp_path), mode="offline")
            is None
        )
