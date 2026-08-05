"""Tests that ezpz.examples.fsdp_tp creates its W&B run EARLY.

The run used to be created by the ``History`` constructor inside
``train()`` — i.e. only after tokenization, model build, FLOP counting,
FSDP wrapping and ``torch.compile``. On a 4-node agpt-2b run that is
~60s of startup that W&B never saw, and a job that died during model
build (the common failure at 20b) uploaded nothing at all.

``main()`` now calls ``setup_wandb`` immediately after the outdir is
resolved. These tests pin the *ordering* contract (wandb before train)
and the rank gate, without needing an accelerator, a process group, or
a real wandb account.

CPU-only: every heavy collaborator in ``main()`` is monkeypatched.
"""

from __future__ import annotations

import argparse
import importlib
from contextlib import contextmanager

import pytest


def _import_fsdp_tp():
    try:
        return importlib.import_module("ezpz.examples.fsdp_tp")
    except Exception as exc:  # heavy optional deps may be missing
        pytest.skip(f"could not import ezpz.examples.fsdp_tp: {exc}")


class _FakeTracker:
    def log(self, *_a, **_kw):
        return None


class _FakeHistory:
    def __init__(self):
        self.tracker = _FakeTracker()

    def finalize(self, *_a, **_kw):
        return None


def _run_main(m, monkeypatch, *, rank: int, events: list[str]):
    """Drive ``main()`` with every heavy collaborator stubbed out.

    Appends a marker to ``events`` as each stub fires so the test can
    assert the call ORDER, which is the whole point of the change.
    """
    import ezpz

    monkeypatch.setattr(ezpz, "silence_noisy_loggers", lambda *a, **k: None)
    monkeypatch.setattr(
        ezpz.distributed, "setup_torch", lambda *a, **k: rank
    )

    def _fake_outdir(*_a, **_kw):
        events.append("outdir")
        return tmp_outdir

    monkeypatch.setattr(m, "get_example_outdir", _fake_outdir)

    captured: dict[str, object] = {}

    def _fake_setup_wandb(*_a, **kw):
        events.append("wandb")
        captured.update(kw)
        return object()

    # main() resolves this as an attribute on the ezpz module at call
    # time, so patching the module attribute is what the code sees.
    monkeypatch.setattr(ezpz, "setup_wandb", _fake_setup_wandb)

    @contextmanager
    def _fake_prof(*_a, **_kw):
        yield None

    monkeypatch.setattr(m, "profiling_context_from_args", _fake_prof)

    def _fake_train(*_a, **_kw):
        events.append("train")
        return _FakeHistory()

    monkeypatch.setattr(m, "train", _fake_train)

    args = argparse.Namespace(tp=1, seed=None, outdir=None)
    m.main(args)
    return captured


# Module-level so the outdir stub can close over a stable value.
tmp_outdir = "/tmp/ezpz-test-early-wandb"


class TestEarlyWandbInit:
    """W&B must exist before train() runs — that is the regression."""

    def test_wandb_initialized_before_train(self, monkeypatch):
        m = _import_fsdp_tp()
        events: list[str] = []
        _run_main(m, monkeypatch, rank=0, events=events)
        assert "wandb" in events, "setup_wandb was never called on rank 0"
        assert "train" in events, "train() was never reached"
        assert events.index("wandb") < events.index("train"), (
            "W&B run must be created BEFORE train(); got order "
            f"{events!r}. This is the whole point of the early-init "
            "change — reverting to History-time init loses ~60s of "
            "startup (tokenization, model build, compile) and uploads "
            "nothing at all for a job that dies during build."
        )

    def test_wandb_initialized_after_outdir_resolved(self, monkeypatch):
        """dir= must point at a real resolved outdir, so ordering matters."""
        m = _import_fsdp_tp()
        events: list[str] = []
        _run_main(m, monkeypatch, rank=0, events=events)
        assert events.index("outdir") < events.index("wandb")

    def test_wandb_receives_project_and_dir(self, monkeypatch):
        m = _import_fsdp_tp()
        events: list[str] = []
        captured = _run_main(m, monkeypatch, rank=0, events=events)
        assert captured.get("project_name") == m.WBPROJ_NAME
        assert str(captured.get("dir")) == tmp_outdir

    def test_no_wandb_on_nonzero_rank(self, monkeypatch):
        """Rank gate: only rank 0 initialises (setup_wandb also self-gates,
        but the explicit check keeps N-1 ranks out of the call entirely)."""
        m = _import_fsdp_tp()
        events: list[str] = []
        _run_main(m, monkeypatch, rank=3, events=events)
        assert "wandb" not in events, "non-zero rank must not init wandb"
        assert "train" in events, "non-zero ranks still train"

    def test_outdir_resolved_on_every_rank(self, monkeypatch):
        """get_example_outdir broadcasts a timestamp — it is collective, so
        it must stay OUTSIDE the rank-0 gate or non-zero ranks hang."""
        m = _import_fsdp_tp()
        events: list[str] = []
        _run_main(m, monkeypatch, rank=3, events=events)
        assert "outdir" in events, (
            "get_example_outdir() must run on ALL ranks — it broadcasts "
            "the shared run timestamp. Moving it inside the rank-0 gate "
            "would hang every other rank."
        )
