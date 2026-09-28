"""``ezpz.examples.profiler`` is the documented profiling entry point.

It is referenced verbatim from the ALCF Aurora docs
(argonne-lcf/user-guides#1183), so the surface these tests pin is a
contract with an external page, not just an internal example:

* ``--profile`` is the flag that turns profiling on (it sets
  ``pytorch_profiler``; there is no ``--pytorch-profiler`` flag);
* without it, the context is a ``nullcontext`` yielding ``None``, so the
  unprofiled path stays free of overhead;
* ``prof.step()`` is called once per training step -- the schedule only
  advances when it is, and omitting it writes no trace at all while
  raising nothing.

Everything here runs on CPU: the XPU/CUDA branch lives in
:func:`ezpz.profile.get_torch_profiler`, which picks activities from what
is available, so CI on CPU still catches a regression in the wiring.
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest

torch = pytest.importorskip("torch")

from ezpz.examples.profiler import parse_args  # noqa: E402


def test_profile_flag_sets_pytorch_profiler():
    """``--profile`` is the documented spelling.

    The dest is ``pytorch_profiler`` but the *flag* is ``--profile``;
    documenting the wrong one sends readers to an argparse error.
    """
    assert parse_args(["--profile"]).pytorch_profiler is True


def test_pytorch_profiler_flag_does_not_exist():
    """Guard the spelling above by pinning the negative case."""
    with pytest.raises(SystemExit):
        parse_args(["--pytorch-profiler"])


def test_profiling_is_off_by_default():
    """No flag: profiling must not engage."""
    args = parse_args([])
    assert args.pytorch_profiler is False
    assert args.pyinstrument_profiler is False


def test_schedule_defaults_match_documented_values(tmp_path):
    """The docs state wait=1, warmup=2, active=3, repeat=5.

    The example's warning text does the arithmetic (1+2+3=6 steps before
    a first trace), so a change to these defaults silently makes that
    message wrong.
    """
    args = parse_args([])
    assert args.pytorch_profiler_wait == 1
    assert args.pytorch_profiler_warmup == 2
    assert args.pytorch_profiler_active == 3
    assert args.pytorch_profiler_repeat == 5


def test_context_is_nullcontext_without_the_flag(tmp_path):
    """Unprofiled runs must yield ``None``, not a profiler."""
    from ezpz.profile import profiling_context_from_args

    ctx = profiling_context_from_args(parse_args([]), tmp_path)
    assert isinstance(ctx, nullcontext)
    with ctx as prof:
        assert prof is None


def test_train_calls_prof_step_once_per_step(tmp_path, monkeypatch):
    """``prof.step()`` must be called once per training step.

    This is the failure that has no error message: a loop that never
    calls ``step()`` leaves the schedule in ``wait`` forever and writes
    no trace, looking like a broken profiler rather than a missing call.
    """
    import ezpz.examples.profiler as mod

    calls = {"n": 0}

    class _FakeProf:
        def step(self) -> None:
            calls["n"] += 1

    class _FakeCtx:
        def __enter__(self):
            return _FakeProf()

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        mod, "profiling_context_from_args", lambda *a, **k: _FakeCtx()
    )

    args = parse_args(["--steps", "7", "--hidden-size", "8", "--layers", "1"])
    # Put the model where train() will put its inputs: on this host that
    # may be mps/cuda, and a bare CPU Linear would raise a device
    # mismatch that has nothing to do with what is under test.
    import ezpz

    model = torch.nn.Linear(8, 8).to(ezpz.get_torch_device_type())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    mod.train(model, optimizer, args, tmp_path)

    assert calls["n"] == 7, (
        f"expected one prof.step() per training step, got {calls['n']}"
    )


def test_train_runs_without_a_profiler(tmp_path, monkeypatch):
    """The ``prof is None`` branch must not raise.

    Pins the guard: calling ``.step()`` on the ``nullcontext`` target
    would be an ``AttributeError`` on every unprofiled run.
    """
    import ezpz.examples.profiler as mod

    monkeypatch.setattr(
        mod, "profiling_context_from_args", lambda *a, **k: nullcontext()
    )
    args = parse_args(["--steps", "3", "--hidden-size", "8", "--layers", "1"])
    import ezpz

    model = torch.nn.Linear(8, 8).to(ezpz.get_torch_device_type())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    mod.train(model, optimizer, args, tmp_path)  # must not raise
