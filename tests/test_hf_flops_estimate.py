"""``hf`` must not run a full-length real forward to count FLOPs.

#264: `ezpz benchmark` could not finish `hf` on any system. The cause was
``try_estimate`` at what is now the FLOP-estimation block -- it runs a
**real** forward+backward under ``FlopCounterMode``, and at
``block_size=2048`` on Llama-3.2-1B that is ~24 s single-threaded on a
fast laptop core. On a compute node with 8-12 ranks contending it
produced **>57 minutes of silence** (Perlmutter job ``59023958``), and
three separate runs died there: SIGKILL at the NERSC ``debug`` 30-minute
cap, ``exit=124`` on Polaris, and an unfinished 90-minute allocation.

``try_estimate_fake`` uses ``FakeTensorMode``: measured 0.6 s vs 23.7 s
on a 1.5B-param model, with the counts agreeing to ~11%. ``fsdp_tp``
already used the fake path, which is why it completes everywhere while
``hf`` did not.

These tests pin the ordering. They do not build a 1B model -- they assert
the call structure, which is what regressed.
"""

from __future__ import annotations

import pathlib

import pytest


def _hf_source() -> str:
    """Read hf.py from disk rather than importing it.

    ``inspect.getsource`` would need the module imported, which drags in
    ``accelerate`` and ``transformers``. These assertions are about the
    call structure, so the file text is both sufficient and more robust.
    """
    import ezpz

    path = pathlib.Path(ezpz.__file__).parent / "examples" / "hf.py"
    return path.read_text()


def test_fake_estimate_is_used():
    """The fake-tensor path must be the primary estimator."""
    assert "try_estimate_fake(" in _hf_source(), (
        "hf must estimate FLOPs via FakeTensorMode; a real full-length "
        "forward took >57 min on a compute node (#264)"
    )


def test_real_estimate_is_never_called_at_full_block_size():
    """A real probe is allowed, but only at a short sequence.

    The pathology is not ``try_estimate`` itself -- it is calling it with
    the full ``block_size``. The fallback must cap the probe length and
    scale the result.
    """
    src = _hf_source()
    assert (
        "try_estimate(\n            model, (training_args.per_device_train_batch_size, block_size)"
        not in src
    ), "real FLOP estimation at full block_size is the #264 pathology"
    if "try_estimate(" in src:
        assert "_probe_seq" in src, (
            "a real-tensor fallback must probe at a capped sequence "
            "length (see _probe_seq) rather than the full block_size"
        )
        assert "min(128, block_size)" in src, (
            "the probe sequence should be capped at 128, matching fsdp_tp"
        )


def test_fake_is_attempted_before_real():
    """Ordering matters: fake first, real only when fake returns 0."""
    src = _hf_source()
    i_fake = src.index("try_estimate_fake(")
    # The real call, if present, must come after the fake one.
    tail = src[i_fake:]
    if "try_estimate(" in tail:
        # and must be guarded by a check on the fake result
        assert "_model_flops <= 0" in tail, (
            "the real probe must run only when the fake estimate failed"
        )


def test_silent_window_has_progress_logging():
    """#264 was reported as a hang because 100 lines logged nothing.

    Between ``group_texts done`` and ``accelerator.prepare()`` there was
    no output at all, so a slow step was indistinguishable from a
    deadlock. Pin the marks that make it legible.
    """
    src = _hf_source()
    for marker in (
        "building dataloaders",
        "building optimizer",
        "estimating model FLOPs",
    ):
        assert marker in src, (
            f"missing progress log {marker!r}: the window between "
            "group_texts and accelerator.prepare() must not be silent"
        )


def test_flops_helpers_agree_on_a_small_model():
    """Sanity-check that the fake count is usable, not just fast.

    Small enough to run in CI; the 1.5B measurement (1.68e13 vs 1.52e13,
    ~11% apart) is in the module docstring.
    """
    torch = pytest.importorskip("torch")
    from ezpz.flops import try_estimate, try_estimate_fake

    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
    )
    real = try_estimate(model, (4, 128))
    fake = try_estimate_fake(model, (4, 128))
    if real > 0 and fake > 0:
        ratio = fake / real
        assert 0.5 < ratio < 2.0, (
            f"fake estimate {fake:.3e} diverges from real {real:.3e}"
        )
