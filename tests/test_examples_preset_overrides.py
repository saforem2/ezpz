"""Regression tests for the `--flag=value` preset-override contract.

PR #166's b2c0b67 fixed `_arg_provided` in `ezpz.examples.fsdp_tp` so
that `ezpz launch ... --model agpt-2b --seq-len=8192` honored the
explicit `--seq-len` override instead of letting the preset's value
silently win. The other four example modules had copy-pasted local
copies of `_arg_provided` with the original broken
``flag in argv`` check, so the same bug persisted for them — caught
only when a user passed `--model xxxl --batch-size=32` to
`ezpz.examples.test` and got `batch_size=8` (the preset value).

The fix consolidated `_arg_provided` into `ezpz.examples._presets.arg_provided`,
imported by all 5 examples. This test pins the contract end-to-end:
both space-separated and ``=``-fused argv shapes must override the
preset across every example.
"""

from __future__ import annotations

import importlib

import pytest


_EXAMPLES = [
    ("ezpz.examples.test", "batch_size", "--batch-size", 32),
    ("ezpz.examples.fsdp", "batch_size", "--batch-size", 64),
    ("ezpz.examples.vit", "batch_size", "--batch_size", 2),
    ("ezpz.examples.fsdp_tp", "batch_size", "--batch-size", 2),
    ("ezpz.examples.diffusion", "batch_size", "--batch-size", 2),
]


def _load(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"could not import {module_name}: {exc}")


@pytest.mark.parametrize(
    "module_name,attr,flag,explicit",
    _EXAMPLES,
    ids=[m for m, *_ in _EXAMPLES],
)
class TestPresetOverrideContract:
    """Each example must honor `--flag value` AND `--flag=value`
    overrides against the `xxxl` preset's value for the same field."""

    def test_space_separated_form_overrides_preset(
        self, module_name, attr, flag, explicit
    ):
        mod = _load(module_name)
        args = mod.parse_args(
            ["--model", "xxxl", flag, str(explicit)]
        )
        assert getattr(args, attr) == explicit, (
            f"{module_name} {flag} {explicit}: got "
            f"{attr}={getattr(args, attr)}, expected {explicit}. "
            "Preset value silently won."
        )

    def test_equals_fused_form_overrides_preset(
        self, module_name, attr, flag, explicit
    ):
        """This was the regression: `--batch-size=32` (one argv token)
        used to fail `flag in argv` and silently let the preset win."""
        mod = _load(module_name)
        args = mod.parse_args(
            ["--model", "xxxl", f"{flag}={explicit}"]
        )
        assert getattr(args, attr) == explicit, (
            f"{module_name} {flag}={explicit}: got "
            f"{attr}={getattr(args, attr)}, expected {explicit}. "
            "Preset value silently won — the `--flag=value` argv "
            "shape must beat the preset."
        )


def test_shared_arg_provided_exists_and_handles_both_shapes():
    """The shared helper itself: both argv shapes must register."""
    from ezpz.examples._presets import arg_provided

    flags = ["--batch-size", "--batch_size"]
    assert arg_provided(["--batch-size", "32"], flags) is True
    assert arg_provided(["--batch-size=32"], flags) is True
    assert arg_provided(["--batch_size=32"], flags) is True  # underscore-eq
    assert arg_provided(["--model", "xxxl"], flags) is False
    # Edge case: a flag that *starts with* but isn't `--batch-size`
    # must NOT match. e.g. `--batch-size-foo=1` should not register
    # as `--batch-size`.
    assert arg_provided(
        ["--batch-size-foo=1"], ["--batch-size"]
    ) is False, "prefix match must require `--batch-size=`, not just `--batch-size`"
