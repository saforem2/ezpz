"""Tests for the ``--compile`` / ``--compile-mode`` flags across ezpz.examples.

Every example exposes a ``parse_args`` callable that builds an
``argparse.Namespace``. Per the contract added in PR #166:

1. ``--compile`` is a boolean flag, default ``False``.
2. ``--compile-mode`` defaults to ``"default"`` and accepts the three
   torch.compile modes: ``default``, ``reduce-overhead``,
   ``max-autotune``.
3. ``argparse`` rejects unknown ``--compile-mode`` values.
4. The single-token form (``--compile-mode=max-autotune``) parses
   identically to the two-token form — pins the regression around the
   ``_arg_provided=`` bug class that historically broke ``--flag=value``.

The same contract holds for ``ezpz.cli.flags.build_test_parser`` (the
shared parser used by ``ezpz test`` / ``ezpz benchmark``). Both
parsers are exercised where they exist so the flag stays in sync.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest


# ---------------------------------------------------------------------
# Example modules under test. Each exposes ``parse_args(argv)``.
# ---------------------------------------------------------------------

_EXAMPLES = [
    "ezpz.examples.vit",
    "ezpz.examples.fsdp",
    "ezpz.examples.fsdp_tp",
    "ezpz.examples.diffusion",
    "ezpz.examples.test",
]


_COMPILE_MODES = ("default", "reduce-overhead", "max-autotune")


def _load(module_name: str) -> Any:
    """Import an example module, skipping the test if it can't load.

    Mirrors the helper in ``tests/test_example_model_sizes.py``: the
    examples import torch / FSDP etc., and on a CI box without those
    deps the import fails. We don't want to gate the CLI-contract
    tests on the heavy deps, so skip cleanly instead of erroring.
    """
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"could not import {module_name}: {exc}")


def _parse(module_name: str, argv: list[str]) -> Any:
    """Run a module's ``parse_args`` with ``argv`` and return the
    resulting ``Namespace``. Skips if the module can't be imported."""
    mod = _load(module_name)
    parse_args = getattr(mod, "parse_args", None)
    if parse_args is None:
        pytest.skip(f"{module_name} has no parse_args()")
    return parse_args(argv)


def _build_test_parser_or_skip() -> Any:
    """Build the shared ``ezpz.cli.flags.build_test_parser`` parser
    or skip if the helper isn't there. ``ezpz.examples.test`` may
    delegate to this parser, in which case both should expose the
    same flags."""
    try:
        flags = importlib.import_module("ezpz.cli.flags")
    except Exception as exc:
        pytest.skip(f"could not import ezpz.cli.flags: {exc}")
    build_test_parser = getattr(flags, "build_test_parser", None)
    if build_test_parser is None:
        pytest.skip("ezpz.cli.flags.build_test_parser missing")
    return build_test_parser()


# =====================================================================
# Default values: compile is opt-in (False), compile_mode is "default"
# =====================================================================


@pytest.mark.parametrize("module_name", _EXAMPLES, ids=_EXAMPLES)
def test_compile_flag_defaults_to_false(module_name):
    """Without ``--compile`` on argv, ``args.compile`` must be False.

    This is the safety net — silently enabling ``torch.compile`` on
    every example would change steady-state perf and break checkpoint
    compat for users that rely on eager mode.
    """
    args = _parse(module_name, [])
    assert hasattr(args, "compile"), (
        f"{module_name}.parse_args() did not register a 'compile' "
        f"attribute on the Namespace."
    )
    assert args.compile is False, (
        f"{module_name}: args.compile default = {args.compile!r}, "
        f"expected False. torch.compile must be opt-in."
    )


@pytest.mark.parametrize("module_name", _EXAMPLES, ids=_EXAMPLES)
def test_compile_mode_default_is_default(module_name):
    """``--compile-mode`` must default to ``"default"`` — the
    plain-vanilla torch.compile mode. Anything else (e.g.
    ``reduce-overhead``) has side-effects (cudagraphs) that
    shouldn't be silently active."""
    args = _parse(module_name, [])
    assert hasattr(args, "compile_mode"), (
        f"{module_name}.parse_args() did not register a "
        f"'compile_mode' attribute on the Namespace."
    )
    assert args.compile_mode == "default", (
        f"{module_name}: args.compile_mode default = "
        f"{args.compile_mode!r}, expected 'default'."
    )


# =====================================================================
# Enabling: --compile flips the bool; --compile-mode picks the mode.
# =====================================================================


@pytest.mark.parametrize("module_name", _EXAMPLES, ids=_EXAMPLES)
def test_compile_can_be_enabled(module_name):
    """``--compile`` alone flips compile to True (mode stays default).
    Pairing with ``--compile-mode reduce-overhead`` flips compile AND
    picks the mode. This is the user-facing contract."""
    args = _parse(module_name, ["--compile"])
    assert args.compile is True, (
        f"{module_name}: --compile did not set args.compile=True "
        f"(got {args.compile!r})."
    )
    assert args.compile_mode == "default", (
        f"{module_name}: --compile alone should leave compile_mode "
        f"at 'default', got {args.compile_mode!r}."
    )

    args2 = _parse(
        module_name, ["--compile", "--compile-mode", "reduce-overhead"]
    )
    assert args2.compile is True, (
        f"{module_name}: --compile + --compile-mode … did not set "
        f"args.compile=True (got {args2.compile!r})."
    )
    assert args2.compile_mode == "reduce-overhead", (
        f"{module_name}: --compile-mode reduce-overhead did not "
        f"propagate (got {args2.compile_mode!r})."
    )


@pytest.mark.parametrize("module_name", _EXAMPLES, ids=_EXAMPLES)
@pytest.mark.parametrize("mode", _COMPILE_MODES, ids=_COMPILE_MODES)
def test_compile_mode_accepts_all_three(module_name, mode):
    """Each of the three valid torch.compile modes must parse
    cleanly and propagate to ``args.compile_mode``."""
    args = _parse(module_name, ["--compile", "--compile-mode", mode])
    assert args.compile_mode == mode, (
        f"{module_name}: --compile-mode {mode} did not propagate "
        f"(got {args.compile_mode!r})."
    )


@pytest.mark.parametrize("module_name", _EXAMPLES, ids=_EXAMPLES)
def test_compile_mode_rejects_unknown(module_name):
    """argparse must reject an unknown ``--compile-mode`` rather than
    silently fall through. ``torch.compile`` accepts only three
    documented modes; a typo here should fail fast."""
    with pytest.raises(SystemExit):
        _parse(module_name, ["--compile-mode", "invalid"])


@pytest.mark.parametrize("module_name", _EXAMPLES, ids=_EXAMPLES)
def test_compile_mode_equals_fused(module_name):
    """``--compile-mode=max-autotune`` (single argv token) must parse
    identically to the two-token form.

    Regression pin against the ``_arg_provided=`` bug class — see
    commit b2c0b67 "fix(examples/fsdp_tp): honour --flag=value
    overrides". When custom override-detection logic forgets to
    handle the ``=``-form, single-token argv silently fails to
    apply.
    """
    args = _parse(module_name, ["--compile", "--compile-mode=max-autotune"])
    assert args.compile is True, (
        f"{module_name}: --compile (with =-form --compile-mode) "
        f"did not set args.compile=True (got {args.compile!r})."
    )
    assert args.compile_mode == "max-autotune", (
        f"{module_name}: --compile-mode=max-autotune (single token) "
        f"did not propagate (got {args.compile_mode!r}). This is "
        f"the =-form override regression — check that the parser "
        f"doesn't have custom override-detection that ignores "
        f"the equals-sign form."
    )


# =====================================================================
# Shared CLI parser: ezpz.cli.flags.build_test_parser
# ---------------------------------------------------------------------
# ``ezpz test`` / ``ezpz benchmark`` go through ``build_test_parser``
# rather than ``ezpz.examples.test.parse_args`` directly. The flag
# must exist in BOTH places (or the parser must be shared) so that
# ``ezpz test --compile`` works the same as ``python -m
# ezpz.examples.test --compile``.
# =====================================================================


def test_build_test_parser_has_compile_flag():
    """The shared CLI parser must register ``--compile`` with the
    same default + behavior as the per-example parsers."""
    parser = _build_test_parser_or_skip()
    ns = parser.parse_args([])
    assert hasattr(ns, "compile"), (
        "ezpz.cli.flags.build_test_parser did not register a "
        "'compile' attribute on the Namespace."
    )
    assert ns.compile is False, (
        f"build_test_parser: args.compile default = {ns.compile!r}, "
        f"expected False."
    )


def test_build_test_parser_has_compile_mode_flag():
    """The shared CLI parser must register ``--compile-mode`` with
    default ``"default"``."""
    parser = _build_test_parser_or_skip()
    ns = parser.parse_args([])
    assert hasattr(ns, "compile_mode"), (
        "ezpz.cli.flags.build_test_parser did not register a "
        "'compile_mode' attribute on the Namespace."
    )
    assert ns.compile_mode == "default", (
        f"build_test_parser: args.compile_mode default = "
        f"{ns.compile_mode!r}, expected 'default'."
    )


@pytest.mark.parametrize("mode", _COMPILE_MODES, ids=_COMPILE_MODES)
def test_build_test_parser_accepts_all_compile_modes(mode):
    """All three documented modes must parse via the shared CLI
    parser too."""
    parser = _build_test_parser_or_skip()
    ns = parser.parse_args(["--compile", "--compile-mode", mode])
    assert ns.compile is True
    assert ns.compile_mode == mode


def test_build_test_parser_rejects_unknown_compile_mode():
    """argparse must reject an unknown ``--compile-mode`` in the
    shared parser too."""
    parser = _build_test_parser_or_skip()
    with pytest.raises(SystemExit):
        parser.parse_args(["--compile-mode", "invalid"])


def test_build_test_parser_equals_form_compile_mode():
    """``--compile-mode=max-autotune`` (single token) must parse via
    the shared parser too. Same regression pin as the per-example
    test."""
    parser = _build_test_parser_or_skip()
    ns = parser.parse_args(["--compile", "--compile-mode=max-autotune"])
    assert ns.compile is True
    assert ns.compile_mode == "max-autotune"
