"""Regression test: cli/flags.py --model choices must stay in sync.

``src/ezpz/cli/flags.py`` declares a hardcoded ``choices=[...]`` list
for the ``--model`` argument on ``ezpz test``. The comment in-source
explains this is a circular-dep workaround for not importing
``ezpz.examples.test.MODEL_PRESETS`` / ``MODEL_ALIASES`` directly.

That hardcoded list silently drifts the next time someone adds a
size to ``test.py`` and forgets to update flags.py. This test fails
loudly in that case.

Mirrors the import-skip pattern in ``tests/test_example_model_sizes.py``
so a torch-less CI box doesn't false-fail.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest


def _load(module_name: str) -> Any:
    """Import a module, skipping the test if it can't load.

    Matches the helper in ``tests/test_example_model_sizes.py``:
    examples + flags pull in torch transitively; we don't want this
    contract test to fail on environments without torch.
    """
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"could not import {module_name}: {exc}")


def _get_model_choices_from_parser() -> list[str]:
    """Build the test parser and reflect the --model action's choices.

    flags.py doesn't expose the choices as a module-level constant
    (verified by reading lines 200-225 — the list is inline in the
    add_argument call). Walking parser._actions is the cleanest way
    to fetch it without re-parsing source.
    """
    flags = _load("ezpz.cli.flags")
    parser = flags.build_test_parser()
    for action in parser._actions:
        if "--model" in action.option_strings:
            assert action.choices is not None, (
                "flags.py --model action has no choices= set; "
                "this test assumes choices is the source of truth"
            )
            return list(action.choices)
    raise AssertionError(
        "no --model action found on flags.build_test_parser()"
    )


def test_flags_model_choices_match_test_module_presets():
    """flags.py choices must equal MODEL_PRESETS ∪ MODEL_ALIASES.

    Each direction matters:
    - extra entries in flags.py → argparse will silently accept a
      size that test.py doesn't recognize, then test.py fails
      mid-run with a confusing KeyError.
    - missing entries → argparse rejects a valid preset before
      test.py even sees it.
    """
    flags_choices = set(_get_model_choices_from_parser())

    test_mod = _load("ezpz.examples.test")
    presets = set(test_mod.MODEL_PRESETS.keys())
    aliases = set(test_mod.MODEL_ALIASES.keys())
    expected = presets | aliases

    assert flags_choices == expected, (
        "ezpz.cli.flags.py --model choices drifted from "
        "ezpz.examples.test.{MODEL_PRESETS, MODEL_ALIASES}.\n"
        f"  flags.py choices:                 {sorted(flags_choices)}\n"
        f"  test.py presets | aliases:        {sorted(expected)}\n"
        f"  in flags.py only (remove these):  "
        f"{sorted(flags_choices - expected)}\n"
        f"  in test.py only (add these to flags.py): "
        f"{sorted(expected - flags_choices)}"
    )
