"""Tests for the unified s/m/l/xl/xxl/xxxl model-size ladder across
ezpz.examples.

Each example exposes a MODEL_PRESETS dict (size_name → preset values)
plus a MODEL_ALIASES dict (alias → canonical_size_name). The canonical
short names are ``s/m/l/xl/xxl/xxxl`` targeting roughly
``100M/250M/500M/1B/5B/10B`` params. Long-form aliases
(``small``/``medium``/``large``/``xlarge``/``extra-large``/...) map
onto the same ladder. ``debug`` is the laptop-runnable smoke-test
preset (sub-MB scale) and is preserved across the ladder change.

These tests pin the contract:
1. Every example has all 6 ladder presets (``s/m/l/xl/xxl/xxxl``).
2. Every example also preserves ``debug``.
3. Long-form aliases (``small``/``medium``/.../``extra-extra-extra-large``)
   resolve to the same preset values as the short forms.
4. Aliases never point at nonexistent presets.
5. End-to-end CLI parse: ``--model xl`` and ``--model xlarge`` produce
   identical Namespaces.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest


# Map of example module name → (presets attr, aliases attr).
# All 5 examples now use the same ladder, so monotonicity / size-signal
# checks are handled by test_example_size_targets.py (which actually
# verifies parameter counts hit the targets).
_EXAMPLES = [
    ("ezpz.examples.test", "MODEL_PRESETS", "MODEL_ALIASES"),
    ("ezpz.examples.fsdp", "MODEL_PRESETS", "MODEL_ALIASES"),
    ("ezpz.examples.vit", "MODEL_PRESETS", "MODEL_ALIASES"),
    ("ezpz.examples.diffusion", "MODEL_PRESETS", "MODEL_ALIASES"),
    ("ezpz.examples.fsdp_tp", "MODEL_PRESETS", "MODEL_ALIASES"),
]


_LADDER_SIZES = ("s", "m", "l", "xl", "xxl", "xxxl")
_EXPECTED_ALIAS_GROUPS = {
    "s": ("small",),
    "m": ("medium",),
    "l": ("large",),
    "xl": ("xlarge", "extra-large"),
    "xxl": ("xxlarge", "extra-extra-large"),
    "xxxl": ("xxxlarge", "extra-extra-extra-large"),
}


def _load(module_name: str) -> Any:
    """Import an example module, skipping the test if it can't load.

    The examples import torch etc. — on a CI box without torch the
    module import fails. We don't want to gate the alias-contract
    test on torch availability, but skipping is cleaner than
    erroring.
    """
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"could not import {module_name}: {exc}")


# ===================================================================
# Per-example contract checks
# ===================================================================


@pytest.mark.parametrize(
    "module_name,presets_attr,aliases_attr",
    _EXAMPLES,
    ids=[m for m, _, _ in _EXAMPLES],
)
class TestModelSizePresets:
    def test_has_all_ladder_sizes(
        self, module_name, presets_attr, aliases_attr
    ):
        """All 6 canonical short-name sizes must appear in MODEL_PRESETS."""
        mod = _load(module_name)
        presets = getattr(mod, presets_attr)
        for size in _LADDER_SIZES:
            assert size in presets, (
                f"{module_name}.{presets_attr} missing '{size}' "
                f"preset. Got: {sorted(presets.keys())}"
            )

    def test_preserves_debug_preset(
        self, module_name, presets_attr, aliases_attr
    ):
        """`debug` is the laptop-runnable smoke-test escape hatch and
        must survive the s/m/l/xl/xxl/xxxl ladder migration."""
        mod = _load(module_name)
        presets = getattr(mod, presets_attr)
        assert "debug" in presets, (
            f"{module_name}.{presets_attr} dropped the `debug` preset. "
            "Keep it — it's the smoke-test entry point."
        )

    def test_has_long_form_aliases(
        self, module_name, presets_attr, aliases_attr
    ):
        """Long-form names map to canonical short names.

        E.g. ``small → s``, ``xlarge → xl``, ``extra-large → xl``.
        """
        mod = _load(module_name)
        aliases = getattr(mod, aliases_attr)
        for canonical, alias_names in _EXPECTED_ALIAS_GROUPS.items():
            for alias in alias_names:
                assert alias in aliases, (
                    f"{module_name}.{aliases_attr} missing alias "
                    f"'{alias}' (should map to '{canonical}'). "
                    f"Got: {sorted(aliases.keys())}"
                )
                assert aliases[alias] == canonical, (
                    f"{module_name}.{aliases_attr}['{alias}'] = "
                    f"{aliases[alias]!r}, expected {canonical!r}"
                )

    def test_aliases_point_to_real_presets(
        self, module_name, presets_attr, aliases_attr
    ):
        """Every alias must point to a key that actually exists in
        MODEL_PRESETS. Catches typos (alias → nonexistent preset)
        that would surface only when a user tried the alias."""
        mod = _load(module_name)
        presets = getattr(mod, presets_attr)
        aliases = getattr(mod, aliases_attr)
        for alias, canonical in aliases.items():
            assert canonical in presets, (
                f"{module_name}.{aliases_attr}['{alias}'] points to "
                f"'{canonical}' but that's not a key in "
                f"{presets_attr}. Got presets: {sorted(presets.keys())}"
            )


# ===================================================================
# End-to-end CLI parse: --model xl etc. actually applies the preset
# ===================================================================


def _parse_with_model(module_name: str, model_value: str) -> Any:
    """Run the example's parse_args with --model <value> and return
    the resulting Namespace."""
    mod = _load(module_name)
    parse_args = mod.parse_args
    return parse_args(["--model", model_value])


@pytest.mark.parametrize(
    "module_name,short,alias",
    [
        # Cover each example × at least one long-form alias.
        ("ezpz.examples.vit", "xl", "xlarge"),
        ("ezpz.examples.vit", "xxl", "extra-extra-large"),
        ("ezpz.examples.vit", "s", "small"),
        ("ezpz.examples.fsdp_tp", "xl", "xlarge"),
        ("ezpz.examples.fsdp_tp", "xxxl", "extra-extra-extra-large"),
        ("ezpz.examples.fsdp_tp", "l", "large"),
        ("ezpz.examples.fsdp", "m", "medium"),
        ("ezpz.examples.diffusion", "xxl", "xxlarge"),
        ("ezpz.examples.test", "s", "small"),
    ],
    ids=lambda v: str(v),
)
def test_short_and_long_alias_produce_same_preset_values(
    module_name, short, alias
):
    """``--model xl`` and ``--model xlarge`` (and the long form)
    should produce identical Namespaces post-apply_model_preset.

    This is the actual user-facing contract: pick whichever spelling
    you like, you get the same training run."""
    args_short = _parse_with_model(module_name, short)
    args_alias = _parse_with_model(module_name, alias)

    # Only compare the fields the MODEL_PRESETS dict actually sets —
    # everything else (e.g. defaults from other flags) should be
    # identical anyway since we passed the same argv shape.
    mod = _load(module_name)
    preset_fields = list(mod.MODEL_PRESETS[short].keys())
    for field in preset_fields:
        v_short = getattr(args_short, field, None)
        v_alias = getattr(args_alias, field, None)
        assert v_short == v_alias, (
            f"{module_name}: field {field!r} differs between "
            f"--model {short} ({v_short!r}) and --model {alias} "
            f"({v_alias!r}). Aliases should be fully transparent."
        )


def test_argparse_rejects_unknown_model_size():
    """argparse must reject an unknown --model value rather than
    silently fall through. Pinning this on vit since vit's parser
    uses ``choices=sorted([*MODEL_PRESETS.keys(), *MODEL_ALIASES.keys()])``,
    which is the right shape for this rejection."""
    with pytest.raises(SystemExit):
        _parse_with_model("ezpz.examples.vit", "doesnotexist")
