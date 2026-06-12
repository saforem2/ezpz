"""Tests for the XL/XXL/XXXL model-size presets across ezpz.examples.

Each example exposes a MODEL_PRESETS dict (size_name → preset values)
plus a MODEL_ALIASES dict (alias → canonical_size_name). Users can
say ``--model xl`` or ``--model xlarge`` or ``--model extra-large``
and all three should resolve to the same preset.

These tests pin the contract:
1. Every example has the three new sizes (``xl``, ``xxl``, ``xxxl``).
2. Every example accepts the long-form aliases (``xlarge`` /
   ``extra-large`` / etc.) and resolves them to the same preset as
   the short form.
3. argparse rejects unknown sizes (no silent fallthrough).
4. Sizes are strictly increasing in their main "model capacity"
   dimension (catches accidental down-scaling).
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest


# Map of example module name → (alias dict attribute, presets dict
# attribute, "size signal" key to use for monotonicity check).
# The "size signal" key is whichever field most cleanly captures
# the model's parameter count for that file's NN topology.
_EXAMPLES = [
    # (module, presets_attr, aliases_attr, monotonic_key)
    ("ezpz.examples.test", "MODEL_PRESETS", "MODEL_ALIASES", None),  # layer_sizes is a list — skip strict mono check
    ("ezpz.examples.fsdp", "MODEL_PRESETS", "MODEL_ALIASES", "fc_dim"),
    ("ezpz.examples.vit", "MODEL_PRESETS", "MODEL_ALIASES", "head_dim"),
    ("ezpz.examples.diffusion", "MODEL_PRESETS", "MODEL_ALIASES", "hidden"),
    ("ezpz.examples.fsdp_tp", "MODEL_PRESETS", "MODEL_ALIASES", "dim"),
]


_EXPECTED_NEW_SIZES = ("xl", "xxl", "xxxl")
_EXPECTED_ALIAS_GROUPS = {
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
    "module_name,presets_attr,aliases_attr,_mono",
    _EXAMPLES,
    ids=[m for m, _, _, _ in _EXAMPLES],
)
class TestModelSizePresets:
    def test_has_xl_xxl_xxxl_presets(
        self, module_name, presets_attr, aliases_attr, _mono
    ):
        """All 3 new size names must appear in MODEL_PRESETS."""
        mod = _load(module_name)
        presets = getattr(mod, presets_attr)
        for size in _EXPECTED_NEW_SIZES:
            assert size in presets, (
                f"{module_name}.{presets_attr} missing '{size}' "
                f"preset. Got: {sorted(presets.keys())}"
            )

    def test_has_long_form_aliases(
        self, module_name, presets_attr, aliases_attr, _mono
    ):
        """Each new size has both short-form and long-form aliases.

        E.g. ``--model xl``, ``--model xlarge``, and
        ``--model extra-large`` must all resolve to the ``xl``
        preset.
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
        self, module_name, presets_attr, aliases_attr, _mono
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

    def test_size_monotonically_increasing(
        self, module_name, presets_attr, aliases_attr, _mono
    ):
        """Each consecutive size (large → xl → xxl → xxxl) should
        scale UP on the chosen monotonicity-signal key.

        Skipped for examples where the size signal is a list
        (e.g. test.py uses layer_sizes — comparing nested lists
        for monotonicity is its own thing). For those, the other
        tests verify the presets exist + aliases resolve.
        """
        if _mono is None:
            pytest.skip(f"no scalar size signal for {module_name}")
        mod = _load(module_name)
        presets = getattr(mod, presets_attr)
        # large is the "old" top size; xl/xxl/xxxl should each be
        # strictly larger.
        sequence = ["large", "xl", "xxl", "xxxl"]
        values = [presets[s][_mono] for s in sequence if s in presets]
        assert values == sorted(values), (
            f"{module_name}: '{_mono}' is not monotonically "
            f"increasing across {sequence}. Got: "
            f"{dict(zip(sequence, values))}"
        )
        # Also: strictly increasing, not just non-decreasing — if
        # two consecutive sizes are equal on the size signal, the
        # XL/XXL/XXXL distinction is meaningless for this metric.
        for a, b in zip(values, values[1:]):
            assert a < b, (
                f"{module_name}: '{_mono}' did not strictly "
                f"increase from {a} → {b} between consecutive "
                f"sizes. XL/XXL/XXXL should be distinct."
            )


# ===================================================================
# End-to-end CLI parse: --model xl etc. actually applies the preset
# ===================================================================


def _parse_with_model(module_name: str, model_value: str) -> Any:
    """Run the example's parse_args with --model <value> and return
    the resulting Namespace. Used to verify the preset values
    actually propagate to argparse output."""
    mod = _load(module_name)
    parse_args = mod.parse_args
    # The vit, fsdp, fsdp_tp, diffusion parse_args take an argv
    # list directly. test.py's parse_args also accepts argv.
    return parse_args(["--model", model_value])


@pytest.mark.parametrize(
    "module_name,short,alias",
    [
        # Just spot-check vit + fsdp_tp end-to-end (the others use the
        # same alias-resolution helper). If those two work, the
        # contract is exercised.
        ("ezpz.examples.vit", "xl", "xlarge"),
        ("ezpz.examples.vit", "xxl", "extra-extra-large"),
        ("ezpz.examples.fsdp_tp", "xl", "xlarge"),
        ("ezpz.examples.fsdp_tp", "xxxl", "extra-extra-extra-large"),
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
