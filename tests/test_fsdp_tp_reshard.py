"""Tests for the `--reshard-after-forward` flag in ezpz.examples.fsdp_tp.

Covers the new `--reshard-after-forward {always,never}` /
`--no-reshard-after-forward` surface, the `_reshard_arg` policy->bool
mapper, and the deprecated `--sharding-strategy` alias (mapping +
hard-error on the removed `hybrid_shard*` values).

CPU-only: pure arg parsing + a pure mapping function; no accelerator or
process group needed.
"""

from __future__ import annotations

import importlib

import pytest


def _import_fsdp_tp():
    try:
        return importlib.import_module("ezpz.examples.fsdp_tp")
    except Exception as exc:  # heavy optional deps may be missing
        pytest.skip(f"could not import ezpz.examples.fsdp_tp: {exc}")


class TestReshardAfterForwardFlag:
    """The new --reshard-after-forward / --no-reshard-after-forward surface."""

    def test_default_is_always(self):
        m = _import_fsdp_tp()
        assert m.parse_args([]).reshard_after_forward == "always"

    def test_bare_flag_is_always(self):
        m = _import_fsdp_tp()
        args = m.parse_args(["--reshard-after-forward"])
        assert args.reshard_after_forward == "always"

    def test_explicit_never(self):
        m = _import_fsdp_tp()
        args = m.parse_args(["--reshard-after-forward", "never"])
        assert args.reshard_after_forward == "never"

    def test_explicit_always(self):
        m = _import_fsdp_tp()
        args = m.parse_args(["--reshard-after-forward", "always"])
        assert args.reshard_after_forward == "always"

    def test_negation_flag_is_never(self):
        m = _import_fsdp_tp()
        args = m.parse_args(["--no-reshard-after-forward"])
        assert args.reshard_after_forward == "never"

    def test_invalid_value_rejected(self):
        m = _import_fsdp_tp()
        with pytest.raises(SystemExit):
            m.parse_args(["--reshard-after-forward", "sometimes"])


class TestReshardArg:
    """The pure policy->bool mapper used for fully_shard."""

    def test_always_true(self):
        m = _import_fsdp_tp()
        assert m._reshard_arg("always") is True

    def test_never_false(self):
        m = _import_fsdp_tp()
        assert m._reshard_arg("never") is False


class TestLegacyShardingStrategyAlias:
    """The deprecated --sharding-strategy alias maps to the new policy."""

    def test_full_shard_maps_to_always(self):
        m = _import_fsdp_tp()
        args = m.parse_args(["--sharding-strategy", "full_shard"])
        assert args.reshard_after_forward == "always"

    def test_shard_grad_op_maps_to_never(self):
        m = _import_fsdp_tp()
        args = m.parse_args(["--sharding-strategy", "shard_grad_op"])
        assert args.reshard_after_forward == "never"

    def test_no_shard_maps_to_never(self):
        m = _import_fsdp_tp()
        args = m.parse_args(["--sharding-strategy", "no_shard"])
        assert args.reshard_after_forward == "never"

    @pytest.mark.parametrize("hv", ["hybrid_shard", "hybrid_shard_zero2"])
    def test_hybrid_values_hard_error(self, hv):
        m = _import_fsdp_tp()
        with pytest.raises(SystemExit, match="dp-replicate|dp-shard|HSDP"):
            m.parse_args(["--sharding-strategy", hv])

    def test_unknown_value_hard_error(self):
        m = _import_fsdp_tp()
        with pytest.raises(SystemExit):
            m.parse_args(["--sharding-strategy", "bogus"])

    def test_both_flags_deprecated_wins(self):
        """When both are passed, the deprecated alias is applied last."""
        m = _import_fsdp_tp()
        args = m.parse_args(
            ["--reshard-after-forward", "always",
             "--sharding-strategy", "shard_grad_op"]
        )
        assert args.reshard_after_forward == "never"


class TestHelpSurface:
    """--help advertises the new flag and hides the deprecated one."""

    def test_help_lists_new_flag_and_hides_legacy(self, capsys):
        m = _import_fsdp_tp()
        with pytest.raises(SystemExit):
            m.parse_args(["--help"])
        out = capsys.readouterr().out
        assert "--reshard-after-forward" in out
        assert "--no-reshard-after-forward" in out
        # Deprecated alias is argparse.SUPPRESS-hidden; removed hybrid names
        # must not appear anywhere in the help text.
        assert "--sharding-strategy" not in out
        assert "hybrid_shard" not in out
