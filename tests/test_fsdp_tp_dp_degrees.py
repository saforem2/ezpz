"""Tests for HSDP data-parallel degree resolution in ezpz.examples.fsdp_tp.

Covers the ``--dp-replicate`` / ``--dp-shard`` CLI flags and the pure
``_resolve_dp_degrees`` helper that turns them into concrete
(dp_replicate, dp_shard) degrees. Mirrors torchtitan semantics:
dp_shard == -1 means "use all remaining ranks", and the product
dp_replicate * dp_shard * tp must equal WORLD_SIZE.

CPU-only: the helper is pure arithmetic, and arg parsing needs no
accelerator or process group.
"""

from __future__ import annotations

import importlib

import pytest


def _import_fsdp_tp():
    try:
        return importlib.import_module("ezpz.examples.fsdp_tp")
    except Exception as exc:  # heavy optional deps may be missing
        pytest.skip(f"could not import ezpz.examples.fsdp_tp: {exc}")


class TestResolveDpDegrees:
    """Pure ``_resolve_dp_degrees`` arithmetic + validation."""

    def test_defaults_reproduce_flat_dp(self):
        """replicate=1, shard=-1 → flat dp = world_size // tp, replicate=1.

        This is the pre-HSDP behavior and MUST be preserved exactly.
        """
        m = _import_fsdp_tp()
        assert m._resolve_dp_degrees(
            world_size=8, tp=2, dp_replicate=1, dp_shard=-1
        ) == (1, 4)
        # tp=1 → whole world is the (flat) shard dim
        assert m._resolve_dp_degrees(
            world_size=8, tp=1, dp_replicate=1, dp_shard=-1
        ) == (1, 8)

    def test_shard_auto_fill(self):
        """dp_shard=-1 fills as world_size // (dp_replicate * tp)."""
        m = _import_fsdp_tp()
        # 16 ranks, tp=2, replicate=2 → shard = 16/(2*2) = 4
        assert m._resolve_dp_degrees(
            world_size=16, tp=2, dp_replicate=2, dp_shard=-1
        ) == (2, 4)

    def test_explicit_hsdp(self):
        """Explicit replicate+shard that satisfies the product constraint."""
        m = _import_fsdp_tp()
        assert m._resolve_dp_degrees(
            world_size=8, tp=2, dp_replicate=2, dp_shard=2
        ) == (2, 2)

    def test_product_must_equal_world_size(self):
        """A config whose product != world_size raises with a clear message."""
        m = _import_fsdp_tp()
        with pytest.raises(AssertionError, match="!= WORLD_SIZE"):
            m._resolve_dp_degrees(
                world_size=8, tp=2, dp_replicate=2, dp_shard=3
            )

    def test_auto_shard_requires_divisibility(self):
        """dp_shard=-1 when (dp_replicate*tp) doesn't divide world_size fails."""
        m = _import_fsdp_tp()
        with pytest.raises(AssertionError, match="not\n?.*divisible|divisible"):
            m._resolve_dp_degrees(
                world_size=6, tp=4, dp_replicate=1, dp_shard=-1
            )

    def test_replicate_must_be_positive(self):
        m = _import_fsdp_tp()
        with pytest.raises(AssertionError, match="dp-replicate must be >= 1"):
            m._resolve_dp_degrees(
                world_size=8, tp=1, dp_replicate=0, dp_shard=-1
            )

    def test_shard_must_be_positive_or_auto(self):
        m = _import_fsdp_tp()
        with pytest.raises(AssertionError, match="dp-shard must be >= 1"):
            m._resolve_dp_degrees(
                world_size=8, tp=1, dp_replicate=1, dp_shard=0
            )

    def test_single_rank(self):
        """world_size=1, tp=1 → (1, 1) (laptop path)."""
        m = _import_fsdp_tp()
        assert m._resolve_dp_degrees(
            world_size=1, tp=1, dp_replicate=1, dp_shard=-1
        ) == (1, 1)


class TestDpArgParsing:
    """The --dp-replicate / --dp-shard CLI flags parse correctly."""

    def test_defaults(self):
        m = _import_fsdp_tp()
        args = m.parse_args([])
        assert args.dp_replicate == 1
        assert args.dp_shard == -1

    def test_explicit_values(self):
        m = _import_fsdp_tp()
        args = m.parse_args(["--dp-replicate", "2", "--dp-shard", "4"])
        assert args.dp_replicate == 2
        assert args.dp_shard == 4

    def test_flags_are_ints(self):
        m = _import_fsdp_tp()
        args = m.parse_args(["--dp-replicate", "3"])
        assert isinstance(args.dp_replicate, int)
        assert isinstance(args.dp_shard, int)
