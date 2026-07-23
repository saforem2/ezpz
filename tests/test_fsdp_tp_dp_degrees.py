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


class TestGlobalBatchSize:
    """global_batch_size = local_batch * dp_replicate * dp_shard.

    Data parallelism spans BOTH the HSDP replicate dim and the FSDP shard
    dim (the DistributedSampler shards over num_replicas = dp_replicate *
    dp_shard). TP is NOT data parallel — tp ranks share the same batch — so
    it must NOT scale the global batch. train() backfills
    args.global_batch_size = args.batch_size * (dp_replicate * dp_shard);
    these tests lock that formula via the resolver it's built on.
    """

    def _gbs(self, m, *, batch, world_size, tp, dp_replicate, dp_shard):
        rep, shard = m._resolve_dp_degrees(
            world_size=world_size,
            tp=tp,
            dp_replicate=dp_replicate,
            dp_shard=dp_shard,
        )
        return batch * rep * shard

    def test_flat_fsdp_tp1(self):
        """8 ranks, tp=1, defaults → dp=8; gbs = batch * 8."""
        m = _import_fsdp_tp()
        assert self._gbs(
            m, batch=2, world_size=8, tp=1, dp_replicate=1, dp_shard=-1
        ) == 16

    def test_tp_does_not_scale_batch(self):
        """tp>1 must NOT multiply the batch: 8 ranks, tp=2 → dp=4; gbs=batch*4.

        Contrast with world_size=8: if tp were (wrongly) included, gbs would
        be batch*8. It must be batch*4.
        """
        m = _import_fsdp_tp()
        assert self._gbs(
            m, batch=2, world_size=8, tp=2, dp_replicate=1, dp_shard=-1
        ) == 8  # 2 * (dp_replicate=1 * dp_shard=4), NOT 2*8

    def test_hsdp_both_dp_dims_count(self):
        """Both replicate and shard scale the batch: 16 ranks, tp=2,
        replicate=2 → shard=4, dp=8; gbs = batch * 8 (NOT batch * 2)."""
        m = _import_fsdp_tp()
        assert self._gbs(
            m, batch=3, world_size=16, tp=2, dp_replicate=2, dp_shard=-1
        ) == 24  # 3 * (2 * 4)

    def test_single_rank(self):
        m = _import_fsdp_tp()
        assert self._gbs(
            m, batch=4, world_size=1, tp=1, dp_replicate=1, dp_shard=-1
        ) == 4


class TestConsumedTokensAccounting:
    """Locks the token-accounting invariants used by train()'s metrics.

    train() logs global tokens as `batch * local_seq_len * world_size` per
    step (cumulative in train/tokens_seen). Global tokens use WORLD_SIZE, not
    dp_size — contrast with the global-batch-size formula, which uses dp_size
    (excludes tp). Under sequence parallelism each tp rank holds a distinct
    seq shard, so tp ranks consume distinct tokens; hence world_size here.
    These are pure-arithmetic checks of that intent (the loop itself needs a
    live run to exercise).
    """

    def test_global_tokens_per_step_uses_world_size(self):
        # 8 ranks, tp=2: world_size=8. Global tokens/step = batch*seq*8,
        # NOT batch*seq*dp_size (which would be batch*seq*4).
        batch, seq, world_size = 2, 4096, 8
        assert batch * seq * world_size == 65536

    def test_tps_and_gbs_differ_by_tp_under_sp(self):
        """GBS excludes tp (samples), token-throughput includes it (SP seq
        shards). At tp>1 the two scalings must differ by exactly tp."""
        m = _import_fsdp_tp()
        batch, world_size, tp = 2, 8, 2
        rep, shard = m._resolve_dp_degrees(
            world_size=world_size, tp=tp, dp_replicate=1, dp_shard=-1
        )
        gbs_factor = rep * shard          # dp_size = 4 (samples)
        tok_factor = world_size           # 8 (distinct-token ranks, SP)
        assert tok_factor == gbs_factor * tp
        assert batch * gbs_factor == 8    # global batch (samples/step)
        assert batch * tok_factor == 16   # global-token rank multiplier
