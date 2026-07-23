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

    def test_explicit_dp_shard(self):
        """Explicit --dp-shard (not -1) still scales the batch: 8 ranks,
        tp=2, replicate=1, shard=4 → dp=4; gbs = batch * 4. (Product
        constraint: 1 * 4 * 2 == world_size=8.)"""
        m = _import_fsdp_tp()
        assert self._gbs(
            m, batch=8, world_size=8, tp=2, dp_replicate=1, dp_shard=4
        ) == 32  # 8 * (1 * 4); explicit shard honored, tp excluded

    def test_single_rank(self):
        m = _import_fsdp_tp()
        assert self._gbs(
            m, batch=4, world_size=1, tp=1, dp_replicate=1, dp_shard=-1
        ) == 4


class TestConsumedTokensAccounting:
    """Locks the token-accounting invariant used by train()'s metrics.

    train() logs global tokens/step as `batch * inp.shape[1] * dpsize`,
    where inp.shape[1] is the FULL pre-shard sequence length (the model
    input is Replicate() across the tp group; only the embedding OUTPUT is
    sequence-sharded). tp does NOT enter: under SP the tp ranks hold shards
    of the SAME sequence (summing the shards recovers the full length), and
    on the HF path tp is forced to 1 with no SP. So the token count is
    exactly the global batch (batch * dpsize) times the sequence length, on
    every parallelism config.

    The earlier `local_seq_len * dpsize * tp` framing overcounted by
    `dpsize * (tp - remainder)` whenever seq_len % tp != 0 — and since
    inp = x[:, :-1] makes seq_len odd, that was the common case. These are
    pure-arithmetic checks of the corrected formula (the loop needs a live
    run to exercise the tensors themselves).
    """

    def _global_tokens(self, m, *, batch, seq_len, world_size, tp,
                       dp_replicate, dp_shard):
        """Global tokens/step exactly as train() computes it:
        batch * seq_len * dpsize, where seq_len is the full (pre-shard)
        length and dpsize is resolved from the ORIGINAL tp."""
        rep, shard = m._resolve_dp_degrees(
            world_size=world_size, tp=tp, dp_replicate=dp_replicate,
            dp_shard=dp_shard,
        )
        return batch * seq_len * (rep * shard)

    def test_sp_path_seq_divisible_by_tp(self):
        """ezpz Transformer, tp=2, seq divisible by tp: 8 ranks → dpsize=4.
        Global tokens = batch * seq * 4 (tp does not enter)."""
        m = _import_fsdp_tp()
        assert self._global_tokens(
            m, batch=2, seq_len=4096, world_size=8, tp=2,
            dp_replicate=1, dp_shard=-1,
        ) == 2 * 4096 * 4

    def test_sp_path_seq_not_divisible_by_tp(self):
        """The bug Copilot flagged: seq_len=4095, tp=2. The full-length
        formula gives batch*4095*dpsize exactly; the old shard*tp framing
        would have given batch*4096*dpsize (overcount by dpsize per step)."""
        m = _import_fsdp_tp()
        batch, seq, tp, dpsize = 2, 4095, 2, 4
        exact = self._global_tokens(
            m, batch=batch, seq_len=seq, world_size=8, tp=tp,
            dp_replicate=1, dp_shard=-1,
        )
        assert exact == batch * seq * dpsize          # 2 * 4095 * 4 = 32760
        # Old (buggy) framing: batch * rank0_shard * dpsize * tp, where
        #   rank0_shard = ceil(seq/tp) = 2048  →  batch*2048*dpsize*2 = 32768,
        # overcounting the true 32760 by batch*dpsize*(tp - remainder) = 8.
        rank0_shard = (seq + tp - 1) // tp            # ceil(4095/2) = 2048
        buggy = batch * rank0_shard * dpsize * tp
        remainder = seq % tp                          # 1
        assert buggy == exact + batch * dpsize * (tp - remainder)
        assert buggy != exact

    def test_hf_forced_tp1_uses_dp_size_not_world_size(self):
        """HF path forces tp=1 with NO SP: dpsize was resolved with the
        ORIGINAL tp (=2), so global tokens use dpsize (=4), NOT world_size
        (=8). Multiplying by world_size would 2x-overcount duplicated ranks."""
        m = _import_fsdp_tp()
        tokens = self._global_tokens(
            m, batch=2, seq_len=1024, world_size=8, tp=2,
            dp_replicate=1, dp_shard=-1,
        )
        assert tokens == 2 * 1024 * 4          # dpsize=4, not world_size=8
        assert tokens != 2 * 1024 * 8

    def test_tp1_path(self):
        """Plain tp=1: dpsize == world_size, regression guard for the
        common path."""
        m = _import_fsdp_tp()
        assert self._global_tokens(
            m, batch=2, seq_len=512, world_size=8, tp=1,
            dp_replicate=1, dp_shard=-1,
        ) == 2 * 512 * 8

    def test_tokens_equal_global_batch_times_seq(self):
        """Tokens/step == global_batch * seq_len on every config: the token
        multiplier (dpsize) is exactly the global-batch multiplier."""
        m = _import_fsdp_tp()
        batch, seq, world_size, tp = 3, 2048, 16, 2
        rep, shard = m._resolve_dp_degrees(
            world_size=world_size, tp=tp, dp_replicate=2, dp_shard=-1
        )
        gbs = batch * rep * shard                     # global batch (samples)
        tokens = self._global_tokens(
            m, batch=batch, seq_len=seq, world_size=world_size, tp=tp,
            dp_replicate=2, dp_shard=-1,
        )
        assert tokens == gbs * seq
