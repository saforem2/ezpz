"""Tests for ``ezpz.examples.inference``.

Covers the pure-Python helpers (no torch/distributed required).
The end-to-end inference loop is tested by running the example on
real hardware — see docs/examples/inference.md.
"""

from __future__ import annotations

import pytest

from ezpz.examples.inference import _normalize, parse_args, shard_indices


class TestShardIndices:
    """Tests for ``shard_indices``."""

    def test_even_split(self):
        """16 samples across 4 ranks → 4 each."""
        chunks = [shard_indices(16, r, 4) for r in range(4)]
        assert chunks == [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ]

    def test_uneven_split(self):
        """10 samples across 4 ranks → 3, 3, 3, 1."""
        chunks = [shard_indices(10, r, 4) for r in range(4)]
        assert chunks == [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9],
        ]
        # Concatenation reconstructs the full set
        flat = [i for c in chunks for i in c]
        assert flat == list(range(10))

    def test_more_ranks_than_samples(self):
        """3 samples across 4 ranks → first 3 ranks get one, last gets none."""
        chunks = [shard_indices(3, r, 4) for r in range(4)]
        assert chunks == [[0], [1], [2], []]

    def test_single_rank(self):
        """Everything goes to rank 0 in single-rank mode."""
        assert shard_indices(7, 0, 1) == list(range(7))

    def test_empty(self):
        """Zero samples returns empty for every rank."""
        for r in range(4):
            assert shard_indices(0, r, 4) == []

    def test_invalid_world_size(self):
        """Non-positive world_size returns empty."""
        assert shard_indices(10, 0, 0) == []


class TestNormalize:
    """Tests for the eval-mode text normalization helper."""

    def test_lowercases(self):
        assert _normalize("Hello World") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize("foo   bar\t\nbaz") == "foo bar baz"

    def test_strips_edges(self):
        assert _normalize("  spaced  ") == "spaced"

    def test_empty(self):
        assert _normalize("") == ""

    def test_none_returns_empty(self):
        assert _normalize(None) == ""  # type: ignore[arg-type]


class TestParseArgsModes:
    """Tests for the --mode argument."""

    def test_default_is_generate(self):
        args = parse_args([])
        assert args.mode == "generate"

    def test_benchmark_mode(self):
        args = parse_args(["--mode", "benchmark"])
        assert args.mode == "benchmark"
        assert args.benchmark_iters == 20  # default
        assert args.benchmark_warmup == 3  # default

    def test_eval_mode(self):
        args = parse_args(["--mode", "eval", "--label-column", "answer"])
        assert args.mode == "eval"
        assert args.label_column == "answer"

    def test_eval_mode_without_label_column_is_allowed(self):
        """Without --label-column, eval falls back to next-token scoring."""
        args = parse_args(["--mode", "eval"])
        assert args.mode == "eval"
        assert args.label_column is None

    def test_invalid_mode_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--mode", "nonsense"])

    def test_flops_default_off(self):
        """--flops is opt-in; without it MFU/TFLOPS are not reported."""
        args = parse_args([])
        assert args.flops is False
        assert args.flops_every_n_steps == 1  # only used when --flops set

    def test_flops_enabled(self):
        args = parse_args(["--flops"])
        assert args.flops is True

    def test_flops_every_n_steps_custom(self):
        args = parse_args(["--flops", "--flops-every-n-steps", "10"])
        assert args.flops is True
        assert args.flops_every_n_steps == 10
