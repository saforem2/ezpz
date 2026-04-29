"""Tests for ``ezpz.examples.inference``.

Covers the pure-Python helpers (no torch/distributed required).
The end-to-end inference loop is tested by running the example on
real hardware — see docs/examples/inference.md.
"""

from __future__ import annotations

from ezpz.examples.inference import shard_indices


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
