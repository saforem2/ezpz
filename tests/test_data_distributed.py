"""Tests for ``ezpz.data.distributed``.

Covers TP-aware distributed data loading helpers and the
``TPBroadcastDataLoader`` wrapper, all with mocked torch.distributed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

import ezpz.data.distributed as ddist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pg(rank: int = 0, world_size: int = 4) -> MagicMock:
    """Create a mock ProcessGroup with deterministic rank/world_size."""
    pg = MagicMock(spec=[])
    pg._rank = rank
    pg._world_size = world_size
    return pg


def _get_rank_side_effect(group=None):
    if group is None:
        return 0
    return group._rank


def _get_world_size_side_effect(group=None):
    if group is None:
        return 4
    return group._world_size


# ===================================================================
# _rank_ws
# ===================================================================


class TestRankWs:
    """Tests for ``_rank_ws``."""

    @patch("torch.distributed.get_world_size", side_effect=_get_world_size_side_effect)
    @patch("torch.distributed.get_rank", side_effect=_get_rank_side_effect)
    def test_with_process_group(self, mock_rank, mock_ws):
        pg = _make_pg(rank=2, world_size=8)
        rank, ws = ddist._rank_ws(pg)
        assert rank == 2
        assert ws == 8

    @patch("torch.distributed.get_world_size", return_value=4)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_without_process_group(self, mock_rank, mock_ws):
        rank, ws = ddist._rank_ws(None)
        assert rank == 0
        assert ws == 4


# ===================================================================
# _is_dist
# ===================================================================


class TestIsDist:
    """Tests for ``_is_dist``."""

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.is_available", return_value=True)
    def test_true_when_available_and_initialized(self, mock_avail, mock_init):
        assert ddist._is_dist() is True

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.is_available", return_value=True)
    def test_false_when_not_initialized(self, mock_avail, mock_init):
        assert ddist._is_dist() is False

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.is_available", return_value=False)
    def test_false_when_not_available(self, mock_avail, mock_init):
        assert ddist._is_dist() is False


# ===================================================================
# _tp_is_leader
# ===================================================================


class TestTpIsLeader:
    """Tests for ``_tp_is_leader``."""

    @patch("ezpz.data.distributed._is_dist", return_value=True)
    @patch("torch.distributed.get_world_size", side_effect=_get_world_size_side_effect)
    @patch("torch.distributed.get_rank", side_effect=_get_rank_side_effect)
    def test_rank_0_is_leader(self, mock_rank, mock_ws, mock_dist):
        pg = _make_pg(rank=0, world_size=2)
        assert ddist._tp_is_leader(pg) is True

    @patch("ezpz.data.distributed._is_dist", return_value=True)
    @patch("torch.distributed.get_world_size", side_effect=_get_world_size_side_effect)
    @patch("torch.distributed.get_rank", side_effect=_get_rank_side_effect)
    def test_rank_nonzero_is_not_leader(self, mock_rank, mock_ws, mock_dist):
        pg = _make_pg(rank=1, world_size=2)
        assert ddist._tp_is_leader(pg) is False

    @patch("ezpz.data.distributed._is_dist", return_value=False)
    def test_non_distributed_is_leader(self, mock_dist):
        pg = _make_pg(rank=1, world_size=2)
        assert ddist._tp_is_leader(pg) is True

    @patch("ezpz.data.distributed._is_dist", return_value=True)
    def test_none_group_is_leader(self, mock_dist):
        assert ddist._tp_is_leader(None) is True


# ===================================================================
# _broadcast_batch
# ===================================================================


class TestBroadcastBatch:
    """Tests for ``_broadcast_batch``."""

    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.get_global_rank", return_value=0)
    def test_dict_batch(self, mock_global_rank, mock_bcast):
        """Broadcasts a dict batch correctly."""
        batch = {"input_ids": [1, 2, 3], "labels": [4, 5, 6]}
        pg = _make_pg()

        def side_effect(obj_list, src, group):
            # Simulate broadcast: obj_list stays the same on leader
            pass

        mock_bcast.side_effect = side_effect
        result = ddist._broadcast_batch(batch, pg)
        mock_global_rank.assert_called_once_with(pg, 0)
        mock_bcast.assert_called_once()
        assert result == batch

    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.get_global_rank", return_value=0)
    def test_list_batch(self, mock_global_rank, mock_bcast):
        """Broadcasts a list batch correctly."""
        batch = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        pg = _make_pg()
        result = ddist._broadcast_batch(batch, pg)
        assert result == batch


# ===================================================================
# TPBroadcastDataLoader
# ===================================================================


class TestTPBroadcastDataLoader:
    """Tests for ``TPBroadcastDataLoader``."""

    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.get_global_rank", return_value=0)
    @patch("ezpz.data.distributed._tp_is_leader", return_value=True)
    def test_leader_iterates_real_dataloader(
        self, mock_leader, mock_global, mock_bcast
    ):
        """Leader yields batches from the real dataloader."""
        mock_dl = MagicMock()
        batches = [{"x": 1}, {"x": 2}, {"x": 3}]
        mock_dl.__iter__ = MagicMock(return_value=iter(batches))
        mock_dl.__len__ = MagicMock(return_value=3)

        pg = _make_pg(rank=0, world_size=2)
        loader = ddist.TPBroadcastDataLoader(mock_dl, pg)
        results = list(loader)
        assert len(results) == 3
        assert results[0] == {"x": 1}

    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.get_global_rank", return_value=0)
    @patch("ezpz.data.distributed._tp_is_leader", return_value=False)
    def test_non_leader_receives_broadcasts(
        self, mock_leader, mock_global, mock_bcast
    ):
        """Non-leader iterates dummy range and receives broadcast results."""
        mock_dl = MagicMock()
        mock_dl.__len__ = MagicMock(return_value=2)

        pg = _make_pg(rank=1, world_size=2)
        loader = ddist.TPBroadcastDataLoader(mock_dl, pg)
        results = list(loader)
        # Should have iterated 2 times (len of dataloader)
        assert len(results) == 2

    def test_len(self):
        """__len__ delegates to underlying dataloader."""
        mock_dl = MagicMock()
        mock_dl.__len__ = MagicMock(return_value=42)
        pg = _make_pg()
        loader = ddist.TPBroadcastDataLoader(mock_dl, pg)
        assert len(loader) == 42


# ===================================================================
# get_random_dataset_fsdp_tp
# ===================================================================


class TestGetRandomDatasetFsdpTp:
    """Tests for ``get_random_dataset_fsdp_tp``."""

    @patch("ezpz.data.distributed._is_dist", return_value=False)
    def test_non_distributed_returns_dict(self, mock_dist):
        """Returns dict with dataset, sampler=None, dataloader."""
        result = ddist.get_random_dataset_fsdp_tp(
            batch_size=4, vocab_size=100, seq_length=32
        )
        assert "dataset" in result
        assert "sampler" in result
        assert "dataloader" in result
        assert result["sampler"] is None

    @patch("ezpz.data.distributed.DistributedSampler")
    @patch("torch.distributed.get_world_size", side_effect=_get_world_size_side_effect)
    @patch("torch.distributed.get_rank", side_effect=_get_rank_side_effect)
    @patch("ezpz.data.distributed._is_dist", return_value=True)
    def test_distributed_creates_sampler(
        self, mock_dist, mock_rank, mock_ws, mock_sampler_cls
    ):
        """Creates DistributedSampler with dp_group rank/world_size."""
        dp_group = _make_pg(rank=1, world_size=4)
        mock_sampler_cls.return_value = MagicMock()

        result = ddist.get_random_dataset_fsdp_tp(
            batch_size=4, vocab_size=100, seq_length=32, dp_group=dp_group
        )
        assert result["sampler"] is not None
        mock_sampler_cls.assert_called_once()
        call_kwargs = mock_sampler_cls.call_args
        assert call_kwargs[1]["num_replicas"] == 4
        assert call_kwargs[1]["rank"] == 1

    @patch("torch.distributed.broadcast_object_list")
    @patch("torch.distributed.get_global_rank", return_value=0)
    @patch("ezpz.data.distributed._tp_is_leader", return_value=True)
    @patch("ezpz.data.distributed.DistributedSampler")
    @patch("torch.distributed.get_world_size", side_effect=_get_world_size_side_effect)
    @patch("torch.distributed.get_rank", side_effect=_get_rank_side_effect)
    @patch("ezpz.data.distributed._is_dist", return_value=True)
    def test_tp_broadcast_wrapping(
        self, mock_dist, mock_rank, mock_ws, mock_sampler_cls,
        mock_leader, mock_global, mock_bcast
    ):
        """When broadcast_within_tp=True, wraps dataloader in TPBroadcastDataLoader."""
        dp_group = _make_pg(rank=0, world_size=2)
        tp_group = _make_pg(rank=0, world_size=2)
        mock_sampler_cls.return_value = MagicMock()

        result = ddist.get_random_dataset_fsdp_tp(
            batch_size=4,
            vocab_size=100,
            seq_length=32,
            dp_group=dp_group,
            tp_group=tp_group,
            broadcast_within_tp=True,
        )
        assert isinstance(result["dataloader"], ddist.TPBroadcastDataLoader)

    @patch("ezpz.data.distributed.DistributedSampler")
    @patch("torch.distributed.get_world_size", side_effect=_get_world_size_side_effect)
    @patch("torch.distributed.get_rank", side_effect=_get_rank_side_effect)
    @patch("ezpz.data.distributed._is_dist", return_value=True)
    def test_no_tp_broadcast_without_flag(
        self, mock_dist, mock_rank, mock_ws, mock_sampler_cls
    ):
        """Without broadcast_within_tp, dataloader is plain DataLoader."""
        dp_group = _make_pg(rank=0, world_size=2)
        tp_group = _make_pg(rank=0, world_size=2)
        mock_sampler_cls.return_value = MagicMock()

        result = ddist.get_random_dataset_fsdp_tp(
            batch_size=4,
            vocab_size=100,
            seq_length=32,
            dp_group=dp_group,
            tp_group=tp_group,
            broadcast_within_tp=False,
        )
        assert not isinstance(result["dataloader"], ddist.TPBroadcastDataLoader)
