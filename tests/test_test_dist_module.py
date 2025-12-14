"""Tests for the ``ezpz.test_dist`` helpers."""

import pytest
import torch

import ezpz.test_dist as test_dist


def test_calc_loss_matches_cross_entropy():
    """calc_loss should match PyTorch cross entropy."""
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    targets = torch.tensor([0, 1], dtype=torch.long)
    expected = torch.nn.functional.cross_entropy(logits.float(), targets)
    loss = test_dist.calc_loss(logits, targets)
    assert pytest.approx(loss.item()) == expected.item()
