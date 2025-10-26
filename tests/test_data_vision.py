"""Tests for the ``ezpz.data.vision`` helpers."""

from __future__ import annotations

import pytest
import torch

import ezpz.data.vision as vision


class _RecordingLoader:
    """Minimal DataLoader stub capturing kwargs for assertions."""

    calls: list[dict[str, object]] = []

    def __init__(self, *args, **kwargs):
        _RecordingLoader.calls.append({"args": args, "kwargs": kwargs})
        self.dataset = kwargs.get("dataset")
        self.sampler = kwargs.get("sampler")


class _DummyMNIST(torch.utils.data.Dataset):
    """Small synthetic dataset to avoid downloads in tests."""

    def __init__(self, *args, **kwargs):
        self.data = torch.zeros((8, 1, 28, 28))
        self.targets = torch.zeros(8, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


@pytest.mark.parametrize("world_size", [1, 2])
def test_get_mnist_handles_shuffle_with_sampler(monkeypatch, world_size):
    """Ensure get_mnist never passes shuffle alongside a sampler."""

    # Capture DataLoader invocations
    monkeypatch.setattr(
        vision.torch.utils.data,
        "DataLoader",
        _RecordingLoader,
    )

    # Prevent barrier from hanging
    monkeypatch.setattr(
        vision.torch.distributed,
        "barrier",
        lambda *args, **kwargs: None,
    )

    # Avoid downloading the dataset
    monkeypatch.setattr(vision.datasets, "MNIST", _DummyMNIST)

    # Simulate distributed setup
    monkeypatch.setattr(vision, "WORLD_SIZE", world_size)
    monkeypatch.setattr(vision, "RANK", 0)

    _RecordingLoader.calls.clear()
    bundle = vision.get_mnist(
        train_batch_size=4,
        test_batch_size=4,
        download=False,
        shuffle=True,
    )

    assert "train" in bundle and "test" in bundle
    assert len(_RecordingLoader.calls) == 2
    train_call, test_call = _RecordingLoader.calls
    train_kwargs = train_call["kwargs"]
    test_kwargs = test_call["kwargs"]

    if world_size > 1:
        assert train_kwargs["sampler"] is not None
        assert "shuffle" not in train_kwargs
        assert test_kwargs["sampler"] is not None
        assert "shuffle" not in test_kwargs
    else:
        assert train_kwargs.get("shuffle") is True
        assert train_kwargs.get("sampler") is None
