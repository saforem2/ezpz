"""Regression tests for TP broadcasts with unsized, streaming dataloaders."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ezpz.data.distributed import TPBroadcastDataLoader


class _StreamingLoader:
    def __init__(self):
        self.loaded_state = None

    def __iter__(self):
        yield torch.tensor([10])
        yield torch.tensor([20])

    def __len__(self):
        return 0

    def save_state(self):
        return {"position": 2}

    def load_state(self, state):
        self.loaded_state = state


def _tp_worker(rank, world_size, init_method, result_dir):
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size
    )
    try:
        source = _StreamingLoader()
        loader = TPBroadcastDataLoader(source, dist.group.WORLD)
        batches = [batch.tolist() for batch in loader]
        saved_state = loader.save_state()
        loader.load_state({"position": 1})
        Path(result_dir, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "batches": batches,
                    "saved_state": saved_state,
                    "loaded_state": source.loaded_state,
                }
            )
        )
    finally:
        dist.destroy_process_group()


def _tp_nccl_worker(rank, world_size, init_method, result_dir):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=45),
    )
    try:
        source = _StreamingLoader()
        loader = TPBroadcastDataLoader(source, dist.group.WORLD)
        batches = [batch.tolist() for batch in loader]
        Path(result_dir, f"nccl-rank-{rank}.json").write_text(
            json.dumps(batches)
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo distributed backend is unavailable",
)
def test_tp_broadcast_streaming_loader_and_checkpoint_state(tmp_path):
    init_file = tmp_path / "gloo-init"
    result_dir = tmp_path
    mp.spawn(
        _tp_worker,
        args=(2, f"file://{init_file}", str(result_dir)),
        nprocs=2,
        join=True,
    )

    results = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text())
        for rank in range(2)
    ]
    assert results[0]["batches"] == [[10], [20]]
    assert results[1]["batches"] == results[0]["batches"]
    assert results[0]["saved_state"] == {"position": 2}
    assert results[1]["saved_state"] is None
    assert results[0]["loaded_state"] == {"position": 1}
    assert results[1]["loaded_state"] is None


@pytest.mark.skipif(
    not dist.is_available()
    or not dist.is_nccl_available()
    or not torch.cuda.is_available()
    or torch.cuda.device_count() < 2,
    reason="two CUDA devices and NCCL are required",
)
def test_tp_broadcast_streaming_loader_nccl(tmp_path):
    init_file = tmp_path / "nccl-init"
    mp.spawn(
        _tp_nccl_worker,
        args=(2, f"file://{init_file}", str(tmp_path)),
        nprocs=2,
        join=True,
    )

    expected = [[10], [20]]
    for rank in range(2):
        batches = json.loads((tmp_path / f"nccl-rank-{rank}.json").read_text())
        assert batches == expected
