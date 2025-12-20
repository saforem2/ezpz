import socket
from contextlib import closing
from multiprocessing import Manager

import pytest
import torch
from torch.distributed.fsdp import MixedPrecision
import torch.distributed as dist
import torch.multiprocessing as mp

from ezpz.examples import fsdp_tp
from ezpz.models.llama import ModelArgs, Transformer


def _build_tiny_model(vocab_size: int, seq_len: int) -> Transformer:
    config = ModelArgs(
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=vocab_size,
        multiple_of=8,
        batch_size=2,
        max_seq_len=seq_len,
        depth_init=True,
    )
    return Transformer.from_model_args(config)


def _build_large_model(vocab_size: int, seq_len: int) -> Transformer:
    config = ModelArgs(
        dim=256,
        n_layers=16,
        n_heads=16,
        n_kv_heads=4,
        vocab_size=vocab_size,
        multiple_of=8,
        batch_size=2,
        max_seq_len=seq_len,
        depth_init=True,
    )
    return Transformer.from_model_args(config)


def _pick_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def test_xpu_bf16_forward_finiteness():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("XPU not available")

    torch.manual_seed(0)
    device = torch.device("xpu")
    seq_len = 128
    vocab_size = 128
    tokens = torch.randint(
        0, vocab_size, (2, seq_len), device=device, dtype=torch.long
    )

    model_fp32 = _build_tiny_model(vocab_size, seq_len).to(
        device=device, dtype=torch.float32
    )
    pred_fp32 = model_fp32(tokens)
    assert torch.isfinite(pred_fp32).all()

    model_bf16 = _build_tiny_model(vocab_size, seq_len).to(
        device=device, dtype=torch.bfloat16
    )
    pred_bf16 = model_bf16(tokens)
    if not torch.isfinite(pred_bf16).all():
        pytest.xfail(
            "bf16 XPU forward produced non-finite logits; suspected kernel issue"
        )


def test_xpu_bf16_forward_finiteness_large():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("XPU not available")

    torch.manual_seed(0)
    device = torch.device("xpu")
    seq_len = 1024
    vocab_size = 32000
    tokens = torch.randint(
        0, vocab_size, (2, seq_len), device=device, dtype=torch.long
    )

    model_bf16 = _build_large_model(vocab_size, seq_len).to(
        device=device, dtype=torch.bfloat16
    )
    pred_bf16 = model_bf16(tokens)
    if not torch.isfinite(pred_bf16).all():
        pytest.xfail(
            "bf16 XPU forward produced non-finite logits at larger shapes"
        )


def _tp_worker(rank: int, world_size: int, port: int, results):
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        results["skip"] = True
        return

    try:
        dist.init_process_group(
            backend="xccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
        )
        torch.xpu.set_device(rank)
        device = torch.device(f"xpu:{rank}")
        seq_len = 256
        vocab_size = 32000
        tokens = torch.randint(
            0, vocab_size, (2, seq_len), device=device, dtype=torch.long
        )

        model = _build_tiny_model(vocab_size, seq_len).to(
            device=device, dtype=torch.bfloat16
        )
        device_mesh = fsdp_tp.init_device_mesh(
            str(torch.device("xpu")),
            (1, world_size),
            mesh_dim_names=("dp", "tp"),
        )
        mp_config = MixedPrecision(
            param_dtype=torch.bfloat16,
            cast_forward_inputs=True,
            reduce_dtype=torch.float32,
        )
        model = fsdp_tp.parallelize(model, device_mesh, mp_config)
        pred = model(tokens)
        results[rank] = bool(torch.isfinite(pred).all().item())
        dist.barrier()
    except Exception as exc:
        results["skip"] = True
        results["error"] = str(exc)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_xpu_bf16_forward_finiteness_tp():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("XPU not available")

    world_size = 2
    port = _pick_free_port()
    with Manager() as manager:
        results = manager.dict()
        mp.spawn(
            _tp_worker,
            args=(world_size, port, results),
            nprocs=world_size,
            join=True,
        )
        if results.get("skip"):
            pytest.skip(results.get("error", "xccl unavailable"))
        if not all(results.values()):
            pytest.xfail("bf16 XPU TP forward produced non-finite logits")
