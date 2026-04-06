"""Multi-rank integration tests using real MPI processes on CPU.

These tests launch 2 MPI ranks via ``ezpz launch`` and exercise the core ezpz
workflow end-to-end: setup_torch, wrap_model (DDP), History with distributed
stats, finalize, and cleanup.  No GPUs required — runs on CPU with gloo.

Tests are marked ``integration`` and ``slow`` so they can be skipped in fast
CI runs via ``-m "not slow"``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import ezpz.distributed

PYTHON = sys.executable
NPROCS = 2

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]


def _run_multirank(
    script: str, tmp_path: Path, nprocs: int = NPROCS
) -> subprocess.CompletedProcess:
    """Write *script* to a file and launch it with ``ezpz launch`` on CPU/gloo."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text(textwrap.dedent(script), encoding="utf-8")

    hostfile = tmp_path / "hostfile"
    ezpz.distributed.write_localhost_to_hostfile(hostfile)

    env = os.environ.copy()
    # Scrub distributed env vars that earlier tests may have leaked into
    # the pytest process so that the mpirun children discover topology
    # from MPI rather than stale env state.
    for var in [
        "MASTER_PORT", "MASTER_ADDR",
        "RANK", "LOCAL_RANK", "WORLD_SIZE",
        "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE",
        "PMI_RANK", "PMI_SIZE", "PMI_LOCAL_RANK", "PMI_LOCAL_SIZE",
        "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ]:
        env.pop(var, None)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["TORCH_DEVICE"] = "cpu"
    env["TORCH_BACKEND"] = "gloo"
    env["WANDB_MODE"] = "disabled"
    env["NO_COLOR"] = "1"
    env["EZPZ_LOG_LEVEL"] = "CRITICAL"

    result = subprocess.run(
        [
            "ezpz", "launch",
            "-n", str(nprocs),
            "-ppn", str(nprocs),
            "--hostfile", str(hostfile),
            PYTHON, str(script_file),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    return result


# ---------------------------------------------------------------------------
# Core workflow: setup_torch → get_rank/world_size → cleanup
# ---------------------------------------------------------------------------


class TestMultiRankSetup:
    """Verify setup_torch initializes distributed correctly across ranks."""

    def test_setup_returns_correct_ranks(self, tmp_path):
        result = _run_multirank(
            """
            import ezpz, json
            rank = ezpz.setup_torch()
            ws = ezpz.get_world_size()
            lr = ezpz.get_local_rank()
            print(json.dumps({"rank": rank, "world_size": ws, "local_rank": lr}))
            ezpz.cleanup()
        """,
            tmp_path,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        lines = [
            l for l in result.stdout.strip().splitlines() if l.startswith("{")
        ]
        assert len(lines) == NPROCS
        ranks = sorted(json.loads(l)["rank"] for l in lines)
        world_sizes = [json.loads(l)["world_size"] for l in lines]
        assert ranks == list(range(NPROCS))
        assert all(ws == NPROCS for ws in world_sizes)

    def test_device_is_cpu(self, tmp_path):
        result = _run_multirank(
            """
            import ezpz, json
            ezpz.setup_torch()
            print(json.dumps({"device": str(ezpz.get_torch_device())}))
            ezpz.cleanup()
        """,
            tmp_path,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        lines = [
            l for l in result.stdout.strip().splitlines() if l.startswith("{")
        ]
        for line in lines:
            assert json.loads(line)["device"] == "cpu"


# ---------------------------------------------------------------------------
# DDP wrapping: wrap_model with use_fsdp=False
# ---------------------------------------------------------------------------


class TestMultiRankDDP:
    """Verify DDP wrapping and gradient sync across ranks."""

    def test_ddp_wrapping_and_forward(self, tmp_path):
        """wrap_model should return a DistributedDataParallel wrapper."""
        result = _run_multirank(
            """
            import ezpz, torch, json
            rank = ezpz.setup_torch()
            model = torch.nn.Linear(10, 5)
            wrapped = ezpz.wrap_model(model, use_fsdp=False)
            class_name = type(wrapped).__name__
            x = torch.randn(4, 10)
            loss = wrapped(x).sum()
            loss.backward()
            print(json.dumps({"rank": rank, "class": class_name, "loss": loss.item()}))
            ezpz.cleanup()
        """,
            tmp_path,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        lines = [
            l for l in result.stdout.strip().splitlines() if l.startswith("{")
        ]
        assert len(lines) == NPROCS
        for line in lines:
            data = json.loads(line)
            assert data["class"] == "DistributedDataParallel"
            assert isinstance(data["loss"], float)

    def test_ddp_gradient_sync(self, tmp_path):
        """After backward, DDP should synchronize gradients across ranks."""
        result = _run_multirank(
            """
            import ezpz, torch, json
            torch.manual_seed(0)  # same init weights
            rank = ezpz.setup_torch()
            model = torch.nn.Linear(4, 2)
            wrapped = ezpz.wrap_model(model, use_fsdp=False)
            # Each rank gets different data
            x = torch.randn(2, 4) + rank
            loss = wrapped(x).sum()
            loss.backward()
            # After DDP backward, gradients should be averaged (identical across ranks)
            grad = wrapped.module.weight.grad.tolist()
            print(json.dumps({"rank": rank, "grad": grad}))
            ezpz.cleanup()
        """,
            tmp_path,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        lines = [
            l for l in result.stdout.strip().splitlines() if l.startswith("{")
        ]
        grads = [json.loads(l)["grad"] for l in lines]
        # DDP averages gradients — they should be identical across ranks
        assert grads[0] == grads[1], (
            f"Gradients should match after DDP sync: {grads}"
        )


# ---------------------------------------------------------------------------
# History with distributed stats across ranks
# ---------------------------------------------------------------------------


class TestMultiRankHistory:
    """Verify History distributed aggregation works across real ranks."""

    def test_distributed_history_computes_stats(self, tmp_path):
        """History.update should compute min/max/mean/std across ranks."""
        outdir = tmp_path / "history_out"
        outdir.mkdir()
        result = _run_multirank(
            f"""
            import ezpz, json
            from ezpz.history import History
            rank = ezpz.setup_torch()
            history = History(
                distributed_history=True,
                report_enabled=False,
                jsonl_path="{outdir}/metrics-r" + str(rank) + ".jsonl",
                jsonl_overwrite=True,
            )
            # Each rank logs a different loss value
            loss_val = float(rank + 1)  # rank 0 → 1.0, rank 1 → 2.0
            summary = history.update({{"loss": loss_val}})
            # Distributed stats are only stored on rank 0
            if rank == 0:
                print(json.dumps({{
                    "loss_mean": history.history.get("loss/mean", [None])[-1],
                    "loss_min": history.history.get("loss/min", [None])[-1],
                    "loss_max": history.history.get("loss/max", [None])[-1],
                }}))
            ezpz.cleanup()
        """,
            tmp_path,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        lines = [
            l for l in result.stdout.strip().splitlines() if l.startswith("{")
        ]
        assert len(lines) == 1  # only rank 0
        data = json.loads(lines[0])
        # mean of [1.0, 2.0] = 1.5, min = 1.0, max = 2.0
        assert data["loss_mean"] == pytest.approx(1.5, abs=0.01)
        assert data["loss_min"] == pytest.approx(1.0, abs=0.01)
        assert data["loss_max"] == pytest.approx(2.0, abs=0.01)

    def test_finalize_produces_outputs(self, tmp_path):
        """History.finalize should produce dataset and JSONL files."""
        outdir = tmp_path / "finalize_out"
        outdir.mkdir()
        result = _run_multirank(
            f"""
            import ezpz, json, os
            from ezpz.history import History
            rank = ezpz.setup_torch()
            history = History(
                distributed_history=True,
                report_enabled=False,
                jsonl_path="{outdir}/metrics.jsonl",
                jsonl_overwrite=True,
            )
            for step in range(10):
                history.update({{"loss": 1.0 / (step + 1), "step": step}})
            if rank == 0:
                ds = history.finalize(
                    outdir="{outdir}",
                    run_name="multirank-test",
                    save=True,
                    plot=False,
                )
                has_loss = "loss" in ds.data_vars
                n_steps = len(ds["loss"])
                print(json.dumps({{"finalized": True, "has_loss": has_loss, "n_steps": n_steps}}))
            ezpz.cleanup()
        """,
            tmp_path,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        lines = [
            l for l in result.stdout.strip().splitlines() if l.startswith("{")
        ]
        # Only rank 0 prints
        assert len(lines) >= 1
        data = json.loads(lines[0])
        assert data["finalized"] is True
        assert data["has_loss"] is True
        assert data["n_steps"] == 10
        # Check output files exist
        assert (outdir / "metrics.jsonl").exists()


# ---------------------------------------------------------------------------
# Full training loop: setup → model → DDP → train → history → finalize
# ---------------------------------------------------------------------------


class TestMultiRankFullLoop:
    """End-to-end multi-rank training loop."""

    def test_full_ddp_training_loop(self, tmp_path):
        """Complete training loop with DDP, History, and finalize."""
        outdir = tmp_path / "full_loop"
        outdir.mkdir()
        result = _run_multirank(
            f"""
            import ezpz, torch, json
            from ezpz.history import History
            rank = ezpz.setup_torch()
            device = ezpz.get_torch_device()
            model = torch.nn.Linear(16, 8).to(device)
            wrapped = ezpz.wrap_model(model, use_fsdp=False)
            optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
            history = History(
                distributed_history=True,
                report_enabled=False,
                jsonl_path="{outdir}/metrics.jsonl",
                jsonl_overwrite=True,
            )
            losses = []
            for step in range(20):
                optimizer.zero_grad()
                x = torch.randn(8, 16, device=device)
                loss = wrapped(x).sum() ** 2
                loss.backward()
                optimizer.step()
                history.update({{"loss": loss.item(), "step": step}})
                losses.append(loss.item())
            if rank == 0:
                ds = history.finalize(
                    outdir="{outdir}",
                    run_name="full-loop",
                    save=True,
                    plot=False,
                )
                print(json.dumps({{
                    "rank": rank,
                    "n_steps": len(ds["loss"]),
                    "first_loss": losses[0],
                    "last_loss": losses[-1],
                }}))
            ezpz.cleanup()
        """,
            tmp_path,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        lines = [
            l for l in result.stdout.strip().splitlines() if l.startswith("{")
        ]
        assert len(lines) >= 1
        data = json.loads(lines[0])
        assert data["n_steps"] == 20
        assert isinstance(data["first_loss"], float)
        assert isinstance(data["last_loss"], float)
