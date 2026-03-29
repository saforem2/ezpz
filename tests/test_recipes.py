"""Tests verifying that the code snippets in docs/recipes.md actually work.

Each test corresponds to a recipe section and exercises the real ezpz API
in single-process mode (no MPI / multi-GPU required).
"""

from __future__ import annotations

import time

import pytest
import torch

import ezpz
import ezpz.distributed as dist
from ezpz.history import History


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _single_process_env(monkeypatch):
    """Force single-process distributed environment for all tests."""
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.fixture()
def device():
    """Return the auto-detected torch device."""
    return ezpz.get_torch_device()


# ---------------------------------------------------------------------------
# Recipe: FSDP Training
# ---------------------------------------------------------------------------


class TestRecipeFSDPTraining:
    """docs/recipes.md — 'FSDP Training' section."""

    def test_setup_wrap_and_cleanup(self):
        """setup_torch -> wrap_model -> optimizer -> cleanup round-trip."""
        rank = ezpz.setup_torch()
        assert isinstance(rank, int)
        device = ezpz.get_torch_device()
        model = torch.nn.Linear(32, 16).to(device)
        # wrap_model returns the model unchanged when world_size <= 1
        wrapped = ezpz.wrap_model(model, use_fsdp=True)
        assert wrapped is not None
        optimizer = torch.optim.Adam(wrapped.parameters(), lr=1e-4)
        assert optimizer is not None
        ezpz.cleanup()

    def test_ddp_flag(self):
        """use_fsdp=False selects DDP wrapping path."""
        ezpz.setup_torch()
        device = ezpz.get_torch_device()
        model = torch.nn.Linear(32, 16).to(device)
        wrapped = ezpz.wrap_model(model, use_fsdp=False)
        assert wrapped is not None
        ezpz.cleanup()


# ---------------------------------------------------------------------------
# Recipe: W&B Logging
# ---------------------------------------------------------------------------


class TestRecipeWandBLogging:
    """docs/recipes.md — 'W&B Logging' section."""

    def test_setup_wandb_with_history(self, tmp_path):
        """setup_wandb + History.update + finalize work end-to-end."""
        ezpz.setup_torch()
        # WANDB_MODE=disabled means this won't actually contact W&B
        ezpz.setup_wandb(project_name="test-recipes")
        history = ezpz.History(
            report_enabled=False,
            jsonl_path=tmp_path / "metrics.jsonl",
            jsonl_overwrite=True,
        )
        num_steps = 10
        for step in range(num_steps):
            loss_val = 1.0 / (step + 1)
            lr_val = 1e-3
            summary = history.update(
                {"loss": loss_val, "lr": lr_val}
            )
            assert isinstance(summary, str)
            assert "loss" in summary
        dataset = history.finalize(
            outdir=tmp_path,
            run_name="test-recipe-wandb",
            save=True,
            plot=False,
        )
        assert dataset is not None
        assert "loss" in dataset.data_vars
        assert "lr" in dataset.data_vars
        ezpz.cleanup()


# ---------------------------------------------------------------------------
# Recipe: Timing with ezpz.synchronize()
# ---------------------------------------------------------------------------


class TestRecipeSynchronizeTiming:
    """docs/recipes.md — 'Timing with ezpz.synchronize()' section."""

    def test_synchronize_timing_pattern(self):
        """synchronize + perf_counter timing produces a positive dt."""
        ezpz.setup_torch()
        device = ezpz.get_torch_device()
        model = torch.nn.Linear(32, 16).to(device)
        batch = torch.randn(8, 32, device=device)

        ezpz.synchronize()
        t0 = time.perf_counter()

        output = model(batch)
        loss = output.sum()
        loss.backward()

        ezpz.synchronize()
        dt = time.perf_counter() - t0

        assert dt > 0, "Elapsed time should be positive"
        assert isinstance(dt, float)
        ezpz.cleanup()

    def test_synchronize_is_callable_without_args(self):
        """synchronize() works with no arguments (auto-detect device)."""
        ezpz.setup_torch()
        # Should not raise
        ezpz.synchronize()
        ezpz.cleanup()


# ---------------------------------------------------------------------------
# Recipe: Disabling Distributed History
# ---------------------------------------------------------------------------


class TestRecipeDisableDistributedHistory:
    """docs/recipes.md — 'Disabling Distributed History' section."""

    def test_distributed_history_false_in_constructor(self, tmp_path):
        """History(distributed_history=False) disables distributed stats."""
        ezpz.setup_torch()
        history = History(
            distributed_history=False,
            report_enabled=False,
            jsonl_path=tmp_path / "metrics.jsonl",
            jsonl_overwrite=True,
        )
        assert history.distributed_history is False
        summary = history.update({"loss": 0.5})
        assert isinstance(summary, str)
        ezpz.cleanup()

    def test_env_var_disables_distributed_history(
        self, monkeypatch, tmp_path
    ):
        """EZPZ_NO_DISTRIBUTED_HISTORY=1 disables distributed stats."""
        monkeypatch.setenv("EZPZ_NO_DISTRIBUTED_HISTORY", "1")
        ezpz.setup_torch()
        history = History(
            report_enabled=False,
            jsonl_path=tmp_path / "metrics.jsonl",
            jsonl_overwrite=True,
        )
        assert history.distributed_history is False
        ezpz.cleanup()


# ---------------------------------------------------------------------------
# Recipe: FSDP Training (full training loop)
# ---------------------------------------------------------------------------


class TestRecipeTrainingLoop:
    """Verify the full training loop pattern from the recipes works."""

    def test_full_training_loop(self, tmp_path):
        """setup -> model -> wrap -> train loop -> history -> finalize."""
        ezpz.setup_torch()
        device = ezpz.get_torch_device()

        model = torch.nn.Linear(128, 64).to(device)
        wrapped = ezpz.wrap_model(model, use_fsdp=True)
        optimizer = torch.optim.Adam(wrapped.parameters(), lr=1e-3)

        history = History(
            report_enabled=False,
            jsonl_path=tmp_path / "metrics.jsonl",
            jsonl_overwrite=True,
        )

        for step in range(20):
            x = torch.randn(32, 128, device=device)
            loss = wrapped(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            history.update({"step": step, "loss": loss.item()})

        dataset = history.finalize(
            outdir=tmp_path,
            run_name="recipe-loop",
            save=True,
            plot=False,
        )
        assert dataset is not None
        assert "loss" in dataset.data_vars
        assert len(dataset["loss"]) == 20
        ezpz.cleanup()


# ---------------------------------------------------------------------------
# Recipe: Forcing a Specific Device/Backend
# ---------------------------------------------------------------------------


class TestRecipeForceDeviceBackend:
    """docs/recipes.md — 'Forcing a Specific Device/Backend' section."""

    def test_force_cpu_gloo(self, monkeypatch):
        """TORCH_DEVICE=cpu TORCH_BACKEND=gloo overrides auto-detection."""
        monkeypatch.setenv("TORCH_DEVICE", "cpu")
        monkeypatch.setenv("TORCH_BACKEND", "gloo")
        device_type = dist.get_torch_device_type()
        backend = dist.get_torch_backend()
        assert device_type == "cpu"
        assert backend == "gloo"
