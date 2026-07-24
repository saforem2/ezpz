"""Tests for ezpz.examples._checkpoint (DCP save/load helpers).

Single-process, CPU-only: DCP works with a 1-rank (or no) process group, so
these round-trip a tiny model + AdamW without a scheduler or GPU. The
sharded/DTensor path is exercised on-hardware (see the fsdp_tp restart
experiment); here we lock the file-layout contract + the meta/marker logic,
which is where the crash-safety semantics live.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _destroy_pg_after():
    """Tear down any process group these tests initialized.

    Without this, `dist.init_process_group` leaves global distributed state
    initialized, which pollutes OTHER test modules that assume no PG (e.g.
    test_distributed's get_local_rank env-fallback, which then sees a real
    rank-0 PG instead of the mocked env). Runs even if the test skips.
    """
    yield
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


def _import():
    try:
        return importlib.import_module("ezpz.examples._checkpoint")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"could not import _checkpoint: {exc}")


def _torch_or_skip():
    try:
        import torch  # noqa: F401

        return torch
    except Exception:  # pragma: no cover
        pytest.skip("torch not available")


def _init_single_rank_pg(torch):
    """Init a 1-rank gloo process group so DCP has a group to collective on.

    Idempotent-ish: skips if already initialized. Passes rank/world_size and
    a TCP ``init_method`` DIRECTLY to init_process_group so it does NOT touch
    os.environ (RANK/WORLD_SIZE/MASTER_* leaking into the process env polluted
    downstream tests — e.g. get_local_rank's env-fallback). Paired with the
    autouse teardown fixture that destroys the group.
    """
    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29557",
            rank=0,
            world_size=1,
        )
    return dist


class _Tiny(object):
    """Factory for a tiny deterministic model + AdamW (built lazily so torch
    import stays inside the test)."""

    @staticmethod
    def build(torch):
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return model, opt


class TestLatestCheckpoint:
    def test_none_when_dir_absent(self, tmp_path):
        m = _import()
        assert m.latest_checkpoint(tmp_path / "nope") is None

    def test_none_when_no_complete_marker(self, tmp_path):
        m = _import()
        # A step dir with no .complete marker must be ignored.
        (tmp_path / "step-10").mkdir()
        assert m.latest_checkpoint(tmp_path) is None

    def test_picks_highest_step_numerically(self, tmp_path):
        m = _import()
        for s in (9, 10, 100):
            d = tmp_path / f"step-{s}"
            d.mkdir()
            (d / ".complete").write_text("")
        got = m.latest_checkpoint(tmp_path)
        assert got is not None and got.name == "step-100"  # not step-9 lexically

    def test_skips_incomplete_prefers_previous(self, tmp_path):
        m = _import()
        good = tmp_path / "step-10"
        good.mkdir()
        (good / ".complete").write_text("")
        bad = tmp_path / "step-20"  # newer but interrupted (no marker)
        bad.mkdir()
        got = m.latest_checkpoint(tmp_path)
        assert got is not None and got.name == "step-10"


class TestSaveLoadRoundTrip:
    def test_round_trip_restores_params_optimizer_and_meta(self, tmp_path):
        torch = _torch_or_skip()
        _init_single_rank_pg(torch)
        m = _import()

        model, opt = _Tiny.build(torch)
        # Take a couple of steps so optimizer state (exp_avg, etc.) is non-empty.
        for _ in range(3):
            opt.zero_grad()
            out = model(torch.randn(4, 8))
            out.sum().backward()
            opt.step()

        ref_params = [p.detach().clone() for p in model.parameters()]
        ref_step0 = opt.state[next(iter(opt.state))]["step"].clone()

        m.save_checkpoint(
            tmp_path, step=30, model=model, optimizer=opt,
            meta={"tokens_seen": 12345, "epoch": 1, "batch_offset": 3},
        )
        assert (tmp_path / "step-30" / ".complete").exists()
        assert (tmp_path / "step-30" / "meta.json").exists()

        # Fresh model+opt (different init), then load the checkpoint into it.
        torch.manual_seed(999)
        model2, opt2 = _Tiny.build(torch)
        # Perturb so we can prove load actually overwrote it.
        with torch.no_grad():
            for p in model2.parameters():
                p.add_(1.0)

        meta = m.load_checkpoint(tmp_path, model2, opt2)
        assert meta is not None
        assert meta["step"] == 30
        assert meta["tokens_seen"] == 12345
        assert meta["epoch"] == 1
        assert meta["batch_offset"] == 3

        # Params match the saved model.
        for a, b in zip(ref_params, model2.parameters()):
            assert torch.allclose(a, b), "loaded params differ from saved"
        # Optimizer step counter restored.
        ld_step0 = opt2.state[next(iter(opt2.state))]["step"]
        assert torch.equal(ref_step0, ld_step0)

    def test_load_returns_none_when_empty(self, tmp_path):
        torch = _torch_or_skip()
        _init_single_rank_pg(torch)
        m = _import()
        model, opt = _Tiny.build(torch)
        assert m.load_checkpoint(tmp_path, model, opt) is None
