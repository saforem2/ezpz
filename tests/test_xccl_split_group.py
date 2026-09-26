"""The xccl ``split_group`` workaround must cover mesh INDEXING, not just
mesh creation.

Background (#252): FSDP + TP>1 stalled on Aurora and Sunspot after a few
iterations, all ranks parked in ``reduce_scatter``. ``--tp 1`` was fine.

``ProcessGroupXCCL`` never overrides ``supportsSplitting()``, so it inherits
``false`` from ``Backend.hpp``, while ``DeviceMesh._init_one_process_group``
routes to ``split_group`` whenever the default PG is device-bound. ezpz binds
``device_id`` deliberately (without it FSDP2's ``foreach_all_gather``
deadlocks), so every nested-mesh construction on XPU took a path xccl cannot
serve.

``_xccl_split_workaround`` already guarded ``init_device_mesh`` and
``DeviceMesh._flatten`` — but ``device_mesh["tp"]`` builds a process group
too and sits outside both. These tests pin the function-level patch that
covers all of them.

They run anywhere: the XPU-specific behaviour is exercised through fakes, so
CI on CPU still catches a regression.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _unpatch_device_mesh():
    """Leave ``DeviceMesh`` as we found it.

    The patch is installed globally and marked idempotent via a class
    attribute; without cleanup the first test would hide failures in the
    rest, and the patch would leak into unrelated tests.
    """
    torch = pytest.importorskip("torch")
    try:
        from torch.distributed.device_mesh import DeviceMesh
    except Exception:  # pragma: no cover - torch without distributed
        pytest.skip("torch.distributed.device_mesh unavailable")

    from ezpz.xccl_split_group import _PATCHED_ATTR

    original = getattr(DeviceMesh, "_init_one_process_group", None)
    had_flag = getattr(DeviceMesh, _PATCHED_ATTR, False)
    yield
    if original is not None:
        DeviceMesh._init_one_process_group = original
    if not had_flag and hasattr(DeviceMesh, _PATCHED_ATTR):
        delattr(DeviceMesh, _PATCHED_ATTR)
    del torch


def test_no_op_off_xpu(monkeypatch):
    """On CUDA/CPU the upstream path must be left alone.

    ``ProcessGroupNCCL`` overrides ``supportsSplitting`` correctly, so
    patching there would add risk for no benefit.
    """
    import ezpz.xccl_split_group as mod

    monkeypatch.setattr(mod, "_should_patch", lambda: False)
    assert mod.maybe_install_xccl_split_group_workaround() is False


def test_install_is_idempotent(monkeypatch):
    """Calling twice must not wrap the wrapper."""
    pytest.importorskip("torch")
    from torch.distributed.device_mesh import DeviceMesh

    import ezpz.xccl_split_group as mod

    monkeypatch.setattr(mod, "_should_patch", lambda: True)
    assert mod.maybe_install_xccl_split_group_workaround() is True
    once = DeviceMesh._init_one_process_group
    assert mod.maybe_install_xccl_split_group_workaround() is True
    assert DeviceMesh._init_one_process_group is once, (
        "second install re-wrapped the already-patched function"
    )


def _install_with_fake_backend(monkeypatch, *, supports_splitting: bool):
    """Patch with a fake default PG whose backend reports *supports_splitting*.

    Returns the fake PG so a test can inspect ``bound_device_id``.
    """
    pytest.importorskip("torch")
    import torch

    import ezpz.xccl_split_group as mod

    class _FakeBackend:
        def __init__(self, splits: bool) -> None:
            self.supports_splitting = splits

    class _FakePG:
        def __init__(self, splits: bool) -> None:
            self.bound_device_id = "xpu:3"
            self._backend = _FakeBackend(splits)
            self.seen_bound_device_id: list = []

        def _get_backend(self, _accel):
            return self._backend

    fake_pg = _FakePG(supports_splitting)

    monkeypatch.setattr(mod, "_should_patch", lambda: True)
    monkeypatch.setattr(
        "torch.distributed.distributed_c10d._get_default_group",
        lambda: fake_pg,
    )
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda: torch.device("xpu")
    )

    from torch.distributed.device_mesh import DeviceMesh

    # Record what bound_device_id looks like from INSIDE the upstream call --
    # that is the whole point of the patch.
    def _spy(sub_layout, rank_map, dim_name, backend_override, *a, **k):
        fake_pg.seen_bound_device_id.append(fake_pg.bound_device_id)
        return "called"

    monkeypatch.setattr(
        DeviceMesh, "_init_one_process_group", staticmethod(_spy)
    )
    mod.maybe_install_xccl_split_group_workaround()
    return fake_pg, DeviceMesh


def test_clears_bound_device_id_when_backend_cannot_split(monkeypatch):
    """The xccl case: force the ``new_group`` fallback."""
    fake_pg, DeviceMesh = _install_with_fake_backend(
        monkeypatch, supports_splitting=False
    )
    assert (
        DeviceMesh._init_one_process_group(None, None, "tp", None) == "called"
    )
    assert fake_pg.seen_bound_device_id == [None], (
        "bound_device_id must be None during the call so torch takes the "
        "new_group branch that xccl supports"
    )
    assert fake_pg.bound_device_id == "xpu:3", (
        "bound_device_id must be RESTORED -- FSDP2 needs it for per-device "
        "PG resolution; leaving it cleared deadlocks foreach_all_gather"
    )


def test_leaves_bound_device_id_when_backend_can_split(monkeypatch):
    """NCCL case: upstream fast path, untouched."""
    fake_pg, DeviceMesh = _install_with_fake_backend(
        monkeypatch, supports_splitting=True
    )
    assert (
        DeviceMesh._init_one_process_group(None, None, "tp", None) == "called"
    )
    assert fake_pg.seen_bound_device_id == ["xpu:3"], (
        "a splitting-capable backend must see the original bound_device_id"
    )


def test_forwards_extra_positional_args(monkeypatch):
    """torch 2.15 added a 5th positional arg to the upstream signature.

    Pinning to four parameters raised ``TypeError`` during device-mesh
    construction, before any model code ran. The shim must pass through
    whatever upstream adds.
    """
    fake_pg, DeviceMesh = _install_with_fake_backend(
        monkeypatch, supports_splitting=False
    )
    got = DeviceMesh._init_one_process_group(
        None, None, "tp", None, True, some_kwarg=1
    )
    assert got == "called"


def test_restores_bound_device_id_on_exception(monkeypatch):
    """A failure inside upstream must not leave the PG unbound.

    Leaving ``bound_device_id=None`` after an error would convert a loud
    mesh failure into a silent FSDP2 all-gather deadlock later.
    """
    pytest.importorskip("torch")
    import torch

    import ezpz.xccl_split_group as mod

    class _FakeBackend:
        supports_splitting = False

    class _FakePG:
        def __init__(self) -> None:
            self.bound_device_id = "xpu:3"

        def _get_backend(self, _accel):
            return _FakeBackend()

    fake_pg = _FakePG()
    monkeypatch.setattr(mod, "_should_patch", lambda: True)
    monkeypatch.setattr(
        "torch.distributed.distributed_c10d._get_default_group",
        lambda: fake_pg,
    )
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda: torch.device("xpu")
    )

    from torch.distributed.device_mesh import DeviceMesh

    def _boom(*_a, **_k):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(
        DeviceMesh, "_init_one_process_group", staticmethod(_boom)
    )
    mod.maybe_install_xccl_split_group_workaround()

    with pytest.raises(RuntimeError, match="upstream exploded"):
        DeviceMesh._init_one_process_group(None, None, "tp", None)
    assert fake_pg.bound_device_id == "xpu:3"


def test_falls_through_when_no_default_group(monkeypatch):
    """No PG yet: call through so upstream raises its own error."""
    pytest.importorskip("torch")

    import ezpz.xccl_split_group as mod

    monkeypatch.setattr(mod, "_should_patch", lambda: True)

    def _no_pg():
        raise RuntimeError("Default process group has not been initialized")

    monkeypatch.setattr(
        "torch.distributed.distributed_c10d._get_default_group", _no_pg
    )

    from torch.distributed.device_mesh import DeviceMesh

    monkeypatch.setattr(
        DeviceMesh,
        "_init_one_process_group",
        staticmethod(lambda *a, **k: "passthrough"),
    )
    mod.maybe_install_xccl_split_group_workaround()
    assert (
        DeviceMesh._init_one_process_group(None, None, "tp", None)
        == "passthrough"
    )
