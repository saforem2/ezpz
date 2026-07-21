"""Equivalence harness for ``ezpz.examples.fsdp_tp`` cross-entropy impls.

The training loop selects a CE implementation via ``--loss-impl``
(dispatched through ``_compute_loss``). Every non-default impl MUST be a
numerically faithful drop-in for the plain ``_cross_entropy_eager`` —
matching both the **loss value** and the **gradient w.r.t. logits**,
including ``ignore_index`` (``-100``) labels.

This is the correctness gate that lets us trust the memory-bounded impls
(``chunked``, ``chunked-backward``, ``fused-linear``, ``loss-parallel``).
It runs on CPU; no XPU/accelerator required.
"""

from __future__ import annotations

import pytest

try:
    import torch

    import ezpz.examples.fsdp_tp as fsdp_tp

    LOSS_AVAILABLE = True
except Exception:  # noqa: BLE001 - heavy optional deps may be missing
    LOSS_AVAILABLE = False


# CE impls that operate on already-materialized logits and are expected to
# match eager loss+grad. (loss-parallel/fused-linear are added with their own
# TP-aware tests when implemented.)
LOGIT_IMPLS = ["chunked", "chunked-backward", "compiled"]


def _loss_and_grad(impl, logits, labels, *, chunk_size=1024):
    lg = logits.clone().detach().requires_grad_(True)
    loss = fsdp_tp._compute_loss(
        lg, labels, impl=impl, ignore_index=-100, chunk_size=chunk_size
    )
    loss.backward()
    return loss.detach(), lg.grad.detach()


@pytest.mark.skipif(not LOSS_AVAILABLE, reason="ezpz.examples.fsdp_tp not importable")
class TestLossImplEquivalence:
    @pytest.mark.parametrize("impl", LOGIT_IMPLS)
    @pytest.mark.parametrize(
        "shape", [(2, 16, 50), (1, 257, 128), (3, 8, 1000)], ids=["small", "long", "wide"]
    )
    def test_matches_eager_loss_and_grad(self, impl, shape):
        if impl == "compiled":
            pytest.importorskip("torch._dynamo")
        torch.manual_seed(0)
        B, T, V = shape
        logits = torch.randn(B, T, V, dtype=torch.float32)
        labels = torch.randint(0, V, (B, T))
        labels[0, ::5] = -100  # exercise ignore_index

        le, ge = _loss_and_grad("eager", logits, labels)
        try:
            lv, gv = _loss_and_grad(impl, logits, labels, chunk_size=7)
        except Exception as exc:  # torch.compile may be unavailable on CPU
            if impl == "compiled":
                pytest.skip(f"compiled CE unavailable: {exc}")
            raise

        assert torch.allclose(lv, le, atol=1e-5, rtol=1e-5), (
            f"{impl} loss {lv} != eager {le}"
        )
        assert torch.allclose(gv, ge, atol=1e-5, rtol=1e-5), (
            f"{impl} grad max-diff {(gv - ge).abs().max()}"
        )

    @pytest.mark.parametrize("impl", ["chunked", "chunked-backward"])
    def test_chunk_size_invariant(self, impl):
        """Chunked impls must be identical regardless of chunk_size (loss +
        grad) — especially chunked-backward's hand-rolled backward.
        """
        torch.manual_seed(1)
        logits = torch.randn(2, 64, 200, dtype=torch.float32)
        labels = torch.randint(0, 200, (2, 64))
        labels[1, ::3] = -100
        ref_l, ref_g = _loss_and_grad("eager", logits, labels)
        for cs in (1, 13, 64, 100000):
            lv, gv = _loss_and_grad(impl, logits, labels, chunk_size=cs)
            assert torch.allclose(lv, ref_l, atol=1e-5), f"{impl} cs={cs} loss"
            assert torch.allclose(gv, ref_g, atol=1e-5), f"{impl} cs={cs} grad"

    def test_all_ignored_is_finite(self):
        """All-ignored microbatch: eager yields NaN (0/0); the memory-bounded
        impls clamp the denominator to 1 and yield a finite 0 loss + finite
        grad. We assert the *finite* behavior (the safer contract); this is a
        documented, intentional divergence from eager's NaN on a pathological
        all-padding microbatch.
        """
        torch.manual_seed(2)
        logits = torch.randn(2, 4, 10, dtype=torch.float32)
        labels = torch.full((2, 4), -100)
        for impl in ("chunked", "chunked-backward"):
            lv, gv = _loss_and_grad(impl, logits, labels, chunk_size=3)
            assert torch.isfinite(lv).all(), f"{impl} loss not finite"
            assert torch.isfinite(gv).all(), f"{impl} grad not finite"


@pytest.mark.skipif(not LOSS_AVAILABLE, reason="ezpz.examples.fsdp_tp not importable")
class TestFusedLinearCE:
    """fused-linear runs the output projection MODULE per row-chunk (under
    checkpoint) and never materializes the full logits. Verify loss AND both
    gradients (grad_h via hidden, grad_W via the module weight) match eager
    CE over the full ``output_module(h)``.
    """

    @staticmethod
    def _make(N, D, V, seed=0):
        torch.manual_seed(seed)
        h = torch.randn(N, D, dtype=torch.float32)
        out = torch.nn.Linear(D, V, bias=False)
        labels = torch.randint(0, V, (N,))
        labels[::5] = -100
        return h, out, labels

    @pytest.mark.parametrize(
        "shape", [(16, 8, 50), (257, 16, 128), (8, 32, 1000)], ids=["small", "long", "wide"]
    )
    def test_matches_eager_loss_and_both_grads(self, shape):
        import torch.nn.functional as F

        N, D, V = shape
        h, out, labels = self._make(N, D, V)

        # reference: eager CE over the full module output
        he = h.clone().requires_grad_(True)
        out_ref = torch.nn.Linear(D, V, bias=False)
        out_ref.load_state_dict(out.state_dict())
        ref = F.cross_entropy(out_ref(he), labels, ignore_index=-100)
        ref.backward()

        hf = h.clone().requires_grad_(True)
        try:
            lf = fsdp_tp._cross_entropy_fused_linear(
                hf, out, labels, ignore_index=-100, chunk_size=7
            )
            lf.backward()
        except OSError as exc:
            # torch.utils.checkpoint probes accelerator devices for RNG state;
            # on a CPU-only host without the XPU/CUDA loader libs that raises.
            # The grad path is exercised in CI / on-node; skip here.
            pytest.skip(f"checkpoint device probe unavailable on this host: {exc}")

        assert torch.allclose(lf, ref, atol=1e-4), f"loss {lf} != {ref}"
        assert torch.allclose(hf.grad, he.grad, atol=1e-4), "grad_h mismatch"
        assert torch.allclose(
            out.weight.grad, out_ref.weight.grad, atol=1e-4
        ), "grad_W mismatch"

    def test_forward_matches_eager_no_checkpoint(self):
        """Forward equivalence on the no-grad path (no checkpoint, so it runs
        even on hosts without an accelerator device loader). Locks the chunked
        loss summation math independent of the checkpoint backward.
        """
        import torch.nn.functional as F

        h, out, labels = self._make(40, 16, 300, seed=3)
        with torch.no_grad():
            ref = F.cross_entropy(out(h), labels, ignore_index=-100)
            for cs in (1, 7, 40, 99999):
                fused = fsdp_tp._cross_entropy_fused_linear(
                    h, out, labels, ignore_index=-100, chunk_size=cs
                )
                assert torch.allclose(fused, ref, atol=1e-5), f"cs={cs} forward"

    def test_chunk_size_invariant(self):
        import torch.nn.functional as F

        h, out, labels = self._make(64, 16, 200, seed=1)
        he = h.clone().requires_grad_(True)
        out_ref = torch.nn.Linear(16, 200, bias=False)
        out_ref.load_state_dict(out.state_dict())
        ref = F.cross_entropy(out_ref(he), labels, ignore_index=-100)
        ref.backward()
        for cs in (1, 9, 64, 100000):
            for p in out.parameters():
                p.grad = None
            hf = h.clone().requires_grad_(True)
            try:
                lf = fsdp_tp._cross_entropy_fused_linear(
                    hf, out, labels, ignore_index=-100, chunk_size=cs
                )
                lf.backward()
            except OSError as exc:
                pytest.skip(f"checkpoint device probe unavailable: {exc}")
            assert torch.allclose(lf, ref, atol=1e-4), f"cs={cs} loss"
            assert torch.allclose(hf.grad, he.grad, atol=1e-4), f"cs={cs} grad_h"
            assert torch.allclose(
                out.weight.grad, out_ref.weight.grad, atol=1e-4
            ), f"cs={cs} grad_W"


# ---------------------------------------------------------------------------
# Vocab-parallel CE: real 2-rank gloo test (the simulation in dev verified the
# math; this exercises the actual funcol all-reduces + process group, so a
# collective/group bug can't slip through). Skipped if gloo isn't usable.
# ---------------------------------------------------------------------------

# Shared fixture data so both ranks + the reference agree (seeded, not random).
_VP_N, _VP_V = 8, 12  # rows, global vocab (split across 2 ranks: 6 + 6)


def _free_port() -> int:
    """Bind to port 0 to let the OS hand us a free TCP port, then release it."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _vp_worker(rank, world_size, vocab, n_rows, master_port, ret):
    import os

    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    # Port is chosen (free) by the parent and passed in, rather than
    # `setdefault`-ing a hard-coded value: setdefault would inherit an
    # already-set (possibly busy) MASTER_PORT from the parent environment and
    # hang/flake on collision.
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        import ezpz.examples.fsdp_tp as m

        torch.manual_seed(123)
        full_logits = torch.randn(n_rows, vocab, dtype=torch.float32)
        labels = torch.randint(0, vocab, (n_rows,))
        labels[::4] = -100

        chunk = (vocab + world_size - 1) // world_size
        s, e = chunk * rank, min(vocab, chunk * (rank + 1))
        local = full_logits[:, s:e].clone().detach().requires_grad_(True)

        loss = m._cross_entropy_vocab_parallel(
            local,
            labels,
            ignore_index=-100,
            global_vocab_size=vocab,
            tp_group=dist.group.WORLD,
        )
        loss.backward()
        # rank 0 reports loss + its grad shard for comparison to eager.
        # NOTE: store grad0 as a plain nested list, NOT a torch.Tensor.
        # Putting a Tensor into a multiprocessing.Manager().dict() routes
        # it through torch's shared-memory / file-descriptor reducer, and
        # handing that FD to the Manager server process HANGS under the
        # `spawn` start method (default on macOS) — gloo then sits at its
        # 30-min rendezvous timeout, looking like the whole suite froze.
        # `.tolist()` serializes as pure Python; the parent rebuilds a
        # tensor for the allclose check.
        if rank == 0:
            ret["loss"] = float(loss.detach())
            ret["grad0"] = local.grad.detach().tolist()
            ret["shard"] = (s, e)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not LOSS_AVAILABLE, reason="ezpz.examples.fsdp_tp not importable")
@pytest.mark.parametrize(
    "vocab",
    [
        12,  # even: 2 ranks -> 6 + 6 (divisible)
        13,  # uneven: 2 ranks -> 7 + 6, exercises the uneven-shard math
        # (local_vocab computation, out-of-range label clamping, ragged gather)
    ],
)
def test_vocab_parallel_matches_eager_2rank(vocab):
    import torch
    import torch.multiprocessing as mp

    port = _free_port()
    try:
        mgr = mp.Manager()
        ret = mgr.dict()
        mp.spawn(
            _vp_worker, args=(2, vocab, _VP_N, port, ret), nprocs=2, join=True
        )
    except Exception as exc:  # noqa: BLE001 - gloo/spawn may be unavailable
        pytest.skip(f"2-rank gloo unavailable: {exc}")

    assert "loss" in ret, "rank 0 did not report"
    # Reference: full-vocab eager CE on the SAME seeded data.
    torch.manual_seed(123)
    full = torch.randn(_VP_N, vocab, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, vocab, (_VP_N,))
    labels[::4] = -100
    import torch.nn.functional as F

    ref = F.cross_entropy(full, labels, ignore_index=-100)
    ref.backward()
    # `full` requires grad, so `ref` tracks grad too; detach before the
    # Python-float conversion to avoid a "converting a tensor with
    # requires_grad=True to a scalar" UserWarning (the value is unaffected).
    ref_val = ref.detach().item()
    s, e = ret["shard"]
    assert abs(ret["loss"] - ref_val) < 1e-4, (
        f"vp loss {ret['loss']} != eager {ref_val} (vocab={vocab})"
    )
    # grad0 came back as a plain nested list (see _vp_worker) — rebuild a
    # tensor for the comparison, matching the eager grad's dtype and device
    # so the check stays accurate if the reference ever runs in another
    # precision / on an accelerator.
    grad0 = torch.tensor(
        ret["grad0"], dtype=full.grad.dtype, device=full.grad.device
    )
    assert torch.allclose(grad0, full.grad[:, s:e], atol=1e-4), (
        f"vocab-parallel grad shard != eager grad shard (vocab={vocab})"
    )


# ---------------------------------------------------------------------------
# Replicate-DTensor loss localization: at tp>1 (non-loss-parallel), the output
# projection returns a REPLICATED DTensor. The train loop must localize it
# (pred.to_local()) before eager/chunked/compiled CE, else F.cross_entropy
# raises "got mixed torch.Tensor and DTensor". This builds a real Replicate
# DTensor on a 1-rank gloo mesh and asserts to_local() -> plain CE matches
# plain eager (loss + grad), and that the differentiable to_local() routes the
# grad back to the DTensor. Skipped if gloo/DeviceMesh aren't usable.
# ---------------------------------------------------------------------------


def _replicate_dtensor_worker(rank, world_size, master_port, ret):
    import os

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        import ezpz.examples.fsdp_tp as fsdp_tp
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Replicate

        n, vocab = 6, 16
        torch.manual_seed(7)
        full = torch.randn(n, vocab, dtype=torch.float32)
        labels = torch.randint(0, vocab, (n,))
        labels[::3] = -100

        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
        # Replicated DTensor built from a differentiable local leaf, mirroring
        # the non-loss-parallel output projection (output_layouts=Replicate(),
        # use_local_output=False). from_local (not distribute_tensor) keeps the
        # autograd chain to `leaf` so we can assert grad routes back through
        # to_local().
        leaf = full.clone().detach().requires_grad_(True)
        dpred = DTensor.from_local(leaf, mesh, [Replicate()], run_check=False)

        # Exercise the ACTUAL production helper the train loop calls, so a
        # revert of the fix (e.g. dropping the to_local localization) fails
        # this test rather than silently passing.
        local = fsdp_tp._localize_logits_for_loss(dpred)
        assert not hasattr(local, "to_local"), (
            "_localize_logits_for_loss must return a plain tensor for a "
            "Replicate DTensor"
        )
        loss = F.cross_entropy(
            local.reshape(-1, local.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()

        ret["loss"] = float(loss.detach().item())
        # grad flowed back through to_local() into the DTensor -> local leaf.
        ret["has_grad"] = leaf.grad is not None
        ret["grad"] = leaf.grad.tolist() if leaf.grad is not None else None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not LOSS_AVAILABLE, reason="ezpz.examples.fsdp_tp not importable"
)
def test_replicate_dtensor_to_local_matches_eager_1rank():
    import torch
    import torch.multiprocessing as mp
    import torch.nn.functional as F

    port = _free_port()
    mgr = mp.Manager()
    ret = mgr.dict()
    try:
        mp.spawn(
            _replicate_dtensor_worker, args=(1, port, ret), nprocs=1, join=True
        )
    except (RuntimeError, OSError) as exc:
        # Only skip for genuine infra/unavailability (gloo transport,
        # DeviceMesh init, spawn). Real regressions in the worker surface as
        # AssertionError (from the in-worker asserts) or other exceptions and
        # must NOT be swallowed — re-raise those.
        msg = str(exc).lower()
        if any(
            k in msg
            for k in ("gloo", "process group", "devicemesh", "backend", "connect")
        ):
            pytest.skip(f"1-rank gloo/DeviceMesh unavailable: {exc}")
        raise

    assert "loss" in ret, "worker did not report"
    assert ret["has_grad"], (
        "to_local() must be differentiable so grad reaches the DTensor"
    )

    # Reference: plain eager CE on the same seeded data (no DTensor).
    torch.manual_seed(7)
    full = torch.randn(6, 16, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, 16, (6,))
    labels[::3] = -100
    ref = F.cross_entropy(full, labels, ignore_index=-100)
    ref.backward()

    assert abs(ret["loss"] - ref.detach().item()) < 1e-5, (
        f"Replicate-DTensor localized loss {ret['loss']} != eager {ref.item()}"
    )
    grad = torch.tensor(ret["grad"], dtype=full.grad.dtype)
    assert torch.allclose(grad, full.grad, atol=1e-5), (
        "grad through to_local() != eager grad"
    )


@pytest.mark.skipif(
    not LOSS_AVAILABLE, reason="ezpz.examples.fsdp_tp not importable"
)
def test_localize_logits_plain_tensor_is_noop():
    """Fast CPU unit test (no process group): a plain tensor passes through
    _localize_logits_for_loss unchanged. Locks the tp=1 / HF no-op contract."""
    x = torch.randn(4, 8)
    out = fsdp_tp._localize_logits_for_loss(x)
    assert out is x  # identity, no copy, no wrap


@pytest.mark.skipif(
    not LOSS_AVAILABLE, reason="ezpz.examples.fsdp_tp not importable"
)
def test_localize_logits_rejects_vocab_sharded_dtensor():
    """A Shard(-1) (vocab-parallel) DTensor must be rejected, not silently
    localized to a partial-vocab tensor. Uses a lightweight stub with the
    DTensor duck-type (to_local + placements) so no process group is needed."""
    from torch.distributed.tensor import Shard

    class _FakeShardedDT:
        placements = (Shard(-1),)

        def to_local(self):  # pragma: no cover - must raise before this
            raise AssertionError("to_local must not be reached for Shard(-1)")

    with pytest.raises(RuntimeError, match="loss-parallel"):
        fsdp_tp._localize_logits_for_loss(_FakeShardedDT())
