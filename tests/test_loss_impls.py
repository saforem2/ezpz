"""Equivalence harness for ``ezpz.examples.fsdp_tp`` cross-entropy impls.

The training loop selects a CE implementation via ``--loss-impl``
(dispatched through ``_compute_loss``). Every non-default impl MUST be a
numerically faithful drop-in for the plain ``_cross_entropy_eager`` —
matching both the **loss value** and the **gradient w.r.t. logits**,
including ``ignore_index`` (``-100``) labels.

This is the correctness gate that lets us trust the memory-bounded impls
(``chunked``, ``chunked-backward``, and — when added — ``fused-linear`` /
``loss-parallel``). It runs on CPU; no XPU/accelerator required.
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

    def test_chunk_size_invariant(self):
        """chunked-backward must be identical regardless of chunk_size."""
        torch.manual_seed(1)
        logits = torch.randn(2, 64, 200, dtype=torch.float32)
        labels = torch.randint(0, 200, (2, 64))
        labels[1, ::3] = -100
        ref_l, ref_g = _loss_and_grad("eager", logits, labels)
        for cs in (1, 13, 64, 100000):
            lv, gv = _loss_and_grad("chunked-backward", logits, labels, chunk_size=cs)
            assert torch.allclose(lv, ref_l, atol=1e-5), f"chunk_size={cs} loss"
            assert torch.allclose(gv, ref_g, atol=1e-5), f"chunk_size={cs} grad"

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
