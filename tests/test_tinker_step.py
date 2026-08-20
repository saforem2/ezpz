"""Tests for the extracted step in :mod:`ezpz.tinker.step`.

Two things need proving, and they pull in opposite directions:

* **Equivalence.** For a 1:1 caller (one ``forward_backward``, one
  ``optim_step``) the split must behave exactly like the fused loop in
  ``fsdp_tp.py``. ``test_split_matches_fused_loop`` runs both against the
  same seeded model and asserts the resulting weights are bit-identical.
* **New capability.** Gradient accumulation must actually work --
  impossible in the fused loop, because ``zero_grad`` sat immediately
  before ``backward()`` and discarded everything but the last microbatch.
  ``test_accumulation_matches_one_big_batch`` is the proof, and
  ``test_old_zero_grad_placement_would_break_accumulation`` demonstrates
  the bug the move fixes.

CPU-only; no distributed init, no accelerator.
"""

from __future__ import annotations

import argparse

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from ezpz.tinker.step import (  # noqa: E402
    TrainState,
    forward_backward,
    optim_step,
    prepare_batch,
)
from ezpz.tinker.types import AdamParams  # noqa: E402


class TinyLM(nn.Module):
    """Smallest thing with the shape the step expects: (B,T) -> (B,T,V)."""

    def __init__(self, vocab: int = 32, dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.output = nn.Linear(dim, vocab, bias=False)

    def forward(self, x, return_hidden: bool = False):
        h = self.emb(x)
        return h if return_hidden else self.output(h)


def _args(**over):
    base = dict(
        loss_impl="eager",
        loss_chunk_size=1024,
        vocab_size=32,
        max_grad_norm=1.0,
        dataset="random",
    )
    base.update(over)
    return argparse.Namespace(**base)


def _eager_ce(logits, labels, *, impl="eager", ignore_index=-100, chunk_size=0):
    """Stand-in for fsdp_tp._compute_loss with the same signature."""
    return nn.functional.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        ignore_index=ignore_index,
    )


def _state(model, optimizer, **over):
    kwargs = dict(
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        args=_args(),
        base_model=model,
        dataset=None,
        compute_loss=_eager_ce,
        localize_logits=lambda t: t,
    )
    kwargs.update(over)
    return TrainState(**kwargs)


def _batch(seed: int, b: int = 2, t: int = 9, vocab: int = 32):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (b, t), generator=g)


def _fresh(seed: int = 0, lr: float = 0.1):
    torch.manual_seed(seed)
    m = TinyLM()
    return m, torch.optim.SGD(m.parameters(), lr=lr)


# ---------------------------------------------------------------------------
# prepare_batch
# ---------------------------------------------------------------------------


class TestPrepareBatch:
    def test_causal_shift(self):
        m, o = _fresh()
        x = torch.arange(12).reshape(2, 6)
        p = prepare_batch(_state(m, o), x)
        assert p.inp.shape == (2, 5) and p.labels.shape == (2, 5)
        torch.testing.assert_close(p.inp[0], torch.arange(0, 5))
        torch.testing.assert_close(p.labels[0], torch.arange(1, 6))

    def test_dict_batch_with_attention_mask(self):
        m, o = _fresh()
        p = prepare_batch(
            _state(m, o),
            {"input_ids": torch.ones(2, 6, dtype=torch.long),
             "attention_mask": torch.ones(2, 6, dtype=torch.long)},
        )
        assert p.attn_mask is not None and p.inp.shape == (2, 5)

    def test_casts_to_long(self):
        m, o = _fresh()
        p = prepare_batch(_state(m, o), torch.ones(2, 6, dtype=torch.int32))
        assert p.inp.dtype == torch.long


# ---------------------------------------------------------------------------
# Equivalence with the fused loop
# ---------------------------------------------------------------------------


class TestEquivalenceWithFusedLoop:
    @staticmethod
    def _fused_reference(model, optimizer, batches, args):
        """The original loop body, verbatim from fsdp_tp.py:3272-3423."""
        losses = []
        for batch in batches:
            x = batch.to(torch.long)
            inp, labels = x[:, :-1], x[:, 1:]
            pred = model(inp)
            loss = _eager_ce(pred, labels)
            optimizer.zero_grad(set_to_none=True)   # :3416 -- BEFORE backward
            loss.backward()                          # :3417
            if args.max_grad_norm > 0:               # :3419-3422
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
            optimizer.step()                         # :3423
            losses.append(float(loss.detach()))
        return losses

    def test_split_matches_fused_loop(self):
        """1:1 caller => bit-identical weights and losses."""
        batches = [_batch(i) for i in range(5)]

        m_ref, o_ref = _fresh()
        ref_losses = self._fused_reference(m_ref, o_ref, batches, _args())

        m_new, o_new = _fresh()
        st = _state(m_new, o_new)
        new_losses = []
        for b in batches:
            new_losses.append(forward_backward(st, b).loss)
            optim_step(st)

        for i, (a, b) in enumerate(zip(ref_losses, new_losses)):
            assert a == b, f"loss diverged at step {i}: {a} != {b}"
        for (n, p_ref), (_, p_new) in zip(
            m_ref.named_parameters(), m_new.named_parameters()
        ):
            torch.testing.assert_close(p_new, p_ref, rtol=0, atol=0,
                                       msg=f"param {n} diverged")

    def test_global_step_advances_on_optim_step_only(self):
        m, o = _fresh()
        st = _state(m, o)
        forward_backward(st, _batch(0))
        forward_backward(st, _batch(1))
        assert st.global_step == 0, "forward_backward must not advance the step"
        assert optim_step(st).step == 1
        assert st.global_step == 1


# ---------------------------------------------------------------------------
# Gradient accumulation -- the capability the split unlocks
# ---------------------------------------------------------------------------


class TestGradientAccumulation:
    def test_accumulation_matches_one_big_batch(self):
        """N microbatches at 1/N scale == one batch of the same total size.

        Only true because zero_grad moved out of the forward path.
        """
        big = _batch(7, b=4)
        micro = [big[:2], big[2:]]

        m_big, o_big = _fresh()
        st_big = _state(m_big, o_big)
        forward_backward(st_big, big)
        optim_step(st_big)

        m_acc, o_acc = _fresh()
        st_acc = _state(m_acc, o_acc)
        for mb in micro:
            forward_backward(st_acc, mb, loss_scale=1.0 / len(micro))
        optim_step(st_acc)

        for (n, p_big), (_, p_acc) in zip(
            m_big.named_parameters(), m_acc.named_parameters()
        ):
            torch.testing.assert_close(
                p_acc, p_big, rtol=1e-5, atol=1e-6,
                msg=f"param {n} differs between accumulated and single batch",
            )

    def test_old_zero_grad_placement_would_break_accumulation(self):
        """Demonstrates the bug the zero_grad move fixes.

        Zeroing before each backward (the old placement) keeps only the
        LAST microbatch's gradient, so the update is wrong. Pinned so
        nobody "tidies" zero_grad back into forward_backward.
        """
        micro = [_batch(11, b=2), _batch(12, b=2)]

        m_ok, o_ok = _fresh()
        st_ok = _state(m_ok, o_ok)
        for mb in micro:
            forward_backward(st_ok, mb, loss_scale=0.5)
        g_accumulated = m_ok.output.weight.grad.clone()

        m_bad, o_bad = _fresh()
        st_bad = _state(m_bad, o_bad)
        for mb in micro:
            o_bad.zero_grad(set_to_none=True)   # the old placement
            forward_backward(st_bad, mb, loss_scale=0.5)
        g_last_only = m_bad.output.weight.grad.clone()

        assert not torch.allclose(g_accumulated, g_last_only), (
            "accumulated grad should differ from last-microbatch-only; if "
            "these match, zero_grad has crept back into the forward path"
        )

    def test_grads_cleared_after_optim_step(self):
        m, o = _fresh()
        st = _state(m, o)
        forward_backward(st, _batch(3))
        assert m.output.weight.grad is not None
        optim_step(st)
        assert m.output.weight.grad is None


# ---------------------------------------------------------------------------
# optim_step
# ---------------------------------------------------------------------------


class TestOptimStep:
    def test_adam_params_override_lr(self):
        m, o = _fresh(lr=0.1)
        st = _state(m, o)
        forward_backward(st, _batch(0))
        resp = optim_step(st, AdamParams(learning_rate=0.005))
        assert resp.learning_rate == pytest.approx(0.005)
        assert o.param_groups[0]["lr"] == pytest.approx(0.005)

    def test_defaults_to_args_max_grad_norm(self):
        m, o = _fresh()
        st = _state(m, o, args=_args(max_grad_norm=0.5))
        forward_backward(st, _batch(0))
        assert optim_step(st).grad_norm is not None

    def test_no_clipping_when_disabled(self):
        m, o = _fresh()
        st = _state(m, o, args=_args(max_grad_norm=0.0))
        forward_backward(st, _batch(0))
        assert optim_step(st).grad_norm is None

    def test_profiler_advances_per_optim_step_not_microbatch(self):
        class P:
            def __init__(self): self.n = 0
            def step(self): self.n += 1

        prof = P()
        m, o = _fresh()
        st = _state(m, o, profiler=prof)
        forward_backward(st, _batch(0))
        forward_backward(st, _batch(1))
        assert prof.n == 0
        optim_step(st)
        assert prof.n == 1


# ---------------------------------------------------------------------------
# hooks / plumbing
# ---------------------------------------------------------------------------


class TestHooks:
    def test_on_forward_done_fires_between_forward_and_loss(self):
        """fsdp_tp's t1 barrier depends on this ordering."""
        m, o = _fresh()
        st = _state(m, o)
        seen = []
        real = st.compute_loss

        def spy(*a, **k):
            seen.append("loss")
            return real(*a, **k)

        st.compute_loss = spy
        forward_backward(
            st, _batch(0),
            on_forward_done=lambda pred, labels, n: seen.append("hook"),
        )
        assert seen == ["hook", "loss"]

    def test_rejects_unimplemented_loss(self):
        m, o = _fresh()
        with pytest.raises(NotImplementedError, match="reserved"):
            forward_backward(_state(m, o), _batch(0), loss_fn="ppo")

    def test_rejects_unknown_loss(self):
        m, o = _fresh()
        with pytest.raises(ValueError, match="unknown loss_fn"):
            forward_backward(_state(m, o), _batch(0), loss_fn="nope")

    def test_output_module_errors_clearly_when_absent(self):
        st = _state(nn.Linear(4, 4), torch.optim.SGD(nn.Linear(4, 4).parameters(), lr=0.1))
        with pytest.raises(AttributeError, match="fused-linear loss needs"):
            _ = st.output_module

    def test_num_tokens_excludes_ignored(self):
        m, o = _fresh()

        class DS:
            pad_id = 0

        st = _state(m, o, dataset=DS())
        x = torch.zeros(2, 6, dtype=torch.long)  # every label is pad
        assert forward_backward(st, x).num_tokens == 0
