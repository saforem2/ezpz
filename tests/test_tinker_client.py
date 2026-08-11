"""Tests for :mod:`ezpz.tinker.client`.

The client is a thin facade, so these focus on the parts where thin
wrappers usually go wrong:

* microbatch scaling (does N batches through the client equal one big
  batch?);
* loss averaging over *ragged* microbatches, where a plain mean is
  subtly wrong;
* the future shape, which exists so a remote backend can be dropped in
  without touching call sites.

CPU-only.
"""

from __future__ import annotations

import argparse

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from ezpz.tinker.client import (  # noqa: E402
    ImmediateFuture,
    LocalSamplingClient,
    LocalTrainingClient,
    build_training_client,
)
from ezpz.tinker.step import TrainState  # noqa: E402
from ezpz.tinker.types import AdamParams  # noqa: E402


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.output = nn.Linear(dim, vocab, bias=False)

    def forward(self, x, return_hidden: bool = False):
        h = self.emb(x)
        return h if return_hidden else self.output(h)


def _eager_ce(logits, labels, *, impl="eager", ignore_index=-100, chunk_size=0):
    return nn.functional.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        ignore_index=ignore_index,
    )


def _client(seed: int = 0, lr: float = 0.1, **over):
    torch.manual_seed(seed)
    model = TinyLM()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    kwargs = dict(
        model=model,
        optimizer=opt,
        device=torch.device("cpu"),
        args=argparse.Namespace(
            loss_impl="eager",
            loss_chunk_size=1024,
            vocab_size=32,
            max_grad_norm=1.0,
            dataset="random",
        ),
        base_model=model,
        dataset=None,
        compute_loss=_eager_ce,
        localize_logits=lambda t: t,
    )
    kwargs.update(over)
    return LocalTrainingClient(TrainState(**kwargs)), model, opt


def _batch(seed: int, b: int = 2, t: int = 9, vocab: int = 32):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (b, t), generator=g)


class TestImmediateFuture:
    def test_result_and_done(self):
        f = ImmediateFuture(42)
        assert f.result() == 42 and f.done()

    def test_result_ignores_timeout(self):
        assert ImmediateFuture("x").result(timeout=0.001) == "x"


class TestForwardBackward:
    def test_returns_future(self):
        c, _, _ = _client()
        assert isinstance(c.forward_backward([_batch(0)]), ImmediateFuture)

    def test_rejects_empty_data(self):
        c, _, _ = _client()
        with pytest.raises(ValueError, match="at least one batch"):
            c.forward_backward([])

    def test_microbatches_equal_one_big_batch(self):
        """The client must scale by 1/N so callers need not remember to."""
        big = _batch(7, b=4)

        c_big, m_big, _ = _client()
        c_big.forward_backward([big]).result()
        c_big.optim_step().result()

        c_acc, m_acc, _ = _client()
        c_acc.forward_backward([big[:2], big[2:]]).result()
        c_acc.optim_step().result()

        for (n, a), (_, b) in zip(
            m_big.named_parameters(), m_acc.named_parameters()
        ):
            torch.testing.assert_close(
                b, a, rtol=1e-5, atol=1e-6, msg=f"param {n} differs"
            )

    def test_loss_is_token_weighted_not_plain_mean(self):
        """Ragged microbatches: a plain mean over-weights the short one.

        Build two microbatches with very different token counts and
        confirm the reported loss is the token-weighted average.
        """
        c, _, _ = _client()
        short = _batch(1, b=1, t=5)
        long = _batch(2, b=3, t=17)
        out = c.forward_backward([short, long]).result()

        # Recompute the two losses independently at the same weights.
        c2, _, _ = _client()
        s_out = c2.forward_backward([short]).result()
        c3, _, _ = _client()
        l_out = c3.forward_backward([long]).result()

        weighted = (
            s_out.loss * s_out.num_tokens + l_out.loss * l_out.num_tokens
        ) / (s_out.num_tokens + l_out.num_tokens)
        plain = (s_out.loss + l_out.loss) / 2

        assert out.loss == pytest.approx(weighted, rel=1e-5)
        if abs(weighted - plain) > 1e-6:
            assert abs(out.loss - plain) > 1e-9, "looks like a plain mean"

    def test_reports_microbatch_count(self):
        c, _, _ = _client()
        out = c.forward_backward([_batch(0), _batch(1), _batch(2)]).result()
        assert out.metrics["num_microbatches"] == 3.0

    def test_accumulates_total_tokens(self):
        c, _, _ = _client()
        out = c.forward_backward([_batch(0, b=2, t=9), _batch(1, b=2, t=9)]).result()
        assert out.num_tokens == 2 * 2 * 8  # (t-1) labels per row


class TestOptimStep:
    def test_advances_step_and_returns_future(self):
        c, _, _ = _client()
        c.forward_backward([_batch(0)]).result()
        resp = c.optim_step().result()
        assert resp.step == 1 and c.step == 1

    def test_adam_params_applied(self):
        c, _, opt = _client(lr=0.1)
        c.forward_backward([_batch(0)]).result()
        c.optim_step(AdamParams(learning_rate=0.007)).result()
        assert opt.param_groups[0]["lr"] == pytest.approx(0.007)

    def test_forward_backward_alone_does_not_step(self):
        c, _, _ = _client()
        c.forward_backward([_batch(0)]).result()
        assert c.step == 0


class TestTokenizer:
    def test_prefers_dataset_tokenizer(self):
        class DS:
            tokenizer = "from-dataset"

        c, _, _ = _client(dataset=DS())
        assert c.get_tokenizer() == "from-dataset"

    def test_falls_back_to_extras(self):
        c, _, _ = _client(extras={"tokenizer": "from-extras"})
        assert c.get_tokenizer() == "from-extras"

    def test_clear_error_when_absent(self):
        c, _, _ = _client()
        with pytest.raises(AttributeError, match="no tokenizer available"):
            c.get_tokenizer()


class TestSamplingClient:
    def test_handoff_returns_sampler_without_saving(self):
        c, model, _ = _client()
        s = c.save_weights_and_get_sampling_client()
        assert isinstance(s, LocalSamplingClient)
        assert s.model is model

    def test_sample_is_honest_about_being_unimplemented(self):
        c, _, _ = _client()
        s = c.save_weights_and_get_sampling_client()
        with pytest.raises(NotImplementedError, match="not wired up yet"):
            s.sample("hello")


class TestBuilder:
    def test_build_training_client(self):
        c, _, _ = _client()
        assert isinstance(build_training_client(c.state), LocalTrainingClient)


class TestRaggedMicrobatchWeighting:
    """Ragged microbatches must weight by tokens, not by count.

    Each microbatch's cross-entropy is already a mean over its own
    tokens, so a uniform `1/N` scale gives a 300-token and a 100-token
    microbatch equal pull where one combined batch would weight them
    0.75 / 0.25. (Codex P1 + Copilot on #207.)
    """

    @staticmethod
    def _padded(rows, t=9, vocab=32, pad=0):
        """rows = list of real-token counts; the rest is pad."""
        out = torch.full((len(rows), t), pad, dtype=torch.long)
        g = torch.Generator().manual_seed(7)
        for i, n in enumerate(rows):
            out[i, :n] = torch.randint(1, vocab, (n,), generator=g)
        return out

    class _DS:
        pad_id = 0

    def test_accumulated_grad_matches_one_combined_batch(self):
        """The whole point of the scale: N microbatches == 1 big batch."""
        long_mb = self._padded([9, 9])
        short_mb = self._padded([3, 3])

        # (a) two ragged microbatches, accumulated
        c1, m1, o1 = _client(seed=1, dataset=self._DS())
        c1.forward_backward([long_mb, short_mb]).result()
        acc = torch.cat([p.grad.flatten() for p in m1.parameters()])

        # (b) the same rows as a single batch
        c2, m2, o2 = _client(seed=1, dataset=self._DS())
        c2.forward_backward([torch.cat([long_mb, short_mb], dim=0)]).result()
        combined = torch.cat([p.grad.flatten() for p in m2.parameters()])

        cos = torch.nn.functional.cosine_similarity(
            acc.unsqueeze(0), combined.unsqueeze(0)
        ).item()
        assert cos > 0.999, (
            f"accumulated gradient diverges from the combined batch "
            f"(cosine {cos:.4f}); microbatches are not token-weighted"
        )

    def test_zero_token_microbatch_contributes_nothing_to_loss(self):
        """`max(num_tokens, 1)` gave an all-pad microbatch full weight."""
        real = self._padded([9, 9])
        empty = torch.zeros((2, 9), dtype=torch.long)  # all pad -> 0 valid

        c, _, _ = _client(seed=2, dataset=self._DS())
        with_empty = c.forward_backward([real, empty]).result().loss

        c2, _, _ = _client(seed=2, dataset=self._DS())
        alone = c2.forward_backward([real]).result().loss

        assert with_empty == pytest.approx(alone, rel=1e-6), (
            f"the empty microbatch shifted the reported loss "
            f"({alone:.6f} -> {with_empty:.6f})"
        )

    def test_all_empty_reports_zero_tokens_and_does_not_crash(self):
        """Every microbatch empty is a caller error, not a case to paper over.

        Cross-entropy over zero valid targets is genuinely undefined, so
        the loss is NaN and `num_tokens` is 0. What matters is that the
        call returns those honestly instead of raising ZeroDivisionError
        or silently reporting a plausible number.
        """
        import math

        empty = torch.zeros((2, 9), dtype=torch.long)
        c, _, _ = _client(seed=3, dataset=self._DS())
        out = c.forward_backward([empty, empty]).result()
        assert out.num_tokens == 0
        assert math.isnan(out.loss), (
            "an all-empty batch reported a finite loss; that would hide "
            "a broken input pipeline"
        )

    def test_num_tokens_excludes_pads(self):
        from ezpz.tinker.step import prepare_batch

        c, _, _ = _client(seed=4, dataset=self._DS())
        pb = prepare_batch(c.state, self._padded([3], t=9))
        # 9 tokens -> 8 labels after the shift; 2 real (positions 1,2)
        assert pb.num_tokens == 2, (
            f"num_tokens={pb.num_tokens} counts pads; the client would "
            "weight microbatches by padding rather than content"
        )


class TestDatumContract:
    """The exported, documented Tinker input type must actually work."""

    def test_datum_is_accepted(self):
        from ezpz.tinker.types import Datum, ModelInput

        c, _, _ = _client(seed=5)
        d = Datum(
            model_input=ModelInput(token_ids=list(range(1, 10))),
            loss_fn_inputs={},
        )
        out = c.forward_backward([d]).result()
        assert out.num_tokens > 0

    def test_model_input_is_accepted(self):
        from ezpz.tinker.types import ModelInput

        c, _, _ = _client(seed=6)
        out = c.forward_backward(
            [ModelInput(token_ids=list(range(1, 10)))]
        ).result()
        assert out.num_tokens > 0

    def test_unsupported_type_names_what_is_allowed(self):
        c, _, _ = _client(seed=7)
        with pytest.raises(TypeError, match="ModelInput.*Datum|Datum"):
            c.forward_backward(["not a batch"]).result()


class TestAdapterCheckpointRoundTrip:
    """save_state defaults to adapters_only, so load_state must too."""

    def test_load_state_defaults_match_save_state(self):
        import inspect

        from ezpz.tinker.client import LocalTrainingClient

        save = inspect.signature(LocalTrainingClient.save_state)
        load = inspect.signature(LocalTrainingClient.load_state)
        assert "adapters_only" in load.parameters, (
            "load_state cannot restore what save_state writes by default"
        )
        assert (
            load.parameters["adapters_only"].default
            == save.parameters["adapters_only"].default
        ), "save/load adapters_only defaults disagree; round-trip breaks"

    def test_load_checkpoint_accepts_state_dict_options(self):
        import inspect

        from ezpz.examples._checkpoint import load_checkpoint

        assert "state_dict_options" in inspect.signature(
            load_checkpoint
        ).parameters
