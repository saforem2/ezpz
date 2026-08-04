"""Tests for the GQA path in `ezpz.models.llama.Attention`.

`Attention.forward` can either materialize the repeated kv heads with
`repeat_kv` or let SDPA broadcast them via `enable_gqa=True`. The two are
mathematically equivalent; these tests pin that equivalence (forward and
backward) and pin `enable_gqa` as the default.
"""

import pytest
import torch

from ezpz.models import llama
from ezpz.models.llama import Attention, ModelArgs, precompute_freqs_cis


def _build_attention(n_heads: int, n_kv_heads: int, head_dim: int = 16):
    torch.manual_seed(0)
    args = ModelArgs(
        dim=n_heads * head_dim,
        n_layers=1,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=32,
        multiple_of=8,
        batch_size=2,
        max_seq_len=32,
        depth_init=True,
    )
    return Attention(args), args


def _run_forward(attn, args, seq_len=16, batch_size=2):
    torch.manual_seed(1)
    x = torch.randn(batch_size, seq_len, args.dim, requires_grad=True)
    freqs_cis = precompute_freqs_cis(
        args.dim // args.n_heads, seq_len, theta=args.rope_theta
    )
    return x, attn(x, freqs_cis)


@pytest.mark.parametrize(("n_heads", "n_kv_heads"), [(4, 1), (8, 2), (4, 4)])
def test_gqa_matches_repeat_kv_forward(monkeypatch, n_heads, n_kv_heads):
    """Broadcasting the kv heads gives the same output as repeating them."""
    attn, args = _build_attention(n_heads, n_kv_heads)

    monkeypatch.setenv("EZPZ_SDPA_ENABLE_GQA", "1")
    _, out_gqa = _run_forward(attn, args)

    monkeypatch.setenv("EZPZ_SDPA_ENABLE_GQA", "0")
    _, out_repeat = _run_forward(attn, args)

    torch.testing.assert_close(out_gqa, out_repeat, rtol=1e-5, atol=1e-6)


def test_gqa_matches_repeat_kv_backward(monkeypatch):
    """Gradients agree between the two paths."""
    attn, args = _build_attention(n_heads=4, n_kv_heads=1)

    grads = {}
    for flag in ("1", "0"):
        monkeypatch.setenv("EZPZ_SDPA_ENABLE_GQA", flag)
        attn.zero_grad(set_to_none=True)
        x, out = _run_forward(attn, args)
        out.sum().backward()
        grads[flag] = (
            x.grad.clone(),
            {name: p.grad.clone() for name, p in attn.named_parameters()},
        )

    x_grad_gqa, param_grads_gqa = grads["1"]
    x_grad_repeat, param_grads_repeat = grads["0"]
    torch.testing.assert_close(x_grad_gqa, x_grad_repeat, rtol=1e-5, atol=1e-6)
    for name, grad in param_grads_gqa.items():
        torch.testing.assert_close(
            grad, param_grads_repeat[name], rtol=1e-5, atol=1e-6
        )


def test_gqa_is_the_default(monkeypatch):
    """With the env var unset, `repeat_kv` is not called."""
    attn, args = _build_attention(n_heads=4, n_kv_heads=1)
    monkeypatch.delenv("EZPZ_SDPA_ENABLE_GQA", raising=False)

    calls = []
    real_repeat_kv = llama.repeat_kv

    def _counting_repeat_kv(x, n_rep):
        calls.append(n_rep)
        return real_repeat_kv(x, n_rep)

    monkeypatch.setattr(llama, "repeat_kv", _counting_repeat_kv)
    _run_forward(attn, args)

    assert calls == []


def test_repeat_kv_used_when_gqa_disabled(monkeypatch):
    """`EZPZ_SDPA_ENABLE_GQA=0` restores the explicit `repeat_kv` path."""
    attn, args = _build_attention(n_heads=4, n_kv_heads=1)
    monkeypatch.setenv("EZPZ_SDPA_ENABLE_GQA", "0")

    calls = []
    real_repeat_kv = llama.repeat_kv

    def _counting_repeat_kv(x, n_rep):
        calls.append(n_rep)
        return real_repeat_kv(x, n_rep)

    monkeypatch.setattr(llama, "repeat_kv", _counting_repeat_kv)
    _run_forward(attn, args)

    # once for the keys, once for the values
    assert calls == [4, 4]
