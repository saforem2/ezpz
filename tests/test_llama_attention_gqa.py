"""Tests for the GQA path in `ezpz.models.llama.Attention`.

`Attention.forward` can either materialize the repeated kv heads with
`repeat_kv` or let SDPA broadcast them via `enable_gqa=True`. The two are
mathematically equivalent; these tests pin that equivalence (forward and
backward) and pin `enable_gqa` as the default.
"""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

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


def _backend_available(backend) -> bool:
    """True when this torch build can run SDPA under *backend* here."""
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 1, 8, 16)
    v = torch.randn(2, 1, 8, 16)
    try:
        with sdpa_kernel(backend):
            torch.nn.functional.scaled_dot_product_attention(
                q, k, v, enable_gqa=True
            )
    except Exception:
        return False
    return True


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
    """Gradients agree between the two paths.

    Pinned to the MATH backend, where the assertion is exact: the two
    paths are bit-identical there (measured max |grad diff| == 0.0).
    Without the pin this picked up whichever backend torch chose, and on
    a build that selects FLASH the fused kernel reassociates the
    backward reduction differently, producing ~7.6e-06 of drift against
    the ``atol=1e-6`` below -- a deterministic failure that looks like a
    correctness bug but is not one. See
    ``test_gqa_matches_repeat_kv_backward_flash`` for FLASH coverage at
    a tolerance appropriate to it.
    """
    attn, args = _build_attention(n_heads=4, n_kv_heads=1)

    grads = {}
    for flag in ("1", "0"):
        monkeypatch.setenv("EZPZ_SDPA_ENABLE_GQA", flag)
        attn.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.MATH):
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


def test_gqa_matches_repeat_kv_backward_flash(monkeypatch):
    """Same equivalence under the FLASH backend, at a realistic tolerance.

    The strict test above runs under MATH, where the two paths are
    bit-identical. FLASH fuses the attention and reassociates the
    backward reduction differently for `enable_gqa=True` vs a
    materialized `repeat_kv`, so its gradients differ by ~1e-5 relative
    — real floating-point reassociation, not a correctness gap.

    Measured on this config (torch 2.12.1, CPU fp32):

        MATH  : max |grad diff| = 0.0
        FLASH : max |grad diff| = 7.6e-06

    Keeping a FLASH case at a fp32-appropriate tolerance preserves
    coverage of the backend that actually runs in production, instead of
    dropping it or loosening the strict MATH assertion to hide the
    difference.
    """
    if not _backend_available(SDPBackend.FLASH_ATTENTION):
        pytest.skip("FLASH SDPA backend unavailable")

    attn, args = _build_attention(n_heads=4, n_kv_heads=1)

    grads = {}
    for flag in ("1", "0"):
        monkeypatch.setenv("EZPZ_SDPA_ENABLE_GQA", flag)
        attn.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x, out = _run_forward(attn, args)
            out.sum().backward()
        grads[flag] = (
            x.grad.clone(),
            {name: p.grad.clone() for name, p in attn.named_parameters()},
        )

    x_grad_gqa, param_grads_gqa = grads["1"]
    x_grad_repeat, param_grads_repeat = grads["0"]
    torch.testing.assert_close(x_grad_gqa, x_grad_repeat, rtol=2e-4, atol=1e-4)
    for name, grad in param_grads_gqa.items():
        torch.testing.assert_close(
            grad, param_grads_repeat[name], rtol=2e-4, atol=1e-4
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
