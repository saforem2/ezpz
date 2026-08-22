"""LoRA on HuggingFace models.

Why this file exists: `--lora-rank 16 --model meta-llama/Llama-3.2-1B`
used to be accepted, logged, and then silently ignored -- a full
fine-tune that looked exactly like a LoRA run. Two separate causes:

1. The `apply_lora` call in ``fsdp_tp.train`` sat inside the native
   model's ``else:`` branch, so the HF path never reached it.
2. The target names were hardcoded to ``ezpz.models.llama``
   (``wq/wk/wv/wo``, ``w1/w2/w3``). HF Llama spells the same roles
   ``q_proj/k_proj/v_proj/o_proj`` and ``gate_proj/up_proj/down_proj``,
   with ``lm_head`` for the unembedding. Zero overlap, so even a
   reachable call would have matched nothing.

The models here are built from an in-memory ``LlamaConfig`` rather than
loaded from disk. A ``tiny-random-llama-2/`` directory exists in the
working tree but is **not tracked by git**, so a test depending on it
would pass locally and fail on a clean clone.

The gate throughout is that every trainable parameter receives a
gradient after a real forward/backward. "It ran without raising" is not
the bar: a LoRA branch that is detached from the graph runs perfectly
well and trains nothing, which is the exact failure being fixed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch.nn as nn  # noqa: E402

from ezpz.tinker.lora import (  # noqa: E402
    LoraConfig,
    UNEMBED_TARGETS,
    apply_lora,
    iter_lora_modules,
)

RANK = 4


def _hf_llama(*, tie: bool = False, vocab: int = 64):
    """A 2-layer HF Llama small enough to train in a unit test."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=tie,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    if tie:
        model.tie_weights()
    return model


def _fwd_bwd(model, vocab: int = 64):
    """Run one real step; return (n_trainable, n_with_grad)."""
    out = model(input_ids=torch.randint(0, vocab, (1, 8))).logits
    out.float().pow(2).mean().backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    return len(trainable), sum(p.grad is not None for p in trainable)


class TestHfAttnMlp:
    def test_wraps_hf_projections(self):
        model = apply_lora(_hf_llama(), LoraConfig(rank=RANK), verbose=False)
        leaves = {n.rsplit(".", 1)[-1] for n, _ in iter_lora_modules(model)}
        assert leaves == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }

    def test_every_adapter_receives_a_gradient(self):
        """The real gate: a detached adapter runs fine and learns nothing."""
        model = apply_lora(_hf_llama(), LoraConfig(rank=RANK), verbose=False)
        n_train, n_grad = _fwd_bwd(model)
        assert n_train > 0, "nothing trainable -- the test proves nothing"
        assert n_grad == n_train, (
            f"{n_train - n_grad} adapter params got no gradient; some "
            "adapter is detached from the graph"
        )

    def test_base_weights_are_frozen(self):
        model = apply_lora(_hf_llama(), LoraConfig(rank=RANK), verbose=False)
        for name, mod in iter_lora_modules(model):
            assert not mod.base.weight.requires_grad, (
                f"{name}: base weight is still trainable, so this is a "
                "full fine-tune wearing LoRA's clothes"
            )

    def test_is_actually_low_rank(self):
        """Trainable params must be a small fraction of the total."""
        model = apply_lora(_hf_llama(), LoraConfig(rank=RANK), verbose=False)
        train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert 0 < train < total / 2, f"{train}/{total} trainable"

    @pytest.mark.parametrize(
        "attn,mlp,expect",
        [
            (True, False, {"q_proj", "k_proj", "v_proj", "o_proj"}),
            (False, True, {"gate_proj", "up_proj", "down_proj"}),
        ],
        ids=["attn-only", "mlp-only"],
    )
    def test_target_selection_is_honored(self, attn, mlp, expect):
        model = apply_lora(
            _hf_llama(),
            LoraConfig(rank=RANK, train_attn=attn, train_mlp=mlp),
            verbose=False,
        )
        leaves = {n.rsplit(".", 1)[-1] for n, _ in iter_lora_modules(model)}
        assert leaves == expect


class TestHfUnembed:
    def test_wraps_lm_head_by_its_own_name(self):
        """The adapter must replace `lm_head`, not create `model.output`.

        Writing to a hardcoded `output` attribute leaves `lm_head` a
        plain Linear: the forward path never sees the adapter, so the
        run silently trains fewer parameters than requested.
        """
        model = apply_lora(
            _hf_llama(),
            LoraConfig(rank=RANK, train_unembed=True),
            verbose=False,
        )
        assert type(model.lm_head).__name__ == "LoRALinear"
        leaves = {n.rsplit(".", 1)[-1] for n, _ in iter_lora_modules(model)}
        assert "lm_head" in leaves
        # The native spelling must NOT have been invented on the side.
        assert not isinstance(getattr(model, "output", None), nn.Module)

    def test_unembed_adapter_trains(self):
        model = apply_lora(
            _hf_llama(),
            LoraConfig(rank=RANK, train_unembed=True),
            verbose=False,
        )
        n_train, n_grad = _fwd_bwd(model)
        assert n_grad == n_train

    def test_tied_embeddings_are_adapted_without_unfreezing_the_embedding(
        self,
    ):
        """Tied `lm_head` works, and must not thaw the shared weight.

        When `tie_word_embeddings=True`, `lm_head.weight` IS
        `embed_tokens.weight`. Wrapping is still correct -- the adapter
        is additive and leaves the shared tensor alone -- but only as
        long as the base stays frozen. If wrapping ever made the base
        trainable it would silently unfreeze the input embedding too,
        turning a "LoRA" run into a partial full fine-tune.
        """
        model = apply_lora(
            _hf_llama(tie=True),
            LoraConfig(rank=RANK, train_unembed=True),
            verbose=False,
        )
        leaves = {n.rsplit(".", 1)[-1] for n, _ in iter_lora_modules(model)}
        assert leaves & set(UNEMBED_TARGETS), "lm_head was not adapted"

        assert model.lm_head.base.weight is model.model.embed_tokens.weight, (
            "wrapping broke weight tying"
        )
        assert not model.model.embed_tokens.weight.requires_grad, (
            "the tied embedding is trainable, so this is no longer LoRA"
        )
        n_train, n_grad = _fwd_bwd(model)
        assert n_grad == n_train


class TestUnsupportedArchitectures:
    def test_fused_qkv_conv1d_raises_with_a_named_error(self):
        """GPT-2 style `Conv1D` is not an nn.Linear, so nothing matches.

        A name table cannot fix this, so the error must say so rather
        than leaving the user to wonder why LoRA "did nothing".
        """
        conv1d = pytest.importorskip("transformers.pytorch_utils").Conv1D
        assert not issubclass(conv1d, nn.Linear), (
            "Conv1D became an nn.Linear subclass upstream; this test's "
            "premise (and the docs' scope note) need revisiting"
        )

        class FusedAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.c_attn = conv1d(3 * 8, 8)

        model = nn.Module()
        model.h = nn.ModuleList([FusedAttn()])
        with pytest.raises(RuntimeError, match="matched no modules") as ei:
            apply_lora(model, LoraConfig(rank=RANK), verbose=False)
        assert "Fused-QKV" in str(ei.value), (
            "the error should name the actual limitation, not just "
            "report a miss"
        )
