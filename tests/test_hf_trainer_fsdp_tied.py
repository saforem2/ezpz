"""FSDP wrap policy for models with tied weights (#237).

Llama-3.2-1B ties `embed_tokens` to `lm_head`. FSDP2 refuses to put a
shared parameter in two wrap groups::

    ValueError: Parameter 'model.embed_tokens.weight' is shared with a
    parameter already managed by another FSDP group.

`--fsdp=auto_wrap` reaches HF as `fsdp=True` with NO policy
(`min_num_params: 0`), so FSDP wraps greedily and eventually splits the
pair. Observed on Perlmutter: all 8 ranks dead before step 1.

Wrapping at the decoder layer keeps the embedding and LM head outside
the wrapped units, so the tied pair stays in one group.
"""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")

from ezpz.examples.hf_trainer import (  # noqa: E402
    _apply_tied_weight_fsdp_policy,
    _decoder_layer_cls_names,
    _tied_weight_keys,
)

def _model(tie: bool):
    """A tiny Llama built in-process, with tying on or off.

    Constructed from `LlamaConfig` rather than loaded from a path or
    the Hub: the first version of this file read a `tiny-random-llama-2`
    directory that exists in one working tree and is not tracked, so CI
    fell through to HuggingFace and failed with 401 Unauthorized. Tests
    must not depend on an untracked fixture or on network access.

    Small enough (2 layers, dim 32) that construction is milliseconds,
    and it is a REAL LlamaForCausalLM, so `LlamaDecoderLayer` detection
    and the tied embed_tokens/lm_head pair are the genuine article.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=tie,
    )
    model = LlamaForCausalLM(cfg)
    if tie:
        model.tie_weights()
    return model


class _Args:
    def __init__(self, fsdp, fsdp_config):
        self.fsdp = fsdp
        self.fsdp_config = fsdp_config


@pytest.fixture(scope="module")
def tied_model():
    return _model(True)


@pytest.fixture(scope="module")
def untied_model():
    return _model(False)


class TestTiedWeightDetection:
    def test_finds_the_embed_lm_head_pair(self, tied_model):
        """The exact pair from the Perlmutter failure."""
        tied = _tied_weight_keys(tied_model)
        assert tied == ["model.embed_tokens.weight <-> lm_head.weight"]

    def test_untied_model_reports_nothing(self, untied_model):
        assert _tied_weight_keys(untied_model) == []

    def test_named_parameters_dedup_would_hide_the_tie(self, tied_model):
        """Pins `remove_duplicate=False`, which is load-bearing.

        `named_parameters()` DEDUPLICATES shared tensors by default, so
        a tied pair appears once and detection returns [] — silently
        disabling the whole fix. The first version of this helper had
        exactly that bug.
        """
        deduped = list(tied_model.named_parameters())
        full = list(tied_model.named_parameters(remove_duplicate=False))
        assert len(full) > len(deduped), (
            "fixture must actually tie a parameter, or this proves nothing"
        )


class TestDecoderLayerDetection:
    def test_reads_the_class_off_the_model(self, tied_model):
        """Discovered from the module tree, not an allowlist, so other
        architectures work without a code change."""
        assert _decoder_layer_cls_names(tied_model) == ["LlamaDecoderLayer"]


class TestPolicyApplication:
    def test_sets_the_wrap_policy_for_a_tied_model(self, tied_model):
        args = _Args(True, {"min_num_params": 0})
        _apply_tied_weight_fsdp_policy(tied_model, args)
        assert args.fsdp_config["transformer_layer_cls_to_wrap"] == [
            "LlamaDecoderLayer"
        ]

    def test_untied_model_is_left_alone(self, untied_model):
        args = _Args(True, {"min_num_params": 0})
        _apply_tied_weight_fsdp_policy(untied_model, args)
        assert "transformer_layer_cls_to_wrap" not in args.fsdp_config

    def test_no_fsdp_means_no_change(self, tied_model):
        args = _Args(False, {"min_num_params": 0})
        _apply_tied_weight_fsdp_policy(tied_model, args)
        assert "transformer_layer_cls_to_wrap" not in args.fsdp_config

    def test_an_explicit_policy_wins(self, tied_model):
        """Someone who set this has a reason; do not second-guess it."""
        args = _Args(True, {"transformer_layer_cls_to_wrap": ["MyBlock"]})
        _apply_tied_weight_fsdp_policy(tied_model, args)
        assert args.fsdp_config["transformer_layer_cls_to_wrap"] == ["MyBlock"]

    def test_an_explicit_min_num_params_wins(self, tied_model):
        """A size-based policy is also a deliberate choice."""
        args = _Args(True, {"min_num_params": 1000})
        _apply_tied_weight_fsdp_policy(tied_model, args)
        assert "transformer_layer_cls_to_wrap" not in args.fsdp_config

    def test_missing_fsdp_config_does_not_crash(self, tied_model):
        args = _Args(True, None)
        _apply_tied_weight_fsdp_policy(tied_model, args)  # no raise
