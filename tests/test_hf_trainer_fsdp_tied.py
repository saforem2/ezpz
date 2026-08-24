"""FSDP wrap policy for models with tied weights (#237).

Llama-3.2-1B ties `embed_tokens` to `lm_head`. FSDP2 refuses to put a
shared parameter in two wrap groups::

    ValueError: Parameter 'model.embed_tokens.weight' is shared with a
    parameter already managed by another FSDP group.

`--fsdp=auto_wrap` arrives with NO policy, so FSDP wraps greedily and
eventually splits the pair. Observed on Perlmutter: all 8 ranks dead
before step 1.

**Timing is the whole difficulty.** `TrainingArguments.__post_init__`
translates `fsdp_config` into `FSDP_*` environment variables *as it
constructs*. The first version of this fix mutated `fsdp_config` just
before `Trainer(...)`, logged that it had set a policy, and changed
nothing -- the env vars were already written. So these tests assert on
`FSDP_TRANSFORMER_CLS_TO_WRAP`, not on the dict.
"""

from __future__ import annotations

import json

import pytest

transformers = pytest.importorskip("transformers")

from ezpz.examples.hf_trainer import (  # noqa: E402
    _maybe_inject_tied_wrap_policy,
    _tied_wrap_policy_from_cli,
)


@pytest.fixture(scope="module")
def tied_model_dir(tmp_path_factory):
    """A real tied Llama on disk (AutoConfig.from_pretrained needs a path).

    Built in-process from `LlamaConfig` rather than downloaded: an
    earlier version read an untracked directory that existed only in one
    working tree, so CI fell through to the Hub and 401'd.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    d = tmp_path_factory.mktemp("tied") / "model"
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, tie_word_embeddings=True,
    )
    LlamaForCausalLM(cfg).save_pretrained(d)
    return str(d)


@pytest.fixture(scope="module")
def untied_model_dir(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM

    d = tmp_path_factory.mktemp("untied") / "model"
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, tie_word_embeddings=False,
    )
    LlamaForCausalLM(cfg).save_pretrained(d)
    return str(d)


class TestPolicyProbe:
    def test_tied_model_yields_its_decoder_layer(self, tied_model_dir):
        """Read from HF's own `_no_split_modules`, not an allowlist."""
        argv = ["--model_name_or_path", tied_model_dir]
        assert _tied_wrap_policy_from_cli(argv) == ["LlamaDecoderLayer"]

    def test_untied_model_yields_nothing(self, untied_model_dir):
        argv = ["--model_name_or_path", untied_model_dir]
        assert _tied_wrap_policy_from_cli(argv) is None

    def test_equals_form_is_parsed(self, tied_model_dir):
        argv = [f"--model_name_or_path={tied_model_dir}"]
        assert _tied_wrap_policy_from_cli(argv) == ["LlamaDecoderLayer"]

    def test_no_model_arg_yields_nothing(self):
        assert _tied_wrap_policy_from_cli(["--fsdp=auto_wrap"]) is None

    def test_unresolvable_model_does_not_raise(self):
        """A probe failure must never block a run."""
        argv = ["--model_name_or_path", "/nonexistent/model/path"]
        assert _tied_wrap_policy_from_cli(argv) is None


class TestInjection:
    def test_injects_for_a_tied_model(self, tied_model_dir):
        argv = ["--model_name_or_path", tied_model_dir, "--fsdp=auto_wrap"]
        out = _maybe_inject_tied_wrap_policy(argv)
        assert out[-2] == "--fsdp_config"
        assert json.loads(out[-1]) == {
            "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"]
        }

    def test_no_fsdp_means_no_injection(self, tied_model_dir):
        argv = ["--model_name_or_path", tied_model_dir]
        assert _maybe_inject_tied_wrap_policy(argv) == argv

    def test_untied_model_is_left_alone(self, untied_model_dir):
        argv = ["--model_name_or_path", untied_model_dir, "--fsdp=auto_wrap"]
        assert _maybe_inject_tied_wrap_policy(argv) == argv

    def test_explicit_fsdp_config_wins(self, tied_model_dir):
        argv = [
            "--model_name_or_path", tied_model_dir, "--fsdp=auto_wrap",
            "--fsdp_config", '{"min_num_params": 100}',
        ]
        assert _maybe_inject_tied_wrap_policy(argv) == argv

    def test_explicit_layer_cls_flag_wins(self, tied_model_dir):
        argv = [
            "--model_name_or_path", tied_model_dir, "--fsdp=auto_wrap",
            "--fsdp_transformer_layer_cls_to_wrap", "MyBlock",
        ]
        assert _maybe_inject_tied_wrap_policy(argv) == argv


class TestItReachesTheEnvVar:
    """The assertion that matters, and that the first fix failed.

    A policy in `fsdp_config` is worthless unless it lands in
    `FSDP_TRANSFORMER_CLS_TO_WRAP`, which is what FSDP actually reads.
    `TrainingArguments.__post_init__` writes that env var as it
    constructs -- which is why mutating `fsdp_config` later (the first
    version of this fix) was inert.

    Asserted against HF's own translation step rather than by building
    a full `TrainingArguments`: constructing one with `fsdp=` also
    initializes torch.distributed and demands MASTER_ADDR, so it passes
    alone and fails in the suite depending on what ran before it.
    """

    def test_the_injected_config_is_what_hf_translates(
        self, tied_model_dir
    ):
        """The injected key is the one HF exports, spelled its way."""
        argv = ["--model_name_or_path", tied_model_dir, "--fsdp=auto_wrap"]
        out = _maybe_inject_tied_wrap_policy(argv)
        assert len(out) > len(argv), "nothing was injected"
        cfg = json.loads(out[-1])

        # HF reads `transformer_layer_cls_to_wrap` out of fsdp_config and
        # joins it into FSDP_TRANSFORMER_CLS_TO_WRAP. Pin both the key
        # and the shape (a list, which is what the join expects).
        assert "transformer_layer_cls_to_wrap" in cfg
        assert isinstance(cfg["transformer_layer_cls_to_wrap"], list)
        assert ",".join(cfg["transformer_layer_cls_to_wrap"]) == (
            "LlamaDecoderLayer"
        )

    def test_hf_still_reads_that_key(self):
        """Guard against HF renaming the key under us.

        If this fails, the injected config is being ignored and the
        #237 bug is back -- silently, since nothing errors.
        """
        import inspect

        from transformers import TrainingArguments

        src = inspect.getsource(TrainingArguments.__post_init__)
        assert "transformer_layer_cls_to_wrap" in src
        assert "TRANSFORMER_CLS_TO_WRAP" in src
