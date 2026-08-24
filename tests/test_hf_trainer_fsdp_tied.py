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
    _model_ties_embeddings,
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
    def test_detects_a_tied_model(self, tied_model_dir):
        argv = ["--model_name_or_path", tied_model_dir]
        assert _model_ties_embeddings(argv) is True

    def test_untied_model_is_not_flagged(self, untied_model_dir):
        argv = ["--model_name_or_path", untied_model_dir]
        assert _model_ties_embeddings(argv) is False

    def test_equals_form_is_parsed(self, tied_model_dir):
        argv = [f"--model_name_or_path={tied_model_dir}"]
        assert _model_ties_embeddings(argv) is True

    def test_no_model_arg_yields_nothing(self):
        assert _model_ties_embeddings(["--fsdp=auto_wrap"]) is False

    def test_unresolvable_model_does_not_raise(self):
        """A probe failure must never block a run."""
        argv = ["--model_name_or_path", "/nonexistent/model/path"]
        assert _model_ties_embeddings(argv) is False


class TestInjection:
    def test_injects_for_a_tied_model(self, tied_model_dir):
        argv = ["--model_name_or_path", tied_model_dir, "--fsdp=auto_wrap"]
        out = _maybe_inject_tied_wrap_policy(argv)
        assert out[-2] == "--fsdp_config"
        assert json.loads(out[-1]) == {"fsdp_version": 1}

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

    def test_explicit_fsdp_version_wins(self, tied_model_dir):
        argv = [
            "--model_name_or_path", tied_model_dir, "--fsdp=auto_wrap",
            "--fsdp_version", "2",
        ]
        assert _maybe_inject_tied_wrap_policy(argv) == argv


class TestTheInjectedConfig:
    """What gets injected, and why it is `fsdp_version: 1`.

    Verified on a Perlmutter compute node (torch 2.13.0, accelerate
    1.14, real Llama-3.2-1B, 2 ranks) by reproducing the failure and
    running each candidate against it:

        transformer_layer_cls_to_wrap   fails
        activation_checkpointing        fails
        use_orig_params                 fails
        fsdp_version: 1                 OK
        untie lm_head before wrapping   OK

    FSDP1 over untying because untying adds a `vocab_size x hidden`
    parameter tensor, changing what a benchmark measures.
    """

    def test_pins_fsdp_v1(self, tied_model_dir):
        argv = ["--model_name_or_path", tied_model_dir, "--fsdp=auto_wrap"]
        out = _maybe_inject_tied_wrap_policy(argv)
        assert out[-2] == "--fsdp_config"
        assert json.loads(out[-1]) == {"fsdp_version": 1}

    def test_accelerate_still_understands_fsdp_version(self):
        """Guard against the key being renamed under us.

        If this fails the injected config is being ignored and #237 is
        back -- silently, since nothing errors.

        Checks **accelerate**, not transformers: `fsdp_version` is a
        field on `FullyShardedDataParallelPlugin` and appears nowhere in
        `TrainingArguments.__post_init__` (verified on transformers
        4.50.1 and 5.14.1). The first version of this guard asserted
        against transformers and failed for that reason.
        """
        import inspect

        accelerate = pytest.importorskip("accelerate")
        from accelerate.utils import dataclasses as accel_dc

        src = inspect.getsource(accel_dc.FullyShardedDataParallelPlugin)
        assert "fsdp_version" in src, (
            f"accelerate {accelerate.__version__} no longer has an "
            "fsdp_version field; the injected config is being ignored"
        )
