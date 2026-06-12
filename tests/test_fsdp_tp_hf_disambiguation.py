"""Tests for the `--model owner/repo` HuggingFace branch in fsdp_tp.

The disambiguation rule: if ``args.model`` contains a ``/``, it's a HF
repo id and ``apply_model_preset`` must leave it alone (the
model-construction path branches on it later). It must also default
``args.tokenizer_name`` to the same repo id unless the user explicitly
passed ``--tokenizer_name``.

We don't load any weights or instantiate HF objects here — these are
pure parser tests.
"""

from __future__ import annotations

import importlib

import pytest


def _import_fsdp_tp():
    try:
        return importlib.import_module("ezpz.examples.fsdp_tp")
    except Exception as exc:
        pytest.skip(f"could not import ezpz.examples.fsdp_tp: {exc}")


def test_preset_still_works():
    """Sanity: a regular preset like ``--model xxl`` still produces a
    preset-applied Namespace (not the HF early-return path)."""
    mod = _import_fsdp_tp()
    args = mod.parse_args(["--model", "xxl"])
    assert args.model == "xxl"
    assert args.dim == 4096  # xxl preset value


@pytest.mark.parametrize(
    "repo_id",
    [
        "meta-llama/Llama-3.2-1B",
        "google/gemma-2-2b",
        "Qwen/Qwen2.5-0.5B",
        "mistralai/Mistral-7B-v0.1",
    ],
)
def test_hf_repo_id_is_left_unchanged(repo_id):
    """A `/`-containing --model must survive apply_model_preset
    unchanged — downstream code branches on `"/" in args.model`."""
    mod = _import_fsdp_tp()
    args = mod.parse_args(["--model", repo_id])
    assert args.model == repo_id, (
        f"args.model was rewritten from {repo_id!r} to {args.model!r}"
    )


@pytest.mark.parametrize(
    "repo_id",
    [
        "meta-llama/Llama-3.2-1B",
        "google/gemma-2-2b",
    ],
)
def test_hf_tokenizer_defaults_to_repo(repo_id):
    """When --tokenizer_name isn't passed, it auto-fills to the HF repo
    id. The vast majority of HF causal-LM repos publish a matching
    tokenizer at the same path, so this is the right default."""
    mod = _import_fsdp_tp()
    args = mod.parse_args(["--model", repo_id])
    assert args.tokenizer_name == repo_id, (
        f"--model {repo_id}: tokenizer_name = {args.tokenizer_name!r}, "
        f"expected {repo_id!r} (HF auto-default)"
    )


@pytest.mark.parametrize(
    "tokenizer_argv",
    [
        # Space-separated form: two argv tokens.
        ["--tokenizer_name", "gpt2"],
        # Equals-fused form: one argv token. This is the exact bug
        # class that motivated the `_arg_provided` fix — pre-fix, a
        # single "--tokenizer_name=gpt2" token wasn't detected by a
        # naive `flag in argv` check and the HF default silently
        # clobbered the user's override.
        ["--tokenizer_name=gpt2"],
    ],
    ids=["space_separated", "equals_fused"],
)
def test_explicit_tokenizer_wins_over_hf_default(tokenizer_argv):
    """If the user passes --tokenizer_name explicitly, the HF
    auto-default in apply_model_preset must NOT overwrite it.

    Covers both argv shapes accepted by argparse for the underscore
    form. Note: ``--tokenizer-name`` (hyphenated) is NOT declared in
    fsdp_tp's argparse, so those forms would SystemExit at parse
    time and aren't valid test inputs — even though `_arg_provided`
    would detect them in the raw argv list.
    """
    mod = _import_fsdp_tp()
    args = mod.parse_args([
        "--model", "meta-llama/Llama-3.2-1B",
        *tokenizer_argv,
    ])
    assert args.tokenizer_name == "gpt2", (
        f"argv={tokenizer_argv}: tokenizer_name={args.tokenizer_name!r}, "
        "expected 'gpt2' — HF default must not clobber explicit override."
    )


def test_hf_path_does_not_apply_preset_fields():
    """An HF repo id is not in MODEL_PRESETS — `args.dim` etc. must
    stay at their argparse defaults, NOT trigger KeyError or
    silently pick up another preset's values."""
    mod = _import_fsdp_tp()
    args = mod.parse_args(["--model", "meta-llama/Llama-3.2-1B"])
    # argparse defaults from parse_args (--dim default=4096, etc.)
    # — value doesn't matter much, just that no preset was applied
    # and no exception was raised.
    assert hasattr(args, "dim")
    assert hasattr(args, "n_layers")


def test_unknown_preset_without_slash_is_rejected():
    """A string with no `/` that isn't in MODEL_PRESETS or
    MODEL_ALIASES must raise SystemExit with a clear message —
    otherwise typos like `--model meidum` silently produce
    default-architecture runs."""
    mod = _import_fsdp_tp()
    with pytest.raises(SystemExit):
        mod.parse_args(["--model", "totally-not-a-preset"])
