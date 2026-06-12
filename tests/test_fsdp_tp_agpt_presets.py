"""Tests for the agpt-2b / agpt-20b presets in ezpz.examples.fsdp_tp.

These presets are reproduced verbatim from torchtitan's
``torchtitan/experiments/ezpz/agpt/__init__.py`` registry at commit
``0aa404543cd5707d21678f26d1d0dc6a13c9c750``. The point of having them
in ezpz is so a user can A/B an ezpz fsdp_tp run against a torchtitan
agpt run on the same architecture.

If these values drift away from torchtitan's spec, this test fires.
"""

from __future__ import annotations

import importlib

import pytest


# Expected per-spec architecture (verbatim from torchtitan).
_AGPT_EXPECTED = {
    "agpt-2b": {
        "dim": 2048,
        "n_layers": 12,
        "n_heads": 16,
        "n_kv_heads": 4,
        "vocab_size": 256128,
        "hidden_dim": 11008,
        "rope_theta": 50000.0,
        "seq_len": 8192,
    },
    "agpt-20b": {
        "dim": 5120,
        "n_layers": 64,
        "n_heads": 40,
        "n_kv_heads": 8,
        "vocab_size": 256128,
        "hidden_dim": 14336,
        "rope_theta": 500000.0,
    },
}

# Long-form alias spellings every preset must accept. Tests check that
# each alias resolves to the same preset values as the canonical key.
_AGPT_ALIASES = {
    "agpt-2b": ["agpt2b", "agpt_2b", "AGPT-2B"],
    "agpt-20b": ["agpt20b", "agpt_20b", "AGPT-20B"],
}


def _import_fsdp_tp():
    try:
        return importlib.import_module("ezpz.examples.fsdp_tp")
    except Exception as exc:
        pytest.skip(f"could not import ezpz.examples.fsdp_tp: {exc}")


@pytest.mark.parametrize("spec_name", list(_AGPT_EXPECTED))
def test_agpt_preset_matches_torchtitan(spec_name):
    """Every architectural field on the preset matches torchtitan exactly."""
    mod = _import_fsdp_tp()
    presets = mod.MODEL_PRESETS
    assert spec_name in presets, (
        f"MODEL_PRESETS missing {spec_name!r}; got {sorted(presets)}"
    )
    preset = presets[spec_name]
    expected = _AGPT_EXPECTED[spec_name]
    for key, expected_value in expected.items():
        assert preset.get(key) == expected_value, (
            f"MODEL_PRESETS[{spec_name!r}][{key!r}] = "
            f"{preset.get(key)!r}, expected {expected_value!r} "
            "(verbatim from torchtitan)"
        )


@pytest.mark.parametrize(
    "alias,canonical",
    [
        (alias, canonical)
        for canonical, aliases in _AGPT_ALIASES.items()
        for alias in aliases
    ],
)
def test_agpt_aliases_resolve(alias, canonical):
    """Each alias must map to the canonical preset key."""
    mod = _import_fsdp_tp()
    assert mod.MODEL_ALIASES.get(alias) == canonical, (
        f"MODEL_ALIASES[{alias!r}] = {mod.MODEL_ALIASES.get(alias)!r}, "
        f"expected {canonical!r}"
    )


@pytest.mark.parametrize(
    "spec_name", list(_AGPT_EXPECTED) + [
        a for aliases in _AGPT_ALIASES.values() for a in aliases
    ]
)
def test_agpt_parse_args_resolves_arch(spec_name):
    """End-to-end: ``--model agpt-2b`` (or any alias) → args.dim,
    args.n_layers, etc. all reflect the preset architecture."""
    mod = _import_fsdp_tp()
    args = mod.parse_args(["--model", spec_name])
    canonical = mod.MODEL_ALIASES.get(spec_name, spec_name)
    expected = _AGPT_EXPECTED[canonical]
    for key, expected_value in expected.items():
        assert getattr(args, key) == expected_value, (
            f"--model {spec_name}: args.{key} = "
            f"{getattr(args, key)!r}, expected {expected_value!r}"
        )


def test_agpt_explicit_cli_flag_overrides_preset():
    """Explicit ``--n-layers 4`` must win over the agpt preset's n_layers=12.

    Without this, the preset application would clobber explicit user input
    and silently surprise anyone trying to scale down a preset for a
    smoke-test run.
    """
    mod = _import_fsdp_tp()
    args = mod.parse_args(["--model", "agpt-2b", "--n-layers", "4"])
    assert args.n_layers == 4
    # Other preset fields still apply
    assert args.dim == 2048
    assert args.hidden_dim == 11008


@pytest.mark.parametrize(
    "argv,expected_seq_len,expected_batch_size",
    [
        # space-separated form (already worked before the fix)
        (
            ["--model", "agpt-2b", "--seq-len", "512", "--batch-size", "4"],
            512,
            4,
        ),
        # `=`-fused form. Pre-fix, _arg_provided('--seq-len' in argv)
        # was False because the actual token is "--seq-len=512", so
        # the preset's seq_len=8192 clobbered the user's 512.
        (
            ["--model", "agpt-2b", "--seq-len=512", "--batch-size=4"],
            512,
            4,
        ),
        # mixed forms — `--seq-len` space-separated, `--batch-size=` fused
        (
            ["--model", "agpt-2b", "--seq-len", "512", "--batch-size=4"],
            512,
            4,
        ),
    ],
    ids=["space-separated", "equals-fused", "mixed"],
)
def test_explicit_flag_wins_for_both_argv_shapes(
    argv, expected_seq_len, expected_batch_size
):
    """Regression: ``--seq-len=8192`` (with ``=``) was being silently
    overwritten by the preset's seq_len, because _arg_provided's
    ``flag in argv`` check fails for ``=``-fused tokens. After the fix,
    both spellings are honoured."""
    mod = _import_fsdp_tp()
    args = mod.parse_args(argv)
    assert args.seq_len == expected_seq_len, (
        f"argv {argv}: seq_len = {args.seq_len}, expected {expected_seq_len}"
    )
    assert args.batch_size == expected_batch_size, (
        f"argv {argv}: batch_size = {args.batch_size}, expected {expected_batch_size}"
    )


@pytest.mark.parametrize("theta", [10000.0, 50000.0, 500000.0])
def test_reshape_for_broadcast_fallback_uses_provided_theta(theta):
    """Regression: when the cached freqs buffer is too short for the
    incoming sequence, `reshape_for_broadcast` recomputes on the fly. That
    recomputation must use the same RoPE base (`theta`) as the original
    buffer — otherwise agpt-2b (50000), agpt-20b (500000), or Llama3
    (500000) silently fall back to the Llama1/Llama2 default of 10000,
    producing a different rotary basis.
    """
    torch = pytest.importorskip("torch")
    llama = pytest.importorskip("ezpz.models.llama")

    rotary_dim = 8  # complex pair count = 4
    short_len = 4
    long_len = 16  # > short_len so the fallback fires

    # Start with a "too short" buffer (built with the right theta).
    short_freqs = llama.precompute_freqs_cis(
        rotary_dim * 2, short_len, theta=theta
    )
    # Fake query tensor that needs `long_len` positions worth of freqs.
    fake_q = torch.zeros((1, long_len, 1, rotary_dim), dtype=torch.complex64)

    reshaped = llama.reshape_for_broadcast(short_freqs, fake_q, theta=theta)

    # Ground truth: what precompute_freqs_cis returns at `long_len` with
    # the SAME theta the caller passed.
    expected = llama.precompute_freqs_cis(
        rotary_dim * 2, long_len, theta=theta
    )
    # reshaped is broadcast-shaped (1, long_len, 1, rotary_dim/2 complex).
    # Flatten it back to (long_len, rotary_dim/2) for comparison.
    reshaped_flat = reshaped.reshape(long_len, -1)
    assert torch.allclose(reshaped_flat, expected, atol=1e-6), (
        f"Fallback freqs at theta={theta} do not match precompute_freqs_cis "
        f"with the same theta — the on-the-fly recomputation is using a "
        f"different RoPE base."
    )

    if theta != 10000.0:
        # And critically: the freqs at theta!=10000 must NOT match the
        # default-base freqs. This is the bug the fix is closing.
        default_base = llama.precompute_freqs_cis(
            rotary_dim * 2, long_len, theta=10000.0
        )
        assert not torch.allclose(reshaped_flat, default_base, atol=1e-3), (
            f"Fallback freqs at theta={theta} accidentally match the "
            f"theta=10000 freqs — the theta kwarg is being ignored."
        )
