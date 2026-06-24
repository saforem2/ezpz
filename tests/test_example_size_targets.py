"""Parameter-count target test for the unified s/m/l/xl/xxl/xxxl ladder.

The ladder targets across all 5 ezpz.examples modules:
  s    ~  100M
  m    ~  250M
  l    ~  500M
  xl   ~ 1.0B
  xxl  ~ 5.0B
  xxxl ~10.0B

These targets are intentionally loose: architectural quantization
(integer channel counts in CNNs, head-dim/depth steps in transformers,
fixed input dims in MNIST-MLPs) makes hitting them exactly impossible.
We use a ±40 pct band so contributors get a green test as long as the
new preset is in the right ballpark.

If a preset drifts beyond the band, this fires. The failure message
shows the actual count, the target, and the configured preset values so
the contributor can fix the preset (or, if the new value is intentional,
update the target here).

Computed analytically (no real model construction) so the test runs in
<1s even for the xxxl tier.
"""

from __future__ import annotations

import importlib

import pytest


# Target parameter count per ladder rung. The same target applies to all
# 5 examples — that's the whole point of the unified ladder.
TARGETS = {
    "s":     int(100e6),    # 100M
    "m":     int(250e6),    # 250M
    "l":     int(500e6),    # 500M
    "xl":    int(1.0e9),    # 1.0B
    "xxl":   int(5.0e9),    # 5.0B
    "xxxl":  int(10.0e9),   # 10.0B
}

# Tolerance for the parameter-count match. Wide because architectural
# quantization makes exact hits impossible — CNNs grow in O(2x) channel
# steps, transformers in O(2x) head_dim steps, etc.
TOLERANCE_PCT = 0.40   # ±40%


# ===================================================================
# Analytic param-count functions, one per example architecture.
# ===================================================================


def _params_test(layer_sizes, input_dim=784, output_dim=10, **_):
    """SequentialLinearNet (MLP, MNIST input/output dims)."""
    sizes = [input_dim] + list(layer_sizes) + [output_dim]
    return sum(
        sizes[i] * sizes[i + 1] + sizes[i + 1]
        for i in range(len(sizes) - 1)
    )


def _params_fsdp(
    conv1_channels, conv2_channels, fc_dim,
    img_size=28, num_classes=10, **_,
):
    """Simple 2-layer CNN (MNIST)."""
    pooled = (img_size - 4) // 2   # two 3x3 valid convs + 2x2 maxpool
    return (
        1 * conv1_channels * 9 + conv1_channels
        + conv1_channels * conv2_channels * 9 + conv2_channels
        + conv2_channels * pooled * pooled * fc_dim + fc_dim
        + fc_dim * num_classes + num_classes
    )


def _params_vit(
    num_heads, head_dim, depth,
    img_size=224, patch_size=16, in_chans=3, num_classes=1000, **_,
):
    """Standard pre-norm ViT (224x224 RGB, 1000 cls)."""
    e = num_heads * head_dim
    patch_embed = in_chans * patch_size ** 2 * e + e
    num_patches = (img_size // patch_size) ** 2
    cls = e
    pos = (num_patches + 1) * e
    # Per AttentionBlock: 2 norms + Attn(QKV+proj) + MLP(4x hidden)
    block = 4 * e + 4 * e * e + 4 * e + 8 * e * e + 5 * e
    blocks = depth * block
    final = 2 * e + e * num_classes + num_classes
    return patch_embed + cls + pos + blocks + final


def _params_diffusion(
    hidden, n_layers, n_heads, seq_len, timesteps,
    vocab_size=10000, **_,
):
    """Toy DiT-style transformer for text diffusion."""
    tok = vocab_size * hidden
    pos = seq_len * hidden
    time_emb = timesteps * hidden
    block = 12 * hidden * hidden + 13 * hidden
    head = hidden * vocab_size + vocab_size
    return tok + pos + time_emb + n_layers * block + head


def _params_fsdp_tp(
    dim, n_layers, n_heads, n_kv_heads,
    vocab_size=32000, hidden_dim=None, multiple_of=256, **_,
):
    """Llama-arch transformer (vocab=32k default from parser)."""
    head_dim = dim // n_heads
    q = n_heads * head_dim
    kv = n_kv_heads * head_dim
    attn = dim * q + 2 * dim * kv + q * dim
    if hidden_dim is not None:
        pre = (hidden_dim * 3 + 1) // 2
        ffn_h = int(2 * pre / 3)
    else:
        ffn_h = int(2 * 4 * dim / 3)
    ffn_h = multiple_of * ((ffn_h + multiple_of - 1) // multiple_of)
    ffn = 3 * dim * ffn_h
    block = attn + ffn + 2 * dim
    return vocab_size * dim + n_layers * block + dim + vocab_size * dim


# Map module → analytic param-count function
_PARAM_FUNCS = {
    "ezpz.examples.test": _params_test,
    "ezpz.examples.fsdp": _params_fsdp,
    "ezpz.examples.vit": _params_vit,
    "ezpz.examples.diffusion": _params_diffusion,
    "ezpz.examples.fsdp_tp": _params_fsdp_tp,
}


def _load_presets(module_name: str):
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"could not import {module_name}: {exc}")
    return mod.MODEL_PRESETS


# ===================================================================
# Parametrized test: 5 modules × 6 ladder rungs = 30 tests
# ===================================================================


@pytest.mark.parametrize(
    "module_name", list(_PARAM_FUNCS),
    ids=lambda v: v.rsplit(".", 1)[-1],
)
@pytest.mark.parametrize(
    "rung,target",
    list(TARGETS.items()),
    ids=[
        f"{r}={t // 10**6}M" if t < 10**9 else f"{r}={t // 10**9}B"
        for r, t in TARGETS.items()
    ],
)
def test_preset_hits_target_param_count(module_name, rung, target):
    """Each (module, rung) pair produces a param count within ±40 pct
    of the unified ladder target.

    Wide tolerance because architectural quantization makes exact hits
    impossible. If a preset is intentionally outside the band — e.g.
    test.py xl maxes out at ~860M because the MLP input layer
    (784→24576) caps growth — bump the tolerance OR update the target
    here.
    """
    presets = _load_presets(module_name)
    if rung not in presets:
        pytest.fail(
            f"{module_name}.MODEL_PRESETS missing rung '{rung}'. "
            f"Got: {sorted(presets.keys())}"
        )
    preset = presets[rung]
    param_fn = _PARAM_FUNCS[module_name]
    actual = param_fn(**preset)

    low = target * (1 - TOLERANCE_PCT)
    high = target * (1 + TOLERANCE_PCT)
    assert low <= actual <= high, (
        f"{module_name} preset '{rung}': got {actual / 1e6:.0f}M "
        f"params, expected ~{target / 1e6:.0f}M "
        f"(±{TOLERANCE_PCT * 100:.0f} pct band: "
        f"{low / 1e6:.0f}M–{high / 1e6:.0f}M). "
        f"Preset values: {preset}"
    )
