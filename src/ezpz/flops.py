"""FLOPS estimation and Model FLOPS Utilization (MFU) calculation.

Provides utilities for measuring how efficiently a model uses the
available hardware compute:

- :func:`get_peak_flops` — peak BF16 FLOPS for common accelerators
- :func:`estimate_model_flops` — count FLOPS for one forward+backward pass
- :func:`compute_mfu` — calculate MFU% from model FLOPS and step timing

Usage::

    import ezpz
    from ezpz.flops import estimate_model_flops, compute_mfu

    rank = ezpz.setup_torch()
    model = MyModel().to(ezpz.get_torch_device())

    # Count FLOPS once before training
    model_flops = estimate_model_flops(model, input_shape=(batch_size, seq_len))

    for step in range(num_steps):
        t0 = time.perf_counter()
        loss = train_step(model, batch)
        ezpz.synchronize()
        dt = time.perf_counter() - t0

        mfu = compute_mfu(model_flops, dt)
        history.update({"loss": loss, "mfu": mfu}, step=step)
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "compute_mfu",
    "estimate_model_flops",
    "get_device_name",
    "get_peak_flops",
]


# ── Peak FLOPS database ─────────────────────────────────────────────────────
#
# BF16 peak FLOPS (without sparsity) for common accelerators.
# Sources linked inline.

# Order matters: more specific matches first (e.g. "H100 NVL" before "H100").
_PEAK_FLOPS: list[tuple[str, float | None]] = [
    # NVIDIA — specific variants first
    ("H100 NVL", 835e12),      # https://www.nvidia.com/en-us/data-center/h100/
    ("H100 PCIE", 756e12),
    ("H100", 989e12),          # SXM (default H100)
    ("H200", 989e12),          # https://www.nvidia.com/en-us/data-center/h200/
    ("B200", 2.25e15),         # https://nvdam.widen.net/s/wwnsxrhm2w
    ("A100", 312e12),          # https://www.nvidia.com/en-us/data-center/a100/
    ("L40S", 362e12),          # https://resources.nvidia.com/en-us-l40s
    # AMD
    ("MI355X", 2500e12),       # https://www.amd.com/.../mi355x
    ("MI325X", 1300e12),       # https://www.amd.com/.../mi325x
    ("MI300X", 1300e12),       # https://www.amd.com/.../mi300x
    ("MI250X", 191.5e12),      # per GCD, https://www.amd.com/.../mi250x
    # Intel
    ("1550", None),            # PVC — computed dynamically
    ("MAX", None),             # "Data Center GPU Max" variants
]


def get_device_name() -> str:
    """Return a human-readable name for the current accelerator.

    Tries ``torch.cuda.get_device_name()`` for NVIDIA/AMD, falls back
    to ``torch.xpu.get_device_properties()`` for Intel, and finally
    ``"cpu"`` if nothing is available.
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        props = torch.xpu.get_device_properties(0)
        return getattr(props, "name", str(props))
    return "cpu"


def get_peak_flops(
    device_name: str | None = None,
) -> float:
    """Return peak BF16 FLOPS for the given device.

    Args:
        device_name: GPU name string (e.g. ``"NVIDIA A100-SXM4-80GB"``).
            If ``None``, auto-detected from the current device.

    Returns:
        Peak FLOPS as a float. Falls back to A100 (312 TFLOPS) if the
        device is not recognized.
    """
    if device_name is None:
        device_name = get_device_name()

    name = device_name.upper()

    # Check known devices (order matters — specific matches first)
    for key, flops in _PEAK_FLOPS:
        if key.upper() in name:
            if flops is not None:
                return flops
            # Dynamic computation for Intel PVC
            return _compute_pvc_peak_flops()

    # Fallback
    logger.warning(
        "Peak FLOPS unknown for %r, falling back to A100 (312 TFLOPS)",
        device_name,
    )
    return 312e12


def _compute_pvc_peak_flops() -> float:
    """Compute peak BF16 FLOPS for Intel Data Center GPU Max 1550 (PVC).

    Uses the actual max_compute_units from the device to handle both
    full-EU (512 CUs = 340.8 TFLOPS) and standard-EU (448 CUs = 298.2
    TFLOPS) modes.

    Formula: 512 ops/cycle × max_compute_units × 1300 MHz
    """
    try:
        max_cu = torch.xpu.get_device_properties(0).max_compute_units
        return 512 * max_cu * 1300 * 10**6
    except Exception:
        # Fallback: assume full-EU mode (512 CUs)
        return 512 * 512 * 1300 * 10**6


# ── Model FLOPS estimation ──────────────────────────────────────────────────


def estimate_model_flops(
    model: torch.nn.Module,
    input_shape: tuple[int, ...] | list[int],
    *,
    device: torch.device | str | None = None,
    backward: bool = True,
) -> int:
    """Count FLOPS for one forward (+ optional backward) pass.

    Uses PyTorch's built-in ``FlopCounterMode`` to count actual
    floating-point operations, not parameter-based estimates.

    Args:
        model: The model to profile.
        input_shape: Shape of the input tensor (e.g. ``(batch, seq_len)``
            or ``(batch, channels, height, width)``).
        device: Device for the dummy input. Auto-detected if ``None``.
        backward: If ``True``, also count the backward pass FLOPS
            (typically ~2× forward for most architectures).

    Returns:
        Total FLOPS (forward + backward if requested).
    """
    from torch.utils.flop_counter import FlopCounterMode

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

    # Detect if model expects integer inputs (has an embedding layer)
    # vs float inputs (CNN, MLP, etc.)
    has_embedding = any(
        isinstance(m, (torch.nn.Embedding, torch.nn.EmbeddingBag))
        for m in model.modules()
    )

    if has_embedding:
        # Language model: expects Long token IDs
        # Find vocab size from the embedding layer
        vocab_size = 32000  # fallback
        for m in model.modules():
            if isinstance(m, torch.nn.Embedding):
                vocab_size = m.num_embeddings
                break
        dummy = torch.randint(0, vocab_size, input_shape, device=device)
    else:
        # Vision/MLP model: expects float tensors matching model dtype
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            dtype = torch.float32
        dummy = torch.randn(*input_shape, device=device, dtype=dtype)
    was_training = model.training
    model.eval()

    with FlopCounterMode(display=False) as counter:
        output = model(dummy)
        if backward:
            loss = output.sum()
            loss.backward()

    # Restore original mode and clean up accumulated grads
    if was_training:
        model.train()
    if backward:
        model.zero_grad()

    return counter.get_total_flops()


# ── MFU calculation ──────────────────────────────────────────────────────────


def compute_mfu(
    model_flops: int | float,
    step_duration: float,
    *,
    world_size: int | None = None,
    device_name: str | None = None,
    peak_flops: float | None = None,
) -> float:
    """Calculate Model FLOPS Utilization (MFU) as a percentage.

    MFU measures what fraction of the hardware's theoretical peak
    compute is used by the model's actual operations::

        MFU = model_flops / (peak_flops_per_device × world_size × step_duration)

    Args:
        model_flops: FLOPS per forward+backward pass (from
            :func:`estimate_model_flops`).
        step_duration: Wall-clock time for one training step (seconds).
        world_size: Number of devices. Auto-detected if ``None``.
        device_name: Device name for peak FLOPS lookup. Auto-detected
            if ``None``.
        peak_flops: Override the peak FLOPS value directly (skips
            device lookup).

    Returns:
        MFU as a percentage (0–100). Returns 0.0 if inputs are invalid.
    """
    if step_duration <= 0 or model_flops <= 0:
        return 0.0

    if world_size is None:
        try:
            from ezpz.distributed import get_world_size
            world_size = get_world_size()
        except Exception:
            world_size = 1

    if peak_flops is None:
        peak_flops = get_peak_flops(device_name)

    achieved_flops = model_flops / step_duration
    theoretical_peak = peak_flops * world_size

    if theoretical_peak <= 0:
        return 0.0

    return (achieved_flops / theoretical_peak) * 100.0
