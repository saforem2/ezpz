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
import math
import warnings

import torch
from torch.utils.flop_counter import FlopCounterMode

logger = logging.getLogger(__name__)

__all__ = [
    "compute_mfu",
    "estimate_model_flops",
    "get_device_name",
    "get_peak_flops",
    "try_estimate",
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
    ("L40S", 183e12),          # BF16 dense; 366 TFLOPS w/ sparsity
    # AMD
    ("MI355X", 5000e12),       # 5.0 PF BF16 dense per AMD MI355X brochure
    ("MI325X", 1300e12),       # https://www.amd.com/.../mi325x
    ("MI300X", 1300e12),       # https://www.amd.com/.../mi300x
    ("MI250X", 191.5e12),      # per GCD, https://www.amd.com/.../mi250x
    # Intel
    ("1550", None),            # PVC — computed dynamically
    ("MAX", None),             # "Data Center GPU Max" variants
]


# Set of device names we've already warned about, so we don't flood
# logs with the same message every time get_peak_flops is called for
# an unrecognized accelerator.
_WARNED_UNKNOWN_DEVICES: set[str] = set()


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
) -> float | None:
    """Return peak BF16 FLOPS for the given device.

    Args:
        device_name: GPU name string (e.g. ``"NVIDIA A100-SXM4-80GB"``).
            If ``None``, auto-detected from the current device.

    Returns:
        Peak FLOPS as a float, or ``None`` if the device is not
        recognized (e.g. CPU).
    """
    if device_name is None:
        device_name = get_device_name()

    name = device_name.upper()

    # CPU — no meaningful peak FLOPS
    if name == "CPU":
        return None

    # Check known devices (order matters — specific matches first)
    for key, flops in _PEAK_FLOPS:
        if key.upper() in name:
            if flops is not None:
                return flops
            # Dynamic computation for Intel PVC
            return _compute_pvc_peak_flops()

    # Unknown accelerator — warn once per device and return None
    if device_name not in _WARNED_UNKNOWN_DEVICES:
        _WARNED_UNKNOWN_DEVICES.add(device_name)
        warnings.warn(
            f"Peak FLOPS unknown for {device_name!r} — MFU tracking disabled",
            stacklevel=2,
        )
    return None


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
        # Fallback: assume full-EU mode (512 CUs = 340.8 TFLOPS)
        logger.info("Cannot read XPU compute units — assuming 512 CUs (full-EU)")
        return 512 * 512 * 1300 * 10**6


# ── Model FLOPS estimation ──────────────────────────────────────────────────


def _extract_loss(output: object) -> torch.Tensor | None:
    """Extract a scalar loss tensor from an arbitrary model output.

    Handles HF dataclass outputs (loss/logits attrs), plain tensors,
    tuples, lists, and dicts.  Returns ``None`` if no usable tensor
    can be found — in which case the caller should skip backward.
    """
    if isinstance(output, torch.Tensor):
        return output.sum()
    loss = getattr(output, "loss", None)
    if isinstance(loss, torch.Tensor):
        return loss
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits.sum()
    if isinstance(output, dict):
        for key in ("loss", "logits"):
            v = output.get(key)
            if isinstance(v, torch.Tensor):
                return v.sum() if key == "logits" else v
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v.sum()
        return None
    if isinstance(output, (list, tuple)):
        for v in output:
            if isinstance(v, torch.Tensor):
                return v.sum()
        return None
    return None


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

    # Stay in training mode so dropout/BN paths are counted realistically.
    # FlopCounterMode is a measurement, not a forward pass we want to
    # ship — flipping to eval() would under-count BN-heavy models.
    flops = 0
    backward_ran = False
    try:
        with FlopCounterMode(display=False) as counter:
            output = model(dummy)
            if backward:
                loss = _extract_loss(output)
                if loss is not None:
                    loss.backward()
                    backward_ran = True
        flops = counter.get_total_flops()
    except Exception as exc:
        logger.debug("FlopCounterMode failed: %s", exc)
        flops = 0
    finally:
        # Only clear gradients if backward actually ran — otherwise
        # we'd clobber caller-set grads even though we never touched
        # them.  This matters for the parameter-fallback path, which
        # never executes loss.backward() and must leave grads alone.
        if backward_ran:
            model.zero_grad(set_to_none=True)

    if flops > 0:
        return flops

    # FlopCounterMode returned 0 (common on XPU / non-CUDA devices).
    # Fall back to parameter-based estimate:
    #   forward ≈ 2 * params * tokens, backward ≈ 4 * params * tokens
    #   total ≈ 6 * params * tokens  (Kaplan et al.)
    num_params = sum(p.numel() for p in model.parameters())
    # For embedding models, last dim is sequence length (tokens per sample)
    # For vision/MLP, use product of spatial dims as "elements per sample"
    batch_size = input_shape[0]
    tokens = (
        input_shape[-1] if has_embedding
        else math.prod(input_shape[1:])
    )
    multiplier = 6 if backward else 2
    fallback = multiplier * num_params * batch_size * tokens
    logger.info(
        "FlopCounterMode returned 0 — using parameter-based estimate: %.2e",
        fallback,
    )
    return fallback


def try_estimate(
    model: torch.nn.Module,
    input_shape: tuple[int, ...] | list[int],
    *,
    device: torch.device | str | None = None,
    backward: bool = True,
) -> int:
    """Estimate model FLOPS, returning 0 on failure.

    Convenience wrapper around :func:`estimate_model_flops` that catches
    exceptions and logs a warning instead of propagating.  Intended to
    replace the repeated try/except/log boilerplate in example scripts.
    """
    try:
        flops = estimate_model_flops(
            model, input_shape, device=device, backward=backward,
        )
        if flops > 0:
            try:
                from ezpz.distributed import get_rank
                rank = get_rank()
            except Exception:
                rank = 0
            if rank == 0:
                logger.info("Model FLOPS (fwd+bwd): %.2e", flops)
        return flops
    except Exception as exc:
        logger.warning("FLOPS estimation failed: %s", exc)
        return 0


# ── MFU calculation ──────────────────────────────────────────────────────────


def compute_mfu(
    model_flops: int | float,
    step_duration: float,
    *,
    device_name: str | None = None,
    peak_flops: float | None = None,
) -> float:
    """Calculate per-device Model FLOPS Utilization (MFU) as a percentage.

    MFU measures what fraction of a single device's theoretical peak
    compute is used by the model's actual operations::

        MFU = model_flops / (peak_flops_per_device × step_duration)

    Both ``model_flops`` and ``peak_flops`` are per-device quantities,
    so ``world_size`` is not needed.  In data-parallel training
    (DDP/FSDP), every device performs the same forward+backward work.

    Args:
        model_flops: FLOPS per forward+backward pass (from
            :func:`estimate_model_flops`).  This is the per-device
            workload — the same value on every rank.
        step_duration: Wall-clock time for one training step (seconds).
        device_name: Device name for peak FLOPS lookup. Auto-detected
            if ``None``.
        peak_flops: Override the peak FLOPS value directly (skips
            device lookup).

    Returns:
        MFU as a percentage (0–100). Returns 0.0 if inputs are invalid.
    """
    if step_duration <= 0 or model_flops <= 0:
        return 0.0

    if peak_flops is None:
        peak_flops = get_peak_flops(device_name)
        if peak_flops is None:
            return 0.0

    if peak_flops <= 0:
        return 0.0

    return (model_flops / step_duration / peak_flops) * 100.0
