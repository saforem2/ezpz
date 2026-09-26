"""Workaround for upstream xccl's missing ``supportsSplitting`` override.

``torch/csrc/distributed/c10d/Backend.hpp`` declares
``virtual bool supportsSplitting() const { return false; }``.
``ProcessGroupNCCL`` overrides it to ``return true;``; ``ProcessGroupXCCL``
does **not**, so it inherits the base ``false``.

``DeviceMesh._init_one_process_group`` gates ``split_group`` vs ``new_group``
on ``bound_device_id is not None`` plus accelerator availability plus a
matching backend name. All three hold for an xccl-backed default group on XPU
(``_setup_ddp`` passes ``device_id=`` so FSDP2's ``foreach_all_gather`` routes
correctly), so torch takes the ``split_group`` branch. That calls
``parent_backend.supports_splitting``, gets ``False``, and raises::

    RuntimeError: No backend for the parent process group or its backend
                  does not support splitting

**This blocks every nested mesh construction on XPU.** A 2D
``(dp_shard, tp)`` mesh is exactly that, which is why ``--tp 1`` was fine and
``--tp > 1`` was not: ezpz #252, where FSDP+TP stalled on Aurora and Sunspot
after a handful of iterations with all ranks parked in ``reduce_scatter``.

Why a monkey-patch rather than more call-site guards
----------------------------------------------------
``ezpz.distributed._xccl_split_workaround`` already clears
``bound_device_id`` around ``init_device_mesh`` and ``DeviceMesh._flatten``.
That covers mesh *creation* but not mesh *indexing*: ``device_mesh["tp"]`` and
``device_mesh["dp_replicate", "dp_shard"]`` lazily construct process groups
too, and those calls sit outside both guards. Patching the function every
path funnels through closes the whole class instead of one case at a time.

The real fix belongs upstream, in two places:

1. ``ProcessGroupXCCL`` declaring ``supportsSplitting() override
   { return true; }``;
2. ``ProcessGroupXCCL::split`` actually being implemented and exercised.

Remove this module once both hold.

Adapted from torchtitan's ``experiments/ezpz/xccl_split_group_workaround.py``
(validated there on Aurora ``next-eval``, job 8868654).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED_ATTR = "_ezpz_xccl_split_group_patched"


def _should_patch() -> bool:
    """True iff this is an XPU build with xccl available.

    A pure no-op on CUDA/CPU: the bug only manifests under xccl, and
    ``ProcessGroupNCCL`` overrides ``supportsSplitting`` correctly, so
    leaving the upstream path untouched elsewhere keeps the blast radius
    minimal.
    """
    try:
        import torch

        return bool(
            torch.distributed.is_xccl_available() and torch.xpu.is_available()
        )
    except (AttributeError, ImportError, RuntimeError):
        # Older torch has no is_xccl_available; a CUDA build has no
        # torch.xpu that initialises. Either way: not our case.
        return False


def maybe_install_xccl_split_group_workaround() -> bool:
    """Install the xccl ``split_group`` workaround if needed.

    Idempotent, and a no-op off XPU.

    Returns:
        ``True`` if the patch is in place (now or from an earlier call),
        ``False`` if it was not needed.
    """
    if not _should_patch():
        return False

    import torch
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.distributed_c10d import _get_default_group

    if getattr(DeviceMesh, _PATCHED_ATTR, False):
        return True

    original_init_one_process_group = DeviceMesh._init_one_process_group

    @staticmethod
    def _patched_init_one_process_group(
        sub_layout,
        rank_map,
        dim_name,
        backend_override,
        *extra_args,
        **extra_kwargs,
    ):
        """Take the ``new_group`` path when the parent backend cannot split.

        ``*extra_args`` / ``**extra_kwargs`` forward whatever upstream adds
        after ``backend_override`` rather than pinning this shim to one torch
        version: torch 2.15 added a fifth positional ``preserve_rank_order``,
        and a four-argument signature raises ``TypeError`` during device-mesh
        construction, before any model code runs.
        """

        def _call():
            return original_init_one_process_group(
                sub_layout,
                rank_map,
                dim_name,
                backend_override,
                *extra_args,
                **extra_kwargs,
            )

        try:
            default_group = _get_default_group()
        except Exception:
            # No PG yet — call through so upstream raises its own message.
            return _call()

        try:
            accel = torch.accelerator.current_accelerator()
        except Exception:
            return _call()
        if accel is None:
            return _call()

        try:
            parent_backend = default_group._get_backend(accel)
        except Exception:
            return _call()

        if getattr(parent_backend, "supports_splitting", False):
            # NCCL and other well-behaved backends: upstream fast path.
            return _call()

        # supports_splitting is False (xccl today). Steer the upstream gate's
        # first clause to False by temporarily stripping bound_device_id, so
        # it falls through to new_group -- which xccl does support.
        #
        # backend_override passes through unchanged so new_group uses the
        # parent PG's backend spec (the multi-backend "cpu:gloo,xpu:xccl"
        # string). bound_device_id is restored in `finally` because FSDP2
        # needs it for per-device PG resolution.
        #
        # Do NOT "simplify" this by dropping device_id at init_process_group
        # time: without device-bound PGs, foreach_all_gather routes some ranks
        # to xpu:0 and others to xpu:LOCAL_RANK and they never meet -- a
        # silent FSDP2 deadlock rather than a loud error.
        saved_bound_device_id = getattr(default_group, "bound_device_id", None)
        try:
            default_group.bound_device_id = None  # type: ignore[attr-defined]
            return _call()
        finally:
            default_group.bound_device_id = saved_bound_device_id  # type: ignore[attr-defined]

    DeviceMesh._init_one_process_group = _patched_init_one_process_group
    setattr(DeviceMesh, _PATCHED_ATTR, True)
    logger.info(
        "Installed xccl split_group workaround on "
        "DeviceMesh._init_one_process_group (upstream ProcessGroupXCCL has "
        "no supportsSplitting() override; see ezpz/xccl_split_group.py)."
    )
    return True
