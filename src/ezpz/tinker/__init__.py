"""A Tinker-style fine-tuning API for ezpz.

Mirrors the shape of `Tinker <https://tinker-docs.thinkingmachines.ai/>`_
-- a decoupled ``forward_backward`` / ``optim_step`` pair over a
LoRA-adapted model -- but runs **in-process on your own allocation**
rather than as a hosted service.

The split is the point. A single fused "train one batch" call can only
express supervised fine-tuning; separating the two lets one client drive
SFT, DPO, GRPO and PPO, with each loss's extra inputs travelling
alongside the data in :class:`~ezpz.tinker.types.Datum`. It also makes
gradient accumulation expressible at all, which the fused loop in
``ezpz.examples.fsdp_tp`` could not do.

Everything composes with the existing distributed stack: FSDP2, tensor
parallelism, HSDP, meta-device init, and the async DCP checkpoint layer.
"""

from ezpz.tinker.client import (
    ImmediateFuture,
    LocalSamplingClient,
    LocalTrainingClient,
    build_training_client,
)
from ezpz.tinker.lora import (
    ATTN_TARGETS,
    MLP_TARGETS,
    LoraConfig,
    LoRALinear,
    adapter_state_dict,
    apply_lora,
    iter_lora_modules,
    lora_tp_plan,
    merge_adapters,
)
from ezpz.tinker.step import (
    TrainState,
    forward_backward,
    optim_step,
)
from ezpz.tinker.types import (
    AdamParams,
    Datum,
    ForwardBackwardOutput,
    LossFnInputs,
    ModelInput,
    OptimStepResponse,
    SamplingParams,
    SaveWeightsResponse,
)

__all__ = [
    "ATTN_TARGETS",
    "AdamParams",
    "Datum",
    "ForwardBackwardOutput",
    "ImmediateFuture",
    "LocalSamplingClient",
    "LocalTrainingClient",
    "LossFnInputs",
    "MLP_TARGETS",
    "LoRALinear",
    "LoraConfig",
    "ModelInput",
    "OptimStepResponse",
    "SamplingParams",
    "SaveWeightsResponse",
    "TrainState",
    "build_training_client",
    "forward_backward",
    "optim_step",
    "adapter_state_dict",
    "apply_lora",
    "iter_lora_modules",
    "lora_tp_plan",
    "merge_adapters",
]
