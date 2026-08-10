"""Data contract for the ezpz Tinker-style API.

Field names and shapes deliberately mirror ``tinker.types`` (verified
against the published SDK, ``tinker 0.25.0``) so code written against one
reads the same against the other:

    Datum        = {model_input: ModelInput, loss_fn_inputs: LossFnInputs}
    AdamParams   = {learning_rate, beta1, beta2, eps, weight_decay,
                    grad_clip_norm}
    LossFnType   = 'cross_entropy' | 'importance_sampling' | 'ppo' |
                   'cispo' | 'dro'

Two choices here are load-bearing rather than cosmetic:

**Loss is named, and its extra inputs ride with the data.** A PPO step
needs advantages and old logprobs; a DPO step needs a chosen/rejected
pair. Putting them in ``Datum.loss_fn_inputs`` instead of in the trainer
means adding an algorithm never changes the client API. Only
``cross_entropy`` is implemented today; the enum reserves the rest.

**``grad_clip_norm`` lives on ``AdamParams``.** Clipping happens between
the last ``forward_backward`` and the update, so it belongs to the step,
not to run-level config -- which is also what makes it tunable per-step
in an RL loop.

Plain dataclasses, not pydantic: ezpz has no pydantic dependency and
these never cross a network boundary in phase A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

LossFnType = Literal[
    "cross_entropy",
    "importance_sampling",
    "ppo",
    "cispo",
    "dro",
]

#: Losses this implementation can actually run. The rest are reserved so
#: the type doesn't churn when they land.
IMPLEMENTED_LOSS_FNS: frozenset[str] = frozenset({"cross_entropy"})


@dataclass
class ModelInput:
    """Tokens for one sequence.

    Tinker's version carries a list of typed chunks (text / images). We
    take pre-tokenized ids, which is what ``ezpz.data.hf`` already
    produces; a chunked form can be added without changing callers.
    """

    token_ids: Sequence[int]

    def __len__(self) -> int:
        return len(self.token_ids)


@dataclass
class LossFnInputs:
    """Per-loss extras that travel with the datum.

    ``cross_entropy`` needs only ``targets`` (and optionally
    ``weights``). The remaining fields are here so an RL loop can be
    written against the same ``Datum`` without a second type.
    """

    targets: Optional[Sequence[int]] = None
    weights: Optional[Sequence[float]] = None
    advantages: Optional[Sequence[float]] = None
    logprobs: Optional[Sequence[float]] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Datum:
    """One training example: the input, plus whatever its loss needs."""

    model_input: ModelInput
    loss_fn_inputs: LossFnInputs = field(default_factory=LossFnInputs)


@dataclass
class AdamParams:
    """Optimizer settings applied at :meth:`optim_step`.

    Defaults match the hardcoded AdamW in ``fsdp_tp.py:3020-3030``
    (betas 0.9/0.95, eps 1e-8, weight_decay 0.1) so routing an existing
    run through this API changes nothing.
    """

    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate < 0:
            raise ValueError(
                f"learning_rate must be >= 0, got {self.learning_rate}"
            )
        for name in ("beta1", "beta2"):
            val = getattr(self, name)
            if not (0.0 <= val < 1.0):
                raise ValueError(f"{name} must be in [0, 1), got {val}")


@dataclass
class SamplingParams:
    """Decoding settings. Mirrors ``tinker.types.SamplingParams``."""

    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: Optional[int] = None
    stop: Optional[Sequence[str]] = None


@dataclass
class ForwardBackwardOutput:
    """What one :meth:`forward_backward` produced.

    ``loss`` is the scalar for this microbatch. ``num_tokens`` lets the
    caller weight a multi-microbatch average correctly -- ragged batches
    would otherwise be silently mis-averaged.
    """

    loss: float
    num_tokens: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class OptimStepResponse:
    """What one :meth:`optim_step` produced."""

    step: int
    grad_norm: Optional[float] = None
    learning_rate: Optional[float] = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class SaveWeightsResponse:
    """Where a checkpoint landed."""

    path: str
    step: int
    adapters_only: bool = False


def validate_loss_fn(loss_fn: str) -> str:
    """Reject unimplemented losses at the call site, with a clear message.

    Better to fail here than deep in the step with a confusing shape
    error -- and it keeps the reserved names honest about not working yet.
    """
    if loss_fn in IMPLEMENTED_LOSS_FNS:
        return loss_fn
    reserved = set(LossFnType.__args__)  # type: ignore[attr-defined]
    if loss_fn in reserved:
        raise NotImplementedError(
            f"loss_fn={loss_fn!r} is reserved but not implemented yet; "
            f"available: {sorted(IMPLEMENTED_LOSS_FNS)}"
        )
    raise ValueError(
        f"unknown loss_fn={loss_fn!r}; expected one of {sorted(reserved)}"
    )
