"""LoRA adapters for :mod:`ezpz.models.llama`, composable with FSDP2 + TP.

Low-Rank Adaptation freezes the pretrained weight ``W`` and learns a
low-rank update, so ``y = Wx + (alpha/r)·B(A(x))`` with
``A: d_in -> r`` and ``B: r -> d_out``. ``B`` is zero-initialized, so an
adapted model is *exactly* the base model at step 0 and training starts
from a known-good point.

Three facts about this codebase shaped the design; each was measured, not
assumed (torch 2.12.1):

1. **FSDP2 tolerates mixed ``requires_grad``.** ``fully_shard`` over a
   module whose base weights are frozen and whose adapters are not works:
   gradients reach the adapters only, and no frozen parameter accumulates
   one. So LoRA needs no special sharding treatment.

2. **The tensor-parallel plan targets modules by NAME, and breaks.**
   ``fsdp_tp.parallelize`` maps ``"attention.wq" -> ColwiseParallel()``
   (fsdp_tp.py:2011-2021). Swapping ``wq`` for a wrapper raises
   ``NotImplementedError: ColwiseParallel currently only support nn.Linear
   and nn.Embedding!``. :func:`lora_tp_plan` rewrites those keys to point
   at the inner ``nn.Linear`` layers instead. Without it, ``tp>1`` dies.

   Getting that rewrite *right* needs a real multi-rank mesh to verify.
   A ``world_size=1`` mesh satisfies every placement trivially, and
   ``ColwiseParallel``'s input hook uses ``DTensor.from_local(...,
   run_check=False)``, so torch accepts a mislabeled layout instead of
   raising -- a single-rank probe reports success for a plan that is
   wrong at ``tp=2``. ``tests/test_tinker_lora_tp.py`` runs a genuine
   2-rank gloo mesh and gates on tp=2 matching tp=1 numerically.

3. **Model init reaches through to ``.weight``.**
   ``Attention.init_weights`` does ``nn.init.trunc_normal_(linear.weight,
   ...)`` (llama.py:359-360, :503-505) on exactly the modules we wrap, so
   :class:`LoRALinear` proxies ``.weight`` to the frozen base. Without the
   proxy, every native model build raises ``AttributeError``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
import torch.nn as nn

import ezpz

logger = ezpz.get_logger(__name__)

# Module attribute names on ezpz.models.llama, grouped by role. Mirrors
# Tinker's LoraConfig, which gates by role (train_attn / train_mlp) rather
# than by name-regex -- the roles are stable across model families even
# when the attribute spellings are not.
ATTN_TARGETS: tuple[str, ...] = ("wq", "wk", "wv", "wo")
MLP_TARGETS: tuple[str, ...] = ("w1", "w2", "w3")


@dataclass
class LoraConfig:
    """Which modules to adapt, and how.

    Field names follow ``tinker.types.LoraConfig`` so the two are
    interchangeable at the call site.
    """

    rank: int = 32
    alpha: float | None = None
    """LoRA scaling numerator. ``None`` -> use ``rank`` (i.e. scale 1.0)."""
    dropout: float = 0.0
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = False
    """Adapt the output/unembedding projection. Off by default: it is the
    single largest matrix in the model and rarely worth the memory."""
    seed: int | None = None
    extra_targets: tuple[str, ...] = field(default_factory=tuple)
    """Additional attribute names to adapt, beyond the role defaults."""

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {self.rank}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"LoRA dropout must be in [0, 1), got {self.dropout}"
            )
        if not (
            self.train_attn
            or self.train_mlp
            or self.train_unembed
            or self.extra_targets
        ):
            raise ValueError(
                "LoraConfig adapts nothing: enable train_attn / train_mlp / "
                "train_unembed, or pass extra_targets."
            )

    def target_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.train_attn:
            names.extend(ATTN_TARGETS)
        if self.train_mlp:
            names.extend(MLP_TARGETS)
        names.extend(self.extra_targets)
        return tuple(dict.fromkeys(names))  # dedupe, keep order


class LoRALinear(nn.Module):
    """``nn.Linear`` with a frozen base and a trainable low-rank update.

    Args:
        base: the ``nn.Linear`` to adapt. Frozen in place.
        rank: inner dimension ``r`` of the update.
        alpha: scaling numerator; the update is scaled by ``alpha / rank``.
            ``None`` means ``alpha = rank`` (scale 1.0).
        dropout: applied to the *input* of the adapter branch only.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 32,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(
                f"LoRALinear expects nn.Linear, got {type(base).__name__}"
            )
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / self.rank

        # Named A/B (not lora_A/lora_B) so the TP plan keys read naturally:
        # "attention.wq.A" / "attention.wq.B".
        self.A = nn.Linear(base.in_features, self.rank, bias=False)
        self.B = nn.Linear(self.rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.reset_lora_parameters()

    # -- init ---------------------------------------------------------------

    def reset_lora_parameters(self) -> None:
        """Kaiming-uniform ``A``, zeros ``B``.

        Zeroing ``B`` makes the adapter a no-op at step 0, so the adapted
        model's output is bit-identical to the base model's. Pinned by
        ``test_zero_init_is_identity``.
        """
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def init_weights(self, init_std: float | None = None) -> None:
        """Match the per-module convention in :mod:`ezpz.models.llama`.

        The base is initialized by its owner (``Attention.init_weights`` et
        al. write through our ``.weight`` proxy); we only (re)initialize the
        adapter. Also called after ``to_empty()`` on the meta-init path,
        where adapter storage is uninitialized and MUST be reset -- garbage
        in ``B`` would silently corrupt the "identity at step 0" property.
        """
        self.reset_lora_parameters()

    # -- transparency to the base -------------------------------------------

    @property
    def weight(self) -> nn.Parameter:
        """Proxy to the frozen base weight.

        REQUIRED: ``Attention.init_weights`` / ``FeedForward.init_weights``
        do ``nn.init.trunc_normal_(linear.weight, ...)`` directly
        (llama.py:359-360, :503-505). Without this, wrapping raises
        ``AttributeError`` during model construction.
        """
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        upd = self.B(self.A(self.dropout(x)))
        # Adapter runs in the base's compute dtype (autocast/FSDP may hand
        # the branches different dtypes under mixed precision).
        return out + self.scaling * upd.to(out.dtype)

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4g}"
        )


def apply_lora(
    model: nn.Module,
    cfg: LoraConfig,
    *,
    verbose: bool = True,
) -> nn.Module:
    """Wrap the configured ``nn.Linear`` targets in-place; freeze the rest.

    Returns the same ``model`` object (mutated), so it composes with the
    existing ``model = parallelize(model, ...)`` style in fsdp_tp.

    Call BEFORE ``parallelize()``: the TP plan and ``fully_shard`` must see
    the final module tree.
    """
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    targets = set(cfg.target_names())
    wrapped: list[str] = []

    # Freeze everything first, then let each adapter un-freeze its own A/B.
    # Simpler to reason about than tracking which params to leave alone.
    for p in model.parameters():
        p.requires_grad_(False)

    for mod_name, module in list(model.named_modules()):
        for attr in list(vars(module).get("_modules", {})):
            if attr not in targets:
                continue
            child = getattr(module, attr, None)
            if not isinstance(child, nn.Linear):
                continue
            setattr(
                module,
                attr,
                LoRALinear(child, cfg.rank, cfg.alpha, cfg.dropout),
            )
            wrapped.append(f"{mod_name}.{attr}" if mod_name else attr)

    if cfg.train_unembed:
        out = getattr(model, "output", None)
        if isinstance(out, nn.Linear):
            model.output = LoRALinear(  # type: ignore[assignment]
                out, cfg.rank, cfg.alpha, cfg.dropout
            )
            wrapped.append("output")

    if not wrapped:
        raise RuntimeError(
            f"apply_lora matched no modules (targets={sorted(targets)}). "
            "Is this an ezpz.models.llama Transformer?"
        )

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    if verbose:
        logger.info(
            "LoRA: wrapped %d modules (rank=%d, alpha=%g); trainable "
            "%s/%s params (%.3f%%)",
            len(wrapped),
            cfg.rank,
            float(cfg.rank if cfg.alpha is None else cfg.alpha),
            f"{n_train:,}",
            f"{n_all:,}",
            100.0 * n_train / max(n_all, 1),
        )
    return model


def lora_tp_plan(
    base_plan: dict[str, Any], module: nn.Module | None = None
) -> dict[str, Any]:
    """Retarget a tensor-parallel plan at the inner Linears of wrapped modules.

    ``parallelize_module`` dispatches on the module *class*, so a plan entry
    for ``"attention.wq"`` fails once ``wq`` is a :class:`LoRALinear`:

        NotImplementedError: ColwiseParallel currently only support
        nn.Linear and nn.Embedding!

    Each wrapped target becomes three entries -- ``base``, ``A`` and
    ``B`` -- whose styles are *derived* from the original rather than
    copied wholesale. ``A`` mirrors what the base consumes and ``B``
    what it produces; see :func:`_adapter_in_style` and
    :func:`_adapter_out_style` for why each of the four layout choices
    is forced.

    Pass ``module`` whenever the model is available. A plan key is
    retargeted only if it resolves to an actual :class:`LoRALinear`,
    which matters because ``--lora-target`` is selective: with the
    default ``attn,mlp`` the unembedding ``output`` is *not* wrapped, so
    rewriting its key would point at ``output.base`` and friends, which
    do not exist. ``parallelize_module`` ignores keys that match no
    module **without raising**, so ``output`` would silently stay an
    unsharded ``nn.Linear`` and fail downstream on its first DTensor
    input::

        RuntimeError: aten.mm.default got mixed torch.Tensor and
        DTensor, need to convert all torch.Tensor to DTensor before
        calling distributed operators!

    Omitting ``module`` falls back to retargeting every key whose leaf
    names a LoRA-able module, which is right only when every one of
    them is in fact wrapped.

    Keys naming something other than a wrapped target -- ``norm``,
    ``PrepareModuleInput`` -- pass through untouched either way.
    """
    adapted = set(ATTN_TARGETS) | set(MLP_TARGETS) | {"output"}
    out: dict[str, Any] = {}
    for key, style in base_plan.items():
        leaf = key.rsplit(".", 1)[-1]
        if leaf not in adapted or not _is_wrapped(module, key):
            out[key] = style
            continue

        out[f"{key}.base"] = style
        out[f"{key}.A"] = _adapter_in_style(style)
        out[f"{key}.B"] = _adapter_out_style(style)
    return out


def _first(layouts: Any) -> Any:
    """``input_layouts``/``output_layouts`` are normalized to tuples."""
    return layouts[0] if isinstance(layouts, tuple) else layouts


def _is_wrapped(module: nn.Module | None, key: str) -> bool:
    """Does ``key`` name a :class:`LoRALinear` inside ``module``?

    ``None`` means "no model to check against", and the caller then
    assumes every LoRA-able key is wrapped -- see :func:`lora_tp_plan`.
    """
    if module is None:
        return True
    target: Any = module
    for part in key.split("."):
        target = getattr(target, part, None)
        if target is None:
            return False
    return isinstance(target, LoRALinear)


def _adapter_in_style(style: Any) -> Any:
    """The style for ``A``, derived from what the base *consumes*.

    ``A`` is fed the same activation as ``base``, so it must agree with
    the base on that tensor's layout, and it must hand ``B`` a
    replicated ``r``-wide result.

    Copying the base's class is not cosmetic. Under
    ``RowwiseParallel``, ``base`` receives an input already sharded on
    the feature dimension, and its weight is sharded to match. A
    ``ColwiseParallel`` ``A`` declares that same input *replicated* and
    keeps a full-width weight, so the contraction dimensions disagree::

        Sharding propagation failed for aten.mm.default(
            Spec(f32[16, 64](R)), Spec(f32[128, 8](S(1))))

    (``attention.wo`` at ``tp=2``: activation 64 wide, weight 128.)
    Mirroring the class shards ``A``'s weight the same way the base's
    is, so the two line up.

    ``output_layouts=Replicate()`` because ``B`` consumes this, and
    ``use_local_output=False`` to keep it a DTensor -- the default
    ``True`` unwraps to this rank's shard, so ``B`` would see ``r/tp``
    instead of ``r``.
    """
    from torch.distributed.tensor import Replicate

    return type(style)(
        input_layouts=_first(style.input_layouts),
        output_layouts=Replicate(),
        use_local_output=False,
    )


def _adapter_out_style(style: Any) -> Any:
    """The style for ``B``, derived from what the base *produces*.

    ``B``'s output is summed with ``base``'s, so it must land in the
    same layout; and its input is ``A``'s replicated output. Reusing
    ``style`` unmodified is wrong whenever the base's declared
    ``input_layouts`` is not ``Replicate()`` -- ``B`` is fed by ``A``,
    not by the base's input.
    """
    from torch.distributed.tensor import Replicate

    return type(style)(
        input_layouts=Replicate(),
        output_layouts=_first(style.output_layouts),
        use_local_output=style.use_local_output,
    )


def iter_lora_modules(model: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            yield name, mod


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Just the trainable adapter tensors, for a small portable export.

    Note the DCP path does NOT use this -- it uses
    ``StateDictOptions(ignore_frozen_params=True)``, which is
    sharding-aware. This is for single-file export/inspection.
    """
    return {
        name: param.detach()
        for name, param in model.named_parameters()
        if param.requires_grad and (".A." in name or ".B." in name)
    }


@torch.no_grad()
def merge_adapters(model: nn.Module) -> nn.Module:
    """Fold every adapter into its base weight, in place, and unwrap.

    After this the model is a plain ``Transformer`` again -- useful for
    export or inference with no adapter overhead. Not safe to call mid-
    training: it discards the separate adapter parameters the optimizer
    holds state for.
    """
    for parent_name, parent in list(model.named_modules()):
        for attr, child in list(parent.named_children()):
            if not isinstance(child, LoRALinear):
                continue
            delta = child.scaling * (child.B.weight @ child.A.weight)
            child.base.weight.add_(delta.to(child.base.weight.dtype))
            child.base.weight.requires_grad_(True)
            setattr(parent, attr, child.base)
            logger.debug(
                "merged adapter %s", f"{parent_name}.{attr}".lstrip(".")
            )
    return model
