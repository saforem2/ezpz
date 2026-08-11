"""A Tinker-shaped client over the local training step.

Method names and argument shapes follow ``tinker.TrainingClient`` (checked
against the published SDK, ``tinker 0.25.0``) so a loop written against
one reads the same against the other::

    client = LocalTrainingClient(state)
    for batch in loader:
        client.forward_backward([batch], "cross_entropy").result()
        client.optim_step(AdamParams(learning_rate=3e-4)).result()

The difference is where the compute lives: Tinker ships your loop to a
hosted cluster, this runs it in your own allocation. So there is no auth,
no queue, no billing, and no network hop.

Calls still return a future-shaped object. That is not ceremony -- it
keeps the call sites identical if a remote backend is ever added, and it
is the one design decision here that would be expensive to retrofit.
:class:`ImmediateFuture` resolves eagerly; only ``.result()`` is
meaningful today.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generic, Optional, Sequence, TypeVar

import torch

import ezpz
from ezpz.tinker.step import (
    TrainState,
    forward_backward,
    optim_step,
    prepare_batch,
)
from ezpz.tinker.types import (
    AdamParams,
    ForwardBackwardOutput,
    OptimStepResponse,
    SaveWeightsResponse,
)

logger = ezpz.get_logger(__name__)

T = TypeVar("T")


class ImmediateFuture(Generic[T]):
    """An already-resolved future.

    Mirrors ``tinker.APIFuture`` so callers write ``.result()`` today and
    keep working unchanged against a genuinely async backend later.
    """

    __slots__ = ("_value",)

    def __init__(self, value: T) -> None:
        self._value = value

    def result(self, timeout: float | None = None) -> T:  # noqa: ARG002
        return self._value

    def done(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"ImmediateFuture({self._value!r})"


class LocalTrainingClient:
    """Drive :mod:`ezpz.tinker.step` with Tinker's method names.

    Args:
        state: a :class:`~ezpz.tinker.step.TrainState` built by the
            caller (``fsdp_tp.train()`` already constructs everything it
            needs). This client does not build models -- keeping
            construction in one place avoids a second, drifting setup path.
    """

    def __init__(self, state: TrainState) -> None:
        self._state = state

    # -- properties ---------------------------------------------------------

    @property
    def state(self) -> TrainState:
        return self._state

    @property
    def step(self) -> int:
        return self._state.global_step

    def get_tokenizer(self) -> Any:
        """The dataset's tokenizer, or one stashed in ``state.extras``."""
        tok = getattr(self._state.dataset, "tokenizer", None)
        if tok is not None:
            return tok
        tok = self._state.extras.get("tokenizer")
        if tok is not None:
            return tok
        raise AttributeError(
            "no tokenizer available: the dataset has none and "
            "TrainState.extras['tokenizer'] is unset"
        )

    # -- training -----------------------------------------------------------

    def forward_backward(
        self,
        data: Sequence[Any],
        loss_fn: str = "cross_entropy",
        loss_fn_config: Optional[dict[str, float]] = None,  # noqa: ARG002
    ) -> ImmediateFuture[ForwardBackwardOutput]:
        """Accumulate gradients over ``data``; do not update.

        ``data`` is a sequence of microbatches, each scaled so the
        accumulated gradient equals what one batch of the same total
        size would produce -- the caller should not have to remember to
        scale.

        The scale is each microbatch's **share of the valid tokens**,
        not ``1/len(data)``. Every microbatch's cross-entropy is already
        a mean over its own tokens, so a uniform ``1/N`` gives a
        300-token and a 100-token microbatch equal pull (0.5 / 0.5)
        where a single combined batch would weight them 0.75 / 0.25.
        That silently changes results for ragged or differently padded
        inputs.

        Counting tokens needs a forward pass, so this runs
        :func:`prepare_batch` first to size every microbatch, then does
        the scaled forward+backward. The extra pass is cheap: it moves
        and shifts tensors, no model call.

        Returns the token-weighted mean loss -- the correct average for
        ragged microbatches.
        """
        if not data:
            raise ValueError("forward_backward requires at least one batch")

        # Pass 1: size each microbatch (no model, no autograd).
        with torch.no_grad():
            counts = [prepare_batch(self._state, b).num_tokens for b in data]
        total_tokens = sum(counts)
        # All-empty (or an unlabeled loss) -> fall back to uniform, which
        # is what "no token information" can honestly support.
        scales = (
            [c / total_tokens for c in counts]
            if total_tokens > 0
            else [1.0 / len(data)] * len(data)
        )

        # Pass 2: the real work.
        total_loss = 0.0
        outputs: list[ForwardBackwardOutput] = []
        for batch, count, scale in zip(data, counts, scales):
            if count == 0 and total_tokens > 0:
                # Nothing to score. Cross-entropy over zero valid targets
                # is 0/0 = NaN, which would poison the weighted mean via
                # `0 * NaN`, and its gradient is meaningless. Skip it --
                # cheaper and correct. (When EVERY microbatch is empty we
                # still run them, so the caller gets a real error rather
                # than a silent no-op.)
                continue
            out = forward_backward(
                self._state, batch, loss_fn, loss_scale=scale
            )
            outputs.append(out)
            # Weight by ACTUAL tokens, not max(n, 1), which gave an empty
            # microbatch full weight whenever any other had tokens.
            total_loss += out.loss * out.num_tokens

        if total_tokens > 0:
            mean_loss = total_loss / total_tokens
        else:
            mean_loss = sum(o.loss for o in outputs) / len(outputs)
        return ImmediateFuture(
            ForwardBackwardOutput(
                loss=mean_loss,
                num_tokens=total_tokens,
                metrics={"num_microbatches": float(len(outputs))},
            )
        )

    def optim_step(
        self, adam_params: Optional[AdamParams] = None
    ) -> ImmediateFuture[OptimStepResponse]:
        """Clip, update, and zero the accumulated gradients."""
        return ImmediateFuture(optim_step(self._state, adam_params))

    # -- persistence --------------------------------------------------------

    def save_state(
        self,
        name: str,
        *,
        adapters_only: bool = True,
        ttl_seconds: Optional[int] = None,  # noqa: ARG002
        overwrite: bool = False,
    ) -> ImmediateFuture[SaveWeightsResponse]:
        """Checkpoint through the existing DCP layer.

        ``adapters_only`` defaults to True because that is the point of
        LoRA -- an adapter checkpoint is ~``rank/dim`` the size of a full
        one. It maps to
        ``StateDictOptions(ignore_frozen_params=True)``; with no frozen
        params it is simply a no-op.

        Args:
            overwrite: remove an existing checkpoint at this step before
                writing. Without it DCP writes into the existing
                directory, which can leave a mix of old and new shards.
            ttl_seconds: **accepted and ignored.** It exists so a loop
                written against Tinker's signature runs here unchanged;
                expiry is a hosted-service concern with no local
                meaning, and silently dropping the argument is better
                than rejecting an otherwise-portable call. Local
                checkpoints never expire -- delete them yourself.
        """
        from ezpz.examples._checkpoint import save_checkpoint

        options = None
        if adapters_only:
            from torch.distributed.checkpoint.state_dict import StateDictOptions

            options = StateDictOptions(ignore_frozen_params=True)

        if overwrite:
            import shutil

            stale = Path(name) / f"step-{self._state.global_step}"
            if stale.is_dir():
                logger.info("overwrite=True: removing %s", stale)
                shutil.rmtree(stale)

        path = save_checkpoint(
            Path(name),
            self._state.global_step,
            self._state.model,
            self._state.optimizer,
            state_dict_options=options,
        )
        return ImmediateFuture(
            SaveWeightsResponse(
                path=str(path),
                step=self._state.global_step,
                adapters_only=adapters_only,
            )
        )

    def load_state(
        self, path: str, *, adapters_only: bool = True
    ) -> ImmediateFuture[Optional[dict[str, Any]]]:
        """Restore a checkpoint written by :meth:`save_state`.

        ``adapters_only`` must MATCH the save. It defaults to True for
        the same reason ``save_state`` does, so the client can round-trip
        its own default: an adapter-only checkpoint holds no frozen base
        weights, and shaping the load with full-state options would ask
        DCP for keys the checkpoint does not contain.
        """
        from ezpz.examples._checkpoint import load_checkpoint

        options = None
        if adapters_only:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
            )

            options = StateDictOptions(ignore_frozen_params=True)

        meta = load_checkpoint(
            Path(path),
            self._state.model,
            self._state.optimizer,
            state_dict_options=options,
        )
        if meta:
            self._state.global_step = int(meta.get("step", 0) or 0)
        return ImmediateFuture(meta)

    def save_weights_and_get_sampling_client(
        self, name: Optional[str] = None
    ) -> "LocalSamplingClient":
        """Persist, then hand back a sampler over the current weights.

        Tinker's version round-trips weights to an inference cluster; in
        process we can sample the live model directly, so the save is
        only for durability and is skipped when ``name`` is None.
        """
        if name is not None:
            self.save_state(name).result()
        return LocalSamplingClient(self._state)


class LocalSamplingClient:
    """Generate from the model currently held in :class:`TrainState`.

    Deliberately thin. ezpz already has generation
    (``ezpz.examples.generate``); duplicating decoding here would create a
    second implementation to keep in sync.
    """

    def __init__(self, state: TrainState) -> None:
        self._state = state

    @property
    def model(self) -> Any:
        return self._state.model

    def get_tokenizer(self) -> Any:
        return getattr(self._state.dataset, "tokenizer", None)

    def sample(
        self,
        prompt: Any,
        num_samples: int = 1,
        sampling_params: Any = None,
    ) -> ImmediateFuture[list[Any]]:
        raise NotImplementedError(
            "LocalSamplingClient.sample is not wired up yet. Use "
            "`ezpz.examples.generate` against a saved checkpoint; the "
            "in-process path lands with the RL loop, which is the first "
            "thing that actually needs train->sample handoff."
        )


def build_training_client(state: TrainState) -> LocalTrainingClient:
    """Convenience constructor mirroring ``ServiceClient.create_*``."""
    return LocalTrainingClient(state)
