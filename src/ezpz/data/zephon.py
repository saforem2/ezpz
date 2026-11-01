"""ezpz/data/zephon.py — optional Zephon adapter for fsdp_tp.

Elastically deterministic dataloader for foundation model training:
packing + mixtures survive DP-size changes across checkpoint/resume, which
our DistributedSampler HF path does not. See
docs/superpowers/specs/2026-11-01-zephon-dataloader-design.md.

Requires ``ezpz[zephon]`` (``zephon>=0.2.1``). Import is deferred so a
vanilla ``import ezpz`` never touches Zephon.

Reference pattern: ``datologyai/torchtitan-zephon``'s
``torchtitan_recipes/zephon/dataloader.py`` — this module is the ezpz
leaf analog (TOML recipe → ZephonDataLoader, save_state/load_state).

Usage
-----
Install the extra, then wire Zephon in either directly or via the
``--dataloader zephon --zephon-recipe <toml>`` path added to the
from-scratch examples:

    uv pip install -U "ezpz[zephon]"
    ezpz launch python3 -m ezpz.examples.fsdp_tp --model small \\
        --dataloader zephon --zephon-recipe examples/zephon/local_jsonl.toml

Checkpoint integration reuses the existing ``save_checkpoint`` /
``load_checkpoint`` payload: the Zephon checkpoint fragment is stored
under the ``dataloader_state`` key and aggregated via torchrun's
TCPStore (or a shared directory on /eagle//flare).

"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Optional globusfs bridge — no-op if globusfs not installed
try:
    from ezpz.data.zephon_gcs_globus import register_globusfs_with_zephon as _register_globus

    _register_globus()
except Exception:
    pass


def _require_zephon() -> Any:  # type: ignore[no-untyped-def]
    try:
        import zephon  # noqa: F401  # presence check

        return zephon
    except ImportError as exc:
        raise ImportError(
            "Zephon is not installed. Install it with "
            '`uv pip install "ezpz[zephon]"` or `uv pip install zephon`.'
        ) from exc


def get_zephon_version() -> str | None:
    try:
        from importlib.metadata import version as _v

        return _v("zephon")
    except Exception:
        return None


# ── HF path compatibility ─────────────────────────────────────────

try:  # optional torch dependency on laptops without torch
    import torch  # type: ignore[no-redef]

    _Tensor = torch.Tensor
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _Tensor = object  # type: ignore[misc,assignment]


# ── Main adapter ──────────────────────────────────────────────────


class ZephonDataLoader:
    """Thin wrapper around a Zephon ``Pipeline`` for the fsdp_tp training loop.

    Parameters
    ----------
    recipe:
        Path to a Zephon TOML recipe (e.g. ``examples/zephon/local_jsonl.toml``).
        Relative paths inside the recipe are resolved relative to the recipe
        file — same semantics as ``torchtitan_recipes.zephon``.
    seq_len:
        Target sequence length (packer window). Should match the model's
        ``max_seq_len`` / ``seq_len``.
    microbatch_tokens:
        Tokens per microbatch per DP rank (for ``num_tokens_per_microbatch_per_dp_rank``).
    tokenizer_id:
        HF tokenizer repo id (default: the recipe's tokenizer).
    aggregate_dir:
        Shared directory for checkpoint aggregation (Polaris/Aurora
        ``/eagle`` or ``/flare``). Mutually exclusive with
        ``aggregate_store="torch://"``.
    aggregate_store:
        ``"torch://"`` to aggregate via torchrun's TCPStore (no shared FS).
    run_id:
        Stable per-run identifier; defaults to ``$ZEPHON_RUN_ID``. Must be
        constant across save and resume phases of one training run.
    extra:
        Passed through to ``Pipeline(...)`` / Zephon operators.

    """

    def __init__(
        self,
        recipe: str | Path,
        *,
        seq_len: int = 2048,
        microbatch_tokens: int | None = None,
        tokenizer_id: str | None = None,
        aggregate_dir: str | Path | None = None,
        aggregate_store: str | None = None,
        run_id: str | None = None,
        **extra: Any,
    ) -> None:
        _require_zephon()
        self.recipe = Path(recipe).resolve()
        if not self.recipe.is_file():
            raise FileNotFoundError(f"Zephon recipe not found: {self.recipe}")
        self.seq_len = int(seq_len)
        self.microbatch_tokens = microbatch_tokens
        self.tokenizer_id = tokenizer_id
        self.aggregate_dir = (
            Path(aggregate_dir).resolve() if aggregate_dir else None
        )
        self.aggregate_store = aggregate_store
        self.run_id = run_id or os.environ.get("ZEPHON_RUN_ID")
        self.extra = extra
        # Lazily built on first iter()
        self._pipeline: Any = None
        self._iterator: Iterator[Any] | None = None

        logger.info(
            "ZephonDataLoader(recipe=%s, seq_len=%d, aggregate=%s)",
            self.recipe,
            self.seq_len,
            self.aggregate_dir or self.aggregate_store or "(none)",
        )

    # ── Construction ──────────────────────────────────────────────

    @classmethod
    def from_toml(
        cls,
        recipe_path: str | Path,
        *,
        aggregate_dir: str | Path | None = None,
        aggregate_store: str | None = None,
        run_id: str | None = None,
        **extra: Any,
    ) -> ZephonDataLoader:
        """Construct from a Zephon TOML recipe file."""
        return cls(
            recipe_path,
            aggregate_dir=aggregate_dir,
            aggregate_store=aggregate_store,
            run_id=run_id,
            **extra,
        )

    def _build_pipeline(self) -> Any:
        """Materialise the Zephon Pipeline from the TOML recipe."""
        from zephon import Pipeline  # type: ignore[import-not-found]
        from zephon.io import Dataset  # type: ignore[import-not-found]
        from zephon.work import MixtureSpec, StaticMixtureWorkSource  # type: ignore[import-not-found]

        try:
            import tomllib

            with open(self.recipe, "rb") as f:
                cfg = tomllib.load(f)
        except ImportError:  # py3.10 fallback — needs tomli in zephon extra
            import tomli as tomllib  # type: ignore[import-not-found,no-redef]

            with open(self.recipe, "rb") as f:
                cfg = tomllib.load(f)

        sources_cfg = cfg.get("sources") or []
        if not sources_cfg:
            raise ValueError(f"No [[sources]] in Zephon recipe {self.recipe}")

        datasets = []
        weights: dict[str, float] = {}
        for src in sources_cfg:
            raw_path = src["path"]
            # hf:// URIs are zephon-native; local paths resolve vs recipe dir.
            resolved = (
                raw_path
                if "://" in str(raw_path)
                else str(Path(self.recipe).parent / raw_path)
            )
            ds = Dataset.from_path(
                src["name"], resolved, fmt=src.get("fmt") or None
            )
            datasets.append(ds)
            weights[src["name"]] = float(src.get("weight", 1.0))

        chunk_size = int(cfg.get("chunk_size", self.seq_len))
        text_field = cfg.get("text_field", "text")
        tok_id = self.tokenizer_id or cfg.get("tokenizer_id") or "gpt2"
        seed = int(cfg.get("seed", 42))

        work = StaticMixtureWorkSource(
            datasets=datasets,
            mixture=MixtureSpec(weights),
            seed=seed,
            chunk_size=chunk_size,
        )

        # Packing + batching — real 0.2.1 API is pack_flat + batch, not Pipeline.pack
        # (see torchtitan_recipes.zephon.dataloader). Mirrors that recipe.
        microbatch = int(self.microbatch_tokens or chunk_size)
        pipeline = (
            Pipeline(work)
            .decode_text()
            .tokenize(tokenizer_id=tok_id, field=text_field, padding=True)
            .pack_flat(
                max_length=self.seq_len,
                algorithm="wrap",
                pad_token_id=None,
                drop_oversized=False,
            )
            .batch(microbatch, drop_last=True)
        )

        # Aggregate settings (elastic checkpoint), DP degree, and any caller options.
        # Must be set before iteration so checkpoint aggregation knows where to store.
        pipe_opts: dict[str, Any] = {}
        if self.aggregate_dir is not None:
            pipe_opts["aggregate_dir"] = str(self.aggregate_dir)
        if self.aggregate_store is not None:
            pipe_opts["aggregate_store"] = self.aggregate_store
        if self.run_id is not None:
            pipe_opts["run_id"] = self.run_id
        # Pass through extra as runtime options (canonical_replicas etc.)
        for k, v in (self.extra or {}).items():
            if not k.startswith("_"):
                pipe_opts[k] = v
        if pipe_opts:
            pipeline = pipeline.options(**pipe_opts)

        # Apply deferred checkpoint state (save_state → new loader → load_state → build)
        pending = (self.extra or {}).pop("_restore_state", None)
        if pending is not None:
            # _pipeline must be set for restore helper below
            self._pipeline = pipeline
            self._restore_pipeline(pending)
            return self._pipeline

        return pipeline

    def _restore_pipeline(self, state: Any) -> None:
        assert self._pipeline is not None
        pipe = self._pipeline
        # Zephon 0.2.1 checkpoint API is checkpoint()/restore(); fall back to load_state variants.
        for attr in ("restore", "load_state", "load_state_dict"):
            if hasattr(pipe, attr):
                try:
                    getattr(pipe, attr)(state)
                    return
                except Exception as exc:
                    raise ValueError(
                        f"Zephon checkpoint incompatible: {exc}"
                    ) from exc
        raise ValueError(
            "Zephon pipeline has no restore method; checkpoint cannot be applied"
        )

    # ── Iter / checkpoint ──────────────────────────────────────────

    def __iter__(self) -> Iterator[Any]:
        if self._pipeline is None:
            self._pipeline = self._build_pipeline()
            # If _build_pipeline already applied a deferred restore, _restore_state is gone;
            # otherwise consume it now (covers direct Pipeline reuse).
            pending = (self.extra or {}).pop("_restore_state", None)
            if pending is not None:
                self._restore_pipeline(pending)
        self._iterator = iter(self._pipeline)
        return self

    def __next__(self) -> Any:
        if self._iterator is None:
            self.__iter__()
        assert self._iterator is not None
        batch = next(self._iterator)  # type: ignore[arg-type]
        # Normalise to (input, labels) if the batch is a Zephon wrapper type.
        if hasattr(batch, "to_training"):
            return batch.to_training()
        return batch

    def save_state(self) -> Any:
        """Return a checkpointable state; raises if checkpoint fails."""
        pipe = self._pipeline
        if pipe is None:
            return None
        for attr in ("checkpoint", "save_state", "state_dict"):
            if hasattr(pipe, attr):
                try:
                    return getattr(pipe, attr)()
                except Exception:
                    # checkpoint/restore failure is not a "fresh loader" signal — re-raise
                    raise
        return None

    def load_state(self, state: Any | None) -> None:
        """Restore from :meth:`save_state` (may be None for fresh runs); raises on corrupt state."""
        if state is None:
            return
        pipe = self._pipeline
        if pipe is None:
            self.extra["_restore_state"] = state
            return
        self._restore_pipeline(state)

    def __len__(self) -> int:
        # Some callers probe len(dataloader); Zephon is streaming/unknown.
        # Returning 0 signals "unknown" and keeps `for batch in loader:` working.
        return 0
