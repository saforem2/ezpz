"""Turn a finished ezpz run into an AmSC at-scale benchmark CSV row.

Emits one row of throughput metrics: ``timestamp, system, config,
nodes, gpus, status, wall_time_sec`` plus the measured figures, with an
empty field meaning "not measured" (``None``), never zero.

.. warning::

   These columns come from the `AmSC at-scale benchmarks
   <https://gitlab.com/amsc2/ai-services/at-scale-services/amsc-atscale-benchmarks>`_
   dashboard's ``build_dashboard.py``, which required ``benchmark.yaml``
   + ``results/runs.csv``. **Re-checked against origin/main 2026-08-25:
   no ``runs.csv`` exists on any branch, and build_dashboard.py has
   never been in that repo's history** -- it lives in the separate
   dashboard project. The two benchmarks that do publish results
   disagree with each other: ``vit-weather`` writes
   ``results/<System>/throughput_metrics.csv`` with three columns
   (``configuration_name, samples_per_sec, iters_per_sec``);
   ``mldocking`` writes free-form ``results/summary_<system>.txt``.

   This is not a competing invention: these columns came from the
   dashboard's own requirements, which makes them the only shape in
   play with a documented provenance. That repo currently has no
   schema, no ``benchmark.yaml`` and no CONTRIBUTING -- its
   ``results/README.md`` asks only for "structured subfolders with
   provenance metadata when possible" -- so the format is being
   proposed upstream rather than forked.

   Meanwhile this writer will not translate columns: contributing to a
   benchmark that already publishes results means matching it.

Two measurements from real Sunspot runs shaped the defaults here, and
both are the kind of thing that silently produces a wrong-but-plausible
number:

**Warmup dominates.** ``train/tps`` over an ``agpt-2b`` run reads
``1045, 34601, 34686, 34715, 34833, 34650`` -- step 1 is a 33x outlier
(compile, allocator warmup, lazy init). A mean over all steps reports
~29k against a true ~34.7k. Hence :data:`DEFAULT_WARMUP` = 1 and a
*median* reducer, which is additionally robust to a straggler step.

**Wall time is not the timestamp span.** JSONL records only exist for
logged steps, so first-to-last spans 1.18 s of a 6.76 s run. ezpz's
``timings/*`` go to ``tracker.log()`` and never reach the JSONL, so they
cannot be recovered offline either. ``sum(train/dt)`` is the honest
measure of time spent in measured steps, and is what this reports. It
still EXCLUDES setup (model build, dist init, dataset load), so it
under-reports true job wall time; pass ``--wall-time-sec`` with the
scheduler's figure when that matters.

Two unit traps, both verified against the source rather than assumed:

* ``train/mfu`` is a **percentage 0-100**, not a fraction --
  ``ezpz.flops.compute_mfu`` returns ``(...) * 100.0``. The manifest
  must say ``unit: "%"`` or every chart is off by 100x.
* ``train/mfu`` and ``train/tflops`` are **per-GPU** (``flops_per_gpu =
  _model_flops / args.tp``), while ``train/tps`` is **global across all
  ranks**. That mix is deliberate -- it matches how both ezpz and
  torchtitan report -- but a row mixing them needs the column names to
  say so, which is why the per-GPU throughput column is named
  explicitly.
"""

from __future__ import annotations

import csv
import io
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

__all__ = [
    "AMSC_REQUIRED_COLUMNS",
    "AMSC_COLUMNS",
    "AmscExportError",
    "DEFAULT_WARMUP",
    "REDUCERS",
    "format_csv",
    "load_run_metrics",
    "load_run_provenance",
    "summarize_run",
    "to_amsc_row",
]


class AmscExportError(RuntimeError):
    """Raised when a run cannot be exported (missing data, bad inputs)."""


#: Columns every emitted row carries. Originally the set
#: ``build_dashboard.py`` required; see the module warning -- that
#: script is no longer in the AmSC repo, so this is now ezpz's own.
AMSC_REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "system",
    "config",
    "nodes",
    "gpus",
    "status",
    "wall_time_sec",
)

#: Full column order we emit. The metric names follow the convention the
#: published manifests already use (``throughput_<unit>_per_sec``, as in
#: vllm-bench and mldocking); ``error`` is last in every published
#: ``runs.csv``.
AMSC_COLUMNS: tuple[str, ...] = AMSC_REQUIRED_COLUMNS + (
    "throughput_tokens_per_sec",
    "throughput_tokens_per_sec_per_gpu",
    "mfu",
    "tflops",
    "final_loss",
    "steps",
    "error",
)

DEFAULT_WARMUP: int = 1

REDUCERS: dict[str, Callable[[Sequence[float]], float]] = {
    "median": statistics.median,
    "mean": statistics.fmean,
    "max": max,
    "min": min,
    "last": lambda xs: xs[-1],
}

# ezpz metric key -> AmSC column.
_METRIC_MAP: dict[str, str] = {
    "train/tps": "throughput_tokens_per_sec",
    "train/tps_per_gpu": "throughput_tokens_per_sec_per_gpu",
    "train/mfu": "mfu",
    "train/tflops": "tflops",
}


def _looks_like_metrics(path: Path, *, probe_lines: int = 5) -> bool:
    """Does this JSONL hold metric records rather than log lines?

    ``finalize`` symlinks the structured LOG into the run directory
    alongside the metrics, and both are ``*.jsonl``. Log records have
    ``message``/``levelname`` and no ``metrics`` key, so a few lines are
    enough to tell them apart without reading the whole file.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            for _, line in zip(range(probe_lines), handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and isinstance(
                    payload.get("metrics"), dict
                ):
                    return True
    except OSError:
        return False
    return False


def load_run_metrics(run_dir: Path | str) -> list[dict[str, Any]]:
    """Read every per-step metrics record under ``run_dir``.

    Handles both layouts ezpz produces: ``metrics-<rank>.jsonl`` (one
    file per rank -- fsdp_tp, fsdp, diffusion) and a single
    ``metrics.jsonl`` (vit, test, minimal, hf).

    Returns records sorted by ``train/iter`` then timestamp, each a dict
    with ``timestamp``, ``rank`` and a flattened ``metrics`` mapping.
    Records carrying no numeric metrics are dropped -- notably the
    ``precision/*`` record ``fsdp_tp`` writes directly through
    ``History._write_jsonl_entry``, whose values are strings.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise AmscExportError(f"not a directory: {root}")

    # Three naming schemes are in play: `metrics-<rank>.jsonl` (fsdp_tp,
    # fsdp, diffusion), `metrics.jsonl` (vit, test, minimal, hf), and
    # `<run_id>.jsonl` -- what History writes when the caller does not
    # name the file (history.py:419). Missing the last one made the
    # exporter fail outright on any default History run.
    files = sorted(root.rglob("metrics-*.jsonl")) or sorted(
        root.rglob("metrics.jsonl")
    )
    if not files:
        # Fall back to any JSONL that actually carries metric records,
        # excluding the structured LOG (which has "message"/"levelname"
        # keys and no "metrics") that finalize symlinks into the run dir.
        files = [
            p
            for p in sorted(root.rglob("*.jsonl"))
            if _looks_like_metrics(p)
        ]
    if not files:
        raise AmscExportError(
            f"no metrics JSONL found under {root}. Expected "
            "metrics-<rank>.jsonl, metrics.jsonl, or a <run_id>.jsonl "
            "containing metric records -- is this an ezpz run directory?"
        )

    records: list[dict[str, Any]] = []
    for path in files:
        # Stream: a long run's JSONL is large, and reading it whole just
        # to split it doubles peak RSS for no benefit.
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    # A killed run leaves a truncated final line.
                    continue
                metrics = payload.get("metrics")
                if not isinstance(metrics, dict):
                    continue
                numeric = {
                    k: v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                    and not isinstance(v, bool)
                }
                if not numeric:
                    continue  # e.g. the all-strings precision/* record
                records.append(
                    {
                        "timestamp": payload.get("timestamp"),
                        "rank": payload.get("rank", 0),
                        "metrics": numeric,
                    }
                )

    if not records:
        raise AmscExportError(f"no numeric metric records under {root}")
    records.sort(
        key=lambda r: (
            r["metrics"].get("train/iter", 0),
            r["timestamp"] or 0,
        )
    )
    return records


def _provenance_from_report(root: Path) -> dict[str, Any]:
    """Scrape the ``### Distributed`` bullets out of ``report-*.md``.

    The fallback that makes this usable on runs that already exist.
    ``run_info.json`` is new, and ``config.json`` only appears under the
    non-default ``csv`` tracker backend -- but ``History.finalize`` has
    always rendered ``get_dist_info()`` into the report as::

        ### Distributed

        - **MACHINE**: SunSpot
        - **NUM_NODES**: 1
        - **NGPUS**: 12

    Parsing prose is not how this should work long-term, which is why
    ``run_info.json`` now exists and is preferred; this tier keeps the
    exporter useful for everything already on disk.
    """
    for report in sorted(root.rglob("report*.md")):
        try:
            text = report.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "### Distributed" not in text:
            continue
        # Take the section, stopping at the next heading of any level.
        section = text.split("### Distributed", 1)[1].split("\n#", 1)[0]
        found: dict[str, Any] = {}
        for line in section.splitlines():
            line = line.strip()
            if not line.startswith("- **") or "**:" not in line:
                continue
            key, _, value = line[4:].partition("**:")
            found[key.strip()] = value.strip()
        if found:
            return found
    return {}


def load_run_provenance(run_dir: Path | str) -> dict[str, Any]:
    """Recover ``get_dist_info()`` for a finished run, best source first.

    1. ``run_info.json`` -- written by ``History.finalize`` (structured,
       preferred, present on runs from this version onward).
    2. ``config.json`` -- only written when the non-default ``csv``
       tracker backend is active, but structured when it is there.
    3. ``report*.md`` -- the ``### Distributed`` markdown section, which
       covers the runs that predate the other two.

    Returns ``{}`` when none is available; callers must then require
    explicit ``--system``/``--nodes``/``--gpus`` rather than guess.
    """
    root = Path(run_dir)

    for candidate in sorted(root.rglob("run_info.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        dist = payload.get("Distributed")
        if isinstance(dist, dict) and dist:
            return dict(dist)

    for candidate in sorted(root.rglob("config.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "MACHINE" in payload:
            return {
                k: v for k, v in payload.items() if not isinstance(v, dict)
            }

    return _provenance_from_report(root)


def _series(records: Iterable[dict[str, Any]], key: str) -> list[float]:
    return [
        float(r["metrics"][key]) for r in records if key in r["metrics"]
    ]


def summarize_run(
    records: Sequence[dict[str, Any]],
    *,
    warmup: int = DEFAULT_WARMUP,
    reducer: str = "median",
) -> dict[str, Any]:
    """Reduce per-step records to the run-level numbers AmSC wants.

    Args:
        records: output of :func:`load_run_metrics`.
        warmup: leading steps to drop before reducing throughput. Step 1
            is reliably an outlier by an order of magnitude or more.
        reducer: one of :data:`REDUCERS`.

    Rank handling: prefer rank 0 when it logged anything, else fall back
    to every rank. Under ezpz every rank writes its own metrics file with
    near-identical values, so mixing them would inflate the sample count
    without adding information.

    ``wall_time_sec`` is ``sum(train/dt)`` over ALL steps including
    warmup -- that time was really spent. Only the throughput reduction
    trims warmup.
    """
    if reducer not in REDUCERS:
        raise AmscExportError(
            f"unknown reducer {reducer!r}; choose from "
            f"{sorted(REDUCERS)}"
        )
    if warmup < 0:
        raise AmscExportError(f"warmup must be >= 0, got {warmup}")
    if not records:
        raise AmscExportError("no records to summarize")

    rank0 = [r for r in records if r.get("rank") == 0]
    chosen = rank0 or list(records)

    reduce = REDUCERS[reducer]
    summary: dict[str, Any] = {"steps": len(chosen)}

    # Wall time over every step, warmup included.
    dts = _series(chosen, "train/dt")
    summary["wall_time_sec"] = round(sum(dts), 3) if dts else None

    trimmed = chosen[warmup:] if len(chosen) > warmup else chosen
    if not trimmed:  # pragma: no cover - guarded by the len() check
        trimmed = chosen
    summary["steps_used"] = len(trimmed)
    summary["warmup_dropped"] = len(chosen) - len(trimmed)

    for metric_key, column in _METRIC_MAP.items():
        values = _series(trimmed, metric_key)
        # Absent (e.g. tflops/mfu when the model FLOPs are unknown) must
        # stay None -> an empty CSV cell, never 0.0, which would read as
        # a measured zero.
        summary[column] = round(reduce(values), 3) if values else None

    losses = _series(chosen, "train/loss")
    summary["final_loss"] = round(losses[-1], 6) if losses else None

    stamps = [r["timestamp"] for r in chosen if r.get("timestamp")]
    summary["started_at"] = min(stamps) if stamps else None
    return summary


def _iso(epoch: Optional[float]) -> str:
    if epoch is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def to_amsc_row(
    summary: dict[str, Any],
    *,
    config: str,
    system: Optional[str] = None,
    nodes: Optional[int] = None,
    gpus: Optional[int] = None,
    provenance: Optional[dict[str, Any]] = None,
    status: str = "pass",
    error: str = "",
    timestamp: Optional[str] = None,
) -> dict[str, Any]:
    """Build one schema-valid row.

    Explicit arguments win over ``provenance``; anything still missing
    raises rather than being guessed. In particular ``gpus`` is NOT
    inferred from the number of rank files: those count ranks, which
    equals GPUs only at one rank per GPU, and a silently wrong GPU count
    corrupts every per-GPU comparison on the dashboard.
    """
    prov = provenance or {}

    def _prov_int(key: str) -> Optional[int]:
        raw = prov.get(key)
        try:
            return int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    system = system or (str(prov["MACHINE"]) if prov.get("MACHINE") else None)

    # Report what the RUN used, not what the JOB was allocated.
    #
    # `NUM_NODES`/`NGPUS` come from the scheduler's allocation, so a
    # 4-node job that runs a 1-node configuration reports NUM_NODES=4,
    # NGPUS=48 while WORLD_SIZE_IN_USE=12. Publishing the allocation
    # would make a 1-node result look like a catastrophically bad
    # 4-node one -- observed exactly this on Sunspot job 12473020,
    # where a scaling sweep ran 1/2/4 nodes inside one 4-node
    # allocation.
    #
    # So derive from the ranks actually participating, and only fall
    # back to the allocation when that is unavailable.
    # `WORLD_SIZE_IN_USE` is MPI-only (`get_world_size_in_use` calls
    # `MPI.COMM_WORLD.Get_size()` and falls back to 1), so under
    # torchrun -- where MPI may be absent or every rank is a singleton
    # COMM_WORLD -- it reads 1 while the lowercase `world_size` field
    # correctly reflects $WORLD_SIZE. Take the larger: both describe the
    # same quantity, and the failure mode of each is to under-report.
    in_use = max(
        _prov_int("WORLD_SIZE_IN_USE") or 0,
        _prov_int("world_size") or 0,
    ) or None
    per_node = _prov_int("GPUS_PER_NODE")
    if gpus is None:
        gpus = in_use if in_use is not None else _prov_int("NGPUS")
    if nodes is None:
        if in_use is not None and per_node:
            # ceil: a partially-filled last node is still a node.
            nodes = -(-in_use // per_node)
        else:
            nodes = _prov_int("NUM_NODES")

    missing = [
        name
        for name, value in (
            ("system", system),
            ("nodes", nodes),
            ("gpus", gpus),
        )
        if value in (None, "")
    ]
    if missing:
        flags = " ".join(f"--{m}" for m in missing)
        raise AmscExportError(
            f"cannot determine {', '.join(missing)} for this run. "
            f"run_info.json is absent or incomplete (runs from before it "
            f"existed, or a non-rank-0 directory), so pass {flags} "
            f"explicitly."
        )
    if not config:
        raise AmscExportError("config is required (e.g. agpt-2b/bs1/seq2048)")

    row: dict[str, Any] = {c: "" for c in AMSC_COLUMNS}
    row.update(
        {
            "timestamp": timestamp or _iso(summary.get("started_at")),
            "system": system,
            "config": config,
            "nodes": nodes,
            "gpus": gpus,
            "status": status,
            "error": error,
        }
    )
    for column in (
        "wall_time_sec",
        "throughput_tokens_per_sec",
        "throughput_tokens_per_sec_per_gpu",
        "mfu",
        "tflops",
        "final_loss",
        "steps",
    ):
        value = summary.get(column)
        row[column] = "" if value is None else value
    return row


def format_csv(
    rows: Sequence[dict[str, Any]], *, header: bool = True
) -> str:
    """Render rows in :data:`AMSC_COLUMNS` order."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=list(AMSC_COLUMNS), lineterminator="\n"
    )
    if header:
        writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in AMSC_COLUMNS})
    return buf.getvalue()
