#!/usr/bin/env python3
"""generate_report.py — Parse benchmark outputs and produce a markdown report.

Reads the artifacts produced by ``run_benchmarks.sh``:

- ``env.json``     — environment metadata
- ``timings.csv``  — per-example wall-clock time and exit code
- ``{name}.log``   — stdout/stderr captured from each example

For each example it locates the JSONL metrics file (by grepping the log for the
"Outputs will be saved to" message), extracts summary statistics, and writes a
combined markdown report to ``{outdir}/report.md``.

Usage::

    python3 scripts/generate_report.py --outdir outputs/benchmarks/2026-03-16-103000
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Optional

# ── Metric key mapping per example ───────────────────────────────────────────
# Each example logs slightly different key names in its history.  We map
# canonical fields to the keys each example actually records.
METRIC_KEYS: dict[str, dict[str, str | tuple[str, ...]]] = {
    "test": {
        "loss": "loss",
        "dt": ("dtf", "dtb"),  # sum of dtf + dtb
    },
    "minimal": {
        "loss": "loss",
        "dt": "dt",
    },
    "fsdp": {
        "loss": "train_loss",
        "dt": "dt",
    },
    "vit": {
        "loss": "train/loss",
        "dt": "train/dt",
    },
    "fsdp_tp": {
        "loss": "train/loss",
        "dt": "train/dt",
    },
    "hf": {
        "loss": "loss",
        "dt": "dts",
        "throughput": "tokens_per_sec",
    },
    "hf_trainer": {
        "loss": "loss",
        "dt": "dts",
        "throughput": "tokens_per_sec",
    },
    "diffusion": {
        "loss": "train/loss",
        "dt": "train/dt",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _find_outdir_in_log(logfile: Path) -> Optional[Path]:
    """Extract the example output directory from a log file.

    Looks for lines matching ``Outputs will be saved to <path>``.
    """
    pattern = re.compile(r"Outputs will be saved to\s+(.+)")
    try:
        text = logfile.read_text(errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            candidate = Path(m.group(1).strip())
            if candidate.is_dir():
                return candidate
    return None


def _find_jsonl(outdir: Path) -> Optional[Path]:
    """Find the first ``.jsonl`` file under *outdir*."""
    candidates = sorted(outdir.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, returning a list of parsed metric dicts.

    Each line written by ``History._write_jsonl_entry`` has the structure::

        {"timestamp": ..., "rank": ..., "metrics": {...}, "aggregated": {...}}

    We extract the ``metrics`` dict from rank-0 entries only.
    """
    entries: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("rank", 0) != 0:
                continue
            metrics = obj.get("metrics", obj)
            entries.append(metrics)
    return entries


def _extract_metric(
    entries: list[dict[str, Any]],
    key: str | tuple[str, ...],
) -> list[float]:
    """Pull numeric values for *key* from every entry.

    If *key* is a tuple, the values are summed (e.g. ``dtf + dtb``).
    """
    vals: list[float] = []
    for entry in entries:
        if isinstance(key, tuple):
            parts = [entry.get(k) for k in key]
            if all(p is not None for p in parts):
                vals.append(sum(float(p) for p in parts))  # type: ignore[arg-type]
        else:
            v = entry.get(key)
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
    return vals


def _find_wandb_url(logfile: Path) -> Optional[str]:
    """Extract a wandb run URL from a log file."""
    pattern = re.compile(r"https://wandb\.ai/\S+")
    try:
        text = logfile.read_text(errors="replace")
    except OSError:
        return None
    urls = pattern.findall(text)
    # Prefer run-specific URL (contains /runs/)
    for url in urls:
        if "/runs/" in url:
            return url
    return urls[0] if urls else None


def _align_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Build a markdown table with columns padded to equal widths."""
    ncols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    def _row_line(cells: list[str]) -> str:
        padded = [cells[i].ljust(widths[i]) for i in range(ncols)]
        return "| " + " | ".join(padded) + " |"
    lines = [_row_line(headers)]
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        lines.append(_row_line(row))
    return lines


def _fmt(val: Optional[float], precision: int = 4) -> str:
    if val is None:
        return "—"
    return f"{val:.{precision}f}"


def _fmt_time(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    return f"{m}m{s:02d}s"


def generate_report(outdir: Path) -> str:
    """Build the markdown report string from benchmark artifacts."""
    # ── Load env + timings ───────────────────────────────────────────────
    env_path = outdir / "env.json"
    env: dict[str, Any] = _read_json(env_path) if env_path.exists() else {}

    timings_path = outdir / "timings.csv"
    timings = _read_csv(timings_path) if timings_path.exists() else []

    # ── Environment table ────────────────────────────────────────────────
    gpu_desc = (
        f"{env.get('num_nodes', '?')} × {env.get('gpus_per_node', '?')} GPUs"
        f" = {env.get('total_gpus', '?')} total"
    )
    git_desc = f"`{env.get('git_commit', '?')}` (branch: {env.get('git_branch', '?')})"

    env_headers = ["Key", "Value"]
    env_rows = [
        ["Date", env.get("date", "?")],
        ["Git Commit", git_desc],
        ["Job ID", f"{env.get('job_id', '?')} ({env.get('scheduler', '?')})"],
        ["Nodes", gpu_desc],
        ["Python", env.get("python", "?")],
        ["PyTorch", env.get("torch", "?")],
        ["ezpz", env.get("ezpz_version", "?")],
    ]

    lines: list[str] = [
        "# ezpz Benchmark Report",
        "",
        "## Environment",
        "",
        *_align_table(env_headers, env_rows),
        "",
    ]

    # ── Per-example results ──────────────────────────────────────────────
    results_headers = [
        "Example", "Status", "Wall Time", "Steps",
        "Final Loss", "Mean dt (s)", "Throughput", "W&B",
    ]
    results_rows: list[list[str]] = []

    for row in timings:
        name = row["name"]
        rc = int(row["exit_code"])
        wall = int(row["wall_seconds"])
        status = "\u2705" if rc == 0 else f"\u274c ({rc})"
        wall_str = _fmt_time(wall)

        logfile = outdir / f"{name}.log"
        wandb_url = _find_wandb_url(logfile)
        wandb_cell = f"[link]({wandb_url})" if wandb_url else "—"

        # Attempt to locate and parse metrics
        example_outdir = _find_outdir_in_log(logfile)
        jsonl_path = _find_jsonl(example_outdir) if example_outdir else None
        entries = _parse_jsonl(jsonl_path) if jsonl_path else []

        key_map = METRIC_KEYS.get(name, METRIC_KEYS.get("minimal", {}))
        loss_key = key_map.get("loss", "loss")
        dt_key = key_map.get("dt", "dt")
        tp_key = key_map.get("throughput")

        # Extract values
        losses = _extract_metric(entries, loss_key)  # type: ignore[arg-type]
        dts = _extract_metric(entries, dt_key)  # type: ignore[arg-type]
        throughputs = _extract_metric(entries, tp_key) if tp_key else []  # type: ignore[arg-type]

        final_loss = _fmt(losses[-1]) if losses else "—"
        mean_dt = _fmt(statistics.mean(dts)) if dts else "—"
        num_steps = str(len(entries)) if entries else "—"
        tp_str = (
            f"{statistics.mean(throughputs):.0f} tok/s" if throughputs else "—"
        )

        results_rows.append([
            name, status, wall_str, num_steps,
            final_loss, mean_dt, tp_str, wandb_cell,
        ])

    lines += ["## Results", "", *_align_table(results_headers, results_rows)]

    # ── Per-example log paths ────────────────────────────────────────────
    lines += [
        "",
        "## Log Files",
        "",
    ]
    for row in timings:
        name = row["name"]
        logfile = outdir / f"{name}.log"
        lines.append(f"- **{name}**: `{logfile}`")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a markdown benchmark report from run artifacts."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Path to the benchmark output directory.",
    )
    args = parser.parse_args()

    if not args.outdir.is_dir():
        parser.error(f"Output directory does not exist: {args.outdir}")

    report = generate_report(args.outdir)

    report_path = args.outdir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    # Also print to stdout so it's visible in the job log
    print(report)
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
