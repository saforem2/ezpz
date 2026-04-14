"""report.py — Parse benchmark outputs and produce a markdown report.

Reads the artifacts produced by ``run_all.py`` (or ``run_benchmarks.sh``):

- ``env.json``     — environment metadata
- ``timings.csv``  — per-example wall-clock time and exit code
- ``{name}.log``   — stdout/stderr captured from each example

For each example it locates the JSONL metrics file (by grepping the log for the
"Outputs will be saved to" message), extracts summary statistics, and writes a
combined markdown report to ``{outdir}/report.md``.

Usage::

    python3 -m ezpz.examples.report --outdir outputs/benchmarks/2026-03-16-103000
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


# Maps example names to their output directory names under outputs/.
_EXAMPLE_OUTPUT_DIRS: dict[str, str] = {
    "test": "ezpz.examples.test",
    "fsdp": "ezpz.examples.fsdp",
    "vit": "ezpz.examples.vit",
    "fsdp_tp": "ezpz.examples.fsdp_tp",
    "diffusion": "ezpz.examples.diffusion",
    "minimal": "ezpz.examples.minimal",
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    return _ANSI_RE.sub("", text)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _find_outdir_in_log(logfile: Path) -> Optional[Path]:
    """Extract the example output directory from a log file.

    Looks for lines matching ``Outputs will be saved to <path>``.
    Strips ANSI escape codes before matching to handle Rich-formatted output.
    """
    pattern = re.compile(r"Outputs will be saved to\s+(.+)")
    try:
        text = logfile.read_text(errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        clean = _strip_ansi(line)
        m = pattern.search(clean)
        if m:
            candidate = Path(m.group(1).strip())
            if candidate.is_dir():
                return candidate
    return None


def _find_outdir_by_name(name: str) -> Optional[Path]:
    """Fallback: search ``outputs/`` for the most recent run of *name*."""
    dir_name = _EXAMPLE_OUTPUT_DIRS.get(name)
    if dir_name is None:
        return None
    outputs_dir = Path.cwd() / "outputs" / dir_name
    if not outputs_dir.is_dir():
        return None
    # Pick the most recently modified subdirectory (each run is timestamped).
    subdirs = sorted(
        (d for d in outputs_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    return subdirs[-1] if subdirs else None


def _find_jsonl(outdir: Path) -> Optional[Path]:
    """Find the most recently modified ``.jsonl`` file under *outdir*.

    Prefers files named ``metrics*.jsonl`` over logging JSONL files.
    """
    candidates = list(outdir.rglob("*.jsonl"))
    if not candidates:
        return None
    # Prefer metrics files over logging JSONL.
    metrics = [c for c in candidates if c.stem.startswith("metrics")]
    pool = metrics if metrics else candidates
    pool.sort(key=lambda p: p.stat().st_mtime)
    return pool[-1]


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


def _find_mlflow_url(logfile: Path) -> Optional[str]:
    """Extract an MLflow run URL from a log file."""
    pattern = re.compile(r"View run at\s+(https?://\S+)")
    try:
        text = _strip_ansi(logfile.read_text(errors="replace"))
    except OSError:
        return None
    m = pattern.search(text)
    return m.group(1) if m else None


_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _display_len(cell: str) -> int:
    """Return the rendered width of *cell*, collapsing markdown links."""
    return len(_MD_LINK_RE.sub(lambda m: m.group(1), cell))


def _align_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Build a markdown table with columns padded to equal widths.

    Column widths are computed from the *display* length of each cell so
    that markdown links (``[text](url)``) are measured by their visible
    text, not the full URL.
    """
    ncols = len(headers)
    widths = [_display_len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _display_len(cell))

    def _pad(cell: str, width: int) -> str:
        return cell + " " * (width - _display_len(cell))

    def _row_line(cells: list[str]) -> str:
        padded = [_pad(cells[i], widths[i]) for i in range(ncols)]
        return "| " + " | ".join(padded) + " |"

    lines = [_row_line(headers)]
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        lines.append(_row_line(row))
    return lines


def _fmt(val: Optional[float], precision: int = 4) -> str:
    if val is None:
        return "\u2014"
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
        f"{env.get('num_nodes', '?')} \u00d7 {env.get('gpus_per_node', '?')} GPUs"
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
        "Final Loss", "Mean dt (s)", "Throughput", "W&B", "MLflow",
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
        wandb_cell = f"[link]({wandb_url})" if wandb_url else "\u2014"
        mlflow_url = _find_mlflow_url(logfile)
        mlflow_cell = f"[link]({mlflow_url})" if mlflow_url else "\u2014"

        # Attempt to locate and parse metrics.
        # Primary: extract output dir from log.  Fallback: search outputs/.
        example_outdir = _find_outdir_in_log(logfile)
        if example_outdir is None:
            example_outdir = _find_outdir_by_name(name)
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

        final_loss = _fmt(losses[-1]) if losses else "\u2014"
        mean_dt = _fmt(statistics.mean(dts)) if dts else "\u2014"
        num_steps = str(len(entries)) if entries else "\u2014"
        tp_str = (
            f"{statistics.mean(throughputs):.0f} tok/s" if throughputs else "\u2014"
        )

        results_rows.append([
            name, status, wall_str, num_steps,
            final_loss, mean_dt, tp_str, wandb_cell, mlflow_cell,
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
