#!/usr/bin/env python3
"""Build the checkpoint-restart plot + report from real ezpz metrics JSONL.

Reproduces the ezpz-genuine lines of the reference figure:
  * Baseline (no events)     -- green straight line
  * Checkpoint Restart       -- red sawtooth (fail -> lose steps -> resume)

Inputs are the per-rank metrics JSONL that fsdp_tp writes (rank-0 file is
enough: every row has top-level `timestamp` + metrics.{train/iter,
train/tokens_seen, train/restart_seconds}).

Usage:
    plot_restart.py <baseline_metrics-0.jsonl> <restart_metrics-0.jsonl> \
        [--chaos chaos.log] [--out restart_plot.png] [--report report.md]

Restart detection: a resume shows up in the restart run as a row whose
train/iter DROPS relative to the previous row's iter (we jumped back to the
last checkpoint), OR a row carrying train/restart_seconds. Lost steps =
(iter just before failure) - (iter at resume).
"""
from __future__ import annotations

import argparse
import json
import sys


def _load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            m = r.get("metrics") or {}
            if "train/iter" not in m:
                continue
            rows.append(
                {
                    "t": float(r.get("timestamp", 0.0)),
                    "iter": int(m["train/iter"]),
                    "tokens_seen": int(m.get("train/tokens_seen", 0) or 0),
                    "restart_seconds": m.get("train/restart_seconds"),
                }
            )
    rows.sort(key=lambda x: x["t"])
    return rows


def _rebase(rows):
    if not rows:
        return rows
    t0 = rows[0]["t"]
    for r in rows:
        r["elapsed_min"] = (r["t"] - t0) / 60.0
    return rows


def _detect_restarts(rows):
    """Return list of (index, prev_iter, resume_iter, restart_seconds)."""
    events = []
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        dropped = cur["iter"] < prev["iter"]
        has_rs = cur["restart_seconds"] is not None
        if dropped or has_rs:
            events.append(
                {
                    "i": i,
                    "prev_iter": prev["iter"],
                    "resume_iter": cur["iter"],
                    "lost_steps": max(0, prev["iter"] - cur["iter"]),
                    "restart_seconds": cur["restart_seconds"],
                }
            )
    return events


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", help="baseline metrics-0.jsonl (single file)")
    ap.add_argument(
        "restart", nargs="+",
        help="restart metrics-0.jsonl file(s) — one per relaunch attempt; "
        "merged in timestamp order to reconstruct the sawtooth",
    )
    ap.add_argument("--out", default="restart_plot.png")
    ap.add_argument("--report", default="restart_report.md")
    args = ap.parse_args(argv)

    base = _rebase(_load(args.baseline))
    # Merge all restart-attempt files, then sort by wall-clock timestamp so the
    # per-attempt segments line up into one continuous step-vs-time trace.
    merged = []
    for p in args.restart:
        merged.extend(_load(p))
    merged.sort(key=lambda x: x["t"])
    rst = _rebase(merged)
    events = _detect_restarts(rst)

    # ---- report (always; works without matplotlib) ----
    lines = ["# Checkpoint-restart experiment (real Sunspot measurements)", ""]
    if base:
        lines.append(
            f"**Baseline:** {base[-1]['iter']} steps in "
            f"{base[-1]['elapsed_min']:.2f} min "
            f"({base[-1]['elapsed_min'] * 60 / max(1, base[-1]['iter']):.2f} s/step)."
        )
    if rst:
        lines.append(
            f"**Checkpoint Restart:** reached step {rst[-1]['iter']} in "
            f"{rst[-1]['elapsed_min']:.2f} min across {len(events)} restart(s)."
        )
    lines.append("")
    lines.append("| # | resume@step | lost steps | restart_seconds |")
    lines.append("|---|-------------|-----------|-----------------|")
    for k, e in enumerate(events, 1):
        rs = (
            f"{e['restart_seconds']:.2f}"
            if e["restart_seconds"] is not None
            else "n/a"
        )
        lines.append(
            f"| {k} | {e['resume_iter']} | {e['lost_steps']} | {rs} |"
        )
    report = "\n".join(lines) + "\n"
    with open(args.report, "w") as f:
        f.write(report)
    print(report)

    # ---- plot (optional; skip cleanly if matplotlib missing) ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot skipped: matplotlib unavailable: {exc}]", file=sys.stderr)
        return 0

    fig, ax = plt.subplots(figsize=(11, 6))
    if base:
        ax.plot(
            [r["elapsed_min"] for r in base],
            [r["iter"] for r in base],
            "-", color="tab:green", label="Baseline (no events)", linewidth=2,
        )
    if rst:
        ax.plot(
            [r["elapsed_min"] for r in rst],
            [r["iter"] for r in rst],
            "-o", color="tab:red", label="Checkpoint Restart", markersize=2,
            linewidth=1.2,
        )
    for e in events:
        r = rst[e["i"]]
        ax.scatter(
            [r["elapsed_min"]], [e["prev_iter"]], marker="x", color="black",
            zorder=5, s=60,
        )
        rs = e["restart_seconds"]
        label = f"lost {e['lost_steps']} steps"
        if rs is not None:
            label += f"\n{rs:.1f}s restart"
        ax.annotate(
            label, (r["elapsed_min"], e["resume_iter"]),
            fontsize=8, color="tab:red",
            xytext=(6, -18), textcoords="offset points",
        )
    ax.set_xlabel("Elapsed time (minutes from run start)")
    ax.set_ylabel("Training step")
    ax.set_title(
        "Training progress over time — Baseline vs Checkpoint Restart "
        "(real Sunspot / XPU, injected failures)"
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"[plot written: {args.out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
