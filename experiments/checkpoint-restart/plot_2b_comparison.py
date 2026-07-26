#!/usr/bin/env python3
"""agpt-2b sync-vs-async checkpoint-restart comparison (real Sunspot data).

Two panels, one figure:

  (left)  Training step vs elapsed time — three lines:
            * Baseline (no ckpt, no failures)  — green
            * Sync restart  (blocking DCP save) — red sawtooth
            * Async restart (/tmp stage → fan-out) — blue sawtooth
          Each failure is an ``x`` at the pre-kill step; the drop to the
          resume step is the lost work.

  (right) Per-step checkpoint stall distribution — the decisive contrast:
            * Sync:  train/ckpt_save_seconds  (blocks the training loop)
            * Async: train/ckpt_stage_seconds (only the CPU-stage blocks;
                     shard writes happen in a background thread)
          Shown as a strip + median line so the ~10x gap is legible even
          with few points.

Inputs are the per-rank metrics JSONL fsdp_tp writes. Pass a GLOB per phase
(one or many attempt files); files are merged in timestamp order so the
per-attempt segments line up into one continuous sawtooth.

Usage:
    plot_2b_comparison.py \
        --baseline 'expt_<jid>/baseline/*/metrics-0.jsonl' \
        --sync     'expt_<jid>/sync/*/metrics-0.jsonl' \
        --async    'expt_<jid>/async/*/metrics-0.jsonl' \
        --out agpt2b_restart.png --report agpt2b_restart.md
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import statistics
import sys


def _load(paths):
    rows = []
    for path in paths:
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
                        "restart_seconds": m.get("train/restart_seconds"),
                        "save_seconds": m.get("train/ckpt_save_seconds"),
                        "stage_seconds": m.get("train/ckpt_stage_seconds"),
                        "drain_seconds": m.get("train/ckpt_drain_seconds"),
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


def _stalls(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return [float(v) for v in vals]


def _expand(patterns):
    """Expand a list of globs/paths into a flat sorted file list."""
    out = []
    for p in patterns:
        hits = _glob.glob(p)
        out.extend(hits if hits else [p])
    return sorted(set(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--sync", nargs="+", required=True)
    ap.add_argument("--async", nargs="+", dest="async_", required=True)
    ap.add_argument("--out", default="agpt2b_restart.png")
    ap.add_argument("--report", default="agpt2b_restart.md")
    ap.add_argument(
        "--title",
        default="agpt-2b (23 GB checkpoint) — sync vs async checkpoint "
        "restart (real Sunspot / XPU)",
    )
    args = ap.parse_args(argv)

    base = _rebase(_load(_expand(args.baseline)))
    sync = _rebase(_load(_expand(args.sync)))
    asyn = _rebase(_load(_expand(args.async_)))
    sync_ev = _detect_restarts(sync)
    asyn_ev = _detect_restarts(asyn)
    sync_save = _stalls(sync, "save_seconds")
    asyn_stage = _stalls(asyn, "stage_seconds")
    asyn_drain = _stalls(asyn, "drain_seconds")
    # The HONEST async per-save stall is stage (CPU copy, blocks the save step)
    # + drain (blocking /tmp -> shared FS fan-out, blocks the NEXT step). Both
    # land on the training thread; only their sum is comparable to a sync save.
    # stage and drain are logged one step apart, so pair them by count.
    _n = min(len(asyn_stage), len(asyn_drain))
    asyn_total = [asyn_stage[i] + asyn_drain[i] for i in range(_n)]

    # ---- report (always; works without matplotlib) ----
    def _med(xs):
        return statistics.median(xs) if xs else float("nan")

    lines = ["# agpt-2b checkpoint-restart: sync vs async (real Sunspot)", ""]
    if base:
        lines.append(
            f"**Baseline:** {base[-1]['iter']} steps in "
            f"{base[-1]['elapsed_min']:.2f} min."
        )
    if sync:
        lines.append(
            f"**Sync restart:** reached step {sync[-1]['iter']} in "
            f"{sync[-1]['elapsed_min']:.2f} min across {len(sync_ev)} restart(s)."
        )
    if asyn:
        lines.append(
            f"**Async restart:** reached step {asyn[-1]['iter']} in "
            f"{asyn[-1]['elapsed_min']:.2f} min across {len(asyn_ev)} restart(s)."
        )
    if sync_save and asyn_total:
        _sync_m = _med(sync_save)
        _stage_m = _med(asyn_stage)
        _drain_m = _med(asyn_drain)
        _tot_m = _med(asyn_total)
        lines += [
            "",
            "**Per-save training-thread stall (median):**",
            "",
            f"- sync `ckpt_save_seconds` = **{_sync_m:.3f}s** "
            f"(n={len(sync_save)}) — blocking write of the full checkpoint.",
            f"- async `ckpt_stage_seconds` = **{_stage_m:.3f}s** "
            f"(n={len(asyn_stage)}) — CPU stage only (the cheap half).",
            f"- async `ckpt_drain_seconds` = **{_drain_m:.3f}s** "
            f"(n={len(asyn_drain)}) — blocking /tmp→shared-FS fan-out at the "
            "next step (previously untimed).",
            f"- async TRUE total (stage+drain) = **{_tot_m:.3f}s** — "
            + (
                f"{(_tot_m / _sync_m):.2f}x the sync stall "
                "(async is SLOWER here)."
                if _tot_m > _sync_m
                else f"{(_sync_m / _tot_m):.2f}x less than sync."
            ),
            "",
            "| phase | # | resume@step | lost steps | restart_seconds |",
            "|-------|---|-------------|-----------|-----------------|",
        ]
    else:
        lines += [
            "",
            "| phase | # | resume@step | lost steps | restart_seconds |",
            "|-------|---|-------------|-----------|-----------------|",
        ]
    for tag, evs, rows in (("sync", sync_ev, sync), ("async", asyn_ev, asyn)):
        for k, e in enumerate(evs, 1):
            rs = (
                f"{e['restart_seconds']:.2f}"
                if e["restart_seconds"] is not None
                else "n/a"
            )
            lines.append(
                f"| {tag} | {k} | {e['resume_iter']} | {e['lost_steps']} | {rs} |"
            )
    report = "\n".join(lines) + "\n"
    with open(args.report, "w") as f:
        f.write(report)
    print(report)

    # ---- plot (optional) ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot skipped: matplotlib unavailable: {exc}]", file=sys.stderr)
        return 0

    try:
        import plot_style

        plot_style.apply_style()
    except Exception as exc:  # noqa: BLE001
        print(f"[plot_style not applied: {exc}]", file=sys.stderr)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2.4, 1]}
    )

    # --- left: step-vs-time sawtooth ---
    if base:
        ax.plot(
            [r["elapsed_min"] for r in base], [r["iter"] for r in base],
            "-", color="tab:green", label="Baseline (no events)", linewidth=2,
        )
    for rows, evs, color, label in (
        (sync, sync_ev, "tab:red", "Sync restart (blocking save)"),
        (asyn, asyn_ev, "tab:blue", "Async restart (/tmp → fan-out)"),
    ):
        if not rows:
            continue
        ax.plot(
            [r["elapsed_min"] for r in rows], [r["iter"] for r in rows],
            "-o", color=color, label=label, markersize=2, linewidth=1.2,
        )
        for e in evs:
            r = rows[e["i"]]
            ax.scatter(
                [r["elapsed_min"]], [e["prev_iter"]], marker="x",
                color=color, zorder=5, s=55,
            )
    ax.set_xlabel("Elapsed time (minutes from run start)")
    ax.set_ylabel("Training step")
    ax.set_title("Training progress over time")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # --- right: HONEST per-save training-thread stall ---
    # sync = one blocking write; async = stage (cheap) + drain (expensive
    # fan-out). A stacked async bar next to the sync bar shows the true total,
    # with the raw points overlaid so the distribution is visible.
    def _strip_points(xpos, vals, color, jitter=0.10):
        import math

        xs = [xpos + jitter * math.sin(2.399963 * k) for k in range(len(vals))]
        ax2.scatter(xs, vals, color=color, s=16, alpha=0.55, zorder=5,
                    edgecolors="none")

    sync_m = statistics.median(sync_save) if sync_save else 0.0
    stage_m = statistics.median(asyn_stage) if asyn_stage else 0.0
    drain_m = statistics.median(asyn_drain) if asyn_drain else 0.0

    # sync bar (single segment) at x=0; async stacked bar at x=1.
    ax2.bar(0, sync_m, width=0.55, color="tab:red", alpha=0.85, zorder=2,
            label="sync save")
    ax2.bar(1, stage_m, width=0.55, color="tab:blue", alpha=0.85, zorder=2,
            label="async stage (CPU)")
    ax2.bar(1, drain_m, width=0.55, bottom=stage_m, color="tab:orange",
            alpha=0.9, zorder=2, label="async drain (fan-out)")
    _strip_points(0, sync_save, "darkred")
    # overlay async per-save totals (stage+drain) as points on the async bar
    _strip_points(1, asyn_total, "black")

    ax2.annotate(f"{sync_m:.2f}s", (0, sync_m), fontsize=9, ha="center",
                 va="bottom", xytext=(0, 2), textcoords="offset points")
    ax2.annotate(f"{stage_m + drain_m:.2f}s total\n({stage_m:.2f}+{drain_m:.2f})",
                 (1, stage_m + drain_m), fontsize=9, ha="center", va="bottom",
                 xytext=(0, 2), textcoords="offset points")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["sync\n(save)", "async\n(stage+drain)"])
    ax2.set_ylabel("Per-save training-thread stall (s)")
    ax2.set_title("True checkpoint stall per save")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_xlim(-0.6, 1.6)

    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=130)
    print(f"[plot written: {args.out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
