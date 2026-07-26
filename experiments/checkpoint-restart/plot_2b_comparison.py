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
    lines += [
        "",
        f"**Per-step save stall (median):** sync "
        f"`ckpt_save_seconds`={_med(sync_save):.3f}s (n={len(sync_save)}) "
        f"vs async `ckpt_stage_seconds`={_med(asyn_stage):.3f}s "
        f"(n={len(asyn_stage)}) — "
        f"{(_med(sync_save) / _med(asyn_stage)):.1f}x less stall."
        if sync_save and asyn_stage
        else "",
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

    # --- right: per-step save-stall strip ---
    def _strip(xpos, vals, color, jitter=0.12):
        # deterministic pseudo-jitter (no RNG) so points don't overlap
        import math

        xs = [
            xpos + jitter * math.sin(2.399963 * k) for k in range(len(vals))
        ]
        ax2.scatter(xs, vals, color=color, s=28, alpha=0.75, zorder=3)
        if vals:
            med = statistics.median(vals)
            ax2.hlines(
                med, xpos - 0.28, xpos + 0.28, color=color, linewidth=2.5,
                zorder=4,
            )
            ax2.annotate(
                f"median\n{med:.3f}s", (xpos, med), fontsize=8, color=color,
                xytext=(10, 0), textcoords="offset points", va="center",
            )

    _strip(0, sync_save, "tab:red")
    _strip(1, asyn_stage, "tab:blue")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["sync\nsave_seconds", "async\nstage_seconds"])
    ax2.set_ylabel("Per-step checkpoint stall (s)")
    ax2.set_title("Checkpoint stall per save")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_xlim(-0.6, 1.6)

    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=130)
    print(f"[plot written: {args.out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
