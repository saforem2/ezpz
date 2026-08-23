#!/usr/bin/env python3
"""Chart a fault-injection run: progress, faults, and work lost to each.

    python3 experiments/fault-injection/run_faults.py --out run.jsonl
    python3 experiments/fault-injection/plot_faults.py run.jsonl -o out.png

Reuses `experiments/checkpoint-restart/plot_style.py` so this figure
matches the other charts in the docs.

**What is plotted, and what deliberately is not.** The x-axis is
*attempt*, not wall-clock. At this scale a whole attempt takes about a
second, and `_run_attempt_with_tee` only notices the child exited on its
next poll tick -- a flat 1.0s with the watchdog off, 0.1-1.0s with it on
-- so measured wall-time is dominated by poll granularity rather than by
the work. Charting it would put a number on the page that mostly
describes a sleep.

Step progression and lost work, by contrast, are read exactly out of the
logs. For real wall-clock recovery cost at scale, see
`docs/guides/checkpoint-restart.md`: 2 Sunspot nodes, a 23 GB
checkpoint, restart measured at 10.4-11.1 s per failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "checkpoint-restart")
)

import matplotlib.pyplot as plt  # noqa: E402

try:
    from plot_style import apply_style  # type: ignore
except Exception:  # pragma: no cover - style is cosmetic

    def apply_style() -> None:  # type: ignore[misc]
        return


FAULT_LABELS = {
    "shepherd_sig9": "shepherd died from signal 9",
    "gloo_peer": "gloo: connection closed by peer",
    "ur_oom": "UR_RESULT_ERROR_OUT_OF_RESOURCES",
    "rank_exit": "rank exited with code 1",
}


def load(path: Path) -> tuple[list[dict], dict]:
    attempts, summary = [], {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("kind") == "summary":
            summary = rec
        else:
            attempts.append(rec)
    return attempts, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("fault-injection.png"))
    args = ap.parse_args()

    attempts, summary = load(args.jsonl)
    if not attempts:
        print(f"no attempts in {args.jsonl}", file=sys.stderr)
        return 1

    apply_style()
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 6.5), height_ratios=[2.2, 1], sharex=True
    )

    # -- top: the step range each attempt covered ----------------------
    for a in attempts:
        n = a["attempt"]
        lo, hi = a["first_step"], a["last_step"]
        if lo is None:
            continue
        ax.plot([n, n], [lo, hi], lw=7, solid_capstyle="butt", alpha=0.85)
        ax.annotate(
            f"{lo}→{hi}",
            (n, hi),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=9,
        )
        if a.get("fault"):
            ax.plot(n, hi, marker="X", ms=13, color="crimson", zorder=5)
        if a.get("resumed_from") is not None:
            # The redone work: from the checkpoint back up to where the
            # previous attempt actually died.
            ax.plot(n, lo, marker="o", ms=8, mfc="none", mew=2, zorder=5)

    ax.set_ylabel("training step")
    ax.set_title(
        "--auto-retry under injected faults: each attempt resumes from the "
        "last checkpoint",
        fontsize=11,
    )

    # -- bottom: work lost to each fault -------------------------------
    lost_x, lost_y = [], []
    for prev, cur in zip(attempts, attempts[1:]):
        if cur.get("resumed_from") is None or prev.get("last_step") is None:
            continue
        lost_x.append(cur["attempt"])
        lost_y.append(prev["last_step"] - cur["resumed_from"])
    if lost_x:
        ax2.bar(lost_x, lost_y, width=0.35, color="crimson", alpha=0.75)
        for x, y in zip(lost_x, lost_y):
            ax2.annotate(
                f"{y} steps",
                (x, y),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=9,
            )
    ax2.set_xlabel("attempt")
    ax2.set_ylabel("steps redone")
    ax2.set_xticks([a["attempt"] for a in attempts])

    every = summary.get("ckpt_every")
    if every:
        ax2.axhline(
            every,
            ls="--",
            lw=1,
            color="0.5",
            label=f"checkpoint interval ({every} steps)",
        )
        ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
