#!/usr/bin/env bash
# bench_trackers.sh — Compare tracker backend overhead across 3 runs each.
#
# Usage:
#   ./bench_trackers.sh
#   ./bench_trackers.sh 5          # 5 runs per backend
#   ./bench_trackers.sh 3 --model debug  # extra args forwarded to test

set -euo pipefail

NRUNS="${1:-3}"
shift 2>/dev/null || true
EXTRA_ARGS=("$@")

BACKENDS=("wandb" "mlflow" "wandb,mlflow")
RESULTS_FILE="bench_trackers_results.csv"

echo "backend,run,wall_seconds" > "$RESULTS_FILE"

for backend in "${BACKENDS[@]}"; do
    times=()
    for run in $(seq 1 "$NRUNS"); do
        echo "────── backend=$backend  run=$run/$NRUNS ──────"
        start=$(python3 -c "import time; print(time.perf_counter())")
        EZPZ_TRACKER_BACKENDS="$backend" \
            ezpz launch python3 -m ezpz.examples.test "${EXTRA_ARGS[@]}" \
            2>&1 | tail -3
        end=$(python3 -c "import time; print(time.perf_counter())")
        dt=$(python3 -c "print(f'{$end - $start:.3f}')")
        times+=("$dt")
        echo "$backend,$run,$dt" >> "$RESULTS_FILE"
        echo "  → ${dt}s"
        echo ""
    done

    # Compute mean/std inline
    IFS=','; times_csv="${times[*]}"; unset IFS
    python3 -c "
ts = [$times_csv]
n = len(ts)
mean = sum(ts) / n
std = (sum((t - mean)**2 for t in ts) / n) ** 0.5
print(f'  {\"$backend\":>16s}:  {mean:.2f} ± {std:.2f}s  (n={n}, raw={ts})')
"
    echo ""
done

echo "──────────────────── Summary ────────────────────"
python3 -c "
import csv
from collections import defaultdict

data = defaultdict(list)
with open('$RESULTS_FILE') as f:
    for row in csv.DictReader(f):
        data[row['backend']].append(float(row['wall_seconds']))

print(f'{\"Backend\":>16s}  {\"Mean\":>8s}  {\"Std\":>6s}  {\"Min\":>8s}  {\"Max\":>8s}  N')
print('-' * 60)
for backend, ts in data.items():
    n = len(ts)
    mean = sum(ts) / n
    std = (sum((t - mean)**2 for t in ts) / n) ** 0.5
    print(f'{backend:>16s}  {mean:8.2f}  {std:6.2f}  {min(ts):8.2f}  {max(ts):8.2f}  {n}')
"
echo ""
echo "Raw results saved to $RESULTS_FILE"
