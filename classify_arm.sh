#!/usr/bin/env bash
# Classify one fsdp_tp arm: STALLED vs HEALTHY vs NO-PROGRESS.
#
# Three classifier bugs in a row led here, so the reasoning is written
# down rather than re-derived:
#
#  1. Exit code is useless: mpiexec teardown makes a crash and a deadlock
#     both rc=143, and a wallclock kill is rc=124 either way.
#  2. Iteration COUNT is useless: a run at 2.6 it/s reaching iter=88 is
#     healthy if it is still going, and dead if it stopped 6 min ago.
#  3. File MTIME is useless: untimestamped CCL warnings and the SIGTERM
#     notice keep touching the file long after real progress stops. P3
#     had mtime 15:35 but its last real log line was 10:30 -- the mtime
#     test passed it as "healthy" when it had died in the first forward.
#
# What actually works: the last TIMESTAMPED line inside the log, compared
# against when the arm was killed. Plus an explicit no-progress case,
# because "never reached iteration 1" is a distinct failure from
# "progressed then stopped".
set -u
f="${1:?usage: classify_arm.sh <logfile> <end_epoch>}"
end_epoch="${2:?}"
stall_after="${3:-60}"

S=$(sed 's/\x1b\[[0-9;]*m//g' "$f")
iters=$(printf '%s' "$S" | grep -coE 'iter=[0-9]+')
last_iter=$(printf '%s' "$S" | grep -oE 'iter=[0-9]+' | tail -1)

# Last bracketed ezpz timestamp, e.g. [2026-09-25 10:30:54]
last_ts=$(printf '%s' "$S" | grep -oE '^\[20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\]' \
    | tail -1 | tr -d '[]')
if [ -z "$last_ts" ]; then
    echo "NO-OUTPUT (no timestamped lines at all)"
    exit 0
fi
last_epoch=$(date -d "$last_ts" +%s 2>/dev/null || echo 0)
quiet=$(( end_epoch - last_epoch ))

if [ "$iters" -eq 0 ]; then
    echo "NO-PROGRESS (0 iterations; last output ${quiet}s before kill)"
elif [ "$quiet" -gt "$stall_after" ]; then
    echo "STALLED (quiet ${quiet}s before kill, reached ${last_iter})"
else
    echo "HEALTHY (still logging at kill, reached ${last_iter})"
fi
