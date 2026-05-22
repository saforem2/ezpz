#!/usr/bin/env bash
#
# Manual test driver for `ezpz launch --timeout` / `--retries`
# (PR #136: feat/launch-timeout-retries)
#
# Run inside an interactive PBS allocation (Sunspot / Aurora / Polaris).
# All scenarios use trivial `sleep`/`echo`/`exit` commands so no GPUs
# are needed — the script exercises the watchdog & retry logic only.
#
# Usage (from a checkout):
#   bash src/ezpz/bin/test_launch_timeout_retries.sh
#
# Or from anywhere (resolves the installed location):
#   bash "$(python3 -c 'import ezpz, pathlib; print(pathlib.Path(ezpz.__file__).parent / "bin" / "test_launch_timeout_retries.sh")')"
#
# Expectation: every scenario prints "PASS" or "FAIL" at the end with a
# brief explanation. Final exit code is 0 iff every scenario passed.
#
# Requires `ezpz launch` from the `feat/launch-timeout-retries` branch
# (or main once #136 is merged). Verify with:
#   ezpz launch --help 2>&1 | grep -E "timeout|retries"

set -u
unset CDPATH

# ---- Setup ----------------------------------------------------------------

TMP="$(mktemp -d -t ezpz-watchdog-XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
LOG="$TMP/run.log"
SUMMARY="$TMP/summary.txt"
: > "$SUMMARY"

PASS=0
FAIL=0

# Print a header for each scenario; tee both to stdout and the master log.
hdr() {
    local n="$1" desc="$2"
    printf "\n========================================================================\n"
    printf "Scenario %d: %s\n" "$n" "$desc"
    printf "========================================================================\n"
}

# Record a result; bump global counters.
record() {
    local n="$1" verdict="$2" detail="$3"
    if [[ "$verdict" == "PASS" ]]; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
    fi
    printf "[Scenario %d] %-4s  %s\n" "$n" "$verdict" "$detail" | tee -a "$SUMMARY"
}

# Quick sanity check that the new flags exist.
hdr 0 "Sanity: --timeout / --retries are in ezpz launch --help"
if ezpz launch --help 2>&1 | grep -q -- "--timeout"; then
    record 0 PASS "--timeout flag visible in help"
else
    record 0 FAIL "--timeout NOT in ezpz launch --help — wrong branch?"
    echo
    echo "Aborting: required flags not present. Are you on feat/launch-timeout-retries?"
    cat "$SUMMARY"
    exit 1
fi

# All scenarios below force --nproc 1 so each "attempt" runs ONE child
# process (mpirun -np 1) and our counter files measure launch-wrapper
# attempts, not rank fan-out. The watchdog/retry logic lives in the
# launcher itself, so single-rank is the right granularity here.
NP=(--nproc 1)

# ---- Scenario 1: happy path — chatty job under a generous timeout ---------
#
# Job emits a line every second for 5s, then exits 0. With --timeout 30
# the watchdog should never fire and we get exit 0.
hdr 1 "Happy path: chatty job, no kill expected"
START=$(date +%s)
ezpz launch "${NP[@]}" --timeout 30 \
    -- sh -c 'for i in 1 2 3 4 5; do echo "line $i"; sleep 1; done; exit 0' \
    > "$LOG" 2>&1
RC=$?
ELAPSED=$(( $(date +%s) - START ))
echo "exit=$RC  elapsed=${ELAPSED}s"
tail -20 "$LOG"
if [[ $RC -eq 0 && $ELAPSED -lt 20 ]]; then
    record 1 PASS "exit=0 in ${ELAPSED}s (expected ~5s)"
else
    record 1 FAIL "exit=$RC elapsed=${ELAPSED}s (wanted exit=0, elapsed<20s)"
fi

# ---- Scenario 2: silent job → watchdog kill ------------------------------
#
# `sleep 60` emits NO stdout. With --timeout 5 we expect:
#   - SIGTERM at ~5s
#   - exit code 124 (GNU timeout convention)
#   - elapsed well under 60s (the sleep would otherwise take a full minute)
hdr 2 "Watchdog kill: silent job, expect SIGTERM at ~5s, exit 124"
START=$(date +%s)
ezpz launch "${NP[@]}" --timeout 5 -- sleep 60 > "$LOG" 2>&1
RC=$?
ELAPSED=$(( $(date +%s) - START ))
echo "exit=$RC  elapsed=${ELAPSED}s"
tail -30 "$LOG" | grep -E "Watchdog|SIGTERM|SIGKILL" || echo "(no watchdog log lines captured)"
if [[ $RC -eq 124 && $ELAPSED -lt 25 ]]; then
    record 2 PASS "exit=124 in ${ELAPSED}s (expected ~5-15s)"
else
    record 2 FAIL "exit=$RC elapsed=${ELAPSED}s (wanted exit=124, elapsed<25s)"
fi

# ---- Scenario 3: retry-until-success --------------------------------------
#
# Counter file: script increments it, succeeds on the 3rd attempt.
# With --retries 5 we expect 3 total attempts and exit 0.
hdr 3 "Retry until success: fail twice then succeed, expect exit 0 + 3 attempts"
COUNTER="$TMP/counter1"
echo 0 > "$COUNTER"
START=$(date +%s)
# Note: shell single-quoted to defer $(cat ...) until the child runs.
ezpz launch "${NP[@]}" --retries 5 -- sh -c '
    n=$(cat "'"$COUNTER"'")
    n=$((n + 1))
    echo "$n" > "'"$COUNTER"'"
    echo "attempt $n"
    if [ "$n" -ge 3 ]; then exit 0; else exit 1; fi
' > "$LOG" 2>&1
RC=$?
ELAPSED=$(( $(date +%s) - START ))
ATTEMPTS=$(cat "$COUNTER")
echo "exit=$RC  elapsed=${ELAPSED}s  attempts=$ATTEMPTS"
tail -30 "$LOG" | grep -E "Retry|attempt" || true
# 5s + 10s backoff between attempts 2 and 3 → ~15s plus child time.
if [[ $RC -eq 0 && $ATTEMPTS -eq 3 ]]; then
    record 3 PASS "exit=0 after 3 attempts in ${ELAPSED}s"
else
    record 3 FAIL "exit=$RC attempts=$ATTEMPTS (wanted exit=0, attempts=3)"
fi

# ---- Scenario 4: watchdog + retry compose ---------------------------------
#
# Each attempt: increment counter, then sleep 30 (gets killed by watchdog).
# --timeout 3 --retries 1 → 2 attempts, both killed, final exit 124, counter=2.
hdr 4 "Compose: watchdog kill + retry, expect exit 124 + 2 attempts"
COUNTER="$TMP/counter2"
echo 0 > "$COUNTER"
START=$(date +%s)
ezpz launch "${NP[@]}" --timeout 3 --retries 1 -- sh -c '
    n=$(cat "'"$COUNTER"'")
    n=$((n + 1))
    echo "$n" > "'"$COUNTER"'"
    echo "attempt $n starting"
    sleep 30
' > "$LOG" 2>&1
RC=$?
ELAPSED=$(( $(date +%s) - START ))
ATTEMPTS=$(cat "$COUNTER")
echo "exit=$RC  elapsed=${ELAPSED}s  attempts=$ATTEMPTS"
tail -40 "$LOG" | grep -E "Watchdog|Retry|attempt" || true
# 2× (3s timeout + ≤10s grace) + 5s backoff ≈ 5-30s
if [[ $RC -eq 124 && $ATTEMPTS -eq 2 ]]; then
    record 4 PASS "exit=124 after 2 attempts in ${ELAPSED}s"
else
    record 4 FAIL "exit=$RC attempts=$ATTEMPTS (wanted exit=124, attempts=2)"
fi

# ---- Scenario 5: negative-value rejection at parse time -------------------
#
# argparse should reject `--timeout -5` before anything runs.
hdr 5 "Argparse validation: --timeout -5 must be rejected"
if ezpz launch --timeout -5 -- echo hi 2>&1 | grep -q "must be >= 0"; then
    record 5 PASS "negative --timeout rejected with expected message"
else
    record 5 FAIL "negative --timeout was accepted (or wrong error message)"
fi

# ---- Final report ---------------------------------------------------------

printf "\n========================================================================\n"
printf "Summary  (PASS=%d  FAIL=%d)\n" "$PASS" "$FAIL"
printf "========================================================================\n"
cat "$SUMMARY"

if [[ $FAIL -eq 0 ]]; then
    printf "\nAll scenarios passed. ✅\n"
    exit 0
else
    printf "\n%d scenario(s) failed. ❌  Logs in: %s\n" "$FAIL" "$TMP"
    # Keep tmpdir for inspection on failure.
    trap - EXIT
    printf "(tmpdir preserved for debugging)\n"
    exit 1
fi
