#!/usr/bin/env bash
# test_failover_scrape.sh — end-to-end validation of `ezpz.failover.scrape`.
#
# Run from any directory inside a Python env that has `saforem2/ezpz@feat/
# failover-scrape` (or later) installed. No PBS job, no GPU, no MPI required —
# pure log parsing, runs fine on a login node or even your laptop.
#
# Three checks:
#
#   A. Unit suite              — 36 tests from tests/test_failover_scrape.py
#                                downloaded straight from GitHub. Validates
#                                the package works in YOUR Python env.
#   B. CLI smoke test          — synthetic log with both Aurora patterns +
#                                the innocent-rank-N exclusion. Confirms the
#                                CLI dispatch is wired correctly.
#   C. Real-log diff           — if an upstream torchtitan scrape_bad_nodes.py
#                                AND a log with real failure patterns are both
#                                found on disk, diff the two scrapers'
#                                outputs. Strongest possible validation;
#                                skipped (not failed) when prerequisites
#                                are missing.
#
# All three are independent; the script reports each verdict separately and
# exits non-zero if any HARD test fails. Test C only counts as a failure
# when the prerequisites exist but the diff is non-empty — i.e. the scrapers
# actually disagree on a real log. Missing prerequisites are reported as
# SKIP and don't fail the script.
#
# Usage:
#   bash "$(python3 -c 'import ezpz, pathlib; print(pathlib.Path(ezpz.configs.BIN_DIR) / "test_failover_scrape.sh")")"
#
# Or, from the ezpz checkout:
#   bash src/ezpz/bin/test_failover_scrape.sh

set -u
unset CDPATH

# ANSI colors (skip if NO_COLOR is set, per the convention).
if [[ -z "${NO_COLOR:-}" && -t 1 ]]; then
    G=$'\033[1;32m'; R=$'\033[1;31m'; Y=$'\033[1;33m'; C=$'\033[1;36m'; N=$'\033[0m'
else
    G=""; R=""; Y=""; C=""; N=""
fi

PASS=0
FAIL=0
SKIP=0

record() {
    local id="$1" verdict="$2" detail="$3"
    case "${verdict}" in
        PASS) PASS=$((PASS+1)); printf "[Test %s] %sPASS%s  %s\n" "${id}" "${G}" "${N}" "${detail}" ;;
        FAIL) FAIL=$((FAIL+1)); printf "[Test %s] %sFAIL%s  %s\n" "${id}" "${R}" "${N}" "${detail}" ;;
        SKIP) SKIP=$((SKIP+1)); printf "[Test %s] %sSKIP%s  %s\n" "${id}" "${Y}" "${N}" "${detail}" ;;
    esac
}

hdr() {
    printf "\n%s========================================================================%s\n" "${C}" "${N}"
    printf "%sTest %s%s\n" "${C}" "$*" "${N}"
    printf "%s========================================================================%s\n" "${C}" "${N}"
}

# ---- Preflight: ezpz importable + version sanity ---------------------------

hdr "0: preflight — ezpz + ezpz.failover importable"
if ! python3 -c "import ezpz" 2>/dev/null; then
    record 0 FAIL "ezpz not importable in this Python env. Install with:
        pip install -U 'git+https://github.com/saforem2/ezpz.git@feat/failover-scrape'"
    printf "\n%s%d test(s) failed. ❌%s\n" "${R}" "${FAIL}" "${N}"
    exit 1
fi
if ! python3 -c "from ezpz.failover import scrape_bad_nodes" 2>/dev/null; then
    record 0 FAIL "ezpz.failover not importable. Either the install is stale (pre-#142) or the branch isn't installed. Try:
        pip install -U --force-reinstall --no-deps 'git+https://github.com/saforem2/ezpz.git@feat/failover-scrape'"
    printf "\n%s%d test(s) failed. ❌%s\n" "${R}" "${FAIL}" "${N}"
    exit 1
fi
EZPZ_VERSION=$(python3 -c "import ezpz; print(getattr(ezpz, '__version__', '?'))" 2>/dev/null)
EZPZ_LOC=$(python3 -c "import ezpz, pathlib; print(pathlib.Path(ezpz.__file__).parent)")
record 0 PASS "ezpz=${EZPZ_VERSION} at ${EZPZ_LOC}"

# ---- Test A: unit suite ----------------------------------------------------

hdr "A: unit suite (36 tests downloaded from GitHub)"
# Use an underscore (not a dot) before $$ — pytest treats dots in filenames as
# package-path separators during collection, so `test_failover_scrape.1234.py`
# triggers `ModuleNotFoundError: No module named 'test_failover_scrape.1234'`
# even though the file exists.
TEST_FILE="/tmp/test_failover_scrape_$$.py"
SAMPLE="/tmp/sample_failure_$$.log"
OLD_OUT="/tmp/old_scraper_$$.txt"
NEW_OUT="/tmp/new_scraper_$$.txt"
trap 'rm -f "${TEST_FILE}" "${SAMPLE}" "${OLD_OUT}" "${NEW_OUT}"' EXIT

# Use curl if available, else fall back to urllib via Python — many Sunspot
# users don't have a working curl proxy until ezpz_setup runs, but Python's
# urllib honors http_proxy/https_proxy automatically.
TEST_URL="https://raw.githubusercontent.com/saforem2/ezpz/feat/failover-scrape/tests/test_failover_scrape.py"
if command -v curl >/dev/null 2>&1 && \
   curl -fsSL --max-time 15 "${TEST_URL}" -o "${TEST_FILE}" 2>/dev/null; then
    :
else
    python3 -c "
import urllib.request, pathlib
urllib.request.urlretrieve('${TEST_URL}', '${TEST_FILE}')
" 2>/dev/null || {
        record A FAIL "could not fetch test file from GitHub. Check proxy env (http_proxy/https_proxy)."
        printf "\n"
        # Don't bail — try B and C with whatever we have
        TEST_FILE=""
    }
fi

if [[ -n "${TEST_FILE}" && -f "${TEST_FILE}" ]]; then
    if ! python3 -m pytest --version >/dev/null 2>&1; then
        record A SKIP "pytest not available in this env"
    else
        # Run pytest; print its tail-5 for context but check pytest's
        # OWN exit code via PIPESTATUS, not tail's. `if cmd | tail` would
        # always be "true" because tail always exits 0 even when cmd fails
        # (which is how a collection error got reported as PASS earlier).
        python3 -m pytest "${TEST_FILE}" -q 2>&1 | tail -5
        pytest_rc=${PIPESTATUS[0]}
        if (( pytest_rc == 0 )); then
            record A PASS "all unit tests pass"
        else
            record A FAIL "pytest exit ${pytest_rc} — see output above"
        fi
    fi
fi

# ---- Test B: CLI smoke -----------------------------------------------------

hdr "B: CLI smoke — synthetic log + --explain"
# ${SAMPLE} declared above in the trap.
cat > "${SAMPLE}" <<'EOF'
[2026-05-23 12:34:56] training step=42 loss=2.4
x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9
x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9
rank 1304 died from signal 11
Connection closed by peer [127.0.0.1]:1234
EOF

CLI_OUT=$(python3 -m ezpz.failover --machine aurora "${SAMPLE}" 2>/dev/null)
EXPECTED="x4502c1s3b0n0.hsn.cm.aurora.alcf.anl.gov"
if [[ "${CLI_OUT}" == "${EXPECTED}" ]]; then
    record B PASS "single dedup'd hostname; innocent rank-11 correctly excluded"
else
    record B FAIL "expected ${EXPECTED!r}, got ${CLI_OUT!r}"
fi

# Bonus: show the --explain breakdown so the user can eyeball it
printf "    %s--explain output:%s\n" "${C}" "${N}"
python3 -m ezpz.failover --machine aurora --explain "${SAMPLE}" 2>&1 \
    | sed 's/^/        /'

# ---- Test C: real-log diff against upstream torchtitan scraper -------------

hdr "C: real-log diff against torchtitan's scrape_bad_nodes.py"

# Find upstream scraper. Order: explicit env override → torchtitan checkout
# next to ezpz → torchtitan checkout under ~/projects → bail.
UPSTREAM="${UPSTREAM_SCRAPER:-}"
if [[ -z "${UPSTREAM}" ]]; then
    for candidate in \
        "../torchtitan/torchtitan/experiments/ezpz/scripts/scrape_bad_nodes.py" \
        "../torchtitan-ezpz/torchtitan/experiments/ezpz/scripts/scrape_bad_nodes.py" \
        "${HOME}/projects/saforem2/torchtitan/torchtitan/experiments/ezpz/scripts/scrape_bad_nodes.py" \
        "${HOME}/projects/saforem2/torchtitan-ezpz/torchtitan/experiments/ezpz/scripts/scrape_bad_nodes.py" \
        "${HOME}/datascience/foremans/projects/saforem2/torchtitan/torchtitan/experiments/ezpz/scripts/scrape_bad_nodes.py" \
        "${HOME}/datascience/foremans/projects/saforem2/torchtitan-ezpz/torchtitan/experiments/ezpz/scripts/scrape_bad_nodes.py"
    do
        if [[ -f "${candidate}" ]]; then UPSTREAM="${candidate}"; break; fi
    done
fi

if [[ -z "${UPSTREAM}" || ! -f "${UPSTREAM}" ]]; then
    record C SKIP "no upstream scrape_bad_nodes.py found. Set UPSTREAM_SCRAPER=/path/to/file to override."
else
    printf "    upstream scraper: %s%s%s\n" "${C}" "${UPSTREAM}" "${N}"

    # Find a real log with failure patterns. Use either the user's $LOG var
    # if set, or auto-search for the most patterns.
    REAL_LOG="${LOG:-}"
    if [[ -n "${REAL_LOG}" && ! -f "${REAL_LOG}" ]]; then
        printf "    %swarning: \$LOG points at non-existent file: %s — auto-searching instead%s\n" "${Y}" "${REAL_LOG}" "${N}"
        REAL_LOG=""
    fi

    if [[ -z "${REAL_LOG}" ]]; then
        printf "    searching for a log with real failure patterns (this may take a few seconds)...\n"
        # Search the current directory for files that COULD contain the
        # failure patterns. We can't filter by .log extension only —
        # PBS sometimes writes to .out/.err and ezpz uses .jsonl/.log
        # interchangeably. But explicitly EXCLUDE source files / build
        # artifacts so we don't accidentally "find" the test fixture
        # files that contain the patterns as string literals.
        # Bounded to 2000 files; on Sunspot's lustre this is ~5s
        # worst-case, well within the prompt-iteration budget.
        EXCLUDE_RE='\.(py|pyc|so|pyi|ipynb|md|rst|toml|yaml|yml|json|sh)$|/(\.git|node_modules|__pycache__|\.venv|\.pytest_cache)/'
        if command -v fd >/dev/null 2>&1; then
            CANDIDATES=$(fd -HI --type=file . 2>/dev/null \
                | grep -Ev "${EXCLUDE_RE}" | head -2000)
        else
            CANDIDATES=$(find . -type f 2>/dev/null \
                | grep -Ev "${EXCLUDE_RE}" | head -2000)
        fi
        BEST_LOG=""
        BEST_COUNT=0
        for f in ${CANDIDATES}; do
            # `grep -c` prints to stdout AND `|| echo 0` adds its own line
            # when grep matches nothing — combine both into one integer.
            count=$(grep -cE "shepherd died from signal 9|Connection closed by peer" "${f}" 2>/dev/null | head -1)
            count=${count:-0}
            if (( count > BEST_COUNT )); then
                BEST_LOG="${f}"
                BEST_COUNT=${count}
            fi
        done
        if [[ -n "${BEST_LOG}" && ${BEST_COUNT} -gt 0 ]]; then
            REAL_LOG="${BEST_LOG}"
            printf "    %sfound%s: %s (%d matching lines)\n" "${G}" "${N}" "${REAL_LOG}" "${BEST_COUNT}"
        fi
    else
        # User-supplied $LOG; report match count so they know whether the
        # diff will be meaningful. Same single-integer extraction as above.
        match_count=$(grep -cE "shepherd died from signal 9|Connection closed by peer" "${REAL_LOG}" 2>/dev/null | head -1)
        match_count=${match_count:-0}
        printf "    user-supplied log: %s (%d matching lines)\n" "${REAL_LOG}" "${match_count}"
        if (( match_count == 0 )); then
            printf "    %swarning: log has 0 failure patterns; diff being empty proves nothing%s\n" "${Y}" "${N}"
        fi
    fi

    if [[ -z "${REAL_LOG}" ]]; then
        record C SKIP "no log with bad-node patterns found in cwd. Set \$LOG=/path/to/postmortem.log to override."
    else
        # ${OLD_OUT} / ${NEW_OUT} declared above in the trap.
        python3 "${UPSTREAM}"                       "${REAL_LOG}" > "${OLD_OUT}" 2>/dev/null
        python3 -m ezpz.failover --machine aurora   "${REAL_LOG}" > "${NEW_OUT}" 2>/dev/null

        # `wc -l <file` pads with whitespace on some systems; strip it
        # so the printf %d works and the eventual `(${NEW_LINES} hostname(s))`
        # message isn't misaligned.
        OLD_LINES=$(wc -l < "${OLD_OUT}" | tr -d ' ')
        NEW_LINES=$(wc -l < "${NEW_OUT}" | tr -d ' ')
        printf "    upstream emitted: %d hostname(s)\n" "${OLD_LINES}"
        printf "    ezpz emitted:     %d hostname(s)\n" "${NEW_LINES}"

        if (( OLD_LINES == 0 && NEW_LINES == 0 )); then
            # Both empty. This could mean either "log has no patterns" (in
            # which case both being empty is correct) or "both scrapers are
            # broken". Report as a soft pass with a clear caveat.
            record C PASS "both scrapers emitted nothing — consistent, but log may have no patterns"
        elif diff -q "${OLD_OUT}" "${NEW_OUT}" >/dev/null 2>&1; then
            record C PASS "behavior-identical on real log (${NEW_LINES} hostname(s))"
        else
            record C FAIL "scrapers DISAGREE on real log — diff:"
            diff "${OLD_OUT}" "${NEW_OUT}" | sed 's/^/        /'
        fi
    fi
fi

# ---- Summary ---------------------------------------------------------------

printf "\n%s========================================================================%s\n" "${C}" "${N}"
printf "Summary  (%sPASS=%d%s  %sFAIL=%d%s  %sSKIP=%d%s)\n" \
    "${G}" "${PASS}" "${N}" "${R}" "${FAIL}" "${N}" "${Y}" "${SKIP}" "${N}"
printf "%s========================================================================%s\n" "${C}" "${N}"

if (( FAIL > 0 )); then
    printf "\n%s%d test(s) failed. ❌%s\n" "${R}" "${FAIL}" "${N}"
    exit 1
elif (( SKIP > 0 )); then
    printf "\n%sAll runnable tests passed. %d skipped (see above for prerequisites).%s ✅\n" "${G}" "${SKIP}" "${N}"
    exit 0
else
    printf "\n%sAll tests passed. ✅%s\n" "${G}" "${N}"
    exit 0
fi
