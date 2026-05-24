#!/usr/bin/env bash
# test_failover_lib.sh — unit tests for src/ezpz/bin/failover.sh.
#
# Run from the repo root:
#   bash tests/test_failover_lib.sh
#
# Each test:
#   - sets up a temp dir with a fixture PBS_NODEFILE
#   - shadows `ezpz` and `python3 -m ezpz.failover` with stub scripts
#     that produce known outputs / exit codes
#   - sources failover.sh, calls the function under test
#   - asserts on the resulting state (hostfile contents, bad_nodes.txt,
#     log messages, exit code)
#
# Exit non-zero if any test fails. Each test is independent — its own
# temp dir, its own PBS_* env vars.

set -u
unset CDPATH

# ---- Test harness ----------------------------------------------------------

PASS=0
FAIL=0
FAILED_TESTS=()

# ANSI colors (skip if NO_COLOR set)
if [[ -z "${NO_COLOR:-}" && -t 1 ]]; then
    G=$'\033[1;32m'; R=$'\033[1;31m'; Y=$'\033[1;33m'; C=$'\033[1;36m'; N=$'\033[0m'
else
    G=""; R=""; Y=""; C=""; N=""
fi

# Locate failover.sh relative to this test file (resolves symlinks too).
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${TEST_DIR}/.." && pwd)"
FAILOVER_LIB="${REPO_ROOT}/src/ezpz/bin/failover.sh"
if [[ ! -f "${FAILOVER_LIB}" ]]; then
    printf "%sFATAL%s: failover.sh not found at %s\n" "${R}" "${N}" "${FAILOVER_LIB}" >&2
    exit 1
fi

run_test() {
    # Usage: run_test "test_name" "test_function"
    local name="$1"
    local fn="$2"
    printf "  %s%-60s%s " "${C}" "${name}" "${N}"

    # Each test gets its own subshell + temp dir so they can't leak env
    # vars or files into each other. Capture stdout+stderr in case the
    # test needs them, and the test's exit code is the verdict.
    local tmpdir
    tmpdir=$(mktemp -d "/tmp/failover-test-XXXXXX")
    local output
    if output=$(
        cd "${tmpdir}"
        export TMPDIR="${tmpdir}"
        export PBS_JOBID="testjob-$$"
        # Re-source failover.sh inside the subshell so each test starts
        # clean. We DO need to set NHOSTS first because the lib reads
        # it; otherwise it stays unset between tests.
        unset NHOSTS PBS_NODEFILE FAILOVER_LOG_DIR FAILOVER_ACTIVE \
              FAILOVER_SPARE FAILOVER_BAD FAILOVER_PBS_NODEFILE_ORIG \
              FAILOVER_IDLE_TIMEOUT FAILOVER_MAX_RETRIES \
              FAILOVER_KEEP_BAD
        # shellcheck disable=SC1090
        source "${FAILOVER_LIB}"
        "${fn}"
    ) 2>&1; then
        PASS=$((PASS+1))
        printf "%sPASS%s\n" "${G}" "${N}"
    else
        FAIL=$((FAIL+1))
        FAILED_TESTS+=("${name}")
        printf "%sFAIL%s\n" "${R}" "${N}"
        # Indent the captured output for readability
        printf "%s\n" "${output}" | sed 's/^/      /'
    fi
    rm -rf "${tmpdir}"
}

# Assertion helpers (called from inside tests). Each `assert_*` exits the
# subshell with non-zero on failure so run_test marks the test as FAIL.
assert_eq() {
    local actual="$1" expected="$2" msg="${3:-}"
    if [[ "${actual}" != "${expected}" ]]; then
        printf "ASSERT FAIL%s: got %q, want %q\n" \
            "${msg:+ ($msg)}" "${actual}" "${expected}" >&2
        exit 1
    fi
}

assert_contains() {
    local haystack="$1" needle="$2" msg="${3:-}"
    if [[ "${haystack}" != *"${needle}"* ]]; then
        printf "ASSERT FAIL%s: %q does not contain %q\n" \
            "${msg:+ ($msg)}" "${haystack}" "${needle}" >&2
        exit 1
    fi
}

assert_file_contents() {
    local file="$1" expected="$2"
    if [[ ! -f "${file}" ]]; then
        printf "ASSERT FAIL: file not found: %s\n" "${file}" >&2
        exit 1
    fi
    local actual
    actual=$(cat "${file}")
    if [[ "${actual}" != "${expected}" ]]; then
        printf "ASSERT FAIL: %s contents mismatch\n  want:\n%s\n  got:\n%s\n" \
            "${file}" "${expected}" "${actual}" >&2
        exit 1
    fi
}

# Set up a fixture PBS_NODEFILE inside the cwd (the subshell's tmpdir).
# Returns the path via stdout.
setup_pbs_nodefile() {
    local lines="$1"  # one host per line, separated by newlines
    local file="${TMPDIR}/PBS_NODEFILE"
    printf "%s\n" "${lines}" > "${file}"
    export PBS_NODEFILE="${file}"
    echo "${file}"
}

# Shadow `python3 -m ezpz.failover` with a script that emits the given
# bad-node hostnames (one per line) when invoked. We create a fake `python3`
# binary in a bin dir that's prepended to PATH; it inspects argv to detect
# `-m ezpz.failover ...` and either emits the canned response or passes
# through to the real python3 for everything else.
shadow_scrape_response() {
    local response="$1"   # multiline bad-host string (may be empty)
    local bindir="${TMPDIR}/bin"
    mkdir -p "${bindir}"
    cat > "${bindir}/python3" <<EOF
#!/usr/bin/env bash
# Detect 'python3 -m ezpz.failover ...' (the scrape invocation).
if [[ "\$1" == "-m" && "\$2" == "ezpz.failover" ]]; then
    printf "%s" "${response//\"/\\\"}"
    exit 0
fi
# Fall through to the real python3 for everything else.
exec /usr/bin/env -i PATH="/usr/bin:/bin:/usr/local/bin" python3 "\$@"
EOF
    chmod +x "${bindir}/python3"
    export PATH="${bindir}:${PATH}"
}

# Shadow `ezpz` itself with a script that records its invocation and exits
# with the given code. Reads exit code from the file passed (so successive
# attempts can return different codes).
shadow_ezpz_launch() {
    local exit_code_file="$1"  # path; we read one int per line, popping
    local bindir="${TMPDIR}/bin"
    mkdir -p "${bindir}"
    cat > "${bindir}/ezpz" <<EOF
#!/usr/bin/env bash
# Record the invocation
echo "ezpz \$*" >> "${TMPDIR}/ezpz_invocations.log"

# Pop the next exit code from the file
ec=\$(head -n 1 "${exit_code_file}" 2>/dev/null || echo 0)
tail -n +2 "${exit_code_file}" > "${exit_code_file}.tmp" 2>/dev/null && mv "${exit_code_file}.tmp" "${exit_code_file}"
exit "\${ec:-0}"
EOF
    chmod +x "${bindir}/ezpz"
    export PATH="${bindir}:${PATH}"
}

# ---- Tests: failover_init --------------------------------------------------

test_init_splits_hostfile() {
    setup_pbs_nodefile "host1
host2
host3
host4
host5" >/dev/null

    failover_init 3 || exit 1

    # Active should be first 3, spare should be last 2.
    assert_file_contents "${FAILOVER_ACTIVE}" "host1
host2
host3"
    assert_file_contents "${FAILOVER_SPARE}" "host4
host5"
    # PBS_NODEFILE redirected to the active subset, NHOSTS set.
    assert_eq "${PBS_NODEFILE}" "${FAILOVER_ACTIVE}" "PBS_NODEFILE points to active"
    assert_eq "${NHOSTS}" "3" "NHOSTS"
    # bad_nodes.txt created empty.
    assert_file_contents "${FAILOVER_BAD}" ""
}

test_init_errors_when_nhosts_train_too_large() {
    setup_pbs_nodefile "host1
host2" >/dev/null
    # Asking for 5 nodes when PBS only gave 2 should fail.
    if failover_init 5 2>/dev/null; then
        echo "ASSERT FAIL: expected failover_init 5 to fail" >&2
        exit 1
    fi
}

test_init_errors_when_no_pbs_nodefile() {
    # PBS_NODEFILE intentionally unset.
    if failover_init 1 2>/dev/null; then
        echo "ASSERT FAIL: expected failover_init to fail without PBS_NODEFILE" >&2
        exit 1
    fi
}

test_init_errors_when_no_arg() {
    setup_pbs_nodefile "host1" >/dev/null
    if failover_init 2>/dev/null; then
        echo "ASSERT FAIL: expected failover_init to fail with no arg" >&2
        exit 1
    fi
}

# ---- Tests: failover_swap_in -----------------------------------------------

test_swap_in_replaces_bad_with_spare() {
    setup_pbs_nodefile "host1
host2
host3
host4
host5" >/dev/null
    failover_init 3 || exit 1
    # Active: host1,host2,host3  Spare: host4,host5
    failover_swap_in host2 || exit 1
    # Active should now be host1,host4,host3 (host2 → host4)
    assert_file_contents "${FAILOVER_ACTIVE}" "host1
host4
host3"
    # Spare popped: just host5 left
    assert_file_contents "${FAILOVER_SPARE}" "host5"
    # bad_nodes.txt records what was swapped
    assert_file_contents "${FAILOVER_BAD}" "host2"
}

test_swap_in_handles_multiple_bad_nodes() {
    setup_pbs_nodefile "host1
host2
host3
host4
host5" >/dev/null
    failover_init 3 || exit 1
    failover_swap_in host1 host3 || exit 1
    # host1 → host4, host3 → host5
    assert_file_contents "${FAILOVER_ACTIVE}" "host4
host2
host5"
    assert_file_contents "${FAILOVER_SPARE}" ""
    assert_file_contents "${FAILOVER_BAD}" "host1
host3"
}

test_swap_in_errors_when_no_spares() {
    setup_pbs_nodefile "host1
host2" >/dev/null
    failover_init 2 || exit 1
    # No spares (all 2 nodes went to active)
    if failover_swap_in host1 2>/dev/null; then
        echo "ASSERT FAIL: expected swap_in to fail with no spares" >&2
        exit 1
    fi
}

test_swap_in_skips_unknown_host() {
    setup_pbs_nodefile "host1
host2
host3
host4" >/dev/null
    failover_init 3 || exit 1
    # 'host_not_in_active' isn't in active; should be silently skipped
    # without consuming a spare.
    failover_swap_in host_not_in_active || exit 1
    assert_file_contents "${FAILOVER_ACTIVE}" "host1
host2
host3"
    assert_file_contents "${FAILOVER_SPARE}" "host4"
    # bad_nodes.txt should NOT have the skip recorded
    assert_file_contents "${FAILOVER_BAD}" ""
}

# ---- Tests: failover_swap_one_blind ----------------------------------------

test_swap_one_blind_rotates_first_active() {
    setup_pbs_nodefile "host1
host2
host3
host4" >/dev/null
    failover_init 3 || exit 1
    failover_swap_one_blind || exit 1
    # host1 (first active) gets replaced with host4 (first spare)
    assert_file_contents "${FAILOVER_ACTIVE}" "host4
host2
host3"
    assert_file_contents "${FAILOVER_BAD}" "host1"
}

test_swap_one_blind_errors_when_no_spares() {
    setup_pbs_nodefile "host1
host2" >/dev/null
    failover_init 2 || exit 1
    if failover_swap_one_blind 2>/dev/null; then
        echo "ASSERT FAIL: expected swap_one_blind to fail with no spares" >&2
        exit 1
    fi
}

# ---- Tests: failover_run ---------------------------------------------------

test_run_succeeds_first_attempt() {
    setup_pbs_nodefile "host1
host2" >/dev/null
    failover_init 2 || exit 1
    # Mock ezpz to exit 0 on first call
    echo "0" > "${TMPDIR}/exits.txt"
    shadow_ezpz_launch "${TMPDIR}/exits.txt"
    shadow_scrape_response ""

    export FAILOVER_MAX_RETRIES=2
    # Use a non-`ezpz launch` command so we don't trigger the flag
    # injection path. `true` resolves via PATH so this works on macOS
    # (where /bin/true exists at /usr/bin/true) and Linux alike.
    failover_run true || exit 1
}

test_run_retries_on_failure_then_succeeds() {
    setup_pbs_nodefile "host1
host2
host3" >/dev/null
    failover_init 2 || exit 1
    # Set up: first attempt fails (exit 1), second succeeds (exit 0).
    # Pre-populate attempt-2.log to be empty so the scraper returns nothing,
    # which makes failover_run call swap_one_blind.
    printf "1\n0\n" > "${TMPDIR}/exits.txt"

    # Shadow `bash` so the tee'd command writes the log we expect to scrape.
    local bindir="${TMPDIR}/bin"
    mkdir -p "${bindir}"
    cat > "${bindir}/failing_cmd" <<'EOF'
#!/usr/bin/env bash
n_file="${TMPDIR}/call_count"
n=$(cat "${n_file}" 2>/dev/null || echo 0)
n=$((n + 1))
echo "${n}" > "${n_file}"
if [[ "${n}" == "1" ]]; then exit 1; else exit 0; fi
EOF
    chmod +x "${bindir}/failing_cmd"
    export PATH="${bindir}:${PATH}"
    shadow_scrape_response ""  # blind rotation each time

    export FAILOVER_MAX_RETRIES=2
    failover_run failing_cmd || exit 1
    # First attempt swapped host1 → host3 blindly
    assert_file_contents "${FAILOVER_BAD}" "host1"
}

test_run_walltime_143_no_retry_when_clean() {
    setup_pbs_nodefile "host1
host2
host3" >/dev/null
    failover_init 2 || exit 1
    # Wrap a script that always exits 143 and writes a "clean" log
    # (no bad-node crash patterns). Walltime kill → should NOT retry.
    local bindir="${TMPDIR}/bin"
    mkdir -p "${bindir}"
    cat > "${bindir}/walltime_cmd" <<'EOF'
#!/usr/bin/env bash
echo "Normal training output"
exit 143
EOF
    chmod +x "${bindir}/walltime_cmd"
    export PATH="${bindir}:${PATH}"
    shadow_scrape_response ""

    export FAILOVER_MAX_RETRIES=3
    # Expect rc=143; the function should NOT retry → bad_nodes.txt empty.
    rc=0
    failover_run walltime_cmd || rc=$?
    assert_eq "${rc}" "143" "rc"
    assert_file_contents "${FAILOVER_BAD}" ""
}

test_run_walltime_143_retries_when_bad_node_pattern_in_log() {
    setup_pbs_nodefile "host1
host2
host3" >/dev/null
    failover_init 2 || exit 1
    # Wrap a script that exits 143 but logs a bad-node pattern. Walltime
    # AND bad-node → SHOULD retry. We mock so the second attempt exits 0.
    local bindir="${TMPDIR}/bin"
    mkdir -p "${bindir}"
    cat > "${bindir}/bad_walltime_cmd" <<'EOF'
#!/usr/bin/env bash
n_file="${TMPDIR}/call_count"
n=$(cat "${n_file}" 2>/dev/null || echo 0)
n=$((n + 1))
echo "${n}" > "${n_file}"
if [[ "${n}" == "1" ]]; then
    echo "RuntimeError: [..gloo..] Connection closed by peer [10.0.0.1]:1234"
    exit 143
else
    echo "Normal training output"
    exit 0
fi
EOF
    chmod +x "${bindir}/bad_walltime_cmd"
    export PATH="${bindir}:${PATH}"
    shadow_scrape_response ""  # blind rotation

    export FAILOVER_MAX_RETRIES=2
    failover_run bad_walltime_cmd || exit 1
    # Should have swapped host1 → host3 on the bad-node retry
    assert_file_contents "${FAILOVER_BAD}" "host1"
}

test_run_walltime_143_no_retry_when_only_innocent_rank_signals() {
    # Regression for the Codex P2 review on PR #143: walltime SIGTERM
    # cascades to ranks as `rank N died from signal {11,15}`. Those
    # lines are NOT bad-node indicators (the cascade originated on a
    # DIFFERENT node — see scraper's test_innocent_rank_signal_11_not_matched).
    # The walltime path must NOT treat them as a reason to retry.
    setup_pbs_nodefile "host1
host2
host3" >/dev/null
    failover_init 2 || exit 1
    local bindir="${TMPDIR}/bin"
    mkdir -p "${bindir}"
    cat > "${bindir}/walltime_with_cascade_cmd" <<'EOF'
#!/usr/bin/env bash
# Real walltime kill: SIGTERM cascade emits rank-signal lines but
# NO shepherd-died-9 and no gloo Connection-closed.
echo "rank 0 died from signal 15"
echo "rank 1 died from signal 11"
echo "rank 2 died from signal 15"
exit 143
EOF
    chmod +x "${bindir}/walltime_with_cascade_cmd"
    export PATH="${bindir}:${PATH}"
    shadow_scrape_response ""  # scraper finds nothing either

    export FAILOVER_MAX_RETRIES=2
    local rc=0
    failover_run walltime_with_cascade_cmd || rc=$?
    assert_eq "${rc}" "143" "rc"
    # Critical: no swap happened — bad_nodes.txt is empty.
    assert_eq "$(wc -l < "${FAILOVER_BAD}" | tr -d ' ')" "0" "bad_nodes_count"
}

test_run_walltime_143_retries_on_real_aurora_ur_oom_with_cascade() {
    # Regression for actual Aurora torchtitan log shape (2026-05-12):
    # 18 training steps complete cleanly, then a real level_zero
    # UR_RESULT_ERROR_OUT_OF_RESOURCES on one rank, then mpiexec
    # teardown emits `rank N died from signal 15` as a cascade
    # alongside `rank N exited with code 1`. Without the innocent-
    # cascade strip, this log would silently be misclassified as
    # walltime — losing the bad-node signal.
    setup_pbs_nodefile "host1
host2
host3" >/dev/null
    failover_init 2 || exit 1
    local bindir="${TMPDIR}/bin"
    mkdir -p "${bindir}"
    cat > "${bindir}/aurora_ur_oom_cmd" <<'EOF'
#!/usr/bin/env bash
n_file="${TMPDIR}/call_count"
n=$(cat "${n_file}" 2>/dev/null || echo 0)
n=$((n + 1))
echo "${n}" > "${n_file}"
if [[ "${n}" == "1" ]]; then
    # Verbatim shape from the real postmortem.
    echo "step:  1  loss: 12.94587  ..."
    echo "step: 18  loss: 10.27772  ..."
    echo "[rank7]: RuntimeError: level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)"
    echo "x4610c4s3b0n0.hsn.cm.aurora.alcf.anl.gov: rank 7 exited with code 1"
    echo "x4610c4s5b0n0.hsn.cm.aurora.alcf.anl.gov: rank 14 died from signal 15"
    exit 143
else
    exit 0
fi
EOF
    chmod +x "${bindir}/aurora_ur_oom_cmd"
    export PATH="${bindir}:${PATH}"
    shadow_scrape_response ""  # blind rotation

    export FAILOVER_MAX_RETRIES=2
    failover_run aurora_ur_oom_cmd || exit 1
    # Verifies we DID retry (cascade didn't mask the real OOM).
    assert_file_contents "${FAILOVER_BAD}" "host1"
}

test_run_walltime_143_retries_on_real_hw_death_mixed_with_innocent_cascade() {
    # Companion to the prior test: even when the log has innocent
    # rank-cascade lines, a REAL hardware death (gloo, OOM, shepherd-9,
    # etc.) elsewhere in the same log must still trigger a retry. The
    # `grep -v` of innocent ranks should only strip the cascade lines,
    # not the genuine signal.
    setup_pbs_nodefile "host1
host2
host3" >/dev/null
    failover_init 2 || exit 1
    local bindir="${TMPDIR}/bin"
    mkdir -p "${bindir}"
    cat > "${bindir}/real_death_with_cascade_cmd" <<'EOF'
#!/usr/bin/env bash
n_file="${TMPDIR}/call_count"
n=$(cat "${n_file}" 2>/dev/null || echo 0)
n=$((n + 1))
echo "${n}" > "${n_file}"
if [[ "${n}" == "1" ]]; then
    # Real failure + innocent downstream cascades. The cascade lines
    # would have masked the real OOM under the old regex; the new one
    # strips the cascades first.
    echo "rank 0 died from signal 15"
    echo "rank 1 died from signal 11"
    echo "torch.cuda.OutOfMemoryError: CUDA out of memory"
    exit 143
else
    exit 0
fi
EOF
    chmod +x "${bindir}/real_death_with_cascade_cmd"
    export PATH="${bindir}:${PATH}"
    shadow_scrape_response ""  # blind rotation

    export FAILOVER_MAX_RETRIES=2
    failover_run real_death_with_cascade_cmd || exit 1
    # Verifies we DID retry — host1 got swapped out.
    assert_file_contents "${FAILOVER_BAD}" "host1"
}

test_run_swaps_named_bad_node_when_scraper_finds_one() {
    setup_pbs_nodefile "host1
host2
host3" >/dev/null
    failover_init 2 || exit 1
    # First attempt fails; scraper identifies host1 as the bad one;
    # second attempt succeeds.
    local bindir="${TMPDIR}/bin"
    mkdir -p "${bindir}"
    cat > "${bindir}/failing_cmd" <<'EOF'
#!/usr/bin/env bash
n_file="${TMPDIR}/call_count"
n=$(cat "${n_file}" 2>/dev/null || echo 0)
n=$((n + 1))
echo "${n}" > "${n_file}"
if [[ "${n}" == "1" ]]; then exit 1; else exit 0; fi
EOF
    chmod +x "${bindir}/failing_cmd"
    export PATH="${bindir}:${PATH}"
    shadow_scrape_response "host1"  # named bad node

    export FAILOVER_MAX_RETRIES=2
    failover_run failing_cmd || exit 1
    # Should have swapped the NAMED bad node (host1), not the first active
    assert_file_contents "${FAILOVER_BAD}" "host1"
}

test_run_exhausts_max_retries() {
    setup_pbs_nodefile "host1
host2
host3
host4
host5" >/dev/null
    failover_init 2 || exit 1
    # Always-failing command should exhaust retries and return the
    # failure exit code (not 0).
    local bindir="${TMPDIR}/bin"
    mkdir -p "${bindir}"
    cat > "${bindir}/always_fails" <<'EOF'
#!/usr/bin/env bash
exit 42
EOF
    chmod +x "${bindir}/always_fails"
    export PATH="${bindir}:${PATH}"
    shadow_scrape_response ""

    export FAILOVER_MAX_RETRIES=2
    rc=0
    failover_run always_fails || rc=$?
    assert_eq "${rc}" "42" "rc — should propagate child's exit code"
}

test_run_ansi_stripping_in_inner_rc_detection() {
    # If `ezpz launch` outputs `Execution finished with \x1b[1;36m127\x1b[0m`,
    # the inner_rc detection must strip ANSI before parsing, otherwise it
    # reads '1' from '[1;36m' and concludes a clean run.
    setup_pbs_nodefile "host1
host2
host3" >/dev/null
    failover_init 2 || exit 1
    local bindir="${TMPDIR}/bin"
    mkdir -p "${bindir}"
    # Command exits 0 (shell rc=0), but log says "Execution finished with 127"
    # wrapped in ANSI. Wrapper should detect rc=127 and treat as failure.
    # Then second attempt exits 0 cleanly.
    cat > "${bindir}/ansi_cmd" <<'EOF'
#!/usr/bin/env bash
n_file="${TMPDIR}/call_count"
n=$(cat "${n_file}" 2>/dev/null || echo 0)
n=$((n + 1))
echo "${n}" > "${n_file}"
if [[ "${n}" == "1" ]]; then
    # The literal $'...' style ANSI escape sequence.
    printf 'Execution finished with \e[1;36m127\e[0m\n'
    exit 0  # exits 0 — the override must catch the inner 127
else
    echo "normal output"
    exit 0
fi
EOF
    chmod +x "${bindir}/ansi_cmd"
    export PATH="${bindir}:${PATH}"
    shadow_scrape_response ""

    export FAILOVER_MAX_RETRIES=2
    failover_run ansi_cmd || exit 1
    # If the ANSI override worked, attempt 1 was treated as a failure,
    # we swapped host1 → host3 blindly, attempt 2 succeeded.
    # If the ANSI override DIDN'T work, we'd have returned 0 after
    # attempt 1 with NO swap, leaving bad_nodes.txt empty.
    assert_file_contents "${FAILOVER_BAD}" "host1"
}

test_run_ezpz_launch_injects_topology_args() {
    setup_pbs_nodefile "host1
host2
host3" >/dev/null
    failover_init 2 || exit 1
    # When the command is `ezpz launch ...`, the wrapper should inject
    # --hostfile, --nnodes, -ppn, -n, --timeout. Capture the invocation
    # via the shadow ezpz and assert.
    echo "0" > "${TMPDIR}/exits.txt"
    shadow_ezpz_launch "${TMPDIR}/exits.txt"
    shadow_scrape_response ""

    export NGPU_PER_HOST=4
    export FAILOVER_IDLE_TIMEOUT=600
    export FAILOVER_MAX_RETRIES=1
    failover_run ezpz launch python3 -m my.app || exit 1

    # Check the invocation log captured the injected args
    local inv
    inv=$(cat "${TMPDIR}/ezpz_invocations.log")
    assert_contains "${inv}" "--hostfile=${FAILOVER_ACTIVE}" "--hostfile"
    assert_contains "${inv}" "--nnodes=2" "--nnodes"
    assert_contains "${inv}" "-ppn 4" "-ppn"
    assert_contains "${inv}" "-n 8" "-n (2 nodes * 4 ppn)"
    assert_contains "${inv}" "--timeout=600" "--timeout"
    assert_contains "${inv}" "python3 -m my.app" "user command preserved"
}

# ---- Run all tests ---------------------------------------------------------

printf "\n%sRunning failover.sh tests%s\n" "${C}" "${N}"
printf "%s========================================================================%s\n" "${C}" "${N}"

printf "\n%sfailover_init%s\n" "${C}" "${N}"
run_test "init splits hostfile correctly"                     test_init_splits_hostfile
run_test "init errors when nhosts_train > total"              test_init_errors_when_nhosts_train_too_large
run_test "init errors when PBS_NODEFILE unset"                test_init_errors_when_no_pbs_nodefile
run_test "init errors when no arg passed"                     test_init_errors_when_no_arg

printf "\n%sfailover_swap_in%s\n" "${C}" "${N}"
run_test "swap_in replaces bad host with spare"               test_swap_in_replaces_bad_with_spare
run_test "swap_in handles multiple bad nodes"                 test_swap_in_handles_multiple_bad_nodes
run_test "swap_in errors when no spares left"                 test_swap_in_errors_when_no_spares
run_test "swap_in skips host not in active set"               test_swap_in_skips_unknown_host

printf "\n%sfailover_swap_one_blind%s\n" "${C}" "${N}"
run_test "swap_one_blind rotates first active node"           test_swap_one_blind_rotates_first_active
run_test "swap_one_blind errors when no spares"               test_swap_one_blind_errors_when_no_spares

printf "\n%sfailover_run%s\n" "${C}" "${N}"
run_test "run succeeds on first attempt (no retry)"           test_run_succeeds_first_attempt
run_test "run retries on failure, succeeds on attempt 2"      test_run_retries_on_failure_then_succeeds
run_test "run does NOT retry on walltime (143) when clean"    test_run_walltime_143_no_retry_when_clean
run_test "run DOES retry on 143 when log has bad-node pattern" test_run_walltime_143_retries_when_bad_node_pattern_in_log
run_test "run does NOT retry on 143 with only innocent rank-signal lines" test_run_walltime_143_no_retry_when_only_innocent_rank_signals
run_test "run DOES retry on 143 when real hw death is mixed with cascade"  test_run_walltime_143_retries_on_real_hw_death_mixed_with_innocent_cascade
run_test "run handles real Aurora UR_OOM + cascade regression"             test_run_walltime_143_retries_on_real_aurora_ur_oom_with_cascade
run_test "run swaps named bad node (from scraper)"            test_run_swaps_named_bad_node_when_scraper_finds_one
run_test "run exhausts max retries, returns final rc"         test_run_exhausts_max_retries
run_test "run detects inner_rc through ANSI escapes"          test_run_ansi_stripping_in_inner_rc_detection
run_test "run injects topology args when wrapping ezpz launch" test_run_ezpz_launch_injects_topology_args

# ---- Summary ---------------------------------------------------------------

printf "\n%s========================================================================%s\n" "${C}" "${N}"
printf "Summary  (%sPASS=%d%s  %sFAIL=%d%s)\n" \
    "${G}" "${PASS}" "${N}" "${R}" "${FAIL}" "${N}"
printf "%s========================================================================%s\n" "${C}" "${N}"

if (( FAIL > 0 )); then
    printf "\n%sFailed tests:%s\n" "${R}" "${N}"
    printf "  - %s\n" "${FAILED_TESTS[@]}"
    exit 1
fi
printf "\n%sAll %d tests passed. ✅%s\n" "${G}" "${PASS}" "${N}"
exit 0
