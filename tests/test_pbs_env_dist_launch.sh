#!/usr/bin/env bash
# Regression: ezpz_get_pbs_env must export the launch command it computes.
#
# The function computed `dist_launch_cmd` and then ran
#
#     export DIST_LAUNCH="${DIST_LAUNCH}"
#
# exporting the variable from ITSELF -- so the freshly computed command was
# discarded and DIST_LAUNCH kept whatever it already held (empty, on a
# clean shell). `launch` then expanded to nothing. num_hosts,
# num_gpus_per_host and num_gpus were computed and dropped the same way.
#
# It hid for two reasons:
#   * the usual caller (ezpz_get_job_env) re-derives everything from the
#     hostfile straight afterwards, masking the loss; and
#   * the branch is unreachable from a login node -- the
#     hostname-in-hostfile check fails first (rc=1) -- so it only misbehaves
#     inside a real allocation.
#
# This test stubs the scheduler so the branch is exercised without one.

set -eu
unset CDPATH

UTILS="${UTILS:-src/ezpz/bin/utils.sh}"
[[ -f "${UTILS}" ]] || {
    printf "FATAL: %s not found (run from repo root)\n" "${UTILS}" >&2
    exit 1
}

extract_function() {
    local name="$1"
    awk -v signature="${name}() {" '
        $0 == signature { found=1 }
        found { print }
        found && /^}/ { exit }
    ' "${UTILS}"
}

FN_FILE="$(mktemp)"
WORK="$(mktemp -d)"
trap 'rm -rf "${FN_FILE}" "${WORK}"' EXIT

extract_function ezpz_get_pbs_env >>"${FN_FILE}"

# Fail closed: an empty extraction would make every assertion vacuous.
grep -q '^ezpz_get_pbs_env() {$' "${FN_FILE}" || {
    printf "failed to extract ezpz_get_pbs_env from %s\n" "${UTILS}" >&2
    exit 1
}

# The stub hostfile must contain this host, or the function returns 1
# before reaching the code under test.
hostname >"${WORK}/hostfile"
printf 'other-host\n' >>"${WORK}/hostfile"

FAKE_LAUNCH="mpiexec -n 8 --ppn 4 --hostfile ${WORK}/hostfile"

run_pbs_env() {
    bash -c "
        log_message() { :; }
        ezpz_get_machine_name()             { echo 'aurora'; }
        ezpz_get_scheduler_type()           { echo 'pbs'; }
        ezpz_get_pbs_jobid()                { echo '12345'; }
        ezpz_get_jobenv_file()              { echo '${WORK}/jobenv'; }
        ezpz_get_num_hosts()                { echo 2; }
        ezpz_get_num_gpus_per_host()        { echo 4; }
        ezpz_get_dist_launch_cmd()          { echo '${FAKE_LAUNCH}'; }
        ezpz_get_pbs_nodefile_from_hostname() { echo '${WORK}/hostfile'; }
        BLUE=''; RESET=''
        source '${FN_FILE}'
        # Clean shell: nothing pre-set, so a self-reference yields empty.
        unset DIST_LAUNCH LAUNCH ezlaunch NHOSTS NGPU_PER_HOST NGPUS
        ezpz_get_pbs_env '${WORK}/hostfile' '${WORK}/jobenv' >/dev/null 2>&1
        printf 'rc=%s DIST_LAUNCH=[%s] ezlaunch=[%s] NHOSTS=[%s] NGPUS=[%s]' \
            \"\$?\" \"\${DIST_LAUNCH:-}\" \"\${ezlaunch:-}\" \
            \"\${NHOSTS:-}\" \"\${NGPUS:-}\"
    " 2>&1
}

FAILURES=0
check() {
    local label="$1" expected="$2" actual="$3"
    if [[ "${expected}" == "${actual}" ]]; then
        printf "  ok   %s\n" "${label}"
    else
        printf "  FAIL %s\n       expected: %s\n       actual:   %s\n" \
            "${label}" "${expected}" "${actual}" >&2
        FAILURES=$((FAILURES + 1))
    fi
}

OUT="$(run_pbs_env)"

printf "1. the computed launch command is exported, not discarded\n"
check "DIST_LAUNCH" \
    "rc=0 DIST_LAUNCH=[${FAKE_LAUNCH}] ezlaunch=[${FAKE_LAUNCH}] NHOSTS=[2] NGPUS=[8]" \
    "${OUT}"

printf "2. DIST_LAUNCH is not empty (the exact symptom of the bug)\n"
if printf '%s' "${OUT}" | grep -q 'DIST_LAUNCH=\[\]'; then
    printf "  FAIL DIST_LAUNCH came back empty -- the computed command was dropped\n" >&2
    FAILURES=$((FAILURES + 1))
else
    printf "  ok   DIST_LAUNCH is populated\n"
fi

printf "3. ezlaunch mirrors DIST_LAUNCH\n"
if printf '%s' "${OUT}" | grep -q "ezlaunch=\[${FAKE_LAUNCH}\]"; then
    printf "  ok   ezlaunch matches\n"
else
    printf "  FAIL ezlaunch did not mirror DIST_LAUNCH\n" >&2
    FAILURES=$((FAILURES + 1))
fi

if ((FAILURES > 0)); then
    printf "\n%d assertion(s) failed\n" "${FAILURES}" >&2
    exit 1
fi

printf "\nezpz_get_pbs_env exports the launch command and topology it computes\n"
