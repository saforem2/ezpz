#!/usr/bin/env bash
# test_setup_env_no_silent_noop.sh — regression guard for issue #216.
#
# Run from the repo root:
#   bash tests/test_setup_env_no_silent_noop.sh
#
# The bug: on ALCF (scheduler=pbs), ezpz_setup_python did
#
#     ezpz_setup_python_alcf
#     return 0            # <-- unconditional
#
# throwing away every failure the ALCF helper detects. ezpz_setup_env saw
# 0 and printed "[OK] Finished" while NOTHING had been activated, leaving
# python3 as /usr/bin/python3. The job then died much later with
# `ModuleNotFoundError: No module named 'torch'`, far from the cause.
# Reproduced in PBS job 12473470 before fixing.
#
# Two properties are guarded here:
#   1. the pbs/slurm branches PROPAGATE their helper's return code;
#   2. ezpz_assert_python_env_active fails when nothing is active, so a
#      silent no-op is caught even if some path wrongly returns 0.

set -u
unset CDPATH

PASS=0
FAIL=0

if [[ -z "${NO_COLOR:-}" && -t 1 ]]; then
    G=$'\033[1;32m'; R=$'\033[1;31m'; N=$'\033[0m'
else
    G=""; R=""; N=""
fi

UTILS="${UTILS:-src/ezpz/bin/utils.sh}"
[[ -f "${UTILS}" ]] || { printf "%sFATAL%s: %s not found (run from repo root)\n" "${R}" "${N}" "${UTILS}" >&2; exit 1; }

FN_FILE="$(mktemp)"
trap 'rm -f "${FN_FILE}"' EXIT
awk '/^ezpz_assert_python_env_active\(\) \{/{f=1} f{print} f&&/^\}/{exit}' \
    "${UTILS}" > "${FN_FILE}"
grep -q "VIRTUAL_ENV" "${FN_FILE}" || {
    printf "%sFATAL%s: could not extract ezpz_assert_python_env_active\n" "${R}" "${N}" >&2
    exit 1
}

# Echo the next non-blank line after the first line that is exactly a
# bare call to the given function (leading tabs stripped).
_line_after() {
    awk -v pat="$2" '
        found && NF { gsub(/^[ \t]+/, ""); print; exit }
        { line=$0; gsub(/^[ \t]+/, "", line); if (line ~ "^" pat) found=1 }
    ' <<< "$1"
}

run_test() {
    local name="$1"; shift
    if ( "$@" ) >/dev/null 2>&1; then
        printf "  %s%-58s PASS%s\n" "${G}" "${name}" "${N}"; PASS=$((PASS + 1))
    else
        printf "  %s%-58s FAIL%s\n" "${R}" "${name}" "${N}"; FAIL=$((FAIL + 1))
    fi
}

# --- the assertion helper ---------------------------------------------

# Nothing active + system python => must FAIL (this is the #216 state).
t_nothing_active_fails() {
    # shellcheck disable=SC1090
    source "${FN_FILE}"
    log_message() { :; }
    unset VIRTUAL_ENV CONDA_PREFIX
    PATH="/usr/bin:/bin"
    ! ezpz_assert_python_env_active
}

t_venv_active_passes() {
    source "${FN_FILE}"
    log_message() { :; }
    unset CONDA_PREFIX
    VIRTUAL_ENV="/tmp/some/venv"
    ezpz_assert_python_env_active
}

t_conda_active_passes() {
    source "${FN_FILE}"
    log_message() { :; }
    unset VIRTUAL_ENV
    CONDA_PREFIX="/tmp/some/conda"
    ezpz_assert_python_env_active
}

# A module-provided python3 (not the system one) is a legitimate setup
# even with no venv/conda -- e.g. `module load frameworks/2026.1.0`.
t_module_python_passes() {
    source "${FN_FILE}"
    log_message() { :; }
    unset VIRTUAL_ENV CONDA_PREFIX
    local d; d="$(mktemp -d)"
    printf '#!/bin/sh\n' > "${d}/python3"; chmod +x "${d}/python3"
    PATH="${d}:/usr/bin:/bin"
    local rc=0; ezpz_assert_python_env_active || rc=1
    rm -rf "${d}"
    return "${rc}"
}

t_error_names_the_cause() {
    source "${FN_FILE}"
    local out=""
    log_message() { out+="$* "; }
    unset VIRTUAL_ENV CONDA_PREFIX
    PATH="/usr/bin:/bin"
    ezpz_assert_python_env_active 2>/dev/null || true
    # Must name the fix, not just report failure.
    [[ "${out}" == *"frameworks"* && "${out}" == *"module load"* ]]
}

# --- the call sites ----------------------------------------------------

# Guard the actual bug: the pbs/slurm branches must not hardcode return 0.
t_pbs_branch_propagates() {
    local body
    body="$(awk '/^ezpz_setup_python\(\) \{/{f=1} f{print} f&&/^\}/{exit}' "${UTILS}")"
    [[ "${body}" == *"ezpz_setup_python_alcf"* ]] || return 1
    # The old shape was: bare call, then an unconditional `return 0` on
    # the very next line. Portable check (no grep -P, which is GNU-only).
    [[ "$(_line_after "${body}" 'ezpz_setup_python_alcf$')" != "return 0" ]]
}

t_slurm_branch_propagates() {
    local body
    body="$(awk '/^ezpz_setup_python\(\) \{/{f=1} f{print} f&&/^\}/{exit}' "${UTILS}")"
    [[ "$(_line_after "${body}" 'ezpz_setup_python_nersc$')" != "return 0" ]]
}

t_both_branches_assert() {
    local body
    body="$(awk '/^ezpz_setup_python\(\) \{/{f=1} f{print} f&&/^\}/{exit}' "${UTILS}")"
    [[ "$(grep -c 'ezpz_assert_python_env_active' <<< "${body}")" -ge 2 ]]
}

printf "\ntest_setup_env_no_silent_noop.sh (issue #216)\n\n"
run_test "nothing active + system python -> FAILS"      t_nothing_active_fails
run_test "active venv -> passes"                        t_venv_active_passes
run_test "active conda -> passes"                       t_conda_active_passes
run_test "module-provided python3 -> passes"            t_module_python_passes
run_test "error message names frameworks + module load" t_error_names_the_cause
run_test "pbs branch propagates its return code"        t_pbs_branch_propagates
run_test "slurm branch propagates its return code"      t_slurm_branch_propagates
run_test "both branches assert the post-condition"      t_both_branches_assert

printf "\n  %s%d passed%s, %s%d failed%s\n\n" "${G}" "${PASS}" "${N}" "${R}" "${FAIL}" "${N}"
[[ "${FAIL}" -eq 0 ]]
