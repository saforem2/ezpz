#!/usr/bin/env bash
# Regression tests for the shared Intel XPU oneCCL environment policy.
#
# Two things are under test:
#
#  1. ezpz does not choose CCL_OP_SYNC. Collective semantics belong to the
#     application, and the setup helpers preserve the caller's exact set/unset
#     state. Matched 4-node TorchTitan controls found async about 6x slower on
#     both Aurora and Sunspot, so that application explicitly chooses `1`.
#
#  2. An EXPLICIT caller value must survive `module load`. This permits other
#     applications to select `0` or `1` without ezpz silently replacing it.
#     The oneAPI modulefile is outside this repo's control and can set or unset
#     arbitrary variables. If it ever injects CCL_OP_SYNC, a caller's
#     explicit 0 is silently overwritten and application policy is changed --
#     the exact regression this PR exists to prevent.
#
# The module stub below is therefore HOSTILE: it mutates CCL_OP_SYNC the
# way a real modulefile could. An earlier revision of this file stubbed
# `module() { :; }`, which made every assertion pass vacuously -- it
# could not observe the bug it was written to catch.

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
trap 'rm -f "${FN_FILE}"' EXIT

extract_function _ezpz_load_xpu_modules_preserving_python >>"${FN_FILE}"
extract_function ezpz_load_modules_aurora >>"${FN_FILE}"
extract_function ezpz_load_modules_sunspot >>"${FN_FILE}"
extract_function ezpz_setup_xpu >>"${FN_FILE}"

SETUP_FNS=(ezpz_load_modules_aurora ezpz_load_modules_sunspot ezpz_setup_xpu)

unset_default() {
    printf '<unset>'
}

# Fail closed: if extraction silently produced nothing (renamed function,
# reformatted brace style), every assertion below would pass against an
# empty file.
for fn in _ezpz_load_xpu_modules_preserving_python "${SETUP_FNS[@]}"; do
    grep -q "^${fn}() {$" "${FN_FILE}" || {
        printf "failed to extract %s from %s\n" "${fn}" "${UTILS}" >&2
        exit 1
    }
done

log_message() { :; }

# The variable the hostile stub writes on each `module load`. Tests set
# this to simulate a modulefile that injects, overwrites, or clears
# CCL_OP_SYNC. Empty string means "stub unsets it".
MODULE_INJECTS=""
MODULE_ACTION="set" # set | unset | noop
module() {
    case "${MODULE_ACTION}" in
        set) export CCL_OP_SYNC="${MODULE_INJECTS}" ;;
        unset) unset CCL_OP_SYNC ;;
        noop) : ;;
    esac
}

# shellcheck disable=SC1090
source "${FN_FILE}"

FAILURES=0
check() {
    local label="$1" expected="$2" actual="$3"
    if [[ "${expected}" == "${actual}" ]]; then
        printf "  ok   %s\n" "${label}"
    else
        printf "  FAIL %s: expected %s, got %s\n" \
            "${label}" "${expected}" "${actual}" >&2
        FAILURES=$((FAILURES + 1))
    fi
}

# Render CCL_OP_SYNC as a value or the literal <unset>, so the two states
# are distinguishable in assertions ("" and unset are NOT the same).
ccl_state() { printf '%s' "${CCL_OP_SYNC-<unset>}"; }

printf "1. module load does not touch CCL_OP_SYNC\n"
MODULE_ACTION="noop"
for fn in "${SETUP_FNS[@]}"; do
    unset CCL_OP_SYNC CONDA_PREFIX VIRTUAL_ENV
    "${fn}"
    check "${fn} unset -> unset" "$(unset_default "${fn}")" "$(ccl_state)"

    export CCL_OP_SYNC=0
    "${fn}"
    check "${fn} 0 -> 0" "0" "$(ccl_state)"

    export CCL_OP_SYNC=1
    "${fn}"
    check "${fn} 1 -> 1" "1" "$(ccl_state)"
done

printf "2. module load INJECTS CCL_OP_SYNC=1 (the leak this guards)\n"
MODULE_ACTION="set"
MODULE_INJECTS="1"
for fn in "${SETUP_FNS[@]}"; do
    # Initially unset must stay unset: ezpz must not let a modulefile
    # smuggle in a synchronous default it no longer sets itself.
    unset CCL_OP_SYNC CONDA_PREFIX VIRTUAL_ENV
    "${fn}"
    check "${fn} unset -> unset (module injected 1)" "$(unset_default "${fn}")" "$(ccl_state)"

    # An explicit 0 is the async opt-in verified on Aurora job 8854547.
    # Losing it here is the failure mode that matters most.
    export CCL_OP_SYNC=0
    "${fn}"
    check "${fn} 0 -> 0 (module injected 1)" "0" "$(ccl_state)"

    export CCL_OP_SYNC=1
    "${fn}"
    check "${fn} 1 -> 1 (module injected 1)" "1" "$(ccl_state)"
done

printf "3. module load OVERWRITES with 0 / clears the variable\n"
MODULE_ACTION="set"
MODULE_INJECTS="0"
for fn in "${SETUP_FNS[@]}"; do
    export CCL_OP_SYNC=1
    "${fn}"
    check "${fn} 1 -> 1 (module injected 0)" "1" "$(ccl_state)"
done

MODULE_ACTION="unset"
for fn in "${SETUP_FNS[@]}"; do
    # A modulefile that UNSETS it must not erase an explicit caller value.
    export CCL_OP_SYNC=1
    "${fn}"
    check "${fn} 1 -> 1 (module unset it)" "1" "$(ccl_state)"

    export CCL_OP_SYNC=0
    "${fn}"
    check "${fn} 0 -> 0 (module unset it)" "0" "$(ccl_state)"

    unset CCL_OP_SYNC
    "${fn}"
    check "${fn} unset -> unset (module unset it)" "$(unset_default "${fn}")" "$(ccl_state)"
done

printf "4. empty string is preserved as empty, not treated as unset\n"
MODULE_ACTION="set"
MODULE_INJECTS="1"
for fn in "${SETUP_FNS[@]}"; do
    export CCL_OP_SYNC=""
    "${fn}"
    check "${fn} '' -> ''" "" "$(ccl_state)"
done

if ((FAILURES > 0)); then
    printf "\n%d assertion(s) failed\n" "${FAILURES}" >&2
    exit 1
fi

printf "\nshared XPU setup preserves exact caller CCL_OP_SYNC state across module load\n"
