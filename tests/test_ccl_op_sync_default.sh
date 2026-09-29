#!/usr/bin/env bash
# Regression tests for the shared Intel XPU oneCCL environment policy.
#
# Two things are under test:
#
#  1. CCL_OP_SYNC defaults to 1 when the caller has not chosen. This is a
#     CORRECTNESS default: on oneCCL 2022.x (frameworks/2026.1.0),
#     FSDP2 + TP>1 hangs without it -- 20/20 completions with
#     CCL_OP_SYNC=1 vs 2/20 without, measured on Sunspot with arms
#     alternating inside single allocations (ezpz #252). The hang is
#     silent: no error, no traceback, no watchdog.
#
#     The competing evidence is throughput: a 64-node / 768-rank
#     TorchTitan run was ~27x slower synchronous. That is why the default
#     is only a DEFAULT -- see (2).
#
#  2. An EXPLICIT caller value must survive `module load`, in BOTH
#     directions. `export CCL_OP_SYNC=0` is the async opt-in for
#     workloads that have measured it (Aurora job 8854547, 30/30), and
#     must not be silently replaced by the default above or by a
#     modulefile. The oneAPI modulefile is outside this repo's control
#     and can set or unset arbitrary variables.
#
# The module stub below is therefore HOSTILE: it mutates CCL_OP_SYNC the
# way a real modulefile could. An earlier revision of this file stubbed
# `module() { :; }`, which made every assertion pass vacuously -- it
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
extract_function ezpz_setup_conda_aurora >>"${FN_FILE}"
extract_function ezpz_setup_conda_sunspot >>"${FN_FILE}"

# The conda helpers are on the RECOMMENDED path
# (ezpz_setup_env -> ezpz_setup_python_alcf -> ezpz_setup_conda_*), which
# reaches none of the ezpz_load_modules_*/ezpz_setup_xpu helpers. Covering
# only the latter left the standard flow async -- i.e. hanging. Review
# caught this on #258; the suite now pins every entry point.
SETUP_FNS=(
    ezpz_load_modules_aurora
    ezpz_load_modules_sunspot
    ezpz_setup_xpu
    ezpz_setup_conda_aurora
    ezpz_setup_conda_sunspot
)

# What an unset CCL_OP_SYNC becomes after setup: the synchronous default.
unset_default() {
    printf '1'
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
    check "${fn} unset -> 1 (default)" "$(unset_default "${fn}")" "$(ccl_state)"

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
    check "${fn} unset -> 1 (default; module injected 1)" "$(unset_default "${fn}")" "$(ccl_state)"

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
