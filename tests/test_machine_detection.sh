#!/usr/bin/env bash
# test_machine_detection.sh — unit tests for ezpz_get_machine_name()
# in src/ezpz/bin/utils.sh.
#
# Run from the repo root:
#   bash tests/test_machine_detection.sh
#
# Regression guard for a Sirius bug found 2026-09-16: the real login
# node is `sirius-uan-0002`, but detection only recognised Sirius as an
# `x3*` COMPUTE node carrying `sirius` in $PBS_O_HOST. A login node
# therefore matched no arm and fell through to `*) machine="${mn}"`,
# returning the raw hostname -- so `ezpz_load_modules` errored with
# "Supported: aurora, sunspot, polaris" and `ezpz_setup_conda` fell back
# to installing micromamba, on the very node the user was sitting on.
#
# The ordering is the subtle part, and the reason for the `uan-*` case:
# the `uan*` arm maps to SUNSPOT (Sunspot login nodes are `uan-*`), so
# the new `sirius*` arm has to be matched BEFORE it. These two machines
# share the `<machine>-uan-N` login-node shape -- cf. `aurora-uan-0009`
# in docs/cli/submit.md -- so a too-broad glob on either side silently
# hands jobs to the wrong machine's module stack.
#
# `hostname` is stubbed; no real cluster needed.

set -u
unset CDPATH

PASS=0
FAIL=0
FAILED_TESTS=()

if [[ -z "${NO_COLOR:-}" && -t 1 ]]; then
    G=$'\033[1;32m'; R=$'\033[1;31m'; C=$'\033[1;36m'; N=$'\033[0m'
else
    G=""; R=""; C=""; N=""
fi

UTILS="${UTILS:-src/ezpz/bin/utils.sh}"
[[ -f "${UTILS}" ]] || { printf "%sFATAL%s: %s not found (run from repo root)\n" "${R}" "${N}" "${UTILS}" >&2; exit 1; }

FN_FILE="$(mktemp)"
trap 'rm -f "${FN_FILE}"' EXIT

# Extract just the function under test -- sourcing all of utils.sh would
# touch real modules and schedulers.
awk '/^ezpz_get_machine_name\(\) \{/{f=1} f{print} f&&/^\}/{exit}' \
    "${UTILS}" > "${FN_FILE}"
grep -q 'sirius\*)' "${FN_FILE}" || {
    printf "%sFATAL%s: extracted function lacks the sirius arm\n" "${R}" "${N}" >&2
    exit 1
}

run_test() {
    local name="$1"; shift
    if ( "$@" ) >/dev/null 2>&1; then
        printf "  %s%-56s PASS%s\n" "${G}" "${name}" "${N}"
        PASS=$((PASS + 1))
    else
        printf "  %s%-56s FAIL%s\n" "${R}" "${name}" "${N}"
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("${name}")
    fi
}

# Assert ezpz_get_machine_name prints $2 for hostname $1, with $3 (if
# given) exported as PBS_O_HOST.
expect_machine() {
    local host="$1" want="$2" pbs_o_host="${3:-}"
    local got
    got="$(
        hostname() { echo "${host}"; }
        if [[ -n "${pbs_o_host}" ]]; then
            export PBS_O_HOST="${pbs_o_host}"
        else
            unset PBS_O_HOST
        fi
        # shellcheck disable=SC1090
        source "${FN_FILE}"
        ezpz_get_machine_name
    )"
    [[ "${got}" == "${want}" ]] || {
        printf 'hostname=%s PBS_O_HOST=%s: want %s, got %s\n' \
            "${host}" "${pbs_o_host:-<unset>}" "${want}" "${got}" >&2
        return 1
    }
}

# ── Sirius: the regression these tests exist for ────────────────────
test_sirius_login_node()      { expect_machine "sirius-uan-0002" "sirius"; }
test_sirius_login_fqdn()      { expect_machine "sirius-uan-0002.sirius.alcf.anl.gov" "sirius"; }
test_sirius_compute_node()    { expect_machine "x3001c0s7b0n0" "sirius" "sirius-uan-0002"; }

# ── The uan-* collision the sirius arm must not break ───────────────
test_sunspot_uan_login()      { expect_machine "uan-0001" "sunspot"; }
test_sunspot_compute_node()   { expect_machine "x1002c0s3b0n0" "sunspot"; }
test_sunspot_named_login()    { expect_machine "sunspot-login-1" "sunspot"; }

# ── Polaris shares x3* with Sirius; only PBS_O_HOST separates them ──
test_polaris_compute_node()   { expect_machine "x3001c0s7b0n0" "polaris"; }
test_polaris_login_node()     { expect_machine "polaris-login-01" "polaris"; }

# ── Everything else must be untouched by the new arm ────────────────
test_aurora()                 { expect_machine "x4105c0s4b0n0" "aurora"; }
test_sophia()                 { expect_machine "sophia-n02" "sophia"; }
test_frontier()               { expect_machine "frontier-login01" "frontier"; }
test_perlmutter()             { expect_machine "nid00123" "perlmutter"; }
test_unknown_falls_through()  { expect_machine "myworkstation" "myworkstation"; }

# Hostnames are lowercased before matching.
test_uppercase_is_folded()    { expect_machine "SIRIUS-UAN-0002" "sirius"; }

printf "\n%sezpz_get_machine_name%s\n" "${C}" "${N}"
run_test "sirius login node (sirius-uan-NNNN)"        test_sirius_login_node
run_test "sirius login node, FQDN form"               test_sirius_login_fqdn
run_test "sirius compute node via PBS_O_HOST"         test_sirius_compute_node
run_test "sunspot uan-* login still sunspot"          test_sunspot_uan_login
run_test "sunspot compute node (x1*)"                 test_sunspot_compute_node
run_test "sunspot named login node"                   test_sunspot_named_login
run_test "polaris compute node (no PBS_O_HOST)"       test_polaris_compute_node
run_test "polaris login node"                         test_polaris_login_node
run_test "aurora compute node (x4*)"                  test_aurora
run_test "sophia"                                     test_sophia
run_test "frontier"                                   test_frontier
run_test "perlmutter (nid*)"                          test_perlmutter
run_test "unknown host returns raw hostname"          test_unknown_falls_through
run_test "hostname case is folded"                    test_uppercase_is_folded

printf "\n%d passed, %d failed\n" "${PASS}" "${FAIL}"
if ((FAIL > 0)); then
    printf "Failed: %s\n" "${FAILED_TESTS[*]}"
    exit 1
fi
