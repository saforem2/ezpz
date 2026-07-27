#!/usr/bin/env bash
# test_ezpz_setup_venv.sh — unit tests for ezpz_setup()'s Flow B (venv-arg)
# auto-create behavior in src/ezpz/bin/utils.sh.
#
# Run from the repo root:
#   bash tests/test_ezpz_setup_venv.sh
#
# ezpz_setup() is embedded in a 3000-line utils.sh whose other helpers touch
# real modules/schedulers, so instead of sourcing the whole file we EXTRACT
# just the ezpz_setup function body and run it against STUBS for its
# dependencies (ezpz_setup_job, ezpz_load_modules, log_message, ezpz_realpath)
# plus a fake `uv` on PATH. This isolates the branching logic:
#
#   1. missing venv + uv present        -> auto-creates, activates
#   2. missing venv + EZPZ_NO_AUTO_VENV -> errors, does NOT create
#   3. missing venv + no uv on PATH     -> errors cleanly
#   4. existing venv                     -> activates, does NOT re-create
#   5. modules loaded BEFORE the venv check (ordering guard)
#
# Each test runs in its own temp dir + subshell.

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

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${TEST_DIR}/.." && pwd)"
UTILS="${REPO_ROOT}/src/ezpz/bin/utils.sh"
if [[ ! -f "${UTILS}" ]]; then
    printf "%sFATAL%s: utils.sh not found at %s\n" "${R}" "${N}" "${UTILS}" >&2
    exit 1
fi

# Extract just the `ezpz_setup() { ... }` function into a temp file we can
# source in isolation. awk from the def line to the first line that is exactly
# a closing brace at column 0.
EZPZ_SETUP_FN="$(mktemp "/tmp/ezpz-setup-fn-XXXXXX.sh")"
awk '/^ezpz_setup\(\) \{/{f=1} f{print} f&&/^\}/{exit}' "${UTILS}" > "${EZPZ_SETUP_FN}"
if ! grep -q "EZPZ_NO_AUTO_VENV" "${EZPZ_SETUP_FN}"; then
    printf "%sFATAL%s: extracted ezpz_setup lacks the auto-venv logic\n" "${R}" "${N}" >&2
    rm -f "${EZPZ_SETUP_FN}"; exit 1
fi

# ---- stubs shared by all tests --------------------------------------------
# A preamble sourced before the function so its dependencies exist. Records
# call order in $ORDER_LOG so we can assert modules load before the venv check.
_make_preamble() {
    cat <<'STUB'
# color/format vars the function references
BRIGHT_GREEN=""; RESET=""; CYAN=""; GREEN=""; RED=""
log_message() { :; }                 # silence; tests assert on state not logs
ezpz_realpath() { ( cd "$(dirname "$1")" 2>/dev/null && printf '%s/%s' "$(pwd)" "$(basename "$1")" ); }
ezpz_setup_job()   { printf 'job\n'  >> "${ORDER_LOG}"; return 0; }
ezpz_load_modules(){ printf 'mods\n' >> "${ORDER_LOG}"; return 0; }
STUB
}

# A fake `uv` that actually creates a minimal venv (bin/activate) so the
# post-create activation succeeds. Records that it ran.
_make_fake_uv() {
    local dir="$1"
    cat > "${dir}/uv" <<'UVEOF'
#!/usr/bin/env bash
# fake uv: parse the final arg as the venv dir, create bin/activate
printf 'uv-ran\n' >> "${ORDER_LOG}"
venv="${@: -1}"
mkdir -p "${venv}/bin"
printf '# fake activate\n' > "${venv}/bin/activate"
exit 0
UVEOF
    chmod +x "${dir}/uv"
}

run_test() {
    local name="$1" fn="$2"
    printf "  %s%-55s%s " "${C}" "${name}" "${N}"
    local tmpdir output
    tmpdir=$(mktemp -d "/tmp/ezpz-setup-test-XXXXXX")
    if output=$(
        cd "${tmpdir}"
        export ORDER_LOG="${tmpdir}/order.log"
        : > "${ORDER_LOG}"
        # shellcheck disable=SC1090
        source <(_make_preamble)
        # shellcheck disable=SC1090
        source "${EZPZ_SETUP_FN}"
        "${fn}"
    ) 2>&1; then
        PASS=$((PASS+1)); printf "%sPASS%s\n" "${G}" "${N}"
    else
        FAIL=$((FAIL+1)); FAILED_TESTS+=("${name}")
        printf "%sFAIL%s\n" "${R}" "${N}"
        printf "%s\n" "${output}" | sed 's/^/      /'
    fi
    rm -rf "${tmpdir}"
}

assert() { # assert <cond-cmd...>; message via $MSG
    if ! "$@"; then printf "ASSERT FAIL: %s\n" "$*" >&2; exit 1; fi
}
assert_file() { [[ -f "$1" ]] || { printf "ASSERT FAIL: missing file %s\n" "$1" >&2; exit 1; }; }
assert_no_file() { [[ ! -f "$1" ]] || { printf "ASSERT FAIL: unexpected file %s\n" "$1" >&2; exit 1; }; }
assert_contains_file() { grep -q "$2" "$1" || { printf "ASSERT FAIL: %s not in %s\n" "$2" "$1" >&2; exit 1; }; }

# ---- tests -----------------------------------------------------------------

test_auto_create_when_missing() {
    _make_fake_uv "${tmpdir}"; export PATH="${tmpdir}:${PATH}"
    unset EZPZ_NO_AUTO_VENV
    ezpz_setup ".venv" || return 1
    assert_file "${tmpdir}/.venv/bin/activate"     # created
    assert_contains_file "${ORDER_LOG}" "uv-ran"   # uv actually invoked
}

test_no_auto_venv_opt_out() {
    _make_fake_uv "${tmpdir}"; export PATH="${tmpdir}:${PATH}"
    export EZPZ_NO_AUTO_VENV=1
    if ezpz_setup ".venv"; then
        printf "ASSERT FAIL: expected nonzero exit under EZPZ_NO_AUTO_VENV=1\n" >&2
        return 1
    fi
    assert_no_file "${tmpdir}/.venv/bin/activate"  # NOT created
    ! grep -q "uv-ran" "${ORDER_LOG}" || { printf "ASSERT FAIL: uv ran despite opt-out\n" >&2; return 1; }
}

test_no_uv_available_errors() {
    # No fake uv on PATH, and neutralize any real uv so creation can't happen.
    unset EZPZ_NO_AUTO_VENV
    uv() { return 127; }; export -f uv 2>/dev/null || true
    # Also shadow command -v uv by putting a PATH without uv is hard; instead
    # rely on the function's `command -v uv` — so ensure no uv on a clean PATH.
    export PATH="/usr/bin:/bin"
    if command -v uv >/dev/null 2>&1; then
        printf "SKIP: uv present on minimal PATH; cannot test no-uv branch\n" >&2
        return 0
    fi
    if ezpz_setup ".venv"; then
        printf "ASSERT FAIL: expected nonzero exit when uv missing\n" >&2
        return 1
    fi
    assert_no_file "${tmpdir}/.venv/bin/activate"
}

test_existing_venv_not_recreated() {
    _make_fake_uv "${tmpdir}"; export PATH="${tmpdir}:${PATH}"
    unset EZPZ_NO_AUTO_VENV
    mkdir -p "${tmpdir}/.venv/bin"
    printf '# preexisting\n' > "${tmpdir}/.venv/bin/activate"
    ezpz_setup ".venv" || return 1
    # uv must NOT have run (venv already existed)
    ! grep -q "uv-ran" "${ORDER_LOG}" || { printf "ASSERT FAIL: uv ran on existing venv\n" >&2; return 1; }
    assert_contains_file "${tmpdir}/.venv/bin/activate" "preexisting"  # untouched
}

test_modules_loaded_before_venv_check() {
    _make_fake_uv "${tmpdir}"; export PATH="${tmpdir}:${PATH}"
    unset EZPZ_NO_AUTO_VENV
    ezpz_setup ".venv" || return 1
    # order.log must show job + mods BEFORE uv-ran (creation)
    local order; order="$(tr '\n' ',' < "${ORDER_LOG}")"
    case "${order}" in
        job,mods,uv-ran,*) : ;;  # correct ordering
        *) printf "ASSERT FAIL: bad order %q (want job,mods,uv-ran)\n" "${order}" >&2; return 1 ;;
    esac
}

# ---- run -------------------------------------------------------------------
printf "\n%sezpz_setup() Flow B auto-venv tests%s\n" "${C}" "${N}"
run_test "auto-create when missing"          test_auto_create_when_missing
run_test "EZPZ_NO_AUTO_VENV opt-out"         test_no_auto_venv_opt_out
run_test "no uv available -> clean error"    test_no_uv_available_errors
run_test "existing venv not recreated"       test_existing_venv_not_recreated
run_test "modules load before venv check"    test_modules_loaded_before_venv_check

rm -f "${EZPZ_SETUP_FN}"
printf "\n%d passed, %d failed\n" "${PASS}" "${FAIL}"
if (( FAIL > 0 )); then
    printf "%sFAILED:%s %s\n" "${R}" "${N}" "${FAILED_TESTS[*]}"
    exit 1
fi
exit 0
