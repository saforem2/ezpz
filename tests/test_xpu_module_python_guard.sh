#!/usr/bin/env bash
# test_xpu_module_python_guard.sh — unit tests for
# _ezpz_load_xpu_modules_preserving_python() in src/ezpz/bin/utils.sh.
#
# Run from the repo root:
#   bash tests/test_xpu_module_python_guard.sh
#
# Regression guard for a bug found on Sunspot 2026-08-05: `module load
# oneapi/release` prepends its own Python to PATH and UNSETS
# CONDA_PREFIX, so an already-active conda env is silently discarded --
#
#   conda activate <RC4 frameworks env> -> python3 = <env>/bin/python3
#   module load oneapi/release          -> python3 = /usr/bin/python3
#                                          CONDA_PREFIX = <empty>
#
# ...turning the helpers that PREPARE the XPU stack
# (ezpz_load_modules_{aurora,sunspot}, ezpz_setup_xpu) into ones that
# destroy it.
#
# The subtle part, and the reason for test 3: the guard must NOT key on
# `python3 -c "import torch"` succeeding. On the real frameworks-RC
# stack torch is installed but not yet importable at this point --
# libpti_view.so.0 comes from the pti-gpu module the function is about
# to load. A torch probe fails there, preserves nothing, and lets the
# eviction happen anyway. Key on env activation instead.
#
# `module` is stubbed to emulate the eviction; no real modules needed.

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

# Extract just the function under test (utils.sh's other helpers touch
# real modules/schedulers, so we do not source the whole file).
awk '/^_ezpz_load_xpu_modules_preserving_python\(\) \{/{f=1} f{print} f&&/^\}/{exit}' \
    "${UTILS}" > "${FN_FILE}"
grep -q "module load oneapi" "${FN_FILE}" || {
    printf "%sFATAL%s: extracted function lacks the module load\n" "${R}" "${N}" >&2
    exit 1
}

run_test() {
    local name="$1"; shift
    # Give each test its own TMPDIR and remove it afterwards, so the
    # per-test `mktemp -d` sandboxes don't leak into /tmp. The test body
    # already runs in a subshell, so exporting TMPDIR here cannot bleed
    # between tests.
    local tmp_root
    tmp_root="$(mktemp -d)"
    if ( export TMPDIR="${tmp_root}"; "$@" ) >/dev/null 2>&1; then
        printf "  %s%-56s PASS%s\n" "${G}" "${name}" "${N}"
        PASS=$((PASS + 1))
    else
        printf "  %s%-56s FAIL%s\n" "${R}" "${name}" "${N}"
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("${name}")
    fi
    rm -rf "${tmp_root}"
}

# Build a sandbox: a fake "conda env" python3 and a fake "system"
# python3, plus a `module` stub that emulates the eviction by putting
# the system dir first and unsetting CONDA_PREFIX.
_setup_sandbox() {
    SB="$(mktemp -d)"
    mkdir -p "${SB}/env/bin" "${SB}/sys/bin"
    printf '#!/bin/sh\necho ENV_PYTHON\n'  > "${SB}/env/bin/python3"
    printf '#!/bin/sh\necho SYS_PYTHON\n'  > "${SB}/sys/bin/python3"
    chmod +x "${SB}/env/bin/python3" "${SB}/sys/bin/python3"
    # `module` stub: emulate oneapi evicting the active env.
    module() {
        export PATH="${SB}/sys/bin:${PATH}"
        unset CONDA_PREFIX
    }
    log_message() { :; }
    export PATH="${SB}/env/bin:/usr/bin:/bin"
    # shellcheck disable=SC1090
    source "${FN_FILE}"
}

_which_py() { dirname "$(command -v python3)"; }

# 1. The core contract: an active conda env survives the module load.
test_conda_env_preserved() {
    _setup_sandbox
    export CONDA_PREFIX="${SB}/env"
    _ezpz_load_xpu_modules_preserving_python
    [[ "$(_which_py)" == "${SB}/env/bin" ]] || {
        echo "python3 was evicted: $(command -v python3)"; return 1; }
    # CONDA_PREFIX (unset by the module) must be restored too.
    [[ "${CONDA_PREFIX:-}" == "${SB}/env" ]] || {
        echo "CONDA_PREFIX not restored: ${CONDA_PREFIX:-<empty>}"; return 1; }
}

# 2. Same for a plain virtualenv.
test_virtualenv_preserved() {
    _setup_sandbox
    export VIRTUAL_ENV="${SB}/env"
    _ezpz_load_xpu_modules_preserving_python
    [[ "$(_which_py)" == "${SB}/env/bin" ]] || return 1
    [[ "${VIRTUAL_ENV:-}" == "${SB}/env" ]] || return 1
}

# 3. THE REGRESSION: preservation must not require torch to import.
#    On the frameworks-RC stack torch is installed but unimportable
#    until pti-gpu (loaded by this very function) provides
#    libpti_view.so.0. A torch-probe predicate skipped preservation and
#    let the eviction through -- the original bug.
test_preserved_even_when_torch_unimportable() {
    _setup_sandbox
    # env python that FAILS every import (stands in for the missing .so)
    printf '#!/bin/sh\nexit 1\n' > "${SB}/env/bin/python3"
    chmod +x "${SB}/env/bin/python3"
    export CONDA_PREFIX="${SB}/env"
    _ezpz_load_xpu_modules_preserving_python
    [[ "$(_which_py)" == "${SB}/env/bin" ]] || {
        echo "env evicted because torch did not import (the original bug)"
        return 1; }
}

# 4. No active env -> nothing to preserve; module python wins.
test_no_active_env_leaves_module_python() {
    _setup_sandbox
    unset CONDA_PREFIX VIRTUAL_ENV
    _ezpz_load_xpu_modules_preserving_python
    [[ "$(_which_py)" == "${SB}/sys/bin" ]] || {
        echo "expected module python, got $(command -v python3)"; return 1; }
}

# 5. Idempotent: a second call must not corrupt PATH or the env vars.
test_idempotent() {
    _setup_sandbox
    export CONDA_PREFIX="${SB}/env"
    _ezpz_load_xpu_modules_preserving_python
    _ezpz_load_xpu_modules_preserving_python
    [[ "$(_which_py)" == "${SB}/env/bin" ]] || return 1
    [[ "${CONDA_PREFIX:-}" == "${SB}/env" ]] || return 1
}

# 6. Returns success even when nothing needed restoring, so callers
#    running under `set -e` do not abort.
test_returns_zero() {
    _setup_sandbox
    unset CONDA_PREFIX VIRTUAL_ENV
    _ezpz_load_xpu_modules_preserving_python || return 1
}

printf "\n%s_ezpz_load_xpu_modules_preserving_python%s\n" "${C}" "${N}"
run_test "active conda env survives module load"      test_conda_env_preserved
run_test "active virtualenv survives module load"     test_virtualenv_preserved
run_test "preserved even when torch is unimportable"  test_preserved_even_when_torch_unimportable
run_test "no active env -> module python kept"        test_no_active_env_leaves_module_python
run_test "idempotent across repeated calls"           test_idempotent
run_test "returns 0 when nothing to restore"          test_returns_zero

printf "\n%d passed, %d failed\n" "${PASS}" "${FAIL}"
if ((FAIL > 0)); then
    printf "Failed: %s\n" "${FAILED_TESTS[*]}"
    exit 1
fi
