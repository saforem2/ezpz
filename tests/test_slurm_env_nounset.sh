#!/usr/bin/env bash
# Regression: ezpz job setup must survive `set -u` on the SLURM path.
#
# ezpz_get_slurm_env set NHOSTS and NGPU_PER_HOST but NOT NGPUS, while the
# PBS path (ezpz_get_pbs_env) exported all three. ezpz_setup_host_slurm
# then ran `export NGPUS="${NGPUS}"` -- a self-reference that is a silent
# no-op normally, but under `set -u` aborts the shell:
#
#   utils.sh: line 2741: NGPUS: unbound variable
#
# A job script using `set -u` (entirely reasonable, and what our own
# example sbatch files do) died mid-setup with no NHOSTS/NGPUS and a
# misleading exit code. Measured on Perlmutter compute node nid008349:
#
#   without set -u : rc=0,   NHOSTS=2
#   with    set -u : rc=127, NHOSTS=unset
#
# It does NOT reproduce on a login node -- only inside a real allocation,
# where ezpz_get_slurm_env actually runs. That is why this test stubs the
# scheduler rather than requiring one.

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

extract_function ezpz_get_slurm_env >>"${FN_FILE}"

# Fail closed: an empty extraction would make every assertion vacuous.
grep -q '^ezpz_get_slurm_env() {$' "${FN_FILE}" || {
    printf "failed to extract ezpz_get_slurm_env from %s\n" "${UTILS}" >&2
    exit 1
}

# Two-host stub nodefile, so NHOSTS=2 like the measured Perlmutter run.
printf 'nid008349\nnid008350\n' >"${WORK}/nodefile"

run_case() {
    # $1: "nounset" | "lax"  -- returns rc, prints NGPUS state
    local mode="$1"
    local guard=""
    [[ "${mode}" == "nounset" ]] && guard="set -u"
    bash -c "
        ${guard}
        log_message() { :; }
        ezpz_get_slurm_running_jobid() { echo 58828088; }
        ezpz_make_slurm_nodefile()     { echo '${WORK}/nodefile'; }
        ezpz_get_num_gpus_per_host()   { echo 4; }
        source '${FN_FILE}'
        ezpz_get_slurm_env >/dev/null 2>&1 || exit \$?
        # The variable whose absence caused the abort downstream.
        printf 'NHOSTS=%s NGPU_PER_HOST=%s NGPUS=%s\n' \
            \"\${NHOSTS:-<unset>}\" \"\${NGPU_PER_HOST:-<unset>}\" \"\${NGPUS:-<unset>}\"
    " 2>&1
}

FAILURES=0
# BSD `wc -l` pads its count with leading whitespace, so on macOS NHOSTS
# comes back as '       2' where GNU coreutils gives '2'. That is a
# property of the test host, not of ezpz: the real Perlmutter run logs a
# clean `NHOSTS=2`. Normalise the whitespace rather than "fixing" a
# product that is not broken.
norm() { printf '%s' "$*" | tr -s ' ' | sed 's/= /=/g'; }
check() {
    local label="$1" expected="$2" actual="$3"
    expected="$(norm "${expected}")"
    actual="$(norm "${actual}")"
    if [[ "${expected}" == "${actual}" ]]; then
        printf "  ok   %s\n" "${label}"
    else
        printf "  FAIL %s\n       expected: %s\n       actual:   %s\n" \
            "${label}" "${expected}" "${actual}" >&2
        FAILURES=$((FAILURES + 1))
    fi
}

EXPECT="NHOSTS=2 NGPU_PER_HOST=4 NGPUS=8"

printf "1. SLURM env sets all three vars (no set -u)\n"
check "lax mode" "${EXPECT}" "$(run_case lax)"

printf "2. SLURM env sets all three vars UNDER set -u\n"
check "nounset mode" "${EXPECT}" "$(run_case nounset)"

printf "3. NGPUS is exported, not merely local\n"
# A `local`-only NGPUS would vanish for the caller -- which is exactly the
# state that made the downstream self-reference explode.
out=$(bash -c "
    set -u
    log_message() { :; }
    ezpz_get_slurm_running_jobid() { echo 1; }
    ezpz_make_slurm_nodefile()     { echo '${WORK}/nodefile'; }
    ezpz_get_num_gpus_per_host()   { echo 4; }
    source '${FN_FILE}'
    ezpz_get_slurm_env >/dev/null 2>&1
    # Subshell inherits only EXPORTED vars.
    bash -c 'printf \"%s\" \"\${NGPUS:-<unset>}\"'
" 2>&1)
check "NGPUS visible to a child process" "8" "${out}"

printf "4. NGPUS == NHOSTS * NGPU_PER_HOST\n"
out=$(bash -c "
    set -u
    log_message() { :; }
    ezpz_get_slurm_running_jobid() { echo 1; }
    ezpz_make_slurm_nodefile()     { echo '${WORK}/nodefile'; }
    ezpz_get_num_gpus_per_host()   { echo 6; }
    source '${FN_FILE}'
    ezpz_get_slurm_env >/dev/null 2>&1
    printf '%s' \"\${NGPUS:-<unset>}\"
" 2>&1)
check "2 hosts x 6 gpus = 12" "12" "${out}"

if ((FAILURES > 0)); then
    printf "\n%d assertion(s) failed\n" "${FAILURES}" >&2
    exit 1
fi

printf "\nSLURM job setup exports NHOSTS/NGPU_PER_HOST/NGPUS and survives set -u\n"
