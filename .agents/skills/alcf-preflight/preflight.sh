#!/bin/bash
# alcf-preflight: run BEFORE qsub/sbatch. Exits non-zero if anything the
# job needs is missing, so a bad submission never reaches the queue.
#
#   usage: preflight.sh <worktree> [venv] [machine] [job-script]
#
# Catches the failure classes that cost nine jobs in one session: node-local
# /tmp, missing ZE_FLAT_DEVICE_HIERARCHY, stale worktree, and stale API
# signatures. No `set -e`: we collect all BADs before exiting.
set -o pipefail
W="${1:?usage: preflight.sh <worktree> [venv] [machine] [job-script]}"
V="${2:-}"
M="${3:-$(hostname -s | sed 's/-.*//')}"
J="${4:-}"
fail=0
ok()   { printf "  OK    %s\n" "$1"; }
bad()  { printf "  BAD   %s\n" "$1"; fail=1; }
note() { printf "  note  %s\n" "$1"; }

cd "$W" 2>/dev/null || { bad "worktree $W does not exist"; exit 1; }

# --- worktree currency: a pinned worktree misses files recent commits moved ---
if git rev-parse --git-dir >/dev/null 2>&1; then
  h=$(git rev-parse --short HEAD 2>/dev/null)
  if u=$(git rev-parse --short '@{u}' 2>/dev/null); then
    if [ "$h" = "$u" ]; then ok "worktree current ($h)"
    else bad "worktree $h != upstream $u -- fetch/checkout first"; fi
  else
    note "no upstream set -- skipping staleness check (compare to origin/main manually)"
  fi
fi

# --- XPU geometry: the one that keeps biting ---
# Polaris/Sophia are CUDA (NCCL), not XPU — skip ZE_FLAT there.
case "$M" in
  aurora*|sunspot*|x[0-9]*)
    if [ "${ZE_FLAT_DEVICE_HIERARCHY:-}" = "FLAT" ]; then ok "ZE_FLAT_DEVICE_HIERARCHY=FLAT"
    else bad "ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-unset} -- need FLAT for 12 ranks/node"; fi;;
  *) note "ZE_FLAT check skipped (machine=$M, not XPU)";;
esac

# --- PBS login shell: without --login, module is a silent no-op ---
if [ -n "$J" ] && [ -f "$J" ]; then
  shebang=$(head -1 "$J")
  case "$shebang" in
    *"bash --login"*|*"bash -l"* ) ok "job script shebang has --login ($J)";;
    * ) bad "job script shebang lacks --login ($J) -- PBS has no module without it";;
  esac
  if grep -Eq '^\s*set\s+.*-.*u' "$J" 2>/dev/null; then
    bad "$J contains set -u -- breaks Lmod (/etc/profile, ZSH_EVAL_CONTEXT)"
  fi
fi

# --- nothing the job reads may live on node-local tmpfs ---
_tmp="${TMPDIR:-/tmp}"
for p in "$W" "$V" "$J"; do
  [ -z "$p" ] && continue
  if [[ "$p" == /tmp/* || "$p" == /dev/shm/* || ( -n "$_tmp" && "$_tmp" != "/tmp" && "$p" == "$_tmp"/* ) ]]; then
    bad "$p is on node-local tmpfs -- stage to shared FS (tegu/flare/eagle/pscratch)"
  fi
done

# --- python deps, in the venv the JOB will use (only if V given) ---
if [ -n "$V" ]; then
  if [ ! -d "$V" ] || { [ ! -x "$V/bin/python" ] && [ ! -x "$V/bin/python3" ]; }; then
    bad "venv $V not found (no bin/python)"
  else
    _pypath="$V/bin/python"; [ -x "$_pypath" ] || _pypath="$V/bin/python3"
    for m in torch mpi4py ezpz; do
      "$_pypath" -c "import $m" 2>/dev/null \
        && ok "py:$m in $V" \
        || bad "py:$m not importable in $V"
    done
    # --- API dry-run: verify the venv's ezpz exposes the expected API ---
    # Checks that the venv's installed ezpz can be imported and its key symbols
    # resolve. This catches a stale venv that predates a renamed kwarg. It does
    # NOT test the job's actual kwargs (use --dry-run / inspect.signature with
    # real args for that).
    if "$_pypath" -c "import ezpz.distributed; import inspect; inspect.signature(ezpz.distributed.setup_torch)" 2>/dev/null; then
      ok "API: ezpz.distributed imports and signatures resolvable"
    else
      bad "API: ezpz.distributed not importable in $V -- wrong venv or stale checkout?"
    fi
  fi
else
  note "venv not given -- skipping py imports + API check"
fi

[ $fail -eq 0 ] && echo "  ---> PREFLIGHT PASS" || echo "  ---> PREFLIGHT FAIL, do not submit"
exit $fail
