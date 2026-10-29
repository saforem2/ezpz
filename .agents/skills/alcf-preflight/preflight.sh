#!/bin/bash
# alcf-preflight: run BEFORE qsub/sbatch. Exits non-zero if anything the
# job needs is missing, so a bad submission never reaches the queue.
#
#   usage: preflight.sh <worktree> [venv] [machine] [job-script]
#
# Catches the failure classes that cost nine jobs in one session: node-local
# /tmp, missing ZE_FLAT_DEVICE_HIERARCHY, stale worktree, and API signatures
# that changed under us.
set -o pipefail
W="${1:?usage: preflight.sh <worktree> [venv] [machine] [job-script]}"
V="${2:-}"
M="${3:-$(hostname -s | sed 's/[0-9-].*//')}"
J="${4:-}"
fail=0
ok()   { printf "  OK    %s\n" "$1"; }
bad()  { printf "  BAD   %s\n" "$1"; fail=1; }
note() { printf "  note  %s\n" "$1"; }

cd "$W" 2>/dev/null || { bad "worktree $W does not exist"; exit 1; }

# --- worktree currency: a pinned worktree misses files recent commits moved ---
if git rev-parse --git-dir >/dev/null 2>&1; then
  h=$(git rev-parse --short HEAD 2>/dev/null)
  u=$(git rev-parse --short '@{u}' 2>/dev/null || echo "$h")
  if [ "$h" = "$u" ]; then ok "worktree current ($h)"
  else bad "worktree $h != upstream $u -- fetch/checkout first"; fi
fi

# --- XPU geometry: the one that keeps biting ---
case "$M" in
  aurora*|sunspot*|x[0-9]*)
    if [ "${ZE_FLAT_DEVICE_HIERARCHY:-}" = "FLAT" ]; then ok "ZE_FLAT_DEVICE_HIERARCHY=FLAT"
    else bad "ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-unset} -- need FLAT for 12 ranks/node"; fi;;
  *) note "ZE_FLAT check skipped (machine=$M, not XPU)";;
esac

# --- PBS login shell + MODULEPATH: without --login, module is a silent no-op ---
if [ -n "$J" ] && [ -f "$J" ]; then
  shebang=$(head -1 "$J")
  case "$shebang" in
    *"bash --login"*|*"bash -l"* ) ok "job script shebang has --login ($J)";;
    * ) bad "job script shebang lacks --login: $shebang -- PBS has no module without it";;
  esac
  if [[ "$J" == /tmp/* ]]; then
    bad "$J is on node-local tmpfs -- invisible to compute nodes"
  fi
  if grep -q "set -u" "$J" 2>/dev/null; then
    bad "$J contains set -u -- breaks Lmod (/etc/profile, ZSH_EVAL_CONTEXT)"
  fi
fi

# --- nothing the job reads may live on node-local tmpfs ---
for p in "$W" "$V" "$J"; do
  [ -z "$p" ] && continue
  case "$p" in /tmp/*) bad "$p is on node-local tmpfs -- stage to shared FS";; esac
done

# --- python deps, in the venv the JOB will use (only if V given) ---
if [ -n "$V" ]; then
  if [ ! -d "$V" ] && [ ! -x "$V/bin/python" ] && [ ! -x "$V/bin/python3" ]; then
    bad "venv $V not found"
  else
    for m in torch mpi4py ezpz; do
      "$V/bin/python" -c "import $m" 2>/dev/null \
        && ok "py:$m in $V" \
        || bad "py:$m not importable in $V"
    done
    # --- API dry-run: bind the exact kwargs the job will use ---
    "$V/bin/python" -c "
import inspect, ezpz.distributed as d
inspect.signature(d.setup_torch).bind()
inspect.signature(d.get_torch_device).bind()
" 2>/dev/null \
      && ok "API: setup_torch/get_torch_device signatures OK" \
      || bad "API: setup_torch or get_torch_device signature check failed -- stale checkout or wrong venv?"
  fi
else
  note "venv not given -- skipping py: imports + API bind check"
fi

[ $fail -eq 0 ] && echo "  ---> PREFLIGHT PASS" || echo "  ---> PREFLIGHT FAIL, do not submit"
exit $fail
