#!/bin/bash
# #239 on ALCF NVIDIA hardware: Polaris (A100 + NCCL, PBS).
#
# Perlmutter is also A100+NCCL, so this is NOT a new collectives stack --
# it is an independent *site*: different MPI, different NCCL build,
# different node topology (4x A100 per node vs Perlmutter's 4x, but a
# different interconnect), and a different torch build. If the boundary
# lands in the same place here, it is a property of FSDP2 + NCCL rather
# than of one machine's software stack.
#
# Pairs with experiments/sunspot/lora_239_xpu.sh, which tests the other
# axis (same torch major, completely different collectives: xccl).
#
# Submit:
#   qsub -l select=2 -l walltime=01:00:00 -l filesystems=eagle:home \
#     -A datascience_collab -q debug \
#     -o $D/lora239.o -e $D/lora239.e -- /bin/bash $D/experiments/polaris/lora_239_a100.sh

set -o pipefail
# `set -u` is deliberately deferred until AFTER /etc/profile is sourced:
# the profile references unset variables, so under `set -u` it aborts
# the script instantly. Job 12473854 died that way -- walltime 00:00:00,
# Exit_status=1, EMPTY .o and .e, not one line printed. Reproduced:
#   bash -c 'set -u; source /etc/profile'   -> silent death

# PBS runs this under a NON-login /bin/bash, where `module` does not
# exist. Sunspot job 12473851 lost a whole allocation to exactly that:
# it reached the training cells, then failed every one in ~1 s with
# `module: command not found`. Initialise Lmod first, and hard-fail
# rather than run cells in an unconfigured environment.
# Lmod's init/bash alone defines `module` but leaves MODULEPATH EMPTY,
# so every `module load` silently no-ops ("No modules loaded") and the
# conda setup then fails with "CONDA_PREFIX still not set" -- that cost
# Sunspot job 12473853. Source the login profile, which populates
# MODULEPATH with the ALCF tree.
# shellcheck disable=SC1091
source /etc/profile 2>/dev/null || true
if ! command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "${MODULESHOME:-/usr/share/lmod/lmod}/init/bash" 2>/dev/null \
        || { echo "FATAL: cannot initialise Lmod"; exit 1; }
fi
command -v module >/dev/null 2>&1 || { echo "FATAL: module still missing"; exit 1; }
[ -n "${MODULEPATH:-}" ] || { echo "FATAL: MODULEPATH is empty; module loads would silently no-op"; exit 1; }

# NOTE: `set -u` stays OFF for the whole script. Re-enabling it after the
# profile was not enough -- ezpz_setup_env calls `module load frameworks`,
# and Lmod's own init/bash reads $ZSH_EVAL_CONTEXT, which is unbound under
# `set -u`:
#   /usr/share/lmod/lmod/init/bash: line 237: ZSH_EVAL_CONTEXT: unbound variable
# That killed job 12473855 mid-setup. The module machinery is simply not
# `set -u`-clean, so this script relies on explicit checks and
# `${VAR:-default}` everywhere instead. `set -o pipefail` is kept.

D="${EZPZ_DIR:-/lus/eagle/projects/datascience_collab/foremans/torchcomms-test/ezpz}"
cd "${D}" || exit 1

# THE canonical ALCF setup: ezpz_setup_env does python/venv selection,
# the module stack AND the hostfile in one call. Do not hand-assemble
# ezpz_load_modules_polaris + ezpz_setup_conda_polaris -- that is how
# four Sunspot allocations were lost. Compute nodes have no outbound
# internet, so source the repo's own copy of utils.sh rather than the
# usual `source <(curl -fsSL https://bit.ly/ezpz-utils)`.
# shellcheck disable=SC1091
source "${D}/src/ezpz/bin/utils.sh" || { echo "FATAL: no utils.sh"; exit 1; }
ezpz_setup_env || { echo "FATAL: ezpz_setup_env failed"; exit 1; }

export PYTHONPATH="${D}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# LOWERING this is the hang diagnostic -- the 3600 s default turns a
# deadlock into a silent timeout that outlives the allocation.
export TORCH_DDP_TIMEOUT=300
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000

PY_BIN="$(command -v python3)"
echo "=== host: $(hostname) ==="
echo "=== python3: ${PY_BIN} ==="

# PREFLIGHT. On the login node `conda activate base` leaves
# /usr/bin/python3 with no torch; if that is also true on the compute
# node, say so and stop -- do NOT run cells that will all report
# INDETERMINATE and waste the allocation.
"${PY_BIN}" -c "
import sys, torch
print('=== torch', torch.__version__, '/ cuda avail', torch.cuda.is_available(), '===')
if not torch.cuda.is_available():
    sys.exit('CUDA unavailable -- environment, not a result')
" || { echo "FATAL: no usable torch on this node (python3=${PY_BIN})"; exit 1; }

# Verify the SHIPPED DEFAULT arm by probing behaviour, not an env var --
# an env check passes vacuously against a checkout predating the
# variable (that cost Perlmutter job 57604070).
"${PY_BIN}" -c "
import os, sys
os.environ.pop('EZPZ_FSDP_KEEP_FROZEN_GATHERED', None)
from ezpz.examples.fsdp_tp import frozen_unit_kwargs
import torch.nn as nn
f = nn.Linear(4, 4, bias=False); f.weight.requires_grad_(False)
if frozen_unit_kwargs(f, {'reshard_after_forward': True})['reshard_after_forward'] is not True:
    sys.exit('checkout predates the opt-in inversion -- wrong arm')
print('=== default arm confirmed; ezpz from', __import__('ezpz').__file__, '===')
" || { echo "FATAL: wrong code under test -- refusing to run"; exit 1; }

OUT="${D}/outputs/lora-239-polaris-${PBS_JOBID%%.*}"
mkdir -p "${OUT}"
NP=$(( $(wc -l < "${PBS_NODEFILE}") * 4 ))   # 4x A100 per Polaris node
ITERS="${ITERS:-20}"
echo "=== NP=${NP} ==="

# $1 = lora rank
probe () {
    local r="$1"
    local label="r${r}"
    echo
    echo "########## --lora-rank ${r} (Polaris A100/NCCL) ##########"
    local t0=$SECONDS
    timeout 900 "${PY_BIN}" -m ezpz.launch --np "${NP}" -- \
        "${PY_BIN}" -m ezpz.examples.fsdp_tp \
            --model agpt-2b --tp 1 --dataset random \
            --train-iters "${ITERS}" --batch-size 1 --seq-len 2048 \
            --lora-rank "${r}" --lora-target attn,mlp \
            --outdir "${OUT}/${label}" \
        > "${OUT}/${label}.log" 2>&1
    local rc=$?
    local dt=$(( SECONDS - t0 ))
    # rc captured BEFORE any pipe: `| head` SIGPIPEs tee and clobbers
    # PIPESTATUS, which once reported rc=1 for a run that trained fine.
    local last
    last=$(grep -aoE "iter=[0-9]+" "${OUT}/${label}.log" | tail -1)

    # Classify on evidence, not rc: Perlmutter's r64 cell exited rc=1
    # having trained all 20 iters, and these runs emit no `iter=` marker
    # (gating on it once mislabelled a clean pass INDETERMINATE).
    local verdict
    if grep -qaE "Watchdog caught collective operation timeout" "${OUT}/${label}.log"; then
        verdict=HANG
    elif grep -qa "Saving plots" "${OUT}/${label}.log" || [ -n "${last}" ]; then
        verdict=TRAINED
    else
        verdict=INDETERMINATE
    fi
    if [ "${verdict}" = "INDETERMINATE" ] && [ ! -s "${OUT}/${label}.log" ]; then
        echo "    (log is EMPTY -- the launch itself failed, not the run)"
    fi
    grep -aE "Watchdog|Timeout at collective|last enqueued|Traceback|Error|command not found" \
        "${OUT}/${label}.log" | head -4
    echo "### ${label} verdict=${verdict} rc=${rc} secs=${dt} last_iter=${last:-NONE}"
}

# r8 is the most-reproduced Perlmutter hang (6/6); r17/r18 straddle the
# measured boundary there.
for r in ${RANKS:-8 17 18}; do
    probe "${r}"
done

echo
echo "=== artifacts: ${OUT} ==="
echo "=== verdicts are the '### ' lines above ==="
