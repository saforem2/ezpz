#!/bin/bash
# #239 on ALCF XPU hardware: does the LoRA r8 deadlock reproduce off NVIDIA?
#
# Everything in #239 so far is Perlmutter: A100 + NCCL. Sunspot is PVC +
# **xccl**, running torch 2.13.0.dev+xpu -- the same major version as the
# hang, on an entirely different collectives stack. So:
#
#   r8 hangs here  -> the bug is in FSDP2's bucketing/scheduling, not in
#                     NCCL, and #239 is much broader than reported.
#   r8 trains here -> it is NVIDIA/NCCL-specific, which is itself a strong
#                     constraint on the mechanism.
#
# Either outcome is worth an allocation. Cells mirror the Perlmutter
# matrix around the measured boundary (r8 hang / r17 hang / r18 train).
#
# Submit (qsub is NOT on $PATH over plain ssh -- absolute path required,
# and all four flags are mandatory):
#
#   /opt/pbs/bin/qsub -l select=2 -l walltime=00:60:00 \
#     -l filesystems=tegu:home -A datascience -q workq \
#     -o $D/lora239.o -e $D/lora239.e -- /bin/bash $D/experiments/sunspot/lora_239_xpu.sh

set -o pipefail
# `set -u` is deliberately deferred until AFTER /etc/profile is sourced:
# the profile references unset variables, so under `set -u` it aborts
# the script instantly. Job 12473854 died that way -- walltime 00:00:00,
# Exit_status=1, EMPTY .o and .e, not one line printed. Reproduced:
#   bash -c 'set -u; source /etc/profile'   -> silent death

# PBS runs this under a NON-login /bin/bash, where `module` does not
# exist -- job 12473851 got all the way to the training cells and then
# every one of them failed in ~1 s with `module: command not found`,
# because ezpz_load_modules_sunspot could not load oneapi. Initialise
# Lmod explicitly before anything tries to use it.
# Sourcing Lmod's init/bash alone is NOT enough: it defines `module` but
# leaves MODULEPATH EMPTY, so `module load frameworks` silently no-ops
# ("No modules loaded") and ezpz_setup_env then fails with
# "CONDA_PREFIX still not set". That cost job 12473853. Source the login
# profile, which sets MODULEPATH to the ALCF tree.
# shellcheck disable=SC1091
source /etc/profile 2>/dev/null || true
if ! command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "${MODULESHOME:-/usr/share/lmod/lmod}/init/bash" 2>/dev/null \
        || { echo "FATAL: cannot initialise Lmod"; exit 1; }
fi
command -v module >/dev/null 2>&1 || { echo "FATAL: module still missing"; exit 1; }
# Assert the PATH is populated too -- an empty MODULEPATH makes every
# subsequent `module load` a silent no-op, which is worse than an error.
[ -n "${MODULEPATH:-}" ] || { echo "FATAL: MODULEPATH is empty; module loads would silently no-op"; exit 1; }

# Safe to be strict again now that the profile is out of the way.
set -u

D="${EZPZ_DIR:-$HOME/datascience/foremans/projects/saforem2/ezpz}"
TT="${TT_DIR:-$HOME/datascience/foremans/projects/saforem2/torchtitan}"
cd "${D}" || exit 1

# THE canonical ALCF setup, exactly as used by hand:
#
#   source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
#
# ezpz_setup_env does everything -- python/venv selection, the module
# stack, and the hostfile -- so do NOT hand-assemble
# ezpz_load_modules_sunspot + a venv activation. Three submissions
# (12473850/51/52) were lost doing exactly that.
#
# Compute nodes have no outbound internet, so the curl form cannot be
# used from inside a batch job (it times out after ~270 s and silently
# leaves the environment unconfigured). Source the repo's own copy,
# which is the same file.
# shellcheck disable=SC1091
source "${D}/src/ezpz/bin/utils.sh" || { echo "FATAL: no utils.sh"; exit 1; }
ezpz_setup_env || { echo "FATAL: ezpz_setup_env failed"; exit 1; }

export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# LOWERING this is the hang diagnostic -- the 3600 s default turns a
# deadlock into a silent timeout that outlives the allocation.
export TORCH_DDP_TIMEOUT=300

PY_BIN="$(command -v python3)"
OUT="${D}/outputs/lora-239-xpu-${PBS_JOBID%%.*}"
mkdir -p "${OUT}"
NP="${NP:-$(wc -l < "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || echo 1)}"
NP=$(( NP * 12 ))   # 12 XPU tiles per Sunspot node
ITERS="${ITERS:-20}"

echo "=== host: $(hostname) ==="
echo "=== ezpz commit: $(git -C "${D}" rev-parse --short HEAD 2>/dev/null) ==="
echo "=== NP=${NP} ==="
"${PY_BIN}" -c "import torch; print('=== torch', torch.__version__, '===')"

# Verify we are on the SHIPPED DEFAULT arm by probing behaviour, not by
# checking an env var -- an env check passes vacuously against a checkout
# that predates the variable (that mistake cost Perlmutter job 57604070).
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

# $1 = lora rank
probe () {
    local r="$1"
    local label="r${r}"
    echo
    echo "########## --lora-rank ${r} (XPU/xccl) ##########"
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

    # Classify on evidence, not rc: on Perlmutter the r64 cell exited
    # rc=1 having trained all 20 iters, and these runs emit no `iter=`
    # marker at all (gating on it once mislabelled a clean pass).
    local verdict
    if grep -qaE "Watchdog caught collective operation timeout|ProcessGroup.*[Tt]imeout" "${OUT}/${label}.log"; then
        verdict=HANG
    elif grep -qa "Saving plots" "${OUT}/${label}.log" || [ -n "${last}" ]; then
        verdict=TRAINED
    else
        verdict=INDETERMINATE
    fi
    # An INDETERMINATE cell that produced NO output at all means the
    # launch never started -- surface that instead of leaving a bare
    # verdict with no cause to chase (12473851 left three empty logs).
    if [ "${verdict}" = "INDETERMINATE" ] && [ ! -s "${OUT}/${label}.log" ]; then
        echo "    (log is EMPTY -- the launch itself failed, not the run)"
    fi
    grep -aE "Watchdog|Timeout|last enqueued|Traceback|Error|command not found" \
        "${OUT}/${label}.log" | head -4
    echo "### ${label} verdict=${verdict} rc=${rc} secs=${dt} last_iter=${last:-NONE}"
}

# r8 first: the most-reproduced Perlmutter hang (6/6). r17/r18 straddle
# the measured NVIDIA boundary, so if r8 hangs here they say whether the
# boundary lands in the same place on a different collectives stack.
for r in ${RANKS:-8 17 18}; do
    probe "${r}"
done

echo
echo "=== artifacts: ${OUT} ==="
echo "=== verdicts are the '### ' lines above ==="
