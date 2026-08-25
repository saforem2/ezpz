#!/bin/bash --login
#SBATCH --account=amsc013_g
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --qos=debug
#SBATCH --time=00:29:00
#SBATCH --job-name=lora-239-bisect
#SBATCH --output=%x-%j.log
#
# #239 lead #1: is "r8/r16 hang, r32/r64 work" a THRESHOLD, or is r just
# a poor proxy for something else?
#
# Everything static about the collective stream is now ruled out --
# LoRA-specificity, payload size, per-rank order, and the frozen-unit
# AG/RS asymmetry itself (jobs 57601590 / 57602201). What survives is
# that the outcome depends on r, deterministically, with an identical
# op sequence. So find where it flips.
#
# BUDGET. A hang costs the full 300s watchdog plus ~90s startup; a
# working run finishes in ~90-120s. In 29 minutes that is at most ~3
# hangs. So this bisects rather than sweeps: binary search on the
# boundary, at most 4 probes, widest-information-first.
#
#   known HANG: 16      known OK: 32       -> probe 24
#   then 20 or 28 depending, then one more.
#
# Overriding: RANKS="24 20 28" sbatch lora_239_rank_bisect.sbatch
#
# A cell that neither hangs nor trains (crash, OOM, import error) is
# reported as INDETERMINATE and must NOT be read as either arm.

set -u
set -o pipefail

D="${EZPZ_DIR:-/pscratch/sd/f/foremans/projects/ezpz}"
cd "${D}" || exit 1

VENV="${VENV:-.venv-213}"
PY_BIN="${D}/${VENV}/bin/python"
[ -x "${PY_BIN}" ] || { echo "FATAL: no interpreter at ${PY_BIN}"; exit 1; }

export PYTHONPATH="${D}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPICH_GPU_SUPPORT_ENABLED=0

# 5-minute watchdog instead of the 3600s default. See
# docs/troubleshooting.md: LOWERING this is the hang diagnostic.
export TORCH_DDP_TIMEOUT=300
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000

# Without this NCCL silently falls back to TCP: 8.3x slower inter-node,
# no error. See docs/guides/perlmutter.md.
set +u; module load nccl/2.24.3 2>/dev/null || echo "WARN: nccl module failed"; set -u
echo "=== NCCL_NET=${NCCL_NET:-<unset>} ==="
case ":${LD_LIBRARY_PATH:-}:" in
    *nccl*plugin*) echo "=== nccl plugin: yes ===" ;;
    *) echo "WARNING: no nccl plugin — expect TCP fallback" ;;
esac

echo "=== ezpz commit: $(git -C "${D}" rev-parse --short HEAD 2>/dev/null || echo unknown) ==="
"${PY_BIN}" -c "import torch; print('=== torch', torch.__version__, '===')"

# This job tests the SHIPPED DEFAULT path. Assert that by asking the
# code what it actually does, not by checking an env var -- an env
# check passes vacuously against a checkout that predates the variable.
#
# That is not hypothetical: job 57604070 was launched against a stale
# checkout where frozen_unit_kwargs still defaulted to ON, so it was
# silently bisecting the FAILED-intervention arm while reporting itself
# as the default arm. Cancelled and relaunched. Probe the behaviour.
EZPZ_FSDP_KEEP_FROZEN_GATHERED="${EZPZ_FSDP_KEEP_FROZEN_GATHERED:-0}" \
"${PY_BIN}" -c "
import os, sys
os.environ.pop('EZPZ_FSDP_KEEP_FROZEN_GATHERED', None)
from ezpz.examples.fsdp_tp import frozen_unit_kwargs
import torch.nn as nn
f = nn.Linear(4, 4, bias=False); f.weight.requires_grad_(False)
base = {'reshard_after_forward': True}
if frozen_unit_kwargs(f, base)['reshard_after_forward'] is not True:
    sys.exit('frozen unit is NOT resharding by default -- this checkout '
             'predates the opt-in inversion, so it would measure the '
             'failed-intervention arm')
print('=== default arm confirmed: frozen units reshard ===')
" || { echo "FATAL: wrong code under test -- refusing to run"; exit 1; }

OUT="${D}/outputs/lora-239-bisect-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
NP=$(( SLURM_NNODES * 4 ))
ITERS="${ITERS:-20}"

# $1 = lora rank
probe () {
    local r="$1"
    local label="r${r}"
    echo
    echo "########## --lora-rank ${r} ##########"
    local t0=$SECONDS
    timeout 600 "${PY_BIN}" -m ezpz.launch --np "${NP}" -- \
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

    # Classify explicitly. "did not hang" is NOT the same as "trained".
    #
    # Do NOT gate on an `iter=` marker: these runs do not emit one, so
    # requiring it labelled a clean 173s r24 pass INDETERMINATE. Gate on
    # "reached the plotting stage", which only happens after the
    # training loop completes. Do not gate on rc either -- the original
    # sweep's r64 cell exited rc=1 having trained all 20 iters.
    local verdict
    if grep -qaE "Watchdog caught collective operation timeout" "${OUT}/${label}.log"; then
        verdict=HANG
    elif grep -qa "Saving plots" "${OUT}/${label}.log" || [ -n "${last}" ]; then
        verdict=TRAINED
    else
        verdict=INDETERMINATE
    fi
    grep -aE "Watchdog|Timeout at collective|last enqueued|Traceback|Error" \
        "${OUT}/${label}.log" | head -4
    echo "### ${label} verdict=${verdict} rc=${rc} secs=${dt} last_iter=${last:-NONE}"
}

# Anchors are already established (r8/r16 HANG 4/4 across jobs 57601590
# and 57602201; r32/r64 reported OK), so spend the allocation entirely
# on the unexplored interval. Order is binary-search, not ascending.
for r in ${RANKS:-24 20 28}; do
    probe "${r}"
done

echo
echo "=== artifacts: ${OUT} ==="
echo "=== VERDICT TABLE ==="
grep -h "^### " "$0".log 2>/dev/null || true
