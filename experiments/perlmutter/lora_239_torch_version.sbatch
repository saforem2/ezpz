#!/bin/bash --login
#SBATCH --account=amsc013_g
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --qos=debug
#SBATCH --time=00:29:00
#SBATCH --job-name=lora-239-torchver
#SBATCH --output=%x-%j.log
#
# #239 lead #2: is the deadlock a torch-2.13 regression?
#
# Every #239 run so far is 2.13.0+cu130. #237 (FSDP2 tied-embedding
# rejection) is also 2.13-only, so a shared regression is plausible --
# but plausible is not tested, and the guide previously overstated this
# from a local CPU/gloo probe that cannot reproduce a GPU NCCL deadlock
# under ANY version. This runs the real config on real GPUs under an
# older torch.
#
# Perlmutter has .venv (2.8.0+cu129) and .venv-213 (2.13.0+cu130). No
# 2.12.1 build exists here; 2.8.0 is an even wider gap, which is fine --
# the question is "does an older torch hang", not "which release
# introduced it". If 2.8.0 also hangs, the 2.13-regression theory dies
# and the search moves to FSDP2 semantics common to both.
#
# CONTROL FIRST. Cell 1 re-runs r8 under 2.13 in THIS allocation, so a
# 2.8.0 pass cannot be confounded by node/topology luck. Do not read
# cell 2 unless cell 1 hangs.

set -u
set -o pipefail

D="${EZPZ_DIR:-/pscratch/sd/f/foremans/projects/ezpz}"
cd "${D}" || exit 1

PY_213="${D}/.venv-213/bin/python"
PY_OLD="${D}/.venv/bin/python"
for p in "${PY_213}" "${PY_OLD}"; do
    [ -x "${p}" ] || { echo "FATAL: no interpreter at ${p}"; exit 1; }
done

export PYTHONPATH="${D}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPICH_GPU_SUPPORT_ENABLED=0

# LOWERING this is the hang diagnostic; see docs/troubleshooting.md.
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

OUT="${D}/outputs/lora-239-torchver-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
NP=$(( SLURM_NNODES * 4 ))
ITERS="${ITERS:-20}"

# $1 = label, $2 = interpreter
probe () {
    local label="$1"; local py="$2"
    echo
    echo "########## ${label} ##########"
    "${py}" -c "import torch; print('=== torch', torch.__version__, '===')"

    # Probe BEHAVIOUR, not a version string or env var: assert this
    # build's default really reshards frozen units, so a stale checkout
    # cannot silently swap which arm is under test. (Job 57604070 was
    # lost to exactly that mistake.)
    "${py}" -c "
import os, sys
os.environ.pop('EZPZ_FSDP_KEEP_FROZEN_GATHERED', None)
from ezpz.examples.fsdp_tp import frozen_unit_kwargs
import torch.nn as nn
f = nn.Linear(4, 4, bias=False); f.weight.requires_grad_(False)
if frozen_unit_kwargs(f, {'reshard_after_forward': True})['reshard_after_forward'] is not True:
    sys.exit('checkout predates the opt-in inversion -- wrong arm')
" || { echo "### ${label} verdict=SKIPPED reason=wrong-code"; return; }

    local t0=$SECONDS
    timeout 600 "${py}" -m ezpz.launch --np "${NP}" -- \
        "${py}" -m ezpz.examples.fsdp_tp \
            --model agpt-2b --tp 1 --dataset random \
            --train-iters "${ITERS}" --batch-size 1 --seq-len 2048 \
            --lora-rank 8 --lora-target attn,mlp \
            --outdir "${OUT}/${label}" \
        > "${OUT}/${label}.log" 2>&1
    local rc=$?
    local dt=$(( SECONDS - t0 ))
    # rc captured BEFORE any pipe: `| head` SIGPIPEs tee and clobbers
    # PIPESTATUS, which once reported rc=1 for a run that trained fine.
    local last
    last=$(grep -aoE "iter=[0-9]+" "${OUT}/${label}.log" | tail -1)

    # Classify on evidence, not rc. The original sweep's r64 cell exited
    # rc=1 having trained all 20 iters (teardown failure), and hanging
    # cells exit 124 or 134 -- rc alone misreads both directions.
    local verdict
    if grep -qaE "Watchdog caught collective operation timeout" "${OUT}/${label}.log"; then
        verdict=HANG
    elif [ -n "${last}" ] || grep -qa "Saving plots" "${OUT}/${label}.log"; then
        verdict=TRAINED
    else
        verdict=INDETERMINATE
    fi
    grep -aE "Watchdog|Timeout at collective|last enqueued|Traceback|Error" \
        "${OUT}/${label}.log" | head -4
    echo "### ${label} verdict=${verdict} rc=${rc} secs=${dt} last_iter=${last:-NONE}"
}

# Cell 1 is the CONTROL and must run first: if r8 does not hang under
# 2.13 in this very allocation, cell 2 proves nothing.
probe "1-control-torch213-r8" "${PY_213}"
probe "2-torch28-r8"          "${PY_OLD}"

echo
echo "=== artifacts: ${OUT} ==="
echo "=== NOTE: cell 2 is only interpretable if cell 1 = HANG ==="
