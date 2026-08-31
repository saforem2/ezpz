#!/bin/bash --login
#SBATCH --account=amsc013_g
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --qos=debug
#SBATCH --time=00:29:00
#SBATCH --job-name=lora-239-proto
#SBATCH --output=%x-%j.log
#
# #239: the DIRECT test of the 256 KiB NCCL-protocol hypothesis.
#
# The rank bisect established the boundary is 17..20, and the per-rank
# reduce-scatter shard crosses 262144 B = 256 KiB exactly between r16
# (209920, hangs) and r20 (262400, trains) -- an NCCL LL/LL128/Simple
# selection boundary.
#
# Rank-bisecting tests that boundary only indirectly: changing r changes
# the payload, the protocol, AND the adapter geometry all at once. This
# job changes ONLY the protocol, holding r=8 fixed, via NCCL_PROTO.
# That isolates the variable the bisect cannot.
#
#   if r8 + a forced protocol TRAINS -> the protocol is the mechanism,
#       and users get a workaround that costs no adapter capacity.
#   if r8 hangs under EVERY forced protocol -> the 256 KiB story is
#       wrong or incomplete, and the r-dependence means something else.
#
# BUDGET. r8 hangs in ~430-500 s (5/5 so far). Two cells worst-case is
# ~1000 s; three would risk being killed mid-cell at 29 min. So: Simple
# first (furthest from the small-payload LL path, most likely to
# differ), then LL128. LL is deliberately omitted -- it is the protocol
# the failing config is already presumed to select, so it is the least
# informative of the three.
#
# No control cell: r8-hangs-by-default is established 5/5 across jobs
# 57601590 (x3), 57602201 and 57604574. Spend the allocation on new
# information instead.

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
"${PY_BIN}" -c "import torch; print('=== torch', torch.__version__, '===')"

# Probe BEHAVIOUR, not an env var: assert this checkout's default really
# reshards frozen units, so a stale checkout cannot silently swap which
# arm is measured. (Job 57604070 was lost to exactly that mistake.)
"${PY_BIN}" -c "
import os, sys
os.environ.pop('EZPZ_FSDP_KEEP_FROZEN_GATHERED', None)
from ezpz.examples.fsdp_tp import frozen_unit_kwargs
import torch.nn as nn
f = nn.Linear(4, 4, bias=False); f.weight.requires_grad_(False)
if frozen_unit_kwargs(f, {'reshard_after_forward': True})['reshard_after_forward'] is not True:
    sys.exit('checkout predates the opt-in inversion -- wrong arm')
print('=== default arm confirmed: frozen units reshard ===')
" || { echo "FATAL: wrong code under test -- refusing to run"; exit 1; }

OUT="${D}/outputs/lora-239-proto-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
NP=$(( SLURM_NNODES * 4 ))
ITERS="${ITERS:-20}"

# $1 = NCCL_PROTO value
probe () {
    local proto="$1"
    local label="proto-${proto}"
    echo
    echo "########## NCCL_PROTO=${proto}  (r8 attn,mlp) ##########"
    local t0=$SECONDS
    NCCL_PROTO="${proto}" NCCL_DEBUG=WARN \
    timeout 600 "${PY_BIN}" -m ezpz.launch --np "${NP}" -- \
        "${PY_BIN}" -m ezpz.examples.fsdp_tp \
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

    # Classify on evidence, never on rc or an `iter=` marker: these runs
    # emit no `iter=` (that mistake reported a clean r24 pass as
    # INDETERMINATE), and the original sweep's r64 cell exited rc=1
    # having trained all 20 iterations.
    local verdict
    if grep -qaE "Watchdog caught collective operation timeout" "${OUT}/${label}.log"; then
        verdict=HANG
    elif grep -qa "Saving plots" "${OUT}/${label}.log" || [ -n "${last}" ]; then
        verdict=TRAINED
    else
        verdict=INDETERMINATE
    fi

    # An invalid/ignored NCCL_PROTO would silently make this a rerun of
    # the default. Surface whatever NCCL said about protocol selection.
    grep -aiE "invalid|unknown proto|NCCL_PROTO" "${OUT}/${label}.log" | head -2
    grep -aE "Watchdog|Timeout at collective|last enqueued|Traceback|Error" \
        "${OUT}/${label}.log" | head -3
    echo "### ${label} verdict=${verdict} rc=${rc} secs=${dt} last_iter=${last:-NONE}"
}

for p in ${PROTOS:-Simple LL128}; do
    probe "${p}"
done

echo
echo "=== artifacts: ${OUT} ==="
echo "=== r8 hangs by DEFAULT 5/5; any TRAINED above implicates the protocol ==="
