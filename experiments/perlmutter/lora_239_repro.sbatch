#!/bin/bash --login
#SBATCH --account=amsc013_g
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --qos=debug
#SBATCH --time=00:29:00
#SBATCH --job-name=lora-239
#SBATCH --output=%x-%j.log
#
# #239: does the r8/r16 attn,mlp hang reproduce, and is it deterministic?
#
# The single most valuable output of this job is NOT whether the fix
# works -- it is whether the baseline hangs all three times. Two prior
# jobs (57540698, 57541459) each hung once, which is consistent with a
# deterministic bug AND with a coin-flip race. Those two cannot be told
# apart from two samples, and every downstream claim depends on which
# it is. If baseline hangs 3/3, rank is a real variable; if it hangs
# 1/3 or 2/3, "r8/r16 hang, r32/r64 work" was never a real boundary and
# the whole framing needs rebuilding.
#
# TORCH_DDP_TIMEOUT=300 so a hang dumps in 5 min, not 60. The original
# runs used the 3600s default, which is why they looked like silent
# hangs instead of watchdog aborts.

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

# 5-minute watchdog instead of the 3600s default: a hang must announce
# itself inside a debug allocation.
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

OUT="${D}/outputs/lora-239-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
NP=$(( SLURM_NNODES * 4 ))
ITERS="${ITERS:-20}"

# Proves the lever is actually in this checkout AND honours the opt-in.
# Without this a job can silently test the wrong code and report a
# result belonging to a different build.
EZPZ_FSDP_KEEP_FROZEN_GATHERED=1 "${PY_BIN}" -c "
from ezpz.examples.fsdp_tp import frozen_unit_kwargs
import torch.nn as nn
f = nn.Linear(4, 4, bias=False); f.weight.requires_grad_(False)
got = frozen_unit_kwargs(f, {'reshard_after_forward': True})
assert got['reshard_after_forward'] is False, got
print('=== frozen_unit_kwargs PRESENT + opt-in honoured ===')
" || { echo "FATAL: frozen_unit_kwargs missing/inert -- refusing to run"; exit 1; }

# $1 label, $2 = 0|1 for EZPZ_FSDP_KEEP_FROZEN_GATHERED (1 = keep frozen
# units gathered, i.e. the asymmetry-removal arm; 0 = shipped default)
run () {
    local label="$1"; local keepgathered="$2"; shift 2
    echo
    echo "########## ${label}  (EZPZ_FSDP_KEEP_FROZEN_GATHERED=${keepgathered}) ##########"
    local t0=$SECONDS
    EZPZ_FSDP_KEEP_FROZEN_GATHERED="${keepgathered}" \
    timeout 600 "${PY_BIN}" -m ezpz.launch --np "${NP}" -- \
        "${PY_BIN}" -m ezpz.examples.fsdp_tp \
            --model agpt-2b --tp 1 --dataset random \
            --train-iters "${ITERS}" --batch-size 1 --seq-len 2048 \
            --outdir "${OUT}/${label}" "$@" \
        > "${OUT}/${label}.log" 2>&1
    local rc=$?
    local dt=$(( SECONDS - t0 ))
    # rc captured BEFORE any pipe: `| head` SIGPIPEs tee and clobbers
    # PIPESTATUS, which once reported rc=1 for a run that trained fine.
    local last
    last=$(grep -aoE "iter=[0-9]+" "${OUT}/${label}.log" | tail -1)
    grep -aE "iter=(1|10|20) |Watchdog|Timeout|NCCL|Traceback|Error" \
        "${OUT}/${label}.log" | head -12 || true
    echo "### ${label} rc=${rc} secs=${dt} last_iter=${last:-NONE}"
}

# Cells 1a/1b/1c: the SAME shipped-default baseline three times. This is
# the point of the job -- determinism first, intervention second.
# RESULT (job 57601590): rc=134 3/3, last_iter=NONE. Deterministic.
run "1a-baseline-r8"  0 --lora-rank 8
run "1b-baseline-r8"  0 --lora-rank 8
run "1c-baseline-r8"  0 --lora-rank 8

# Cells 2-3: asymmetry removed, against the two reported-hanging configs.
# RESULT (job 57602201): still hung. rc=134 (r8) / rc=124 (r16),
# last_iter=NONE. Same collective, same 419840->52480 payload, merely
# renumbered. This is a REFUTATION, kept runnable, not a fix.
run "2-nofrozenreshard-r8"   1 --lora-rank 8
run "3-nofrozenreshard-r16"  1 --lora-rank 16

echo
echo "=== artifacts: ${OUT} ==="
echo "=== VERDICT TABLE ==="
grep -h "^### " "${OUT}"/../lora-239-${SLURM_JOB_ID}.log 2>/dev/null || true
