#!/bin/bash --login
#SBATCH --account=amsc013_g
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --qos=debug
#SBATCH --time=00:29:00
#SBATCH --job-name=lora-sweep
#SBATCH --output=%x-%j.log
#
# LoRA on Perlmutter: first CUDA validation. LoRA was validated on
# Sunspot/Aurora (XPU) only, so every adapter path here -- rank sweep,
# role targeting, HF-vs-native naming -- is untested on NVIDIA.
#
# One allocation, several rank/target points, so a single debug-QOS job
# yields a curve rather than a point.

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

# Without this NCCL silently falls back to TCP: 8.3x slower inter-node,
# no error. See docs/guides/perlmutter.md.
set +u; module load nccl/2.24.3 2>/dev/null || echo "WARN: nccl module failed"; set -u
echo "=== NCCL_NET=${NCCL_NET:-<unset>} ==="
case ":${LD_LIBRARY_PATH:-}:" in
    *nccl*plugin*) echo "=== nccl plugin: yes ===" ;;
    *) echo "WARNING: no nccl plugin — expect TCP fallback" ;;
esac

OUT="${D}/outputs/lora-sweep-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
NP=$(( SLURM_NNODES * 4 ))
ITERS="${ITERS:-20}"

# rank 0 = full fine-tune (the baseline every LoRA point is measured
# against); the rest sweep adapter capacity. Targets vary too, since
# role selection is its own code path (#227 dedented it out of a
# native-only branch, so HF/native divergence is worth a data point).
run () {
    local label="$1"; shift
    echo
    echo "########## ${label} ##########"
    # 600s: each run pays full ezpz.launch startup (~90s to first
    # iter on 2 nodes) before it trains at all. 420 was not enough to
    # distinguish "slow" from "hung" -- job 57540435 hit it having
    # already died on a missing seaborn.
    timeout 600 "${PY_BIN}" -m ezpz.launch --np "${NP}" -- \
        "${PY_BIN}" -m ezpz.examples.fsdp_tp \
            --model agpt-2b --tp 1 --dataset random \
            --train-iters "${ITERS}" --batch-size 1 --seq-len 2048 \
            --outdir "${OUT}/${label}" "$@" 2>&1 \
        > "${OUT}/${label}.log" 2>&1
    local rc=$?
    # Capture rc BEFORE filtering. Piping into `grep | head` did two
    # bad things: `head -20` closes the pipe early and SIGPIPEs tee,
    # which can kill a still-running python, and PIPESTATUS is clobbered
    # by the time the echo reads it. Job 57540698 reported rc=1 for a
    # run that trained all 20 iters at 19.56% MFU -- a false failure.
    grep -aE "iter=(1|5|10|15|20) |lora|adapter|Error|Traceback|ModuleNotFound" \
        "${OUT}/${label}.log" | head -20 || true
    echo "### ${label} rc=${rc}"
}

# Which points run is selectable: five sequential runs plus ~90s of
# ezpz.launch startup each overran the 29-minute debug allocation, and
# r8/r16-attn-mlp were killed mid-run (job 57540698). Split across two
# submissions rather than silently losing points.
case "${SWEEP:-a}" in
  a)
    run "full-baseline"      --lora-rank 0
    run "lora-r16-attn-only" --lora-rank 16 --lora-target attn
    run "lora-r64-attn-mlp"  --lora-rank 64
    ;;
  b)
    run "lora-r8-attn-mlp"   --lora-rank 8
    run "lora-r16-attn-mlp"  --lora-rank 16
    run "lora-r32-attn-mlp"  --lora-rank 32
    ;;
esac

echo
echo "=== artifacts: ${OUT} ==="
