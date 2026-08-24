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
    timeout 420 "${PY_BIN}" -m ezpz.launch --np "${NP}" -- \
        "${PY_BIN}" -m ezpz.examples.fsdp_tp \
            --model agpt-2b --tp 1 --dataset random \
            --train-iters "${ITERS}" --batch-size 1 --seq-len 2048 \
            --outdir "${OUT}/${label}" "$@" 2>&1 \
        | tee "${OUT}/${label}.log" | grep -aE "iter=(1|5|10|15|20) |lora|adapter|Error|Traceback" | head -20
    echo "### ${label} rc=${PIPESTATUS[0]}"
}

run "full-baseline"      --lora-rank 0
run "lora-r8-attn-mlp"   --lora-rank 8
run "lora-r16-attn-mlp"  --lora-rank 16
run "lora-r16-attn-only" --lora-rank 16 --lora-target attn
run "lora-r64-attn-mlp"  --lora-rank 64

echo
echo "=== artifacts: ${OUT} ==="
