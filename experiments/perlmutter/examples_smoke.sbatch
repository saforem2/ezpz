#!/bin/bash --login
#SBATCH --account=amsc013_g
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --qos=debug
#SBATCH --time=00:29:00
#SBATCH --job-name=ex-smoke
#SBATCH --output=%x-%j.log
#
# Do ezpz's examples run on NVIDIA at all?
#
# They are exercised on Intel XPU (Aurora/Sunspot) and on CPU in CI.
# Nothing has run them multi-node on CUDA, so device selection, NCCL
# collectives and the dataloaders are all unverified there. A failure
# here is a real portability bug, not a benchmark result -- this is a
# smoke test and is reported as pass/fail, not as performance.

set -u
set -o pipefail

D="${EZPZ_DIR:-/pscratch/sd/f/foremans/projects/ezpz}"
cd "${D}" || exit 1
PY_BIN="${D}/${VENV:-.venv-213}/bin/python"
[ -x "${PY_BIN}" ] || { echo "FATAL: no interpreter at ${PY_BIN}"; exit 1; }

export PYTHONPATH="${D}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPICH_GPU_SUPPORT_ENABLED=0
set +u; module load nccl/2.24.3 2>/dev/null; set -u
echo "=== NCCL_NET=${NCCL_NET:-<unset>} ==="

OUT="${D}/outputs/ex-smoke-${SLURM_JOB_ID}"
mkdir -p "${OUT}"
NP=$(( SLURM_NNODES * 4 ))

declare -A RESULT
smoke () {
    local mod="$1"; shift
    echo; echo "########## ${mod} ##########"
    timeout 300 "${PY_BIN}" -m ezpz.launch --np "${NP}" -- \
        "${PY_BIN}" -m "ezpz.examples.${mod}" "$@" 2>&1 \
        | tee "${OUT}/${mod}.log" | tail -6
    local rc=${PIPESTATUS[0]}
    RESULT[$mod]=$rc
    echo "### ${mod} rc=${rc}"
}

smoke test      --train-iters 20
smoke minimal
smoke vit       --train-iters 10
smoke diffusion --train-iters 10

echo
echo "======================= SMOKE RESULTS ======================="
fail=0
for m in "${!RESULT[@]}"; do
    if [ "${RESULT[$m]}" -eq 0 ]; then echo "  PASS  ${m}"
    else echo "  FAIL  ${m} (rc=${RESULT[$m]})"; fail=1; fi
done
echo "============================================================"
echo "artifacts: ${OUT}"
exit "${fail}"
