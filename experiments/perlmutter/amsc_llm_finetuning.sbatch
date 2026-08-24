#!/bin/bash --login
#SBATCH --account=amsc013_g
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --qos=debug
#SBATCH --time=00:28:00
#SBATCH --job-name=amsc-llm-ft
#SBATCH --output=%x-%j.log
#
# The AmSC `training/llm-finetuning` benchmark AS DEFINED by that repo:
# Llama-3.2-1B via ezpz.examples.hf_trainer at block_size 8192, bs 1.
#
# NOT the agpt-2b/seq2048 fsdp_tp pretraining in amsc_bench.sbatch --
# different model, task and sequence length, so those numbers do not
# belong in this benchmark's results even though both are "ezpz on
# Perlmutter".

set -u
set -o pipefail

D="${EZPZ_DIR:-/pscratch/sd/f/foremans/projects/ezpz}"
cd "${D}" || exit 1

VENV="${VENV:-.venv-213}"
PY_BIN="${D}/${VENV}/bin/python"
[ -x "${PY_BIN}" ] || { echo "FATAL: no interpreter at ${PY_BIN}"; exit 1; }

export PYTHONPATH="${D}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# pytorch/2.13.0's mpi4py is not GTL-linked; without this every rank
# segfaults in MPIDI_CRAY_init (see docs/guides/perlmutter.md).
export MPICH_GPU_SUPPORT_ENABLED=0
export WANDB_MODE=disabled
export HF_HOME="${SCRATCH}/.cache/hf"

STEPS="${STEPS:-20}"
BLOCK="${BLOCK:-8192}"
BS="${BS:-1}"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"

OUT="${D}/outputs/amsc-llm-ft-${SLURM_JOB_ID}"
mkdir -p "${OUT}"

echo "=== config: model=${MODEL} bs=${BS} block=${BLOCK} steps=${STEPS} ==="
echo "=== venv=${VENV} torch=$("${PY_BIN}" -c 'import torch;print(torch.__version__)' 2>/dev/null) ==="
echo "=== nodes=${SLURM_NNODES} nodelist=${SLURM_JOB_NODELIST} ==="

NP=$(( SLURM_NNODES * 4 ))
"${PY_BIN}" -m ezpz.launch --np "${NP}" -- \
    "${PY_BIN}" -m ezpz.examples.hf_trainer \
        --streaming \
        --dataset_name=eliplutchok/fineweb-small-sample \
        --tokenizer_name "${MODEL}" \
        --model_name_or_path "${MODEL}" \
        --bf16=true \
        --do_train=true \
        --logging-steps=1 \
        --include-tokens-per-second=true \
        --max-steps="${STEPS}" \
        --include-num-input-tokens-seen=true \
        --optim=adamw_torch \
        --logging-first-step \
        --include-for-metrics='inputs,loss' \
        --per_device_train_batch_size="${BS}" \
        --block-size="${BLOCK}" \
        --gradient_checkpointing=true \
        --fsdp=shard_grad_op \
        --output_dir="${OUT}" 2>&1 | tee "${OUT}/train.log"
RC=${PIPESTATUS[0]}

echo
echo "=== rc=${RC} ==="
echo "artifacts: ${OUT}"
exit "${RC}"
