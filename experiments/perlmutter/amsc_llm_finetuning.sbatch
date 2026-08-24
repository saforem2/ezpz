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

# ---- Slingshot networking -------------------------------------------------
# WITHOUT this, NCCL finds no `libnccl-net` plugin and falls back to TCP
# over the host stack. The cost is not subtle: identical config,
# agpt-2b/bs1/seq2048, differs 13x per GPU between 1 and 2 nodes --
#
#     1 node / 4 A100   40,302 tok/s   30.28% MFU
#     2 node / 8 A100    6,175 tok/s    2.31% MFU
#
# -- and adding the second node made the job 6.5x SLOWER in absolute
# terms. The same inter-node path is what hung hf_trainer with
# TCPStore `recvValue failed` (job 57524750).
#
# The plugin lives in the nccl module, which sets NCCL_NET="AWS
# Libfabric" and puts plugin/lib on LD_LIBRARY_PATH. torch bundles NCCL
# 2.29.7 while the plugin is built against 2.24.3; the net-plugin ABI is
# versioned and stable across those, so torch's NCCL loads it.
set +u
module load nccl/2.24.3 2>/dev/null || echo "WARNING: nccl module failed to load"
set -u
echo "=== NCCL_NET=${NCCL_NET:-<unset>} ==="
case ":${LD_LIBRARY_PATH:-}:" in
    *nccl*plugin*) echo "=== nccl plugin on LD_LIBRARY_PATH: yes ===" ;;
    *) echo "WARNING: nccl plugin NOT on LD_LIBRARY_PATH -- expect TCP fallback" ;;
esac

export WANDB_MODE=disabled
export HF_HOME="${SCRATCH}/.cache/hf"
# Read from the local cache. Job 57523900 loaded the model fine and then
# hung at step 0 on an HTTP GET for the dataset until walltime (rc=143,
# zero loss lines) -- `--streaming` fetches per batch over the network
# no matter what is cached, and the compute nodes cannot sustain that.
# Both the model and dataset are already in HF_HOME.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

STEPS="${STEPS:-20}"
BLOCK="${BLOCK:-8192}"
BS="${BS:-1}"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"

OUT="${D}/outputs/amsc-llm-ft-${SLURM_JOB_ID}"
# HF Trainer refuses a non-empty --output_dir, and this script's own
# `tee train.log` was enough to trip it (job 57523702: ValueError on
# all 8 ranks). Keep the log OUTSIDE the trainer's directory.
LOGDIR="${D}/outputs/amsc-llm-ft-${SLURM_JOB_ID}-logs"
mkdir -p "${LOGDIR}"

echo "=== config: model=${MODEL} bs=${BS} block=${BLOCK} steps=${STEPS} fsdp=${FSDP:-auto_wrap} ==="
echo "=== venv=${VENV} torch=$("${PY_BIN}" -c 'import torch;print(torch.__version__)' 2>/dev/null) ==="
echo "=== nodes=${SLURM_NNODES} nodelist=${SLURM_JOB_NODELIST} ==="

NP=$(( SLURM_NNODES * 4 ))
"${PY_BIN}" -m ezpz.launch --np "${NP}" -- \
    "${PY_BIN}" -m ezpz.examples.hf_trainer \
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
        --fsdp="${FSDP:-auto_wrap}" \
        --output_dir="${OUT}" 2>&1 | tee "${LOGDIR}/train.log"
RC=${PIPESTATUS[0]}

echo
echo "=== rc=${RC} ==="
echo "artifacts: ${OUT} (logs: ${LOGDIR})"
exit "${RC}"
