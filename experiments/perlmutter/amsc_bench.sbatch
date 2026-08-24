#!/bin/bash --login
#SBATCH --account=amsc013_g
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --qos=debug
# 25 min, not 45: the `debug` QOS caps MaxWall at 00:30:00 and a longer
# request is rejected outright with QOSMaxWallDurationPerJobLimit. A
# 20-iteration benchmark needs a couple of minutes; the rest is queue
# and startup headroom.
#SBATCH --time=00:25:00
#SBATCH --job-name=ezpz-amsc-bench
#SBATCH --output=%x-%j.log
#
# AmSC at-scale benchmark on Perlmutter (A100), matching the shape of the
# Aurora/Sunspot runs so the numbers are comparable:
#   agpt-2b, tp=1, bs=1, seq 2048  ->  config agpt-2b/bs1/seq2048/tp1
#
# Perlmutter is the first NON-PBS, NON-XPU system this has run on, so it
# also exercises the SLURM launch path and NVIDIA device discovery that
# the ALCF machines never touch.

set -u
set -o pipefail

D="${EZPZ_DIR:-/pscratch/sd/f/foremans/projects/ezpz}"
cd "${D}" || exit 1
echo "=== commit: $(git rev-parse --short HEAD) ==="
echo "=== nodes: ${SLURM_NNODES} x ${SLURM_GPUS_PER_NODE:-?} GPU ==="
echo "=== nodelist: ${SLURM_JOB_NODELIST} ==="

# Working-tree code, not whatever is installed in the venv. The Sunspot
# validate clone had a stale non-editable install and silently ran a
# months-old ezpz; PYTHONPATH avoids repeating that.
export PYTHONPATH="${D}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

OUT="${D}/outputs/amsc-perlmutter-${SLURM_JOB_ID}"
mkdir -p "${OUT}"

# --dataset random: the benchmark measures throughput, and a real corpus
# would add dataloader variance plus an offline-cache dependency that
# says nothing about the machine.
# Through `ezpz.launch`, NOT a bare `python3 -m`. Job 57519659 called
# the module directly: that runs ONE process, which then blocks in
# init_process_group waiting for a world_size=8 that nothing ever
# started -- 18 minutes of silence at `master_addr=...` before I killed
# it. ezpz.launch is what builds the `srun -N.. -n..` command line.
NP=$(( SLURM_NNODES * 4 ))
"${D}/.venv/bin/python" -m ezpz.launch --np "${NP}" -- \
    "${D}/.venv/bin/python" -m ezpz.examples.fsdp_tp \
        --model agpt-2b --tp 1 --dataset random \
        --train-iters 20 --batch-size 1 --seq-len 2048 \
        --outdir "${OUT}" 2>&1 | tee "${OUT}/train.log"
RC=${PIPESTATUS[0]}

echo
echo "=== rc=${RC} ==="
echo "=== run dirs ==="
find "${OUT}" -maxdepth 3 -name '*.jsonl' -o -maxdepth 3 -type d -name '2026-*' 2>/dev/null | head
echo "artifacts: ${OUT}"
exit "${RC}"
