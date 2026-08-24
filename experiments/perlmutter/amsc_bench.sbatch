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

# Which torch. `.venv` inherits NERSC's pytorch/2.8.0, whose Dynamo
# cannot trace the FSDP2 backward hook -- `--compile` aborts every rank
# before iteration 1 (#236). `.venv-213` is built on the pytorch/2.13.0
# module (torch 2.13.0+cu130) to see whether that is fixed. Both use
# the module's torch rather than a pip build, so CUDA and the Slingshot
# stack stay matched.
VENV="${VENV:-.venv-213}"
PY_BIN="${D}/${VENV}/bin/python"
[ -x "${PY_BIN}" ] || { echo "FATAL: no interpreter at ${PY_BIN}"; exit 1; }
cd "${D}" || exit 1
echo "=== commit: $(git rev-parse --short HEAD) ==="
echo "=== nodes: ${SLURM_NNODES} x ${SLURM_GPUS_PER_NODE:-?} GPU ==="
echo "=== nodelist: ${SLURM_JOB_NODELIST} ==="

# Working-tree code, not whatever is installed in the venv. The Sunspot
# validate clone had a stale non-editable install and silently ran a
# months-old ezpz; PYTHONPATH avoids repeating that.
export PYTHONPATH="${D}/src:${PYTHONPATH:-}"
# The pytorch/2.13.0 module exports MPICH_GPU_SUPPORT_ENABLED=1, but its
# mpi4py is not linked against the GTL library, so MPIDI_CRAY_init
# aborts and every rank segfaults -- job 57521863: "GPU_SUPPORT_ENABLED
# is requested, but GTL library is not linked", then SIGSEGV on all 8
# ranks, rc=139, zero iterations. The 2.8.0 build does not hit this.
#
# Turning it off disables GPU-aware MPI, which costs nothing here:
# collectives go over NCCL, and ezpz uses MPI only for rank discovery.
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
# Config knobs, overridable so a second point on the curve is one
# `sbatch --export` away rather than an edit.
MODEL="${MODEL:-agpt-2b}"
BS="${BS:-8}"
SEQ="${SEQ:-2048}"
ITERS="${ITERS:-20}"
# An explicit yes/no, NOT a flag string. `COMPILE="${COMPILE:---compile}"`
# substitutes when the value is empty OR unset, so `--export=...,COMPILE=`
# still produced `--compile` -- job 57521116 ran compiled while claiming
# to test the uncompiled path, and only the config echo (`"compile":
# true`) gave it away.
# Default OFF on Perlmutter: `--compile` aborts before iteration 1 on
# NERSC's torch 2.8.0 with `Dynamo: Attempted to call function marked
# as skipped` in the FSDP2 backward hook (#236). Defaulting to yes
# would make every run here fail. Flip with COMPILE=yes once a torch
# build that traces it is available.
COMPILE="${COMPILE:-no}"
COMPILE_FLAG=""
[ "${COMPILE}" = "yes" ] && COMPILE_FLAG="--compile"

# bs=1/seq2048 (job 57520568) left the A100s starved: 13.68 of 80 GiB
# used, MFU 2.32%, and dtb ~3x dtf -- collective overhead dominating a
# batch too small to fill the device. That measures the config, not the
# machine. bs=8 + --compile is a fairer read of the hardware; the
# bs=1 point is kept in the results table for comparison.
echo "=== config: model=${MODEL} bs=${BS} seq=${SEQ} iters=${ITERS} compile=${COMPILE} ==="
echo "=== venv=${VENV} torch=$("${PY_BIN}" -c 'import torch;print(torch.__version__)' 2>/dev/null) ==="
NP=$(( SLURM_NNODES * 4 ))
"${PY_BIN}" -m ezpz.launch --np "${NP}" -- \
    "${PY_BIN}" -m ezpz.examples.fsdp_tp \
        --model "${MODEL}" --tp 1 --dataset random \
        --train-iters "${ITERS}" --batch-size "${BS}" --seq-len "${SEQ}" \
        ${COMPILE_FLAG} \
        --outdir "${OUT}" 2>&1 | tee "${OUT}/train.log"
RC=${PIPESTATUS[0]}

echo
echo "=== rc=${RC} ==="
echo "=== run dirs ==="
find "${OUT}" -maxdepth 3 -name '*.jsonl' -o -maxdepth 3 -type d -name '2026-*' 2>/dev/null | head
echo "artifacts: ${OUT}"
exit "${RC}"
