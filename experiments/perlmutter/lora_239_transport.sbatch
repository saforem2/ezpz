#!/bin/bash --login
#SBATCH --account=amsc013_g
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --qos=debug
#SBATCH --time=00:29:00
#SBATCH --job-name=i239transport
#SBATCH --output=%x-%j.log
#
# ezpz #239, transport isolation. Reproduces the 2026-09-15 result:
# a ~4 MiB reduce-scatter deadlocks on aws-ofi-nccl/cxi and completes
# over TCP. Everything is pinned at --lora-rank 17 (the hang point);
# the ONLY variable is the NCCL network layer.
#
# Measured (jobs 58368557, 58369713):
#   default plugin  6/6 HANG, always NumelIn=1055232
#   NCCL_NET=Socket 3/3 TRAIN, ~7x slower (1.38 vs 0.20 s/step)
set -u
set -o pipefail

C="${EZPZ_DIR:-/pscratch/sd/f/foremans/projects/saforem2/ezpz}"
PY="${PY_BIN:-/pscratch/sd/f/foremans/projects/ezpz/.venv-213/bin/python}"
cd "${C}" || exit 1
[ -x "${PY}" ] || { echo "FATAL: no interpreter at ${PY}"; exit 1; }

export PYTHONPATH="${C}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPICH_GPU_SUPPORT_ENABLED=0
# Lowering this is the DIAGNOSTIC: a deadlock aborts in 3 min, not 60.
export TORCH_DDP_TIMEOUT=180

run() {
  L=$1; shift
  echo "===== BEGIN $L ====="
  G="${SLURM_JOB_ID}-${L}.log"
  ( "$@" timeout 300 srun -N2 -n8 --gpus-per-node=4 "${PY}" -m ezpz.examples.fsdp_tp \
      --model agpt-2b --dataset random --tp 1 --train-iters 4 --batch-size 1 \
      --seq-len 2048 --lora-target attn,mlp --lora-rank 17 ) > "${G}" 2>&1
  rc=$?
  # Classify on EVIDENCE, not exit code. Two traps, both hit during the
  # original investigation:
  #   * srun tears down steps with SIGTERM, so rc=143 means nothing --
  #     a 12 s argparse crash and a 5 min deadlock look identical.
  #   * ezpz's init banner prints "timeout=0:05:00", so a bare grep for
  #     "timeout" reports a clean run as a hang. Strip ANSI, then match
  #     the watchdog string exactly.
  S=$(sed 's/\x1b\[[0-9;]*m//g' "${G}")
  if printf '%s' "${S}" | grep -qE "unrecognized arguments|ConnectionError|Traceback" \
     && ! printf '%s' "${S}" | grep -qE "iter=[0-9]"; then
      echo "RESULT $L = SETUP-ERROR :: $(printf '%s' "${S}" | grep -oE 'error: .*' | tail -1 | head -c 120)"
  elif printf '%s' "${S}" | grep -qE "iter=3|iter=4"; then
      echo "RESULT $L = TRAINS :: $(printf '%s' "${S}" | grep -oE 'train_dt.*' | tail -1 | head -c 70)"
  elif printf '%s' "${S}" | grep -q "Watchdog caught collective operation timeout"; then
      echo "RESULT $L = HANG :: $(printf '%s' "${S}" | grep -oE 'SeqNum=[0-9]+, OpType=[A-Z_]+, NumelIn=[0-9]+' | head -1)"
  else
      echo "RESULT $L = OTHER rc=$rc :: $(printf '%s' "${S}" | grep -vE '^srun:' | tail -1 | head -c 120)"
  fi
  echo "===== END $L ====="
}

# Identify the stack first -- the NCCL version is NOT the one the module
# system reports (see the NOTE in the other scripts in this directory).
echo "=== stack ==="
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET timeout 120 srun -N1 -n1 "${PY}" -c \
  'import torch.distributed' 2>&1 | grep -oiE "NCCL version [0-9.]+|Selected Provider is [a-z]+|Using network [A-Za-z]+" | sort -u

run baseline          env                    # expect HANG
run socket            env NCCL_NET=Socket    # expect TRAINS, ~7x slower
run proto-simple      env NCCL_PROTO=Simple  # expect HANG (not the protocol)
run algo-ring         env NCCL_ALGO=Ring     # expect HANG (not the algorithm)
echo "ALL DONE"
