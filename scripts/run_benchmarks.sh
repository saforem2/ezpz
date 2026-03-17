#!/bin/bash --login
# run_benchmarks.sh — Run all ezpz examples sequentially and generate a report.
#
# Usage:
#   bash scripts/run_benchmarks.sh          # local (mpirun fallback)
#   qsub scripts/run_benchmarks.sh          # PBS
#   sbatch scripts/run_benchmarks.sh        # SLURM
#
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle:grand
#PBS -A <project>
#PBS -q debug
#SBATCH --nodes=2
#SBATCH --time=01:00:00
set -euo pipefail

# ── Resolve project root (works whether invoked from repo root or via scheduler) ─
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env

# ── Source user env setup if available ────────────────────────────────────────
if [[ -n "${EZPZ_SETUP_ENV:-}" ]] && [[ -f "${EZPZ_SETUP_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${EZPZ_SETUP_ENV}"
fi

# ── Python interpreter ───────────────────────────────────────────────────────
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python3)"
fi

# ── Create timestamped output directory ──────────────────────────────────────
TIMESTAMP="$(date +%Y-%m-%d-%H%M%S)"
BENCH_DIR="${PROJECT_ROOT}/outputs/benchmarks/${TIMESTAMP}"
mkdir -p "${BENCH_DIR}"

echo "Benchmark output directory: ${BENCH_DIR}"

# ── Detect scheduler ────────────────────────────────────────────────────────
if [[ -n "${PBS_JOBID:-}" ]]; then
    SCHEDULER="PBS"
    JOB_ID="${PBS_JOBID}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SCHEDULER="SLURM"
    JOB_ID="${SLURM_JOB_ID}"
else
    SCHEDULER="local"
    JOB_ID="$$"
fi

# ── Capture environment info ────────────────────────────────────────────────

NOW="$(date "+%Y-%m-%d-%H%M%S")"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
HOSTNAME_STR="$(hostname)"
DATE_ISO="$(date -u +%Y-%m-%dT%H:%M:%S)"
PYTHON_VER="$(${PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
TORCH_VER="$(${PYTHON} -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
EZPZ_VER="$(${PYTHON} -c 'from ezpz.__about__ import __version__; print(__version__)' 2>/dev/null || echo 'unknown')"

# GPU/node detection
if [[ "${SCHEDULER}" == "PBS" ]]; then
    NUM_NODES="${PBS_NUM_NODES:-$(wc -l < "${PBS_NODEFILE}" 2>/dev/null || echo 1)}"
elif [[ "${SCHEDULER}" == "SLURM" ]]; then
    NUM_NODES="${SLURM_NNODES:-1}"
else
    NUM_NODES=1
fi


GPUS_PER_NODE="${NGPU_PER_HOST:-"$(${PYTHON} -c 'import ezpz; print(ezpz.distributed.get_gpus_per_node())' 2>/dev/null || echo 0)"}"
TOTAL_GPUS=$(( NUM_NODES * GPUS_PER_NODE ))

# Write env.json
cat > "${BENCH_DIR}/env.json" <<ENVJSON
{
  "git_commit": "${GIT_COMMIT}",
  "git_branch": "${GIT_BRANCH}",
  "job_id": "${JOB_ID}",
  "scheduler": "${SCHEDULER}",
  "num_nodes": ${NUM_NODES},
  "gpus_per_node": ${GPUS_PER_NODE},
  "total_gpus": ${TOTAL_GPUS},
  "hostname": "${HOSTNAME_STR}",
  "date": "${DATE_ISO}",
  "python": "${PYTHON_VER}",
  "torch": "${TORCH_VER}",
  "ezpz_version": "${EZPZ_VER}"
}
ENVJSON

echo "Environment info written to ${BENCH_DIR}/env.json"

# ── Runner function ─────────────────────────────────────────────────────────
run_example() {
    local name="$1"; shift
    local logfile="${BENCH_DIR}/${name}.log"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Running: ${name}"
    echo "════════════════════════════════════════════════════════════════"
    local t0=${SECONDS}
    # Run the command, tee stdout+stderr to the log file.
    # Use set +e so a failing example doesn't abort the whole suite.
    set +e
    "$@" > >(tee "${logfile}") 2>&1
    local rc=$?
    set -e
    local elapsed=$(( SECONDS - t0 ))
    echo "${name},${rc},${elapsed}" >> "${BENCH_DIR}/timings.csv"
    if [[ ${rc} -eq 0 ]]; then
        echo "  ✓ ${name} completed in ${elapsed}s"
    else
        echo "  ✗ ${name} FAILED (exit ${rc}) after ${elapsed}s"
    fi
    return 0  # always continue to the next example
}

# ── Write CSV header ────────────────────────────────────────────────────────
echo "name,exit_code,wall_seconds" > "${BENCH_DIR}/timings.csv"

# ── Run examples ────────────────────────────────────────────────────────────
run_example test \
    ezpz launch python3 -m ezpz.examples.test --model small

run_example fsdp \
    ezpz launch python3 -m ezpz.examples.fsdp --model small

run_example vit \
    ezpz launch python3 -m ezpz.examples.vit --model small --warmup 0 --fsdp

run_example fsdp_tp \
    ezpz launch python3 -m ezpz.examples.fsdp_tp --model small --dataset stanfordnlp/imdb

run_example diffusion \
    ezpz launch python3 -m ezpz.examples.diffusion --model small --dataset standfordnlp/imdb

run_example hf \
    ezpz launch python3 -m ezpz.examples.hf \
        --dataset_name=eliplutchok/fineweb-small-sample \
        --streaming \
        --model_name_or_path meta-llama/Llama-3.2-1B \
        --bf16=true \
        --do_train=true \
        --do_eval=true \
        --report-to=wandb \
        --logging-steps=1 \
        --max-steps=100 \
        --optim=adamw_torch \
        --logging-first-step \
        --include-for-metrics='inputs,loss' \
        --max-eval-samples=100 \
        --per_device_train_batch_size=1 \
        --per_device_eval_batch_size=1 \
        --block_size=2048 \
        --fsdp=auto_wrap \
        --output_dir="${BENCH_DIR}/outputs/ezpz.hf/${NOW}"

run_example hf_trainer \
    ezpz launch python3 -m ezpz.examples.hf_trainer \
      --dataset_name=eliplutchok/fineweb-small-sample \
      --streaming \
      --model_name_or_path meta-llama/Llama-3.2-1B \
      --bf16=true \
      --do_train=true \
      --do_eval=true \
      --report-to=wandb \
      --logging-steps=1 \
      --max-steps=100 \
      --optim=adamw_torch \
      --logging-first-step \
      --include-for-metrics='inputs,loss' \
      --max-eval-samples=100 \
      --per_device_train_batch_size=1 \
      --per_device_eval_batch_size=1 \
      --block_size=2048 \
      --fsdp=auto_wrap \
      --output_dir="${BENCH_DIR}/outputs/ezpz.hf_trainer/${NOW}"

# ── Generate report ─────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Generating report"
echo "════════════════════════════════════════════════════════════════"
"${PYTHON}" "${PROJECT_ROOT}/scripts/generate_report.py" --outdir "${BENCH_DIR}"

echo ""
echo "Done. Results in: ${BENCH_DIR}"
