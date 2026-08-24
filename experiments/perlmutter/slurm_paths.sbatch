#!/bin/bash --login
#SBATCH --account=amsc013_g
#SBATCH --constraint=gpu
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --qos=debug
#SBATCH --time=00:29:00
#SBATCH --job-name=slurm-paths
#SBATCH --output=%x-%j.log
#
# Every SLURM code path that has only ever been exercised against a
# mock, plus the --auto-retry node-kill that has only ever run on PBS.
#
# 4 nodes so --auto-retry has a real active+spare split (2+2), matching
# the Sunspot experiment that found three stacked attribution bugs.

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

OUT="${D}/outputs/slurm-paths-${SLURM_JOB_ID}"
mkdir -p "${OUT}"

echo "############ PART 1: SLURM discovery functions ############"
# These are unit-tested against mocked subprocess output only. This is
# the first time they run against a real scheduler.
"${PY_BIN}" - <<'PY' 2>&1 | grep -viE '^\[2026' | tail -20
import ezpz.slurm as S, os
jid = os.environ["SLURM_JOB_ID"]
checks = [
    ("get_slurm_jobid_of_active_job", lambda: S.get_slurm_jobid_of_active_job()),
    ("get_nodelist_from_slurm_jobid", lambda: S.get_nodelist_from_slurm_jobid(jid)),
    ("get_slurm_running_jobs",        lambda: list(S.get_slurm_running_jobs())[:2]),
    ("get_slurm_nodefile_of_active_job", lambda: S.get_slurm_nodefile_of_active_job()),
    ("build_launch_cmd",              lambda: S.build_launch_cmd(ngpu_per_host=4)),
]
for name, fn in checks:
    try:
        v = fn()
        vs = str(v)
        print(f"  OK    {name}: {vs[:90]}")
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {str(e)[:70]}")

# Does the real nodelist match what SLURM says?
want = sorted(os.popen("scontrol show hostnames").read().split())
got  = sorted(S.get_nodelist_from_slurm_jobid(jid) or [])
print(f"  nodelist matches scontrol: {want == got}  ({len(got)} hosts)")
PY

echo
echo "############ PART 2: --auto-retry node-kill on SLURM ############"
# The PBS version of this (Sunspot 12473751) found three stacked bugs.
# Perlmutter has NO bad-node patterns registered, so attribution is
# expected to be blind -- that is the result, not a failure.
ACTIVE=2
KILL_INDEX=1
mapfile -t HOSTS < <(scontrol show hostnames)
VICTIM="${HOSTS[$KILL_INDEX]}"
echo "hosts: ${HOSTS[*]}"
echo "victim (active[${KILL_INDEX}]): ${VICTIM}"

CKPT="${OUT}/ckpt"; LOG="${OUT}/train.log"; DONE="${OUT}/.done"
RECEIPT="${OUT}/kill_receipt.txt"

chaos () {
    # Wait for a checkpoint, then kill the ranks on ONE node.
    for _ in $(seq 1 60); do
        [ -f "${DONE}" ] && { echo "[CHAOS] training ended first"; return 0; }
        ls -d "${CKPT}"/step-* >/dev/null 2>&1 && break
        sleep 3
    done
    sleep 5
    echo "[CHAOS] killing ranks on ${VICTIM}"
    # --overlap: join the running allocation without a new resource req.
    srun --jobid="${SLURM_JOB_ID}" --overlap -w "${VICTIM}" -N1 -n1 bash -c '
        n=0
        for pid in $(pgrep -f "ezpz[.]examples[.]fsdp_tp" 2>/dev/null); do
            grep -qa "ezpz.launch" "/proc/${pid}/cmdline" 2>/dev/null && continue
            kill -9 "${pid}" 2>/dev/null && n=$((n+1))
        done
        echo "$(hostname) killed ${n}"
    ' > "${RECEIPT}" 2>&1
    echo "[CHAOS] receipt: $(cat "${RECEIPT}" 2>/dev/null)"
}
chaos & CPID=$!

# 400 iters / save every 10: the job must still be TRAINING when the
# kill lands, and must have runway left to relaunch afterwards. Job
# 57540535 used 120 iters and hit the 29-min walltime first --
# --auto-retry then correctly returned `FAILOVER STOP: walltime` and
# declined to retry, which is the WALLTIME guard working, not a bug.
timeout 1100 "${PY_BIN}" -m ezpz.launch --auto-retry --nhosts "${ACTIVE}" -- \
    "${PY_BIN}" -m ezpz.examples.fsdp_tp \
        --model agpt-2b --tp 1 --dataset random \
        --train-iters 400 --batch-size 1 --seq-len 2048 \
        --ckpt-dir "${CKPT}" --save-interval 10 \
        --outdir "${OUT}/out" 2>&1 | tee "${LOG}"
RC=${PIPESTATUS[0]}
touch "${DONE}"; kill "${CPID}" 2>/dev/null || true

echo
echo "======================= RESULT ======================="
echo "rc=${RC}  victim=${VICTIM}"
FO="${D}/logs/failover-${SLURM_JOB_ID}"
[ -d "${FO}" ] || FO="$(ls -dt "${D}"/logs/failover-* 2>/dev/null | head -1)"
echo "failover dir: ${FO:-<none>}"
[ -n "${FO}" ] && {
    echo "--- bad_nodes.txt ---";   cat "${FO}/bad_nodes.txt"   2>/dev/null
    echo "--- active.hostfile ---"; cat "${FO}/active.hostfile" 2>/dev/null
    echo "--- attempts: $(ls -1 "${FO}"/attempt-*.log 2>/dev/null | wc -l) ---"
}
K="$(cat "${RECEIPT}" 2>/dev/null)"
if grep -qE "killed [1-9]" <<< "${K}"; then
    echo "  PASS  the kill landed"
    N=$(ls -1 "${FO}"/attempt-*.log 2>/dev/null | wc -l)
    [ "${N}" -ge 2 ] && echo "  PASS  relaunched (${N} attempts)" \
                     || echo "  FAIL  no relaunch (${N} attempts)"
    grep -qa "blind rotation" "${LOG}" 2>/dev/null \
        && echo "  INFO  swap was BLIND (expected: no Perlmutter patterns)" \
        || echo "  INFO  swap was scraper-identified"
else
    echo "  FAIL  the kill never landed — everything below is vacuous"
fi
echo "======================================================"
echo "artifacts: ${OUT}"
