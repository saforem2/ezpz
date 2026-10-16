#!/bin/bash --login
#SBATCH --account=amsc013_g
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --qos=debug
#SBATCH --time=00:12:00
#SBATCH --mem=224G
#SBATCH --job-name=pm-setu
#SBATCH --output=/pscratch/sd/f/foremans/%x-%j.log
#
# Isolate ONE variable: does ezpz_setup_job survive `set -u` on a compute
# node? The verify run had `set -u` and died here with no message; the
# diagnostic had no `set -u` and returned rc=0. Same node type, same
# staged utils.sh, same allocation shape -- so run both, in one job, and
# name the unbound variable if it aborts.
#
# Login-node checks already said "rc=0" for BOTH, which is exactly why
# that evidence was worthless: this only reproduces on compute.
P=/pscratch/sd/f/foremans
STAGE=$P/ezpzmain
echo "host=$(hostname)"

echo "=== A. WITHOUT set -u (expected: rc=0) ==="
bash -c "
source $STAGE/utils.sh >/dev/null 2>&1
ezpz_setup_job >/dev/null 2>&1
echo \"  rc=\$? NHOSTS=\${NHOSTS:-unset}\"
"

echo "=== B. WITH set -u, stderr KEPT (names the unbound var if it aborts) ==="
bash -c "
set -u
source $STAGE/utils.sh >/dev/null 2>&1
ezpz_setup_job 2>&1 | tail -5
echo \"  rc=\${PIPESTATUS[0]} NHOSTS=\${NHOSTS:-unset}\"
" 2>&1 | tail -8
echo "  outer rc=$?"

echo "=== C. WITH set -u, no pipe (so the abort is not masked) ==="
bash -c "
set -u
source $STAGE/utils.sh >/dev/null 2>&1
ezpz_setup_job >/dev/null
echo \"  reached-after rc=\$?\"
" 2>&1 | tail -6
echo "  outer rc=$?"

echo "SETU_DONE"
