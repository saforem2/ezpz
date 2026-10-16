#!/bin/bash --login
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home
#PBS -A AuroraGPT
#PBS -k doe
#PBS -j oe
#PBS -q debug
#PBS -N aurora

set -eo pipefail
cd /Users/sam/projects/saforem2/ezpz

# ── Environment setup ──
source /Users/sam/projects/saforem2/ezpz/src/ezpz/bin/utils.sh && ezpz_setup_env

ezpz launch aurora /tmp/murali-test.sh
