#!/bin/sh

export LOCAL_RANK=$PMIX_LOCAL_RANK
export RANK=$PMIX_RANK
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
export SIZE=$((PALS_LOCAL_SIZE*PBS_JOBSIZE))
export LOCAL_RANK=$PALS_LOCAL_RANKID
export RANK=$PALS_RANKID
export WORLD_SIZE=$SIZE

export MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"

echo "I am $RANK of $SIZE: $LOCAL_RANK on `hostname`"

$@
