#!/bin/sh

export LOCAL_RANK="${PMIX_LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK}}"
export RANK="${PMIX_RANK:-${OMPI_COMM_WORLD_RANK}}"
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
export LOCAL_SIZE="${PALS_LOCAL_SIZE:-${OMPI_COMM_WORLD_LOCAL_SIZE}}"
export SIZE=$((LOCAL_SIZE * PBS_JOBSIZE))
export NODE_IDX=$((RANK % PBS_JOBSIZE))
# export SIZE=$((PALS_LOCAL_SIZE*PBS_JOBSIZE))
# export LOCAL_RANK=$PALS_LOCAL_RANKID
# export RANK=$PALS_RANKID
export WORLD_SIZE=$SIZE

export MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"

#echo "I am $RANK of $SIZE: $LOCAL_RANK on `hostname`"
printf "[%s]: global: [%s/%s], local: [%s/%s], node: [%s/%s]\n" "$(hostname)" "${RANK}" "${SIZE}" "${LOCAL_RANK}" "${LOCAL_SIZE}" "${NODE_IDX}" "${PBS_JOBSIZE}"

$@
