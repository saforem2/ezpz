#!/bin/bash -l
# ----------------------------------------------------------------------------
#PBS -k doe
#PBS -V exports all the environment variables in your environnment to the
#compute node The rest is an example of how an MPI job might be set up
# echo Working directory is $PBS_O_WORKDIR
# cd $PBS_O_WORKDIR
# ----------------------------------------------------------------------------
#
resolveDir() {
  SOURCE=${BASH_SOURCE[0]}
  while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
    DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
    SOURCE=$(readlink "$SOURCE")
    [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
  done
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  HERE="${DIR}/train.sh"
  PARENT=$(dirname "${DIR}")
  MAIN="${PARENT}/main.py"
  GRANDPARENT=$(dirname "${PARENT}")
  ROOT=$(dirname "${GRANDPARENT}")
  echo "DIR: ${DIR}"
  echo "HERE: ${HERE}"
  echo "PARENT: ${PARENT}"
  echo "GRANDPARENT: ${GRANDPARENT}"
  echo "ROOT: ${ROOT}"
}


function sourceFile() {
  fpath="${1:-${DIR}/setup.sh}"
  # SETUP_FILE="${DIR}/setup.sh"
  if [[ -f "${fpath}" ]]; then
    echo "source-ing ${fpath}"
    # shellcheck source=./setup.sh
    source "${fpath}"
  else
    echo "ERROR: UNABLE TO SOURCE ${fpath}"
  fi
}

#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃ Make sure we're not already running; if so, exit here ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
function killIfRunning() {
  PIDS=$(ps aux | grep -E 'mpi.+main.+py' | grep -v grep | awk '{print $2}')
  if [ -n "${PIDS}" ]; then
    echo "Already running! Exiting!"
    exit 1
  fi
}


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ SETUP CONDA + MPI ENVIRONMENT @ ALCF ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
function setup() {
  resolveDir
  killIfRunning
  # setupEnv
  # setupLogs
  sourceFile "${DIR}/setup.sh"
  setupJob "$@" | tee -a "${LOGFILE}"
  export NODE_RANK=0
  export NNODES=$NHOSTS
  export GPUS_PER_NODE=$NGPU_PER_HOST
  export WORLD_SIZE=$NGPUS
  # export LC_ALL=$(locale -a | grep UTF-8)
  # printJobInfo "$@" | tee -a "${LOGFILE}"
}

setup "$@"
mpilaunch "$(which python3)" "${MAIN}" "$@" > "${LOGFILE}" 2>&1 &
# wait $!
