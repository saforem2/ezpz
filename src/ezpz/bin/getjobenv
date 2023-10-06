#!/bin/bash --login

HOSTNAME=$(hostname)
export HOMEDIR="$HOME"


getCOBALT_NODEFILE() {
  if [[ "${HOSTNAME}" == x3* ]]; then
    JOBENV_FILE="${HOME}/.jobenv-polaris"
    export JOBENV_FILE="$JOBENV_FILE"
    # if [[ -f "$HOME/.jobenv-polaris" ]] ; then
    #   source "~/.jobenv-polaris"
    # fi
    # export MPI_COMMAND="mpiexec --verbose -n 2 --ppn $NGPU_PER_HOST --hostfile $PBS_NODEFILE --envall"
  elif [[ "${HOSTNAME}" == thetagpu* ]]; then
    RUNNING_JOB_FILE="/var/tmp/cobalt-running-job"
    echo "RUNNING_JOB_FILE: $RUNNING_JOB_FILE"
    if [[ -f "$RUNNING_JOB_FILE" ]]; then
      JOBID=$(sed "s/:$USER//" /var/tmp/cobalt-running-job)
      COBALT_NODEFILE="/var/tmp/cobalt.${JOBID}"
      export JOBID="${JOBID}"
      export HOSTFILE="${HOSTFILE}"
      export COBALT_NODEFILE="${COBALT_NODEFILE}"
      echo "JOBID: ${JOBID}"
      echo "HOSTFILE: ${HOSTFILE}"
    fi
  else
    echo "NO RUNNING JOB FILE"
  fi
}

getJobEnv() {
  if [[ "${HOSTNAME}" == x3* ]]; then
    JOBENV_FILE="${HOME}/.jobenv-polaris"
    export JOBENV_FILE="$JOBENV_FILE"
    # if [[ -f "$HOME/.jobenv-polaris" ]] ; then
    #   source "~/.jobenv-polaris"
    # fi
    # export MPI_COMMAND="mpiexec --verbose -n 2 --ppn $NGPU_PER_HOST --hostfile $PBS_NODEFILE --envall"
  elif [[ "${HOSTNAME}" == thetagpu* ]]; then
    getCOBALT_NODEFILE
    JOBENV_FILE="${HOME}/.jobenv-thetaGPU"
    # if [[ -f "$HOME/.jobenv-thetaGPU" ]] ; then
    #   source "~/.jobenv-thetaGPU"
    # fi
  else
    JOBENV_FILE="$HOME/.jobenv"
  fi
  export JOBENV_FILE="$JOBENV_FILE"
  echo "Loading job env from: ${JOBENV_FILE}"
  # cat "${JOBENV_FILE}"
  # test condition && echo "true" || echo "false"
  source "${JOBENV_FILE}"
  export HOSTFILE="${HOSTFILE}"
  export NHOSTS="${NHOSTS}"
  export NGPU_PER_HOST="${NGPU_PER_HOST}"
  export NGPUS="${NGPUS}"

  echo "HOSTFILE: ${HOSTFILE}"
  echo "NHOSTS: ${NHOSTS}"
  echo "NGPU_PER_HOST: ${NGPU_PER_HOST}"
  echo "NGPUS (NHOSTS x NGPU_PER_HOST): ${NGPUS}"
  echo "HOSTS: $(cat "$HOSTFILE")"
}

getJobEnv
