#!/bin/bash --login

if [[ "$(command -v setopt)" ]]; then
    setopt aliases
elif [[ "$(command -v shopt)" ]]; then
    shopt -s expand_aliases
fi

RESET="\e[0m"
BLACK="\e[1;30m"
RED="\e[1;31m"
GREEN="\e[1;32m"
YELLOW="\e[1;33m"
BLUE="\e[1;34m"
MAGENTA="\e[1;35m"
CYAN="\e[1;36m"
WHITE="\e[1;37m"

HOSTNAME=$(hostname)

PBS_ENV_FILE="${HOME}/.pbsenv"
SLURM_ENV_FILE="${HOME}/.slurmenv"

# HEADER_LINE="┌─────────────────────────────────────────────────────────────────────┐\n"
# FOOTER_LINE="└─────────────────────────────────────────────────────────────────────┘\n"
HEADER="\n"
FOOTER="\n"

###############################################################################
# Check if running in DEBUG=1 mode.
#   - If so, this will print each command before it is ran and exit if any of
#   them return a nonzero exit status.
###############################################################################
if [[ -n "${DEBUG-}" ]]; then  # to use: `DEBUG=1 bash train_llama_alcf.sh`
    printf "\e[1;31m%s\e[0m\n" "!! RUNNING IN DEBUG MODE !!"
    set -euxo pipefail
fi

###############################################################################
# Print (but DO NOT EXECUTE !!) each command that would be ran.
#
# Enable with: NOOP=1 PBS_O_WORKDIR=$(pwd) bash train_llama_alcf.sh
###############################################################################
if [[ -v NOOP ]]; then         # to use: `NOOP=1 bash train_llama_alcf.sh`
  echo "Run NOOP mode"
  set -o noexec                # same as set -n
fi

ezpz_get_tstamp () {
    printf "%s" "$(date "+%Y-%m-%d-%H%M%S")"
}

####################
# ezpz_qsme_running
#
# prints 1 line for each running job owned by $USER
#
# each line of the form:
#
# <jobid> <elapsed_time> <node0> <node1> <node2> ...
####################
ezpz_qsme_running() {
    qstat -u $USER -n1rw | sed -e "s/\/0\*208/\ /g" | tr "+|." "\ " | awk '{a = ""; for (i = 13 ; i <= NF ; i++) a = a " " $i; print $1 a}' | egrep -v "aurora-pbs|Req|Job|\-\-"
}

###############################
# ezpz_get_jobid_from_hostname
#
# Identify jobid containing "$(hostname)" from all active (running) jobs owned
# by the $USER.
#
# Example:
# --------
# Look for `$(hostname)` in output from `ezpz_qsme_running`, and print the first
# column
#
#  |   jobid   |   host0  |   host1   |  host2   |
#  |:---------:|:--------:|:---------:|:--------:|
#  |  jobid0   |  host00  |  host10   |  host20  |
#  |  jobid1   |  host01  |  host11   |  host21  |
#  |  jobid2   |  host02  |  host12   |  host22  |
#
###############################
ezpz_get_jobid_from_hostname() {
    jobid=$(ezpz_qsme_running | grep "$(hostname)" | awk '{print $1}')
    echo "${jobid}"
}


#######################
# Unset all:
#
# - `PBS_*`
# - {host,HOST}file
#
# environment variables
ezpz_reset_pbs_vars() {
    wd="${PBS_O_WORKDIR:-${WORKING_DIR:-$(pwd)}}"
    vars=($(printenv | grep -iE "^PBS" | tr "=" " " | awk '{print $1}'))
    for v in "$vars[@]"; do echo "Unsetting $v" && unset -v "${v}"; done
    # matches=($(printenv | grep -iE "^PBS|host"))
    # for v in "${matches[*]}" ; do
    # for v in ${vars}; do
    #     echo "Unsetting $v" && unset -v ${v}
    #     # vname=$(echo "${v}" | tr "=" " " | awk '{print $1}')
    #     # echo "unsetting $vname" && unset $vname
    #     # unset $vname
    # done
    export PBS_O_WORKDIR="${wd}"
    # for v in $(printenv | grep -ie "PBS|host"); do
    #     # if [[ "${v}" != "PBS_O_WORKDIR" ]]; then

}


######################################
# ezpz_get_pbs_nodefile_from_hostname
#
# Return path to PBS_NODEFILE corresponding to the jobid that was identified as
# containing the (currently active, determined by `$(hostname)`) host.
#
# Example:
# --------
# Look for $(hostname) in output from `ezpz_qsme_running`
#
#  |   jobid   |   host0  |   host1   |  host2   |
#  |:---------:|:--------:|:---------:|:--------:|
#  |  jobid0   |  host00  |  host10   |  host20  |
#  |  jobid1   |  host01  |  host11   |  host21  |
#  |  jobid2   |  host02  |  host12   |  host22  |
#
# then, once we've identified the `jobid` containing `$(hostname)`, we can use
# that to reconstruct the path to our jobs' `PBS_NODEFILE`, which is located at
#
#     ```bash
#     /var/spool/pbs/aux/${jobid}
#     ````
######################################
ezpz_get_pbs_nodefile_from_hostname() {
    jobid=$(ezpz_get_jobid_from_hostname)
    if [[ -n "${jobid}" ]]; then
        match=$(/bin/ls /var/spool/pbs/aux/ | grep ${jobid})
        hostfile="/var/spool/pbs/aux/${match}"
        if [[ -f "${hostfile}" ]]; then
            export PBS_NODEFILE="${hostfile}"
            export PBS_JOBID=$(echo "${PBS_NODEFILE}" | tr "/" " " | awk '{print $NF}')
            echo "${hostfile}"
        fi
    fi
}


ezpz_save_dotenv() {
    if [[ "$#" -ne 1 ]]; then
        estr="[error]"
        # echo "Expected exactly one argument, specifying outputdir. Received $#"
        printf "%s Expected one argument (outdir). Received: %s" "$(printRed "${estr}")" "$#"
    else
        outdir="$1"
        mkdir -p "${outdir}"
        module list
        dotenv_file="${outdir}/.env"
        echo "Saving environment to ${dotenv_file}"
        printenv | grep -v "LS_COLORS" > "${dotenv_file}"
        export DOTENV_FILE="${dotenv_file}"
    fi
}

######################################################################
# ezpz_get_machine_name: Return current machine name, as lowercase string
######################################################################
ezpz_get_machine_name() {
    if [[ $(hostname) == x4* || $(hostname) == aurora* ]]; then
        machine="aurora"
    elif [[ $(hostname) == x1* || $(hostname) == uan* ]]; then
        machine="sunspot"
    elif [[ $(hostname) == x3* || $(hostname) == polaris* ]]; then
        if [[ "${PBS_O_HOST}" == sirius* ]]; then
            machine="sirius"
        else
            machine="polaris"
        fi
    elif [[ $(hostname) == nid* ]]; then
        machine="perlmutter"
    else
        machine=$(hostname)
    fi
    echo "${machine}"
}

ezpz_check_and_kill_if_running() {
    # kill $(ps aux | grep -E "$USER.+(mpi|main.py)" | grep -v grep | awk '{print $2}')
    RUNNING_PIDS=$(lsof -i:29500 -Fp | head -n 1 | sed 's/^p//')
    if [[ -n "${RUNNING_PIDS}" ]];
        then echo "Caught ${RUNNING_PIDS}" && kill "${RUNNING_PIDS}";
    else
        echo "Not currently running. Continuing!"
    fi
}


ezpz_setup_srun() {
    if [[ $(hostname) == login* || $(hostname) == nid* ]]; then
        export NHOSTS="${SLURM_NNODES:-1}"
        export NGPU_PER_HOST="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
        export NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
        export SRUN_EXEC="srun --gpus ${NGPUS} --gpus-per-node ${NGPU_PER_HOST} -N ${NHOSTS} -n ${NGPUS} -l -u --verbose"
    else
        echo "Skipping ezpz_setup_srun() on $(hostname)"
    fi
}

############################################################################
# save_ds_env
#
# Save important environment variables to .deepspeed_env, which will be
# forwarded to ALL ranks with DeepSpeed 
############################################################################
ezpz_save_ds_env() {
    echo "Saving {PATH, LD_LIBRARY_PATH, htt{p,ps}_proxy, CFLAGS, PYTHONUSERBASE} to .deepspeed_env"
    {
        echo "PATH=${PATH}" ;
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" ;
        echo "http_proxy=${http_proxy:-}" ;
        echo "https_proxy=${https_proxy:-}" ;
        echo "CFLAGS=${CFLAGS}" ;
        echo "PYTHONUSERBASE=$PYTHONUSERBASE" ;
    } > .deepspeed_env
}


###########################
# Setup conda on Sunspot
###########################
ezpz_setup_conda_sunspot() {
    ###### check if CONDA_PREFIX non-empty ################
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module use /soft/preview-modulefiles/24.086.0
        module load frameworks/2024.04.15.002.lua
        # module use /soft/preview-modulefiles/24.086.0 ; module load frameworks/2024.04.15.002.lua
        # source "${WORKING_DIR}/ALCF/sunspot-env-2024-q2.sh"
    fi
}

###########################
# Setup conda on Aurora
###########################
ezpz_setup_conda_aurora() {
    # if [[ -n $(command -v mm) ]]; then
    #     mm activate anl_2024_5_v2
    if [[ -n "${MAMBA_ROOT_PREFIX:-}" ]]; then
        micromamba activate anl_2024_5_v2
    elif [[ -z "${CONDA_PREFIX:-}" ]]; then
        module use -a /soft/modulefiles ; module load frameworks/2024.1
    fi
}

########################
# Setup conda on Sirius
########################
ezpz_setup_conda_sirius() {
    if [[ -z "${CONDA_PREFIX-}" && -z "${VIRTUAL_ENV-}" ]]; then
        export MAMBA_ROOT_PREFIX=/lus/tegu/projects/PolarisAT/foremans/micromamba
        shell_name=$(echo "${SHELL}" | tr "\/" "\t" | awk '{print $NF}')
        eval "$("${MAMBA_ROOT_PREFIX}/bin/micromamba" shell hook --shell "${shell_name}")"
        micromamba activate 2024-04-23
    else
        echo "Found existing python at: $(which python3)"
    fi
}

########################
# Setup conda on Polaris
########################
ezpz_setup_conda_polaris() {
    # unset MPICH_GPU_SUPPORT_ENABLED
    ###### check if CONDA_PREFIX non-empty ################
    if [[ -z "${CONDA_PREFIX-}" ]]; then
        # if so, load the default conda/2024-04-29
        # module and activate base environment
        module use /soft/modulefiles ; module load conda ; conda activate base
    else
        echo "Caught CONDA_PREFIX=${CONDA_PREFIX}"
    fi
}

ezpz_setup_conda() {
    # machine_name=$(ezpz_get_machine_name)
    # if [[ "${machine_name}" == "aurora" ]]; then
    machine_name=$(ezpz_get_machine_name)
    # echo "machine name: ${machine_name}"
    # if [[ $(hostname) == x4* || $(hostname) == aurora* ]]; then
    if [[ "${machine_name}" == "aurora" ]]; then
        ezpz_setup_conda_aurora
    # elif [[ $(hostname) == x1* || $(hostname) == uan* ]]; then
    elif [[ "${machine_name}" == "sunspot" ]]; then
        ezpz_setup_conda_sunspot
    # elif [[ $(hostname) == x3*  || $(hostname) == polaris* ]]; then
    elif [[ "${machine_name}" == "polaris" ]]; then
        if [[ "${PBS_O_HOST:-}" == sirius* ]]; then
            ezpz_setup_conda_sirius
        else
            ezpz_setup_conda_polaris
        fi
    elif [[ $(hostname) == login* || $(hostname) == nid* ]]; then
        echo "Running on Perlmutter !!"
        module load pytorch
        source "${SLURM_SUBMIT_DIR}/venvs/perlmutter/pytorch-2.1.0-cu12/bin/activate"
    else # ------------------------------------- [Unknown] -------------------
        echo "Unknown hostname $(hostname)"
        exit 1
    fi
    # # ----- [Perlmutter @ NERSC] -------------------------------------
}

########################
# setup_venv_from_conda
#
# Build (if necessary) a virtual environment
# on top of the active conda and
# activate it.
# ######################
ezpz_setup_venv_from_conda() {
    if [[ -z "${CONDA_PREFIX}" ]]; then
        echo "!! No ${CONDA_PREFIX} found."  #  Exiting."
        # exit 1
    else
        echo "Found conda at: ${CONDA_PREFIX}"
        CONDA_NAME=$(echo "${CONDA_PREFIX}" | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
        export CONDA_NAME
        if [[ -z "${VIRTUAL_ENV}" ]]; then
            echo "No VIRTUAL_ENV found in environment!"
            echo "    - Trying to setup from ${CONDA_PREFIX}"
            export VENV_DIR="${WORKING_DIR}/venvs/${CONDA_NAME}"
            echo "    - Using VENV_DIR=${VENV_DIR}"
            if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
                printf "\n    - Creating a new virtual env on top of %s in %s\n" "$(printBlue "${CONDA_NAME}")" "$(printGreen "${VENV_DIR}")"
                mkdir -p "${VENV_DIR}"
                python3 -m venv "${VENV_DIR}" --system-site-packages
                source "${VENV_DIR}/bin/activate" || exit
            elif [[ -f "${VENV_DIR}/bin/activate" ]]; then
                echo "    - Found existing venv, activating from $(printBlue "${VENV_DIR}")"
                source "${VENV_DIR}/bin/activate" || exit
            else
                printf "\n    [!! %s]: Unable to locate %s\n" "$(printRed "ERROR")" "$(printMagenta "${VENV_DIR}/bin/activate")"
            fi
        fi
    fi

}

##############################################################################
# `setup_python`:
#
# 1. Setup `conda`
#    - if `conda` nonempty, and `venv` empty, use `conda` to setup `venv`.
#    - if `venv` nonempty, and `conda` empty, what do (???)
#    - if `venv` nonempty and `conda` nonempty, use these
#    - if `conda` empty and `venv` empty:
#       - if `hostname == x4*`, we're on Aurora
#       - if `hostname == x1*`, we're on Sunspot
#       - if `hostname == x3*`, we're on Polaris
#       - if `hostname == nid*`, we're on Perlmutter
#       - otherwise, you're on you're own
#
# 2. Activate (creating, if necessary) a `venv` on top of `base` conda
#    - use the $CONDA_PREFIX to create a venv in
#      `Megatron-DeepSpeed/venvs/${CONDA_PREFIX}`
#      - activate and use this
#
# 3. Print info about which python we're using
##############################################################################
ezpz_setup_python() {
    virtual_env="${VIRTUAL_ENV:-}"
    conda_prefix="${CONDA_PREFIX:-}"
    if [[ -z "${conda_prefix}" && -z "${virtual_env}" ]]; then
        echo "No conda_prefix OR virtual_env found in environment..."
        echo "Setting up conda..."
        ezpz_setup_conda
    elif [[ -n "${conda_prefix}" && -z "${virtual_env}" ]]; then
        echo "No virtual environment found."
        echo "Using conda from: ${conda_prefix}"
        echo "Setting up venv from ${CONDA_PROMPT_MODIFIER:-}"
        ezpz_setup_venv_from_conda
    elif [[ -n "${virtual_env}" && -z "${conda_prefix}" ]]; then
        echo "No conda found."
        echo "Using virtual_env from: ${virtual_env}"
    elif [[ -n "${virtual_env}" && -n "${conda_prefix}" ]]; then
        echo "Using virtual_env: ${virtual_env} on top of conda from: ${conda_prefix}"
    else
        echo "Unable to setup python environment. Exiting"
        exit 1
    fi
    if [[ -z "${virtual_env}" ]]; then
        ezpz_setup_venv_from_conda
    fi
    printf "[python] Using ${MAGENTA}%s${RESET}\n" "$(which python3)"
    export PYTHON_EXEC=$(which python3)
}


whereAmI() {
    python3 -c 'import os; print(os.getcwd())'
}

join_by() {
    local d=${1-} f=${2-}
    if shift 2; then
        printf %s "$f" "${@/#/$d}"
    fi
}


ezpz_parse_hostfile() {
    if [[ "$#" != 1 ]]; then
        echo "Expected exactly one argument: hostfile"
        echo "Received: $#"
    fi
    hf="$1"
    num_hosts=$(ezpz_get_num_hosts "${hf}")
    num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
    num_gpus=$(( num_gpus * num_gpus_per_host ))
    ret=("${num_hosts}" "${num_gpus_per_host}" "${num_gpus}")
    # echo ("${num_hosts}" "${num_gpus_per_host}" "${num_gpus}")
    echo "${ret}"
}

ezpz_get_dist_launch_cmd() {
    if [[ "$#" != 3 ]]; then
        echo "Expected exactly three arguments: hostfile, num_gpus_per_host, num_gpus"
        echo "Received: $#"
    fi
    # hf="$1"
    # # local num_hosts=$(ezpz_get_num_hosts "${hf}")
    # # local num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
    # # local num_gpus=$(( num_gpus * num_gpus_per_host ))
    # local dist_env=$(ezpz_parse_hostfile "${hf}")
    # local num_hosts="${dist_env[1]}"
    # local num_gpus_per_host="${dist_env[2]}"
    # local num_gpus="${dist_env[3]}"
    echo "mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hf} --cpu-bind depth -d 16"
}

ezpz_save_pbs_env() {
    printf "\n[${BLUE}%s${RESET}]\n" "ezpz_save_pbs_env"
    if [[ "$#" == 0 ]]; then
        hostfile="${HOSTFILE:-${PBS_NODEFILE}}"
        jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
    elif [[ "$#" == 1 ]]; then
        printf "    • Caught ${BLUE}%s${RESET} arguments\n" "$#"
        hostfile="$1"
    elif [[ "$#" == 2 ]]; then
        printf "    • Caught ${BLUE}%s${RESET} arguments\n" "$#"
        hostfile="$1"
        jobenv_file="$2"
    else
        hostfile="${HOSTFILE:-${PBS_NODEFILE}}"
        jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
    fi
    if [[ -n $(printenv | grep PBS_JOBID ) ]]; then
        PBS_VARS=$(env | grep PBS)
        echo "${PBS_VARS[*]}" > "${jobenv_file}"
        if [[ "${hostfile}" != "${PBS_NODEFILE:-}" ]]; then
            printf "\n"
            printf "    • Caught ${RED}%s${RESET} != ${RED}%s${RESET} \n" "hostfile" "PBS_NODEFILE"
            printf "        • hostfile: ${RED}%s${RESET}\n" "${hostfile}"
            printf "        • PBS_NODEFILE: ${RED}%s${RESET}\n" "${PBS_NODEFILE}"
            printf "\n"
        fi
        printf "    • Using:\n"
        printf "        • hostfile: ${BLUE}%s${RESET}\n" "${hostfile}"
        printf "        • jobenv_file: ${BLUE}%s${RESET}\n" "${jobenv_file}"
        sed -i 's/^PBS/export\ PBS/g' "${jobenv_file}"
        sed -i 's/^HOSTFILE/export\ HOSTFILE/g' "${jobenv_file}"
        # dist_env=$(ezpz_parse_hostfile "${hostfile}")
        dist_env=() ; dist_env+=($(ezpz_parse_hostfile "$(ezpz_get_pbs_nodefile_from_hostname)"))
        num_hosts="${dist_env[1]}"
        num_gpus_per_host="${dist_env[2]}"
        num_gpus="${dist_env[3]}"
        # dist_launch_cmd=$(ezpz_get_dist_launch_cmd "${hostfile}")
        dist_launch_cmd="mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hostfile} --cpu-bind depth -d 16"
        # num_hosts=$(ezpz_get_num_hosts "${hostfile}")
        # num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
        # num_gpus=$(( num_hosts * num_gpus_per_host ))
        printf "      to calculate:\n"
        printf "        • num_hosts: ${BLUE}%s${RESET}\n" "${num_hosts}"
        printf "        • num_gpus_per_host: ${BLUE}%s${RESET}\n" "${num_gpus_per_host}"
        printf "        • num_gpus: ${BLUE}%s${RESET}\n" "${num_gpus}"
        # getNumGPUs
        # NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
        # export DIST_LAUNCH="mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hostfile} --cpu-bind depth -d 16"
        export DIST_LAUNCH="${dist_launch_cmd}"
        export ezlaunch="${DIST_LAUNCH}"
        # printf "Caught ${BLUE}HOSTFILE${RESET} != ${BLUE}PBS_NODEFILE${RESET} \n"
        printf "        • DIST_LAUNCH: ${BLUE}%s${RESET}\n" "${DIST_LAUNCH}"
    fi
    export HOSTFILE="${hostfile}"
    export JOBENV_FILE="${jobenv_file}"
    printf "    • Setting:\n" 
    printf "        • HOSTFILE: ${BLUE}%s${RESET}\n" "${HOSTFILE}"
    printf "        • JOBENV_FILE: ${BLUE}%s${RESET}\n\n" "${JOBENV_FILE}"
}


ezpz_save_slurm_env() {
    if [[ $(hostname) == nid* || $(hostname) == login* ]]; then
        echo "  Saving SLURM_* to ${SLURM_ENV_FILE}"
        SLURM_VARS=$(env | grep SLU)
        echo "${SLURM_VARS[*]}" > "${SLURM_ENV_FILE}"
        export HOSTFILE="${HOME}/.slurm-nodefile"
        export JOBENV_FILE="${SLURM_ENV_FILE}"
        export SLURM_NODES="${SLURM_NODES}"
        SLURM_NODES=$(scontrol show hostname $SLURM_NODELIST)
        printf "%s\n" "${SLURM_NODES[@]}" > $HOSTFILE
        sed -i 's/^SLURM/export\ SLURM/g' "${SLURM_ENV_FILE}"
        sed -i 's/(x2)//g' "${SLURM_ENV_FILE}"
        export DIST_LAUNCH="srun --gpus ${NGPUS} --gpus-per-node ${NGPU_PER_HOST} -N ${NHOSTS} -n ${NGPUS} -l -u --verbose"  #  "$@"
        export ezlaunch="${DIST_LAUNCH}"
    fi
}


###########################
# ezpz_setup_host
#
# takes 0, 1, or 2 arguments
#
# 0.
#   - hostfile: Look for $HOSTFILE or $PBS_NODEFILE from environment
#   - jobenv_file: Look for $JOBENV_FILE or $PBS_ENV_FILE from environment
#
# 1. hostfile: Specific hostfile to use
# 
# 2.
#   - hostfile: Specific hostfile to use
#   - jobenv_file: Specific `.jobenv` file to use
#
#
# Then, if `hostname` starts with:
#
# - `x3*`: We're on Polaris, with 4 Nvidia A100s per node
# - `x4*` or `x1`: We're on Aurora or Sunspot with 12 Intel PVCs per node
# - `nid` or `login`: We're on Perlmutter with 4 Nvidia A100s per node
#
# if we're on any of the ALCF systems (`x[1-4]*`), we call `ezpz_save_pbs_env`,
# passing along any received arguments
###########################
ezpz_setup_host() {
    printf "[${CYAN}%s${RESET}]\n" "ezpz_setup_host"
    #########################################
    # If no arguments passed ("$#" == 0):
    #
    # - `hostfile` assigned to to the first non-zero variable from:
    #       1. `HOSTFILE`
    #       2. `PBS_NODEFILE`
    # - `jobenv_file` assigned to first non-zero variable from:
    #       1. `JOBENV_FILE`
    #       2. `PBS_ENV_FILE`
    if [[ "$#" == 0 ]]; then
        hostfile="${HOSTFILE:-${PBS_NODEFILE}}"
        jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
        printf "    • Using hostfile: ${CYAN}%s${RESET}\n" "${hostfile}"
        printf "    • Found in environment:\n"
        if [[ -n "${HOSTFILE}" ]]; then
            printf "        • HOSTFILE: ${CYAN}%s${RESET}\n" "${HOSTFILE}"
        fi
        if [[ "${hostfile}" != "${PBS_NODEFILE}" ]]; then
            printf "        • PBS_NODEFILE: ${CYAN}%s${RESET}\n" "${PBS_NODEFILE}"
        fi
    elif [[ "$#" == 1 ]]; then
        printf "    • Caught ${CYAN}%s${RESET} arguments\n" "$#"
        hostfile="$1"
        printf "    • Caught ${CYAN}%s${RESET} arguments\n" "$#"
        printf "    • hostfile=${CYAN}%s${RESET}\n" "${hostfile}"
    elif [[ "$#" == 2 ]]; then
        hostfile="$1"
        jobenv_file="$2"
        printf "    • Caught ${CYAN}%s${RESET} arguments\n" "$#"
        printf "        • hostfile=${CYAN}%s${RESET}\n" "${hostfile}"
        printf "        • jobenv_file=${CYAN}%s${RESET}\n" "${jobenv_file}"
    else
        echo "Expected exactly 0, 1, or 2 arguments, received: $#"
    fi
    printf "        • Writing PBS vars to: ${CYAN}%s${RESET}\n" "${jobenv_file}"
    if [[ $(hostname) == x3* ]]; then
        export GPU_TYPE="NVIDIA"
        # export HOSTFILE="${HOSTFILE:-${PBS_NODEFILE}}"
        ezpz_save_pbs_env "$@"
    elif [[ $(hostname) == x4* || $(hostname) == x1* ]]; then
        export GPU_TYPE="INTEL"
        # export HOSTFILE="${HOSTFILE:-${PBS_NODEFILE}}"
        export NGPU_PER_TILE=6
        export NTILE_PER_HOST=2
        export NGPU_PER_HOST=$(( NGPU_PER_TILE * NTILE_PER_HOST ))
        ezpz_save_pbs_env "$@"
    elif [[ $(hostname) == nid* || $(hostname) == login* ]]; then 
        export GPU_TYPE="NVIDIA"
        export HOSTFILE="${HOME}/.slurm-nodefile"
        ezpz_save_slurm_env
    else
        echo "    Unexpected hostname: $(hostname)"
        export GPU_TYPE="NONE"
        HOSTFILE="hostfile"
        hostname > "${HOSTFILE}"
    fi
}

ezpz_print_hosts() {
    if [[ "$#" == 0 ]]; then
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE}}}"
    elif [[ "$#" == 1 ]]; then
        hostfile="$1"
    else
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE}}}"
    fi
    counter=0
    for f in $(/bin/cat "${hostfile}"); do
        # printf "│     • [host:%s] - \e[1;34m%s\e[0m\n" "${counter}" "${f}"
        printf "    • [host:%s] - ${MAGENTA}%s${RESET}\n" "${counter}" "${f}"
        counter=$((counter+1))
    done
}


ezpz_get_num_xpus() {
    python3 -c 'import intel_extension_for_pytorch as ipex; print(ipex.xpu.device_count())'
}

ezpz_get_num_gpus_nvidia() {
    nGPU=$(nvidia-smi -L | wc -l)
    export NGPU_PER_HOST="${nGPU}"
    echo "${nGPU}"
}

ezpz_get_num_gpus_per_host() {
    if [[ $(hostname) == x4* || $(hostname) == x1* ]]; then
        # export NGPU_PER_HOST=12
        ngpu_per_host=12
    # elif [[ $(hostname) == x3* ]]; then
    #     ngpu_per_host=$(ezpz_get_num_gpus_nvidia)
    else
        # echo "Unknown host $(hostname)"
        ngpu_per_host=$(ezpz_get_num_gpus_nvidia)
    fi
    export NGPU_PER_HOST="${ngpu_per_host}"
    echo "${ngpu_per_host}"
}

ezpz_get_num_hosts() {
    if [[ "$#" == 0 ]]; then
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE}}}"
    elif [[ "$#" == 1 ]]; then
        hostfile="$1"
    else
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE}}}"
    fi
    if [[ -n "${hostfile}" ]]; then
        nhosts=$(wc -l < "${hostfile}")
    elif [[ -n "${SLURM_NNODES:-}" ]]; then
        nhosts=${SLURM_NNODES:-1}
    else
        nhosts=1
    fi
    if [[ -n "${nhosts}" ]]; then
        export NHOSTS="${nhosts}"
    fi
    echo "${nhosts}"
}

ezpz_get_num_gpus_total() {
    num_hosts=$(ezpz_get_num_hosts "$@")
    num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
    num_gpus=$(( num_hosts * num_gpus_per_host ))
    echo "${num_gpus}"
}


ezpz_write_job_info() {
    if [[ "$#" == 0 ]]; then
        hostfile="${HOSTFILE:-${PBS_NODEFILE}}"
        jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
    elif [[ "$#" == 1 ]]; then
        hostfile="$1"
    elif [[ "$#" == 2 ]]; then
        hostfile="$1"
        jobenv_file="$2"
    else
        hostfile="${HOSTFILE:-${PBS_NODEFILE}}"
        jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
    fi
    # printf "[ezpz_write_job_info] Caught jobenv_file: %s\n" "${jobenv_file}"
    # printf "[ezpz_write_job_info] Caught hostfile: %s\n" "${hostfile}"
    # getNumGPUs
    # dist_env=$(ezpz_parse_hostfile "${hostfile}")
    num_hosts=$(ezpz_get_num_hosts "${hostfile}")
    num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
    num_gpus="$(( num_hosts * num_gpus_per_host ))"
    # num_hosts="${dist_env[1]}"
    # num_gpus_per_host="${dist_env[2]}"
    # num_gpus="${dist_env[3]}"
    # dist_launch_cmd=$(ezpz_get_dist_launch_cmd "${hostfile}")
    dist_launch_cmd="mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hostfile} --cpu-bind depth -d 16"
    HOSTS=$(join_by ', ' $(/bin/cat ${hostfile}))
    export NHOSTS="${num_hosts}"
    export NGPU_PER_HOST="${num_gpus_per_host}"
    export NGPUS="${num_gpus}"
    {
        echo "export HOSTFILE=${hostfile}"
        echo "export NHOSTS=${NHOSTS}"
        echo "export NGPU_PER_HOST=${NGPU_PER_HOST}"
        echo "export NGPUS=${NGPUS}"
    } >> "${jobenv_file}"
    export LAUNCH="${dist_launch_cmd}"
    export DIST_LAUNCH="${dist_launch_cmd}"
    export ezlaunch="${DIST_LAUNCH}"
    if [[ -n "${DIST_LAUNCH:-}" ]]; then
        echo "alias LAUNCH='${DIST_LAUNCH}'"
    fi
    # if [[ -n "${DIST_LAUNCH:-}" ]]; then
    #     echo "alias LAUNCH='${DIST_LAUNCH}'"
    # else
    #     export DIST_LAUNCH="${dist_launch_cmd}"
    #     export ezlaunch="${DIST_LAUNCH}"
    # fi
    export LAUNCH="${DIST_LAUNCH}"
    export ezlaunch="${DIST_LAUNCH}"
    alias launch="${LAUNCH}"
    printf "[${MAGENTA}%s${RESET}]\n" "HOSTS"
    # printf "hostfile: ${MAGENTA}%s${RESET}\n" "${hostfile}"
    ezpz_print_hosts
    printf "\n"
    printf "[${YELLOW}%s${RESET}]\n" "DIST INFO"
    printf "    • HOSTFILE=${YELLOW}%s${RESET}\n" "${hostfile}"
    printf "    • NHOSTS=${YELLOW}%s${RESET}\n" "${NHOSTS}"
    printf "    • NGPU_PER_HOST=${YELLOW}%s${RESET}\n" "${NGPU_PER_HOST}"
    printf "    • NGPUS=${YELLOW}%s${RESET}\n" "$NGPUS"
    printf "    • DIST_LAUNCH=${YELLOW}%s${RESET}\n" "${DIST_LAUNCH}"
    printf "\n"
    if [[ -n "$(command -v launch)" ]]; then
        printf "[${GREEN}%s${RESET}]:\n" "LAUNCH"
        printf "    • To launch across all available GPUs, use: ${GREEN}%s${RESET}\n" "launch"
        printf "      ${GREEN}launch${RESET} = ${GREEN}%s${RESET}\n" "${LAUNCH}"
        # printf "      '${GREEN}launch${RESET}' ( = ${GREEN}%s${RESET} )\n" "${LAUNCH}"
    fi
    # echo "export HOSTFILE=${hostfile}" >> "${JOBENV_FILE}"
    # echo "┌────────────────────────────────────────────────────────────────────────────────"
    # echo "│ YOU ARE HERE: $(whereAmI)"
    # echo "│ Run 'source ./bin/getjobenv' in a NEW SHELL to automatically set env vars      "
    # echo "└────────────────────────────────────────────────────────────────────────────────"
    export NHOSTS="${NHOSTS}"
    export NGPU_PER_HOST="${NGPU_PER_HOST}"
    export NGPUS="${NGPUS}"
}

ezpz_save_deepspeed_env() {
    echo "Saving to .deepspeed_env"
    echo "PATH=${PATH}" > .deepspeed_env
    [ "${LD_LIBRARY_PATH}" ] && echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
    [ "${CFLAGS}" ] && echo "CFLAGS=${CFLAGS}" >> .deepspeed_env
    [ "${PYTHONUSERBASE}" ] && echo "PYTHONUSERBASE=${PYTHONUSERBASE}" >> .deepspeed_env
    [ "${http_proxy}" ] && echo "http_proxy=${http_proxy}" >> .deepspeed_env
    [ "${https_proxy}" ] && echo "https_proxy=${https_proxy}" >> .deepspeed_env
}

ezpz_get_pbs_env() {
    if [[ "$#" == 1 ]]; then
        hostfile="$1"
    elif [[ "$#" == 2 ]]; then
        hostfile="$1"
        jobenv_file="$2"
    else
        hostfile="${HOSTFILE:-$(ezpz_get_pbs_nodefile_from_hostname)}"
        jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
    fi
    printf "\n"
    printf "  [${BLUE}ezpz_get_pbs_env${RESET}]: Caught ${BLUE}%s${RESET} arguments\n" "$#"
    printf "      • hostfile: ${BLUE}%s${RESET}\n" "${hostfile}"
    printf "      • jobenv_file: ${BLUE}%s${RESET}\n" "${jobenv_file}"
    if [[ $(hostname) == x3* || $(hostname) == x1* || $(hostname) == x4* ]]; then
        if [[ -n $(cat "${hostfile:-}" | grep "$(hostname)") ]]; then
            num_hosts=$(ezpz_get_num_hosts "${nodefile}")
            num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
            num_gpus="$(( num_hosts * num_gpus_per_host ))"
            dist_launch="mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hostfile} --cpu-bind depth -d 16"
            export DIST_LAUNCH="${dist_launch}"
            export ezlaunch="${DIST_LAUNCH}"
        else
            echo "$(hostname) not found in ${nodefile} ... ?"
        fi
    else
        echo "Skipping ezpz_get_pbs_env() on $(hostname)"
    fi
    printf "${FOOTER}"
}

ezpz_get_slurm_env() {
    if [[ $(hostname) == nid* || $(hostname) == login* ]]; then
        export JOBENV_FILE="${SLURM_ENV_FILE}"
        # shellcheck source="${HOME}/.slurmenv"
        source "${SLURM_ENV_FILE}"
        export DIST_LAUNCH="srun --gpus ${NGPUS} --gpus-per-node ${NGPU_PER_HOST} -N ${NHOSTS} -n ${NGPUS} -l -u --verbose"
        export ezlaunch="${DIST_LAUNCH}"
    else
        echo "Skipping ezpz_get_slurm_env() on $(hostname)"
    fi
}


ezpz_get_job_env() {
    if [[ "$#" == 1 ]]; then
        hostfile="$1"
    elif [[ "$#" == 2 ]]; then
        hostfile="$1"
        jobenv_file="$2"
    else
        # jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-$(ezpz_get_pbs_nodefile_from_hostname)}}"
        # if [[ -z "${hostfile}" ]]; then
        #     hostfile=$(ezpz_get_pbs_nodefile_from_hostname)
        #     if [[ -f "${hostfile}" ]]; then
        #         export PBS_NODEFILE="${hostfile}"
        #     fi
        # fi
        mn=$(ezpz_get_machine_name)
        # if [[ $(hostname) == x1* || $(hostname) == x3* || $(hostname) == x4* ]]; then
        # elif [[ $(hostname) == nid* || $(hostname) == login* ]]; then
        if [[ "${mn}" == "aurora" || "${mn}" == "polaris" || "${mn}" == "sunspot" ]]; then
            jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
            ezpz_get_pbs_env "$@"
        elif [[ "${mn}" == "perlmutter" ]]; then
            jobenv_file="${SLURM_ENV_FILE}"
            ezpz_get_slurm_env
        else
            jobenv_file="${UNKNOWN_ENV_FILE}"
            echo "Unexpected hostname ${HOSTNAME}"
        fi
    fi
    # if [[ -f "${jobenv_file}" ]]; then
    #     source "${jobenv_file}" || exit
    # else
    #     echo "Unable to find ${jobenv_file} on $(hostname)"
    #     exit 1
    # fi
    nhosts=$(wc -l < "${hostfile}")
    local nhosts="${nhosts}"
    export LAUNCH="${DIST_LAUNCH}"
    export ezlaunch="${DIST_LAUNCH}"
    alias launch="${DIST_LAUNCH}"
    export HOSTFILE="${hostfile}"
    export NHOSTS="${nhosts}"
    export NGPU_PER_HOST="${NGPU_PER_HOST}"
    export NGPUS="${NGPUS}"
    export WORLD_SIZE="${NGPUS}"
    hosts_arr=$(/bin/cat "${HOSTFILE}")
    export HOSTS_ARR="${hosts_arr}"
    HOSTS="$(join_by ', ' "$(/bin/cat "${HOSTFILE}")")"
    export HOSTS="${HOSTS}"
}

ezpz_print_job_env() {
    if [[ "$#" == 0 ]]; then
        hostfile="${HOSTFILE:-${PBS_NODEFILE}}"
        jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
    elif [[ "$#" == 1 ]]; then
        hostfile="$1"
    elif [[ "$#" == 2 ]]; then
        hostfile="$1"
        jobenv_file="$2"
    else
        hostfile="${HOSTFILE:-${PBS_NODEFILE}}"
        jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
    fi
    num_hosts=$(ezpz_get_num_hosts "${hostfile}")
    num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
    num_gpus="$(( num_hosts * num_gpus_per_host ))"
    printf "\n"
    printf "  [${MAGENTA}%s${RESET}]:\n" "HOSTS"
    ezpz_print_hosts "${hostfile}"
    printf "\n"
    printf "  [${YELLOW}%s${RESET}]:\n" "DIST INFO"
    printf "      • HOSTFILE=${YELLOW}%s${RESET}\n" "${hostfile}"
    printf "      • NHOSTS=${YELLOW}%s${RESET}\n" "${num_hosts}"
    printf "      • NGPU_PER_HOST=${YELLOW}%s${RESET}\n" "${num_gpus_per_host}"
    printf "      • NGPUS=${YELLOW}%s${RESET}\n" "${num_gpus}"
    printf "      • LAUNCH=${YELLOW}%s${RESET}\n" "${LAUNCH}"
    printf "      • DIST_LAUNCH=${YELLOW}%s${RESET}\n" "${DIST_LAUNCH}"
    printf "\n"
    printf "  [${GREEN}%s${RESET}]:\n" "LAUNCH"
    printf "      • To launch across all available GPUs, use:\n"
    printf "        '${GREEN}launch${RESET}' ( = ${GREEN}%s${RESET} )\n" "${LAUNCH}"
    printf "\n"
}

ezpz_getjobenv_main() {
    ezpz_get_job_env "$@"
    ezpz_setup_host "$@"
    ezpz_write_job_info "$@"
    # ezpz_print_job_env "$@"
}

ezpz_savejobenv_main() {
    # printf "${BLACK}%s${RESET}\n" "${LOGO}"
    # printf "${BLACK}[ezpz]${RESET}\n" "${LOGO_DOOM}"
    ezpz_setup_host "$@"
    ezpz_write_job_info "$@"
}


ezpz_setup_alcf() {
    mn=$(ezpz_get_machine_name)
    hn=$(hostname)
    local mn="${mn}"
    local hn="${hn}"
    printf "\n"
    printf "[${RED}%s${RESET}]\n" "ezpz/bin/utils.sh"
    printf "\n"
    # printf "      • %s: ${BLACK}%s${RESET} ${BLACK}@${RESET} %s\n" "${mn}" $(echo $USER) "${hn}"
    # [tstamp] user @ machine (hostname) in $()
    printf "[${BLACK}%s${RESET}]\n" $(ezpz_get_tstamp)
    printf "    • USER=${BLACK}%s${RESET}\n" "$(echo ${USER})"
    printf "    • MACHINE=${BLACK}%s${RESET}\n" "${mn}"
    printf "    • HOST=${BLACK}%s${RESET}\n" "${hn}"
    #
    # printf "%s ${BLACK}@${RESET} %s (${BLACK}%s${RESET}) in %s" "$(echo $USER)" "${mn}" "${hn}" "${WORKING_DIR}"
    # # printf "      • %s: ${BLACK}%s${RESET} ${BLACK}@${RESET} %s\n" "${mn}" $(echo $USER) "${hn}"
    # printf "      • timestamp: ${BLACK}%s${RESET}\n" "$(ezpz_get_tstamp)"
    printf "\n"
    if [[ "${hn}" == x1* || "${hn}" == x3* || "${hn}" == x4* ]]; then
        # if [[ -z "${PBS_NODEFILE}" ]]; then
        #     pbs_nodefile=$(ezpz_get_pbs_nodefile_from_hostname)
        #     if  [[ -f "${pbs_nodefile}" ]]; then
        #         export PBS_NODEFILE="${pbs_nodefile}"
        #     else
        #         echo "Unable to determine PBS_NODEFILE from hostname"
        #     fi
        # fi
        # if [[ -n "${PBS_NODEFILE:-}" || -f "$(ezpz_get_pbs_nodefile_from_hostname)}" ]]; then
        if [[ -n "${PBS_NODEFILE:-}" ]]; then # || -f "$(ezpz_get_pbs_nodefile_from_hostname)}" ]]; then
            ezpz_savejobenv_main "$@"
        else
            export PBS_NODEFILE=$(ezpz_get_pbs_nodefile_from_hostname)
            ezpz_getjobenv_main "$@"
        fi
    fi
}

###############################################
# Helper functions for printing colored text
###############################################
printBlack() {
    printf "\e[1;30m%s\e[0m\n" "$@"
}

printRed() {
    printf "\e[1;31m%s\e[0m\n" "$@"
}

printGreen() {
    printf "\e[1;32m%s\e[0m\n" "$@"
}

printYellow() {
    printf "\e[1;33m%s\e[0m\n" "$@"
}

printBlue() {
    printf "\e[1;34m%s\e[0m\n" "$@"
}

printMagenta() {
    printf "\e[1;35m%s\e[0m\n" "$@"
}

printCyan() {
    printf "\e[1;36m%s\e[0m\n" "$@"
}

##################
# utils_main
#
# This will get called automatically when running:
#
# ```bash
# $ cd Megatron-DeepSpeed
# $ PBS_O_WORKDIR=$(pwd) source ALCF/utils.sh
# ```
#
# - This will set `"${WORKING_DIR}"`, according to:
#       1. `${PBS_O_WORKDIR}` is nonzero, use this
#       2. else, if `${SLURM_SUBMIT_DIR}` is nonzero use this
#       3. else, use `$(pwd)`
#
#   this is crucial since many of the functions below use paths 
#   which are defined relative to this "${WORKING_DIR}"
#   (e.g. virtual environment, location of executables, etc.)
##################
utils_main() {
    # for debug mode, run with `DEBUG=1`
    if [[ -n "${DEBUG:-}" ]]; then
        set -euxo
    fi
    if [[ -n "${PBS_O_WORKDIR}" ]]; then
        WORKING_DIR="${PBS_O_WORKDIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
        WORKING_DIR="${SLURM_SUBMIT_DIR}"
    else
        echo "Unable to detect PBS or SLURM working directory info..."
        WORKING_DIR=$(python3 -c 'import os; print(os.getcwd())')
        echo "Using ${WORKING_DIR} as working directory..."
    fi
    export WORKING_DIR="${WORKING_DIR}"
    printf "Using WORKING_DIR: %s\n" "${WORKING_DIR}"
}


utils_main
