#!/bin/bash --login
# @file utils.sh
# @brief `ezpz` helper script with functions to make life ez.
# @description
#     This file provides multiple helper functions, all prefixed with "ezpz_"
#      - `ezpz_setup_job`
#      - `ezpz_setup_python`
#      - ...
#

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
# WHITE="\e[1;37m"

# BACKGROUND_BLACK="\e[1;40m"
# BACKGROUND_RED="\e[1;41m"
# BACKGROUND_GREEN="\e[1;42m"
# BACKGROUND_YELLOW="\e[1;43m"
# BACKGROUND_BLUE="\e[1;44m"
# BACKGROUND_MAGENTA="\e[1;45m"
# BACKGROUND_CYAN="\e[1;46m"
# BACKGROUND_WHITE="\e[1;47m"

# BRIGHT_BLACK="\e[1;90m"
# BRIGHT_RED="\e[1;91m"
# BRIGHT_GREEN="\e[1;92m"
# BRIGHT_YELLOW="\e[1;93m"
BRIGHT_BLUE="\e[1;94m"
# BRIGHT_MAGENTA="\e[1;95m"
# BRIGHT_CYAN="\e[1;96m"
# BRIGHT_WHITE="\e[1;97m"

# BACKGROUND_BRIGHT_BLACK="\e[1;100m"
# BACKGROUND_BRIGHT_RED="\e[1;101m"
# BACKGROUND_BRIGHT_GREEN="\e[1;102m"
# BACKGROUND_BRIGHT_YELLOW="\e[1;103m"
# BACKGROUND_BRIGHT_BLUE="\e[1;104m"
# BACKGROUND_BRIGHT_MAGENTA="\e[1;105m"
# BACKGROUND_BRIGHT_CYAN="\e[1;106m"
# BACKGROUND_BRIGHT_WHITE="\e[1;107m"

HOSTNAME=$(hostname)

PBS_ENV_FILE="${HOME}/.pbsenv"
SLURM_ENV_FILE="${HOME}/.slurmenv"

# HEADER_LINE="┌─────────────────────────────────────────────────────────────────────┐\n"
# FOOTER_LINE="└─────────────────────────────────────────────────────────────────────┘\n"
# HEADER="\n"
# FOOTER="\n"

###############################################################################
# Check if running in DEBUG=1 mode.
#   - If so, this will print each command before it is ran and exit if any of
#   them return a nonzero exit status.
###############################################################################
if [[ -n "${DEBUG-}" ]]; then # to use: `DEBUG=1 bash train_llama_alcf.sh`
    printf "\e[1;31m%s\e[0m\n" "!! RUNNING IN DEBUG MODE !!"
    _shell_name=$(ezpz_get_shell_name)
    if [[ "${_shell_name}" == "zsh" ]]; then
        echo "No debug"
        # set -x # o pipefail
    else
        set -euxo pipefail
    fi
fi

###############################################################################
# Print (but DO NOT EXECUTE !!) each command that would be ran.
#
# Enable with: NOOP=1 PBS_O_WORKDIR=$(pwd) bash train_llama_alcf.sh
###############################################################################
if [[ -v NOOP ]]; then # to use: `NOOP=1 bash train_llama_alcf.sh`
    echo "Run NOOP mode"
    set -o noexec # same as set -n
fi

# @description Get name of shell.
# Strip off `/bin/` substr from "${SHELL}" env var and return this string.
#
# @example
#    $ echo "${SHELL}"
#    /bin/zsh
#    $ ezpz_get_shell_name
#    zsh
ezpz_get_shell_name() {
    echo "${SHELL}" | sed -e "s/\/bin\///g"
}

ezpz_get_tstamp() {
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
    qstat -u "${USER}" -n1rw | sed -e "s/\/0\*208/\ /g" | tr "+|." "\ " | awk '{a = ""; for (i = 13 ; i <= NF ; i++) a = a " " $i; print $1 a}' | grep -vE "aurora-pbs|Req|Job|\-\-"
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
#######################
ezpz_reset_pbs_vars() {
    wd="${PBS_O_WORKDIR:-${WORKING_DIR:-$(pwd)}}"
    vars=($(printenv | grep -iE "^PBS" | tr "=" " " | awk '{print $1}'))
    for v in "$vars[@]"; do echo "Unsetting $v" && unset -v "${v}"; done
    export PBS_O_WORKDIR="${wd}"
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
        match=$(/bin/ls /var/spool/pbs/aux/ | grep "${jobid}")
        hostfile="/var/spool/pbs/aux/${match}"
        if [[ -f "${hostfile}" ]]; then
            export PBS_NODEFILE="${hostfile}"
            _pbs_jobid=$(echo "${PBS_NODEFILE}" | tr "/" " " | awk '{print $NF}')
            export PBS_JOBID="${_pbs_jobid}"
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
        printenv | grep -v "LS_COLORS" >"${dotenv_file}"
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
    elif [[ $(hostname) == sophia* ]]; then
        machine="sophia"
    elif [[ $(hostname) == x3* || $(hostname) == polaris* ]]; then
        if [[ "${PBS_O_HOST}" == sirius* ]]; then
            machine="sirius"
        else
            machine="polaris"
        fi
    elif [[ $(hostname) == frontier* ]]; then
        machine="frontier"
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
    if [[ -n "${RUNNING_PIDS}" ]]; then
        echo "Caught ${RUNNING_PIDS}" && kill "${RUNNING_PIDS}"
    else
        echo "Not currently running. Continuing!"
    fi
}

#################################
# ezpz_get_slurm_running_jobid
# Retruns SLURM_JOBID of running slurm jobs
#################################
ezpz_get_slurm_running_jobid() {
    if [[ -n $(command -v sacct) ]]; then
        jobid=$(sacct --format=JobID,NodeList%-30,state%20 --user "${USER}" -s R | grep -Ev "\.int|\.ext|^JobID|^---" | awk '{print $1}')
        echo "${jobid}"
    fi
}

ezpz_get_slurm_running_nodelist() {
    if [[ -n $(command -v sacct) ]]; then
        slurm_nodelist=$(sacct --format=JobID,NodeList%-30,state%20 --user $USER -s R | grep -Ev "\.int|\.ext|^JobID|^---" | awk '{print $2}')
        echo "${slurm_nodelist}"
    fi
}

ezpz_make_slurm_nodefile() {
    if [[ "$#" == 1 ]]; then
        outfile="$1"
    else
        outfile="nodefile"
    fi
    snodelist="${SLURM_NODELIST:-$(ezpz_get_slurm_running_nodelist)}"
    if [[ -n $(command -v scontrol) ]]; then
        scontrol show hostname "${snodelist}" >"${outfile}"
        echo "${outfile}"
    fi
}

ezpz_setup_srun() {
    # if [[ $(hostname) == login* || $(hostname) == nid* ]]; then
    export NHOSTS="${SLURM_NNODES:-1}"
    export NGPU_PER_HOST="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
    export HOSTFILE="${HOSTFILE:-$(ezpz_make_slurm_nodefile "$@")}"
    export NGPUS="$((NHOSTS * NGPU_PER_HOST))"
    export SRUN_EXEC="srun -l -u --verbose -N${SLURM_NNODES} -n$((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
    # export SRUN_EXEC="srun --gpus ${NGPUS} --gpus-per-node ${NGPU_PER_HOST} -N ${NHOSTS} -n ${NGPUS} -l -u --verbose"
    # else
    #     echo "Skipping ezpz_setup_srun() on $(hostname)"
    # fi
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
        echo "PATH=${PATH}"
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
        echo "http_proxy=${http_proxy:-}"
        echo "https_proxy=${https_proxy:-}"
        echo "CFLAGS=${CFLAGS}"
        echo "PYTHONUSERBASE=$PYTHONUSERBASE"
    } >.deepspeed_env
}

###########################
# Setup conda on Frontier
###########################
ezpz_setup_conda_frontier() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module load PrgEnv-gnu/8.5.0
        module load craype-accel-amd-gfx90a
        module load rocm
        micromamba activate /lustre/orion/csc613/scratch/foremans/envs/micromamba/py3.10-torch2.2-rocm5.7
        # module load miniforge3/23.11.0-0
        # eval "$(/autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/bin/conda "shell.$(echo $SHELL | tr '/' ' ' | awk '{print $NF}')" hook)"
        # conda activate pytorch2.2-rocm5.7
    fi
}

###########################
# Setup conda on Sunspot
###########################
ezpz_setup_conda_sunspot() {
    ###### check if CONDA_PREFIX non-empty ################
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module use /opt/aurora/24.180.1/modulefiles
        module load frameworks/2024.2.1_u1
        # module use /soft/preview-modulefiles/24.086.0
        # module load frameworks/2024.04.15.002.lua
        # module use /soft/preview-modulefiles/24.086.0 ; module load frameworks/2024.04.15.002.lua
        # source "${WORKING_DIR}/ALCF/sunspot-env-2024-q2.sh"
    fi
}

###########################
# Setup conda on Aurora
###########################
ezpz_setup_conda_aurora() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        # NOTE: Updated 2024-10-08 [@saforem2]
        module load frameworks
        module load mpich
    else
        printf "Caught CONDA_PREFIX=%s from environment, using this!" "${CONDA_PREFIX}"
    fi
}

########################
# Setup conda on Sirius
########################
ezpz_setup_conda_sirius() {
    if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV-}" ]]; then
        export MAMBA_ROOT_PREFIX=/lus/tegu/projects/PolarisAT/foremans/micromamba
        shell_name=$(echo "${SHELL}" | tr "\/" "\t" | awk '{print $NF}')
        eval "$("${MAMBA_ROOT_PREFIX}/bin/micromamba" shell hook --shell "${shell_name}")"
        micromamba activate 2024-04-23
    else
        echo "Found existing python at: $(which python3)"
    fi
}

# ########################
# # Setup conda on Sophia
# ########################
ezpz_setup_conda_sophia() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module load conda
        conda activate base
    else
        echo "Caught CONDA_PREFIX=${CONDA_PREFIX}"
    fi
}

########################
# Setup conda on Polaris
########################
ezpz_setup_conda_polaris() {
    # unset MPICH_GPU_SUPPORT_ENABLED
    ###### check if CONDA_PREFIX non-empty ################
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        # if so, load the default conda/2024-04-29
        # module and activate base environment
        module use /soft/modulefiles
        module load conda
        conda activate base
    else
        echo "Caught CONDA_PREFIX=${CONDA_PREFIX}"
    fi
}

ezpz_setup_conda() {
    # machine_name=$(ezpz_get_machine_name)
    # if [[ "${machine_name}" == "aurora" ]]; then
    machine_name=$(ezpz_get_machine_name)
    # echo "machine name: ${machine_name}"
    if [[ "${machine_name}" == "aurora" ]]; then
        ezpz_setup_conda_aurora
    elif [[ "${machine_name}" == "sophia" ]]; then
        ezpz_setup_conda_sophia
    elif [[ "${machine_name}" == "sunspot" ]]; then
        ezpz_setup_conda_sunspot
    elif [[ "${machine_name}" == "polaris" ]]; then
        if [[ "${PBS_O_HOST:-}" == sirius* ]]; then
            ezpz_setup_conda_sirius
        else
            ezpz_setup_conda_polaris
        fi
    elif [[ $(hostname) == frontier* ]]; then
        ezpz_setup_conda_frontier
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
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        echo "!! No CONDA_PREFIX var found." #  Exiting."
        # exit 1
    else
        echo "Found conda at: ${CONDA_PREFIX}"
        CONDA_NAME=$(echo "${CONDA_PREFIX}" | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
        export CONDA_NAME
        if [[ -z "${VIRTUAL_ENV:-}" ]]; then
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
    python_exec=$(which python3)
    export PYTHON_EXEC="${python_exec}"
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
    num_gpus=$((num_hosts * num_gpus_per_host))
    echo "${num_hosts}" "${num_gpus_per_host}" "${num_gpus}"
}

ezpz_get_dist_launch_cmd() {
    if [[ "$#" != 1 ]]; then
        echo "Expected exactly one argument: hostfile"
        echo "Received: $#"
    fi
    hf="$1"
    mn=$(ezpz_get_machine_name)
    num_hosts=$(ezpz_get_num_hosts "${hf}")
    num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
    num_gpus="$((num_hosts * num_gpus_per_host))"
    num_cores_per_host=$(getconf _NPROCESSORS_ONLN)
    num_cpus_per_host=$((num_cores_per_host / 2))
    depth=$((num_cpus_per_host / num_gpus_per_host))
    if [[ "${mn}" == "sophia" ]]; then
        dist_launch_cmd="mpirun -n ${num_gpus} -N ${num_gpus_per_host} --hostfile ${hostfile} -x PATH -x LD_LIBRARY_PATH"
    else
        dist_launch_cmd="mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hostfile} --cpu-bind depth -d ${depth}"
    fi
    if [[ "${mn}" == "aurora" ]]; then
        dist_launch_cmd="${dist_launch_cmd} --no-vni"
    fi
    echo "${dist_launch_cmd}"
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
    if [[ -n $(printenv | grep PBS_JOBID) ]]; then
        PBS_VARS=$(env | grep PBS)
        echo "${PBS_VARS[*]}" >"${jobenv_file}"
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
        num_hosts=$(ezpz_get_num_hosts "${hostfile}")
        num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
        num_gpus="$((num_hosts * num_gpus_per_host))"
        num_cores_per_host=$(getconf _NPROCESSORS_ONLN)
        num_cpus_per_host=$((num_cores_per_host / 2))
        depth=$((num_cpus_per_host / num_gpus_per_host))
        dist_launch_cmd=$(ezpz_get_dist_launch_cmd "${hostfile}")

        # dist_launch_cmd="mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hostfile} --cpu-bind depth -d ${depth}"
        # dist_env=()
        # dist_env+=($(ezpz_parse_hostfile "$(ezpz_get_pbs_nodefile_from_hostname)"))
        # num_hosts="${dist_env[1]}"
        # num_gpus_per_host="${dist_env[2]}"
        # num_gpus="${dist_env[3]}"
        # dist_launch_cmd=$(ezpz_get_dist_launch_cmd "${hostfile}")
        # num_hosts=$(ezpz_get_num_hosts "${hostfile}")
        # num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
        # num_gpus=$(( num_hosts * num_gpus_per_host ))
        printf "      to calculate:\n"
        printf "        • num_hosts: ${BLUE}%s${RESET}\n" "${num_hosts}"
        printf "        • num_cores_per_host: ${BLUE}%s${RESET}\n" "${num_cores_per_host}"
        printf "        • num_cpus_per_host: ${BLUE}%s${RESET}\n" "${num_cpus_per_host}"
        printf "        • num_gpus_per_host: ${BLUE}%s${RESET}\n" "${num_gpus_per_host}"
        printf "        • depth: ${BLUE}%s${RESET}\n" "${depth}"
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
    printf "\n[${BLUE}%s${RESET}]\n" "ezpz_save_slurm_env"
    if [[ "$#" == 0 ]]; then
        # hostfile="${HOSTFILE:-${PBS_NODEFILE}}"
        hostfile="${HOSTFILE:-$(ezpz_make_slurm_nodefile)}"
        jobenv_file="${JOBENV_FILE:-${SLURM_ENV_FILE}}"
    elif [[ "$#" == 1 ]]; then
        printf "    • Caught ${BLUE}%s${RESET} arguments\n" "$#"
        hostfile="$1"
        jobenv_file="${JOBENV_FILE:-${SLURM_ENV_FILE}}"
    elif [[ "$#" == 2 ]]; then
        printf "    • Caught ${BLUE}%s${RESET} arguments\n" "$#"
        hostfile="$1"
        jobenv_file="$2"
    else
        hostfile="${HOSTFILE:-$(ezpz_make_slurm_nodefile)}"
        jobenv_file="${JOBENV_FILE:-${SLURM_ENV_FILE}}"
    fi
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        SLURM_VARS=$(env | grep SLU)
        echo "${SLURM_VARS[*]}" >"${jobenv_file}"
        # if [[ "${hostfile}" != "${SLURM_NODEFILE:-}" ]]; then
        #     printf "\n"
        #     printf "    • Caught ${RED}%s${RESET} != ${RED}%s${RESET} \n" "hostfile" "SLURM_NODEFILE"
        #     printf "        • hostfile: ${RED}%s${RESET}\n" "${hostfile}"
        #     printf "        • SLURM_NODEFILE: ${RED}%s${RESET}\n" "${SLURM_NODEFILE}"
        #     printf "\n"
        # fi
        printf "    • Using:\n"
        printf "        • hostfile: ${BLUE}%s${RESET}\n" "${hostfile}"
        printf "        • jobenv_file: ${BLUE}%s${RESET}\n" "${jobenv_file}"

        sed -i 's/^SLURM/export\ SLURM/g' "${jobenv_file}"
        sed -i 's/^HOSTFILE/export\ HOSTFILE/g' "${jobenv_file}"
        num_hosts=$(ezpz_get_num_hosts "${hostfile}")
        num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
        num_gpus="$((num_hosts * num_gpus_per_host))"
        # dist_env=$(ezpz_parse_hostfile "${hostfile}")
        # dist_env=()
        # dist_env+=("$(ezpz_parse_hostfile "$(ezpz_make_slurm_nodefile)")")
        # num_hosts="${dist_env[1]}"
        # num_gpus_per_host="${dist_env[2]}"
        # num_gpus="${dist_env[3]}"
        # dist_launch_cmd="srun -N ${num_hosts} -n ${num_gpus} -l -u --verbose"
        dist_launch_cmd="srun -l -u --verbose -N${SLURM_NNODES} -n$((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
        printf "      to calculate:\n"
        printf "        • num_hosts: ${BLUE}%s${RESET}\n" "${num_hosts}"
        printf "        • num_gpus_per_host: ${BLUE}%s${RESET}\n" "${num_gpus_per_host}"
        printf "        • num_gpus: ${BLUE}%s${RESET}\n" "${num_gpus}"
        export DIST_LAUNCH="${dist_launch_cmd}"
        export ezlaunch="${DIST_LAUNCH}"
        printf "        • DIST_LAUNCH: ${BLUE}%s${RESET}\n" "${DIST_LAUNCH}"
    fi
    export HOSTFILE="${hostfile}"
    export JOBENV_FILE="${jobenv_file}"
    printf "    • Setting:\n"
    printf "        • HOSTFILE: ${BLUE}%s${RESET}\n" "${HOSTFILE}"
    printf "        • JOBENV_FILE: ${BLUE}%s${RESET}\n\n" "${JOBENV_FILE}"
}

ezpz_setup_host_slurm() {
    printf "[${CYAN}%s${RESET}]\n" "ezpz_setup_host_slurm"
    mn=$(ezpz_get_machine_name)
    scheduler_type=$(ezpz_get_scheduler_type)
    if [[ "${scheduler_type}" == "slurm" ]]; then
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
            hostfile="${HOSTFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}"
            jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
            printf "    • Using hostfile: ${CYAN}%s${RESET}\n" "${hostfile}"
            printf "    • Found in environment:\n"
            if [[ -n "${HOSTFILE:-}" ]]; then
                printf "        • HOSTFILE: ${CYAN}%s${RESET}\n" "${HOSTFILE}"
            fi
            # if [[ "${hostfile}" != "${PBS_NODEFILE}" ]]; then
        elif [[ "$#" == 1 ]]; then
            printf "    • Caught ${CYAN}%s${RESET} arguments\n" "$#"
            hostfile="$1"
            jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
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
        printf "        • Writing SLURM vars to: ${CYAN}%s${RESET}\n" "${jobenv_file}"
        if [[ "${mn}" == "frontier" ]]; then
            export GPU_TYPE="AMD"
            _hostfile=$(ezpz_make_slurm_nodefile)
            export HOSTFILE="${_hostfile}"
            ezpz_save_slurm_env "$@"
        elif [[ $(hostname) == nid* || $(hostname) == login* ]]; then
            export GPU_TYPE="NVIDIA"
            _hostfile="$(ezpz_make_slurm_nodefile)"
            export HOSTFILE="${_hostfile}"
            ezpz_save_slurm_env "$@"
        fi
    fi
}

ezpz_setup_host_pbs() {
    printf "[${CYAN}%s${RESET}]\n" "ezpz_setup_host_pbs"
    mn=$(ezpz_get_machine_name)
    scheduler_type=$(ezpz_get_scheduler_type)
    if [[ "${scheduler_type}" == "pbs" ]]; then
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
            # hostfile="${HOSTFILE:-${PBS_NODEFILE}}"
            # jobenv_file="$(ezpz_get_jobenv_file)"
            # jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
            hostfile="${HOSTFILE:-$(ezpz_get_pbs_nodefile_from_hostname)}"
            jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
            printf "    • Using hostfile: ${CYAN}%s${RESET}\n" "${hostfile}"
            printf "    • Found in environment:\n"
            if [[ -n "${HOSTFILE:-}" ]]; then
                printf "        • HOSTFILE: ${CYAN}%s${RESET}\n" "${HOSTFILE}"
            fi
            # if [[ "${hostfile}" != "${PBS_NODEFILE}" ]]; then
            if [[ "${scheduler_type}" == "pbs" && "${hostfile}" != "${PBS_NODEFILE:-}" ]]; then
                printf "        • PBS_NODEFILE: ${CYAN}%s${RESET}\n" "${PBS_NODEFILE}"
            fi
        elif [[ "$#" == 1 ]]; then
            printf "    • Caught ${CYAN}%s${RESET} arguments\n" "$#"
            hostfile="$1"
            jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
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
        hn=$(hostname)
        if [[ "${hn}" == x* || "${hn}" == "sophia*" ]]; then
            printf "        • Writing PBS vars to: ${CYAN}%s${RESET}\n" "${jobenv_file}"
            if [[ "${mn}" == "polaris" || "${mn}" == "sirius" || "${mn}" == "sophia*" ]]; then
                export GPU_TYPE="NVIDIA"
                ezpz_save_pbs_env "$@"
            elif [[ "${mn}" == "aurora" || "${mn}" == "sunspot" ]]; then
                export GPU_TYPE="INTEL"
                export NGPU_PER_TILE=6
                export NTILE_PER_HOST=2
                export NGPU_PER_HOST=$((NGPU_PER_TILE * NTILE_PER_HOST))
                ezpz_save_pbs_env "$@"
            fi
        fi
    fi
}

ezpz_setup_host() {
    mn=$(ezpz_get_machine_name)
    scheduler_type=$(ezpz_get_scheduler_type)
    # printf "[${CYAN}%s${RESET}]\n" "ezpz_setup_host"
    if [[ "${scheduler_type}" == "pbs" ]]; then
        ezpz_setup_host_pbs "$@"
    elif [[ "${scheduler_type}" ]]; then
        ezpz_setup_host_slurm "$@"
    else
        echo "Unknown scheduler: ${scheduler_type} on ${mn}"
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
ezpz_setup_host_old() {
    printf "[${CYAN}%s${RESET}]\n" "ezpz_setup_host"
    mn=$(ezpz_get_machine_name)
    scheduler_type=$(ezpz_get_scheduler_type)
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
        # hostfile="${HOSTFILE:-${PBS_NODEFILE}}"
        # jobenv_file="$(ezpz_get_jobenv_file)"
        # jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
        if [[ "${scheduler_type}" == "slurm" ]]; then
            hostfile="${HOSTFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}"
        elif [[ "${scheduler_type}" == "pbs" ]]; then
            hostfile="${HOSTFILE:-$(ezpz_get_pbs_nodefile_from_hostname)}"
        else
            echo "Unknown scheduler: $(ezpz_get_scheduler_type)"
        fi
        jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
        printf "    • Using hostfile: ${CYAN}%s${RESET}\n" "${hostfile}"
        printf "    • Found in environment:\n"
        if [[ -n "${HOSTFILE:-}" ]]; then
            printf "        • HOSTFILE: ${CYAN}%s${RESET}\n" "${HOSTFILE}"
        fi
        # if [[ "${hostfile}" != "${PBS_NODEFILE}" ]]; then
        if [[ "${scheduler_type}" == "pbs" && "${hostfile}" != "${PBS_NODEFILE:-}" ]]; then
            printf "        • PBS_NODEFILE: ${CYAN}%s${RESET}\n" "${PBS_NODEFILE}"
        fi
    elif [[ "$#" == 1 ]]; then
        printf "    • Caught ${CYAN}%s${RESET} arguments\n" "$#"
        hostfile="$1"
        jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
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
    if [[ "${scheduler_type:-}" == "pbs" ]]; then # && "${hostfile}" != "${PBS_NODEFILE:-}" ]]; then
        # if [[ $(hostname) == x3* ]]; then
        hn=$(hostname)
        if [[ "${hn}" == x* || "${hn}" == "sophia*" ]]; then
            printf "        • Writing PBS vars to: ${CYAN}%s${RESET}\n" "${jobenv_file}"
            if [[ "${mn}" == "polaris" || "${mn}" == "sirius" || "${mn}" == "sophia" ]]; then
                export GPU_TYPE="NVIDIA"
                ezpz_save_pbs_env "$@"
            elif [[ "${mn}" == "aurora" || "${mn}" == "sunspot" ]]; then
                export GPU_TYPE="INTEL"
                export NGPU_PER_TILE=6
                export NTILE_PER_HOST=2
                export NGPU_PER_HOST=$((NGPU_PER_TILE * NTILE_PER_HOST))
                ezpz_save_pbs_env "$@"
            fi
        fi
    elif [[ "${scheduler_type:-}" == "slurm" ]]; then
        printf "        • Writing SLURM vars to: ${CYAN}%s${RESET}\n" "${jobenv_file}"
        if [[ "${mn}" == "frontier" ]]; then
            export GPU_TYPE="AMD"
            _hostfile=$(ezpz_make_slurm_nodefile)
            export HOSTFILE="${_hostfile}"
            ezpz_save_slurm_env "$@"
        elif [[ $(hostname) == nid* || $(hostname) == login* ]]; then
            export GPU_TYPE="NVIDIA"
            _hostfile="$(ezpz_make_slurm_nodefile)"
            export HOSTFILE="${_hostfile}"
            ezpz_save_slurm_env "$@"
        fi
    else
        echo "!! Unknown scheduler !! Neither 'pbs' nor 'slurm' ?? ${scheduler_type:-}"
        echo "    Unexpected hostname: $(hostname)"
        export GPU_TYPE="NONE"
        HOSTFILE="hostfile"
        hostname >"${HOSTFILE}"
    fi
}

ezpz_print_hosts() {
    if [[ "$#" == 0 ]]; then
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}}"
    elif [[ "$#" == 1 ]]; then
        hostfile="$1"
    else
        # hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE}}}"
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}}"
    fi
    counter=0
    for f in $(/bin/cat "${hostfile}"); do
        # printf "│     • [host:%s] - \e[1;34m%s\e[0m\n" "${counter}" "${f}"
        printf "    • [host:${MAGENTA}%s${RESET}] - ${MAGENTA}%s${RESET}\n" "${counter}" "${f}"
        counter=$((counter + 1))
    done
}

ezpz_get_num_xpus() {
    python3 -c 'import intel_extension_for_pytorch as ipex; print(ipex.xpu.device_count())'
}

ezpz_get_num_gpus_nvidia() {
    if [[ -n "$(command -v nvidia-smi)" ]]; then
        num_gpus=$(nvidia-smi -L | wc -l)
    else
        num_gpus=$(python3 -c 'import torch; print(torch.cuda.device_count())')
    fi
    export NGPU_PER_HOST="${num_gpus}"
    echo "${num_gpus}"
}

ezpz_get_num_gpus_per_host() {
    mn=$(ezpz_get_machine_name)
    # export NGPU_PER_HOST=12
    if [[ "${mn}" == "aurora" || "${mn}" == "sunspot" ]]; then
        ngpu_per_host=12
    elif [[ "${mn}" == "frontier" ]]; then
        ngpu_per_host=8
    else
        ngpu_per_host=$(ezpz_get_num_gpus_nvidia)
    fi
    export NGPU_PER_HOST="${ngpu_per_host}"
    echo "${ngpu_per_host}"
}

ezpz_get_num_hosts() {
    if [[ "$#" == 0 ]]; then
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}}"
    elif [[ "$#" == 1 ]]; then
        hostfile="$1"
    else
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}}"
    fi
    if [[ -n "${hostfile}" ]]; then
        nhosts=$(wc -l <"${hostfile}")
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
    num_gpus=$((num_hosts * num_gpus_per_host))
    echo "${num_gpus}"
}

ezpz_get_jobenv_file() {
    mn=$(ezpz_get_machine_name)
    if [[ "${mn}" == "aurora" || "${mn}" == "polaris" || "${mn}" == "sunspot" || "${mn}" == "sirius" || "${mn}" == "sophia" ]]; then
        echo "${JOBENV_FILE:-${PBS_ENV_FILE}}"
    elif [[ "${mn}" == "frontier" || "${mn}" == "perlmutter" || -n "${SLURM_JOB_ID:-}" ]]; then
        echo "${JOBENV_FILE:-${SLURM_ENV_FILE}}"
    fi
}

ezpz_get_scheduler_type() {
    mn=$(ezpz_get_machine_name)
    if [[ "${mn}" == "aurora" || "${mn}" == "polaris" || "${mn}" == "sunspot" || "${mn}" == "sirius" || "${mn}" == "sophia" ]]; then
        echo "pbs"
    elif [[ "${mn}" == "frontier" || "${mn}" == "perlmutter" || -n "${SLURM_JOB_ID:-}" ]]; then
        echo "slurm"
    fi

}

ezpz_write_job_info() {
    if [[ "$#" == 0 ]]; then
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}}"
        jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
        # jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
    elif [[ "$#" == 1 ]]; then
        hostfile="$1"
        jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
    elif [[ "$#" == 2 ]]; then
        hostfile="$1"
        jobenv_file="$2"
    else
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}}"
        # jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
        jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
    fi
    # printf "[ezpz_write_job_info] Caught jobenv_file: %s\n" "${jobenv_file}"
    # printf "[ezpz_write_job_info] Caught hostfile: %s\n" "${hostfile}"
    # getNumGPUs
    # dist_env=$(ezpz_parse_hostfile "${hostfile}")
    # num_hosts=$(ezpz_get_num_hosts "${hostfile}")
    # num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
    # num_gpus="$((num_hosts * num_gpus_per_host))"
    # num_hosts="${dist_env[1]}"
    # num_gpus_per_host="${dist_env[2]}"
    # num_gpus="${dist_env[3]}"
    # dist_launch_cmd=$(ezpz_get_dist_launch_cmd "${hostfile}")
    scheduler_type=$(ezpz_get_scheduler_type)
    num_hosts=$(ezpz_get_num_hosts "${hostfile}")
    num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
    num_gpus="$((num_hosts * num_gpus_per_host))"
    num_cores_per_host=$(getconf _NPROCESSORS_ONLN)
    num_cpus_per_host=$((num_cores_per_host / 2))
    depth=$((num_cpus_per_host / num_gpus_per_host))
    if [[ "${scheduler_type}" == "pbs" ]]; then
        # dist_launch_cmd="mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hostfile} --cpu-bind depth -d ${depth}"
        dist_launch_cmd=$(ezpz_get_dist_launch_cmd "${hostfile}")
    elif [[ "${scheduler_type}" == "slurm" ]]; then
        # dist_launch_cmd="srun -N ${num_hosts} -n ${num_gpus} -l -u --verbose"
        dist_launch_cmd="srun -l -u --verbose -N${SLURM_NNODES} -n$((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
    else
        echo Unknown scheduler!
    fi
    if [[ -f "${hostfile:-}" ]]; then
        HOSTS=$(join_by ', ' "$(/bin/cat "${hostfile}")")
        export NHOSTS="${num_hosts}"
        export NGPU_PER_HOST="${num_gpus_per_host}"
        export NGPUS="${num_gpus}"
        {
            echo "export HOSTFILE=${hostfile}"
            echo "export NHOSTS=${NHOSTS}"
            echo "export NGPU_PER_HOST=${NGPU_PER_HOST}"
            echo "export NGPUS=${NGPUS}"
        } >>"${jobenv_file}"
        export LAUNCH="${dist_launch_cmd}"
        export DIST_LAUNCH="${dist_launch_cmd}"
        export ezlaunch="${DIST_LAUNCH}"
        # if [[ -n "${DIST_LAUNCH:-}" ]]; then
        #     echo "alias LAUNCH='${DIST_LAUNCH}'"
        # fi
        export LAUNCH="${DIST_LAUNCH}"
        export ezlaunch="${DIST_LAUNCH}"
        alias launch="${LAUNCH}"
        printf "[${MAGENTA}%s${RESET}]\n" "HOSTS"
        # printf "hostfile: ${MAGENTA}%s${RESET}\n" "${hostfile}"
        ezpz_print_hosts "${hostfile}"
        printf "\n"
        printf "[${BRIGHT_BLUE}%s${RESET}]\n" "DIST INFO"
        printf "    • NGPUS=${BRIGHT_BLUE}%s${RESET}\n" "$NGPUS"
        printf "    • NHOSTS=${BRIGHT_BLUE}%s${RESET}\n" "${NHOSTS}"
        printf "    • NGPU_PER_HOST=${BRIGHT_BLUE}%s${RESET}\n" "${NGPU_PER_HOST}"
        printf "    • HOSTFILE=${BRIGHT_BLUE}%s${RESET}\n" "${hostfile}"
        printf "    • DIST_LAUNCH=${BRIGHT_BLUE}%s${RESET}\n" "${DIST_LAUNCH}"
        printf "\n"
        if [[ -n "$(command -v launch)" ]]; then
            printf "[${GREEN}%s${RESET}]:\n" "LAUNCH"
            printf "    • To launch across all available GPUs, use: ${GREEN}%s${RESET}\n" "launch"
            printf "\n"
            printf "      ${GREEN}launch${RESET} = ${GREEN}%s${RESET}\n" "${LAUNCH}"
            # printf "      '${GREEN}launch${RESET}' ( = ${GREEN}%s${RESET} )\n" "${LAUNCH}"
        fi
        printf "\n"
        # echo "export HOSTFILE=${hostfile}" >> "${JOBENV_FILE}"
        # echo "┌────────────────────────────────────────────────────────────────────────────────"
        # echo "│ YOU ARE HERE: $(whereAmI)"
        # echo "│ Run 'source ./bin/getjobenv' in a NEW SHELL to automatically set env vars      "
        # echo "└────────────────────────────────────────────────────────────────────────────────"
        # export NHOSTS="${NHOSTS}"
        # export NGPU_PER_HOST="${NGPU_PER_HOST}"
        # export NGPUS="${NGPUS}"
    fi
}

ezpz_save_deepspeed_env() {
    echo "Saving to .deepspeed_env"
    echo "PATH=${PATH}" >.deepspeed_env
    [ "${LD_LIBRARY_PATH}" ] && echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >>.deepspeed_env
    [ "${CFLAGS}" ] && echo "CFLAGS=${CFLAGS}" >>.deepspeed_env
    [ "${PYTHONUSERBASE}" ] && echo "PYTHONUSERBASE=${PYTHONUSERBASE}" >>.deepspeed_env
    [ "${http_proxy}" ] && echo "http_proxy=${http_proxy}" >>.deepspeed_env
    [ "${https_proxy}" ] && echo "https_proxy=${https_proxy}" >>.deepspeed_env
}

ezpz_get_pbs_env() {
    if [[ "$#" == 1 ]]; then
        hostfile="$1"
        jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
    elif [[ "$#" == 2 ]]; then
        hostfile="$1"
        jobenv_file="$2"
    else
        hostfile="${HOSTFILE:-$(ezpz_get_pbs_nodefile_from_hostname)}"
        jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
    fi
    printf "\n"
    printf "[${BLUE}ezpz_get_pbs_env${RESET}]: Caught ${BLUE}%s${RESET} arguments\n" "$#"
    printf "    • hostfile: ${BLUE}%s${RESET}\n" "${hostfile}"
    printf "    • jobenv_file: ${BLUE}%s${RESET}\n" "${jobenv_file}"
    if [[ $(hostname) == x3* || $(hostname) == x1* || $(hostname) == x4* || $(hostname) == sophia* ]]; then
        if [[ -n $(/bin/cat "${hostfile:-}" | grep "$(hostname)") ]]; then
            num_hosts=$(ezpz_get_num_hosts "${hostfile}")
            num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
            num_gpus="$((num_hosts * num_gpus_per_host))"
            dist_launch=$(ezpz_get_dist_launch_cmd "${hostfile}")
            export DIST_LAUNCH="${dist_launch}"
            export ezlaunch="${DIST_LAUNCH}"
        else
            echo "$(hostname) not found in ${hostfile} ... ?"
        fi
    else
        echo "Skipping ezpz_get_pbs_env() on $(hostname)"
    fi
    # printf "%s" "${FOOTER}"
}

ezpz_get_slurm_env() {
    if [[ -n "${SLURM_JOB_ID}" ]]; then
        export JOBENV_FILE="${SLURM_ENV_FILE}"
        # export jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
        # shellcheck source="${HOME}/.slurmenv"
        [ -f "${JOBENV_FILE}" ] && source "${JOBENV_FILE}"
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
        scheduler_type=$(ezpz_get_scheduler_type)
        if [[ "${scheduler_type}" == pbs ]]; then
            hostfile="${HOSTFILE:-${PBS_NODEFILE:-$(ezpz_get_pbs_nodefile_from_hostname)}}"
            jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
            ezpz_get_pbs_env "$@"
        elif [[ "${scheduler_type}" == "slurm" ]]; then
            hostfile="${HOSTFILE:-$(ezpz_make_slurm_nodefile)}"
            jobenv_file="${SLURM_ENV_FILE}"
            ezpz_get_slurm_env "$@"
        else
            echo "[ezpz_get_job_env] Unknown scheduler ${scheduler_type}"
        fi
    fi
    if [[ -f "${hostfile:-}" ]]; then
        nhosts=$(wc -l <"${hostfile}")
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
    fi
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
    num_gpus="$((num_hosts * num_gpus_per_host))"
    printf "\n"
    printf "  [${MAGENTA}%s${RESET}]:\n" "HOSTS"
    ezpz_print_hosts "${hostfile}"
    printf "\n"
    printf "  [${BRIGHT_BLUE}%s${RESET}]:\n" "DIST INFO"
    printf "      • NGPUS=${BRIGHT_BLUE}%s${RESET}\n" "${num_gpus}"
    printf "      • NHOSTS=${BRIGHT_BLUE}%s${RESET}\n" "${num_hosts}"
    printf "      • NGPU_PER_HOST=${BRIGHT_BLUE}%s${RESET}\n" "${num_gpus_per_host}"
    printf "      • HOSTFILE=${BRIGHT_BLUE}%s${RESET}\n" "${hostfile}"
    printf "      • LAUNCH=${BRIGHT_BLUE}%s${RESET}\n" "${LAUNCH}"
    printf "      • DIST_LAUNCH=${BRIGHT_BLUE}%s${RESET}\n" "${DIST_LAUNCH}"
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
    printf "[%s ${YELLOW}%s${RESET}]\n" "🍋" "ezpz/bin/utils.sh"
    printf "\n"
    printf "    • USER=${BLACK}%s${RESET}\n" "${USER}"
    printf "    • MACHINE=${BLACK}%s${RESET}\n" "${mn}"
    printf "    • HOST=${BLACK}%s${RESET}\n" "${hn}"
    printf "    • TSTAMP=${BLACK}%s${RESET}\n" "$(ezpz_get_tstamp)"
    printf "\n"
    if [[ -n "${PBS_NODEFILE:-}" ]]; then
        ezpz_savejobenv_main "$@"
    elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
        ezpz_savejobenv_main "$@"
    else
        scheduler_type=$(ezpz_get_scheduler_type)
        if [[ "${scheduler_type}" == "pbs" ]]; then
            _pbs_nodefile=$(ezpz_get_pbs_nodefile_from_hostname)
            export PBS_NODEFILE="${_pbs_nodefile}"
            ezpz_getjobenv_main "$@"
        elif [[ "${scheduler_type}" == "slurm" ]]; then
            running_nodes=$(ezpz_get_slurm_running_nodelist)
            if [[ -n "${running_nodes}" ]]; then
                snodelist=$(scontrol show hostname "${running_nodes}")
                _slurm_job_id=$(ezpz_get_slurm_running_jobid)
                export SLURM_JOB_ID="${_slurm_job_id}"
                export SLURM_NODELIST="${running_nodes}"
                ezpz_getjobenv_main "$@"
            fi
        fi
    fi
}

ezpz_setup_job() {
    mn=$(ezpz_get_machine_name)
    hn=$(hostname)
    local mn="${mn}"
    local hn="${hn}"
    printf "\n"
    printf "[%s ${YELLOW}%s${RESET}]\n" "🍋" "ezpz/bin/utils.sh"
    # printf "[${RED}%s${RESET}]\n" "ezpz/bin/utils.sh"
    # printf "\n"
    # printf "[${BLACK}%s${RESET}]\n" "$(ezpz_get_tstamp)"
    printf "    • USER=${YELLOW}%s${RESET}\n" "${USER}"
    printf "    • MACHINE=${YELLOW}%s${RESET}\n" "${mn}"
    printf "    • HOST=${YELLOW}%s${RESET}\n" "${hn}"
    printf "    • TSTAMP=${YELLOW}%s${RESET}\n" "$(ezpz_get_tstamp)"
    printf "\n"
    if [[ -n "${PBS_NODEFILE:-}" ]]; then
        ezpz_savejobenv_main "$@"
    elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
        ezpz_savejobenv_main "$@"
    else
        scheduler_type=$(ezpz_get_scheduler_type)
        if [[ "${scheduler_type}" == "pbs" ]]; then
            _pbs_nodefile=$(ezpz_get_pbs_nodefile_from_hostname)
            if [[ -f "${_pbs_nodefile}" ]]; then
                export PBS_NODEFILE="${_pbs_nodefile}"
                ezpz_getjobenv_main "$@"
            else
                echo "[${mn}] @ [${hn}] No compute node found !!"
            fi
        elif [[ "${scheduler_type}" == "slurm" ]]; then
            running_nodes=$(ezpz_get_slurm_running_nodelist)
            if [[ -n "${running_nodes}" ]]; then
                snodelist=$(scontrol show hostname "${running_nodes}")
                _slurm_job_id=$(ezpz_get_slurm_running_jobid)
                export SLURM_JOB_ID="${_slurm_job_id}"
                export SLURM_NODELIST="${running_nodes}"
                ezpz_getjobenv_main "$@"
            fi
        fi
    fi
}


ezpz_setup_env() {
    ezpz_setup_python && ezpz_setup_job
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
    if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
        WORKING_DIR="${PBS_O_WORKDIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
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

if [[ -n "${DEBUG:-}" ]]; then set +x; fi
