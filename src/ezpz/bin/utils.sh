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

get_tstamp () {
    printf "%s" "$(date "+%Y-%m-%d-%H%M%S")"
}

qsme_running() {
    qstat -u $USER -n1rw | sed -e "s/\/0\*208/\ /g" | tr "+|." "\ " | awk '{a = ""; for (i = 13 ; i <= NF ; i++) a = a " " $i; print $1 a}' | egrep -v "aurora-pbs|Req|Job|\-\-"
}


get_jobid_from_hostname() {
    jobid=$(qsme_running | grep "$(hostname)" | awk '{print $1}')
    echo "${jobid}"
}

save_dotenv() {
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
# get_machine_name: Return current machine name, as lowercase string
######################################################################
get_machine_name() {
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

check_and_kill_if_running() {
    # kill $(ps aux | grep -E "$USER.+(mpi|main.py)" | grep -v grep | awk '{print $2}')
    RUNNING_PIDS=$(lsof -i:29500 -Fp | head -n 1 | sed 's/^p//')
    if [[ -n "${RUNNING_PIDS}" ]];
        then echo "Caught ${RUNNING_PIDS}" && kill "${RUNNING_PIDS}";
    else
        echo "Not currently running. Continuing!"
    fi
}


setupSrun() {
    if [[ $(hostname) == login* || $(hostname) == nid* ]]; then
        export NHOSTS="${SLURM_NNODES:-1}"
        export NGPU_PER_HOST="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
        export NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
        export SRUN_EXEC="srun --gpus ${NGPUS} --gpus-per-node ${NGPU_PER_HOST} -N ${NHOSTS} -n ${NGPUS} -l -u --verbose"
    else
        echo "Skipping setupSrun() on $(hostname)"
    fi
}

############################################################################
# save_ds_env
#
# Save important environment variables to .deepspeed_env, which will be
# forwarded to ALL ranks with DeepSpeed 
############################################################################
save_ds_env() {
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
setup_conda_sunspot() {
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
setup_conda_aurora() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module use -a /soft/modulefiles ; module load frameworks/2024.1
    fi
}

########################
# Setup conda on Sirius
########################
setup_conda_sirius() {
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
setup_conda_polaris() {
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

########################
# setup_venv_from_conda
#
# Build (if necessary) a virtual environment
# on top of the active conda and
# activate it.
# ######################
setup_venv_from_conda() {
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
setup_python() {
    virtual_env="${VIRTUAL_ENV:-}"
    conda_prefix="${CONDA_PREFIX:-}"
    if [[ -z "${conda_prefix}" && -z "${virtual_env}" ]]; then
        echo "No conda_prefix OR virtual_env found in environment..."
        echo "Setting up conda..."
        machine_name=$(get_machine_name)
        echo "machine name: ${machine_name}"
        # if [[ $(hostname) == x4* || $(hostname) == aurora* ]]; then
        if [[ "${machine_name}" == "aurora" ]]; then
            setup_conda_aurora
        # elif [[ $(hostname) == x1* || $(hostname) == uan* ]]; then
        elif [[ "${machine_name}" == "sunspot" ]]; then
            setup_conda_sunspot
        # elif [[ $(hostname) == x3*  || $(hostname) == polaris* ]]; then
        elif [[ "${machine_name}" == "polaris" ]]; then
            if [[ "${PBS_O_HOST:-}" == sirius* ]]; then
                setup_conda_sirius
            else
                setup_conda_polaris
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
    elif [[ -n "${conda_prefix}" && -z "${virtual_env}" ]]; then
        echo "No virtual environment found."
        echo "Using conda from: ${conda_prefix}"
        echo "Setting up venv from ${CONDA_PROMPT_MODIFIER:-}"
        setup_venv_from_conda
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
        setup_venv_from_conda
    fi
    pystr="Using: $(which python3)"
    printf "[python] %s" "$(printMagenta "${pystr}")"
    printf "\n"
    export "PYTHON_EXEC=$(which python3)"
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


save_pbs_env() {
    printf "\n[${BLUE}%s${RESET}]\n" "save_pbs_env"
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
        num_hosts=$(get_num_hosts "${hostfile}")
        num_gpus_per_host=$(get_num_gpus_per_host)
        num_gpus=$(( num_hosts * num_gpus_per_host ))
        printf "      to calculate:\n"
        printf "        • num_hosts: ${BLUE}%s${RESET}\n" "${num_hosts}"
        printf "        • num_gpus_per_host: ${BLUE}%s${RESET}\n" "${num_gpus_per_host}"
        printf "        • num_gpus: ${BLUE}%s${RESET}\n" "${num_gpus}"
        # getNumGPUs
        # NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
        export DIST_LAUNCH="mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hostfile} --cpu-bind depth -d 16"
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


save_slurm_env() {
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
# setupHost
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
# if we're on any of the ALCF systems (`x[1-4]*`), we call `save_pbs_env`,
# passing along any received arguments
###########################
setupHost() {
    printf "[${CYAN}%s${RESET}]\n" "setupHost"
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
        save_pbs_env "$@"
    elif [[ $(hostname) == x4* || $(hostname) == x1* ]]; then
        export GPU_TYPE="INTEL"
        # export HOSTFILE="${HOSTFILE:-${PBS_NODEFILE}}"
        export NGPU_PER_TILE=6
        export NTILE_PER_HOST=2
        export NGPU_PER_HOST=$(( NGPU_PER_TILE * NTILE_PER_HOST ))
        save_pbs_env "$@"
    elif [[ $(hostname) == nid* || $(hostname) == login* ]]; then 
        export GPU_TYPE="NVIDIA"
        export HOSTFILE="${HOME}/.slurm-nodefile"
        save_slurm_env
    else
        echo "    Unexpected hostname: $(hostname)"
        export GPU_TYPE="NONE"
        HOSTFILE="hostfile"
        hostname > "${HOSTFILE}"
    fi
}

printHosts() {
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


getNumXPUs() {
    python3 -c 'import intel_extension_for_pytorch as ipex; print(ipex.xpu.device_count())'
}

getNumGPUsNVIDIA() {
    nGPU=$(nvidia-smi -L | wc -l)
    export NGPU_PER_HOST="${nGPU}"
    echo "${nGPU}"
}

get_num_gpus_per_host() {
    if [[ $(hostname) == x4* || $(hostname) == x1* ]]; then
        # export NGPU_PER_HOST=12
        ngpu_per_host=12
    # elif [[ $(hostname) == x3* ]]; then
    #     ngpu_per_host=$(getNumGPUsNVIDIA)
    else
        # echo "Unknown host $(hostname)"
        ngpu_per_host=$(getNumGPUsNVIDIA)
    fi
    export NGPU_PER_HOST="${ngpu_per_host}"
    echo "${ngpu_per_host}"
}

get_num_hosts() {
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

get_num_gpus_total() {
    num_hosts=$(get_num_hosts "$@")
    num_gpus_per_host=$(get_num_gpus_per_host)
    num_gpus=$(( num_hosts * num_gpus_per_host ))
    echo "${num_gpus}"
}


writeJobInfo() {
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
    # printf "[writeJobInfo] Caught jobenv_file: %s\n" "${jobenv_file}"
    # printf "[writeJobInfo] Caught hostfile: %s\n" "${hostfile}"
    num_hosts=$(get_num_hosts "${hostfile}")
    num_gpus_per_host=$(get_num_gpus_per_host)
    # getNumGPUs
    num_gpus="$(( num_hosts * num_gpus_per_host ))"
    HOSTS=$(join_by ', ' $(/bin/cat ${hostfile}))
    export NHOSTS="${num_hosts}"
    export NGPU_PER_HOST="${num_gpus_per_host}"
    export NGPUS="${num_gpus}"
    {
        echo "export HOSTFILE=${hostfile}"
        echo "export NHOSTS=${NHOSTS}"
        echo "export NGPU_PER_HOST=${NGPU_PER_HOST}"
        echo "export NGPUS=${NGPUS}"
        echo "alias LAUNCH='${DIST_LAUNCH}'"
    } >> "${jobenv_file}"
    export LAUNCH="${DIST_LAUNCH}"
    export ezlaunch="${DIST_LAUNCH}"
    alias launch="${LAUNCH}"
    printf "[${MAGENTA}%s${RESET}]\n" "HOSTS"
    # printf "hostfile: ${MAGENTA}%s${RESET}\n" "${hostfile}"
    printHosts
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

saveDeepSpeedEnv() {
    echo "Saving to .deepspeed_env"
    echo "PATH=${PATH}" > .deepspeed_env
    [ "${LD_LIBRARY_PATH}" ] && echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
    [ "${CFLAGS}" ] && echo "CFLAGS=${CFLAGS}" >> .deepspeed_env
    [ "${PYTHONUSERBASE}" ] && echo "PYTHONUSERBASE=${PYTHONUSERBASE}" >> .deepspeed_env
    [ "${http_proxy}" ] && echo "http_proxy=${http_proxy}" >> .deepspeed_env
    [ "${https_proxy}" ] && echo "https_proxy=${https_proxy}" >> .deepspeed_env
}

get_pbs_env() {
    # if [[ "$#" == 0 ]]; then
    #     jobenv_file="${PBS_ENV_FILE}"
    #     hostfile="${HOSTFILE:-$PBS_NODEFILE}"
    if [[ "$#" == 1 ]]; then
        hostfile="$1"
    elif [[ "$#" == 2 ]]; then
        hostfile="$1"
        jobenv_file="$2"
    else
        hostfile="${HOSTFILE:-${PBS_NODEFILE}}"
        jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
    fi
    printf "\n"
    printf "  [${BLUE}get_pbs_env${RESET}]: Caught ${BLUE}%s${RESET} arguments\n" "$#"
    printf "      • hostfile: ${BLUE}%s${RESET}\n" "${hostfile}"
    printf "      • jobenv_file: ${BLUE}%s${RESET}\n" "${jobenv_file}"
    if [[ $(hostname) == x3* || $(hostname) == x1* || $(hostname) == x4* ]]; then
        # export JOBENV_FILE="${PBS_ENV_FILE}"
        if [[ -f "${jobenv_file}" ]]; then
            # envfile="${PBS_ENV_FILE:-}"
            # nodefile=$(/bin/cat "${envfile}" | grep PBS_NODEFILE | sed 's/export\ PBS_NODEFILE=//g')
            # nodefile=$(/bin/cat "${jobenv_file}" | grep "${hostfile}" | sed 's/export\ PBS_NODEFILE=//g')
            nodefile=$(cat "${jobenv_file}" | grep "export HOSTFILE=${hostfile}" | uniq | tr '=' ' ' | awk '{print $NF}')
        else
            nodefile="${nodefile:-${HOSTFILE:-${PBS_NODEFILE}}}"
        fi
        # printf "[${BLUE}get_pbs_env${RESET}] Using nodefile: ${BLUE}%s${RESET}\n" "${nodefile}"
        # printf "      • NGPU_PER_HOST=${MAGENTA}${num_gpus_per_host}${RESET}\n"
        # if [[ -n $(cat "${nodefile:-${PBS_NODEFILE:-HOSTFILE}}" | grep $(hostname)) ]]; then
        if [[ -n $(cat "${nodefile:-}" | grep "$(hostname)") ]]; then
            source "${jobenv_file}" || exit
            # source "${PBS_ENV_FILE}"
            num_hosts=$(get_num_hosts "${nodefile}")
            num_gpus_per_host=$(get_num_gpus_per_host)
            num_gpus="$(( num_hosts * num_gpus_per_host ))"
            dist_launch="mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hostfile} --cpu-bind depth -d 16"
            export DIST_LAUNCH="${dist_launch}"
            export ezlaunch="${DIST_LAUNCH}"
        else
            echo "$(hostname) not found in ${nodefile} ... ?"
        fi
    else
        echo "Skipping get_pbs_env() on $(hostname)"
    fi
    printf "${FOOTER}"
}

get_slurm_env() {
    if [[ $(hostname) == nid* || $(hostname) == login* ]]; then
        export JOBENV_FILE="${SLURM_ENV_FILE}"
        # shellcheck source="${HOME}/.slurmenv"
        source "${SLURM_ENV_FILE}"
        export DIST_LAUNCH="srun --gpus ${NGPUS} --gpus-per-node ${NGPU_PER_HOST} -N ${NHOSTS} -n ${NGPUS} -l -u --verbose"
        export ezlaunch="${DIST_LAUNCH}"
    else
        echo "Skipping get_slurm_env() on $(hostname)"
    fi
}

get_job_env() {
    if [[ "$#" == 1 ]]; then
        hostfile="$1"
    elif [[ "$#" == 2 ]]; then
        hostfile="$1"
        jobenv_file="$2"
    else
        # jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
        hostfile="${HOSTFILE:-${PBS_NODEFILE}}"
        if [[ -z "${hostfile}" ]]; then
            jobid=$(get_jobid_from_hostname)
            if [[ -n "${jobid}" ]]; then
                match=$(/bin/ls /var/spool/pbs/aux/ | grep ${jobid})
                hostfile="/var/spool/pbs/aux/${match}"
                if [[ -f "${hostfile}" ]]; then
                    export PBS_NODEFILE="${hostfile}"
                fi
            fi
        fi
        echo "hostfile: ${hostfile}"
        if [[ $(hostname) == x1* || $(hostname) == x3* || $(hostname) == x4* ]]; then
            jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
            get_pbs_env "$@"
        elif [[ $(hostname) == nid* || $(hostname) == login* ]]; then
            jobenv_file="${SLURM_ENV_FILE}"
            get_slurm_env
        else
            jobenv_file="${UNKNOWN_ENV_FILE}"
            echo "Unexpected hostname ${HOSTNAME}"
        fi
    fi
    if [[ -f "${jobenv_file}" ]]; then
        source "${jobenv_file}" || exit
    else
        echo "Unable to find ${jobenv_file} on $(hostname)"
        exit 1
    fi
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

print_job_env() {
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
    num_hosts=$(get_num_hosts "${hostfile}")
    num_gpus_per_host=$(get_num_gpus_per_host)
    num_gpus="$(( num_hosts * num_gpus_per_host ))"
    printf "\n"
    printf "  [${MAGENTA}%s${RESET}]:\n" "HOSTS"
    printHosts "${hostfile}"
    printf "\n"
    printf "  [${YELLOW}%s${RESET}]:\n" "DIST INFO"
    printf "      • HOSTFILE=${YELLOW}%s${RESET}\n" "${hostfile}"
    printf "      • NHOSTS=${YELLOW}%s${RESET}\n" "${num_hosts}"
    printf "      • NGPU_PER_HOST=${YELLOW}%s${RESET}\n" "${num_gpus_per_host}"
    printf "      • NGPUS=${YELLOW}%s${RESET}\n" "${num_gpus}"
    printf "      • DIST_LAUNCH=${YELLOW}%s${RESET}\n" "${DIST_LAUNCH}"
    printf "\n"
    printf "  [${GREEN}%s${RESET}]:\n" "LAUNCH"
    printf "      • To launch across all available GPUs, use:\n"
    printf "        '${GREEN}launch${RESET}' ( = ${GREEN}%s${RESET} )\n" "${LAUNCH}"
    printf "\n"
}

getjobenv_main() {
    get_job_env "$@"
    print_job_env "$@"
}

savejobenv_main() {
    # printf "${BLACK}%s${RESET}\n" "${LOGO}"
    # printf "${BLACK}[ezpz]${RESET}\n" "${LOGO_DOOM}"
    setupHost "$@"
    writeJobInfo "$@"
}


setup_alcf() {
    mn=$(get_machine_name)
    hn=$(hostname)
    local mn="${mn}"
    local hn="${hn}"
    printf "\n"
    printf "[${RED}%s${RESET}]\n" "ezpz/bin/utils.sh"
    printf "\n"
    # printf "      • %s: ${BLACK}%s${RESET} ${BLACK}@${RESET} %s\n" "${mn}" $(echo $USER) "${hn}"
    # [tstamp] user @ machine (hostname) in $()
    printf "[${BLACK}%s${RESET}]\n" $(get_tstamp)
    printf "    • USER=${BLACK}%s${RESET}\n" "$(echo ${USER})"
    printf "    • MACHINE=${BLACK}%s${RESET}\n" "${mn}"
    printf "    • HOST=${BLACK}%s${RESET}\n" "${hn}"
    #
    # printf "%s ${BLACK}@${RESET} %s (${BLACK}%s${RESET}) in %s" "$(echo $USER)" "${mn}" "${hn}" "${WORKING_DIR}"
    # # printf "      • %s: ${BLACK}%s${RESET} ${BLACK}@${RESET} %s\n" "${mn}" $(echo $USER) "${hn}"
    # printf "      • timestamp: ${BLACK}%s${RESET}\n" "$(get_tstamp)"
    printf "\n"
    if [[ "${hn}" == x1* || "${hn}" == x3* || "${hn}" == x4* ]]; then
        if [[ -n "${PBS_NODEFILE}" ]]; then
            savejobenv_main "$@"
        else
            getjobenv_main "$@"
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
