#!/usr/bin/env bash --login
IFS=$'\n\t'

################################################################################
# Configuration and Initialization
################################################################################

# If DEBUG mode is enabled, print a banner and enable trace
if [[ -n "${DEBUG:-}" ]]; then
    printf "\e[1;31m%s\e[0m\n" "!! RUNNING IN DEBUG MODE !!"
    set -x
    set -euo pipefail
fi

# Define color constants for output
RESET="\e[0m" BLACK="\e[1;30m" RED="\e[1;31m" GREEN="\e[1;32m" YELLOW="\e[1;33m"
BLUE="\e[1;34m" MAGENTA="\e[1;35m" CYAN="\e[1;36m"

################################################################################
# Core Utility Functions
################################################################################

# ezpz_get_shell_name: Return the current shell name (strip path).
# Example: /bin/zsh -> zsh
ezpz_get_shell_name() {
    local shell_path="${SHELL:-}"
    printf '%s' "${shell_path##*/}"
}

# ezpz_get_tstamp: Print current timestamp (YYYY-MM-DD-HHMMSS).
ezpz_get_tstamp() {
    date +"%Y-%m-%d-%H%M%S"
}

# ezpz_qsme_running: Print one line per running PBS job owned by $USER.
# Filters out headers and merges fields 13+ into a single string.
ezpz_qsme_running() {
    qstat -u "$USER" -n1rw 2>/dev/null | tr '+|.' ' ' |
        awk '$0 !~ /aurora-pbs|Req|Job|--/ {
                printf "%s", $1
                for (i = 13; i <= NF; i++) printf " %s", $i
                printf "\n"
            }'
}

# ezpz_get_jobid_from_hostname: Return the PBS job ID for the current host.
ezpz_get_jobid_from_hostname() {
    local host_short jobid
    host_short=$(hostname)
    host_short="${host_short%%.*}"
    jobid=$(ezpz_qsme_running | awk -v h="$host_short" '$0 ~ h && /^[0-9]/ {print $1; exit}')
    printf '%s' "$jobid"
}

# ezpz_reset_pbs_vars: Unset all PBS_* environment vars, preserving working dir.
ezpz_reset_pbs_vars() {
    local current_dir
    current_dir="${PBS_O_WORKDIR:-${WORKING_DIR:-$(pwd)}}"
    # Unset all variables starting with PBS
    while IFS='=' read -r name _; do
        if [[ "$name" == PBS* ]]; then
            echo "Unsetting $name"
            unset "$name"
        fi
    done < <(env)
    export PBS_O_WORKDIR="$current_dir"
}

# ezpz_get_pbs_nodefile_from_hostname: Identify PBS_NODEFILE for this host.
# Exports PBS_NODEFILE and PBS_JOBID if found, and echoes the path.
ezpz_get_pbs_nodefile_from_hostname() {
    local jobid hostfile
    jobid=$(ezpz_get_jobid_from_hostname)
    if [[ -n "$jobid" ]]; then
        # Find a nodefile in /var/spool/pbs/aux matching the jobid prefix
        for f in /var/spool/pbs/aux/"$jobid"*; do
            if [[ -f "$f" ]]; then
                hostfile="$f"
                break
            fi
        done
        if [[ -n "$hostfile" ]]; then
            export PBS_NODEFILE="$hostfile"
            export PBS_JOBID="$(basename "$PBS_NODEFILE")"
            printf '%s' "$hostfile"
        fi
    fi
}

# ezpz_save_dotenv: Save environment (excluding LS_COLORS) to .env in outdir.
# Args: output_directory
ezpz_save_dotenv() {
    if [[ "$#" -ne 1 ]]; then
        printf "%s Expected one argument (output dir). Received: %s\n" "[error]" "$#"
        return 1
    fi
    local outdir="$1"
    mkdir -p "$outdir"
    local dotenv_file="${outdir}/.env"
    echo "Saving environment to ${dotenv_file}"
    printenv | grep -v "LS_COLORS" >"$dotenv_file"
    export DOTENV_FILE="$dotenv_file"
}

################################################################################
# Cluster Machine Identification
################################################################################

# ezpz_get_machine_name: Return a short machine name, based on hostname.
ezpz_get_machine_name() {
    local hn machine
    hn=$(hostname)
    case "$hn" in
    x4* | aurora*) machine="aurora" ;;
    x1* | uan*) machine="sunspot" ;;
    sophia*) machine="sophia" ;;
    x3* | polaris*)
        if [[ "${PBS_O_HOST:-}" == sirius* ]]; then
            machine="sirius"
        else
            machine="polaris"
        fi
        ;;
    frontier*) machine="frontier" ;;
    nid*) machine="perlmutter" ;;
    *) machine="${hn%%.*}" ;;
    esac
    printf '%s' "$machine"
}

################################################################################
# Process and Job Controls
################################################################################

# ezpz_check_and_kill_if_running: Kill process listening on port 29500, if any.
ezpz_check_and_kill_if_running() {
    local pid
    pid=$(lsof -ti:29500 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
        echo "Caught $pid"
        kill "$pid"
    else
        echo "Not currently running. Continuing!"
    fi
}

# ezpz_get_slurm_running_jobid: Print running Slurm job ID(s) for current user.
ezpz_get_slurm_running_jobid() {
    if command -v sacct >/dev/null; then
        local jobid
        jobid=$(sacct --format=JobID --noheader --user "$USER" -s R | awk '/^[0-9]/ {print $1; exit}')
        printf '%s' "$jobid"
    fi
}

# ezpz_get_slurm_running_nodelist: Print node list for running Slurm jobs of user.
ezpz_get_slurm_running_nodelist() {
    if command -v sacct >/dev/null; then
        local sn
        sn=$(sacct --format=NodeList --noheader --user "$USER" -s R | awk '/^[0-9]/ {print $1; exit}')
        printf '%s' "$sn"
    fi
}

# ezpz_make_slurm_nodefile: Create a file listing Slurm hostnames.
# Args: [output_file] (default: "nodefile")
ezpz_make_slurm_nodefile() {
    local outfile="${1:-nodefile}"
    local node_list="${SLURM_NODELIST:-$(ezpz_get_slurm_running_nodelist)}"
    if command -v scontrol >/dev/null && [[ -n "$node_list" ]]; then
        scontrol show hostname "$node_list" >"$outfile"
        printf '%s' "$outfile"
    fi
}

# ezpz_setup_srun: Export environment variables for use with srun.
ezpz_setup_srun() {
    if [[ $(hostname) == login* || $(hostname) == nid* ]]; then
        local nhosts="${SLURM_NNODES:-1}"
        local ngpu_per_host="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
        export NHOSTS="$nhosts"
        export NGPU_PER_HOST="$ngpu_per_host"
        local hostfile="${HOSTFILE:-$(ezpz_make_slurm_nodefile)}"
        export HOSTFILE="$hostfile"
        export NGPUS=$((nhosts * ngpu_per_host))
        export SRUN_EXEC="srun -l -u --verbose -N${SLURM_NNODES} -n$((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
    else
        echo "Skipping ezpz_setup_srun() on $(hostname)"
    fi
}

# ezpz_save_ds_env: Save selected env vars (PATH, proxies, etc.) to .deepspeed_env.
ezpz_save_ds_env() {
    echo "Saving {PATH, LD_LIBRARY_PATH, http_proxy, https_proxy, CFLAGS, PYTHONUSERBASE} to .deepspeed_env"
    {
        echo "PATH=${PATH}"
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
        echo "http_proxy=${http_proxy:-}"
        echo "https_proxy=${https_proxy:-}"
        echo "CFLAGS=${CFLAGS:-}"
        echo "PYTHONUSERBASE=${PYTHONUSERBASE:-}"
    } >.deepspeed_env
}

################################################################################
# Conda and Environment Setup (Machine-specific)
################################################################################

# ezpz_setup_conda_frontier: Load modules and activate conda on Frontier.
ezpz_setup_conda_frontier() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module load PrgEnv-gnu/8.5.0
        module load craype-accel-amd-gfx90a
        module load rocm
        micromamba activate /lustre/orion/csc613/scratch/foremans/envs/micromamba/py3.10-torch2.2-rocm5.7
    fi
}

# ezpz_setup_conda_sunspot: Load modules and activate conda on Sunspot.
ezpz_setup_conda_sunspot() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module use /opt/aurora/24.180.1/modulefiles
        module load frameworks/2024.2.1_u1
    fi
}

# ezpz_setup_conda_aurora: Load modules and activate conda on Aurora.
ezpz_setup_conda_aurora() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module load frameworks
        module load mpich/opt/4.3.0rc3
    else
        printf "Caught CONDA_PREFIX=%s from environment, using this!\n" "${CONDA_PREFIX}"
    fi
}

# ezpz_setup_conda_sirius: Activate micromamba on Polaris (Sirius).
ezpz_setup_conda_sirius() {
    if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
        export MAMBA_ROOT_PREFIX=/lus/tegu/projects/PolarisAT/foremans/micromamba
        local shell_name
        shell_name=$(basename "$SHELL")
        eval "$("$MAMBA_ROOT_PREFIX/bin/micromamba" shell hook --shell "$shell_name")"
        micromamba activate 2024-04-23
    else
        echo "Found existing Python at: $(which python3)"
    fi
}

# ezpz_setup_conda_sophia: Load modules and activate conda on Sophia.
ezpz_setup_conda_sophia() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module load conda
        conda activate base
    else
        echo "Caught CONDA_PREFIX=${CONDA_PREFIX}"
    fi
}

# ezpz_setup_conda_polaris: Load modules and activate conda on Polaris.
ezpz_setup_conda_polaris() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module use /soft/modulefiles
        module load conda/2024-04-29
        conda activate base
    else
        echo "Caught CONDA_PREFIX=${CONDA_PREFIX}"
    fi
}

# ezpz_setup_conda: Load/activate conda environment based on detected machine.
ezpz_setup_conda() {
    local machine_name
    machine_name=$(ezpz_get_machine_name)
    case "$machine_name" in
    aurora) ezpz_setup_conda_aurora ;;
    sophia) ezpz_setup_conda_sophia ;;
    sunspot) ezpz_setup_conda_sunspot ;;
    polaris)
        if [[ "${PBS_O_HOST:-}" == sirius* ]]; then
            ezpz_setup_conda_sirius
        else
            ezpz_setup_conda_polaris
        fi
        ;;
    esac
    if [[ $(hostname) == frontier* ]]; then
        ezpz_setup_conda_frontier
    elif [[ $(hostname) == login* || $(hostname) == nid* ]]; then
        echo "Running on Perlmutter!"
        module load pytorch
        source "${SLURM_SUBMIT_DIR}/venvs/perlmutter/pytorch-2.1.0-cu12/bin/activate"
    else
        echo "Unknown hostname $(hostname)"
        exit 1
    fi
}

################################################################################
# Python Virtual Environment Setup
################################################################################

# ezpz_setup_venp_from_conda: Create (if needed) and activate a Python venv on top of conda base.
ezpz_setup_venv_from_conda() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        echo "!! No CONDA_PREFIX var found."
        return 1
    fi
    echo "Found conda at: ${CONDA_PREFIX}"
    local CONDA_NAME
    CONDA_NAME=$(basename "${CONDA_PREFIX}" | sed -E 's/mconda3|\/base//g')
    export CONDA_NAME
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        echo "No VIRTUAL_ENV found. Setting up from conda."
        export VENV_DIR="${WORKING_DIR}/venvs/${CONDA_NAME}"
        echo " - Using VENV_DIR=${VENV_DIR}"
        if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
            printf "\n- Creating a new virtual env using %s in %s\n" "$CONDA_NAME" "$VENV_DIR"
            mkdir -p "$VENV_DIR"
            python3 -m venv "$VENV_DIR" --system-site-packages
            source "${VENV_DIR}/bin/activate"
        else
            echo " - Found existing venv, activating ${VENV_DIR}"
            source "${VENV_DIR}/bin/activate"
        fi
    fi
}

# ezpz_setup_python: Set up Python environment (conda + venv) and report python path.
ezpz_setup_python() {
    local virtual_env="${VIRTUAL_ENV:-}"
    local conda_prefix="${CONDA_PREFIX:-}"
    if [[ -z "$virtual_env" && -z "$conda_prefix" ]]; then
        echo "No conda_prefix OR virtual_env found in environment..."
        echo "Setting up conda..."
        ezpz_setup_conda
    elif [[ -n "$conda_prefix" && -z "$virtual_env" ]]; then
        echo "No virtual environment found."
        echo "Using conda from: $conda_prefix"
        echo "Setting up venv from conda"
        ezpz_setup_venv_from_conda
    elif [[ -n "$virtual_env" && -z "$conda_prefix" ]]; then
        echo "No conda found. Using virtual_env: $virtual_env"
    elif [[ -n "$virtual_env" && -n "$conda_prefix" ]]; then
        echo "Using virtual_env: $virtual_env on top of conda from: $conda_prefix"
    else
        echo "Unable to setup python environment. Exiting"
        exit 1
    fi
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        ezpz_setup_venv_from_conda
    fi
    printf "[python] Using ${MAGENTA}%s${RESET}\n" "$(which python3)"
    export PYTHON_EXEC="$(which python3)"
}

################################################################################
# Hostfile and GPU Utilities
################################################################################

whereAmI() {
    python3 -c 'import os; print(os.getcwd())'
}

join_by() {
    local d="$1" f="$2"
    if shift 2; then
        printf '%s' "$f" "${@/#/$d}"
    fi
}

# ezpz_parse_hostfile: Given a hostfile, output "num_hosts num_gpus_per_host total_gpus".
# Args: hostfile
ezpz_parse_hostfile() {
    if [[ "$#" -ne 1 ]]; then
        echo "Expected exactly one argument: hostfile"
        echo "Received: $#"
        return 1
    fi
    local hf="$1"
    local num_hosts num_gpus_per_host num_gpus
    num_hosts=$(ezpz_get_num_hosts "$hf")
    num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
    num_gpus=$((num_hosts * num_gpus_per_host))
    echo "${num_hosts} ${num_gpus_per_host} ${num_gpus}"
}

# ezpz_get_dist_launch_cmd: Construct an MPI launch command based on hostfile.
# Args: hostfile
ezpz_get_dist_launch_cmd() {
    if [[ "$#" -ne 1 ]]; then
        echo "Expected exactly one argument: hostfile"
        echo "Received: $#"
        return 1
    fi
    local hostfile="$1"
    local machine num_hosts num_gpus_per_host num_gpus num_cores depth dist_cmd
    machine=$(ezpz_get_machine_name)
    num_hosts=$(ezpz_get_num_hosts "$hostfile")
    num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
    num_gpus=$((num_hosts * num_gpus_per_host))
    local num_cores_per_host=$(getconf _NPROCESSORS_ONLN)
    local num_cpus_per_host=$((num_cores_per_host / 2))
    depth=$((num_cpus_per_host / num_gpus_per_host))
    if [[ "$machine" == "sophia" ]]; then
        dist_cmd="mpirun -n ${num_gpus} -N ${num_gpus_per_host} --hostfile ${hostfile} -x PATH -x LD_LIBRARY_PATH"
    else
        dist_cmd="mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hostfile} --cpu-bind depth -d ${depth}"
    fi
    if [[ "$machine" == "aurora" ]]; then
        dist_cmd="${dist_cmd} --no-vni"
    fi
    printf '%s' "$dist_cmd"
}

################################################################################
# Saving PBS and SLURM Job Environments
################################################################################

# ezpz_save_pbs_env: Save PBS job environment variables and set DIST_LAUNCH for PBS.
# Args: [hostfile [jobenv_file]]
ezpz_save_pbs_env() {
    printf "\n[${BLUE}%s${RESET}]\n" "ezpz_save_pbs_env"
    local hostfile jobenv_file dist_launch
    if [[ "$#" -eq 0 ]]; then
        hostfile="${HOSTFILE:-${PBS_NODEFILE}}"
        jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
    elif [[ "$#" -eq 1 ]]; then
        hostfile="$1"
        jobenv_file="${JOBENV_FILE:-${PBS_ENV_FILE}}"
    elif [[ "$#" -eq 2 ]]; then
        hostfile="$1"
        jobenv_file="$2"
    else
        echo "Expected at most 2 arguments. Received: $#"
        return 1
    fi

    if [[ -n "${PBS_JOBID:-}" ]]; then
        env | grep '^PBS' >"${jobenv_file}"
        if [[ "$hostfile" != "${PBS_NODEFILE:-}" ]]; then
            printf "\n"
            printf " • hostfile: ${RED}%s${RESET}\n" "$hostfile"
            printf " • PBS_NODEFILE: ${RED}%s${RESET}\n" "${PBS_NODEFILE}"
            printf "\n"
        fi
        printf " • Using:\n"
        printf "   hostfile: ${BLUE}%s${RESET}\n" "$hostfile"
        printf "   jobenv_file: ${BLUE}%s${RESET}\n" "$jobenv_file"
        # Compute DIST_LAUNCH for PBS
        local num_hosts
        num_hosts=$(ezpz_get_num_hosts "$hostfile")
        local num_gpus_per_host
        num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
        dist_launch=$(ezpz_get_dist_launch_cmd "$hostfile")
        export DIST_LAUNCH="$dist_launch"
        printf " • DIST_LAUNCH: ${BLUE}%s${RESET}\n" "${DIST_LAUNCH}"
    fi
    export HOSTFILE="$hostfile"
    export JOBENV_FILE="$jobenv_file"
    printf " • HOSTFILE: ${BLUE}%s${RESET}\n" "${HOSTFILE}"
    printf " • JOBENV_FILE: ${BLUE}%s${RESET}\n" "${JOBENV_FILE}"
}

# ezpz_save_slurm_env: Save SLURM job environment vars and set DIST_LAUNCH for SLURM.
# Args: [hostfile [jobenv_file]]
ezpz_save_slurm_env() {
    printf "\n[${BLUE}%s${RESET}]\n" "ezpz_save_slurm_env"
    local hostfile jobenv_file dist_launch
    if [[ "$#" -eq 0 ]]; then
        hostfile="${HOSTFILE:-$(ezpz_make_slurm_nodefile)}"
        jobenv_file="${JOBENV_FILE:-${SLURM_ENV_FILE}}"
    elif [[ "$#" -eq 1 ]]; then
        hostfile="$1"
        jobenv_file="${JOBENV_FILE:-${SLURM_ENV_FILE}}"
    elif [[ "$#" -eq 2 ]]; then
        hostfile="$1"
        jobenv_file="$2"
    else
        echo "Expected at most 2 arguments. Received: $#"
        return 1
    fi

    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        env | grep '^SLU' >"${jobenv_file}"
        printf " • Using:\n"
        printf "   hostfile: ${BLUE}%s${RESET}\n" "$hostfile"
        printf "   jobenv_file: ${BLUE}%s${RESET}\n" "$jobenv_file"
        # Compute DIST_LAUNCH for SLURM
        local num_hosts
        num_hosts=$(ezpz_get_num_hosts "$hostfile")
        local num_gpus_per_host
        num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
        dist_launch="srun -l -u --verbose -N${SLURM_NNODES} -n$((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
        export DIST_LAUNCH="$dist_launch"
        printf " • DIST_LAUNCH: ${BLUE}%s${RESET}\n" "${DIST_LAUNCH}"
    fi
    export HOSTFILE="$hostfile"
    export JOBENV_FILE="$jobenv_file"
    printf " • HOSTFILE: ${BLUE}%s${RESET}\n" "${HOSTFILE}"
    printf " • JOBENV_FILE: ${BLUE}%s${RESET}\n" "${JOBENV_FILE}"
}

################################################################################
# Hostfile and GPU Utilities
################################################################################

# ezpz_print_hosts: Print hosts from a hostfile (or current HOSTFILE) with indices.
# Args: [hostfile]
ezpz_print_hosts() {
    local hostfile
    if [[ "$#" -eq 1 ]]; then
        hostfile="$1"
    else
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}}"
    fi
    local counter=0
    while IFS= read -r host; do
        printf " • [host:${MAGENTA}%s${RESET}] - ${MAGENTA}%s${RESET}\n" "$counter" "$host"
        counter=$((counter + 1))
    done <"$hostfile"
}

# ezpz_get_num_gpus_nvidia: Return number of NVIDIA GPUs on this host.
ezpz_get_num_gpus_nvidia() {
    local num_gpus
    if command -v nvidia-smi >/dev/null; then
        num_gpus=$(nvidia-smi -L | wc -l)
    else
        num_gpus=$(python3 -c 'import torch; print(torch.cuda.device_count())')
    fi
    export NGPU_PER_HOST="$num_gpus"
    echo "$num_gpus"
}

# ezpz_get_num_gpus_per_host: Determine GPUs per host by machine type.
ezpz_get_num_gpus_per_host() {
    local mn ngpu_per_host
    mn=$(ezpz_get_machine_name)
    case "$mn" in
    aurora | sunspot) ngpu_per_host=12 ;;
    frontier) ngpu_per_host=8 ;;
    *) ngpu_per_host=$(ezpz_get_num_gpus_nvidia) ;;
    esac
    export NGPU_PER_HOST="$ngpu_per_host"
    echo "$ngpu_per_host"
}

# ezpz_get_num_hosts: Count hosts from a hostfile or use SLURM_NNODES.
# Args: [hostfile]
ezpz_get_num_hosts() {
    local hostfile nhosts
    if [[ "$#" -eq 1 ]]; then
        hostfile="$1"
    else
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}}"
    fi
    if [[ -n "$hostfile" && -f "$hostfile" ]]; then
        nhosts=$(wc -l <"$hostfile")
    elif [[ -n "${SLURM_NNODES:-}" ]]; then
        nhosts="${SLURM_NNODES}"
    else
        nhosts=1
    fi
    export NHOSTS="$nhosts"
    echo "$nhosts"
}

# ezpz_get_num_gpus_total: Total GPUs across all hosts.
ezpz_get_num_gpus_total() {
    local num_hosts num_gpus_per_host num_gpus
    num_hosts=$(ezpz_get_num_hosts "$@")
    num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
    num_gpus=$((num_hosts * num_gpus_per_host))
    echo "$num_gpus"
}

# ezpz_get_jobenv_file: Return the appropriate job environment file variable.
ezpz_get_jobenv_file() {
    local mn
    mn=$(ezpz_get_machine_name)
    if [[ "$mn" == "aurora" || "$mn" == "polaris" || "$mn" == "sunspot" || "$mn" == "sirius" || "$mn" == "sophia" ]]; then
        echo "${JOBENV_FILE:-${PBS_ENV_FILE}}"
    elif [[ "$mn" == "frontier" || "$mn" == "perlmutter" || -n "${SLURM_JOB_ID:-}" ]]; then
        echo "${JOBENV_FILE:-${SLURM_ENV_FILE}}"
    fi
}

# ezpz_get_scheduler_type: Return 'pbs' or 'slurm' based on machine name.
ezpz_get_scheduler_type() {
    local mn
    mn=$(ezpz_get_machine_name)
    case "$mn" in
    aurora | polaris | sunspot | sirius | sophia) echo "pbs" ;;
    frontier | perlmutter) echo "slurm" ;;
    *) echo "slurm" ;;
    esac
}

# ezpz_write_job_info: Print job info summary (placeholder implementation).
# Args: [hostfile [jobenv_file]]
ezpz_write_job_info() {
    if [[ "$#" -eq 0 ]]; then
        local hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}}"
        local jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
    elif [[ "$#" -eq 1 ]]; then
        hostfile="$1"
        jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
    elif [[ "$#" -eq 2 ]]; then
        hostfile="$1"
        jobenv_file="$2"
    else
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}}"
        jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
    fi
    echo "Hostfile: ${hostfile}"
    echo "Jobenv file: ${jobenv_file}"
}

################################################################################
# Helper functions for colored output
################################################################################

printBlack() { printf "\e[1;30m%s\e[0m\n" "$*"; }
printRed() { printf "\e[1;31m%s\e[0m\n" "$*"; }
printGreen() { printf "\e[1;32m%s\e[0m\n" "$*"; }
printYellow() { printf "\e[1;33m%s\e[0m\n" "$*"; }
printBlue() { printf "\e[1;34m%s\e[0m\n" "$*"; }
printMagenta() { printf "\e[1;35m%s\e[0m\n" "$*"; }
printCyan() { printf "\e[1;36m%s\e[0m\n" "$*"; }

################################################################################
# Main Initialization
################################################################################

utils_main() {
    # Set WORKING_DIR based on PBS or SLURM environment
    if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
        WORKING_DIR="${PBS_O_WORKDIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        WORKING_DIR="${SLURM_SUBMIT_DIR}"
    else
        echo "Unable to detect PBS or SLURM working directory info..."
        WORKING_DIR=$(python3 -c 'import os; print(os.getcwd())')
        echo "Using ${WORKING_DIR} as working directory..."
    fi
    export WORKING_DIR
    printf "Using WORKING_DIR: %s\n" "${WORKING_DIR}"
}

utils_main

# Turn off xtrace if DEBUG was set
if [[ -n "${DEBUG:-}" ]]; then
    set +x
    # set +euo pipefail
fi
