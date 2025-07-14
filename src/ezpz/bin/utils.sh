#!/bin/bash --login
# @file utils.sh
# @brief `ezpz` helper script with functions to make life ez.
# @description
#     This file provides multiple helper functions, all prefixed with "ezpz_"
#      - `ezpz_setup_job`
#      - `ezpz_setup_python`
#      - ...

# --- Strict Mode & Options ---
# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error when substituting.
# The return value of a pipeline is the status of the last command to exit
# with a non-zero status, or zero if no command exited with a non-zero status.
# set -euo pipefail

# Allow aliases to be expanded (needed for `launch` alias)
# shopt -s expand_aliases

if [[ -n "${NO_COLOR:-}" || -n "${NOCOLOR:-}" || "${COLOR:-}" == 0 || "${TERM}" == "dumb" ]]; then
    # Enable color support for `ls` and `grep`
    # shopt -s dircolors
    # shopt -s colorize
    # shopt -s colorize_grep
    export RESET=''
    export BLACK=''
    export RED=''
    export BRIGHT_RED=''
    export GREEN=''
    export BRIGHT_GREEN=''
    export YELLOW=''
    export BRIGHT_YELLOW=''
    export BLUE=''
    export BRIGHT_BLUE=''
    export MAGENTA=''
    export BRIGHT_MAGENTA=''
    export CYAN=''
    export BRIGHT_CYAN=''
    export WHITE=''
    export BRIGHT_WHITE=''
else
    # --- Color Codes ---
    # Usage: printf "${RED}This is red text${RESET}\n"
    export RESET='\e[0m'
    # BLACK='\e[1;30m' # Avoid black text
    export RED='\e[1;31m'
    export BRIGHT_RED='\e[1;91m'
    export GREEN='\e[1;32m'
    export BRIGHT_GREEN='\e[1;92m'
    export YELLOW='\e[1;33m'
    export BRIGHT_YELLOW='\e[1;93m'
    export BLUE='\e[1;34m'
    export BRIGHT_BLUE='\e[1;94m'
    export MAGENTA='\e[1;35m'
    export BRIGHT_MAGENTA='\e[1;95m'
    export CYAN='\e[1;36m'
    export BRIGHT_CYAN='\e[1;96m'
    export WHITE='\e[1;37m'        # Avoid white on light terminals
    export BRIGHT_WHITE='\e[1;97m' # Added for emphasis
fi

# --- Helper Functions ---

# # Set the default log level to INFO if the
# # environment variable isn't already set.
DEFAULT_LOG_LEVEL="${DEFAULT_LOG_LEVEL:-INFO}"
export DEFAULT_LOG_LEVEL
log_info() {
    args=("$@")
    printf "[${GREEN}I${RESET}][%s] - %s\n" "$(ezpz_get_tstamp)" "${args[*]}"
}

log_warn() {
    args=("$@")
    printf "[%s][${YELLOW}W${RESET}] - %s\n" "$(ezpz_get_tstamp)" "${args[*]}"
}

log_error() {
    args=("$@")
    printf "[%s][${RED}E${RESET}] - %s\n" "$(ezpz_get_tstamp)" "${args[*]}" >&2
}

# @description Log a message to a file and to standared error.
log_message() {
    local level="$1"
    shift
    local string="$*"
    local date
    date=$(ezpz_get_tstamp)
    local log_level="${level:-$DEFAULT_LOG_LEVEL}"
    case "${log_level}" in
    D) log_level="${CYAN}D${RESET}" ;;
    DEBUG) log_level="${CYAN}D${RESET}" ;;
    I) log_level="${GREEN}I${RESET}" ;;
    INFO) log_level="${GREEN}I${RESET}" ;;
    W) log_level="${YELLOW}W${RESET}" ;;
    WARN) log_level="${YELLOW}W${RESET}" ;;
    WARNING) log_level="${YELLOW}W${RESET}" ;;
    E) log_level="${RED}E${RESET}" ;;
    ERROR) log_level="${RED}E${RESET}" ;;
    F) log_level="${RED}F${RESET}" ;;
    FATAL) log_level="${RED}F${RESET}" ;;
    *) log_level="${INFO}I${RESET}" ;; # Default to INFO
    esac
    log_msg="[${date}][$log_level] ${string}"
    echo -e "$log_msg"
}

# --- Global Variables ---
# These are determined dynamically or expected from the environment
# HOSTNAME="$(hostname)"
PBS_ENV_FILE="${HOME}/.pbsenv"
SLURM_ENV_FILE="${HOME}/.slurmenv"
WORKING_DIR="" # Determined in utils_main

# --- Debugging and No-Op ---
# Check if running in DEBUG=1 mode.
# Usage: DEBUG=1 source utils_modern.sh
if [[ -n "${DEBUG:-}" ]]; then
    log_message DEBUG "!! RUNNING IN DEBUG MODE !!"
    set -x # Print command traces before executing command.
fi

# Print (but DO NOT EXECUTE !!) each command that would be run.
# Usage: NOOP=1 source utils_modern.sh
if [[ -v NOOP ]]; then
    log_message WARN "!! RUNNING IN NOOP MODE (Dry Run) !!"
    set -o noexec # Read commands but do not execute them.
fi

# @description Kill existing mpi processes
ezpz_kill_mpi() {
    # find any mpi or python or pals processes and kill them
    ps aux | grep -E "$USER.+(pals|mpi|python)" | grep -v grep | awk '{print $2}' | xargs -r kill
}


# Use the first `n` lines from `PBS_NODEFILE` to create a new hostfile
ezpz_head_n_from_pbs_nodefile() {
    if [[ "$#" -ne 1 ]]; then
        log_message ERROR "Usage: $0 <NHOSTS>"
        return 1
    fi
    local _num_nodes="$1"
    if  [[ -z "${PBS_NODEFILE}" ]]; then
        log_message ERROR "${RED}PBS_NODEFILE is not set. Cannot tail nodefile.${RESET}"
        return 1
    fi
    if [[ ! -f "${PBS_NODEFILE}" ]]; then
        log_message ERROR "${RED}PBS_NODEFILE does not exist: ${PBS_NODEFILE}${RESET}"
        return 1
    fi
    _of="nodefile-0-${_num_nodes}"
    head -n "${_num_nodes}" "${PBS_NODEFILE}" > "${_of}" && wc -l "${_of}"
}

# Function to use the last `n` lins from `PBS_NODEFILE` to create a new hostfile.
ezpz_tail_n_from_pbs_nodefile() {
    if [[ "$#" -ne 1 ]]; then
        log_message ERROR "Usage: $0 <NHOSTS>"
        return 1
    fi
    local _num_nodes="$1"
    if  [[ -z "${PBS_NODEFILE}" ]]; then
        log_message ERROR "${RED}PBS_NODEFILE is not set. Cannot tail nodefile.${RESET}"
        return 1
    fi
    if [[ ! -f "${PBS_NODEFILE}" ]]; then
        log_message ERROR "${RED}PBS_NODEFILE does not exist: ${PBS_NODEFILE}${RESET}"
        return 1
    fi
    _of="nodefile-$((NHOSTS - _num_nodes))-${NHOSTS}"
    tail -n "${_num_nodes}" "${PBS_NODEFILE}" > "${_of}" && wc -l "${_of}"
}

# @description Get name of shell.
# Strip off `/bin/` substr from "${SHELL}" env var and return this string.
#
# @example
#    $ echo "${SHELL}"
#    /bin/zsh
#    $ ezpz_get_shell_name
#    zsh
ezpz_get_shell_name() {
    basename "${SHELL}"
}

# @description Get current timestamp.
# Format: `YYYY-MM-DD-HHMMSS`
#
# @example
#    local timestamp
#    timestamp=$(ezpz_get_tstamp)
#    echo "${timestamp}"
#
# @output
#    The current timestamp string
ezpz_get_tstamp() {
    date "+%Y-%m-%d-%H%M%S"
}

# --- PBS Related Functions ---

# -----------------------------------------------------------------------------
# @brief Prints information about running PBS jobs owned by the current user.
# @description prints 1 line for each running job owned by $USER
#    each line of the form:
#
#    <jobid> <elapsed_time> <node0> <node1> <node2> ...
#
#
#    Parses `qstat` output to show Job ID, elapsed time, and assigned nodes.
#
#    Note: This function relies on the specific output format of `qstat -n1rw`
#          and uses `sed`, `tr`, `awk`, and `grep` for parsing. Changes in
#          `qstat` output might break this function. Consider using `qstat -f -F json`
#          or `qstat -x` (XML) and a proper parser (like `jq` or `xmlstarlet`)
#          if available and more robustness is needed.
#
# @example:
#   ezpz_qsme_running
#
# @output:
#      <jobid0> <elapsed_time0> <node0> <node1> ...
#      <jobid1> <elapsed_time1> <nodeA> <nodeB> ...
#
#    Outputs:
#      Lines describing running jobs, one job per line. Format: <jobid> <nodes...>
#      Returns 1 if qstat command is not found.
# -----------------------------------------------------------------------------
ezpz_qsme_running() {
    # Check if qstat exists
    if ! command -v qstat &>/dev/null; then
        log_message ERROR "'qstat' command not found. Cannot list PBS jobs."
        return 1
    fi
    # -u "${USER}": Filter for the current user.
    # -n1: Show nodes assigned to the job on the first line.
    # -r: Show running jobs.
    # -w: Wide format.
    qstat -u "${USER}" -n1rw |
        sed -e "s/\/0\*208/\ /g" | # Remove CPU/core counts like /8*16
        tr "+|." "\ " |            # Replace '+', '|', '.' with spaces
        awk '{
                a = "";
                # Fields from 13 onwards are node names in this specific format
                for (i = 13; i <= NF; i++) { 
                    a = a " " $i; 
                }
                # Print the first field (Job ID) and the rest of the line
                print $1 a 
            }' |
        grep -vE 'aurora-pbs|Req|Job|-----' # Filter out headers / separators
}

# -----------------------------------------------------------------------------
# @description Identify jobid containing "$(hostname)"
# from all active (running) jobs owned by the $USER.
#
# @example
#    Look for `$(hostname)` in output from `ezpz_qsme_running`, and print the first
#    column
#
#     |   jobid   |   host0  |   host1   |  host2   |
#     |:---------:|:--------:|:---------:|:--------:|
#     |  jobid0   |  host00  |  host10   |  host20  |
#     |  jobid1   |  host01  |  host11   |  host21  |
#     |  jobid2   |  host02  |  host12   |  host22  |
# ezpz_get_jobid_from_hostname() {
#     # jobid=$(ezpz_qsme_running | sed 's/\/.*\ /\ /g' | sed 's/\/.*//g' | grep "$(hostname | sed 's/\..*//g')" | awk '{print $1}')
#     jobid=$(ezpz_qsme_running | grep "^[0-9]" | grep "$(hostname)" | awk '{print $1}')
#     echo "${jobid}"
# }

# -----------------------------------------------------------------------------
# Get the PBS Job ID associated with the current hostname.
# It identifies the job by finding the current hostname in the list of nodes
# assigned to the user's running jobs obtained via `ezpz_qsme_running`.
#
# Relies on:
#   - `ezpz_qsme_running`
#
# Usage:
#   local jobid
#   jobid=$(ezpz_get_jobid_from_hostname)
#   if [[ -n "${jobid}" ]]; then
#       printf "Current host is part of Job ID: %s\n" "${jobid}"
#   fi
#
# Example:
#    Look for `$(hostname)` in output from `ezpz_qsme_running`, and print the first
#    column
#
#     |   jobid   |   host0  |   host1   |  host2   |
#     |:---------:|:--------:|:---------:|:--------:|
#     |  jobid0   |  host00  |  host10   |  host20  |
#     |  jobid1   |  host01  |  host11   |  host21  |
#     |  jobid2   |  host02  |  host12   |  host22  |
#
# Outputs:
#   The PBS Job ID if the current hostname is found in a running job's node list.
#   An empty string otherwise. Returns 1 if `ezpz_qsme_running` fails.
# -----------------------------------------------------------------------------
ezpz_get_jobid_from_hostname() {
    local jobid=""
    local running_jobs_output
    # Capture output or handle error
    if ! running_jobs_output=$(ezpz_qsme_running); then
        return 1 # Propagate error from ezpz_qsme_running
    fi

    # Grep for lines starting with a digit (likely Job IDs)
    # Then grep for the current hostname
    # Awk prints the first field (Job ID)
    # Use grep -m 1 to stop after the first match for efficiency
    jobid=$(echo "${running_jobs_output}" | grep "^[0-9]" | grep -m 1 "$(hostname)" | awk '{print $1}')
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
    for v in ${vars[@]}; do echo "Unsetting $v" && unset -v "${v}"; done
    export PBS_O_WORKDIR="${wd}"
}

ezpz_show_env() {
    log_message INFO "Current environment:"
    log_message INFO "Loaded modules:"
    module list
    vars=(
        "CONDA_PREFIX"
        "CONDA_DEFAULT_ENV"
        "CONDA_NAME"
        "VENV_DIR"
        "VIRTUAL_ENV"
        "PYTHON_EXEC"
        "PBS_JOBID"
        "PBS_NODEFILE"
        "HOSTFILE"
        "NGPUS"
        "NHOSTS"
        "NTILE_PER_HOST"
        "WORLD_SIZE"
    )
    log_message INFO "Showing important environment variables:"
    printenv | grep -E "($(IFS='|'; echo "${vars[*]}"))"
}

ezpz_reset() {
    log_message INFO "Current environment before reset:"
    log_message INFO "$(ezpz_show_env)"
    module reset
    vars=(
        "CONDA_PREFIX"
        "CONDA_DEFAULT_ENV"
        "CONDA_NAME"
        "VENV_DIR"
        "VIRTUAL_ENV"
        "VIRTUAL_ENV_PROMPT"
        "PYTHON_EXEC"
        "CONDA_SHLVL"
        "DIST_LAUNCH"
        "ezlaunch"
        "HOSTFILE"
        "HOSTS"
        "HOSTS_ARR"
        "JOBENV_FILE"
        "LAUNCH"
        "NGPU_PER_HOST"
        "NGPU_PER_TILE"
        "NGPUS"
        "NHOSTS"
        "NTILE_PER_HOST"
        "PYTHONPATH"
        "VENV_DIR"
        "WORLD_SIZE"
    )
    for v in "${vars[@]}"; do
        echo "Unsetting ${v}"
        unset -v "${v}"
    done
    log_message INFO "Resetting PBS-related environment variables..."
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
        # log_info "Saving environment to ${dotenv_file}"
        log_message INFO "Saving environment to ${dotenv_file}"
        printenv | grep -v "LS_COLORS" >"${dotenv_file}"
        export DOTENV_FILE="${dotenv_file}"
    fi
}



# Function to activate (or create, if not found) a conda environment using
# [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
#
# - Usage: `activate_or_create_micromamba_env <envdir> [<python_version>]`
# - Parameters:
#   - `<envdir>`: Directory to look for the conda environment.
#   - `[<python_version>]`: Optional Python version to use for the environment.
#     If not specified, it defaults to the value of `${DEFAULT_PYTHON_VERSION:-3.11}`.
ezpz_activate_or_create_micromamba_env() {
    if ! command -v micromamba &>/dev/null; then
        log_message INFO "micromamba not found. Installing micromamba..."
        ezpz_install_micromamba || {
            log_message INFO "Failed to install micromamba. Please ensure you have curl installed."
            return 1
        }
    fi
    if [[ "$#" -eq 2 ]]; then
        log_message INFO "Received two arguments: envdir=$1, python_version=$2"
        envdir="$(realpath "$1")"
        python_version="$2"
    elif [[ "$#" -eq 1 ]]; then
        log_message INFO "Received one argument: envdir=$1"
        envdir="$(realpath "$1")"
        python_version="${DEFAULT_PYTHON_VERSION:-3.11}"
    else
        log_message INFO "Usage: $0 <envdir> [<python_version>]"
        log_message INFO "If no python version is specified, it defaults to ${DEFAULT_PYTHON_VERSION:-3.11}."
        return 1
    fi

    # Initialize shell for micromamba
    shell_type="$(basename "${SHELL}")"
    eval "$(micromamba shell hook --shell "${shell_type}")"
    # Check if the environment already exists
    if [[ -d "${envdir}" ]] && [[ -n "$(ls -A "${envdir}")" ]]; then
        log_message INFO "Found existing conda environment at ${envdir}. Activating it..."
        micromamba activate "${envdir}" || {
            log_message INFO "Failed to activate existing conda environment at ${envdir}."
            return 1
        }
    else
        log_message INFO "Creating conda environment in: ${envdir}"
        micromamba create --prefix "${envdir}" \
            --yes \
            --verbose \
            --override-channels \
            --channel https://software.repos.intel.com/python/conda/linux-64 \
            --channel conda-forge \
            --strict-channel-priority \
            "python=${python_version}" || {
            log_message INFO "Failed to create conda environment at ${envdir}."
            return 1
        }
        # Activate the newly created environment
        log_message INFO "Activating the conda environment at ${envdir}..."
        micromamba activate "${envdir}" || {
            log_message INFO "Failed to create or activate conda environment at ${envdir}."
            return 1
        }
    fi
}

# Generic function to clone a GitHub repository and build a wheel from it.
# NOTE: THIS HASNT BEEN TESTED YET
# But, something like this should work and could _possibly_ be used as a
# generic replacement instead of having to manually build each library
# one-by-one as we're doing now.
ezpz_build_bdist_wheel_from_github_repo() {
    if [[ "$#" -ne 1 ]]; then
        log_message INFO "Usage: $0 <wheel_name>"
        return 1
    fi
    local repo_url="$1"
    git clone "${repo_url}" && cd "${repo_url##*/}" || return 1
    git submodule sync
    git submodule update --init --recursive
    if [[ -f "requirements.txt" ]]; then
        uv pip install -r requirements.txt
    fi
    if [[ -f "setup.py" ]]; then
        log_message INFO "Building wheel from setup.py..."
        uv pip install --upgrade pip setuptools wheel
        python3 setup.py bdist_wheel
        uv pip install dist/*.whl
    elif [[ -f "pyproject.toml" ]]; then
        log_message INFO "Building wheel from pyproject.toml..."
        python3 -m build --installer=uv
    fi
    uv pip install dist/*.whl || {
        log_message INFO "Failed to install the built wheel."
        return 1
    }
    log_message INFO "Successfully built and installed the wheel from ${repo_url}."
    cd - || return 1
}

# Function to prepare a repository in the specified build directory.
#
# - Usage: prepare_repo_in_build_dir `<build_dir>` `<repo_url>`
#   Where <build_dir> is the directory where the repository will be cloned.
#
# - Example:
#
#   ```bash
#   prepare_repo_in_build_dir build-2025-07-05-203137 "https://github.com/pytorch/pytorch"
#   ```
ezpz_prepare_repo_in_build_dir() {
    # build_dir, repo_url
    if [[ "$#" -ne 2 ]]; then
        log_message INFO "Usage: $0 <build_dir> <repo_url>"
        log_message INFO "Where <build_dir> is the directory where the repository will be cloned."
        return 1
    fi
    local bd
    bd="$(realpath "$1")"
    local src
    src="$2"
    local name
    name="${src##*/}" # Extract the repository name from the URL
    local fp
    fp="${bd}/${name}" # Full path
    if [[ ! -d "${fp}" ]]; then
        log_message INFO "Cloning ${name} from ${src} into ${fp}"
        git clone "${src}" "${fp}" || {
            log_message INFO "Failed to clone ${src}."
            return 1
        }
    else
        log_message INFO "${name} already exists in ${bd}. Skipping clone."
    fi
    cd "${fp}" || {
        log_message INFO "Failed to change directory to ${fp}. Please ensure it exists."
        return 1
    }
    git submodule sync
    git submodule update --init --recursive
    cd - || return 1
}

# Function to check if the wheel file already exists in the build directory.
# - Usage: check_if_already_built `<libdir>`
ezpz_check_if_already_built() {
    # Check if the wheel file already exists in the build directory
    if [[ "$#" -ne 1 ]]; then
        log_message INFO "Usage: $0 libdir"
        return 1
    fi

    local ldir; ldir="$(realpath "$1")"
    log_message INFO "Checking for existing wheels in ${ldir}/dist..."

    if [[ -d "${ldir}/dist" ]] && [[ -n "$(ls -A "${ldir}/dist")" ]]; then
        log_message INFO "Found existing wheels in ${ldir}/dist:"
        ls "${ldir}/dist"/*.whl
        return 0
    else
        return 1
    fi
}

# function to get name of machine, as lowercase string, based on hostname
# Returns one of:
#  aurora
#  sunspot
#  sophia
#  polaris
#  sirius
#  frontier
#  perlmutter
#  <other hostname>
ezpz_get_machine_name() {
    if [[ $(hostname) == x4* || $(hostname) == aurora* ]]; then
        machine="aurora"
    elif [[ $(hostname) == x1* || $(hostname) == uan* ]]; then
        machine="sunspot"
    elif [[ $(hostname) == sophia* ]]; then
        machine="sophia"
    elif [[ $(hostname) == x3* || $(hostname) == polaris* ]]; then
        if [[ "${PBS_O_HOST:-}" == sirius* ]]; then
            machine="sirius"
        else
            machine="polaris"
        fi
    elif [[ $(hostname) == frontier* ]]; then
        machine="frontier"
    elif [[ $(hostname) == nid* ]] || [[ $(hostname) == login* ]]; then
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

ezpz_get_slurm_running_jobid() {
    if [[ -n $(command -v sacct) ]]; then
        jobid=$(sacct --format=JobID,NodeList%-30,state%20 --user "${USER}" -s R | grep -Ev "\.int|\.ext|^JobID|^---" | awk '{print $1}')
        echo "${jobid}"
    fi
}

ezpz_get_slurm_running_nodelist() {
    if [[ -n $(command -v sacct) ]]; then
        slurm_nodelist=$(sacct --format=JobID,NodeList%-30,state%20 --user "${USER}" -s R | grep -Ev "\.int|\.ext|^JobID|^---" | awk '{print $2}')
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

ezpz_set_proxy_alcf() {
    #############################################################################
    # ezpz_set_proxy_alcf
    #
    # Set proxy variables for ALCF
    #
    export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
    export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
    export http_proxy="http://proxy.alcf.anl.gov:3128"
    export https_proxy="http://proxy.alcf.anl.gov:3128"
    export ftp_proxy="http://proxy.alcf.anl.gov:3128"
}

ezpz_save_ds_env() {
    ############################################################################
    # ezpz_save_ds_env
    #
    # Save important environment variables to .deepspeed_env, which will be
    # forwarded to ALL ranks with DeepSpeed
    ############################################################################
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

ezpz_setup_conda_frontier() {
    ###########################
    # Setup conda on Frontier
    ###########################
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

ezpz_setup_conda_sunspot() {
    ###########################
    # Setup conda on Sunspot
    ###########################
    ###### check if CONDA_PREFIX non-empty ################
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module use /opt/aurora/24.180.1/modulefiles
        module load frameworks/2024.2.1_u1
    fi
}

ezpz_setup_conda_aurora() {
    ###########################
    # Setup conda on Aurora
    ###########################
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        # NOTE: Updated 2024-10-08 [@saforem2]
        module load frameworks
        # module load mpich/opt/4.3.0rc3
    else
        printf "Caught CONDA_PREFIX=%s from environment, using this!" "${CONDA_PREFIX}"
    fi
    log_message INFO "Setting FI_MR_CACHE_MONITOR=userfaultfd"
    export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"
}

ezpz_setup_conda_sirius() {
    ########################
    # Setup conda on Sirius
    ########################
    if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV-}" ]]; then
        export MAMBA_ROOT_PREFIX=/lus/tegu/projects/PolarisAT/foremans/micromamba
        shell_name=$(basename "${SHELL}")
        # shell_name=$(echo "${SHELL}" | tr "\/" "" | awk '{print $NF}')
        eval "$("${MAMBA_ROOT_PREFIX}/bin/micromamba" shell hook --shell "${shell_name}")"
        micromamba activate 2024-04-23
    else
        echo "Found existing python at: $(which python3)"
    fi
}

ezpz_setup_conda_sophia() {
    # ########################
    # # Setup conda on Sophia
    # ########################
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module load conda
        conda activate base
    else
        echo "Caught CONDA_PREFIX=${CONDA_PREFIX}"
    fi
}

_ezpz_setup_conda_polaris() {
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

ezpz_setup_conda_polaris() {
    ########################
    # Setup conda on Polaris
    ########################
    # unset MPICH_GPU_SUPPORT_ENABLED
    if [[ "${PBS_O_HOST:-}" == sirius* ]]; then
        ezpz_setup_conda_sirius
    else
        _ezpz_setup_conda_polaris
    fi
}

ezpz_setup_conda_perlmutter() {
    module load pytorch
    # if [[ -f "${SLURM_SUBMIT_DIR}/venvs/perlmutter/pytorch-2.1.0-cu12/bin/activate" ]]; then
    #     log_message INFO "Found existing venv at ${SLURM_SUBMIT_DIR}/venvs/perlmutter/pytorch-2.1.0-cu12"
    # else
    #     log_message INFO "Creating venv at ${SLURM_SUBMIT_DIR}/venvs/perlmutter/pytorch-2.1.0-cu12"
    #     mkdir -p "${SLURM_SUBMIT_DIR}/venvs/perlmutter"
    #     # python3 -m venv "${SLURM_SUBMIT_DIR}/venvs/perlmutter/pytorch-2.1.0-cu12" --system-site-packages
    # fi
    # source "${SLURM_SUBMIT_DIR}/venvs/perlmutter/pytorch-2.1.0-cu12/bin/activate"
}

# ezpz_setup_conda() {
#     local machine_name
#     machine_name=$(ezpz_get_machine_name)
#     log_message INFO "Setting up conda on ${machine_name}"
#     if [[ "${machine_name}" == "aurora" ]]; then
#         ezpz_setup_conda_aurora
#     elif [[ "${machine_name}" == "sophia" ]]; then
#         ezpz_setup_conda_sophia
#     elif [[ "${machine_name}" == "sunspot" ]]; then
#         ezpz_setup_conda_sunspot
#     elif [[ "${machine_name}" == "polaris" ]]; then
#         ezpz_setup_conda_polaris
#     elif [[ $(hostname) == frontier* ]]; then
#         ezpz_setup_conda_frontier
#     elif [[ $(hostname) == login* || $(hostname) == nid* ]]; then
#         echo "Running on Perlmutter !!"
#         module load pytorch
#         source "${SLURM_SUBMIT_DIR}/venvs/perlmutter/pytorch-2.1.0-cu12/bin/activate"
#     else # ------------------------------------- [Unknown] -------------------
#         echo "Unknown hostname $(hostname)"
#         return 1
#     fi
#     # local machine_name
#     # machine_name=$(ezpz_get_machine_name)
#     # case "${machine_name}" in
#     #     aurora) log_message INFO "On Aurora, loaded conda env: ${CYAN}${CONDA_PREFIX:-none}${RESET}" ;;
#     #     sophia) log_message INFO "On Sophia, loaded conda env: ${CYAN}${CONDA_PREFIX:-none}${RESET}" ;;
#     #     sunspot) log_message INFO "On Sunspot, loaded conda env: ${CYAN}${CONDA_PREFIX:-none}${RESET}" ;;
#     #     sirius) log_message INFO "On Sirius, loaded conda env: ${CYAN}${CONDA_PREFIX:-none}${RESET}" ;;
#     #     polaris) log_message INFO "On Polaris, loaded conda env: ${CYAN}${CONDA_PREFIX:-none}${RESET}" ;;
#     #     frontier) log_message INFO "On Frontier, loaded conda env: ${CYAN}${CONDA_PREFIX:-none}${RESET}" ;;
#     #     perlmutter) log_message INFO "On Perlmutter, loaded conda env: ${CYAN}${CONDA_PREFIX:-none}${RESET}" ;;
#     #
#     # esac
#     log_message INFO "List of active modules:"
#     if [[ -n $(command -v module) ]]; then
#         module list
#     else
#         echo "Module command not found. Skipping module list."
#     fi
#     # # ----- [Perlmutter @ NERSC] -------------------------------------
# }

ezpz_setup_conda() {
    if command -v conda &>/dev/null; then
        log_message INFO "Conda version: $(conda --version)"
    else
        log_message WARN "Conda command not found, despite CONDA_PREFIX being set."
        ezpz_install_micromamba
    fi
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        case "$(ezpz_get_machine_name)" in
        aurora) ezpz_setup_conda_aurora ;;
        sunspot) ezpz_setup_conda_sunspot ;;
        polaris) ezpz_setup_conda_polaris ;;
        sirius) ezpz_setup_conda_sirius ;;
        sophia) ezpz_setup_conda_sophia ;;
        frontier) ezpz_setup_conda_frontier ;;
        perlmutter) log_message INFO "Skipping conda setup on Perlmutter" ;;
        *) log_message ERROR "Unknown machine for conda setup: $(hostname)" && return 1 ;;
        esac
    else
        log_message INFO "Skipping conda setup, CONDA_PREFIX already set to ${CYAN}${CONDA_PREFIX}${RESET}"
    fi
    log_message INFO "List of active modules:"
    if [[ -n $(command -v module) ]]; then
        module list
    else
        echo "Module command not found. Skipping module list."
    fi
}

# ezpz_install_uv
#
# Install `uv` package.
# See: https://docs.astral.sh/uv/#installation
ezpz_install_uv() {
    curl -LsSf https://astral.sh/uv/install.sh | sh
}

ezpz_install_micromamba() {
    if ! command -v micromamba &>/dev/null; then
        if ! command -v curl &>/dev/null; then
            log_message ERROR "curl command not found. Cannot install micromamba."
            return 1
        fi
        log_message INFO "micromamba not found in PATH."
        log_message INFO "Installing micromamba to ${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}..."
        "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
        log_message INFO "Installed micromamba. Adding to PATH..."
        shell_type="$(basename "${SHELL}")"
        eval "$(micromamba shell hook --shell "${shell_type}")"
    else
        log_message INFO "micromamba already installed at: $(command -v micromamba). Skipping installation."
        return 0
    fi
}

# Function to set up a Python virtual environment using `uv`.
# - Relies on:
#   - `uv` command must be available.
#   - `CONDA_PREFIX` environment variable must be set.
#   - `WORKING_DIR` environment variable must be set.
#   - `python3` command must exist and point to the Conda Python.
# - Usage:
#   `ezpz_setup_uv_venv`
# - Side Effects:
#   Creates a virtual environment under "${WORKING_DIR}/venvs/".
#   Activates the created virtual environment. Prints status messages.
#   Returns 1 on failure. Exports CONDA_NAME, VENV_DIR.
ezpz_setup_uv_venv() {
    if [[ -n "$(command -v uv)" ]]; then
        echo "uv already installed. Skipping..."
    else
        echo "Installing uv..."
        ezpz_install_uv
    fi
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        log_message ERROR "  - CONDA_PREFIX is not set. Cannot create venv."
        return 1
    else
        log_message INFO "  - Found conda at ${CYAN}${CONDA_PREFIX}${RESET}"
        CONDA_NAME=$(basename "${CONDA_PREFIX}") && export CONDA_NAME
        if [[ -z "${WORKING_DIR:-}" ]]; then
            log_message ERROR "  - WORKING_DIR is not set. Cannot create venv."
            return 1
        else
            log_message INFO "  - Found WORKING_DIR=${CYAN}${WORKING_DIR}${RESET}"
        fi

    fi
    local mn
    local env_name
    env_name=$(basename "${CONDA_PREFIX}")
    VENV_DIR="${WORKING_DIR}/venvs/$(ezpz_get_machine_name)/${env_name}"
    fpactivate="${VENV_DIR}/bin/activate"
    mn=$(ezpz_get_machine_name)
    [ ! -f "${fpactivate}" ] && log_message INFO "  - Creating venv in ${VENV_DIR} on ${mn}..." && uv venv --python="$(which python3)" --system-site-packages "${VENV_DIR}"
    # shellcheck disable=SC1090
    [ -f "${fpactivate}" ] && log_message INFO "  - Activating: ${fpactivate}" && source "${fpactivate}"
}

# -----------------------------------------------------------------------------
# @description Set up a standard Python `venv` on top of an active Conda environment.
# Creates a venv named after the Conda environment in a central 'venvs' directory.
# Activates the created venv. Inherits system site packages.
#
# Note: Similar purpose to `ezpz_setup_uv_venv` but uses the built-in `venv` module.
#
# Relies on:
#   - `CONDA_PREFIX` environment variable must be set.
#   - `WORKING_DIR` environment variable must be set.
#   - `python3` command must exist and point to the Conda Python.
#
# Usage:
#   ezpz_setup_venv_from_conda
#
# Side Effects:
#   Creates a virtual environment under "${WORKING_DIR}/venvs/".
#   Activates the created virtual environment. Prints status messages.
#   Returns 1 on failure. Exports CONDA_NAME, VENV_DIR.
# -----------------------------------------------------------------------------
# ezpz_setup_venv_from_conda() {
#     if [[ -z "${CONDA_PREFIX:-}" ]]; then
#         log_message ERROR "  - ${RED}CONDA_PREFIX${RESET} is not set. Cannot create venv. Returning 1"
#         return 1
#     else
#         log_message INFO "  - Found conda at ${CYAN}${CONDA_PREFIX}${RESET}"
#         CONDA_NAME=$(basename "${CONDA_PREFIX}") && export CONDA_NAME
#         local mn
#         mn="$(ezpz_get_machine_name)"
#         if [[ -z "${VIRTUAL_ENV:-}" ]]; then
#             log_message INFO "  - No VIRTUAL_ENV found in environment!"
#             VENV_DIR="${WORKING_DIR}/venvs/$(ezpz_get_machine_name)/${CONDA_NAME}"
#             export VENV_DIR
#             log_message INFO "  - Looking for venv in VENV_DIR=${CYAN}${VENV_DIR}${RESET}..."
#             local fpactivate
#             fpactivate="${VENV_DIR}/bin/activate"
#             # make directory if it doesn't exist
#             [[ ! -d "${VENV_DIR}" ]] && mkdir -p "${VENV_DIR}"
#             if [[ ! -f "${fpactivate}" ]]; then
#                 log_message INFO "  - Creating venv (on top of ${GREEN}${CONDA_NAME}${RESET}) in VENV_DIR..."
#                 python3 -m venv "${VENV_DIR}" --system-site-packages
#                 source "${fpactivate}" || {
#                     log_message ERROR "  - Failed to source ${fpactivate} after creation."
#                     return 1
#                 }
#                 # --seed ensures that pip is installed into the venv
#                 # uv venv --seed --python="$(which python3)" --system-site-packages "${VENV_DIR}" || {
#                 #     log_message WARN "  - uv venv failed, falling back to python3 -m venv"
#                 #     python3 -m venv "${VENV_DIR}" --system-site-packages
#                 # }
#                 # if [[ -f "${fpactivate}" ]]; then
#                 #     log_message INFO "  - Activating newly created venv..."
#                 #     # shellcheck disable=SC1090
#                 #     [ -f "${fpactivate}" ] && {
#                 #         log_message INFO "  - Found ${fpactivate}" && {
#                 #             source "${fpactivate}" && {
#                 #                 return 0
#                 #             }
#                 #         }
#                 #     }
#                 # else
#                 #     log_message ERROR "  - Failed to create venv at ${RED}${VENV_DIR}${RESET}"
#                 #     return 1
#                 # fi
#             elif [[ -f "${fpactivate}" ]]; then
#                 log_message INFO "  - Activating existing venv in VENV_DIR=${CYAN}${VENV_DIR}${RESET}"
#                 # shellcheck disable=SC1090
#                 [ -f "${fpactivate}" ] && {
#                     log_message INFO "  - Found ${fpactivate}" && {
#                         source "${fpactivate}" && {
#                             return 0
#                         }
#                     }
#                 }
#             else
#                 log_message ERROR "  - Unable to locate ${RED}${fpactivate}${RESET}"
#                 return 1
#             fi
#         fi
#     fi
#
# }

# ezpz_setup_venv_from_conda1() {
#     # Check prerequisites
#     if [[ -z "${CONDA_PREFIX:-}" ]]; then
#         printf "${RED}Error: CONDA_PREFIX is not set. Cannot create venv.${RESET}\n" >&2
#         return 1
#     fi
#     # if [[ -z "${WORKING_DIR:-}" ]]; then
#     #     printf "${RED}Error: WORKING_DIR is not set. Cannot determine where to create venvs.${RESET}\n" >&2
#     #     return 1
#     # fi
#     if ! command -v python3 &> /dev/null; then
#          printf "${RED}Error: python3 command not found in PATH.${RESET}\n" >&2
#          return 1
#     fi
#
#     log_info $(echo "Found conda at: %s\n" "${CONDA_PREFIX}")
#     local conda_name
#     conda_name=$(basename "${CONDA_PREFIX}") # Get conda env name
#     export CONDA_NAME="${conda_name}" # Export for potential use elsewhere
#
#     # Check if already inside a venv
#     if [[ -n "${VIRTUAL_ENV:-}" ]]; then
#         printf "Already inside a virtual environment: %s\n" "${VIRTUAL_ENV}"
#         # Ensure VENV_DIR is set if we are already in one
#         export VENV_DIR="${VIRTUAL_ENV}"
#         return 0
#     fi
#
#     export VENV_DIR="$(ezpz_get_working_dir)/venvs/${CONDA_NAME}"
#     log_info "No VIRTUAL_ENV found in environment!"
#     log_info $(echo "    - Setting up venv from Conda env '%s'\n" "${CONDA_NAME}")
#     log_Info $(echo "    - Using VENV_DIR=%s\n" "${VENV_DIR}")
#
#     local activate_script="${VENV_DIR}/bin/activate"
#
#     # Check if venv needs creation
#     if [[ ! -f "${activate_script}" ]]; then
#         log_info "    - Creating new virtual env in" "$(printf "${GREEN}%s${RESET}" "${VENV_DIR}")"
#
#         if ! mkdir -p "${VENV_DIR}"; then
#              printf "${RED}Error: Failed to create directory '%s'.${RESET}\n" "${VENV_DIR}" >&2
#              return 1
#         fi
#         # Create venv using the current python3, inheriting system site packages
#         if ! python3 -m venv "${VENV_DIR}" --system-site-packages; then
#              printf "${RED}Error: Failed to create venv at '%s'.${RESET}\n" "${VENV_DIR}" >&2
#              rm -rf "${VENV_DIR}" # Clean up partial venv
#              return 1
#         fi
#          printf "    - Activating newly created venv...\n"
#          # Source the activate script
#          source "${activate_script}" || {
#               printf "${RED}Error: Failed to source activate script '%s' after creation.${RESET}\n" "${activate_script}" >&2
#               return 1
#          }
#          printf "${GREEN}Successfully created and activated venv.${RESET}\n"
#          return 0
#     else # Venv already exists
#          printf "    - Found existing venv, activating from %s\n" "$(printf "${BLUE}%s${RESET}" "${VENV_DIR}")"
#          source "${activate_script}" || {
#               printf "${RED}Error: Failed to activate existing venv '%s'.${RESET}\n" "${VENV_DIR}" >&2
#               return 1
#          }
#          printf "${GREEN}Successfully activated existing venv.${RESET}\n"
#          return 0
#     fi
# }

# -----------------------------------------------------------------------------
# @description Set up a standard Python `venv` on top of an active Conda environment.
# Creates a venv named after the Conda environment in a central 'venvs' directory.
# Activates the created venv. Inherits system site packages.
#
# Note: Similar purpose to `ezpz_setup_uv_venv` but uses the built-in `venv` module.
#
# Relies on:
#   - `CONDA_PREFIX` environment variable must be set.
#   - `WORKING_DIR` environment variable must be set.
#   - `python3` command must exist and point to the Conda Python.
#
# Usage:
#   ezpz_setup_venv_from_conda
#
# Side Effects:
#   Creates a virtual environment under "${WORKING_DIR}/venvs/".
#   Activates the created virtual environment. Prints status messages.
#   Returns 1 on failure. Exports CONDA_NAME, VENV_DIR.
# -----------------------------------------------------------------------------
ezpz_setup_venv_from_conda() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        log_message ERROR "  - CONDA_PREFIX is not set. Cannot create venv."
        return 1
    else
        log_message INFO "  - Found conda at ${CYAN}${CONDA_PREFIX}${RESET}"
        CONDA_NAME=$(basename "${CONDA_PREFIX}") && export CONDA_NAME
        if [[ -z "${VIRTUAL_ENV:-}" ]]; then
            log_message INFO "  - No VIRTUAL_ENV found in environment!"
            # log_message INFO "Trying to setup venv from ${GREEN}${CYAN}${RESET}..."
            VENV_DIR="${WORKING_DIR}/venvs/$(ezpz_get_machine_name)/${CONDA_NAME}"
            log_message INFO "  - Looking for venv in venvs/$(ezpz_get_machine_name)/${CYAN}${CONDA_NAME}${RESET}..."
            local fpactivate
            fpactivate="${VENV_DIR}/bin/activate"
            export VENV_DIR
            # make directory if it doesn't exist
            [[ ! -d "${VENV_DIR}" ]] && mkdir -p "${VENV_DIR}"
            if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
                log_message INFO "  - Creating venv (on top of ${GREEN}${CONDA_NAME}${RESET}) in VENV_DIR..."
                python3 -m venv "${VENV_DIR}" --system-site-packages
                source "${VENV_DIR}/bin/activate" || {
                    log_message ERROR "  - Failed to source ${fpactivate} after creation."
                    return 1
                }
                # if [[ -f "${VENV_DIR}/bin/activate" ]]; then
                #     log_message INFO "  - Activating newly created venv..."
                #     # shellcheck disable=SC1090
                #     [ -f "${fpactivate}" ] && log_message INFO "  - Found ${fpactivate}" && source "${fpactivate}" && return 0
                # else
                #     log_message ERROR "  - Failed to create venv at ${VENV_DIR}"
                #     return 1
                # fi
            elif [[ -f "${VENV_DIR}/bin/activate" ]]; then
                log_message INFO "  - Activating existing venv in VENV_DIR=venvs/${CYAN}${CONDA_NAME}${RESET}"
                # shellcheck disable=SC1090
                [ -f "${fpactivate}" ] && log_message INFO "  - Found ${fpactivate}" && source "${fpactivate}" && return 0
            else
                log_message ERROR "  - Unable to locate ${VENV_DIR}/bin/activate"
                return 1
            fi
        fi
    fi

}

# -----------------------------------------------------------------------------
# @brief Main Python environment setup function.
#
# @example:
#   ezpz_setup_python
#
# @description
#    Relies on:
#      - `ezpz_setup_conda`
#      - ~~`ezpz_setup_venv_from_conda` (or `ezpz_setup_uv_venv`)~~
#      - `ezpz_setup_uv_venv`
#      - `CONDA_PREFIX`, `VIRTUAL_ENV` environment variables.
#      - `which python3`
#
#    Side Effects:
#      Activates Conda and/or venv environments. Exports PYTHON_EXEC.
#      Prints status messages. Returns 1 on failure.
#
#    1. Setup `conda`
#       - if `conda` nonempty, and `venv` empty, use `conda` to setup `venv`.
#       - if `venv` nonempty, and `conda` empty, what do (???)
#       - if `venv` nonempty and `conda` nonempty, use these
#       - if `conda` empty and `venv` empty:
#          - if `hostname == x4*`, we're on Aurora
#          - if `hostname == x1*`, we're on Sunspot
#          - if `hostname == x3*`, we're on Polaris
#          - if `hostname == nid*`, we're on Perlmutter
#          - otherwise, you're on you're own
#
#    2. Activate (creating, if necessary) a `venv` on top of `base` conda
#       - use the $CONDA_PREFIX to create a venv in
#         `Megatron-DeepSpeed/venvs/${CONDA_PREFIX}`
#         - activate and use this
#
#    3. Print info about which python we're using
# -----------------------------------------------------------------------------
ezpz_setup_python() {
    # virtual_env="${VIRTUAL_ENV:-}"
    # conda_prefix="${CONDA_PREFIX:-}"


    local virtual_env="${VIRTUAL_ENV:-}"
    local conda_prefix="${CONDA_PREFIX:-}"
    # local setup_status=0


    # log_message INFO "${CYAN}[ezpz_setup_python]${RESET} Checking Python environment..."
    log_message INFO "[${CYAN}PYTHON${RESET}]"


    # Scenario 1: Neither Conda nor venv active -> Setup Conda then venv
    if [[ -z "${conda_prefix}" && -z "${virtual_env}" ]]; then
        log_message INFO "  - No conda_prefix OR virtual_env found in environment. Setting up conda..."
        if ! ezpz_setup_conda; then
            log_message ERROR "  - ezpz_setup_conda failed."
            return 1
        fi
        # Re-check conda_prefix after setup attempt
        conda_prefix="${CONDA_PREFIX:-}"
        if [[ -z "${conda_prefix}" ]]; then
            log_message ERROR "  - CONDA_PREFIX still not set after ezpz_setup_conda."
            return 1
        fi
        # Now attempt to set up venv on top of the activated conda


        # log_message INFO "  - Setting up venv from conda=${CYAN}${conda_prefix}${RESET}..."
        if ! ezpz_setup_venv_from_conda; then
            log_message ERROR "  - ezpz_setup_venv_from_conda failed."
            return 1
        fi


    # Scenario 2: Conda active, venv not active -> Setup venv
    elif [[ -n "${conda_prefix}" && -z "${virtual_env}" ]]; then
        ezpz_setup_venv_from_conda
        # ezpz_setup_uv_venv


    # Scenario 2: Conda active, venv not active -> Setup venv
    elif [[ -n "${conda_prefix}" && -z "${virtual_env}" ]]; then
        log_message INFO "  - Conda active, conda=${GREEN}${conda_prefix}${RESET}..."
        # log_message INFO "Setting up venv from"
        if ! ezpz_setup_venv_from_conda; then
            log_message ERROR "  - ezpz_setup_venv_from_conda failed."
            return 1
        fi


    # Scenario 3: Venv active, Conda not active (less common/intended)
    elif [[ -n "${virtual_env}" && -z "${conda_prefix}" ]]; then
        log_message INFO "  - No conda_prefix found."
        log_message INFO "  - Using virtual_env from: ${CYAN}${virtual_env}${RESET}"


    # Scenario 4: Both Conda and venv active
    elif [[ -n "${virtual_env}" && -n "${conda_prefix}" ]]; then
        log_message INFO "  - Found both conda_prefix and virtual_env in environment."
        log_message INFO "  - Using conda from: ${GREEN}${conda_prefix}${RESET}"
        log_message INFO "  - Using venv from: ${CYAN}${virtual_env}${RESET}"
    fi


    # Verify python3 is available and export path
    local python_exec
    if ! python_exec=$(which python3); then
        log_message ERROR "  - python3 command not found in PATH."
        return 1
    fi
    log_message INFO "  - Using python from: ${CYAN}$(which python3)${RESET}"
    export PYTHON_EXEC="${python_exec}"
}

# # -----------------------------------------------------------------------------
# # @brief Main Python environment setup function.
# #
# # @example:
# #   ezpz_setup_python
# #
# # @description
# #    Relies on:
# #      - `ezpz_setup_conda`
# #      - ~~`ezpz_setup_venv_from_conda` (or `ezpz_setup_venv_from_conda`)~~
# #      - `ezpz_setup_venv_from_conda`
# #      - `CONDA_PREFIX`, `VIRTUAL_ENV` environment variables.
# #      - `which python3`
# #
# #    Side Effects:
# #      Activates Conda and/or venv environments. Exports PYTHON_EXEC.
# #      Prints status messages. Returns 1 on failure.
# #
# #    1. Setup `conda`
# #       - if `conda` nonempty, and `venv` empty, use `conda` to setup `venv`.
# #       - if `venv` nonempty, and `conda` empty, what do (???)
# #       - if `venv` nonempty and `conda` nonempty, use these
# #       - if `conda` empty and `venv` empty:
# #          - if `hostname == x4*`, we're on Aurora
# #          - if `hostname == x1*`, we're on Sunspot
# #          - if `hostname == x3*`, we're on Polaris
# #          - if `hostname == nid*`, we're on Perlmutter
# #          - otherwise, you're on you're own
# #
# #    2. Activate (creating, if necessary) a `venv` on top of `base` conda
# #       - use the $CONDA_PREFIX to create a venv in
# #         `Megatron-DeepSpeed/venvs/${CONDA_PREFIX}`
# #         - activate and use this
# #
# #    3. Print info about which python we're using
# # -----------------------------------------------------------------------------
# ezpz_setup_python() {
#     local virtual_env="${VIRTUAL_ENV:-}"
#     local conda_prefix="${CONDA_PREFIX:-}"
#     log_message INFO "[${CYAN}PYTHON${RESET}]"
#     # Scenario 1: Neither Conda nor venv active -> Setup Conda then venv
#     if [[ -z "${conda_prefix}" && -z "${virtual_env}" ]]; then
#         log_message INFO "  - No conda_prefix OR virtual_env found in environment. Setting up conda..."
#         if ! ezpz_setup_conda; then
#             log_message ERROR "  - ezpz_setup_conda failed."
#             return 1
#         fi
#         # Re-check conda_prefix after setup attempt
#         conda_prefix="${CONDA_PREFIX:-}"
#         if [[ -z "${conda_prefix}" ]]; then
#             log_message ERROR "  - CONDA_PREFIX still not set after ezpz_setup_conda."
#             return 1
#         fi
#         # Now attempt to set up venv on top of the activated conda
#
#         # log_message INFO "  - Setting up venv from conda=${CYAN}${conda_prefix}${RESET}..."
#         if ! ezpz_setup_venv_from_conda; then
#             log_message ERROR "  - ezpz_setup_venv_from_conda failed."
#             return 1
#         fi
#
#     # Scenario 2: Conda active, venv not active -> Setup venv
#     elif [[ -n "${conda_prefix}" && -z "${virtual_env}" ]]; then
#         ezpz_setup_venv_from_conda
#
#     # Scenario 2: Conda active, venv not active -> Setup venv
#     elif [[ -n "${conda_prefix}" && -z "${virtual_env}" ]]; then
#         log_message INFO "  - Conda active, conda=${GREEN}${conda_prefix}${RESET}..."
#         # log_message INFO "Setting up venv from"
#         if ! ezpz_setup_venv_from_conda; then
#             log_message ERROR "  - ezpz_setup_venv_from_conda failed."
#             return 1
#         fi
#
#     # Scenario 3: Venv active, Conda not active (less common/intended)
#     elif [[ -n "${virtual_env}" && -z "${conda_prefix}" ]]; then
#         log_message INFO "  - No conda_prefix found."
#         log_message INFO "  - Using virtual_env from: ${CYAN}${virtual_env}${RESET}"
#
#     # Scenario 4: Both Conda and venv active
#     elif [[ -n "${virtual_env}" && -n "${conda_prefix}" ]]; then
#         log_message INFO "  - Found both conda_prefix and virtual_env in environment."
#         log_message INFO "  - Using conda from: ${GREEN}${conda_prefix}${RESET}"
#         log_message INFO "  - Using venv from: ${CYAN}${virtual_env}${RESET}"
#     fi
#
#     # Verify python3 is available and export path
#     local python_exec
#     if ! python_exec=$(which python3); then
#         log_message ERROR "  - python3 command not found in PATH."
#         return 1
#     fi
#     log_message INFO "  - Using python from: ${CYAN}$(which python3)${RESET}"
#     export PYTHON_EXEC="${python_exec}"
# }

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

    scheduler_type=$(ezpz_get_scheduler_type)
    if [[ "${scheduler_type}" == "pbs" ]]; then
        # dist_launch_cmd="mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hostfile} --cpu-bind depth -d ${depth}"
        if [[ "${mn}" == "sophia" ]]; then
            dist_launch_cmd="mpirun -n ${num_gpus} -N ${num_gpus_per_host} --hostfile ${hostfile} -x PATH -x LD_LIBRARY_PATH"
        else
            dist_launch_cmd="mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hostfile} --cpu-bind depth -d ${depth}"
        fi
        if [[ "${mn}" == "aurora" ]]; then
            dist_launch_cmd="${dist_launch_cmd} --no-vni"
        fi
        # dist_launch_cmd=$(ezpz_get_dist_launch_cmd "${hostfile}")
    elif [[ "${scheduler_type}" == "slurm" ]]; then
        # dist_launch_cmd="srun -N ${num_hosts} -n ${num_gpus} -l -u --verbose"
        dist_launch_cmd="srun -l -u --verbose -N${SLURM_NNODES} -n$((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
    else
        printf "\n[!! %s]: Unable to determine scheduler type. Exiting.\n" "$(printRed "ERROR")"
        exit 1
    fi
    echo "${dist_launch_cmd}"

}

ezpz_save_pbs_env() {
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
    if [[ -n $(printenv | grep PBS_JOBID) ]]; then
        PBS_VARS=$(env | grep PBS)
        if [[ "$#" == 1 || "$#" == 2 ]]; then
            printf "\n[${BLUE}%s${RESET}]\n" "ezpz_save_pbs_env"
            printf "• Caught ${BLUE}%s${RESET} arguments\n" "$#"
            printf "• Using:\n"
            printf "  - hostfile: ${BLUE}%s${RESET}\n" "${hostfile}"
            printf "  - jobenv_file: ${BLUE}%s${RESET}\n" "${jobenv_file}"
            printf "• Setting:\n"
            printf "  - HOSTFILE: ${BLUE}%s${RESET}\n" "${HOSTFILE}"
            printf "  - JOBENV_FILE: ${BLUE}%s${RESET}\n\n" "${JOBENV_FILE}"
        fi
        echo "${PBS_VARS[*]}" >"${jobenv_file}"
        if [[ "${hostfile}" != "${PBS_NODEFILE:-}" ]]; then
            printf "\n"
            printf "  - Caught ${RED}%s${RESET} != ${RED}%s${RESET} \n" "hostfile" "PBS_NODEFILE"
            printf "      - hostfile: ${RED}%s${RESET}\n" "${hostfile}"
            printf "      - PBS_NODEFILE: ${RED}%s${RESET}\n" "${PBS_NODEFILE}"
            printf "\n"
        fi
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

        printf "    to calculate:\n"
        printf "      - num_hosts: ${BLUE}%s${RESET}\n" "${num_hosts}"
        printf "      - num_cores_per_host: ${BLUE}%s${RESET}\n" "${num_cores_per_host}"
        printf "      - num_cpus_per_host: ${BLUE}%s${RESET}\n" "${num_cpus_per_host}"
        printf "      - num_gpus_per_host: ${BLUE}%s${RESET}\n" "${num_gpus_per_host}"
        printf "      - depth: ${BLUE}%s${RESET}\n" "${depth}"
        printf "      - num_gpus: ${BLUE}%s${RESET}\n" "${num_gpus}"
        export DIST_LAUNCH="${dist_launch_cmd}"
        export ezlaunch="${DIST_LAUNCH}"
        # printf "      - DIST_LAUNCH: ${BLUE}%s${RESET}\n" "${DIST_LAUNCH}"
    fi
    export HOSTFILE="${hostfile}"
    export JOBENV_FILE="${jobenv_file}"
}

ezpz_save_slurm_env() {
    printf "\n[${BLUE}%s${RESET}]\n" "ezpz_save_slurm_env"
    if [[ "$#" == 0 ]]; then
        # hostfile="${HOSTFILE:-${PBS_NODEFILE}}"
        hostfile="${HOSTFILE:-$(ezpz_make_slurm_nodefile)}"
        jobenv_file="${JOBENV_FILE:-${SLURM_ENV_FILE}}"
    elif [[ "$#" == 1 ]]; then
        printf "  - Caught ${BLUE}%s${RESET} arguments\n" "$#"
        hostfile="$1"
        jobenv_file="${JOBENV_FILE:-${SLURM_ENV_FILE}}"
    elif [[ "$#" == 2 ]]; then
        printf "  - Caught ${BLUE}%s${RESET} arguments\n" "$#"
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
        #     printf "  - Caught ${RED}%s${RESET} != ${RED}%s${RESET} \n" "hostfile" "SLURM_NODEFILE"
        #     printf "      - hostfile: ${RED}%s${RESET}\n" "${hostfile}"
        #     printf "      - SLURM_NODEFILE: ${RED}%s${RESET}\n" "${SLURM_NODEFILE}"
        #     printf "\n"
        # fi
        printf "  - Using:\n"
        printf "      - hostfile: ${BLUE}%s${RESET}\n" "${hostfile}"
        printf "      - jobenv_file: ${BLUE}%s${RESET}\n" "${jobenv_file}"

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
        printf "    to calculate:\n"
        printf "      - num_hosts: ${BLUE}%s${RESET}\n" "${num_hosts}"
        printf "      - num_gpus_per_host: ${BLUE}%s${RESET}\n" "${num_gpus_per_host}"
        printf "      - num_gpus: ${BLUE}%s${RESET}\n" "${num_gpus}"
        export DIST_LAUNCH="${dist_launch_cmd}"
        export ezlaunch="${DIST_LAUNCH}"
        # printf "      - DIST_LAUNCH: ${BLUE}%s${RESET}\n" "${DIST_LAUNCH}"
    fi
    export HOSTFILE="${hostfile}"
    export JOBENV_FILE="${jobenv_file}"
    printf "  - Setting:\n"
    printf "      - HOSTFILE: ${BLUE}%s${RESET}\n" "${HOSTFILE}"
    printf "      - JOBENV_FILE: ${BLUE}%s${RESET}\n\n" "${JOBENV_FILE}"
}

ezpz_setup_host_slurm() {
    log_message INFO "[${CYAN}ezpz_setup_host_slurm${RESET}]"
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
            log_message INFO "  - Using hostfile: ${CYAN}%s${RESET}\n" "${hostfile}"
            log_message INFO "  - Found in environment:\n"
            if [[ -n "${HOSTFILE:-}" ]]; then
                log_message INFO "      - HOSTFILE: ${CYAN}%s${RESET}\n" "${HOSTFILE}"
            fi
            # if [[ "${hostfile}" != "${PBS_NODEFILE}" ]]; then
        elif [[ "$#" == 1 ]]; then
            log_message INFO "  - Caught ${CYAN}%s${RESET} arguments\n" "$#"
            hostfile="$1"
            jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
            log_message INFO "  - Caught ${CYAN}%s${RESET} arguments\n" "$#"
            log_message INFO "  - hostfile=${CYAN}%s${RESET}\n" "${hostfile}"
        elif [[ "$#" == 2 ]]; then
            hostfile="$1"
            jobenv_file="$2"
            log_message INFO "  - Caught ${CYAN}%s${RESET} arguments\n" "$#"
            log_message INFO "      - hostfile=${CYAN}%s${RESET}\n" "${hostfile}"
            log_message INFO "      - jobenv_file=${CYAN}%s${RESET}\n" "${jobenv_file}"
        else
            echo "Expected exactly 0, 1, or 2 arguments, received: $#"
        fi
        log_message INFO "      - Writing SLURM vars to: ${CYAN}%s${RESET}\n" "${jobenv_file}"
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

# -----------------------------------------------------------------------------
# @description Set up the host environment for PBS scheduler.
#
# @arg $1 hostfile
# @arg $2 jobenv_file
ezpz_setup_host_pbs() {
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
        # Scenario 1: No arguments passed
        # log_message INFO "[${BLUE}ezpz_setup_host_pbs${RESET}]"
        # log_message INFO "  - Caught ${BLUE}${#}${RESET} arguments"
        if [[ "$#" == 0 ]]; then
            hostfile="${HOSTFILE:-$(ezpz_get_pbs_nodefile_from_hostname)}"
            jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
        # Scenario 2: One argument passed: hostfile
        elif [[ "$#" == 1 ]]; then
            hostfile="$1"
            jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
            log_message INFO "  - hostfile=${BLUE}${hostfile}${RESET}"
        # Scenario 3: Two arguments passed: hostfile and jobenv_file
        elif [[ "$#" == 2 ]]; then
            hostfile="$1"
            jobenv_file="$2"
            log_message INFO "  - hostfile=${BLUE}${hostfile}${RESET}"
            log_message INFO "  - jobenv_file=${BLUE}${jobenv_file}${RESET}"
        else
            log_message ERROR "Expected exactly 0, 1, or 2 arguments, received: $#"
        fi
        hn=$(hostname)
        if [[ "${hn}" == x* || "${hn}" == "sophia*" ]]; then
            if [[ "${mn}" == "polaris" || "${mn}" == "sirius" || "${mn}" == "sophia*" ]]; then
                export GPU_TYPE="NVIDIA"
            elif [[ "${mn}" == "aurora" || "${mn}" == "sunspot" ]]; then
                # Each Aurora node has 12 Intel XPU devices
                export GPU_TYPE="INTEL"
                export NGPU_PER_TILE=6
                export NTILE_PER_HOST=2
                export NGPU_PER_HOST=$((NGPU_PER_TILE * NTILE_PER_HOST))
            fi
            ezpz_save_pbs_env "$@"
        fi
    fi
}

# -----------------------------------------------------------------------------
# @description ezpz_setup_host
#
# @example:
#  ezpz_setup_host
#
# @output:
#  - Sets up the host environment for the current machine.
ezpz_setup_host() {
    mn=$(ezpz_get_machine_name)
    scheduler_type=$(ezpz_get_scheduler_type)
    if [[ "${scheduler_type}" == "pbs" ]]; then
        ezpz_setup_host_pbs "$@"
    elif [[ "${scheduler_type}" ]]; then
        ezpz_setup_host_slurm "$@"
    else
        log_message ERROR "Unknown scheduler: ${scheduler_type} on ${mn}"
    fi
}

ezpz_print_hosts() {
    local hostfile
    if [[ "$#" == 0 ]]; then
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}}"
    elif [[ "$#" == 1 ]]; then
        hostfile="$1"
    else
        # hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE}}}"
        hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}}"
    fi
    log_message INFO "[${MAGENTA}HOSTS${RESET}]"
    log_message INFO "  - HOSTFILE=${MAGENTA}${hostfile}${RESET}"
    log_message INFO "  - NHOSTS=${MAGENTA}$(ezpz_get_num_hosts "${hostfile}")${RESET}"
    log_message INFO "  - HOSTS:"
    counter=0
    for f in $(/bin/cat "${hostfile}"); do
        log_message INFO "    - [host:${MAGENTA}${counter}${RESET}] - ${MAGENTA}${f}${RESET}"
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
        export LAUNCH="${DIST_LAUNCH}"
        export ezlaunch="${DIST_LAUNCH}"
        alias launch="${LAUNCH}"
        # printf "[${MAGENTA}%s${RESET}]\n" "HOSTS"
        # printf "hostfile: ${MAGENTA}%s${RESET}\n" "${hostfile}"
        ezpz_print_hosts "${hostfile}"
        log_message INFO "[${BRIGHT_BLUE}DIST_INFO${RESET}]"
        log_message INFO "  - HOSTFILE=${BRIGHT_BLUE}${hostfile}${RESET}"
        log_message INFO "  - NHOSTS=${BRIGHT_BLUE}${NHOSTS}${RESET}"
        log_message INFO "  - NGPU_PER_HOST=${BRIGHT_BLUE}${NGPU_PER_HOST}${RESET}"
        log_message INFO "  - NGPUS=${BRIGHT_BLUE}${NGPUS}${RESET}"
        # log_message INFO "  - DIST_LAUNCH=${BRIGHT_BLUE}${DIST_LAUNCH}${RESET}"
        # printf "\n"
        # printf "[${BRIGHT_BLUE}%s${RESET}]\n" "DIST INFO"
        # printf "  - NGPUS=${BRIGHT_BLUE}%s${RESET}\n" "$NGPUS"
        # printf "  - NHOSTS=${BRIGHT_BLUE}%s${RESET}\n" "${NHOSTS}"
        # printf "  - NGPU_PER_HOST=${BRIGHT_BLUE}%s${RESET}\n" "${NGPU_PER_HOST}"
        # printf "  - HOSTFILE=${BRIGHT_BLUE}%s${RESET}\n" "${hostfile}"
        # printf "  - DIST_LAUNCH=${BRIGHT_BLUE}%s${RESET}\n" "${DIST_LAUNCH}"
        # printf "\n"
        if [[ -n "$(command -v launch)" ]]; then
            log_message INFO "[${GREEN}LAUNCH${RESET}]"
            log_message INFO "  - To launch across all available GPUs, use: '${GREEN}launch${RESET}'"
            log_message INFO "    ${GREEN}launch${RESET} = ${GREEN}${LAUNCH}${RESET}"
            log_message INFO "  - Run '${GREEN}which launch${RESET}' to ensure that the alias is set correctly"
        fi
        # echo "┌────────────────────────────────────────────────────────────────────────────────"
        # echo "│ YOU ARE $(whereAmI)"
        # echo "│ Run 'source ./bin/getjobenv' in a NEW SHELL to automatically set env vars      "
        # echo "└────────────────────────────────────────────────────────────────────────────────"
    fi
}

ezpz_launch() {
    if [[ -v WORLD_SIZE ]]; then
        dlaunch="$(echo "${DIST_LAUNCH}" | sed "s/-n\ ${NGPUS}/-n\ ${WORLD_SIZE}/g")"
    else
        dlaunch="${DIST_LAUNCH}"
    fi
    _args=("${@}")
    printf "[yeet]:\n"
    printf "evaluating:\n${GREEN}%s${RESET}\n" "${dlaunch}"
    printf "with arguments:\n${BLUE}%s${RESET}\n" "${_args[*]}"
    eval "${dlaunch} ${*}"
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

ezpz_get_pbs_jobid() {
    local _pbs_nodefile
    _pbs_nodefile=$(ezpz_get_pbs_nodefile_from_hostname)
    local _pbs_jobid
    _pbs_jobid="$(basename "${_pbs_nodefile}")"
    echo "${_pbs_jobid}"
}

# -----------------------------------------------------------------------------
# @description Get the PBS environment variables.
#
# @example:
#   ezpz_get_pbs_env
#
# @arg $1 hostfile
# @arg $2 jobenv_file
#
#
# @set DIST_LAUNCH string Distributed launch command.
# @set ezplaunch string Distributed launch command.
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
    log_message INFO "[${BLUE}ezpz_get_pbs_env${RESET}]"
    log_message INFO "  - Caught ${BLUE}${#}${RESET} arguments"
    log_message INFO "  - hostfile=${BLUE}${hostfile}${RESET}"
    log_message INFO "  - jobenv_file=${BLUE}${jobenv_file}${RESET}"
    mn=$(ezpz_get_machine_name)
    scheduler_type=$(ezpz_get_scheduler_type)
    _pbs_jobid=$(ezpz_get_pbs_jobid)
    export PBS_JOBID="${_pbs_jobid}"
    if [[ "${scheduler_type}" == "pbs" ]]; then
        if grep -q "$(hostname)" "${hostfile:-}"; then
            log_message INFO "  - Host ${BLUE}$(hostname)${RESET} found in ${BLUE}${hostfile}${RESET}"
            num_hosts=$(ezpz_get_num_hosts "${hostfile}")
            num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
            num_gpus="$((num_hosts * num_gpus_per_host))"
            dist_launch_cmd=$(ezpz_get_dist_launch_cmd "${hostfile}")
            export DIST_LAUNCH="${DIST_LAUNCH}"
            export ezlaunch="${DIST_LAUNCH}"
            return 0
        else
            log_message ERROR "Host $(hostname) not found in ${hostfile:-}"
            return 1
        fi
    else
        log_message ERROR "Skipping ezpz_get_pbs_env() on $(hostname)"
        return 1
    fi
}

ezpz_get_slurm_env() {
    if [[ -n "${SLURM_JOB_ID}" ]]; then
        export JOBENV_FILE="${SLURM_ENV_FILE}"
        # shellcheck source="${HOME}/.slurmenv"
        # shellcheck disable=SC1091
        [ -f "${HOME}/.slurmenv" ] && source "${HOME}/.slurmenv"
        # [ -f "${JOBENV_FILE}" ] && source "${JOBENV_FILE}"
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
    # printf "\n[${MAGENTA}%s${RESET}]:\n" "HOSTS"
    ezpz_print_hosts "${hostfile}"
    printf "\n[${BRIGHT_BLUE}%s${RESET}]:\n" "DIST INFO"
    printf "  - NGPUS=${BRIGHT_BLUE}%s${RESET}\n" "${num_gpus}"
    printf "  - NHOSTS=${BRIGHT_BLUE}%s${RESET}\n" "${num_hosts}"
    printf "  - NGPU_PER_HOST=${BRIGHT_BLUE}%s${RESET}\n" "${num_gpus_per_host}"
    printf "  - HOSTFILE=${BRIGHT_BLUE}%s${RESET}\n" "${hostfile}"
    printf "  - LAUNCH=${BRIGHT_BLUE}%s${RESET}\n" "${LAUNCH}"
    printf "  - DIST_LAUNCH=${BRIGHT_BLUE}%s${RESET}\n" "${DIST_LAUNCH}"
    printf "\n[${GREEN}%s${RESET}]:\n" "LAUNCH"
    printf "  - To launch across all available GPUs, use:\n"
    printf "  '${GREEN}launch${RESET}' ( = ${GREEN}%s${RESET} )\n\n" "${LAUNCH}"
    # •
}

# @description: ezpz_setup_alcf
# Setups the environment for ALCF systems
#
# @example
#    ezpz_setup_alcf
#
# @arg $1 string hostfile
# @stdout Output the job environment information
ezpz_setup_alcf() {
    mn=$(ezpz_get_machine_name)
    hn=$(hostname)
    local mn="${mn}"
    local hn="${hn}"
    printf "\n"
    printf "[%s ${YELLOW}%s${RESET}]\n" "🍋" "ezpz/bin/utils.sh"
    printf "\n"
    printf "  - USER=${BLACK}%s${RESET}\n" "${USER}"
    printf "  - MACHINE=${BLACK}%s${RESET}\n" "${mn}"
    printf "  - HOST=${BLACK}%s${RESET}\n" "${hn}"
    printf "  - TSTAMP=${BLACK}%s${RESET}\n\n" "$(ezpz_get_tstamp)"
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

# --- Main Orchestration Functions ---
#
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

# -----------------------------------------------------------------------------
# Main entry point for SAVING job environment variables and info.
# Calls `ezpz_setup_host` (saves scheduler vars) and `ezpz_write_job_info`
# (saves calculated vars, defines launch func, prints summary).
#
# Args:
#   $@: Arguments passed to underlying functions (hostfile, jobenv_file).
#
# Outputs: Creates/appends jobenv file, exports vars, defines launch func, prints summary.
# -----------------------------------------------------------------------------
# ezpz_savejobenv_main1() {
#     printf "${MAGENTA}==== Running ezpz_savejobenv_main ====${RESET}\n"
#     # Setup host first (detects scheduler, saves SLURM/PBS vars, determines hostfile/jobenv)
#     if ! ezpz_setup_host "$@"; then
#          printf "${RED}Error during ezpz_setup_host. Aborting savejobenv.${RESET}\n" >&2
#          return 1
#     fi
#
#     # Write calculated info (NGPUS, launch cmd etc.) using determined hostfile/jobenv
#     # Pass HOSTFILE and JOBENV_FILE explicitly if they were set by ezpz_setup_host
#     # Ensure HOSTFILE is available before calling write_job_info
#     if [[ -z "${HOSTFILE:-}" || ! -f "${HOSTFILE:-}" ]]; then
#         printf "${RED}Error: HOSTFILE not valid after ezpz_setup_host. Cannot write job info.${RESET}\n" >&2
#         return 1
#     fi
#     if ! ezpz_write_job_info "${HOSTFILE:-}" "${JOBENV_FILE:-}"; then
#          printf "${RED}Error during ezpz_write_job_info.${RESET}\n" >&2
#          return 1 # Return failure status
#     fi
#
#     printf "${MAGENTA}==== Finished ezpz_savejobenv_main (Status: 0) ====${RESET}\n"
#     return 0
# }
#
ezpz_get_pbs_jobid_from_nodefile() {
    if [[ -z "${PBS_NODEFILE:-}" ]]; then
        echo "No PBS_NODEFILE found"
        return 1
    else
        _pbs_jobid="$(echo "${PBS_NODEFILE//#/}" | tr "\/" " " | awk '{print $NF}' | tr "." " " | awk '{print $1}')"
        export PBS_JOBID="${_pbs_jobid}"
    fi

}

ezpz_setup_job() {
    mn=$(ezpz_get_machine_name)
    hn=$(hostname)
    local mn="${mn}"
    local hn="${hn}"
    local _pbs_jobid
    _pbs_jobid=$(ezpz_get_pbs_jobid)
    log_message INFO "[${YELLOW}JOB${RESET}]"
    log_message INFO "  - Setting up job for ${YELLOW}${USER}${RESET}"
    log_message INFO "  - Machine: ${YELLOW}${mn}${RESET}"
    log_message INFO "  - Hostname: ${YELLOW}${hn}${RESET}"
    export PBS_JOBID="${_pbs_jobid}"
    # log_message INFO "PBS_NODEFILE: ${YELLOW}${PBS_NODEFILE:-<not set>}${RESET}"
    # printf "\n[%s ${YELLOW}%s${RESET}]\n" "🍋" "ezpz/bin/utils.sh"
    # printf "  - USER=${YELLOW}%s${RESET}\n" "${USER}"
    # printf "  - MACHINE=${YELLOW}%s${RESET}\n" "${mn}"
    # printf "  - HOST=${YELLOW}%s${RESET}\n" "${hn}"
    # printf "  - TSTAMP=${YELLOW}%s${RESET}\n\n" "$(ezpz_get_tstamp)"
    if [[ -n "${PBS_JOBID:-}" ]]; then
        log_message INFO "  - PBS_JOBID=${YELLOW}${PBS_JOBID}${RESET}"
    elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
        log_message INFO "  - SLURM_JOB_ID=${YELLOW}${SLURM_JOB_ID}${RESET}"
    else
        if [[ "$(ezpz_get_scheduler_type)" == "pbs" ]]; then
            ezpz_get_pbs_jobid
            export PBS_JOBID="${_pbs_jobid}"
            log_message INFO "  - PBS_JOBID=${YELLOW}${PBS_JOBID}${RESET}"
        fi
    fi
    if [[ -n "${PBS_NODEFILE:-}" ]]; then
        ezpz_savejobenv_main "$@"
    elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
        ezpz_savejobenv_main "$@"
    else
        scheduler_type=$(ezpz_get_scheduler_type)
        if [[ "${scheduler_type}" == "pbs" ]]; then
            _pbs_nodefile=$(ezpz_get_pbs_nodefile_from_hostname)
            _pbs_jobid="$(echo "${PBS_NODEFILE//#/}" | tr "\/" " " | awk '{print $NF}' | tr "." " " | awk '{print $1}')"
            export PBS_JOBID="${_pbs_jobid}"
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
    if [[ "${mn}" == "aurora" ]]; then
        export ITEX_VERBOSE="${ITEX_VERBOSE:-0}"
        export LOG_LEVEL_ALL="${LOG_LEVEL_ALL:-5}"
        export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-5}"
        export ITEX_CPP_MIN_LOG_LEVEL="${ITEX_CPP_MIN_LOG_LEVEL:-5}"
        export CCL_LOG_LEVEL="${CCL_LOG_LEVEL:-ERROR}"
    fi
}

# Set up the necessary modules for the new PyTorch 2.{7,8} environments.
# It unloads existing modules, loads the required ones, and sets environment variables.
ezpz_load_new_pt_modules_aurora() {
    module restore
    module unload oneapi mpich
    module use /soft/compilers/oneapi/2025.1.3/modulefiles
    module use /soft/compilers/oneapi/nope/modulefiles
    module add mpich/nope/develop-git.6037a7a
    module load cmake
    unset CMAKE_ROOT
    export A21_SDK_PTIROOT_OVERRIDE=/home/cchannui/debug5/pti-gpu-test/tools/pti-gpu/d5c2e2e
    module add oneapi/public/2025.1.3
    export ZE_FLAT_DEVICE_HIERARCHY="FLAT"
}


# Set up the Python environment for new PyTorch 2.{7,8} on Aurora.
# It activates the specified conda environment and loads the necessary modules.
ezpz_setup_python_pt_new_aurora() {
    if [[ "$#" -ne 1 ]]; then
        log_message ERROR "${RED}Usage: ezpz_setup_python_pt_new <conda_env>${RESET}"
        return 1
    fi
    local conda_env="$1"
    log_message INFO "  - Running ${BRIGHT_GREEN}ezpz_setup_python_pt_new_aurora${RESET}..."
    log_message INFO "  - Using conda environment: ${BRIGHT_GREEN}${conda_env}${RESET}"
    ezpz_load_new_pt_modules_aurora
    micromamba activate "${conda_env}" || {
        log_message ERROR "Failed to micromamba activate ${RED}${conda_env}${RESET}. Returning 1"
        return 1
    }
    ezpz_setup_python || {
        log_message ERROR "Failed to call ${RED}ezpz_setup_python${RESET}. Returning 1"
        return 1
    }
    log_message INFO "  - ${GREEN}[✓] Finished${RESET} [${BRIGHT_GREEN}ezpz_setup_python_pt_new_aurora${RESET}]"
    return 0
}

ezpz_setup_python_pt29_aurora() {
    pt29env="${PT29_ENV:-/flare/datascience_collab/foremans/micromamba/envs/pt29-2025-07}"
    # log_message INFO "  - Running ${BRIGHT_GREEN}ezpz_setup_python_pt29${RESET}..."
    log_message INFO "[${BRIGHT_GREEN}ezpz_setup_python_pt29_aurora${RESET}]"
    log_message INFO "  - Using PT29_ENV=${BRIGHT_GREEN}${pt29env}${RESET}"
    ezpz_setup_python_pt_new_aurora "${pt29env}" || {
        log_message ERROR "Failed to call ${RED}ezpz_setup_python_pt_new${RESET} ${pt29env}. Returning 1"
        return 1
    }
    return 0
}

ezpz_setup_python_pt28_aurora() {
    pt28env="${PT28_ENV:-/flare/datascience_collab/foremans/micromamba/envs/pt28-2025-07}"
    # log_message INFO "  - Running ${BRIGHT_GREEN}ezpz_setup_python_pt28${RESET}..."
    log_message INFO "[${BRIGHT_GREEN}ezpz_setup_python_pt28_aurora${RESET}]"
    log_message INFO "  - Using PT28_ENV=${BRIGHT_GREEN}${pt28env}${RESET}"
    ezpz_setup_python_pt_new_aurora "${pt28env}" || {
        log_message ERROR "Failed to call ${RED}ezpz_setup_python_pt_new${RESET} ${pt28env}. Returning 1"
        return 1
    }
}

# Helper function 
ezpz_setup_env_pt29_aurora() {
    ezpz_setup_python_pt29_aurora || {
        log_message ERROR "Python setup for pt29 failed. Aborting."
        return 1
    }
    ezpz_setup_job "$@" || {
        log_message ERROR "Job setup failed @ ${RED}ezpz_setup_job${RESET}. Aborting."
        return 1
    }
    return 0
}

ezpz_setup_env_pt28_aurora() {
    ezpz_setup_python_pt28_aurora || {
        log_message ERROR "Python setup for pt28 failed. Aborting."
        return 1
    }
    ezpz_setup_job "$@" || {
        log_message ERROR "Job setup failed @ ${RED}ezpz_setup_job${RESET}. Aborting."
        return 1
    }
    return 0
}


_ezpz_install_from_git() {
    python3 -m pip install "git+https://github.com/saforem2/ezpz" --require-virtualenv
}


# Function to install ezpz into the currently active virtualenv.
# Checks if a virtualenv or conda env is active, and if PYTHON_EXEC is set.
# If checks fail, it logs an error and returns 1.
# If checks pass, it attempts to install ezpz from GitHub using pip.
ezpz_install() {
    if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
        log_message ERROR "No virtual environment or conda environment is active. Please activate one before installing ezpz."
        return 1
    fi
    if [[ -z "${PYTHON_EXEC:-}" ]]; then
        log_message ERROR "PYTHON_EXEC is not set. Please ensure Python is available in your environment."
        return 1
    fi
    log_message INFO "Installing ezpz into environment at ${VIRTUAL_ENV:-${CONDA_PREFIX}}"
    if ! _ezpz_install_from_git; then
        log_message ERROR "${RED}✘${RESET} Failed to install ezpz. Please check the error messages above."
        log_message ERROR "If you see a 'No module named pip' error, please ensure pip is installed in your environment."
        return 1
    fi
}

# Usage: setup_modules
# ezpz_setup_env_pt27() {
#     source /opt/aurora/24.347.0/spack/unified/0.9.2/install/linux-sles15-x86_64/gcc-13.3.0/miniforge3-24.3.0-0-gfganax/bin/activate
#     conda activate /lus/flare/projects/datascience/foremans/miniforge/2025-07-pt27
#     ezpz_load_new_pt_modules
# }

# -----------------------------------------------------------------------------
# Comprehensive setup: Python environment AND Job environment.
# Calls `ezpz_setup_python` then `ezpz_setup_job`. Recommended entry point.
#
# Usage:
#   source utils_modern.sh && ezpz_setup_env
#
# Args:
#   $@: Arguments passed to `ezpz_setup_job` (hostfile, jobenv_file).
# Outputs: Sets up Python & Job envs. Prints summaries. Returns 1 on failure.
# -----------------------------------------------------------------------------
ezpz_setup_env() {
    if ! ezpz_check_working_dir; then
        log_mesasge ERROR "Failed to set WORKING_DIR. Please check your environment."
    fi
    log_message info "running [${BRIGHT_YELLOW}ezpz_setup_env${RESET}]..."
    if ! ezpz_setup_python; then
        log_message ERROR "Python setup failed. Aborting."
        return 1
    fi
    if ! ezpz_setup_job "$@"; then
        log_message ERROR "Job setup failed. Aborting."
        return 1
    fi
    log_message INFO "${GREEN}[✓] Finished${RESET} [${BRIGHT_YELLOW}ezpz_setup_env${RESET}]"
    return 0
}

# ezpz_setup_install() {
#     printf "[ezpz] Loading python modules and looking for virtual environment...\n"
#     ezpz_setup_python
#     printf "[ezpz] Determining job information from hostname=%s...\n" "$(hostname)"
#     ezpz_setup_job
#     printf "[ezpz] Installing https://github.com/saforem2/ezpz into %s\n" "${VIRTUAL_ENV}"
#     if ! python3 -m pip install "git+https://github.com/saforem2/ezpz" --require-virtualenv; then
#         printf "[ezpz] :x: Failed to install ezpz into %s\n" "${VIRTUAL_ENV}"
#         exit 1
#     fi
#     printf "[ezpz] :check: Done!"
# }

# -----------------------------------------------------------------------------
# Setup environment and install the `ezpz` package itself using pip.
# Calls `ezpz_setup_python`, `ezpz_setup_job`, then `pip install`.
#
# Usage:
#   source utils_modern.sh && ezpz_setup_install
#
# Args:
#   $@: Arguments passed to `ezpz_setup_job` (hostfile, jobenv_file).
# Outputs: Sets up envs, installs `ezpz`. Prints status. Exits(1) on failure.
# -----------------------------------------------------------------------------
ezpz_setup_install() {
    printf "[ezpz] Setting up Python environment\n"
    ezpz_setup_python || {
        log_message ERROR "Python setup failed. Aborting."
        return 1
    }

    printf "[ezpz] Setting up Job environment\n"
    ezpz_setup_job "$@" || {
        log_message ERROR "Job setup failed. Aborting."
        return 1
    }

    local target_env_path="${VIRTUAL_ENV:-${CONDA_PREFIX:-<unknown>}}"
    printf "[ezpz] Installing ezpz from GitHub into %s\n" "${target_env_path}"

    if [[ -z "${PYTHON_EXEC:-}" ]]; then
        log_message WARN "PYTHON_EXEC not set. Attempting to set it now..."
        return 1
    fi

    # Install using pip, requiring virtualenv to avoid global installs if VIRTUAL_ENV is set
    local pip_cmd=("${PYTHON_EXEC}" -m pip install "git+https://github.com/saforem2/ezpz")
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        pip_cmd+=("--require-virtualenv")
    fi

    if ! "${pip_cmd[@]}"; then
        log_message ERROR "${RED}✘${RESET} Failed to install ezpz into ${target_env_path}"
        log_message ERROR "Please check the error messages above."
        return 1
    fi
    printf "[ezpz] ${GREEN}:check: Done!${RESET}\n"
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

# --- Main Execution Block (when sourced) ---

# -----------------------------------------------------------------------------
# Main logic executed when the script is sourced.
# Determines the working directory based on PBS_O_WORKDIR, SLURM_SUBMIT_DIR, or pwd.
# Exports WORKING_DIR.
# -----------------------------------------------------------------------------
ezpz_get_working_dir() {
    python3 -c 'import os; print(os.getcwd())'
    # echo $(python3 -c 'from pathlib import Path; print(Path().absolute())') | tr "/lus" ""
}

ezpz_check_working_dir() {
    GIT_BRANCH=$(git branch --show-current) && export GIT_BRANCH
    WORKING_DIR=$(ezpz_get_working_dir)
    export WORKING_DIR="${WORKING_DIR}"

    # NOTE: [Scenario 1]
    # - If PBS_O_WORKDIR is empty (not set) use $(pwd)
    if [[ -z "${PBS_O_WORKDIR:-}" ]]; then
        # Set PBS_O_WORKDIR as WORKING_DIR
        log_message WARN "PBS_O_WORKDIR is not set! Setting it to current working directory"
        log_message INFO "Exporting PBS_O_WORKDIR=${GREEN}${WORKING_DIR}${RESET}"
        export PBS_O_WORKDIR="${WORKING_DIR}"

    # NOTE: [Scenario 2]
    # - If PBS_O_WORKDIR is set, check if it matches the current working directory
    elif [[ -n "${PBS_O_WORKDIR:-}" ]]; then
        if [[ "${WORKING_DIR}" != "${PBS_O_WORKDIR:-}" ]]; then
            log_message WARN "Current working directory does not match PBS_O_WORKDIR! This may cause issues with the job submission."
            log_message WARN "PBS_O_WORKDIR" "$(printf "${RED}%s${RESET}" "${PBS_O_WORKDIR}")"
            log_message WARN "WORKING_DIR" "$(printf "${GREEN}%s${RESET}" "${WORKING_DIR}")"
            log_message WARN "Exporting PBS_O_WORKDIR=WORKING_DIR=$(printf "${BLUE}%s${RESET}" "${WORKING_DIR}") and continuing..."
            export PBS_O_WORKDIR="${WORKING_DIR}"
        fi

    # NOTE: [Scenario 3]
    # - If SLURM_SUBMIT_DIR is set, check if it matches the current working directory
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        # TODO: Add similar logic for SLURM environments
        # WORKING_DIR="${SLURM_SUBMIT_DIR}"
        if [[ "${WORKING_DIR}" != "${SLURM_SUBMIT_DIR:-}" ]]; then
            log_message WARN "Current working directory does not match SLURM_SUBMIT_DIR! This may cause issues with the job submission."
            log_message WARN "SLURM_SUBMIT_DIR=${RED}${SLURM_SUBMIT_DIR}${RESET}"
            log_message WARN "WORKING_DIR=${GREEN}${WORKING_DIR}${RESET}"
            log_message WARN "Exporting SLURM_SUBMIT_DIR=WORKING_DIR=${BLUE}${WORKING_DIR}${RESET} and continuing..."
            export SLURM_SUBMIT_DIR="${WORKING_DIR}"
        fi

    # NOTE: [Scenario 4]
    # - If neither PBS_O_WORKDIR nor SLURM_SUBMIT_DIR are set, use the current working directory
    else
        log_message INFO "Unable to detect PBS or SLURM working directory info..."
        log_message INFO "Using current working directory (${GREEN}${WORKING_DIR}${RESET}) as working directory..."
    fi
}

# --- Script Entry Point ---
# Call utils_main when the script is sourced to set WORKING_DIR.
# If it fails, print an error but allow sourcing to continue (individual functions might still work).
if ! ezpz_check_working_dir; then
    log_mesasge ERROR "Failed to set WORKING_DIR. Please check your environment."
fi

# If DEBUG mode was enabled, turn off command tracing now that setup is done.
if [[ -n "${DEBUG:-}" ]]; then
    log_message WARN "DEBUG MODE IS ${RED}OFF${RESET}"
    set +x
fi
