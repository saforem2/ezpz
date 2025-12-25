#!/bin/bash
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
#
EZPZ_SHELL_TYPE="$(basename "${SHELL}")"

if [[ "${EZPZ_SHELL_TYPE}" == "bash" ]]; then
	# Allow aliases to be expanded (needed for `launch` alias)
	shopt -s expand_aliases
elif [[ "${EZPZ_SHELL_TYPE}" == "zsh" ]]; then
	setopt LOCAL_OPTIONS # make sure options are local to this script
	setopt KSH_ARRAYS    # arrays are 0-indexed
fi

###############################################
# Helper functions for printing colored text
###############################################
if [[ -n "${NO_COLOR:-}" || -n "${NOCOLOR:-}" || "${COLOR:-}" == 0 || "${TERM}" == "dumb" ]]; then
	RESET=''
	BLACK=''
	RED=''
	BRIGHT_RED=''
	GREEN=''
	BRIGHT_GREEN=''
	YELLOW=''
	BRIGHT_YELLOW=''
	BLUE=''
	BRIGHT_BLUE=''
	MAGENTA=''
	BRIGHT_MAGENTA=''
	CYAN=''
	BRIGHT_CYAN=''
	WHITE=''
	BRIGHT_WHITE=''
else
	# --- Color Codes ---
	# Usage: printf "${RED}This is red text${RESET}\n"
	RESET='\e[0m'
	# BLACK='\e[1;30m' # Avoid black text
	RED='\e[1;31m'
	BRIGHT_RED='\e[1;91m'
	GREEN='\e[1;32m'
	BRIGHT_GREEN='\e[1;92m'
	YELLOW='\e[1;33m'
	BRIGHT_YELLOW='\e[1;93m'
	BLUE='\e[1;34m'
	BRIGHT_BLUE='\e[1;94m'
	MAGENTA='\e[1;35m'
	BRIGHT_MAGENTA='\e[1;95m'
	CYAN='\e[1;36m'
	BRIGHT_CYAN='\e[1;96m'
	WHITE='\e[1;37m'        # Avoid white on light terminals
	BRIGHT_WHITE='\e[1;97m' # Added for emphasis
fi

# --- tiny helpers (safe in bash + zsh) ---------------------------------------
ezpz_is_sourced() { [[ "${BASH_SOURCE[0]:-}" != "${0}" ]]; }

ezpz_has() { command -v "$1" >/dev/null 2>&1; }

ezpz_realpath() {
	# realpath isn't always there (mac, minimal images)
	# if ezpz_has realpath; then realpath "$1"; else python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$1"; fi
	# p
	python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$1"
}

ezpz_require_file() {
	local fp="$1" what="${2:-file}"
	[[ -n "${fp}" && -f "${fp}" ]] || {
		log_message ERROR "${what} not found: ${fp}"
		return 1
	}
}

ezpz_ensure_micromamba_hook() {
	local shell_type
	shell_type="$(basename "${SHELL:-bash}")"
	eval "$(micromamba shell hook --shell "${shell_type}")"
}

# --- Helper Functions ---

# # Set the default log level to INFO if the
# # environment variable isn't already set.
EZPZ_LOG_LEVEL="${EZPZ_LOG_LEVEL:-INFO}"
export EZPZ_LOG_LEVEL

log_info() {
	args=("$@")
	printf "[%s][${GREEN}I${RESET}] - %s\n" "$(ezpz_get_tstamp)" "${args[*]}"
}

log_warn() {
	args=("$@")
	printf "[%s][${YELLOW}W${RESET}] - %s\n" "$(ezpz_get_tstamp)" "${args[*]}"
}

log_error() {
	args=("$@")
	printf "[%s][${RED}E${RESET}] - %s\n" "$(ezpz_get_tstamp)" "${args[*]}" >&2
}

# alias log_message='log_message_stdout ${FUNCNAME} ${LINENO}'
#
log_message() {
	local level="$1"
	shift || true
	local string="$*"
	local date log_level log_msg
	date="$(ezpz_get_tstamp)"
	log_level="${level:-$EZPZ_LOG_LEVEL}"

	case "${log_level}" in
	D | DEBUG) log_level="${CYAN}D${RESET}" ;;
	I | INFO) log_level="${GREEN}I${RESET}" ;;
	W | WARN | WARNING) log_level="${YELLOW}W${RESET}" ;;
	E | ERROR) log_level="${RED}E${RESET}" ;;
	F | FATAL) log_level="${RED}F${RESET}" ;;
	*) log_level="${GREEN}I${RESET}" ;;
	esac

	if [[ "${EZPZ_SHELL_TYPE}" == "bash" ]]; then
		log_msg="[${date}][${log_level}][${BASH_SOURCE[1]}:${BASH_LINENO[0]}] ${string}"
	elif [[ "${EZPZ_SHELL_TYPE}" == "zsh" ]]; then
		local fft=("${funcfiletrace[@]}")
		log_msg="[${date}][${log_level}][${fft[1]}] ${string}"
	else
		log_msg="[${date}][${log_level}] ${string}"
	fi

	# printf is predictable; preserves backslashes unless you add %b intentionally
	printf "%b\n" "${log_msg}"
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

ezpz_kill_mpi() {
	# Kill matching processes owned by $USER (pals|mpi|python), excluding grep itself
	local pids
	pids="$(ps -u "${USER}" -o pid=,comm=,args= | awk '/(pals|mpi|python)/ && !/awk/ {print $1}')"
	[[ -n "${pids}" ]] && echo "${pids}" | xargs -r kill
}

# Use the first `n` lines from `PBS_NODEFILE` to create a new hostfile
ezpz_head_n_from_pbs_nodefile() {
	if [[ "$#" -ne 1 ]]; then
		log_message ERROR "Usage: $0 <NHOSTS>"
		return 1
	fi
	local _num_nodes="$1"
	if [[ -z "${PBS_NODEFILE}" ]]; then
		log_message ERROR "${RED}PBS_NODEFILE is not set. Cannot tail nodefile.${RESET}"
		return 1
	fi
	if [[ ! -f "${PBS_NODEFILE}" ]]; then
		log_message ERROR "${RED}PBS_NODEFILE does not exist: ${PBS_NODEFILE}${RESET}"
		return 1
	fi
	_of="nodefile-0-${_num_nodes}"
	head -n "${_num_nodes}" "${PBS_NODEFILE}" >"${_of}" && wc -l "${_of}"
}

# Function to use the last `n` lins from `PBS_NODEFILE` to create a new hostfile.
ezpz_tail_n_from_pbs_nodefile() {
	if [[ "$#" -ne 1 ]]; then
		log_message ERROR "Usage: $0 <NHOSTS>"
		return 1
	fi
	local _num_nodes="$1"
	[[ -n "${PBS_NODEFILE:-}" ]] || {
		log_message ERROR "${RED}PBS_NODEFILE is not set. Cannot tail nodefile.${RESET}"
		return 1
	}
	ezpz_require_file "${PBS_NODEFILE}" "PBS_NODEFILE" || return 1

	local total _of
	total="$(wc -l <"${PBS_NODEFILE}")"
	_of="nodefile-$((total - _num_nodes))-${total}"
	tail -n "${_num_nodes}" "${PBS_NODEFILE}" >"${_of}" && wc -l "${_of}"
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
	if command -v module &>/dev/null; then
		module list
	else
		log_message WARN "lmod 'module' not found. Skipping module list."
	fi
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
	printenv | grep -E "($(
		IFS='|'
		echo "${vars[*]}"
	))"
}

ezpz_reset() {
	log_message INFO "Current environment before reset:"
	log_message INFO "$(ezpz_show_env)"

	if command -v module &>/dev/null; then
		log_message INFO "Unloading all loaded modules..."
		module reset
	else
		log_message WARN "lmod 'module' not found. Skipping module purge."
	fi

	if [[ -n "${VIRTUAL_ENV:-}" ]]; then
		log_message INFO "Deactivating virtual environment at: ${VIRTUAL_ENV}"
		deactivate
	else
		log_message WARN "No virtual environment is active."
		log_message INFO "Skipping virtual environment deactivation step."
	fi

	if [[ -n "${CONDA_PREFIX:-}" ]]; then
		log_message INFO "Deactivating conda environment at: ${CONDA_PREFIX}"
		if command -v conda &>/dev/null; then
			conda deactivate
		fi
	else
		log_message WARN "No conda environment is active."
		log_message INFO "Skipping conda environment deactivation step."
	fi

	# if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
	#     log_message ERROR "No virtual environment or conda environment is active. Please activate one before installing ezpz."
	#     return 1
	# fi
	vars=(
		"ezlaunch"
		"CONDA_DEFAULT_ENV"
		"CONDA_NAME"
		"CONDA_PREFIX"
		"CONDA_SHLVL"
		"DEFAULT_PYTHON_VERSION"
		"DIST_LAUNCH"
		"EZPZ_LOG_LEVEL"
		"GPU_TYPE"
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
		"PYTHON_EXEC"
		"PYTHON_ROOT"
		"VENV_DIR"
		"VIRTUAL_ENV"
		"VIRTUAL_ENV_PROMPT"
		"WORLD_SIZE"
		"WORKING_DIR"
		# "RESET"
		# "RED"
		# "BRIGHT_RED"
		# "GREEN"
		# "BRIGHT_GREEN"
		# "YELLOW"
		# "BRIGHT_YELLOW"
		# "BLUE"
		# "BRIGHT_BLUE"
		# "MAGENTA"
		# "BRIGHT_MAGENTA"
		# "CYAN"
		# "BRIGHT_CYAN"
		# "WHITE"
		# "BRIGHT_WHITE"
	)
	for v in "${vars[@]}"; do
		echo "Unsetting ${v}"
		unset -v "${v}"
	done
	log_message INFO "ezpz_reset completed!"
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
		log_message ERROR "${RED}${estr}${RESET} Expected one argument (outdir). Received: ${#}"
	else
		outdir="$1"
		mkdir -p "${outdir}"
		if ! command -v module &>/dev/null; then
			log_message WARN "lmod 'module' not found. Skipping module list save."
			return 0
		fi
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

	local ldir
	ldir="$(realpath "$1")"
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
	local mn
	mn="$(hostname | tr '[:upper:]' '[:lower:]')"
	case "${mn}" in
	sophia*) machine="sophia" ;;
	x1* | uan* | sunspot*) machine="sunspot" ;;
	x3* | polaris*)
		if [[ "${PBS_O_HOST:-}" == sirius* ]]; then
			machine="sirius"
		else
			machine="polaris"
		fi
		;;
	x4* | aurora*) machine="aurora" ;;
	frontier*) machine="frontier" ;;
	nid* | login*) machine="perlmutter" ;;
	*) machine="${mn}" ;;
	esac
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
		jobid="$(echo "${jobid}" | tr "." " " | awk '{print $1}' | uniq)"
		echo "${jobid}"
	fi
}

ezpz_get_slurm_running_nodelist() {
	# export jobid=$(ezpz_get_slurm_running_jobid | tr "." " " | awk '{print $1}' | uniq)

	if [[ -n $(command -v sacct) ]]; then
		local jobid
		local nodelist
		jobid=$(ezpz_get_slurm_running_jobid)
		nodelist=$(scontrol show job "${jobid}" | grep -E "\ NodeList=" | tr ',' '\n' | tr '-' '\n' | tr -d 'NodeList=nid[]' | sed 's/^/nid/g;s/\ //g')
		echo "${nodelist}"
	fi
}

ezpz_make_slurm_nodefile() {
	if [[ "$#" == 1 ]]; then
		outfile="$1"
	else
		outfile="nodefile-$(ezpz_get_slurm_running_jobid)"
	fi
	ezpz_get_slurm_running_nodelist >"${outfile}"
	# nodelist="$(ezpz_get_slurm_running_nodelist)"
	# log_message INFO "Detected $(echo "${snodelist}" | wc -l) nodes, saving to: ${outfile}"
	# echo "${snodelist}" >"${outfile}"
	echo "${outfile}"
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
	if [[ -z "${CONDA_PREFIX:-}" ]] || [[ -z "${PYTHON_ROOT:-}" ]]; then
		module load frameworks
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
		log_message WARN "Caught CONDA_PREFIX=${CONDA_PREFIX} from environment, using this!"
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
		module load conda/2025-09-25
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
}

ezpz_install_and_setup_micromamba() {
	ezpz_install_micromamba
}

ezpz_setup_conda() {
	if command -v conda &>/dev/null; then
		log_message INFO "Conda version: $(conda --version)"
	fi
	if [[ -z "${CONDA_PREFIX:-}" ]]; then
		case "$(ezpz_get_machine_name)" in
		aurora*) ezpz_setup_conda_aurora ;;
		sunspot*) ezpz_setup_conda_sunspot ;;
		polaris*) ezpz_setup_conda_polaris ;;
		sirius*) ezpz_setup_conda_sirius ;;
		sophia*) ezpz_setup_conda_sophia ;;
		frontier*) ezpz_setup_conda_frontier ;;
		perlmutter*) log_message INFO "Skipping conda setup on Perlmutter" ;;
		*) log_message WARN "Unknown machine for conda setup: $(hostname)" && ezpz_install_micromamba ;;
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

# ezpz_ensure_uv() {
# 	curl -LsSf https://astral.sh/uv/install.sh | sh
# 	# # if ezpz_has uv; then
# 	# # 	return 0
# 	# # fi
# 	#    if ! command -v uv &>/dev/null; then
# 	#        log_message INFO "Installing uv..."
# 	#        curl -LsSf https://astral.sh/uv/install.sh | sh
# 	#        ezpz_install_uv
# 	#    fi
# }

ezpz_ensure_uv() {
	if ! command -v uv &>/dev/null; then
		log_message INFO "Installing uv..."
		ezpz_install_uv
	fi
}

# Function to setup (build + activate) a new `uv` virtual environment.
#
# - Usage:
#
#  `ezpz_build_new_uv_venv [<venv_dir>] [<python_version>]`
#
#   - Parameters:
#     - `<python_version>`: Optional Python version to use for the environment.
#       - If not specified, it defaults to the value of `${DEFAULT_PYTHON_VERSION:-3.12}`.
#     - `<venv_dir>`: Directory to create the virtual environment in.
#       - If not specified, it will be placed in the working directory.
#
# - Side Effects:
#  - Installs `uv` if not already installed.
#  - Prints status messages.
#  - Example:
#
#    ```bash
#    ezpz_build_new_uv_venv 3.10 ~/.venv
#    ```
#
#  - Relies on:
#    - `uv` command must be available or will be installed.
ezpz_link_dotvenv() {
	local target="$1"
	local force="${EZPZ_FORCE_DOTVENV:-0}"
	if [[ -z "${target:-}" ]]; then
		return
	fi
	# If the caller explicitly points to ".venv", there's nothing to link.
	if [[ "$(basename "${target}")" == ".venv" ]]; then
		return
	fi
	local wd="${WORKING_DIR:-$(pwd)}"
	local link="${wd}/.venv"
	if [[ -e "${link}" && "${force}" != "1" ]]; then
		log_message INFO "  - Skipping .venv link (already exists: ${CYAN}${link}${RESET}); set EZPZ_FORCE_DOTVENV=1 to repoint."
		return
	fi
	log_message INFO "  - Linking ${CYAN}${link}${RESET} -> ${CYAN}${target}${RESET}"
	ln -sfn "${target}" "${link}"
}

ezpz_setup_new_uv_venv() {
	ezpz_ensure_uv || return 1

	local py_version venv_dir fpactivate mn
	mn="$(ezpz_get_machine_name)"

	if [[ "$#" -eq 2 ]]; then
		py_version="$1"
		venv_dir="$2"
	elif [[ "$#" -eq 1 ]]; then
		py_version="$1"
		venv_dir=$(ezpz_get_venv_dir)
		# venv_dir="${WORKING_DIR:-$(pwd)}/venvs/${mn}/py${py_version}"
	else
		py_version="${DEFAULT_PYTHON_VERSION:-3.12}"
		venv_dir=$(ezpz_get_venv_dir)
		# venv_dir="${WORKING_DIR:-$(pwd)}/venvs/${mn}/py${py_version}"
	fi

	fpactivate="${venv_dir}/bin/activate"

	if [[ -f "${fpactivate}" ]]; then
		log_message INFO "  - venv already exists at: ${CYAN}${venv_dir}${RESET}"
		log_message INFO "  - Activating existing venv..."
		# shellcheck disable=SC1090
		source "${fpactivate}"
		return $?
	fi

	log_message INFO "  - Creating (new) venv in ${CYAN}${venv_dir}${RESET}..."
	mkdir -p "${venv_dir%/*}" 2>/dev/null || true
	uv venv --python="python${py_version}" --system-site-packages "${venv_dir}" || return 1
	ezpz_link_dotvenv "${venv_dir}"
	# shellcheck disable=SC1090
	source "${fpactivate}"
}

# Function to set up a Python virtual environment using `uv`.
#
# - Relies on:
#   - `uv` command must be available.
#   - `CONDA_PREFIX` environment variable must be set.
#   - `WORKING_DIR` environment variable must be set.
#   - `python3` command must exist and point to the Conda Python.
#
# - Usage:
#
#   `ezpz_setup_uv_venv`
#
# - Side Effects:
#   - Creates a virtual environment under "${WORKING_DIR}/venvs/".
#   - Activates the created virtual environment. Prints status messages.
#   - Returns 1 on failure.
#   - Exports CONDA_NAME, VENV_DIR.
ezpz_setup_uv_venv() {
	if ! command -v uv &>/dev/null; then
		echo "uv already installed. Skipping..."
	else
		echo "Installing uv..."
		ezpz_install_uv
	fi
	if [[ -z "${CONDA_PREFIX:-${PYTHON_ROOT:-${PYTHONUSERBASE}}}" ]]; then
		WORKING_DIR="${WORKING_DIR:-$(pwd)}"
		log_message ERROR "  - None of {CONDA_PREFIX, PYTHON_ROOT, PYTHONUSERBASE} are set."
		log_message WARN " - Creating (NEW!) venv without conda base. This may lead to unexpected behavior."
		if [[ "$#" -eq 1 ]]; then
			py_version="$1"
			log_message INFO "  - Using python version: ${CYAN}${py_version}${RESET}"
			CONDA_PREFIX="python${py_version}"
		else
			log_message INFO "  - Using default python version: ${CYAN}python${DEFAULT_PYTHON_VERSION:-3.12}${RESET}"
			CONDA_PREFIX="python${DEFAULT_PYTHON_VERSION:-3.12}"
		fi
		# return 1
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
	# local mn
	# local env_name
	# local ptmodstr
	# mn=$(ezpz_get_machine_name)
	# env_name=$(basename "${CONDA_PREFIX}")
	#
	# ptmodstr="$(module list 2>&1 | grep -E "py-torch" | awk '{print $NF}')"
	# if [[ -n "${ptmodstr}" ]]; then
	# 	env_name="${env_name}-pt$(basename "${ptmodstr}")"
	# fi
	#
	# VENV_DIR="${WORKING_DIR:-$(pwd)}/venvs/$(ezpz_get_machine_name)/${env_name}"
	venv_dir=$(ezpz_get_venv_dir)
	# fpactivate="${VENV_DIR}/bin/activate"
	fpactivate="${venv_dir}/bin/activate"
	if [[ ! -f "${fpactivate}" ]]; then
		log_message INFO "  - Creating venv in ${CYAN}${venv_dir}${RESET}..."
		uv venv --python="$(which python3)" --system-site-packages "${venv_dir}"
		ezpz_link_dotvenv "${venv_dir}"
	fi
	# shellcheck disable=SC1090
	[ -f "${fpactivate}" ] && log_message INFO "  - Activating: ${fpactivate}" && source "${fpactivate}"
}

ezpz_get_venv_dir() {
	local python_root
	python_root=$(ezpz_get_python_root)
	local wd
	wd=$(ezpz_get_working_dir)
	local wdname
	wdname="$(basename "$(ezpz_get_working_dir)")"
	local pyname
	local env_name
	env_name="${wdname}"
	if [[ -n "${python_root}" ]]; then
		log_message INFO "  - Found python root at: ${CYAN}${python_root}${RESET}"
		pyname="$(basename "${python_root}")"
		env_name="${env_name}-${pyname}"
	fi
	# else
	#     log_message WARN "  - Python root is not set. Using only working dir name for venv."
	# fi
	local ptmodstr
	if command -v module &>/dev/null; then
		ptmodstr="$(module list 2>&1 | grep -E "py-torch" | awk '{print $NF}')"
		if [[ -n "${ptmodstr}" ]]; then
			env_name="${env_name}-pt$(basename "${ptmodstr}")"
		fi
	fi
	# env_name=$(echo "${env_name}" | sed -E 's/python([0-9\.]+)/py\1/')
	echo "${wd}/venvs/$(ezpz_get_machine_name)/${env_name}"
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
ezpz_setup_venv_from_conda() {
	# local python_root="${CONDA_PREFIX:-${PYTHON_ROOT:-${PYTHONUSERBASE}}}"
	local python_root
	python_root=$(ezpz_get_python_root)
	if [[ -z "${python_root}" ]]; then
		log_message ERROR "  - Python root is not set. Cannot create venv."
		return 1
	fi
	local require_venv="${REQUIRE_VIRTUAL_ENV:-${REQUIRE_VENV:-1}}"
	if [[ "${require_venv}" -ne 1 ]]; then
		log_message INFO "  - NO_REQUIRE_VIRTUAL_ENV not set, not requiring VIRTUAL_ENV to be set."
		log_message INFO "  - Detected python root at: ${CYAN}${python_root}${RESET}"
		return 0
	else
		log_message INFO "  - Found python root at ${CYAN}${python_root}${RESET}"
		local wdname
		wdname="$(basename "$(ezpz_get_working_dir)")"
		local pyname
		pyname="$(basename "${python_root}")"
		local env_name
		env_name="${wdname}-${pyname}"
		local ptmodstr
		ptmodstr="$(module list 2>&1 | grep -E "py-torch" | awk '{print $NF}')"
		if [[ -n "${ptmodstr}" ]]; then
			env_name="${env_name}-pt$(basename "${ptmodstr}")"
		fi
		env_name=$(echo "${env_name}" | sed -E 's/python([0-9\.]+)/py\1/')
		# CONDA_NAME=$(basename "${python_root}") && export CONDA_NAME
		if [[ -z "${VIRTUAL_ENV:-}" ]]; then
			log_message INFO "  - No VIRTUAL_ENV found in environment!"
			# log_message INFO "Trying to setup venv from ${GREEN}${CYAN}${RESET}..."
			VENV_DIR="${WORKING_DIR}/venvs/$(ezpz_get_machine_name)/${env_name}"
			log_message INFO "  - Looking for venv in venvs/$(ezpz_get_machine_name)/${CYAN}${env_name}${RESET}..."
			local fpactivate
			fpactivate="${VENV_DIR}/bin/activate"
			# make directory if it doesn't exist
			[[ ! -d "${VENV_DIR}" ]] && mkdir -p "${VENV_DIR}"
			if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
				log_message INFO "  - Creating venv (on top of ${GREEN}${env_name}${RESET}) in ${VENV_DIR}..."
				if command -v uv >/dev/null 2>&1; then
					log_message INFO "  - Using uv for venv creation"
				else
					log_message INFO "  - uv not found, installing..."
					ezpz_install_uv
					# else
					# log_message INFO "  - Using python for venv creation"
					# python3 -m venv "${VENV_DIR}" --system-site-packages
				fi
				uv venv --python="$(which python3)" --system-site-packages "${VENV_DIR}"
				ezpz_link_dotvenv "${VENV_DIR}"
				# shellcheck disable=SC1090,SC1091
				source "${VENV_DIR}/bin/activate" || {
					log_message ERROR "  - Failed to source ${fpactivate} after creation."
					return 1
				}
			elif [[ -f "${VENV_DIR}/bin/activate" ]]; then
				log_message INFO "  - Activating existing venv in VENV_DIR=venvs/${CYAN}${env_name}${RESET}"
				# shellcheck disable=SC1090
				[ -f "${fpactivate}" ] && log_message INFO "  - Found ${fpactivate}" && source "${fpactivate}" && return 0
			else
				log_message ERROR "  - Unable to locate ${VENV_DIR}/bin/activate"
				return 1
			fi
		fi
	fi
	# fi

}

# -----------------------------------------------------------------------------
# @description Set up a standard Python `venv` on top of an active python build.
# Creates a venv named after the PYTHONUSERBASE environment in a central 'venvs' directory.
# Activates the created venv. Inherits system site packages.
#
# Note: Similar purpose to `ezpz_setup_uv_venv` but uses the built-in `venv` module.
#
# Relies on:
#   - `PYTHONUSERBASE` environment variable must be set.
#   - `WORKING_DIR` environment variable must be set.
#   - `python3` command must exist and point to the Conda Python.
#
# Usage:
#   ezpz_setup_venv_from_pythonuserbase
#
# Side Effects:
#   Creates a virtual environment under "${WORKING_DIR}/venvs/".
#   Activates the created virtual environment. Prints status messages.
#   Returns 1 on failure. Exports CONDA_NAME, VENV_DIR.
# -----------------------------------------------------------------------------
ezpz_setup_venv_from_pythonuserbase() {
	python_prefix="$(basename "${PYTHONUSERBASE:-}")"
	local wd="${WORKING_DIR:-$(python3 -c 'import os; print(os.getcwd())')}"
	if [[ -z "${python_prefix:-}" ]]; then
		log_message ERROR "  - python_prefix is not set. Cannot create venv."
		return 1
	else
		log_message INFO "  - Found python at ${CYAN}${python_prefix}${RESET}"
		PYTHON_NAME=$(basename "${python_prefix}") && export PYTHON_NAME
		if [[ -z "${VIRTUAL_ENV:-}" ]]; then
			log_message INFO "  - No VIRTUAL_ENV found in environment!"
			# log_message INFO "Trying to setup venv from ${GREEN}${CYAN}${RESET}..."
			VENV_DIR="${wd}/venvs/$(ezpz_get_machine_name)/${PYTHON_NAME}"
			log_message INFO "  - Looking for venv in venvs/$(ezpz_get_machine_name)/${CYAN}${PYTHON_NAME}${RESET}..."
			local fpactivate
			fpactivate="${VENV_DIR}/bin/activate"
			export VENV_DIR
			# make directory if it doesn't exist
			[[ ! -d "${VENV_DIR}" ]] && mkdir -p "${VENV_DIR}"
			if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
				log_message INFO "  - Creating venv (on top of ${GREEN}${PYTHON_NAME}${RESET}) in VENV_DIR..."
				python3 -m venv "${VENV_DIR}" --system-site-packages
				ezpz_link_dotvenv "${VENV_DIR}"
				# shellcheck disable=SC1090,SC1091
				source "${VENV_DIR}/bin/activate" || {
					log_message ERROR "  - Failed to source ${fpactivate} after creation."
					return 1
				}
			elif [[ -f "${VENV_DIR}/bin/activate" ]]; then
				log_message INFO "  - Activating existing venv in VENV_DIR=venvs/${CYAN}${PYTHON_NAME}${RESET}"
				# shellcheck disable=SC1090
				[ -f "${fpactivate}" ] && log_message INFO "  - Found ${fpactivate}" && source "${fpactivate}" && return 0
			else
				log_message ERROR "  - Unable to locate ${VENV_DIR}/bin/activate"
				return 1
			fi
		fi
	fi

}

ezpz_get_python_root() {
	local python_root
	python_root="${CONDA_PREFIX:-${PYTHON_ROOT:-${PYTHONUSERBASE:-${VIRTUAL_ENV:-}}}}"
	echo "${python_root}"
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
ezpz_setup_python_alcf() {
	log_message INFO "[${CYAN}PYTHON${RESET}]"
	local python_root
	python_root=$(ezpz_get_python_root)
	local virtual_env="${VIRTUAL_ENV:-}"

	# Scenario 1: Neither Conda nor venv active -> Setup Conda then venv
	if [[ -z "${python_root}" && -z "${virtual_env}" ]]; then
		log_message INFO "  - No conda_prefix OR pythonuserbase OR virtual_env found in environment. Setting up conda..."
		if ! ezpz_setup_conda; then
			log_message ERROR "  - ezpz_setup_conda failed."
			return 1
		fi
		# Re-check conda_prefix after setup attempt
		# conda_prefix="${CONDA_PREFIX:-}"
		python_root=$(ezpz_get_python_root)
		if [[ -z "${python_root}" ]]; then
			log_message ERROR "  - CONDA_PREFIX still not set after ezpz_setup_conda."
			return 1
		else
			# Now attempt to set up venv on top of the activated conda
			log_message INFO "  - Found Python at ${CYAN}${python_root}${RESET}"
			if ! ezpz_setup_venv_from_conda; then
				log_message ERROR "  - ezpz_setup_venv_from_conda failed."
				return 1
			fi
		fi

		# Scenario 2: Conda active, venv not active -> Setup venv
	elif [[ -n "${python_root}" && -z "${virtual_env}" ]]; then
		log_message INFO "  - Conda active, conda=${GREEN}${python_root}${RESET}..."
		log_message INFO "  - No virtual_env found in environment"
		# log_message INFO "Setting up venv from"
		if ! ezpz_setup_venv_from_conda; then
			log_message ERROR "  - ezpz_setup_venv_from_conda failed."
			return 1
		fi

		# Scenario 3: Venv active, Conda not active (less common/intended)
	elif [[ -n "${virtual_env}" && -z "${python_root}" ]]; then
		log_message INFO "  - No conda_prefix found."
		log_message INFO "  - Using virtual_env from: ${CYAN}${virtual_env}${RESET}"

		# Scenario 4: Both Conda and venv active
	elif [[ -n "${virtual_env}" && -n "${python_root}" ]]; then
		log_message INFO "  - Found both conda_prefix and virtual_env in environment."
		log_message INFO "  - Using conda from: ${GREEN}${python_root}${RESET}"
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

ezpz_load_python_modules_nersc() {
	module load cudatoolkit gcc-native pytorch cudnn nccl
	module load PrgEnv-nvidia mpich gcc-native cudatoolkit pytorch cudnn nccl
}

ezpz_get_python_prefix_nersc() {
	local pythonuserbase
	local python_prefix
	pythonuserbase=$(basename "${PYTHONUSERBASE:-}")
	python_prefix="${CONDA_PREFIX:-${python_prefix:-${pythonuserbase}}}"
	echo "${python_prefix}"
}

ezpz_setup_python_nersc() {
	# log_message INFO "${CYAN}[ezpz_setup_python]${RESET} Setting up Python environment for NERSC..."
	ezpz_load_python_modules_nersc
	local virtual_env
	local pythonuserbase
	local python_prefix
	virtual_env="${VIRTUAL_ENV:-}"
	python_prefix="$(ezpz_get_python_prefix_nersc)"
	log_message INFO "[${CYAN}PYTHON${RESET}]"
	# Scenario 1: Neither PYTHONUSERBASE nor venv active -> Setup Conda then venv
	if [[ -z "${python_prefix}" && -z "${virtual_env}" ]]; then
		log_message INFO "  - No python_prefix OR virtual_env found in environment. Setting up python..."
		if ! ezpz_load_python_modules_nersc; then
			log_message ERROR "  - ezpz_load_python_modules_nersc failed."
			return 1
		fi
		# Re-check python_prefix after setup attempt
		python_prefix="$(ezpz_get_python_prefix_nersc)"
		if [[ -z "${python_prefix}" ]]; then
			log_message ERROR "  - python_prefix still not set after ezpz_load_python_modules_nersc."
			return 1
		fi
		# Now attempt to set up venv on top of the activated pythonuserbase
		if ! ezpz_setup_venv_from_pythonuserbase; then
			log_message ERROR "  - ezpz_setup_venv_from_pythonuserbase failed."
			return 1
		fi
		# Scenario 2: python_prefix active, venv not active -> Setup venv
	elif [[ -n "${python_prefix}" && -z "${virtual_env}" ]]; then
		log_message INFO "  - Python prefix found python_prefix=${GREEN}${python_prefix}${RESET}..."
		log_message INFO "  - No virtual_env found in environment"
		if ! ezpz_setup_venv_from_pythonuserbase; then
			log_message ERROR "  - ezpz_setup_venv_from_pythonuserbase failed."
			return 1
		fi
		# Scenario 3: Venv active, python_prefix not active (less common/intended)
	elif [[ -n "${virtual_env}" && -z "${python_prefix}" ]]; then
		log_message INFO "  - No python_prefix found."
		log_message INFO "  - Using virtual_env from: ${CYAN}${virtual_env}${RESET}"
		# Scenario 4: Both python_prefix and venv active
	elif [[ -n "${virtual_env}" && -n "${python_prefix}" ]]; then
		log_message INFO "  - Found both python_prefix and virtual_env in environment."
		log_message INFO "  - Using python_prefix from: ${GREEN}${python_prefix}${RESET}"
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

ezpz_setup_python() {
	local venv_override="${1:-}"
	local scheduler_type
	scheduler_type=$(ezpz_get_scheduler_type)
	if [[ "${scheduler_type}" == "pbs" ]]; then
		ezpz_setup_python_alcf
		return 0
	elif [[ "${scheduler_type}" == "slurm" ]]; then
		ezpz_setup_python_nersc
		return 0
	else
		# if [[ "${scheduler_type}" == "unknown" ]]; then
		log_message WARN "  - Unable to determine scheduler type."
		log_message INFO "  - Using uv to create a virtual environment with PYTHON_VERSION=${PYTHON_VERSION:-${DEFAULT_PYTHON_VERSION:-3.12}}"
		if [[ -n "${venv_override}" ]]; then
			log_message INFO "  - Using venv override: ${CYAN}${venv_override}${RESET}"
			if ! ezpz_setup_new_uv_venv "${PYTHON_VERSION:-${DEFAULT_PYTHON_VERSION:-3.12}}" "${venv_override}"; then
				log_message ERROR "  - ezpz_setup_new_uv_venv failed."
				return 1
			else
				return 0
			fi
		fi
		if ! ezpz_setup_new_uv_venv "${PYTHON_VERSION:-${DEFAULT_PYTHON_VERSION:-3.12}}"; then
			log_message ERROR "  - ezpz_setup_new_uv_venv failed."
			return 1
		else
			return 0
		fi
	fi
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
	if [[ "${num_gpus_per_host}" -ge 0 ]]; then
		depth="$((num_cpus_per_host / num_gpus_per_host))"
		# depth=$((num_cpus_per_host / num_gpus_per_host))
	else
		log_message WARN "  - num_gpus_per_host is zero or negative, setting depth=1"
		depth=1
	fi

	scheduler_type=$(ezpz_get_scheduler_type)
	if [[ "${scheduler_type}" == "pbs" ]]; then
		if [[ "${mn}" == "sophia" ]]; then
			dist_launch_cmd="mpirun -n ${num_gpus} -N ${num_gpus_per_host} --hostfile ${hostfile} -x PATH -x LD_LIBRARY_PATH"
		else
			dist_launch_cmd="mpiexec --verbose --envall -n ${num_gpus} -ppn ${num_gpus_per_host} --hostfile ${hostfile}"
		fi
		if [[ "${mn}" == "aurora" || "${mn}" == "sunspot" ]]; then
			CPU_BIND="verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96"
			dist_launch_cmd="${dist_launch_cmd} --no-vni --cpu-bind=${CPU_BIND}"
		else
			dist_launch_cmd="${dist_launch_cmd} --cpu-bind=depth -d ${depth}"
		fi
		# dist_launch_cmd=$(ezpz_get_dist_launch_cmd "${hostfile}")
	elif [[ "${scheduler_type}" == "slurm" ]]; then
		# dist_launch_cmd="srun -N ${num_hosts} -n ${num_gpus} -l -u --verbose"
		dist_launch_cmd="srun -u --verbose -N${SLURM_NNODES} -n$((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
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
	log_message INFO "[${BLUE}ezpz_save_slurm_env${RESET}]"
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
		log_message INFO "  - hostfile: ${BLUE}${hostfile}${RESET}"
		log_message INFO "  - jobenv_file: ${BLUE}${jobenv_file}${RESET}"
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
		log_message INFO "  to calculate:"
		log_message INFO "      - num_hosts: ${BLUE}${num_hosts}${RESET}"
		log_message INFO "      - num_gpus_per_host: ${BLUE}${num_gpus_per_host}${RESET}"
		log_message INFO "      - num_gpus: ${BLUE}${num_gpus}${RESET}"
		log_message INFO "  - Setting DIST_LAUNCH and ezlaunch..."
		# printf "    to calculate:\n"
		# printf "      - num_hosts: ${BLUE}%s${RESET}\n" "${num_hosts}"
		# printf "      - num_gpus_per_host: ${BLUE}%s${RESET}\n" "${num_gpus_per_host}"
		# printf "      - num_gpus: ${BLUE}%s${RESET}\n" "${num_gpus}"
		export DIST_LAUNCH="${dist_launch_cmd}"
		export ezlaunch="${DIST_LAUNCH}"
		# printf "      - DIST_LAUNCH: ${BLUE}%s${RESET}\n" "${DIST_LAUNCH}"
	fi
	export HOSTFILE="${hostfile}"
	export JOBENV_FILE="${jobenv_file}"
	log_message INFO "      - HOSTFILE: ${BLUE}${HOSTFILE}${RESET}"
	log_message INFO "      - JOBENV_FILE: ${BLUE}${JOBENV_FILE}${RESET}"
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
			log_message INFO "  - Using hostfile: ${CYAN}${hostfile}${RESET}"
			log_message INFO "  - Found in environment:"
			if [[ -n "${HOSTFILE:-}" ]]; then
				log_message INFO "      - HOSTFILE: ${CYAN}${HOSTFILE}${RESET}"
			fi
			# if [[ "${hostfile}" != "${PBS_NODEFILE}" ]]; then
		elif [[ "$#" == 1 ]]; then
			log_message INFO "  - Caught ${CYAN}${#}${RESET} arguments"
			hostfile="$1"
			jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
			log_message INFO "  - Caught ${CYAN}${#}${RESET} arguments"
			log_message INFO "  - hostfile=${CYAN}${hostfile}${RESET}"
		elif [[ "$#" == 2 ]]; then
			hostfile="$1"
			jobenv_file="$2"
			log_message INFO "  - Caught ${CYAN}${#}${RESET} arguments"
			log_message INFO "      - hostfile=${CYAN}${hostfile}${RESET}"
			log_message INFO "      - jobenv_file=${CYAN}${jobenv_file}${RESET}"
		else
			echo "Expected exactly 0, 1, or 2 arguments, received: $#"
			return 1
		fi
		log_message INFO "      - Writing SLURM vars to: ${CYAN}${jobenv_file}${RESET}"
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
	local mn scheduler_type
	mn=$(ezpz_get_machine_name)
	scheduler_type=$(ezpz_get_scheduler_type)
	if [[ "${scheduler_type}" == "pbs" ]]; then
		ezpz_setup_host_pbs "$@"
	elif [[ "${scheduler_type}" == "slurm" ]]; then
		ezpz_setup_host_slurm "$@"
	else
		log_message ERROR "Unknown scheduler: ${scheduler_type} on ${mn}"
	fi
}

ezpz_print_hosts() {
	local hostfile
	local scheduler_type
	scheduler_type=$(ezpz_get_scheduler_type)
	log_message INFO "[${MAGENTA}HOSTS${RESET}]"
	if [[ "${scheduler_type}" == "pbs" ]]; then
		log_message INFO "  - Detected PBS Scheduler"
		# log_message INFO "[${MAGENTA}HOSTS${RESET}] - PBS Scheduler"
	elif [[ "${scheduler_type}" == "slurm" ]]; then
		log_message INFO "  - Detected SLURM Scheduler"
		# log_message INFO "[${MAGENTA}HOSTS${RESET}] - SLURM Scheduler"
		HOSTFILE="$(ezpz_make_slurm_nodefile)"
	else
		log_message INFO "[${MAGENTA}HOSTS${RESET}] - Unknown Scheduler"
	fi

	if [[ "$#" == 0 ]]; then
		hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}}"
	elif [[ "$#" == 1 ]]; then
		hostfile="$1"
	else
		# hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE}}}"
		hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}}"
	fi
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
	case "${mn}" in
	aurora* | polaris* | sunspot* | sirius* | sophia*)
		echo "pbs"
		;;
	frontier* | perlmutter*)
		echo "slurm"
		;;
	*)
		if [[ -n "${PBS_JOBID:-}" ]]; then
			echo "pbs"
		elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
			echo "slurm"
		else
			echo "unknown"
		fi
		;;
	esac
}

ezpz_write_job_info_slurm() {
	local hostfile
	hostfile="$(ezpz_make_slurm_nodefile)"
	local num_hosts
	num_hosts=$(wc -l <"${hostfile}")
	local num_gpus_per_host
	num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
	local num_gpus
	num_gpus=$((num_hosts * num_gpus_per_host))
	dist_launch_cmd="srun -N ${num_hosts} -n ${num_gpus} -u --verbose"
	HOSTS
}

ezpz_write_job_info() {
	if [[ "$#" == 0 ]]; then
		hostfile="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-$(ezpz_make_slurm_nodefile)}}}"
		jobenv_file="${JOBENV_FILE:-$(ezpz_get_jobenv_file)}"
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
		dist_launch_cmd=$(ezpz_get_dist_launch_cmd "${hostfile}")
	elif [[ "${scheduler_type}" == "slurm" ]]; then
		# dist_launch_cmd="srun -N ${num_hosts} -n ${num_gpus} -l -u --verbose"
		dist_launch_cmd="srun -u --verbose -N${num_hosts} -n${num_gpus}"
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
		ezpz_print_hosts "${hostfile}"
		log_message INFO "[${BRIGHT_BLUE}DIST_INFO${RESET}]"
		log_message INFO "  - HOSTFILE=${BRIGHT_BLUE}${hostfile}${RESET}"
		log_message INFO "  - NHOSTS=${BRIGHT_BLUE}${NHOSTS}${RESET}"
		log_message INFO "  - NGPU_PER_HOST=${BRIGHT_BLUE}${NGPU_PER_HOST}${RESET}"
		log_message INFO "  - NGPUS=${BRIGHT_BLUE}${NGPUS}${RESET}"
		if [[ -n "$(command -v launch)" ]]; then
			log_message INFO "[${GREEN}LAUNCH${RESET}]"
			log_message INFO "  - To launch across all available GPUs, use: '${GREEN}launch${RESET}'"
			log_message INFO "    ${GREEN}launch${RESET} = ${GREEN}${LAUNCH}${RESET}"
			log_message INFO "  - Run '${GREEN}which launch${RESET}' to ensure that the alias is set correctly"
		fi
	fi
}

ezpz_launch() {
	if [[ -v WORLD_SIZE ]]; then
		dlaunch="$(echo "${DIST_LAUNCH}" | sed "s/-n\ ${NGPUS}/-n\ ${WORLD_SIZE}/g")"
	else
		dlaunch="${DIST_LAUNCH}"
	fi
	_args=("${@}")
	log_message INFO "[yeet]:\n"
	log_message INFO "evaluating:\n${GREEN}${dlaunch}${RESET}"
	log_message INFO "with arguments:\n${BLUE}${_args[*]}${RESET}"
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
	local slurm_job_id
	local hostfile
	slurm_job_id="$(ezpz_get_slurm_running_jobid)"
	hostfile="$(ezpz_make_slurm_nodefile)"
	NHOSTS=$(wc -l <"${hostfile}")
	NGPU_PER_HOST=$(ezpz_get_num_gpus_per_host)
	JOBENV_FILE="${SLURM_ENV_FILE:-${HOME}/.slurmenv}"
	DIST_LAUNCH="srun -l -u --verbose -N${NHOSTS} -n$((NHOSTS * NGPU_PER_HOST))"
	ezlaunch="${DIST_LAUNCH}"
	export NHOSTS
	export NGPU_PER_HOST
	export JOBENV_FILE
	export DIST_LAUNCH
	export ezlaunch
	# if [[ -n "${SLURM_JOB_ID}" ]]; then
	#   export JOBENV_FILE="${SLURM_ENV_FILE}"
	#   # shellcheck source="${HOME}/.slurmenv"
	#   # shellcheck disable=SC1091
	#   [ -f "${HOME}/.slurmenv" ] && source "${HOME}/.slurmenv"
	#   # [ -f "${JOBENV_FILE}" ] && source "${JOBENV_FILE}"
	#   export DIST_LAUNCH="srun -u --verbose -N${NHOSTS} -n${NGPUS}"
	#   # export DIST_LAUNCH="srun --gpus ${NGPUS} --gpus-per-node ${NGPU_PER_HOST} -N ${NHOSTS} -n ${NGPUS} -u --verbose"
	#   export ezlaunch="${DIST_LAUNCH}"
	# else
	#   echo "Skipping ezpz_get_slurm_env() on $(hostname)"
	# fi
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
	ezpz_print_hosts "${hostfile}"
	log_message INFO "\n[${BRIGHT_BLUE}DIST INFO${RESET}]:"
	log_message INFO "  - NGPUS=${BRIGHT_BLUE}${num_gpus}${RESET}"
	log_message INFO "  - NHOSTS=${BRIGHT_BLUE}${num_hosts}${RESET}"
	log_message INFO "  - NGPU_PER_HOST=${BRIGHT_BLUE}${num_gpus_per_host}${RESET}"
	log_message INFO "  - HOSTFILE=${BRIGHT_BLUE}${hostfile}${RESET}"
	log_message INFO "  - LAUNCH=${BRIGHT_BLUE}${LAUNCH}${RESET}"
	log_message INFO "  - DIST_LAUNCH=${BRIGHT_BLUE}${DIST_LAUNCH}${RESET}"
	log_message INFO "[${GREEN}LAUNCH${RESET}]:"
	log_message INFO "  - To launch across all available GPUs, use:"
	log_message INFO "  '${GREEN}launch${RESET}' ( = ${GREEN}${LAUNCH}${RESET} )"
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
	log_message INFO "\n"
	log_message INFO "[🍋 ${YELLOW}ezpz/bin/utils.sh${RESET}]"
	log_message INFO "\n"
	log_message INFO "  - USER=${BLACK}${USER}${RESET}"
	log_message INFO "  - MACHINE=${BLACK}${mn}${RESET}"
	log_message INFO "  - HOST=${BLACK}${hn}${RESET}"
	log_message INFO "  - TSTAMP=${BLACK}$(ezpz_get_tstamp)${RESET}"
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

ezpz_setup_job_alcf() {
	local scheduler_type
	scheduler_type=$(ezpz_get_scheduler_type)
	if [[ "${scheduler_type}" != "pbs" ]]; then
		log_message ERROR "Unknown scheduler: ${scheduler_type} on ${mn}"
		return 1
	fi
	local _pbs_jobid
	_pbs_jobid=$(ezpz_get_pbs_jobid)
	export PBS_JOBID="${_pbs_jobid}"
	if [[ -n "${PBS_JOBID:-}" ]]; then
		log_message INFO "  - PBS_JOBID=${YELLOW}${PBS_JOBID}${RESET}"
	elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
		log_message INFO "  - SLURM_JOB_ID=${YELLOW}${SLURM_JOB_ID}${RESET}"
	else
		ezpz_get_pbs_jobid
		export PBS_JOBID="${_pbs_jobid}"
		log_message INFO "  - PBS_JOBID=${YELLOW}${PBS_JOBID}${RESET}"
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
				# HOSTFILE="$(ezpz_make_slurm_nodefile)"
				local _slurm_nodefile
				_slurm_nodefile=$(ezpz_make_slurm_nodefile)
				if [[ -f "${_slurm_nodefile}" ]]; then
					export HOSTFILE="${_slurm_nodefile}"
					ezpz_getjobenv_main "$@"
				else
					echo "[${mn}] @ [${hn}] No compute node found !!"
				fi
				# export HOSTFILE
				# export SLURM_NODELIST="${running_nodes}"
				# ezpz_getjobenv_main "$@"
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

ezpz_setup_job_slurm() {
	# if [[ -n "${SLURM_JOB_ID:-}" ]]; then
	#   ezpz_savejobenv_main "$@"
	# else
	local slurm_job_id
	slurm_job_id="$(ezpz_get_slurm_running_jobid)"
	# if [[ -z "${slurm_job_id}" ]]; then
	#   log_message ERROR "No running SLURM job ID found on $(hostname). Cannot setup job."
	#   return 1
	if [[ -n "${slurm_job_id}" ]]; then
		log_message INFO "  - SLURM_JOB_ID=${YELLOW}${SLURM_JOB_ID}${RESET}"
		local running_nodes
		running_nodes=$(ezpz_get_slurm_running_nodelist)
		local _hostfile
		_hostfile="$(ezpz_make_slurm_nodefile)"
		if [[ "$#" -ge 1 ]]; then
			ezpz_getjobenv_main "$@"
		else
			ezpz_getjobenv_main
		fi
	fi
}

ezpz_setup_job() {
	# local mn="$(ezpz_get_machine_name)"
	# local hn="$(hostname)"
	local scheduler_type
	scheduler_type=$(ezpz_get_scheduler_type)
	local mn
	mn="$(ezpz_get_machine_name)"
	local hn
	hn="$(hostname)"
	log_message INFO "[${YELLOW}JOB${RESET}]"
	log_message INFO "  - Setting up env for ${YELLOW}${USER}${RESET}"
	log_message INFO "  - Detected ${YELLOW}${scheduler_type}${RESET} scheduler"
	log_message INFO "  - Machine: ${YELLOW}${mn}${RESET}"
	log_message INFO "  - Hostname: ${YELLOW}${hn}${RESET}"
	if [[ "${scheduler_type}" == "pbs" ]]; then
		# local pbs_vars
		# pbs_vars="$(printenv | grep -i "PBS")"
		ezpz_setup_job_alcf "$@"
	elif [[ "${scheduler_type}" == "slurm" ]]; then
		# local slurm_vars
		# slurm_vars="$(printenv | grep -i "SLURM")"
		ezpz_setup_job_slurm "$@"
	else
		log_message ERROR "Unknown scheduler: ${scheduler_type} on ${mn}"
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
	log_message INFO "  - ${GREEN}[✓]${RESET} Finished [${BRIGHT_GREEN}ezpz_setup_python_pt_new_aurora${RESET}]"
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
# Args (optional):
#   $1: Path to virtual env (default: auto-chosen by ezpz_setup_python)
#   $2: Path to hostfile (default: auto-chosen by ezpz_setup_job)
# Outputs: Sets up Python & Job envs. Prints summaries. Returns 1 on failure.
# -----------------------------------------------------------------------------
ezpz_setup_env() {
	# Positional args:
	#   $1 (optional): venv path override (defaults to auto selection)
	#   $2 (optional): hostfile override (defaults to auto selection)
	local venv_override hostfile_override
	venv_override="${1:-}"
	hostfile_override="${2:-}"
	if [[ "$#" -gt 2 ]]; then
		log_message WARN "Ignoring extra arguments to ezpz_setup_env (expected at most 2)"
	fi

	if ! ezpz_check_working_dir; then
		log_message ERROR "Failed to set WORKING_DIR. Please check your environment."
	fi
	log_message INFO "[${BRIGHT_YELLOW}ezpz_setup_env${RESET}]..."
	if [[ -n "${venv_override}" ]]; then
		if ! ezpz_setup_python "${venv_override}"; then
			log_message ERROR "Python setup failed. Aborting."
			return 1
		fi
	else
		if ! ezpz_setup_python; then
			log_message ERROR "Python setup failed. Aborting."
			return 1
		fi
	fi

	if [[ -n "${hostfile_override}" ]]; then
		if ! ezpz_setup_job "${hostfile_override}"; then
			log_message ERROR "Job setup failed. Aborting."
			return 1
		fi
	else
		if ! ezpz_setup_job; then
			log_message ERROR "Job setup failed. Aborting."
			return 1
		fi
	fi
	log_message INFO "${GREEN}[✓]${RESET} Finished [${BRIGHT_YELLOW}ezpz_setup_env${RESET}]"
	return 0
}

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
	log_message INFO "[ezpz] Setting up Python environment\n"
	ezpz_setup_python || {
		log_message ERROR "Python setup failed. Aborting."
		return 1
	}

	log_message INFO "[ezpz] Setting up Job environment\n"
	ezpz_setup_job "$@" || {
		log_message ERROR "Job setup failed. Aborting."
		return 1
	}

	local target_env_path="${VIRTUAL_ENV:-${CONDA_PREFIX:-<unknown>}}"
	log_message INFO "[ezpz] Installing ezpz from GitHub into ${target_env_path}\n"

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
	# printf "[ezpz] ${GREEN}:check: Done!${RESET}\n"
}

# Function to generate CPU ranges
ezpz_generate_cpu_ranges() {
	local cores_physical_start=$1
	local cores_logical_start=$2
	local ranks_per_socket=$3
	local cpu_ranges=""
	if [[ "${ranks_per_socket}" -le 0 ]]; then
		echo "Error: ranks_per_socket must be greater than 0"
		return 1
	fi
	local cores_per_rank=$((cores_per_socket_physical / ranks_per_socket))

	if [ "$ranks_per_socket" -gt "$cores_per_socket_physical" ]; then
		local remaining_ranks=$((ranks_per_socket - cores_per_socket_physical))

		# Assign ranks to physical cores
		for ((rank = 0; rank < cores_per_socket_physical; rank++)); do
			local physical_core=$((cores_physical_start + rank))
			cpu_ranges+="$physical_core:"
		done

		# Assign remaining ranks to logical cores
		for ((rank = 0; rank < remaining_ranks; rank++)); do
			local logical_core=$((cores_logical_start + rank))
			cpu_ranges+="$logical_core:"
		done
	else
		for ((rank = 0; rank < ranks_per_socket; rank++)); do
			local physical_start=$((cores_physical_start + rank * cores_per_rank + shift_amount))
			local logical_start=$((cores_logical_start + rank * cores_per_rank + shift_amount))

			if [[ $cores_per_rank -gt 1 ]]; then
				local physical_end=$((physical_start + cores_per_rank - 1))
				local logical_end=$((logical_start + cores_per_rank - 1))
				cpu_ranges+="$physical_start-$physical_end,$logical_start-$logical_end:"
			else
				cpu_ranges+="$physical_start,$logical_start:"
			fi
		done
	fi

	echo "${cpu_ranges%:}"
}

ezpz_get_cpu_bind_aurora() {
	# Constants
	local cores_per_socket_physical=52
	local cores_per_socket_logical=52
	local sockets=2
	local total_physical_cores=$((cores_per_socket_physical * sockets))

	# Check if number of ranks per node is provided
	if [[ "$#" == 1 ]]; then
		ranks_per_node=$1
	elif [ "$#" -lt 1 ]; then
		ranks_per_node="$(ezpz_get_num_gpus_per_host)"
		# echo "Usage: $0 <ranks_per_node> [shift_amount]"
		# return 1
	fi

	shift_amount=${2:-0} # Default shift amount is 0 if not provided

	if [ "$ranks_per_node" -eq 1 ]; then
		cpu_bind_list="0-$((cores_per_socket_physical * sockets + cores_per_socket_logical * sockets - 1))"
		echo "--cpu-bind=verbose,list:$cpu_bind_list"
		return 0
	fi

	# Round up ranks_per_node to the next even number if it's odd.
	# If adjustment is made, warn the user to prevent confusion about resource allocation.
	local was_odd=0
	if [ $((ranks_per_node % 2)) -ne 0 ]; then
		# log_message "Warning: ranks_per_node ($((ranks_per_node))) is odd. Rounding up to $((ranks_per_node + 1)) to ensure even allocation."
		ranks_per_node=$((ranks_per_node + 1))
		was_odd=1
	fi

	# Calculate the maximum allowable shift based on the remaining cores
	local ranks_per_socket
	local max_shift
	ranks_per_socket=$((ranks_per_node / sockets))
	max_shift=$((cores_per_socket_physical % ranks_per_socket))

	# Check if the shift amount is greater than the max allowable shift
	if [ "$shift_amount" -gt "$max_shift" ]; then
		# N.B. Uncomment to throw error, otherwise shift_amount is silently set to zero
		#echo "Error: Shift amount ($shift_amount) is greater than the maximum allowable shift ($max_shift)."
		#exit 1
		shift_amount=0
	fi
	local cpu_ranges_socket0
	local cpu_ranges_socket1
	cpu_ranges_socket0=$(ezpz_generate_cpu_ranges 0 104 $ranks_per_socket)
	cpu_ranges_socket1=$(ezpz_generate_cpu_ranges 52 156 $ranks_per_socket)

	# Combine the CPU ranges for both sockets in the correct order
	local cpu_bind_list
	cpu_bind_list="${cpu_ranges_socket0}:${cpu_ranges_socket1}"

	# Conditionally trim the last group if ranks_per_node was originally odd
	if [ "$was_odd" -eq 1 ]; then
		cpu_bind_list="${cpu_bind_list%:*}"
	fi
	echo "--cpu-bind=verbose,list:$cpu_bind_list"
}

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

# A helper function to time the execution of a given command or function.
ezpz_timeit() {
	local start_time=${SECONDS}
	local cmd="$@"

	log_message INFO "Running: $cmd"
	# Execute the command passed as arguments
	"$@"

	# Check the exit status of the executed command
	# if [ $? -eq 0 ]; then
	if $? -eq 0; then
		local duration=$((SECONDS - start_time))
		log_message INFO "${GREEN}->${RESET} $(printf "['%s'] completed in %02d:%02d:%02d\n" "$cmd" $((duration / 3600)) $((duration % 3600 / 60)) $((duration % 60)))"
		# printf "-> '%s' completed in %02d:%02d:%02d\n" "$cmd" $(($duration/3600)) $(($duration%3600/60)) $(($duration%60))
	else
		echo "-> '%s' failed to execute." "$cmd"
		log_message "${RED}->${RESET} '${cmd}' failed to execute."
	fi
}

# ezpz_suite() {
#   ezpz-launch python3 -m ezpz.examples.vit --compile &&
#       ezpz-launch python3 -m ezpz.examples.fsdp &&
#       ezpz-launch python3 -m ezpz.examples.fsdp_tp --dataset=random --tp=2 --batch-size=2 --epochs=5 &&
#       ezpz-launch python3 -m ezpz.examples.diffusion --batch_size 1 --hf-dataset stanfordnlp/imdb &&
#       ezpz-launch python3 -m ezpz.examples.hf_trainer --streaming --dataset_name=eliplutchok/fineweb-small-sample --tokenizer_name meta-llama/Llama-3.2-1B --model_name_or_path meta-llama/Llama-3.2-1B --bf16=true --do_trai>
# }

# --- Main Execution Block (when sourced) ---
ezpz_check_working_dir_slurm() {
	# NOTE: [Scenario 1]
	# - If SLURM_SUBMIT_DIR is empty (not set) use $(pwd)
	local jobdir_name="SLURM_SUBMIT_DIR"
	if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
		# Set SLURM_SUBMIT_DIR as WORKING_DIR
		log_message WARN "${jobdir_name} is not set! Setting it to current working directory"
		log_message INFO "Exporting ${jobdir_name}=${GREEN}${WORKING_DIR}${RESET}"
		export SLURM_SUBMIT_DIR="${WORKING_DIR}"

		# NOTE: [Scenario 2]
		# - If SLURM_SUBMIT_DIR is set, check if it matches the current working directory
	elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
		if [[ "${WORKING_DIR}" != "${SLURM_SUBMIT_DIR:-}" ]]; then
			log_message WARN "Current working directory does not match SLURM_SUBMIT_DIR! This may cause issues with the job submission."
			log_message WARN "SLURM_SUBMIT_DIR" "$(printf "${RED}%s${RESET}" "${SLURM_SUBMIT_DIR}")"
			log_message WARN "WORKING_DIR" "$(printf "${GREEN}%s${RESET}" "${WORKING_DIR}")"
			log_message WARN "Exporting SLURM_SUBMIT_DIR=WORKING_DIR=$(printf "${BLUE}%s${RESET}" "${WORKING_DIR}") and continuing..."
			export SLURM_SUBMIT_DIR="${WORKING_DIR}"
		fi
	fi
}

ezpz_check_working_dir() {
	WORKING_DIR=$(ezpz_get_working_dir)
	export WORKING_DIR="${WORKING_DIR}"

	if [[ -d .git ]]; then
		GIT_COMMIT_HASH=$(git rev-parse HEAD) && export GIT_COMMIT_HASH
		GIT_BRANCH=$(git branch --show-current) && export GIT_BRANCH
	else
		log_message WARN "No .git directory found in WORKING_DIR (${GREEN}${WORKING_DIR}${RESET}). Skipping Git info export."
	fi

	scheduler_type=$(ezpz_get_scheduler_type)
	if [[ "${scheduler_type}" == "pbs" ]]; then
		log_message INFO "Detected PBS scheduler environment."
		ezpz_check_working_dir_pbs
	elif [[ "${scheduler_type}" == "slurm" ]]; then
		log_message INFO "Detected SLURM scheduler environment."
		ezpz_check_working_dir_slurm
	else
		log_message INFO "No PBS or SLURM scheduler environment detected."
		log_message INFO "Unable to detect PBS or SLURM working directory info..."
		log_message INFO "Using current working directory (${GREEN}${WORKING_DIR}${RESET}) as working directory..."
	fi

}

# -----------------------------------------------------------------------------
# Main logic executed when the script is sourced.
# Determines the working directory based on PBS_O_WORKDIR, SLURM_SUBMIT_DIR, or pwd.
# Exports WORKING_DIR.
# -----------------------------------------------------------------------------
ezpz_get_working_dir() {
	python3 -c 'import os; print(os.getcwd())'
}

ezpz_check_working_dir_pbs() {
	# NOTE: [Scenario 1]
	# - If PBS_O_WORKDIR is empty (not set) use $(pwd)
	local jobdir_name="PBS_O_WORKDIR"
	if [[ -z "${PBS_O_WORKDIR:-}" ]]; then
		# Set PBS_O_WORKDIR as WORKING_DIR
		log_message WARN "${jobdir_name} is not set! Setting it to current working directory"
		log_message INFO "Exporting ${jobdir_name}=${GREEN}${WORKING_DIR}${RESET}"
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
	fi
}

# --- Script Entry Point ---
if [[ -n "${EZPZ_CHECK_WORKING_DIR:-}" ]]; then
	if ! ezpz_check_working_dir; then
		log_message ERROR "Failed to set WORKING_DIR. Please check your environment."
	fi
fi

if [[ "${EZPZ_SHELL_TYPE}" == "zsh" ]]; then
	unsetopt KSH_ARRAYS
fi

# If DEBUG mode was enabled, turn off command tracing now that setup is done.
# if [[ -n "${DEBUG:-}" ]]; then
#   set -x
#   log_message WARN "DEBUG MODE IS ${RED}OFF${RESET}"
#   set +x
# fi
