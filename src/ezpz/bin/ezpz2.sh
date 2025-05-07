#!/bin/bash --login
# Modernized version of utils.sh
# @file utils_modern.sh
# @brief `ezpz` helper script with functions to make life ez.
# @description
#       This file provides multiple helper functions, all prefixed with "ezpz_"
#       - `ezpz_setup_job`
#       - `ezpz_setup_python`
#       - ...
#       It aims to follow modern Bash practices including error handling,
#       proper quoting, local variables, and clearer documentation.

# --- Strict Mode & Options ---
# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error when substituting.
# The return value of a pipeline is the status of the last command to exit
# with a non-zero status, or zero if no command exited with a non-zero status.
# set -euo pipefail

# Allow aliases to be expanded (needed for `launch` alias)
# shopt -s expand_aliases

# --- Color Codes ---
# Usage: printf "${RED}This is red text${RESET}\n"
RESET='\e[0m'
# BLACK='\e[1;30m' # Avoid black text
RED='\e[1;31m'
GREEN='\e[1;32m'
YELLOW='\e[1;33m'
BLUE='\e[1;34m'
MAGENTA='\e[1;35m'
CYAN='\e[1;36m'
BRIGHT_BLUE='\e[1;94m' # Added for emphasis
# WHITE='\e[1;37m' # Avoid white on light terminals

# --- Global Variables ---
# These are determined dynamically or expected from the environment
HOSTNAME="$(hostname)"
PBS_ENV_FILE="${HOME}/.pbsenv"
SLURM_ENV_FILE="${HOME}/.slurmenv"
WORKING_DIR="" # Determined in utils_main

# --- Debugging and No-Op ---
# Check if running in DEBUG=1 mode.
# Usage: DEBUG=1 source utils_modern.sh
if [[ -n "${DEBUG:-}" ]]; then
    printf "${RED}!! RUNNING IN DEBUG MODE !!${RESET}\n"
    set -x # Print command traces before executing command.
fi

# Print (but DO NOT EXECUTE !!) each command that would be run.
# Usage: NOOP=1 source utils_modern.sh
if [[ -v NOOP ]]; then
    printf "${YELLOW}!! RUNNING IN NOOP MODE (Dry Run) !!${RESET}\n"
    set -o noexec # Read commands but do not execute them.
fi


# --- Helper Functions ---


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
    printf "[%s][${RED}E${RESET}] - %s\n" "$(ezpz_get_tstamp)" "${args[*]}"
}

# -----------------------------------------------------------------------------
# Get the name of the current shell.
# Strips off directory prefix from the $SHELL environment variable.
#
# Usage:
#   local shell_name
#   shell_name=$(ezpz_get_shell_name)
#   printf "Current shell: %s\n" "${shell_name}"
#
# Outputs:
#   The base name of the shell (e.g., "bash", "zsh").
# -----------------------------------------------------------------------------
ezpz_get_shell_name() {
    basename "${SHELL:-/bin/sh}" # Provide a default if SHELL is unset
}

# -----------------------------------------------------------------------------
# Get the current timestamp in YYYY-MM-DD-HHMMSS format.
#
# Usage:
#   local timestamp
#   timestamp=$(ezpz_get_tstamp)
#   printf "Timestamp: %s\n" "${timestamp}"
#
# Outputs:
#   The current timestamp string.
# -----------------------------------------------------------------------------
ezpz_get_tstamp() {
    date "+%Y-%m-%d-%H%M%S"
}

# --- PBS Related Functions ---

# -----------------------------------------------------------------------------
# Prints information about running PBS jobs owned by the current user.
# Parses `qstat` output to show Job ID, elapsed time, and assigned nodes.
#
# Note: This function relies on the specific output format of `qstat -n1rw`
#       and uses `sed`, `tr`, `awk`, and `grep` for parsing. Changes in
#       `qstat` output might break this function. Consider using `qstat -f -F json`
#       or `qstat -x` (XML) and a proper parser (like `jq` or `xmlstarlet`)
#       if available and more robustness is needed.
#
# Usage:
#   ezpz_qsme_running
#
# Example Output:
#   <jobid0> <elapsed_time0> <node0> <node1> ...
#   <jobid1> <elapsed_time1> <nodeA> <nodeB> ...
#
# Outputs:
#   Lines describing running jobs, one job per line. Format: <jobid> <nodes...>
#   Returns 1 if qstat command is not found.
# -----------------------------------------------------------------------------
ezpz_qsme_running() {
    # Check if qstat exists
    if ! command -v qstat &> /dev/null; then
        # printf "${RED}Error: 'qstat' command not found. Cannot list PBS jobs.${RESET}\n" >&2
        # printf "${RED}Error: 'qstat' command not found. Cannot list PBS jobs.${RESET}\n" >&2
        log_error "Error: 'qstat' command not found. Cannot list PBS jobs."
        return 1
    fi

    # -u "${USER}": Filter for the current user.
    # -n1: Show nodes assigned to the job on the first line.
    # -r: Show running jobs.
    # -w: Wide format.
    qstat -u "${USER}" -n1rw | \
        sed -e 's/\/[0-9]*\*[^ ]*//g' | # Remove CPU/core counts like /8*16
        tr '+|.' '   ' |                # Replace '+', '|', '.' with spaces
        awk '{
               a = "";
               # Fields from 13 onwards are node names in this specific format
               for (i = 13; i <= NF; i++) {
                 a = a " " $i;
               }
               # Print JobID (field 1) and the concatenated node list
               print $1 a
             }' | \
        grep -vE 'aurora-pbs|Req|Job|-----' # Filter out headers/separators
}

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
    jobid=$(echo "${running_jobs_output}" | grep "^[0-9]" | grep -m 1 "${HOSTNAME}" | awk '{print $1}')
    echo "${jobid}"
}

# -----------------------------------------------------------------------------
# Unset all environment variables starting with PBS_, except PBS_O_WORKDIR.
# Preserves the original working directory where the job was submitted.
#
# Usage:
#   ezpz_reset_pbs_vars
#
# Side Effects:
#   Unsets environment variables matching PBS_*.
#   Exports PBS_O_WORKDIR with its original or current value.
# -----------------------------------------------------------------------------
ezpz_reset_pbs_vars() {
    local wd="${PBS_O_WORKDIR:-${WORKING_DIR:-$(pwd)}}" # Preserve original workdir or use current
    local var_name

    # Use parameter expansion with prefix matching (requires Bash 4+)
    if (( BASH_VERSINFO[0] >= 4 )); then
        for var_name in "${!PBS_@}"; do
            if [[ "${var_name}" != "PBS_O_WORKDIR" ]]; then
                 printf "Unsetting %s\n" "${var_name}" >&2 # Log to stderr
                 # log_info $(echo "Unsetting ${var_name}")
                 unset "${var_name}"
            fi
        done
    else
        # Fallback for older Bash versions using printenv
        log_warn "Using printenv fallback for unsetting PBS vars (Bash < 4)."
        local vars_to_unset
        # `printenv`: Prints environment variables.
        # `grep -E "^PBS"`: Filters for lines starting with PBS.
        # `grep -v "^PBS_O_WORKDIR="`: Excludes PBS_O_WORKDIR.
        # `cut -d'=' -f1`: Extracts the variable name before the first '='.
        vars_to_unset=$(printenv | grep -E "^PBS" | grep -v "^PBS_O_WORKDIR=" | cut -d'=' -f1)
        for var_name in ${vars_to_unset}; do # Use loop over potentially multi-line output
             printf "Unsetting %s\n" "${var_name}" >&2 # Log to stderr
             unset "${var_name}"
        done
    fi
    export PBS_O_WORKDIR="${wd}"
}

# -----------------------------------------------------------------------------
# Get the path to the PBS nodefile corresponding to the current job.
# Determines the job ID based on the current hostname and finds the
# corresponding file in `/var/spool/pbs/aux/`.
#
# Relies on:
#   - `ezpz_get_jobid_from_hostname`
#
# Usage:
#   local nodefile
#   nodefile=$(ezpz_get_pbs_nodefile_from_hostname)
#   if [[ -f "${nodefile}" ]]; then
#       export PBS_NODEFILE="${nodefile}"
#       # Extract job ID from filename (handle potential suffixes)
#       export PBS_JOBID=$(basename "${nodefile}" | cut -d. -f1)
#       printf "Found nodefile: %s\n" "${nodefile}"
#   fi
#
# Outputs:
#   The full path to the PBS nodefile if found.
#   Exports PBS_NODEFILE and PBS_JOBID if found.
#   Prints the nodefile path to stdout.
#   Returns 1 if job ID or nodefile cannot be found.
# -----------------------------------------------------------------------------
ezpz_get_pbs_nodefile_from_hostname() {
    local jobid
    local nodefile_path=""
    local pbs_aux_dir="/var/spool/pbs/aux" # Standard location

    if ! jobid=$(ezpz_get_jobid_from_hostname); then
        # printf "${YELLOW}Warning: Could not determine PBS Job ID for hostname '%s'. Cannot find nodefile.${RESET}\n" "${HOSTNAME}" >&2
        log_warn $(echo "Warning: Could not determine PBS Job ID for hostname ${HOSTNAME}. Cannot find nodefile.")
        return 1
    fi
     if [[ -z "${jobid}" ]]; then
         # ezpz_get_jobid_from_hostname succeeded but found no job for this host
         printf "Info: Host '%s' not found in any running PBS jobs.\n" "${HOSTNAME}" >&2
         return 1
     fi

    # Construct potential nodefile path
    # Note: PBS job IDs might have suffixes like '.server-name'. Get the base ID.
    local base_jobid="${jobid%%.*}" # Remove suffix starting from the first '.'
    nodefile_path="${pbs_aux_dir}/${base_jobid}"

    # Check if the specific file exists (most reliable)
    if [[ -f "${nodefile_path}" ]]; then
        export PBS_NODEFILE="${nodefile_path}"
        export PBS_JOBID="${base_jobid}" # Use the base job ID
        echo "${nodefile_path}"
        return 0
    fi

    # Fallback: Search for files STARTING with the base job ID in the name
    # This is less reliable than a direct path match. Use find instead of parsing ls.
    local found_file
    # Use find -print -quit to stop after the first match
    found_file=$(find "${pbs_aux_dir}" -maxdepth 1 -name "${base_jobid}*" -print -quit)

    if [[ -f "${found_file}" ]]; then
        printf "${YELLOW}Warning: Direct nodefile '%s' not found. Using matched file '%s'.${RESET}\n" "${nodefile_path}" "${found_file}" >&2
        export PBS_NODEFILE="${found_file}"
        # Extract job ID from the found filename, assuming it's the base name before any '.'
        export PBS_JOBID=$(basename "${found_file}" | cut -d. -f1)
        echo "${found_file}"
        return 0
    else
        # printf "${RED}Error: Could not find PBS nodefile for Job ID '%s' (base: '%s') in '%s'.${RESET}\n" "${jobid}" "${base_jobid}" "${pbs_aux_dir}" >&2
        log_error $(echo "Error: Could not find PBS nodefile for Job ID ${jobid} (base: ${base_jobid}) in ${pbs_aux_dir}.")
        return 1
    fi
}

# --- SLURM Related Functions ---

# -----------------------------------------------------------------------------
# Get the Job IDs of running SLURM jobs for the current user.
# Parses `sacct` output. Filters out job steps.
#
# Note: Relies on `sacct` command and its output format.
#
# Usage:
#   local running_jobids
#   running_jobids=$(ezpz_get_slurm_running_jobid)
#   printf "Running SLURM jobs:\n%s\n" "${running_jobids}"
#
# Outputs:
#   A newline-separated list of running SLURM Job IDs.
#   Returns 1 if `sacct` is not found.
# -----------------------------------------------------------------------------
ezpz_get_slurm_running_jobid() {
    if ! command -v sacct &> /dev/null; then
        log_error "Error: 'sacct' command not found. Cannot list SLURM jobs."
        # printf "${RED}Error: 'sacct' command not found. Cannot list SLURM jobs.${RESET}\n" >&2
        return 1
    fi

    # --format=JobID: Specify output columns
    # --user "${USER}": Filter by user
    # -s R: Filter by state RUNNING (R)
    # --noheader: Suppress header line
    # -P: Parseable output (uses '|' delimiter, easier than fixed width)
    # grep -Ev '\.(batch|extern)$': Filter out step IDs (.batch, .extern) using ERE
    # cut -d'|' -f1: Extract the first field (JobID)
    sacct --format=JobID -P --user "${USER}" -s R --noheader | \
        grep -Ev '\.(batch|extern)$' | \
        cut -d'|' -f1
}

# -----------------------------------------------------------------------------
# Get the node list of running SLURM jobs for the current user.
# Parses `sacct` output. Filters out job steps.
#
# Note: Relies on `sacct` command and its output format. The nodelist might
#       be compressed (e.g., "nid[001-004]").
#
# Usage:
#   local nodelist
#   nodelist=$(ezpz_get_slurm_running_nodelist)
#   printf "Nodes for running SLURM jobs:\n%s\n" "${nodelist}"
#
# Outputs:
#   A newline-separated list of nodelists for running jobs.
#   Returns 1 if `sacct` is not found.
# -----------------------------------------------------------------------------
ezpz_get_slurm_running_nodelist() {
     if ! command -v sacct &> /dev/null; then
        # printf "${RED}Error: 'sacct' command not found. Cannot list SLURM jobs.${RESET}\n" >&2
        log_error "Error: 'sacct' command not found. Cannot list SLURM jobs."
        return 1
    fi

    # Use parseable output format
    sacct --format=JobID,NodeList -P --user "${USER}" -s R --noheader | \
        grep -Ev '\.(batch|extern)$' | \
        cut -d'|' -f2 # Extract the second field (NodeList)
}

# -----------------------------------------------------------------------------
# Create a SLURM nodefile containing the expanded list of hostnames for the current job.
# Uses `scontrol show hostnames` on the nodelist obtained from the environment
# ($SLURM_NODELIST) or by querying the primary running job via `sacct`.
#
# Usage:
#   local nodefile
#   nodefile=$(ezpz_make_slurm_nodefile [output_filename])
#   # or rely on default:
#   nodefile=$(ezpz_make_slurm_nodefile)
#   export HOSTFILE="${nodefile}"
#
# Args:
#   $1 (optional): output_filename - The name for the generated nodefile.
#                                    Defaults to "nodefile".
#
# Outputs:
#   Creates the nodefile with one hostname per line.
#   Prints the path to the created nodefile to stdout.
#   Returns 1 if `scontrol` is not found, nodelist is empty, or command fails.
# -----------------------------------------------------------------------------
ezpz_make_slurm_nodefile() {
    local outfile="${1:-nodefile}" # Default filename is "nodefile"
    local snodelist=""

    # Use SLURM_NODELIST from env if available and non-empty
    if [[ -n "${SLURM_NODELIST:-}" ]]; then
        snodelist="${SLURM_NODELIST}"
    else
        # Otherwise, query running jobs and take the nodelist of the first one found
        printf "${YELLOW}Warning: SLURM_NODELIST not set. Querying running jobs...${RESET}\n" >&2
        local running_nodelists
        if ! running_nodelists=$(ezpz_get_slurm_running_nodelist); then
             return 1 # Propagate error
        fi
        # Take the first line of output
        snodelist=$(echo "${running_nodelists}" | head -n 1)
    fi

    if [[ -z "${snodelist}" ]]; then
         printf "${RED}Error: Could not determine SLURM nodelist.${RESET}\n" >&2
         return 1
    fi

    if ! command -v scontrol &> /dev/null; then
        printf "${RED}Error: 'scontrol' command not found. Cannot expand nodelist.${RESET}\n" >&2
        return 1
    fi

    # Expand the potentially compressed nodelist (e.g., "nid[001-003]")
    if scontrol show hostnames "${snodelist}" > "${outfile}"; then
        echo "${outfile}" # Print path to created file
    else
        printf "${RED}Error: 'scontrol show hostnames %s' failed.${RESET}\n" "${snodelist}" >&2
        # Clean up potentially empty/partial file
        rm -f "${outfile}"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Set up environment variables for running with `srun` on SLURM systems.
# Calculates NHOSTS, NGPUS_PER_HOST, NGPUS, and constructs the SRUN_EXEC command.
# Prioritizes SLURM environment variables, falls back to detection.
#
# Args:
#   $@ (optional): Arguments to pass to `ezpz_make_slurm_nodefile` (e.g., output filename).
#
# Relies on:
#   - SLURM environment variables (SLURM_NNODES, SLURM_GPUS_ON_NODE, SLURM_NODELIST) if available.
#   - `ezpz_make_slurm_nodefile` to generate HOSTFILE if needed.
#   - `ezpz_get_num_gpus_per_host` as a fallback for GPU count.
#   - `wc -l` to count hosts from file if needed.
#
# Exports:
#   NHOSTS: Number of nodes in the allocation.
#   NGPUS_PER_HOST: Number of GPUs per node.
#   HOSTFILE: Path to the nodefile.
#   NGPUS: Total number of GPUs in the allocation.
#   SRUN_EXEC: The base `srun` command string.
#
# Side effects:
#   Creates a nodefile via `ezpz_make_slurm_nodefile` if HOSTFILE is not set or found.
#   Prints status messages. Returns 1 on failure.
# -----------------------------------------------------------------------------
ezpz_setup_srun() {
    local node_count="${SLURM_NNODES:-}"
    local gpus_on_node="${SLURM_GPUS_ON_NODE:-}"
    local hostfile_path="${HOSTFILE:-}" # Use existing HOSTFILE if already set

    # Determine HOSTFILE path if not already set
    if [[ -z "${hostfile_path}" || ! -f "${hostfile_path}" ]]; then
        printf "HOSTFILE not set or invalid, attempting creation...\n" >&2
        if ! hostfile_path=$(ezpz_make_slurm_nodefile "$@"); then
             printf "${RED}Error: Failed to create HOSTFILE for srun setup.${RESET}\n" >&2
             return 1
        fi
        printf "Created HOSTFILE: %s\n" "${hostfile_path}" >&2
    fi
    export HOSTFILE="${hostfile_path}" # Ensure it's exported

    # Determine NHOSTS
    if [[ -z "${node_count}" ]]; then
        printf "${YELLOW}Warning: SLURM_NNODES not set. Counting hosts from %s.${RESET}\n" "${HOSTFILE}" >&2
        node_count=$(wc -l < "${HOSTFILE}")
        if ! [[ "${node_count}" =~ ^[0-9]+$ ]] || [[ "${node_count}" -lt 1 ]]; then
             printf "${RED}Error: Invalid host count '%s' from %s.${RESET}\n" "${node_count}" "${HOSTFILE}" >&2
             node_count=1 # Fallback
        fi
    fi
    export NHOSTS="${node_count}"

    # Determine NGPUS_PER_HOST
    if [[ -z "${gpus_on_node}" ]]; then
         printf "${YELLOW}Warning: SLURM_GPUS_ON_NODE not set. Detecting GPUs per host...${RESET}\n" >&2
         gpus_on_node=$(ezpz_get_num_gpus_per_host) # Detect based on machine type
    fi
    export NGPUS_PER_HOST="${gpus_on_node}"

    # Calculate total GPUs
    local total_gpus=$(( NHOSTS * NGPUS_PER_HOST ))
    export NGPUS="${total_gpus}"
    export WORLD_SIZE="${total_gpus}" # Set WORLD_SIZE too

    # Construct the srun command
    # -l: Label output lines with task ID.
    # -u: Unbuffered output.
    # --verbose: Increase verbosity.
    # -N: Number of nodes.
    # -n: Total number of tasks (usually == NGPUS for GPU jobs).
    # Using calculated NGPUS is safer than relying only on SLURM vars for -n
    export SRUN_EXEC="srun -l -u --verbose -N${NHOSTS} -n${NGPUS}"
    # Consider adding --exact if tasks must match nodes/gpus precisely
    # Consider adding GPU binding flags e.g. --gpus-per-task=1 --gpu-bind=closest

     printf "${CYAN}[ezpz_setup_srun]${RESET} Configuration:\n"
     printf "  NHOSTS: %s\n" "${NHOSTS}"
     printf "  NGPUS_PER_HOST: %s\n" "${NGPUS_PER_HOST}"
     printf "  NGPUS (WORLD_SIZE): %s\n" "${NGPUS}"
     printf "  HOSTFILE: %s\n" "${HOSTFILE}"
     printf "  SRUN_EXEC: %s\n" "${SRUN_EXEC}"
     return 0
}

# --- General Environment Setup ---

# -----------------------------------------------------------------------------
# Save current environment variables to a specified .env file.
# Creates the output directory if it doesn't exist. Excludes LS_COLORS.
#
# Usage:
#   ezpz_save_dotenv <output_directory>
#
# Args:
#   $1: output_directory - The directory where the `.env` file will be saved.
#
# Outputs:
#   Creates or overwrites the `.env` file in the specified directory.
#   Exports DOTENV_FILE variable with the path to the created file.
#   Prints status messages to stdout.
#   Returns 1 if the incorrect number of arguments is provided or write fails.
# -----------------------------------------------------------------------------
ezpz_save_dotenv() {
    if [[ "$#" -ne 1 ]]; then
        printf "${RED}[Error] %s: Expected one argument (outdir). Received: %d${RESET}\n" "${FUNCNAME[0]}" "$#" >&2
        return 1
    fi

    local outdir="$1"
    local dotenv_file="${outdir}/.env"

    # Create directory, fail if cannot
    if ! mkdir -p "${outdir}"; then
        printf "${RED}Error: Failed to create directory '%s'.${RESET}\n" "${outdir}" >&2
        return 1
    fi

    printf "Saving environment to %s\n" "${dotenv_file}"
    # Use printenv and grep -v to exclude LS_COLORS
    if ! printenv | grep -v "LS_COLORS" > "${dotenv_file}"; then
         printf "${RED}Error: Failed to write to '%s'.${RESET}\n" "${dotenv_file}" >&2
         return 1
    fi

    export DOTENV_FILE="${dotenv_file}"
    printf "Successfully saved environment to %s\n" "${dotenv_file}"
}

# -----------------------------------------------------------------------------
# Determine the machine name based on hostname patterns.
# Maps specific hostname patterns (e.g., x4*, aurora*) to canonical names
# (e.g., "aurora", "sunspot"). Uses 'polaris'/'sirius' distinction based on PBS_O_HOST.
#
# Usage:
#   local machine
#   machine=$(ezpz_get_machine_name)
#   printf "Running on machine: %s\n" "${machine}"
#
# Outputs:
#   The canonical machine name (lowercase) (e.g., "aurora", "sunspot",
#   "polaris", "sirius", "frontier", "perlmutter", "sophia").
#   Returns the full hostname if no pattern matches.
# -----------------------------------------------------------------------------
ezpz_get_machine_name() {
    local machine=""
    # Use a case statement for clarity and efficiency
    case "${HOSTNAME}" in
        x4*|aurora*)
            machine="aurora"
            ;;
        x1*|uan*) # Sunspot compute nodes (uan*) or testbeds (x1*)
            machine="sunspot"
            ;;
        sophia*)
             machine="sophia"
             ;;
        x3*|polaris*) # Polaris compute nodes (x3*) or login nodes
            # Distinguish between Polaris compute nodes and Sirius login nodes
            # PBS_O_HOST is set on jobs launched from login nodes
            if [[ "${PBS_O_HOST:-}" == sirius* ]]; then
                 machine="sirius"
            else
                 machine="polaris" # Assume compute node or direct login to polaris
            fi
            ;;
        frontier*)
             machine="frontier"
             ;;
        nid*|login*) # Perlmutter compute (nid*) or login nodes
             machine="perlmutter"
             ;;
        *) # Default to the full hostname if no match
             machine="${HOSTNAME}"
             printf "${YELLOW}Warning: Unrecognized hostname pattern '%s'. Using full hostname.${RESET}\n" "${HOSTNAME}" >&2
             ;;
    esac
    echo "${machine}"
}

# -----------------------------------------------------------------------------
# Check if a process is listening on TCP port 29500 and kill it if found.
# Used to prevent multiple conflicting processes (e.g., Torchrun/DeepSpeed).
#
# Note: Relies on `lsof`. The port 29500 seems specific to a particular setup.
#
# Usage:
#   ezpz_check_and_kill_if_running
#
# Side Effects:
#   Kills the process(es) found listening on TCP port 29500.
#   Prints messages to stdout indicating whether a process was found/killed.
#   Returns 1 if lsof is not found or kill fails.
# -----------------------------------------------------------------------------
ezpz_check_and_kill_if_running() {
    if ! command -v lsof &> /dev/null; then
        printf "${RED}Error: 'lsof' command not found. Cannot check for running process.${RESET}\n" >&2
        return 1
    fi

    # -i tcp:29500 : Look for processes using TCP port 29500
    # -t : Output only process IDs
    # -n : Do not resolve hostnames (faster)
    # -P : Do not resolve port names (faster)
    local running_pids
    # Use process substitution and mapfile for safer PID handling than simple command substitution
    mapfile -t running_pids < <(lsof -i tcp:29500 -t -n -P 2>/dev/null)

    if [[ "${#running_pids[@]}" -gt 0 ]]; then
        printf "${YELLOW}Found conflicting process(es) (PID(s): %s) on port 29500. Killing...${RESET}\n" "${running_pids[*]}"
        local kill_failed=0
        for pid in "${running_pids[@]}"; do
             if kill "${pid}"; then
                 printf "${GREEN}Successfully sent SIGTERM to PID %s.${RESET}\n" "${pid}"
                 # Optionally wait briefly and check if killed, send SIGKILL if needed
             else
                 printf "${RED}Error: Failed to send SIGTERM to PID %s.${RESET}\n" "${pid}" >&2
                 kill_failed=1
             fi
        done
        return ${kill_failed} # Return 0 if all kills succeeded, 1 otherwise
    else
        printf "${GREEN}No conflicting process found on port 29500. Continuing.${RESET}\n"
        return 0
    fi
}

# -----------------------------------------------------------------------------
# Set ALCF proxy environment variables.
#
# Usage:
#   ezpz_set_proxy_alcf
#
# Exports:
#   HTTP_PROXY, HTTPS_PROXY, http_proxy, https_proxy, ftp_proxy
# -----------------------------------------------------------------------------
ezpz_set_proxy_alcf() {
    export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
    export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
    export http_proxy="http://proxy.alcf.anl.gov:3128"
    export https_proxy="http://proxy.alcf.anl.gov:3128"
    export ftp_proxy="http://proxy.alcf.anl.gov:3128"
    printf "ALCF proxy variables set.\n"
}

# -----------------------------------------------------------------------------
# Save essential environment variables to `.deepspeed_env` for DeepSpeed.
# This file is often sourced by DeepSpeed runners on each rank.
#
# Usage:
#   ezpz_save_ds_env
#
# Outputs:
#   Creates or overwrites the `.deepspeed_env` file in the current directory.
#   Prints a status message. Returns 1 on failure to write.
# -----------------------------------------------------------------------------
ezpz_save_ds_env() {
    local ds_env_file=".deepspeed_env"
    printf "Saving {PATH, LD_LIBRARY_PATH, http{,s}_proxy, CFLAGS, PYTHONUSERBASE} to %s\n" "${ds_env_file}"
    # Use redirection group for cleaner output
    {
        echo "PATH=${PATH}"
        # Use parameter expansion to provide empty value if unset
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
        echo "http_proxy=${http_proxy:-}"
        echo "https_proxy=${https_proxy:-}"
        echo "CFLAGS=${CFLAGS:-}"
        echo "PYTHONUSERBASE=${PYTHONUSERBASE:-}"
    } > "${ds_env_file}" || {
        printf "${RED}Error: Failed to write to '%s'.${RESET}\n" "${ds_env_file}" >&2
        return 1
    }
}

# --- Machine-Specific Conda Setup Functions ---
# These functions load modules and potentially activate conda environments
# specific to each HPC system. They generally check if CONDA_PREFIX is
# already set and skip if it is. Hardcoded paths/module names may need
# updating as systems evolve.

ezpz_setup_conda_frontier() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        printf "Setting up Conda/ROCm environment on Frontier...\n"
        module load PrgEnv-gnu/8.5.0 # Specify version for reproducibility
        module load craype-accel-amd-gfx90a
        module load rocm # Load latest compatible ROCm
        # Use fully qualified path for micromamba activation
        # !! Adjust these paths for your specific Frontier setup !!
        local MMBIN="/path/to/your/micromamba/bin/micromamba"
        local MMENV="/lustre/orion/csc613/scratch/foremans/envs/micromamba/py3.10-torch2.2-rocm5.7"
        if [[ ! -x "${MMBIN}" ]]; then
             printf "${RED}Error: Micromamba executable not found at '%s'.${RESET}\n" "${MMBIN}" >&2
             return 1
        fi
        # Use 'shell init' and eval for robust activation
        eval "$(${MMBIN} shell init -s bash -p ${MMENV})" || {
            printf "${RED}Error: Failed to init micromamba shell for env '%s'.${RESET}\n" "${MMENV}" >&2
            return 1
        }
        micromamba activate "${MMENV}" || { # Activate by path
             printf "${RED}Error: Failed to activate micromamba environment '%s'.${RESET}\n" "${MMENV}" >&2
             return 1
        }
        printf "Activated: %s\n" "${MMENV}"
    else
        printf "CONDA_PREFIX ('%s') already set. Skipping Conda setup.\n" "${CONDA_PREFIX}"
    fi
}

ezpz_setup_conda_sunspot() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        printf "Setting up Conda/Level Zero environment on Sunspot...\n"
        # !! Adjust module path/version as needed for Sunspot !!
        module use /soft/preview-modulefiles/24.147.0 # Example
        module load frameworks/2024.05.15.001     # Example
        # Activate base conda env if needed after module load
        # if command -v conda &> /dev/null; then conda activate base; fi
    else
        printf "CONDA_PREFIX ('%s') already set. Skipping Conda setup.\n" "${CONDA_PREFIX}"
    fi
}

ezpz_setup_conda_aurora() {
    if [[ "${CONDA_SHLVL}" -gt 0 ]]; then
        printf "Conda already activated (level %d). Skipping setup.\n" "${CONDA_SHLVL}"
        return 0
    else
        printf "Setting up Conda/Level Zero environment on Aurora...\n"
        # !! Adjust module versions as needed for Aurora !!
        module load frameworks
        # Activate base conda env if needed after module load
        # if command -v conda &> /dev/null; then conda activate base; fi
    fi
}

ezpz_setup_conda_sirius() {
    if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
        printf "Setting up Micromamba environment on Sirius...\n"
        # !! Adjust paths for your Sirius setup !!
        export MAMBA_ROOT_PREFIX=/lus/tegu/projects/PolarisAT/foremans/micromamba
        local micromamba_exe="${MAMBA_ROOT_PREFIX}/bin/micromamba"
        local target_env="2024-04-23" # Adjust environment name

        if [[ ! -x "${micromamba_exe}" ]]; then
             printf "${RED}Error: Micromamba executable not found at '%s'.${RESET}\n" "${micromamba_exe}" >&2
             return 1
        fi
        # Activate the environment using shell init/hook
        eval "$(${micromamba_exe} shell hook -s bash)" # Use hook for current shell
        micromamba activate "${target_env}" || {
            printf "${RED}Error: Failed to activate micromamba environment '%s'.${RESET}\n" "${target_env}" >&2
            return 1
        }
         printf "Activated Micromamba env: %s\n" "${target_env}"
    else
        printf "CONDA_PREFIX ('%s') or VIRTUAL_ENV ('%s') already set. Skipping Conda setup.\n" "${CONDA_PREFIX:-<unset>}" "${VIRTUAL_ENV:-<unset>}"
    fi
}

ezpz_setup_conda_sophia() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        printf "Setting up Conda environment on Sophia...\n"
        module load conda || {
             printf "${RED}Error: Failed to load 'conda' module.${RESET}\n" >&2
             return 1
        }
        conda activate base || {
             printf "${RED}Error: Failed to activate conda 'base' environment.${RESET}\n" >&2
             return 1
        }
         printf "Loaded 'conda' module and activated 'base' environment.\n"
    else
        printf "CONDA_PREFIX ('%s') already set. Skipping Conda setup.\n" "${CONDA_PREFIX}"
    fi
}

ezpz_setup_conda_polaris() {
    # Original script had `unset MPICH_GPU_SUPPORT_ENABLED` - keep if necessary
    # unset MPICH_GPU_SUPPORT_ENABLED

    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        printf "Setting up Conda environment on Polaris...\n"
        module use /soft/modulefiles # Ensure standard modules are available
        module load conda || { # Load default conda module, specify version if needed
             printf "${RED}Error: Failed to load 'conda' module.${RESET}\n" >&2
             return 1
        }
        conda activate base || {
             printf "${RED}Error: Failed to activate conda 'base' environment.${RESET}\n" >&2
             return 1
        }
         printf "Loaded 'conda' module and activated 'base' environment.\n"
    else
        printf "CONDA_PREFIX ('%s') already set. Skipping Conda setup.\n" "${CONDA_PREFIX}"
    fi
}

# -----------------------------------------------------------------------------
# Generic Conda setup dispatcher.
# Calls the appropriate machine-specific conda setup function based on the
# output of `ezpz_get_machine_name`. Handles Perlmutter separately.
#
# Usage:
#   ezpz_setup_conda
#
# Relies on:
#   - `ezpz_get_machine_name`
#   - Machine-specific setup functions (ezpz_setup_conda_*)
#
# Side Effects:
#   Calls one of the machine-specific setup functions or handles Perlmutter/unknown hosts.
#   Prints status messages. Returns 1 on failure.
# -----------------------------------------------------------------------------
ezpz_setup_conda() {
    local machine_name
    machine_name=$(ezpz_get_machine_name)
    printf "Attempting Conda setup for machine: %s\n" "${machine_name}"
    local setup_func=""
    local setup_status=0

    case "${machine_name}" in
        aurora)     setup_func="ezpz_setup_conda_aurora" ;;
        sophia)     setup_func="ezpz_setup_conda_sophia" ;;
        sunspot)    setup_func="ezpz_setup_conda_sunspot" ;;
        polaris)
             # Check if on compute node (x3*) vs login (sirius handled separately)
             if [[ "${HOSTNAME}" == x3* ]]; then
                 setup_func="ezpz_setup_conda_polaris"
             else
                 # Assume login node, treat like Sirius
                 printf "${YELLOW}On Polaris login node (%s), using Sirius setup.${RESET}\n" "${HOSTNAME}"
                 setup_func="ezpz_setup_conda_sirius"
             fi
            ;;
         sirius) setup_func="ezpz_setup_conda_sirius" ;;
        frontier) setup_func="ezpz_setup_conda_frontier" ;;
        perlmutter)
             # Perlmutter specific setup
             printf "Setting up environment on NERSC Perlmutter...\n"
             if [[ -z "${CONDA_PREFIX:-}" ]]; then
                  # !! Adjust Perlmutter module load as needed !!
                  module load pytorch # Or specific version: pytorch/2.1.0-gpu
                  # Activate associated venv if needed
                  # local venv_path="/path/to/your/perlmutter/venv"
                  # if [[ -f "${venv_path}/bin/activate" ]]; then
                  #    source "${venv_path}/bin/activate" || { setup_status=1; }
                  #    printf "Activated venv: %s\n" "${venv_path}"
                  # fi
                  printf "Loaded Perlmutter modules.\n"
             else
                  printf "CONDA_PREFIX ('%s') already set. Skipping Perlmutter setup.\n" "${CONDA_PREFIX}"
             fi
             # Skip calling a function if setup is done here
             setup_func=""
            ;;
        *)
            printf "${RED}Error: Unknown machine '%s' (%s). Cannot automatically set up Conda.${RESET}\n" "${machine_name}" "${HOSTNAME}" >&2
             return 1
             ;;
    esac

    # Call the determined setup function if one was assigned
    if [[ -n "${setup_func}" ]]; then
        if ! "${setup_func}"; then
             printf "${RED}Error during ${setup_func}.${RESET}\n" >&2
             setup_status=1
        fi
    fi
    return ${setup_status}
}

# --- Python Virtual Environment Setup ---

# -----------------------------------------------------------------------------
# Install the 'uv' Python package manager/installer using pip.
# Assumes a Python environment (conda/venv) is already active. Uses proxy if set.
#
# Usage:
#   ezpz_install_uv
#
# Relies on:
#   - `python3 -m pip`
#   - Proxy environment variables (http_proxy, https_proxy) if needed.
#
# Side Effects:
#   Installs or upgrades `uv` into the current Python environment.
#   Prints installation messages. Returns 1 on failure.
# -----------------------------------------------------------------------------
ezpz_install_uv() {
    if command -v uv &> /dev/null; then
        printf "'uv' is already installed: $(command -v uv)\n"
        # Optionally upgrade: python3 -m pip install -U uv
        return 0
    fi

    printf "Attempting to install 'uv' using pip...\n"
    # Use the active python's pip
    if python3 -m pip install uv; then
         printf "${GREEN}'uv' installation successful.${RESET}\n"
         # Verify command is now available
         if ! command -v uv &> /dev/null; then
              printf "${YELLOW}Warning: 'uv' installed but command not found in PATH immediately. Check environment.${RESET}\n" >&2
              # Installation might be in ~/.local/bin, ensure it's in PATH
         fi
         return 0
    else
         printf "${RED}Error: 'uv' installation via pip failed.${RESET}\n" >&2
         return 1
    fi
}

# -----------------------------------------------------------------------------
# Set up a Python virtual environment using 'uv'.
# Creates a venv named after the base conda environment in a central 'venvs' directory.
# Uses the Python from the underlying Conda environment and inherits site packages.
#
# Note: Intended to create a venv *on top of* an existing Conda environment.
#
# Relies on:
#   - `uv` command (installs it if missing via `ezpz_install_uv`).
#   - `CONDA_PREFIX` environment variable must be set.
#   - `WORKING_DIR` environment variable must be set.
#   - `which python3`
#
# Usage:
#   ezpz_setup_uv_venv
#
# Side Effects:
#   Installs `uv` if not present.
#   Creates a virtual environment under "${WORKING_DIR}/venvs/".
#   Activates the created virtual environment.
#   Prints status messages. Returns 1 on failure.
# -----------------------------------------------------------------------------
ezpz_setup_uv_venv() {
    # Ensure uv is available
    ezpz_install_uv || return 1

    # Check prerequisites
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        printf "${RED}Error: CONDA_PREFIX is not set. Cannot determine base environment for uv venv.${RESET}\n" >&2
        return 1
    fi
     if [[ -z "${WORKING_DIR:-}" ]]; then
        printf "${RED}Error: WORKING_DIR is not set. Cannot determine where to create venvs.${RESET}\n" >&2
        return 1
    fi

    local python_executable
    if ! python_executable=$(which python3); then
         printf "${RED}Error: python3 not found in PATH.${RESET}\n" >&2
         return 1
    fi

    # Derive venv name from the base name of the conda prefix
    local conda_base_name
    conda_base_name=$(basename "${CONDA_PREFIX}")
    local venv_dir="${WORKING_DIR}/venvs/${conda_base_name}"

    printf "Setting up 'uv' virtual environment based on Conda env '%s'...\n" "${conda_base_name}"
    printf "Target venv directory: %s\n" "${venv_dir}"

    # Create venv using the python from the underlying conda env
    # --system-site-packages allows inheriting packages like mpi4py, torch
    if uv venv --python "${python_executable}" --system-site-packages "${venv_dir}"; then
         printf "Activating venv: %s\n" "${venv_dir}"
         # Use 'source' to activate in the current shell
         # Check if activate script exists before sourcing
         if [[ -f "${venv_dir}/bin/activate" ]]; then
              source "${venv_dir}/bin/activate" || {
                   printf "${RED}Error: Failed to source activate script '%s/bin/activate'.${RESET}\n" "${venv_dir}" >&2
                   return 1
              }
              printf "${GREEN}Successfully created and activated uv venv.${RESET}\n"
              return 0
         else
              printf "${RED}Error: Activate script not found at '%s/bin/activate' after venv creation.${RESET}\n" "${venv_dir}" >&2
              return 1
         fi
    else
         printf "${RED}Error: Failed to create uv venv at '%s'.${RESET}\n" "${venv_dir}" >&2
         return 1
    fi
}

# -----------------------------------------------------------------------------
# Set up a standard Python `venv` on top of an active Conda environment.
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
    # Check prerequisites
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        printf "${RED}Error: CONDA_PREFIX is not set. Cannot create venv.${RESET}\n" >&2
        return 1
    fi
     if [[ -z "${WORKING_DIR:-}" ]]; then
        printf "${RED}Error: WORKING_DIR is not set. Cannot determine where to create venvs.${RESET}\n" >&2
        return 1
    fi
    if ! command -v python3 &> /dev/null; then
         printf "${RED}Error: python3 command not found in PATH.${RESET}\n" >&2
         return 1
    fi

    printf "Found conda at: %s\n" "${CONDA_PREFIX}"
    local conda_name
    conda_name=$(basename "${CONDA_PREFIX}") # Get conda env name
    export CONDA_NAME="${conda_name}" # Export for potential use elsewhere

    # Check if already inside a venv
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        printf "Already inside a virtual environment: %s\n" "${VIRTUAL_ENV}"
        # Ensure VENV_DIR is set if we are already in one
        export VENV_DIR="${VIRTUAL_ENV}"
        return 0
    fi

    printf "No VIRTUAL_ENV found in environment!\n"
    printf "    - Setting up venv from Conda env '%s'\n" "${CONDA_NAME}"
    export VENV_DIR="${WORKING_DIR}/venvs/${CONDA_NAME}"
    printf "    - Using VENV_DIR=%s\n" "${VENV_DIR}"

    local activate_script="${VENV_DIR}/bin/activate"

    # Check if venv needs creation
    if [[ ! -f "${activate_script}" ]]; then
        printf "\n    - Creating new virtual env in %s\n" \
            "$(printf "${GREEN}%s${RESET}" "${VENV_DIR}")"

        if ! mkdir -p "${VENV_DIR}"; then
             printf "${RED}Error: Failed to create directory '%s'.${RESET}\n" "${VENV_DIR}" >&2
             return 1
        fi
        # Create venv using the current python3, inheriting system site packages
        if ! python3 -m venv "${VENV_DIR}" --system-site-packages; then
             printf "${RED}Error: Failed to create venv at '%s'.${RESET}\n" "${VENV_DIR}" >&2
             rm -rf "${VENV_DIR}" # Clean up partial venv
             return 1
        fi
         printf "    - Activating newly created venv...\n"
         # Source the activate script
         source "${activate_script}" || {
              printf "${RED}Error: Failed to source activate script '%s' after creation.${RESET}\n" "${activate_script}" >&2
              return 1
         }
         printf "${GREEN}Successfully created and activated venv.${RESET}\n"
         return 0
    else # Venv already exists
         printf "    - Found existing venv, activating from %s\n" "$(printf "${BLUE}%s${RESET}" "${VENV_DIR}")"
         source "${activate_script}" || {
              printf "${RED}Error: Failed to activate existing venv '%s'.${RESET}\n" "${VENV_DIR}" >&2
              return 1
         }
         printf "${GREEN}Successfully activated existing venv.${RESET}\n"
         return 0
    fi
}

# -----------------------------------------------------------------------------
# Main Python environment setup function.
# Orchestrates setting up Conda (via `ezpz_setup_conda`) and then a venv on top
# (via `ezpz_setup_venv_from_conda`). Exports PYTHON_EXEC.
#
# Usage:
#   ezpz_setup_python
#
# Relies on:
#   - `ezpz_setup_conda`
#   - `ezpz_setup_venv_from_conda` (or `ezpz_setup_uv_venv`)
#   - `CONDA_PREFIX`, `VIRTUAL_ENV` environment variables.
#   - `which python3`
#
# Side Effects:
#   Activates Conda and/or venv environments. Exports PYTHON_EXEC.
#   Prints status messages. Returns 1 on failure.
# -----------------------------------------------------------------------------
ezpz_setup_python() {
    local virtual_env="${VIRTUAL_ENV:-}"
    local conda_prefix="${CONDA_PREFIX:-}"
    local setup_status=0

    printf "${CYAN}[ezpz_setup_python]${RESET} Checking Python environment...\n"

    # Scenario 1: Neither Conda nor venv active -> Setup Conda then venv
    if [[ -z "${conda_prefix}" && -z "${virtual_env}" ]]; then
        printf "Neither conda nor venv active. Setting up conda...\n"
        if ! ezpz_setup_conda; then
             printf "${RED}Error: ezpz_setup_conda failed.${RESET}\n" >&2
             return 1
        fi
        # Re-check conda_prefix after setup attempt
        conda_prefix="${CONDA_PREFIX:-}"
        if [[ -z "${conda_prefix}" ]]; then
             printf "${RED}Error: CONDA_PREFIX still not set after ezpz_setup_conda.${RESET}\n" >&2
             return 1
        fi
         # Now attempt to set up venv on top of the activated conda
         printf "Setting up venv from conda '%s'...\n" "${conda_prefix}"
         if ! ezpz_setup_venv_from_conda; then
              printf "${RED}Error: ezpz_setup_venv_from_conda failed.${RESET}\n" >&2
              return 1
         fi

    # Scenario 2: Conda active, venv not active -> Setup venv
    elif [[ -n "${conda_prefix}" && -z "${virtual_env}" ]]; then
        printf "Conda active, venv inactive. Setting up venv from conda...\n"
        if ! ezpz_setup_venv_from_conda; then
             printf "${RED}Error: ezpz_setup_venv_from_conda failed.${RESET}\n" >&2
             return 1
        fi

    # Scenario 3: Venv active, Conda not active (less common/intended)
    elif [[ -z "${conda_prefix}" && -n "${virtual_env}" ]]; then
        printf "${YELLOW}Warning: Venv active at '%s' but no underlying Conda env detected.${RESET}\n" "${virtual_env}"
        # Proceed, assuming the venv is sufficient

    # Scenario 4: Both Conda and venv active
    elif [[ -n "${conda_prefix}" && -n "${virtual_env}" ]]; then
        printf "Venv active at '%s' on top of Conda at '%s'.\n" "${virtual_env}" "${conda_prefix}"
        # Environment seems correctly set up
    fi

    # Verify python3 is available and export path
    local python_exec
    if ! python_exec=$(which python3); then
         printf "${RED}Error: python3 command not found or not executable after setup.${RESET}\n" >&2
         return 1
    fi

    printf "${GREEN}[python] Using ${MAGENTA}%s${RESET}\n" "${python_exec}"
    export PYTHON_EXEC="${python_exec}"
    return 0
}

# --- Utility Functions ---

# -----------------------------------------------------------------------------
# Print the current working directory using Python.
# Useful if `pwd` behaves unexpectedly or for consistency.
#
# Usage:
#   local current_dir
#   current_dir=$(whereAmI)
# -----------------------------------------------------------------------------
whereAmI() {
    # Prefer python3 if available
    if command -v python3 &> /dev/null; then
        python3 -c 'import os; print(os.getcwd())' || {
             printf "${RED}Error executing Python for whereAmI. Falling back to pwd.${RESET}\n" >&2
             pwd -P # Use physical directory path
        }
    else
         pwd -P # Fallback if python3 not found
    fi
}

# -----------------------------------------------------------------------------
# Join command line arguments with a specified delimiter.
#
# Usage:
#   local joined_string
#   joined_string=$(join_by <delimiter> <arg1> <arg2> ...)
#
# Args:
#   $1: delimiter - The character or string to join arguments with.
#   $@: arguments - The arguments to join (starting from $2).
#
# Example:
#   join_by ',' foo bar baz  # Output: foo,bar,baz
#
# Outputs:
#   The joined string. Does not print a trailing newline.
# -----------------------------------------------------------------------------
join_by() {
    local d=${1-} f=${2-} # Delimiter and first element
    if shift 2; then      # Shift off the delimiter and the first element
        # Print first element, then print rest prefixed with delimiter
        printf %s "$f" "${@/#/$d}"
    elif [[ -n "$f" ]]; then # Handle case with only one element (and delimiter)
         printf %s "$f"
    fi
    # No trailing newline
}

# -----------------------------------------------------------------------------
# Get the number of Intel XPUs (PVC GPUs) available on the node.
# Uses Python with `intel_extension_for_pytorch`. Returns 0 on failure.
#
# Usage:
#   local num_xpus
#   num_xpus=$(ezpz_get_num_xpus)
#
# Relies on:
#   - `python3`
#   - `intel_extension_for_pytorch` Python package installed.
#
# Outputs:
#   The number of XPUs detected, or 0 if detection fails.
# -----------------------------------------------------------------------------
ezpz_get_num_xpus() {
    # Ensure python is available
    if ! command -v python3 &> /dev/null; then
         printf "${YELLOW}Warning: python3 not found. Cannot detect XPUs. Assuming 0.${RESET}\n" >&2
         echo 0
         return
    fi
    # Try to import and run; redirect stderr to avoid noise if ipex not installed
    python3 -c 'import intel_extension_for_pytorch as ipex; print(ipex.xpu.device_count())' 2>/dev/null || {
         # If python command fails (e.g., ipex not installed), return 0
         printf "${YELLOW}Warning: Failed to get XPU count via ipex (maybe not installed?). Assuming 0.${RESET}\n" >&2
         echo 0
    }
}

# -----------------------------------------------------------------------------
# Get the number of Nvidia GPUs available on the node.
# Tries `nvidia-smi -L | wc -l` first, falls back to `torch.cuda.device_count()`.
# Returns 0 on failure.
#
# Usage:
#   local num_gpus
#   num_gpus=$(ezpz_get_num_gpus_nvidia)
#   export NGPUS_PER_HOST="${num_gpus}" # Optionally export
#
# Relies on:
#   - `nvidia-smi` command (optional)
#   - `python3` and `torch` package (optional fallback)
#
# Outputs:
#   The number of Nvidia GPUs detected. Defaults to 0 if detection fails.
# -----------------------------------------------------------------------------
ezpz_get_num_gpus_nvidia() {
    local num_gpus=0
    if command -v nvidia-smi &> /dev/null; then
        # Count lines in nvidia-smi -L output
        num_gpus=$(nvidia-smi -L | wc -l)
    elif command -v python3 &> /dev/null; then
         # Fallback to torch, check cuda availability
         num_gpus=$(python3 -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null)
         # Handle potential Python errors gracefully
         num_gpus=${num_gpus:-0}
    else
         printf "${YELLOW}Warning: Cannot detect Nvidia GPUs (nvidia-smi and python3/torch unavailable). Assuming 0.${RESET}\n" >&2
    fi
    # Ensure it's a valid non-negative number
    if ! [[ "${num_gpus}" =~ ^[0-9]+$ ]]; then
         printf "${YELLOW}Warning: Detected Nvidia GPU count ('%s') is not a number. Assuming 0.${RESET}\n" "${num_gpus}" >&2
         num_gpus=0
    fi
    echo "${num_gpus}"
}

# -----------------------------------------------------------------------------
# Get the number of GPUs/Accelerators per host based on the machine type.
# Calls machine-specific detection functions (Nvidia, Intel XPU, AMD MI).
# Returns 0 on failure or if unknown.
#
# Usage:
#   local gpus_per_host
#   gpus_per_host=$(ezpz_get_num_gpus_per_host)
#   export NGPUS_PER_HOST="${gpus_per_host}" # Optionally export
#
# Relies on:
#   - `ezpz_get_machine_name`
#   - `ezpz_get_num_gpus_nvidia`
#   - `ezpz_get_num_xpus`
#   - `rocminfo` command (for AMD)
#
# Outputs:
#   The number of GPUs/accelerators per host. Defaults to 0.
# -----------------------------------------------------------------------------
ezpz_get_num_gpus_per_host() {
    local mn
    mn=$(ezpz_get_machine_name)
    local ngpu_per_host=0

    printf "Detecting GPUs/XPUs per host for machine '%s'...\n" "${mn}" >&2 # Log detection attempt

    case "${mn}" in
        aurora|sunspot) # Intel PVC
            # ngpu_per_host=$(ezpz_get_num_xpus) # Use ipex detection
             ngpu_per_host=12 # Keep hardcoded value if detection is unreliable/slow
             printf "Using hardcoded value for Intel PVC: %d\n" "${ngpu_per_host}" >&2
            ;;
        frontier) # AMD MI Instinct (gfx90a)
             if command -v rocminfo &> /dev/null; then
                 # Count lines containing 'gfx90a'
                 ngpu_per_host=$(rocminfo | grep -c 'gfx90a')
                 if [[ "${ngpu_per_host}" -eq 0 ]]; then
                      printf "${YELLOW}Warning: rocminfo found 0 'gfx90a' GPUs on Frontier. Check installation. Falling back to 8.${RESET}\n" >&2
                      ngpu_per_host=8 # Fallback to hardcoded value
                 fi
             else
                  printf "${YELLOW}Warning: 'rocminfo' not found on Frontier. Assuming 8 GPUs/node.${RESET}\n" >&2
                  ngpu_per_host=8 # Fallback to hardcoded value
             fi
            ;;
        polaris|sirius|perlmutter|sophia|*) # Assume Nvidia or fallback if unknown
            ngpu_per_host=$(ezpz_get_num_gpus_nvidia)
            ;;
    esac

    # Final validation
    if ! [[ "${ngpu_per_host}" =~ ^[0-9]+$ ]]; then
         printf "${YELLOW}Warning: Detected GPUs per host ('%s') is not a number. Defaulting to 0.${RESET}\n" "${ngpu_per_host}" >&2
         ngpu_per_host=0
    fi

    printf "Detected %d GPUs/XPUs per host.\n" "${ngpu_per_host}" >&2
    echo "${ngpu_per_host}"
}

# -----------------------------------------------------------------------------
# Get the number of hosts (nodes) in the current job allocation.
# Uses specified file, HOSTFILE, PBS_NODEFILE, NODEFILE, SLURM_NNODES, or defaults to 1.
# Returns 1 as minimum.
#
# Args:
#   $1 (optional): Path to a specific hostfile to use.
#
# Usage:
#   local num_nodes
#   num_nodes=$(ezpz_get_num_hosts [path/to/hostfile])
#   export NHOSTS="${num_nodes}" # Optionally export
#
# Relies on:
#   - Environment variables: HOSTFILE, PBS_NODEFILE, NODEFILE, SLURM_NNODES
#   - `wc -l`
#
# Outputs:
#   The number of hosts (minimum 1).
# -----------------------------------------------------------------------------
ezpz_get_num_hosts() {
    local hostfile_to_check="${1:-}"
    local nhosts=1 # Default to 1

    # Determine which file to check based on priority
    local effective_hostfile=""
    if [[ -n "${hostfile_to_check}" && -f "${hostfile_to_check}" ]]; then
         effective_hostfile="${hostfile_to_check}"
         printf "Using provided hostfile: %s\n" "${effective_hostfile}" >&2
    elif [[ -n "${HOSTFILE:-}" && -f "${HOSTFILE:-}" ]]; then
        effective_hostfile="${HOSTFILE}"
        printf "Using HOSTFILE env var: %s\n" "${effective_hostfile}" >&2
    elif [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE:-}" ]]; then
        effective_hostfile="${PBS_NODEFILE}"
        printf "Using PBS_NODEFILE env var: %s\n" "${effective_hostfile}" >&2
    elif [[ -n "${NODEFILE:-}" && -f "${NODEFILE:-}" ]]; then
        effective_hostfile="${NODEFILE}"
        printf "Using NODEFILE env var: %s\n" "${effective_hostfile}" >&2
    fi

    # Count lines if we found a valid file
    if [[ -n "${effective_hostfile}" ]]; then
         nhosts=$(wc -l < "${effective_hostfile}")
    # If no file, check SLURM variable
    elif [[ -n "${SLURM_NNODES:-}" ]]; then
        nhosts="${SLURM_NNODES}"
        printf "Using SLURM_NNODES env var: %s\n" "${nhosts}" >&2
    else
         printf "${YELLOW}Warning: Could not determine number of hosts from files or SLURM. Assuming 1.${RESET}\n" >&2
         nhosts=1
    fi

     # Ensure it's a valid number >= 1
    if ! [[ "${nhosts}" =~ ^[0-9]+$ ]] || [[ "${nhosts}" -lt 1 ]]; then
         printf "${YELLOW}Warning: Calculated host count ('%s') is invalid. Defaulting to 1.${RESET}\n" "${nhosts}" >&2
         nhosts=1
    fi

    echo "${nhosts}"
}

# -----------------------------------------------------------------------------
# Calculate the total number of GPUs/Accelerators across all hosts in the allocation.
# Multiplies NHOSTS by NGPUS_PER_HOST. Returns 0 on failure.
#
# Args:
#   $@ (optional): Arguments passed to `ezpz_get_num_hosts` (e.g., hostfile path).
#
# Usage:
#   local total_gpus
#   total_gpus=$(ezpz_get_num_gpus_total [path/to/hostfile])
#   export NGPUS="${total_gpus}" # Optionally export
#   export WORLD_SIZE="${total_gpus}" # Common alias
#
# Relies on:
#   - `ezpz_get_num_hosts`
#   - `ezpz_get_num_gpus_per_host`
#
# Outputs:
#   The total number of GPUs/accelerators. Defaults to 0.
# -----------------------------------------------------------------------------
ezpz_get_num_gpus_total() {
    local num_hosts
    local num_gpus_per_host
    # Pass hostfile argument, if any, to ezpz_get_num_hosts
    num_hosts=$(ezpz_get_num_hosts "$@")
    # ezpz_get_num_gpus_per_host doesn't depend on the hostfile arg
    num_gpus_per_host=$(ezpz_get_num_gpus_per_host)

    # Perform calculation only if both values are valid numbers
    local num_gpus=0
    if [[ "${num_hosts}" =~ ^[0-9]+$ && "${num_gpus_per_host}" =~ ^[0-9]+$ ]]; then
         num_gpus=$(( num_hosts * num_gpus_per_host ))
    else
         printf "${RED}Error: Invalid inputs for total GPU calculation (hosts='%s', gpus/host='%s').${RESET}\n" "${num_hosts}" "${num_gpus_per_host}" >&2
    fi

    echo "${num_gpus}"
}

# -----------------------------------------------------------------------------
# Determine the scheduler type (pbs or slurm) based on machine name or SLURM env vars.
#
# Usage:
#   local scheduler
#   scheduler=$(ezpz_get_scheduler_type)
#
# Relies on:
#   - `ezpz_get_machine_name`
#   - `SLURM_JOB_ID` environment variable.
#
# Outputs:
#   "pbs", "slurm", or an empty string if undetermined.
# -----------------------------------------------------------------------------
ezpz_get_scheduler_type() {
    local mn
    mn=$(ezpz_get_machine_name)
    local scheduler=""

    # First, check for active SLURM job ID - strongest indicator
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        scheduler="slurm"
    # Then, check for active PBS job ID
    elif [[ -n "${PBS_JOBID:-}" ]]; then # Use PBS_JOBID which might be set by ezpz_get_pbs_nodefile...
         scheduler="pbs"
    elif [[ -n "${PBS_JOBID:-}" ]]; then # Or PBS_JOBID directly from env
         scheduler="pbs"
    else
        # If no active job vars, guess based on machine type
        printf "${YELLOW}Warning: No active SLURM/PBS job vars detected. Guessing scheduler based on machine '%s'.${RESET}\n" "${mn}" >&2
        case "${mn}" in
            aurora|polaris|sunspot|sirius|sophia) scheduler="pbs" ;;
            frontier|perlmutter) scheduler="slurm" ;;
            *) scheduler="" ;; # Unknown
        esac
    fi

    if [[ -z "${scheduler}" ]]; then
         printf "${YELLOW}Warning: Cannot determine scheduler type for machine '%s'.${RESET}\n" "${mn}" >&2
    fi
    echo "${scheduler}"
}

# -----------------------------------------------------------------------------
# Determine the path to the job environment file (.pbsenv or .slurmenv).
# Uses scheduler type and default file locations in $HOME, unless JOBENV_FILE is set.
#
# Usage:
#   local jobenv_file
#   jobenv_file=$(ezpz_get_jobenv_file)
#   export JOBENV_FILE="${jobenv_file}" # Optionally export
#
# Relies on:
#   - `ezpz_get_scheduler_type`
#   - Global variables: PBS_ENV_FILE, SLURM_ENV_FILE
#   - Environment variable: JOBENV_FILE (uses if already set)
#
# Outputs:
#   The path to the appropriate job environment file (e.g., ~/.pbsenv).
#   Returns an empty string if the scheduler is unknown.
# -----------------------------------------------------------------------------
ezpz_get_jobenv_file() {
    # Return immediately if JOBENV_FILE is already explicitly set
    if [[ -n "${JOBENV_FILE:-}" ]]; then
        echo "${JOBENV_FILE}"
        return 0
    fi

    local scheduler
    scheduler=$(ezpz_get_scheduler_type)
    local jobenv_file=""

    case "${scheduler}" in
        pbs)   jobenv_file="${PBS_ENV_FILE}" ;;  # Use global default path
        slurm) jobenv_file="${SLURM_ENV_FILE}" ;; # Use global default path
        *)
             printf "${YELLOW}Warning: Unknown scheduler type. Cannot determine jobenv file path.${RESET}\n" >&2
             jobenv_file=""
             ;;
    esac
    echo "${jobenv_file}"
}

# --- Job Setup and Information ---

# -----------------------------------------------------------------------------
# Get the distributed launch command (mpiexec/mpirun/srun) based on machine/hostfile.
# Constructs the appropriate command line with necessary flags for process count,
# placement, and binding, prioritizing system specifics.
#
# Args:
#   $1: hostfile - Path to the hostfile. Required.
#
# Relies on:
#   - `ezpz_get_machine_name`
#   - `ezpz_get_num_hosts`, `ezpz_get_num_gpus_per_host`, `ezpz_get_num_gpus_total`
#   - `getconf _NPROCESSORS_ONLN` (for CPU core info, optional)
#   - `SRUN_EXEC` (if set by `ezpz_setup_srun` for Slurm systems)
#
# Outputs:
#   The full distributed launch command string.
#   Returns 1 if hostfile is missing or machine/scheduler is unknown.
# -----------------------------------------------------------------------------
ezpz_get_dist_launch_cmd() {
    if [[ "$#" -ne 1 ]]; then
        printf "${RED}Error: %s requires exactly one argument: hostfile${RESET}\n" "${FUNCNAME[0]}" >&2
        return 1
    fi
    local hf="$1"
     if [[ ! -f "${hf}" ]]; then
         printf "${RED}Error: Hostfile '%s' not found in %s.${RESET}\n" "${hf}" "${FUNCNAME[0]}" >&2
         return 1
    fi

    local mn
    mn=$(ezpz_get_machine_name)
    # Ensure counts are up-to-date based on the provided hostfile
    local num_hosts
    local num_gpus_per_host
    local num_gpus
    num_hosts=$(ezpz_get_num_hosts "${hf}")
    num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
    num_gpus=$(( num_hosts * num_gpus_per_host ))

    # If total GPUs is 0, cannot form a meaningful launch command
    if [[ "${num_gpus}" -le 0 ]]; then
         printf "${RED}Error: Calculated total GPUs is %d. Cannot form launch command.${RESET}\n" "${num_gpus}" >&2
         return 1
    fi

    # Calculate CPU binding depth (optional, improves performance)
    local depth=1 # Default binding depth
    if command -v getconf &> /dev/null; then
        local num_cores_per_host=0
        num_cores_per_host=$(getconf _NPROCESSORS_ONLN 2>/dev/null) || num_cores_per_host=0
        # Estimate physical cores (often half of logical processors if HT enabled)
        local num_cpus_per_host=$(( num_cores_per_host > 0 ? num_cores_per_host / 2 : 0 ))
        if [[ "${num_cpus_per_host}" -gt 0 && "${num_gpus_per_host}" -gt 0 ]]; then
             depth=$(( num_cpus_per_host / num_gpus_per_host ))
        fi
    fi
     # Ensure depth is at least 1
     depth=$(( depth > 0 ? depth : 1 ))
     printf "Using CPU binding depth: %d\n" "${depth}" >&2

    local dist_launch_cmd=""

    # Determine command based on machine/scheduler
    case "${mn}" in
        sophia)
            # Sophia specific MPI command (example, adjust as needed)
            dist_launch_cmd="mpirun -n ${num_gpus} -npernode ${num_gpus_per_host} --hostfile \"${hf}\" -x PATH -x LD_LIBRARY_PATH"
            ;;
        aurora|sunspot|polaris|sirius) # Systems using Cray MPICH (mpiexec)
             # Base command with process counts and hostfile
             dist_launch_cmd="mpiexec --verbose --envall -n ${num_gpus} --ppn ${num_gpus_per_host} --hostfile \"${hf}\""
             # Add binding options for performance
             dist_launch_cmd+=" --cpu-bind depth -d ${depth}"
             # Add Aurora specific flag if needed
             if [[ "${mn}" == "aurora" ]]; then
                 dist_launch_cmd+=" --no-vni" # Required for Aurora network interface
             fi
            ;;
        frontier|perlmutter) # Systems using Slurm (srun)
             # Use pre-constructed SRUN_EXEC if available from ezpz_setup_srun
             if [[ -n "${SRUN_EXEC:-}" ]]; then
                  dist_launch_cmd="${SRUN_EXEC}"
                  # We might need to update -n if WORLD_SIZE differs, but handle that in ezpz_launch function
             else
                  # Fallback construction if ezpz_setup_srun wasn't called (less ideal)
                  printf "${YELLOW}Warning: SRUN_EXEC not set. Constructing basic srun command.${RESET}\n" >&2
                  dist_launch_cmd="srun -l -u --verbose -N${num_hosts} -n${num_gpus}"
                  # Add GPU binding flags if appropriate for the system
                  # e.g., --gpus-per-task=1 --gpu-bind=closest
             fi
             ;;
        *)
             printf "${RED}Error: Unknown machine '%s'. Cannot determine distributed launch command.${RESET}\n" "${mn}" >&2
             return 1
             ;;
    esac

    echo "${dist_launch_cmd}"
}

# -----------------------------------------------------------------------------
# Save relevant PBS environment variables and calculated launch info to the jobenv file.
#
# Args:
#   $1 (optional): Path to the hostfile (defaults to $HOSTFILE or $PBS_NODEFILE).
#   $2 (optional): Path to the jobenv file (defaults based on `ezpz_get_jobenv_file`).
#
# Relies on:
#   - PBS environment variables (PBS_JOBID, PBS_NODEFILE, etc.)
#   - `ezpz_get_num_hosts`, `ezpz_get_num_gpus_per_host`, `ezpz_get_num_gpus_total`
#   - `ezpz_get_dist_launch_cmd`
#   - `env`, `grep`, `sed`
#
# Outputs:
#   Appends exports to the jobenv file. Exports DIST_LAUNCH, ezlaunch, HOSTFILE, JOBENV_FILE.
#   Prints status messages. Returns 1 on failure.
# -----------------------------------------------------------------------------
ezpz_save_pbs_env() {
    printf "\n${BLUE}[%s]${RESET}\n" "${FUNCNAME[0]}"
    # Determine hostfile and jobenv file paths
    local hostfile="${1:-${HOSTFILE:-${PBS_NODEFILE:-}}}"
    local jobenv_file="${2:-$(ezpz_get_jobenv_file)}"

    # Check if running under PBS (using PBS_JOBID as indicator)
    if [[ -z "${PBS_JOBID:-}" ]]; then
        printf "${YELLOW}Warning: PBS_JOBID not set. Not saving PBS environment.${RESET}\n" >&2
        return 0 # Not an error, just not applicable
    fi

    # Validate required files
    if [[ -z "${hostfile}" || ! -f "${hostfile}" ]]; then
         printf "${RED}Error: Hostfile not found or specified (%s). Cannot save PBS env.${RESET}\n" "${hostfile:-<not set>}" >&2
         return 1
    fi
     if [[ -z "${jobenv_file}" ]]; then
         printf "${RED}Error: Job environment file path not determined. Cannot save PBS env.${RESET}\n" >&2
         return 1
    fi

    printf "    • Using:\n"
    printf "        • hostfile: ${BLUE}%s${RESET}\n" "${hostfile}"
    printf "        • jobenv_file: ${BLUE}%s${RESET}\n" "${jobenv_file}"

    # Save PBS_* variables to jobenv file, converting PBS_* to export PBS_*
    # Clear the file first or append carefully? Append for now.
    local save_status=0
    if ! env | grep '^PBS' | sed 's/^PBS/export PBS/' >> "${jobenv_file}"; then
         printf "${RED}Error: Failed to write PBS variables to '%s'.${RESET}\n" "${jobenv_file}" >&2
         save_status=1 # Mark error but continue
    fi
    # Also explicitly save HOSTFILE variable if it was determined
    if [[ -n "${hostfile}" ]]; then
         echo "export HOSTFILE=\"${hostfile}\"" >> "${jobenv_file}"
    fi

    # Calculate and export distributed launch parameters
    local num_hosts
    local num_gpus_per_host
    local num_gpus
    local dist_launch_cmd

    num_hosts=$(ezpz_get_num_hosts "${hostfile}")
    num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
    num_gpus=$(ezpz_get_num_gpus_total "${hostfile}") # Recalculate total
    dist_launch_cmd=$(ezpz_get_dist_launch_cmd "${hostfile}") || return 1 # Fail if launch cmd fails

    printf "    Calculated:\n"
    printf "        • num_hosts: ${BLUE}%s${RESET}\n" "${num_hosts}"
    printf "        • num_gpus_per_host: ${BLUE}%s${RESET}\n" "${num_gpus_per_host}"
    printf "        • num_gpus (WORLD_SIZE): ${BLUE}%s${RESET}\n" "${num_gpus}"
    printf "        • DIST_LAUNCH: %s\n" "${dist_launch_cmd}" # No color for easier copy-paste

    # Append calculated values to jobenv file
    {
        echo "export NHOSTS=${num_hosts}"
        echo "export NGPUS_PER_HOST=${num_gpus_per_host}"
        echo "export NGPUS=${num_gpus}"
        echo "export WORLD_SIZE=${num_gpus}"
        # Quote command robustly for export line
        printf "export DIST_LAUNCH=%q\n" "${dist_launch_cmd}"
        printf "export ezlaunch=%q\n" "${dist_launch_cmd}" # Use same quoted command
    } >> "${jobenv_file}" || {
         printf "${RED}Error: Failed to write calculated variables to '%s'.${RESET}\n" "${jobenv_file}" >&2
         save_status=1
    }

    # Export variables into the current environment as well
    export NHOSTS="${num_hosts}"
    export NGPUS_PER_HOST="${num_gpus_per_host}"
    export NGPUS="${num_gpus}"
    export WORLD_SIZE="${num_gpus}"
    export DIST_LAUNCH="${dist_launch_cmd}"
    export ezlaunch="${dist_launch_cmd}" # Use variable instead of alias for exportability
    export HOSTFILE="${hostfile}"
    export JOBENV_FILE="${jobenv_file}"

    printf "    Set in current environment:\n"
    printf "        • HOSTFILE: ${BLUE}%s${RESET}\n" "${HOSTFILE}"
    printf "        • JOBENV_FILE: ${BLUE}%s${RESET}\n\n" "${JOBENV_FILE}"

    return ${save_status} # Return 0 if all writes succeeded, 1 otherwise
}

# -----------------------------------------------------------------------------
# Save relevant SLURM environment variables and calculated launch info to the jobenv file.
#
# Args:
#   $1 (optional): Path to the hostfile (defaults based on `ezpz_make_slurm_nodefile`).
#   $2 (optional): Path to the jobenv file (defaults based on `ezpz_get_jobenv_file`).
#
# Relies on:
#   - SLURM environment variables (SLURM_JOB_ID, SLURM_NNODES, etc.)
#   - `ezpz_make_slurm_nodefile`, `ezpz_setup_srun`
#   - `ezpz_get_dist_launch_cmd` (implicitly via `ezpz_setup_srun`)
#   - `env`, `grep`, `sed`
#
# Outputs:
#   Appends exports to the jobenv file. Exports DIST_LAUNCH, ezlaunch, HOSTFILE, JOBENV_FILE.
#   Prints status messages. Returns 1 on failure.
# -----------------------------------------------------------------------------
ezpz_save_slurm_env() {
    printf "\n${BLUE}[%s]${RESET}\n" "${FUNCNAME[0]}"
    # Check if running under SLURM
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        printf "${YELLOW}Warning: SLURM_JOB_ID not set. Not saving SLURM environment.${RESET}\n" >&2
        return 0 # Not an error, just not applicable
    fi

    # Determine hostfile and jobenv file paths
    # Create hostfile if it doesn't exist or isn't specified
    local hostfile="${1:-${HOSTFILE:-}}"
    if [[ -z "${hostfile}" || ! -f "${hostfile}" ]]; then
         printf "Hostfile not specified or found, attempting to create...\n" >&2
         if ! hostfile=$(ezpz_make_slurm_nodefile); then # Pass optional args if any were given? No, use default name.
              printf "${RED}Error: Failed to create SLURM nodefile. Cannot save SLURM env.${RESET}\n" >&2
              return 1
         fi
         printf "Created hostfile: %s\n" "${hostfile}" >&2
    fi

    local jobenv_file="${2:-$(ezpz_get_jobenv_file)}"
     if [[ -z "${jobenv_file}" ]]; then
         printf "${RED}Error: Job environment file path not determined. Cannot save SLURM env.${RESET}\n" >&2
         return 1
    fi

    printf "    • Using:\n"
    printf "        • hostfile: ${BLUE}%s${RESET}\n" "${hostfile}"
    printf "        • jobenv_file: ${BLUE}%s${RESET}\n" "${jobenv_file}"

    local save_status=0
    # Save SLURM_* variables to jobenv file
    if ! env | grep '^SLURM' | sed 's/^SLURM/export SLURM/' >> "${jobenv_file}"; then
         printf "${RED}Error: Failed to write SLURM variables to '%s'.${RESET}\n" "${jobenv_file}" >&2
         save_status=1
    fi
    # Also save HOSTFILE variable itself
     echo "export HOSTFILE=\"${hostfile}\"" >> "${jobenv_file}"

    # Calculate and export distributed launch parameters using ezpz_setup_srun
    if ! ezpz_setup_srun "${hostfile}"; then # Pass hostfile explicitly
         printf "${RED}Error: ezpz_setup_srun failed.${RESET}\n" >&2
         return 1
    fi
    # Get values from exported variables set by ezpz_setup_srun
    local num_hosts="${NHOSTS}"
    local num_gpus_per_host="${NGPUS_PER_HOST}"
    local num_gpus="${NGPUS}"
    local dist_launch_cmd="${SRUN_EXEC}" # Use the command from ezpz_setup_srun

    printf "    Calculated:\n"
    printf "        • num_hosts: ${BLUE}%s${RESET}\n" "${num_hosts}"
    printf "        • num_gpus_per_host: ${BLUE}%s${RESET}\n" "${num_gpus_per_host}"
    printf "        • num_gpus (WORLD_SIZE): ${BLUE}%s${RESET}\n" "${num_gpus}"
    printf "        • DIST_LAUNCH: %s\n" "${dist_launch_cmd}"

    # Append calculated values to jobenv file
    {
        echo "export NHOSTS=${num_hosts}"
        echo "export NGPUS_PER_HOST=${num_gpus_per_host}"
        echo "export NGPUS=${num_gpus}"
        echo "export WORLD_SIZE=${num_gpus}"
        printf "export DIST_LAUNCH=%q\n" "${dist_launch_cmd}"
        printf "export ezlaunch=%q\n" "${dist_launch_cmd}"
    } >> "${jobenv_file}" || {
         printf "${RED}Error: Failed to write calculated variables to '%s'.${RESET}\n" "${jobenv_file}" >&2
         save_status=1
    }

    # Export variables into the current environment
    # These should already be exported by ezpz_setup_srun, but ensure consistency
    export NHOSTS="${num_hosts}"
    export NGPUS_PER_HOST="${num_gpus_per_host}"
    export NGPUS="${num_gpus}"
    export WORLD_SIZE="${num_gpus}"
    export DIST_LAUNCH="${dist_launch_cmd}"
    export ezlaunch="${dist_launch_cmd}"
    export HOSTFILE="${hostfile}" # Ensure HOSTFILE is exported
    export JOBENV_FILE="${jobenv_file}"

    printf "    Set in current environment:\n"
    printf "        • HOSTFILE: ${BLUE}%s${RESET}\n" "${HOSTFILE}"
    printf "        • JOBENV_FILE: ${BLUE}%s${RESET}\n\n" "${JOBENV_FILE}"

    return ${save_status}
}

# -----------------------------------------------------------------------------
# Set up host-specific configurations based on the scheduler (SLURM).
# Calls `ezpz_save_slurm_env`.
#
# Args:
#   $@: Arguments passed directly to `ezpz_save_slurm_env` (hostfile, jobenv_file).
#
# Relies on: `ezpz_get_scheduler_type`, `ezpz_save_slurm_env`
# Outputs: Prints status messages. Returns status of `ezpz_save_slurm_env`.
# -----------------------------------------------------------------------------
ezpz_setup_host_slurm() {
    printf "[%s]\n" "${CYAN}${FUNCNAME[0]}${RESET}"
    local scheduler_type
    scheduler_type=$(ezpz_get_scheduler_type)

    if [[ "${scheduler_type}" == "slurm" ]]; then
        ezpz_save_slurm_env "$@" # Pass along any arguments
        return $?
    else
         printf "${YELLOW}Scheduler is '%s', not 'slurm'. Skipping SLURM host setup.${RESET}\n" "${scheduler_type:-<unknown>}" >&2
         return 0 # Not an error in this context
    fi
}

# -----------------------------------------------------------------------------
# Set up host-specific configurations based on the scheduler (PBS).
# Calls `ezpz_save_pbs_env`.
#
# Args:
#   $@: Arguments passed directly to `ezpz_save_pbs_env` (hostfile, jobenv_file).
#
# Relies on: `ezpz_get_scheduler_type`, `ezpz_save_pbs_env`
# Outputs: Prints status messages. Returns status of `ezpz_save_pbs_env`.
# -----------------------------------------------------------------------------
ezpz_setup_host_pbs() {
    printf "[%s]\n" "${CYAN}${FUNCNAME[0]}${RESET}"
    local scheduler_type
    scheduler_type=$(ezpz_get_scheduler_type)

    if [[ "${scheduler_type}" == "pbs" ]]; then
        ezpz_save_pbs_env "$@" # Pass along any arguments
        return $?
    else
         printf "${YELLOW}Scheduler is '%s', not 'pbs'. Skipping PBS host setup.${RESET}\n" "${scheduler_type:-<unknown>}" >&2
         return 0 # Not an error in this context
    fi
}

# -----------------------------------------------------------------------------
# Dispatcher function for host setup based on the detected scheduler.
# Calls either `ezpz_setup_host_pbs` or `ezpz_setup_host_slurm`.
#
# Args:
#   $@: Arguments passed to the underlying setup function (hostfile, jobenv_file).
#
# Relies on: `ezpz_get_scheduler_type`, `ezpz_setup_host_pbs`, `ezpz_setup_host_slurm`
# Outputs: Prints status messages. Returns status of the called setup function or 1.
# -----------------------------------------------------------------------------
ezpz_setup_host() {
    local scheduler_type
    scheduler_type=$(ezpz_get_scheduler_type)
    printf "${MAGENTA}---- Setting up Host Environment (Scheduler: %s) ----${RESET}\n" "${scheduler_type:-Unknown}"
    local setup_status=0

    case "${scheduler_type}" in
        pbs)   ezpz_setup_host_pbs "$@" || setup_status=$? ;;
        slurm) ezpz_setup_host_slurm "$@" || setup_status=$? ;;
        *)
            printf "${RED}Error: Unknown scheduler type '%s'. Cannot perform host setup.${RESET}\n" "${scheduler_type}" >&2
            return 1
            ;;
    esac
    return ${setup_status}
}

# -----------------------------------------------------------------------------
# Print the list of hosts from a given hostfile, numbered.
#
# Args:
#   $1 (optional): Path to the hostfile. Defaults to determining via environment
#                  variables or creating a SLURM nodefile.
#
# Usage:
#   ezpz_print_hosts [path/to/hostfile]
#
# Relies on: `ezpz_make_slurm_nodefile` (if needed), `wc -l`, `mapfile` (Bash 4+)
# Outputs: Prints numbered list of hosts. Returns 1 if hostfile cannot be found/created.
# -----------------------------------------------------------------------------
ezpz_print_hosts() {
    local hostfile_to_use="${1:-}"

    # Determine the hostfile if not provided
    if [[ -z "${hostfile_to_use}" ]]; then
        hostfile_to_use="${HOSTFILE:-${PBS_NODEFILE:-${NODEFILE:-}}}"
        # If still empty, and on Slurm, try to create one
        if [[ -z "${hostfile_to_use}" && -n "${SLURM_JOB_ID:-}" ]]; then
             printf "Hostfile not found, attempting to create from SLURM...\n" >&2
             if ! hostfile_to_use=$(ezpz_make_slurm_nodefile); then
                  printf "${RED}Error: Failed to create hostfile for printing.${RESET}\n" >&2
                  return 1
             fi
        fi
    fi

    if [[ -z "${hostfile_to_use}" || ! -f "${hostfile_to_use}" ]]; then
         printf "${RED}Error: Hostfile not found or specified ('%s'). Cannot print hosts.${RESET}\n" "${hostfile_to_use:-<unset>}" >&2
         return 1
    fi

    local counter=0
    local host_line
    # Use mapfile (Bash 4+) for safer reading than while read if possible
    if (( BASH_VERSINFO[0] >= 4 )); then
        local hosts_array=()
        mapfile -t hosts_array < "${hostfile_to_use}"
        if [[ "${#hosts_array[@]}" -eq 0 ]]; then
             printf "${YELLOW}Warning: Hostfile '%s' appears to be empty.${RESET}\n" "${hostfile_to_use}" >&2
             return 0
        fi
        for host_line in "${hosts_array[@]}"; do
            # Skip empty lines just in case
            [[ -n "${host_line}" ]] || continue
            printf "    • [host:${MAGENTA}%d${RESET}] - ${MAGENTA}%s${RESET}\n" "${counter}" "${host_line}"
            counter=$(( counter + 1 ))
        done
    else
        # Fallback for Bash < 4
        printf "${YELLOW}Warning: Using 'while read' loop for hostfile (Bash < 4).${RESET}\n" >&2
        while IFS= read -r host_line || [[ -n "$host_line" ]]; do
            [[ -n "${host_line}" ]] || continue
            printf "    • [host:${MAGENTA}%d${RESET}] - ${MAGENTA}%s${RESET}\n" "${counter}" "${host_line}"
            counter=$(( counter + 1 ))
        done < "${hostfile_to_use}"
         if [[ "${counter}" -eq 0 ]]; then
              printf "${YELLOW}Warning: Hostfile '%s' appears to be empty.${RESET}\n" "${hostfile_to_use}" >&2
         fi
    fi
    return 0
}

# -----------------------------------------------------------------------------
# Write comprehensive job information (hosts, GPUs, launch command) to the jobenv file.
# Also exports key variables and defines a helper `ezpz_launch` function.
#
# Args:
#   $1 (optional): Path to the hostfile.
#   $2 (optional): Path to the jobenv file.
#
# Relies on: Many `ezpz_get_*` functions, `ezpz_get_dist_launch_cmd`, `ezpz_print_hosts`.
# Outputs: Appends to jobenv file, exports vars, defines `ezpz_launch` func, prints summary.
#          Returns 1 on critical failure (missing hostfile, cannot get launch cmd).
# -----------------------------------------------------------------------------
ezpz_write_job_info() {
    local hostfile="${1:-${HOSTFILE:-}}" # Use existing HOSTFILE if set
    local jobenv_file="${2:-$(ezpz_get_jobenv_file)}"
    local save_status=0

    # Determine hostfile if not explicitly provided or set
    if [[ -z "${hostfile}" || ! -f "${hostfile}" ]]; then
        local scheduler_type
        scheduler_type=$(ezpz_get_scheduler_type)
        printf "Hostfile not set, determining based on scheduler (%s)...\n" "${scheduler_type:-<unknown>}" >&2
        case "${scheduler_type}" in
            pbs)   hostfile="${PBS_NODEFILE:-$(ezpz_get_pbs_nodefile_from_hostname || echo "")}" ;;
            slurm) hostfile=$(ezpz_make_slurm_nodefile || echo "") ;;
            *)     hostfile="" ;;
        esac
         if [[ -z "${hostfile}" || ! -f "${hostfile}" ]]; then
              printf "${RED}Error: Failed to determine or create hostfile. Cannot write job info.${RESET}\n" >&2
              return 1
         fi
         export HOSTFILE="${hostfile}" # Export the determined hostfile path
         printf "Determined HOSTFILE: %s\n" "${HOSTFILE}" >&2
    fi

    if [[ -z "${jobenv_file}" ]]; then
         printf "${RED}Error: Job environment file path not determined ('%s'). Cannot write job info.${RESET}\n" "${jobenv_file:-<unset>}" >&2
         return 1
    fi

    # Get job parameters
    local num_hosts
    local num_gpus_per_host
    local num_gpus
    local dist_launch_cmd

    num_hosts=$(ezpz_get_num_hosts "${hostfile}")
    num_gpus_per_host=$(ezpz_get_num_gpus_per_host)
    num_gpus=$(ezpz_get_num_gpus_total "${hostfile}") # Use hostfile arg
    dist_launch_cmd=$(ezpz_get_dist_launch_cmd "${hostfile}") || return 1 # Exit if cmd fails

    # Prepare host list variables
    local hosts_arr=() # Bash array
    local hosts_str="" # Comma-separated string
    if (( BASH_VERSINFO[0] >= 4 )); then
        mapfile -t hosts_arr < "${hostfile}" # Read hosts into array
        hosts_str=$(join_by ',' "${hosts_arr[@]}")
    else
         # Fallback for Bash < 4 (less robust for names with spaces)
         hosts_str=$(paste -sd, "${hostfile}")
    fi

    # Export variables needed by many frameworks/launchers
    export NHOSTS="${num_hosts}"
    export NGPUS_PER_HOST="${num_gpus_per_host}"
    export NGPUS="${num_gpus}"
    export WORLD_SIZE="${num_gpus}"
    export DIST_LAUNCH="${dist_launch_cmd}"
    export LAUNCH="${dist_launch_cmd}" # Alias
    export ezlaunch="${dist_launch_cmd}" # Another alias
    export HOSTS="${hosts_str}"
    # Exporting arrays requires specific syntax if needed by sub-processes, often avoided
    # If needed: declare -a HOSTS_ARR=("${hosts_arr[@]}") export HOSTS_ARR

    # Append exports to jobenv file, quoting carefully
    # Use printf %q for robust quoting
    {
        printf "export HOSTFILE=%q\n" "${hostfile}"
        printf "export NHOSTS=%q\n" "${NHOSTS}"
        printf "export NGPUS_PER_HOST=%q\n" "${NGPUS_PER_HOST}"
        printf "export NGPUS=%q\n" "${NGPUS}"
        printf "export WORLD_SIZE=%q\n" "${WORLD_SIZE}"
        printf "export DIST_LAUNCH=%q\n" "${DIST_LAUNCH}"
        printf "export LAUNCH=%q\n" "${LAUNCH}"
        printf "export ezlaunch=%q\n" "${ezlaunch}"
        printf "export HOSTS=%q\n" "${HOSTS}"
    } >> "${jobenv_file}" || {
         printf "${RED}Error writing exports to %s${RESET}\n" "${jobenv_file}" >&2
         save_status=1
    }


    # Print summary
    printf "[${MAGENTA}%s${RESET}]\n" "HOSTS"
    ezpz_print_hosts "${hostfile}" # Print numbered list
    printf "\n"
    printf "[${BRIGHT_BLUE}%s${RESET}]\n" "DIST INFO"
    printf "    • NGPUS (WORLD_SIZE)=${BRIGHT_BLUE}%s${RESET}\n" "${NGPUS}"
    printf "    • NHOSTS=${BRIGHT_BLUE}%s${RESET}\n" "${NHOSTS}"
    printf "    • NGPUS_PER_HOST=${BRIGHT_BLUE}%s${RESET}\n" "${NGPUS_PER_HOST}"
    printf "    • HOSTFILE=${BRIGHT_BLUE}%s${RESET}\n" "${hostfile}"
    printf "    • DIST_LAUNCH=${BRIGHT_BLUE}%s${RESET}\n" "${DIST_LAUNCH}"
    printf "\n"

    # Print launch helper info only if function/alias seems available
    if command -v ezpz_launch &> /dev/null || alias launch &> /dev/null; then
        printf "[${GREEN}%s${RESET}]:\n" "LAUNCH"
        printf "    • To launch across all available GPUs, use: ${GREEN}%s${RESET}\n" "ezpz_launch"
        printf "    • (Alias 'launch' also available in this shell)\n"
        printf "\n"
        printf "      ${GREEN}ezpz_launch${RESET} ≈ ${GREEN}%s${RESET}\n" "${LAUNCH}"
    fi
    printf "\n"
    return ${save_status}
}

# Define a helper function in the current shell for convenience
# This function dynamically adjusts the number of processes if WORLD_SIZE differs
ezpz_launch() {
    local effective_launch_cmd="${DIST_LAUNCH}"
    local calculated_ngpus="${NGPUS}" # Store original calculation
    local target_world_size="${WORLD_SIZE:-${calculated_ngpus}}" # Use WORLD_SIZE if set

    # Check if user explicitly set WORLD_SIZE differently from calculated NGPUS
    if [[ "${target_world_size}" -ne "${calculated_ngpus}" ]]; then
        printf "${YELLOW}Warning: WORLD_SIZE (%s) differs from calculated NGPUS (%s). Adjusting launch command...${RESET}\n" "${target_world_size}" "${calculated_ngpus}" >&2
        # Attempt to replace the process count (-n <num>) in the launch command
        # This is fragile and depends heavily on the command format.
        # Handle common cases: ' -n <num> ', ' -n<num> ', '--processes <num>' etc.
        # Using extended regex for flexibility
        if [[ "${effective_launch_cmd}" =~ (-n[[:space:]]+)[0-9]+ ]]; then
                effective_launch_cmd="${effective_launch_cmd/${BASH_REMATCH[0]}/${BASH_REMATCH[1]}${target_world_size}}"
        elif [[ "${effective_launch_cmd}" =~ (--processes[[:space:]]+)[0-9]+ ]]; then
                effective_launch_cmd="${effective_launch_cmd/${BASH_REMATCH[0]}/${BASH_REMATCH[1]}${target_world_size}}"
        else
                printf "${YELLOW}Warning: Could not automatically adjust process count in launch command for WORLD_SIZE override.${RESET}\n" >&2
        fi
    fi

    local _args=("${@}") # Capture arguments passed to ezpz_launch
    printf "${GREEN}[ezpz_launch]${RESET}:\n"
    printf "  Executing: ${YELLOW}%s${RESET}\n" "${effective_launch_cmd}"
    printf "  With args: ${BLUE}%s${RESET}\n" "${_args[*]:-<none>}"
    # Use eval to execute the command string with arguments
    eval "${effective_launch_cmd}" "${_args[@]}"
}

# Export the function so it's available in subshells (Bash specific)
export -f ezpz_launch

# # Define a simple alias 'launch' pointing to the function
# # Note: Aliases are not exported to subshells by default. Use the function name.
# alias launch='ezpz_launch'


# -----------------------------------------------------------------------------
# Save environment variables specifically needed by DeepSpeed to `.deepspeed_env`.
# DEPRECATED - Use `ezpz_save_ds_env`. Kept for compatibility.
# -----------------------------------------------------------------------------
ezpz_save_deepspeed_env() {
    printf "${YELLOW}Warning: %s is deprecated. Use ezpz_save_ds_env.${RESET}\n" "${FUNCNAME[0]}" >&2
    ezpz_save_ds_env
}

# --- Job Environment Getters (Simplified) ---

# -----------------------------------------------------------------------------
# Set up environment if running under PBS by sourcing the jobenv file.
# Assumes `ezpz_save_pbs_env` or `ezpz_setup_host` was run previously to create the file.
#
# Args:
#   $1 (optional): Path to the jobenv file (defaults based on `ezpz_get_jobenv_file`).
#
# Relies on: `ezpz_get_jobenv_file`
# Outputs: Sources the jobenv file. Returns 1 if file not found.
# -----------------------------------------------------------------------------
ezpz_get_pbs_env() {
    local jobenv_file="${1:-$(ezpz_get_jobenv_file)}"
    if [[ -z "${jobenv_file}" ]]; then
         printf "${RED}Error [get_pbs_env]: Job environment file path not determined.${RESET}\n" >&2
         return 1
    fi

    if [[ -f "${jobenv_file}" ]]; then
        printf "\n${BLUE}[ezpz_get_pbs_env]${RESET}: Sourcing ${BLUE}%s${RESET}\n" "${jobenv_file}"
        # shellcheck source=/dev/null # Tell shellcheck we are intentionally sourcing a dynamic file
        source "${jobenv_file}" || {
             printf "${RED}Error sourcing '%s'.${RESET}\n" "${jobenv_file}" >&2
             return 1
        }
        # Define launch alias/function if DIST_LAUNCH was sourced
        if [[ -n "${DIST_LAUNCH:-}" ]]; then
             export ezlaunch="${DIST_LAUNCH}" # Ensure function var is exported
             export -f ezpz_launch # Re-export function if needed
             alias launch='ezpz_launch'
             printf "  ✓ Sourced env and defined launch helpers.\n"
        else
             printf "${YELLOW}Warning: Sourced '%s', but DIST_LAUNCH not defined.${RESET}\n" "${jobenv_file}" >&2
        fi
        return 0
    else
         printf "${YELLOW}Warning: Jobenv file '%s' not found. Cannot source PBS env.${RESET}\n" "${jobenv_file}" >&2
         return 1 # Return error if file doesn't exist
    fi
}

# -----------------------------------------------------------------------------
# Set up environment if running under SLURM by sourcing the jobenv file.
# Assumes `ezpz_save_slurm_env` or `ezpz_setup_host` was run previously.
#
# Args:
#   $1 (optional): Path to the jobenv file (defaults based on `ezpz_get_jobenv_file`).
#
# Relies on: `SLURM_JOB_ID`, `ezpz_get_jobenv_file`
# Outputs: Sources the jobenv file. Returns 1 if not in Slurm or file not found.
# -----------------------------------------------------------------------------
ezpz_get_slurm_env() {
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        printf "${YELLOW}Warning [get_slurm_env]: SLURM_JOB_ID not set. Skipping.${RESET}\n" >&2
        return 0 # Not in Slurm job, not an error for 'get'
    fi

    local jobenv_file="${1:-$(ezpz_get_jobenv_file)}"
     if [[ -z "${jobenv_file}" ]]; then
         printf "${RED}Error [get_slurm_env]: Job environment file path not determined.${RESET}\n" >&2
         return 1
    fi

    if [[ -f "${jobenv_file}" ]]; then
        printf "\n${BLUE}[ezpz_get_slurm_env]${RESET}: Sourcing ${BLUE}%s${RESET}\n" "${jobenv_file}"
        # shellcheck source=/dev/null
        source "${jobenv_file}" || {
             printf "${RED}Error sourcing '%s'.${RESET}\n" "${jobenv_file}" >&2
             return 1
        }
        # Define launch alias/function if DIST_LAUNCH was sourced
        if [[ -n "${DIST_LAUNCH:-}" ]]; then
             export ezlaunch="${DIST_LAUNCH}"
             export -f ezpz_launch
             alias launch='ezpz_launch'
             printf "  ✓ Sourced env and defined launch helpers.\n"
        else
             printf "${YELLOW}Warning: Sourced '%s', but DIST_LAUNCH not defined.${RESET}\n" "${jobenv_file}" >&2
        fi
        return 0
    else
         printf "${YELLOW}Warning: Jobenv file '%s' not found. Cannot source SLURM env.${RESET}\n" "${jobenv_file}" >&2
         return 1 # Return error if file doesn't exist
    fi
}

# -----------------------------------------------------------------------------
# Main dispatcher to GET job environment by sourcing the appropriate jobenv file.
# Calls `ezpz_get_pbs_env` or `ezpz_get_slurm_env`.
#
# Args:
#   $@: Arguments passed to underlying get functions (jobenv_file).
#
# Relies on: `ezpz_get_scheduler_type`, `ezpz_get_pbs_env`, `ezpz_get_slurm_env`
# Outputs: Sources env file. Defines launch helpers. Returns status.
# -----------------------------------------------------------------------------
ezpz_get_job_env() {
    local scheduler_type
    scheduler_type=$(ezpz_get_scheduler_type)
    local ret_status=0

    printf "${MAGENTA}---- Getting Job Environment (Scheduler: %s) ----${RESET}\n" "${scheduler_type:-Unknown}"

    case "${scheduler_type}" in
        pbs)   ezpz_get_pbs_env "$@" || ret_status=$? ;;
        slurm) ezpz_get_slurm_env "$@" || ret_status=$? ;;
        *)
            printf "${RED}[ezpz_get_job_env] Unknown scheduler %s. Cannot get environment.${RESET}\n" "${scheduler_type}" >&2
            ret_status=1
            ;;
    esac
    return "${ret_status}"
}

# -----------------------------------------------------------------------------
# Print a summary of the determined job environment based on currently set variables.
# Shows hosts, GPU counts, hostfile, and launch command.
#
# Args: None (uses exported environment variables).
# Relies on: HOSTFILE, NHOSTS, NGPUS_PER_HOST, NGPUS, LAUNCH/DIST_LAUNCH vars being set.
# Outputs: Prints formatted job summary to stdout.
# -----------------------------------------------------------------------------
ezpz_print_job_env() {
    # Use currently set environment variables
    local hostfile="${HOSTFILE:-<Not Set>}"
    local num_hosts="${NHOSTS:-?}"
    local num_gpus_per_host="${NGPUS_PER_HOST:-?}"
    local num_gpus="${NGPUS:-?}"
    local launch_cmd="${ezlaunch:-${LAUNCH:-${DIST_LAUNCH:-<Not Set>}}}" # Prefer ezlaunch function var

    printf "\n"
    printf "  [${MAGENTA}%s${RESET}]:\n" "HOSTS"
    if [[ -f "${hostfile}" ]]; then
        ezpz_print_hosts "${hostfile}"
    elif [[ "${hostfile}" != "<Not Set>" ]]; then
         printf "    (Hostfile '%s' not found)\n" "${hostfile}"
    else
         printf "    (Hostfile not set)\n"
    fi
    printf "\n"
    printf "  [${BRIGHT_BLUE}%s${RESET}]:\n" "DIST INFO (Current Env)"
    printf "      • NGPUS (WORLD_SIZE)=${BRIGHT_BLUE}%s${RESET}\n" "${num_gpus}"
    printf "      • NHOSTS=${BRIGHT_BLUE}%s${RESET}\n" "${num_hosts}"
    printf "      • NGPUS_PER_HOST=${BRIGHT_BLUE}%s${RESET}\n" "${num_gpus_per_host}"
    printf "      • HOSTFILE=${BRIGHT_BLUE}%s${RESET}\n" "${hostfile}"
    printf "      • LAUNCH CMD=${BRIGHT_BLUE}%s${RESET}\n" "${launch_cmd}"
    printf "\n"
    # Print launch helper info
    if [[ "${launch_cmd}" != "<Not Set>" ]]; then
         printf "  [${GREEN}%s${RESET}]:\n" "LAUNCH"
         printf "      • Use function: ${GREEN}ezpz_launch ... ${RESET}\n"
         printf "      • (Alias 'launch' may also be available)\n"
    fi
    printf "\n"
}

# --- Main Orchestration Functions ---

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
ezpz_savejobenv_main() {
    printf "${MAGENTA}==== Running ezpz_savejobenv_main ====${RESET}\n"
    # Setup host first (detects scheduler, saves SLURM/PBS vars, determines hostfile/jobenv)
    if ! ezpz_setup_host "$@"; then
         printf "${RED}Error during ezpz_setup_host. Aborting savejobenv.${RESET}\n" >&2
         return 1
    fi

    # Write calculated info (NGPUS, launch cmd etc.) using determined hostfile/jobenv
    # Pass HOSTFILE and JOBENV_FILE explicitly if they were set by ezpz_setup_host
    # Ensure HOSTFILE is available before calling write_job_info
    if [[ -z "${HOSTFILE:-}" || ! -f "${HOSTFILE:-}" ]]; then
        printf "${RED}Error: HOSTFILE not valid after ezpz_setup_host. Cannot write job info.${RESET}\n" >&2
        return 1
    fi
    if ! ezpz_write_job_info "${HOSTFILE:-}" "${JOBENV_FILE:-}"; then
         printf "${RED}Error during ezpz_write_job_info.${RESET}\n" >&2
         return 1 # Return failure status
    fi

    printf "${MAGENTA}==== Finished ezpz_savejobenv_main (Status: 0) ====${RESET}\n"
    return 0
}

# -----------------------------------------------------------------------------
# Top-level setup for a job environment. Calls `ezpz_savejobenv_main`.
#
# Args:
#   $@: Arguments passed to `ezpz_savejobenv_main` (hostfile, jobenv_file).
# Outputs: Sets up/saves job env info. Prints summary. Returns status.
# -----------------------------------------------------------------------------
ezpz_setup_job() {
    local mn
    local hn
    mn=$(ezpz_get_machine_name)
    hn=$(hostname)
    printf "\n"
    printf "[%s ${YELLOW}%s${RESET}]\n" "🚀" "ezpz/bin/utils_modern.sh"
    printf "    • USER=${YELLOW}%s${RESET}\n" "${USER:-<unknown>}"
    printf "    • MACHINE=${YELLOW}%s${RESET}\n" "${mn}"
    printf "    • HOST=${YELLOW}%s${RESET}\n" "${hn}"
    printf "    • TSTAMP=${YELLOW}%s${RESET}\n" "$(ezpz_get_tstamp)"
    printf "\n"

    # Call the main save/setup function
    ezpz_savejobenv_main "$@"
    return $?
}

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
    printf "${MAGENTA}===== Running Full Environment Setup =====${RESET}\n"
    if ! ezpz_setup_python; then
         printf "${RED}!! Python setup failed. Aborting.${RESET}\n" >&2
         return 1
    fi
    if ! ezpz_setup_job "$@"; then
         printf "${RED}!! Job setup failed.${RESET}\n" >&2
         return 1
    fi
    printf "${MAGENTA}===== Environment Setup Complete =====${RESET}\n"
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
    printf "[ezpz] Setting up Python environment...\n"
    ezpz_setup_python || { printf "${RED}Python setup failed.${RESET}\n"; exit 1; }

    printf "[ezpz] Setting up Job environment...\n"
    ezpz_setup_job "$@" || { printf "${RED}Job setup failed.${RESET}\n"; exit 1; }

    local target_env_path="${VIRTUAL_ENV:-${CONDA_PREFIX:-<unknown>}}"
    printf "[ezpz] Installing ezpz from GitHub into %s\n" "${target_env_path}"

    if [[ -z "${PYTHON_EXEC:-}" ]]; then
        printf "${RED}Error: PYTHON_EXEC not set after setup. Cannot install.${RESET}\n" >&2
        exit 1
    fi

    # Install using pip, requiring virtualenv to avoid global installs if VIRTUAL_ENV is set
    local pip_cmd=("${PYTHON_EXEC}" -m pip install "git+https://github.com/saforem2/ezpz")
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
         pip_cmd+=("--require-virtualenv")
    fi

    if ! "${pip_cmd[@]}"; then
        printf "[ezpz] ${RED}✘ Failed to install ezpz into %s${RESET}\n" "${target_env_path}"
        exit 1
    fi
    printf "[ezpz] ${GREEN}✔ Done!${RESET}\n"
}

# --- Color Printing Helpers ---
printBlack()   { printf "${BLACK}%s${RESET}\n" "$@"; }
printRed()     { printf "${RED}%s${RESET}\n" "$@"; }
printGreen()   { printf "${GREEN}%s${RESET}\n" "$@"; }
printYellow()  { printf "${YELLOW}%s${RESET}\n" "$@"; }
printBlue()    { printf "${BLUE}%s${RESET}\n" "$@"; }
printMagenta() { printf "${MAGENTA}%s${RESET}\n" "$@"; }
printCyan()    { printf "${CYAN}%s${RESET}\n" "$@"; }

# --- Main Execution Block (when sourced) ---

# -----------------------------------------------------------------------------
# Main logic executed when the script is sourced.
# Determines the working directory based on PBS_O_WORKDIR, SLURM_SUBMIT_DIR, or pwd.
# Exports WORKING_DIR.
# -----------------------------------------------------------------------------
utils_main() {
    # Determine working directory, prioritizing scheduler variables
    local wd=""
    if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
        wd="${PBS_O_WORKDIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        wd="${SLURM_SUBMIT_DIR}"
    else
        # Fallback to current directory using Python or pwd
        if command -v python3 &> /dev/null; then
             wd=$(python3 -c 'import os; print(os.getcwd())' 2>/dev/null)
        fi
        # If Python failed or not available, use pwd
        if [[ -z "${wd}" ]]; then
             wd=$(pwd -P) # Physical path
        fi

        if [[ -z "${wd}" ]]; then
             printf "${RED}Error: Cannot determine working directory.${RESET}\n" >&2
             return 1 # Indicate failure
        fi
        printf "${YELLOW}Warning: Using current directory '%s' as WORKING_DIR.${RESET}\n" "${wd}" >&2
    fi

    # Ensure directory exists (it should, but check)
    if [[ ! -d "${wd}" ]]; then
         printf "${RED}Error: Determined WORKING_DIR '%s' does not exist or is not a directory.${RESET}\n" "${wd}" >&2
         return 1
    fi

    export WORKING_DIR="${wd}"
    printf "Exporting WORKING_DIR=${GREEN}%s${RESET}\n" "${WORKING_DIR}"
    return 0
}

# --- Script Entry Point ---
# Call utils_main when the script is sourced to set WORKING_DIR.
# If it fails, print an error but allow sourcing to continue (individual functions might still work).
if ! utils_main; then
     printf "${RED}Critical error during utils_main setup. WORKING_DIR may be incorrect.${RESET}\n" >&2
fi

# If DEBUG mode was enabled, turn off command tracing now that setup is done.
if [[ -n "${DEBUG:-}" ]]; then set +x; fi

# Indicate sourcing completion (optional)
printf "${GREEN}ezpz2.sh sourced successfully.${RESET}\n"
