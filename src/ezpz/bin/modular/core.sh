#!/bin/bash
# @file core.sh
# @brief Core utilities for ezpz shell functions
# @description
#     This file contains the most fundamental utilities that are needed by all other components.

# --- Strict Mode & Options ---
# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error when substituting.
# The return value of a pipeline is the status of the last command to exit
# with a non-zero status, or zero if no command exited with a non-zero status.
# set -euo pipefail

EZPZ_SHELL_TYPE="$(basename "${SHELL}")"

if [[ "${EZPZ_SHELL_TYPE}" == "bash" ]]; then
    # Allow aliases to be expanded (needed for `launch` alias)
    shopt -s expand_aliases
fi

if [[ -n "${NO_COLOR:-}" || -n "${NOCOLOR:-}" || "${COLOR:-}" == 0 || "${TERM}" == "dumb" ]]; then
    # --- Color Codes ---
    # Usage: printf "${RED}This is red text${RESET}\n"
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
    export WHITE='\e[1;37m'
    export BRIGHT_WHITE='\e[1;97m'
fi

# Function to check working directory
ezpz_check_working_dir() {
    if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
        export WORKING_DIR="${PBS_O_WORKDIR}"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        export WORKING_DIR="${SLURM_SUBMIT_DIR}"
    else
        export WORKING_DIR="$(pwd)"
    fi

    if [[ ! -d "${WORKING_DIR}" ]]; then
        return 1
    fi

    return 0
}

# Function to get timestamp
ezpz_get_tstamp() {
    date '+%Y-%m-%d-%H%M%S'
}

# Function to get shell name
ezpz_get_shell_name() {
    basename "${SHELL}"
}

# Function to kill MPI processes
ezpz_kill_mpi() {
    pkill -f "mpiexec\|mpirun" 2>/dev/null || true
    pkill -f "orted\|orterun" 2>/dev/null || true
}
