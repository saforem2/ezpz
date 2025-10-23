#!/bin/bash
# @file logging.sh
# @brief Logging utilities for ezpz shell functions
# @description
#     This file contains logging functions for consistent messaging.

# Function to show environment variables
ezpz_show_env() {
    env | sort
}

# Function to log messages with consistent formatting
log_message() {
    local level="${1:-INFO}"
    local message="${2:-}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case "${level}" in
        "ERROR"|"error")
            printf "${RED}[${timestamp}] [ERROR] ${message}${RESET}\n" >&2
            ;;
        "WARN"|"WARNING"|"warn"|"warning")
            printf "${YELLOW}[${timestamp}] [WARN] ${message}${RESET}\n" >&2
            ;;
        "INFO"|"info")
            printf "${GREEN}[${timestamp}] [INFO] ${message}${RESET}\n"
            ;;
        "DEBUG"|"debug")
            if [[ "${EZPZ_DEBUG:-}" == "1" ]]; then
                printf "${BLUE}[${timestamp}] [DEBUG] ${message}${RESET}\n"
            fi
            ;;
        *)
            printf "[${timestamp}] [${level}] ${message}\n"
            ;;
    esac
}

# Function to reset environment
ezpz_reset() {
    unset PBS_JOBID PBS_NODEFILE PBS_O_WORKDIR SLURM_JOB_ID SLURM_NODELIST
    log_message INFO "Environment reset"
}
