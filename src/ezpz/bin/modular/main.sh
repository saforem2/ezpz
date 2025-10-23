#!/bin/bash
# @file main.sh
# @brief Main entry point functions
# @description
#     This file contains the main entry point functions that users call.

# Import all modular components
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source all modular components if not already sourced
if [[ -z "${EZPZ_MODULAR_LOADED:-}" ]]; then
    source "${SCRIPT_DIR}/core.sh"
    source "${SCRIPT_DIR}/logging.sh"
    source "${SCRIPT_DIR}/scheduler.sh"
    source "${SCRIPT_DIR}/environment.sh"
    source "${SCRIPT_DIR}/machines.sh"
    source "${SCRIPT_DIR}/python.sh"
    source "${SCRIPT_DIR}/jobs.sh"
    source "${SCRIPT_DIR}/utilities.sh"
    export EZPZ_MODULAR_LOADED=1
fi

# Main setup function - this is the primary entry point
ezpz_setup_env() {
    if ! ezpz_check_working_dir; then
        log_message ERROR "Failed to set WORKING_DIR. Please check your environment."
        return 1
    fi

    log_message INFO "Running [${BRIGHT_YELLOW}ezpz_setup_env${RESET}]..."

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

# Function to setup job - this needs to be implemented based on the original
ezpz_setup_job() {
    log_message INFO "Setting up job environment..."

    local scheduler_type=$(ezpz_get_scheduler_type)
    local machine=$(ezpz_get_machine_name)

    log_message INFO "Detected scheduler: ${scheduler_type}, machine: ${machine}"

    # Save environment
    ezpz_save_dotenv

    # Additional job-specific setup can go here
    case "${scheduler_type}" in
        "pbs")
            if [[ -n "${PBS_NODEFILE:-}" ]]; then
                log_message INFO "PBS nodefile: ${PBS_NODEFILE}"
            fi
            ;;
        "slurm")
            if [[ -n "${SLURM_NODELIST:-}" ]]; then
                log_message INFO "SLURM nodelist: ${SLURM_NODELIST}"
            fi
            ;;
    esac

    log_message INFO "Job setup completed"
}
