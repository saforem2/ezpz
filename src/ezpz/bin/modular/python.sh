#!/bin/bash
# @file python.sh
# @brief Python and conda setup functions
# @description
#     This file contains functions for setting up Python environments.

# Function to setup Python for ALCF systems
ezpz_setup_python_alcf() {
    log_message INFO "Setting up Python for ALCF systems"
    local machine=$(ezpz_get_machine_name)

    case "${machine}" in
        "aurora")
            ezpz_setup_conda_aurora
            ;;
        "polaris"|"sirius")
            ezpz_setup_conda_polaris
            ;;
        *)
            log_message WARN "No specific ALCF setup for ${machine}"
            ;;
    esac
}

# Function to setup Python for NERSC systems
ezpz_setup_python_nersc() {
    log_message INFO "Setting up Python for NERSC systems"
    local machine=$(ezpz_get_machine_name)

    case "${machine}" in
        "perlmutter")
            ezpz_setup_conda_perlmutter
            ;;
        *)
            log_message WARN "No specific NERSC setup for ${machine}"
            ;;
    esac
}

# Main Python setup function
ezpz_setup_python() {
    local scheduler_type=$(ezpz_get_scheduler_type)

    case "${scheduler_type}" in
        "pbs")
            ezpz_setup_python_alcf
            ;;
        "slurm")
            ezpz_setup_python_nersc
            ;;
        *)
            log_message WARN "Unknown scheduler type: ${scheduler_type}"
            # Try to detect machine and setup accordingly
            ezpz_setup_conda
            ;;
    esac
}

# Function to setup Python with specific PyTorch version for Aurora
ezpz_setup_python_pt_new_aurora() {
    local conda_env="${1:-}"

    if [[ -z "${conda_env}" ]]; then
        log_message ERROR "Usage: ezpz_setup_python_pt_new_aurora <conda_env>"
        return 1
    fi

    log_message INFO "Setting up Python with PyTorch environment: ${conda_env}"
    ezpz_setup_python
}

# Function to setup Python with PyTorch 2.9 for Aurora
ezpz_setup_python_pt29_aurora() {
    log_message INFO "[ezpz_setup_python_pt29_aurora]"
    local pt29env="2025-08-04-pt29"
    ezpz_setup_python_pt_new_aurora "${pt29env}"
}

# Function to setup Python with PyTorch 2.8 for Aurora
ezpz_setup_python_pt28_aurora() {
    log_message INFO "[ezpz_setup_python_pt28_aurora]"
    local pt28env="2025-08-04-pt28"
    ezpz_setup_python_pt_new_aurora "${pt28env}"
}
