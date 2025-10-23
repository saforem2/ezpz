#!/bin/bash
# @file machines.sh
# @brief Machine-specific setup functions
# @description
#     This file contains functions for setting up specific HPC machines.

# Function to get machine name
ezpz_get_machine_name() {
    local hostname="${1:-$(hostname)}"

    case "${hostname}" in
        x3*)
            echo "aurora"
            ;;
        x4*)
            echo "polaris"
            ;;
        nid*)
            # Check if it's Perlmutter or Sirius
            if [[ "${hostname}" == *sirius* ]]; then
                echo "sirius"
            else
                echo "perlmutter"
            fi
            ;;
        frontier*)
            echo "frontier"
            ;;
        sunspot*)
            echo "sunspot"
            ;;
        sophia*)
            echo "sophia"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Function to setup conda for Frontier
ezpz_setup_conda_frontier() {
    log_message INFO "Setting up conda for Frontier"
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module load PrgEnv-gnu/8.5.0
        module load craype-accel-amd-gfx90a
        module load rocm
        micromamba activate /lustre/orion/csc613/scratch/foremans/envs/micromamba/py3.10-torch2.2-rocm5.7
    fi
}

# Function to setup conda for Sunspot
ezpz_setup_conda_sunspot() {
    log_message INFO "Setting up conda for Sunspot"
    if [[ -z "${CONDA_PREFIX:-}" ]] || [[ -z "${PYTHON_ROOT:-}" ]]; then
        module load oneapi/release/2025.2.0
        module load py-torch/2.8
        module load py-ipex/2.8.10xpu
        export ZE_FLAT_DEVICE_HIERARCHY=FLAT
    fi
}

# Function to setup conda for Aurora
ezpz_setup_conda_aurora() {
    log_message INFO "Setting up conda for Aurora"
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module load frameworks
        log_message INFO "Setting FI_MR_CACHE_MONITOR=userfaultfd"
        export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"
    else
        log_message WARN "Using existing CONDA_PREFIX=${CONDA_PREFIX}"
    fi
}

# Function to setup conda for Sirius
ezpz_setup_conda_sirius() {
    log_message INFO "Setting up conda for Sirius"
    if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV-}" ]]; then
        export MAMBA_ROOT_PREFIX=/lus/tegu/projects/PolarisAT/foremans/micromamba
        local shell_name=$(ezpz_get_shell_name)
        eval "$("${MAMBA_ROOT_PREFIX}/bin/micromamba" shell hook --shell "${shell_name}")"
        micromamba activate 2024-04-23
    else
        log_message INFO "Found existing python at: $(which python3)"
    fi
}

# Function to setup conda for Polaris
ezpz_setup_conda_polaris() {
    log_message INFO "Setting up conda for Polaris"
    if [[ "${PBS_O_HOST:-}" == sirius* ]]; then
        ezpz_setup_conda_sirius
    else
        # Setup for standard Polaris
        if [[ -z "${CONDA_PREFIX:-}" ]]; then
            module use /soft/modulefiles
            module load conda
            conda activate base
        else
            log_message INFO "Using existing CONDA_PREFIX=${CONDA_PREFIX}"
        fi
    fi
}

# Function to setup conda for Perlmutter
ezpz_setup_conda_perlmutter() {
    log_message INFO "Setting up conda for Perlmutter"
    module load pytorch
}

# Function to setup conda for Sophia
ezpz_setup_conda_sophia() {
    log_message INFO "Setting up conda for Sophia"
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module load conda
        conda activate base
    else
        log_message INFO "Using existing CONDA_PREFIX=${CONDA_PREFIX}"
    fi
}

# Main function to setup conda based on machine
ezpz_setup_conda() {
    local machine=$(ezpz_get_machine_name)

    case "${machine}" in
        "aurora")
            ezpz_setup_conda_aurora
            ;;
        "polaris"|"sirius")
            ezpz_setup_conda_polaris
            ;;
        "frontier")
            ezpz_setup_conda_frontier
            ;;
        "sunspot")
            ezpz_setup_conda_sunspot
            ;;
        "perlmutter")
            ezpz_setup_conda_perlmutter
            ;;
        "sophia")
            ezpz_setup_conda_sophia
            ;;
        *)
            log_message WARN "Unknown machine: ${machine}. No specific conda setup available."
            return 1
            ;;
    esac
}
