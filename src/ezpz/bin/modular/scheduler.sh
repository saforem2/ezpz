#!/bin/bash
# @file scheduler.sh
# @brief Scheduler detection and management functions
# @description
#     This file contains functions for detecting and working with job schedulers.

# Function to get scheduler type
ezpz_get_scheduler_type() {
    if [[ -n "${PBS_JOBID:-}" ]]; then
        echo "pbs"
    elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "slurm"
    else
        echo "unknown"
    fi
}

# Function to get job ID from hostname
ezpz_get_jobid_from_hostname() {
    local hostname="${1:-$(hostname)}"
    case "${hostname}" in
        x3*|x4*)
            # Aurora/Polaris format
            if [[ -n "${PBS_JOBID:-}" ]]; then
                echo "${PBS_JOBID}" | cut -d'.' -f1
            fi
            ;;
        nid*)
            # Perlmutter/Sirius format
            if [[ -n "${SLURM_JOB_ID:-}" ]]; then
                echo "${SLURM_JOB_ID}"
            fi
            ;;
        *)
            # Try PBS first, then SLURM
            if [[ -n "${PBS_JOBID:-}" ]]; then
                echo "${PBS_JOBID}" | cut -d'.' -f1
            elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
                echo "${SLURM_JOB_ID}"
            fi
            ;;
    esac
}

# Function to reset PBS variables
ezpz_reset_pbs_vars() {
    unset PBS_JOBID PBS_NODEFILE PBS_O_WORKDIR
    log_message INFO "PBS variables reset"
}

# Function to get PBS nodefile from hostname
ezpz_get_pbs_nodefile_from_hostname() {
    local hostname="${1:-$(hostname)}"
    local jobid=$(ezpz_get_jobid_from_hostname "${hostname}")
    
    if [[ -n "${jobid}" ]]; then
        echo "/var/spool/pbs/aux/${jobid}.${hostname}"
    fi
}