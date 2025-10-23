#!/bin/bash
# @file jobs.sh
# @brief Job management functions
# @description
#     This file contains functions for managing jobs and job environments.

# Function to setup srun for SLURM
ezpz_setup_srun() {
    log_message INFO "Setting up srun environment"
    export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
    export SRUN_NTASKS="${SLURM_NTASKS:-1}"
    log_message INFO "SRUN_CPUS_PER_TASK=${SRUN_CPUS_PER_TASK}, SRUN_NTASKS=${SRUN_NTASKS}"
}

# Function to save DeepSpeed environment
ezpz_save_ds_env() {
    local outfile="${1:-${WORKING_DIR}/ds_env.json}"
    local jobid=$(ezpz_get_jobid_from_hostname)
    local machine=$(ezpz_get_machine_name)
    
    cat > "${outfile}" << EOF
{
    "job_id": "${jobid}",
    "machine": "${machine}",
    "timestamp": "$(ezpz_get_tstamp)",
    "working_dir": "${WORKING_DIR}",
    "hostname": "$(hostname)"
}
EOF

    log_message INFO "DeepSpeed environment saved to ${outfile}"
}

# Function to check if job is already running
ezpz_check_and_kill_if_running() {
    local job_name="${1:-ezpz_job}"
    local pid_file="${WORKING_DIR}/${job_name}.pid"
    
    if [[ -f "${pid_file}" ]]; then
        local pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            log_message WARN "Job ${job_name} is already running (PID: ${pid}). Killing it..."
            kill "${pid}"
            rm -f "${pid_file}"
        else
            log_message INFO "Stale PID file found. Removing..."
            rm -f "${pid_file}"
        fi
    fi
}

# Function to get SLURM running job ID
ezpz_get_slurm_running_jobid() {
    if command -v squeue >/dev/null 2>&1; then
        squeue -u "${USER}" -h -o "%A" | head -1
    fi
}

# Function to get SLURM running nodelist
ezpz_get_slurm_running_nodelist() {
    local jobid="${1:-$(ezpz_get_slurm_running_jobid)}"
    if [[ -n "${jobid}" ]] && command -v squeue >/dev/null 2>&1; then
        squeue -j "${jobid}" -h -o "%N"
    fi
}