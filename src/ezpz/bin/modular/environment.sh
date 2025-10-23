#!/bin/bash
# @file environment.sh
# @brief Environment setup and management functions
# @description
#     This file contains functions for setting up and managing environments.

# Function to save dotenv file
ezpz_save_dotenv() {
    local outfile="${1:-${WORKING_DIR}/.env}"
    env > "${outfile}"
    log_message INFO "Environment saved to ${outfile}"
}

# Function to activate or create micromamba environment
ezpz_activate_or_create_micromamba_env() {
    local env_name="${1:-base}"
    local root_prefix="${MAMBA_ROOT_PREFIX:-/usr/local/micromamba}"

    if [[ -d "${root_prefix}" ]]; then
        local shell_name=$(ezpz_get_shell_name)
        eval "$("${root_prefix}/bin/micromamba" shell hook --shell "${shell_name}")"
        micromamba activate "${env_name}"
        log_message INFO "Activated micromamba environment: ${env_name}"
    else
        log_message WARN "Micromamba not found at ${root_prefix}"
        return 1
    fi
}

# Function to check if already built
ezpz_check_if_already_built() {
    local build_dir="${1:-${WORKING_DIR}/build}"
    if [[ -d "${build_dir}" && -n "$(ls -A "${build_dir}")" ]]; then
        log_message INFO "Build directory ${build_dir} already exists and is not empty"
        return 0
    else
        log_message INFO "Build directory ${build_dir} is empty or does not exist"
        return 1
    fi
}

# Function to prepare repo in build directory
ezpz_prepare_repo_in_build_dir() {
    local repo_url="${1:-}"
    local build_dir="${2:-${WORKING_DIR}/build}"

    if [[ -z "${repo_url}" ]]; then
        log_message ERROR "Repository URL is required"
        return 1
    fi

    mkdir -p "${build_dir}"
    cd "${build_dir}"

    if [[ ! -d ".git" ]]; then
        git clone "${repo_url}" .
    fi

    log_message INFO "Repository prepared in ${build_dir}"
}
