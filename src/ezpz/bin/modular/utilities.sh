#!/bin/bash
# @file utilities.sh
# @brief General utility functions
# @description
#     This file contains various utility functions used throughout ezpz.

# Function to get head n lines from PBS nodefile
ezpz_head_n_from_pbs_nodefile() {
    local n="${1:-1}"
    local nodefile="${2:-${PBS_NODEFILE:-}}"

    if [[ -n "${nodefile}" && -f "${nodefile}" ]]; then
        head -n "${n}" "${nodefile}"
    fi
}

# Function to get tail n lines from PBS nodefile
ezpz_tail_n_from_pbs_nodefile() {
    local n="${1:-1}"
    local nodefile="${2:-${PBS_NODEFILE:-}}"

    if [[ -n "${nodefile}" && -f "${nodefile}" ]]; then
        tail -n "${n}" "${nodefile}"
    fi
}

# Function to check if qsme is running
ezpz_qsme_running() {
    pgrep -f "qsme" >/dev/null 2>&1
}

# Function to join array elements with delimiter
join_by() {
    local d=${1-} f=${2-}
    if shift 2; then
        printf %s "$f" "${@/#/$d}"
    fi
}

# Function to parse hostfile
ezpz_parse_hostfile() {
    if [[ "$#" != 1 ]]; then
        echo "Expected exactly one argument: hostfile"
        echo "Received: $#"
        return 1
    fi

    local hf="$1"
    if [[ ! -f "${hf}" ]]; then
        log_message ERROR "Hostfile ${hf} does not exist"
        return 1
    fi

    # Return the hostfile content
    cat "${hf}"
}

# Function to build bdist wheel from GitHub repo
ezpz_build_bdist_wheel_from_github_repo() {
    local repo_url="${1:-}"
    local build_dir="${2:-${WORKING_DIR}/build}"

    if [[ -z "${repo_url}" ]]; then
        log_message ERROR "Repository URL is required"
        return 1
    fi

    ezpz_prepare_repo_in_build_dir "${repo_url}" "${build_dir}"
    cd "${build_dir}"

    if [[ ! -f "setup.py" && ! -f "pyproject.toml" ]]; then
        log_message ERROR "No setup.py or pyproject.toml found in repository"
        return 1
    fi

    python3 -m pip wheel . --wheel-dir dist/
    log_message INFO "Wheel built successfully in ${build_dir}/dist/"
}
