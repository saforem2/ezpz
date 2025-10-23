#!/bin/bash
# @file ezpz.sh
# @brief Main entry point for ezpz shell utilities
# @description
#     This file provides the main entry point that sources all modular components
#     of the ezpz utility functions.

# Source all modular components
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Core utilities (always loaded)
source "${SCRIPT_DIR}/core.sh"

# Scheduler-specific functions
source "${SCRIPT_DIR}/scheduler.sh"

# Environment setup functions
source "${SCRIPT_DIR}/environment.sh"

# Machine-specific setup functions
source "${SCRIPT_DIR}/machines.sh"

# Job management functions
source "${SCRIPT_DIR}/jobs.sh"

# Logging utilities
source "${SCRIPT_DIR}/logging.sh"

# Python/Conda setup functions
source "${SCRIPT_DIR}/python.sh"

# Utility functions
source "${SCRIPT_DIR}/utilities.sh"

# Backward compatibility - export main functions
export -f ezpz_setup_env
export -f ezpz_setup_python
export -f ezpz_setup_job
export -f ezpz_get_machine_name
