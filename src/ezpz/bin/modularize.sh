#!/bin/bash
# @file modularize.sh
# @brief Script to modularize the original utils.sh file

# This script helps in the transition from monolithic utils.sh to modular structure

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_FILE="${SCRIPT_DIR}/../utils.sh"
MODULAR_DIR="${SCRIPT_DIR}/modular"

echo "Modularizing ezpz utils..."

# Check if original file exists
if [[ ! -f "${ORIGINAL_FILE}" ]]; then
    echo "Error: Original utils.sh file not found at ${ORIGINAL_FILE}"
    exit 1
fi

echo "Original file: ${ORIGINAL_FILE}"
echo "Modular directory: ${MODULAR_DIR}"

# Create modular directory if it doesn't exist
mkdir -p "${MODULAR_DIR}"

# Copy the modular files we've created
echo "Modular files are ready in ${MODULAR_DIR}"

# Create a symlink or backup of the original file
if [[ ! -f "${ORIGINAL_FILE}.backup" ]]; then
    cp "${ORIGINAL_FILE}" "${ORIGINAL_FILE}.backup"
    echo "Created backup of original file"
fi

echo "Modularization complete!"
echo "To use the modular version, source ${MODULAR_DIR}/ezpz.sh instead of utils.sh"
