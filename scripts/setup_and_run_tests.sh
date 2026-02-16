#!/bin/bash
# Script to set up FlyDSL environment and run tests
# This script will:
#   1. Create a virtual environment (flydsl_venv)
#   2. Activate it
#   3. Install flydsl package
#   4. Install torch and test requirements
#   5. Run the test suite

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory (scripts/) and project root (one level up)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set venv name: use first argument if provided, otherwise default to "flydsl_venv"
VENV_NAME="${1:-flydsl_venv}"

# Set venv path: use second argument if provided, otherwise default to user home directory
VENV_BASE_PATH="${2:-${HOME}}"
VENV_PATH="${VENV_BASE_PATH}/${VENV_NAME}"

REQUIREMENTS_TEST_FILE="${PROJECT_ROOT}/flydsl/requirements_tests.txt"
RUN_TESTS_SCRIPT="${SCRIPT_DIR}/run_tests.sh"

# Function to print error and exit
error_exit() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

# Function to print success message
success_msg() {
    echo -e "${GREEN}$1${NC}"
}

# Function to print info message
info_msg() {
    echo -e "${YELLOW}$1${NC}"
}

echo "=========================================="
echo "FlyDSL Setup and Test Script"
echo "=========================================="
echo ""
echo "Virtual environment name: ${VENV_NAME}"
echo "Virtual environment path: ${VENV_PATH}"
echo "Usage: $0 [venv_name] [venv_path]"
echo "  venv_name: Name of virtual environment (default: flydsl_venv)"
echo "  venv_path: Path where venv will be created (default: \$HOME)"
echo ""

# Validate venv base path exists
if [ ! -d "${VENV_BASE_PATH}" ]; then
    error_exit "Venv base path does not exist: ${VENV_BASE_PATH}"
fi

# Step 1: Create virtual environment
info_msg "[1/5] Creating virtual environment: ${VENV_NAME}..."
if [ -d "${VENV_PATH}" ]; then
    info_msg "  Virtual environment already exists. Removing old one..."
    rm -rf "${VENV_PATH}"
fi

if ! python3 -m venv "${VENV_PATH}"; then
    error_exit "Failed to create virtual environment. Make sure python3-venv is installed."
fi
success_msg "  Virtual environment created successfully"

# Step 2: Activate virtual environment
info_msg "[2/5] Activating virtual environment..."
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
    error_exit "Virtual environment activation script not found at ${VENV_PATH}/bin/activate"
fi

# Source the activation script
source "${VENV_PATH}/bin/activate"

# Verify activation
if [ -z "${VIRTUAL_ENV}" ]; then
    error_exit "Failed to activate virtual environment"
fi
success_msg "  Virtual environment activated: ${VIRTUAL_ENV}"

# Upgrade pip
info_msg "  Upgrading pip..."
pip install --upgrade pip --quiet || error_exit "Failed to upgrade pip"

# Step 3: Install flydsl package
info_msg "[3/5] Installing flydsl package..."
cd "${PROJECT_ROOT}"
if [ ! -f "setup.py" ]; then
    error_exit "setup.py not found. Are you in the FlyDSL root directory?"
fi

if ! pip install -e . --quiet; then
    error_exit "Failed to install flydsl package"
fi
success_msg "  flydsl package installed successfully"

# Step 4: Install torch and test requirements
info_msg "[4/5] Installing torch and test requirements..."
if [ ! -f "${REQUIREMENTS_TEST_FILE}" ]; then
    error_exit "requirements_tests.txt not found at ${REQUIREMENTS_TEST_FILE}"
fi

if ! pip install --no-cache-dir -r "${REQUIREMENTS_TEST_FILE}"; then
    error_exit "Failed to install test requirements"
fi
success_msg "  Test requirements installed successfully"

# Step 5: Run tests
info_msg "[5/5] Running test suite..."
if [ ! -f "${RUN_TESTS_SCRIPT}" ]; then
    error_exit "run_tests.sh not found at ${RUN_TESTS_SCRIPT}"
fi

if [ ! -x "${RUN_TESTS_SCRIPT}" ]; then
    info_msg "  Making run_tests.sh executable..."
    chmod +x "${RUN_TESTS_SCRIPT}"
fi

echo ""
echo "=========================================="
echo "Running Tests"
echo "=========================================="
echo ""

if ! bash "${RUN_TESTS_SCRIPT}"; then
    error_exit "Tests failed. Check the output above for details."
fi

echo ""
echo "=========================================="
success_msg "All steps completed successfully!"
echo "=========================================="
echo ""
info_msg "Virtual environment: ${VENV_PATH}"
info_msg "To activate manually: source ${VENV_PATH}/bin/activate"
echo ""
