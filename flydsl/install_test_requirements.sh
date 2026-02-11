#!/bin/bash
# Script to install test requirements, only installing torch if not already present

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements_tests.txt"
PYTORCH_ROCM_INDEX="https://download.pytorch.org/whl/rocm7.1"

echo "Installing test requirements..."

# Install base and test requirements (excluding torch)
echo "Installing base and test dependencies..."
pip install -r "${REQUIREMENTS_FILE}"

# --no-cache-dir is used when installing torch to avoid using cached packages, and fix the error:
# packages/pip/_vendor/msgpack/fallback.py", line 821, in _pack
#    raise ValueError("Memoryview is too large")
# ValueError: Memoryview is too large

# Check if torch is already installed
if python -c "import torch" 2>/dev/null; then
    echo "[OK] torch is already installed, skipping..."
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    echo "  Current torch version: ${TORCH_VERSION}"
else
    echo "Installing torch from ROCm index..."
    pip install --no-cache-dir torch --index-url "${PYTORCH_ROCM_INDEX}"
    echo "[OK] torch installed successfully"
fi

echo ""
echo "All test requirements installed!"
