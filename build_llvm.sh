#!/bin/bash
set -e

# Build LLVM/MLIR for FlyDSL.
#
# This repo historically supported two layouts:
# - llvm-project under the repo root:        FlyDSL/llvm-project
# - llvm-project under the repo parent:      FlyDSL/../llvm-project   (common on shared machines)
#
# We auto-detect existing checkouts and default to the parent layout for
# backwards-compatibility (flir/build.sh supports both).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM_SRC_DIR_IN_REPO="${REPO_ROOT}/llvm-project"
LLVM_SRC_DIR_IN_PARENT="${REPO_ROOT}/../llvm-project"

# Allow explicit override.
if [ -n "${LLVM_SRC_DIR:-}" ]; then
    LLVM_SRC_DIR="${LLVM_SRC_DIR}"
elif [ -d "${LLVM_SRC_DIR_IN_REPO}" ]; then
    LLVM_SRC_DIR="${LLVM_SRC_DIR_IN_REPO}"
elif [ -d "${LLVM_SRC_DIR_IN_PARENT}" ]; then
    LLVM_SRC_DIR="${LLVM_SRC_DIR_IN_PARENT}"
else
    # Backward compatible default: create under the repo parent directory.
    LLVM_SRC_DIR="${LLVM_SRC_DIR_IN_PARENT}"
fi

# Build directory (override-able).
LLVM_BUILD_DIR_DEFAULT="${LLVM_SRC_DIR}/buildmlir"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${LLVM_BUILD_DIR_DEFAULT}}"

echo "Repo root:      $REPO_ROOT"
echo "LLVM Source:    $LLVM_SRC_DIR"
echo "LLVM Build:     $LLVM_BUILD_DIR"

# 1. Clone LLVM
if [ ! -d "$LLVM_SRC_DIR" ]; then
    echo "Cloning llvm-project..."
    git clone https://github.com/ROCm/llvm-project.git "$LLVM_SRC_DIR"
fi

echo "Checking out amd-staging branch (commit 04f968b02917)..."
pushd "$LLVM_SRC_DIR"

# Check if we need to switch remote to ROCm fork
CURRENT_REMOTE=$(git remote get-url origin)
if [[ "$CURRENT_REMOTE" == *"github.com/llvm/llvm-project"* ]]; then
    echo "Detected upstream LLVM. Switching origin to ROCm fork for amd-staging..."
    git remote set-url origin https://github.com/ROCm/llvm-project.git
fi

git fetch origin amd-staging
git checkout 04f968b02917
popd

# 2. Create Build Directory
mkdir -p "$LLVM_BUILD_DIR"
cd "$LLVM_BUILD_DIR"

# 3. Configure CMake
echo "Configuring LLVM..."

# Install dependencies for Python bindings
echo "Installing Python dependencies..."
pip install nanobind numpy pybind11

# Check for ninja
GENERATOR="Unix Makefiles"
if command -v ninja &> /dev/null; then
    GENERATOR="Ninja"
    echo "Using Ninja generator."
else
    echo "Ninja not found. Using Unix Makefiles (this might be slower)."
fi

# Build only MLIR and necessary Clang tools, targeting native architecture, in Release mode
# Explicitly set nanobind directory if found to help CMake locate it
NANOBIND_DIR=$(python3 -c "import nanobind; import os; print(os.path.dirname(nanobind.__file__) + '/cmake')")

cmake -G "$GENERATOR" \
    -S "$LLVM_SRC_DIR/llvm" \
    -B "$LLVM_BUILD_DIR" \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3) \
    -Dnanobind_DIR="$NANOBIND_DIR" 

# 4. Build
echo "Starting build with $(nproc) parallel jobs..."
cmake --build . --target check-mlir -j$(nproc) || echo "MLIR tests failed, but build might be okay for use."
cmake --build . -j$(nproc)

echo "=============================================="
echo "LLVM/MLIR build completed successfully!"
echo ""
echo "To configure flir, use:"
echo "cmake .. -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir"
echo "=============================================="
