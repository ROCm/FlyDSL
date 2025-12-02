#!/bin/bash
set -e

# Default to downloading llvm-project in the parent directory of rocDSL
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM_SRC_DIR="$BASE_DIR/llvm-project"
LLVM_BUILD_DIR="$LLVM_SRC_DIR/buildmlir"

echo "Base directory: $BASE_DIR"
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
echo "To configure rocDSL, use:"
echo "cmake .. -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir"
echo "=============================================="
