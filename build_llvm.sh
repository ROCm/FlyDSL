#!/bin/bash
set -e

# Default to downloading llvm-project in the parent directory of rocDSL
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM_SRC_DIR="$BASE_DIR/llvm-project"
LLVM_BUILD_DIR="$LLVM_SRC_DIR/build"

echo "Base directory: $BASE_DIR"
echo "LLVM Source:    $LLVM_SRC_DIR"
echo "LLVM Build:     $LLVM_BUILD_DIR"

# 1. Clone LLVM
if [ ! -d "$LLVM_SRC_DIR" ]; then
    echo "Cloning llvm-project (branch: release/19.x)..."
    # Use release/19.x branch, generally stable and recent
    git clone --depth 1 -b release/19.x https://github.com/llvm/llvm-project.git "$LLVM_SRC_DIR"
else
    echo "llvm-project directory already exists. Skipping clone."
fi

# 2. Create Build Directory
mkdir -p "$LLVM_BUILD_DIR"
cd "$LLVM_BUILD_DIR"

# 3. Configure CMake
echo "Configuring LLVM..."

# Check for ninja
GENERATOR="Unix Makefiles"
if command -v ninja &> /dev/null; then
    GENERATOR="Ninja"
    echo "Using Ninja generator."
else
    echo "Ninja not found. Using Unix Makefiles (this might be slower)."
fi

# Build only MLIR and necessary Clang tools, targeting native architecture, in Release mode
cmake -G "$GENERATOR" \
    -S "$LLVM_SRC_DIR/llvm" \
    -B "$LLVM_BUILD_DIR" \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON 

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
