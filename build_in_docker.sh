#!/bin/bash
# Build CuTe IR inside felixatt Docker container (CUDA-only)

set -e

CONTAINER_NAME="felixatt"
PROJECT_DIR="/mnt/raid0/felix/cute_ir_tablegen"
MLIR_DIR="/mnt/raid0/felix/llvm-project/buildmlir/lib/cmake/mlir"
LLVM_DIR="/mnt/raid0/felix/llvm-project/buildmlir/lib/cmake/llvm"

echo "========================================="
echo "Building CuTe IR in Docker: ${CONTAINER_NAME}"
echo "ROCm enabled"
echo "========================================="

# Check if container is running
if ! docker ps | grep -q ${CONTAINER_NAME}; then
    echo "Error: Container ${CONTAINER_NAME} is not running"
    echo "Please start the container first"
    exit 1
fi

# Execute build commands inside the container
docker exec -it ${CONTAINER_NAME} bash -c "
    cd ${PROJECT_DIR} &&     echo '--- Cleaning previous build ---' &&     rm -rf build &&     mkdir -p build &&     cd build &&     echo '' &&     echo '--- CMake Configuration (with MLIR) ---' &&     cmake .. \
        -DMLIR_DIR=${MLIR_DIR} \
        -DLLVM_DIR=${LLVM_DIR} \
        -DBUILD_RUNTIME=OFF \
        -DBUILD_PYTHON_BINDINGS=OFF \
        -DENABLE_ROCM=ON \
        -DCMAKE_BUILD_TYPE=Release &&     echo '' &&     echo '--- Building TableGen targets (First Pass) ---' &&     # We expect this to fail on C++ compilation but succeed on TableGen
    make -j$(nproc) || true &&     echo '' &&     echo '--- Patching Generated Headers ---' &&     cd .. && python3 patch_ops.py && cd build &&     echo '' &&     echo '--- Resuming Build ---' &&     make -j$(nproc) &&     echo '' &&     echo '=========================================' &&     echo 'Build completed successfully!' &&     echo '========================================='
"

echo ""
echo "Build finished."
