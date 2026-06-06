#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
set -e

# Use local llvm-pipeline directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_DIR="$(cd "${REPO_ROOT}/.." && pwd)"
LLVM_SRC_DIR="$BASE_DIR/llvm-pipeline"
LLVM_BUILD_DIR="$LLVM_SRC_DIR/build-coexec-samebank"
LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-$LLVM_SRC_DIR/mlir_install_coexec_samebank}"
LLVM_INSTALL_TGZ="${LLVM_INSTALL_TGZ:-$LLVM_SRC_DIR/mlir_install_coexec_samebank.tgz}"
LLVM_PACKAGE_INSTALL="${LLVM_PACKAGE_INSTALL:-1}"

if [ ! -d "$LLVM_SRC_DIR" ]; then
    echo "Error: LLVM source directory not found: $LLVM_SRC_DIR"
    exit 1
fi

echo "Base directory: $BASE_DIR"
echo "LLVM Source:    $LLVM_SRC_DIR"
echo "LLVM Build:     $LLVM_BUILD_DIR"
echo "LLVM Install:   $LLVM_INSTALL_DIR"
echo "LLVM Tarball:   $LLVM_INSTALL_TGZ"

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

export HIP_PLATFORM="${HIP_PLATFORM:-amd}"

cmake -G "$GENERATOR" \
    -S "$LLVM_SRC_DIR/llvm" \
    -B "$LLVM_BUILD_DIR" \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
    -DLLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD:-X86;NVPTX;AMDGPU}" \
    -DLLVM_ENABLE_RUNTIMES="" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DMLIR_ENABLE_ROCM_RUNNER=ON \
    -DMLIR_ENABLE_EXECUTION_ENGINE=ON \
    -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir \
    -DPython3_EXECUTABLE=$(which python3) \
    -Dnanobind_DIR="$NANOBIND_DIR" \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF \
    -DMLIR_INCLUDE_TESTS=OFF \
    -DHIP_PLATFORM=amd \
    -DCMAKE_INSTALL_RPATH="\$ORIGIN"

# 4. Build
PARALLEL_JOBS=$(( $(nproc) / 2 ))
for arg in "$@"; do
    if [[ "$arg" =~ ^-j([0-9]+)$ ]]; then
        PARALLEL_JOBS="${BASH_REMATCH[1]}"
    elif [[ "$arg" == "--no-install" ]]; then
        LLVM_PACKAGE_INSTALL=0
    fi
done
echo "Starting build with ${PARALLEL_JOBS} parallel jobs..."
cmake --build . -j${PARALLEL_JOBS}

if [[ "${LLVM_PACKAGE_INSTALL}" == "1" ]]; then
  echo "=============================================="
  echo "Installing MLIR/LLVM to a clean prefix..."
  rm -rf "${LLVM_INSTALL_DIR}"
  mkdir -p "${LLVM_INSTALL_DIR}"
  cmake --install "${LLVM_BUILD_DIR}" --prefix "${LLVM_INSTALL_DIR}"

  if [[ ! -d "${LLVM_INSTALL_DIR}/lib/cmake/mlir" ]]; then
    echo "Error: install prefix missing lib/cmake/mlir: ${LLVM_INSTALL_DIR}" >&2
    exit 1
  fi

  echo "Creating tarball..."
  tar --warning=no-file-changed --warning=no-file-removed --ignore-failed-read \
      -C "$(dirname "${LLVM_INSTALL_DIR}")" \
      -czf "${LLVM_INSTALL_TGZ}" "$(basename "${LLVM_INSTALL_DIR}")"
fi

echo "=============================================="
echo "LLVM/MLIR build completed successfully!"
echo ""
echo "To configure flydsl, use:"
echo "cmake .. -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir"
if [[ "${LLVM_PACKAGE_INSTALL}" == "1" ]]; then
  echo ""
  echo "Packaged install prefix:"
  echo "  ${LLVM_INSTALL_DIR}"
  echo "Use with:"
  echo "  export MLIR_PATH=${LLVM_INSTALL_DIR}"
  echo "Tarball:"
  echo "  ${LLVM_INSTALL_TGZ}"
fi
echo "=============================================="
