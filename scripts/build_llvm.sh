#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
set -e

# Default to downloading llvm-project in the parent directory of flydsl
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_DIR="$(cd "${REPO_ROOT}/.." && pwd)"
LLVM_SRC_DIR="$BASE_DIR/llvm-project"
LLVM_BUILD_DIR="$LLVM_SRC_DIR/build-flydsl"
LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-$LLVM_SRC_DIR/mlir_install}"
LLVM_INSTALL_TGZ="${LLVM_INSTALL_TGZ:-$LLVM_SRC_DIR/mlir_install.tgz}"
LLVM_PACKAGE_INSTALL="${LLVM_PACKAGE_INSTALL:-1}"

# Read LLVM commit hash from thirdparty/llvm-hash.txt
LLVM_HASH_FILE="${REPO_ROOT}/thirdparty/llvm-hash.txt"
LLVM_COMMIT_DEFAULT=$(cat "${LLVM_HASH_FILE}" | tr -d '[:space:]')
LLVM_COMMIT="${LLVM_COMMIT:-$LLVM_COMMIT_DEFAULT}"

if [[ ${#LLVM_COMMIT} -lt 40 ]]; then
    echo "Error: LLVM_COMMIT must be a full 40-char SHA (got '${LLVM_COMMIT}')"
    echo "Shallow fetch does not support short hashes."
    exit 1
fi

echo "Base directory: $BASE_DIR"
echo "LLVM Source:    $LLVM_SRC_DIR"
echo "LLVM Build:     $LLVM_BUILD_DIR"
echo "LLVM Install:   $LLVM_INSTALL_DIR"
echo "LLVM Tarball:   $LLVM_INSTALL_TGZ"
echo "LLVM Commit:    $LLVM_COMMIT"

# 1. Clone LLVM
LLVM_REMOTE="${LLVM_REMOTE:-https://github.com/llvm/llvm-project.git}"

if [ ! -d "$LLVM_SRC_DIR" ]; then
    echo "Fetching llvm-project commit ${LLVM_COMMIT} (shallow, single commit)..."
    git init "$LLVM_SRC_DIR"
    pushd "$LLVM_SRC_DIR"
    git remote add origin "$LLVM_REMOTE"
else
    pushd "$LLVM_SRC_DIR"
fi

if ! git cat-file -e "${LLVM_COMMIT}^{commit}" 2>/dev/null; then
    echo "Fetching commit ${LLVM_COMMIT} ..."
    git fetch --depth 1 origin "${LLVM_COMMIT}"
fi
git checkout "${LLVM_COMMIT}"
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
    -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DMLIR_ENABLE_ROCM_RUNNER=ON \
    -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir \
    -DPython3_EXECUTABLE=$(which python3) \
    -Dnanobind_DIR="$NANOBIND_DIR" \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF \
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
  # The install tree may still have files whose mtimes change (e.g. Python bytecode caches),
  # which can cause GNU tar to exit(1) with "file changed as we read it". Treat those as
  # non-fatal for packaging.
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
