#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_DIR="$(cd "${REPO_ROOT}/.." && pwd)"

LLVM_HASH_FILE="${REPO_ROOT}/cmake/llvm-hash.txt"
if [[ -f "${LLVM_HASH_FILE}" ]]; then
    LLVM_COMMIT_DEFAULT=$(cat "${LLVM_HASH_FILE}" | tr -d '[:space:]')
else
    LLVM_COMMIT_DEFAULT="edf06d742821"
fi

LLVM_SRC_DIR="${LLVM_SRC_DIR:-$BASE_DIR/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$LLVM_SRC_DIR/build}"
LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-$LLVM_SRC_DIR/install}"
LLVM_COMMIT="${LLVM_COMMIT:-$LLVM_COMMIT_DEFAULT}"
LLVM_PACKAGE_INSTALL="${LLVM_PACKAGE_INSTALL:-0}"

echo "=============================================="
echo "FlyDSL LLVM/MLIR Build Script"
echo "=============================================="
echo "LLVM Source:  ${LLVM_SRC_DIR}"
echo "LLVM Build:   ${LLVM_BUILD_DIR}"
echo "LLVM Commit:  ${LLVM_COMMIT}"
echo "LLVM Install: ${LLVM_INSTALL_DIR}"
echo "=============================================="

if [ ! -d "$LLVM_SRC_DIR" ]; then
    echo "Cloning llvm-project from ROCm fork..."
    git clone --depth 1 https://github.com/ROCm/llvm-project.git "$LLVM_SRC_DIR"
fi

pushd "$LLVM_SRC_DIR" > /dev/null

CURRENT_REMOTE=$(git remote get-url origin)
if [[ "$CURRENT_REMOTE" == *"github.com/llvm/llvm-project"* ]]; then
    echo "Switching origin to ROCm fork..."
    git remote set-url origin https://github.com/ROCm/llvm-project.git
fi

CURRENT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "none")
SHORT_CURRENT=$(echo "$CURRENT_COMMIT" | cut -c1-12)
SHORT_TARGET=$(echo "$LLVM_COMMIT" | cut -c1-12)

if [[ "$SHORT_CURRENT" != "$SHORT_TARGET"* && "$SHORT_TARGET" != "$SHORT_CURRENT"* ]]; then
    echo "Fetching and checking out commit ${LLVM_COMMIT}..."
    git fetch --depth 1 origin "${LLVM_COMMIT}"
    git checkout "${LLVM_COMMIT}"
else
    echo "Already at commit ${SHORT_CURRENT}"
fi

popd > /dev/null

mkdir -p "$LLVM_BUILD_DIR"

GENERATOR="Unix Makefiles"
if command -v ninja &> /dev/null; then
    GENERATOR="Ninja"
fi

echo "Installing Python build dependencies..."
pip install -q nanobind numpy pybind11

NANOBIND_DIR=$(python3 -c "import nanobind; import os; print(os.path.dirname(nanobind.__file__) + '/cmake')")

echo "Configuring LLVM with ${GENERATOR}..."
cmake -G "$GENERATOR" \
    -S "$LLVM_SRC_DIR/llvm" \
    -B "$LLVM_BUILD_DIR" \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DMLIR_ENABLE_ROCM_RUNNER=ON \
    -DPython3_EXECUTABLE=$(which python3) \
    -Dnanobind_DIR="$NANOBIND_DIR" \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF 

NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Building with ${NPROC} parallel jobs..."
cmake --build "$LLVM_BUILD_DIR" -j"$NPROC"

if [[ "${LLVM_PACKAGE_INSTALL}" == "1" ]]; then
    echo "Installing to ${LLVM_INSTALL_DIR}..."
    rm -rf "${LLVM_INSTALL_DIR}"
    mkdir -p "${LLVM_INSTALL_DIR}"
    cmake --install "${LLVM_BUILD_DIR}" --prefix "${LLVM_INSTALL_DIR}"

    if [[ ! -d "${LLVM_INSTALL_DIR}/lib/cmake/mlir" ]]; then
        echo "Error: install prefix missing lib/cmake/mlir" >&2
        exit 1
    fi

    LLVM_INSTALL_TGZ="${LLVM_INSTALL_DIR}.tar.gz"
    echo "Creating tarball: ${LLVM_INSTALL_TGZ}"
    tar -C "$(dirname "${LLVM_INSTALL_DIR}")" -czf "${LLVM_INSTALL_TGZ}" "$(basename "${LLVM_INSTALL_DIR}")"
fi

echo "=============================================="
echo "LLVM/MLIR build complete!"
echo "MLIR_DIR: ${LLVM_BUILD_DIR}/lib/cmake/mlir"
echo "=============================================="
