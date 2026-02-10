#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Keep all generated artifacts under one directory by default.
# - You can override with:
#   FLIR_OUT_DIR=.flir          (relative to repo root) or an absolute path
#   FLIR_BUILD_DIR=...          (absolute path to CMake build dir)
DEFAULT_OUT_DIR="${REPO_ROOT}/.flir"
# Backward compatible: honor legacy FLIR_OUT_DIR/FLIR_BUILD_DIR if FLIR_* not set.
OUT_DIR="${FLIR_OUT_DIR:-${DEFAULT_OUT_DIR}}"
if [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${REPO_ROOT}/${OUT_DIR}"
fi
BUILD_DIR="${FLIR_BUILD_DIR:-${OUT_DIR}/build}"
if [[ "${BUILD_DIR}" != /* ]]; then
  BUILD_DIR="${REPO_ROOT}/${BUILD_DIR}"
fi

# Set up environment
if [ -z "$MLIR_PATH" ]; then
    # Prefer packaged install prefix if present (created by scripts/build_llvm.sh).
    candidates=(
      "${REPO_ROOT}/llvm-project/mlir_install"
    )
    for p in "${candidates[@]}"; do
      if [ -d "${p}/lib/cmake/mlir" ]; then
        echo "MLIR_PATH not set. Using: ${p}"
        export MLIR_PATH="${p}"
        break
      fi
    done
    if [ -z "${MLIR_PATH:-}" ]; then
      echo "Error: MLIR_PATH not set and no default MLIR install/build dir found." >&2
      echo "Tried:" >&2
      for p in "${candidates[@]}"; do
        echo "  - ${p}" >&2
      done
      echo "Please run: bash scripts/build_llvm.sh (which can package an install prefix), or set MLIR_PATH." >&2
      exit 1
    fi
fi

# Build C++ components
mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"

# Enable ROCm by default when ROCm is present on the system.
ENABLE_ROCM_FLAG=OFF
if [[ -d "/opt/rocm" ]] || command -v hipcc &> /dev/null; then
  ENABLE_ROCM_FLAG=ON
fi

cmake "${SCRIPT_DIR}" \
    -DMLIR_DIR="$MLIR_PATH/lib/cmake/mlir" \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_RUNTIME=OFF \
    -DENABLE_ROCM="${ENABLE_ROCM_FLAG}"

# Build core targets
echo "Building core libraries..."
cmake --build . --target FlirDialect -j$(nproc) || { echo "Failed to build FlirDialect"; exit 1; }
cmake --build . --target FlirTransforms -j$(nproc) || { echo "Failed to build FlirTransforms"; exit 1; }

# Build unified Python extension (_flir_ir)
echo "Building unified Python extension..."
cmake --build . --target _flir_ir -j$(nproc) || { echo "Failed to build _flir_ir"; exit 1; }

# Build flir-opt tool (used by run_tests.sh MLIR file tests)
cmake --build . --target flir-opt -j$(nproc) || true

# Build FLIR dialect TableGen sources
cmake --build . --target FlirPythonSources.flir.ops_gen -j$(nproc) || true

# Set up Python package structure
echo "Setting up Python package structure..."
bash "${SCRIPT_DIR}/python_bindings/setup_python_package.sh" "${BUILD_DIR}" "${MLIR_PATH}/../buildmlir"

# Set up PYTHONPATH for the embedded Python package root (contains `flydsl/_mlir/`)
PYTHON_PACKAGE_DIR="${BUILD_DIR}/python_packages/flydsl"

# Ensure the python package root contains the embedded MLIR package (flydsl/_mlir).
if [ ! -d "${PYTHON_PACKAGE_DIR}/flydsl/_mlir" ]; then
    echo "Error: expected python package not found: ${PYTHON_PACKAGE_DIR}/flydsl/_mlir"
    echo "   (Did the build generate embedded MLIR python modules?)"
    exit 1
fi

# Clean any previously overlaid sources at the root (keep embedded flydsl/_mlir and include/).
find "${PYTHON_PACKAGE_DIR}" -mindepth 1 -maxdepth 1 \
    ! -name "flydsl" \
    ! -name "include" \
    -exec rm -rf {} +

cd "${REPO_ROOT}"

echo ""
echo "✓ Build complete!"
echo "✓ flir-opt: ${BUILD_DIR}/bin/flir-opt"
echo "✓ Python bindings: unified _flir_ir module (1 exported symbol)"
echo ""

# Build a compliant manylinux wheel if possible
if [[ "${FLIR_BUILD_WHEEL:-0}" == "1" ]]; then
    echo "Building and repairing wheel and sdist for release..."
    rm -rf "${REPO_ROOT}/dist"
    cd "${REPO_ROOT}"
    export FLIR_IN_BUILD_SH=1

    # Strip debug symbols from shared libraries
    echo "Stripping shared libraries..."
    if command -v strip &> /dev/null; then
        find "${PYTHON_PACKAGE_DIR}" -name "*.so*" -exec strip --strip-unneeded {} + || true
    else
        echo "Warning: strip not found; skipping binary stripping."
    fi

    # Generate both Wheel and Source distribution
    python3 setup.py bdist_wheel sdist

    if command -v auditwheel &> /dev/null; then
        echo "Repairing wheel with auditwheel..."
        WHEELHOUSE="${REPO_ROOT}/dist/wheelhouse"
        mkdir -p "${WHEELHOUSE}"
        
        WHEEL_FILE=$(ls dist/*.whl | head -n 1)
        auditwheel repair "$WHEEL_FILE" -w "${WHEELHOUSE}" \
            --exclude "libamdhip64.so.*" \
            --exclude "libhsa-runtime64.so.*" \
            --exclude "libdrm_amdgpu.so.*" \
            || { echo "Warning: auditwheel repair failed; leaving the original wheel in dist/"; rm -rf "${WHEELHOUSE}"; }
        
        if ls "${WHEELHOUSE}"/*.whl &> /dev/null; then
            rm -f dist/*linux_x86_64.whl
            mv "${WHEELHOUSE}"/*.whl dist/
            rm -rf "${WHEELHOUSE}"
            echo "✓ Compliant manylinux wheel and sdist are ready in dist/"
        fi
    else
        echo "Warning: auditwheel not found. Original dist files remain in dist/."
    fi
fi

echo "Embedded MLIR runtime location: ${PYTHON_PACKAGE_DIR}/flydsl/_mlir"
echo ""
echo "Recommended (no manual PYTHONPATH):"
echo "  cd ${REPO_ROOT} && python3 -m pip install -e ."
echo ""
echo "Build a wheel:"
echo "  cd ${REPO_ROOT} && python3 setup.py bdist_wheel"
echo "  # wheel will be under: ${REPO_ROOT}/dist/"
echo ""
echo "Fallback (no install):"
echo "  export PYTHONPATH=${PYTHON_PACKAGE_DIR}:${REPO_ROOT}/flydsl/src:${REPO_ROOT}:\$PYTHONPATH"
