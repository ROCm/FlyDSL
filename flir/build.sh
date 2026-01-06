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
OUT_DIR="${FLIR_OUT_DIR:-${FLIR_OUT_DIR:-${DEFAULT_OUT_DIR}}}"
if [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${REPO_ROOT}/${OUT_DIR}"
fi
BUILD_DIR="${FLIR_BUILD_DIR:-${FLIR_BUILD_DIR:-${OUT_DIR}/build}}"
if [[ "${BUILD_DIR}" != /* ]]; then
  BUILD_DIR="${REPO_ROOT}/${BUILD_DIR}"
fi

# Set up environment
if [ -z "$MLIR_PATH" ]; then
    # Default path based on build_llvm.sh
    DEFAULT_MLIR_PATH="${REPO_ROOT}/llvm-project/buildmlir"
    if [ -d "$DEFAULT_MLIR_PATH" ]; then
        echo "MLIR_PATH not set. Using default: $DEFAULT_MLIR_PATH"
        export MLIR_PATH="$DEFAULT_MLIR_PATH"
    else
        echo "Error: MLIR_PATH not set and default location ($DEFAULT_MLIR_PATH) not found."
        echo "Please run bash scripts/build_llvm.sh from the repo root first, or set MLIR_PATH."
        exit 1
    fi
fi

# Note: No longer need to install MLIR Python packages separately!
# The new build system embeds all MLIR Python dependencies

# Build C++ components
mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"

# Enable ROCm by default when ROCm is present on the system.
# This is required for GPU execution tests (HIP runtime) to work.
ENABLE_ROCM_FLAG=OFF
if [[ -d "/opt/rocm" ]] || command -v hipcc &> /dev/null; then
  ENABLE_ROCM_FLAG=ON
fi

cmake "${SCRIPT_DIR}" \
    -DMLIR_DIR="$MLIR_PATH/lib/cmake/mlir" \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_RUNTIME=OFF \
    -DENABLE_ROCM="${ENABLE_ROCM_FLAG}"

# Build core targets (skip type stub generation which may fail)
echo "Building core libraries..."
cmake --build . --target FlirDialect -j$(nproc) || { echo "Failed to build FlirDialect"; exit 1; }
cmake --build . --target FlirTransforms -j$(nproc) || { echo "Failed to build FlirTransforms"; exit 1; }
cmake --build . --target FlirPythonCAPI -j$(nproc) || { echo "Failed to build FlirPythonCAPI"; exit 1; }

# Build Python extension modules
echo "Building Python extensions..."
cmake --build . --target FlirPythonModules.extension._mlir.dso -j$(nproc) || { echo "Failed to build _mlir.dso"; exit 1; }
cmake --build . --target FlirPythonModules.extension._mlirDialectsGPU.dso -j$(nproc) || { echo "Failed to build _mlirDialectsGPU.dso"; exit 1; }
cmake --build . --target FlirPythonModules.extension._mlirDialectsLLVM.dso -j$(nproc) || { echo "Failed to build _mlirDialectsLLVM.dso"; exit 1; }
cmake --build . --target FlirPythonModules.extension._mlirGPUPasses.dso -j$(nproc) || true
cmake --build . --target FlirPythonModules.extension._mlirExecutionEngine.dso -j$(nproc) || true
cmake --build . --target FlirPythonModules.extension._flirPasses.dso -j$(nproc) || { echo "Failed to build _flirPasses.dso"; exit 1; }

# Build flir-opt tool (used by run_tests.sh MLIR file tests)
cmake --build . --target flir-opt -j$(nproc) || true

# Copy Python source files and MLIR modules
echo "Copying Python sources..."
# Core python modules (ir.py, passmanager.py, rewrite.py, etc)
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Core.Python -j$(nproc) || { echo "Failed to build MLIR core python sources"; exit 1; }
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Core.Python.Extras -j$(nproc) || true

# Upstream dialect wrappers (_mlir.dialects.*) used by tests/utilities
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.builtin -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.builtin.ops_gen -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.arith -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.arith.ops_gen -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.math -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.math.ops_gen -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.memref -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.memref.ops_gen -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.func -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.func.ops_gen -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.cf -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.cf.ops_gen -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.scf -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.scf.ops_gen -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.gpu -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.gpu.ops_gen -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.vector -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.vector.ops_gen -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.llvm -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.llvm.ops_gen -j$(nproc) || true

# Project dialect python bindings
cmake --build . --target FlirPythonModules.sources.FlirPythonSources.flir -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.FlirPythonSources.flir.ops_gen -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.rocdl -j$(nproc) || true
cmake --build . --target FlirPythonModules.sources.MLIRPythonSources.Dialects.rocdl.ops_gen -j$(nproc) || true

# Set up PYTHONPATH for the embedded Python package root (contains `_mlir/` and `flydsl/`)
PYTHON_PACKAGE_DIR="${BUILD_DIR}/python_packages/flydsl"

# Ensure the python package root contains the embedded MLIR package (_mlir) and our sources (flydsl, mlir shim).
if [ ! -d "${PYTHON_PACKAGE_DIR}" ]; then
    echo "Error: expected python package root not found: ${PYTHON_PACKAGE_DIR}"
    echo "   (Did the build generate embedded MLIR python modules?)"
    exit 1
fi

# Clean any previously overlaid sources at the root (keep embedded _mlir and include/).
find "${PYTHON_PACKAGE_DIR}" -mindepth 1 -maxdepth 1 \
    ! -name "_mlir" \
    ! -name "include" \
    -exec rm -rf {} +

# Copy flydsl python package into the package root as flydsl/
cp -r "${REPO_ROOT}/flydsl/src/flydsl" "${PYTHON_PACKAGE_DIR}/" || { echo "Failed to copy flydsl/src/flydsl"; exit 1; }

cd "${REPO_ROOT}"

echo ""
echo "✓ Build complete!"
echo "✓ flir-opt: ${BUILD_DIR}/bin/flir-opt"
echo "✓ Python bindings built with embedded MLIR dependencies"
echo ""

# Build a compliant manylinux wheel if possible
# DEFAULT: 0 (Fast build, no wheel). Set FLIR_BUILD_WHEEL=1 for PyPI release.
if [[ "${FLIR_BUILD_WHEEL:-0}" == "1" ]]; then
    echo "Building and repairing wheel and sdist for release..."
    # Clean up old dist artifacts to avoid confusion
    rm -rf "${REPO_ROOT}/dist"
    cd "${REPO_ROOT}"
    # Set FLIR_IN_BUILD_SH=1 to prevent setup.py from recursively calling build.sh
    export FLIR_IN_BUILD_SH=1

    # Reduce wheel size (wheel build only; does not affect local editable installs):
    # - Drop the non-versioned CAPI .so which is typically a symlink/copy of the
    #   versioned library (packaging both can double wheel size).
    # - Strip debug symbols from shared libraries (huge savings).
    echo "Stripping shared libraries..."
    if command -v strip &> /dev/null; then
        rm -f "${PYTHON_PACKAGE_DIR}/_mlir/_mlir_libs/libFlirPythonCAPI.so" || true
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
        
        # Repair the specific wheel generated (avoiding glob issues)
        WHEEL_FILE=$(ls dist/*.whl | head -n 1)
        # IMPORTANT:
        # We intentionally do NOT bundle ROCm user-space runtime libraries into the wheel.
        # If bundled, they can conflict with an existing ROCm runtime (e.g. PyTorch),
        # leading to runtime failures like hipErrorNoDevice. Instead, rely on the
        # system ROCm installation in ROCm-enabled environments.
        #
        # Excluding these libs also avoids auditwheel failing manylinux checks due
        # to too-recent symbol versions inside ROCm-provided shared libraries.
        auditwheel repair "$WHEEL_FILE" -w "${WHEELHOUSE}" \
            --exclude "libamdhip64.so.*" \
            --exclude "libhsa-runtime64.so.*" \
            --exclude "libdrm_amdgpu.so.*" \
            || { echo "Warning: auditwheel repair failed; leaving the original wheel in dist/"; rm -rf "${WHEELHOUSE}"; }
        
        # Replace the original wheel with the repaired ones
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

echo "Embedded MLIR runtime location: ${PYTHON_PACKAGE_DIR}/_mlir"
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

