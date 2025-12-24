#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep all generated artifacts under one directory by default.
# - You can override with:
#   FLIR_OUT_DIR=.flir   (relative to repo root) or an absolute path
#   FLIR_BUILD_DIR=...     (absolute path to CMake build dir)
DEFAULT_OUT_DIR="${SCRIPT_DIR}/.flir"
# Backward compatible: honor legacy ROCDSL_OUT_DIR/ROCDSL_BUILD_DIR if FLIR_* not set.
OUT_DIR="${FLIR_OUT_DIR:-${ROCDSL_OUT_DIR:-${DEFAULT_OUT_DIR}}}"
if [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${SCRIPT_DIR}/${OUT_DIR}"
fi
BUILD_DIR="${FLIR_BUILD_DIR:-${ROCDSL_BUILD_DIR:-${OUT_DIR}/build}}"
if [[ "${BUILD_DIR}" != /* ]]; then
  BUILD_DIR="${SCRIPT_DIR}/${BUILD_DIR}"
fi

# Set up environment
if [ -z "$MLIR_PATH" ]; then
    # Default path based on build_llvm.sh
    DEFAULT_MLIR_PATH="$(cd "${SCRIPT_DIR}/.." && pwd)/llvm-project/buildmlir"
    if [ -d "$DEFAULT_MLIR_PATH" ]; then
        echo "MLIR_PATH not set. Using default: $DEFAULT_MLIR_PATH"
        export MLIR_PATH="$DEFAULT_MLIR_PATH"
    else
        echo "Error: MLIR_PATH not set and default location ($DEFAULT_MLIR_PATH) not found."
        echo "Please run ./build_llvm.sh first or set MLIR_PATH."
        exit 1
    fi
fi

# Note: No longer need to install MLIR Python packages separately!
# The new build system embeds all MLIR Python dependencies

# Build C++ components
mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"
# NOTE: build dir may be nested (e.g. .flir/build), so `..` may not be repo root.
cmake "${SCRIPT_DIR}" \
    -DMLIR_DIR="$MLIR_PATH/lib/cmake/mlir" \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_RUNTIME=OFF

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

# Set up PYTHONPATH for the embedded Python package root (contains `_mlir/` and `pyflir/`)
PYTHON_PACKAGE_DIR="${BUILD_DIR}/python_packages/pyflir"

# Ensure the python package root contains the embedded MLIR package (_mlir) and our sources (pyflir, mlir shim).
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

# Copy pyflir python package into the package root as pyflir/
cp -r "${SCRIPT_DIR}/pyflir/src/pyflir" "${PYTHON_PACKAGE_DIR}/" || { echo "Failed to copy pyflir/src/pyflir"; exit 1; }

cd "${SCRIPT_DIR}"

echo ""
echo "✓ Build complete!"
echo "✓ flir-opt: ${BUILD_DIR}/bin/flir-opt"
echo "✓ Python bindings built with embedded MLIR dependencies"
echo ""
echo "Embedded MLIR runtime location: ${PYTHON_PACKAGE_DIR}/_mlir"
echo ""
echo "Recommended (no manual PYTHONPATH):"
echo "  cd ${SCRIPT_DIR} && python3 -m pip install -e ."
echo ""
echo "Build a wheel:"
echo "  cd ${SCRIPT_DIR} && python3 setup.py bdist_wheel"
echo "  # wheel will be under: ${SCRIPT_DIR}/dist/"
echo ""
echo "Fallback (no install):"
echo "  export PYTHONPATH=${PYTHON_PACKAGE_DIR}:${SCRIPT_DIR}/pyflir/src:${SCRIPT_DIR}:\$PYTHONPATH"
