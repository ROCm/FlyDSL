#!/bin/bash
# Setup Python package structure for unified _flir_ir module
#
# This script copies MLIR Python sources into our package tree so that
# relative imports (e.g. `from ._mlir_libs import _mlir`) resolve correctly
# under the flydsl._mlir namespace.
#
# Usage: ./setup_python_package.sh <build_dir> <mlir_build_dir>

set -e

BUILD_DIR="${1:-.flir/build}"
MLIR_BUILD_DIR="${2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Derive MLIR source directory from MLIR_PATH (set by build.sh) or MLIR_BUILD_DIR
if [ -n "${MLIR_PATH:-}" ]; then
    MLIR_SRC_DIR="${MLIR_PATH}/../mlir/python/mlir"
elif [ -n "${MLIR_BUILD_DIR}" ]; then
    # Assume MLIR source is a sibling: llvm-project/mlir/python/mlir
    MLIR_SRC_DIR="$(cd "${MLIR_BUILD_DIR}/.." && pwd)/mlir/python/mlir"
else
    echo "Error: MLIR_BUILD_DIR must be specified as the second argument"
    exit 1
fi

if [ ! -d "${MLIR_SRC_DIR}" ]; then
    echo "Error: MLIR Python sources not found at ${MLIR_SRC_DIR}"
    exit 1
fi

PYTHON_PKG="${BUILD_DIR}/python_packages/flydsl/flydsl/_mlir"
DIALECTS_DIR="${PYTHON_PKG}/dialects"
EXTRAS_DIR="${PYTHON_PKG}/extras"

echo "Setting up Python package structure..."

# Helper: copy all .py files and subdirectories from $1 into $2
# Skips __init__.py and __pycache__ so we can provide our own __init__.py.
copy_all() {
  local src="$1" dst="$2"
  # Copy .py files (excluding __init__.py)
  for f in "${src}"/*.py; do
    [ -f "$f" ] || continue
    [ "$(basename "$f")" = "__init__.py" ] && continue
    cp -f "$f" "${dst}/$(basename "$f")" 2>/dev/null || true
  done
  # Copy subdirectories (excluding __pycache__ and _mlir_libs which is managed by CMake)
  for d in "${src}"/*/; do
    [ -d "$d" ] || continue
    local name
    name="$(basename "$d")"
    [ "$name" = "__pycache__" ] && continue
    [ "$name" = "_mlir_libs" ] && continue
    rm -rf "${dst}/${name}" 2>/dev/null || true
    cp -rf "${src}/${name}" "${dst}/${name}"
  done
}

# Create directories
mkdir -p "${DIALECTS_DIR}" "${EXTRAS_DIR}"

# --- Core Python modules ---
copy_all "${MLIR_SRC_DIR}" "${PYTHON_PKG}"

# --- Extras package ---
echo "# Extras package" > "${EXTRAS_DIR}/__init__.py"
copy_all "${MLIR_SRC_DIR}/extras" "${EXTRAS_DIR}"

# --- Dialect package ---
echo "# Dialects package" > "${DIALECTS_DIR}/__init__.py"
copy_all "${MLIR_SRC_DIR}/dialects" "${DIALECTS_DIR}"

# --- TableGen generated files (must copy from MLIR build) ---
MLIR_GEN="${MLIR_BUILD_DIR}/tools/mlir/python/dialects"
for f in gpu arith scf memref vector math func cf builtin llvm rocdl; do
  cp "${MLIR_GEN}/_${f}_ops_gen.py" "${DIALECTS_DIR}/" 2>/dev/null || true
  cp "${MLIR_GEN}/_${f}_enum_gen.py" "${DIALECTS_DIR}/" 2>/dev/null || true
done

# --- FLIR dialect ---
FLIR_GEN="${BUILD_DIR}/python_bindings/dialects"
cp "${FLIR_GEN}/_flir_ops_gen.py" "${DIALECTS_DIR}/" 2>/dev/null || true
cp "${FLIR_GEN}/_flir_enum_gen.py" "${DIALECTS_DIR}/" 2>/dev/null || true
cp "${SCRIPT_DIR}/dialects/flir.py" "${DIALECTS_DIR}/"

# --- Proxy modules for _mlir_libs ---
# LLVM's wrappers expect `_mlir_libs._mlir.ir`, `_mlir_libs._mlirDialectsGPU`, etc.
# These don't exist as real files — they're nanobind submodules inside _flir_ir.so.
# Instead of hacking sys.modules, we place thin proxy packages/modules that
# re-export from _flir_ir's auto-registered paths. This way LLVM's original
# wrapper files work unmodified.
PROXIES_DIR="${SCRIPT_DIR}/mlir_proxies"
MLIR_LIBS_DIR="${PYTHON_PKG}/_mlir_libs"
rm -rf "${MLIR_LIBS_DIR}/_mlir"
cp -rf "${PROXIES_DIR}/_mlir" "${MLIR_LIBS_DIR}/_mlir"
cp -f "${PROXIES_DIR}/_mlirDialectsGPU.py" "${MLIR_LIBS_DIR}/_mlirDialectsGPU.py"
cp -f "${PROXIES_DIR}/_mlirDialectsLLVM.py" "${MLIR_LIBS_DIR}/_mlirDialectsLLVM.py"
cp -f "${PROXIES_DIR}/_mlirExecutionEngine.py" "${MLIR_LIBS_DIR}/_mlirExecutionEngine.py"

echo "Python package setup complete!"
