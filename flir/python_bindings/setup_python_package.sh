#!/bin/bash
# Setup Python package structure for unified _flir_ir module
#
# This script sets up symlinks and copies files to create a complete Python
# package that uses our monolithic _flir_ir.so module.
#
# Key insight: LLVM's gpu/llvm dialects are directories (gpu/__init__.py),
# not single files. This allows us to symlink them directly because the
# relative import depth matches (... from 3 levels deep works correctly).
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

# Create directories
mkdir -p "${DIALECTS_DIR}" "${EXTRAS_DIR}"

# --- Core Python modules (symlinks) ---
for f in ir.py passmanager.py rewrite.py execution_engine.py; do
  ln -sf "${MLIR_SRC_DIR}/${f}" "${PYTHON_PKG}/${f}"
done

# --- Extras package (symlinks) ---
echo "# Extras package" > "${EXTRAS_DIR}/__init__.py"
for f in meta.py types.py; do
  ln -sf "${MLIR_SRC_DIR}/extras/${f}" "${EXTRAS_DIR}/${f}" 2>/dev/null || true
done
rm -rf "${EXTRAS_DIR}/dialects" 2>/dev/null || true
ln -sf "${MLIR_SRC_DIR}/extras/dialects" "${EXTRAS_DIR}/dialects" 2>/dev/null || true

# --- Dialect package setup ---
echo "# Dialects package" > "${DIALECTS_DIR}/__init__.py"
ln -sf "${MLIR_SRC_DIR}/dialects/_ods_common.py" "${DIALECTS_DIR}/_ods_common.py"

# Single-file dialects (symlinks)
for f in arith builtin cf func math memref scf vector rocdl llvm; do
  ln -sf "${MLIR_SRC_DIR}/dialects/${f}.py" "${DIALECTS_DIR}/${f}.py" 2>/dev/null || true
done

# Directory dialects (gpu) - symlink the directories
# These use ... imports which require 3-level depth
for d in gpu; do
  rm -rf "${DIALECTS_DIR}/${d}" 2>/dev/null || true
  if [ -d "${MLIR_SRC_DIR}/dialects/${d}" ]; then
    ln -sf "${MLIR_SRC_DIR}/dialects/${d}" "${DIALECTS_DIR}/${d}"
  fi
done

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

echo "Python package setup complete!"
