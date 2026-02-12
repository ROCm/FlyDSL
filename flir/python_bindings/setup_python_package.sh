#!/bin/bash
# Setup Python package structure for unified _flir_ir module
#
# This script copies MLIR Python sources into our package tree so that
# relative imports (e.g. `from ._mlir_libs import _mlir`) resolve correctly
# under the flydsl._mlir namespace.
#
# Usage: ./setup_python_package.sh <build_dir> [mlir_build_dir] [mlir_src_dir]
#
# Arguments:
#   build_dir       FLIR CMake build directory (required)
#   mlir_build_dir  MLIR CMake build directory (optional, for TableGen output)
#   mlir_src_dir    MLIR Python source directory, i.e. <llvm-project>/mlir/python/mlir
#                   (optional, derived from MLIR_PATH env var if not provided)
#
# Environment variables (used as fallbacks when arguments are not provided):
#   MLIR_PATH       MLIR install prefix (e.g. llvm-project/mlir_install)
#   MLIR_SRC_DIR    Explicit path to <llvm-project>/mlir/python/mlir
#   MLIR_GEN_DIR    Explicit path to MLIR TableGen output (dialects directory)

set -e

BUILD_DIR="${1:-.flir/build}"
MLIR_BUILD_DIR="${2:-}"
MLIR_SRC_DIR_ARG="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Resolve MLIR Python source directory
# Priority: argument > MLIR_SRC_DIR env > derive from MLIR_PATH env
# ---------------------------------------------------------------------------
resolve_mlir_src_dir() {
    # 1. Explicit argument
    if [ -n "${MLIR_SRC_DIR_ARG}" ]; then
        echo "${MLIR_SRC_DIR_ARG}"
        return
    fi

    # 2. Explicit env var
    if [ -n "${MLIR_SRC_DIR:-}" ]; then
        echo "${MLIR_SRC_DIR}"
        return
    fi

    # 3. Derive from MLIR_PATH (install prefix -> sibling mlir source)
    if [ -n "${MLIR_PATH:-}" ]; then
        local candidate
        candidate="$(cd "${MLIR_PATH}/.." 2>/dev/null && pwd)/mlir/python/mlir"
        if [ -d "${candidate}" ]; then
            echo "${candidate}"
            return
        fi
    fi

    # 4. Derive from MLIR_BUILD_DIR (build dir -> sibling mlir source)
    if [ -n "${MLIR_BUILD_DIR}" ]; then
        local candidate
        candidate="$(cd "${MLIR_BUILD_DIR}/.." 2>/dev/null && pwd)/mlir/python/mlir"
        if [ -d "${candidate}" ]; then
            echo "${candidate}"
            return
        fi
    fi

    return 1
}

MLIR_SRC_DIR="$(resolve_mlir_src_dir)" || {
    echo "Error: Cannot find MLIR Python source directory." >&2
    echo "" >&2
    echo "Provide it via one of (in priority order):" >&2
    echo "  1. Third argument:   $0 <build_dir> <mlir_build_dir> <mlir_src_dir>" >&2
    echo "  2. Env var:          export MLIR_SRC_DIR=/path/to/llvm-project/mlir/python/mlir" >&2
    echo "  3. Env var:          export MLIR_PATH=/path/to/mlir_install  (derives source as sibling)" >&2
    exit 1
}

if [ ! -d "${MLIR_SRC_DIR}" ]; then
    echo "Error: MLIR Python sources not found at ${MLIR_SRC_DIR}" >&2
    exit 1
fi
echo "MLIR Python sources: ${MLIR_SRC_DIR}"

# ---------------------------------------------------------------------------
# Resolve MLIR TableGen output directory (for _*_ops_gen.py / _*_enum_gen.py)
# Priority: MLIR_GEN_DIR env > derive from MLIR_BUILD_DIR
# ---------------------------------------------------------------------------
if [ -n "${MLIR_GEN_DIR:-}" ]; then
    MLIR_GEN="${MLIR_GEN_DIR}"
elif [ -n "${MLIR_BUILD_DIR}" ]; then
    MLIR_GEN="${MLIR_BUILD_DIR}/tools/mlir/python/dialects"
else
    MLIR_GEN=""
fi

if [ -n "${MLIR_GEN}" ] && [ ! -d "${MLIR_GEN}" ]; then
    echo "Warning: MLIR TableGen output not found at ${MLIR_GEN}" >&2
    echo "         Dialect *_ops_gen.py files will not be copied." >&2
    MLIR_GEN=""
fi

# ---------------------------------------------------------------------------
# Package directories
# ---------------------------------------------------------------------------
PYTHON_PKG="${BUILD_DIR}/python_packages/flydsl/flydsl/_mlir"
DIALECTS_DIR="${PYTHON_PKG}/dialects"
EXTRAS_DIR="${PYTHON_PKG}/extras"
MLIR_LIBS_DIR="${PYTHON_PKG}/_mlir_libs"

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
mkdir -p "${DIALECTS_DIR}" "${EXTRAS_DIR}" "${MLIR_LIBS_DIR}"

# --- Core Python modules ---
copy_all "${MLIR_SRC_DIR}" "${PYTHON_PKG}"

# --- Extras package ---
echo "# Extras package" > "${EXTRAS_DIR}/__init__.py"
copy_all "${MLIR_SRC_DIR}/extras" "${EXTRAS_DIR}"

# --- Dialect package ---
echo "# Dialects package" > "${DIALECTS_DIR}/__init__.py"
copy_all "${MLIR_SRC_DIR}/dialects" "${DIALECTS_DIR}"

# --- TableGen generated files (must copy from MLIR build) ---
if [ -n "${MLIR_GEN}" ]; then
    echo "Copying MLIR TableGen generated files from ${MLIR_GEN}..."
    for f in gpu arith scf memref vector math func cf builtin llvm rocdl; do
      cp "${MLIR_GEN}/_${f}_ops_gen.py" "${DIALECTS_DIR}/" 2>/dev/null || true
      cp "${MLIR_GEN}/_${f}_enum_gen.py" "${DIALECTS_DIR}/" 2>/dev/null || true
    done
fi

# --- FLIR dialect ---
FLIR_GEN="${BUILD_DIR}/python_bindings/dialects"
cp "${FLIR_GEN}/_flir_ops_gen.py" "${DIALECTS_DIR}/" 2>/dev/null || true
cp "${FLIR_GEN}/_flir_enum_gen.py" "${DIALECTS_DIR}/" 2>/dev/null || true
cp "${SCRIPT_DIR}/dialects/flir.py" "${DIALECTS_DIR}/"

# --- _mlir_libs package init ---
# Always copy the canonical _mlir_libs_init.py as __init__.py.
# This ensures the latest version is deployed even when CMake's post-build
# step doesn't re-run (e.g. when only this script is invoked, or the
# _flir_ir target is already up-to-date).
INIT_SRC="${SCRIPT_DIR}/flir_ir/_mlir_libs_init.py"
if [ -f "${INIT_SRC}" ]; then
    cp -f "${INIT_SRC}" "${MLIR_LIBS_DIR}/__init__.py"
else
    echo "Warning: _mlir_libs_init.py not found at ${INIT_SRC}" >&2
    echo "         _mlir_libs/__init__.py may be outdated or missing" >&2
fi

# --- Proxy modules for _mlir_libs ---
# LLVM's wrappers expect `_mlir_libs._mlir.ir`, `_mlir_libs._mlirDialectsGPU`, etc.
# These don't exist as real files — they're nanobind submodules inside _flir_ir.so.
# Instead of hacking sys.modules, we place thin proxy packages/modules that
# re-export from _flir_ir's auto-registered paths. This way LLVM's original
# wrapper files work unmodified.
PROXIES_DIR="${SCRIPT_DIR}/mlir_proxies"
rm -rf "${MLIR_LIBS_DIR}/_mlir"
cp -rf "${PROXIES_DIR}/_mlir" "${MLIR_LIBS_DIR}/_mlir"
cp -f "${PROXIES_DIR}/_mlirDialectsGPU.py" "${MLIR_LIBS_DIR}/_mlirDialectsGPU.py"
cp -f "${PROXIES_DIR}/_mlirDialectsLLVM.py" "${MLIR_LIBS_DIR}/_mlirDialectsLLVM.py"
cp -f "${PROXIES_DIR}/_mlirExecutionEngine.py" "${MLIR_LIBS_DIR}/_mlirExecutionEngine.py"

echo "Python package setup complete!"
