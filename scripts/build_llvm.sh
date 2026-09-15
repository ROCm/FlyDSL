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

# Read LLVM commit hash from thirdparty/llvm-build-info.json (upstream entry)
LLVM_BUILD_INFO="${REPO_ROOT}/thirdparty/llvm-build-info.json"
LLVM_COMMIT_DEFAULT=$(python3 -c "import json; print(json.load(open('${LLVM_BUILD_INFO}'))['upstream']['llvm_hash'])")
LLVM_REF="${LLVM_REF:-${LLVM_COMMIT:-$LLVM_COMMIT_DEFAULT}}"
LLVM_EXT_DIR="${REPO_ROOT}/thirdparty/llvm-extensions"

# FLYDSL_LLVM_NO_EXT=1 builds stock upstream, applying nothing. It is the
# control arm for "is this regression ours?": same pin, same everything,
# extensions the only variable. The series is all-or-nothing on purpose -- a
# per-extension skip would make "which extensions were in this build?" a
# question with more than two answers, and a stale extension is meant to be
# rebased (scripts/llvm_extension.sh --rebase), not routed around. An extension
# that is applied can still be switched off at run time; see LLVM_EXTENSIONS.
LLVM_NO_EXT="${FLYDSL_LLVM_NO_EXT:-0}"

# Local LLVM extensions, applied in the order listed here. These carry the parts
# of FlyDSL's end-to-end performance work that belong in LLVM rather than in the
# DSL. Only two kinds of change belong here: one already submitted upstream and
# waiting on review, or one specific enough to FlyDSL that upstream would not
# take it. Every entry says which, so the next pin bump can drop what landed.
# Every entry must also be switchable at run time (cl::opt or getenv), so a
# regression can be turned off without rebuilding LLVM -- enforced by
# scripts/check_llvm_extensions.py.
#
# Each extension is a diff against the tree with its predecessors already
# applied, so the order is part of their meaning -- keep it deliberate, and
# append rather than insert unless a new one genuinely has to precede another.
#
# To add one: bash scripts/llvm_extension.sh <slug>
# See CONTRIBUTING.md, "Add an LLVM Extension".
LLVM_EXTENSIONS=(
    # Pass the resolved lld path as argv[0] instead of the literal "ld.lld", so
    # MLIR's ROCDL target finds the linker when it is not on PATH under that
    # name. Case 1 (upstreamable). Upstream: not yet submitted.
    rocdl-lld-argv0.patch
)
LLVM_BUILD_PROFILE="${LLVM_BUILD_PROFILE:-full}"

case "${LLVM_BUILD_PROFILE}" in
  full)
    LLVM_ENABLE_PROJECTS="${LLVM_ENABLE_PROJECTS:-mlir;clang;lld}"
    LLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD:-X86;NVPTX;AMDGPU}"
    # Use `-` rather than `:-` so callers can explicitly disable runtimes.
    LLVM_ENABLE_RUNTIMES="${LLVM_ENABLE_RUNTIMES-compiler-rt}"
    ;;
  amd-minimal)
    LLVM_ENABLE_PROJECTS="mlir"
    LLVM_TARGETS_TO_BUILD="X86;AMDGPU"
    LLVM_ENABLE_RUNTIMES=
    ;;
  *)
    echo "Unknown LLVM_BUILD_PROFILE: ${LLVM_BUILD_PROFILE}" >&2
    exit 2
    ;;
esac

echo "Base directory: $BASE_DIR"
echo "LLVM Source:    $LLVM_SRC_DIR"
echo "LLVM Build:     $LLVM_BUILD_DIR"
echo "LLVM Install:   $LLVM_INSTALL_DIR"
echo "LLVM Tarball:   $LLVM_INSTALL_TGZ"
echo "LLVM Ref:       $LLVM_REF"
echo "LLVM Profile:   $LLVM_BUILD_PROFILE"
echo "LLVM Projects:  $LLVM_ENABLE_PROJECTS"
echo "LLVM Targets:   $LLVM_TARGETS_TO_BUILD"
echo "LLVM Runtimes:  ${LLVM_ENABLE_RUNTIMES:-<none>}"

# 1. Clone LLVM
LLVM_REMOTE="${LLVM_REMOTE:-https://github.com/llvm/llvm-project.git}"

# A leftover partial ("promisor") clone is unusable here: every checkout, patch
# and rev-parse would trigger per-blob lazy fetches against github.com. Unsetting
# the config does not bring the missing blobs back, so start over instead.
if [ -d "$LLVM_SRC_DIR/.git" ] && \
   [ -n "$(git -C "$LLVM_SRC_DIR" config --get remote.origin.promisor || true)" ]; then
    echo "Discarding partial (promisor) llvm-project checkout at ${LLVM_SRC_DIR} ..."
    rm -rf "$LLVM_SRC_DIR"
fi

if [ ! -d "$LLVM_SRC_DIR" ]; then
    echo "Preparing llvm-project checkout for ${LLVM_REF} ..."
    git init "$LLVM_SRC_DIR"
    pushd "$LLVM_SRC_DIR"
    git remote add origin "$LLVM_REMOTE"
else
    pushd "$LLVM_SRC_DIR"
fi

# Plain shallow fetch. Do NOT add --filter=blob:none here: a blob-filtered fetch
# of an arbitrary SHA makes the server build an uncached pack, and the checkout
# that follows then lazily re-fetches every file in the tree one batch at a time.
# Measured on CI: `--depth 1` alone downloads llvm-project in ~100s, while
# `--depth 1 --filter=blob:none` did not finish within 100 minutes.
LLVM_FETCH_ARGS=(--depth 1)

if [[ "$LLVM_REF" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Checking for local LLVM commit ${LLVM_REF} ..."
    if ! git cat-file -e "${LLVM_REF}^{commit}" 2>/dev/null; then
        echo "Fetching commit ${LLVM_REF} ..."
        git fetch "${LLVM_FETCH_ARGS[@]}" origin "${LLVM_REF}"
    else
        echo "LLVM commit ${LLVM_REF} is already available locally."
    fi
    echo "Checking out LLVM commit ${LLVM_REF} ..."
    git checkout --force "${LLVM_REF}"
else
    echo "Fetching ref ${LLVM_REF} ..."
    git fetch "${LLVM_FETCH_ARGS[@]}" origin "${LLVM_REF}"
    git checkout --force FETCH_HEAD
fi

# Patches are replayed from a pristine tree rather than detected as
# already-applied. A per-patch `git apply --reverse --check` probe cannot work
# for a series: once patch N+1 has rewritten the same lines, patch N no longer
# reverse-applies, so the probe reports it as missing and the re-apply fails.
# The forced checkout above is what makes the replay total: it discards the
# previous run's applied extensions and moves HEAD even when the authoring
# script left a commit behind. A plain checkout refuses both, and `set -e`
# would then abort here on every subsequent run with no way to recover.
#
# Tracked files only -- deliberately NOT `git clean -fd`, which would delete
# build-flydsl/ and mlir_install/ (both live inside this checkout, and
# mlir_install/ is not covered by LLVM's ignore rules) and turn every
# incremental build into a full one.

# A .patch file present but absent from LLVM_EXTENSIONS would be silently
# ignored, which is the most likely way an added extension goes missing (the
# array edit is forgotten in review). Check both directions before applying.
for ext_file in "${LLVM_EXT_DIR}"/*.patch; do
    [ -e "${ext_file}" ] || continue
    ext_base="$(basename "${ext_file}")"
    listed=0
    for ext_name in "${LLVM_EXTENSIONS[@]}"; do
        [ "${ext_name}" = "${ext_base}" ] && listed=1 && break
    done
    if [ "${listed}" -eq 0 ]; then
        echo "Error: extension file is not listed in LLVM_EXTENSIONS: ${ext_base}" >&2
        echo "       Add it to the LLVM_EXTENSIONS array in scripts/build_llvm.sh, or delete it." >&2
        exit 1
    fi
done
for ext_name in "${LLVM_EXTENSIONS[@]}"; do
    if [ ! -f "${LLVM_EXT_DIR}/${ext_name}" ]; then
        echo "Error: LLVM_EXTENSIONS lists a file that does not exist: ${ext_name}" >&2
        echo "       Expected at: ${LLVM_EXT_DIR}/${ext_name}" >&2
        exit 1
    fi
done

# Validation above is deliberately unconditional: building without the
# extensions must not also suppress "this file is not listed", which is a
# mistake in the tree rather than a choice made at build time.
if [[ "${LLVM_NO_EXT}" == "1" ]]; then
    echo "LLVM Extensions: none (FLYDSL_LLVM_NO_EXT=1)"
    LLVM_EXTENSIONS=()
else
    echo "LLVM Extensions: ${#LLVM_EXTENSIONS[@]}"
fi
for ext_name in "${LLVM_EXTENSIONS[@]}"; do
    echo "  applying ${ext_name}"
    if ! git apply "${LLVM_EXT_DIR}/${ext_name}"; then
        echo "" >&2
        echo "Error: LLVM extension failed to apply: ${ext_name}" >&2
        echo "       LLVM pin: ${LLVM_REF}" >&2
        echo "" >&2
        echo "  Either the extension is stale against this pin, or it landed upstream." >&2
        echo "  To rebase it:   bash scripts/llvm_extension.sh --rebase ${ext_name}" >&2
        echo "  If it landed:   delete it and drop it from LLVM_EXTENSIONS in scripts/build_llvm.sh" >&2
        exit 1
    fi
done

LLVM_COMMIT_RESOLVED=$(git rev-parse HEAD)
popd
echo "LLVM Commit:    $LLVM_COMMIT_RESOLVED"

# 2. Create Build Directory
mkdir -p "$LLVM_BUILD_DIR"
cd "$LLVM_BUILD_DIR"

# 3. Configure CMake
echo "Configuring LLVM..."

# Install dependencies for Python bindings
echo "Installing Python dependencies..."
NANOBIND_VERSION="${NANOBIND_VERSION:-2.12.0}"
pip install "nanobind==${NANOBIND_VERSION}" numpy pybind11

# Check for ninja
GENERATOR="Unix Makefiles"
if command -v ninja &> /dev/null; then
    GENERATOR="Ninja"
    echo "Using Ninja generator."
else
    echo "Ninja not found. Using Unix Makefiles (this might be slower)."
fi

# Build the selected LLVM projects and targets in Release mode.
# Explicitly set nanobind directory if found to help CMake locate it
NANOBIND_DIR=$(python3 -c "import nanobind; import os; print(os.path.dirname(nanobind.__file__) + '/cmake')")

cmake -G "$GENERATOR" \
    -S "$LLVM_SRC_DIR/llvm" \
    -B "$LLVM_BUILD_DIR" \
    -DLLVM_ENABLE_PROJECTS="${LLVM_ENABLE_PROJECTS}" \
    -DLLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD}" \
    -DLLVM_ENABLE_RUNTIMES="${LLVM_ENABLE_RUNTIMES}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
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

  # The install tree is ~80% bin/, and those binaries carry a symbol table the
  # build does not need: stripping takes ~20% off the tarball, which the CI
  # cache transfers on every job. Static archives are left alone - they gained
  # nothing when measured, and stripping an archive can drop symbols the link
  # still needs. Set LLVM_STRIP_INSTALL=0 to keep symbols for crash backtraces.
  if [[ "${LLVM_STRIP_INSTALL:-1}" == "1" ]] && command -v strip >/dev/null 2>&1; then
    echo "Stripping installed binaries and shared libraries..."
    before_kb=$(du -sk "${LLVM_INSTALL_DIR}" | cut -f1)
    # -type f skips the symlinks in bin/; non-ELF entries (the Python and Perl
    # helper scripts) simply fail to strip and are skipped.
    find "${LLVM_INSTALL_DIR}/bin" "${LLVM_INSTALL_DIR}/lib" \
         "${LLVM_INSTALL_DIR}/python_packages" \
         -type f ! -name '*.a' -print0 2>/dev/null |
      while IFS= read -r -d '' f; do
        strip --strip-unneeded "${f}" 2>/dev/null || true
      done
    after_kb=$(du -sk "${LLVM_INSTALL_DIR}" | cut -f1)
    echo "Install tree: $((before_kb / 1024)) MB -> $((after_kb / 1024)) MB"
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
