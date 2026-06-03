#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
JOBS="${JOBS:-$(nproc)}"

AITER_DIR=""
LLVM_DIR=""
FLYDSL_DIR=""

usage() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} --aiter-dir <dir> --llvm-dir <dir> --flydsl-dir <dir> [options]

Required:
  --aiter-dir <dir>   Path to the aiter repository
  --llvm-dir <dir>    Path to the llvm-project repository
  --flydsl-dir <dir>  Path to the FlyDSL repository

Options:
  -j, --jobs <N>             Parallel build jobs, default: JOBS env or nproc
  -h, --help                 Show this help

Environment:
  ARCH                       Optional GPU arch used by opus_attn/install_python.sh
  OPUS_INCLUDE_DIR           Optional override; defaults to <aiter-dir>/csrc/include
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_value() {
  local opt="$1"
  local value="${2:-}"
  [ -n "$value" ] || die "${opt} requires a value"
}

abs_path() {
  local path="$1"
  if [ -d "$path" ]; then
    cd "$path"
    pwd
    cd - >/dev/null
  else
    return 1
  fi
}

parse_args() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --aiter-dir)
        require_value "$1" "${2:-}"
        AITER_DIR="${2:-}"
        shift 2
        ;;
      --llvm-dir)
        require_value "$1" "${2:-}"
        LLVM_DIR="${2:-}"
        shift 2
        ;;
      --flydsl-dir)
        require_value "$1" "${2:-}"
        FLYDSL_DIR="${2:-}"
        shift 2
        ;;
      -j|--jobs)
        require_value "$1" "${2:-}"
        JOBS="${2:-}"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

validate_inputs() {
  [ -n "$AITER_DIR" ] || die "--aiter-dir is required"
  [ -n "$LLVM_DIR" ] || die "--llvm-dir is required"
  [ -n "$FLYDSL_DIR" ] || die "--flydsl-dir is required"
  [[ "$JOBS" =~ ^[0-9]+$ ]] || die "--jobs must be a positive integer"

  AITER_DIR="$(abs_path "$AITER_DIR")" || die "aiter directory not found: $AITER_DIR"
  LLVM_DIR="$(abs_path "$LLVM_DIR")" || die "llvm-project directory not found: $LLVM_DIR"
  FLYDSL_DIR="$(abs_path "$FLYDSL_DIR")" || die "FlyDSL directory not found: $FLYDSL_DIR"

  [ -f "${AITER_DIR}/setup.py" ] || die "aiter setup.py not found under: $AITER_DIR"
  [ -d "${LLVM_DIR}/llvm" ] || die "llvm/ directory not found under: $LLVM_DIR"
  [ -f "${FLYDSL_DIR}/scripts/build.sh" ] || die "FlyDSL scripts/build.sh not found under: $FLYDSL_DIR"
  [ -f "${FLYDSL_DIR}/opus_attn/install_python.sh" ] || die "FlyDSL opus_attn/install_python.sh not found"
  [ -f "${FLYDSL_DIR}/exp_isa/build.sh" ] || die "FlyDSL exp_isa/build.sh not found"
}

install_prereqs() {
  echo "=== Installing build prerequisites ==="
  python3 -m pip install nanobind numpy pybind11 vcs_versioning

  if ! command -v cmake >/dev/null 2>&1 ||
     ! command -v ninja >/dev/null 2>&1 ||
     ! command -v patchelf >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y cmake ninja-build patchelf
  fi

  python3 -c "import nanobind, numpy, pybind11"
  cmake --version
  ninja --version
  patchelf --version
  echo ""
}

build_aiter() {
  echo "=== Building/installing aiter ==="
  cd "$AITER_DIR"
  rm -f aiter/jit/*.so
  rm -rf aiter/jit/build
  python3 setup.py develop
  echo "=== aiter complete ==="
  echo ""
}

build_llvm() {
  echo "=== Building/installing LLVM/MLIR ==="
  cd "$LLVM_DIR"

  echo "Removing ${LLVM_DIR}/buildmlir"
  rm -rf buildmlir
  mkdir -p buildmlir
  cd buildmlir
  cmake -G Ninja \
    -S ../llvm \
    -DLLVM_ENABLE_PROJECTS='mlir;clang' \
    -DLLVM_TARGETS_TO_BUILD='X86;NVPTX;AMDGPU' \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE="$(which python3)" \
    -Dnanobind_DIR="$(python3 -c "import nanobind, os; print(os.path.dirname(nanobind.__file__) + '/cmake')")" \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF
  cd ..

  [ -f buildmlir/build.ninja ] || die "LLVM build directory exists but build.ninja is missing: ${LLVM_DIR}/buildmlir"

  cmake --build buildmlir -j "$JOBS"
  rm -rf mlir_install
  cmake --install buildmlir --prefix mlir_install
  echo "=== LLVM/MLIR complete ==="
  echo ""
}

build_flydsl() {
  echo "=== Building/installing FlyDSL ==="
  cd "$FLYDSL_DIR"
  rm -rf build-fly
  export MLIR_PATH="${LLVM_DIR}/mlir_install"
  bash scripts/build.sh -j"$JOBS"
  python3 -m pip install -e .
  echo "=== FlyDSL complete ==="
  echo ""
}

build_opus_attn() {
  echo "=== Building/installing opus_attn ==="
  export OPUS_INCLUDE_DIR="${OPUS_INCLUDE_DIR:-${AITER_DIR}/csrc/include}"
  export JOBS
  (
    cd "${FLYDSL_DIR}/opus_attn"
    bash install_python.sh
  )
  echo "=== opus_attn complete ==="
  echo ""
}

build_exp_isa() {
  echo "=== Building exp_isa ==="
  (
    cd "${FLYDSL_DIR}/exp_isa"
    bash build.sh
  )
  echo "=== exp_isa complete ==="
  echo ""
}

main() {
  parse_args "$@"
  validate_inputs

  echo "=== Build configuration ==="
  echo "AITER_DIR=${AITER_DIR}"
  echo "LLVM_DIR=${LLVM_DIR}"
  echo "FLYDSL_DIR=${FLYDSL_DIR}"
  echo "JOBS=${JOBS}"
  echo ""

  install_prereqs
  build_aiter
  build_llvm
  build_flydsl
  build_opus_attn
  build_exp_isa

  echo "=== All builds complete ==="

  echo "=== Run benchmark ==="
  python tests/kernels/test_flash_attn_fwd.py --iters 100 --compare
}

main "$@"
