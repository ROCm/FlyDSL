#!/usr/bin/env bash
set -euo pipefail

TEST_CMD=(
  python tests/kernels/test_flash_attn_fwd.py
  --causal
  --compare
  --dtype bf16
  --batch 2
  --num_heads 64
  --num_kv_heads 64
  --seq_len 1024
  --head_dim 128
  --iters 100
)

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <rocprof_input_yaml> <output_trace_tar_path>"
  echo "Example: $0 ./input_opus_attn_thread_trace.yaml thread_trace/flash_attn_opus_kernel_0_b2_s1024_wpe2.tar.gz"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

YAML_FILE="$1"
OUTPUT_TAR="$2"

if [[ ! -f "${YAML_FILE}" ]]; then
  echo "Missing rocprofv3 input YAML: ${YAML_FILE}" >&2
  exit 1
fi

ensure_rocprof_trace_decoder() {
  local decoder_lib="/opt/rocm/lib/librocprof-trace-decoder.so"
  local installer_name="rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh"
  local installer_url="https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/${installer_name}"

  if [[ -f "${decoder_lib}" ]]; then
    echo "rocprof trace decoder found: ${decoder_lib}"
    return
  fi

  echo "rocprof trace decoder missing; installing to /opt/rocm/lib/"
  (
    cd /tmp
    rm -f "${installer_name}"
    wget -q "${installer_url}"
    chmod a+x "${installer_name}"
    echo -e 'y\nn' | "./${installer_name}" --prefix=/opt/rocm/
    cp /opt/rocm/opt/rocm/lib/librocprof-trace-decoder.so /opt/rocm/lib/
    ls -lah "${decoder_lib}"
  )
}

OUTPUT_DIR="$(dirname "${OUTPUT_TAR}")"
OUTPUT_BASE="$(basename "${OUTPUT_TAR}")"
TRACE_NAME="${OUTPUT_BASE%.tar.gz}"
TRACE_NAME="${TRACE_NAME%.tgz}"
FINAL_DIR="${OUTPUT_DIR}/${TRACE_NAME}"

RPF_DIR="$(python3 - "${YAML_FILE}" <<'PY'
import sys
from pathlib import Path

yaml_file = sys.argv[1]
for line in Path(yaml_file).read_text().splitlines():
    stripped = line.strip()
    if stripped.startswith("output_directory:"):
        print(stripped.split(":", 1)[1].strip())
        break
else:
    raise SystemExit(f"output_directory not found in {yaml_file}")
PY
)"

MLIR_LIBS_DIR="${REPO_ROOT}/build-fly/python_packages/flydsl/_mlir/_mlir_libs"
if [[ ":${LD_LIBRARY_PATH:-}:" != *":${MLIR_LIBS_DIR}:"* ]]; then
  export LD_LIBRARY_PATH="${MLIR_LIBS_DIR}:${LD_LIBRARY_PATH:-}"
fi

# export HIP_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=1
export FLYDSL_WAVES_PER_EU=2
export FLYDSL_ENABLE_OPUS_PATH=1
export FLYDSL_LOG_MORE="${FLYDSL_LOG_MORE:-1}"
export FLYDSL_DEBUG_LOG_TO_CONSOLE="${FLYDSL_DEBUG_LOG_TO_CONSOLE:-1}"
export FLYDSL_DEBUG_LOG_LEVEL="${FLYDSL_DEBUG_LOG_LEVEL:-INFO}"
unset FLYDSL_OPUS_TRIGGER_LAZY_ELSE FLYDSL_OPUS_DEBUG_LAZY_ELSE FLYDSL_OPUS_DEBUG_LAZY_COUNTS

rocm-smi | egrep "$HIP_VISIBLE_DEVICES    |Device"

rm -rf ~/.flydsl/cache/*
mkdir -p "${RPF_DIR}" "${OUTPUT_DIR}"
rm -rf "${RPF_DIR}" "${FINAL_DIR}" "${OUTPUT_TAR}"

ensure_rocprof_trace_decoder

set +e
rocprofv3 -i "${YAML_FILE}" -- "${TEST_CMD[@]}"
status=$?
set -e

mkdir -p "${FINAL_DIR}"
shopt -s nullglob dotglob
trace_files=("${RPF_DIR}"/*)
if (( ${#trace_files[@]} > 0 )); then
  cp -r "${trace_files[@]}" "${FINAL_DIR}/"
fi
shopt -u nullglob dotglob

tar -zcf "${OUTPUT_TAR}" -C "${OUTPUT_DIR}" "${TRACE_NAME}"

echo "Trace directory: ${FINAL_DIR}"
echo "Trace archive:   ${OUTPUT_TAR}"
ls -lah ${FINAL_DIR}
ls -lah ${OUTPUT_TAR}
echo "rocprofv3/test exit code: ${status}"
exit "${status}"
