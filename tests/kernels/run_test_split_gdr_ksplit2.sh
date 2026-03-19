#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_TARGET="tests/kernels/test_fused_split_gdr_update_ksplit2_flyc.py::test_split_gdr_ksplit2_correctness_and_perf"
TARGET="${1:-${DEFAULT_TARGET}}"

if [[ $# -gt 0 ]]; then
  shift
fi

# Default behavior:
# - kill stale pytest processes for this test file
# - keep a stable torch extension cache for faster reruns
# - allow opt-in isolated cache via ISOLATE_EXT=1
if [[ "${CLEAN_STALE_PYTEST:-1}" == "1" ]]; then
  pkill -f "pytest tests/kernels/test_fused_split_gdr_update_ksplit2_flyc.py" 2>/dev/null || true
fi

if [[ "${ISOLATE_EXT:-0}" == "1" ]]; then
  EXT_DIR="/tmp/torch_ext_split_gdr_$(date +%s)"
else
  EXT_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_ext_split_gdr}"
fi

FAULT_TIMEOUT="${FAULT_TIMEOUT:-180}"

echo "[run_test_split_gdr_ksplit2] repo: ${REPO_ROOT}"
echo "[run_test_split_gdr_ksplit2] target: ${TARGET}"
echo "[run_test_split_gdr_ksplit2] TORCH_EXTENSIONS_DIR: ${EXT_DIR}"
echo "[run_test_split_gdr_ksplit2] faulthandler_timeout: ${FAULT_TIMEOUT}"

exec env \
  TORCH_EXTENSIONS_DIR="${EXT_DIR}" \
  PYTHONFAULTHANDLER=1 \
  HIP_VISIBLE_DEVICES=7 rocprofv3 --hip-runtime-trace --kernel-trace --output-format csv pftrace -d trace -o flydsl_gdr_ksplit2_0311 --stats -- python3 -m pytest "${TARGET}" -v -s -o "faulthandler_timeout=${FAULT_TIMEOUT}" "$@"
  # HIP_VISIBLE_DEVICES=7 rocprofv3 --hip-runtime-trace --kernel-trace --output-format csv pftrace -d trace -o flydsl_gdr_ksplit2_0311 --stats -- python3 -m pytest "${TARGET}" -v -s -o "faulthandler_timeout=${FAULT_TIMEOUT}" "$@"

