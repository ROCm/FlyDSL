#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-hyg_trn_rocm7.1}"
REPO_ROOT="${REPO_ROOT:-/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_sp1/FlyDSL}"
TEST_FILE="${REPO_ROOT}/tests/kernels/test_flash_opus_attn.py"
BACKUP_FILE="${TEST_FILE}.isolated_compare.bak.$$"
GPU_ID="${HIP_VISIBLE_DEVICES:-1}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/isolated_compare_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/isolated_compare_$(date +%Y%m%d_%H%M%S).log}"

# Set the isolated benchmark execution order here.
# Valid names: flydsl, opus, ck, asm.
# BENCH_ORDER=(flydsl opus ck asm)
BENCH_ORDER=(ck asm flydsl opus)

rocm-smi | egrep "$GPU_ID    |Device"

BENCH_ARGS=(
  --causal
  --dtype bf16
  --batch 2
  --num_heads 64
  --num_kv_heads 64
  --seq_len 1024
  --head_dim 128
  --iters 100
  --compare
)

docker_bash() {
  docker exec "${CONTAINER}" bash -lc "$1"
}

validate_bench_order() {
  local backend
  if [[ "${#BENCH_ORDER[@]}" -ne 4 ]]; then
    echo "BENCH_ORDER must contain exactly 4 entries: flydsl opus ck asm" >&2
    exit 1
  fi
  for backend in "${BENCH_ORDER[@]}"; do
    case "${backend}" in
      flydsl|opus|ck|asm) ;;
      *)
        echo "Invalid backend in BENCH_ORDER: ${backend}" >&2
        echo "Valid names: flydsl opus ck asm" >&2
        exit 1
        ;;
    esac
  done
}

restore_test_file() {
  docker_bash "if [[ -f '${BACKUP_FILE}' ]]; then cp '${BACKUP_FILE}' '${TEST_FILE}'; rm -f '${BACKUP_FILE}'; fi"
}

patch_test_backend() {
  local backend="$1"
  docker exec -i \
    -e "ISOLATED_BACKEND=${backend}" \
    -e "TEST_FILE=${TEST_FILE}" \
    "${CONTAINER}" \
    python3 - <<'PY'
import os
from pathlib import Path

backend = os.environ["ISOLATED_BACKEND"]
path = Path(os.environ["TEST_FILE"])
text = path.read_text()

call_blocks = {
    "flydsl": [
        "res = run_config(",
        "    batch, seq_len, nh, hd, dtype, causal,",
        "    warmup=args.warmup, iters=args.iters,",
        "    seed=args.seed, dtype_str=dtype_str, verbose=False,",
        "    num_kv_heads=nh_kv,",
        ")",
    ],
    "opus": [
        "res = run_opus_attn_bench(",
        "    batch, seq_len, nh, hd, dtype, causal,",
        "    warmup=args.warmup, iters=args.iters,",
        "    seed=args.seed, num_kv_heads=nh_kv,",
        ")",
    ],
    "ck": [
        "res = run_aiter_bench(",
        "    batch, seq_len, nh, hd, dtype, causal,",
        "    warmup=args.warmup, iters=args.iters,",
        '    seed=args.seed, backend="ck",',
        "    num_kv_heads=nh_kv,",
        ")",
    ],
    "asm": [
        "res = run_aiter_bench(",
        "    batch, seq_len, nh, hd, dtype, causal,",
        "    warmup=args.warmup, iters=args.iters,",
        '    seed=args.seed, backend="asm",',
        "    num_kv_heads=nh_kv,",
        ")",
    ],
}

if backend not in call_blocks:
    raise SystemExit(f"unknown backend: {backend}")

outer_indent = " " * 20
inner_indent = " " * 24
lines = [f"{outer_indent}for i in range(4):"]
for name in ("flydsl", "opus", "ck", "asm"):
    active = name == backend
    for line in call_blocks[name]:
        prefix = inner_indent if active else f"{inner_indent}# "
        lines.append(prefix + line)
    lines.append("")

lines.extend(
    [
        f"{inner_indent}if i == 0:",
        f"{inner_indent}    fly_r = res",
        f"{inner_indent}elif i == 1:",
        f"{inner_indent}    opus_r = res",
        f"{inner_indent}elif i == 2:",
        f"{inner_indent}    ck_r = res",
        f"{inner_indent}elif i == 3:",
        f"{inner_indent}    asm_r = res",
    ]
)
replacement = "\n".join(lines) + "\n"

start_marker = f"{outer_indent}for i in range(4):\n"
end_marker = f"{outer_indent}rows.append((cfg, fly_r, opus_r, ck_r, asm_r))"

try:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
except ValueError as exc:
    raise SystemExit(f"could not locate benchmark switch block in {path}") from exc

path.write_text(text[:start] + replacement + text[end:])
print(f"Enabled only {backend}; other benchmark calls are commented out in {path}")
PY
}

trap restore_test_file EXIT

docker_bash "test -f '${TEST_FILE}'"
docker_bash "cp '${TEST_FILE}' '${BACKUP_FILE}' && mkdir -p '${LOG_DIR}' && : > '${LOG_FILE}'"
validate_bench_order

echo "Container: ${CONTAINER}"
echo "GPU: ${GPU_ID}"
echo "Log file: ${LOG_FILE}"
echo "Benchmark order: ${BENCH_ORDER[*]}"

for backend in "${BENCH_ORDER[@]}"; do
    # export FLYDSL_DUMP_IR=1
  echo
  echo "===== Running isolated backend: ${backend} ====="
  patch_test_backend "${backend}"
  docker_bash "
    set -euo pipefail
    cd '${REPO_ROOT}'
    export HIP_VISIBLE_DEVICES='${GPU_ID}'
    export FLYDSL_LOG_MORE=1
    export FLYDSL_DEBUG_LOG_TO_CONSOLE=1
    export FLYDSL_DEBUG_LOG_LEVEL=INFO
    export FLYDSL_DUMP_DIR=./flydsl_dump
    export FLYDSL_ENABLE_OPUS_PATH=1
    {
      printf '\n===== Running isolated backend: ${backend} =====\n'
      python tests/kernels/test_flash_opus_attn.py ${BENCH_ARGS[*]} 2>&1
    #   python tests/kernels/test_flash_opus_attn.py ${BENCH_ARGS[*]} 2>&1
    #   python tests/kernels/test_flash_opus_attn.py ${BENCH_ARGS[*]} 2>&1
    #   python tests/kernels/test_flash_opus_attn.py ${BENCH_ARGS[*]} 2>&1
    } | tee -a '${LOG_FILE}'
  "
done

restore_test_file
trap - EXIT

echo
echo "Done. Restored ${TEST_FILE}"
echo "Logs are in ${LOG_FILE}"
