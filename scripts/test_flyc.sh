#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/build-fly/python_packages:${REPO_ROOT}"
export LD_LIBRARY_PATH="${REPO_ROOT}/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH:-}"

echo "=== vecadd ==="
python3 examples/01-vectorAdd.py 2>&1 | tail -2
echo ""

echo "=== gemm ==="
for dtype in fp8 int8 bf16; do
  for args in \
    "-M 5120 -N 5120 -K 8320 --tile_m 64 --tile_n 256 --tile_k 128" \
    "-M 9728 -N 8192 -K 8320 --tile_m 64 --tile_n 256 --tile_k 128" \
    "-M 5133 -N 5120 -K 8320 --tile_m 64 --tile_n 256 --tile_k 128" \
    "-M 16 -N 5120 -K 8192 --tile_m 16 --tile_n 64 --tile_k 512"; do
    shape=$(echo $args | awk '{print $2"x"$4"x"$6}')
    result=$(python3 tests/kernels/test_preshuffle_gemm.py --flyc --in_dtype $dtype $args --num_iters 10 --num_warmup 3 2>&1)
    tf=$(echo "$result" | grep -oP '[0-9]+\.[0-9]+ TFLOPS' || true)
    st=$(echo "$result" | grep -q "passed" && echo "PASS" || echo "FAIL")
    printf "%-5s %-20s %15s  %s\n" "$dtype" "$shape" "$tf" "$st"
  done
done
