#!/bin/sh
TOP=$(cd "$(dirname "$0")" && pwd)
cd "$TOP"

make clean 2>/dev/null

echo "=== Parallel build (make -j) ==="
START=$(date +%s%N)
make -j 2>&1
RC=$?
END=$(date +%s%N)
TOTAL_MS=$(( (END - START) / 1000000 ))

if [ $RC -ne 0 ]; then
    echo "Build FAILED (rc=$RC)"
    exit 1
fi

echo ""
echo "=== Build time: ${TOTAL_MS} ms ==="
echo "Output: $TOP/build/gqa_attn.exe"
echo ""

# Run benchmarks: d=512 (current default) then d=128
echo "=== d=512 Causal ==="
./build/gqa_attn.exe -d 512 --causal
echo ""
echo "=== d=512 Causal N=16384 ==="
./build/gqa_attn.exe -d 512 --causal -n 16384
echo ""
echo "=== d=512 Non-causal ==="
./build/gqa_attn.exe -d 512 --no-causal
echo ""
echo "=== d=512 Non-causal N=16384 ==="
./build/gqa_attn.exe -d 512 --no-causal -n 16384
echo ""
echo "=== d=128 Causal ==="
./build/gqa_attn.exe -d 128 --causal
echo ""
echo "=== d=128 Non-causal ==="
./build/gqa_attn.exe -d 128 --no-causal
