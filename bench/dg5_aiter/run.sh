#!/usr/bin/env bash
# DG5 (aiter sorting): bs ∈ {16,32,64} × 4 routing × 3 builds = 36 runs
#   builds:
#     - baseline       : --bench-op baseline
#     - fused_prev     : --bench-op fused --no-token-flag-sync   (D-flag C-1 OFF, 等价于上一个 fused 版本)
#     - fused_c1       : --bench-op fused --token-flag-sync      (D-flag C-1 ON, 当前路径)
# Output: bench/dg5_aiter/*.log + bench/dg5_aiter/summary.csv
set -u
cd /home/yashao/FlyDSL
mkdir -p bench/dg5_aiter

SUMMARY=bench/dg5_aiter/summary.csv
echo "bs,routing,build,gemm2_us,combine_us,fused_us,e2e_us,host_us,status" > "$SUMMARY"

export PYTHONPATH=/home/yashao/aiter:${PYTHONPATH:-}

run_one() {
  local bs=$1 routing=$2 build=$3
  local logfile="bench/dg5_aiter/${build}_bs${bs}_${routing}.log"
  local extra_args
  case "$build" in
    baseline)    extra_args="--bench-op baseline --no-token-flag-sync" ;;
    fused_prev)  extra_args="--bench-op fused --fuse-mode auto --no-token-flag-sync" ;;
    fused_c1)    extra_args="--bench-op fused --fuse-mode auto --token-flag-sync" ;;
    *) echo "unknown build: $build"; return 1 ;;
  esac

  echo "=== [DG5] bs=$bs routing=$routing build=$build ==="
  if timeout 300 torchrun --nproc_per_node=8 \
      tests/kernels/test_profiler_moe_gemm2_combine.py \
      --mode profile --cudagraph $extra_args \
      --max-tokens $bs --k 8 --hidden-dim 7168 --inter-dim 2048 \
      --routing $routing --sorting aiter >"$logfile" 2>&1; then
    status=ok
  else
    status=fail
  fi

  local g2=$(grep '\[Device\] moe_gemm2 kernel GPU time' "$logfile" | awk '{print $6}')
  local cb=$(grep '\[Device\] combine kernel GPU time'   "$logfile" | awk '{print $6}')
  local fu=$(grep '\[Device\] fused_gemm2_combine GPU time' "$logfile" | awk '{print $5}')
  local e2=$(grep '\[E2E\]'   "$logfile" | awk '{print $6}')
  local ho=$(grep '\[Host\]'  "$logfile" | awk '{print $5}')
  echo "$bs,$routing,$build,${g2:-NA},${cb:-NA},${fu:-NA},${e2:-NA},${ho:-NA},$status" >> "$SUMMARY"
  echo "    -> gemm2=${g2:-NA} combine=${cb:-NA} fused=${fu:-NA} e2e=${e2:-NA} status=$status"
}

START=$(date +%s)
for bs in 16 32 64; do
  for routing in random atomic1_8pe atomic8_1pe atomic2_4pe; do
    for build in baseline fused_prev fused_c1; do
      run_one "$bs" "$routing" "$build"
    done
  done
done
END=$(date +%s)
echo "DG5 done in $((END-START)) s. Summary: $SUMMARY"
cat "$SUMMARY"
