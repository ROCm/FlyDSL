#!/bin/bash
# Full-map verification of the batch-aware dense routing (PR #685).
#
# For every dense family cell it measures:
#   auto    = default routing (the NEW batch-aware gate, what users get)
#   generic = FLYDSL_DISABLE_DUALWAVE_SWP=1 (the generic fallback)
# and records both times + the aiter_ck reference (--compare) + PASS/FAIL.
#
# A companion python pass then checks, per cell, that `auto` is within noise of
# the FASTER of {auto-as-routed, generic}, i.e. the routing did not leave a
# materially faster provider unused. The cells the new gate changes vs the old
# flat S<256 gate (large-batch S=192/255) are flagged explicitly.
set -uo pipefail
cd /sgl-workspace/FlyDSL-pr685
export PYTHONPATH="/sgl-workspace/FlyDSL-pr685:/sgl-workspace/FlyDSL-lab/build-fly/python_packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/sgl-workspace/FlyDSL-lab/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH:-}"
GPU=${GPU:-0}; H=tests/kernels/test_flash_attn_fwd.py
OUT=${OUT:-/sgl-workspace/FlyDSL-pr685/scripts/dense_routing_verify.csv}
PROG=${OUT%.csv}.progress
: > "$PROG"
log(){ echo "[$(date -u +%H:%M:%S)] $*" >> "$PROG"; }
echo "mode,B,S,H,Hkv,dtype,causal,rc,pass,fail,FlyDSL_us,aiter_ck_us,Fly/ck%" > "$OUT"

emit(){ # mode rc
  local mode="$1" rc="$2"
  if ls fmha_perf_compare_*.csv >/dev/null 2>&1; then
    python3 - "$(ls fmha_perf_compare_*.csv|head -1)" "$mode" "$rc" >> "$OUT" <<'PY'
import csv,sys
f,mode,rc=sys.argv[1:4]
r=[x for x in csv.DictReader(open(f)) if x.get("B","").strip() and "AVG" not in x["B"]]
if r:
    x=r[0]
    try: me=float(x["FlyDSL_MaxErr"]); st="PASS" if (rc=="0" and me<1e-2) else "FAIL"
    except: st="FAIL"
    print(",".join([mode,x["B"],x["S"],x["H"],x["Hkv"],x["dtype"],x["causal"],rc,
        "1" if st=="PASS" else "0","0" if st=="PASS" else "1",
        x["FlyDSL_Time(us)"],x.get("aiter_ck_Time(us)",""),x.get("Fly/aiter_ck_TFLOPS%","")]))
PY
  fi
}
run(){ # mode env B S Hq Hkv dt causalflag
  local mode="$1" envv="$2" B="$3" S="$4" Hq="$5" Hkv="$6" dt="$7" cf="$8"
  rm -f fmha_perf_compare_*.csv
  env $envv HIP_VISIBLE_DEVICES=$GPU python3 "$H" --compare --batch $B --seq_len $S \
    --num_heads $Hq --num_kv_heads $Hkv --head_dim 128 $cf --dtype $dt \
    --warmup 8 --iters 30 >> "${OUT%.csv}.log" 2>&1
  emit "$mode" "$?"
}

for B in 1 8; do
  for S in 128 192 256 384 512; do
    for heads in "64 64" "64 8"; do
      set -- $heads; Hq=$1; Hkv=$2
      for dt in bf16 fp16; do
        for cf in "--causal" "--no-causal"; do
          run auto    ""                             $B $S $Hq $Hkv $dt "$cf"
          run generic "FLYDSL_DISABLE_DUALWAVE_SWP=1" $B $S $Hq $Hkv $dt "$cf"
        done
      done
    done
  done
  log "B=$B done"
done
n=$(awk 'NR>1{c++}END{print c+0}' "$OUT")
echo "DONE rows=$n" | tee -a "$PROG"
