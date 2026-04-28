#!/bin/bash
#
# Benchmark FlyDSL A8W8 blockscale GEMM kernel on the FFM simulator.
#
# Shape and config come from the CONFIG block inside
# run_gemm_a8w8_blockscale.py — EDIT THAT FILE to change what's benchmarked.
# This script just drives capture + replay + timing + traces.
#
# Usage:
#   ./perf_a8w8_blockscale.sh [--stats-only]
#
# Examples:
#   # Full run (capture, replay, bandwidth, TFLOPS, Perfetto trace, SP3):
#   ./perf_a8w8_blockscale.sh
#
#   # Quick pass — skip Perfetto + SP3 traces, just get timing:
#   ./perf_a8w8_blockscale.sh --stats-only
#
#   # Try a different shape:
#   # 1. Edit run_gemm_a8w8_blockscale.py, change M/N/K in CONFIG
#   # 2. ./perf_a8w8_blockscale.sh
#
# Environment:
#   TRITON_GFX1250_MODEL_PATH  — path to the FFM/rocdtif installation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLYDSL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TRITON_GFX1250_MODEL_PATH="${TRITON_GFX1250_MODEL_PATH:-/root/rocdtif-7.12-am+ffmlite-mi400-r4.03}"
DRAW_LOG="${DRAW_LOG:-./draw.log}"

# --- Parse options ---
SKIP_TRACES=0
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --stats-only) SKIP_TRACES=1; shift ;;
    --) shift; break ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

LAUNCHER="$SCRIPT_DIR/run_gemm_a8w8_blockscale.py"
FLYDSL_DEBUG_DIR="$HOME/.flydsl/debug"

if [[ ! -f "$LAUNCHER" ]]; then
  echo "Error: launcher not found at $LAUNCHER"
  exit 1
fi

echo "=== FlyDSL A8W8 Blockscale GEMM Benchmark ==="
echo "Launcher:  $LAUNCHER"
echo "Config from: run_gemm_a8w8_blockscale.py (edit that file for shape/tile changes)"
echo ""

# --- 0) Clear stale debug dumps ---
if [[ -d "$FLYDSL_DEBUG_DIR" ]]; then
  read -rp "Clear existing debug dumps at $FLYDSL_DEBUG_DIR? [y/N] " answer
  if [[ "$answer" =~ ^[Yy]$ ]]; then
    rm -rf "$FLYDSL_DEBUG_DIR"
    echo "Debug dumps cleared."
  else
    echo "Debug dumps kept."
  fi
fi

# --- 1) Capture: run the kernel under roccap ---
echo ""
echo "=== Step 1: Capture kernel dispatch ==="
: "${LD_PRELOAD:=}"
set +u
source "$TRITON_GFX1250_MODEL_PATH/ffmlite_env.sh"
set -u

CAPTURE_LOG=$(mktemp)
set +e
# Note: FLYDSL_RUNTIME_ENABLE_CACHE=0 intentionally omitted — it triggers a
# known FlyDSL bug (UnboundLocalError: result in jit_function.py) when the
# cache is disabled. Each fresh python invocation recompiles anyway since
# the in-memory cache dies with the process, so leaving the cache enabled
# changes nothing about benchmark behavior.
FLYDSL_DUMP_IR=1 \
FLYDSL_DEBUG_DUMP_ASM=1 \
PYTHONPATH="$FLYDSL_ROOT" \
"$TRITON_GFX1250_MODEL_PATH/tools/roccap/bin/roccap" capture \
  --loglevel error \
  --disp "kernel_gemm_a8w8_blockscale/0" \
  --file gemm_a8w8_blockscale.cap \
  python3 "$LAUNCHER" 2>&1 | tee "$CAPTURE_LOG"
set -e

echo "MLIR/ASM dumps written to $FLYDSL_DEBUG_DIR"

# --- 1b) Parse PERF_STATS line from the runner's output ---
STATS_LINE=$(grep -E '^PERF_STATS ' "$CAPTURE_LOG" | head -n1 || true)
if [[ -z "$STATS_LINE" ]]; then
  echo "Error: could not find 'PERF_STATS' line in launcher output."
  echo "       The runner must print a line like:"
  echo "       PERF_STATS M=.. N=.. K=.. scale_k=.. scale_n=.. elem_bytes_out=.."
  exit 1
fi

# Extract each field
extract_field() { echo "$STATS_LINE" | grep -oE "$1=[0-9]+" | head -n1 | cut -d= -f2; }
M=$(extract_field "M")
N=$(extract_field "N")
K=$(extract_field "K")
SCALE_K=$(extract_field "scale_k")
SCALE_N=$(extract_field "scale_n")
ELEM_OUT=$(extract_field "elem_bytes_out")

TAG="${M}_${N}_${K}"
echo "Parsed shape: M=$M N=$N K=$K scale_k=$SCALE_K scale_n=$SCALE_N out_bytes=$ELEM_OUT"
rm -f "$CAPTURE_LOG"

# --- 2) Play: replay on the model ---
echo ""
echo "=== Step 2: Replay on FFM model ==="
# am_env.sh references $LD_PRELOAD without a default; our `set -u` triggers
# on the unbound variable. Initialize it to empty, then source.
: "${LD_PRELOAD:=}"
set +u
source "$TRITON_GFX1250_MODEL_PATH/am_env.sh"
set -u
export DtifFbBaseLocation=0x200000000

CAP_FILE=""
largest_size=0
for f in gemm_a8w8_blockscale*.cap; do
  [[ -f "$f" ]] || continue
  fsize=$(stat --format="%s" "$f")
  if (( fsize > largest_size )); then
    largest_size=$fsize
    CAP_FILE="$f"
  fi
done

if [[ -z "$CAP_FILE" ]]; then
  echo "Error: no gemm_a8w8_blockscale*.cap files found"
  exit 1
fi
echo "Using cap file: $CAP_FILE ($largest_size bytes)"

"$TRITON_GFX1250_MODEL_PATH/tools/roccap/bin/roccap" play \
  -r "0x200000000-0xF00000000" "./$CAP_FILE"

# --- 3) Parse draw.log for start/end timestamps (picoseconds) ---
echo ""
echo "=== Step 3: Parse timing ==="
if [[ ! -f "$DRAW_LOG" ]]; then
  echo "Error: draw.log not found at $DRAW_LOG"
  exit 1
fi

start_ps=""
end_ps=""
while IFS= read -r line; do
  if [[ "$line" =~ Time:([0-9]+)\ DrawId: ]]; then
    start_ps="${BASH_REMATCH[1]}"
  elif [[ "$line" =~ Time:([0-9]+)\ DrawDone: ]]; then
    end_ps="${BASH_REMATCH[1]}"
  fi
done < "$DRAW_LOG"

if [[ -z "$start_ps" || -z "$end_ps" ]]; then
  echo "Error: could not parse start/end time from $DRAW_LOG"
  exit 1
fi

time_taken_ps=$(( end_ps - start_ps ))
time_us=$(echo "scale=2; $time_taken_ps / 1000000" | bc)

# --- 4) Compute bandwidth and TFLOPS ---
# Byte accounting for A8W8 blockscale:
#   X (FP8):       M * K * 1
#   W (FP8):       N * K * 1
#   x_scale (f32): M * scale_k * 4
#   w_scale (f32): scale_n * scale_k * 4
#   Y (out):       M * N * elem_bytes_out
x_bytes=$(( M * K ))
w_bytes=$(( N * K ))
xs_bytes=$(( M * SCALE_K * 4 ))
ws_bytes=$(( SCALE_N * SCALE_K * 4 ))
y_bytes=$(( M * N * ELEM_OUT ))
total_bytes=$(( x_bytes + w_bytes + xs_bytes + ws_bytes + y_bytes ))

bw_tb_s=$(echo "scale=4; $total_bytes / $time_taken_ps" | bc)

total_flops=$(( 2 * M * N * K ))
tflops=$(echo "scale=2; $total_flops / $time_taken_ps" | bc)

echo ""
echo "=== Results ==="
echo "Time:      ${time_us} us"
echo "Bandwidth: ${bw_tb_s} TB/s  (X=${x_bytes} + W=${w_bytes} + x_scale=${xs_bytes} + w_scale=${ws_bytes} + Y=${y_bytes} = ${total_bytes} B)"
echo "TFLOPS:    ${tflops}"
echo ""

# --- Write stats file ---
STATS_FILE="stats_flydsl_a8w8_blockscale_${TAG}.txt"
cat > "$STATS_FILE" <<EOF
FlyDSL A8W8 Blockscale GEMM Benchmark
M=$M  N=$N  K=$K
scale_k=$SCALE_K scale_n=$SCALE_N elem_bytes_out=$ELEM_OUT
Time (us): $time_us
Time (ps): $time_taken_ps
X bytes: $x_bytes
W bytes: $w_bytes
x_scale bytes: $xs_bytes
w_scale bytes: $ws_bytes
Y bytes: $y_bytes
Total bytes: $total_bytes
Bandwidth: $bw_tb_s TB/s
Total FLOPs: $total_flops
TFLOPS: $tflops
EOF
echo "Stats written to $STATS_FILE"

if [[ "$SKIP_TRACES" -eq 1 ]]; then
  exit 0
fi

# --- 5) Collect WGP00 instruction trace -> Perfetto ---
echo ""
echo "=== Collecting WGP00 instruction trace ==="
if [[ -f "xcc0se0sa0_itrace_emu.mon" ]]; then
  grep -A1 "WGP00" xcc0se0sa0_itrace_emu.mon > wgp0.txt
  if [[ -f "$SCRIPT_DIR/gen_perfetto.py" ]]; then
    python3 "$SCRIPT_DIR/gen_perfetto.py" wgp0.txt "itrace_flydsl_a8w8_blockscale_${TAG}.json"
    echo "Perfetto trace written to itrace_flydsl_a8w8_blockscale_${TAG}.json"
  else
    echo "Warning: gen_perfetto.py not found, skipping Perfetto trace generation"
  fi
else
  echo "Warning: xcc0se0sa0_itrace_emu.mon not found, skipping WGP00 trace"
fi

# --- 6) SP3 disassembly + amtool ---
echo ""
echo "=== Collecting SP3 disassembly trace ==="
"$TRITON_GFX1250_MODEL_PATH/tools/roccap/bin/roccap" extract --sp3 0- "./$CAP_FILE"

ISA_BIN=$(ls roc-dump-*-isa-data.bin 2>/dev/null | head -n1)
if [[ -z "$ISA_BIN" ]]; then
  echo "Error: no roc-dump-*-isa-data.bin file found after extract"
  exit 1
fi
echo "Found ISA binary: $ISA_BIN"

"$TRITON_GFX1250_MODEL_PATH/ffm-lite/sp3disasm" "./$ISA_BIN" gemm_a8w8_blockscale.sp3
echo "SP3 disassembly written to gemm_a8w8_blockscale.sp3"

"$TRITON_GFX1250_MODEL_PATH/tools/rcv/amtool" "rcv_flydsl_a8w8_blockscale_${TAG}/" *.mon gemm_a8w8_blockscale.sp3
echo "amtool output written to rcv_flydsl_a8w8_blockscale_${TAG}/"

# --- 7) Pack traces ---
echo ""
echo "=== Packing traces ==="
PACK_LIST=("rcv_flydsl_a8w8_blockscale_${TAG}/" "$STATS_FILE")
if [[ -f "itrace_flydsl_a8w8_blockscale_${TAG}.json" ]]; then
  PACK_LIST+=("itrace_flydsl_a8w8_blockscale_${TAG}.json")
fi
if [[ -d "$FLYDSL_DEBUG_DIR" ]]; then
  PACK_LIST+=("$FLYDSL_DEBUG_DIR")
fi

tar czf "traces_flydsl_a8w8_blockscale_${TAG}.tar.gz" "${PACK_LIST[@]}"
echo "Traces packed into traces_flydsl_a8w8_blockscale_${TAG}.tar.gz"
