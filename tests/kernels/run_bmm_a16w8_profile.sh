#!/usr/bin/env bash
# Run rocprofv3 counter collection for all bmm_a16w8 configs.
#
# Usage:
#   cd ~/flydsl
#   bash tests/kernels/run_bmm_a16w8_profile.sh
#   bash tests/kernels/run_bmm_a16w8_profile.sh /tmp/bmm_results   # custom output dir
#
# Output: one subdirectory per config under OUTPUT_DIR.
# Each dir contains rocprofv3's CSV with counter values for dispatch indices 10..14.
#
# After collection, aggregate with:
#   python tests/kernels/parse_bmm_counters.py /tmp/bmm_results

set -euo pipefail

OUTPUT_DIR="${1:-/tmp/bmm_profile}"
YAML="tests/kernels/bmm_a16w8_counters.yaml"
SCRIPT="tests/kernels/profile_bmm_a16w8.py"

# All configs defined in profile_bmm_a16w8.py
CONFIGS=(
    "dec_e8m0"          # decode baseline, memory-bound
    "dec_e8m0_cl8"      # decode + cluster_n=8 (A multicast)
    "pre_e8m0_m256"     # prefill M=256, near compute-bound
    "dec_noscale"       # no_scale mode (plain fp8→bf16)
    "dec_tm32"          # tile_m=32 (smaller tile, compare vs 64)
)

mkdir -p "$OUTPUT_DIR"

echo "=== bmm_a16w8 counter collection ==="
echo "  YAML:       $YAML"
echo "  Script:     $SCRIPT"
echo "  Output dir: $OUTPUT_DIR"
echo "  Configs:    ${CONFIGS[*]}"
echo ""

FAILED=()

for CFG in "${CONFIGS[@]}"; do
    OUT="$OUTPUT_DIR/$CFG"
    mkdir -p "$OUT"
    LOG="$OUT/rocprofv3.log"

    echo "--- [$CFG] start ---"
    echo "    output: $OUT"

    if rocprofv3 \
        -i "$YAML" \
        --output-dir "$OUT" \
        -- python "$SCRIPT" "$CFG" \
        2>&1 | tee "$LOG"; then
        echo "    [$CFG] OK"
    else
        echo "    [$CFG] FAILED (see $LOG)"
        FAILED+=("$CFG")
    fi
    echo ""
done

echo "=== Done ==="
echo "Results: $OUTPUT_DIR"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED configs: ${FAILED[*]}"
    exit 1
fi

echo ""
echo "Next: parse counter CSVs:"
echo "  python tests/kernels/parse_bmm_counters.py $OUTPUT_DIR"
