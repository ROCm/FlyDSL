#!/bin/bash
INPUT="$1"
OUTPUT="$2"
python3 - "$INPUT" "$OUTPUT" << 'PYEOF'
import sys, re

with open(sys.argv[1]) as f:
    content = f.read()

# ============================================================
# GEMM1 double-buffering: replace serialized ds_read/MFMA chain
# with pipelined version using v[152:153] and v[154:155]
# ============================================================

# The serialized GEMM1 pattern (16 repetitions):
#   ds_read2_b32 v[152:153], v99 offset0:X offset1:Y
#   s_waitcnt lgkmcnt(0)
#   v_mfma_f32_32x32x8_f16 v[64:79], v[152:153], v[Q:Q+1], {0|v[64:79]}

# Find all 16 MFMA blocks in GEMM1
gemm1_pattern = re.compile(
    r'(\tds_read2_b32 v\[152:153\], v99 (offset[^\n]*)\n'
    r'\ts_waitcnt lgkmcnt\(0\)\n'
    r'\tv_mfma_f32_32x32x8_f16 v\[64:79\], v\[152:153\], (v\[\d+:\d+\]), ([^\n]*)\n)',
    re.MULTILINE
)

matches = list(gemm1_pattern.finditer(content))

if len(matches) == 16:
    # Build the double-buffered replacement
    # Extract offset strings and Q-pack registers
    offsets = []
    qpacks = []
    accs = []
    for m in matches:
        offsets.append(m.group(2))
        qpacks.append(m.group(3))
        accs.append(m.group(4))
    
    # Build new GEMM1 block
    lines = []
    # Prefetch K[0] into v[152:153] and K[1] into v[154:155]
    lines.append(f"\tds_read2_b32 v[152:153], v99 {offsets[0]}")
    lines.append(f"\tds_read2_b32 v[154:155], v99 {offsets[1]}")
    lines.append(f"\ts_waitcnt lgkmcnt(1)")
    lines.append(f"\tv_mfma_f32_32x32x8_f16 v[64:79], v[152:153], {qpacks[0]}, {accs[0]}")
    
    # Middle MFMAs (2..14): alternate buffers, prefetch next
    for i in range(1, 15):
        if i % 2 == 1:
            read_buf = "v[152:153]"  # prefetch into 152 (154 being consumed)
            use_buf = "v[154:155]"   # use 154 (prefetched last step)
        else:
            read_buf = "v[154:155]"  # prefetch into 154 (152 being consumed)
            use_buf = "v[152:153]"   # use 152 (prefetched last step)
        
        lines.append(f"\tds_read2_b32 {read_buf}, v99 {offsets[i+1]}")
        lines.append(f"\ts_waitcnt lgkmcnt(1)")
        lines.append(f"\tv_mfma_f32_32x32x8_f16 v[64:79], {use_buf}, {qpacks[i]}, v[64:79]")
    
    # Last MFMA (#16, index 15): no more prefetch, wait for everything
    last_buf = "v[154:155]" if 15 % 2 == 1 else "v[152:153]"
    lines.append(f"\ts_waitcnt lgkmcnt(0)")
    lines.append(f"\tv_mfma_f32_32x32x8_f16 v[64:79], {last_buf}, {qpacks[15]}, v[64:79]")
    
    new_gemm1 = '\n'.join(lines) + '\n'
    
    # Replace from first match start to last match end
    start = matches[0].start()
    end = matches[-1].end()
    content = content[:start] + new_gemm1 + content[end:]

with open(sys.argv[2], 'w') as f:
    f.write(content)

import os
dump_dir = os.environ.get('FLYDSL_DUMP_DIR', '')
if dump_dir:
    dump_path = os.path.join(dump_dir, 'postprocessed_final_isa.s')
    os.makedirs(dump_dir, exist_ok=True)
    with open(dump_path, 'w') as f:
        f.write(content)
    print(f"[postprocess_pipeline] saved to {dump_path}", file=sys.stderr)
PYEOF
