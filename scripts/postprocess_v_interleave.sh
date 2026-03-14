#!/bin/bash
INPUT="$1"
OUTPUT="$2"
python3 - "$INPUT" "$OUTPUT" << 'PYEOF'
import sys, re

with open(sys.argv[1]) as f:
    lines = f.readlines()

content = ''.join(lines)

loop_start_idx = None
loop_end_idx = None
for i, line in enumerate(lines):
    if '.LBB0_5:' in line and loop_start_idx is None:
        loop_start_idx = i
    if loop_start_idx is not None and 's_cbranch_vccnz .LBB0_5' in line:
        loop_end_idx = i
        break

if loop_start_idx is None or loop_end_idx is None:
    with open(sys.argv[2], 'w') as f:
        f.write(content)
    sys.exit(0)

loop = lines[loop_start_idx:loop_end_idx + 1]

def find_line(start, pattern):
    for i in range(start, len(loop)):
        if pattern in loop[i]:
            return i
    return None

def find_line_back(end, pattern):
    for i in range(end, -1, -1):
        if pattern in loop[i]:
            return i
    return None

def extract(start, end):
    return loop[start:end + 1]

# --- Identify sections ---
sec_a_end = find_line(0, 'v_lshl_add_u64 v[68:69], s[6:7]')
sec_b_start = find_line(sec_a_end + 1, 'global_load_dwordx4 v[64:67], v[68:69]')
sec_b_end = find_line_back(find_line(sec_b_start, 's_barrier') - 1, 'ds_write2_b32 v98')
sec_c_end = find_line(sec_b_end + 1, 's_barrier')

sec_d_start = find_line(sec_c_end + 1, 'ds_read2_b32 v[152:153], v99')
sec_d_end = find_line(sec_d_start, 'v[150:151], v[64:79]')

sec_e_start = find_line(sec_d_end + 1, 'v_lshl_add_u64 v[152:153], v[88:89]')
sec_e_end = find_line(sec_e_start, 's_nop 6')

sec_fg_start = sec_e_end + 1
sec_fg_end = find_line(sec_fg_start, 'v_add_f32_e32 v93, v119, v67')

sec_h_start = sec_fg_end + 1
sec_h_end = find_line_back(
    find_line(sec_h_start, 'v_lshl_add_u64 v[66:67], s[8:9]') - 1,
    'v_pk_mul_f32')

sec_i_start = find_line(sec_h_end + 1, 'v_lshl_add_u64 v[66:67], s[8:9]')
sec_i_end = find_line(sec_i_start, 'global_load_dwordx4 v[158:161]')

sec_j_start = sec_i_end + 1
sec_j_end_marker = find_line(sec_j_start, 's_waitcnt vmcnt(1)')
if sec_j_end_marker is None:
    sec_j_end_marker = find_line(sec_j_start, 's_waitcnt vmcnt(0)')
sec_j_end = sec_j_end_marker - 1

sec_k_start = sec_j_end_marker
sec_k_end = find_line_back(find_line(sec_k_start, 's_barrier') - 1, 'ds_write_b16')

sec_l_start = sec_k_end + 1
sec_l_end = find_line(sec_l_start, 's_barrier')

sec_m_start = sec_l_end + 1
sec_m_end = find_line_back(loop_end_idx - loop_start_idx, 'v_mfma_f32_32x32x8_f16')

sec_n = loop_end_idx - loop_start_idx

markers = [sec_a_end, sec_b_start, sec_b_end, sec_c_end,
           sec_d_start, sec_d_end, sec_e_start, sec_e_end,
           sec_fg_start, sec_fg_end, sec_h_start, sec_h_end,
           sec_i_start, sec_i_end, sec_j_start, sec_j_end,
           sec_k_start, sec_k_end, sec_l_start, sec_l_end,
           sec_m_start, sec_m_end, sec_n]

if any(m is None for m in markers):
    print("[postprocess] WARNING: markers not found, copying unchanged",
          file=sys.stderr)
    with open(sys.argv[2], 'w') as f:
        f.write(content)
    sys.exit(0)

# --- Parse GEMM1 ---
gemm1_lines = extract(sec_d_start, sec_d_end)
gemm1_reads = []
gemm1_mfmas = []
for gl in gemm1_lines:
    stripped = gl.strip()
    if stripped.startswith('ds_read2_b32 v[152:153], v99'):
        gemm1_reads.append(stripped.split('v99')[1].strip())
    elif stripped.startswith('v_mfma_f32_32x32x8_f16 v[64:79], v[152:153],'):
        parts = stripped.split(',')
        gemm1_mfmas.append((parts[2].strip(), parts[3].strip() if len(parts) > 3 else '0'))

if len(gemm1_reads) != 16 or len(gemm1_mfmas) != 16:
    print(f"[postprocess] WARNING: GEMM1 {len(gemm1_reads)}/{len(gemm1_mfmas)}, copying unchanged",
          file=sys.stderr)
    with open(sys.argv[2], 'w') as f:
        f.write(content)
    sys.exit(0)

# --- V write ops (grouped by source register) ---
v_hi = [
    ('\tds_write_b16 v101, v154 offset:8864\n',
     '\tds_write_b16_d16_hi v101, v154 offset:8932\n'),
    ('\tds_write_b16 v101, v155 offset:9000\n',
     '\tds_write_b16_d16_hi v101, v155 offset:9068\n'),
    ('\tds_write_b16 v101, v156 offset:9136\n',
     '\tds_write_b16_d16_hi v101, v156 offset:9204\n'),
    ('\tds_write_b16 v101, v157 offset:9272\n',
     '\tds_write_b16_d16_hi v101, v157 offset:9340\n'),
]
v_lo = [
    ('\tds_write_b16 v101, v158 offset:8320\n',
     '\tds_write_b16_d16_hi v101, v158 offset:8388\n'),
    ('\tds_write_b16 v101, v159 offset:8456\n',
     '\tds_write_b16_d16_hi v101, v159 offset:8524\n'),
    ('\tds_write_b16 v101, v160 offset:8592\n',
     '\tds_write_b16_d16_hi v101, v160 offset:8660\n'),
    ('\tds_write_b16 v101, v161 offset:8728\n',
     '\tds_write_b16_d16_hi v101, v161 offset:8796\n'),
]

# --- Build new inner loop ---
new_loop = []

# (A) Addr computation (unchanged)
new_loop.extend(extract(0, sec_a_end))

# (B) K load + LDS write (unchanged)
new_loop.extend(extract(sec_b_start, sec_b_end))

# (B') V addr + V loads (before barrier, hiding latency in GEMM1)
new_loop.append('\tv_lshl_add_u64 v[64:65], s[8:9], 0, v[96:97]\n')
new_loop.append('\tglobal_load_dwordx4 v[154:157], v[64:65], off offset:16\n')
new_loop.append('\tglobal_load_dwordx4 v[158:161], v[64:65], off\n')

# (C) Barrier 1
new_loop.extend(extract(sec_b_end + 1, sec_c_end))

# ============================================================
# GEMM1: K double-buffer + V writes interleaved
# Phase 1 (#0-#3): serial K in v[152:153], flush V hi to LDS
# Phase 2 (#4-#9): K double-buffer v[152:153]/v[154:155]
# Phase 3 (#10-#13): K double-buffer + V lo writes
# Phase 4 (#14-#15): K double-buffer tail
# ============================================================

# --- Phase 1: MFMA #0 (serial, + wait V hi + 2 V hi writes) ---
new_loop.append(f'\tds_read2_b32 v[152:153], v99 {gemm1_reads[0]}\n')
new_loop.append('\ts_waitcnt vmcnt(1)\n')
new_loop.append(v_hi[0][0])
new_loop.append(v_hi[0][1])
new_loop.append('\ts_waitcnt lgkmcnt(2)\n')
new_loop.append(f'\tv_mfma_f32_32x32x8_f16 v[64:79], v[152:153], {gemm1_mfmas[0][0]}, {gemm1_mfmas[0][1]}\n')

# --- Phase 1: MFMA #1-#3 (serial K + 2 V hi writes each) ---
for i in range(1, 4):
    new_loop.append(f'\tds_read2_b32 v[152:153], v99 {gemm1_reads[i]}\n')
    new_loop.append(v_hi[i][0])
    new_loop.append(v_hi[i][1])
    new_loop.append('\ts_waitcnt lgkmcnt(2)\n')
    new_loop.append(f'\tv_mfma_f32_32x32x8_f16 v[64:79], v[152:153], {gemm1_mfmas[i][0]}, v[64:79]\n')

# --- Phase 2: MFMA #4 (start double-buffer: prefetch K[5] into v[154:155]) ---
new_loop.append(f'\tds_read2_b32 v[152:153], v99 {gemm1_reads[4]}\n')
new_loop.append(f'\tds_read2_b32 v[154:155], v99 {gemm1_reads[5]}\n')
new_loop.append('\ts_waitcnt lgkmcnt(1)\n')
new_loop.append(f'\tv_mfma_f32_32x32x8_f16 v[64:79], v[152:153], {gemm1_mfmas[4][0]}, v[64:79]\n')

# --- Phase 2: MFMA #5-#9 (double-buffer alternating) ---
for i in range(5, 10):
    if i % 2 == 1:
        use_buf = 'v[154:155]'
        read_buf = 'v[152:153]'
    else:
        use_buf = 'v[152:153]'
        read_buf = 'v[154:155]'
    new_loop.append(f'\tds_read2_b32 {read_buf}, v99 {gemm1_reads[i + 1]}\n')
    new_loop.append('\ts_waitcnt lgkmcnt(1)\n')
    new_loop.append(f'\tv_mfma_f32_32x32x8_f16 v[64:79], {use_buf}, {gemm1_mfmas[i][0]}, v[64:79]\n')

# --- Phase 3: MFMA #10-#13 (double-buffer + V lo writes) ---
for i in range(10, 14):
    if i % 2 == 0:
        use_buf = 'v[152:153]'
        read_buf = 'v[154:155]'
    else:
        use_buf = 'v[154:155]'
        read_buf = 'v[152:153]'
    v_idx = i - 10
    new_loop.append(f'\tds_read2_b32 {read_buf}, v99 {gemm1_reads[i + 1]}\n')
    if i == 10:
        new_loop.append('\ts_waitcnt vmcnt(0)\n')
    new_loop.append(v_lo[v_idx][0])
    new_loop.append(v_lo[v_idx][1])
    new_loop.append('\ts_waitcnt lgkmcnt(1)\n')
    new_loop.append(f'\tv_mfma_f32_32x32x8_f16 v[64:79], {use_buf}, {gemm1_mfmas[i][0]}, v[64:79]\n')

# --- Phase 4: MFMA #14 (double-buffer, read last K into other buf) ---
new_loop.append(f'\tds_read2_b32 v[154:155], v99 {gemm1_reads[15]}\n')
new_loop.append('\ts_waitcnt lgkmcnt(1)\n')
new_loop.append(f'\tv_mfma_f32_32x32x8_f16 v[64:79], v[152:153], {gemm1_mfmas[14][0]}, v[64:79]\n')

# --- Phase 4: MFMA #15 (last, drain) ---
new_loop.append('\ts_waitcnt lgkmcnt(0)\n')
new_loop.append(f'\tv_mfma_f32_32x32x8_f16 v[64:79], v[154:155], {gemm1_mfmas[15][0]}, v[64:79]\n')

# (E) Loop counter + mask
new_loop.extend(extract(sec_e_start, sec_e_end))

# (F+G) Score scaling + masking + softmax
new_loop.extend(extract(sec_fg_start, sec_fg_end))

# (H) O rescale
new_loop.extend(extract(sec_h_start, sec_h_end))

# (J) P cvt/pack (V load section I removed)
new_loop.extend(extract(sec_j_start, sec_j_end))

# (L) Barrier 2 (V write section K removed)
new_loop.extend(extract(sec_l_start, sec_l_end))

# (M) GEMM2
new_loop.extend(extract(sec_m_start, sec_m_end))

# (N) Loop branch
new_loop.extend(extract(sec_n, sec_n))

# --- Reassemble ---
output_lines = lines[:loop_start_idx] + new_loop + lines[loop_end_idx + 1:]
final_isa = ''.join(output_lines)
with open(sys.argv[2], 'w') as f:
    f.write(final_isa)

import os
dump_dir = os.environ.get('FLYDSL_DUMP_DIR', '')
if dump_dir:
    dump_path = os.path.join(dump_dir, 'postprocessed_final_isa.s')
    os.makedirs(dump_dir, exist_ok=True)
    with open(dump_path, 'w') as f:
        f.write(final_isa)
    print(f"[postprocess] saved to {dump_path}", file=sys.stderr)

print("[postprocess] V loads before barrier + GEMM1 K double-buffer + V writes interleaved",
      file=sys.stderr)
PYEOF
