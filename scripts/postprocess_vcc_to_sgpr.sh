#!/bin/bash
INPUT="$1"
OUTPUT="$2"
python3 - "$INPUT" "$OUTPUT" << 'PYEOF'
import sys, re

with open(sys.argv[1]) as f:
    lines = f.readlines()

sgpr_base = 42
sgpr_idx = 0

# Only convert causal mask comparisons (unsigned u64), NOT loop control (signed i64)
CAUSAL_CMP_PATTERN = re.compile(r'(v_cmp_(?:le|lt|ge|gt)_u(?:32|64))_e32\s+vcc,\s*(.*)')

result = []
i = 0
while i < len(lines):
    line = lines[i].rstrip('\n')
    stripped = line.strip()

    m = CAUSAL_CMP_PATTERN.match(stripped)
    if m:
        indent = line[:len(line) - len(line.lstrip())]
        op, operands = m.group(1), m.group(2)
        pair = sgpr_base + sgpr_idx * 2
        sgpr_pair = f"s[{pair}:{pair+1}]"
        result.append(f"{indent}{op}_e64 {sgpr_pair}, {operands}")

        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if nxt == 's_nop 0':
                j += 1
                continue
            mc = re.match(r'v_cndmask_b32_e32\s+(.*),\s*vcc', nxt)
            if mc:
                nxt_indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
                result.append(f"{nxt_indent}v_cndmask_b32_e64 {mc.group(1)}, {sgpr_pair}")
                sgpr_idx += 1
                i = j + 1
                break
            else:
                result.append(lines[j].rstrip('\n'))
                j += 1
        else:
            sgpr_idx += 1
            i = j
        continue

    result.append(line)
    i += 1

max_sgpr = sgpr_base + sgpr_idx * 2
for idx, line in enumerate(result):
    if 'numbered_sgpr,' in line:
        result[idx] = re.sub(r'numbered_sgpr,\s*\d+', f'numbered_sgpr, {max_sgpr}', line)
    if '.sgpr_count:' in line:
        result[idx] = re.sub(r'sgpr_count:\s*\d+', f'sgpr_count:     {max_sgpr + 6}', line)

final_isa = '\n'.join(result) + '\n'
with open(sys.argv[2], 'w') as f:
    f.write(final_isa)

import os
dump_dir = os.environ.get('FLYDSL_DUMP_DIR', '')
if dump_dir:
    dump_path = os.path.join(dump_dir, 'postprocessed_final_isa.s')
    os.makedirs(dump_dir, exist_ok=True)
    with open(dump_path, 'w') as f:
        f.write(final_isa)
    print(f"[postprocess_vcc_to_sgpr] saved to {dump_path}", file=sys.stderr)
PYEOF
