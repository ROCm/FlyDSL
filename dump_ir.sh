#!/usr/bin/bash
# Usage:  ./dump_ir.sh <example.py> [output_dir]
#
# Runs the example with IR printing enabled, then splits each pass's
# IR dump into a numbered file under <output_dir> (default: ./ir_dump).
set -euo pipefail

EXAMPLE="${1:?Usage: $0 <example.py> [output_dir]}"
OUTDIR="${2:-./ir_dump}"

rm -rf "$OUTDIR" ~/.flydsl/cache/
mkdir -p "$OUTDIR"

cd /home/jli10004/flydsl/flydsl-prev
source after_build.sh 2>/dev/null || true

export HIP_VISIBLE_DEVICES=0
export FLYDSL_DEBUG_PRINT_ORIGIN_IR=1
export FLYDSL_DEBUG_PRINT_AFTER_ALL=1
export FLYDSL_DEBUG_LOG_TO_CONSOLE=1
export FLYDSL_DEBUG_LOG_LEVEL=INFO

python "$EXAMPLE" >"$OUTDIR/_raw.txt" 2>&1

python3 -c "
import re, sys, os

outdir = '$OUTDIR'
with open(os.path.join(outdir, '_raw.txt')) as f:
    text = f.read()

sections = []

# Origin IR
m = re.search(r'Origin IR:\s*\n(module\b.*?)(?=\n// -----// IR Dump|\Z)', text, re.DOTALL)
if m:
    sections.append(('origin_ir', m.group(1).rstrip()))

# Per-pass IR
marker = re.compile(r'^// -----// IR Dump After (\S+)(?: \(.*?\))? //----- //\$', re.MULTILINE)
hits = list(marker.finditer(text))
for i, h in enumerate(hits):
    end = hits[i+1].start() if i+1 < len(hits) else len(text)
    body = text[h.end()+1:end].rstrip()
    sections.append((h.group(1), body))

for seq, (name, body) in enumerate(sections):
    safe = re.sub(r'[^\w-]', '_', name)
    fn = f'{seq:02d}_{safe}.mlir'
    with open(os.path.join(outdir, fn), 'w') as f:
        f.write(body + '\n')
    print(f'  {fn}')

os.remove(os.path.join(outdir, '_raw.txt'))
print(f'\n{len(sections)} IR files written to {outdir}/')
"
