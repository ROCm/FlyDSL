---
name: kernel-trace-analysis
description: >
  Profile GPU kernels using rocprofv3 to collect ATT instruction-level traces, then
  analyze the trace data using hotspot_analyzer.py to identify top-K stall hotspots
  (VMEM-load, VMEM-wait, LDS/SMEM-wait, barrier, MFMA stalls) mapped back to source
  lines, and produce an actionable optimization plan.
  Usage: /kernel-trace-analysis <cmd>
  Can also analyze an existing dispatch dir directly: /kernel-trace-analysis --dir <path>
tools: Read,Edit,Bash,Grep,Glob,Agent,Write
note: All analysis is done programmatically via hotspot_analyzer.py + code.json. Do NOT use GUI tools.
---

# Kernel Trace Analysis

Profile and analyze GPU kernel ATT traces to identify stall hotspots and produce
an optimization plan.

## Arguments

| Argument | Description |
|----------|-------------|
| `<CMD>` | Command to profile. Example: `python bench_pa.py --batch 32` |
| `--dir <path>` | Skip collection; analyze existing `ui_output_agent_*_dispatch_*` directory |
| `--topk N` | Show top-N hotspots (default: 15) |

---

## Hotspot Analyzer Script

**Always write this script to `/tmp/hotspot_analyzer.py` before analysis.**
It reads a `ui_output_agent_*_dispatch_*` directory and reports top-K stall hotspots.

```python
"""
GPU Kernel Hotspot Analyzer
Reads rocprof-compute ATT trace output and identifies top-K stall hotspots.

Usage:
    python hotspot_analyzer.py <dispatch_dir> [--topk N] [--mode {asm,src,both}]
    python hotspot_analyzer.py <dispatch_dir> --topk 5 --mode src --detail --context 4
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Instruction:
    asm: str
    pc_index: int
    source_loc: str
    pc_addr: int
    exec_count: int
    total_cycles: int
    stall_cycles: int
    issue_cycles: int

    @property
    def stall_pct(self):
        return 100.0 * self.stall_cycles / self.total_cycles if self.total_cycles else 0.0

    @property
    def stall_type(self):
        asm = self.asm.lower()
        if "s_waitcnt" in asm:
            if "vmcnt" in asm:   return "VMEM-wait"
            if "lgkmcnt" in asm: return "LDS/SMEM-wait"
            if "expcnt" in asm:  return "EXP-wait"
            return "waitcnt"
        if "s_barrier" in asm or "s_wait_idle" in asm: return "barrier"
        if "buffer_load" in asm or "global_load" in asm or "flat_load" in asm: return "VMEM-load"
        if "buffer_store" in asm or "global_store" in asm: return "VMEM-store"
        if "ds_read" in asm or "ds_write" in asm: return "LDS"
        if "s_load" in asm or "s_store" in asm: return "SMEM"
        if "v_mfma" in asm or "v_fma" in asm: return "MFMA/FMA"
        return "other"


@dataclass
class SourceLineHotspot:
    source_loc: str
    total_stall_cycles: int = 0
    total_cycles: int = 0
    instructions: list = field(default_factory=list)

    @property
    def stall_pct(self):
        return 100.0 * self.total_stall_cycles / self.total_cycles if self.total_cycles else 0.0

    @property
    def dominant_stall_type(self):
        by_type = defaultdict(int)
        for inst in self.instructions:
            by_type[inst.stall_type] += inst.stall_cycles
        return max(by_type, key=by_type.get) if by_type else "other"


def load_source_map(dispatch_dir):
    """Parse snapshots.json nested tree -> {virtual_path: [source_lines]}."""
    snap_path = os.path.join(dispatch_dir, "snapshots.json")
    if not os.path.exists(snap_path):
        return {}
    with open(snap_path) as f:
        tree = json.load(f)

    path_map = {}
    def _walk(node, prefix):
        for key, val in node.items():
            segment = "" if key == "/" else key
            path = prefix.rstrip("/") + "/" + segment if segment else prefix
            if isinstance(val, dict):
                _walk(val, path)
            else:
                path_map[path] = val
    _walk(tree, "")

    source_cache = {}
    for vpath, local_name in path_map.items():
        local_path = os.path.join(dispatch_dir, local_name)
        if os.path.exists(local_path):
            with open(local_path) as f:
                source_cache[vpath] = f.readlines()
    return source_cache


def get_source_snippet(source_cache, source_loc, context=3):
    if ":" not in source_loc:
        return []
    path, lineno_str = source_loc.rsplit(":", 1)
    try:
        lineno = int(lineno_str)
    except ValueError:
        return []
    lines = source_cache.get(path)
    if not lines:
        return []
    start = max(0, lineno - context - 1)
    end = min(len(lines), lineno + context)
    return [(i + 1, lines[i].rstrip(), i + 1 == lineno) for i in range(start, end)]


def load_instructions(dispatch_dir):
    with open(os.path.join(dispatch_dir, "code.json")) as f:
        data = json.load(f)
    instructions = []
    for row in data["code"]:
        if not isinstance(row[2], int) or row[2] == 0:
            continue
        instructions.append(Instruction(
            asm=row[0],
            pc_index=row[2],
            source_loc=row[3] if row[3] else "<unknown>",
            pc_addr=row[5],
            exec_count=row[6] if isinstance(row[6], int) else 0,
            total_cycles=row[7] if isinstance(row[7], int) else 0,
            stall_cycles=row[8] if isinstance(row[8], int) else 0,
            issue_cycles=row[9] if isinstance(row[9], int) else 0,
        ))
    return instructions


def aggregate_by_source(instructions):
    by_src = {}
    for inst in instructions:
        loc = inst.source_loc
        if loc not in by_src:
            by_src[loc] = SourceLineHotspot(source_loc=loc)
        hs = by_src[loc]
        hs.total_stall_cycles += inst.stall_cycles
        hs.total_cycles += inst.total_cycles
        if inst.stall_cycles > 0:
            hs.instructions.append(inst)
    return sorted(by_src.values(), key=lambda x: x.total_stall_cycles, reverse=True)


BAR_WIDTH = 30

def stall_bar(pct):
    filled = int(pct / 100 * BAR_WIDTH)
    return f"[{'#' * filled}{'-' * (BAR_WIDTH - filled)}] {pct:5.1f}%"

def fmt_cycles(n):
    if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(n)

def print_header(title):
    print(f"\n{'=' * 90}\n  {title}\n{'=' * 90}")


def print_stall_type_summary(instructions, total_stall):
    print_header("Stall Breakdown by Type")
    by_type = defaultdict(int)
    for inst in instructions:
        if inst.stall_cycles > 0:
            by_type[inst.stall_type] += inst.stall_cycles
    print(f"  {'Type':<14}  {'Stall':>8}  Bar")
    print(f"  {'-'*14}  {'-'*8}  {'-'*38}")
    for stype, cycles in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        pct = 100.0 * cycles / total_stall if total_stall else 0
        print(f"  {stype:<14}  {fmt_cycles(cycles):>8}  {stall_bar(pct)}")


def print_source_hotspots(hotspots, topk, total_stall):
    print_header(f"Top-{topk} Hotspot Source Lines  (stall cycles aggregated)")
    print(f"  {'#':>3}  {'Stall':>8}  {'%Total':>7}  {'StallBar':<38}  {'DomType':<12}  Source")
    print(f"  {'-'*3}  {'-'*8}  {'-'*7}  {'-'*38}  {'-'*12}  {'-'*40}")
    for rank, hs in enumerate(hotspots[:topk], 1):
        if hs.total_stall_cycles == 0:
            break
        pct = 100.0 * hs.total_stall_cycles / total_stall if total_stall else 0
        src_short = hs.source_loc[-48:] if len(hs.source_loc) > 48 else hs.source_loc
        print(f"  {rank:>3}  {fmt_cycles(hs.total_stall_cycles):>8}  {pct:>6.2f}%  "
              f"{stall_bar(hs.stall_pct):<38}  {hs.dominant_stall_type:<12}  {src_short}")


def print_asm_hotspots(instructions, topk, total_stall):
    print_header(f"Top-{topk} Hotspot Instructions  (by stall cycles)")
    print(f"  {'#':>3}  {'Stall':>8}  {'%Total':>7}  {'Type':<12}  {'ASM':<48}  Source")
    print(f"  {'-'*3}  {'-'*8}  {'-'*7}  {'-'*12}  {'-'*48}  {'-'*30}")
    ranked = sorted([i for i in instructions if i.stall_cycles > 0],
                    key=lambda x: x.stall_cycles, reverse=True)[:topk]
    for rank, inst in enumerate(ranked, 1):
        pct = 100.0 * inst.stall_cycles / total_stall if total_stall else 0
        asm_short = inst.asm[:47] + "..." if len(inst.asm) > 48 else inst.asm
        src_short = inst.source_loc[-38:] if len(inst.source_loc) > 38 else inst.source_loc
        print(f"  {rank:>3}  {fmt_cycles(inst.stall_cycles):>8}  {pct:>6.2f}%  "
              f"{inst.stall_type:<12}  {asm_short:<48}  {src_short}")


def print_source_detail(hotspot, source_cache, context=3):
    print(f"\n    -- {hotspot.source_loc}  "
          f"(stall={fmt_cycles(hotspot.total_stall_cycles)}, {hotspot.stall_pct:.0f}% stall rate)")
    snippet = get_source_snippet(source_cache, hotspot.source_loc, context=context)
    if snippet:
        print("    Source:")
        for lineno, text, is_hot in snippet:
            marker = ">>>" if is_hot else "   "
            print(f"      {marker} {lineno:4d} | {text}")
    print("    Stalling instructions:")
    for inst in sorted(hotspot.instructions, key=lambda x: x.stall_cycles, reverse=True)[:6]:
        print(f"      stall={fmt_cycles(inst.stall_cycles):>7}  type={inst.stall_type:<12}  {inst.asm}")


def main():
    parser = argparse.ArgumentParser(description="GPU kernel hotspot analyzer")
    parser.add_argument("dispatch_dir", help="Path to ATT dispatch output directory")
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--mode", choices=["asm", "src", "both"], default="both")
    parser.add_argument("--detail", action="store_true",
                        help="Show source snippet + instruction breakdown under each source hotspot")
    parser.add_argument("--context", type=int, default=3,
                        help="Source lines of context around hotspot (default: 3)")
    args = parser.parse_args()

    if not os.path.isdir(args.dispatch_dir):
        print(f"Error: directory not found: {args.dispatch_dir}")
        return 1

    print(f"\nLoading: {args.dispatch_dir}")
    instructions = load_instructions(args.dispatch_dir)
    source_hotspots = aggregate_by_source(instructions)
    source_cache = load_source_map(args.dispatch_dir)

    total_stall  = sum(i.stall_cycles  for i in instructions)
    total_cycles = sum(i.total_cycles  for i in instructions)

    print(f"\n  Kernel:        {os.path.basename(args.dispatch_dir)}")
    print(f"  Instructions:  {len(instructions):,}")
    print(f"  Total cycles:  {fmt_cycles(total_cycles)}")
    print(f"  Total stalls:  {fmt_cycles(total_stall)}  ({100*total_stall/total_cycles:.1f}% of total cycles)")

    print_stall_type_summary(instructions, total_stall)

    if args.mode in ("src", "both"):
        print_source_hotspots(source_hotspots, args.topk, total_stall)
        if args.detail:
            for hs in source_hotspots[:min(5, args.topk)]:
                if hs.total_stall_cycles > 0:
                    print_source_detail(hs, source_cache, context=args.context)

    if args.mode in ("asm", "both"):
        print_asm_hotspots(instructions, args.topk, total_stall)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## Workflow

### Mode A: Analyze existing dispatch directory

If the user provides `--dir <path>` or already has a `ui_output_agent_*_dispatch_*` directory:

```bash
# Write hotspot_analyzer.py (see above), then:
python /tmp/hotspot_analyzer.py <dispatch_dir> --topk 15 --mode both
python /tmp/hotspot_analyzer.py <dispatch_dir> --topk 5 --mode src --detail --context 4
```

Skip to **Step 4: Interpret Results**.

---

### Mode B: Full collection workflow

#### Step 1: Kernel Discovery

```bash
touch /tmp/trace_ts
rocprofv3 --stats --kernel-trace -f csv -- <CMD> 2>&1
find . -maxdepth 3 -name "*stats*" -newer /tmp/trace_ts -type f 2>/dev/null
```

Parse the stats CSV and present a kernel table:

| Rank | Kernel Name | Calls | Total (us) | Avg (us) | % GPU Time |
|------|-------------|-------|------------|----------|------------|

Ask the user which kernel to trace if not obvious.

**Prefer `results.db`** if available -- use sqlite3 for structured queries:
```bash
sqlite3 results.db "
SELECT ks.KernelName, COUNT(*) calls,
       ROUND(AVG(kd.end-kd.start)/1000.0,1) avg_us
FROM rocpd_kernel_dispatch kd
JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id=ks.id
GROUP BY ks.KernelName ORDER BY avg_us DESC LIMIT 20;"
```

#### Step 2: Configure input.yaml

```yaml
jobs:
   -
       kernel_include_regex: <KERNEL_NAME_PATTERN>
       kernel_iteration_range: "[1, [3-4]]"
       output_file: out
       output_directory: kernel_trace_output
       output_format: [csv]
       truncate_kernels: true
       sys_trace: true
       advanced_thread_trace: true
       att_target_cu: 1
       att_shader_engine_mask: "0xf"
       att_simd_select: "0xf"
       att_buffer_size: "0x6000000"
```

Key notes:
- `kernel_iteration_range`: `"[1, [3-4]]"` skips warmup, traces dispatches 3-4
- `att_buffer_size`: 96MB per SE; increase to `"0xC000000"` if truncated
- `att_target_cu: 1`: single CU keeps output manageable

#### Step 3: Collect ATT Trace

```bash
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 rocprofv3 -i /tmp/trace_input.yaml -- <CMD> 2>&1
find . -type d -name "ui_output_agent_*" -newer /tmp/trace_ts 2>/dev/null
```

If `rocprof-trace-decoder` library is missing:
```bash
wget -q https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh
chmod +x rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh
./rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh --skip-license --prefix=/tmp/rtd-install
find /tmp/rtd-install -name '*.so*' -exec cp -a {} /opt/rocm/lib/ \;
ldconfig
```

**Output structure:**
```
ui_output_agent_<PID>_dispatch_<N>/
├── code.json          <- PRIMARY: per-instruction stall/cycle data
├── snapshots.json     <- source file path mapping (virtual -> local filename)
├── source_0_*.py      <- embedded source files
├── filenames.json     <- wave file index
├── occupancy.json     <- occupancy timeline
└── se*_sm*_sl*_wv*.json  <- per-wave raw traces
```

---

## Step 4: Run hotspot_analyzer.py

Write the script (see above), then run:

```bash
# Full report
python /tmp/hotspot_analyzer.py <dispatch_dir> --topk 15 --mode both

# Source-level with code context (best for optimization)
python /tmp/hotspot_analyzer.py <dispatch_dir> --topk 5 --mode src --detail --context 4

# ASM-only for instruction-level detail
python /tmp/hotspot_analyzer.py <dispatch_dir> --mode asm --topk 20
```

---

## Step 5: Interpret Results

### code.json field reference

Each row in `code["code"]` is:
```
[asm, _, pc_index, source_loc, _, pc_addr, exec_count, total_cycles, stall_cycles, issue_cycles]
  0   1     2          3       4     5          6            7              8             9
```

- **col[8] `stall_cycles`**: cycles the instruction was blocked from issuing -- **primary hotspot metric**
- **col[7] `total_cycles`**: total cycles charged to this instruction across all waves
- **col[3] `source_loc`**: `"/path/to/file.py:LINE"` -- virtual path resolved via `snapshots.json`
- **col[6] `exec_count`**: number of wave-threads that executed this instruction

### snapshots.json: resolving source paths

`snapshots.json` encodes a nested dict tree mapping virtual paths to local filenames:
```json
{"/": {"FlyDSL": {"kernels": {"pa_decode_sw_fp8_ps.py": "source_0_pa_decode_sw_fp8_ps.py"}}}}
```
Flatten recursively: `/FlyDSL/kernels/pa_decode_sw_fp8_ps.py` -> `source_0_pa_decode_sw_fp8_ps.py`

### Stall type classification

| Type | Instructions | Root Cause |
|------|-------------|------------|
| `VMEM-load` | `buffer_load_*`, `global_load_*` | Load itself stalled (VMEM queue full or back-pressure from no compute to hide behind) |
| `VMEM-wait` | `s_waitcnt vmcnt(N)` | Waiting for outstanding VMEM loads to complete |
| `LDS/SMEM-wait` | `s_waitcnt lgkmcnt(N)` | Waiting for LDS or SMEM ops |
| `barrier` | `s_barrier` | Cross-wave sync -- slowest wave dominates |
| `MFMA/FMA` | `v_mfma_*` | MFMA dependency chain (RAW hazard) |
| `LDS` | `ds_read_*`, `ds_write_*` | LDS access latency |

### Common hotspot patterns

#### Pattern 1: V/K loads inside MFMA loop -> very high stall rate (80-95%)

```python
# BAD: load and MFMA alternate -- only 1 MFMA of hiding time
for k_step in range_constexpr(QKHELOOP * 2):
    if k_step % 2 == 0:
        v_data = buffer_ops.buffer_load(...)   # stall_rate ~92%
    acc = rocdl.mfma_f32_16x16x32_fp8_fp8(...)

# GOOD: batch all loads before the MFMA loop
for td in range_constexpr(TLOOP):
    v_prefetch[td] = [buffer_ops.buffer_load(...) for _ in range_constexpr(QKHELOOP)]

for td in range_constexpr(TLOOP):
    for k_step in range_constexpr(QKHELOOP * 2):
        acc = rocdl.mfma_f32_16x16x32_fp8_fp8(...)   # entire QK MFMA hides VMEM latency
    v_results[td] = v_prefetch[td]   # already in registers
```

#### Pattern 2: Sequential loads with no compute -> VMEM queue saturation

```python
# BAD: all loads back-to-back, no compute interleaved
for td in range_constexpr(TLOOP):
    for qkhe in range_constexpr(QKHELOOP):
        k4 = buffer_ops.buffer_load(k_rsrc, ka_dw, ...)   # queue fills up

# GOOD: prefetch next tile's K loads during current tile's MFMA computation
```

#### Pattern 3: LDS prob reads immediately before PV MFMA -> lgkmcnt stall

```python
# BAD: LDS reads and MFMA in same loop
for vhe in ...:
    for vt in ...:
        p_i64 = lds_read(...)    # issued here
        tmp = mfma(v_i64, p_i64, ...)   # immediately consumed -> lgkmcnt stall

# GOOD: batch all LDS reads first, then all MFMAs
for vhe in ...:
    for vt in ...:
        p_i64s.append(lds_read(...))    # all LDS reads issued first

for vhe in ...:
    for vt in ...:
        tmp = mfma(v_i64s[...], p_i64s[...], ...)   # LDS data already ready
```

#### Pattern 4: Scale loads too close to usage

```python
# BAD: scale load and usage separated by only TLOOP MFMAs
for td in range_constexpr(TLOOP):
    k_scale = buffer_ops.buffer_load(ks_rsrc, ...)   # issued here
# ... small compute gap ...
    result = acc * k_scale   # used too soon -> stall

# GOOD: issue scale loads at the very beginning of the block,
# before K loads, to maximise latency hiding distance
```

#### Pattern 5: Hotspot attributed to kernel entry line

When `@flyc.kernel` / kernel decorator line appears as the top hotspot with a mix of
VMEM-wait + barrier stall types -- this is a **debug info aggregation artifact**.
MLIR/compiler-generated instructions (address arithmetic, cndmask, prologue setup) map
to the outermost scope line. Ignore this line; focus on lines with explicit user ops.

### Register pressure check (CDNA3 / gfx942)

CDNA3 has two separate register files: `arch_vgpr` (VALU/VMEM) and `accum_vgpr` (MFMA results).
Occupancy = `256 / max(arch_vgpr, accum_vgpr)`.

```bash
sqlite3 results.db "
SELECT ks.KernelName, ki.arch_vgpr_count, ki.accum_vgpr_count, ki.lds_size
FROM rocpd_kernel_dispatch kd
JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id=ks.id
JOIN rocpd_info_kernel ki ON kd.kernel_id=ki.id LIMIT 5;"
```

Or estimate from ISA trace:
```python
import re, json
with open("code.json") as f: code = json.load(f)["code"]
asms = [r[0] for r in code]
max_vgpr = max((int(m.group(1)) for a in asms for m in re.finditer(r'\bv\[?(\d+)', a)), default=0)
max_agpr = max((int(m.group(1)) for a in asms for m in re.finditer(r'\ba\[?(\d+)', a)), default=0)
print(f"arch_vgpr ~{max_vgpr+1}, accum_vgpr ~{max_agpr+1}, occupancy ~{256//max(max_vgpr+1,max_agpr+1,1)} waves/SIMD")
```

**Warning**: `maxnreg` forcing `accum_vgpr=0` doubles occupancy but causes MFMA spills through
arch_vgpr -- measured 4.5x GPU slowdown. Do not use `maxnreg` for MFMA-heavy kernels.

---

## Step 6: Optimization Plan

After running `hotspot_analyzer.py --detail`, produce a prioritized plan:

```
## Stall Summary
- Total stalls: X cycles (Y% of kernel)
- Top stall type: VMEM-load (Z%)

## Hotspot Analysis

### #1 :LINE  stall=XK (N%)  VMEM-load  stall_rate=92%
Root cause: buffer_load inside QK MFMA loop -- only 1 MFMA of hiding time.
Fix: Move all V loads before the QK MFMA loop.
Estimated gain: ~20% kernel cycle reduction.

### #2 :LINE  stall=XK (N%)  VMEM-load  stall_rate=80%
Root cause: K loads sequential with no compute interleaved.
Fix: Prefetch next tile's K during current tile's MFMA (double-buffer pattern).
See /prefetch-data-load skill.

### #3 ...

## Priority Order
1. [HIGH]  Fix V-load position (24% of all stalls, easy refactor)
2. [HIGH]  K-load cross-tile prefetch (8% of stalls, needs _process_block restructure)
3. [MED]   Move scale loads earlier (8% of stalls, trivial move)
4. [LOW]   Batch LDS reads before PV MFMA (4% of stalls, loop split)
```

---

## Error Handling

| Error | Fix |
|-------|-----|
| `rocprof-trace-decoder library path not found` | Install decoder .so (see Step 3) |
| Trace output empty | Check `kernel_include_regex` matches exactly |
| Trace truncated | Increase `att_buffer_size` to `"0xC000000"` |
| `kernel_iteration_range` mismatch | Adjust range; try `"[0, [1-2]]"` |
| `INVALID_SHADER_DATA` | aqlprofile/decoder version mismatch -- update both |
| Source loc all `""` | Set `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`; check `-g` flag in compile pipeline |
| Top hotspot is kernel decorator line | Debug info artifact -- skip it, focus on op lines |
| `--att` flag error | `--att` is boolean, no value; use `-i input.yaml` for full config |
