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
import re
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
    return f"[{'█' * filled}{'░' * (BAR_WIDTH - filled)}] {pct:5.1f}%"

def fmt_cycles(n):
    if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(n)

def print_header(title):
    print(f"\n{'═' * 90}\n  {title}\n{'═' * 90}")


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
        asm_short = inst.asm[:47] + "…" if len(inst.asm) > 48 else inst.asm
        src_short = inst.source_loc[-38:] if len(inst.source_loc) > 38 else inst.source_loc
        print(f"  {rank:>3}  {fmt_cycles(inst.stall_cycles):>8}  {pct:>6.2f}%  "
              f"{inst.stall_type:<12}  {asm_short:<48}  {src_short}")


def print_source_detail(hotspot, source_cache, context=3):
    print(f"\n    ── {hotspot.source_loc}  "
          f"(stall={fmt_cycles(hotspot.total_stall_cycles)}, {hotspot.stall_pct:.0f}% stall rate)")
    snippet = get_source_snippet(source_cache, hotspot.source_loc, context=context)
    if snippet:
        print("    Source:")
        for lineno, text, is_hot in snippet:
            marker = ">>>" if is_hot else "   "
            print(f"      {marker} {lineno:4d} │ {text}")
    print("    Stalling instructions:")
    for inst in sorted(hotspot.instructions, key=lambda x: x.stall_cycles, reverse=True)[:6]:
        print(f"      stall={fmt_cycles(inst.stall_cycles):>7}  type={inst.stall_type:<12}  {inst.asm}")


def detect_arch_and_reg_pressure(instructions):
    """Detect GPU architecture from ISA and estimate VGPR usage + occupancy.

    Architecture differences (from CDNA4 ISA Reference Guide):
      - CDNA3 (gfx942): 256 arch_vgpr + 256 accum_vgpr, two SEPARATE pools.
        Occupancy = 256 / max(arch_vgpr_alloc, accum_vgpr_alloc).
      - CDNA4 (gfx950): 256 arch_vgpr (V0-V255) + 256 accum_vgpr (AV0-AV255),
        one COMBINED pool of 512, flexibly split between the two types.
        Occupancy = 512 / (arch_vgpr_alloc + accum_vgpr_alloc).
    VGPRs are allocated in groups of 8 on both architectures.
    """
    asms = [inst.asm for inst in instructions]

    # Detect architecture from gfx950-specific instructions
    is_gfx950 = any(
        "v_mfma_scale_f32" in a or "v_mfma_f32_16x16x128" in a or "v_mfma_f32_32x32x64" in a
        for a in asms
    )
    arch = "gfx950 (CDNA4)" if is_gfx950 else "gfx942 (CDNA3)"

    # Scan for max VGPR/AccVGPR indices
    max_vgpr = 0
    max_agpr = 0
    for a in asms:
        for m in re.finditer(r'\bv(\d+)\b', a):
            max_vgpr = max(max_vgpr, int(m.group(1)))
        for m in re.finditer(r'\bv\[(\d+)', a):
            max_vgpr = max(max_vgpr, int(m.group(1)))
        for m in re.finditer(r'\ba(\d+)\b', a):
            max_agpr = max(max_agpr, int(m.group(1)))
        for m in re.finditer(r'\ba\[(\d+)', a):
            max_agpr = max(max_agpr, int(m.group(1)))

    arch_vgpr_count = max_vgpr + 1
    accum_vgpr_count = max_agpr + 1 if max_agpr > 0 else 0

    # Round up to allocation granularity of 8
    arch_vgpr_alloc = ((arch_vgpr_count + 7) // 8) * 8
    accum_vgpr_alloc = ((accum_vgpr_count + 7) // 8) * 8 if accum_vgpr_count > 0 else 0

    # Occupancy calculation (architecture-dependent)
    max_occupancy = 8
    if is_gfx950:
        # CDNA4: combined pool of 512, flexibly split
        total_vgpr_alloc = arch_vgpr_alloc + accum_vgpr_alloc
        occupancy = min(512 // total_vgpr_alloc, max_occupancy) if total_vgpr_alloc > 0 else max_occupancy
        next_occ = occupancy + 1
        target_total = 512 // next_occ if next_occ <= max_occupancy else None
    else:
        # CDNA3: two separate pools of 256
        limiting = max(arch_vgpr_alloc, accum_vgpr_alloc)
        occupancy = min(256 // limiting, max_occupancy) if limiting > 0 else max_occupancy
        next_occ = occupancy + 1
        target_total = 256 // next_occ if next_occ <= max_occupancy else None

    # Instruction mix counts
    mfma_count = sum(1 for a in asms if "v_mfma_" in a)
    buf_load = sum(1 for a in asms if "buffer_load" in a)
    buf_store = sum(1 for a in asms if "buffer_store" in a)
    ds_read = sum(1 for a in asms if "ds_read" in a or "ds_load" in a)
    ds_write = sum(1 for a in asms if "ds_write" in a or "ds_store" in a)

    return {
        "arch": arch,
        "is_gfx950": is_gfx950,
        "arch_vgpr": arch_vgpr_count,
        "arch_vgpr_alloc": arch_vgpr_alloc,
        "accum_vgpr": accum_vgpr_count,
        "accum_vgpr_alloc": accum_vgpr_alloc,
        "occupancy": occupancy,
        "target_for_next_occ": target_total,
        "next_occ": next_occ if next_occ <= max_occupancy else None,
        "mfma_count": mfma_count,
        "buffer_load": buf_load,
        "buffer_store": buf_store,
        "ds_read": ds_read,
        "ds_write": ds_write,
    }


def print_reg_pressure(reg_info):
    print_header("Register Pressure & Occupancy")
    print(f"  Architecture:   {reg_info['arch']}")
    print(f"  arch_vgpr:      ~{reg_info['arch_vgpr']} (alloc {reg_info['arch_vgpr_alloc']})")
    if reg_info["accum_vgpr"] > 0:
        print(f"  accum_vgpr:     ~{reg_info['accum_vgpr']} (alloc {reg_info['accum_vgpr_alloc']})")
    else:
        print(f"  accum_vgpr:     0 (not used)")

    if reg_info["is_gfx950"]:
        total = reg_info["arch_vgpr_alloc"] + reg_info["accum_vgpr_alloc"]
        print(f"  total_vgpr:     {total} / 512 (combined pool)")
    else:
        lim = max(reg_info["arch_vgpr_alloc"], reg_info["accum_vgpr_alloc"])
        print(f"  limiting pool:  {lim} / 256")

    print(f"  occupancy:      {reg_info['occupancy']} waves/SIMD")
    if reg_info["next_occ"] is not None:
        if reg_info["is_gfx950"]:
            print(f"  -> {reg_info['next_occ']} waves requires total_vgpr <= {reg_info['target_for_next_occ']}")
        else:
            print(f"  -> {reg_info['next_occ']} waves requires max(arch,accum) <= {reg_info['target_for_next_occ']}")

    print(f"\n  Instruction mix:")
    print(f"    MFMA: {reg_info['mfma_count']},  buffer_load: {reg_info['buffer_load']},"
          f"  buffer_store: {reg_info['buffer_store']}")
    print(f"    ds_read: {reg_info['ds_read']},  ds_write: {reg_info['ds_write']}")


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

    reg_info = detect_arch_and_reg_pressure(instructions)
    print_reg_pressure(reg_info)

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
