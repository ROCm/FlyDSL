#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""FlyDSL DWARF line mapper — show the ISA ↔ Python-line mapping for a kernel.

Extracts the HSACO ELF binary from a FlyDSL MLIR dump file (produced by
FLYDSL_DUMP_IR=1) or reads a pre-extracted HSACO, then parses the DWARF
line table to show which ISA addresses correspond to which Python source lines.

Usage:
    # From an MLIR dump (FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/dump ...):
    python3 scripts/flydsl_dwarf_mapper.py /tmp/dump/<kernel>/20_gpu_module_to_binary.mlir

    # From a pre-extracted HSACO:
    python3 scripts/flydsl_dwarf_mapper.py --hsaco kernel.hsaco

    # Save HSACO while mapping:
    python3 scripts/flydsl_dwarf_mapper.py dump.mlir --save-hsaco kernel.hsaco

Requires:
    pip install pyelftools
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# HSACO extraction from MLIR text
# ---------------------------------------------------------------------------

def extract_hsaco_from_mlir(mlir_path: str) -> bytes:
    """Parse the MLIR LLVM-escaped string in the gpu.binary bin= attribute."""
    with open(mlir_path) as f:
        text = f.read()

    m = re.search(r'bin\s*=\s*"', text)
    if not m:
        raise ValueError(f"No 'bin = \"...\"' attribute found in {mlir_path}\n"
                         "Is this the gpu-module-to-binary output file?")

    start = m.end()
    raw = bytearray()
    i = start
    while i < len(text):
        ch = text[i]
        if ch == '"':
            break
        if ch == '\\' and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == '\\':
                raw.append(ord('\\'));  i += 2; continue
            if nxt == '"':
                raw.append(ord('"'));   i += 2; continue
            if nxt == 'n':
                raw.append(ord('\n')); i += 2; continue
            if i + 2 < len(text):
                try:
                    raw.append(int(text[i + 1:i + 3], 16))
                    i += 3
                    continue
                except ValueError:
                    pass
        raw.append(ord(ch))
        i += 1

    if raw[:4] != b'\x7fELF':
        raise ValueError("Extracted data is not a valid ELF binary (bad magic).")
    return bytes(raw)


# ---------------------------------------------------------------------------
# DWARF parsing
# ---------------------------------------------------------------------------

def parse_dwarf_lines(hsaco_bytes: bytes):
    """Return (entries, file_map) from the DWARF .debug_line section."""
    try:
        import io
        from elftools.elf.elffile import ELFFile
    except ImportError:
        sys.exit("pyelftools is required: pip install pyelftools")

    elf = ELFFile(io.BytesIO(hsaco_bytes))
    if not elf.has_dwarf_info():
        sys.exit(
            "HSACO has no DWARF info.\n"
            "Recompile with FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1."
        )

    dwarfinfo = elf.get_dwarf_info()
    entries = []
    file_map = {}

    for CU in dwarfinfo.iter_CUs():
        lp = dwarfinfo.line_program_for_CU(CU)
        if lp is None:
            continue

        # Build file index → path mapping
        inc_dirs = [b"."]
        if lp["include_directory"]:
            inc_dirs.extend(lp["include_directory"])

        local_file_map = {}
        for idx, fe in enumerate(lp["file_entry"], 1):
            d = inc_dirs[fe.dir_index] if fe.dir_index < len(inc_dirs) else b"."
            d = d.decode("utf-8", errors="replace") if isinstance(d, bytes) else d
            n = fe.name.decode("utf-8", errors="replace") if isinstance(fe.name, bytes) else fe.name
            local_file_map[idx] = (d, n, f"{d}/{n}")

        file_map.update(local_file_map)

        for e in lp.get_entries():
            s = e.state
            if s is None or s.end_sequence:
                continue
            fi = local_file_map.get(s.file, ("?", "?", "?"))
            entries.append({
                "address": s.address,
                "line":    s.line,
                "column":  s.column,
                "file":    fi[2],
                "fname":   fi[1],
                "is_stmt": s.is_stmt,
            })

    return entries, file_map


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_report(entries, file_map):
    print("\n=== Source files ===")
    for idx, (d, n, full) in sorted(file_map.items()):
        print(f"  [{idx}] {full}")

    print(f"\n=== Full line table ===")
    print(f"  {'Address':>18}  {'Line':>5}  {'Col':>3}  {'Stmt':>4}  File")
    print("  " + "-" * 65)
    for e in entries:
        stmt = "*" if e["is_stmt"] else " "
        print(f"  0x{e['address']:016x}  {e['line']:5d}  {e['column']:3d}  {stmt:>4s}  {e['fname']}")

    by_line: dict = defaultdict(list)
    for e in entries:
        if e["line"] > 0:
            by_line[(e["file"], e["line"])].append(e["address"])

    print(f"\n=== Python-line summary ===")
    print(f"  {'Line':>6}  {'ISA start':>18}  {'ISA end':>18}  {'#entries':>8}  File")
    print("  " + "-" * 75)
    for (fpath, line), addrs in sorted(by_line.items()):
        fname = Path(fpath).name
        print(f"  {line:6d}  0x{min(addrs):016x}  0x{max(addrs):016x}  {len(addrs):8d}  {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("mlir_path", nargs="?",
                     help="Path to the gpu-module-to-binary MLIR dump file")
    src.add_argument("--hsaco", metavar="FILE",
                     help="Path to a pre-extracted HSACO ELF binary")
    ap.add_argument("--save-hsaco", metavar="FILE",
                    help="Save extracted HSACO to this path")
    args = ap.parse_args()

    if args.hsaco:
        data = Path(args.hsaco).read_bytes()
        print(f"Loaded {len(data):,} bytes from {args.hsaco}")
    else:
        data = extract_hsaco_from_mlir(args.mlir_path)
        print(f"Extracted {len(data):,}-byte HSACO from {args.mlir_path}")

    if args.save_hsaco:
        Path(args.save_hsaco).write_bytes(data)
        print(f"Saved HSACO to {args.save_hsaco}")

    entries, file_map = parse_dwarf_lines(data)
    print_report(entries, file_map)


if __name__ == "__main__":
    main()
