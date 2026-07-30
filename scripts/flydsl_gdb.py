# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""FlyDSL rocgdb Python extension — GPU source-level stepping.

Load inside rocgdb with:
    (gdb) source scripts/flydsl_gdb.py

Commands added:
    flydsl-where            Show current Python source location and context
    flydsl-step [N]         Step to the next Python source line (≤ N ISA insns)
    flydsl-regs             Show VGPRs v0-v15, SGPRs s0-s15, EXEC mask
    flydsl-line-table       Dump the ISA ↔ Python line mapping for the kernel

Requirements:
    - rocgdb (ROCm 6.x+, found at /opt/rocm/bin/rocgdb)
    - Kernel compiled with FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1
    - GPU wavefront halted at a breakpoint

Limitations:
    - Only LineTablesOnly DWARF is available — no Python variable names.
    - Source lines shown are the fx.* call sites in your kernel .py file.
    - Use FLYDSL_DUMP_IR=1 to get the ISA dump and manually correlate
      registers to variables (see scripts/flydsl_dwarf_mapper.py).

Typical session:
    $ FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \\
          /opt/rocm/bin/rocgdb -x scripts/flydsl_gdb.py --args python3 repro.py
    (gdb) set breakpoint pending on
    (gdb) break my_kernel_0        # symbol = <func_name>_<sequential_id>
    (gdb) run
    (gdb) flydsl-where
    (gdb) flydsl-step
    (gdb) flydsl-regs
"""

import gdb
import os
import re
import struct


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_pc():
    """Return the current $pc for the active GPU wavefront, or None."""
    try:
        return int(gdb.parse_and_eval("$pc"))
    except gdb.error:
        return None


def _source_for_pc(pc):
    """Return (filepath, lineno) from DWARF for the given PC, or (None, None)."""
    try:
        sal = gdb.find_pc_line(pc)
        if sal.symtab and sal.line > 0:
            return sal.symtab.fullname(), sal.line
    except gdb.error:
        pass
    return None, None


def _source_context(filepath, line, radius=3):
    """Return a string showing `radius` lines of context around `line`."""
    try:
        with open(filepath) as f:
            lines = f.readlines()
    except OSError:
        return f"  (cannot read {filepath})"
    start = max(0, line - radius - 1)
    end = min(len(lines), line + radius)
    out = []
    for i in range(start, end):
        ln = i + 1
        marker = ">>>" if ln == line else "   "
        out.append(f"{marker} {ln:4d} | {lines[i].rstrip()}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

class FlyDSLWhereCommand(gdb.Command):
    """Show current Python source location for the halted GPU wavefront.

Usage: flydsl-where
    Resolves $pc via the DWARF line table to a Python source line and
    displays surrounding context from the .py file."""

    def __init__(self):
        super().__init__("flydsl-where", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        pc = _current_pc()
        if pc is None:
            gdb.write("Error: cannot read $pc — is a GPU wavefront selected?\n")
            return
        filepath, line = _source_for_pc(pc)
        if filepath is None:
            gdb.write(f"No source info for PC=0x{pc:x}\n")
            gdb.write("Hint: compile with FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1\n")
            return
        gdb.write(f"\n  PC  0x{pc:x}\n")
        gdb.write(f"  At  {filepath}:{line}\n\n")
        gdb.write(_source_context(filepath, line) + "\n\n")


class FlyDSLStepCommand(gdb.Command):
    """Step the GPU wavefront to the next Python source line.

Usage: flydsl-step [max_insns]
    Issues stepi until the DWARF line mapping changes.
    max_insns: safety limit on ISA instructions (default 200).
    If the limit is hit, run 'flydsl-step 1000' (may be in a long
    compiler-unrolled sequence)."""

    def __init__(self):
        super().__init__("flydsl-step", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        limit = 200
        if arg.strip():
            try:
                limit = int(arg.strip())
            except ValueError:
                gdb.write("Usage: flydsl-step [max_insns]\n")
                return

        pc = _current_pc()
        if pc is None:
            gdb.write("Error: cannot read $pc\n")
            return
        start_file, start_line = _source_for_pc(pc)
        if start_file is None:
            gdb.write("No source info at current PC; try 'stepi' manually.\n")
            return

        gdb.write(f"Stepping from {os.path.basename(start_file)}:{start_line} …\n")
        for i in range(limit):
            gdb.execute("stepi", to_string=True)
            new_pc = _current_pc()
            if new_pc is None:
                gdb.write("Lost GPU wavefront context.\n")
                return
            new_file, new_line = _source_for_pc(new_pc)
            if new_file is None or new_line == 0:
                continue
            if new_line != start_line or new_file != start_file:
                gdb.write(f"→ {os.path.basename(new_file)}:{new_line}  (0x{new_pc:x}, {i+1} ISA insns)\n\n")
                gdb.write(_source_context(new_file, new_line) + "\n\n")
                return

        gdb.write(f"Reached {limit} ISA instructions without a line change.\n")
        gdb.write("The kernel may be in a long unrolled sequence; try 'flydsl-step 2000'.\n")


class FlyDSLRegsCommand(gdb.Command):
    """Show GPU register state for the current wavefront.

Usage: flydsl-regs
    Displays VGPRs v0-v15 (hex + float interpretation), SGPRs s0-s15,
    and the EXEC mask.  Use the ISA dump to map registers to variables."""

    def __init__(self):
        super().__init__("flydsl-regs", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        try:
            exec_val = int(gdb.parse_and_eval("$exec"))
            gdb.write(f"  EXEC = 0x{exec_val:016x}\n")
        except gdb.error:
            pass

        gdb.write("\n  VGPRs (hex / float32):\n")
        for i in range(16):
            try:
                raw = int(gdb.parse_and_eval(f"$v{i}")) & 0xFFFFFFFF
                fval = struct.unpack("f", struct.pack("I", raw))[0]
                gdb.write(f"    v{i:2d} = 0x{raw:08x}  ({fval:+.6g})\n")
            except gdb.error:
                break

        gdb.write("\n  SGPRs:\n")
        for i in range(16):
            try:
                raw = int(gdb.parse_and_eval(f"$s{i}")) & 0xFFFFFFFF
                gdb.write(f"    s{i:2d} = 0x{raw:08x}\n")
            except gdb.error:
                break
        gdb.write("\n")


class FlyDSLLineTableCommand(gdb.Command):
    """Dump the ISA ↔ Python line mapping for the current kernel.

Usage: flydsl-line-table
    Probes GDB's internal line table and shows which Python source lines
    have ISA code with their address ranges."""

    def __init__(self):
        super().__init__("flydsl-line-table", gdb.COMMAND_USER)

    def invoke(self, arg, from_tty):
        pc = _current_pc()
        if pc is None:
            gdb.write("Error: cannot read $pc\n")
            return
        filepath, _ = _source_for_pc(pc)
        if filepath is None:
            gdb.write("No debug info at current PC.\n")
            return

        gdb.write(f"\nISA ↔ Python line table\n")
        gdb.write(f"Source: {filepath}\n\n")
        seen = set()
        for probe in range(1, 500):
            try:
                out = gdb.execute(f"info line {filepath}:{probe}", to_string=True)
                if "No line" in out or "out of range" in out:
                    continue
                if probe in seen:
                    continue
                seen.add(probe)
                m = re.search(r"starts at address (0x[0-9a-f]+).*ends at (0x[0-9a-f]+)", out)
                if m:
                    gdb.write(f"  line {probe:4d}: {m.group(1)} – {m.group(2)}\n")
            except gdb.error:
                pass
        if not seen:
            gdb.write("  (no line info found)\n")
        gdb.write("\n")


# ---------------------------------------------------------------------------
# Register and print banner
# ---------------------------------------------------------------------------

FlyDSLWhereCommand()
FlyDSLStepCommand()
FlyDSLRegsCommand()
FlyDSLLineTableCommand()

gdb.write(
    "\nFlyDSL GPU debugger extensions loaded.\n"
    "  flydsl-where            current Python source location\n"
    "  flydsl-step [N]         step to next Python line (≤ N ISA insns)\n"
    "  flydsl-regs             VGPRs / SGPRs / EXEC\n"
    "  flydsl-line-table       full ISA ↔ Python line map\n"
    "Requires: FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 at compile time.\n"
    "No Python variable names available (LineTablesOnly DWARF);\n"
    "use 'scripts/flydsl_dwarf_mapper.py' to correlate registers.\n\n"
)
