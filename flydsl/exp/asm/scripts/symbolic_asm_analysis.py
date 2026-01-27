#!/usr/bin/env python3
"""
Symbolic analysis of assembly code using SymPy.
Builds symbolic expressions for registers to analyze instruction operands.

Usage:
    python symbolic_asm_analysis.py <asm_file> <line_number_or_pattern>

Examples:
    python symbolic_asm_analysis.py wave_asm.s 124
    python symbolic_asm_analysis.py wave_asm.s "buffer_load_dword"
    python symbolic_asm_analysis.py llvm_asm.s "buffer_load_dword.*lds"
"""

import re
import sys
import sympy
from sympy import Symbol, floor, Integer, simplify, Function
from typing import Dict, Optional, Tuple


# Custom function for bitwise AND (symbolic)
class BitAnd(Function):
    """Symbolic bitwise AND."""

    @classmethod
    def eval(cls, x, y):
        if isinstance(x, Integer) and isinstance(y, Integer):
            return Integer(int(x) & int(y))
        return None


class SymbolicAsmAnalyzer:
    """Analyzes assembly and builds SymPy expressions for registers."""

    def __init__(self):
        # Base symbols
        self.tid_x = Symbol("tid_x", integer=True, nonnegative=True)
        self.tid_y = Symbol("tid_y", integer=True, nonnegative=True)
        self.wgid_x = Symbol("wgid_x", integer=True, nonnegative=True)
        self.wgid_y = Symbol("wgid_y", integer=True, nonnegative=True)
        self.loop_idx = Symbol("loop_idx", integer=True, nonnegative=True)

        # Register state: maps register name -> SymPy expression
        self.vregs: Dict[int, sympy.Expr] = {}
        self.sregs: Dict[int, sympy.Expr] = {}

    def reset(self):
        """Reset register state."""
        self.vregs = {}
        self.sregs = {}

    def set_initial_state(self, v0_is_packed_tid=True, wgid_x_reg=2, wgid_y_reg=3):
        """Set initial register values based on kernel ABI.

        Args:
            v0_is_packed_tid: If True, v0 = (tid_y << 10) | tid_x
            wgid_x_reg: SGPR number containing workgroup_id_x (default 2 for ASM, 8 for LLVM)
            wgid_y_reg: SGPR number containing workgroup_id_y (default 3 for ASM, 9 for LLVM)
        """
        if v0_is_packed_tid:
            # v0 = (tid_y << 10) | tid_x on gfx9
            self.vregs[0] = self.tid_y * 1024 + self.tid_x
        if wgid_x_reg is not None:
            self.sregs[wgid_x_reg] = self.wgid_x
        if wgid_y_reg is not None:
            self.sregs[wgid_y_reg] = self.wgid_y

    def parse_register(self, reg_str: str) -> Tuple[Optional[str], Optional[int]]:
        """Parse register string like 'v0', 's2', 'v[4:5]' -> (type, number)."""
        reg_str = reg_str.strip().rstrip(",")
        if reg_str.startswith("v["):
            m = re.match(r"v\[(\d+):(\d+)\]", reg_str)
            if m:
                return ("v", int(m.group(1)))
        elif reg_str.startswith("s["):
            m = re.match(r"s\[(\d+):(\d+)\]", reg_str)
            if m:
                return ("s", int(m.group(1)))
        elif reg_str.startswith("v") and len(reg_str) > 1:
            try:
                return ("v", int(reg_str[1:]))
            except ValueError:
                return (None, None)
        elif reg_str.startswith("s") and len(reg_str) > 1:
            try:
                return ("s", int(reg_str[1:]))
            except ValueError:
                return (None, None)
        return (None, None)

    def get_reg_value(self, reg_str: str) -> Optional[sympy.Expr]:
        """Get symbolic value of a register."""
        rtype, rnum = self.parse_register(reg_str)
        if rtype == "v":
            return self.vregs.get(rnum, Symbol(f"v{rnum}"))
        elif rtype == "s":
            return self.sregs.get(rnum, Symbol(f"s{rnum}"))
        return None

    def set_reg_value(self, reg_str: str, value: sympy.Expr):
        """Set symbolic value of a register."""
        rtype, rnum = self.parse_register(reg_str)
        if rtype == "v" and rnum is not None:
            self.vregs[rnum] = value
        elif rtype == "s" and rnum is not None:
            self.sregs[rnum] = value

    def parse_operand(self, op: str) -> sympy.Expr:
        """Parse an operand (register or immediate) to SymPy expression."""
        op = op.strip().rstrip(",")

        # Handle hex immediates
        if op.startswith("0x") or op.startswith("-0x"):
            return Integer(int(op, 16))
        if op.startswith("0X") or op.startswith("-0X"):
            return Integer(int(op, 16))

        # Handle decimal immediates (including negative)
        try:
            return Integer(int(op))
        except ValueError:
            pass

        # Handle register
        rtype, rnum = self.parse_register(op)
        if rtype == "v":
            return self.vregs.get(rnum, Symbol(f"v{rnum}"))
        elif rtype == "s":
            return self.sregs.get(rnum, Symbol(f"s{rnum}"))

        # Unknown - return as symbol
        return Symbol(op.replace(".", "_").replace("[", "_").replace("]", "_"))

    def process_instruction(self, line: str) -> bool:
        """Process a single assembly instruction and update register state.
        Returns True if instruction was processed."""
        line = line.strip()

        # Skip comments and labels
        if not line or line.startswith("#") or line.startswith("//"):
            return False

        # Remove inline comments BEFORE checking for labels
        if "//" in line:
            line = line.split("//")[0].strip()
        if " #" in line:
            line = line.split(" #")[0].strip()

        if not line:
            return False

        # Check for labels (word followed by colon at start of line)
        if ":" in line:
            # Labels are "word:" at start, not instructions containing ':'
            first_word = line.split()[0] if line.split() else ""
            if first_word.endswith(":"):
                return False

        # Parse instruction
        parts = line.replace(",", " ").split()
        if not parts:
            return False

        op = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        # Skip directives
        if (
            op.startswith(".")
            or op.startswith("s_waitcnt")
            or op.startswith("s_barrier")
        ):
            return False
        if op in ("s_endpgm", "s_nop", "s_branch", "s_cbranch_scc0", "s_cbranch_scc1"):
            return False

        # Map instructions to operations
        if op in ("v_and_b32", "v_and_b32_e32"):
            if len(args) >= 3:
                dst, src0, src1 = args[0], args[1], args[2]
                val0 = self.parse_operand(src0)
                val1 = self.parse_operand(src1)
                if isinstance(val0, Integer):
                    mask = int(val0)
                    if mask == 0x3FF:
                        result = self.tid_x
                    elif (mask & (mask + 1)) == 0:  # Power of 2 minus 1
                        result = val1 % (mask + 1)
                    else:
                        result = BitAnd(val1, val0)
                else:
                    result = BitAnd(val0, val1)
                self.set_reg_value(dst, result)
                return True

        elif op == "v_bfe_u32":
            if len(args) >= 4:
                dst, src, offset, width = args[0], args[1], args[2], args[3]
                val = self.parse_operand(src)
                off = int(self.parse_operand(offset))
                w = int(self.parse_operand(width))
                if off == 10 and w == 10:
                    result = self.tid_y
                else:
                    result = floor(val / (2**off)) % (2**w)
                self.set_reg_value(dst, result)
                return True

        elif op in ("v_lshrrev_b32", "v_lshrrev_b32_e32"):
            if len(args) >= 3:
                dst, shift, src = args[0], args[1], args[2]
                shift_val = self.parse_operand(shift)
                src_val = self.parse_operand(src)
                if isinstance(shift_val, Integer):
                    result = floor(src_val / (2 ** int(shift_val)))
                else:
                    result = floor(src_val / (2**shift_val))
                self.set_reg_value(dst, result)
                return True

        elif op in ("v_lshlrev_b32", "v_lshlrev_b32_e32"):
            if len(args) >= 3:
                dst, shift, src = args[0], args[1], args[2]
                shift_val = self.parse_operand(shift)
                src_val = self.parse_operand(src)
                if isinstance(shift_val, Integer):
                    result = src_val * (2 ** int(shift_val))
                else:
                    result = src_val * (2**shift_val)
                self.set_reg_value(dst, result)
                return True

        elif op in ("v_add_u32", "v_add_u32_e32"):
            if len(args) >= 3:
                dst, src0, src1 = args[0], args[1], args[2]
                self.set_reg_value(
                    dst, self.parse_operand(src0) + self.parse_operand(src1)
                )
                return True

        elif op in ("v_sub_u32", "v_sub_u32_e32"):
            if len(args) >= 3:
                dst, src0, src1 = args[0], args[1], args[2]
                self.set_reg_value(
                    dst, self.parse_operand(src0) - self.parse_operand(src1)
                )
                return True

        elif op == "v_mul_lo_u32":
            if len(args) >= 3:
                dst, src0, src1 = args[0], args[1], args[2]
                self.set_reg_value(
                    dst, self.parse_operand(src0) * self.parse_operand(src1)
                )
                return True

        elif op in ("v_or_b32", "v_or_b32_e32"):
            if len(args) >= 3:
                dst, src0, src1 = args[0], args[1], args[2]
                # Assume non-overlapping for simplification
                self.set_reg_value(
                    dst, self.parse_operand(src0) + self.parse_operand(src1)
                )
                return True

        elif op == "v_lshl_or_b32":
            if len(args) >= 4:
                dst, src0, shift, src2 = args[0], args[1], args[2], args[3]
                val0 = self.parse_operand(src0)
                shift_val = self.parse_operand(shift)
                val2 = self.parse_operand(src2)
                if isinstance(shift_val, Integer):
                    result = val0 * (2 ** int(shift_val)) + val2
                else:
                    result = val0 * (2**shift_val) + val2
                self.set_reg_value(dst, result)
                return True

        elif op == "v_lshl_add_u32":
            if len(args) >= 4:
                dst, src0, shift, src2 = args[0], args[1], args[2], args[3]
                val0 = self.parse_operand(src0)
                shift_val = self.parse_operand(shift)
                val2 = self.parse_operand(src2)
                if isinstance(shift_val, Integer):
                    result = val0 * (2 ** int(shift_val)) + val2
                else:
                    result = val0 * (2**shift_val) + val2
                self.set_reg_value(dst, result)
                return True

        elif op == "v_or3_b32":
            if len(args) >= 4:
                dst = args[0]
                vals = [self.parse_operand(a) for a in args[1:4]]
                self.set_reg_value(dst, sum(vals))
                return True

        elif op in ("v_mov_b32", "v_mov_b32_e32"):
            if len(args) >= 2:
                dst, src = args[0], args[1]
                self.set_reg_value(dst, self.parse_operand(src))
                return True

        elif op == "v_readfirstlane_b32":
            if len(args) >= 2:
                dst, src = args[0], args[1]
                self.set_reg_value(dst, self.parse_operand(src))
                return True

        elif op == "v_mbcnt_lo_u32_b32":
            if len(args) >= 3:
                dst = args[0]
                self.set_reg_value(dst, Symbol("lane_id_lo"))
                return True

        elif op == "v_mbcnt_hi_u32_b32":
            if len(args) >= 3:
                dst = args[0]
                self.set_reg_value(dst, Symbol("lane_id"))
                return True

        elif op == "s_mov_b32":
            if len(args) >= 2:
                dst, src = args[0], args[1]
                self.set_reg_value(dst, self.parse_operand(src))
                return True

        elif op == "s_mov_b64":
            if len(args) >= 2:
                dst, src = args[0], args[1]
                self.set_reg_value(dst, self.parse_operand(src))
                return True

        elif op == "s_lshl_b32":
            if len(args) >= 3:
                dst, src, shift = args[0], args[1], args[2]
                val = self.parse_operand(src)
                shift_val = self.parse_operand(shift)
                if isinstance(shift_val, Integer):
                    self.set_reg_value(dst, val * (2 ** int(shift_val)))
                else:
                    self.set_reg_value(dst, val * (2**shift_val))
                return True

        elif op in ("s_add_i32", "s_add_u32"):
            if len(args) >= 3:
                dst, src0, src1 = args[0], args[1], args[2]
                self.set_reg_value(
                    dst, self.parse_operand(src0) + self.parse_operand(src1)
                )
                return True

        elif op == "s_and_b32":
            if len(args) >= 3:
                dst, src0, src1 = args[0], args[1], args[2]
                val0 = self.parse_operand(src0)
                val1 = self.parse_operand(src1)
                if isinstance(val0, Integer) and isinstance(val1, Integer):
                    self.set_reg_value(dst, Integer(int(val0) & int(val1)))
                else:
                    self.set_reg_value(dst, BitAnd(val0, val1))
                return True

        elif op == "s_or_b32":
            if len(args) >= 3:
                dst, src0, src1 = args[0], args[1], args[2]
                self.set_reg_value(
                    dst, self.parse_operand(src0) + self.parse_operand(src1)
                )
                return True

        elif op.startswith("s_load"):
            # Track loads from kernel args
            if len(args) >= 3:
                dst = args[0]
                # Keep as symbol since we don't know the value
                self.set_reg_value(
                    dst,
                    Symbol(dst.replace("[", "_").replace("]", "_").replace(":", "_")),
                )
                return True

        return False


def analyze_instruction(asm_file: str, target: str):
    """
    Analyze an instruction in an assembly file.

    Args:
        asm_file: Path to assembly file
        target: Line number (int) or regex pattern to match
    """
    # Read the file
    with open(asm_file, "r") as f:
        lines = f.readlines()

    # Find target line(s)
    target_lines = []
    try:
        line_num = int(target)
        if 1 <= line_num <= len(lines):
            target_lines = [(line_num, lines[line_num - 1].strip())]
    except ValueError:
        # Treat as regex pattern
        pattern = re.compile(target, re.IGNORECASE)
        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                target_lines.append((i, line.strip()))

    if not target_lines:
        print(f"Error: No lines matching '{target}' found in {asm_file}")
        return

    print("=" * 80)
    print(f"Symbolic Analysis: {asm_file}")
    print("=" * 80)

    for line_num, target_line in target_lines:
        print(f"\n{'─' * 80}")
        print(f"Line {line_num}: {target_line}")
        print(f"{'─' * 80}")

        # Create analyzer and set initial state based on file type
        analyzer = SymbolicAsmAnalyzer()

        # Detect SGPR layout: LLVM uses s8/s9 for wgid, ASM backend uses s2/s3
        is_llvm = "hsaco" in asm_file.lower() or any("s_nop" in l for l in lines[:70])
        if is_llvm:
            analyzer.set_initial_state(wgid_x_reg=8, wgid_y_reg=9)
        else:
            analyzer.set_initial_state(wgid_x_reg=2, wgid_y_reg=3)

        # Process all instructions up to AND INCLUDING target line
        for i, line in enumerate(lines[:line_num], 1):
            analyzer.process_instruction(line)

        # Parse the target instruction to get operands
        clean_line = target_line
        if "//" in clean_line:
            clean_line = clean_line.split("//")[0].strip()
        if " #" in clean_line:
            clean_line = clean_line.split(" #")[0].strip()

        parts = clean_line.replace(",", " ").split()
        if not parts:
            continue

        instr = parts[0]
        operands = [p.rstrip(",") for p in parts[1:]]

        print(f"\nInstruction: {instr}")
        print(f"Operands: {operands}")
        print(f"\nSymbolic values:")
        print("-" * 60)

        for op in operands:
            # Skip non-register operands
            if op.startswith("0x") or op.startswith("-"):
                print(f"  {op:<12} = {op} (immediate)")
                continue
            try:
                int(op)
                print(f"  {op:<12} = {op} (immediate)")
                continue
            except ValueError:
                pass

            # Skip special operands
            if op in ("offen", "lds", "offset:0", "off") or op.startswith("offset:"):
                print(f"  {op:<12} = {op} (modifier)")
                continue

            # Get symbolic value
            val = analyzer.get_reg_value(op)
            if val is not None:
                simplified = simplify(val)
                print(f"  {op:<12} = {simplified}")
            else:
                print(f"  {op:<12} = <unknown>")

        # Special handling for common instruction types
        if "buffer_load" in instr.lower() or "buffer_store" in instr.lower():
            print(f"\n{'─' * 40}")
            print("Buffer operation analysis:")
            print("─" * 40)

            # First operand is usually vdata/vaddr for loads
            if len(operands) >= 2:
                vaddr = operands[0]
                srd = operands[1]

                vaddr_val = analyzer.get_reg_value(vaddr)
                if vaddr_val:
                    print(f"\nvaddr ({vaddr}) formula:")
                    print(f"  {simplify(vaddr_val)}")

                    # Evaluate for sample thread IDs
                    print(f"\nSample evaluations:")
                    samples = [
                        (0, 0, "Wave(0,0) lane 0"),
                        (1, 0, "Wave(0,0) lane 1"),
                        (64, 0, "Wave(1,0) lane 0"),
                        (0, 1, "Wave(0,1) lane 0"),
                        (64, 1, "Wave(1,1) lane 0"),
                    ]

                    for tid_x, tid_y, desc in samples:
                        try:
                            subs = {
                                analyzer.tid_x: tid_x,
                                analyzer.tid_y: tid_y,
                                analyzer.wgid_x: 0,
                                analyzer.wgid_y: 0,
                            }
                            # Try to substitute known symbols
                            result = vaddr_val.subs(subs)
                            # Try to evaluate if all symbols are resolved
                            try:
                                result = int(result)
                                print(
                                    f"  {desc}: tid_x={tid_x}, tid_y={tid_y} -> vaddr={result}"
                                )
                            except:
                                print(
                                    f"  {desc}: tid_x={tid_x}, tid_y={tid_y} -> vaddr={result}"
                                )
                        except Exception as e:
                            print(f"  {desc}: Error - {e}")

        elif "ds_read" in instr.lower() or "ds_write" in instr.lower():
            print(f"\n{'─' * 40}")
            print("LDS operation analysis:")
            print("─" * 40)

            # Find addr operand (second operand for ds_read, after vdst)
            if len(operands) >= 2:
                addr = operands[1]  # Address is always second operand (after vdst)
                if addr.startswith("v"):
                    addr_val = analyzer.get_reg_value(addr)
                    if addr_val:
                        print(f"\naddr ({addr}) formula:")
                        print(f"  {simplify(addr_val)}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nExample usage:")
        print(f"  python {sys.argv[0]} wave_asm.s 124")
        print(f"  python {sys.argv[0]} wave_asm.s 'buffer_load_dword.*lds'")
        print(f"  python {sys.argv[0]} llvm_asm.s 'buffer_load_dword'")
        sys.exit(1)

    asm_file = sys.argv[1]
    target = sys.argv[2]

    analyze_instruction(asm_file, target)


if __name__ == "__main__":
    main()
