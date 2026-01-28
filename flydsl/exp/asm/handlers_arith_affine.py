# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import operator
import struct

import sympy
from .ir_imports import affine_d, arith_d, gpu_d

from .handlers_shared import *
from .instruction_registry import Instruction
from .kernel_ir import KSReg, KInstr, KImm
from .kernel_model import KernelInfo
from .utils import simplify_expression, tid_upper_bound_from_thread_id


class _ArithAffineHandlers:
    def _get_lane_regs(self, ssa_name: str):
        regs = self.walker.kernel_ctx.ssa_to_reg.get(ssa_name)
        return tuple(regs) if regs is not None else None

    def _get_scalar_shift_imm(self, value_ssa: str, kernel_info: KernelInfo) -> int | None:
        # Prefer tracked constants.
        v = kernel_info.index_env.get(value_ssa)
        if isinstance(v, int):
            return int(v)
        # Or a scalar reg that was materialized from constant.
        regs = self.walker.kernel_ctx.ssa_to_reg.get(value_ssa)
        if regs and len(regs) == 1:
            # Can't recover immediate; require constant in index_env.
            return None
        return None

    def handle_arith_constant_op(
        self, operation: arith_d.ConstantOp, kernel_info: KernelInfo
    ):
        """Handle arith.constant.

        - Track integer-like constants in `index_env` (for address algebra).
        - Materialize f32 constants into VGPRs (for arithmetic/select).
        """
        import re

        ctx = self.walker.kernel_ctx
        res_ssa = ssa(operation.result)
        val_str = str(operation.value).strip()

        # Vector dense integer constants (e.g., dense<16> : vector<8xi32>).
        if "dense<" in val_str and ": vector<" in val_str and ("xi" in val_str):
            import re

            m = re.search(r"dense<\s*([-]?\d+)\s*>", val_str)
            n = re.search(r":\s*vector<\s*(\d+)\s*x", val_str)
            if m and n:
                v = int(m.group(1))
                lanes = int(n.group(1))
                regs = []
                for _ in range(lanes):
                    r = ctx.vreg()
                    ctx.program.emit(
                        KInstr(
                            Instruction.V_MOV_B32,
                            defs=(r,),
                            uses=(KImm(int(v)),),
                            comment=f"const vec {v}",
                        )
                    )
                    regs.append(r)
                ctx.ssa_to_reg[res_ssa] = tuple(regs)
                return

        # Vector dense float splats (e.g., dense<1.0> : vector<8xf32>).
        if "dense<" in val_str and ": vector<" in val_str and ("xf32" in val_str):
            m = re.search(r"dense<\s*([^>]+)\s*>", val_str)
            n = re.search(r":\s*vector<\s*(\d+)\s*x", val_str)
            if m and n:
                raw = m.group(1).strip()
                lanes = int(n.group(1))
                if raw.startswith(("0x", "0X")):
                    bits = int(raw, 16) & 0xFFFFFFFF
                else:
                    bits = struct.unpack("<I", struct.pack("<f", float(raw)))[0]
                regs = []
                for _ in range(lanes):
                    r = ctx.vreg()
                    ctx.program.emit(
                        KInstr(
                            Instruction.V_MOV_B32,
                            defs=(r,),
                            uses=(KImm(int(bits)),),
                            comment=f"const vec f32 {raw}",
                        )
                    )
                    regs.append(r)
                ctx.ssa_to_reg[res_ssa] = tuple(regs)
                return

        # Float constants: f32/f16/bf16.
        if ": f32" in val_str or ": f16" in val_str or ": bf16" in val_str:
            if ": f32" in val_str:
                raw = val_str.split(": f32", 1)[0].strip()
                kind = "f32"
            elif ": f16" in val_str:
                raw = val_str.split(": f16", 1)[0].strip()
                kind = "f16"
            else:
                raw = val_str.split(": bf16", 1)[0].strip()
                kind = "bf16"

            if raw.startswith(("0x", "0X")):
                if kind == "f32":
                    bits = int(raw, 16) & 0xFFFFFFFF
                elif kind == "bf16":
                    bits16 = int(raw, 16) & 0xFFFF
                    bits = (bits16 << 16) & 0xFFFFFFFF
                else:  # f16
                    bits16 = int(raw, 16) & 0xFFFF
                    try:
                        import numpy as np  # type: ignore

                        f = np.frombuffer(np.uint16(bits16).tobytes(), dtype=np.float16)[0].astype(np.float32)
                        bits = struct.unpack("<I", struct.pack("<f", float(f)))[0]
                    except Exception:
                        # Fallback: approximate via float() parsing (may be wrong for hex).
                        bits = struct.unpack("<I", struct.pack("<f", float("nan")))[0]
            else:
                # Decimal literal.
                bits = struct.unpack("<I", struct.pack("<f", float(raw)))[0]
            r = ctx.vreg()
            ctx.program.emit(
                KInstr(
                    Instruction.V_MOV_B32,
                    defs=(r,),
                    uses=(KImm(int(bits)),),
                    comment=f"const {kind} {raw}",
                )
            )
            ctx.ssa_to_reg[res_ssa] = (r,)
            return

        # Integer constants: leading int.
        m = re.match(r"^-?\d+", val_str)
        if m:
            kernel_info.index_env[res_ssa] = int(m.group(0))
            # Also materialize as a VGPR for runtime index math when needed.
            r = ctx.vreg()
            ctx.program.emit(
                KInstr(
                    Instruction.V_MOV_B32,
                    defs=(r,),
                    uses=(KImm(int(m.group(0))),),
                    comment=f"const {m.group(0)}",
                )
            )
            ctx.ssa_to_reg[res_ssa] = (r,)
            return

    def handle_arith_mulf_op(self, operation: arith_d.MulFOp, kernel_info: KernelInfo):
        """Lower arith.mulf for scalar/vector f32 (opview or generic op)."""
        ctx = self.walker.kernel_ctx
        try:
            lhs_ssa = ssa(operation.lhs)  # type: ignore[attr-defined]
            rhs_ssa = ssa(operation.rhs)  # type: ignore[attr-defined]
            res_ssa = ssa(operation.result)  # type: ignore[attr-defined]
        except Exception:
            lhs_ssa = ssa(operation.operands[0])
            rhs_ssa = ssa(operation.operands[1])
            res_ssa = ssa(operation.results[0])

        lhs = ctx.ssa_to_reg.get(lhs_ssa)
        rhs = ctx.ssa_to_reg.get(rhs_ssa)
        if not lhs or not rhs or len(lhs) != len(rhs):
            return
        out = tuple(ctx.v_mul_f32(a, b, comment="mulf") for a, b in zip(lhs, rhs))
        ctx.ssa_to_reg[res_ssa] = out

    def handle_arith_subf_op(self, operation: arith_d.SubFOp, kernel_info: KernelInfo):
        ctx = self.walker.kernel_ctx
        lhs = ctx.ssa_to_reg.get(ssa(operation.lhs))
        rhs = ctx.ssa_to_reg.get(ssa(operation.rhs))
        if not lhs or not rhs or len(lhs) != len(rhs):
            return
        out = tuple(ctx.v_sub_f32(a, b, comment="subf") for a, b in zip(lhs, rhs))
        ctx.ssa_to_reg[ssa(operation.result)] = out

    def handle_arith_maximumf_op(self, operation: arith_d.MaximumFOp, kernel_info: KernelInfo):
        ctx = self.walker.kernel_ctx
        lhs = ctx.ssa_to_reg.get(ssa(operation.lhs))
        rhs = ctx.ssa_to_reg.get(ssa(operation.rhs))
        if not lhs or not rhs or len(lhs) != len(rhs):
            return
        out = tuple(ctx.v_max_f32(a, b, comment="maxf") for a, b in zip(lhs, rhs))
        ctx.ssa_to_reg[ssa(operation.result)] = out

    def handle_math_exp2_op(self, operation, kernel_info: KernelInfo):
        """Lower math.exp2 for scalar or vector f32 values."""
        ctx = self.walker.kernel_ctx
        src = ctx.ssa_to_reg.get(ssa(operation.operands[0]))
        if not src:
            return

        outs = []
        for x in src:
            n_i = ctx.vreg()
            ctx.program.emit(KInstr("v_cvt_i32_f32", defs=(n_i,), uses=(x,), comment="exp2 n=int(x)"))
            n_f = ctx.vreg()
            ctx.program.emit(KInstr("v_cvt_f32_i32", defs=(n_f,), uses=(n_i,), comment="exp2 float(n)"))
            frac = ctx.v_sub_f32(x, n_f, comment="exp2 frac")
            r = ctx.v_exp_f32(frac, comment="exp2 2^frac")
            out = ctx.vreg()
            ctx.program.emit(KInstr("v_ldexp_f32", defs=(out,), uses=(r, n_i), comment="exp2 scale"))
            outs.append(out)

        # math.exp2 has one result.
        ctx.ssa_to_reg[ssa(operation.results[0])] = tuple(outs)

    def _handle_arith_binop(self, operation, kernel_info: KernelInfo, op_func):
        """Handle binary arithmetic operations (addi, muli) in index_env.

        Args:
            operation: The MLIR operation (AddIOp or MulIOp)
            kernel_info: Kernel info containing index_env
            op_func: Binary function to apply (e.g., operator.add, operator.mul)
        """
        lhs = kernel_info.index_env.get(ssa(operation.operands[0]))
        rhs = kernel_info.index_env.get(ssa(operation.operands[1]))

        # Operands not tracked - can't compute result
        if lhs is None or rhs is None:
            return

        # Convert symbolic strings (tid_x, wgid_x, etc.) and loop counters (KSReg)
        # to SymPy symbols.
        if isinstance(lhs, str):
            lhs = sympy.Symbol(lhs)
        if isinstance(lhs, KSReg):
            lhs = sympy.Symbol(f"ks{lhs.id}", nonnegative=True)
        if isinstance(rhs, str):
            rhs = sympy.Symbol(rhs)
        if isinstance(rhs, KSReg):
            rhs = sympy.Symbol(f"ks{rhs.id}", nonnegative=True)

        if isinstance(lhs, (int, sympy.Expr)) and isinstance(rhs, (int, sympy.Expr)):
            kernel_info.index_env[ssa(operation.result)] = op_func(lhs, rhs)

    def handle_arith_addi_op(self, operation: arith_d.AddIOp, kernel_info: KernelInfo):
        """Handle arith.addi - track integer addition in index_env."""
        self._handle_arith_binop(operation, kernel_info, operator.add)
        # Runtime (VGPR) add for index/i32 values.
        ctx = self.walker.kernel_ctx
        lhs = ctx.ssa_to_reg.get(ssa(operation.operands[0]))
        rhs = ctx.ssa_to_reg.get(ssa(operation.operands[1]))
        if lhs and rhs:
            if len(lhs) == 1 and len(rhs) == 1:
                out = ctx.v_add_u32(lhs[0], rhs[0], comment="addi")
                ctx.ssa_to_reg[ssa(operation.result)] = (out,)
            elif len(lhs) == len(rhs) and len(lhs) > 1:
                out = tuple(ctx.v_add_u32(a, b, comment="addi") for a, b in zip(lhs, rhs))
                ctx.ssa_to_reg[ssa(operation.result)] = out

    def handle_arith_muli_op(self, operation: arith_d.MulIOp, kernel_info: KernelInfo):
        """Handle arith.muli - track integer multiplication in index_env."""
        self._handle_arith_binop(operation, kernel_info, operator.mul)
        ctx = self.walker.kernel_ctx
        lhs = ctx.ssa_to_reg.get(ssa(operation.operands[0]))
        rhs = ctx.ssa_to_reg.get(ssa(operation.operands[1]))
        if lhs and rhs and len(lhs) == 1 and len(rhs) == 1:
            out = ctx.v_mul_lo_u32(lhs[0], rhs[0], comment="muli")
            ctx.ssa_to_reg[ssa(operation.result)] = (out,)

    def handle_arith_divui_op(self, operation: arith_d.DivUIOp, kernel_info: KernelInfo):
        """Handle arith.divui for simple power-of-two divisors (VGPR runtime)."""
        ctx = self.walker.kernel_ctx
        lhs = ctx.ssa_to_reg.get(ssa(operation.lhs))
        rhs_ssa = ssa(operation.rhs)
        rhs_const = kernel_info.index_env.get(rhs_ssa)
        if lhs is None or len(lhs) != 1:
            return
        if isinstance(rhs_const, int) and rhs_const > 0 and (rhs_const & (rhs_const - 1)) == 0:
            shift = int(rhs_const.bit_length() - 1)
            out = ctx.v_lshrrev_b32(KImm(shift), lhs[0], comment=f"divui /{rhs_const}")
            ctx.ssa_to_reg[ssa(operation.result)] = (out,)

    def handle_arith_divsi_op(self, operation: arith_d.DivSIOp, kernel_info: KernelInfo):
        """Handle arith.divsi for simple power-of-two divisors (VGPR runtime).

        NOTE: For our current kernels (thread/block indices), values are non-negative,
        so an unsigned shift is sufficient.
        """
        ctx = self.walker.kernel_ctx
        lhs = ctx.ssa_to_reg.get(ssa(operation.lhs))
        rhs_ssa = ssa(operation.rhs)
        rhs_const = kernel_info.index_env.get(rhs_ssa)
        if lhs is None or len(lhs) != 1:
            return
        if isinstance(rhs_const, int) and rhs_const > 0 and (rhs_const & (rhs_const - 1)) == 0:
            shift = int(rhs_const.bit_length() - 1)
            out = ctx.v_lshrrev_b32(KImm(shift), lhs[0], comment=f"divsi /{rhs_const}")
            ctx.ssa_to_reg[ssa(operation.result)] = (out,)

    def handle_arith_remsi_op(self, operation: arith_d.RemSIOp, kernel_info: KernelInfo):
        """Handle arith.remsi for simple power-of-two divisors (VGPR runtime)."""
        # Reuse the same mask trick as remui; for non-negative values this matches.
        ctx = self.walker.kernel_ctx
        lhs_r = ctx.ssa_to_reg.get(ssa(operation.operands[0]))
        rhs_ssa = ssa(operation.operands[1])
        rhs_c = kernel_info.index_env.get(rhs_ssa)
        if lhs_r and len(lhs_r) == 1 and isinstance(rhs_c, int) and rhs_c > 0 and (rhs_c & (rhs_c - 1)) == 0:
            mask = rhs_c - 1
            mask_r = ctx.vreg()
            ctx.program.emit(
                KInstr(
                    Instruction.V_MOV_B32,
                    defs=(mask_r,),
                    uses=(KImm(int(mask)),),
                    comment=f"remsi mask {mask:#x}",
                )
            )
            out = ctx.v_and_b32(lhs_r[0], mask_r, comment=f"remsi %{rhs_c}")
            ctx.ssa_to_reg[ssa(operation.result)] = (out,)

    def handle_arith_xori_op(self, operation: arith_d.XOrIOp, kernel_info: KernelInfo):
        """Handle arith.xori (VGPR runtime)."""
        ctx = self.walker.kernel_ctx
        lhs = ctx.ssa_to_reg.get(ssa(operation.lhs))
        rhs = ctx.ssa_to_reg.get(ssa(operation.rhs))
        if lhs and rhs and len(lhs) == 1 and len(rhs) == 1:
            out = ctx.v_xor_b32(lhs[0], rhs[0], comment="xori")
            ctx.ssa_to_reg[ssa(operation.result)] = (out,)

    def handle_arith_remui_op(
        self, operation: arith_d.RemUIOp, kernel_info: KernelInfo
    ):
        """Handle arith.remui for index arithmetic."""
        lhs = kernel_info.index_env.get(ssa(operation.operands[0]))
        rhs = kernel_info.index_env.get(ssa(operation.operands[1]))
        if lhs is None or rhs is None:
            return

        if isinstance(lhs, str):
            lhs = sympy.Symbol(lhs, nonnegative=True)
        if isinstance(lhs, KSReg):
            lhs = sympy.Symbol(f"ks{lhs.id}", nonnegative=True)
        if isinstance(rhs, str):
            rhs = sympy.Symbol(rhs, nonnegative=True)
        if isinstance(rhs, KSReg):
            rhs = sympy.Symbol(f"ks{rhs.id}", nonnegative=True)

        if isinstance(rhs, int):
            kernel_info.index_env[ssa(operation.result)] = sympy.Mod(
                lhs, sympy.Integer(rhs)
            )
        else:
            kernel_info.index_env[ssa(operation.result)] = sympy.Mod(lhs, rhs)

        # Runtime (VGPR) rem for power-of-two divisors: x & (m-1).
        ctx = self.walker.kernel_ctx
        lhs_r = ctx.ssa_to_reg.get(ssa(operation.operands[0]))
        rhs_ssa = ssa(operation.operands[1])
        rhs_c = kernel_info.index_env.get(rhs_ssa)
        if lhs_r and len(lhs_r) == 1 and isinstance(rhs_c, int) and rhs_c > 0 and (rhs_c & (rhs_c - 1)) == 0:
            mask = rhs_c - 1
            # Some assemblers reject literal operands for v_and_b32; materialize mask.
            mask_r = ctx.vreg()
            ctx.program.emit(
                KInstr(
                    Instruction.V_MOV_B32,
                    defs=(mask_r,),
                    uses=(KImm(int(mask)),),
                    comment=f"remui mask {mask:#x}",
                )
            )
            out = ctx.v_and_b32(lhs_r[0], mask_r, comment=f"remui %{rhs_c}")
            ctx.ssa_to_reg[ssa(operation.result)] = (out,)

    def handle_arith_index_cast_op(
        self, operation: arith_d.IndexCastOp, kernel_info: KernelInfo
    ):
        """Handle arith.index_cast operations - propagate values through cast.

        Propagates integers, SymPy expressions, and symbolic strings (tid_x, etc.).
        """
        result_ssa = ssa(operation.result)
        src_ssa = ssa(operation.operands[0])

        # Runtime: treat index/i32 casts as no-op on the underlying registers.
        ctx = self.walker.kernel_ctx
        src_r = ctx.ssa_to_reg.get(src_ssa)
        if src_r and len(src_r) in (1, 2):
            ctx.ssa_to_reg[result_ssa] = src_r

        src_val = kernel_info.index_env.get(src_ssa)
        if src_val is None:
            return

        # Propagate numeric values and symbolic strings
        if isinstance(src_val, (int, sympy.Expr, str)):
            kernel_info.index_env[result_ssa] = src_val

    def handle_gpu_thread_id_op(
        self, operation: gpu_d.ThreadIdOp, kernel_info: KernelInfo
    ):
        """Handle gpu.thread_id operations - extract thread ID information."""
        upper_bound = tid_upper_bound_from_thread_id(operation)
        # Get the actual dimension from the operation
        dimension = operation.dimension
        # Extract dimension from MLIR attribute string like "#gpu<dim x>"
        dimension_string = str(dimension)
        if "dim x" in dimension_string:
            kernel_info.index_env[ssa(operation.result)] = "tid_x"
            if upper_bound is not None:
                kernel_info.tid_ub_x = upper_bound
            self.walker.kernel_ctx.ssa_to_reg[ssa(operation.result)] = (
                self.walker.kernel_ctx.ensure_tid_x(),
            )
        elif "dim y" in dimension_string:
            kernel_info.index_env[ssa(operation.result)] = "tid_y"
            if upper_bound is not None:
                kernel_info.tid_ub_y = upper_bound
            self.walker.kernel_ctx.ssa_to_reg[ssa(operation.result)] = (
                self.walker.kernel_ctx.ensure_tid_y(),
            )
        elif "dim z" in dimension_string:
            if upper_bound is not None:
                kernel_info.tid_ub_z = upper_bound
            kernel_info.index_env[ssa(operation.result)] = "tid_z"

    def handle_gpu_block_id_op(
        self, operation: gpu_d.BlockIdOp, kernel_info: KernelInfo
    ):
        """
        Handle gpu.block_id operations - map to workgroup ID symbols.

        Maps block IDs to symbolic names that the expression emitter can use:
        - block_id x -> wgid_x
        - block_id y -> wgid_y
        - block_id z -> wgid_z
        """
        dimension = operation.dimension
        dimension_string = str(dimension)

        if "dim x" in dimension_string:
            kernel_info.index_env[ssa(operation.result)] = "wgid_x"
            self.walker.kernel_ctx.ssa_to_reg[ssa(operation.result)] = (
                self.walker.kernel_ctx.ensure_wgid_x(),
            )
        elif "dim y" in dimension_string:
            kernel_info.index_env[ssa(operation.result)] = "wgid_y"
            self.walker.kernel_ctx.ssa_to_reg[ssa(operation.result)] = (
                self.walker.kernel_ctx.ensure_wgid_y(),
            )
        elif "dim z" in dimension_string:
            kernel_info.index_env[ssa(operation.result)] = "wgid_z"
            self.walker.kernel_ctx.ssa_to_reg[ssa(operation.result)] = (
                self.walker.kernel_ctx.ensure_wgid_z(),
            )

    def handle_gpu_block_dim_op(
        self, operation: gpu_d.BlockDimOp, kernel_info: KernelInfo
    ):
        """Handle gpu.block_dim by materializing workgroup size constants."""
        dimension_string = str(operation.dimension)
        wg_size = getattr(kernel_info, "wg_size", None) or (1, 1, 1)
        if "dim x" in dimension_string:
            kernel_info.index_env[ssa(operation.result)] = int(wg_size[0])
        elif "dim y" in dimension_string:
            kernel_info.index_env[ssa(operation.result)] = int(wg_size[1])
        elif "dim z" in dimension_string:
            kernel_info.index_env[ssa(operation.result)] = int(wg_size[2])

    def handle_arith_addf_op(self, operation: arith_d.AddFOp, kernel_info: KernelInfo):
        """Handle arith.addf for register-backed values (scalar or lanes)."""
        ctx = self.walker.kernel_ctx
        lhs_ssa = ssa(operation.operands[0])
        rhs_ssa = ssa(operation.operands[1])
        lhs_regs = ctx.ssa_to_reg.get(lhs_ssa)
        rhs_regs = ctx.ssa_to_reg.get(rhs_ssa)

        if lhs_regs is None or rhs_regs is None:
            return
        if len(lhs_regs) != len(rhs_regs):
            raise ValueError(
                f"arith.addf operand register arity mismatch: lhs={len(lhs_regs)} rhs={len(rhs_regs)} "
                f"(lhs_ssa={lhs_ssa}, rhs_ssa={rhs_ssa})"
            )

        out_regs = []
        for a, b in zip(lhs_regs, rhs_regs):
            out_regs.append(ctx.v_add_f32(a, b, comment="fadd"))
        ctx.ssa_to_reg[ssa(operation.result)] = tuple(out_regs)

    # -------------------------------------------------------------------------
    # Scalar float ops (softmax)
    # -------------------------------------------------------------------------

    def handle_arith_subf_op(self, operation: arith_d.SubFOp, kernel_info: KernelInfo):
        ctx = self.walker.kernel_ctx
        lhs = ctx.ssa_to_reg.get(ssa(operation.lhs))
        rhs = ctx.ssa_to_reg.get(ssa(operation.rhs))
        if not lhs or not rhs or len(lhs) != len(rhs):
            return
        out = tuple(ctx.v_sub_f32(a, b, comment="subf") for a, b in zip(lhs, rhs))
        ctx.ssa_to_reg[ssa(operation.result)] = out

    def handle_arith_mulf_op(self, operation: arith_d.MulFOp, kernel_info: KernelInfo):
        ctx = self.walker.kernel_ctx
        lhs = ctx.ssa_to_reg.get(ssa(operation.lhs))
        rhs = ctx.ssa_to_reg.get(ssa(operation.rhs))
        if not lhs or not rhs or len(lhs) != len(rhs):
            return
        out = tuple(ctx.v_mul_f32(a, b, comment="mulf") for a, b in zip(lhs, rhs))
        ctx.ssa_to_reg[ssa(operation.result)] = out

    def handle_arith_maximumf_op(self, operation: arith_d.MaximumFOp, kernel_info: KernelInfo):
        ctx = self.walker.kernel_ctx
        lhs = ctx.ssa_to_reg.get(ssa(operation.lhs))
        rhs = ctx.ssa_to_reg.get(ssa(operation.rhs))
        if not lhs or not rhs or len(lhs) != len(rhs):
            return
        out = tuple(ctx.v_max_f32(a, b, comment="maxf") for a, b in zip(lhs, rhs))
        ctx.ssa_to_reg[ssa(operation.result)] = out

    def handle_arith_divf_op(self, operation: arith_d.DivFOp, kernel_info: KernelInfo):
        ctx = self.walker.kernel_ctx
        lhs = ctx.ssa_to_reg.get(ssa(operation.lhs))
        rhs = ctx.ssa_to_reg.get(ssa(operation.rhs))
        if not lhs or not rhs or len(lhs) != 1 or len(rhs) != 1:
            return
        inv = ctx.v_rcp_f32(rhs[0], comment="rcp")
        out = ctx.v_mul_f32(lhs[0], inv, comment="divf via rcp")
        ctx.ssa_to_reg[ssa(operation.result)] = (out,)

        # Debug: capture divf inputs/outputs (thread0 only) into output buffer.
        import os
        if os.environ.get("FLIR_ASM_DEBUG_SOFTMAX", "0") == "1":
            out_ssa = "%arg1"
            if out_ssa in kernel_info.subspans:
                from .kernel_ir import (
                    KInstr,
                    KImm,
                    KRegRange,
                    KPhysVReg,
                    EXEC,
                    VCC,
                    KInstrConstraints,
                )

                # Ensure output SRD exists.
                self.walker.handlers._ensure_global_store_srd(kernel_info, out_ssa)  # type: ignore[attr-defined]

                saved_exec = ctx.program.alloc_sreg_range(2, alignment=2)
                ctx.program.emit(KInstr("s_mov_b64", defs=(saved_exec,), uses=(EXEC,), comment="dbgD save exec"))
                vcc_clobber = KInstrConstraints(vcc_clobber=True)
                ctx.program.emit(
                    KInstr(
                        "v_cmp_eq_u32",
                        defs=(VCC,),
                        uses=(KPhysVReg(0), KImm(0)),
                        constraints=vcc_clobber,
                        comment="dbgD v0==0",
                    )
                )
                ctx.program.emit(KInstr("s_nop", defs=(), uses=(KImm(1),), comment="dbgD vcc hazard"))
                ctx.program.emit(KInstr("s_and_b64", defs=(EXEC,), uses=(EXEC, VCC), comment="dbgD exec &= vcc"))

                # Layout at byte offset 0x5000: [rhs, inv, lhs, out]
                base = ctx.vreg()
                ctx.program.emit(KInstr("v_mov_b32", defs=(base,), uses=(KImm(0x5000),), comment="dbgD base"))
                ctx.emit_buffer_store(out_ssa, (KRegRange(rhs[0], 1),), base, 0)
                b4 = ctx.v_add_u32(base, KImm(4), comment="dbgD +4")
                ctx.emit_buffer_store(out_ssa, (KRegRange(inv, 1),), b4, 0)
                b8 = ctx.v_add_u32(base, KImm(8), comment="dbgD +8")
                ctx.emit_buffer_store(out_ssa, (KRegRange(lhs[0], 1),), b8, 0)
                b12 = ctx.v_add_u32(base, KImm(12), comment="dbgD +12")
                ctx.emit_buffer_store(out_ssa, (KRegRange(out, 1),), b12, 0)
                ctx.program.emit(KInstr("s_mov_b64", defs=(EXEC,), uses=(saved_exec,), comment="dbgD restore exec"))

    def handle_arith_extf_op(self, operation: arith_d.ExtFOp, kernel_info: KernelInfo):
        """Widen f16/bf16 to f32."""
        ctx = self.walker.kernel_ctx
        src_ssa = ssa(operation.in_)
        src = ctx.ssa_to_reg.get(src_ssa)
        if not src or len(src) != 1:
            return
        src_r = src[0]
        src_ty = str(operation.in_.type)
        if src_ty == "f16":
            out = ctx.v_cvt_f32_f16(src_r, comment="extf f16->f32")
        elif src_ty == "bf16":
            # bf16 bits are in low 16 of src_r; convert by shifting to high bits.
            bits = ctx.v_lshlrev_b32(KImm(16), src_r, comment="bf16<<16")
            out = bits  # f32 bits in VGPR
        else:
            # Already f32 (or something unexpected).
            out = src_r
        ctx.ssa_to_reg[ssa(operation.out)] = (out,)

    def handle_arith_truncf_op(self, operation: arith_d.TruncFOp, kernel_info: KernelInfo):
        """Narrow f32 to bf16/f16.

        - Scalar: keep value in f32 regs; packing may be deferred to stores.
        - Vector f16: convert each lane to f16 bits and pack two lanes per dword,
          producing 4 VGPRs for vector<8xf16>.
        """
        ctx = self.walker.kernel_ctx
        # Support both opview and generic Operation shapes.
        try:
            src_ssa = ssa(operation.in_)  # type: ignore[attr-defined]
            dst_ssa = ssa(operation.out)  # type: ignore[attr-defined]
            dst_ty = str(operation.out.type)  # type: ignore[attr-defined]
        except Exception:
            src_ssa = ssa(operation.operands[0])
            dst_ssa = ssa(operation.results[0])
            dst_ty = str(operation.results[0].type)

        src = ctx.ssa_to_reg.get(src_ssa)
        if not src:
            return

        # Vector f16 path: pack into dwords (2x16-bit per VGPR).
        if dst_ty.startswith("vector<") and "xf16" in dst_ty:
            # Expect f32 lanes.
            lanes = list(src)
            if len(lanes) % 2 != 0:
                return
            from .kernel_ir import KInstr, KImm

            # Materialize 0xFFFF mask once.
            mask = ctx.vreg()
            ctx.program.emit(
                KInstr(
                    Instruction.V_MOV_B32,
                    defs=(mask,),
                    uses=(KImm(0xFFFF),),
                    comment="truncf f16 mask",
                )
            )
            packed = []
            for i in range(0, len(lanes), 2):
                lo16 = ctx.vreg()
                hi16 = ctx.vreg()
                ctx.program.emit(KInstr("v_cvt_f16_f32", defs=(lo16,), uses=(lanes[i],), comment="f32->f16 lo"))
                ctx.program.emit(KInstr("v_cvt_f16_f32", defs=(hi16,), uses=(lanes[i + 1],), comment="f32->f16 hi"))
                lo = ctx.v_and_b32(lo16, mask, comment="pack lo16")
                hi = ctx.v_and_b32(hi16, mask, comment="pack hi16")
                hi_sh = ctx.v_lshlrev_b32(KImm(16), hi, comment="hi<<16")
                dw = ctx.v_or_b32(lo, hi_sh, comment="pack f16x2")
                packed.append(dw)
            ctx.ssa_to_reg[ssa(operation.out)] = tuple(packed)
            return

        # Scalar f16/bf16: convert now (needed for bitcasts/packing).
        if len(src) == 1 and (dst_ty == "f16" or dst_ty.endswith(" f16")):
            from .kernel_ir import KInstr
            out = ctx.vreg()
            ctx.program.emit(
                KInstr(
                    "v_cvt_f16_f32",
                    defs=(out,),
                    uses=(src[0],),
                    comment="truncf f32->f16",
                )
            )
            ctx.ssa_to_reg[dst_ssa] = (out,)
            return

        # Scalar fallback: forward.
        if len(src) == 1:
            ctx.ssa_to_reg[dst_ssa] = src
            return

    # -------------------------------------------------------------------------
    # Vector/int ops used by wide bf16 softmax (packing)
    # -------------------------------------------------------------------------

    def handle_arith_bitcast_op(self, operation: arith_d.BitcastOp, kernel_info: KernelInfo):
        ctx = self.walker.kernel_ctx
        src = ctx.ssa_to_reg.get(ssa(operation.in_))
        if src is None:
            return
        ctx.ssa_to_reg[ssa(operation.out)] = tuple(src)

    def handle_arith_shrui_op(self, operation, kernel_info: KernelInfo):
        """Vector/scalar logical shift right by immediate."""
        ctx = self.walker.kernel_ctx
        lhs_ssa = ssa(operation.lhs)
        rhs_ssa = ssa(operation.rhs)
        lhs = ctx.ssa_to_reg.get(lhs_ssa)
        if lhs is None:
            return
        rhs = ctx.ssa_to_reg.get(rhs_ssa)
        if rhs is not None and len(rhs) == len(lhs):
            out = tuple(ctx.v_lshrrev_b32(s, r, comment="shru") for r, s in zip(lhs, rhs))
        else:
            sh = self._get_scalar_shift_imm(rhs_ssa, kernel_info)
            if sh is None:
                return
            out = tuple(ctx.v_lshrrev_b32(KImm(sh), r, comment="shru") for r in lhs)
        ctx.ssa_to_reg[ssa(operation.result)] = out

    def handle_arith_shli_op(self, operation, kernel_info: KernelInfo):
        ctx = self.walker.kernel_ctx
        lhs_ssa = ssa(operation.lhs)
        rhs_ssa = ssa(operation.rhs)
        lhs = ctx.ssa_to_reg.get(lhs_ssa)
        if lhs is None:
            return
        rhs = ctx.ssa_to_reg.get(rhs_ssa)
        if rhs is not None and len(rhs) == len(lhs):
            out = tuple(ctx.v_lshlrev_b32(s, r, comment="shl") for r, s in zip(lhs, rhs))
        else:
            sh = self._get_scalar_shift_imm(rhs_ssa, kernel_info)
            if sh is None:
                return
            out = tuple(ctx.v_lshlrev_b32(KImm(sh), r, comment="shl") for r in lhs)
        ctx.ssa_to_reg[ssa(operation.result)] = out

    def handle_arith_andi_op(self, operation, kernel_info: KernelInfo):
        ctx = self.walker.kernel_ctx
        lhs = ctx.ssa_to_reg.get(ssa(operation.lhs))
        rhs_ssa = ssa(operation.rhs)
        if lhs is None:
            return
        rhs = ctx.ssa_to_reg.get(rhs_ssa)
        if rhs is not None and len(rhs) == len(lhs):
            out = tuple(ctx.v_and_b32(a, b, comment="and") for a, b in zip(lhs, rhs))
        else:
            m = kernel_info.index_env.get(rhs_ssa)
            if not isinstance(m, int):
                return
            mask_r = ctx.vreg()
            ctx.program.emit(
                KInstr(Instruction.V_MOV_B32, defs=(mask_r,), uses=(KImm(int(m)),), comment="and mask")
            )
            out = tuple(ctx.v_and_b32(r, mask_r, comment="and") for r in lhs)
        ctx.ssa_to_reg[ssa(operation.result)] = out

    def handle_arith_extf_op(self, operation: arith_d.ExtFOp, kernel_info: KernelInfo):
        """Widen f16/bf16 to f32 (scalar or vector)."""
        ctx = self.walker.kernel_ctx
        src = ctx.ssa_to_reg.get(ssa(operation.in_))
        if not src:
            return
        src_ty = str(operation.in_.type)
        # Vector types: vector<...xbf16> / vector<...xf16>
        if src_ty.startswith("vector<") and ("bf16" in src_ty or "f16" in src_ty):
            # Expect packed dwords in groups of 2 lanes.
            mask = ctx.vreg()
            ctx.program.emit(KInstr(Instruction.V_MOV_B32, defs=(mask,), uses=(KImm(0xFFFF),), comment="u16 mask"))
            outs = []
            for p in src:
                lo = ctx.v_and_b32(p, mask, comment="u16 lo")
                hi = ctx.v_lshrrev_b32(KImm(16), p, comment="u16 hi")
                if "bf16" in src_ty:
                    outs.append(ctx.v_lshlrev_b32(KImm(16), lo, comment="bf16->f32 lo"))
                    outs.append(ctx.v_lshlrev_b32(KImm(16), hi, comment="bf16->f32 hi"))
                else:
                    outs.append(ctx.v_cvt_f32_f16(lo, comment="f16->f32 lo"))
                    outs.append(ctx.v_cvt_f32_f16(hi, comment="f16->f32 hi"))
            ctx.ssa_to_reg[ssa(operation.out)] = tuple(outs)
            return

        # Scalar extf.
        src_r = src[0]
        if src_ty == "f16":
            out = ctx.v_cvt_f32_f16(src_r, comment="extf f16->f32")
        elif src_ty == "bf16":
            out = ctx.v_lshlrev_b32(KImm(16), src_r, comment="bf16<<16")
        else:
            out = src_r
        ctx.ssa_to_reg[ssa(operation.out)] = (out,)

    def handle_arith_ori_op(self, operation, kernel_info: KernelInfo):
        ctx = self.walker.kernel_ctx
        lhs = ctx.ssa_to_reg.get(ssa(operation.lhs))
        rhs = ctx.ssa_to_reg.get(ssa(operation.rhs))
        if lhs is None or rhs is None or len(lhs) != len(rhs):
            return
        out = tuple(ctx.v_or_b32(a, b, comment="or") for a, b in zip(lhs, rhs))
        ctx.ssa_to_reg[ssa(operation.result)] = out

    # -------------------------------------------------------------------------
    # Predicates + select (VCC-based)
    # -------------------------------------------------------------------------

    def handle_arith_cmpi_op(self, operation: arith_d.CmpIOp, kernel_info: KernelInfo):
        """Record + (best-effort) compute a predicate into VCC."""
        ctx = self.walker.kernel_ctx
        res_ssa = ssa(operation.result)
        # `arith.cmpi` predicate can stringify as an enum ("ult") or as an
        # integer attribute like "6 : i64" depending on bindings/build.
        pred_raw = str(operation.predicate).strip()
        pred = pred_raw
        try:
            import re

            m = re.match(r"^-?\d+", pred_raw)
            if m:
                code = int(m.group(0))
                # MLIR arith.cmpi predicate codes:
                # 0 eq, 1 ne, 2 slt, 3 sle, 4 sgt, 5 sge, 6 ult, 7 ule, 8 ugt, 9 uge
                pred = {
                    0: "eq",
                    1: "ne",
                    2: "slt",
                    3: "sle",
                    4: "sgt",
                    5: "sge",
                    6: "ult",
                    7: "ule",
                    8: "ugt",
                    9: "uge",
                }.get(code, pred_raw)
        except Exception:
            pred = pred_raw
        lhs_ssa = ssa(operation.lhs)
        rhs_ssa = ssa(operation.rhs)
        self.walker._cmpi_predicates = getattr(self.walker, "_cmpi_predicates", {})
        self.walker._cmpi_predicates[res_ssa] = (pred, lhs_ssa, rhs_ssa)

        # Emit VCC now (most uses are immediate).
        self._emit_vcc_for(res_ssa, kernel_info)

    def _emit_vcc_for(self, cond_ssa: str, kernel_info: KernelInfo):
        """Emit compare and set VCC for a previously-seen cmpi result SSA."""
        from .kernel_ir import KInstr, KImm, VCC, KInstrConstraints

        info = getattr(self.walker, "_cmpi_predicates", {}).get(cond_ssa)
        if info is None:
            return
        pred, lhs_ssa, rhs_ssa = info
        ctx = self.walker.kernel_ctx

        def _opnd(ssa_name):
            r = ctx.ssa_to_reg.get(ssa_name)
            if r and len(r) == 1:
                return r[0]
            c = kernel_info.index_env.get(ssa_name)
            if isinstance(c, int):
                return KImm(int(c))
            return None

        lhs = _opnd(lhs_ssa)
        rhs = _opnd(rhs_ssa)
        if lhs is None or rhs is None:
            return

        # NOTE: We emit compare ops directly as KInstr (string names), so we must
        # explicitly mark VCC clobbering for downstream hazard mitigation.
        vcc_clobber = KInstrConstraints(vcc_clobber=True)

        if "eq" in pred:
            ctx.program.emit(
                KInstr(
                    "v_cmp_eq_u32",
                    defs=(VCC,),
                    uses=(lhs, rhs),
                    constraints=vcc_clobber,
                    comment="cmp eq",
                )
            )
        elif "ne" in pred:
            ctx.program.emit(
                KInstr(
                    "v_cmp_ne_u32",
                    defs=(VCC,),
                    uses=(lhs, rhs),
                    constraints=vcc_clobber,
                    comment="cmp ne",
                )
            )
        elif "ult" in pred:
            ctx.program.emit(
                KInstr(
                    "v_cmp_lt_u32",
                    defs=(VCC,),
                    uses=(lhs, rhs),
                    constraints=vcc_clobber,
                    comment="cmp ult",
                )
            )

    def handle_arith_select_op(self, operation: arith_d.SelectOp, kernel_info: KernelInfo):
        """Lower select via v_cndmask_b32 using VCC."""
        from .kernel_ir import VCC

        ctx = self.walker.kernel_ctx
        cond_ssa = ssa(operation.condition)
        self._emit_vcc_for(cond_ssa, kernel_info)

        t = ctx.ssa_to_reg.get(ssa(operation.true_value))
        f = ctx.ssa_to_reg.get(ssa(operation.false_value))
        if not t or not f or len(t) != len(f):
            return

        # v_cndmask selects src0 when VCC=0, else src1.
        out = tuple(ctx.v_cndmask_b32(fv, tv, VCC, comment="select") for fv, tv in zip(f, t))
        ctx.ssa_to_reg[ssa(operation.result)] = out

    # -------------------------------------------------------------------------
    # math + gpu shuffle (softmax)
    # -------------------------------------------------------------------------

    def handle_math_exp2_op(self, operation, kernel_info: KernelInfo):
        """Lower math.exp2 for f32 with range reduction.

        ROCm codegen typically lowers `exp2f(x)` as:
          n = (int)x          (truncate toward 0)
          f = x - (float)n    (fractional part in (-1,1))
          r = v_exp_f32(f)    (2^f approximation)
          y = v_ldexp_f32(r,n)  (scale by 2^n)

        This matches `exp2f` semantics for our value ranges and is needed because
        `v_exp_f32` alone is not accurate for large |x|.
        """
        ctx = self.walker.kernel_ctx
        src = ctx.ssa_to_reg.get(ssa(operation.operands[0]))
        if not src:
            return
        outs = []
        for x in src:
            n_i = ctx.vreg()
            ctx.program.emit(KInstr("v_cvt_i32_f32", defs=(n_i,), uses=(x,), comment="exp2 n=int(x)"))
            n_f = ctx.vreg()
            ctx.program.emit(KInstr("v_cvt_f32_i32", defs=(n_f,), uses=(n_i,), comment="exp2 float(n)"))
            frac = ctx.v_sub_f32(x, n_f, comment="exp2 frac")
            r = ctx.v_exp_f32(frac, comment="exp2 2^frac")
            out = ctx.vreg()
            ctx.program.emit(KInstr("v_ldexp_f32", defs=(out,), uses=(r, n_i), comment="exp2 scale"))
            outs.append(out)
        ctx.ssa_to_reg[ssa(operation.results[0])] = tuple(outs)

    def handle_gpu_shuffle_op(self, operation, kernel_info: KernelInfo):
        """Lower gpu.shuffle xor for f32 via ds_bpermute_b32.

        Only supports `xor` mode with width=64 (wave64), which is what the
        softmax reductions use.
        """
        from .kernel_ir import KVReg

        ctx = self.walker.kernel_ctx
        # gpu.shuffle operands: (value, offset, width)
        val_ssa = ssa(operation.operands[0])
        off_ssa = ssa(operation.operands[1])
        val = ctx.ssa_to_reg.get(val_ssa)
        if not val or len(val) != 1:
            return

        # Offset is typically a constant i32.
        off_c = kernel_info.index_env.get(off_ssa)
        off_r = ctx.ssa_to_reg.get(off_ssa)
        if isinstance(off_c, int):
            off_op = KImm(int(off_c))
        elif off_r and len(off_r) == 1:
            off_op = off_r[0]
        else:
            return

        # Compute lane_id within the wave: (tid_x & 63) for multi-wave.
        lane = getattr(self.walker, "_lane_id_v", None)
        if lane is None:
            tid = ctx.ensure_tid_x()
            lane = ctx.v_and_b32(tid, KImm(63), comment="lane_id = tid_x & 63")
            self.walker._lane_id_v = lane

        # target_lane = lane_id xor offset
        tgt = ctx.v_xor_b32(lane, off_op, comment="shuffle lane xor")
        # Match ROCm codegen: address is byte offset (lane * 4).
        addr = ctx.v_lshlrev_b32(KImm(2), tgt, comment="shuffle addr bytes")

        dst = ctx.vreg()
        ctx.program.emit(
            KInstr(
                "ds_bpermute_b32",
                defs=(dst,),
                uses=(addr, val[0]),
                comment="gpu.shuffle xor",
            )
        )
        # Conservative: ensure shuffle result is ready.
        self.walker.unified.s_waitcnt(waitcnt="lgkmcnt(0)")
        ctx.ssa_to_reg[ssa(operation.results[0])] = (dst,)

        # valid result is unused in our kernels; materialize as 'true' (1).
        if len(operation.results) > 1:
            v = ctx.sreg()
            ctx.program.emit(
                KInstr(Instruction.S_MOV_B32, defs=(v,), uses=(KImm(1),), comment="shuffle valid=true")
            )
            ctx.ssa_to_reg[ssa(operation.results[1])] = (v,)

    def handle_subgroup_broadcast_op(self, operation, kernel_info: KernelInfo):
        """Handle gpu.subgroup_broadcast - propagate value for uniform broadcast.

        The expression is preserved as-is (not evaluated) because each wavefront
        has different tid values. v_readfirstlane is emitted during code generation.
        """
        result_ssa = str(operation.results[0])
        source_ssa = str(operation.src)

        if source_ssa in kernel_info.index_env:
            kernel_info.index_env[result_ssa] = kernel_info.index_env[source_ssa]

    def _extract_dimension_values(
        self,
        operation: affine_d.AffineApplyOp,
        kernel_info: KernelInfo,
        num_dimensions: int,
    ) -> list:
        """Extract dimension values from the first num_dimensions operands."""
        import sympy

        dimension_values = []

        for i in range(num_dimensions):
            if i < len(operation.operands):
                operand_ssa = str(operation.operands[i])
                operand_value = kernel_info.index_env.get(operand_ssa)

                if isinstance(operand_value, int):
                    dimension_values.append(operand_value)
                elif isinstance(operand_value, sympy.Expr):
                    # SymPy expressions from previous affine.apply results
                    dimension_values.append(operand_value)
                elif operand_value in [
                    "tid_x",
                    "tid_y",
                    "tid_z",
                    "wgid_x",
                    "wgid_y",
                    "wgid_z",
                ]:
                    # Thread IDs and workgroup IDs can be represented as symbols in the expression
                    dimension_values.append(operand_value)
                else:
                    # If we can't resolve the dimension value, we can't simplify
                    return None
            else:
                # Not enough operands for the expected number of dimensions
                return None

        return dimension_values

    def _extract_symbol_values(
        self,
        operation: affine_d.AffineApplyOp,
        kernel_info: KernelInfo,
        num_dimensions: int,
        num_symbols: int,
    ) -> list:
        """Extract symbol values from the next num_symbols operands."""
        import sympy

        symbol_values = []

        for i in range(num_symbols):
            operand_index = num_dimensions + i
            if operand_index < len(operation.operands):
                operand_ssa = str(operation.operands[operand_index])
                operand_value = kernel_info.index_env.get(operand_ssa)

                if isinstance(operand_value, int):
                    symbol_values.append(operand_value)
                elif isinstance(operand_value, sympy.Expr):
                    # SymPy expressions from previous affine.apply results
                    symbol_values.append(operand_value)
                elif operand_value in [
                    "tid_x",
                    "tid_y",
                    "tid_z",
                    "wgid_x",
                    "wgid_y",
                    "wgid_z",
                ]:
                    # Thread IDs and workgroup IDs can be used as symbol values
                    symbol_values.append(operand_value)
                elif (
                    isinstance(operand_value, str)
                    and operand_value.startswith("s")
                    and operand_value[1:].isdigit()
                ):
                    # SGPR references (e.g., "s4" for loop counter) can be used as symbol values
                    symbol_values.append(operand_value)
                elif isinstance(operand_value, KSReg):
                    # Virtual SGPR references (loop counters) can be used as symbol values
                    # Convert to ks{id} symbol format for expression evaluation
                    symbol_values.append(
                        sympy.Symbol(f"ks{operand_value.id}", nonnegative=True)
                    )
                else:
                    # If we can't resolve the symbol value, we can't simplify
                    return None
            else:
                # Not enough operands for the expected number of symbols
                return None

        return symbol_values

    def handle_affine_apply_op(
        self, operation: affine_d.AffineApplyOp, kernel_info: KernelInfo
    ):
        """Handle affine.apply operations - simplify affine expressions."""
        # The first operands correspond to dimensions, the rest to symbols
        affine_map_attribute = operation.map
        affine_map = affine_map_attribute.value
        num_dimensions = affine_map.n_dims
        num_symbols = affine_map.n_symbols

        # Extract dimension and symbol values from operands
        dimension_values = self._extract_dimension_values(
            operation, kernel_info, num_dimensions
        )
        symbol_values = self._extract_symbol_values(
            operation, kernel_info, num_dimensions, num_symbols
        )

        # Try to simplify the expression with actual values
        simplified_expression = simplify_expression(
            operation.map, kernel_info.tid_ub_x, dimension_values, symbol_values
        )

        destination_ssa = str(operation.result)
        if simplified_expression is not None:
            # Check if the simplified expression is a constant
            if len(simplified_expression.free_symbols) == 0:
                # Expression has no free symbols, so it's a constant - convert to int
                assert hasattr(simplified_expression, "__int__") or hasattr(
                    simplified_expression, "is_integer"
                ), f"Simplified expression without free symbols should be convertible to int: {simplified_expression}"
                constant_value = int(simplified_expression)
                kernel_info.index_env[destination_ssa] = constant_value
            # Check if the simplified expression is a thread ID symbol
            else:
                import sympy

                # Check for all thread ID types
                thread_id_symbols = {
                    "tid_x": sympy.Symbol("tid_x", nonnegative=True),
                    "tid_y": sympy.Symbol("tid_y", nonnegative=True),
                    "tid_z": sympy.Symbol("tid_z", nonnegative=True),
                }

                matched_tid = False
                for thread_id_name, thread_id_symbol in thread_id_symbols.items():
                    if simplified_expression == thread_id_symbol:
                        # Map back to the original thread ID format
                        original_thread_id = thread_id_name.replace("_", ".")
                        source_ssa = (
                            str(operation.operands[0])
                            if len(operation.operands) > 0
                            else None
                        )
                        if (
                            source_ssa
                            and kernel_info.index_env.get(source_ssa)
                            == original_thread_id
                        ):
                            kernel_info.index_env[destination_ssa] = original_thread_id
                            matched_tid = True
                        break

                if not matched_tid:
                    # Store the simplified SymPy expression for later ASM emission
                    kernel_info.index_env[destination_ssa] = simplified_expression
