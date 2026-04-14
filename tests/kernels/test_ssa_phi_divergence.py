#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Reproduce the SSA phi explosion that causes VGPR waste and spills on AMDGPU.

Problem
-------
When a large accumulator (N x f32x4 = 4N VGPRs) flows through an ``scf.IfOp``
phi, LLVM's register allocator makes independent allocation choices per branch.
With 32 x f32x4 = 128 VGPRs crossing phi, the two branches may assign disjoint
VGPR ranges, effectively doubling register consumption at the merge point.  The
allocator must insert copies to coalesce them, and under pressure this causes
scratch spills.

This was the historically catastrophic issue in the MLA decode kernel:
  - oaccu (32 x f32x4 = 128 VGPRs) flowed through phi across 6 branch instances
  - LLVM produced 71 unique MFMA destination register groups
  - 170 scratch spills (vgpr_spill_count = 170)
  - Fix: restructure so oaccu never enters phi nodes -- keep it branch-local

This test yields NUM_ACCU_VECS x f32x4 through ``scf.IfOp`` phi.  Each branch
performs structurally different computation (different chain lengths and
intermediate ops) so LLVM cannot merge them via v_cndmask pointer selection.

How to inspect
--------------
::

    FLYDSL_DUMP_IR=1 python tests/kernels/test_ssa_phi_divergence.py
    grep 'v_mfma' ~/.flydsl/debug/ssa_phi_accu_kernel_0/17_final_isa.s
    grep 'vgpr_spill_count' ~/.flydsl/debug/ssa_phi_accu_kernel_0/17_final_isa.s
"""

import os
import sys

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

try:
    import torch
except ImportError:
    torch = None
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.expr.vector import full
from flydsl.expr.numeric import Float16, Float32
from flydsl.expr import buffer_ops

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects import arith as _std_arith
from flydsl._mlir.dialects import vector as _vd

from flydsl.runtime.device import get_rocm_arch

from kernels.kernels_common import get_warp_size

WARP_SIZE = get_warp_size()
BLOCK_THREADS = 64

# Number of f32x4 vectors yielded through phi.
NUM_ACCU_VECS = 32


def _raw(v):
    """Unwrap to ir.Value."""
    if hasattr(v, "ir_value"):
        return v.ir_value()
    if hasattr(v, "result"):
        return v.result
    return v


def _i32_const(val):
    """Create an i32 constant in the current insertion point."""
    return _std_arith.ConstantOp(T.i32, val).result


def build_ssa_phi_accu(N: int):
    """Build a kernel where a large accumulator crosses scf.IfOp phi.

    Each branch does structurally different computation:
      - Branch A: load from A, chain 3 MFMAs per accumulator
      - Branch B: load from B, chain 2 MFMAs + MulFOp per accumulator
    This prevents LLVM from merging branches via v_cndmask pointer selection.
    """

    @flyc.kernel
    def ssa_phi_accu_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32x4_type = T.f32x4

        rsrc_a = buffer_ops.create_buffer_resource(A)
        rsrc_b = buffer_ops.create_buffer_resource(B)
        rsrc_c = buffer_ops.create_buffer_resource(C)

        tid_i32 = _raw(ArithValue(tid))
        bid_i32 = _raw(ArithValue(bid))
        row_elem_offset = _std_arith.MulIOp(bid_i32, _i32_const(N)).result
        tid_elem_offset = _std_arith.MulIOp(tid_i32, _i32_const(4)).result
        base_elem_offset = _std_arith.AddIOp(row_elem_offset, tid_elem_offset).result

        b_frag = _raw(full(4, Float16(1.0), Float16))
        c_zero = _raw(full(4, Float32(0.0), Float32))

        # Scale factor for branch B's MulFOp (different from 1.0 to prevent DCE)
        scale_vec = _raw(full(4, Float32(0.5), Float32))

        warp_id = ArithValue(tid) // WARP_SIZE
        is_first_warp = _std_arith.CmpIOp(
            _std_arith.CmpIPredicate.eq,
            _raw(warp_id),
            _i32_const(0),
        ).result

        result_types = [f32x4_type] * NUM_ACCU_VECS
        if_op = scf.IfOp(is_first_warp, result_types, has_else=True)

        # Branch A (warp 0): load from A, 3-MFMA chain per accumulator
        with ir.InsertionPoint(if_op.regions[0].blocks[0]):
            yields_a = []
            for vec_idx in range_constexpr(NUM_ACCU_VECS):
                elem_off = _std_arith.AddIOp(
                    base_elem_offset,
                    _i32_const(vec_idx * BLOCK_THREADS * 4),
                ).result
                a_frag = buffer_ops.buffer_load(
                    rsrc_a, elem_off, vec_width=4, dtype=T.f16
                )
                # 3-MFMA chain (structurally different from branch B)
                accu = _raw(fx.rocdl.mfma_f32_16x16x16f16(
                    f32x4_type, [a_frag, b_frag, c_zero, 0, 0, 0]
                ))
                accu = _raw(fx.rocdl.mfma_f32_16x16x16f16(
                    f32x4_type, [a_frag, b_frag, accu, 0, 0, 0]
                ))
                accu = _raw(fx.rocdl.mfma_f32_16x16x16f16(
                    f32x4_type, [a_frag, b_frag, accu, 0, 0, 0]
                ))
                yields_a.append(accu)
            scf.YieldOp(yields_a)

        # Branch B (other warps): load from B, 2-MFMA + MulF per accumulator
        with ir.InsertionPoint(if_op.regions[1].blocks[0]):
            yields_b = []
            for vec_idx in range_constexpr(NUM_ACCU_VECS):
                elem_off = _std_arith.AddIOp(
                    base_elem_offset,
                    _i32_const(vec_idx * BLOCK_THREADS * 4),
                ).result
                a_frag = buffer_ops.buffer_load(
                    rsrc_b, elem_off, vec_width=4, dtype=T.f16
                )
                # 2-MFMA + MulF chain (structurally different from branch A)
                accu = _raw(fx.rocdl.mfma_f32_16x16x16f16(
                    f32x4_type, [a_frag, b_frag, c_zero, 0, 0, 0]
                ))
                accu = _raw(fx.rocdl.mfma_f32_16x16x16f16(
                    f32x4_type, [a_frag, b_frag, accu, 0, 0, 0]
                ))
                # Multiply by scale (this extra op makes the DAG different)
                accu = _std_arith.MulFOp(accu, scale_vec).result
                yields_b.append(accu)
            scf.YieldOp(yields_b)

        # After merge: all NUM_ACCU_VECS f32x4 values came through phi.
        merged = [if_op.results[i] for i in range(NUM_ACCU_VECS)]

        # Sum all f32x4 to one (prevents DCE)
        total = merged[0]
        for i in range_constexpr(1, NUM_ACCU_VECS):
            total = _std_arith.AddFOp(total, merged[i]).result

        # Store to C
        for elem in range_constexpr(4):
            val = _vd.extract(total, [], [elem])
            store_idx = _std_arith.AddIOp(
                _std_arith.AddIOp(row_elem_offset, tid_elem_offset).result,
                _i32_const(elem),
            ).result
            buffer_ops.buffer_store(val, rsrc_c, store_idx)

    @flyc.jit
    def launch_ssa_phi_accu(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        idx_m = ArithValue(m_in).index_cast(T.index)
        launcher = ssa_phi_accu_kernel(A, B, C)
        launcher.launch(
            grid=(idx_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_ssa_phi_accu


def _parse_isa_stats(isa_path):
    """Parse the ISA file for MFMA destinations and spill count."""
    stats = {
        "vgpr_spill_count": 0,
        "vgpr_count": 0,
        "total_mfma_count": 0,
        "unique_mfma_dsts": set(),
    }

    try:
        with open(isa_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    for line in lines:
        stripped = line.strip()

        if "v_mfma" in stripped:
            stats["total_mfma_count"] += 1
            parts = stripped.split()
            for p in parts:
                if p.startswith("v[") and ":" in p:
                    dst = p.rstrip(",")
                    stats["unique_mfma_dsts"].add(dst)

        if "vgpr_spill_count" in stripped:
            try:
                val = stripped.split(":")[-1].strip()
                stats["vgpr_spill_count"] = int(val)
            except ValueError:
                pass
        if ".vgpr_count" in stripped:
            try:
                val = stripped.split(":")[-1].strip()
                stats["vgpr_count"] = int(val)
            except ValueError:
                pass

    return stats


def test_ssa_phi_accu_explosion():
    """Compile and run the SSA phi accumulator explosion demo."""
    M = 4
    N = NUM_ACCU_VECS * BLOCK_THREADS * 4

    print(f"\n{'='*70}")
    print(f"SSA Phi Accumulator Explosion Demo")
    print(f"  M={M}, N={N}, BLOCK_THREADS={BLOCK_THREADS}")
    print(f"  NUM_ACCU_VECS={NUM_ACCU_VECS} ({NUM_ACCU_VECS * 4} VGPRs through phi)")
    print(f"  arch={get_rocm_arch()}")
    print(f"{'='*70}")

    launch_fn = build_ssa_phi_accu(N)

    a_dev = torch.randn((M, N), device="cuda", dtype=torch.float16)
    b_dev = torch.randn((M, N), device="cuda", dtype=torch.float16)
    c_dev = torch.zeros((M, N), device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream()

    launch_fn(a_dev, b_dev, c_dev, M, stream=stream)
    torch.cuda.synchronize()

    nonzero_count = (c_dev[:, :BLOCK_THREADS * 4] != 0).sum().item()
    print(f"  Non-zero outputs: {nonzero_count}/{BLOCK_THREADS * 4 * M}")
    assert nonzero_count > 0, "All outputs are zero -- kernel may not have run"

    isa_path = os.path.expanduser(
        "~/.flydsl/debug/ssa_phi_accu_kernel_0/17_final_isa.s"
    )
    stats = _parse_isa_stats(isa_path)

    if stats:
        # With NUM_ACCU_VECS=32:
        # Branch A: 32 * 3 = 96 MFMAs, each writing to f32x4
        # Branch B: 32 * 2 = 64 MFMAs
        # Ideal (shared VGPRs): 32 unique dst groups
        # SSA phi explosion: many more unique dst groups (up to 160)
        print(f"\n  ISA analysis ({isa_path}):")
        print(f"    Total MFMA instructions: {stats['total_mfma_count']}")
        print(f"    Unique MFMA dst groups: {len(stats['unique_mfma_dsts'])}")
        print(f"    VGPR count: {stats['vgpr_count']}")
        print(f"    VGPR spill count: {stats['vgpr_spill_count']}")

        # The ideal case (branches share VGPR ranges): ~32 unique dst groups
        # The SSA phi case (branches use different ranges): >> 32 unique dst groups
        if len(stats['unique_mfma_dsts']) > 40:
            print(f"\n  ** PHI EXPLOSION: {len(stats['unique_mfma_dsts'])} unique MFMA dst groups "
                  f"(ideal: ~32 if branches shared VGPRs) **")
        if stats["vgpr_spill_count"] > 0:
            print(f"  ** SPILLS DETECTED: {stats['vgpr_spill_count']} spills **")
    else:
        print(f"\n  ISA file not found at {isa_path}")
        print(f"  Run with FLYDSL_DUMP_IR=1 to generate ISA output")

    # Count phi nodes in LLVM IR
    llvm_ir_path = os.path.expanduser(
        "~/.flydsl/debug/ssa_phi_accu_kernel_0/16_llvm_ir.ll"
    )
    try:
        with open(llvm_ir_path) as f:
            ir_text = f.read()
        phi_f32x4 = ir_text.count("phi <4 x float>")
        phi_total = ir_text.count(" phi ")
        print(f"\n  LLVM IR phi nodes: {phi_total} total, {phi_f32x4} x <4 x float>")
    except FileNotFoundError:
        pass

    print(f"\n  PASSED (kernel compiled and produced output)")
    return True


if __name__ == "__main__":
    if "FLYDSL_DUMP_IR" not in os.environ:
        os.environ["FLYDSL_DUMP_IR"] = "1"

    ok = test_ssa_phi_accu_explosion()
    sys.exit(0 if ok else 1)
