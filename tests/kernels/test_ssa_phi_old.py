#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""OLD test: small phi (1 x i64), identical branch structure.

This is the original test that could NOT reproduce the SSA phi explosion.
Each branch chains 8 MFMAs producing f32x4, packs it to i64 within the
branch (f32x4 is branch-local), and only yields the single i64 through phi.

Expected result: NO phi explosion.
  - LLVM merges identical branches via v_cndmask pointer selection
  - Even if branches survive, only 1 x i64 crosses phi (trivial to coalesce)
  - Both branches use the same VGPR range for MFMA destinations

Used as baseline for factorial analysis in test_ssa_phi_factors.py.
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
NUM_MFMA = 8  # chain length per branch


def _raw(v):
    if hasattr(v, "ir_value"):
        return v.ir_value()
    if hasattr(v, "result"):
        return v.result
    return v


def _i32_const(val):
    return _std_arith.ConstantOp(T.i32, val).result


def build_small_phi_identical(N: int):
    """Small phi, identical branch structure.

    Each branch:
      1. Loads 4 x f16 from a buffer (A or B)
      2. Chains NUM_MFMA MFMAs producing f32x4 (branch-local)
      3. Packs f32x4 -> f16x4 -> i64 within the branch
      4. Yields only the single i64 through phi
    """

    @flyc.kernel
    def ssa_phi_old_kernel(
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
        row_off = _std_arith.MulIOp(bid_i32, _i32_const(N)).result
        tid_off = _std_arith.MulIOp(tid_i32, _i32_const(4)).result
        base_off = _std_arith.AddIOp(row_off, tid_off).result

        b_frag = _raw(full(4, Float16(1.0), Float16))
        c_zero = _raw(full(4, Float32(0.0), Float32))

        def _mfma_chain(a_frag):
            """Chain NUM_MFMA MFMAs. Returns f32x4 (branch-local)."""
            accu = _raw(fx.rocdl.mfma_f32_16x16x16f16(
                f32x4_type, [a_frag, b_frag, c_zero, 0, 0, 0]
            ))
            for _ in range_constexpr(NUM_MFMA - 1):
                accu = _raw(fx.rocdl.mfma_f32_16x16x16f16(
                    f32x4_type, [a_frag, b_frag, accu, 0, 0, 0]
                ))
            return accu

        def _pack_to_i64(p_comp_f32x4):
            """Pack f32x4 -> f16x4 -> i64 (branch-local consumption of f32x4)."""
            f16x4_type = T.f16x4
            truncated = _std_arith.TruncFOp(f16x4_type, p_comp_f32x4).result
            packed = _vd.bitcast(T.vec(1, T.i64), truncated)
            return _vd.extract(packed, [], [0])

        # Branch on warp_id — identical computation, different data
        warp_id = ArithValue(tid) // WARP_SIZE
        is_first_warp = _std_arith.CmpIOp(
            _std_arith.CmpIPredicate.eq,
            _raw(warp_id),
            _i32_const(0),
        ).result

        i64_type = T.i64
        result_types = [i64_type]  # only 1 x i64 crosses phi
        if_op = scf.IfOp(is_first_warp, result_types, has_else=True)

        # Branch A: load from A, MFMA chain, pack to i64
        with ir.InsertionPoint(if_op.regions[0].blocks[0]):
            a_frag_a = buffer_ops.buffer_load(
                rsrc_a, base_off, vec_width=4, dtype=T.f16
            )
            p_comp_a = _mfma_chain(a_frag_a)
            p_pack_a = _pack_to_i64(p_comp_a)
            scf.YieldOp([p_pack_a])

        # Branch B: load from B, same MFMA chain, pack to i64
        with ir.InsertionPoint(if_op.regions[1].blocks[0]):
            a_frag_b = buffer_ops.buffer_load(
                rsrc_b, base_off, vec_width=4, dtype=T.f16
            )
            p_comp_b = _mfma_chain(a_frag_b)
            p_pack_b = _pack_to_i64(p_comp_b)
            scf.YieldOp([p_pack_b])

        # After merge: only i64 came through phi, f32x4 was consumed within branches
        p_pack = if_op.results[0]

        # Store packed result (as 2 x f32) to prevent DCE
        p_vec1 = _vd.broadcast(T.vec(1, T.i64), p_pack)
        p_as_f32x2 = _vd.bitcast(T.vec(2, T.f32), p_vec1)

        for elem in range_constexpr(2):
            val = _vd.extract(p_as_f32x2, [], [elem])
            store_idx = _std_arith.AddIOp(
                _std_arith.AddIOp(row_off, _std_arith.MulIOp(tid_i32, _i32_const(2)).result).result,
                _i32_const(elem),
            ).result
            buffer_ops.buffer_store(val, rsrc_c, store_idx)

    @flyc.jit
    def launch_old(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        idx_m = ArithValue(m_in).index_cast(T.index)
        launcher = ssa_phi_old_kernel(A, B, C)
        launcher.launch(
            grid=(idx_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_old


def _parse_isa_stats(isa_path):
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
            for p in stripped.split():
                if p.startswith("v[") and ":" in p:
                    stats["unique_mfma_dsts"].add(p.rstrip(","))
        if "vgpr_spill_count" in stripped:
            try:
                stats["vgpr_spill_count"] = int(stripped.split(":")[-1].strip())
            except ValueError:
                pass
        if ".vgpr_count" in stripped:
            try:
                stats["vgpr_count"] = int(stripped.split(":")[-1].strip())
            except ValueError:
                pass
    return stats


def test_small_phi_identical():
    """OLD test: small phi, identical branches. Should NOT show phi explosion."""
    M = 4
    N = BLOCK_THREADS * 8

    print(f"\n{'='*70}")
    print(f"OLD: Small phi (1 x i64), identical branches")
    print(f"  M={M}, N={N}, BLOCK_THREADS={BLOCK_THREADS}")
    print(f"  arch={get_rocm_arch()}")
    print(f"{'='*70}")

    launch_fn = build_small_phi_identical(N)

    a_dev = torch.randn((M, N), device="cuda", dtype=torch.float16)
    b_dev = torch.randn((M, N), device="cuda", dtype=torch.float16)
    c_dev = torch.zeros((M, BLOCK_THREADS * 2), device="cuda", dtype=torch.float32)
    stream = torch.cuda.current_stream()

    launch_fn(a_dev, b_dev, c_dev, M, stream=stream)
    torch.cuda.synchronize()

    nonzero = (c_dev != 0).sum().item()
    print(f"  Non-zero outputs: {nonzero}/{c_dev.numel()}")
    assert nonzero > 0

    isa_path = os.path.expanduser("~/.flydsl/debug/ssa_phi_old_kernel_0/17_final_isa.s")
    stats = _parse_isa_stats(isa_path)
    if stats:
        print(f"\n  ISA: {stats['total_mfma_count']} MFMAs, "
              f"{len(stats['unique_mfma_dsts'])} unique dst groups, "
              f"{stats['vgpr_count']} VGPRs, "
              f"{stats['vgpr_spill_count']} spills")

    llvm_path = os.path.expanduser("~/.flydsl/debug/ssa_phi_old_kernel_0/16_llvm_ir.ll")
    try:
        with open(llvm_path) as f:
            ir_text = f.read()
        phi_f32x4 = ir_text.count("phi <4 x float>")
        phi_total = ir_text.count(" phi ")
        print(f"  LLVM IR: {phi_total} phi nodes, {phi_f32x4} x <4 x float>")
    except FileNotFoundError:
        pass

    print(f"  PASSED")
    return True


if __name__ == "__main__":
    if "FLYDSL_DUMP_IR" not in os.environ:
        os.environ["FLYDSL_DUMP_IR"] = "1"
    ok = test_small_phi_identical()
    sys.exit(0 if ok else 1)
