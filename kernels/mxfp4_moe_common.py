# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Shared raw helpers / constants / K-derived size formulas for the layout-API
MXFP4 MoE gemm (down-proj gemm2 + up/gate gemm1).

Extracted verbatim from the original aiter port so both ``mxfp4_moe_gemm1`` and
``mxfp4_moe_gemm2`` share one definition of the pointer/LDS helpers, the e8m0
scale-layout size formulas, and the atomic bf16 epilogue. The gemm bodies
themselves live in the sibling kernel modules.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

# -- shape constants (BM-independent; KIMI defaults, per-shape via compile args) --
MAX_M = 655360
NE = 385
K = 512  # gemm2 contraction = inter_dim (DEFAULT / KIMI)
N_OUT = 7168  # default gemm2 output dim = model_dim
BN = 256
BK = 256
kStages = 2
# e8m0 scale-layout K-independent stride.
kBS_stride_k0_dw = 64


# -- K-derived sizes (parametrized over the contraction dim K = inter_dim) -----
def k_half_for(k):
    return k // 2  # packed-fp4 bytes along K (KIMI: 256)


def k_tiles_total_for(k):
    return k // BK  # KIMI: 2


def kunroll_for(k):
    # streaming main-loop trip count: kUnroll = K_TILES_TOTAL - kStages.
    return k_tiles_total_for(k) - kStages


def kbs_c_k1_for(k):
    return (k // 32) // 4 // 2  # KIMI: 2


def kbs_stride_n0_dw_for(k):
    return kbs_c_k1_for(k) * 64  # KIMI: 128


def kas_c_k1_for(k):
    return (k // 32) // 4 // 2  # KIMI: 2


def kas_per_chunk_dw_for(k):
    return kas_c_k1_for(k) * 64  # KIMI: 128


# -- shape-parametrized sizes (NE/N_OUT vary per instance; N_OUT % 256 == 0) ----
def num_n_blocks_for(n_out):
    return n_out // 256


def kbs_per_expert_dw_for(n_out, k=K):
    return (n_out // 16 // 2) * kbs_stride_n0_dw_for(k)


def kmchunks(BM):
    return 1 if const_expr(BM == 16) else BM // 16


_PTR3 = "!llvm.ptr<3>"


def _raw(v):
    """Unwrap an fx value to a raw ir.Value for raw llvm/arith ops."""
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _udiv(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.divui(_raw(a), _raw(cc)))


def _lds_ptr3(base_i32, byte_off_i32):
    """ptr<3> = inttoptr(i64(base_i32 + byte_off_i32))."""
    addr_i64 = fx.Int64(base_i32 + byte_off_i32)
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(addr_i64))


def _lds_base_ptr3(lds_view):
    """One ptr<3> for the LDS base; offsets via GEP."""
    base_i32 = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(lds_view))
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(fx.Int64(base_i32)))


def _gep3(base_ptr, byte_off_i32):
    """getelementptr i8, base_ptr, byte_off_i32  (ptr<3>)."""
    return buffer_ops.get_element_ptr(
        base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8
    )


def _global_base_ptr1(addr_i64):
    """One ptr<1> base from a raw i64 device address (bare data_ptr() kernarg)."""
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr_i64)))


def _gep1(base_ptr, byte_off_i32):
    """getelementptr i8, base_ptr, byte_off_i32  (ptr<1>)."""
    return buffer_ops.get_element_ptr(
        base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8
    )


def _global_ptr1(arg, byte_off_i32):
    return _gep1(_global_base_ptr1(arg), byte_off_i32)


def _lds_swizzle_mask(row):
    """lds_swizzle_mask<ROW_BYTES=BK/2=128>(row): mask = (row & 14) << 3."""
    return (row & fx.Int32(14)) << fx.Int32(3)


def _atomic_bf16_epilog(
    lds_acc,
    accm,
    arg_out,
    arg_stids,
    arg_sweights,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
):
    _kMChunks = kmchunks(BM)
    M_REPS = BM // 8  # BM32: 4, BM16: 2
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_base_ptr3(lds_acc.get())

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    stids_base = _global_base_ptr1(arg_stids)
    sweights_base = _global_base_ptr1(arg_sweights)
    out_base = _global_base_ptr1(arg_out)

    # Prefetch sorted_token_ids / sorted_weights BEFORE the cshuffle stores and
    # both LDS barriers (invariant => freely hoistable), overlapping their global
    # latency with the store + barriers instead of exposing it in the atomic loop.
    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(
            llvm.load(
                T.i32, _gep1(stids_base, sorted_pos * fx.Int32(4)), invariant=True
            )
        )
        weight.append(
            llvm.load(
                T.f32, _gep1(sweights_base, sorted_pos * fx.Int32(4)), invariant=True
            )
        )

    # pre-store fence+barrier (HIP run_one __syncthreads() before the epilog).
    gpu.barrier()

    # write accm -> lds_acc cshuffle (scalar f32 stores, as HIP does)
    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            col = wave * fx.Int32(64) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))

    gpu.barrier()

    # read back + weighted atomic add (token_id / weight prefetched above)
    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            row_base_addr = (
                token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            )
            for s in range_constexpr(4):
                # adjacent ee=0,1 are contiguous -> one <2xf32> load (as HIP vectorizes)
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(
                    llvm.load(T.vec(2, T.f32), _gep3(lds_base, idx0 * fx.Int32(4)))
                )
                pk = Vec.from_elements(
                    [v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32
                ).to(fx.BFloat16)
                off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)  # bf16 byte off
                out_ptr = _gep1(out_base, off)
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    out_ptr,
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
