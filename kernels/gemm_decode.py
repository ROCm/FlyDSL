# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# gemm_decode: warp-per-scalar small-M GEMM for LLM autoregressive decode.
#
# C[M, N] = A[M, K] @ B[N, K]^T   (B row-major weight matrix)
#
# Optimizations applied:
#   KVEC=16     : wider loads (16 BF16 per lane)
#   NP=2        : A-reuse (2 output columns per wavefront)
#   K_BATCH=2   : split-K → 2× more wavefronts → better latency hiding
#                 uses FP32 atomic_add to scratch, then BF16 convert pass
#   XCD swizzle : L2 locality — consecutive wgids on same XCD share B data
#
# Reference: SILOTIGER-669, gemm_decode_kernel_reference.md §16

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import buffer_ops
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T, Vector as Vec
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _arith_dialect
from flydsl._mlir.dialects import llvm as _llvm

KVEC       = 16   # BF16 elements per lane per K-iteration
NP         = 2    # output columns per wavefront (A-reuse)
K_BATCH    = 2    # split-K shards
NUM_XCDS   = 8    # XCD chiplets on MI355X
CHUNK_SIZE = 8    # wgids per XCD per chunk for swizzle


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_ir(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _const_f32(val: float) -> ir.Value:
    return _arith_dialect.ConstantOp(
        ir.F32Type.get(),
        ir.FloatAttr.get(ir.F32Type.get(), val)
    ).result


def _add_f32(a: ir.Value, b) -> ir.Value:
    return _arith_dialect.AddFOp(lhs=a, rhs=_to_ir(b)).result


def dot2_f32_bf16(acc, a_i32, b_i32):
    return _llvm.inline_asm(
        ir.F32Type.get(),
        [acc, _to_ir(a_i32), _to_ir(b_i32)],
        "v_dot2_f32_bf16 $0, $2, $3, $1\ns_nop 2",
        "=v,0,v,v",
        has_side_effects=False,
    )


def pack_bf16x2(lo, hi):
    lo_i32 = ArithValue(ArithValue(_to_ir(lo)).bitcast(T.i16)).extui(T.i32)
    hi_i32 = ArithValue(ArithValue(_to_ir(hi)).bitcast(T.i16)).extui(T.i32)
    return ArithValue(lo_i32) | (ArithValue(hi_i32) << fx.Int32(16))


def wavefront_reduce_sum_f32(val, lane):
    for stage in range_constexpr(6):
        partner  = lane ^ fx.Int32(1 << stage)
        val_i32  = ArithValue(val).bitcast(T.i32)
        peer_i32 = fx.rocdl.ds_bpermute(T.i32, partner * fx.Int32(4), val_i32)
        val      = _add_f32(val, ArithValue(peer_i32).bitcast(T.f32))
    return val


def load_kvec(rsrc, base_elem):
    v0 = Vec(buffer_ops.buffer_load(rsrc, base_elem,                vec_width=4, dtype=T.bf16))
    v1 = Vec(buffer_ops.buffer_load(rsrc, base_elem + fx.Int32(4),  vec_width=4, dtype=T.bf16))
    v2 = Vec(buffer_ops.buffer_load(rsrc, base_elem + fx.Int32(8),  vec_width=4, dtype=T.bf16))
    v3 = Vec(buffer_ops.buffer_load(rsrc, base_elem + fx.Int32(12), vec_width=4, dtype=T.bf16))
    return v0, v1, v2, v3


def dot2_kvec(acc, av0, av1, av2, av3, bv0, bv1, bv2, bv3):
    acc = dot2_f32_bf16(acc, pack_bf16x2(av0[0], av0[1]), pack_bf16x2(bv0[0], bv0[1]))
    acc = dot2_f32_bf16(acc, pack_bf16x2(av0[2], av0[3]), pack_bf16x2(bv0[2], bv0[3]))
    acc = dot2_f32_bf16(acc, pack_bf16x2(av1[0], av1[1]), pack_bf16x2(bv1[0], bv1[1]))
    acc = dot2_f32_bf16(acc, pack_bf16x2(av1[2], av1[3]), pack_bf16x2(bv1[2], bv1[3]))
    acc = dot2_f32_bf16(acc, pack_bf16x2(av2[0], av2[1]), pack_bf16x2(bv2[0], bv2[1]))
    acc = dot2_f32_bf16(acc, pack_bf16x2(av2[2], av2[3]), pack_bf16x2(bv2[2], bv2[3]))
    acc = dot2_f32_bf16(acc, pack_bf16x2(av3[0], av3[1]), pack_bf16x2(bv3[0], bv3[1]))
    acc = dot2_f32_bf16(acc, pack_bf16x2(av3[2], av3[3]), pack_bf16x2(bv3[2], bv3[3]))
    return acc


def xcd_swizzle(wgid, num_wg):
    """XCD-aware workgroup remapping (gemm_decode_kernel_reference §10).

    Remaps so consecutive wgids land on the same XCD → B-matrix L2 reuse.
    Tail wgids past the aligned block pass through unchanged.
    """
    block    = fx.Int32(NUM_XCDS * CHUNK_SIZE)    # 64
    full_blk = (num_wg // (NUM_XCDS * CHUNK_SIZE)) * block

    xcd      = wgid % fx.Int32(NUM_XCDS)
    local    = wgid // fx.Int32(NUM_XCDS)
    chunk    = local // fx.Int32(CHUNK_SIZE)
    off      = local % fx.Int32(CHUNK_SIZE)
    swizzled = chunk * block + xcd * fx.Int32(CHUNK_SIZE) + off

    in_full  = wgid < full_blk
    return swizzled * in_full + wgid * (fx.Int32(1) - in_full)


# ── kernel 1: split-K GEMM → FP32 scratch (atomic add) ───────────────────────

@flyc.kernel
def gemm_decode_splitk_kernel(
    A:         fx.Tensor,      # [M, K]   BF16
    B:         fx.Tensor,      # [N, K]   BF16
    C:         fx.Tensor,      # [M, N]   BF16 output (pre-zeroed, bf16x2 atomic add on gfx950)
    K:         fx.Constexpr[int],
    N:         fx.Constexpr[int],
):
    lane   = gpu.thread_idx.x
    m      = gpu.block_idx.x
    k_id   = gpu.block_idx.z       # split-K shard 0..K_BATCH-1

    # XCD swizzle on the N-tile dimension (blockIdx.y)
    num_wg  = fx.Int32(N // NP)
    n_tile  = xcd_swizzle(gpu.block_idx.y, num_wg)
    n_base  = n_tile * fx.Int32(NP)

    rsrc_a = buffer_ops.create_buffer_resource(A)
    rsrc_b = buffer_ops.create_buffer_resource(B)
    rsrc_c = buffer_ops.create_buffer_resource(C)

    acc0 = _const_f32(0.0)
    acc1 = _const_f32(0.0)

    K_per_shard = K // K_BATCH
    kTileN      = 64 * KVEC
    num_iter    = K_per_shard // kTileN
    k_offset    = k_id * fx.Int32(K_per_shard)

    for i in range_constexpr(num_iter):
        k_elem = k_offset + fx.Int32(i * kTileN) + lane * fx.Int32(KVEC)

        a_elem = m * fx.Int32(K) + k_elem
        av0, av1, av2, av3 = load_kvec(rsrc_a, a_elem)

        b0_elem = n_base * fx.Int32(K) + k_elem
        bv0_0, bv1_0, bv2_0, bv3_0 = load_kvec(rsrc_b, b0_elem)

        b1_elem = (n_base + fx.Int32(1)) * fx.Int32(K) + k_elem
        bv0_1, bv1_1, bv2_1, bv3_1 = load_kvec(rsrc_b, b1_elem)

        acc0 = dot2_kvec(acc0, av0, av1, av2, av3, bv0_0, bv1_0, bv2_0, bv3_0)
        acc1 = dot2_kvec(acc1, av0, av1, av2, av3, bv0_1, bv1_1, bv2_1, bv3_1)

    acc0 = wavefront_reduce_sum_f32(acc0, lane)
    acc1 = wavefront_reduce_sum_f32(acc1, lane)

    if lane == fx.Int32(0):
        # gfx950: buffer_atomic_pk_add_bf16 — atomically adds bf16x2 (packed pair)
        # NP=2 means acc0 and acc1 are adjacent in C → pack them as bf16x2
        # Convert each FP32 accumulator to BF16 (truncate)
        def f32_to_bf16_i16(acc):
            return ArithValue(_to_ir(ArithValue(acc).bitcast(T.i32)).shrui(fx.Int32(16))).trunci(T.i16)

        bf16_0 = ArithValue(f32_to_bf16_i16(acc0)).extui(T.i32)
        bf16_1 = ArithValue(f32_to_bf16_i16(acc1)).extui(T.i32)
        # Pack as bf16x2: lo=acc0 bits[15:0], hi=acc1 bits[31:16]
        bf16x2_i32 = ArithValue(bf16_0) | (ArithValue(bf16_1) << fx.Int32(16))
        # Reinterpret packed i32 as vector<2xbf16> via llvm bitcast
        bf16x2_vec = _llvm.BitcastOp(
            ir.VectorType.get([2], T.bf16),
            _to_ir(bf16x2_i32)
        ).result

        # Byte offset: C[m, n_base] — the two adjacent BF16 elements form one i32 word
        zero_i32  = fx.Int32(0)
        c_byte    = (m * fx.Int32(N) + n_base) * fx.Int32(2)   # BF16 = 2 bytes each
        fx.rocdl.raw_ptr_buffer_atomic_fadd(
            _to_ir(bf16x2_vec), rsrc_c, _to_ir(c_byte), _to_ir(zero_i32), _to_ir(zero_i32)
        )


# ── JIT launcher ──────────────────────────────────────────────────────────────

@flyc.jit
def gemm_decode_bf16(
    A:  fx.Tensor,
    B:  fx.Tensor,
    C:  fx.Tensor,          # [M, N]  BF16 (pre-zeroed, bf16x2 atomic add)
    M:  fx.Constexpr[int],
    N:  fx.Constexpr[int],
    K:  fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    # Single kernel: split-K shards atomic-add BF16x2 directly to C
    gemm_decode_splitk_kernel(A, B, C, K, N).launch(
        grid=(M, N // NP, K_BATCH),
        block=(64, 1, 1),
        stream=stream,
    )


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    # K divisible by 64*KVEC*K_BATCH = 2048
    M, N, K = 1, 64, 2048

    A = torch.randn(M, K, dtype=torch.bfloat16).cuda()
    B = torch.randn(N, K, dtype=torch.bfloat16).cuda()
    C = torch.zeros(M, N, dtype=torch.bfloat16).cuda()   # pre-zeroed!

    stream = torch.cuda.Stream()
    gemm_decode_bf16(A, B, C, M, N, K, stream=stream)
    torch.cuda.synchronize()

    ref = (A.float() @ B.float().T).bfloat16()
    ok  = torch.allclose(C, ref, atol=1.0, rtol=0.1)
    print(f"Result correct: {ok}")
    if not ok:
        print(f"Max diff: {(C.float()-ref.float()).abs().max().item():.4f}")
    print(f"C[:4]:   {C[0, :4]}")
    print(f"ref[:4]: {ref[0, :4]}")
