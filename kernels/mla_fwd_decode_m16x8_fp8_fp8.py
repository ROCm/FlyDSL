# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MLA decode kernel (nhead=128, fp8 Q, fp8 KV, bf16 output).

Transplanted from csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h128_fp8_fp8.cuh.

Architecture: 8 warps / 512 threads, persistent-thread dispatch.
Per work item: load Q -> iterate KV tiles (BLOCK_N=32) -> QK GEMM (nope+rope)
-> online softmax -> PV GEMM -> output (final bf16 or split f32 + LSE).

NOTE: Do NOT use ``from __future__ import annotations`` here -- it breaks
``fx.Constexpr`` detection in the FlyDSL AST rewriter.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, memref
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator

# ---------------------------------------------------------------------------
# Compile-time constants (mirroring HkMlaDecodeFwdTraits)
# ---------------------------------------------------------------------------
NUM_QO_HEADS: int = 128
NUM_KV_HEADS: int = 1
KV_LORA_RANK: int = 512
QK_NOPE_HEAD_DIM: int = KV_LORA_RANK  # 512
QK_ROPE_HEAD_DIM: int = 64
QK_HEAD_DIM: int = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM: int = KV_LORA_RANK  # 512
PAGE_SIZE: int = 1
NUM_WARPS: int = 8
WARP_SIZE: int = 64
NUM_THREADS: int = NUM_WARPS * WARP_SIZE  # 512
BLOCK_M: int = 128  # == NUM_QO_HEADS
BLOCK_N: int = 32
BLOCK_K: int = 32
TILE_M: int = BLOCK_M // NUM_WARPS  # 16
OCCUPANCY: int = 1

SIZE_MLA_WORK_INFO_IN_DW: int = 8
LOG2E: float = 1.4426950408889634

# ---------------------------------------------------------------------------
# KvManagerV2 LDS layout constants
# ---------------------------------------------------------------------------
# KV tile: 32 rows x 576 cols (fp8), split into 9 blocks of 64 cols each.
# Each block: 8 sub-blocks (one per warp) of 4 rows x 64 cols + 2 DW padding.
KV_NUM_COLS: int = 64
KV_NUM_BLOCKS: int = QK_HEAD_DIM // KV_NUM_COLS  # 576 / 64 = 9
KV_ROWS_PER_SUB: int = BLOCK_N // NUM_WARPS  # 32 / 8 = 4
KV_BYTES_PER_ROW: int = KV_NUM_COLS  # 64 * 1 (fp8)
KV_PAD_DW: int = 2
KV_SUB_BYTES: int = KV_ROWS_PER_SUB * KV_BYTES_PER_ROW + KV_PAD_DW * 4  # 264
KV_NUM_SUBS: int = BLOCK_N // KV_ROWS_PER_SUB  # 8
KV_BLOCK_BYTES: int = KV_SUB_BYTES * KV_NUM_SUBS  # 2112
SZ_LDS_KV: int = KV_BLOCK_BYTES * KV_NUM_BLOCKS  # 2112 * 9 = 19008

# ---------------------------------------------------------------------------
# VtManagerV1 LDS layout constants
# ---------------------------------------------------------------------------
VT_ROWS_PER_THR: int = 4
VT_COLS_PER_THR: int = 8
VT_ELEMS_PER_BLK: int = VT_ROWS_PER_THR * VT_COLS_PER_THR  # 32
VT_BLKS_PER_ROW: int = V_HEAD_DIM // VT_COLS_PER_THR  # 64
VT_BLKS_PER_ROW_PAD: int = VT_BLKS_PER_ROW + 2  # 66
VT_NUM_SUB_BLKS: int = 8
SZ_LDS_VT: int = VT_NUM_SUB_BLKS * (
    (BLOCK_N // VT_NUM_SUB_BLKS) * V_HEAD_DIM + 16 * 4
)  # 8 * (4*512 + 64) = 16896

# ---------------------------------------------------------------------------
# QManagerV3 LDS layout constants (per-warp staging for VRAM->LDS->GPR)
# ---------------------------------------------------------------------------
Q_ELEM_PER_ROW: int = 64
Q_ELEM_PER_COL: int = 16
Q_PAD_BYTES_PER_2ROWS: int = 8  # 2 DW
Q_BYTES_PER_2ROWS: int = Q_ELEM_PER_ROW * 2 + Q_PAD_BYTES_PER_2ROWS  # 136
SZ_LDS_Q_PER_WARP: int = Q_ELEM_PER_COL // 2 * Q_BYTES_PER_2ROWS  # 1088
SZ_LDS_Q: int = NUM_WARPS * SZ_LDS_Q_PER_WARP  # 8704

# ---------------------------------------------------------------------------
# OManager16bitsV2 (bf16 output via LDS reshape)
# ---------------------------------------------------------------------------
O16_NUM_ROWS: int = 16
O16_NUM_COLS: int = 32
O16_PAD_ELEM_PER_2ROWS: int = 4  # padded 2-row stride in bf16 elements
O16_ELEM_PER_PAD_2ROWS: int = 2 * O16_NUM_COLS + O16_PAD_ELEM_PER_2ROWS  # 68
O16_LDS_PER_WARP: int = (O16_NUM_ROWS // 2) * O16_ELEM_PER_PAD_2ROWS * 2  # 1088
SZ_LDS_O16: int = NUM_WARPS * O16_LDS_PER_WARP  # 8704  (reuses p_lds_kv region)

# ---------------------------------------------------------------------------
# OManager32bitsV2 (f32 split output via LDS reshape)
# ---------------------------------------------------------------------------
O32_NUM_ROWS: int = 16
O32_NUM_COLS: int = 32
O32_PAD_ELEM_PER_ROW: int = 4
O32_ELEM_PER_PAD_ROW: int = O32_NUM_COLS + O32_PAD_ELEM_PER_ROW  # 36
O32_LDS_PER_WARP: int = O32_NUM_ROWS * O32_ELEM_PER_PAD_ROW * 4  # 2304
SZ_LDS_O32: int = NUM_WARPS * O32_LDS_PER_WARP  # 18432

# Overall LDS layout (byte offsets):
#   [0, SZ_LDS_VT) = Vt staging buffer
#   [SZ_LDS_VT, SZ_LDS_VT + SZ_LDS_Q) = Q staging buffer
#   [SZ_LDS_VT + SZ_LDS_Q, +SZ_LDS_KV) = KV double-buffer 0
#   [SZ_LDS_VT + SZ_LDS_Q + SZ_LDS_KV, +SZ_LDS_KV) = KV double-buffer 1
# Output reuses the KV double-buffer 0 region.
P_LDS_VT: int = 0
P_LDS_Q: int = SZ_LDS_VT  # 16896
P_LDS_KV_0: int = P_LDS_Q + SZ_LDS_Q  # 25600
P_LDS_KV_1: int = P_LDS_KV_0 + SZ_LDS_KV  # 44608
TOTAL_LDS_BYTES: int = P_LDS_KV_1 + SZ_LDS_KV  # 63616

assert (
    max(SZ_LDS_O16, SZ_LDS_O32) <= SZ_LDS_KV
), "Output LDS must fit in one KV buffer region"

# ---------------------------------------------------------------------------
# MFMA tile constants
# ---------------------------------------------------------------------------
MFMA_M: int = 16
MFMA_N: int = 16
MFMA_K: int = 32  # mfma_f32_16x16x32_fp8_fp8
MFMA_ELEM_PER_THR: int = MFMA_M * MFMA_K // WARP_SIZE  # 8

# Number of QK sub-tile iterations
NUM_NOPE_ITERS: int = QK_NOPE_HEAD_DIM // (MFMA_K * 2)  # 512/64 = 8
NUM_ROPE_ITERS: int = QK_ROPE_HEAD_DIM // (MFMA_K * 2)  # 64/64 = 1
NUM_PV_ITERS: int = V_HEAD_DIM // (MFMA_N * 2)  # 512/32 = 16


# ---------------------------------------------------------------------------
# Utility helpers (ported from FlyDSL/kernels/mla_decode_fp8.py)
# ---------------------------------------------------------------------------


def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    """Encode s_waitcnt bitfield for CDNA3 (gfx94x)."""
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


def _barrier(vmcnt=63, lgkmcnt=63):
    """Emit s_waitcnt + s_barrier via inline asm."""
    parts = []
    needs_waitcnt = vmcnt < 63 or lgkmcnt < 63
    if needs_waitcnt:
        wc = []
        if vmcnt < 63:
            wc.append(f"vmcnt({vmcnt})")
        if lgkmcnt < 63:
            wc.append(f"lgkmcnt({lgkmcnt})")
        parts.append("s_waitcnt " + " ".join(wc))
    parts.append("s_barrier")
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="\n".join(parts),
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )


_LDS_PTR_TYPE = None


def _inttoptr_lds(i64_val):
    """Convert i64 scalar to !llvm.ptr<3> (LDS pointer)."""
    global _LDS_PTR_TYPE
    if _LDS_PTR_TYPE is None:
        _LDS_PTR_TYPE = ir.Type.parse("!llvm.ptr<3>")
    return llvm.inttoptr(_LDS_PTR_TYPE, i64_val)


_gep = buffer_ops.get_element_ptr


def _lds_load(byte_addr_index, vec_type, static_byte_offset=0):
    """LDS load via raw llvm.LoadOp on an LDS pointer (addr space 3)."""
    addr_i64 = _raw(fx.Int64(byte_addr_index))
    lds_ptr = _inttoptr_lds(addr_i64)
    if static_byte_offset != 0:
        lds_ptr = _gep(lds_ptr, static_byte_offset=static_byte_offset)
    return llvm.LoadOp(vec_type, lds_ptr, alignment=16, nontemporal=True).result


def _lds_load_volatile(base_i32, vec_type, byte_offset=0):
    """Volatile LDS load forcing ds_read_b64/b32 with immediate offset.

    Unlike _lds_load, uses volatile to prevent LLVM from merging adjacent
    loads into ds_read2 variants (which have limited 8-bit offsets).
    LLVM still tracks these as LDS loads for lgkmcnt.
    Input: base_i32 must be an i32 ir.Value (LDS byte address).
    """
    addr_i64 = _raw(ArithValue(base_i32).extui(T.i64))
    lds_ptr = _inttoptr_lds(addr_i64)
    if byte_offset != 0:
        lds_ptr = _gep(lds_ptr, static_byte_offset=byte_offset)
    return llvm.LoadOp(vec_type, lds_ptr, alignment=8, volatile_=True).result


def _lds_ptr_from_i32(addr_i32, byte_offset=0):
    """Build an LDS pointer (ptr<3>) from an i32 byte address + optional static offset."""
    ptr = _inttoptr_lds(_raw(ArithValue(addr_i32).extui(T.i64)))
    if byte_offset != 0:
        ptr = _gep(ptr, static_byte_offset=byte_offset)
    return ptr


def _i32(value):
    """Cast index/ArithValue to i32.  No-op if already i32."""
    raw = _raw(value) if not isinstance(value, ir.Value) else value
    if raw.type == T.i32:
        return raw
    return _raw(fx.Int32(raw))


def _fast_exp2(val):
    """Bare v_exp_f32 via rocdl.exp2 -- no range reduction."""
    return rocdl.exp2(T.f32, _raw(val))


def _to_mlir(val):
    """Convert Python int/float, ArithValue, or ir.Value to raw MLIR Value."""
    if isinstance(val, int):
        return _raw(fx.Int32(val))
    if isinstance(val, float):
        return _raw(fx.Float32(val))
    return _raw(val) if not isinstance(val, ir.Value) else val


def _pack_i32x2(lo, hi):
    """Pack two i32 values into a single i64: lo | (hi << 32)."""
    return _raw(ArithValue(lo).extui(T.i64) | (ArithValue(hi).extui(T.i64) << 32))


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
def kn_mla_fwd_decode_m16x8_fp8_fp8(
    # --- inputs ---
    query: fx.Tensor,  # [num_seqs * num_heads, qk_head_dim]  (fp8)
    kv_buffer: fx.Tensor,  # [num_pages, qk_head_dim]  (fp8)
    kv_page_indices: fx.Tensor,  # [num_page_used]  (i32)
    # --- metadata ---
    work_indptr: fx.Tensor,  # [num_workers + 1]  (i32)
    work_info_set: fx.Tensor,  # [num_work_items * 8]  (i32)
    # --- outputs ---
    final_output: fx.Tensor,  # [num_seqs * num_heads, v_head_dim]  (bf16)
    split_output: fx.Tensor,  # [num_partial_slots * num_heads, v_head_dim]  (f32)
    split_lse: fx.Tensor,  # [num_partial_slots * num_heads]  (f32)
    # --- parameters ---
    softmax_scale: fx.Float32,
):
    """MLA decode forward kernel (nhead=128, fp8/fp8 -> bf16).

    Persistent-thread kernel: each workgroup picks up work items
    from ``work_indptr`` / ``work_info_set`` and processes them sequentially.
    """
    _STUB_EARLY_RETURN = False  # Set True to skip all kernel body for testing launch
    if const_expr(_STUB_EARLY_RETURN):
        return

    # ---- Types ----
    fm_fast = arith.FastMathFlags.fast
    # fastmath without ninf: safe for operations that may encounter -inf
    # (boundary masking sets OOB attention scores to -inf)
    fm_no_inf = (
        arith.FastMathFlags.nnan
        | arith.FastMathFlags.nsz
        | arith.FastMathFlags.arcp
        | arith.FastMathFlags.contract
        | arith.FastMathFlags.afn
        | arith.FastMathFlags.reassoc
    )

    def _mfma_fp8(result_type, operands, **kw):
        return rocdl.mfma_f32_16x16x32_fp8_fp8(result_type, operands, **kw)

    # ---- LDS setup ----
    arch = get_hip_arch()
    lds_allocator = SmemAllocator(None, arch=arch)
    lds_allocator.ptr = TOTAL_LDS_BYTES  # reserve LDS bytes

    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        lds_allocator.finalize()

    lds_buffer = lds_allocator.get_base()
    lds_base_idx = memref.extract_aligned_pointer_as_index(lds_buffer)

    # ---- V^T transpose perm constants ----
    c_perm0 = fx.Int32(0x05010400)
    c_perm1 = fx.Int32(0x07030602)
    c_perm2 = fx.Int32(0x05040100)
    c_perm3 = fx.Int32(0x07060302)

    def _vt_perm(src_hi, src_lo, sel):
        return llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.perm",
            [_raw(src_hi), _raw(src_lo), _raw(sel)],
            [],
            [],
        )

    # ---- Constants ----
    c_neg_inf = fx.Float32(float("-inf"))
    c_zero_f32 = fx.Float32(0.0)
    c_one_f32 = fx.Float32(1.0)
    c_zero_i32 = fx.Int32(0)
    c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)
    c_log2e = fx.Float32(LOG2E)
    c_inv_log2e = fx.Float32(1.0 / LOG2E)

    # ---- Buffer resources ----
    query_rsrc = buffer_ops.create_buffer_resource(query)
    kv_rsrc = buffer_ops.create_buffer_resource(kv_buffer)
    kv_page_indices_rsrc = buffer_ops.create_buffer_resource(kv_page_indices)
    work_indptr_rsrc = buffer_ops.create_buffer_resource(work_indptr)
    work_info_set_rsrc = buffer_ops.create_buffer_resource(work_info_set)
    final_output_rsrc = buffer_ops.create_buffer_resource(final_output)
    split_output_rsrc = buffer_ops.create_buffer_resource(split_output)
    split_lse_rsrc = buffer_ops.create_buffer_resource(split_lse)

    # ---- Thread indices ----
    worker_idx = gpu.block_idx.x
    tid = gpu.thread_id("x")
    warp_idx = tid / WARP_SIZE
    lane_idx = tid % WARP_SIZE
    warp_idx_i32 = rocdl.readfirstlane(T.i32, _i32(warp_idx))
    lane_idx_i32 = _i32(lane_idx)

    # ---- Work range ----
    work_range = buffer_ops.buffer_load(
        work_indptr_rsrc, _i32(worker_idx), vec_width=2, dtype=T.i32
    )
    work_range_vec = Vec(work_range)
    work_start_i32 = rocdl.readfirstlane(T.i32, work_range_vec[0])
    work_end_i32 = rocdl.readfirstlane(T.i32, work_range_vec[1])
    work_start_idx = fx.Index(work_start_i32)
    work_end_idx = fx.Index(work_end_i32)

    # ---- KvManagerV2 thread-to-data mapping ----
    # Each warp takes 4 rows: warp w -> rows {w*2, w*2+1, w*2+16, w*2+17}
    # lane mapping: (lane/32)*16 + (lane/16)%2 + warp*2
    kv_ld_row_base_i32 = _i32(
        lane_idx / 32 * 16 + (lane_idx / 16) % 2 + warp_idx * 2
    )
    kv_ld_col_base_i32 = _i32((lane_idx % 16) * 4)

    # ---- Helper: resolve KV page index -> physical row ----
    def _get_kv_ld_row(kv_tile_start_i32, kv_tile_end_i32, check_boundary):
        """Resolve physical KV row for this thread's assigned row.

        For OOB rows (row >= kv_end), returns -1 WITHOUT issuing a
        buffer_load -- avoids reading garbage from kv_page_indices.
        """
        row_idx_i32 = _i32(ArithValue(kv_ld_row_base_i32) + kv_tile_start_i32)
        if const_expr(check_boundary):
            phys_row = fx.Int32(-1)
            if ArithValue(row_idx_i32) < ArithValue(kv_tile_end_i32):
                phys_row = buffer_ops.buffer_load(
                    kv_page_indices_rsrc, row_idx_i32, vec_width=1, dtype=T.i32
                )
            return _raw(phys_row)
        else:
            return buffer_ops.buffer_load(
                kv_page_indices_rsrc, row_idx_i32, vec_width=1, dtype=T.i32
            )

    # ---- Helper: async_load_k_tile (VRAM->LDS via buffer_load_dword_lds) ----
    def _async_load_k_tile(
        p_lds_kv_warp_i32, row_i32, col_base_i32, block_idx_const, check_boundary=False
    ):
        """Load one 32x64 block of KV data from VRAM to LDS.

        block_idx_const: Python int [0..8], which 64-col block.
        """
        lds_warp_offset = block_idx_const * KV_BLOCK_BYTES
        # p_lds_kv_warp points to warp's sub-block start.
        # Actual LDS target: p_lds_kv_warp + block*KV_BLOCK_BYTES - block*64
        lds_adjust = lds_warp_offset - block_idx_const * KV_NUM_COLS
        lds_base_i32 = _i32(ArithValue(p_lds_kv_warp_i32) + lds_adjust)

        def _emit_vram_to_lds():
            voff = _i32(ArithValue(row_i32) * QK_HEAD_DIM + col_base_i32)
            rocdl.buffer_load_to_lds(
                kv_rsrc, _lds_ptr_from_i32(lds_base_i32), voff,
                offset=block_idx_const * KV_NUM_COLS,
            )

        if const_expr(check_boundary):
            is_oob = ArithValue(row_i32) == -1
            if is_oob:
                # Write zero via ds_write_b32 at lane's position
                lds_addr = _i32(
                    ArithValue(lds_base_i32)
                    + block_idx_const * KV_NUM_COLS
                    + ArithValue(lane_idx_i32) * 4
                )
                lds_ptr = _lds_ptr_from_i32(lds_addr)
                llvm.StoreOp(_raw(c_zero_i32), lds_ptr, alignment=4)
            else:
                _emit_vram_to_lds()
        else:
            _emit_vram_to_lds()

    def _async_load_kv_all(
        p_lds_kv_warp_i32, row_i32, col_base_i32, check_boundary=False
    ):
        """Load all 9 blocks of a KV tile."""
        for blk in range_constexpr(KV_NUM_BLOCKS):
            _async_load_k_tile(
                p_lds_kv_warp_i32,
                row_i32,
                col_base_i32,
                blk,
                check_boundary=check_boundary,
            )

    # ---- Inline-asm prefetch: fully opaque to LLVM waitcnt analysis ----
    def _prefetch_k_tile_asm(
        p_lds_kv_warp_i32,
        row_i32,
        col_base_i32,
        block_idx_const,
        check_boundary=True,
    ):
        """Prefetch one KV block via inline asm buffer_load_dword lds.

        Uses inline asm for BOTH the normal load AND the OOB zero-write
        so LLVM sees no LDS operations and won't insert spurious
        s_waitcnt vmcnt(0) before subsequent ds_read ops.

        check_boundary: controls OOB row==-1 check.
          - False (Python): skips check entirely -- caller guarantees valid row.
          - True (Python): always emits a branch on row==-1.
          - ir.Value (i1): emits a branch on check_boundary AND row==-1,
            allowing runtime bypass.
        """
        lds_adjust = block_idx_const * KV_BLOCK_BYTES - block_idx_const * KV_NUM_COLS
        lds_base_i32 = _i32(ArithValue(p_lds_kv_warp_i32) + lds_adjust)

        def _emit_normal_load():
            voff = _i32(ArithValue(row_i32) * QK_HEAD_DIM + col_base_i32)
            col_off_imm = block_idx_const * KV_NUM_COLS
            asm_str = (
                "s_mov_b32 m0, $0\n"
                "s_nop 0\n"
                f"buffer_load_dword $1, $2, 0 offen offset:{col_off_imm} lds"
            )
            llvm.InlineAsmOp(
                res=None,
                operands_=[lds_base_i32, voff, _raw(kv_rsrc)],
                asm_string=asm_str,
                constraints="s,v,s",
                has_side_effects=True,
                is_align_stack=False,
            )

        if const_expr(check_boundary is False):
            _emit_normal_load()
        else:
            # Build OOB condition: row == -1
            is_oob = ArithValue(row_i32) == -1
            # If check_boundary is a runtime i1, AND it in
            if const_expr(check_boundary is not True):
                is_oob = _raw(ArithValue(check_boundary) & ArithValue(is_oob))

            if is_oob:
                # OOB: write zero to LDS via inline asm ds_write_b32
                lds_zero_addr = _i32(
                    ArithValue(lds_base_i32)
                    + block_idx_const * KV_NUM_COLS
                    + ArithValue(lane_idx_i32) * 4
                )
                llvm.InlineAsmOp(
                    res=None,
                    operands_=[lds_zero_addr, _raw(c_zero_i32)],
                    asm_string="ds_write_b32 $0, $1",
                    constraints="v,v",
                    has_side_effects=True,
                    is_align_stack=False,
                )
            else:
                _emit_normal_load()

    # ---- K LDS lane base pointer (computed once, shared across all K loads) ----
    # Per-lane dynamic part of the K LDS address, stored as an LDS pointer.
    # All K loads use this as base + GEP(fixed_offset), so LLVM can fold
    # the fixed_offset into ds_read's 16-bit immediate offset field.
    k_row_in_mfma = lane_idx % MFMA_M
    k_row_phy = (k_row_in_mfma / 2) * 4 + k_row_in_mfma % 2
    k_col_in_lane = (lane_idx / MFMA_M) * MFMA_ELEM_PER_THR
    k_lds_lane_offset = (
        (k_row_phy / 4) * KV_SUB_BYTES
        + (k_row_phy % 4) * KV_BYTES_PER_ROW
        + (k_col_in_lane % KV_NUM_COLS)
    )

    # ---- Vt LDS lane base offset (computed once, shared across all Vt loads) ----
    vt_row_blk = lane_idx / 16
    vt_col_blk = (lane_idx % 16) / VT_COLS_PER_THR
    vt_row_inblk = lane_idx % VT_ROWS_PER_THR
    vt_col_inblk = ((lane_idx % 8) / VT_ROWS_PER_THR) * VT_ROWS_PER_THR
    vt_block_offset = (vt_row_blk * VT_BLKS_PER_ROW_PAD + vt_col_blk) * VT_ELEMS_PER_BLK
    vt_inblock_offset = vt_row_inblk * VT_COLS_PER_THR + vt_col_inblk
    vt_lds_lane_offset = vt_block_offset + vt_inblock_offset

    # ---- Helper: load K sub-tile from LDS (16x32 for MFMA) ----
    def _load_k_from_lds(k_base_i32, row_offset, col_offset):
        """Read 16x32 K sub-tile from LDS -> i64 for MFMA.

        row_offset: 0 or 16 (which half of BLOCK_N=32)
        col_offset: column offset in elements (multiple of 32)

        KvManagerV2 LDS address formula:
          row_phy = (row/2)*4 + (row%2)  where row = lane_idx % 16
          p = p_lds_kv + (row_phy/4)*KV_SUB_BYTES + (row_phy%4)*KV_BYTES_PER_ROW
              + (col%64)*sizeof(kv_t) + (col/64)*KV_BLOCK_BYTES
          fixed_offset = (row_offset/16)*2*KV_BYTES_PER_ROW
                       + (col_offset%64)*sizeof(kv_t)
                       + (col_offset/64)*KV_BLOCK_BYTES

        NOTE: The fixed_offset is passed via static_byte_offset so LLVM
        can potentially fold it into ds_read's immediate. Currently LLVM
        lowers this to ds_read2_b64 due to inttoptr; a proper fix needs
        FlyDSL infrastructure changes to emit ds_read_b64 with large offsets.
        """
        # Fixed part: compile-time constant byte offset
        fixed_offset = (
            (row_offset // 16) * 2 * KV_BYTES_PER_ROW
            + (col_offset % KV_NUM_COLS)
            + (col_offset // KV_NUM_COLS) * KV_BLOCK_BYTES
        )

        # ds_read_b64 with immediate offset (volatile prevents ds_read2 merge)
        data = _lds_load_volatile(k_base_i32, T.i64, byte_offset=fixed_offset)
        return data

    # ---- Helper: load V from KV LDS (un-transposed) ----
    def _load_v_from_lds(p_lds_kv_base_idx, warp_idx_val, lane_idx_val):
        """Load un-transposed V: each warp reads 16x128 region.

        KvManagerV2::load_v_to_gpr pattern:
          row = (warp%2)*16 + lane/16*4
          row_phy = ((row%16)/2)*4 + 2*(row/16) + (row%2)
          col = (lane%16)*8 + (warp/2)*128
        Returns 8 i32 values.
        """
        row = (warp_idx_val % 2) * 16 + (lane_idx_val / 16) * 4
        row_mod16 = row % 16
        row_phy = (
            (row_mod16 / 2) * 4
            + 2 * (row / 16)
            + row % 2
        )
        col = (lane_idx_val % 16) * 8 + (warp_idx_val / 2) * 128

        lds_v_offset = (
            (row_phy / 4) * KV_SUB_BYTES
            + (row_phy % 4) * KV_BYTES_PER_ROW
            + (col / KV_NUM_COLS) * KV_BLOCK_BYTES
            + (col % KV_NUM_COLS)
        )

        lds_addr = p_lds_kv_base_idx + lds_v_offset

        # 4 x ds_read_b64: load 8 dwords at strides matching KvManagerV2
        v_vals = []
        for pass_idx in range_constexpr(4):
            if const_expr(pass_idx == 0):
                off = 0
            elif const_expr(pass_idx == 1):
                off = KV_BYTES_PER_ROW
            elif const_expr(pass_idx == 2):
                off = KV_SUB_BYTES
            else:
                off = KV_SUB_BYTES + KV_BYTES_PER_ROW
            data = _lds_load(
                lds_addr,
                T.i32x2,
                static_byte_offset=off,
            )
            data_vec = Vec(data)
            v_vals.append(data_vec[0])
            v_vals.append(data_vec[1])
        return v_vals  # 8 i32 values

    # ---- Helper: transpose V in-register ----
    def _transpose_v(v8):
        """12x v_perm_b32 to transpose 4x8 fp8 block.

        Ported from VtManagerV1::transpose_v.
        Input:  v8[0..7] in row-major 4x8 layout
        Output: v8[0..7] in transposed layout for Vt storage
        """
        # Phase 1: perm_0 (c_perm0=0x05010400) and perm_3 (c_perm1=0x07030602)
        t0_0 = _vt_perm(v8[2], v8[0], c_perm0)
        t2_0 = _vt_perm(v8[2], v8[0], c_perm1)
        t0_1 = _vt_perm(v8[3], v8[1], c_perm0)
        t2_1 = _vt_perm(v8[3], v8[1], c_perm1)

        t1_0 = _vt_perm(v8[6], v8[4], c_perm0)
        t3_0 = _vt_perm(v8[6], v8[4], c_perm1)
        t1_1 = _vt_perm(v8[7], v8[5], c_perm0)
        t3_1 = _vt_perm(v8[7], v8[5], c_perm1)

        # Phase 2: perm_1 (c_perm2=0x05040100) and perm_2 (c_perm3=0x07060302)
        # Output order: r0_0, r0_1, r1_0, r1_1, r2_0, r2_1, r3_0, r3_1
        r = [None] * 8
        r[0] = _vt_perm(t1_0, t0_0, c_perm2)  # r0_0
        r[1] = _vt_perm(t1_1, t0_1, c_perm2)  # r0_1
        r[2] = _vt_perm(t1_0, t0_0, c_perm3)  # r1_0
        r[3] = _vt_perm(t1_1, t0_1, c_perm3)  # r1_1
        r[4] = _vt_perm(t3_0, t2_0, c_perm2)  # r2_0
        r[5] = _vt_perm(t3_1, t2_1, c_perm2)  # r2_1
        r[6] = _vt_perm(t3_0, t2_0, c_perm3)  # r3_0
        r[7] = _vt_perm(t3_1, t2_1, c_perm3)  # r3_1
        return r

    # ---- Helper: store transposed V to Vt LDS ----
    def _store_vt_to_lds(vt_lds_base_idx, warp_idx_val, lane_idx_val, vt8):
        """VtManagerV1::store_transposed_v_to_lds.

        4x8 block-wise row-major layout, no padding between rows/cols.
        row_blk = (warp%2)*4 + lane/16
        col_blk = (lane%16) + (warp/2)*16
        block_offset = (row_blk * VT_BLKS_PER_ROW_PAD + col_blk) * VT_ELEMS_PER_BLK
        """
        row_blk = (warp_idx_val % 2) * 4 + lane_idx_val / 16
        col_blk = (lane_idx_val % 16) + (warp_idx_val / 2) * 16
        block_offset = (row_blk * VT_BLKS_PER_ROW_PAD + col_blk) * VT_ELEMS_PER_BLK
        lds_vt_addr = vt_lds_base_idx + block_offset

        # ds_write_b128 x 2 (4 dwords each = 32 fp8)
        lo_packed = Vec.from_elements(vt8[0:4], fx.Int32)
        lo_i8 = Vec(lo_packed).bitcast(fx.Int8)
        lo_i8.store(lds_buffer, [lds_vt_addr])

        hi_packed = Vec.from_elements(vt8[4:8], fx.Int32)
        hi_i8 = Vec(hi_packed).bitcast(fx.Int8)
        hi_i8.store(lds_buffer, [lds_vt_addr + 16])

    # ---- Helper: load transposed V from Vt LDS ----
    def _load_vt_from_lds(vt_base_i32, col_offset):
        """VtManagerV1::load_transposed_v_to_gpr.

        Each warp reads 32x16 block from Vt LDS. Returns 2 i32 via ds_read_b32.
        vt_base_i32: i32 LDS byte address with lane offset pre-baked.
        col_offset: Python int, multiple of 16, in [0, 512).

        Lane offset pre-computed in vt_lds_lane_offset (top level).
        Only col_offset contributes a fixed immediate offset here.
        offset_tl_bl = 4 * VT_BLKS_PER_ROW_PAD * VT_ELEMS_PER_BLK = 8448
        """
        fixed_col_blk = col_offset // VT_COLS_PER_THR
        fixed_block_offset = fixed_col_blk * VT_ELEMS_PER_BLK
        offset_tl_bl = 4 * VT_BLKS_PER_ROW_PAD * VT_ELEMS_PER_BLK  # 8448

        # ds_read_b32 x 2 with immediate offsets (volatile prevents ds_read2 merge)
        v0 = _lds_load_volatile(vt_base_i32, T.i32, byte_offset=fixed_block_offset)
        v1 = _lds_load_volatile(
            vt_base_i32, T.i32, byte_offset=fixed_block_offset + offset_tl_bl
        )
        return v0, v1

    # ---- Helper: warp reduce (butterfly XOR) ----
    def _shfl_xor_f32(val_f32, offset, width=WARP_SIZE):
        """XOR shuffle for f32 via bitcast to i32 and back."""
        val_i32 = _raw(ArithValue(val_f32).bitcast(T.i32))
        peer_i32 = ArithValue(val_i32).shuffle_xor(offset, width)
        return _raw(ArithValue(peer_i32).bitcast(T.f32))

    def _warp_reduce_max_16(val):
        """Butterfly max reduce across MFMA column groups (strides 32, 16)."""
        w = _to_mlir(val)
        for sh in [32, 16]:
            w = arith.MaximumFOp(w, _shfl_xor_f32(w, sh), fastmath=fm_no_inf).result
        return w

    def _warp_reduce_add_16(val):
        """Butterfly sum reduce across MFMA column groups (strides 32, 16)."""
        w = _to_mlir(val)
        for sh in [32, 16]:
            w = ArithValue(w).addf(_shfl_xor_f32(w, sh), fastmath=fm_fast)
        return w

    # ---- Helper: Q loading (QManagerV3) ----
    def _load_q_to_regs(qo_start_i32):
        """Load Q from VRAM to registers via LDS staging.

        QManagerV3: each warp loads 16x64 per pass, 9 passes total.
        VRAM -> LDS (ds_write_b128), then LDS -> register (ds_read_b64).
        Returns (q_nope_regs, q_rope_regs):
          q_nope_regs: list of 16 v2i64 (16 sub-tiles x 32 cols each)
          q_rope_regs: list of 2 v2i64 (2 sub-tiles x 32 cols each)
        """
        p_lds_q_warp = (
            lds_base_idx
            + P_LDS_Q
            + warp_idx * SZ_LDS_Q_PER_WARP
        )

        # VRAM addressing: row = lane/4, col = (lane%4)*16
        # s_offset = warp * 16 * QK_HEAD_DIM * sizeof(fp8)
        # v_offset = (row * QK_HEAD_DIM + col) * sizeof(fp8)
        # s_offset = warp * 16 * QK_HEAD_DIM + qo_start * NUM_QO_HEADS * QK_HEAD_DIM
        s_offset_i32 = _i32(
            ArithValue(warp_idx_i32) * (16 * QK_HEAD_DIM)
            + ArithValue(qo_start_i32) * (NUM_QO_HEADS * QK_HEAD_DIM)
        )

        row = lane_idx / 4
        col = (lane_idx % 4) * 16
        v_offset_i32 = _i32(row * QK_HEAD_DIM + col)

        # LDS store layout (QManagerV3):
        # row_st = lane/4, col_st = (lane%4)*16
        # v_offset_st = (row_st/2)*Q_BYTES_PER_2ROWS + ((row_st%2)*64 + col_st)
        row_st = lane_idx / 4
        col_st = (lane_idx % 4) * 16
        lds_st_offset = (
            (row_st / 2) * Q_BYTES_PER_2ROWS
            + (row_st % 2) * Q_ELEM_PER_ROW
            + col_st
        )

        # LDS read layout (MFMA-compatible):
        # row_ld = lane%16, col_ld = (lane/16)*8
        # v_offset_ld = (row_ld/2)*Q_BYTES_PER_2ROWS + ((row_ld%2)*64 + col_ld)
        row_ld = lane_idx % 16
        col_ld = (lane_idx / 16) * 8
        lds_ld_offset = (
            (row_ld / 2) * Q_BYTES_PER_2ROWS
            + (row_ld % 2) * Q_ELEM_PER_ROW
            + col_ld
        )

        q_regs = []  # Will hold 18 v2i64 = 16 nope + 2 rope

        # Fold s_offset and per-pass ioffset into voffset so that soffset=0.
        # LLVM ISel only extracts immediate offsets when soffset is literal 0.
        # v_offset_i32 is in bytes; buffer_load auto-scales by element_bytes
        # (i32 = 4), so divide by 4.  s_offset_i32 is also in bytes.
        voff_dw = _i32(
            (ArithValue(v_offset_i32) + ArithValue(s_offset_i32)) // 4
        )

        # Pre-compute LDS pointers (constant across passes)
        lds_st_addr = p_lds_q_warp + lds_st_offset
        lds_st_ptr = _inttoptr_lds(_raw(fx.Int64(lds_st_addr)))
        lds_rd_addr = p_lds_q_warp + lds_ld_offset

        def _q_buf_load(pass_idx):
            voff_pass = _i32(
                ArithValue(voff_dw) + pass_idx * Q_ELEM_PER_ROW // 4
            )
            return buffer_ops.buffer_load(
                query_rsrc,
                voff_pass,
                vec_width=4,
                dtype=T.i32,
            )

        def _shuffle_q_through_lds(q_vram_data):
            """LDS write (ds_write_b128) + barrier + LDS read (2x ds_read_b64)."""
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            llvm.StoreOp(_raw(q_vram_data), lds_st_ptr, alignment=16)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            q0 = _lds_load(lds_rd_addr, T.i64, static_byte_offset=0)
            q1 = _lds_load(lds_rd_addr, T.i64, static_byte_offset=MFMA_K)
            return (q0, q1)

        # 3-deep pipeline: keep 2 buffer_loads in flight while shuffling
        # the completed one through LDS (matches HK QManagerV3).
        #   Before loop: issue passes 0, 1
        #   Iteration i: wait(1), issue pass i+2, shuffle pass i
        #   Last 2 iters: wait(0), shuffle (no new issue)
        loads = [None, None, None]
        loads[0] = _q_buf_load(0)
        loads[1] = _q_buf_load(1)

        for i in range_constexpr(9):
            slot = i % 3
            issue_pass = i + 2

            if const_expr(issue_pass < 9):
                rocdl.s_waitcnt(_encode_waitcnt(vmcnt=1))
                loads[issue_pass % 3] = _q_buf_load(issue_pass)
            else:
                rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))

            q_regs.append(_shuffle_q_through_lds(loads[slot]))

        # Split into nope (passes 0-7 -> 16 sub-tiles) and rope (pass 8 -> 2 sub-tiles)
        q_nope_packs = []
        for i in range_constexpr(8):
            q_nope_packs.append(q_regs[i][0])  # sub-tile 0
            q_nope_packs.append(q_regs[i][1])  # sub-tile 1
        q_rope_packs = [q_regs[8][0], q_regs[8][1]]
        return q_nope_packs, q_rope_packs

    # ---- Helper: softmax scale + boundary masking ----
    def _softmax_scale_p(p_vals, col_0_start_i32, kv_end_i32, check_boundary):
        """Scale p_vals by softmax_scale, mask OOB to -inf.

        check_boundary: False (skip), True (always mask), or ir.Value i1
        (runtime: mask only when True at runtime).
        """
        result = [None] * 8
        for i in range_constexpr(8):
            result[i] = arith.MulFOp(_raw(p_vals[i]), _raw(softmax_scale), fastmath=fm_fast).result

        if const_expr(check_boundary is not False):
            for i in range_constexpr(8):
                sub_offset = (i // 4) * 16 + (i % 4)
                pos_i32 = _i32(ArithValue(col_0_start_i32) + sub_offset)
                is_oob = ArithValue(pos_i32) >= ArithValue(kv_end_i32)
                if const_expr(check_boundary is not True):
                    is_oob = _raw(ArithValue(check_boundary) & ArithValue(is_oob))
                result[i] = _raw(arith.select(is_oob, _raw(c_neg_inf), result[i]))
        return result

    # ---- Helper: online softmax ----
    def _softmax(
        p_vals,
        row_max_old,
        row_sum_e_old,
        is_first_iter,
        kv_tile_start_i32,
        kv_end_i32,
        check_boundary,
    ):
        """Online softmax: scale -> max -> exp2 -> sum -> rescale.

        p_vals: 8 f32 attention scores for this thread
        Returns: (p_exp_vals, row_max_new, row_sum_e_new, rescale)
        """
        # Column index for this thread's first element
        col_0_start_i32 = _i32(ArithValue(_i32(lane_idx / 16 * 4)) + kv_tile_start_i32)

        # Scale and mask
        scaled = _softmax_scale_p(p_vals, col_0_start_i32, kv_end_i32, check_boundary)

        # Local max of 8 values
        local_max = scaled[0]
        for i in range_constexpr(1, 8):
            local_max = arith.MaximumFOp(local_max, _raw(scaled[i]), fastmath=fm_no_inf).result

        # Warp reduce max (within 16-lane groups)
        local_max = _warp_reduce_max_16(local_max)

        # New row max
        if const_expr(is_first_iter):
            new_row_max = local_max
            rescale = _raw(c_one_f32)
        else:
            new_row_max = arith.MaximumFOp(local_max, _raw(row_max_old), fastmath=fm_no_inf).result
            # rescale = exp2((old_max - new_max) * log2e)
            diff = arith.SubFOp(_raw(row_max_old), new_row_max, fastmath=fm_no_inf).result
            rescale = _fast_exp2(arith.MulFOp(diff, _raw(c_log2e), fastmath=fm_no_inf).result)

        # exp(p - max) for each value, and sum
        p_exp_vals = [None] * 8
        local_sum = _raw(c_zero_f32)
        for i in range_constexpr(8):
            exp_arg = arith.MulFOp(
                arith.SubFOp(_raw(scaled[i]), new_row_max, fastmath=fm_no_inf).result,
                _raw(c_log2e),
                fastmath=fm_no_inf,
            ).result
            p_exp_vals[i] = _fast_exp2(exp_arg)
            local_sum = arith.AddFOp(local_sum, p_exp_vals[i], fastmath=fm_fast).result

        # Warp reduce sum
        local_sum = _warp_reduce_add_16(local_sum)

        # Update row_sum_e
        if const_expr(is_first_iter):
            row_sum_e_new = local_sum
        else:
            row_sum_e_new = arith.AddFOp(
                arith.MulFOp(rescale, _raw(row_sum_e_old), fastmath=fm_fast).result,
                local_sum,
                fastmath=fm_fast,
            ).result

        return p_exp_vals, new_row_max, row_sum_e_new, rescale

    # ---- Helper: pack P from f32 to fp8 ----
    def _pack_p_to_fp8(p_exp_vals):
        """Pack 8 f32 -> 2 i32 (4x cvt_pk_fp8_f32) -> 1 i64 for MFMA."""
        w0 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(p_exp_vals[0]), _raw(p_exp_vals[1]), c_zero_i32, 0)
        w0 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(p_exp_vals[2]), _raw(p_exp_vals[3]), w0, 1)
        w1 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(p_exp_vals[4]), _raw(p_exp_vals[5]), c_zero_i32, 0)
        w1 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(p_exp_vals[6]), _raw(p_exp_vals[7]), w1, 1)
        return _pack_i32x2(w0, w1)

    # ---- Helper: rescale oaccu ----
    def _rescale_oaccu(oaccu, rescale):
        """Multiply all oaccu accumulators by rescale factor.
        Descending s_setprio 3->0 across 4 groups of 8 muls."""
        rv = _raw(Vec.filled(4, rescale, fx.Float32))
        result = [None] * len(oaccu)
        for group in range_constexpr(4):
            rocdl.s_setprio(3 - group)
            for j in range_constexpr(8):
                i = group * 8 + j
                result[i] = arith.MulFOp(_raw(oaccu[i]), rv, fastmath=fm_fast).result
        return result

    # ---- Helper: process one KV tile (GEMM1 + softmax + V + GEMM2) ----
    # Interleaves async prefetch of the NEXT tile's KV data
    # into the GEMM1 NoPE loop (1 block per iteration, 9 total).
    def _process_tile_gemm1(
        p_lds_kv_base,
        kv_tile_start_i32,
        kv_end_i32,
        q_nope,
        q_rope,
        row_max_in,
        row_sum_e_in,
        is_first_iter,
        check_boundary,
        p_lds_kv_next_warp_i32=None,
        row_kv_ld_next=None,
        kv_ld_col_base_i32_arg=None,
        check_boundary_next=True,
        # 2-ahead row resolution (match HK's row_kv_ld_next_next pattern)
        nn_resolve_start=None,
        nn_resolve_end=None,
        do_resolve_nn=None,
    ):
        """Process one KV tile: QK GEMM -> softmax -> V transpose -> pack P.

        GEMM2 (PV accumulation) is NOT included -- call _gemm2_with_rescale
        after the branch merge to keep oaccu out of phi nodes.

        Returns (row_max, row_sum_e, p_pack, rescale).
        """
        # ---- K base VGPR (baked-in lane offset) ----
        k_base_i32 = _i32(ArithValue(_i32(p_lds_kv_base)) + ArithValue(_i32(k_lds_lane_offset)))

        do_prefetch = p_lds_kv_next_warp_i32 is not None

        def _maybe_prefetch(block_idx):
            """Issue prefetch (OOB check controlled by check_boundary_next)."""
            if const_expr(not do_prefetch):
                return
            _prefetch_k_tile_asm(
                p_lds_kv_next_warp_i32,
                row_kv_ld_next,
                kv_ld_col_base_i32_arg,
                block_idx,
                check_boundary=check_boundary_next,
            )

        # ---- Prefetch block 0 of next tile (inline asm, opaque to LLVM) ----
        _maybe_prefetch(0)

        # ---- GEMM1: QK attention scores ----
        p_comp = [_raw(c_zero_v4f32), _raw(c_zero_v4f32)]

        for nope_pair in range_constexpr(NUM_NOPE_ITERS):
            tile_0 = nope_pair * 2
            tile_1 = nope_pair * 2 + 1

            k0_lo = _load_k_from_lds(k_base_i32, 0, tile_0 * BLOCK_K)
            k0_hi = _load_k_from_lds(k_base_i32, 16, tile_0 * BLOCK_K)
            k1_lo = _load_k_from_lds(k_base_i32, 0, tile_1 * BLOCK_K)
            k1_hi = _load_k_from_lds(k_base_i32, 16, tile_1 * BLOCK_K)

            # Prefetch block nope_pair+1 of next tile (inline asm)
            _maybe_prefetch(nope_pair + 1)

            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=2))

            q_0 = q_nope[tile_0]
            q_1 = q_nope[tile_1]

            if const_expr(nope_pair == 0):
                p_comp[0] = _mfma_fp8(
                    T.f32x4, [k0_lo, q_0, _raw(c_zero_v4f32), 0, 0, 0]
                )
                p_comp[1] = _mfma_fp8(
                    T.f32x4, [k0_hi, q_0, _raw(c_zero_v4f32), 0, 0, 0]
                )
                rocdl.s_setprio(15)
            else:
                p_comp[0] = _mfma_fp8(T.f32x4, [k0_lo, q_0, p_comp[0], 0, 0, 0])
                p_comp[1] = _mfma_fp8(T.f32x4, [k0_hi, q_0, p_comp[1], 0, 0, 0])

            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

            p_comp[0] = _mfma_fp8(T.f32x4, [k1_lo, q_1, p_comp[0], 0, 0, 0])
            p_comp[1] = _mfma_fp8(T.f32x4, [k1_hi, q_1, p_comp[1], 0, 0, 0])

        for rope_pair in range_constexpr(NUM_ROPE_ITERS):
            tile_0 = rope_pair * 2
            tile_1 = rope_pair * 2 + 1

            k0_lo = _load_k_from_lds(k_base_i32, 0, (tile_0 + 16) * BLOCK_K)
            k0_hi = _load_k_from_lds(k_base_i32, 16, (tile_0 + 16) * BLOCK_K)
            k1_lo = _load_k_from_lds(k_base_i32, 0, (tile_1 + 16) * BLOCK_K)
            k1_hi = _load_k_from_lds(k_base_i32, 16, (tile_1 + 16) * BLOCK_K)

            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=2))

            p_comp[0] = _mfma_fp8(T.f32x4, [k0_lo, q_rope[tile_0], p_comp[0], 0, 0, 0])
            p_comp[1] = _mfma_fp8(T.f32x4, [k0_hi, q_rope[tile_0], p_comp[1], 0, 0, 0])

            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

            p_comp[0] = _mfma_fp8(T.f32x4, [k1_lo, q_rope[tile_1], p_comp[0], 0, 0, 0])
            p_comp[1] = _mfma_fp8(T.f32x4, [k1_hi, q_rope[tile_1], p_comp[1], 0, 0, 0])

        rocdl.s_setprio(14)

        # ---- Extract p_comp values for softmax ----
        p_vals = []
        for sub in range_constexpr(2):
            p_comp_sub = Vec(p_comp[sub])
            for ii in range_constexpr(4):
                p_vals.append(p_comp_sub[ii])

        # ---- Load V from KV LDS ----
        v8_raw = _load_v_from_lds(p_lds_kv_base, warp_idx, lane_idx)
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
        rocdl.sched_barrier(0)

        # ---- Resolve row for tile+2 (2-ahead, matches HK line 407-426) ----
        # The buffer_load has softmax+V-transpose+GEMM2+barrier to complete.
        if const_expr(do_resolve_nn is not None):
            row_kv_ld_nn = fx.Int32(-1)
            if do_resolve_nn:
                row_kv_ld_nn = _get_kv_ld_row(nn_resolve_start, nn_resolve_end, True)
        else:
            row_kv_ld_nn = fx.Int32(-1)

        # ---- Softmax ----
        p_exp_vals, row_max_new, row_sum_e_new, rescale = _softmax(
            p_vals,
            row_max_in,
            row_sum_e_in,
            is_first_iter,
            kv_tile_start_i32,
            kv_end_i32,
            check_boundary,
        )

        # ---- Transpose V and store to Vt LDS ----
        vt8 = _transpose_v(v8_raw)
        vt_lds_base = lds_base_idx + P_LDS_VT
        _store_vt_to_lds(vt_lds_base, warp_idx, lane_idx, vt8)

        # ---- Pack P to fp8 ----
        p_pack = _pack_p_to_fp8(p_exp_vals)

        return row_max_new, row_sum_e_new, p_pack, rescale, row_kv_ld_nn

    def _gemm2_core(p_pack, oaccu, vt_base_i32):
        """GEMM2 PV accumulation loop (shared by first-iter and rescale paths).

        Matches HK interleaving: 8x ds_read_b32 burst (2 PV iters),
        lgkmcnt(4) -> 2 MFMA, lgkmcnt(0) -> 2 MFMA.
        """
        rocdl.s_setprio(15)
        for pv_pair in range_constexpr(NUM_PV_ITERS // 2):
            # Load 8 values: vt for 2 consecutive PV iterations
            iter_a = pv_pair * 2
            iter_b = pv_pair * 2 + 1
            col_a0 = iter_a * MFMA_N * 2
            col_a1 = col_a0 + MFMA_N
            col_b0 = iter_b * MFMA_N * 2
            col_b1 = col_b0 + MFMA_N

            # 8x ds_read_b32 burst
            vta0_lo, vta0_hi = _load_vt_from_lds(vt_base_i32, col_a0)
            vta1_lo, vta1_hi = _load_vt_from_lds(vt_base_i32, col_a1)
            vtb0_lo, vtb0_hi = _load_vt_from_lds(vt_base_i32, col_b0)
            vtb1_lo, vtb1_hi = _load_vt_from_lds(vt_base_i32, col_b1)

            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=4))

            # MFMA pair A
            oaccu[iter_a * 2] = _mfma_fp8(
                T.f32x4, [_pack_i32x2(vta0_lo, vta0_hi), p_pack, oaccu[iter_a * 2], 0, 0, 0]
            )
            oaccu[iter_a * 2 + 1] = _mfma_fp8(
                T.f32x4, [_pack_i32x2(vta1_lo, vta1_hi), p_pack, oaccu[iter_a * 2 + 1], 0, 0, 0]
            )
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

            # MFMA pair B
            oaccu[iter_b * 2] = _mfma_fp8(
                T.f32x4, [_pack_i32x2(vtb0_lo, vtb0_hi), p_pack, oaccu[iter_b * 2], 0, 0, 0]
            )
            oaccu[iter_b * 2 + 1] = _mfma_fp8(
                T.f32x4, [_pack_i32x2(vtb1_lo, vtb1_hi), p_pack, oaccu[iter_b * 2 + 1], 0, 0, 0]
            )
            rocdl.sched_barrier(0)

            if const_expr(pv_pair < NUM_PV_ITERS // 2 - 1):
                rocdl.s_nop(1)

        rocdl.s_setprio(0)
        return oaccu

    def _gemm2_first_iter(p_pack, vt_base_i32):
        """GEMM2 for first iteration: C=0 (hardcoded), no rescale.

        The MFMA C input is literal c_zero_v4f32, so LLVM doesn't need
        oaccu registers live -- results go to fresh registers.
        """
        _barrier(lgkmcnt=0)
        rocdl.sched_barrier(0)
        oaccu = [_raw(c_zero_v4f32)] * (NUM_PV_ITERS * 2)
        return _gemm2_core(p_pack, oaccu, vt_base_i32)

    def _gemm2_with_rescale(p_pack, rescale, oaccu_in, vt_base_i32):
        """Rescale oaccu, barrier, then GEMM2 PV accumulation.

        This runs after the branch merge so oaccu never enters phi nodes.
        """
        oaccu = _rescale_oaccu(oaccu_in, rescale)
        _barrier(lgkmcnt=0)
        rocdl.sched_barrier(0)
        return _gemm2_core(p_pack, oaccu, vt_base_i32)

    def _pack_f32x4_to_bf16_2dw(acc_val):
        """Convert f32x4 accumulator to 2 packed bf16 dwords."""
        i16s = Vec(acc_val).to(fx.BFloat16).bitcast(fx.Int16)
        i16_0, i16_1, i16_2, i16_3 = (_raw(i16s[j]) for j in range(4))
        dw0 = _raw(ArithValue(i16_0).extui(T.i32) | (ArithValue(i16_1).extui(T.i32) << 16))
        dw1 = _raw(ArithValue(i16_2).extui(T.i32) | (ArithValue(i16_3).extui(T.i32) << 16))
        return dw0, dw1

    # ---- Pre-compute LDS reshape addresses (computed once, reused per store) ----
    # Need explicit DivUI/RemUI since ArithValue `/` `%` are signed.
    def _udiv(a, b):
        return arith.DivUIOp(_raw(a), _raw(fx.Int32(b))).result

    def _urem(a, b):
        return arith.RemUIOp(_raw(a), _raw(fx.Int32(b))).result

    lane_u = ArithValue(lane_idx_i32)

    # bf16 LDS write address (MFMA layout): row_st = lane%16, col_st = (lane/16)*4
    o16_row_st = _urem(lane_u, 16)
    o16_col_st = _i32(ArithValue(_udiv(lane_u, 16)) * 4)
    # get_v_offset_lds(r,c) = ((r/2)*68 + (r%2)*32 + c) * 2  [bytes]
    o16_st_offset = _raw(
        (ArithValue(_udiv(o16_row_st, 2)) * O16_ELEM_PER_PAD_2ROWS
         + ArithValue(_urem(o16_row_st, 2)) * O16_NUM_COLS
         + ArithValue(o16_col_st)) * 2
    )

    # bf16 LDS read address (coalesced layout): row_ld = lane/4, col_ld = (lane%4)*8
    o16_row_ld = _udiv(lane_u, 4)
    o16_col_ld = _i32(ArithValue(_urem(lane_u, 4)) * 8)
    o16_rd_offset = _raw(
        (ArithValue(_udiv(o16_row_ld, 2)) * O16_ELEM_PER_PAD_2ROWS
         + ArithValue(_urem(o16_row_ld, 2)) * O16_NUM_COLS
         + ArithValue(o16_col_ld)) * 2
    )

    # f32 LDS write address: row_st/col_st reused, different padding
    o32_st_offset = _i32((ArithValue(o16_row_st) * O32_ELEM_PER_PAD_ROW + ArithValue(o16_col_st)) * 4)

    # f32 LDS read address: row_ld = lane/8, col_ld = (lane%8)*4
    o32_row_ld = _udiv(lane_u, 8)
    o32_col_ld = _i32(ArithValue(_urem(lane_u, 8)) * 4)
    o32_rd_offset = _i32((ArithValue(o32_row_ld) * O32_ELEM_PER_PAD_ROW + ArithValue(o32_col_ld)) * 4)

    def _store_oaccu_pair_bf16(oaccu_a, oaccu_b, tile_idx, p_lds_o_i32, row_base_i32):
        """Store 2 oaccu groups (1 PV iter) as bf16 via LDS reshape.

        Matches HK OManager16bitsV2: writes MFMA-layout data to LDS,
        reads back in row-major coalesced layout, then buffer_store_dwordx4.
        """
        # Per-warp LDS base
        lds_warp = _i32(ArithValue(p_lds_o_i32) + ArithValue(warp_idx_i32) * O16_LDS_PER_WARP)
        lds_st_addr = _i32(ArithValue(lds_warp) + ArithValue(o16_st_offset))

        # LDS write: 2 sub-blocks -> 2x ds_write_b64
        for sub, acc_val in enumerate([oaccu_a, oaccu_b]):
            dw0, dw1 = _pack_f32x4_to_bf16_2dw(acc_val)
            vec_2dw = Vec.from_elements([dw0, dw1], fx.Int32)
            sub_offset = sub * O16_NUM_COLS
            st_addr_sub = _i32(ArithValue(lds_st_addr) + sub_offset)
            st_ptr = _lds_ptr_from_i32(st_addr_sub)
            llvm.StoreOp(vec_2dw, st_ptr, alignment=8, volatile_=True)

        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

        # LDS read: ds_read_b128 (4 dwords = 8 bf16 in coalesced layout)
        lds_rd_addr = _i32(ArithValue(lds_warp) + ArithValue(o16_rd_offset))
        rd_ptr = _lds_ptr_from_i32(lds_rd_addr)
        data = llvm.LoadOp(T.i32x4, rd_ptr, alignment=16).result

        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

        # Coalesced VRAM store: buffer_store_dwordx4
        row_vram = ArithValue(row_base_i32) + ArithValue(o16_row_ld)
        col_vram = ArithValue(o16_col_ld) + tile_idx * MFMA_N * 2
        vram_offset = _raw((row_vram * V_HEAD_DIM + col_vram) * 2)
        buffer_ops.buffer_store(data, final_output_rsrc, vram_offset, offset_is_bytes=True)

    def _store_oaccu_pair_split(oaccu_a, oaccu_b, tile_idx, p_lds_o_i32, row_base_i32):
        """Store 2 oaccu groups (1 PV iter) as f32 via LDS reshape.

        Matches HK OManager32bitsV2: writes MFMA-layout f32 data to LDS,
        reads back in row-major coalesced layout, then buffer_store_dwordx4.
        16 rows need 2 rounds (8 rows each) because 64 lanes / 8 lanes-per-row = 8.
        """
        # Per-warp LDS base
        lds_warp = _i32(ArithValue(p_lds_o_i32) + ArithValue(warp_idx_i32) * O32_LDS_PER_WARP)
        lds_st_addr = _i32(ArithValue(lds_warp) + ArithValue(o32_st_offset))

        col_offset_i32 = tile_idx * MFMA_N * 2
        O32_LD_DELTA = 8 * O32_ELEM_PER_PAD_ROW * 4  # 1152 bytes between round 0/1

        # LDS write: 2 sub-blocks -> 2x ds_write_b128
        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
        for sub, acc_val in enumerate([oaccu_a, oaccu_b]):
            sub_offset = sub * O32_NUM_COLS // 2 * 4
            st_addr_sub = _i32(ArithValue(lds_st_addr) + sub_offset)
            st_ptr = _lds_ptr_from_i32(st_addr_sub)
            llvm.StoreOp(acc_val, st_ptr, alignment=16)

        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

        # LDS read: 2x ds_read_b128 (round 0 = rows 0-7, round 1 = rows 8-15)
        lds_rd_addr = _i32(ArithValue(lds_warp) + ArithValue(o32_rd_offset))
        rd_ptr = _lds_ptr_from_i32(lds_rd_addr)
        data_0 = llvm.LoadOp(T.f32x4, rd_ptr, alignment=16).result
        data_1 = llvm.LoadOp(T.f32x4, _gep(rd_ptr, static_byte_offset=O32_LD_DELTA), alignment=16).result

        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

        # 2x coalesced VRAM store
        row_vram_0 = ArithValue(row_base_i32) + ArithValue(o32_row_ld)
        col_vram = ArithValue(o32_col_ld) + col_offset_i32
        vram_off_0 = _raw((row_vram_0 * V_HEAD_DIM + col_vram) * 4)
        buffer_ops.buffer_store(_raw(Vec(data_0).bitcast(fx.Int32)), split_output_rsrc, vram_off_0, offset_is_bytes=True)

        row_vram_1 = row_vram_0 + 8
        vram_off_1 = _raw((row_vram_1 * V_HEAD_DIM + col_vram) * 4)
        buffer_ops.buffer_store(_raw(Vec(data_1).bitcast(fx.Int32)), split_output_rsrc, vram_off_1, offset_is_bytes=True)

    def _gemm2_last_with_store(
        p_pack,
        rescale,
        oaccu_in,
        vt_base_i32,
        reci_sum,
        is_split,
        p_lds_o_i32,
        row_base_i32,
        is_first_iter_flag,
    ):
        """Last-tile GEMM2: interleave rescale + MFMA + normalize + store.

        Matches HK's kIsLastIter pattern. For each of 8 PV pairs:
        1. Rescale 2 oaccu groups (skip if first iter)
        2. Load Vt from LDS (4x ds_read)
        3. 2 MFMAs (accumulate or init)
        4. Multiply by reci_sum
        5. Store immediately (bf16 or f32 split)
        """
        rescale_vec = _raw(Vec.filled(4, fx.Float32(rescale), fx.Float32))
        reci_vec = _raw(Vec.filled(4, fx.Float32(reci_sum), fx.Float32))

        _barrier(lgkmcnt=0)
        rocdl.sched_barrier(0)
        rocdl.s_setprio(15)
        for pv_pair in range_constexpr(NUM_PV_ITERS // 2):
            iter_a = pv_pair * 2
            iter_b = pv_pair * 2 + 1
            col_a0 = iter_a * MFMA_N * 2
            col_a1 = col_a0 + MFMA_N
            col_b0 = iter_b * MFMA_N * 2
            col_b1 = col_b0 + MFMA_N

            # Rescale 4 oaccu groups for this pair (skip if first iter)
            if const_expr(not is_first_iter_flag):
                for idx in [iter_a * 2, iter_a * 2 + 1, iter_b * 2, iter_b * 2 + 1]:
                    oaccu_in[idx] = arith.MulFOp(_raw(oaccu_in[idx]), rescale_vec, fastmath=fm_fast).result

            # 8x ds_read_b32 burst
            vta0_lo, vta0_hi = _load_vt_from_lds(vt_base_i32, col_a0)
            vta1_lo, vta1_hi = _load_vt_from_lds(vt_base_i32, col_a1)
            vtb0_lo, vtb0_hi = _load_vt_from_lds(vt_base_i32, col_b0)
            vtb1_lo, vtb1_hi = _load_vt_from_lds(vt_base_i32, col_b1)

            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=4))

            # MFMA pair A
            oaccu_in[iter_a * 2] = _mfma_fp8(
                T.f32x4, [_pack_i32x2(vta0_lo, vta0_hi), p_pack, oaccu_in[iter_a * 2], 0, 0, 0]
            )
            oaccu_in[iter_a * 2 + 1] = _mfma_fp8(
                T.f32x4, [_pack_i32x2(vta1_lo, vta1_hi), p_pack, oaccu_in[iter_a * 2 + 1], 0, 0, 0]
            )
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))

            # MFMA pair B
            oaccu_in[iter_b * 2] = _mfma_fp8(
                T.f32x4, [_pack_i32x2(vtb0_lo, vtb0_hi), p_pack, oaccu_in[iter_b * 2], 0, 0, 0]
            )
            oaccu_in[iter_b * 2 + 1] = _mfma_fp8(
                T.f32x4, [_pack_i32x2(vtb1_lo, vtb1_hi), p_pack, oaccu_in[iter_b * 2 + 1], 0, 0, 0]
            )
            rocdl.sched_barrier(0)

            # Normalize by reci_sum
            for idx in [iter_a * 2, iter_a * 2 + 1, iter_b * 2, iter_b * 2 + 1]:
                oaccu_in[idx] = arith.MulFOp(oaccu_in[idx], reci_vec, fastmath=fm_fast).result

            # Store immediately via LDS reshape (coalesced)
            if const_expr(is_split):
                _store_oaccu_pair_split(
                    oaccu_in[iter_a * 2],
                    oaccu_in[iter_a * 2 + 1],
                    iter_a,
                    p_lds_o_i32,
                    row_base_i32,
                )
                _store_oaccu_pair_split(
                    oaccu_in[iter_b * 2],
                    oaccu_in[iter_b * 2 + 1],
                    iter_b,
                    p_lds_o_i32,
                    row_base_i32,
                )
            else:
                _store_oaccu_pair_bf16(
                    oaccu_in[iter_a * 2],
                    oaccu_in[iter_a * 2 + 1],
                    iter_a,
                    p_lds_o_i32,
                    row_base_i32,
                )
                _store_oaccu_pair_bf16(
                    oaccu_in[iter_b * 2],
                    oaccu_in[iter_b * 2 + 1],
                    iter_b,
                    p_lds_o_i32,
                    row_base_i32,
                )

        rocdl.s_setprio(0)

    # ==================================================================
    # KV LDS buffer pointers -- computed once, persist across work items
    # ==================================================================
    p_lds_kv_0_base = lds_base_idx + P_LDS_KV_0
    p_lds_kv_1_base = lds_base_idx + P_LDS_KV_1

    kv_warp_offset_i32 = _raw(ArithValue(warp_idx_i32) * KV_SUB_BYTES)

    p_lds_kv_0_warp_i32 = _raw(ArithValue(_i32(p_lds_kv_0_base)) + ArithValue(kv_warp_offset_i32))
    p_lds_kv_1_warp_i32 = _raw(ArithValue(_i32(p_lds_kv_1_base)) + ArithValue(kv_warp_offset_i32))

    vt_base_i32 = _raw(ArithValue(_i32(lds_base_idx + P_LDS_VT)) + ArithValue(_i32(vt_lds_lane_offset)))

    # ==================================================================
    # Main kernel body: persistent-thread work loop
    # ==================================================================
    for work_idx in range(work_start_idx, work_end_idx):
        # Load MlaWorkInfo
        wi_base_i32 = _i32(work_idx * SIZE_MLA_WORK_INFO_IN_DW)
        wi_dw1_4 = buffer_ops.buffer_load(
            work_info_set_rsrc,
            wi_base_i32 + 1,
            vec_width=4,
            dtype=T.i32,
        )
        wi_dw5 = buffer_ops.buffer_load(
            work_info_set_rsrc,
            wi_base_i32 + 5,
            vec_width=1,
            dtype=T.i32,
        )
        wi_dw1_4_vec = Vec(wi_dw1_4)
        partial_qo_loc = rocdl.readfirstlane(T.i32, wi_dw1_4_vec[0])
        qo_start = rocdl.readfirstlane(T.i32, wi_dw1_4_vec[1])
        kv_start = rocdl.readfirstlane(T.i32, wi_dw1_4_vec[3])
        kv_end = rocdl.readfirstlane(T.i32, _raw(wi_dw5))
        kv_len = kv_end - kv_start

        # ---- KV tile iteration ----
        # Initialize softmax state
        row_max = _raw(c_neg_inf)
        row_sum_e = _raw(c_zero_f32)

        # Compute number of tiles
        kv_len_v = ArithValue(kv_len)
        num_tiles = _udiv(kv_len_v + BLOCK_N - 1, BLOCK_N)

        # --- Pre-compute boundary flags ---
        first_tile_needs_boundary = kv_len_v < BLOCK_N
        has_multi_tiles = kv_len_v > BLOCK_N
        last_tile_partial = (kv_len_v & (BLOCK_N - 1)) != 0

        # --- First tile: resolve KV row (branched on boundary) ---
        row_kv_ld_first = fx.Int32(-1)
        if first_tile_needs_boundary:
            row_kv_ld_first = _get_kv_ld_row(kv_start, kv_end, True)
        else:
            row_kv_ld_first = _get_kv_ld_row(kv_start, _raw(ArithValue(kv_start) + BLOCK_N), False)

        # Load Q to GPR (independent of boundary check)
        q_nope_packs, q_rope_packs = _load_q_to_regs(qo_start)

        # Async load first tile KV to LDS (branched)
        if first_tile_needs_boundary:
            _async_load_kv_all(
                p_lds_kv_0_warp_i32,
                row_kv_ld_first,
                kv_ld_col_base_i32,
                check_boundary=True,
            )
        else:
            _async_load_kv_all(
                p_lds_kv_0_warp_i32,
                row_kv_ld_first,
                kv_ld_col_base_i32,
                check_boundary=False,
            )

        # --- Tile-1 row resolution (only meaningful for multi-tile) ---
        kv_start_v = ArithValue(kv_start)
        kv_start_plus_bn = _raw(kv_start_v + BLOCK_N)
        kv_start_plus_2bn = _raw(kv_start_v + 2 * BLOCK_N)
        tile1_is_full = ArithValue(kv_start_plus_2bn) <= ArithValue(kv_end)
        row_kv_ld_tile1 = fx.Int32(-1)
        if tile1_is_full:
            row_kv_ld_tile1 = _get_kv_ld_row(kv_start_plus_bn, kv_start_plus_2bn, False)
        else:
            row_kv_ld_tile1 = _get_kv_ld_row(kv_start_plus_bn, _raw(kv_end), True)

        # check_boundary_next for first tile: True only when
        # num_tiles==2 AND last_tile_partial (next tile is partial last)
        # Equiv: !tile1_is_full AND last_tile_partial
        # But simpler: cbn = !tile1_is_full (when num_tiles>=2, !tile1_is_full
        # means num_tiles==2, and if num_tiles==2 and tile1 not full then
        # last_tile_partial must be true). Actually just use: !tile1_is_full AND has_multi_tiles AND last_tile_partial.
        # Simplest correct: HK uses (kv_1st_end + BN - 1) < kv_end -> !(kv_start+2*BN <= kv_end) -> !tile1_is_full
        # Wait: HK condition for cbn=False is (kv_1st_end + BN - 1) < kv_end  i.e. kv_start+2*BN-1 < kv_end
        # i.e. kv_start+2*BN <= kv_end i.e. tile1_is_full. So cbn=False when tile1_is_full.
        # cbn=True when !tile1_is_full. This is correct regardless of last_tile_partial because
        # when num_tiles==2 and !tile1_is_full, the next tile IS the last and IS partial.
        # !tile1_is_full: kv_start + 2*BN > kv_end (num_tiles == 2, next tile partial)
        first_tile_cbn = ArithValue(kv_start_plus_2bn) > ArithValue(kv_end)
        do_resolve_nn_first = ArithValue(kv_start_plus_2bn) < ArithValue(kv_end)

        # Branch on has_multi_tiles: multi-tile gets prefetch, single doesn't
        p_pack_first = fx.Int64(0)
        rescale_first = c_one_f32
        row_kv_ld_nn_first = fx.Int32(-1)
        if _raw(has_multi_tiles):
            # Multi-tile: first tile is always full, prefetch tile 1.
            # Sub-branch on first_tile_cbn for compile-time check_boundary_next.
            if first_tile_cbn:
                # cbn=True: next tile needs boundary check (num_tiles==2, partial)
                _barrier(vmcnt=0, lgkmcnt=0)
                rocdl.sched_barrier(0)
                row_max, row_sum_e, p_pack_first, rescale_first, row_kv_ld_nn_first = _process_tile_gemm1(
                    p_lds_kv_0_base,
                    kv_start,
                    kv_end,
                    q_nope_packs,
                    q_rope_packs,
                    row_max,
                    row_sum_e,
                    is_first_iter=True,
                    check_boundary=False,
                    p_lds_kv_next_warp_i32=p_lds_kv_1_warp_i32,
                    row_kv_ld_next=row_kv_ld_tile1,
                    kv_ld_col_base_i32_arg=kv_ld_col_base_i32,
                    check_boundary_next=True,
                    nn_resolve_start=kv_start_plus_2bn,
                    nn_resolve_end=kv_end,
                    do_resolve_nn=do_resolve_nn_first,
                )
            else:
                # cbn=False: next tile is full, no boundary check
                _barrier(vmcnt=0, lgkmcnt=0)
                rocdl.sched_barrier(0)
                row_max, row_sum_e, p_pack_first, rescale_first, row_kv_ld_nn_first = _process_tile_gemm1(
                    p_lds_kv_0_base,
                    kv_start,
                    kv_end,
                    q_nope_packs,
                    q_rope_packs,
                    row_max,
                    row_sum_e,
                    is_first_iter=True,
                    check_boundary=False,
                    p_lds_kv_next_warp_i32=p_lds_kv_1_warp_i32,
                    row_kv_ld_next=row_kv_ld_tile1,
                    kv_ld_col_base_i32_arg=kv_ld_col_base_i32,
                    check_boundary_next=False,
                    nn_resolve_start=kv_start_plus_2bn,
                    nn_resolve_end=kv_end,
                    do_resolve_nn=do_resolve_nn_first,
                )
        else:
            # Single tile: no prefetch, no 2-ahead resolve
            _barrier(vmcnt=0, lgkmcnt=0)
            rocdl.sched_barrier(0)
            row_max, row_sum_e, p_pack_first, rescale_first, row_kv_ld_nn_first = _process_tile_gemm1(
                p_lds_kv_0_base,
                kv_start,
                kv_end,
                q_nope_packs,
                q_rope_packs,
                row_max,
                row_sum_e,
                is_first_iter=True,
                check_boundary=first_tile_needs_boundary,
            )

        def _write_lse(pqo_loc_i32, rm, rse):
            """Write LSE for split output (first 16 lanes per warp)."""
            if ArithValue(lane_idx_i32) < 16:
                log2_sum = fmath.log2(rse, fastmath=fm_fast)
                lse = fmath.fma(log2_sum, c_inv_log2e, rm, fastmath=fm_fast)
                row_idx_i32 = _raw(
                    ArithValue(lane_idx_i32) + ArithValue(warp_idx_i32) * 16
                    + ArithValue(pqo_loc_i32) * NUM_QO_HEADS
                )
                buffer_ops.buffer_store(lse, split_lse_rsrc, row_idx_i32)

        # LDS base for output reshape (reuse KV buffer 0 region)
        p_lds_o_i32 = _i32(p_lds_kv_0_base)

        def _do_last_gemm2_and_store(
            pp,
            rs,
            oaccu_list,
            rm,
            rse,
            is_first_iter_flag,
        ):
            """GEMM2 last tile with interleaved store + LSE write.

            Branches on partial_qo_loc to select bf16 vs f32 split output.
            """
            reci = arith.DivFOp(_raw(c_one_f32), _raw(rse), fastmath=fm_fast).result
            is_not_split = ArithValue(partial_qo_loc) < 0
            if is_not_split:
                # bf16 final output: row_base = qo_start * NUM_QO_HEADS + warp*16
                rb_bf16 = _raw(ArithValue(qo_start) * NUM_QO_HEADS + ArithValue(warp_idx_i32) * 16)
                _gemm2_last_with_store(
                    pp,
                    rs,
                    list(oaccu_list),
                    vt_base_i32,
                    reci,
                    False,
                    p_lds_o_i32,
                    rb_bf16,
                    is_first_iter_flag,
                )
            else:
                # f32 split output: row_base = pqo_loc * NUM_QO_HEADS + warp*16
                rb_split = _raw(ArithValue(partial_qo_loc) * NUM_QO_HEADS + ArithValue(warp_idx_i32) * 16)
                _gemm2_last_with_store(
                    pp,
                    rs,
                    list(oaccu_list),
                    vt_base_i32,
                    reci,
                    True,
                    p_lds_o_i32,
                    rb_split,
                    is_first_iter_flag,
                )
                _write_lse(_raw(partial_qo_loc), rm, rse)

        # ---- Multi-tile vs single-tile dispatch ----
        def _multi_tile_path():
            # === Multi-tile path ===

            # GEMM2 for first tile: C=0 hardcoded, no rescale needed
            oaccu_mt = _gemm2_first_iter(p_pack_first, vt_base_i32)

            # --- Middle tiles [1, num_tiles-1) via loop-carried range ---
            num_tiles_v = ArithValue(num_tiles)
            num_tiles_m1 = _raw(num_tiles_v - 1)
            num_tiles_m2 = _raw(num_tiles_v - 2)

            init_args = [row_max, row_sum_e] + oaccu_mt + [row_kv_ld_nn_first]

            for tile_iv, state in range(fx.Index(1), fx.Index(num_tiles_m1), fx.Index(1), init=init_args):
                tile_iv_i32 = ArithValue(fx.Int32(tile_iv))
                kv_tile_start_i32 = _raw(kv_start_v + tile_iv_i32 * BLOCK_N)

                # Unpack carried state
                rm_carried = state[0]
                rse_carried = state[1]
                oaccu_carried = [state[2 + i] for i in range(NUM_PV_ITERS * 2)]
                row_kv_ld_next = state[2 + NUM_PV_ITERS * 2]

                # Buffer parity
                is_odd = (tile_iv_i32 & 1) != 0
                curr_base_idx = ArithValue(is_odd).select(p_lds_kv_1_base, p_lds_kv_0_base)
                next_warp = ArithValue(is_odd).select(p_lds_kv_0_warp_i32, p_lds_kv_1_warp_i32)

                # check_boundary_next: True when tile_idx == num_tiles-2 AND last_tile_partial
                is_second_to_last = tile_iv_i32 == ArithValue(num_tiles_m2)
                mid_cbn = _raw(ArithValue(is_second_to_last) & ArithValue(last_tile_partial))

                # 2-ahead resolve params
                nn_start_mid = _raw(ArithValue(kv_tile_start_i32) + 2 * BLOCK_N)
                do_resolve_nn_mid = ArithValue(nn_start_mid) < ArithValue(kv_end)

                rm_m = c_neg_inf
                rse_m = c_zero_f32
                pp_m = fx.Int64(0)
                rs_m = c_one_f32
                nn_m = fx.Int32(-1)
                if mid_cbn:
                    # cbn=True: next tile needs boundary check
                    _barrier(vmcnt=0, lgkmcnt=0)
                    rocdl.sched_barrier(0)
                    rm_m, rse_m, pp_m, rs_m, nn_m = _process_tile_gemm1(
                        curr_base_idx,
                        kv_tile_start_i32,
                        kv_end,
                        q_nope_packs,
                        q_rope_packs,
                        rm_carried,
                        rse_carried,
                        is_first_iter=False,
                        check_boundary=False,
                        p_lds_kv_next_warp_i32=next_warp,
                        row_kv_ld_next=row_kv_ld_next,
                        kv_ld_col_base_i32_arg=kv_ld_col_base_i32,
                        check_boundary_next=True,
                        nn_resolve_start=nn_start_mid,
                        nn_resolve_end=kv_end,
                        do_resolve_nn=do_resolve_nn_mid,
                    )
                else:
                    # cbn=False: next tile is full, no boundary check
                    _barrier(vmcnt=0, lgkmcnt=0)
                    rocdl.sched_barrier(0)
                    rm_m, rse_m, pp_m, rs_m, nn_m = _process_tile_gemm1(
                        curr_base_idx,
                        kv_tile_start_i32,
                        kv_end,
                        q_nope_packs,
                        q_rope_packs,
                        rm_carried,
                        rse_carried,
                        is_first_iter=False,
                        check_boundary=False,
                        p_lds_kv_next_warp_i32=next_warp,
                        row_kv_ld_next=row_kv_ld_next,
                        kv_ld_col_base_i32_arg=kv_ld_col_base_i32,
                        check_boundary_next=False,
                        nn_resolve_start=nn_start_mid,
                        nn_resolve_end=kv_end,
                        do_resolve_nn=do_resolve_nn_mid,
                    )
                oa_m = _gemm2_with_rescale(pp_m, rs_m, oaccu_carried, vt_base_i32)
                yield_vals = [rm_m, rse_m] + oa_m + [nn_m]
                results = yield yield_vals

            # Unpack results from middle tiles loop
            row_max_mt = results[0]
            row_sum_e_mt = results[1]
            oaccu_mt = [results[2 + i] for i in range(NUM_PV_ITERS * 2)]

            # --- Last tile: GEMM1 + interleaved GEMM2 store ---
            last_tile_iv = ArithValue(num_tiles_m1)
            kv_last_start = _raw(kv_start_v + last_tile_iv * BLOCK_N)
            last_is_odd = (last_tile_iv & 1) != 0
            last_curr_base = ArithValue(last_is_odd).select(p_lds_kv_1_base, p_lds_kv_0_base)

            _barrier(vmcnt=0, lgkmcnt=0)
            rocdl.sched_barrier(0)
            rm_l, rse_l, pp_l, rs_l, _nn_l = _process_tile_gemm1(
                last_curr_base,
                kv_last_start,
                kv_end,
                q_nope_packs,
                q_rope_packs,
                row_max_mt,
                row_sum_e_mt,
                is_first_iter=False,
                check_boundary=last_tile_partial,
            )
            _do_last_gemm2_and_store(
                pp_l,
                rs_l,
                oaccu_mt,
                rm_l,
                rse_l,
                is_first_iter_flag=False,
            )

        def _single_tile_path():
            # === Single tile path: GEMM2 with interleaved store ===
            oaccu_st = [_raw(c_zero_v4f32)] * (NUM_PV_ITERS * 2)
            _do_last_gemm2_and_store(
                p_pack_first,
                rescale_first,
                oaccu_st,
                row_max,
                row_sum_e,
                is_first_iter_flag=True,
            )

        @flyc.jit
        def _dispatch_multi_single():
            if has_multi_tiles:
                _multi_tile_path()
            else:
                _single_tile_path()

        _dispatch_multi_single()


# ---------------------------------------------------------------------------
# JIT launcher
# ---------------------------------------------------------------------------
@flyc.jit
def launch_mla_fwd_decode_m16x8_fp8_fp8(
    query: fx.Tensor,
    kv_buffer: fx.Tensor,
    kv_page_indices: fx.Tensor,
    work_indptr: fx.Tensor,
    work_info_set: fx.Tensor,
    final_output: fx.Tensor,
    split_output: fx.Tensor,
    split_lse: fx.Tensor,
    softmax_scale: fx.Float32,
    num_cus: fx.Constexpr,
    lds_size: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """JIT host function: configures grid/block and launches the kernel."""
    assert TOTAL_LDS_BYTES <= lds_size, (
        f"Kernel requires {TOTAL_LDS_BYTES} bytes LDS but CU budget is {lds_size}"
    )
    kn_mla_fwd_decode_m16x8_fp8_fp8(
        query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        final_output,
        split_output,
        split_lse,
        softmax_scale,
    ).launch(
        grid=(num_cus, 1, 1),
        block=(NUM_THREADS, 1, 1),
        smem=0,  # LDS is statically allocated via SmemAllocator
        stream=stream,
    )
