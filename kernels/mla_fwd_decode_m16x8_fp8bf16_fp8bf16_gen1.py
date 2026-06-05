# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MLA V4.0 Gen.1 decode kernel (mi35x m16x8, fp8 NoPE + bf16 RoPE).

Transplanted from
``csrc/kernels/mla/hk/mi35x_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1.cuh``.

Architecture
------------
* gfx950 (MI350), wave size 64, 8 warps / 512 threads per workgroup,
  ``kOccupancy=1`` (one workgroup per CU).
* Tile: ``m16x8`` -- 8 ptiles per WG, each ptile = 1 wave owning
  ``kTileM=16`` query rows.  ``H * mtp == 128`` total work items.
* Q split:
    - ``Q[:, 0:256]`` lives PINNED in v72..v103 (Phase A QK).
    - ``Q[:, 256:512]`` lives in a 64 KB per-WG bf16 LDS region (Phase B QK).
* KV double-buffered ("pong") in two 32 KB LDS slots.  Single new tile per
  iteration, swap pointers at iter end.
* ``oaccu`` (16 rows x 512 fp32 cols = 128 fp32/lane) lives PINNED in
  v128..v255 across the whole work loop.

Pinned VGPR layout (per spec ch. 5)
-----------------------------------
======== ================== ===================== ===========================
v0..63   compiler scratch   amdgpu_num_vgpr(64)   cvt, ds_read, sgpr arithm.
v64..71  q_lds              Phase-B Q from LDS    8 vgprs (2 paired tiles)
v72..103 q_vgpr             Q[:, 0:256]           32 vgprs (8 base tiles)
v104..111 pv_v_aux/kv_alt   Dual role (QK/PV)    8 vgprs
v112..119 kv                K tile carrier        8 vgprs (2 base tiles)
v120..127 p_comp             fp32 softmax output  8 vgprs
v120..123 p_mfma (overlay)   bf16 P for PV mfma   4 vgprs (overlay on p_comp)
v128..255 oaccu              fp32 output accum    128 vgprs (16 PV iters x 8)
======== ================== ===================== ===========================

Sub-tile-of-8 ("sb8") permutation
---------------------------------
Each warp's K and Q wave-tile spans 64 cols organised as 8 sub-tiles of 8.
Storing the sub-tiles in LDS in permuted order ``[0,2,4,6,1,3,5,7]``
eliminates the 2-way ds_write_b128 bank conflict.  The QK reduction is
unaffected (sums are commutative); the OManager V3 epilogue applies the
inverse perm to recover natural column order in VRAM.

Site C bank-conflict row XOR
----------------------------
Composed on top of sb8 (disjoint bit).  Rows 4..7 and 12..15 of each
16-row sub-block get bit-5 of their ``byte_in_sb`` flipped.  Applied to
both writers (NoPE cvt+store, RoPE direct vmem->LDS) and the QK reader.

NOTE: do NOT use ``from __future__ import annotations`` -- it breaks
``fx.Constexpr`` detection in the FlyDSL AST rewriter.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, memref, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import pinned_vgpr as pv
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator

# ---------------------------------------------------------------------------
# Trait constants  (HkMlaV40DecodeFwdTraits, H=128, mtp=1)
# ---------------------------------------------------------------------------
NUM_QO_HEADS: int = 128
NUM_KV_HEADS: int = 1
KV_LORA_RANK: int = 448
QK_NOPE_HEAD_DIM: int = KV_LORA_RANK           # 448
QK_ROPE_HEAD_DIM: int = 64
QK_HEAD_DIM: int = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 512
V_HEAD_DIM: int = QK_HEAD_DIM                  # 512 (= NoPE + RoPE)
# PAGE_SIZE is a compile-time parameter. FlyDSL captures module globals into
# the JIT cache key, so mutating this between launches automatically yields
# a separately-specialized kernel per value (no factory required).
# Supported values: {1, 16, 32, 64}.  Mutate via _set_page_size() below.
PAGE_SIZE: int = 1
NUM_WARPS: int = 8
WARP_SIZE: int = 64
NUM_THREADS: int = NUM_WARPS * WARP_SIZE       # 512
BLOCK_M: int = 128                             # == NUM_QO_HEADS * mtp
BLOCK_N: int = 32                              # KV tile rows
BLOCK_K: int = 32                              # mfma K-dim
TILE_M: int = BLOCK_M // NUM_WARPS             # 16
OCCUPANCY: int = 1

# Packed Q-NoPE record: 448 fp8 + 16 dup-E8M0 + 112 pad = 576 B / token.
QK_PACKED_NOPE_BYTES: int = 576
SIZE_MLA_WORK_INFO_IN_DW: int = 8

# fp32 log2(e) (used to fuse exp into v_exp_f32 base-2).
LOG2E: float = 1.4426950408889634

# ---------------------------------------------------------------------------
# QManager Phase 1 constants (vmem fp8 -> staging LDS -> pinned q_vgpr)
# ---------------------------------------------------------------------------
Q_P1_CHUNK_COLS: int = 64
Q_P1_NUM_CHUNKS: int = 256 // Q_P1_CHUNK_COLS                            # 4
Q_P1_STAGING_BYTES_PER_WARP: int = TILE_M * Q_P1_CHUNK_COLS              # 1024 (16 rows x 64 fp8)
Q_P1_NUM_STAGING_BUFFERS: int = 2                                        # double-buffer
Q_P1_STAGING_BYTES_PER_WARP_TOTAL: int = (
    Q_P1_NUM_STAGING_BUFFERS * Q_P1_STAGING_BYTES_PER_WARP               # 2048
)
Q_SCALE_BASE_OFF: int = 448  # E8M0 scales start at byte 448 of the 576-byte record

# ---------------------------------------------------------------------------
# QManager Phase 2 LDS layout (final Q[:, 256:512] residence, bf16)
# ---------------------------------------------------------------------------
Q_SUB_BLOCK_ROWS: int = 16
Q_SUB_BLOCK_COLS: int = 32                  # one QK A-tile (bf16)
Q_SUB_BLOCK_BYTES: int = Q_SUB_BLOCK_ROWS * Q_SUB_BLOCK_COLS * 2  # 1024
Q_LDS_HALF_COLS: int = 256                  # 192 NoPE + 64 RoPE
Q_FINAL_COL_TILES: int = Q_LDS_HALF_COLS // Q_SUB_BLOCK_COLS  # 8
Q_WARP_FINAL_BYTES: int = Q_FINAL_COL_TILES * Q_SUB_BLOCK_BYTES  # 8 * 1024 = 8192
Q_FINAL_LDS_BYTES: int = BLOCK_M * Q_LDS_HALF_COLS * 2     # 128*256*2 = 64 KiB
Q_LDS_HEAD_PAD_BYTES: int = 192             # max kColInRecord pre-subtract

# ---------------------------------------------------------------------------
# KvManager v40 LDS layout (one pong = 32 KiB; double-buffered)
# ---------------------------------------------------------------------------
KV_SUB_BLOCK_ROWS: int = 16
KV_SUB_BLOCK_COLS: int = 32
KV_SUB_BLOCK_BYTES: int = KV_SUB_BLOCK_ROWS * KV_SUB_BLOCK_COLS * 2  # 1024
KV_NUM_ROW_TILES: int = BLOCK_N // KV_SUB_BLOCK_ROWS  # 2
KV_NUM_COL_TILES: int = QK_HEAD_DIM // KV_SUB_BLOCK_COLS  # 16
KV_NUM_COL_TILES_NOPE: int = QK_NOPE_HEAD_DIM // KV_SUB_BLOCK_COLS  # 14
KV_NUM_COL_TILES_ROPE: int = (QK_HEAD_DIM - QK_NOPE_HEAD_DIM) // KV_SUB_BLOCK_COLS  # 2
KV_PONG_BYTES: int = BLOCK_N * QK_HEAD_DIM * 2  # 32*512*2 = 32 KiB

# ---------------------------------------------------------------------------
# OManager bounce-LDS budgets (overlays p_lds_kv_next region)
# ---------------------------------------------------------------------------
O16_BYTES_PER_WARP: int = 2112   # OManager16bitsV3 (bf16) bounce: 2 KiB / warp
O16_LDS_BYTES: int = NUM_WARPS * O16_BYTES_PER_WARP  # 16,896 B
O32_BYTES_PER_WARP: int = 4352   # OManager32bitsV3 (fp32 split) bounce
O32_LDS_BYTES: int = NUM_WARPS * O32_BYTES_PER_WARP  # 34,816 B
O_LDS_BYTES: int = max(O16_LDS_BYTES, O32_LDS_BYTES)  # 34,816

# ---------------------------------------------------------------------------
# Overall LDS layout
#   [0,          KV_PONG_BYTES)              = p_lds_kv_0  (32 KiB)
#   [KV_PONG,    KV_PONG + max(O, KV))       = p_lds_kv_1  (32 KiB)
#                                              OVERLAID by OMgr bounce on the
#                                              global last iter.
#   [KV+max(O,KV), +Q_FINAL_LDS_BYTES)       = p_lds_q     (64 KiB)
# Total at H=128/mtp=1: 32 + max(34, 32) + 64 = ~98 KiB (well under 160).
#
# Q is placed AFTER both KV pongs so warp 0's Phase-1 staging
# (= p_lds_q + 0) can absorb the up-to-192-byte pre-subtract trick used by
# QManager::p1_vmem_to_staging_chunk without underflowing m0 mod 2^32.
# ---------------------------------------------------------------------------
P_LDS_KV_0: int = 0
P_LDS_KV_1: int = P_LDS_KV_0 + KV_PONG_BYTES
_KV_NEXT_BUDGET: int = max(KV_PONG_BYTES, O_LDS_BYTES)
P_LDS_Q: int = P_LDS_KV_1 + _KV_NEXT_BUDGET - KV_PONG_BYTES + KV_PONG_BYTES
# Reduce above: P_LDS_Q = P_LDS_KV_1 + max(KV_PONG_BYTES, O_LDS_BYTES)
P_LDS_Q = P_LDS_KV_1 + _KV_NEXT_BUDGET
TOTAL_LDS_BYTES: int = P_LDS_Q + Q_FINAL_LDS_BYTES

assert KV_PONG_BYTES + _KV_NEXT_BUDGET >= Q_LDS_HEAD_PAD_BYTES, (
    "KV region must precede p_lds_q with at least kLdsHeadPadBytes=192 B "
    "to absorb QManager P1 pre-subtract."
)
assert TOTAL_LDS_BYTES <= 160 * 1024, (
    f"V4 Gen.1 LDS budget {TOTAL_LDS_BYTES} bytes exceeds 160 KiB at kOccupancy=1."
)

# ---------------------------------------------------------------------------
# MFMA tile constants
# ---------------------------------------------------------------------------
MFMA_M: int = 16
MFMA_N: int = 16
MFMA_K: int = 32

# Per-iter QK / PV sub-tile counts
NUM_QK_VGPR_ITER: int = 8     # Q[:, 0:256] = 8 pinned A-tiles (Phase A)
NUM_QK_LDS_ITER: int = 8      # Q[:, 256:512] = 8 LDS A-tiles  (Phase B)
NUM_PV_ITER: int = V_HEAD_DIM // BLOCK_N  # 16

# ---------------------------------------------------------------------------
# Pinned VGPR layout (per spec ch. 5).  Declared via PinnedLayout so the
# union clobber is computed once.
# ---------------------------------------------------------------------------
PL = pv.PinnedLayout()
Q_LDS_RANGE   = PL.declare(64,  71,  name="q_lds")
Q_VGPR_RANGE  = PL.declare(72,  103, name="q_vgpr")
PV_V_AUX_RANGE = PL.declare(104, 111, name="pv_v_aux")
KV_RANGE      = PL.declare(112, 119, name="kv")
P_COMP_RANGE  = PL.declare(120, 127, name="p_comp")
P_MFMA_RANGE  = PL.declare(120, 123, name="p_mfma")   # overlay (low half of p_comp)
OACCU_RANGE   = PL.declare(128, 255, name="oaccu")

# Sub-tile views (q_vgpr split into 8 base tiles of 4 vgprs = 8 QK A-tiles).
Q_VGPR_TILES = Q_VGPR_RANGE.split(4)        # 8 tiles of 4 vgprs each
# Phase 1 cvt writes one dword (= 1 pinned vgpr) per call, so we also need
# a flat single-vgpr view of q_vgpr.
Q_VGPR_TILES_BY_VGPR = Q_VGPR_RANGE.split(1)  # 32 singletons
Q_LDS_TILES  = Q_LDS_RANGE.split(4)         # 2 tiles (q_k0 + q_k1)
KV_TOP, KV_BOT = KV_RANGE.split(4)          # 2 sub-tiles of (16x32)
KV_ALT_TOP, KV_ALT_BOT = PV_V_AUX_RANGE.split(4)  # alternate-tile carrier
P_COMP_LO = pv.PinnedRange(120, 123, name="p_comp_lo")  # 4 vgprs
P_COMP_HI = pv.PinnedRange(124, 127, name="p_comp_hi")
OACCU_SUBS = OACCU_RANGE.split(4)           # 32 sub-tiles of 4 vgprs each

# Debug isolation switch.  When True, the kernel dumps the raw p_comp
# (QK output) of tile 0 into final_output and skips softmax / PV / OMgr.
# Layout in final_output[1, qo_start, head_row, kv_col] (bf16):
#   head_row = warp * 16 + k * 4 + lane // 16  (k in 0..3)
#   kv_col   = (0..15) for the LO half, (16..31) for the HI half
#              from p_comp_lo[k] and p_comp_hi[k] respectively.
# Cols [32..512) are untouched (whatever previous launch left there).
# Set to False for normal runs.
DEBUG_FORCE_PCOMP_PATTERN: bool = False  # When True (with DEBUG_DUMP_QK),
                                         # overwrite the SSA p_comp_lo/hi
                                         # values with a known pattern
                                         # (= lane * 1000 + dw_idx) before
                                         # dumping.  Used to verify the
                                         # dump's (head_row, kv_col) lane
                                         # mapping matches mfma D layout.
DEBUG_DUMP_QK: bool = False          # When True, bypass softmax/PV/OMgr and
                                     # dump the 8 fp32 p_comp (QK output) per
                                     # lane into final_output cols [0..31].
                                     # See _debug_dump_p_comp for the layout.
DEBUG_QK_PHASE_A_ONLY: bool = False  # When True (with DEBUG_DUMP_QK=True),
                                     # also skip Phase B of _do_qk_gemm so
                                     # the dump shows Q[:,0:256] * K only.
DEBUG_QK_PHASE_B_ONLY: bool = False  # When True (with DEBUG_DUMP_QK=True),
                                     # skip Phase A so p_comp starts at 0
                                     # and only Phase B contributes.  Used
                                     # to isolate Phase B's correctness.
DEBUG_FAKE_Q_LDS_AS_ONES: bool = False  # When True, replace _load_q_lds_to_gpr
                                        # with inline-asm v_mov_b32 of bf16
                                        # pair {1.0, 1.0} (= 0x3F803F80) into
                                        # the pinned q_lds slot.  Bypasses
                                        # Phase 2 + ds_read path entirely.
DEBUG_FAKE_K_LDS_AS_ONES: bool = False  # Same idea for _load_k_lds_to_gpr.
DEBUG_FORCE_P2_SCALE_ONE: bool = False  # When True, replace scale_f in
                                        # _p2_cvt_store_nope_chunk with const
                                        # 1.0 (= fp32 0x3F800000).  If the
                                        # Q LDS NaN disappears with this, the
                                        # bug is in scale loading / lifetime.
DEBUG_DUMP_KV_NOPE_DW: bool = False # When True, dump the 4 i32 fp8 dwords
                                    # loaded by _kv_prefetch_tile (per lane,
                                    # 2 tiles).  Layout per-lane (1 of 2 tiles):
                                    #   head_row = warp*16 + lane//4
                                    #   byte_off  = tile_idx*64 + (lane%4)*16 + i*4
                                    # for i in 0..3.  Skips QK/softmax/PV/OMgr.
DEBUG_DUMP_P2_NOPE_DW: bool = False # When True, dump the 4 i32 fp8 dwords
                                    # loaded by Phase 2 vmem (per lane) into
                                    # final_output BEFORE the cvts run.
                                    # Layout per-lane (1 of 3 chunks):
                                    #   head_row = warp*16 + lane//4
                                    #   byte_off  = chunk*64 + (lane%4)*16 + i*4
                                    # for i in 0..3 (1 dw per i, = 4 fp8 bytes).
                                    # Skips QK/softmax/PV/OMgr (returns early).
DEBUG_DUMP_PMFMA: bool = False      # When True dump p_mfma post-softmax.
DEBUG_DUMP_LOCALMAX: bool = False   # When True dump local_max pre+post _warp_reduce_max.
DEBUG_DUMP_KV_TOP: bool = False     # When True, dump pinned KV_TOP (v112..v115)
                                    # right after _load_k_lds_to_gpr(KV_TOP,
                                    # base, k_row=0, k_col=0) (= the very
                                    # first K-side mfma A-operand) and skip
                                    # downstream.  Layout: lane T writes 4
                                    # dwords at final_output[0, warp*16 + T//4,
                                    # (T%4)*16 + i*4] for i in 0..3 (1 dw / i).
                                    # Compare against expected K[kv_rows of
                                    # warp's row_tile, cols 0..32] in mfma
                                    # A-tile layout.
DEBUG_DUMP_K_LDS: bool = False      # When True, dump K LDS pong 0 contents
                                    # after _kv_async_load_k completes.
                                    # Layout per warp: 16 col_tiles × 2 row_tiles
                                    # × 1024B = 32KB.  Each warp dumps a slice
                                    # of K LDS to its head_row band in
                                    # final_output.
DEBUG_DUMP_Q_LDS: bool = False      # When True, after Phase 2 + s_barrier,
                                    # dump each warp's 8 KiB slice of p_lds_q
                                    # to final_output and skip QK/softmax/PV/
                                    # OMgr.  Layout: lane T of warp w writes
                                    # 16 bytes (1 dwordx4 = 8 bf16) at
                                    #   row = w*16 + (T//8)
                                    #   col_byte_start = (T%8)*16
                                    # giving each warp its own 16 rows of
                                    # 128 cols (16 lanes per row in the b128
                                    # cycle). Two passes cover the full
                                    # 256-col Q[:, 256:512] region per warp.
DEBUG_BYPASS_P1_STAGING: bool = False # When True, _load_q_phase1 loads fp8
                                      # directly into VGPRs (no staging LDS)
                                      # and cvts to pinned q_vgpr in one pass.
                                      # Simpler / lower-perf form for debugging
                                      # the staged pipeline (see task #24).
DEBUG_DUMP_Q_VGPR: bool = False      # When True, dump raw q_vgpr (v72..v103)
                                     # of each warp right after _load_q_phase1
                                     # / _load_q_phase2.  Skips QK/softmax/PV/
                                     # OMgr.  Layout per lane: 32 dwords (=64
                                     # bf16) written contiguously into
                                     # final_output at
                                     #   row = warp*16 + (lane // 4)
                                     #   col = (lane % 4) * 64 + bf16_idx
                                     # so each warp's 16 rows × 256 cols cover
                                     # one warp's full q_vgpr contents in a
                                     # readable linear layout.

_DEBUG_RUN_EPILOGUE: bool = not (
    DEBUG_DUMP_QK or DEBUG_DUMP_Q_VGPR or DEBUG_DUMP_Q_LDS
    or DEBUG_DUMP_P2_NOPE_DW or DEBUG_DUMP_KV_NOPE_DW
    or DEBUG_DUMP_K_LDS or DEBUG_DUMP_KV_TOP or DEBUG_DUMP_PMFMA
    or DEBUG_DUMP_LOCALMAX
)


# ---------------------------------------------------------------------------
# Small utility shims (mirror v32 transplant for consistency)
# ---------------------------------------------------------------------------
def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    """Encode s_waitcnt bitfield for CDNA3/CDNA4 (gfx94x/gfx95x)."""
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


def _i32(value):
    raw = _raw(value) if not isinstance(value, ir.Value) else value
    if raw.type == T.i32:
        return raw
    return _raw(fx.Int32(raw))


def _uniform_i32(value):
    return rocdl.readfirstlane(T.i32, _i32(value))


def _idx(val):
    if isinstance(val, fx.Index):
        return val
    return fx.Index(val)


# ---------------------------------------------------------------------------
# LDS pointer / load helpers (copied from v32 conventions).
# ---------------------------------------------------------------------------
_LDS_PTR_TYPE = None


def _inttoptr_lds(byte_addr):
    """Convert an integer byte address to !llvm.ptr<3> (LDS pointer)."""
    global _LDS_PTR_TYPE
    if _LDS_PTR_TYPE is None:
        _LDS_PTR_TYPE = ir.Type.parse("!llvm.ptr<3>")
    return llvm.inttoptr(_LDS_PTR_TYPE, _raw(fx.Int64(byte_addr)))


_gep = buffer_ops.get_element_ptr


def _ptr_load(result_type, ptr, *, alignment=None, volatile_=False, nontemporal=False):
    return llvm.LoadOp(result_type, ptr, alignment=alignment,
                       volatile_=volatile_, nontemporal=nontemporal).result


def _ptr_store(value, ptr, *, alignment=None, volatile_=False):
    return llvm.StoreOp(_raw(value), ptr, alignment=alignment, volatile_=volatile_)


def _lds_ptr_from_i32(addr_i32, byte_offset=0):
    ptr = _inttoptr_lds(ArithValue(addr_i32).extui(T.i64))
    if byte_offset != 0:
        ptr = _gep(ptr, static_byte_offset=byte_offset)
    return ptr


def _lds_load_b128(addr_i32, byte_offset=0):
    """One ds_read_b128 of i32x4 from an i32 LDS byte address + static offset.

    NOTE: volatile=True prevents LLVM from CSE'ing two loads at the same LDS
    address.  In Phase 1's double-buffered pipeline, chunks 0 and 2 both
    read from buf 0 (same staging address); without volatile, LLVM treats
    the two loads as identical and reuses chunk 0's loaded value for
    chunk 2 -- so chunk 2's cvt sees chunk 0's stale bytes.  The same
    issue applies to chunks 1 and 3 sharing buf 1.
    """
    ptr = _lds_ptr_from_i32(addr_i32, byte_offset=byte_offset)
    return _ptr_load(T.i32x4, ptr, alignment=16, volatile_=True)


def _pack_i32_pair_to_i64(lo, hi):
    """Pack two i32 SSA values into an i64."""
    return _raw(
        ArithValue(lo).extui(T.i64) | (ArithValue(hi).extui(T.i64) << 32)
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
def kn_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1(
    # --- inputs (split NoPE / RoPE buffers per V4.0 layout) ---
    query_nope: fx.Tensor,       # [total_q, num_qheads, kQkPackedNopeQElems]  (fp8)
    query_rope: fx.Tensor,       # [total_q, num_qheads, kQkRopeHeadDim]       (bf16)
    kv_buffer_nope: fx.Tensor,   # [num_pages, page_size, num_kv_heads, kQkPackedNopeKvElems] (fp8)
    kv_buffer_rope: fx.Tensor,   # [num_pages, page_size, num_kv_heads, kQkRopeHeadDim]      (bf16)
    kv_page_indices: fx.Tensor,  # [num_pages_used]                            (i32)
    kv_last_page_lens: fx.Tensor,  # [num_seqs]                                (i32)
    # --- metadata ---
    work_indptr: fx.Tensor,      # [num_workers + 1]                           (i32)
    work_info_set: fx.Tensor,    # [num_work_items * 8]                        (i32)
    # --- outputs ---
    final_output: fx.Tensor,     # [1, total_q, num_qheads, V_HEAD_DIM]        (bf16)
    split_output: fx.Tensor,     # [1, partial_slots, num_qheads, V_HEAD_DIM]  (f32)
    split_lse: fx.Tensor,        # [1, partial_slots, num_qheads, 1]           (f32)
    # --- parameters ---
    softmax_scale: fx.Float32,
    log2_num_qheads: fx.Int32,
):
    """V4.0 Gen.1 MLA decode forward kernel.

    Persistent-thread kernel: each workgroup processes multiple work items
    from ``work_indptr[wg .. wg+1]``.  Per work item: load Q (vmem ->
    pinned VGPR + bf16 LDS), iterate KV tiles (single-pong double-buffer,
    sb8 perm), QK Phase A/B, online softmax, PV gemm + oaccu rescale,
    OMgr V3 epilogue.

    THIS IS STAGE 1: file scaffolding only.  All compute helpers are
    stubs.  Goal is to validate import + JIT-launch end-to-end with the
    pinned_vgpr helper before adding real logic.
    """
    # ---- Pinned VGPR clobber (union) ----------------------------------
    # Single function-wide clobber.  Per-call wrappers do NOT re-clobber.
    PL.emit_clobber()

    # Force PAGE_SIZE into this kernel's co_names so FlyDSL's
    # _discover_global_refs (which walks co_names from the launcher) sees the
    # dependency even though it's only used inside nested helpers like
    # _get_kv_ld_row.  Without this, the JIT cache key misses PAGE_SIZE
    # changes and a kernel compiled for PAGE_SIZE=1 is silently reused for
    # PAGE_SIZE=16/32/64 (different work-info parse + index conversion).
    _ = PAGE_SIZE

    # ---- LDS allocator ------------------------------------------------
    arch = get_hip_arch()
    lds_allocator = SmemAllocator(None, arch=arch)
    lds_allocator.ptr = TOTAL_LDS_BYTES

    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        lds_allocator.finalize()

    lds_buffer = lds_allocator.get_base()
    lds_base_idx = memref.extract_aligned_pointer_as_index(lds_buffer)

    # ---- Buffer resources ---------------------------------------------
    query_nope_rsrc = buffer_ops.create_buffer_resource(query_nope)
    query_rope_rsrc = buffer_ops.create_buffer_resource(query_rope)
    kv_nope_rsrc = buffer_ops.create_buffer_resource(kv_buffer_nope)
    kv_rope_rsrc = buffer_ops.create_buffer_resource(kv_buffer_rope)
    kv_page_indices_rsrc = buffer_ops.create_buffer_resource(kv_page_indices)
    kv_last_page_lens_rsrc = buffer_ops.create_buffer_resource(kv_last_page_lens)
    work_indptr_rsrc = buffer_ops.create_buffer_resource(work_indptr)
    work_info_set_rsrc = buffer_ops.create_buffer_resource(work_info_set)
    final_output_rsrc = buffer_ops.create_buffer_resource(final_output)
    split_output_rsrc = buffer_ops.create_buffer_resource(split_output)
    split_lse_rsrc = buffer_ops.create_buffer_resource(split_lse)

    # Silence "unused" diagnostics until stages 3-12 land.
    _ = (
        query_nope_rsrc, query_rope_rsrc, kv_nope_rsrc, kv_rope_rsrc,
        kv_page_indices_rsrc, kv_last_page_lens_rsrc, work_info_set_rsrc,
        final_output_rsrc, split_output_rsrc, split_lse_rsrc,
        lds_base_idx, softmax_scale, log2_num_qheads,
    )

    # ---- Thread / work indices ----------------------------------------
    worker_idx = gpu.block_idx.x
    tid = gpu.thread_id("x")
    warp_idx = tid / WARP_SIZE
    lane_idx = tid % WARP_SIZE
    _ = (warp_idx, lane_idx)

    # ---- Work range ---------------------------------------------------
    work_range = buffer_ops.buffer_load(
        work_indptr_rsrc, worker_idx, vec_width=2, dtype=T.i32
    )
    work_range_vec = Vec(work_range)
    work_start_i32 = rocdl.readfirstlane(T.i32, work_range_vec[0])
    work_end_i32 = rocdl.readfirstlane(T.i32, work_range_vec[1])
    work_start_idx = _idx(work_start_i32)
    work_end_idx = _idx(work_end_i32)

    # ---- e8m0 -> f32 scale conversion ---------------------------------
    # E8M0 is a 1-byte exponent (2^N); fp32 form is just (scale_byte << 23)
    # bit-cast as fp32 with mantissa=0.  Per spec Ch. 13.1, this MUST be
    # `asm volatile` to keep cross-BB ordering against the matching
    # buffer_load_ubyte; the pure-SSA form gets hoisted past s_waitcnt.
    def _e8m0_to_f32(scale_dw):
        return llvm.inline_asm(
            T.f32, [_raw(scale_dw)],
            "v_lshlrev_b32 $0, 23, $1", "=v,v",
            has_side_effects=True,
        )

    # ---- QManager Phase 1 helpers (vmem fp8 -> staging LDS -> q_vgpr) -
    # Per-warp Q vmem base byte offset.  Q layout = [total_q, num_qheads,
    # 576-byte packed records]; each warp owns TILE_M=16 query rows of a
    # single qo_start slice.
    # Wave-uniform i32 forms of warp_idx / lane_idx for byte-offset math.
    warp_idx_i32 = _uniform_i32(warp_idx)
    lane_idx_i32 = _i32(lane_idx)

    def _q_warp_vmem_base(qo_start_i32):
        # i32 byte offsets fit for the shapes V40 targets.
        return _raw(
            ArithValue(qo_start_i32) * fx.Int32(NUM_QO_HEADS * QK_PACKED_NOPE_BYTES)
            + ArithValue(warp_idx_i32) * fx.Int32(TILE_M * QK_PACKED_NOPE_BYTES)
        )

    def _q_warp_staging_base():
        """Per-warp Phase-1 staging base = first 2 KiB of this warp's 8 KiB
        slice of the final 64 KiB Q-LDS region.  (Wave-major contiguous
        layout; staging gets overwritten by Phase 2 in the same region.)
        """
        return _i32(
            _idx(lds_base_idx)
            + P_LDS_Q
            + warp_idx * fx.Int32(Q_WARP_FINAL_BYTES)
        )

    def _p1_vmem_to_staging_chunk(chunk_idx, buf_idx, q_vmem_base_i32):
        """Issue vmem fp8 + scale loads for chunk ``chunk_idx`` into staging
        buffer ``buf_idx``.  Returns the scale dword SSA i32 (one per lane,
        zero-extended from i8) for later cvt consumption.

        Per-lane vmem offset (NoPE 16 fp8 = 16 B):
            v_off = (lane>>2) * 576
                    + ((lane&3) XOR ((lane>>4)&1) << 1) * 16
            i_off = chunk * 64           (folded into soffset_bytes)
        Per-lane vmem offset (E8M0 scale, 1 byte):
            v_off_scale = (lane & 15) * 576
            i_off_scale = 448 + 2*chunk

        Spec Ch. 6.3 says the XOR on the col-quad swaps sub-tile row-bands
        4..7 / 12..15 so the reader's b128 cycles in Step 2 stay conflict-
        free (Site C swizzle, applied vmem-side).

        Scale-row attribution MUST be `lane & 15` (not `lane >> 2`) -- the
        consumer uses `lane & 15` to pick scale.  Mismatch is silently
        wrong on near-uniform data, catastrophic on outliers.
        """
        col_in_record = chunk_idx * Q_P1_CHUNK_COLS    # 0/64/128/192
        scale_byte_in_rec = Q_SCALE_BASE_OFF + 2 * chunk_idx  # 448/450/452/454
        staging_i = buf_idx * Q_P1_STAGING_BYTES_PER_WARP    # 0 or 1024

        # All per-lane arithmetic in i32 (no index/i32 mixing).
        row_in_warp_i32 = _raw(ArithValue(lane_idx_i32) // fx.Int32(4))
        col_quad_i32 = _raw(ArithValue(lane_idx_i32) % fx.Int32(4))
        s_band_i32 = _raw((ArithValue(lane_idx_i32) // fx.Int32(16)) % fx.Int32(2))
        col_quad_swz_i32 = _raw(
            ArithValue(col_quad_i32) ^ (ArithValue(s_band_i32) * fx.Int32(2))
        )

        v_off = _raw(
            ArithValue(row_in_warp_i32) * fx.Int32(QK_PACKED_NOPE_BYTES)
            + ArithValue(col_quad_swz_i32) * fx.Int32(16)
        )
        v_off_with_warp = _raw(ArithValue(q_vmem_base_i32) + ArithValue(v_off))

        # LDS dst = per-warp staging + buf-slot offset.  lds_base_idx is
        # index; convert via _i32 chain that goes through extui.
        lds_base_i32 = _i32(lds_base_idx)
        lds_dst = _raw(
            ArithValue(lds_base_i32) + fx.Int32(P_LDS_Q)
            + ArithValue(warp_idx_i32) * fx.Int32(Q_WARP_FINAL_BYTES)
            + fx.Int32(staging_i)
        )
        # IMPORTANT: route col_in_record via s_offset (NOT i_offset).  On
        # gfx950 the buffer_load_dwordx4 ... offset:N lds form silently
        # IGNORES the i_offset literal -- only the non-lds variant honors
        # `offset:`.  HK does the same: `async_load<16>(p_dst, v_off,
        # s_os=kColInRecord)`.  Symptom of the wrong form: chunks 1, 2, 3
        # all load chunk 0's data (since col_in_record is dropped); chunk 0
        # appears bit-exact vs the direct path by coincidence.
        rocdl.buffer_load_to_lds(
            query_nope_rsrc,
            _lds_ptr_from_i32(lds_dst),
            v_off_with_warp,
            size_bytes=16,
            soffset=fx.Int32(col_in_record),
            offset=0,
        )

        # Scale byte: 1 byte/lane via buffer_load i8.  soffset_bytes folds
        # the wave-uniform piece (kScaleByteInRec) into the instruction.
        scale_row_i32 = _raw(ArithValue(lane_idx_i32) % fx.Int32(16))
        v_off_scale = _raw(
            ArithValue(q_vmem_base_i32)
            + ArithValue(scale_row_i32) * fx.Int32(QK_PACKED_NOPE_BYTES)
        )
        scale_byte = buffer_ops.buffer_load(
            query_nope_rsrc, v_off_scale,
            vec_width=1, dtype=T.i8,
            soffset_bytes=scale_byte_in_rec,
        )
        # Zero-extend i8 to i32 so the cvt input is a clean v-class dword.
        return _raw(ArithValue(scale_byte).extui(T.i32))

    def _p1_vmem_to_vgpr_chunk_direct(chunk_idx, q_vmem_base_i32):
        """Direct vmem -> pinned q_vgpr (no staging LDS).

        Each lane loads 16 fp8 bytes (= 4 i32 dwords) directly from VMEM into
        SSA vgprs, then 8 cvts produce 8 bf16 dwords landing in pinned
        ``q_vgpr[8*chunk_idx .. +7]``.

        Per HK reader, lane (row_in_warp=lane&15, cb=(lane>>4)&3) needs fp8
        bytes from row=row_in_warp, chunk_byte = 16*cb + 0..15.  We load
        exactly that.
        """
        col_in_record = chunk_idx * Q_P1_CHUNK_COLS
        scale_byte_in_rec = Q_SCALE_BASE_OFF + 2 * chunk_idx

        row_in_warp_i32 = _raw(ArithValue(lane_idx_i32) % fx.Int32(16))
        cb_i32 = _raw((ArithValue(lane_idx_i32) // fx.Int32(16)) % fx.Int32(4))

        # VMEM byte offset: head_byte_base + chunk_byte
        # head_byte_base = row_in_warp * 576  (warp base already in q_vmem_base)
        # chunk_byte = 16 * cb  (within the chunk)
        # Plus soffset = col_in_record (folded into the buffer_load instruction).
        # buffer_load(dtype=T.i32) multiplies offset by 4, so pass element offsets.
        v_off_dw = _raw(
            ArithValue(q_vmem_base_i32) // fx.Int32(4)
            + ArithValue(row_in_warp_i32) * fx.Int32(QK_PACKED_NOPE_BYTES // 4)
            + ArithValue(cb_i32) * fx.Int32(4)   # 16 bytes / 4 bytes-per-i32 = 4 dwords
        )
        fp8_dw = buffer_ops.buffer_load(
            query_nope_rsrc, v_off_dw,
            vec_width=4, dtype=T.i32,
            soffset_bytes=col_in_record,
        )

        # Scale byte: same per-lane address as before; consumer uses lane&15.
        scale_row_i32 = _raw(ArithValue(lane_idx_i32) % fx.Int32(16))
        v_off_scale = _raw(
            ArithValue(q_vmem_base_i32)
            + ArithValue(scale_row_i32) * fx.Int32(QK_PACKED_NOPE_BYTES)
        )
        scale_byte = buffer_ops.buffer_load(
            query_nope_rsrc, v_off_scale,
            vec_width=1, dtype=T.i8,
            soffset_bytes=scale_byte_in_rec,
        )
        scale_dw = _raw(ArithValue(scale_byte).extui(T.i32))

        # Drain vmem before cvts read the SSA values.
        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
        rocdl.sched_barrier(0)

        scale_f = _e8m0_to_f32(scale_dw)
        fp8 = Vec(fp8_dw)

        # 8 cvts, same ordering as the staged path: src_dw=0..3, opsel=False/True.
        base = chunk_idx * 8
        for src_dw_idx in range_constexpr(4):
            dst_lo = Q_VGPR_TILES_BY_VGPR[base + 2 * src_dw_idx + 0]
            dst_hi = Q_VGPR_TILES_BY_VGPR[base + 2 * src_dw_idx + 1]
            pv.pinned_cvt_scalef32_pk_bf16_fp8(
                dst_lo, _raw(fp8[src_dw_idx]), scale_f, opsel=False,
            )
            pv.pinned_cvt_scalef32_pk_bf16_fp8(
                dst_hi, _raw(fp8[src_dw_idx]), scale_f, opsel=True,
            )

    def _p1_staging_to_vgpr_chunk(chunk_idx, buf_idx, scale_dw):
        """Drain vmcnt+lgkmcnt, ds_read_b128 the staging slot, run 8 pinned
        cvts that land in ``q_vgpr[8*chunk_idx .. 8*chunk_idx+7]``.

        Address mirror of the writer (Ch. 6.5):
            row_in_warp = lane & 15
            cb = (lane >> 4) & 3
            S  = (lane >> 2) & 1
            C_phys = cb XOR (S << 1)
            addr = staging + row_in_warp*64 + C_phys*16    (+kStagingI imm)
        """
        staging_i = buf_idx * Q_P1_STAGING_BYTES_PER_WARP

        row_in_warp_i32 = _raw(ArithValue(lane_idx_i32) % fx.Int32(16))
        s_bit_i32 = _raw((ArithValue(lane_idx_i32) // fx.Int32(4)) % fx.Int32(2))
        cb_i32 = _raw((ArithValue(lane_idx_i32) // fx.Int32(16)) % fx.Int32(4))
        c_phys_i32 = _raw(ArithValue(cb_i32) ^ (ArithValue(s_bit_i32) * fx.Int32(2)))

        lds_base_i32 = _i32(lds_base_idx)
        addr_base_i32 = _raw(
            ArithValue(lds_base_i32) + fx.Int32(P_LDS_Q)
            + ArithValue(warp_idx_i32) * fx.Int32(Q_WARP_FINAL_BYTES)
            + ArithValue(row_in_warp_i32) * fx.Int32(Q_P1_CHUNK_COLS)
            + ArithValue(c_phys_i32) * fx.Int32(16)
        )

        # Drain both vmcnt (staging vmem traffic + scale) AND lgkmcnt
        # (Phase 1's prior consumers).  HK orders these as one s_waitcnt
        # per chunk; we keep that contract.
        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0, lgkmcnt=0))
        rocdl.sched_barrier(0)

        # ds_read_b128 = 16 fp8 = 4 i32 dwords (both iters of this chunk).
        fp8_vec = _lds_load_b128(addr_base_i32, byte_offset=staging_i)
        fp8 = Vec(fp8_vec)

        # V4 shares one E8M0 scale across the full 64-col chunk.
        scale_f = _e8m0_to_f32(scale_dw)

        # 8 pinned cvts -> 4 dwords of bf16 / iter, 2 iters / chunk = 8
        # vgprs in q_vgpr[8*chunk_idx .. +7].  opsel=false reads the lo
        # fp8 pair (lanes 0,1), opsel=true reads the hi pair (lanes 2,3).
        # Pinned-DST form is required (see [[v40-cvt-to-pinned-inline-asm-gotcha]]).
        base = chunk_idx * 8
        for src_dw_idx in range_constexpr(4):
            dst_lo = Q_VGPR_TILES_BY_VGPR[base + 2 * src_dw_idx + 0]
            dst_hi = Q_VGPR_TILES_BY_VGPR[base + 2 * src_dw_idx + 1]
            pv.pinned_cvt_scalef32_pk_bf16_fp8(
                dst_lo, _raw(fp8[src_dw_idx]), scale_f, opsel=False,
            )
            pv.pinned_cvt_scalef32_pk_bf16_fp8(
                dst_hi, _raw(fp8[src_dw_idx]), scale_f, opsel=True,
            )

    # ---- QManager Phase 2 helpers (vmem -> final Q-LDS, bf16) ---------
    # 3 NoPE chunks (Q cols 256..448) + 1 RoPE chunk (Q cols 448..512).
    # Per warp: writes into its OWN 8 KiB slice of the 64 KiB Q-LDS
    # region (wave-major contiguous layout); no inter-wave barrier
    # needed.  Staging from Phase 1 lives in the first 2 KiB of the same
    # 8 KiB region and gets overwritten in place by Phase 2 -- safe
    # because intra-wave program order sequences it.
    def _q_warp_rope_vmem_base(qo_start_i32):
        """Per-warp vmem base for the BF16 RoPE Q buffer.

        Q-RoPE layout: [total_q, num_qheads, kQkRopeHeadDim] in bf16
        (= 2 bytes per element).  Each warp owns TILE_M=16 rows.
        """
        rope_stride_bytes = QK_ROPE_HEAD_DIM * 2  # 128
        return _raw(
            ArithValue(qo_start_i32) * fx.Int32(NUM_QO_HEADS * rope_stride_bytes)
            + ArithValue(warp_idx_i32) * fx.Int32(TILE_M * rope_stride_bytes)
        )

    def _p2_vmem_to_vgpr_nope_chunk(chunk_idx, q_vmem_base_i32):
        """Issue the NoPE Phase-2 vmem ops for chunk ``chunk_idx``.

        Returns ``(nope_dw_vec, scale_dw_i32)`` where ``nope_dw_vec`` is
        an SSA i32x4 (16 fp8 per lane).  vmem-load addressing is straight
        (no swizzle on this side); the swizzle is applied on the LDS-write
        side inside ``_p2_cvt_store_nope_chunk``.
        """
        col_in_record = 256 + chunk_idx * 64       # 256, 320, 384
        scale_byte_base = 448 + col_in_record // 32  # 456, 458, 460

        row_in_warp_i32 = _raw(ArithValue(lane_idx_i32) // fx.Int32(4))
        col_group_i32 = _raw(ArithValue(lane_idx_i32) % fx.Int32(4))

        # buffer_load with dtype=T.i32 multiplies offset by 4 (element size),
        # so we must pass an *element* (i32-dword) offset, not bytes.
        # QK_PACKED_NOPE_BYTES=576 is a multiple of 4 (=144 i32 dwords/row).
        # col_group * 16 bytes = col_group * 4 dwords.
        # q_vmem_base is in bytes; divide by 4 (it's also a multiple of 4).
        v_off_nope_dw = _raw(
            ArithValue(q_vmem_base_i32) // fx.Int32(4)
            + ArithValue(row_in_warp_i32) * fx.Int32(QK_PACKED_NOPE_BYTES // 4)
            + ArithValue(col_group_i32) * fx.Int32(4)
        )
        nope_dw = buffer_ops.buffer_load(
            query_nope_rsrc, v_off_nope_dw,
            vec_width=4, dtype=T.i32,
            soffset_bytes=col_in_record,
        )

        # Scale byte: per-lane row stride, lane's col_group/2 (so two adjacent
        # col_groups share one scale byte = the V4 dup-by-2 layout).
        v_off_scale = _raw(
            ArithValue(q_vmem_base_i32)
            + ArithValue(row_in_warp_i32) * fx.Int32(QK_PACKED_NOPE_BYTES)
            + ArithValue(col_group_i32) // fx.Int32(2)
        )
        scale_byte = buffer_ops.buffer_load(
            query_nope_rsrc, v_off_scale,
            vec_width=1, dtype=T.i8,
            soffset_bytes=scale_byte_base,
        )
        scale_dw = _raw(ArithValue(scale_byte).extui(T.i32))
        return nope_dw, scale_dw

    def _p2_dump_nope_dw(chunk_idx, nope_dw, qo_start_i32):
        """Dump the 4 i32 fp8 dwords loaded by Phase 2 vmem for this lane
        and chunk into final_output.  Per-lane:
          head_row  = warp*16 + lane//4
          byte_off  = chunk*64 + (lane%4)*16 + i*4   (i in 0..3, 1 dw / i)
        Used to verify per-lane addressing in Phase 2 -- if the dumped
        bytes don't match q_packed[head, col_in_record + col_group*16 + i*4],
        Phase 2's v_off is wrong for this lane.
        """
        nope_vec = Vec(nope_dw)
        head_row_i32 = _raw(
            ArithValue(warp_idx_i32) * fx.Int32(16)
            + ArithValue(lane_idx_i32) // fx.Int32(4)
        )
        lane_byte_base_i32 = _raw(
            (ArithValue(lane_idx_i32) % fx.Int32(4)) * fx.Int32(16)
        )
        row_base_byte = _raw(
            ArithValue(head_row_i32) * fx.Int32(V_HEAD_DIM * 2)
        )
        batch_byte_off = _raw(
            ArithValue(qo_start_i32)
            * fx.Int32(NUM_QO_HEADS * V_HEAD_DIM * 2)
        )
        for i in range_constexpr(4):
            byte_off = _raw(
                ArithValue(row_base_byte)
                + fx.Int32(chunk_idx * 64)
                + ArithValue(lane_byte_base_i32)
                + fx.Int32(i * 4)
            )
            buffer_ops.buffer_store(
                _raw(nope_vec[i]), final_output_rsrc, byte_off,
                offset_is_bytes=True, soffset_bytes=batch_byte_off,
            )

    def _kv_dump_nope_dw(tile_idx, nope_dw, qo_start_i32):
        """Dump 4 i32 fp8 dwords loaded by _kv_prefetch_tile per (warp, lane).
        Mirror of _p2_dump_nope_dw but for KV NoPE.  Same layout in
        final_output:
          head_row = warp*16 + lane//4
          byte_off = tile_idx*64 + (lane%4)*16 + i*4   (i in 0..3)
        Compare against kv_packed[row_kv_ld, col_in_record + col_group_swz*16 + i*4]
        where col_in_record = tile_idx*256.
        """
        nope_vec = Vec(nope_dw)
        head_row_i32 = _raw(
            ArithValue(warp_idx_i32) * fx.Int32(16)
            + ArithValue(lane_idx_i32) // fx.Int32(4)
        )
        lane_byte_base_i32 = _raw(
            (ArithValue(lane_idx_i32) % fx.Int32(4)) * fx.Int32(16)
        )
        row_base_byte = _raw(
            ArithValue(head_row_i32) * fx.Int32(V_HEAD_DIM * 2)
        )
        batch_byte_off = _raw(
            ArithValue(qo_start_i32)
            * fx.Int32(NUM_QO_HEADS * V_HEAD_DIM * 2)
        )
        for i in range_constexpr(4):
            byte_off = _raw(
                ArithValue(row_base_byte)
                + fx.Int32(tile_idx * 64)
                + ArithValue(lane_byte_base_i32)
                + fx.Int32(i * 4)
            )
            buffer_ops.buffer_store(
                _raw(nope_vec[i]), final_output_rsrc, byte_off,
                offset_is_bytes=True, soffset_bytes=batch_byte_off,
            )

    def _p2_cvt_store_nope_chunk(chunk_idx, nope_dw, scale_dw):
        """Cvt the 4 fp8 dwords to 8 bf16 dwords and ds_write_b128 the lo/hi
        halves to LDS sub-blocks (kColTileBase, kColTileBase+1).

        sb8 perm baked into the partition: lane's lo_dw covers data sub-tile
        2*col_group (= even ones), hi_dw covers 2*col_group+1 (= odd ones).
        After perm: lo lands at LDS col_tile = kColTileBase, hi lands at
        kColTileBase+1 (delta = 1024 B = kSubBlockBytes, fits in ds imm).

        Site C row XOR (Method 1, LDS-dst side, disjoint bit from sb8):
        rows whose ``(row_in_warp>>2)&1`` is set get bit 5 of byte_in_sb
        flipped (XOR by 32).
        """
        col_in_lds = chunk_idx * 64        # 0, 64, 128
        col_tile_base = col_in_lds // Q_SUB_BLOCK_COLS  # 0, 2, 4

        row_in_warp_i32 = _raw(ArithValue(lane_idx_i32) // fx.Int32(4))
        col_group_i32 = _raw(ArithValue(lane_idx_i32) % fx.Int32(4))

        # E8M0 -> fp32 scale (one per chunk).
        if const_expr(DEBUG_FORCE_P2_SCALE_ONE):
            # Hard-code scale = 1.0 (= fp32 bits 0x3F800000) to test whether
            # the NaN cluster in q_lds comes from scale loading vs other.
            scale_f = _raw(fx.Float32(1.0))
        else:
            scale_f = _e8m0_to_f32(scale_dw)

        # 8 cvts: lo_dw[0..3] from nope_dw[0,1] (lo/hi pairs), hi_dw[0..3]
        # from nope_dw[2,3].  All cvts share the same scale_f.
        nope_vec = Vec(nope_dw)
        lo_dws = []
        hi_dws = []
        for src in range_constexpr(2):
            for opsel in range_constexpr(2):
                # Issue cvt directly with SSA dst (LLVM allocates; we then
                # bitcast back to i32 dword for the ds_write).
                bf16_pair = llvm.inline_asm(
                    T.i32, [_raw(nope_vec[src]), _raw(scale_f)],
                    "v_cvt_scalef32_pk_bf16_fp8 $0, $1, $2"
                    + (" op_sel:[1,0,0]" if opsel == 1 else ""),
                    "=v,v,v",
                    has_side_effects=True,
                )
                lo_dws.append(bf16_pair)
        for src in range_constexpr(2):
            for opsel in range_constexpr(2):
                bf16_pair = llvm.inline_asm(
                    T.i32, [_raw(nope_vec[2 + src]), _raw(scale_f)],
                    "v_cvt_scalef32_pk_bf16_fp8 $0, $1, $2"
                    + (" op_sel:[1,0,0]" if opsel == 1 else ""),
                    "=v,v,v",
                    has_side_effects=True,
                )
                hi_dws.append(bf16_pair)

        # LDS dst address.
        # sub_block_byte_offset(warp, col_tile) = warp * 8192 + col_tile * 1024.
        byte_in_sb = _raw(ArithValue(col_group_i32) * fx.Int32(16))    # 0/16/32/48
        row_bank_swap = _raw(
            (ArithValue(row_in_warp_i32) // fx.Int32(4) % fx.Int32(2)) * fx.Int32(32)
        )
        byte_in_sb_swz = _raw(ArithValue(byte_in_sb) ^ ArithValue(row_bank_swap))

        lds_base_i32 = _i32(lds_base_idx)
        p_dst_lane = _raw(
            ArithValue(lds_base_i32) + fx.Int32(P_LDS_Q)
            + ArithValue(warp_idx_i32) * fx.Int32(Q_WARP_FINAL_BYTES)
            + fx.Int32(col_tile_base * Q_SUB_BLOCK_BYTES)
            + ArithValue(row_in_warp_i32) * fx.Int32(Q_SUB_BLOCK_COLS * 2)
            + ArithValue(byte_in_sb_swz)
        )

        # Two ds_write_b128: lo at +0, hi at +kSubBlockBytes.
        lo_vec = Vec.from_elements([lo_dws[i] for i in range(4)], fx.Int32)
        hi_vec = Vec.from_elements([hi_dws[i] for i in range(4)], fx.Int32)
        _ptr_store(_raw(lo_vec), _lds_ptr_from_i32(p_dst_lane), alignment=16)
        _ptr_store(_raw(hi_vec), _lds_ptr_from_i32(p_dst_lane,
                                                   byte_offset=Q_SUB_BLOCK_BYTES),
                   alignment=16)

    def _p2_load_rope_chunk(q_rope_vmem_base_i32):
        """Direct vmem(bf16) -> LDS for the 64-col RoPE chunk.

        Two buffer_load_lds_b128 cover the 64-col RoPE patch as two
        16x32 sub-blocks at LDS col-tiles 6 and 7.  sb8 perm applied on
        the vmem-src side (LDS dst is HW-fixed by buffer_load_lds), and
        Site-C row-conditional half-swap on col_quad.
        """
        col_tile_lo = (QK_NOPE_HEAD_DIM - 256) // Q_SUB_BLOCK_COLS    # 192/32 = 6
        col_tile_hi = col_tile_lo + 1                                  # 7
        rope_stride_bytes = QK_ROPE_HEAD_DIM * 2                       # 128
        v_stride = Q_SUB_BLOCK_COLS * 2                                # 64

        row_in_warp_i32 = _raw(ArithValue(lane_idx_i32) // fx.Int32(4))
        col_quad_i32 = _raw(ArithValue(lane_idx_i32) % fx.Int32(4))

        # Site-C row XOR on bit 1 of col_quad (vmem-side, RoPE uses Method 2).
        col_quad_swz_i32 = _raw(
            ArithValue(col_quad_i32)
            ^ ((ArithValue(row_in_warp_i32) // fx.Int32(4) % fx.Int32(2)) * fx.Int32(2))
        )

        v_off_lo = _raw(
            ArithValue(q_rope_vmem_base_i32)
            + ArithValue(row_in_warp_i32) * fx.Int32(rope_stride_bytes)
            + ArithValue(col_quad_swz_i32) * fx.Int32(32)
        )

        # LDS dst (lo) per-lane: lane T -> T*16 inside the sub-block.
        lds_off = _raw(ArithValue(lane_idx_i32) * fx.Int32(16))
        lds_base_i32 = _i32(lds_base_idx)
        p_dst_lo = _raw(
            ArithValue(lds_base_i32) + fx.Int32(P_LDS_Q)
            + ArithValue(warp_idx_i32) * fx.Int32(Q_WARP_FINAL_BYTES)
            + fx.Int32(col_tile_lo * Q_SUB_BLOCK_BYTES)
            + ArithValue(lds_off)
        )
        # LDS dst (hi): pre-subtract 16 B so the i_off=16 on the 2nd load
        # cancels and lands in col-tile 7.
        p_dst_hi_adj = _raw(
            ArithValue(lds_base_i32) + fx.Int32(P_LDS_Q)
            + ArithValue(warp_idx_i32) * fx.Int32(Q_WARP_FINAL_BYTES)
            + fx.Int32(col_tile_hi * Q_SUB_BLOCK_BYTES)
            + ArithValue(lds_off)
            - fx.Int32(16)
        )
        rocdl.buffer_load_to_lds(
            query_rope_rsrc, _lds_ptr_from_i32(p_dst_lo),
            v_off_lo, size_bytes=16, soffset=fx.Int32(0), offset=0,
        )
        rocdl.buffer_load_to_lds(
            query_rope_rsrc, _lds_ptr_from_i32(p_dst_hi_adj),
            v_off_lo, size_bytes=16, soffset=fx.Int32(0), offset=16,
        )

    def _load_q_phase2(qo_start_i32):
        """Phase 2: 3 NoPE chunks + 1 RoPE chunk, double-buffered.

        Matches HK load_q lines 663-676: prefetch chunks 0+1, cvt+store
        chunks 0+1 back-to-back; prefetch chunk 2 alongside RoPE prefetch,
        cvt+store chunk 2.
        """
        q_nope_base = _q_warp_vmem_base(qo_start_i32)
        q_rope_base = _q_warp_rope_vmem_base(qo_start_i32)

        nope_dw_0, scale_dw_0 = _p2_vmem_to_vgpr_nope_chunk(0, q_nope_base)
        nope_dw_1, scale_dw_1 = _p2_vmem_to_vgpr_nope_chunk(1, q_nope_base)
        if const_expr(DEBUG_DUMP_P2_NOPE_DW):
            # Drain vmem before reading nope_dw for the dump.
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
            _p2_dump_nope_dw(0, nope_dw_0, qo_start_i32)
            _p2_dump_nope_dw(1, nope_dw_1, qo_start_i32)
        _p2_cvt_store_nope_chunk(0, nope_dw_0, scale_dw_0)
        _p2_cvt_store_nope_chunk(1, nope_dw_1, scale_dw_1)

        nope_dw_2, scale_dw_2 = _p2_vmem_to_vgpr_nope_chunk(2, q_nope_base)
        _p2_load_rope_chunk(q_rope_base)
        if const_expr(DEBUG_DUMP_P2_NOPE_DW):
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
            _p2_dump_nope_dw(2, nope_dw_2, qo_start_i32)
        _p2_cvt_store_nope_chunk(2, nope_dw_2, scale_dw_2)

    # ===================================================================
    # KvManager v40 (spec Ch. 8)
    # ===================================================================
    # Wave -> tile partition (Option 2, branchless):
    #   row_tile        = (warp >> 1) & 1            # 0 or 1
    #   col_tile_in_tile = ((warp >> 1) & 2) | (warp & 1)   # 0..3
    # Each wave owns one 16-row x 64-col wave-tile per half-tile.  Half-tile
    # kTileIdx in {0, 1} covers KV cols [0,256) / [256,512).  For kTileIdx=1,
    # waves 5 and 7 land on col_tile_in_tile=3 = global col-tile 14/15 = RoPE
    # (bf16); they take the direct vmem->LDS path (TODO stage 6.5; deferred).

    def _kv_row_tile_i32(warp_i32):
        return _raw(
            (ArithValue(warp_i32) // fx.Int32(2)) & fx.Int32(1)
        )

    def _kv_col_tile_in_tile_i32(warp_i32):
        return _raw(
            ((ArithValue(warp_i32) // fx.Int32(2)) & fx.Int32(2))
            | (ArithValue(warp_i32) & fx.Int32(1))
        )

    row_tile_i32 = _kv_row_tile_i32(warp_idx_i32)
    col_tile_in_tile_i32 = _kv_col_tile_in_tile_i32(warp_idx_i32)

    # ---- Row resolver (per-lane local row 0..31 -> physical KV row) ----
    # Per-lane base index into the 32-row tile: row_tile*16 + (lane>>2).
    kv_ld_row_base_i32 = _raw(
        ArithValue(row_tile_i32) * fx.Int32(16)
        + ArithValue(lane_idx_i32) // fx.Int32(4)
    )

    def _get_kv_ld_row(kv_tile_start_i32, kv_tile_end_i32, check_boundary):
        """Resolve the physical KV row for this lane's row in the 32-row tile.

        Returns ``i32`` SSA value: the physical token row, or -1 when the
        lane's row is past ``kv_tile_end`` (only when ``check_boundary=True``).

        - PAGE_SIZE == 1: kv_indices[i] is the physical token row; direct lookup.
        - PAGE_SIZE  > 1: kv_indices[page_idx] is the physical PAGE number;
                          physical row = page_phys * PAGE_SIZE + intra_page.
        """
        row_idx_i32 = _raw(ArithValue(kv_ld_row_base_i32)
                          + ArithValue(kv_tile_start_i32))

        def _lookup(token_row_i32):
            # token_row -> physical-token via kv_indices.
            if const_expr(PAGE_SIZE == 1):
                return _raw(buffer_ops.buffer_load(
                    kv_page_indices_rsrc, _idx(token_row_i32),
                    vec_width=1, dtype=T.i32,
                ))
            else:
                page_idx_i32 = _raw(
                    ArithValue(token_row_i32) // fx.Int32(PAGE_SIZE)
                )
                intra_page_i32 = _raw(
                    ArithValue(token_row_i32) % fx.Int32(PAGE_SIZE)
                )
                page_phys_i32 = _raw(buffer_ops.buffer_load(
                    kv_page_indices_rsrc, _idx(page_idx_i32),
                    vec_width=1, dtype=T.i32,
                ))
                return _raw(
                    ArithValue(page_phys_i32) * fx.Int32(PAGE_SIZE)
                    + ArithValue(intra_page_i32)
                )

        if const_expr(check_boundary):
            cond = ArithValue(row_idx_i32) < ArithValue(kv_tile_end_i32)
            in_bounds_row = fx.Int32(-1)
            if cond:
                in_bounds_row = _lookup(row_idx_i32)
            return _raw(in_bounds_row)
        else:
            return _lookup(row_idx_i32)

    # ---- RoPE prefetch (direct vmem -> LDS for waves 5,7 on tile_idx=1) ----
    def _kv_prefetch_rope(p_lds_kv_base_i32, row_kv_ld_i32, check_boundary):
        """Issue 2× buffer_load_dwordx4_lds covering the 16x64 RoPE patch
        (LDS sub-blocks at col-tiles 14 & 15 of this warp's row_tile).

        Wave-5/wave-7 only; caller must gate.  vmem-src sb8 perm +
        Site-C row XOR (Method 2), per spec Ch. 8.8.
        """
        rope_stride = QK_ROPE_HEAD_DIM * 2                # 128
        col_quad_i32 = _raw(ArithValue(lane_idx_i32) & fx.Int32(3))
        col_quad_swz_i32 = _raw(
            ArithValue(col_quad_i32) ^ (
                ((ArithValue(lane_idx_i32) // fx.Int32(16)) & fx.Int32(1)) * fx.Int32(2)
            )
        )
        if const_expr(check_boundary):
            is_oob = ArithValue(row_kv_ld_i32) == fx.Int32(-1)
            safe_row = ArithValue(is_oob).select(_raw(fx.Int32(0)), row_kv_ld_i32)
        else:
            safe_row = row_kv_ld_i32

        v_off_lo = _raw(
            ArithValue(safe_row) * fx.Int32(rope_stride)
            + ArithValue(col_quad_swz_i32) * fx.Int32(32)
        )
        lds_off = _raw(ArithValue(lane_idx_i32) * fx.Int32(16))
        col_tile_lo = 14
        col_tile_hi = 15
        sb_off_lo_i32 = _raw(
            (fx.Int32(col_tile_lo * 2) + ArithValue(row_tile_i32))
            * fx.Int32(KV_SUB_BLOCK_BYTES)
        )
        sb_off_hi_i32 = _raw(
            (fx.Int32(col_tile_hi * 2) + ArithValue(row_tile_i32))
            * fx.Int32(KV_SUB_BLOCK_BYTES)
        )
        p_dst_lo = _raw(
            ArithValue(p_lds_kv_base_i32) + ArithValue(sb_off_lo_i32)
            + ArithValue(lds_off)
        )
        # Pre-subtract 16 from hi dst so i_off=16 lands at sub-block 15.
        p_dst_hi_adj = _raw(
            ArithValue(p_lds_kv_base_i32) + ArithValue(sb_off_hi_i32)
            + ArithValue(lds_off) - fx.Int32(16)
        )
        rocdl.buffer_load_to_lds(
            kv_rope_rsrc, _lds_ptr_from_i32(p_dst_lo),
            v_off_lo, size_bytes=16, soffset=fx.Int32(0), offset=0,
        )
        rocdl.buffer_load_to_lds(
            kv_rope_rsrc, _lds_ptr_from_i32(p_dst_hi_adj),
            v_off_lo, size_bytes=16, soffset=fx.Int32(0), offset=16,
        )

    # Wave-5/wave-7 dispatch helper for tile_idx=1.
    def _is_rope_owner_warp():
        """Returns SSA i1: True for warps 5 and 7."""
        w = ArithValue(warp_idx_i32)
        return _raw((w == fx.Int32(5)) | (w == fx.Int32(7)))

    # ---- Prefetch (NoPE branch only; RoPE waves 5/7 deferred) ----
    # KvTilePrefetch = (nope_dw: i32x4, scale_dw: i32) — returned as a tuple.
    def _kv_prefetch_tile(tile_idx, row_kv_ld_i32, check_boundary):
        """Issue NoPE fp8 (16 B/lane) + E8M0 scale (1 B/lane) loads for one
        half-tile (kTileIdx in {0,1}).

        vmem-side bank-conflict swizzle (Method 2, applied here so the LDS
        write side stays straight):
            col_group_swz = col_group XOR (((lane>>4) & 1) << 1)
        """
        # Address fields per spec Ch. 8.5:
        #   v_off_nope  = row_kv_ld * 576 + col_group_swz * 16
        #   s_off_nope  = col_tile_in_tile * 64
        #   i_off_nope  = kTileIdx * 256
        kv_packed_stride = QK_PACKED_NOPE_BYTES                          # 576
        wave_tile_cols = 2 * KV_SUB_BLOCK_COLS                           # 64

        col_group_i32 = _raw(ArithValue(lane_idx_i32) & fx.Int32(3))
        col_group_swz_i32 = _raw(
            ArithValue(col_group_i32) ^ (
                ((ArithValue(lane_idx_i32) // fx.Int32(16)) & fx.Int32(1)) * fx.Int32(2)
            )
        )

        # When check_boundary and row is -1, force v_off=0 (we'll zero the
        # carrier downstream via the buffer's OOB behaviour; HK uses a Python
        # ternary on the i32 value).
        if const_expr(check_boundary):
            is_oob = ArithValue(row_kv_ld_i32) == fx.Int32(-1)
            safe_row = ArithValue(is_oob).select(_raw(fx.Int32(0)), row_kv_ld_i32)
        else:
            safe_row = row_kv_ld_i32
        # buffer_load with dtype=T.i32 multiplies offset by 4, so pass an
        # *element* (i32-dword) offset.  kv_packed_stride=576 and
        # col_group_swz*16 are both multiples of 4.
        v_off_nope_dw = _raw(
            ArithValue(safe_row) * fx.Int32(kv_packed_stride // 4)
            + ArithValue(col_group_swz_i32) * fx.Int32(4)
        )
        s_off_nope = _raw(
            ArithValue(col_tile_in_tile_i32) * fx.Int32(wave_tile_cols)
        )
        i_off_nope = tile_idx * 256

        # Dump per (warp, lane, tile_idx) -- 4 dwords / 16 bytes per lane.
        # Layout: head_row = warp*16 + lane//4, byte_off = tile_idx*64 +
        # (lane%4)*16 + i*4 (i in 0..3, 1 dw per i).
        # Caller is responsible for passing qo_start.
        nope_dw = buffer_ops.buffer_load(
            kv_nope_rsrc, v_off_nope_dw,
            vec_width=4, dtype=T.i32,
            soffset_bytes=_raw(
                ArithValue(s_off_nope) + fx.Int32(i_off_nope)
            ),
        )

        # Scale: 1 byte/lane.  v_off_scale = safe_row * 576.
        v_off_scale = _raw(
            ArithValue(safe_row) * fx.Int32(kv_packed_stride)
        )
        s_off_scale = _raw(
            ArithValue(col_tile_in_tile_i32) * fx.Int32(2)
        )
        i_off_scale = 448 + tile_idx * 8
        scale_byte = buffer_ops.buffer_load(
            kv_nope_rsrc, v_off_scale,
            vec_width=1, dtype=T.i8,
            soffset_bytes=_raw(
                ArithValue(s_off_scale) + fx.Int32(i_off_scale)
            ),
        )
        scale_dw = _raw(ArithValue(scale_byte).extui(T.i32))
        return nope_dw, scale_dw

    # ---- Wait (vmcnt drain + sched_barrier, skip for RoPE waves on tile 1) ----
    def _kv_wait_loads(tile_idx, vmcnt):
        """Drain vmcnt to ``vmcnt`` and emit sched_barrier.

        For ``tile_idx==1``, waves 5 and 7 (the RoPE owners) skip the wait
        per spec Ch. 8.12 -- their direct vmem->LDS path is gated by the
        s_barrier the consumer issues, not by per-tile vmcnt.  TODO when we
        wire the RoPE branch in stage 6.5.
        """
        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=vmcnt))
        rocdl.sched_barrier(0)

    # ---- cvt + ds_write_b128 of a half-tile (spec Ch. 8.6) ----
    def _kv_cvt_and_store_tile(tile_idx, nope_dw, scale_dw):
        """Cvt 16 fp8/lane (= 4 dwords) to 16 bf16/lane (= 8 dwords) and
        issue 2 ds_write_b128 (lo at +0, hi at +2048 = kNumRowTiles *
        kSubBlockBytes).

        LDS dst is straight: the writer-side sb8 perm is baked into the
        wave->tile partition (Option 2), so the per-lane bytes go to
        ``sub_block_byte_offset(row_tile, col_tile_global_lo)`` and the
        col-major sub-block order (col*2 + row_tile) takes care of the rest.
        """
        kv_pong = None  # overridden by caller via _kv_cvt_store_tile_at()
        raise RuntimeError("call _kv_cvt_store_tile_at() to pass p_lds_kv_base")

    def _kv_cvt_store_tile_at(tile_idx, p_lds_kv_base_i32, nope_dw, scale_dw):
        """Same as above but takes the destination pong base as an i32 SSA."""
        scale_f = _e8m0_to_f32(scale_dw)
        nope_vec = Vec(nope_dw)

        # 8 cvts -> 8 bf16 dwords.  Match HK ordering: nope[0,1] -> lo[0..3],
        # nope[2,3] -> hi[0..3], each with opsel false/true.
        lo_dws = []
        hi_dws = []
        for src_idx in range_constexpr(2):
            for opsel in range_constexpr(2):
                lo_dws.append(llvm.inline_asm(
                    T.i32, [_raw(nope_vec[src_idx]), _raw(scale_f)],
                    "v_cvt_scalef32_pk_bf16_fp8 $0, $1, $2"
                    + (" op_sel:[1,0,0]" if opsel == 1 else ""),
                    "=v,v,v", has_side_effects=True,
                ))
        for src_idx in range_constexpr(2):
            for opsel in range_constexpr(2):
                hi_dws.append(llvm.inline_asm(
                    T.i32, [_raw(nope_vec[2 + src_idx]), _raw(scale_f)],
                    "v_cvt_scalef32_pk_bf16_fp8 $0, $1, $2"
                    + (" op_sel:[1,0,0]" if opsel == 1 else ""),
                    "=v,v,v", has_side_effects=True,
                ))

        # LDS dst per spec Ch. 8.6:
        #   col_tile_global_lo = kTileIdx * 8 + col_tile_in_tile * 2
        #   addr = p_lds_kv + sub_block_byte_offset(row_tile, col_tile_global_lo)
        #          + row_in_tile * 64 + col_group * 16
        col_tile_global_lo_i32 = _raw(
            fx.Int32(tile_idx * 8)
            + ArithValue(col_tile_in_tile_i32) * fx.Int32(2)
        )
        # sub_block_byte_offset(row_tile, col_tile) = (col_tile * 2 + row_tile) * 1024
        sub_block_off_i32 = _raw(
            (ArithValue(col_tile_global_lo_i32) * fx.Int32(2)
             + ArithValue(row_tile_i32)) * fx.Int32(KV_SUB_BLOCK_BYTES)
        )
        row_in_tile_i32 = _raw(ArithValue(lane_idx_i32) // fx.Int32(4))
        col_group_i32 = _raw(ArithValue(lane_idx_i32) & fx.Int32(3))
        # KV LDS write goes straight (no row-bank XOR on the write address).
        # The reader's row-bank XOR is paired with the vmem-load Method-2 swz
        # in _kv_prefetch_tile (col_group_swz = col_group ^ ((lane>>4)&1)<<1);
        # adding a writer XOR here would double-XOR and corrupt the odd
        # head_rows.  Matches HK store_kv_tile_step.
        byte_in_sb_i32 = _raw(ArithValue(col_group_i32) * fx.Int32(16))

        p_dst_lane = _raw(
            ArithValue(p_lds_kv_base_i32)
            + ArithValue(sub_block_off_i32)
            + ArithValue(row_in_tile_i32) * fx.Int32(KV_SUB_BLOCK_COLS * 2)
            + ArithValue(byte_in_sb_i32)
        )

        lo_vec = Vec.from_elements([lo_dws[i] for i in range(4)], fx.Int32)
        hi_vec = Vec.from_elements([hi_dws[i] for i in range(4)], fx.Int32)
        _ptr_store(_raw(lo_vec), _lds_ptr_from_i32(p_dst_lane), alignment=16)
        # hi sub-block is at +kNumRowTiles * kSubBlockBytes = 2 * 1024 = 2048 B
        _ptr_store(_raw(hi_vec),
                   _lds_ptr_from_i32(p_dst_lane,
                                     byte_offset=KV_NUM_ROW_TILES * KV_SUB_BLOCK_BYTES),
                   alignment=16)

    # ---- Convenience wrapper: non-overlapped full-pong load (prologue) ----
    def _kv_async_load_k(p_lds_kv_base_i32, row_kv_ld_i32, check_boundary,
                         qo_start_i32=None):
        """Prologue helper: prefetch both half-tiles, wait, cvt+store both.

        Equivalent to HK ``KvManager8to16bitsV1::async_load_k``.  Used at
        the start of each work item to fill the first pong; main loop
        uses the split form (prefetch, wait, cvt+store) interleaved with
        QK mfmas (stage 7).

        Tile 0: NoPE-only for all 8 waves.
        Tile 1: NoPE for waves 0..4,6; RoPE direct vmem->LDS for waves 5,7.
        """
        is_rope_owner = _is_rope_owner_warp()

        # --- Tile 0: NoPE for all 8 waves ---
        n0, s0 = _kv_prefetch_tile(0, row_kv_ld_i32, check_boundary)

        # --- Tile 1: NoPE for waves 0..4,6; RoPE direct for 5,7. ---
        # We issue NoPE vmem for all waves (the buffer-load is cheap and the
        # OOB check on rsrc gracefully handles the case where the cvt+store
        # is skipped); only the cvt+store + RoPE direct path branch on the
        # rope-owner predicate to avoid double-writing LDS sub-blocks 14/15.
        n1, s1 = _kv_prefetch_tile(1, row_kv_ld_i32, check_boundary)
        if is_rope_owner:
            _kv_prefetch_rope(p_lds_kv_base_i32, row_kv_ld_i32, check_boundary)

        # --- Drain + cvt+store tile 0 ---
        _kv_wait_loads(0, vmcnt=2)
        if const_expr(DEBUG_DUMP_KV_NOPE_DW):
            rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0))
            _kv_dump_nope_dw(0, n0, qo_start_i32)
        _kv_cvt_store_tile_at(0, p_lds_kv_base_i32, n0, s0)

        # --- Drain + cvt+store tile 1 (NoPE waves only) ---
        _kv_wait_loads(1, vmcnt=0)
        if const_expr(DEBUG_DUMP_KV_NOPE_DW):
            _kv_dump_nope_dw(1, n1, qo_start_i32)
        if ArithValue(is_rope_owner) == 0:
            _kv_cvt_store_tile_at(1, p_lds_kv_base_i32, n1, s1)

    # ===================================================================
    # QK GEMM: Phase A (Q pinned in q_vgpr) + Phase B (Q from LDS)
    # ===================================================================
    # Per spec Ch. 5 / 8.9:
    #   load_k_to_gpr(kv_dst, p_lds_kv, kRowOffset, kColOffset):
    #     row = lane % 16; col = (lane / 16) * 8
    #     row_bank_swap = ((row >> 2) & 1) << 5
    #     in_sb_byte = row * 64 + ((col*2) ^ row_bank_swap)
    #     addr = p_lds_kv + in_sb_byte
    #     fixed_offset = sub_block_byte_offset(kRowOffset/16, kColOffset/32)
    #     ds_read_b128 into `kv_dst` (4 vgprs)
    #
    # Phase A pair (i): kv_top = K[row 0..15, col 2i*32..+32], kv_bot = K[row 16..31, ...],
    #                   kv_alt_top = K[row 0..15, col (2i+1)*32..+32], etc.  Then:
    #                       mma(p_comp_lo, kv_top,     qP_i_0)
    #                       mma(p_comp_hi, kv_bot,     qP_i_0)
    #                       mma(p_comp_lo, kv_alt_top, qP_i_1)
    #                       mma(p_comp_hi, kv_alt_bot, qP_i_1)
    # qP_i_j = q_vgpr split into 8 base tiles of 4 vgprs each
    # (q_vgpr.split(4)[2*i + j]).  Iter index in {0..7} runs through the
    # 8 Phase-A QK A-tiles covering Q[:, 0:256].
    #
    # Phase B is the same shape but Q comes from LDS via load_q_lds_to_gpr
    # into q_lds (v64..v71 = 2 tiles: q_k0, q_k1).  K col offsets are
    # NUM_QK_VGPR_ITER*kBK + 2i*kBK .. = 8*32+...  (continuing the K-axis
    # sweep into Q[:, 256:512] cols).

    # Pre-compute reader-side address pieces (lane-private, identical for
    # every load_k_to_gpr / load_q_lds_to_gpr call this iteration).
    qk_row_i32 = _raw(ArithValue(lane_idx_i32) % fx.Int32(16))
    qk_col_i32 = _raw((ArithValue(lane_idx_i32) // fx.Int32(16)) * fx.Int32(8))
    qk_row_bank_swap_i32 = _raw(
        ((ArithValue(qk_row_i32) // fx.Int32(4)) & fx.Int32(1)) * fx.Int32(32)
    )
    # in_sb_byte = row * 64 + ((col * 2) XOR row_bank_swap)
    qk_in_sb_byte_i32 = _raw(
        ArithValue(qk_row_i32) * fx.Int32(KV_SUB_BLOCK_COLS * 2)
        + (ArithValue(qk_col_i32) * fx.Int32(2) ^ ArithValue(qk_row_bank_swap_i32))
    )

    def _pinned_ds_read_b128(dst_range, addr_i32, byte_offset):
        """``ds_read_b128 v[lo:hi], addr offset:byte_offset`` writing into
        a 4-vgpr pinned range.

        Mirrors ``_pinned_ds_read_tr_b16`` -- forces the destination of the
        ds_read to a fixed pinned vgpr quartet (rather than letting LLVM
        allocate SSA dwords into v[64..255], which would clobber our pinned
        regions like q_lds / q_vgpr).
        """
        if dst_range.size != 4:
            raise ValueError(
                f"ds_read_b128 needs a size-4 dst; got {dst_range.size}"
            )
        off_str = f"offset:{byte_offset}" if byte_offset != 0 else ""
        asm = f"ds_read_b128 {dst_range.asm_name}, $0 {off_str}"
        llvm.inline_asm(
            None, [_raw(addr_i32)], asm, "v",
            has_side_effects=True,
        )

    def _load_k_lds_to_gpr(dst_range, p_lds_kv_base_i32, k_row_offset, k_col_offset):
        """ds_read_b128 of one 16x32 K sub-block into pinned ``dst_range``
        (4 vgprs).

        ``k_row_offset`` in {0, 16} -> row_tile ∈ {0, 1}.
        ``k_col_offset`` multiple of 32 in [0, 512) -> col_tile ∈ {0..15}.
        """
        if const_expr(DEBUG_FAKE_K_LDS_AS_ONES):
            for i in range_constexpr(4):
                v_idx = dst_range.lo + i
                llvm.inline_asm(
                    None, [], f"v_mov_b32 v[{v_idx}], 0x3F803F80",
                    "", has_side_effects=True,
                )
            return
        row_tile = k_row_offset // 16
        col_tile = k_col_offset // 32
        # sub_block_byte_offset(row_tile, col_tile) = (col_tile*2 + row_tile) * 1024
        fixed_off = (col_tile * 2 + row_tile) * KV_SUB_BLOCK_BYTES

        addr_i32 = _raw(
            ArithValue(p_lds_kv_base_i32) + ArithValue(qk_in_sb_byte_i32)
        )
        _pinned_ds_read_b128(dst_range, addr_i32, fixed_off)

    def _debug_dump_kv_top(qo_start_i32, p_lds_kv_base_i32):
        """Dump TWO views of K LDS for the same per-lane address:
          cols [0..16)  bf16  = pinned KV_TOP after _pinned_ds_read_b128
          cols [16..32) bf16  = SSA _lds_load_b128 from the SAME addr
        If the two match -> read mechanism is consistent.
        If they differ -> _pinned_ds_read_b128 returns different bytes.
        """
        head_row_i32 = _raw(
            ArithValue(warp_idx_i32) * fx.Int32(16)
            + ArithValue(lane_idx_i32) // fx.Int32(4)
        )
        lane_byte_base_i32 = _raw(
            (ArithValue(lane_idx_i32) % fx.Int32(4)) * fx.Int32(16)
        )
        row_base_byte = _raw(
            ArithValue(head_row_i32) * fx.Int32(V_HEAD_DIM * 2)
        )
        batch_byte_off = _raw(
            ArithValue(qo_start_i32)
            * fx.Int32(NUM_QO_HEADS * V_HEAD_DIM * 2)
        )
        # 1) Dump pinned KV_TOP (already written by _pinned_ds_read_b128 caller)
        for i in range_constexpr(4):
            slot = pv.PinnedRange(KV_RANGE.lo + i, KV_RANGE.lo + i)
            dw_i32 = pv.read_pinned(slot, as_i64=False)
            byte_off = _raw(
                ArithValue(row_base_byte)
                + ArithValue(lane_byte_base_i32)
                + fx.Int32(i * 4)
            )
            buffer_ops.buffer_store(
                dw_i32, final_output_rsrc, byte_off,
                offset_is_bytes=True, soffset_bytes=batch_byte_off,
            )
        # 2) Do an independent _lds_load_b128 from the SAME per-lane addr the
        # QK reader used.  Then dump that SSA value to cols [16..32) of the
        # same head row.
        addr_i32 = _raw(
            ArithValue(p_lds_kv_base_i32) + ArithValue(qk_in_sb_byte_i32)
        )
        ssa_dws = _lds_load_b128(addr_i32, byte_offset=0)
        ssa_vec = Vec(ssa_dws)
        for i in range_constexpr(4):
            byte_off = _raw(
                ArithValue(row_base_byte)
                + ArithValue(lane_byte_base_i32)
                + fx.Int32(16 * 2)            # offset by 16 bf16 = 32 bytes
                + fx.Int32(i * 4)
            )
            buffer_ops.buffer_store(
                _raw(ssa_vec[i]), final_output_rsrc, byte_off,
                offset_is_bytes=True, soffset_bytes=batch_byte_off,
            )

    def _load_q_lds_to_gpr(dst_range, col_tile):
        """ds_read_b128 of one 16x32 bf16 Q sub-block from p_lds_q into
        pinned ``dst_range`` (4 vgprs).  ``col_tile`` in [0, 8) (selects
        col-tile inside Q[:, 256:512]'s 8-col-tile grid).
        """
        if const_expr(DEBUG_FAKE_Q_LDS_AS_ONES):
            # Bypass Phase 2 + ds_read entirely.  v_mov 0x3F803F80 (= bf16
            # pair {1.0, 1.0}) directly into each pinned dst vgpr.
            for i in range_constexpr(4):
                v_idx = dst_range.lo + i
                llvm.inline_asm(
                    None, [],
                    f"v_mov_b32 v[{v_idx}], 0x3F803F80",
                    "", has_side_effects=True,
                )
            return
        # Reader Site-C XOR same as load_k_to_gpr.
        # Wave-major: per-warp base = p_lds_q + warp * Q_WARP_FINAL_BYTES
        lds_base_i32 = _i32(lds_base_idx)
        addr_i32 = _raw(
            ArithValue(lds_base_i32) + fx.Int32(P_LDS_Q)
            + ArithValue(warp_idx_i32) * fx.Int32(Q_WARP_FINAL_BYTES)
            + ArithValue(qk_in_sb_byte_i32)
        )
        fixed_off = col_tile * Q_SUB_BLOCK_BYTES
        _pinned_ds_read_b128(dst_range, addr_i32, fixed_off)

    def _do_qk_gemm(p_lds_kv_curr_i32, qo_start_i32=None):
        """Issue all 8 QK pairs (Phase A x4 + Phase B x4).

        Accumulates **in place** into pinned P_COMP_LO (v120..v123) and
        P_COMP_HI (v124..v127).  Returns None -- the caller reads p_comp
        back from its pinned slot via `pv.read_pinned(...)` for softmax.

        This was the source of the v_mov_b64 v[76..83], v[120..127]
        scratch-spill bug: the prior return_ssa=True form emitted 2
        v_mov_b64 trampolines per mfma (=64 across 8 pairs x 4 mfmas)
        that LLVM allocated into v[64..255] -- corrupting q_vgpr.
        return_ssa=False here drops those trampolines entirely.

        Per-pair structure: 4 K reads + 4 mfmas, same as HK.

        Q reads happen *in-place* on the pinned VGPRs (Phase A: q_vgpr
        v72..v103 from Phase 1 cvts; Phase B: q_lds v64..v71 from
        load_q_lds_to_gpr below).  K is moved into kv/kv_alt every pair.
        """
        q_k0_range, q_k1_range = Q_LDS_TILES

        # Two separate accumulators (P_COMP_LO, P_COMP_HI); each needs
        # its OWN first-mfma init.  Caller passes is_first=True only on
        # the very first mfma to that particular d.
        def _qk_mfma(a_pin, b_pin, d_pin, a_src, b_src, is_first):
            if const_expr(is_first):
                pv.pinned_mfma_bf16_init(
                    a_pin, b_pin, d_pin, a_src, b_src, return_ssa=False,
                )
            else:
                pv.pinned_mfma_bf16(
                    a_pin, b_pin, d_pin, a_src, b_src, None,
                    return_ssa=False,
                )

        if const_expr(DEBUG_DUMP_KV_TOP):
            # Load chunk 0 K into KV_TOP, dump, skip the rest.
            _load_k_lds_to_gpr(KV_TOP, p_lds_kv_curr_i32, 0, 0)
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            _debug_dump_kv_top(qo_start_i32, p_lds_kv_curr_i32)
            return

        # ---- Phase A: 4 pairs over q_vgpr (Q[:, 0:256]) ----
        # K is loaded *directly into pinned KV_TOP/KV_BOT/KV_ALT_TOP/KV_ALT_BOT*
        # so the mfma reads pinned A in place (a_src=None).  Letting LLVM
        # allocate the K dwords as SSA would put them into v[64..71] and
        # clobber q_lds.
        if const_expr(not DEBUG_QK_PHASE_B_ONLY):
            for pair in range_constexpr(4):
                k_col_0 = pair * 2 * BLOCK_K               # 0/64/128/192
                k_col_1 = k_col_0 + BLOCK_K                # 32/96/160/224

                _load_k_lds_to_gpr(KV_TOP,     p_lds_kv_curr_i32, 0,  k_col_0)
                _load_k_lds_to_gpr(KV_BOT,     p_lds_kv_curr_i32, 16, k_col_0)
                _load_k_lds_to_gpr(KV_ALT_TOP, p_lds_kv_curr_i32, 0,  k_col_1)
                _load_k_lds_to_gpr(KV_ALT_BOT, p_lds_kv_curr_i32, 16, k_col_1)

                rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
                rocdl.sched_barrier(0)

                q_0 = Q_VGPR_TILES[pair * 2 + 0]
                q_1 = Q_VGPR_TILES[pair * 2 + 1]

                is_first = (pair == 0)
                _qk_mfma(KV_TOP,     q_0, P_COMP_LO, None, None, is_first)
                _qk_mfma(KV_BOT,     q_0, P_COMP_HI, None, None, is_first)
                _qk_mfma(KV_ALT_TOP, q_1, P_COMP_LO, None, None, False)
                _qk_mfma(KV_ALT_BOT, q_1, P_COMP_HI, None, None, False)

        # ---- Phase B: 4 pairs over q_lds (Q[:, 256:512]) ----
        # Q is loaded directly into the pinned q_lds tiles (q_k0_range /
        # q_k1_range); K direct-into KV_* as in Phase A.
        if const_expr(DEBUG_QK_PHASE_A_ONLY):
            return
        for pair in range_constexpr(4):
            k_col_0 = (NUM_QK_VGPR_ITER + pair * 2 + 0) * BLOCK_K
            k_col_1 = (NUM_QK_VGPR_ITER + pair * 2 + 1) * BLOCK_K

            _load_q_lds_to_gpr(q_k0_range, pair * 2 + 0)
            _load_q_lds_to_gpr(q_k1_range, pair * 2 + 1)

            _load_k_lds_to_gpr(KV_TOP,     p_lds_kv_curr_i32, 0,  k_col_0)
            _load_k_lds_to_gpr(KV_BOT,     p_lds_kv_curr_i32, 16, k_col_0)
            _load_k_lds_to_gpr(KV_ALT_TOP, p_lds_kv_curr_i32, 0,  k_col_1)
            _load_k_lds_to_gpr(KV_ALT_BOT, p_lds_kv_curr_i32, 16, k_col_1)

            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            rocdl.sched_barrier(0)

            # Phase B is always accumulate (Phase A already inited P_COMP).
            # Exception: when DEBUG_QK_PHASE_B_ONLY is on, Phase A is skipped
            # so Phase B's first pair must init.
            phase_b_is_first = const_expr(DEBUG_QK_PHASE_B_ONLY) and (pair == 0)
            _qk_mfma(KV_TOP,     q_k0_range, P_COMP_LO, None, None, phase_b_is_first)
            _qk_mfma(KV_BOT,     q_k0_range, P_COMP_HI, None, None, phase_b_is_first)
            _qk_mfma(KV_ALT_TOP, q_k1_range, P_COMP_LO, None, None, False)
            _qk_mfma(KV_ALT_BOT, q_k1_range, P_COMP_HI, None, None, False)

        # Result lives pinned in P_COMP_LO + P_COMP_HI = v[120..127].

    # ===================================================================
    # Online softmax (spec Ch. 9)
    # ===================================================================
    fm_no_inf = (
        arith.FastMathFlags.nnan | arith.FastMathFlags.nsz
        | arith.FastMathFlags.arcp | arith.FastMathFlags.contract
        | arith.FastMathFlags.afn | arith.FastMathFlags.reassoc
    )

    def _max3_f32(a, b, c):
        return llvm.inline_asm(
            T.f32, [_raw(a), _raw(b), _raw(c)],
            "v_max3_f32 $0, $1, $2, $3", "=v,v,v,v",
            has_side_effects=False,
        )

    def _shfl_xor_f32(val, offset):
        val_i32 = _raw(ArithValue(val).bitcast(T.i32))
        peer_i32 = ArithValue(val_i32).shuffle_xor(offset, WARP_SIZE)
        return _raw(ArithValue(peer_i32).bitcast(T.f32))

    def _warp_reduce_max(val):
        """Butterfly max reduce across the 4 lanes-per-row group (strides 32, 16)."""
        v = val
        for sh in [32, 16]:
            peer = _shfl_xor_f32(v, sh)
            v = _raw(arith.maximumf(v, peer, fastmath=fm_no_inf))
        return v

    def _warp_reduce_add(val):
        v = val
        for sh in [32, 16]:
            peer = _shfl_xor_f32(v, sh)
            v = _raw(arith.addf(v, peer, fastmath=fm_no_inf))
        return v

    def _softmax_and_pack(p_comp_lo_val, p_comp_hi_val,
                          row_max_old, row_sum_e_old,
                          kv_tile_start_i32, kv_end_i32,
                          is_first_iter, check_boundary,
                          qo_start_i32=None):
        """Full online softmax for one KV tile.

        Inputs:
          p_comp_lo_val, p_comp_hi_val : SSA f32x4 each (QK output for this
                                          warp's N=0..15 / N=16..31 halves)
          row_max_old, row_sum_e_old   : SSA f32 each (running stats)
          is_first_iter                : Python bool (compile-time)
          check_boundary               : Python bool (compile-time)

        Output:
          (row_max_new, row_sum_e_new, rescale) — all SSA f32.

        Side effect: writes the post-exp bf16 P-operand into pinned p_mfma
        (v120..v123) ready for PV mfma to consume in stage 9.
        """
        # 1) Extract 8 fp32 values per lane (4 from lo, 4 from hi).
        lo_vec = Vec(p_comp_lo_val)
        hi_vec = Vec(p_comp_hi_val)
        p_vals = [_raw(lo_vec[i]) for i in range(4)] + [_raw(hi_vec[i]) for i in range(4)]

        # 2) softmax_scale_p: multiply each by softmax_scale.  Boundary mask
        # for OOB cols on partial last tile (set to -inf).  HK's layout for
        # the 8 vals: lane's col_0_group covers cols [col_0_idx*4,+4) (vgprs
        # +0..+3), col_1_group covers [col_0_idx*4 + 16, +20) (vgprs +4..+7).
        # col_0_idx = lane / 16.
        ss = _raw(softmax_scale)
        scaled = [_raw(arith.mulf(v, ss, fastmath=fm_no_inf)) for v in p_vals]

        if const_expr(check_boundary):
            col_0_idx_i32 = _raw(ArithValue(lane_idx_i32) // fx.Int32(16))
            kv_tile_start_idx = _idx(kv_tile_start_i32)
            kv_end_idx = _idx(kv_end_i32)
            c_neg_inf = _raw(fx.Float32(float('-inf')))
            for i in range_constexpr(8):
                # Vgpr layout: i in [0,4) -> sub_offset = i; i in [4,8) -> 16 + (i-4)
                if const_expr(i < 4):
                    sub_off = i
                else:
                    sub_off = 16 + (i - 4)
                # col_idx = col_0_idx*4 + sub_off + kv_tile_start
                col_idx_i32 = _raw(
                    ArithValue(col_0_idx_i32) * fx.Int32(4)
                    + fx.Int32(sub_off)
                    + ArithValue(kv_tile_start_i32)
                )
                # is_oob = col_idx >= kv_end
                is_oob = _raw(ArithValue(col_idx_i32) >= ArithValue(kv_end_i32))
                scaled[i] = _raw(ArithValue(is_oob).select(c_neg_inf, scaled[i]))

        # 3) local max over 8 values via v_max3 ladder (4 instructions).
        tmp0 = _max3_f32(scaled[0], scaled[1], scaled[2])
        tmp1 = _max3_f32(scaled[3], scaled[4], scaled[5])
        tmp2 = _raw(arith.maximumf(scaled[6], scaled[7], fastmath=fm_no_inf))
        local_max = _max3_f32(tmp0, tmp1, tmp2)

        # 4) warp_reduce max across the 4-lane M-group.
        local_max = _warp_reduce_max(local_max)


        # 5) new_row_max + rescale.
        c_one = _raw(fx.Float32(1.0))
        c_log2e = _raw(fx.Float32(LOG2E))
        if const_expr(is_first_iter):
            new_row_max = local_max
            rescale = c_one
        else:
            new_row_max = _raw(arith.maximumf(local_max, row_max_old, fastmath=fm_no_inf))
            diff = _raw(arith.subf(row_max_old, new_row_max, fastmath=fm_no_inf))
            arg = _raw(arith.mulf(diff, c_log2e, fastmath=fm_no_inf))
            rescale = rocdl.exp2(T.f32, arg)

        # 6) softmax_p1: p[i] = exp2((p[i] - new_row_max) * log2e); sum + reduce.
        neg_max = _raw(arith.mulf(new_row_max, _raw(fx.Float32(-1.0)),
                                  fastmath=fm_no_inf))
        p_exp = [None] * 8
        local_sum = _raw(fx.Float32(0.0))
        for i in range_constexpr(8):
            diff = _raw(arith.addf(scaled[i], neg_max, fastmath=fm_no_inf))
            arg = _raw(arith.mulf(diff, c_log2e, fastmath=fm_no_inf))
            p_exp[i] = rocdl.exp2(T.f32, arg)
            local_sum = _raw(arith.addf(local_sum, p_exp[i], fastmath=fm_no_inf))
        local_sum = _warp_reduce_add(local_sum)

        if const_expr(DEBUG_DUMP_LOCALMAX):
            # slot 0..3 = p_exp[0], p_exp[1], p_exp[2], p_exp[3]
            _debug_dump_localmax(p_exp[0], 0, qo_start_i32)
            _debug_dump_localmax(p_exp[1], 1, qo_start_i32)
            _debug_dump_localmax(p_exp[2], 2, qo_start_i32)
            _debug_dump_localmax(p_exp[3], 3, qo_start_i32)

        if const_expr(is_first_iter):
            row_sum_e_new = local_sum
        else:
            row_sum_e_new = _raw(arith.addf(
                _raw(arith.mulf(rescale, row_sum_e_old, fastmath=fm_no_inf)),
                local_sum, fastmath=fm_no_inf,
            ))

        # 7) Pack p_exp (8 fp32) -> p_mfma overlay (4 bf16x2 dwords in v120..v123).
        # Low-to-high pack order is hazard-free per spec Ch. 9.6:
        #   p_mfma[0] = cvt(p_exp[0], p_exp[1])  -- reads p_comp[0,1], writes v120
        #   p_mfma[1] = cvt(p_exp[2], p_exp[3])  -- reads p_comp[2,3], writes v121
        #   ...
        # The pinned helper does the cvt with destination v120..v123 explicitly
        # via inline asm `v_cvt_pk_bf16_f32 v[N], $0, $1`.
        p_mfma_sub = P_MFMA_RANGE.split(1)   # 4 singleton ranges v120, v121, v122, v123
        for i in range_constexpr(4):
            pv.pinned_cvt_pk_bf16_f32(
                p_mfma_sub[i],
                p_exp[2 * i + 0],
                p_exp[2 * i + 1],
            )

        return new_row_max, row_sum_e_new, rescale

    # ===================================================================
    # PV gemm + oaccu rescale (spec Ch. 10)
    # ===================================================================
    # PV computes O^T = V^T * P^T = mma_ABt(oaccu, kv, p_mfma), with
    # `kv` holding V^T values in mfma A-operand layout.  v40 PV is
    # `v_mfma_f32_16x16x32_f16` (bf16 inputs, A/B = 4 vgprs each, D = 4).
    #
    # Per iter (16 total): 4 ds_read_b64_tr_b16 fill kv (8 vgprs, = 2
    # sub-tiles of 16x32 bf16 = kv_top + kv_bot), then 2 mfmas write
    # adjacent oaccu sub-tiles.
    #
    # Stage 9 = single-buffered (Gen.1 form); dual-buffer V deviation
    # (overlap next-iter ds_read with current mfma via pv_v_aux carrier)
    # is left as a follow-up perf tune (stage 9.5).

    # ---- ds_read_b64_tr_b16 pinned-DST helper ----
    def _pinned_ds_read_tr_b16(dst_range, addr_i32, byte_offset):
        """``ds_read_b64_tr_b16 dst, addr offset:byte_offset`` writing into
        a 2-vgpr pinned range.

        Per spec Ch. 10.2: this HW-transpose-reads 4 bf16/lane (= 2 dwords)
        such that the lane layout matches the mfma A-operand for a 16x16
        sub-tile of V^T.
        """
        if dst_range.size != 2:
            raise ValueError(
                f"ds_read_b64_tr_b16 needs a size-2 dst; got {dst_range.size}"
            )
        # Move addr into an SGPR-class operand?  The intrinsic is VMEM-style;
        # 'addr' is a vgpr (per-lane address) and the offset is an immediate.
        # Asm form: ds_read_b64_tr_b16 v[N:N+1], <vgpr_addr> offset:<imm>
        # We must write the dst directly via inline asm — no SSA result.
        off_str = f"offset:{byte_offset}" if byte_offset != 0 else ""
        asm = f"ds_read_b64_tr_b16 {dst_range.asm_name}, $0 {off_str}"
        llvm.inline_asm(
            None, [_raw(addr_i32)], asm, "v",
            has_side_effects=True,
        )

    # Per-lane V-side address pieces (PV reader, spec Ch. 10.2 / cuh 1296-1300):
    #   row_in_sb = lane >> 2
    #   is_swz    = (row_in_sb >> 2) & 1
    #   in_sb     = row_in_sb * 64 + (lane & 3) * 8 + is_swz * kSwzDelta
    pv_row_in_sb_i32 = _raw(ArithValue(lane_idx_i32) // fx.Int32(4))
    pv_is_swz_i32 = _raw((ArithValue(pv_row_in_sb_i32) // fx.Int32(4)) & fx.Int32(1))

    def _pv_in_sb_byte(k_col_offset):
        """Per-lane in-sub-block byte offset for a load_transposed_v call.

        ``k_col_offset`` (multiple of 16) selects the 16-col half within
        a 32-col sub-block.  kSwzDelta = -32 if (k_col_offset & 32), else +32.
        """
        k_col_in_sb_bytes = (k_col_offset % KV_SUB_BLOCK_COLS) * 2
        swz_delta = -32 if (k_col_in_sb_bytes & 32) else +32
        return _raw(
            ArithValue(pv_row_in_sb_i32) * fx.Int32(KV_SUB_BLOCK_COLS * 2)
            + (ArithValue(lane_idx_i32) & fx.Int32(3)) * fx.Int32(8)
            + ArithValue(pv_is_swz_i32) * fx.Int32(swz_delta)
        )

    def _load_tr_v_lds_to_gpr(dst_range, p_lds_kv_base_i32, k_row_offset, k_col_offset):
        """Fill a 2-vgpr pinned range with one 16-row x 16-col transposed
        V sub-tile via ds_read_b64_tr_b16.

        k_row_offset in {0, 16} -> row_tile ∈ {0, 1}.
        k_col_offset multiple of 16 in [0, kVoHeadDim=512).
        """
        row_tile = k_row_offset // 16
        col_tile = k_col_offset // 32
        col_in_sb_bytes = (k_col_offset % 32) * 2
        fixed_off = ((col_tile * 2 + row_tile) * KV_SUB_BLOCK_BYTES
                     + col_in_sb_bytes)
        in_sb = _pv_in_sb_byte(k_col_offset)
        addr_i32 = _raw(
            ArithValue(p_lds_kv_base_i32) + ArithValue(in_sb)
        )
        _pinned_ds_read_tr_b16(dst_range, addr_i32, fixed_off)

    def _do_pv_gemm(p_lds_kv_curr_i32, rescale_val,
                    is_first_iter):
        """Run 16 PV mfmas accumulating into pinned oaccu (v128..v255).

        On non-first iters, every oaccu sub-tile is first multiplied by
        ``rescale`` (= α from softmax).  Spec Ch. 10.3's interleaved
        rescale-during-mfma is a perf optimization; here we do the simple
        form (full rescale loop before PV).  Gen.2 dual-buffer V is also
        TBD.

        ``is_first_iter`` skips the rescale entirely (oaccu starts at 0).
        """
        # Entry fence: pin PV reads of p_mfma (set by softmax) and
        # ensures any in-place oaccu update happens after softmax's pack.
        rocdl.sched_barrier(0)

        # ---- oaccu rescale (skip on first iter) ----
        # 32 oaccu sub-tiles of 4 vgprs each = 128 vgprs total.  Each rescale
        # is v_pk_mul_f32 of a 2-vgpr pair * {rescale, rescale}.
        if const_expr(not is_first_iter):
            # Drain previous iter's PV mfma writes to oaccu before SIMD reads
            # (MFMA -> SIMD-read hazard on v[128..255]).
            llvm.inline_asm(None, [], "s_nop 15\n\t s_nop 15", "",
                            has_side_effects=True)
            # Pack rescale into a packed-fp32 pair (i64).
            rescale_bits = _raw(ArithValue(rescale_val).bitcast(T.i32))
            rescale_pk = _raw(
                ArithValue(rescale_bits).extui(T.i64)
                | (ArithValue(rescale_bits).extui(T.i64) << 32)
            )
            for sub in range_constexpr(32):
                # Two pairs per 4-vgpr sub-tile.
                pair0 = pv.PinnedRange(
                    OACCU_RANGE.lo + sub * 4 + 0,
                    OACCU_RANGE.lo + sub * 4 + 1,
                )
                pair1 = pv.PinnedRange(
                    OACCU_RANGE.lo + sub * 4 + 2,
                    OACCU_RANGE.lo + sub * 4 + 3,
                )
                pv.pinned_v_mul_f32_pair(pair0, rescale_pk)
                pv.pinned_v_mul_f32_pair(pair1, rescale_pk)

        # ---- PV mfma loop ----
        # 16 iters; each iter covers 32 V-cols = 2 mfma A-tiles (top + bot).
        # Per iter:
        #   load_v(kv_top, row=0,  col=iter*32+0)   [-> v112:115]
        #   load_v(kv_bot, row=16, col=iter*32+0)   [-> v116:119]
        #   load_v(kv_top2, row=0,  col=iter*32+16) [-> v112:115 -- reuse]
        #   load_v(kv_bot2, row=16, col=iter*32+16) [-> v116:119]
        # Wait, that double-uses kv.  HK spreads them into kv (v112..v119)
        # = 4 sub-tiles via the 4 ds_read calls per iter; let me re-check.
        # cuh lines 862-872: kv_begin+0/+2/+4/+6 for the 4 reads.  So kv
        # is 8 vgprs split into 4 sub-tiles of 2 vgprs each.  Good.
        kv_subs = KV_RANGE.split(2)        # 4 sub-tiles of 2 vgprs each
        oaccu_subs_local = OACCU_SUBS      # 32 sub-tiles of 4 vgprs each
        for it in range_constexpr(NUM_PV_ITER):
            kcol_a = it * BLOCK_N + 0
            kcol_b = it * BLOCK_N + MFMA_N

            _load_tr_v_lds_to_gpr(kv_subs[0], p_lds_kv_curr_i32, 0,  kcol_a)
            _load_tr_v_lds_to_gpr(kv_subs[1], p_lds_kv_curr_i32, 16, kcol_a)
            _load_tr_v_lds_to_gpr(kv_subs[2], p_lds_kv_curr_i32, 0,  kcol_b)
            _load_tr_v_lds_to_gpr(kv_subs[3], p_lds_kv_curr_i32, 16, kcol_b)

            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            rocdl.sched_barrier(0)

            # kv_top = sub[0]+sub[1] (4 vgprs total, = one 16x32 mfma A-tile)
            # kv_bot = sub[2]+sub[3] (= second 16x32 mfma A-tile)
            kv_top_range = pv.PinnedRange(KV_RANGE.lo + 0, KV_RANGE.lo + 3)
            kv_bot_range = pv.PinnedRange(KV_RANGE.lo + 4, KV_RANGE.lo + 7)
            oaccu_a = oaccu_subs_local[it * 2 + 0]   # 4 vgprs
            oaccu_b = oaccu_subs_local[it * 2 + 1]

            if const_expr(is_first_iter):
                # 3-arg init form: D = A * B (C=0), leave result in pinned d.
                pv.pinned_mfma_bf16_init(
                    kv_top_range, P_MFMA_RANGE, oaccu_a, None, None,
                    return_ssa=False,
                )
                pv.pinned_mfma_bf16_init(
                    kv_bot_range, P_MFMA_RANGE, oaccu_b, None, None,
                    return_ssa=False,
                )
            else:
                # 4-arg accumulate, in place: a/b/acc all from pinned slots.
                pv.pinned_mfma_bf16(
                    kv_top_range, P_MFMA_RANGE, oaccu_a, None, None, None,
                )
                pv.pinned_mfma_bf16(
                    kv_bot_range, P_MFMA_RANGE, oaccu_b, None, None, None,
                )

        # End-of-PV scheduling fence: pinned oaccu writes done in this
        # function must not be reordered past the OMgr epilogue's reads.
        rocdl.sched_barrier(0)

    # ===================================================================
    # OManager V3 epilogue (bf16 output, spec Ch. 11)
    # ===================================================================
    # Per-warp 2112-byte bounce LDS slot.  Overlays p_lds_kv_next (= the
    # second pong, dead on the global last iter).  Per-call: pack 16 fp32
    # oaccu vgprs to 8 bf16 dwords (4 b16-pair), ds_write_b64 x4 in mfma
    # layout, ds_read_b128 x2 with inverse sb8 perm on col, store dwordx4 x2.
    #
    # 8 calls per work-item cover oaccu's 128 vgprs in 16-vgpr strides
    # (each call = 1 wave-tile = 64 V-cols).

    O16_NUM_ROWS = 16
    O16_NUM_COLS = 64
    O16_PAD_ELEM_PER_2_ROWS = 4
    O16_NUM_ELEM_PER_PAD_2_ROWS = 2 * O16_NUM_COLS + O16_PAD_ELEM_PER_2_ROWS  # 132
    O16_VRAM_ST_ELEM_PER_LANE = 8                  # buffer_store_dwordx4 = 8 bf16
    O16_VRAM_ST_LANE_PER_ROW = O16_NUM_COLS // O16_VRAM_ST_ELEM_PER_LANE     # 8
    O16_VRAM_ST_ROWS_PER_RND = WARP_SIZE // O16_VRAM_ST_LANE_PER_ROW          # 8
    O16_VRAM_ST_NUM_RNDS = O16_NUM_ROWS // O16_VRAM_ST_ROWS_PER_RND           # 2
    O16_LDS_LD_OFFSET_DELTA_BYTES = (
        (O16_VRAM_ST_ROWS_PER_RND // 2) * O16_NUM_ELEM_PER_PAD_2_ROWS * 2     # 1056
    )
    O16_VRAM_ST_OFFSET_DELTA_BYTES = (
        O16_VRAM_ST_ROWS_PER_RND * V_HEAD_DIM * 2                             # 8192
    )
    O16_BYTES_PER_WARP_LDS = (O16_NUM_ROWS // 2) * O16_NUM_ELEM_PER_PAD_2_ROWS * 2  # 2112
    O16_MFMA_COLS = MFMA_N                                                    # 16
    O16_MFMA_BYTE_STRIDE = O16_MFMA_COLS * 2                                  # 32

    def _omgr16_output_pair(iter_idx, p_lds_o_base_i32, qo_start_i32):
        """OManager16bitsV3::output_to_vram_pair.

        ``iter_idx`` in [0, 8): selects which 16-vgpr slice of oaccu
        (= which wave-tile col-range).
        """
        oaccu_base = OACCU_RANGE.lo + iter_idx * 16            # v128 + iter*16
        col_off_bytes = iter_idx * 2 * BLOCK_N * 2             # iter * 64 * 2 = 128 B/iter

        # ---- 1) Pack fp32 -> bf16 (8 cvts per call) ----
        # HK pattern: 8 packs (oaccu[2k], oaccu[2k+1]) -> b16_pair_m{k/2}[k%2],
        # where the b16_pair locals are allocator-managed scratch.  Writing
        # back into oaccu slots is unsafe -- the next iter would read its
        # already-overwritten source.  Pack into SSA i32 dwords here; the
        # ds_write_b64 below assembles them into i64 pairs.
        b16_dws = []
        for k in range_constexpr(8):
            src_pair = pv.PinnedRange(oaccu_base + 2 * k, oaccu_base + 2 * k + 1)
            b16_dws.append(pv.pinned_cvt_pk_bf16_f32_from_pinned_to_ssa(src_pair))

        # ---- 2) ds_write_b64 x 4 (each writes 2 dwords = 4 bf16x2) ----
        # Lane mapping: row_st = lane % 16, col_st_base = (lane / 16) * 4
        # v_off_lds_st = ((row_st/2)*132 + (row_st%2)*64 + col_st_base) * 2 B
        row_lds_st_i32 = _raw(ArithValue(lane_idx_i32) % fx.Int32(O16_NUM_ROWS))
        col_lds_st_base_i32 = _raw(
            (ArithValue(lane_idx_i32) // fx.Int32(O16_NUM_ROWS)) * fx.Int32(4)
        )
        v_off_lds_st_i32 = _raw(
            ((ArithValue(row_lds_st_i32) // fx.Int32(2))
             * fx.Int32(O16_NUM_ELEM_PER_PAD_2_ROWS)
             + (ArithValue(row_lds_st_i32) % fx.Int32(2)) * fx.Int32(O16_NUM_COLS)
             + ArithValue(col_lds_st_base_i32)) * fx.Int32(2)
        )
        addr_st_i32 = _raw(
            ArithValue(p_lds_o_base_i32)
            + ArithValue(warp_idx_i32) * fx.Int32(O16_BYTES_PER_WARP_LDS)
            + ArithValue(v_off_lds_st_i32)
        )

        # Issue 4 ds_write_b64.  Each writes a vgpr-pair (2 dwords)
        # constructed from SSA b16_dws[2m, 2m+1] -- LLVM will assign them
        # to a consecutive vgpr pair via the "v" constraint with i64 type.
        for m in range_constexpr(4):
            lo_dw = b16_dws[2 * m]
            hi_dw = b16_dws[2 * m + 1]
            pair_i64 = _pack_i32_pair_to_i64(lo_dw, hi_dw)
            byte_off = m * O16_MFMA_BYTE_STRIDE
            asm = (
                f"ds_write_b64 $0, $1"
                + (f" offset:{byte_off}" if byte_off else "")
            )
            llvm.inline_asm(
                None, [addr_st_i32, pair_i64], asm, "v,v",
                has_side_effects=True,
            )

        # ---- 3) Wait, then 2x ds_read_b128 with INVERSE sb8 perm ----
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
        # lane_in_row = lane % 8 (0..7), row_lds_ld = lane / 8
        lane_in_row_i32 = _raw(ArithValue(lane_idx_i32) % fx.Int32(O16_VRAM_ST_LANE_PER_ROW))
        row_lds_ld_i32 = _raw(ArithValue(lane_idx_i32) // fx.Int32(O16_VRAM_ST_LANE_PER_ROW))
        # lds_subtile = ((lane_in_row & 1) << 2) | ((lane_in_row & 6) >> 1)
        lds_subtile_i32 = _raw(
            ((ArithValue(lane_in_row_i32) & fx.Int32(1)) * fx.Int32(4))
            | ((ArithValue(lane_in_row_i32) & fx.Int32(6)) // fx.Int32(2))
        )
        col_lds_ld_i32 = _raw(ArithValue(lds_subtile_i32) * fx.Int32(8))
        v_off_lds_ld_i32 = _raw(
            ((ArithValue(row_lds_ld_i32) // fx.Int32(2))
             * fx.Int32(O16_NUM_ELEM_PER_PAD_2_ROWS)
             + (ArithValue(row_lds_ld_i32) % fx.Int32(2)) * fx.Int32(O16_NUM_COLS)
             + ArithValue(col_lds_ld_i32)) * fx.Int32(2)
        )
        addr_ld_i32 = _raw(
            ArithValue(p_lds_o_base_i32)
            + ArithValue(warp_idx_i32) * fx.Int32(O16_BYTES_PER_WARP_LDS)
            + ArithValue(v_off_lds_ld_i32)
        )

        # HK reuses oaccu_base..+7 as the ds_read destination (those
        # source vgprs are dead after the pack).  We write the read result
        # into pinned oaccu slots oaccu_base..oaccu_base+7 via pinned-DST
        # ds_read_b128 (mirrors load_k_to_gpr's pattern).
        # Round 0: oaccu_base..+3 ; Round 1: oaccu_base+4..+7
        for round_idx in range_constexpr(O16_VRAM_ST_NUM_RNDS):
            dst_lo = oaccu_base + 4 * round_idx
            dst_hi = oaccu_base + 4 * round_idx + 3
            byte_off = round_idx * O16_LDS_LD_OFFSET_DELTA_BYTES
            asm = (
                f"ds_read_b128 v[{dst_lo}:{dst_hi}], $0"
                + (f" offset:{byte_off}" if byte_off else "")
            )
            llvm.inline_asm(
                None, [addr_ld_i32], asm, "v",
                has_side_effects=True,
            )

        # ---- 4) buffer_store_dwordx4 x 2 (coalesced VRAM stores) ----
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
        # row_vram_st = row_lds_ld + warp * 16
        # col_vram_st = lane_in_row * 8
        # vram_offset = (row_vram_st * V_HEAD_DIM + col_vram_st) * 2 + col_off_bytes
        row_vram_st_i32 = _raw(
            ArithValue(row_lds_ld_i32)
            + ArithValue(warp_idx_i32) * fx.Int32(O16_NUM_ROWS)
        )
        col_vram_st_i32 = _raw(
            ArithValue(lane_in_row_i32) * fx.Int32(O16_VRAM_ST_ELEM_PER_LANE)
        )
        # The buffer base needs adjustment for qo_start (per-batch offset).
        # vram_offset = (row * V_HEAD_DIM + col) * 2 + col_off_bytes
        # batch_off (added via soffset) = qo_start * NUM_QO_HEADS * V_HEAD_DIM * 2
        vram_off_round0_i32 = _raw(
            (ArithValue(row_vram_st_i32) * fx.Int32(V_HEAD_DIM)
             + ArithValue(col_vram_st_i32)) * fx.Int32(2)
            + fx.Int32(col_off_bytes)
        )
        # We need to read the pinned oaccu slots (where the ds_read landed)
        # back into an SSA value before passing to buffer_store.  Use
        # read_pinned -- this v_mov_b32's them out.  For rounds 0/1 we read
        # 4 vgprs each.
        for round_idx in range_constexpr(O16_VRAM_ST_NUM_RNDS):
            src_lo = oaccu_base + 4 * round_idx
            src_hi = oaccu_base + 4 * round_idx + 3
            src_range = pv.PinnedRange(src_lo, src_hi)
            # Round 0 stores at v_off + 0, round 1 at + 8192 B.
            extra_off = round_idx * O16_VRAM_ST_OFFSET_DELTA_BYTES
            vram_off_i32 = _raw(
                ArithValue(vram_off_round0_i32) + fx.Int32(extra_off)
            )
            # Read pinned vgprs as 4 separate i32 dwords via read_pinned.
            dws = pv.read_pinned(src_range)
            data_vec = Vec.from_elements(list(dws), fx.Int32)
            # qo_start_i32 batch offset folded into soffset_bytes.
            batch_soff = _raw(
                ArithValue(qo_start_i32)
                * fx.Int32(NUM_QO_HEADS * V_HEAD_DIM * 2)
            )
            buffer_ops.buffer_store(
                _raw(data_vec), final_output_rsrc,
                vram_off_i32,
                offset_is_bytes=True,
                soffset_bytes=batch_soff,
            )

    # -------------------------------------------------------------------
    # OManager 32-bit (fp32 split-output) — mirrors HK
    # OManager32bitsV4Gen1Swizzle::output_to_vram_pair.
    # -------------------------------------------------------------------
    # Layout per call (= 64 V-cols, 4 mfma sub-tiles):
    #   - 16 oaccu vgprs (4 per mfma sub-tile, fp32)
    #   - LDS: per-warp 16 rows x 68 elements (4-elem pad after each row);
    #          warp's slice = warp * O32_BYTES_PER_WARP
    #   - ds_write_b128 x4 (one per sub-tile, 64-byte stride along col axis)
    #   - ds_read_b128 x4 with 1088-byte LDS-row stride (4 rows x 68 elem x 4 B)
    #   - buffer_store_dwordx4 x4 with 8192-byte VRAM-row stride (4 rows x 512 cols x 4 B)
    O32_NUM_ROWS = 16
    O32_NUM_COLS = 64
    O32_PAD_ELEM_PER_ROW = 4
    O32_NUM_ELEM_PER_PAD_ROW = O32_NUM_COLS + O32_PAD_ELEM_PER_ROW       # 68
    O32_VRAM_ST_ELEM_PER_LANE = 4                                         # dwordx4 = 4 fp32
    O32_VRAM_ST_LANE_PER_ROW = O32_NUM_COLS // O32_VRAM_ST_ELEM_PER_LANE  # 16
    O32_VRAM_ST_ROWS_PER_RND = WARP_SIZE // O32_VRAM_ST_LANE_PER_ROW      # 4
    O32_VRAM_ST_NUM_RNDS = O32_NUM_ROWS // O32_VRAM_ST_ROWS_PER_RND       # 4
    O32_LDS_LD_OFFSET_DELTA_BYTES = (
        O32_VRAM_ST_ROWS_PER_RND * O32_NUM_ELEM_PER_PAD_ROW * 4           # 1088
    )
    O32_VRAM_ST_OFFSET_DELTA_BYTES = (
        O32_VRAM_ST_ROWS_PER_RND * V_HEAD_DIM * 4                         # 8192
    )
    O32_BYTES_PER_WARP_LDS = O32_NUM_ROWS * O32_NUM_ELEM_PER_PAD_ROW * 4  # 4352
    O32_MFMA_COLS = MFMA_N                                                # 16
    O32_MFMA_BYTE_STRIDE = O32_MFMA_COLS * 4                              # 64

    def _omgr32_output_pair(iter_idx, p_lds_o_base_i32, partial_qo_loc_i32):
        """OManager32bitsV4Gen1Swizzle::output_to_vram_pair (fp32 split path).

        Writes 16 oaccu fp32 vgprs (one wave-tile = 64 V-cols) to
        ``split_output[partial_qo_loc, warp*16 + row, col_off + lane*4]``
        via a per-warp LDS bounce that applies the inverse sb8 sub-tile
        permutation.

        ``iter_idx`` in [0, 8): selects the wave-tile col stripe.
        """
        oaccu_base = OACCU_RANGE.lo + iter_idx * 16        # v128 + iter*16
        col_off_bytes = iter_idx * 2 * BLOCK_N * 4         # iter * 64 * 4 = 256 B/iter

        # ---- 1) ds_write_b128 x 4 (one per mfma sub-tile, straight layout) ----
        # row_lds_st = lane % 16, col_lds_st_base = (lane/16) * 4
        # v_off_lds_st = (row_lds_st * 68 + col_lds_st_base) * 4 B
        row_lds_st_i32 = _raw(ArithValue(lane_idx_i32) % fx.Int32(O32_NUM_ROWS))
        col_lds_st_base_i32 = _raw(
            (ArithValue(lane_idx_i32) // fx.Int32(O32_NUM_ROWS)) * fx.Int32(4)
        )
        v_off_lds_st_i32 = _raw(
            (ArithValue(row_lds_st_i32) * fx.Int32(O32_NUM_ELEM_PER_PAD_ROW)
             + ArithValue(col_lds_st_base_i32)) * fx.Int32(4)
        )
        addr_st_i32 = _raw(
            ArithValue(p_lds_o_base_i32)
            + ArithValue(warp_idx_i32) * fx.Int32(O32_BYTES_PER_WARP_LDS)
            + ArithValue(v_off_lds_st_i32)
        )
        # 4 ds_write_b128: each writes 4 fp32 dwords from oaccu sub-tile m.
        for m in range_constexpr(4):
            src_lo = oaccu_base + 4 * m
            src_hi = oaccu_base + 4 * m + 3
            byte_off = m * O32_MFMA_BYTE_STRIDE
            asm = (
                f"ds_write_b128 $0, v[{src_lo}:{src_hi}]"
                + (f" offset:{byte_off}" if byte_off else "")
            )
            llvm.inline_asm(
                None, [addr_st_i32], asm, "v",
                has_side_effects=True,
            )

        # ---- 2) ds_read_b128 x 4 with inverse sb8 sub-tile perm ----
        # row_lds_ld = lane / 16, lane_in_row = lane % 16
        # data_subtile = lane_in_row >> 1  (0..7)
        # lds_subtile = ((data_subtile & 1) << 2) | ((data_subtile & 6) >> 1)
        # intra_off   = (lane_in_row & 1) * 4   (= 0 or 4)
        # col_lds_ld  = lds_subtile * 8 + intra_off
        row_lds_ld_i32 = _raw(
            ArithValue(lane_idx_i32) // fx.Int32(O32_VRAM_ST_LANE_PER_ROW)
        )
        lane_in_row_i32 = _raw(
            ArithValue(lane_idx_i32) % fx.Int32(O32_VRAM_ST_LANE_PER_ROW)
        )
        data_subtile_i32 = _raw(
            ArithValue(lane_in_row_i32) // fx.Int32(2)
        )
        lds_subtile_i32 = _raw(
            ((ArithValue(data_subtile_i32) & fx.Int32(1)) * fx.Int32(4))
            | ((ArithValue(data_subtile_i32) & fx.Int32(6)) // fx.Int32(2))
        )
        intra_off_i32 = _raw(
            (ArithValue(lane_in_row_i32) & fx.Int32(1)) * fx.Int32(4)
        )
        col_lds_ld_i32 = _raw(
            ArithValue(lds_subtile_i32) * fx.Int32(8) + ArithValue(intra_off_i32)
        )
        v_off_lds_ld_i32 = _raw(
            (ArithValue(row_lds_ld_i32) * fx.Int32(O32_NUM_ELEM_PER_PAD_ROW)
             + ArithValue(col_lds_ld_i32)) * fx.Int32(4)
        )
        addr_ld_i32 = _raw(
            ArithValue(p_lds_o_base_i32)
            + ArithValue(warp_idx_i32) * fx.Int32(O32_BYTES_PER_WARP_LDS)
            + ArithValue(v_off_lds_ld_i32)
        )
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
        # Reuse the same oaccu pinned VGPRs (oaccu_base..oaccu_base+15) as
        # ds_read destinations — matches HK's pattern (the source bytes were
        # just written to LDS so the registers are dead).
        for round_idx in range_constexpr(O32_VRAM_ST_NUM_RNDS):
            dst_lo = oaccu_base + 4 * round_idx
            dst_hi = oaccu_base + 4 * round_idx + 3
            byte_off = round_idx * O32_LDS_LD_OFFSET_DELTA_BYTES
            asm = (
                f"ds_read_b128 v[{dst_lo}:{dst_hi}], $0"
                + (f" offset:{byte_off}" if byte_off else "")
            )
            llvm.inline_asm(
                None, [addr_ld_i32], asm, "v",
                has_side_effects=True,
            )

        # ---- 3) buffer_store_dwordx4 x 4 (fp32 to split_output) ----
        # row_vram_st = row_lds_ld + warp * 16
        # col_vram_st = lane_in_row * 4
        # vram_off = (row_vram_st * V_HEAD_DIM + col_vram_st) * 4 + col_off_bytes
        # batch base (soffset) = partial_qo_loc * NUM_QO_HEADS * V_HEAD_DIM * 4
        row_vram_st_i32 = _raw(
            ArithValue(row_lds_ld_i32)
            + ArithValue(warp_idx_i32) * fx.Int32(O32_NUM_ROWS)
        )
        col_vram_st_i32 = _raw(
            ArithValue(lane_in_row_i32) * fx.Int32(O32_VRAM_ST_ELEM_PER_LANE)
        )
        vram_off_round0_i32 = _raw(
            (ArithValue(row_vram_st_i32) * fx.Int32(V_HEAD_DIM)
             + ArithValue(col_vram_st_i32)) * fx.Int32(4)
            + fx.Int32(col_off_bytes)
        )
        batch_soff = _raw(
            ArithValue(partial_qo_loc_i32)
            * fx.Int32(NUM_QO_HEADS * V_HEAD_DIM * 4)
        )
        # HK staggers s_waitcnt lgkmcnt(N) per round (3,2,1,0).  Mirror that.
        for round_idx in range_constexpr(O32_VRAM_ST_NUM_RNDS):
            wait_n = O32_VRAM_ST_NUM_RNDS - 1 - round_idx  # 3,2,1,0
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=wait_n))
            src_lo = oaccu_base + 4 * round_idx
            src_hi = oaccu_base + 4 * round_idx + 3
            src_range = pv.PinnedRange(src_lo, src_hi)
            extra_off = round_idx * O32_VRAM_ST_OFFSET_DELTA_BYTES
            vram_off_i32 = _raw(
                ArithValue(vram_off_round0_i32) + fx.Int32(extra_off)
            )
            dws = pv.read_pinned(src_range)
            data_vec = Vec.from_elements(list(dws), fx.Int32)
            buffer_ops.buffer_store(
                _raw(data_vec), split_output_rsrc,
                vram_off_i32,
                offset_is_bytes=True,
                soffset_bytes=batch_soff,
            )

    def _do_omgr_v3_epilogue(qo_start_i32, row_sum_e_val):
        """Full OMgr V3 epilogue: normalize oaccu by 1/row_sum_e, then 8
        output_to_vram_pair calls covering oaccu's 128 vgprs.

        Bounce LDS overlays p_lds_kv_next (p_lds_kv_1 in our wiring -- safe
        on the global-last iter since the swap is a no-op).
        """
        # Hard scheduling fence: block LLVM from hoisting the OMgr cvts
        # (which READ oaccu) above the PV mfmas (which WRITE oaccu).
        # has_side_effects + clobbers were not sufficient -- the AMDGPU
        # MachineScheduler was reordering pinned-asm ops freely because
        # the vgpr literals in our asm strings are opaque to it.
        # __builtin_amdgcn_sched_barrier(0) is the only construct that
        # blocks the scheduler from moving instructions across the fence.
        rocdl.sched_barrier(0)

        # Drain PV mfma pipeline before SIMD reads of oaccu (MFMA->SIMD hazard).
        llvm.inline_asm(None, [], "s_nop 15\n\t s_nop 15", "",
                        has_side_effects=True)

        # 1/row_sum_e via rocdl.rcp
        reci = rocdl.rcp(T.f32, _raw(row_sum_e_val))
        pv.pinned_v_mul_f32(OACCU_RANGE, reci)

        # Fence again: normalize WRITES oaccu, OMgr cvts must read the
        # post-normalize value.  Without this, OMgr cvts could be moved
        # above the v_mul_f32 sequence (and they were, by ~4000
        # instructions, per yesterday's diagnosis).
        rocdl.sched_barrier(0)

        p_lds_o_base_i32 = _raw(
            ArithValue(_i32(lds_base_idx)) + fx.Int32(P_LDS_KV_1)
        )
        for it in range_constexpr(8):
            _omgr16_output_pair(it, p_lds_o_base_i32, qo_start_i32)
            rocdl.sched_barrier(0)

    def _do_omgr_v3_split_epilogue(partial_qo_loc_i32, row_max_val,
                                   row_sum_e_val):
        """Split (multi-split) epilogue.  Mirrors HK lines 938-988:

          - oaccu /= row_sum_e (same as the bf16 path; reduce kernel does NOT
            re-divide, so the partial fp32 stored to split_output is the
            already-normalized weighted-V).
          - 8 calls to _omgr32_output_pair into split_output.
          - Lanes 0..15 of each warp write per-Q-head LSE to
              split_lse[partial_qo_loc * NUM_QO_HEADS + warp*16 + lane]
            with lse = row_max + ln(row_sum_e) * (1/log2e).
        """
        from flydsl.expr import math as _math

        rocdl.sched_barrier(0)
        llvm.inline_asm(None, [], "s_nop 15\n\t s_nop 15", "",
                        has_side_effects=True)

        reci = rocdl.rcp(T.f32, _raw(row_sum_e_val))
        pv.pinned_v_mul_f32(OACCU_RANGE, reci)
        rocdl.sched_barrier(0)

        p_lds_o_base_i32 = _raw(
            ArithValue(_i32(lds_base_idx)) + fx.Int32(P_LDS_KV_1)
        )
        for it in range_constexpr(8):
            _omgr32_output_pair(it, p_lds_o_base_i32, partial_qo_loc_i32)
            rocdl.sched_barrier(0)

        # ---- LSE write (lanes 0..15 only) -----------------------------
        # lse = row_max + ln(row_sum_e)
        # HK uses __builtin_amdgcn_logf (= v_log_f32 = log2 in HW) * inv_log2e
        # which simplifies to ln(row_sum_e). We use math.log2 + same multiply
        # to match HK bit-for-bit.
        inv_log2e_f32 = _raw(fx.Float32(1.0 / 1.4426950408889634))
        log2_rse = _math.log2(_raw(row_sum_e_val), fastmath=fm_no_inf)
        scaled = _raw(arith.mulf(log2_rse, inv_log2e_f32, fastmath=fm_no_inf))
        lse_val = _raw(arith.addf(_raw(row_max_val), scaled,
                                  fastmath=fm_no_inf))

        # row_idx = partial_qo_loc * NUM_QO_HEADS + warp*16 + lane
        row_idx_i32 = _raw(
            ArithValue(partial_qo_loc_i32) * fx.Int32(NUM_QO_HEADS)
            + ArithValue(warp_idx_i32) * fx.Int32(16)
            + ArithValue(lane_idx_i32)
        )
        byte_off_i32 = _raw(ArithValue(row_idx_i32) * fx.Int32(4))

        # Gate by lane_idx < 16 (per HK predicate)
        in_range = _raw(arith.cmpi(
            arith.CmpIPredicate.slt,
            ArithValue(lane_idx_i32),
            fx.Int32(16),
        ))
        if_op = scf.IfOp(in_range)
        with ir.InsertionPoint(if_op.then_block):
            buffer_ops.buffer_store(
                lse_val, split_lse_rsrc,
                byte_off_i32,
                offset_is_bytes=True,
            )
            scf.YieldOp([])

    def _load_q_phase1(qo_start_i32):
        """Full Phase 1: 4-chunk double-buffered pipeline.

        Pattern (matches HK QManager8to16bitsV1::load_q lines 622-647):
            stage chunk 0 (buf 0)
            stage chunk 1 (buf 1)
            wait vmcnt(2)              -- drain c0's 2 vmem ops
            consume chunk 0
            wait lgkmcnt(0)            -- drain c0's ds_read before buf 0 reuse
            stage chunk 2 (buf 0)
            wait vmcnt(2)              -- drain c1
            consume chunk 1
            wait lgkmcnt(0)
            stage chunk 3 (buf 1)
            wait vmcnt(2)              -- drain c2
            consume chunk 2
            wait vmcnt(0)              -- drain c3
            consume chunk 3
        """
        q_vmem_base = _q_warp_vmem_base(qo_start_i32)

        if const_expr(DEBUG_BYPASS_P1_STAGING):
            for chunk in range_constexpr(4):
                _p1_vmem_to_vgpr_chunk_direct(chunk, q_vmem_base)
            return

        # Stage chunks 0, 1 (vmem ops in flight = 4).
        s_0 = _p1_vmem_to_staging_chunk(0, 0, q_vmem_base)
        s_1 = _p1_vmem_to_staging_chunk(1, 1, q_vmem_base)

        # Consume chunk 0 (waits vmcnt(0) + lgkmcnt(0) inside, issues ds_read,
        # then cvts).  The ds_read it issues is still IN FLIGHT after return;
        # we must drain it (lgkmcnt=0) BEFORE re-staging into buf 0, else the
        # next vmem->buf0 write races with c0's ds_read.  HK orders this
        # explicitly (see QManager8to16bitsV1::load_q).
        _p1_staging_to_vgpr_chunk(0, 0, s_0)
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
        rocdl.sched_barrier(0)
        s_2 = _p1_vmem_to_staging_chunk(2, 0, q_vmem_base)

        _p1_staging_to_vgpr_chunk(1, 1, s_1)
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
        rocdl.sched_barrier(0)
        s_3 = _p1_vmem_to_staging_chunk(3, 1, q_vmem_base)

        _p1_staging_to_vgpr_chunk(2, 0, s_2)
        _p1_staging_to_vgpr_chunk(3, 1, s_3)

    # ---- Per-work-item helper: one KV-tile iteration (load + qk + softmax + pv) ----
    # Single-pong form (always p_lds_kv_0): pong-swap deferred to a later
    # stage.  Operates on the persistent pinned oaccu (the PV mfmas write
    # to oaccu in-place, so no SSA carry needed for the matrix accumulator).
    p_lds_kv_0_base_i32_outer = _raw(
        ArithValue(_i32(lds_base_idx)) + fx.Int32(P_LDS_KV_0)
    )

    def _debug_dump_localmax_i32(val_i32, slot_idx, qo_start_i32):
        """Dump a raw i32 per lane (no bitcast)."""
        head_row_i32 = _raw(
            ArithValue(warp_idx_i32) * fx.Int32(16)
            + ArithValue(lane_idx_i32) // fx.Int32(4)
        )
        lane_byte_base_i32 = _raw(
            (ArithValue(lane_idx_i32) % fx.Int32(4)) * fx.Int32(4)
            + fx.Int32(slot_idx * 256)
        )
        row_base_byte = _raw(
            ArithValue(head_row_i32) * fx.Int32(V_HEAD_DIM * 2)
        )
        batch_byte_off = _raw(
            ArithValue(qo_start_i32) * fx.Int32(NUM_QO_HEADS * V_HEAD_DIM * 2)
        )
        byte_off = _raw(
            ArithValue(row_base_byte) + ArithValue(lane_byte_base_i32)
        )
        buffer_ops.buffer_store(
            val_i32, final_output_rsrc, byte_off,
            offset_is_bytes=True, soffset_bytes=batch_byte_off,
        )

    def _debug_dump_localmax(val_f32, slot_idx, qo_start_i32):
        """Dump a single fp32 per lane to final_output.

        Layout: each lane writes 4 bytes (one fp32) at
          head_row = warp*16 + lane // 4
          col_byte = slot_idx * 256 + (lane % 4) * 4
        slot 0 -> cols 0..63 of head_row (pre-reduce per-lane)
        slot 1 -> cols 64..127 of head_row (post-reduce per-lane)
        Each warp's 16 head_rows hold all 64 lanes' values (4 lanes per head_row).
        """
        head_row_i32 = _raw(
            ArithValue(warp_idx_i32) * fx.Int32(16)
            + ArithValue(lane_idx_i32) // fx.Int32(4)
        )
        lane_byte_base_i32 = _raw(
            (ArithValue(lane_idx_i32) % fx.Int32(4)) * fx.Int32(4)
            + fx.Int32(slot_idx * 256)
        )
        row_base_byte = _raw(
            ArithValue(head_row_i32) * fx.Int32(V_HEAD_DIM * 2)
        )
        batch_byte_off = _raw(
            ArithValue(qo_start_i32) * fx.Int32(NUM_QO_HEADS * V_HEAD_DIM * 2)
        )
        byte_off = _raw(
            ArithValue(row_base_byte) + ArithValue(lane_byte_base_i32)
        )
        # Store fp32 directly (4 bytes). Reader interprets as fp32 by re-viewing.
        val_i32 = _raw(ArithValue(val_f32).bitcast(T.i32))
        buffer_ops.buffer_store(
            val_i32, final_output_rsrc, byte_off,
            offset_is_bytes=True, soffset_bytes=batch_byte_off,
        )

    def _debug_dump_pmfma(qo_start_i32):
        """Dump pinned p_mfma (v[P_MFMA_RANGE.lo..+3]) in KV_TOP-style layout."""
        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0, lgkmcnt=0))
        rocdl.sched_barrier(0)
        head_row_i32 = _raw(
            ArithValue(warp_idx_i32) * fx.Int32(16)
            + ArithValue(lane_idx_i32) // fx.Int32(4)
        )
        lane_byte_base_i32 = _raw(
            (ArithValue(lane_idx_i32) % fx.Int32(4)) * fx.Int32(16)
        )
        row_base_byte_pm = _raw(
            ArithValue(head_row_i32) * fx.Int32(V_HEAD_DIM * 2)
        )
        batch_byte_off_pm = _raw(
            ArithValue(qo_start_i32) * fx.Int32(NUM_QO_HEADS * V_HEAD_DIM * 2)
        )
        for i in range_constexpr(4):
            slot = pv.PinnedRange(P_MFMA_RANGE.lo + i, P_MFMA_RANGE.lo + i)
            dw_i32 = pv.read_pinned(slot, as_i64=False)
            byte_off_pm = _raw(
                ArithValue(row_base_byte_pm)
                + ArithValue(lane_byte_base_i32)
                + fx.Int32(i * 4)
            )
            buffer_ops.buffer_store(
                dw_i32, final_output_rsrc, byte_off_pm,
                offset_is_bytes=True, soffset_bytes=batch_byte_off_pm,
            )

    def _debug_dump_p_comp(p_comp_lo_val, p_comp_hi_val, qo_start_i32):
        """Dump 8 fp32 p_comp values per lane (= mfma D-tile of tile 0) as
        bf16 into final_output[0, qo_start, head_row, kv_col].

        Per cdna3 v_mfma_f32_16x16x32_f16 D layout, lane l holds D[k*4+l/16,
        l%16] for k in 0..3.  warp_idx_i32 picks the head-row 16-row band.
        """
        if const_expr(DEBUG_FORCE_PCOMP_PATTERN):
            # Override with known pattern: value at lane l, k = l*1000 + k (LO)
            # and l*1000 + k + 100 (HI).  Cast to fp32 SSA Vec.
            lane_f = arith.sitofp(T.f32, lane_idx_i32)
            c1000 = _raw(fx.Float32(1000.0))
            base = _raw(arith.mulf(lane_f, c1000, fastmath=fm_no_inf))
            lo_elems = [_raw(arith.addf(base, _raw(fx.Float32(float(k))),
                                        fastmath=fm_no_inf)) for k in range(4)]
            hi_elems = [_raw(arith.addf(base, _raw(fx.Float32(float(k + 100))),
                                        fastmath=fm_no_inf)) for k in range(4)]
            p_comp_lo_val = _raw(Vec.from_elements(lo_elems, fx.Float32))
            p_comp_hi_val = _raw(Vec.from_elements(hi_elems, fx.Float32))
        lo_vec = Vec(p_comp_lo_val)
        hi_vec = Vec(p_comp_hi_val)
        head_row_base_i32 = _raw(
            ArithValue(warp_idx_i32) * fx.Int32(16)
            + ArithValue(lane_idx_i32) // fx.Int32(16)
        )
        kv_col_lo_i32 = _raw(ArithValue(lane_idx_i32) % fx.Int32(16))
        kv_col_hi_i32 = _raw(ArithValue(kv_col_lo_i32) + fx.Int32(16))
        batch_byte_off = _raw(
            ArithValue(qo_start_i32)
            * fx.Int32(NUM_QO_HEADS * V_HEAD_DIM * 2)
        )
        for k in range_constexpr(4):
            head_row_i32 = _raw(
                ArithValue(head_row_base_i32) + fx.Int32(k * 4)
            )
            row_base_byte = _raw(
                ArithValue(head_row_i32)
                * fx.Int32(V_HEAD_DIM * 2)
            )
            # LO half: col in [0..15]
            off_lo = _raw(
                ArithValue(row_base_byte)
                + ArithValue(kv_col_lo_i32) * fx.Int32(2)
            )
            val_lo_bf16 = _raw(
                ArithValue(_raw(lo_vec[k])).truncf(T.bf16)
            )
            buffer_ops.buffer_store(
                val_lo_bf16, final_output_rsrc, off_lo,
                offset_is_bytes=True, soffset_bytes=batch_byte_off,
            )
            # HI half: col in [16..31]
            off_hi = _raw(
                ArithValue(row_base_byte)
                + ArithValue(kv_col_hi_i32) * fx.Int32(2)
            )
            val_hi_bf16 = _raw(
                ArithValue(_raw(hi_vec[k])).truncf(T.bf16)
            )
            buffer_ops.buffer_store(
                val_hi_bf16, final_output_rsrc, off_hi,
                offset_is_bytes=True, soffset_bytes=batch_byte_off,
            )

    def _debug_dump_q_vgpr(qo_start_i32):
        """Dump raw q_vgpr (v72..v103) at the silver Q position each lane's
        data corresponds to.

        Per HK Phase 1 cvt ordering and reader formula:
          chunk = vgpr_idx // 8           (in [0..4))   -> chunk cols base = chunk * 64
          chunk_vgpr = vgpr_idx % 8       (in [0..8))
          iter = chunk_vgpr // 4          (0 or 1) -> iter j adds 32 to col base
          dw_in_iter = (chunk_vgpr // 2) % 4   (in [0..4))
          opsel = chunk_vgpr % 2          (0 or 1)
          For lane l:
            row_in_warp = l & 15
            cb = (l >> 4) & 3
            col within chunk = 16 * cb + 8 * iter ? NO -- HK says
                col base for iter j (within chunk) = 16 * cb + 8 * j (only 2 sub-blocks per chunk)
            But we have 4 sub-tiles of 16 cols each = 4*16 = 64 cols/chunk.  So cb in
            {0..3} cycles through sub-blocks of the chunk.  iter j adds 8 cols inside
            the cb-selected sub-block.
            full chunk_col = 16 * cb + 8 * iter + dw_in_iter * 2 + opsel
          Absolute Q col = chunk * 64 + chunk_col
          Silver head row = warp * 16 + row_in_warp
        """
        row_in_warp_i32 = _raw(ArithValue(lane_idx_i32) % fx.Int32(16))
        cb_i32 = _raw((ArithValue(lane_idx_i32) // fx.Int32(16)) & fx.Int32(3))
        head_row_i32 = _raw(
            ArithValue(warp_idx_i32) * fx.Int32(16)
            + ArithValue(row_in_warp_i32)
        )
        row_base_byte = _raw(
            ArithValue(head_row_i32) * fx.Int32(V_HEAD_DIM * 2)
        )
        batch_byte_off = _raw(
            ArithValue(qo_start_i32)
            * fx.Int32(NUM_QO_HEADS * V_HEAD_DIM * 2)
        )
        for vgpr_idx in range_constexpr(32):
            chunk = vgpr_idx // 8
            chunk_vgpr = vgpr_idx % 8
            iter_ = chunk_vgpr // 4         # 0 or 1
            dw_in_iter = (chunk_vgpr // 2) % 2   # 0 or 1 within 2-dword-per-iter-per-row?
            # Actually each iter has 4 dwords (4 cols * 2 bf16/col = 8 cols).
            # vgpr_idx 0,1 -> dw 0 of iter 0; vgpr_idx 2,3 -> dw 1 of iter 0; etc.
            dw_in_iter = (chunk_vgpr % 4) // 2   # 0,0,1,1 within iter
            opsel = chunk_vgpr % 2          # 0 or 1
            slot = pv.PinnedRange(
                Q_VGPR_RANGE.lo + vgpr_idx,
                Q_VGPR_RANGE.lo + vgpr_idx,
            )
            dw_i32 = pv.read_pinned(slot, as_i64=False)
            # chunk_col = 16 * cb + 8 * iter + dw_in_iter * 2 + 0 (the dword holds 2 bf16,
            # writing those 2 bf16 covers opsel-low and opsel-high which represent col*2 and col*2+1)
            # Each dword stores 2 consecutive bf16 (low and high of the dword) which correspond to
            # 2 consecutive chunk_cols: base and base+1.  We write the dword (4 bytes) starting at
            # bf16 col `base` = 16*cb + 8*iter + dw_in_iter*2 + opsel*?
            # But there are TWO vgprs per (chunk, iter, dw_in_iter): one for opsel=False, one for True.
            # Per HK, opsel selects WHICH 2 fp8 bytes of the source dword we read.  The OUTPUT
            # bf16 pair goes into the WHOLE dword.  So:
            #   vgpr opsel=False: bf16 pair at col base + 0, base + 1 ? or base + 0..1 with opsel = byte select
            # The cleanest is: each vgpr holds a 2-bf16 pair representing 2 consecutive chunk_cols.
            # The bytes (col base in the staging) for vgpr (dw_in_iter=d, opsel=o) =
            # iter*8 + d*4 + o*2  -- but we need to confirm by experiment.
            # For now assume: chunk_col_lo = 16*cb + 8*iter + (dw_in_iter*2 + opsel) * 2
            #                 chunk_col_hi = chunk_col_lo + 1
            chunk_col_base_const_part = 8 * iter_ + (dw_in_iter * 2 + opsel) * 2
            q_col_i32 = _raw(
                ArithValue(cb_i32) * fx.Int32(16)
                + fx.Int32(chunk * 64 + chunk_col_base_const_part)
            )
            byte_off = _raw(
                ArithValue(row_base_byte)
                + ArithValue(q_col_i32) * fx.Int32(2)   # 2 bytes / bf16
            )
            buffer_ops.buffer_store(
                dw_i32, final_output_rsrc, byte_off,
                offset_is_bytes=True, soffset_bytes=batch_byte_off,
            )

    def _debug_dump_k_lds_to_vram(p_lds_kv_base_i32, qo_start_i32):
        """Dump K LDS pong (32 KB = 16 col_tiles × 2 row_tiles × 1024 B each)
        to final_output.  Layout: warp ``w`` dumps the 4 KB slice at
        ``p_lds_kv + w * 4096`` (= 4 sub-blocks per warp).  Within each warp,
        lane ``T`` reads 16 bytes (1 b128) at ``slice_base + T * 16`` and
        writes them to ``final_output[0, w*16 + T//4, (T%4)*16 .. +8 bf16]``.
        Two passes (each = 2KB) cover the warp's 4 KB.
        """
        head_row_base_i32 = _raw(
            ArithValue(warp_idx_i32) * fx.Int32(16)
            + ArithValue(lane_idx_i32) // fx.Int32(4)
        )
        lane_byte_in_pass_i32 = _raw(
            (ArithValue(lane_idx_i32) % fx.Int32(4)) * fx.Int32(16)
        )
        row_base_byte = _raw(
            ArithValue(head_row_base_i32) * fx.Int32(V_HEAD_DIM * 2)
        )
        batch_byte_off = _raw(
            ArithValue(qo_start_i32)
            * fx.Int32(NUM_QO_HEADS * V_HEAD_DIM * 2)
        )
        # Warp w slice base = w * 4096 within the 32 KB pong.
        warp_slice_base_i32 = _raw(
            ArithValue(warp_idx_i32) * fx.Int32(4096)
        )
        for pass_idx in range_constexpr(4):
            # Each pass = 1024 bytes per warp = 64 lanes × 16 bytes.
            lds_addr = _raw(
                ArithValue(p_lds_kv_base_i32)
                + ArithValue(warp_slice_base_i32)
                + fx.Int32(pass_idx * 1024)
                + ArithValue(lane_idx_i32) * fx.Int32(16)
            )
            data = _lds_load_b128(lds_addr, byte_offset=0)
            byte_off = _raw(
                ArithValue(row_base_byte)
                + fx.Int32(pass_idx * 64)
                + ArithValue(lane_byte_in_pass_i32)
            )
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            buffer_ops.buffer_store(
                data, final_output_rsrc, byte_off,
                offset_is_bytes=True, soffset_bytes=batch_byte_off,
            )

    def _debug_dump_q_lds_to_vram(qo_start_i32):
        """Read each lane's bytes from p_lds_q and write them to final_output.

        For each of 8 passes (covering 256 cols per warp), each lane reads 16
        bytes (= 8 bf16 = one b128) from
          p_lds_q + warp*Q_WARP_FINAL_BYTES + pass*1024 + (lane>>2)*64 + (lane&3)*16
        which is exactly the byte layout Phase 2's NoPE cvt+store wrote.
        (No row XOR -- we read raw bytes; comparison against expected requires
        un-applying the writer's XOR in the host.  For now we just dump raw.)

        Store target in final_output[0, head_row, col]:
          head_row = warp*16 + (lane // 4)
          byte_col_start = pass*32 + (lane & 3)*8   (= 8 bf16 = 16 bytes)
        So per warp we write to cols [0..256) bf16 of head rows [w*16..+16).
        """
        for pass_idx in range_constexpr(8):
            # LDS source addr per lane.
            lds_base_i32 = _i32(lds_base_idx)
            row_in_warp_i32 = _raw(ArithValue(lane_idx_i32) // fx.Int32(4))
            col_group_i32 = _raw(ArithValue(lane_idx_i32) & fx.Int32(3))
            byte_in_sb_i32 = _raw(ArithValue(col_group_i32) * fx.Int32(16))
            lds_addr = _raw(
                ArithValue(lds_base_i32)
                + fx.Int32(P_LDS_Q)
                + ArithValue(warp_idx_i32) * fx.Int32(Q_WARP_FINAL_BYTES)
                + fx.Int32(pass_idx * Q_SUB_BLOCK_BYTES)
                + ArithValue(row_in_warp_i32) * fx.Int32(Q_SUB_BLOCK_COLS * 2)
                + ArithValue(byte_in_sb_i32)
            )
            # ds_read_b128 = 16 bytes -> i32x4 SSA value.
            data = _lds_load_b128(lds_addr, byte_offset=0)

            # VRAM dst: head_row = warp*16 + lane//4, byte_col_start = pass*32 + (lane&3)*8.
            head_row_i32 = _raw(
                ArithValue(warp_idx_i32) * fx.Int32(16)
                + ArithValue(lane_idx_i32) // fx.Int32(4)
            )
            byte_col_start_i32 = _raw(
                fx.Int32(pass_idx * 32 + 0)  # filled in below per lane via col_group
                + ArithValue(col_group_i32) * fx.Int32(8)  # 8 bf16 = 16 bytes per lane
            )
            # final_output[0, head, col] is bf16; col_byte = head_row * V_HEAD_DIM*2
            # + byte_col_start*2.
            vram_byte_off = _raw(
                ArithValue(head_row_i32) * fx.Int32(V_HEAD_DIM * 2)
                + ArithValue(byte_col_start_i32) * fx.Int32(2)
            )
            batch_byte_off = _raw(
                ArithValue(qo_start_i32)
                * fx.Int32(NUM_QO_HEADS * V_HEAD_DIM * 2)
            )
            # Drain the ds_read first.
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            buffer_ops.buffer_store(
                data, final_output_rsrc, vram_byte_off,
                offset_is_bytes=True, soffset_bytes=batch_byte_off,
            )

    def _process_one_kv_tile(kv_tile_start_i32, kv_tile_end_i32,
                              row_kv_ld_i32,
                              row_max_in, row_sum_e_in,
                              qo_start_i32,
                              is_first_iter, check_boundary):
        """Load KV tile, do QK + softmax + PV, return (row_max, row_sum_e)."""
        if const_expr(DEBUG_DUMP_Q_VGPR):
            # Dump pinned q_vgpr (set by _load_q_phase1) and short-circuit.
            # We're on tile 0 only (caller ensures num_tiles==1 path is the
            # one exercised; multi-tile would also dump but only tile 0's
            # entry matters since q_vgpr is loaded once per work_idx).
            _debug_dump_q_vgpr(qo_start_i32)
            return row_max_in, row_sum_e_in

        if const_expr(DEBUG_DUMP_Q_LDS):
            _debug_dump_q_lds_to_vram(qo_start_i32)
            return row_max_in, row_sum_e_in

        if const_expr(DEBUG_DUMP_P2_NOPE_DW):
            # Phase 2's dump already wrote final_output; skip downstream.
            return row_max_in, row_sum_e_in

        # Single-pong K LDS: barrier prior iter's PV (which reads K LDS as V)
        # against this iter's _kv_async_load_k (which overwrites K LDS).
        # HK uses double-buffered ping-pong so it doesn't need this; FlyDSL
        # only has one K LDS region.
        if const_expr(not is_first_iter):
            rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
            rocdl.s_barrier()
        # Async load (prefetch + cvt+store, plus RoPE direct for waves 5,7).
        _kv_async_load_k(p_lds_kv_0_base_i32_outer, row_kv_ld_i32,
                         check_boundary=check_boundary,
                         qo_start_i32=qo_start_i32)
        if const_expr(DEBUG_DUMP_KV_NOPE_DW):
            # Dumps already wrote final_output; skip QK / softmax / PV / OMgr.
            return row_max_in, row_sum_e_in
        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=0, lgkmcnt=0))
        rocdl.s_barrier()

        if const_expr(DEBUG_DUMP_K_LDS):
            _debug_dump_k_lds_to_vram(p_lds_kv_0_base_i32_outer, qo_start_i32)
            return row_max_in, row_sum_e_in

        _do_qk_gemm(p_lds_kv_0_base_i32_outer, qo_start_i32=qo_start_i32)
        if const_expr(DEBUG_DUMP_KV_TOP):
            return row_max_in, row_sum_e_in

        # Drain MFMA pipeline so the upcoming v_mov_b32 reads of v[120..127]
        # see retired values (CDNA3 MFMA -> SIMD read hazard).  Required for
        # both dump path and production softmax read.
        llvm.inline_asm(None, [], "s_nop 15\n\t s_nop 15", "",
                        has_side_effects=True)

        # QK leaves the result in pinned P_COMP_LO + P_COMP_HI = v[120:127].
        # Read it back as SSA for softmax math.
        p_comp_lo_dws = pv.read_pinned(P_COMP_LO)   # tuple of 4 i32
        p_comp_hi_dws = pv.read_pinned(P_COMP_HI)
        p_comp_lo_val = _raw(Vec.from_elements(
            [_raw(ArithValue(d).bitcast(T.f32)) for d in p_comp_lo_dws],
            fx.Float32,
        ))
        p_comp_hi_val = _raw(Vec.from_elements(
            [_raw(ArithValue(d).bitcast(T.f32)) for d in p_comp_hi_dws],
            fx.Float32,
        ))

        if const_expr(DEBUG_DUMP_QK):
            # Bypass softmax/PV/OMgr.  Write the 8 fp32 p_comp values per
            # lane as bf16 into final_output[0, qo_start, head_row, kv_col],
            # for head_row covered by this warp/lane and kv_col in [0:32).
            # Layout (mfma F32_16x16x32_F16 D-tile):
            #   head_row = warp * 16 + k * 4 + lane // 16        (k in 0..3)
            #   kv_col   = (lane & 15) for LO, 16 + (lane & 15) for HI
            _debug_dump_p_comp(p_comp_lo_val, p_comp_hi_val, qo_start_i32)
            # Short-circuit: return the input carry unchanged so caller's
            # downstream code becomes a no-op for the values we care about.
            return row_max_in, row_sum_e_in

        row_max_new, row_sum_e_new, rescale = _softmax_and_pack(
            p_comp_lo_val, p_comp_hi_val,
            row_max_in, row_sum_e_in,
            kv_tile_start_i32, kv_tile_end_i32,
            is_first_iter=is_first_iter,
            check_boundary=check_boundary,
            qo_start_i32=qo_start_i32,
        )

        if const_expr(DEBUG_DUMP_PMFMA):
            _debug_dump_pmfma(qo_start_i32)
            return row_max_in, row_sum_e_in

        if const_expr(DEBUG_DUMP_LOCALMAX):
            return row_max_in, row_sum_e_in

        _do_pv_gemm(p_lds_kv_0_base_i32_outer, rescale,
                    is_first_iter=is_first_iter)

        return row_max_new, row_sum_e_new

    # ---- Persistent work loop ------------------------------------------
    for work_idx in range(work_start_idx, work_end_idx):
        # MlaWorkInfo dword layout:
        #   [0] batch_idx
        #   [1] partial_qo_loc       (<0 = final path, >=0 = split slot)
        #   [2] qo_start
        #   [3] qo_end               (unused; num_qheads is fixed)
        #   [4] kv_start_page        (page units when PAGE_SIZE>1, else token)
        #   [5] kv_end_page          (page units when PAGE_SIZE>1, else token)
        #   [6] kv_offset            (0 = work item ends at batch tail)
        #   [7] padding
        wi_base = _idx(work_idx) * SIZE_MLA_WORK_INFO_IN_DW
        # PAGE_SIZE==1: skip batch_idx + kv_offset, single 4-dw + 1-dw load.
        # PAGE_SIZE >1: also need batch_idx (dw0) and kv_offset (dw6) for
        # the tail-clip branch.
        wi_dw1_4 = buffer_ops.buffer_load(
            work_info_set_rsrc, wi_base + _idx(1),
            vec_width=4, dtype=T.i32,
        )
        wi_dw5 = buffer_ops.buffer_load(
            work_info_set_rsrc, wi_base + _idx(5),
            vec_width=1, dtype=T.i32,
        )
        wi_vec = Vec(wi_dw1_4)
        partial_qo_loc = rocdl.readfirstlane(T.i32, _raw(wi_vec[0]))
        qo_start = rocdl.readfirstlane(T.i32, _raw(wi_vec[1]))
        kv_start_page = rocdl.readfirstlane(T.i32, _raw(wi_vec[3]))
        kv_end_page = rocdl.readfirstlane(T.i32, _raw(wi_dw5))

        # Token-range conversion (HK lines 272-284).
        if const_expr(PAGE_SIZE == 1):
            kv_start = kv_start_page
            kv_end = kv_end_page
        else:
            kv_start = rocdl.readfirstlane(T.i32, _raw(
                ArithValue(kv_start_page) * fx.Int32(PAGE_SIZE)
            ))
            batch_idx = rocdl.readfirstlane(T.i32, _raw(buffer_ops.buffer_load(
                work_info_set_rsrc, wi_base + _idx(0),
                vec_width=1, dtype=T.i32,
            )))
            kv_offset = rocdl.readfirstlane(T.i32, _raw(buffer_ops.buffer_load(
                work_info_set_rsrc, wi_base + _idx(6),
                vec_width=1, dtype=T.i32,
            )))
            kv_end_full = _raw(
                ArithValue(kv_end_page) * fx.Int32(PAGE_SIZE)
            )
            last_page_len = rocdl.readfirstlane(T.i32, _raw(buffer_ops.buffer_load(
                kv_last_page_lens_rsrc, _idx(batch_idx),
                vec_width=1, dtype=T.i32,
            )))
            kv_end_tail_clip = _raw(
                (ArithValue(kv_end_page) - fx.Int32(1)) * fx.Int32(PAGE_SIZE)
                + ArithValue(last_page_len)
            )
            # If kv_offset != 0 -> work item not at batch tail -> full last page.
            # If kv_offset == 0 -> tail-clip via kv_last_page_lens[batch_idx].
            use_full = _raw(arith.cmpi(
                arith.CmpIPredicate.ne,
                ArithValue(kv_offset),
                fx.Int32(0),
            ))
            kv_end = rocdl.readfirstlane(T.i32, _raw(
                ArithValue(use_full).select(
                    _raw(kv_end_full),
                    _raw(kv_end_tail_clip),
                )
            ))
        # partial_qo_loc < 0 -> bf16 final-output path (single split).
        # partial_qo_loc >= 0 -> fp32 split-output + per-warp natural-log LSE.
        is_split = _raw(arith.cmpi(
            arith.CmpIPredicate.sge,
            ArithValue(partial_qo_loc),
            fx.Int32(0),
        ))

        kv_len_v = ArithValue(kv_end) - ArithValue(kv_start)
        num_tiles_v = (kv_len_v + (BLOCK_N - 1)).with_signedness(False) // BLOCK_N
        num_tiles = _raw(num_tiles_v)

        # ---- Q load (Phase 1 + Phase 2 + cross-warp barrier) ----
        _load_q_phase1(qo_start)
        _load_q_phase2(qo_start)
        rocdl.s_waitcnt(_encode_waitcnt(lgkmcnt=0))
        rocdl.s_barrier()

        # Single-tile fast path vs multi-tile: scf.for needs num_tiles >= 1
        # which we don't guarantee at trace time, but the metadata invariant
        # is num_tiles >= 1 for every dispatched work item.
        kv_start_v = ArithValue(kv_start)
        kv_end_v = ArithValue(kv_end)
        last_tile_partial = _raw((kv_len_v & (BLOCK_N - 1)) != 0)

        # Resolve tile-0 row index.
        first_tile_needs_boundary = _raw(kv_len_v < BLOCK_N)
        row_kv_ld_first = fx.Int32(-1)
        if first_tile_needs_boundary:
            row_kv_ld_first = _get_kv_ld_row(kv_start, kv_end, True)
        else:
            row_kv_ld_first = _get_kv_ld_row(
                kv_start, _raw(kv_start_v + BLOCK_N), False,
            )

        # ---- Tile 0: always runs as the "first" iter (initializes oaccu). ----
        # IMPORTANT: even when the softmax mask handles OOB cols, the KV
        # prefetch must still be boundary-aware -- without it, OOB lanes'
        # row_kv_ld=-1 cast to unsigned wraps around to a huge offset and
        # the vmem load page-faults.  Pass check_boundary=True whenever
        # row_kv_ld may contain -1 sentinels (i.e. the partial-first-tile
        # path).
        row_max_init = _raw(fx.Float32(float('-inf')))
        row_sum_e_init = _raw(fx.Float32(0.0))

        # check_boundary for the prefetch matches first_tile_needs_boundary
        # so OOB lanes are zero-clamped instead of wrapping around.  This
        # is a *Python compile-time* split (FlyDSL traces both branches).
        row_max_t0 = row_max_init
        row_sum_e_t0 = row_sum_e_init
        if first_tile_needs_boundary:
            row_max_t0, row_sum_e_t0 = _process_one_kv_tile(
                kv_start, kv_end, _raw(row_kv_ld_first),
                row_max_init, row_sum_e_init,
                qo_start,
                is_first_iter=True, check_boundary=True,
            )
        else:
            row_max_t0, row_sum_e_t0 = _process_one_kv_tile(
                kv_start, kv_end, _raw(row_kv_ld_first),
                row_max_init, row_sum_e_init,
                qo_start,
                is_first_iter=True, check_boundary=False,
            )

        # ---- Tiles 1..num_tiles-1: scf.for with carried (row_max, row_sum_e) ----
        # v32-style: wrap multi-tile work in a Python helper so the
        # post-loop "results" carry is well-defined.  Single-tile (num_tiles==1)
        # skips the loop entirely and uses the tile-0 values.
        has_multi_tiles = _raw(ArithValue(num_tiles) > 1)

        def _multi_tile_finish():
            init_args = [row_max_t0, row_sum_e_t0]
            for tile_iv, state in range(_idx(1), _idx(num_tiles), _idx(1),
                                         init=init_args):
                rm = state[0]
                rse = state[1]
                tile_iv_i32 = _i32(tile_iv)
                kv_tile_start_iv = _raw(
                    ArithValue(kv_start) + ArithValue(tile_iv_i32) * fx.Int32(BLOCK_N)
                )
                row_kv_ld_iv = _get_kv_ld_row(kv_tile_start_iv, kv_end, True)
                rm_new, rse_new = _process_one_kv_tile(
                    kv_tile_start_iv, kv_end, row_kv_ld_iv,
                    rm, rse,
                    qo_start,
                    is_first_iter=False, check_boundary=True,
                )
                results = yield [rm_new, rse_new]
            # Run epilogue with the loop's final carry; branch on split vs final.
            if const_expr(_DEBUG_RUN_EPILOGUE):
                rm_final = results[0]
                rse_final = results[1]
                _dispatch_epilogue(rm_final, rse_final)

        def _single_tile_finish():
            if const_expr(_DEBUG_RUN_EPILOGUE):
                _dispatch_epilogue(row_max_t0, row_sum_e_t0)

        def _dispatch_epilogue(rm_final, rse_final):
            """Runtime branch on partial_qo_loc:
              < 0  -> bf16 final OMgr (writes final_output)
              >= 0 -> fp32 split OMgr (writes split_output + split_lse)
            """
            if_op = scf.IfOp(is_split, [], has_else=True)
            with ir.InsertionPoint(if_op.then_block):
                _do_omgr_v3_split_epilogue(partial_qo_loc, rm_final, rse_final)
                scf.YieldOp([])
            with ir.InsertionPoint(if_op.else_block):
                _do_omgr_v3_epilogue(qo_start, rse_final)
                scf.YieldOp([])

        @flyc.jit
        def _epilogue_dispatch():
            if has_multi_tiles:
                _multi_tile_finish()
            else:
                _single_tile_finish()

        _epilogue_dispatch()


# ---------------------------------------------------------------------------
# JIT launcher
# ---------------------------------------------------------------------------
@flyc.jit
def launch_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1(
    query_nope: fx.Tensor,
    query_rope: fx.Tensor,
    kv_buffer_nope: fx.Tensor,
    kv_buffer_rope: fx.Tensor,
    kv_page_indices: fx.Tensor,
    kv_last_page_lens: fx.Tensor,
    work_indptr: fx.Tensor,
    work_info_set: fx.Tensor,
    final_output: fx.Tensor,
    split_output: fx.Tensor,
    split_lse: fx.Tensor,
    softmax_scale: fx.Float32,
    log2_num_qheads: fx.Int32,
    num_cus: fx.Constexpr,
    lds_size: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch the V4.0 Gen.1 decode kernel.

    Grid = (num_cus, 1, 1).  All LDS is statically allocated via the
    SmemAllocator, so the launch `smem` is 0.
    """
    assert TOTAL_LDS_BYTES <= lds_size, (
        f"Kernel requires {TOTAL_LDS_BYTES} B LDS but CU budget is {lds_size}"
    )
    kn_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1(
        query_nope,
        query_rope,
        kv_buffer_nope,
        kv_buffer_rope,
        kv_page_indices,
        kv_last_page_lens,
        work_indptr,
        work_info_set,
        final_output,
        split_output,
        split_lse,
        softmax_scale,
        log2_num_qheads,
        value_attrs={
            "passthrough": [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(
        grid=(num_cus, 1, 1),
        block=(NUM_THREADS, 1, 1),
        smem=0,
        stream=stream,
    )


# Compiler hint: cap allocator scratch at v0..v63 so LLVM-allocated SSA
# never leaks into the pinned region v64..v255.  Mirrors HK's
# ``__attribute__((amdgpu_num_vgpr(64)))`` on the __global__.  Without
# this, the function-wide ``emit_clobber()`` keeps the right scheduling
# ordering but does NOT cap the allocator's vgpr pool -- SSA scratch can
# still land in v64+ and silently clobber pinned q_vgpr / oaccu.
launch_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1.compile_hints = {
    "maxnreg": 64,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


# ---------------------------------------------------------------------------
# aiter-compatible wrapper: drop-in replacement for ``aiter.hk_mla_v40_decode_fwd``
# ---------------------------------------------------------------------------
# Same positional signature as ``aiter.hk_mla_v40_decode_fwd`` so callers
# in ``aiter/mla.py`` can swap implementations without computing num_cus /
# lds_size / log2_num_qheads / tensor reshapes themselves.
def mla_v40_decode_fwd(
    query,            # [total_q, num_heads, 576] FP8 (NoPE + dup-E8M0 + pad)
    query_rope,       # [total_q, num_heads, 64]  BF16
    kv_buffer,        # [num_page, page_size, num_kv_heads=1, 576] FP8
    kv_buffer_rope,   # [num_page, page_size, num_kv_heads=1, 64]  BF16
    qo_indptr,        # unused (FlyDSL kernel takes per-warp work via work_indptr/work_info_set)
    kv_indptr,        # unused (same)
    kv_page_indices,
    kv_last_page_lens,
    work_indptr,
    work_info_set,
    max_seqlen_q,
    softmax_scale,
    split_output,
    split_lse,
    final_output,
):
    """Drop-in replacement for ``aiter.hk_mla_v40_decode_fwd`` that dispatches
    to the FlyDSL V4.0 m16x8 decode kernel.

    Computes ``num_cus`` / ``lds_size`` / ``log2_num_qheads`` from the
    runtime device + tensor shapes so the caller does not have to.
    ``qo_indptr`` / ``kv_indptr`` are accepted for signature parity but
    unused -- the FlyDSL kernel reads per-warp work from
    ``work_indptr`` + ``work_info_set``.
    """
    import math as _math
    import torch as _torch

    del qo_indptr, kv_indptr, max_seqlen_q  # accepted for sig parity only

    total_q, nhead = query.shape[0], query.shape[1]
    num_page, page_sz, nhead_kv = (
        kv_buffer.shape[0], kv_buffer.shape[1], kv_buffer.shape[2],
    )
    device_idx = (
        query.device.index if query.device.index is not None else 0
    )
    props = _torch.cuda.get_device_properties(device_idx)
    num_cus = props.multi_processor_count
    # gfx950 has 160 KiB shared memory per CU; OCCUPANCY=1 in this kernel,
    # so the full budget is available.
    lds_size = 160 * 1024
    assert nhead & (nhead - 1) == 0, f"nhead must be power of 2, got {nhead}"
    log2_num_qheads = int(_math.log2(nhead))

    # Specialize the kernel for this page_size (FlyDSL captures module
    # globals into its JIT cache key, so mutating PAGE_SIZE here triggers
    # a fresh compile + cache entry per value).  Supported: {1, 16, 32, 64}.
    assert page_sz in (1, 16, 32, 64), (
        f"FlyDSL V40 supports page_size in (1, 16, 32, 64); got {page_sz}"
    )
    global PAGE_SIZE
    PAGE_SIZE = page_sz

    launch_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1(
        query.view(total_q, nhead, QK_PACKED_NOPE_BYTES),
        query_rope.view(total_q, nhead, QK_ROPE_HEAD_DIM),
        kv_buffer.view(num_page, page_sz, nhead_kv, QK_PACKED_NOPE_BYTES),
        kv_buffer_rope.view(num_page, page_sz, nhead_kv, QK_ROPE_HEAD_DIM),
        kv_page_indices,
        kv_last_page_lens,
        work_indptr,
        work_info_set,
        final_output,
        split_output,
        split_lse,
        softmax_scale,
        log2_num_qheads,
        num_cus=num_cus,
        lds_size=lds_size,
        stream=_torch.cuda.current_stream(),
    )
