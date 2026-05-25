"""MLA decode Stage1 — GFX1250 fp8 WMMA (``v_wmma_f32_16x16x64_fp8_fp8``).

Online softmax + paged KV with ``ds_load_tr8_b64`` transpose V load from K nope LDS.
K/V share the nope portion in LDS; V is read via hardware transpose after GEMM1 is done.

Input/output shapes (row-major, matching ``compile_mla_chaindot_decode_fp8``):
  Q         (batch, num_q_heads, HEAD_DIM_QK)  fp8
  K_Buffer  paged (num_blocks, page_block_size, num_kv_heads, HEAD_DIM_QK)  fp8
  V_buffer  paged, shares c_kv with K_Buffer  fp8
  Mid_O     (batch, num_kv_splits, num_q_heads, kv_lora_rank)  f32
  Mid_lse   (batch, num_kv_splits, num_q_heads)  f32

Grid: (batch, num_head_blocks, num_kv_splits).  Block: (WAVE_SIZE, NUM_WAVES, 1).
"""

from typing import Any, Callable, Optional
import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import (
    arith as _std_arith,
    fly as fly_d,
    llvm as llvm_dialect,
    math as _mlir_math,
    memref as memref_dialect,
    scf as _scf,
)
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, tdm_ops, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from kernels.layout_utils import crd2idx, idx2crd, get as layout_get


def _exp_f32(x):
    log2e = arith.constant(1.4426950408889634, type=T.f32)
    scaled = _std_arith.MulFOp(_raw(x), _raw(log2e)).result
    return _mlir_math.Exp2Op(_raw(scaled)).result

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WAVE_SIZE = 32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 64
WMMA_VEC = 8
WMMA_K_LANES = 2
WMMA_K_GRP = WMMA_K // (WMMA_K_LANES * WMMA_VEC)  # 4
DS_LOAD_TR_VEC = 8
# gfx1250 LDS budget: WGP$ is 384 KB unified, up to 320 KB can be allocated
# as LDS per workgroup (PDF §0.* lines 399, 2419, 18298; AM config
# `lds_320k`). This was previously set to 94208 (92 KB) — overly conservative.
LDS_SIZE_LIMIT = 320 * 1024

# Number of LDS ping-pong buffers for the KV-loop. KV nope (32 KB) + KV rope
# (4 KB) = 36 KB per stage; persistent Q + scratch = 11 KB; LDS budget 92 KB.
# So we can fit 2 stages (82 KB total, ~90% LDS), but not 3 (would need 118 KB).
# NUM_KV_BUFS=1 collapses to legacy single-buffer (issue→wait→consume).
NUM_KV_BUFS = 2


def compile_mla_decode_fp8_gfx1250(
    nheads_q: int = 16,
    seqlen_kv: int = 512,
    num_kv_heads: int = 1,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    block_h: int = 16,
    block_n: int = 64,
    block_c: int = 512,
    block_r: int = 64,
    sm_scale: Optional[float] = None,
    NUM_WAVES: int = 4,
    num_kv_splits: int = 1,
) -> Callable[..., Any]:
    """Compile Stage1 MLA decode for GFX1250.

    Returns a ``@flyc.jit`` callable that launches the kernel.
    """
    headdim_qk = kv_lora_rank + qk_rope_head_dim
    max_kv_steps = seqlen_kv // block_n
    assert max_kv_steps % num_kv_splits == 0, (
        f"max_kv_steps ({max_kv_steps}) must be divisible by num_kv_splits ({num_kv_splits})"
    )
    max_kv_steps_per_split = max_kv_steps // num_kv_splits
    WAVES_WMMA_N = NUM_WAVES * WMMA_N

    n_m_blk = block_h // WMMA_M
    n_c_og = block_c // WAVES_WMMA_N

    # LDS bank-conflict pad. gfx1250 LDS = 32 banks × 4B/bank = 128 B/cycle.
    # Most natural row strides (64, 128, 512 ...) are multiples of 128 B, so
    # consecutive rows hit the same bank → 8-way conflict. Adding 16 B pad
    # per row breaks the periodicity (gcd(stride/4, 32)=4 → ~4-way reduces
    # to ~1-way). Only LDS slots whose layout we fully control go here; for
    # K-nope (TDM) padding has to be done via TDM pad_interval/pad_amount
    # (see copy_kv_nope_to_lds).
    LDS_PAD_BYTES = 16

    # Padded row strides for the bank-conflict-prone layouts.
    lds_p_row_stride       = block_n + LDS_PAD_BYTES   # 64 + 16 = 80 B
    # softmax row width is NUM_WAVES * WMMA_K_LANES = 8 f32 = 32 B per row.
    # Adding 1 f32 (4 B) pad → 9 f32 = 36 B per row → gcd(9, 32) = 1 in
    # 4-byte units, fully bank-uniform.
    lds_softmax_row_stride_elems = NUM_WAVES * WMMA_K_LANES + 1   # 9 f32 / row
    # K-nope and Q-nope use TDM with row pad (pad_interval=block_c=128 dwords).
    # K-rope and Q-rope ALSO move via TDM (separate small tile, no pad)
    # instead of `global_load_async_to_lds` to eliminate loadcnt stalls
    # (am profile: wave_loadcnt_waitcnt_stall was 11 308 cycles / 6.8%
    # of kernel time when rope used async path). TDM tile_dim0=block_r=64 B
    # is below the 128 B "direct copy" threshold so the rope TDM falls back
    # to indirect path, but it stays on tensorcnt — no loadcnt at all.
    lds_kv_nope_row_stride = block_c + LDS_PAD_BYTES   # 528 B
    lds_q_nope_row_stride  = block_c + LDS_PAD_BYTES   # same
    # K-rope / Q-rope also pad row (LDS write addresses computed by TDM via
    # the tile_dim0 walk, no extra pad needed since one row per pad_interval=64).
    # Use the same +16 B pad to break the 8-way bank conflict on rope reads.
    lds_kv_rope_row_stride = block_r + LDS_PAD_BYTES   # 80 B
    lds_q_rope_row_stride  = block_r + LDS_PAD_BYTES   # 80 B

    # softmax has 2 separate scratch slots (one for max-reduce, one for
    # sum-reduce). With shared slot, reduce_row would need an extra barrier
    # before each store to keep prior iter's reads from racing this iter's
    # writes; with disjoint slots that pre-barrier disappears (saves 16
    # barriers per launch — 1 per reduce × 2 reduces × 8 iters).
    NUM_SOFTMAX_SLOTS = 2

    # LDS element counts and byte sizes (all fp8 = 1 B/elem unless noted)
    lds_q_nope_elems  = block_h * lds_q_nope_row_stride
    lds_q_rope_elems  = block_h * lds_q_rope_row_stride
    lds_kv_nope_elems = block_n * lds_kv_nope_row_stride
    lds_kv_rope_elems = block_n * lds_kv_rope_row_stride
    lds_softmax_elems = block_h * lds_softmax_row_stride_elems
    lds_p_elems       = block_h * lds_p_row_stride

    lds_q_nope_bytes  = lds_q_nope_elems
    lds_q_rope_bytes  = lds_q_rope_elems
    lds_kv_nope_bytes = lds_kv_nope_elems
    lds_kv_rope_bytes = lds_kv_rope_elems
    lds_softmax_bytes = NUM_SOFTMAX_SLOTS * lds_softmax_elems * 4   # fp32
    lds_p_bytes       = lds_p_elems

    gpu_arch = get_hip_arch()
    if not str(gpu_arch).lower().startswith("gfx12"):
        raise RuntimeError(
            f"compile_mla_decode_fp8_gfx1250: requires GFX12, got {gpu_arch!r}"
        )
    assert nheads_q % block_h == 0
    assert seqlen_kv % block_n == 0
    assert block_n % WAVES_WMMA_N == 0
    # Nope path uses K=128 scaled WMMA — block_c must be a multiple of 128.
    assert block_c % 128 == 0, (
        f"block_c={block_c} must be a multiple of 128 for K=128 wmma_scale path"
    )
    num_head_blocks = nheads_q // block_h

    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="mla_decode_fp8_gfx1250_smem"
    )
    allocator.ptr = (
        lds_q_nope_bytes + lds_q_rope_bytes
        + NUM_KV_BUFS * (lds_kv_nope_bytes + lds_kv_rope_bytes)
        + lds_softmax_bytes + lds_p_bytes
    )
    assert allocator.ptr <= LDS_SIZE_LIMIT, (
        f"LDS total {allocator.ptr} exceeds limit {LDS_SIZE_LIMIT} "
        f"(NUM_KV_BUFS={NUM_KV_BUFS})"
    )

    _logit_scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(float(headdim_qk))
    # Fuse log2(e) into the sm_scale so all softmax math runs in log2-scaled
    # space: P = exp(sm·S - max) = exp2(log2e·(sm·S - max)) = exp2(c·S - c·max)
    # where c = sm_scale·log2e and the per-row max we track is c·max_orig.
    # The kernel uses bare exp2 in the hot loop (saves one mul per softmax
    # step) and converts max back to ln-space only when emitting LSE.
    _LOG2E = 1.4426950408889634
    _logit_scale_log2 = _logit_scale * _LOG2E

    # -------------------------------------------------------------------
    # Kernel
    # -------------------------------------------------------------------
    @flyc.kernel
    def mla_decode_fp8_gfx1250_kernel(
        Q: fx.Tensor,
        K_Buffer: fx.Tensor,
        V_buffer: fx.Tensor,
        Mid_O: fx.Tensor,
        Mid_lse: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_indices: fx.Tensor,
        batch: fx.Int32,
        num_kv_splits: fx.Int32,
        stride_b_block_table: fx.Int32,
        page_block_size: fx.Int32,
    ):
        # -- layouts (must be built inside @kernel for JIT MLIR context) --
        layout_wmma = fx.make_layout((2, 16), (16, 1))
        layout_tr_load = fx.make_layout((2, 4, 4), (4, 1, 8))
        # Q nope/rope and K nope/rope each have their own LDS slab; both
        # populated via TDM (no async global_load → no loadcnt stall).
        layout_lds_q_nope = fx.make_layout(
            (block_h, block_c), (lds_q_nope_row_stride, 1)
        )
        layout_lds_q_rope = fx.make_layout(
            (block_h, block_r), (lds_q_rope_row_stride, 1)
        )
        layout_lds_k_nope = fx.make_layout(
            (block_n, block_c), (lds_kv_nope_row_stride, 1)
        )
        layout_lds_k_rope = fx.make_layout(
            (block_n, block_r), (lds_kv_rope_row_stride, 1)
        )
        # V shares the K-nope slab (V = K[..., :block_c]). Col stride matches
        # K-nope row stride. Transpose-read via ds_load_tr8_b64.
        layout_lds_v = fx.make_layout(
            (block_c, block_n), (1, lds_kv_nope_row_stride)
        )
        layout_lds_softmax = fx.make_layout(
            (block_h, NUM_WAVES * WMMA_K_LANES),
            (lds_softmax_row_stride_elems, 1),
        )
        # P LDS uses padded row stride to break the 8-way bank conflict.
        layout_lds_p = fx.make_layout((block_h, block_n), (lds_p_row_stride, 1))

        # -- type aliases --
        ty_32xf8 = T.vec(32, T.f8)
        ty_64xf8 = T.vec(64, T.f8)
        ty_8xf32 = T.vec(8, T.f32)
        ty_2xi32 = T.vec(2, T.i32)
        ty_16xi32 = T.vec(16, T.i32)

        # -- fragment packing helpers --
        def _pack_4x8xfp8_to_32xf8(v0, v1, v2, v3):
            v01 = vector.shuffle(v0, v1, list(range(16)))
            v23 = vector.shuffle(v2, v3, list(range(16)))
            return vector.bitcast(ty_32xf8, vector.shuffle(v01, v23, list(range(32))))

        def _pack_8x8xfp8_to_64xf8(v0, v1, v2, v3, v4, v5, v6, v7):
            # K=128 fragment per lane = 64 fp8 = 8 × vec<8xfp8>
            v01 = vector.shuffle(v0, v1, list(range(16)))
            v23 = vector.shuffle(v2, v3, list(range(16)))
            v45 = vector.shuffle(v4, v5, list(range(16)))
            v67 = vector.shuffle(v6, v7, list(range(16)))
            v0_31 = vector.shuffle(v01, v23, list(range(32)))
            v32_63 = vector.shuffle(v45, v67, list(range(32)))
            return vector.bitcast(ty_64xf8, vector.shuffle(v0_31, v32_63, list(range(64))))

        def _fp8_vec32_to_8xi32(v32_f8):
            return vector.bitcast(T.vec(8, T.i32), v32_f8)

        def _fp8_vec64_to_16xi32(v64_f8):
            return vector.bitcast(ty_16xi32, v64_f8)

        # -- WMMA fragment load: row-major LDS (K-contiguous) --
        def load_wmma_frag_kmajor(lds_memref, layout_lds, lane_id_v, tile_nonK, tile_K):
            lc = idx2crd(lane_id_v, layout_wmma)
            lane_Kgrp = layout_get(lc, 0)
            nonK_idx = tile_nonK + layout_get(lc, 1)

            def _load_step(step):
                K_idx = tile_K + arith.index(step * 16) + lane_Kgrp * arith.index(8)
                return vector.load_op(T.f8x8, lds_memref, [crd2idx([nonK_idx, K_idx], layout_lds)])

            return _pack_4x8xfp8_to_32xf8(*[_load_step(s) for s in range_constexpr(4)])

        # K=128 variant: 8 step × 16 K each = 128 K total per (lane_kgrp half).
        # lane_kgrp ∈ {0,1} keeps the same 8-elem stride so the K=128 fragment
        # stays compatible with v_wmma_scale_*_16x16x128_f8f6f4 expectations.
        def load_wmma_frag_kmajor_k128(lds_memref, layout_lds, lane_id_v, tile_nonK, tile_K):
            lc = idx2crd(lane_id_v, layout_wmma)
            lane_Kgrp = layout_get(lc, 0)
            nonK_idx = tile_nonK + layout_get(lc, 1)

            def _load_step(step):
                K_idx = tile_K + arith.index(step * 16) + lane_Kgrp * arith.index(8)
                return vector.load_op(T.f8x8, lds_memref, [crd2idx([nonK_idx, K_idx], layout_lds)])

            return _pack_8x8xfp8_to_64xf8(*[_load_step(s) for s in range_constexpr(8)])

        # -- WMMA fragment load: transpose-read via ds_load_tr8_b64 --
        def load_wmma_frag_tr(lds_memref, layout_lds, lane_id_v, tile_nonK, tile_K):
            lc = idx2crd(lane_id_v, layout_tr_load)
            lane_nonKgrp = layout_get(lc, 0)
            lane_Kid = layout_get(lc, 1)
            lane_Kgrp = layout_get(lc, 2)
            nonK_idx = tile_nonK + lane_nonKgrp * arith.index(DS_LOAD_TR_VEC)

            def _tr_load_step(step):
                K_idx = tile_K + arith.index(step * 16) + lane_Kgrp * arith.index(4) + lane_Kid
                elem_off = crd2idx([nonK_idx, K_idx], layout_lds)
                base_ptr = memref_dialect.extract_aligned_pointer_as_index(lds_memref)
                ptr_i64 = arith.index_cast(T.i64, arith.addi(base_ptr, _raw(elem_off)))
                ds_ptr = llvm_dialect.inttoptr(ir.Type.parse("!llvm.ptr<3>"), ptr_i64)
                return vector.bitcast(T.f8x8, rocdl.ds_load_tr8_b64(ty_2xi32, ds_ptr))

            return _pack_4x8xfp8_to_32xf8(*[_tr_load_step(s) for s in range_constexpr(WMMA_K_GRP)])

        # -- WMMA instruction wrapper --
        def _wmma_fp8(a_f8x32, b_f8x32, c_8xf32):
            a_i = _fp8_vec32_to_8xi32(a_f8x32)
            b_i = _fp8_vec32_to_8xi32(b_f8x32)
            out = rocdl.wmma_f32_16x16x64_fp8_fp8(ty_8xf32, a_i, b_i, c_8xf32)
            return out.result if hasattr(out, "result") else out

        # -- K=128 WMMA via v_wmma_scale_f32_16x16x128_f8f6f4 with identity
        # E8M0 scales (each byte = 0x7F = 2^0 = 1.0; 4 K-blocks of 32 per call,
        # packed as 0x7F7F7F7F per i32). 2× compute throughput per call vs the
        # K=64 form. fmtA=fmtB=0 → FP8/E4M3, fmtScaleA/B=0 → E8M0 scale format.
        identity_e8m0_i32 = arith.constant(0x7F7F7F7F, type=T.i32)

        def _wmma_fp8_scaled_k128(a_f8x64, b_f8x64, c_8xf32):
            a_i = _fp8_vec64_to_16xi32(a_f8x64)
            b_i = _fp8_vec64_to_16xi32(b_f8x64)
            out = rocdl.wmma_scale_f32_16x16x128_f8f6f4(
                ty_8xf32, a_i, b_i, c_8xf32,
                identity_e8m0_i32, identity_e8m0_i32,
                fmtA=0, fmtB=0,
                fmtScaleA=0, fmtScaleB=0,
            )
            return out.result if hasattr(out, "result") else out

        # -- LDS allocation --
        # Q nope/rope: separate slabs; K nope/rope: separate slabs × NUM_KV_BUFS
        # for ping-pong prefetch. Both rope slots are populated via TDM (same
        # tensorcnt path as nope) — eliminates the loadcnt stall completely.
        base = allocator.get_base()
        lds_off = 0
        smem_q_nope = SmemPtr(base, lds_off, T.f8, shape=(lds_q_nope_elems,))
        lds_off += lds_q_nope_bytes
        smem_q_rope = SmemPtr(base, lds_off, T.f8, shape=(lds_q_rope_elems,))
        lds_off += lds_q_rope_bytes
        smem_kv_nope_bufs = [
            SmemPtr(base, lds_off + i * lds_kv_nope_bytes,
                    T.f8, shape=(lds_kv_nope_elems,))
            for i in range(NUM_KV_BUFS)
        ]
        lds_off += NUM_KV_BUFS * lds_kv_nope_bytes
        smem_kv_rope_bufs = [
            SmemPtr(base, lds_off + i * lds_kv_rope_bytes,
                    T.f8, shape=(lds_kv_rope_elems,))
            for i in range(NUM_KV_BUFS)
        ]
        lds_off += NUM_KV_BUFS * lds_kv_rope_bytes
        smem_softmax_slots = [
            SmemPtr(base, lds_off + i * lds_softmax_elems * 4,
                    T.f32, shape=(lds_softmax_elems,))
            for i in range(NUM_SOFTMAX_SLOTS)
        ]
        lds_off += lds_softmax_bytes
        smem_p = SmemPtr(base, lds_off, T.f8, shape=(lds_p_elems,))

        lds_q_nope_mem = smem_q_nope.get()
        lds_q_rope_mem = smem_q_rope.get()
        lds_kv_nope_mems = [smem_kv_nope_bufs[i].get() for i in range(NUM_KV_BUFS)]
        lds_kv_rope_mems = [smem_kv_rope_bufs[i].get() for i in range(NUM_KV_BUFS)]
        lds_softmax_max_mem, lds_softmax_sum_mem = (
            smem_softmax_slots[0].get(), smem_softmax_slots[1].get()
        )
        lds_p_mem = smem_p.get()

        # -- HBM buffer resources --
        mid_o_rsrc = buffer_ops.create_buffer_resource(Mid_O, max_size=True)
        mid_lse_rsrc = buffer_ops.create_buffer_resource(Mid_lse, max_size=True)

        # -- compile-time index values --
        block_h_idx = arith.index(block_h)
        block_c_idx = arith.index(block_c)
        block_r_idx = arith.index(block_r)
        num_q_heads_idx = arith.index(nheads_q)
        num_kv_splits_v = arith.index_cast(T.index, num_kv_splits.ir_value())
        stride_mid_o_split = arith.muli(num_q_heads_idx, block_c_idx)
        stride_mid_o_token = arith.muli(num_kv_splits_v, stride_mid_o_split)
        stride_mid_lse_token = arith.muli(num_kv_splits_v, num_q_heads_idx)

        # -- thread / block indexing --
        lane_id = gpu.thread_id("x")
        wave_id = gpu.thread_id("y")
        batch_idx = gpu.block_id("x")
        head_group_idx = gpu.block_id("y")
        split_id = gpu.block_id("z")

        _lc = idx2crd(lane_id, layout_wmma)
        lane_kgrp = layout_get(_lc, 0)
        lane16 = layout_get(_lc, 1)

        # -- accumulator init --
        acc_zero = arith.constant_vector(0.0, ty_8xf32)
        acc_o = [
            [arith.constant_vector(0.0, ty_8xf32) for _ in range(n_c_og)]
            for _ in range(n_m_blk)
        ]
        e_sum_ty = T.vec(n_m_blk, T.f32)
        e_sum = arith.constant_vector(0.0, e_sum_ty)
        e_max = arith.constant_vector(-float("inf"), e_sum_ty)

        # -- paged KV index computation --
        page_block_size_v = arith.index_cast(T.index, page_block_size.ir_value())
        kv_indptr_rsrc = buffer_ops.create_buffer_resource(kv_indptr, max_size=True)
        indptr_start_i32 = buffer_ops.buffer_load(
            kv_indptr_rsrc, batch_idx, vec_width=1, dtype=T.i32
        )
        indptr_end_i32 = buffer_ops.buffer_load(
            kv_indptr_rsrc, batch_idx + arith.index(1), vec_width=1, dtype=T.i32
        )
        cur_batch_seq_len = (
            arith.index_cast(T.index, indptr_end_i32)
            - arith.index_cast(T.index, indptr_start_i32)
        )
        cur_batch_block_nums = (
            cur_batch_seq_len + page_block_size_v - arith.index(1)
        ) / page_block_size_v
        blocks_per_split = (
            cur_batch_block_nums + num_kv_splits_v - arith.index(1)
        ) / num_kv_splits_v
        split_kv_start = blocks_per_split * split_id
        split_kv_end = arith.select(
            split_kv_start + blocks_per_split > cur_batch_block_nums,
            cur_batch_block_nums,
            split_kv_start + blocks_per_split,
        )

        stride_bt = arith.index_cast(T.index, stride_b_block_table.ir_value())
        kv_indices_rsrc = buffer_ops.create_buffer_resource(kv_indices, max_size=True)
        nvh_idx = arith.index(num_kv_heads)

        def k_hbm_row_tok0(ks: int):
            off_p = arith.addi(
                arith.muli(batch_idx, stride_bt),
                arith.addi(split_kv_start, arith.index(ks)),
            )
            phys_i32 = buffer_ops.buffer_load(
                kv_indices_rsrc, off_p, vec_width=1, dtype=T.i32
            )
            phys = arith.index_cast(T.index, phys_i32)
            return arith.muli(phys, arith.muli(page_block_size_v, nvh_idx))

        # -- global / LDS pointer helpers (for async rope loads) --
        def _global_byte_ptr(tensor, g_byte_elem):
            glb_ptr_ty = ir.Type.parse("!llvm.ptr<1>")
            i64_ty = ir.IntegerType.get_signless(64)
            raw = tensor.__fly_values__()[0]
            base = fly_d.extract_aligned_pointer_as_index(glb_ptr_ty, raw)
            base_i64 = llvm_dialect.ptrtoint(i64_ty, base)
            g_i64 = arith.index_cast(T.i64, g_byte_elem)
            addr_i64 = llvm_dialect.AddOp(
                base_i64, g_i64, llvm_dialect.IntegerOverflowFlags(0),
            ).result
            return llvm_dialect.inttoptr(glb_ptr_ty, addr_i64)

        def _lds_byte_ptr(lds_mem, lds_byte_off):
            lds_base = memref_dialect.extract_aligned_pointer_as_index(lds_mem)
            ptr_i64 = arith.index_cast(T.i64, arith.addi(lds_base, lds_byte_off))
            return llvm_dialect.inttoptr(ir.Type.parse("!llvm.ptr<3>"), ptr_i64)

        # ---------------------------------------------------------------
        # TDM / async-copy helpers
        # ---------------------------------------------------------------
        zero_off = arith.index(0)
        head_d = arith.index(headdim_qk)
        q_row_g = arith.addi(
            arith.muli(batch_idx, arith.index(nheads_q)),
            arith.muli(head_group_idx, block_h_idx),
        )

        # Issue-only TDM helpers. Each (Q, KV) load is now SPLIT into two TDM
        # transfers — one for nope (with row pad) and one for rope (no extra
        # pad). The rope path that used to use global_load_async (loadcnt) is
        # now also TDM (tensorcnt), eliminating the wave_loadcnt_waitcnt_stall
        # that the am profile flagged as the dominant remaining bottleneck.
        # Each copy_*_to_lds() therefore issues 2 tensorcnt operations.
        TDM_OPS_PER_KV_COPY = 2
        TDM_OPS_PER_Q_COPY  = 2

        def copy_q_to_lds():
            # nope: tile (block_h, block_c=512) padded
            tdm_n = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=Q, lds_memref=lds_q_nope_mem,
                global_offset=(q_row_g, zero_off),
                tensor_shape=(nheads_q, headdim_qk),
                strides=(headdim_qk, 1),
                tile_shape=(block_h, block_c),
                elem_bytes=1,
                pad_interval=block_c, pad_amount=LDS_PAD_BYTES,
                num_warps=NUM_WAVES,
            )
            tdm_ops.tensor_load_2d(tdm_n)
            # rope: tile (block_h, block_r=64), starting at column kv_lora_rank
            tdm_r = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=Q, lds_memref=lds_q_rope_mem,
                global_offset=(q_row_g, arith.index(kv_lora_rank)),
                tensor_shape=(nheads_q, headdim_qk),
                strides=(headdim_qk, 1),
                tile_shape=(block_h, block_r),
                elem_bytes=1,
                pad_interval=block_r, pad_amount=LDS_PAD_BYTES,
                num_warps=NUM_WAVES,
            )
            tdm_ops.tensor_load_2d(tdm_r)

        def copy_kv_to_lds(buf_idx, k_row_glob):
            tdm_n = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=K_Buffer, lds_memref=lds_kv_nope_mems[buf_idx],
                global_offset=(k_row_glob, zero_off),
                tensor_shape=(seqlen_kv, headdim_qk),
                strides=(headdim_qk, 1),
                tile_shape=(block_n, block_c),
                elem_bytes=1,
                pad_interval=block_c, pad_amount=LDS_PAD_BYTES,
                num_warps=NUM_WAVES,
            )
            tdm_ops.tensor_load_2d(tdm_n)
            tdm_r = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=K_Buffer, lds_memref=lds_kv_rope_mems[buf_idx],
                global_offset=(k_row_glob, arith.index(kv_lora_rank)),
                tensor_shape=(seqlen_kv, headdim_qk),
                strides=(headdim_qk, 1),
                tile_shape=(block_n, block_r),
                elem_bytes=1,
                pad_interval=block_r, pad_amount=LDS_PAD_BYTES,
                num_warps=NUM_WAVES,
            )
            tdm_ops.tensor_load_2d(tdm_r)

        def wait_kv_buf_ready(prefetch_in_flight):
            """Block until the KV buffer is ready (and visible cross-wave
            when needed).

            When block_n == WAVES_WMMA_N (=64): TDM partitions
            (compute_warp_distribution → [4, 1]) load exactly the rows each
            wave will then read in GEMM1 (`wave_id * WMMA_N`); wave-local
            tensor_wait suffices, the V cross-wave visibility is already
            handled by the post-`store_p_to_lds` barrier.

            When block_n > WAVES_WMMA_N (e.g., 128 with 4 waves): GEMM1
            access is INTERLEAVED — wave w reads rows {w*16, w*16+64, ...},
            spanning multiple TDM-partitioned slices owned by other waves.
            We need an explicit cross-wave barrier here.
            """
            tdm_ops.tensor_wait(prefetch_in_flight * TDM_OPS_PER_KV_COPY)
            if block_n // WAVES_WMMA_N > 1:
                gpu.barrier()

        # ---------------------------------------------------------------
        # LDS → register fragment load functions
        # ---------------------------------------------------------------
        # Nope path uses K=128 fragments to drive v_wmma_scale_*_16x16x128.
        # block_c (kv_lora_rank=512) must be a multiple of 128 for clean
        # iteration; we assert that at compile time below.
        WMMA_K_NOPE = 128

        def load_q_nope():
            return [
                [load_wmma_frag_kmajor_k128(lds_q_nope_mem, layout_lds_q_nope, lane_id,
                     arith.index(off_m), arith.index(off_k))
                 for off_k in range(0, block_c, WMMA_K_NOPE)]
                for off_m in range(0, block_h, WMMA_M)
            ]

        # rope and nope live in separate LDS slabs again — base column
        # offsets are 0 for both.
        def load_q_rope():
            return [
                [load_wmma_frag_kmajor(lds_q_rope_mem, layout_lds_q_rope, lane_id,
                     arith.index(off_m), arith.index(off_k))
                 for off_k in range(0, block_r, WMMA_K)]
                for off_m in range(0, block_h, WMMA_M)
            ]

        def load_k_nope(buf_idx):
            return [
                [load_wmma_frag_kmajor_k128(lds_kv_nope_mems[buf_idx], layout_lds_k_nope, lane_id,
                     arith.index(off_n) + wave_id * arith.index(WMMA_N),
                     arith.index(off_k))
                 for off_k in range(0, block_c, WMMA_K_NOPE)]
                for off_n in range(0, block_n, WAVES_WMMA_N)
            ]

        def load_k_rope(buf_idx):
            return [
                [load_wmma_frag_kmajor(lds_kv_rope_mems[buf_idx], layout_lds_k_rope, lane_id,
                     arith.index(off_n) + wave_id * arith.index(WMMA_N),
                     arith.index(off_k))
                 for off_k in range(0, block_r, WMMA_K)]
                for off_n in range(0, block_n, WAVES_WMMA_N)
            ]

        def load_v_from_lds(buf_idx):
            return [
                [load_wmma_frag_tr(
                    lds_kv_nope_mems[buf_idx], layout_lds_v, lane_id,
                    arith.addi(
                        arith.muli(arith.index(og), arith.index(WAVES_WMMA_N)),
                        arith.muli(wave_id, arith.index(WMMA_N)),
                    ),
                    arith.index(off_k),
                 )
                 for og in range_constexpr(n_c_og)]
                for off_k in range_constexpr(block_n // WMMA_K)
            ]

        def load_p_from_lds():
            return [
                [load_wmma_frag_kmajor(lds_p_mem, layout_lds_p, lane_id,
                     arith.index(off_m), arith.index(off_k))
                 for off_k in range_constexpr(block_n // WMMA_K)]
                for off_m in range_constexpr(n_m_blk)
            ]

        # ---------------------------------------------------------------
        # Softmax helpers
        # ---------------------------------------------------------------
        def all_reduce(vals, reduce_op):
            elems = [
                vector.extract(vals[ns], static_position=[i], dynamic_position=[])
                for ns in range_constexpr(block_n // WAVES_WMMA_N)
                for i in range_constexpr(WMMA_VEC)
            ]
            return functools.reduce(reduce_op, elems)

        def reduce_row(lane_vals, reduce_op, slot_mem):
            """Cross-wave reduce via LDS scratch slot.

            Caller must pass a `slot_mem` that is NOT aliased with anything
            still being read/written by other waves at the time of this
            call. The two slots `lds_softmax_max_mem` / `lds_softmax_sum_mem`
            are dedicated for max- and sum-reduces respectively, so the
            previous iteration's reads can never collide with this iter's
            writes — the pre-barrier (B1) needed for shared-slot reuse is
            therefore eliminated.
            """
            reduced = all_reduce(lane_vals, reduce_op)
            store_idx = crd2idx(
                [lane_id % 16, (lane_id // 16) + wave_id * WMMA_K_LANES],
                layout_lds_softmax,
            )
            memref_dialect.store(reduced, slot_mem, [store_idx])
            gpu.barrier()
            load_idx = crd2idx([lane_id % 16, 0], layout_lds_softmax)
            row_vals = vector.load_op(ty_8xf32, slot_mem, [load_idx])
            row_elems = [
                vector.extract(row_vals, static_position=[i], dynamic_position=[])
                for i in range_constexpr(WMMA_VEC)
            ]
            return functools.reduce(reduce_op, row_elems)

        def reduce_max(row_acc):
            return reduce_row(row_acc, arith.maximumf, lds_softmax_max_mem)

        def reduce_sum(row_acc):
            return reduce_row(row_acc, arith.addf, lds_softmax_sum_mem)

        # ---------------------------------------------------------------
        # GEMM1: S = Q_nope @ K_nope^T + Q_PE @ K_PE^T
        # ---------------------------------------------------------------
        def gemm1(q_nope_frags, q_rope_frags, k_nope_frags, k_rope_frags):
            # Rope-first: rope (K=64) consumes K-rope LDS which has a strictly
            # shorter lifetime than K-nope (V is also nope). Issuing the rope
            # WMMA first frees rope LDS earlier and lets the K=128 nope WMMAs
            # co-execute with rope's cleanup. Nope WMMAs are big (4 K=128
            # scaled WMMAs per (m,n) tile) so they want to dominate the back
            # half of gemm1.
            res = [
                [arith.constant_vector(0.0, ty_8xf32) for _ in range_constexpr(block_n // WAVES_WMMA_N)]
                for _ in range_constexpr(n_m_blk)
            ]
            for bm in range_constexpr(n_m_blk):
                for bn in range_constexpr(block_n // WAVES_WMMA_N):
                    cfrag = res[bm][bn]
                    # Rope first (block_r=64, 1 K=64 WMMA per tile).
                    for bk in range_constexpr(block_r // WMMA_K):
                        cfrag = _wmma_fp8(k_rope_frags[bn][bk], q_rope_frags[bm][bk], cfrag)
                    # Nope after (block_c=512, 4 K=128 scaled WMMAs per tile).
                    for bk in range_constexpr(block_c // WMMA_K_NOPE):
                        cfrag = _wmma_fp8_scaled_k128(
                            k_nope_frags[bn][bk], q_nope_frags[bm][bk], cfrag,
                        )
                    res[bm][bn] = cfrag
            return res

        # ---------------------------------------------------------------
        # GEMM2: O += P @ V
        # ---------------------------------------------------------------
        def gemm2(p_frags, v_frags, acc_in):
            res = acc_in
            for bm in range_constexpr(n_m_blk):
                for bc in range_constexpr(n_c_og):
                    cfrag = res[bm][bc]
                    for bk in range_constexpr(block_n // WMMA_K):
                        cfrag = _wmma_fp8(v_frags[bk][bc], p_frags[bm][bk], cfrag)
                    res[bm][bc] = cfrag
            return res

        # ---------------------------------------------------------------
        # Local exp2 helpers — used in the hot-loop softmax. sm_scale·log2e
        # is already baked into the pre-softmax `scale_s_logits`, so the
        # softmax's max/diff are in log2-scaled space and just need exp2.
        # ---------------------------------------------------------------
        def _exp2(x):
            return _mlir_math.Exp2Op(_raw(x)).result

        def _exp2_v8(v8):
            return vector.from_elements(ty_8xf32, [
                _exp2(vector.extract(v8, static_position=[i], dynamic_position=[]))
                for i in range_constexpr(8)
            ])

        # ---------------------------------------------------------------
        # Scale vector<8xf32> by scalar
        # ---------------------------------------------------------------
        def _scale_vec8(v, s):
            return vector.from_elements(ty_8xf32, [
                _std_arith.MulFOp(
                    _raw(vector.extract(v, static_position=[i], dynamic_position=[])),
                    _raw(s),
                ).result
                for i in range_constexpr(8)
            ])

        # sm_scale * log2e baked together so softmax can use bare exp2.
        sm_logit_log2 = arith.constant(_logit_scale_log2, type=T.f32)

        def scale_s_logits(s_in):
            return [[_scale_vec8(frag, sm_logit_log2) for frag in row] for row in s_in]

        # ---------------------------------------------------------------
        # Softmax: online running-max/sum with accumulator rescaling
        # ---------------------------------------------------------------
        def softmax(S, e_sum_old, e_max_old, acc_old):
            P_out, e_sum_list, e_max_list, acc_out = [], [], [], []
            for bm in range_constexpr(n_m_blk):
                max_old = vector.extract(e_max_old, static_position=[bm], dynamic_position=[])
                max_new = arith.maximumf(max_old, reduce_max(S[bm]))
                e_max_list.append(max_new)

                # exp2 (not exp) here — sm_scale·log2e is pre-baked, so
                # max_old/max_new and S all sit in log2-scaled space.
                alpha = _exp2(_std_arith.SubFOp(_raw(max_old), _raw(max_new)).result)
                max_splat = vector.broadcast(ty_8xf32, max_new)
                P_row = [
                    _exp2_v8(arith.subf(S[bm][bn], max_splat))
                    for bn in range_constexpr(block_n // WAVES_WMMA_N)
                ]
                P_out.append(P_row)

                sum_old = vector.extract(e_sum_old, static_position=[bm], dynamic_position=[])
                sum_new = sum_old * alpha + reduce_sum(P_row)
                e_sum_list.append(sum_new)

                acc_row = [
                    arith.mulf(acc_old[bm][bc], vector.broadcast(ty_8xf32, alpha))
                    for bc in range_constexpr(n_c_og)
                ]
                acc_out.append(acc_row)
            return (
                P_out,
                vector.from_elements(e_sum_ty, e_sum_list),
                vector.from_elements(e_sum_ty, e_max_list),
                acc_out,
            )

        # ---------------------------------------------------------------
        # Store P (f32 → fp8) to LDS
        # ---------------------------------------------------------------
        def store_p_to_lds(p):
            i32_cvt = ir.IntegerType.get_signless(32)
            _z_i32 = _std_arith.ConstantOp(i32_cvt, ir.IntegerAttr.get(i32_cvt, 0)).result

            def _cvt_wsel(a, b, wsel):
                pp = rocdl.cvt_pk_fp8_f32(i32_cvt, a, b, _z_i32, wsel)
                return pp.result if hasattr(pp, "result") else pp

            for bm in range_constexpr(n_m_blk):
                for bn in range_constexpr(block_n // WAVES_WMMA_N):
                    p_frag = p[bm][bn]
                    row_lds = arith.addi(arith.muli(arith.index(bm), arith.index(WMMA_M)), lane16)
                    # Each bn writes its own WAVES_WMMA_N-wide stripe along
                    # the col axis (= block_n axis of P). With block_n=128
                    # we have 2 bn iters → 2 stripes per (bm, wave).
                    base_c = arith.addi(
                        arith.addi(
                            arith.muli(lane_kgrp, arith.index(WMMA_VEC)),
                            arith.muli(wave_id, arith.index(WMMA_N)),
                        ),
                        arith.index(bn * WAVES_WMMA_N),
                    )
                    elems = [
                        vector.extract(p_frag, static_position=[i], dynamic_position=[])
                        for i in range_constexpr(8)
                    ]
                    pk0 = _cvt_wsel(elems[0], elems[1], False)
                    pk1 = _cvt_wsel(elems[2], elems[3], True)
                    pk2 = _cvt_wsel(elems[4], elems[5], False)
                    pk3 = _cvt_wsel(elems[6], elems[7], True)
                    v4_i32 = vector.from_elements(T.i32x4, [pk0, pk1, pk2, pk3])
                    v16_i8 = vector.bitcast(T.i8x16, v4_i32)
                    v8_i8 = vector.shuffle(v16_i8, v16_i8, [0, 1, 6, 7, 8, 9, 14, 15])
                    f8x8 = vector.bitcast(T.f8x8, v8_i8)
                    vector.store(f8x8, lds_p_mem, [crd2idx([row_lds, base_c], layout_lds_p)])

        # ---------------------------------------------------------------
        # Writeback: Mid_O and Mid_lse
        # ---------------------------------------------------------------
        def store_lse_to_global(lse_row):
            is_writer = _std_arith.CmpIOp(
                _std_arith.CmpIPredicate.eq, _raw(lane_kgrp), _raw(arith.index(0)),
            ).result
            w_ok = _std_arith.CmpIOp(
                _std_arith.CmpIPredicate.eq, _raw(wave_id), _raw(arith.index(0)),
            ).result
            writer_ok = _std_arith.AndIOp(is_writer, w_ok).result
            if_wr = _scf.IfOp(writer_ok, [], has_else=False)
            with ir.InsertionPoint(if_wr.regions[0].blocks[0]):
                for wm in range_constexpr(n_m_blk):
                    row_m = arith.addi(arith.muli(arith.index(wm), arith.index(WMMA_M)), lane16)
                    global_h = arith.addi(arith.muli(head_group_idx, block_h_idx), row_m)
                    in_b = _std_arith.CmpIOp(
                        _std_arith.CmpIPredicate.ult, _raw(global_h), _raw(num_q_heads_idx),
                    ).result
                    if_row = _scf.IfOp(in_b, [], has_else=False)
                    with ir.InsertionPoint(if_row.regions[0].blocks[0]):
                        v0 = vector.extract(lse_row, static_position=[wm], dynamic_position=[])
                        off = arith.addi(
                            arith.addi(
                                arith.muli(batch_idx, stride_mid_lse_token),
                                arith.muli(split_id, num_q_heads_idx),
                            ),
                            global_h,
                        )
                        buffer_ops.buffer_store(v0, mid_lse_rsrc, off)
                        _scf.YieldOp([])
                _scf.YieldOp([])

        def store_acc_o_to_global(acc_o_final):
            for bm in range_constexpr(n_m_blk):
                row_attn = arith.addi(arith.muli(arith.index(bm), arith.index(WMMA_M)), lane16)
                global_head = arith.addi(arith.muli(head_group_idx, block_h_idx), row_attn)
                for bc in range_constexpr(n_c_og):
                    og = arith.addi(
                        arith.muli(arith.index(bc), arith.index(NUM_WAVES)),
                        wave_id,
                    )
                    n_base = arith.muli(og, arith.index(WMMA_N))
                    acc_tile = acc_o_final[bm][bc]
                    for mi in range_constexpr(8):
                        col = arith.addi(
                            arith.addi(n_base, arith.muli(lane_kgrp, arith.index(8))),
                            arith.index(mi),
                        )
                        c_val = vector.extract(acc_tile, static_position=[mi], dynamic_position=[])
                        in_h = _std_arith.CmpIOp(
                            _std_arith.CmpIPredicate.ult, _raw(global_head), _raw(num_q_heads_idx),
                        ).result
                        in_d = _std_arith.CmpIOp(
                            _std_arith.CmpIPredicate.ult, _raw(col), _raw(block_c_idx),
                        ).result
                        in_b = _std_arith.AndIOp(in_h, in_d).result
                        if_o = _scf.IfOp(in_b, [], has_else=False)
                        with ir.InsertionPoint(if_o.regions[0].blocks[0]):
                            mid_o_off = arith.addi(
                                arith.addi(
                                    arith.addi(
                                        arith.muli(batch_idx, stride_mid_o_token),
                                        arith.muli(split_id, stride_mid_o_split),
                                    ),
                                    arith.muli(global_head, block_c_idx),
                                ),
                                col,
                            )
                            buffer_ops.buffer_store(c_val, mid_o_rsrc, mid_o_off)
                            _scf.YieldOp([])

        # ===============================================================
        # Main body
        # ===============================================================
        # Prologue — issue Q (nope + rope) and the first KV slot ALL together
        # so they share HBM bandwidth in flight; one combined wait + barrier
        # below brings Q ready before load_q_*. The KV[0] prefetch is allowed
        # to stay in flight; the iter-0 wait_kv_buf_ready inside the loop
        # finalizes it just before consumption.
        # Q copy issues 2 TDM (nope+rope), KV copy issues 2 TDM. After both,
        # wait for Q to land (leave KV in flight) — all on tensorcnt only.
        copy_q_to_lds()                                   # +TDM_OPS_PER_Q_COPY tensorcnt
        if NUM_KV_BUFS > 1:
            copy_kv_to_lds(0, k_hbm_row_tok0(0))          # +TDM_OPS_PER_KV_COPY tensorcnt
            # Drain Q (=TDM_OPS_PER_Q_COPY), leave KV[0] in flight.
            tdm_ops.tensor_wait(TDM_OPS_PER_KV_COPY)
        else:
            tdm_ops.tensor_wait(0)
        gpu.barrier()
        # Q rope first (tiny — drives DSRD into rope WMMA's XDL pipe sooner).
        q_rope_frags = load_q_rope()
        q_nope_frags = load_q_nope()

        # Per-iteration counts of scheduling-relevant ops (used to size sched
        # hints). One (m,n) tile per (bm, bn) pair; for the default config
        # n_m_blk=1, n_warps=4, block_n/(NUM_WAVES*WMMA_N)=1 → 1 tile/iter.
        # Conservative — doubling actual count is fine; LLVM clamps to actual
        # available instructions.
        _n_tiles = n_m_blk * (block_n // WAVES_WMMA_N)
        _gemm1_mfma_per_tile = (block_r // WMMA_K) + (block_c // WMMA_K_NOPE)  # 1 + 4 = 5
        _gemm2_mfma_per_tile = block_n // WMMA_K                                # 1
        _gemm1_mfma_total = _n_tiles * _gemm1_mfma_per_tile
        _gemm2_mfma_total = n_m_blk * n_c_og * _gemm2_mfma_per_tile             # 1 * 8 * 1 = 8
        # ds_load_b128 estimates per (m,n) tile: rope frag = 4 loads
        # (load_wmma_frag_kmajor ×1 step×4 → actually loop 4×8 = ds_load count
        # is hardware-dependent from packing — use 4 as ballpark).
        _gemm1_dsrd_rope = 4 * _n_tiles
        _gemm1_dsrd_nope = 8 * _n_tiles
        _gemm2_dsrd = 8 * 2  # 8 K=64 wmmas × 2 frag loads each (V + P)

        for kstep in range_constexpr(max_kv_steps_per_split):
            cur_buf = kstep % NUM_KV_BUFS
            is_last = kstep == max_kv_steps_per_split - 1

            # Lock the schedule for this iteration so LLVM doesn't reorder
            # ops across iter boundaries (would defeat the explicit prefetch
            # placement).
            rocdl.sched_barrier(0)

            if NUM_KV_BUFS > 1:
                # Multi-buffer: issue prefetch for kstep+1 BEFORE waiting,
                # so the next iter's HBM→LDS overlaps with this iter's compute.
                if not is_last:
                    next_buf = (kstep + 1) % NUM_KV_BUFS
                    copy_kv_to_lds(next_buf, k_hbm_row_tok0(kstep + 1))
                    wait_kv_buf_ready(prefetch_in_flight=1)
                else:
                    wait_kv_buf_ready(prefetch_in_flight=0)
            else:
                # Single-buffer baseline: issue cur, wait, then consume.
                copy_kv_to_lds(0, k_hbm_row_tok0(kstep))
                wait_kv_buf_ready(prefetch_in_flight=0)

            # Rope LDS reads first — they're tiny (block_n×block_r=4 KB) and
            # feed gemm1's first WMMA, letting nope LDS reads overlap with
            # the rope WMMA via co-execution (XDL pipeline + DSRD pipeline).
            k_rope_frags = load_k_rope(cur_buf)
            k_nope_frags = load_k_nope(cur_buf)

            # gemm1 schedule hint: rope DSRD → rope MFMA → nope DSRD → nope MFMA
            # interleaved so the K=128 nope WMMAs (which run on the XDL pipe)
            # can co-execute with the next round's nope DSRDs.
            rocdl.sched_dsrd(_gemm1_dsrd_rope)
            rocdl.sched_mfma(_n_tiles)                  # rope WMMAs
            rocdl.sched_dsrd(_gemm1_dsrd_nope)
            rocdl.sched_mfma(_n_tiles * (block_c // WMMA_K_NOPE))  # nope WMMAs

            s_raw = gemm1(q_nope_frags, q_rope_frags, k_nope_frags, k_rope_frags)
            s = scale_s_logits(s_raw)
            p, e_sum, e_max, acc_o = softmax(s, e_sum, e_max, acc_o)

            store_p_to_lds(p)
            gpu.barrier()
            v_frags = load_v_from_lds(cur_buf)
            p_frags = load_p_from_lds()
            # gemm2 schedule hint: V/P DSRD followed by gemm2 MFMA chain.
            rocdl.sched_dsrd(_gemm2_dsrd)
            rocdl.sched_mfma(_gemm2_mfma_total)
            acc_o = gemm2(p_frags, v_frags, acc_o)

            rocdl.sched_barrier(0)

        # Epilogue: divide by e_sum, compute LSE, write to global
        one_f = arith.constant(1.0, type=T.f32)
        log2e = arith.constant(1.4426950408889634, type=T.f32)
        acc_o = [
            [
                _scale_vec8(
                    acc_o[bm][bc],
                    arith.divf(
                        one_f,
                        vector.extract(e_sum, static_position=[bm], dynamic_position=[]),
                    ),
                )
                for bc in range_constexpr(n_c_og)
            ]
            for bm in range_constexpr(n_m_blk)
        ]
        # LSE = max_orig + ln(e_sum). Our running `m` is in log2-scaled
        # space (= max_orig · log2e), so:
        #   max_orig + ln(d) = (m + log2(d)) / log2e
        lse_elems = []
        for bm in range_constexpr(n_m_blk):
            d = vector.extract(e_sum, static_position=[bm], dynamic_position=[])
            m = vector.extract(e_max, static_position=[bm], dynamic_position=[])
            log2_d = _mlir_math.Log2Op(_raw(d)).result
            sum_log2 = _std_arith.AddFOp(_raw(m), log2_d).result
            lse_e = _std_arith.DivFOp(sum_log2, _raw(log2e)).result
            lse_elems.append(lse_e)
        lse_row = vector.from_elements(e_sum_ty, lse_elems)

        store_lse_to_global(lse_row)
        store_acc_o_to_global(acc_o)

    # -------------------------------------------------------------------
    # Launch wrapper
    # -------------------------------------------------------------------
    @flyc.jit
    def launch_mla_decode_fp8_gfx1250(
        Q: fx.Tensor,
        K_Buffer: fx.Tensor,
        V_buffer: fx.Tensor,
        Mid_O: fx.Tensor,
        Mid_lse: fx.Tensor,
        kv_indptr: fx.Tensor,
        kv_indices: fx.Tensor,
        batch: fx.Int32,
        num_kv_splits: fx.Int32,
        stride_b_block_table: fx.Int32,
        page_block_size: fx.Int32,
        stream: fx.Stream,
    ):
        _ = (
            f"mla_decode_fp8_gfx1250_{seqlen_kv}_{block_n}_{nheads_q}_"
            f"{max_kv_steps}_sm{_logit_scale:.6f}_"
            f"nvh{num_kv_heads}_paged"
        )
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        gx = arith.index_cast(T.index, batch.ir_value())
        gy = arith.index(num_head_blocks)
        gz = arith.index_cast(T.index, num_kv_splits.ir_value())
        mla_decode_fp8_gfx1250_kernel(
            Q, K_Buffer, V_buffer, Mid_O, Mid_lse,
            kv_indptr, kv_indices,
            batch, num_kv_splits, stride_b_block_table, page_block_size,
        ).launch(
            grid=(gx, gy, gz),
            block=(WAVE_SIZE, NUM_WAVES, 1),
            stream=stream,
        )

    return launch_mla_decode_fp8_gfx1250


def compile_mla_decode_fp8_gfx1250_stage2(
    batch: int,
    num_q_heads: int,
    kv_lora_rank: int,
    num_kv_splits: int,
    NUM_WAVES: int = 4,
) -> Callable[..., Any]:
    """Stage2 — split-LSE merge for the GFX1250 MLA decode pipeline.

    Reads ``Mid_O [B, num_kv_splits, Hq, D_V]`` / ``Mid_lse [B, num_kv_splits, Hq]``
    (row-major, fp32) and writes ``O [B, Hq, D_V]`` (fp32). Each CTA handles one
    ``(batch, head)`` pair; the per-split scalar recurrence (``e_max`` / ``e_sum``)
    runs redundantly in every thread's registers (``num_kv_splits`` is small),
    so no LDS / barrier is needed. Each thread owns ``d_per_thread`` lanes of
    the output along D_V.

    Constraint: ``kv_lora_rank`` must be divisible by ``WAVE_SIZE * NUM_WAVES``.
    """
    assert (
        batch >= 1 and num_q_heads >= 1 and kv_lora_rank >= 1 and num_kv_splits >= 1
    )
    threads_cta = WAVE_SIZE * NUM_WAVES
    assert kv_lora_rank % threads_cta == 0, (
        f"compile_mla_decode_fp8_gfx1250_stage2: kv_lora_rank ({kv_lora_rank}) "
        f"must be divisible by {threads_cta} (WAVE_SIZE*NUM_WAVES)"
    )
    d_per_thread = kv_lora_rank // threads_cta

    stride_mid_o_batch = num_kv_splits * num_q_heads * kv_lora_rank
    stride_mid_o_split = num_q_heads * kv_lora_rank
    stride_mid_o_head = kv_lora_rank
    stride_mid_lse_batch = num_kv_splits * num_q_heads
    stride_mid_lse_split = num_q_heads
    stride_o_batch = num_q_heads * kv_lora_rank

    @flyc.kernel
    def mla_decode_fp8_gfx1250_stage2_kernel(
        Mid_O: fx.Tensor,
        Mid_lse: fx.Tensor,
        O: fx.Tensor,
    ):
        batch_idx = gpu.block_id("x")
        head_idx = gpu.block_id("y")
        lane_id = gpu.thread_id("x")
        wave_id = gpu.thread_id("y")
        tid = arith.addi(
            lane_id,
            arith.muli(wave_id, arith.index(WAVE_SIZE)),
        )
        threads_cta_i = arith.index(threads_cta)

        zero_f = arith.constant(0.0, type=T.f32)
        neg_inf = arith.constant(float("-inf"), type=T.f32)

        stride_mid_lse_batch_i = arith.index(stride_mid_lse_batch)
        stride_mid_lse_split_i = arith.index(stride_mid_lse_split)
        stride_mid_o_batch_i = arith.index(stride_mid_o_batch)
        stride_mid_o_split_i = arith.index(stride_mid_o_split)
        stride_mid_o_head_i = arith.index(stride_mid_o_head)
        stride_o_batch_i = arith.index(stride_o_batch)

        mid_o_rsrc = buffer_ops.create_buffer_resource(Mid_O, max_size=True)
        mid_lse_rsrc = buffer_ops.create_buffer_resource(Mid_lse, max_size=True)
        o_rsrc = buffer_ops.create_buffer_resource(O, max_size=True)

        acc_chunk_ty = ir.Type.parse(f"memref<{d_per_thread}xf32>")
        acc_chunk = memref_dialect.alloca(acc_chunk_ty, [], [])
        for kk in range_constexpr(d_per_thread):
            memref_dialect.store(zero_f, acc_chunk, [arith.index(kk)])

        e_max = neg_inf
        e_sum = zero_f

        for s in range_constexpr(num_kv_splits):
            off_lse = arith.addi(
                arith.addi(
                    arith.muli(batch_idx, stride_mid_lse_batch_i),
                    arith.muli(arith.index(s), stride_mid_lse_split_i),
                ),
                head_idx,
            )
            m_scalar = buffer_ops.buffer_load(
                mid_lse_rsrc, off_lse, vec_width=1, dtype=T.f32
            )
            n_max = _std_arith.MaximumFOp(_raw(e_max), _raw(m_scalar)).result
            d_em = _std_arith.SubFOp(_raw(e_max), _raw(n_max)).result
            d_mn = _std_arith.SubFOp(_raw(m_scalar), _raw(n_max)).result
            scale_old = _exp_f32(d_em)
            scale_new = _exp_f32(d_mn)
            e_sum = _std_arith.AddFOp(
                _std_arith.MulFOp(_raw(e_sum), _raw(scale_old)).result,
                _raw(scale_new),
            ).result
            e_max = n_max

            for kk in range_constexpr(d_per_thread):
                d_idx = arith.addi(
                    tid,
                    arith.muli(arith.index(kk), threads_cta_i),
                )
                off_o = arith.addi(
                    arith.addi(
                        arith.addi(
                            arith.muli(batch_idx, stride_mid_o_batch_i),
                            arith.muli(arith.index(s), stride_mid_o_split_i),
                        ),
                        arith.muli(head_idx, stride_mid_o_head_i),
                    ),
                    d_idx,
                )
                v_el = buffer_ops.buffer_load(
                    mid_o_rsrc, off_o, vec_width=1, dtype=T.f32
                )
                old_a = memref_dialect.load(acc_chunk, [arith.index(kk)])
                t0 = _std_arith.MulFOp(_raw(old_a), _raw(scale_old)).result
                t1 = _std_arith.MulFOp(_raw(scale_new), _raw(v_el)).result
                new_a = _std_arith.AddFOp(_raw(t0), _raw(t1)).result
                memref_dialect.store(new_a, acc_chunk, [arith.index(kk)])

        pos = _std_arith.CmpFOp(
            _std_arith.CmpFPredicate.OGT,
            _raw(e_sum),
            _raw(zero_f),
        ).result
        if_done = _scf.IfOp(pos, [], has_else=True)
        with ir.InsertionPoint(if_done.regions[0].blocks[0]):
            for kk in range_constexpr(d_per_thread):
                d_idx = arith.addi(
                    tid,
                    arith.muli(arith.index(kk), threads_cta_i),
                )
                acc_d = memref_dialect.load(acc_chunk, [arith.index(kk)])
                out_d = _std_arith.DivFOp(_raw(acc_d), _raw(e_sum)).result
                off_out = arith.addi(
                    arith.addi(
                        arith.muli(batch_idx, stride_o_batch_i),
                        arith.muli(head_idx, stride_mid_o_head_i),
                    ),
                    d_idx,
                )
                buffer_ops.buffer_store(out_d, o_rsrc, off_out)
            _scf.YieldOp([])
        with ir.InsertionPoint(if_done.regions[1].blocks[0]):
            for kk in range_constexpr(d_per_thread):
                d_idx = arith.addi(
                    tid,
                    arith.muli(arith.index(kk), threads_cta_i),
                )
                off_out = arith.addi(
                    arith.addi(
                        arith.muli(batch_idx, stride_o_batch_i),
                        arith.muli(head_idx, stride_mid_o_head_i),
                    ),
                    d_idx,
                )
                buffer_ops.buffer_store(zero_f, o_rsrc, off_out)
            _scf.YieldOp([])

    @flyc.jit
    def launch_mla_decode_fp8_gfx1250_stage2(
        Mid_O: fx.Tensor,
        Mid_lse: fx.Tensor,
        O: fx.Tensor,
        stream: fx.Stream,
    ):
        _ = (
            f"mla_decode_fp8_gfx1250_stage2_b{batch}_h{num_q_heads}_"
            f"d{kv_lora_rank}_s{num_kv_splits}_w{NUM_WAVES}"
        )
        gx = arith.index(batch)
        gy = arith.index(num_q_heads)
        gz = arith.index(1)
        mla_decode_fp8_gfx1250_stage2_kernel(Mid_O, Mid_lse, O).launch(
            grid=(gx, gy, gz),
            block=(WAVE_SIZE, NUM_WAVES, 1),
            stream=stream,
        )

    return launch_mla_decode_fp8_gfx1250_stage2


def compile_mla_decode_fp8_gfx1250_2stage(
    batch: int,
    num_q_heads: int,
    num_kv_splits: int,
    seqlen_kv: int = 512,
    num_kv_heads: int = 1,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    block_h: int = 16,
    block_n: int = 64,
    block_c: int = 512,
    block_r: int = 64,
    sm_scale: Optional[float] = None,
    NUM_WAVES: int = 4,
):
    """Compile both Stage1 and Stage2 of the GFX1250 fp8 MLA decode pipeline.

    Returns ``(launch_stage1, launch_stage2)``. The two launches must be invoked
    with the same ``batch`` / ``num_kv_splits``.
    """
    launch_s1 = compile_mla_decode_fp8_gfx1250(
        nheads_q=num_q_heads,
        seqlen_kv=seqlen_kv,
        num_kv_heads=num_kv_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_h=block_h,
        block_n=block_n,
        block_c=block_c,
        block_r=block_r,
        sm_scale=sm_scale,
        NUM_WAVES=NUM_WAVES,
        num_kv_splits=num_kv_splits,
    )
    launch_s2 = compile_mla_decode_fp8_gfx1250_stage2(
        batch=batch,
        num_q_heads=num_q_heads,
        kv_lora_rank=kv_lora_rank,
        num_kv_splits=num_kv_splits,
        NUM_WAVES=NUM_WAVES,
    )
    return launch_s1, launch_s2
