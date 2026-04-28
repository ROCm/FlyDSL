# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FP4 MQA Logits kernel (gfx950, paged preshuffle KV cache).

Computes: logits[b, t] = sum_h(relu(Q[b,h,:] · K[b,t,:]) * weight[b,h])

PoC constraints: H=64, D=128, next_n=1, gfx950 only.

Data format:
  Q:        [B, H=64, D/2=64] uint8 (packed fp4 e2m1)
  Q_scale:  [B, H=64, D/32=4] uint8 (float8_e8m0fnu)
  KV cache: paged preshuffle fp4, [num_blocks, 4, block_size, 16] uint8
  KV_scale: [num_blocks, block_size, 4] uint8 (e8m0fnu)
  weights:  [B, H=64] fp32
  output:   [B, T_max] fp32

MFMA thread mapping (mfma_scale_f32_16x16x128_f8f6f4, cbsz=4):
  lane_id & 15  → M row (A) or N col (B)
  lane_id >> 4  → K chunk index (0..3), each chunk = 16 bytes = 32 FP4 elements
  i32x8 lower 128 bits = 16 bytes of FP4 data per thread
  scale: i32 = 4 packed e8m0 bytes covering 4×32=128 FP4 elements

MFMA sub-tiling for FP4 (_fp4_pack_M=2, _fp4_pack_N=2):
  Each logical 16×16 MFMA tile is decomposed into 2×2 sub-tiles.
  opselA (0..1) selects the M sub-tile, opselB (0..1) selects the N sub-tile.
  Total 4 MFMA calls per logical tile.
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, rocdl, vector
from flydsl.expr.typing import T, Int32
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import scf as _scf

HEADS = 64
HEAD_DIM = 128
HEAD_DIM_PACKED = 64       # D/2 bytes (fp4 packed)
HEAD_DIM_SCALES = 4        # D/32 scale groups
MFMA_M = 16
MFMA_N = 16
FP4_PACK_M = 2             # sub-tiles in M per MFMA logical tile
FP4_PACK_N = 2             # sub-tiles in N per MFMA logical tile
NUM_WARPS = 4
WARP_SIZE = 64
BLOCK_THREADS = NUM_WARPS * WARP_SIZE  # 256
M_TILES = HEADS // MFMA_M  # 4
M_TILES_PACKED = M_TILES // FP4_PACK_M  # 2


def _pack_i32_pair_to_i64(a_i32, b_i32):
    v = vector.from_elements(T.vec(2, T.i32), [a_i32, b_i32])
    v1 = vector.bitcast(T.vec(1, T.i64), v)
    return vector.extract(v1, static_position=[0])


def _pack_i64x4_to_i32x8(x0, x1, x2, x3):
    v4 = vector.from_elements(T.vec(4, T.i64), [x0, x1, x2, x3])
    return vector.bitcast(T.vec(8, T.i32), v4)


allocator = None


# ── Host-side schedule for varctx + persistent CTA assignment ──────
# Inspired by gluon's `safe_chunks_per_cta`: pick the smallest "chunks per
# CTA" such that total CTAs ≤ available parallel units, then build a
# (cta → batch, chunk_start, chunk_count) lookup table so each CTA loads
# its assignment in one shot (vs. gluon's in-kernel scf.while walk).


def compute_varctx_schedule(
    context_lens,
    block_k,
    parallel_unit_num,
):
    """Compute persistent-grid schedule for varctx MQA logits.

    Args:
        context_lens: int32 CUDA tensor [batch], per-batch context length.
        block_k: chunk size in tokens.
        parallel_unit_num: target CTA count (typically TotalCuCount * WavePerEU).

    Returns:
        safe_chunks_per_cta: int — chunks each CTA processes (≤ this many).
        cta_batch: int32 CUDA tensor [total_ctas] — which batch each CTA serves.
        cta_chunk_start: int32 CUDA tensor [total_ctas] — first chunk index in batch.
        cta_chunk_count: int32 CUDA tensor [total_ctas] — actual chunks for this CTA.
        total_ctas: int — grid.x size.
    """
    device = context_lens.device
    ctx_list = context_lens.cpu().tolist()
    chunks_per_batch = [(c + block_k - 1) // block_k for c in ctx_list]
    max_chunks = max(chunks_per_batch) if chunks_per_batch else 1

    safe = max_chunks  # worst case: 1 CTA does all chunks of biggest batch
    for s in range(1, max_chunks + 1):
        ctas_per_b = [(c + s - 1) // s for c in chunks_per_batch]
        if sum(ctas_per_b) <= parallel_unit_num:
            safe = s
            break

    cta_batch, cta_chunk_start, cta_chunk_count = [], [], []
    for b, n_chunks in enumerate(chunks_per_batch):
        if n_chunks == 0:
            continue
        ctas_b = (n_chunks + safe - 1) // safe
        for split in range(ctas_b):
            start = split * safe
            count = min(safe, n_chunks - start)
            cta_batch.append(b)
            cta_chunk_start.append(start)
            cta_chunk_count.append(count)

    total = len(cta_batch)
    if total == 0:  # all-zero context — launch one no-op CTA
        cta_batch, cta_chunk_start, cta_chunk_count, total = [0], [0], [0], 1

    return (
        safe,
        torch.tensor(cta_batch, dtype=torch.int32, device=device),
        torch.tensor(cta_chunk_start, dtype=torch.int32, device=device),
        torch.tensor(cta_chunk_count, dtype=torch.int32, device=device),
        total,
    )


def build_pa_mqa_logits_fp4_module(
    block_k=64,
    kv_block_size=16,
    max_blocks_per_seq=256,
):
    """Build FP4 MQA logits kernel.

    Returns (kernel_fn, allocator).

    Grid: (batch_size, cdiv(T_max, block_k))
    Block: (BLOCK_THREADS=256,)
    """
    global allocator

    N_TILES = block_k // MFMA_N
    N_TILES_PACKED = N_TILES // FP4_PACK_N  # packed N-tile groups

    _stride_q_batch = HEADS * HEAD_DIM_PACKED
    _stride_qs_batch = HEADS * HEAD_DIM_SCALES
    _stride_w_batch = HEADS
    _stride_bt = max_blocks_per_seq

    # KV preshuffle layout: [block_id, tile=4, block_size, 16] uint8
    _stride_kv_block = 4 * kv_block_size * 16
    # KV_scale: [block_id, block_size, 4]
    _stride_kvs_block = kv_block_size * HEAD_DIM_SCALES

    # LDS for cross-warp logit accumulation
    # After per-warp head reduction, each warp has logits for its N-tile.
    # No cross-warp reduction needed since warps handle different N-tiles.
    allocator = SmemAllocator(None, arch="gfx950", global_sym_name="mqa_fp4_smem")
    allocator.ptr = 16  # minimal, no LDS needed for this approach

    @flyc.kernel
    def pa_mqa_logits_fp4_kernel(
        out_logits_ptr: fx.Tensor,
        q_ptr: fx.Tensor,
        q_scale_ptr: fx.Tensor,
        kv_cache_ptr: fx.Tensor,
        kv_scale_ptr: fx.Tensor,
        kv_indices_ptr: fx.Tensor,
        weights_ptr: fx.Tensor,
        context_lens_ptr: fx.Tensor,    # varctx: [B] int32
        cta_batch_ptr: fx.Tensor,       # [total_ctas] i32 — which batch
        cta_chunk_start_ptr: fx.Tensor, # [total_ctas] i32 — first chunk in batch
        cta_chunk_count_ptr: fx.Tensor, # [total_ctas] i32 — chunks to process
        stride_out_batch: Int32,
    ):
        tid = gpu.thread_idx.x
        pid = gpu.block_idx.x

        warp_id = tid >> 6
        lane_id = tid % WARP_SIZE
        lane_mod_16 = lane_id & 15
        lane_div_16 = (lane_id >> 4) & 3

        c_w = fx.Int32(WARP_SIZE)

        q_rsrc = buffer_ops.create_buffer_resource(q_ptr, max_size=True)
        qs_rsrc = buffer_ops.create_buffer_resource(q_scale_ptr, max_size=True)
        kv_rsrc = buffer_ops.create_buffer_resource(kv_cache_ptr, max_size=True)
        kvs_rsrc = buffer_ops.create_buffer_resource(kv_scale_ptr, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(kv_indices_ptr, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(weights_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_logits_ptr, max_size=True)
        ctx_lens_rsrc = buffer_ops.create_buffer_resource(context_lens_ptr, max_size=True)
        cta_b_rsrc = buffer_ops.create_buffer_resource(cta_batch_ptr, max_size=True)
        cta_cs_rsrc = buffer_ops.create_buffer_resource(cta_chunk_start_ptr, max_size=True)
        cta_cc_rsrc = buffer_ops.create_buffer_resource(cta_chunk_count_ptr, max_size=True)

        NEG_INF = arith.constant(float("-inf"), type=T.f32)
        ZERO_F = fx.Float32(0.0)
        c0_i64 = arith.constant(0, type=T.i64)
        c0_i32 = fx.Int32(0)

        # ── Persistent CTA assignment lookup ───────────────────────
        pid_b = buffer_ops.buffer_load(cta_b_rsrc, pid, vec_width=1, dtype=T.i32)
        chunk_start = buffer_ops.buffer_load(cta_cs_rsrc, pid, vec_width=1, dtype=T.i32)
        chunk_count = buffer_ops.buffer_load(cta_cc_rsrc, pid, vec_width=1, dtype=T.i32)
        context_len = buffer_ops.buffer_load(
            ctx_lens_rsrc, pid_b, vec_width=1, dtype=T.i32)

        # ── Q load (HOISTED out of chunk loop — reused across chunks) ──
        # Each thread holds its slice of Q for this batch's 64 heads.
        q_a_ops = []
        q_scale_ops = []
        for mi_idx in [0, 1, 2, 3]:
            q_row = mi_idx * MFMA_M + lane_mod_16
            q_col_bytes = lane_div_16 * 16
            q_off_bytes = pid_b * _stride_q_batch + q_row * HEAD_DIM_PACKED + q_col_bytes
            q_4xi32 = buffer_ops.buffer_load(
                q_rsrc, q_off_bytes // 4, vec_width=4, dtype=T.i32)
            q_i64_0 = _pack_i32_pair_to_i64(
                vector.extract(q_4xi32, static_position=[0]),
                vector.extract(q_4xi32, static_position=[1]),
            )
            q_i64_1 = _pack_i32_pair_to_i64(
                vector.extract(q_4xi32, static_position=[2]),
                vector.extract(q_4xi32, static_position=[3]),
            )
            q_a_ops.append(_pack_i64x4_to_i32x8(q_i64_0, q_i64_1, c0_i64, c0_i64))

            qs_off_i32 = pid_b * HEADS + q_row
            qs_4b = buffer_ops.buffer_load(qs_rsrc, qs_off_i32, vec_width=1, dtype=T.i32)
            q_scale_ops.append(
                (qs_4b >> (lane_div_16 * fx.Int32(8))) & fx.Int32(0xFF))

        # Weights for this batch (HOISTED out of chunk loop too).
        w_per_lane = []
        for mi_idx in [0, 1, 2, 3]:
            for elem in [0, 1, 2, 3]:
                h_idx = mi_idx * MFMA_M + lane_div_16 * 4 + elem
                w_per_lane.append(
                    buffer_ops.buffer_load(
                        w_rsrc, pid_b * _stride_w_batch + h_idx,
                        vec_width=1, dtype=T.f32))

        # ── Loop over this CTA's chunks (1..safe_chunks_per_cta) ──
        # FlyDSL scf.for needs at least one carried value; pass a dummy.
        # Loop bounds must be index type; loop var c_idx is index too.
        _dummy = arith.constant(0, type=T.i32)
        chunk_count_idx = arith.index_cast(T.index, chunk_count)
        for c_idx, _state in range(0, chunk_count_idx, 1, init=[_dummy]):
            c_idx_i32 = arith.index_cast(T.i32, c_idx)
            kv_chunk_start = (chunk_start + c_idx_i32) * fx.Int32(block_k)

            # Each warp processes one N-tile of 16 tokens within this chunk.
            ni_warp = warp_id
            token_base = kv_chunk_start + ni_warp * MFMA_N
            token_global = token_base + lane_mod_16
            token_valid = token_global < context_len

            safe_token = token_valid.select(token_global, c0_i32)
            block_idx_in_seq = safe_token // kv_block_size
            token_in_block = safe_token % kv_block_size

            phys_block = buffer_ops.buffer_load(
                bt_rsrc, pid_b * _stride_bt + block_idx_in_seq,
                vec_width=1, dtype=T.i32)
            phys_block = token_valid.select(phys_block, c0_i32)

            kv_off_bytes = (
                phys_block * _stride_kv_block
                + lane_div_16 * kv_block_size * 16
                + token_in_block * 16
            )
            kv_4xi32 = buffer_ops.buffer_load(
                kv_rsrc, kv_off_bytes // 4, vec_width=4, dtype=T.i32)

            kv_0 = token_valid.select(vector.extract(kv_4xi32, static_position=[0]), c0_i32)
            kv_1 = token_valid.select(vector.extract(kv_4xi32, static_position=[1]), c0_i32)
            kv_2 = token_valid.select(vector.extract(kv_4xi32, static_position=[2]), c0_i32)
            kv_3 = token_valid.select(vector.extract(kv_4xi32, static_position=[3]), c0_i32)

            kv_i64_0 = _pack_i32_pair_to_i64(kv_0, kv_1)
            kv_i64_1 = _pack_i32_pair_to_i64(kv_2, kv_3)
            kv_b = _pack_i64x4_to_i32x8(kv_i64_0, kv_i64_1, c0_i64, c0_i64)

            kvs_off_i32 = phys_block * kv_block_size + token_in_block
            kvs_4b = buffer_ops.buffer_load(
                kvs_rsrc, kvs_off_i32, vec_width=1, dtype=T.i32)
            kv_scale_val = (kvs_4b >> (lane_div_16 * fx.Int32(8))) & fx.Int32(0xFF)
            kv_scale_val = token_valid.select(kv_scale_val, c0_i32)

            zero = arith.constant_vector(0.0, T.f32x4)
            accs = []
            for mi_idx in [0, 1, 2, 3]:
                acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    T.f32x4,
                    [q_a_ops[mi_idx], kv_b, zero,
                     4, 4,
                     0, q_scale_ops[mi_idx],
                     0, kv_scale_val],
                )
                accs.append(acc)

            for mi_idx in [0, 1, 2, 3]:
                for elem in [0, 1, 2, 3]:
                    v = vector.extract(accs[mi_idx], static_position=[elem])
                    v = v.maximumf(ZERO_F)
                    v = v * w_per_lane[mi_idx * 4 + elem]
                    accs[mi_idx] = vector.insert(
                        v, accs[mi_idx], static_position=[elem], dynamic_position=[])

            logit_val = ZERO_F
            for mi_idx in [0, 1, 2, 3]:
                for elem in [0, 1, 2, 3]:
                    v = vector.extract(accs[mi_idx], static_position=[elem])
                    logit_val = logit_val + v

            for sh in [16, 32]:
                peer = logit_val.shuffle_xor(fx.Int32(sh), c_w)
                logit_val = logit_val + peer

            out_token = token_base + lane_mod_16
            in_bounds = out_token < context_len
            logit_val = in_bounds.select(logit_val, NEG_INF)
            out_off = pid_b * stride_out_batch + out_token
            buffer_ops.buffer_store(logit_val, out_rsrc, out_off)

            yield [_dummy]   # carry the dummy through the loop

    return pa_mqa_logits_fp4_kernel, allocator
