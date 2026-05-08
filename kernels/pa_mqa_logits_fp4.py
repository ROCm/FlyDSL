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
from flydsl.expr.primitive import range_constexpr
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


def compute_varctx_schedule(
    context_lens,
    block_k,
    parallel_unit_num,
    next_n=1,
):
    """Compute persistent-grid schedule for varctx MQA logits.

    Returns a SINGLE packed [total_ctas, 4] int32 tensor so the kernel can
    fetch its assignment in one buffer_load_dwordx4 instead of four separate
    dword loads. Layout per CTA: [batch_packed, chunk_start, chunk_count, context_len]
    where batch_packed = batch * next_n + next_n_idx (kernel decodes via /, %).

    Each (batch, chunk-split) is expanded into next_n CTAs — one per next_n
    query. KV is shared across them via L2 (matching gluon's approach).

    Args:
        context_lens: int32 CUDA tensor [batch], per-batch context length.
        block_k: chunk size in tokens.
        parallel_unit_num: target CTA count (typically TotalCuCount * WavePerEU).
        next_n: number of MTP queries per batch (1 = standard, 2 = MTP-1, ...).

    Returns:
        safe_chunks_per_cta: int — chunks each CTA processes (≤ this many).
        cta_info: int32 CUDA tensor [total_ctas, 4] — packed CTA assignment.
        total_ctas: int — grid.x size.
    """
    device = context_lens.device
    ctx_list = context_lens.cpu().tolist()
    chunks_per_batch = [(c + block_k - 1) // block_k for c in ctx_list]
    max_chunks = max(chunks_per_batch) if chunks_per_batch else 1

    safe = max_chunks  # worst case: 1 CTA does all chunks of biggest batch
    for s in range(1, max_chunks + 1):
        ctas_per_b = [(c + s - 1) // s for c in chunks_per_batch]
        if sum(ctas_per_b) * next_n <= parallel_unit_num:
            safe = s
            break

    rows = []  # each row: [batch_packed, chunk_start, chunk_count, context_len]
    for b, n_chunks in enumerate(chunks_per_batch):
        if n_chunks == 0:
            continue
        ctas_b = (n_chunks + safe - 1) // safe
        for split in range(ctas_b):
            start = split * safe
            count = min(safe, n_chunks - start)
            for n in range(next_n):
                rows.append([b * next_n + n, start, count, ctx_list[b]])

    if not rows:  # all-zero context — launch one no-op CTA
        rows = [[0, 0, 0, 0]]

    return (
        safe,
        torch.tensor(rows, dtype=torch.int32, device=device).contiguous(),
        len(rows),
    )


def build_pa_mqa_logits_fp4_module(
    block_k=128,
    kv_block_size=16,
    max_blocks_per_seq=256,
    max_chunks_per_cta=16,
    num_warps=4,
    next_n=1,
):
    """Build FP4 MQA logits kernel (pipelined, block_k=128).

    Returns (kernel_fn, allocator).

    Grid: (total_ctas,) from compute_varctx_schedule(..., next_n=next_n)
    Block: (BLOCK_THREADS=256,)

    `max_chunks_per_cta`: compile-time upper bound on the number of chunks any
    CTA will process. Used to statically unroll block-table prefetch in the
    prologue. Must be >= host-side `safe_chunks_per_cta`.

    `next_n`: number of MTP queries per batch (default 1 = standard MQA).
    Following gluon's design, each (batch, next_n_idx) is a separate CTA;
    KV is shared across the next_n CTAs via L2 cache. cta_info[0] holds
    batch_packed = batch * next_n + next_n_idx; the kernel decodes it.

    With block_k=128 and 4 warps, each warp processes N_TILES_PER_WARP=2 N-tiles
    per chunk (= 32 tokens), giving more compute window per chunk to hide load
    latency.
    """
    MAX_CHUNKS_PER_CTA = max_chunks_per_cta
    NUM_WARPS_K = num_warps          # build-time scoped (overrides module default)
    BLOCK_THREADS_K = NUM_WARPS_K * WARP_SIZE
    NEXT_N = next_n                  # build-time constexpr
    global allocator

    N_TILES = block_k // MFMA_N
    N_TILES_PACKED = N_TILES // FP4_PACK_N  # packed N-tile groups
    assert N_TILES % NUM_WARPS_K == 0, \
        f"block_k={block_k} → N_TILES={N_TILES} must be multiple of num_warps={NUM_WARPS_K}"
    N_TILES_PER_WARP = N_TILES // NUM_WARPS_K

    # Q/Q_scale layout: [B, NEXT_N, H, D/2 or D/32]
    _stride_q_next_n = HEADS * HEAD_DIM_PACKED            # bytes per next_n slice
    _stride_q_batch = NEXT_N * _stride_q_next_n           # bytes per batch
    _stride_qs_next_n = HEADS * HEAD_DIM_SCALES           # bytes per next_n slice
    _stride_qs_batch = NEXT_N * _stride_qs_next_n         # bytes per batch
    # qs_off is in i32 units (×4 bytes). _stride_qs_*_i32 = bytes / 4.
    _stride_qs_next_n_i32 = _stride_qs_next_n // 4        # = HEADS * HEAD_DIM_SCALES // 4
    # Weights/output addressed by batch_packed (= b*NEXT_N + n) directly.
    _stride_w_batch = HEADS
    _stride_bt = max_blocks_per_seq

    # KV preshuffle layout: [block_id, K_chunk=4, block_size, 32] uint8 (FP8 E4M3)
    # 32 bytes per K-chunk per token (vs 16 for FP4) — each chunk holds 32 FP8 K-elements.
    _kv_chunk_bytes = 32
    _stride_kv_block = 4 * kv_block_size * _kv_chunk_bytes
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
        cta_info_ptr: fx.Tensor,        # [total_ctas, 4] i32: [batch, chunk_start, chunk_count, ctx_len]
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
        cta_info_rsrc = buffer_ops.create_buffer_resource(cta_info_ptr, max_size=True)

        NEG_INF = arith.constant(float("-inf"), type=T.f32)
        ZERO_F = fx.Float32(0.0)
        c0_i64 = arith.constant(0, type=T.i64)
        c0_i32 = fx.Int32(0)

        # ── Persistent CTA assignment lookup (one dwordx4 load) ────
        # Layout per CTA: [batch_packed, chunk_start, chunk_count, ctx_len]
        # batch_packed = pid_b * NEXT_N + pid_next_n. Decoded below.
        cta_info_4xi32 = buffer_ops.buffer_load(
            cta_info_rsrc, pid * fx.Int32(4), vec_width=4, dtype=T.i32)
        batch_packed = vector.extract(cta_info_4xi32, static_position=[0])
        chunk_start = vector.extract(cta_info_4xi32, static_position=[1])
        chunk_count = vector.extract(cta_info_4xi32, static_position=[2])
        context_len = vector.extract(cta_info_4xi32, static_position=[3])

        # Decode batch + next_n. NEXT_N=1 ⇒ /1, %1: MLIR canonicalizer folds
        # the divide-by-1 to identity and the mod-by-1 to 0. NEXT_N=power-of-2
        # ⇒ shift/and. No Python-if here because @flyc.kernel rewrites all
        # `if` to scf.if (variables defined inside become branch-local).
        pid_b = batch_packed // fx.Int32(NEXT_N)
        pid_next_n = batch_packed % fx.Int32(NEXT_N)

        # ── Q load (HOISTED out of chunk loop — reused across chunks) ──
        # Q layout: [B, NEXT_N, H, D/2]. Each thread holds its slice of Q
        # for this (batch, next_n_idx)'s 64 heads.
        q_a_ops = []
        for mi_idx in [0, 1, 2, 3]:
            q_row = mi_idx * MFMA_M + lane_mod_16
            q_col_bytes = lane_div_16 * 16
            q_off_bytes = (
                pid_b * _stride_q_batch
                + pid_next_n * fx.Int32(_stride_q_next_n)
                + q_row * HEAD_DIM_PACKED
                + q_col_bytes
            )
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

        # Q scale: pre-shuffled host-side as [B, NEXT_N, K_chunks=4, lane_mod_16=16, mi_idx=4].
        # ONE dword load gives this thread's 4 mi_idx scales packed as bytes
        # [m0, m1, m2, m3]. Per-mi_idx extraction is a compile-time shift —
        # MFMA only reads byte 0, so upper-byte garbage is irrelevant.
        qs_off_i32 = (
            pid_b * fx.Int32(NEXT_N * HEADS)
            + pid_next_n * fx.Int32(_stride_qs_next_n_i32)
            + lane_div_16 * fx.Int32(16)
            + lane_mod_16
        )
        qs_4b = buffer_ops.buffer_load(qs_rsrc, qs_off_i32, vec_width=1, dtype=T.i32)
        q_scale_ops = [
            qs_4b,                          # mi_idx=0: byte 0 already correct
            qs_4b >> fx.Int32(8),           # mi_idx=1
            qs_4b >> fx.Int32(16),          # mi_idx=2
            qs_4b >> fx.Int32(24),          # mi_idx=3
        ]

        # Weights (HOISTED). mfma(kv,q) swap: per thread holds head =
        # mi_idx*16 + lane_mod_16 (4 heads, one per mi_idx). Each lane in the
        # warp loads 4 scalar weights addressed by lane_mod_16. (We can't
        # vec4-load: 4 mi_idx slices are 16 heads apart, not contiguous;
        # within a slice 16 lanes load 16 distinct heads via scalar dwords.)
        # weights shape: [B*NEXT_N, H] — addressed by batch_packed directly.
        w_per_lane_swap = []
        for mi_idx in [0, 1, 2, 3]:
            h_swap = mi_idx * MFMA_M + lane_mod_16
            w_scalar = buffer_ops.buffer_load(
                w_rsrc, batch_packed * _stride_w_batch + h_swap,
                vec_width=1, dtype=T.f32)
            w_per_lane_swap.append(w_scalar)

        # ── Step 3 (block_k=128): prologue + N-1 prefetch loop + epilogue ──
        # Per chunk: each warp handles N_TILES_PER_WARP=2 N-tiles (32 tokens).
        # Carry across iters: kv_cur, kvs_cur (consumed this iter) and
        # phys_next (used to issue NEXT iter's KV prefetch — pre-loaded one
        # iter ahead so its load latency is hidden by current iter's compute).
        # Splitting phys load from KV prefetch lets the compiler issue both
        # buffer_loads in parallel rather than serializing on the dependency
        # phys → kv_off → kv_load.

        def _load_phys(c_i32_arg):
            """Load phys_block for chunk c, all N_TILES_PER_WARP N-tiles."""
            phys_list = []
            for nt in range_constexpr(N_TILES_PER_WARP):
                ni_c = warp_id * fx.Int32(N_TILES_PER_WARP) + fx.Int32(nt)
                token_global_c = (
                    (chunk_start + c_i32_arg) * fx.Int32(block_k)
                    + ni_c * fx.Int32(MFMA_N)
                    + lane_mod_16
                )
                bi_c = token_global_c // kv_block_size
                phys_list.append(buffer_ops.buffer_load(
                    bt_rsrc, pid_b * _stride_bt + bi_c, vec_width=1, dtype=T.i32))
            return phys_list

        def _prefetch_chunk(c_i32_arg, phys_list):
            """Issue KV+scale loads for chunk c using pre-loaded phys_list.
            FP8 KV: 32 bytes per thread per K-chunk → two dwordx4 loads at
            byte offsets [+0, +16] within the 32-byte chunk.
            Returns (kv_lo_list, kv_hi_list, kvs_list) — each length N_TILES_PER_WARP."""
            kv_lo_list = []
            kv_hi_list = []
            kvs_list = []
            for nt in range_constexpr(N_TILES_PER_WARP):
                ni_c = warp_id * fx.Int32(N_TILES_PER_WARP) + fx.Int32(nt)
                token_global_c = (
                    (chunk_start + c_i32_arg) * fx.Int32(block_k)
                    + ni_c * fx.Int32(MFMA_N)
                    + lane_mod_16
                )
                # No address clamping — OOB tokens read garbage that is later
                # overwritten by NEG_INF via in_bounds.select on the store path.
                token_in_block_c = token_global_c % kv_block_size
                phys_block_c = phys_list[nt]
                kv_off_bytes_c = (
                    phys_block_c * _stride_kv_block
                    + lane_div_16 * kv_block_size * _kv_chunk_bytes
                    + token_in_block_c * _kv_chunk_bytes
                )
                kv_lo_c = buffer_ops.buffer_load(
                    kv_rsrc, kv_off_bytes_c // 4, vec_width=4, dtype=T.i32)
                kv_hi_c = buffer_ops.buffer_load(
                    kv_rsrc, (kv_off_bytes_c + 16) // 4, vec_width=4, dtype=T.i32)
                # KV scale pre-shuffled host-side as
                # [num_blocks, K_chunks=4, kv_block_size]. Each thread loads
                # its single byte directly — MFMA reads byte 0, no extraction.
                kvs_off_byte = (
                    phys_block_c * (kv_block_size * HEAD_DIM_SCALES)
                    + lane_div_16 * kv_block_size
                    + token_in_block_c
                )
                kvs_byte_c = buffer_ops.buffer_load(
                    kvs_rsrc, kvs_off_byte, vec_width=1, dtype=T.i8)
                kv_lo_list.append(kv_lo_c)
                kv_hi_list.append(kv_hi_c)
                kvs_list.append(kvs_byte_c)
            return kv_lo_list, kv_hi_list, kvs_list

        def _compute_chunk(kv_lo_list_in, kv_hi_list_in, kvs_list_in, c_i32_arg):
            """Process chunk c using prefetched (kv_lo, kv_hi, kvs).

            mfma(KV as A, Q as B) — output layout is (M=token, N=head):
              acc[mi_idx][elem] = s[t = lane_div_16*4 + elem,
                                    h = mi_idx*16 + lane_mod_16]
            """
            for nt in range_constexpr(N_TILES_PER_WARP):
                ni_warp = warp_id * fx.Int32(N_TILES_PER_WARP) + fx.Int32(nt)
                token_base = (
                    (chunk_start + c_i32_arg) * fx.Int32(block_k)
                    + ni_warp * fx.Int32(MFMA_N)
                )
                kv_lo = kv_lo_list_in[nt]
                kv_hi = kv_hi_list_in[nt]
                kvs_byte = kvs_list_in[nt]

                # Pack 32 KV bytes (lo: bytes 0..15, hi: bytes 16..31) as full
                # i32x8 — FP8 A operand consumes all 32 bytes (one full
                # 32-element K-chunk per thread).
                kv_i64_0 = _pack_i32_pair_to_i64(
                    vector.extract(kv_lo, static_position=[0]),
                    vector.extract(kv_lo, static_position=[1]),
                )
                kv_i64_1 = _pack_i32_pair_to_i64(
                    vector.extract(kv_lo, static_position=[2]),
                    vector.extract(kv_lo, static_position=[3]),
                )
                kv_i64_2 = _pack_i32_pair_to_i64(
                    vector.extract(kv_hi, static_position=[0]),
                    vector.extract(kv_hi, static_position=[1]),
                )
                kv_i64_3 = _pack_i32_pair_to_i64(
                    vector.extract(kv_hi, static_position=[2]),
                    vector.extract(kv_hi, static_position=[3]),
                )
                kv_a = _pack_i64x4_to_i32x8(kv_i64_0, kv_i64_1, kv_i64_2, kv_i64_3)
                kv_scale_val = arith.ArithValue(kvs_byte).extui(T.i32)

                zero = arith.constant_vector(0.0, T.f32x4)
                accs = []
                for mi_idx in [0, 1, 2, 3]:
                    # A=KV (FP8 E4M3, cbsz=0), B=Q (FP4, blgp=4).
                    # scale_A=kv_scale (UE8M0), scale_B=q_scale (UE8M0).
                    acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.f32x4,
                        [kv_a, q_a_ops[mi_idx], zero,
                         0, 4,
                         0, kv_scale_val,
                         0, q_scale_ops[mi_idx]],
                    )
                    accs.append(acc)

                # relu + per-head weight. acc[mi_idx][elem] head index depends
                # only on mi_idx and lane_mod_16 (not elem), so the weight
                # factor is the same for all 4 elem values within a mi_idx.
                for mi_idx in [0, 1, 2, 3]:
                    for elem in [0, 1, 2, 3]:
                        v = vector.extract(accs[mi_idx], static_position=[elem])
                        v = v.maximumf(ZERO_F)
                        v = v * w_per_lane_swap[mi_idx]
                        accs[mi_idx] = vector.insert(
                            v, accs[mi_idx], static_position=[elem], dynamic_position=[])

                # Per-thread reduction: for each elem (= different token),
                # sum over 4 mi_idx (= 4 heads at this lane_mod_16).
                # Yields 4 partial logits per thread, one per token.
                partials = []
                for elem in [0, 1, 2, 3]:
                    v0 = vector.extract(accs[0], static_position=[elem])
                    v1 = vector.extract(accs[1], static_position=[elem])
                    v2 = vector.extract(accs[2], static_position=[elem])
                    v3 = vector.extract(accs[3], static_position=[elem])
                    partials.append((v0 + v1) + (v2 + v3))

                # Cross-lane reduction: XOR butterfly over 16 lane_mod_16 lanes
                # (XOR by 1,2,4,8 toggles only the low 4 bits of lane_id, so
                # we stay within the same lane_div_16 group). After this each
                # thread's `partials[elem]` holds the full 64-head sum for
                # token = lane_div_16*4 + elem within this N-tile.
                #
                # XOR by 1, 2 stay within 4-lane quad — use DPP quad_perm
                # (~4 cyc, no LDS). XOR by 4, 8 cross quads — ds_bpermute.
                #   XOR 1: perm [1,0,3,2] = 1 | 0<<2 | 3<<4 | 2<<6 = 0xB1
                #   XOR 2: perm [2,3,0,1] = 2 | 3<<2 | 0<<4 | 1<<6 = 0x4E
                lane_raw = lane_id if not hasattr(lane_id, 'ir_value') else lane_id.ir_value()
                c4_i32 = arith.constant(4, type=T.i32)

                def _dpp_xor_add(val, dpp_ctrl):
                    val_raw = val if not hasattr(val, 'ir_value') else val.ir_value()
                    val_i32 = arith.ArithValue(val_raw).bitcast(T.i32)
                    peer_i32 = rocdl.update_dpp(
                        T.i32, val_i32, val_i32, dpp_ctrl, 0xF, 0xF, True)
                    peer_f32 = arith.ArithValue(peer_i32).bitcast(T.f32)
                    return arith.ArithValue(val_raw).addf(peer_f32)

                def _bperm_xor_add(val, sh):
                    sh_c = arith.constant(sh, type=T.i32)
                    peer_lane = arith.XOrIOp(lane_raw, sh_c).result
                    peer_byte = arith.MulIOp(peer_lane, c4_i32).result
                    val_raw = val if not hasattr(val, 'ir_value') else val.ir_value()
                    val_i32 = arith.ArithValue(val_raw).bitcast(T.i32)
                    peer_i32 = rocdl.ds_bpermute(T.i32, peer_byte, val_i32)
                    peer_f32 = arith.ArithValue(peer_i32).bitcast(T.f32)
                    return arith.ArithValue(val_raw).addf(peer_f32)

                for elem_idx in range_constexpr(4):
                    val = partials[elem_idx]
                    val = _dpp_xor_add(val, 0xB1)  # XOR by 1
                    val = _dpp_xor_add(val, 0x4E)  # XOR by 2
                    val = _bperm_xor_add(val, 4)
                    val = _bperm_xor_add(val, 8)
                    partials[elem_idx] = val

                # Each thread holds 4 final logits for tokens
                # [token_base + lane_div_16*4 + 0..3]. After the cross-lane
                # reduction the value is replicated across all 16 lane_mod_16
                # lanes within the same lane_div_16 group, so only one writer
                # per group is needed. Pack the 4 partials into a vec4 and
                # emit ONE buffer_store_dwordx4 instead of four scalar dwords.
                #
                # Per-(next_n) causal mask: query at next_n_idx n sees tokens
                # up to and including (context_len - NEXT_N + n). Equivalent:
                # out_token + (NEXT_N - 1 - n) < context_len. With NEXT_N>1
                # the effective bound is NOT a multiple of 4 (e.g. n=0 ⇒
                # context_len-1), so the boundary can split a vec4 group.
                # Apply mask per-elem BEFORE packing — OOB elems get NEG_INF
                # so the single vec4 store still works without losing BW.
                # NEXT_N=1: mask_off=0 → equivalent to old single base check.
                out_token0 = token_base + lane_div_16 * fx.Int32(4)
                mask_off = fx.Int32(NEXT_N - 1) - pid_next_n
                for elem in range_constexpr(4):
                    out_token_e = out_token0 + fx.Int32(elem)
                    in_ctx_e = (out_token_e + mask_off) < context_len
                    partials[elem] = in_ctx_e.select(partials[elem], NEG_INF)

                val_vec = vector.from_elements(
                    T.vec(4, T.f32),
                    [partials[0], partials[1], partials[2], partials[3]],
                )
                # Mask via address: non-writer lanes get OOB offset that the
                # buffer resource silently drops. The 4 tokens are contiguous
                # and the base is 16-byte aligned (token_base ≡ 0 mod 16,
                # lane_div_16*4 advances in 16-byte steps).
                oob_off = fx.Int32(-1)
                is_writer = lane_mod_16 < fx.Int32(1)
                out_off_real = batch_packed * stride_out_batch + out_token0
                out_off = is_writer.select(out_off_real, oob_off)
                buffer_ops.buffer_store(val_vec, out_rsrc, out_off)

        # === Prologue ===
        # (1) Load phys for chunk 0, then issue KV[0] prefetch using it.
        # (2) Pre-load phys for chunk 1 (carried into the loop's first iter
        #     so KV[1] prefetch doesn't have to wait on a fresh phys load).
        phys_pre = _load_phys(c0_i32)
        kv_lo_pre, kv_hi_pre, kvs_pre = _prefetch_chunk(c0_i32, phys_pre)
        phys_next_pre = _load_phys(fx.Int32(1))

        # === Main loop: chunk_count - 1 iterations ===
        # Carry = [kv_lo_cur_*, kv_hi_cur_*, kvs_cur_*, phys_next_*]
        # = 4 * N_TILES_PER_WARP entries (FP8 needs both kv halves carried).
        chunk_count_minus_1_i32 = chunk_count - fx.Int32(1)
        chunk_count_minus_1_idx = arith.index_cast(T.index, chunk_count_minus_1_i32)
        init_args = (
            list(kv_lo_pre) + list(kv_hi_pre) + list(kvs_pre) + list(phys_next_pre)
        )
        for c_idx, state in range(
                0, chunk_count_minus_1_idx, 1, init=init_args):
            kv_lo_cur_list = [state[i] for i in range(N_TILES_PER_WARP)]
            kv_hi_cur_list = [state[N_TILES_PER_WARP + i] for i in range(N_TILES_PER_WARP)]
            kvs_cur_list = [state[2 * N_TILES_PER_WARP + i] for i in range(N_TILES_PER_WARP)]
            phys_next_list = [state[3 * N_TILES_PER_WARP + i] for i in range(N_TILES_PER_WARP)]
            c_idx_i32 = arith.index_cast(T.i32, c_idx)
            c_next_i32 = c_idx_i32 + fx.Int32(1)
            c_next_next_i32 = c_next_i32 + fx.Int32(1)

            # Issue KV prefetch for chunk c+1 using carry phys (no extra wait).
            kv_lo_next, kv_hi_next, kvs_next = _prefetch_chunk(c_next_i32, phys_next_list)

            # Issue phys load for chunk c+2 — its latency overlaps with
            # _compute_chunk on the current chunk below.
            phys_next_next_list = _load_phys(c_next_next_i32)

            # Compute MFMA + store on current chunk
            _compute_chunk(kv_lo_cur_list, kv_hi_cur_list, kvs_cur_list, c_idx_i32)

            results = yield (
                list(kv_lo_next) + list(kv_hi_next) + list(kvs_next) + list(phys_next_next_list)
            )

        # === Epilogue: process last chunk (chunk_count - 1) ===
        kv_lo_last_list = [results[i] for i in range(N_TILES_PER_WARP)]
        kv_hi_last_list = [results[N_TILES_PER_WARP + i] for i in range(N_TILES_PER_WARP)]
        kvs_last_list = [results[2 * N_TILES_PER_WARP + i] for i in range(N_TILES_PER_WARP)]
        last_c_i32 = chunk_count - fx.Int32(1)
        _compute_chunk(kv_lo_last_list, kv_hi_last_list, kvs_last_list, last_c_i32)

    # Attach actual block threads count for the launcher (so the test can use
    # the right block dim when num_warps != module-level default).
    allocator.block_threads = BLOCK_THREADS_K
    return pa_mqa_logits_fp4_kernel, allocator
