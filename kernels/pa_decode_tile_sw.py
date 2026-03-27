"""
PA Decode Tile (Sliding Window): Unified f16/bf16 FlyDSL paged attention decode.

CuTe-style layout algebra with TiledMma + TiledCopy + fx.gemm().
Online softmax, split-K partitioning, GQA, paged KV cache.
Supports sliding window attention, attention sinks, and one-shot mode.

Features:
  - Unified f16/bf16 via dtype parameter
  - Sliding window: restrict attention to last SLIDING_WINDOW tokens
  - Sinks: pre-computed sink token attention scores (ONE_SHOT only)
  - ONE_SHOT: single partition writes directly to output (no reduce)

Data layouts:
  - K cache: [num_blocks, num_kv_heads, head_size//x, kv_block_size, x] (x=8)
  - V cache: [num_blocks, num_kv_heads, head_size, kv_block_size]
  - Query:   [batch_size, num_query_heads, head_size]
  - Output:  [batch_size, num_query_heads, head_size]

Grid mapping:
  - grid.x = num_partitions (split-K) or 1 (one_shot)
  - grid.y = num_kv_heads
  - grid.z = batch_size
"""
import functools
import math
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, gpu
from flydsl.expr.primitive import (make_view, add_offset, memref_load, memref_store)
from flydsl.expr.typing import T
from flydsl.expr import buffer_ops


# Fixed architecture constants
NUM_QUERIES = 16  # = query_group_size, fixed
HEAD_SIZE = 128
X_PACK = 8        # 2-byte elements: 16 bytes / 2 = 8
NUM_WARPS = 4
WARP_SIZE = 64
NUM_THREADS = NUM_WARPS * WARP_SIZE  # 256
BLOCK_N_TILE = 64
PV_N_ITERS = HEAD_SIZE // BLOCK_N_TILE  # 2

LOG2E = 1.4426950408889634

THREADS_PER_ROW_SM = 16
ELEMS_PER_THREAD_SM = BLOCK_N_TILE // THREADS_PER_ROW_SM  # 4
SM_SHUFFLE_BITS = [1, 2, 4, 8]

# LDS sizes (fixed for BLOCK_N_TILE=64, NUM_QUERIES=16, HEAD_SIZE=128)
QK_SCORES_ELEMS = NUM_QUERIES * BLOCK_N_TILE
QK_SCORES_BYTES = QK_SCORES_ELEMS * 4

PROBS_TILE_ELEMS = NUM_QUERIES * BLOCK_N_TILE
PROBS_TILE_BYTES = PROBS_TILE_ELEMS * 2  # 2 bytes per f16/bf16

K_TILE_ELEMS = BLOCK_N_TILE * HEAD_SIZE
V_CHUNK_ELEMS = BLOCK_N_TILE * BLOCK_N_TILE
STAGING_ELEMS = max(K_TILE_ELEMS, V_CHUNK_ELEMS)
STAGING_BYTES = STAGING_ELEMS * 2

ACCUM_ELEMS = NUM_QUERIES * HEAD_SIZE
ACCUM_BYTES = ACCUM_ELEMS * 4

STAGING_OFFSET_BYTES = QK_SCORES_BYTES + PROBS_TILE_BYTES
STAGING_OFFSET_HALF = STAGING_OFFSET_BYTES // 2  # works for both f16 and bf16

ACCUM_OFFSET_BYTES = STAGING_OFFSET_BYTES + STAGING_BYTES
ACCUM_OFFSET_F32 = ACCUM_OFFSET_BYTES // 4

# Q staging in LDS (bf16 BufferCopy from buffer_desc unsupported, stage through LDS)
Q_TILE_ELEMS = NUM_QUERIES * HEAD_SIZE   # 2048
Q_TILE_BYTES = Q_TILE_ELEMS * 2          # 4096
Q_OFFSET_BYTES = ACCUM_OFFSET_BYTES + (ACCUM_ELEMS * 4)
Q_OFFSET_HALF = Q_OFFSET_BYTES // 2
Q_ELEMS_PER_THREAD = Q_TILE_ELEMS // NUM_THREADS  # 8

K_ELEMS_PER_THREAD = K_TILE_ELEMS // NUM_THREADS
V_CHUNK_ELEMS_PER_THREAD = V_CHUNK_ELEMS // NUM_THREADS
ACCUM_PER_THREAD = ACCUM_ELEMS // NUM_THREADS

TOTAL_LDS = QK_SCORES_BYTES + PROBS_TILE_BYTES + STAGING_BYTES + ACCUM_BYTES + Q_TILE_BYTES

# Masking constant for out-of-window tokens
NEG_INF_MASK = -3.4e38


def _get_dtype_config(dtype):
    """Return MFMA atom type, copy atom type for dtype.

    Note: T_ELEM (MLIR IR type) cannot be created outside a kernel context,
    so we return the dtype string and resolve T_ELEM inside the kernel.
    """
    if dtype == "f16":
        return fx.Float16, fx.Float16
    elif dtype == "bf16":
        return fx.BFloat16, fx.BFloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Use 'f16' or 'bf16'.")


@functools.lru_cache(maxsize=64)
def compile_pa_decode_tile_sw(
    num_kv_heads,
    num_qh_per_kvh,
    kv_block_size,
    partition_size,
    max_blocks_per_seq,
    num_partitions,
    softmax_scale,
    sliding_window=0,
    one_shot=False,
    use_sinks=False,
    dtype="bf16",
):
    """Compile split-K and reduce kernels for PA decode tile (f16/bf16).

    Args:
        sliding_window: If > 0, restrict attention to last sliding_window tokens.
        one_shot: If True, single partition writes final output directly.
        use_sinks: If True, include sinks contribution in one_shot mode.
        dtype: "f16" or "bf16".
    """
    assert num_qh_per_kvh == NUM_QUERIES, \
        f"query_group_size ({num_qh_per_kvh}) must equal NUM_QUERIES ({NUM_QUERIES})"

    MFMA_TYPE, COPY_TYPE = _get_dtype_config(dtype)

    # K cache strides [num_blocks, num_kv_heads, head_size//x, kv_block_size, x]
    K_STRIDE_TOKEN = X_PACK
    K_STRIDE_HSPLIT = kv_block_size * X_PACK
    K_STRIDE_KVHEAD = (HEAD_SIZE // X_PACK) * kv_block_size * X_PACK
    K_STRIDE_BLOCK = num_kv_heads * K_STRIDE_KVHEAD

    # V cache strides [num_blocks, num_kv_heads, head_size, kv_block_size]
    V_STRIDE_TOKEN = 1
    V_STRIDE_HD = kv_block_size
    V_STRIDE_KVHEAD = HEAD_SIZE * kv_block_size
    V_STRIDE_BLOCK = num_kv_heads * V_STRIDE_KVHEAD

    SCALE = softmax_scale
    _sliding_window = sliding_window
    _one_shot = one_shot
    _use_sinks = use_sinks

    @flyc.kernel
    def pa_splitk_kernel(
        Q: fx.Tensor,
        K_cache: fx.Tensor,
        V_cache: fx.Tensor,
        Block_table: fx.Tensor,
        Context_lens: fx.Tensor,
        Partial_out: fx.Tensor,
        Max_logits: fx.Tensor,
        Exp_sums: fx.Tensor,
        Out: fx.Tensor,
        Sinks: fx.Tensor,
        NUM_QUERIES: fx.Constexpr[int],
        HEAD_SIZE: fx.Constexpr[int],
        BLOCK_N_TILE: fx.Constexpr[int],
        PV_N_ITERS: fx.Constexpr[int],
        PARTITION_SIZE: fx.Constexpr[int],
    ):
        tid = fx.thread_idx.x
        partition_idx = fx.block_idx.x
        kv_head_idx = fx.block_idx.y
        seq_idx = fx.block_idx.z

        # Resolve MLIR element type inside kernel context
        T_ELEM = T.bf16 if dtype == "bf16" else T.f16

        # KV cache head offsets
        k_head_offset = kv_head_idx * K_STRIDE_KVHEAD
        v_head_offset = kv_head_idx * V_STRIDE_KVHEAD

        # ═══ MMA and Copy Setup ═══
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, MFMA_TYPE))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((1, 4, 1), (1, 1, 0))
        )
        thr_mma = tiled_mma.thr_slice(tid)

        copy_atom_lds_half = fx.make_copy_atom(fx.UniversalCopy(32), COPY_TYPE)
        # Use LDS-based copy for Q (BufferCopy32b from buffer_desc fails for bf16)
        tiled_copy_A = fx.make_tiled_copy_A(copy_atom_lds_half, tiled_mma)
        thr_copy_A = tiled_copy_A.get_slice(tid)
        tiled_copy_B_lds = fx.make_tiled_copy_B(copy_atom_lds_half, tiled_mma)
        thr_copy_B_lds = tiled_copy_B_lds.get_slice(tid)
        tiled_copy_PV_A = fx.make_tiled_copy_A(copy_atom_lds_half, tiled_mma)
        thr_copy_PV_A = tiled_copy_PV_A.get_slice(tid)

        copy_atom_lds_f32 = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
        tiled_copy_C_lds = fx.make_tiled_copy_C(copy_atom_lds_f32, tiled_mma)
        thr_copy_C_lds = tiled_copy_C_lds.get_slice(tid)

        copy_atom_out = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        tiled_copy_C_out = fx.make_tiled_copy_C(copy_atom_out, tiled_mma)
        thr_copy_C_out = tiled_copy_C_out.get_slice(tid)

        # ═══ LDS Views ═══
        lds_raw = fx.get_dyn_shared()

        scores_ptr = fx.recast_iter(
            fx.PointerType.get(T.f32, fx.AddressSpace.Shared, QK_SCORES_ELEMS),
            lds_raw)
        scores_lds = make_view(scores_ptr,
                                fx.make_layout((NUM_QUERIES, BLOCK_N_TILE),
                                                (BLOCK_N_TILE, 1)))
        scores_1d = make_view(scores_ptr, fx.make_layout(QK_SCORES_ELEMS, 1))

        probs_base_ptr = fx.recast_iter(
            fx.PointerType.get(T_ELEM, fx.AddressSpace.Shared, PROBS_TILE_ELEMS),
            lds_raw)
        probs_ptr = add_offset(probs_base_ptr, QK_SCORES_BYTES // 2)
        probs_lds = make_view(probs_ptr,
                               fx.make_layout((NUM_QUERIES, BLOCK_N_TILE),
                                                (BLOCK_N_TILE, 1)))
        probs_1d = make_view(probs_ptr, fx.make_layout(PROBS_TILE_ELEMS, 1))

        staging_base_ptr = fx.recast_iter(
            fx.PointerType.get(T_ELEM, fx.AddressSpace.Shared, STAGING_ELEMS),
            lds_raw)
        staging_ptr = add_offset(staging_base_ptr, STAGING_OFFSET_HALF)
        k_staging_lds = make_view(staging_ptr,
                                   fx.make_layout((BLOCK_N_TILE, HEAD_SIZE),
                                                    (HEAD_SIZE, 1)))
        k_staging_1d = make_view(staging_ptr, fx.make_layout(K_TILE_ELEMS, 1))
        v_staging_lds = make_view(staging_ptr,
                                   fx.make_layout((BLOCK_N_TILE, BLOCK_N_TILE),
                                                    (BLOCK_N_TILE, 1)))
        v_staging_1d = make_view(staging_ptr, fx.make_layout(V_CHUNK_ELEMS, 1))

        accum_base_ptr = fx.recast_iter(
            fx.PointerType.get(T.f32, fx.AddressSpace.Shared, ACCUM_ELEMS),
            lds_raw)
        accum_ptr_v = add_offset(accum_base_ptr, ACCUM_OFFSET_F32)
        accum_lds = make_view(accum_ptr_v,
                               fx.make_layout((NUM_QUERIES, HEAD_SIZE),
                                                (HEAD_SIZE, 1)))
        accum_1d = make_view(accum_ptr_v, fx.make_layout(ACCUM_ELEMS, 1))

        # ═══ Q staging LDS view ═══
        q_staging_base_ptr = fx.recast_iter(
            fx.PointerType.get(T_ELEM, fx.AddressSpace.Shared, Q_TILE_ELEMS),
            lds_raw)
        q_staging_ptr = add_offset(q_staging_base_ptr, Q_OFFSET_HALF)
        q_staging_lds = make_view(q_staging_ptr,
                                   fx.make_layout((NUM_QUERIES, HEAD_SIZE),
                                                    (HEAD_SIZE, 1)))
        q_staging_1d = make_view(q_staging_ptr, fx.make_layout(Q_TILE_ELEMS, 1))

        rsrc_bt = buffer_ops.create_buffer_resource(Block_table)
        rsrc_ctx = buffer_ops.create_buffer_resource(Context_lens)

        # Initialize accum to zero
        for e in range_constexpr(ACCUM_PER_THREAD):
            flat_idx = tid * ACCUM_PER_THREAD + e
            memref_store(arith.constant(0.0, type=T.f32), accum_1d, flat_idx)

        # ═══ Load Q from global → LDS staging ═══
        rsrc_Q = buffer_ops.create_buffer_resource(Q)
        q_tile_idx = seq_idx * num_kv_heads + kv_head_idx
        q_global_base = q_tile_idx * Q_TILE_ELEMS  # offset in elements

        for e in range_constexpr(Q_ELEMS_PER_THREAD):
            flat_idx = tid * Q_ELEMS_PER_THREAD + e
            q_val = buffer_ops.buffer_load(rsrc_Q, q_global_base + flat_idx,
                                            vec_width=1, dtype=T_ELEM)
            memref_store(q_val, q_staging_1d, flat_idx)

        gpu.barrier()

        # ═══ Load runtime context length ═══
        context_len = buffer_ops.buffer_load(rsrc_ctx, seq_idx,
                                              vec_width=1, dtype=T.i32)

        # ═══ Compute attention window bounds ═══
        _zero_i32 = arith.constant(0, type=T.i32)
        if _sliding_window > 0:
            sw_const = arith.constant(_sliding_window, type=T.i32)
            raw_start = context_len - sw_const
            seq_start = arith.select(raw_start > _zero_i32, raw_start, _zero_i32)
            seq_end = context_len
        else:
            seq_start = _zero_i32
            seq_end = context_len

        # ═══ Q setup (from LDS) ═══
        rsrc_K = buffer_ops.create_buffer_resource(K_cache)
        rsrc_V = buffer_ops.create_buffer_resource(V_cache)

        tileQ = fx.make_tile(NUM_QUERIES, HEAD_SIZE)
        bQ = fx.slice(fx.zipped_divide(q_staging_lds, tileQ), (None, 0))
        copy_src_Q = thr_copy_A.partition_S(bQ)
        frag_Q = thr_mma.make_fragment_A(bQ)
        copy_frag_Q = thr_copy_A.retile(frag_Q)

        sm_row = tid // THREADS_PER_ROW_SM
        sm_col = tid % THREADS_PER_ROW_SM

        SCALE_C = arith.constant(SCALE, type=T.f32)
        LOG2E_C = arith.constant(LOG2E, type=T.f32)
        NEG_INF_C = arith.constant(NEG_INF_MASK, type=T.f32)

        # ═══ Compute partition token bounds (runtime) ═══
        # Each partition covers [part_tok_start, part_tok_end) in global token space.
        # partition_idx selects which partition this workgroup processes.
        c_part_size = arith.constant(partition_size, type=T.i32)
        partition_token_start = seq_start + partition_idx * c_part_size
        part_end_raw = partition_token_start + c_part_size
        partition_token_end = arith.select(seq_end < part_end_raw,
                                            seq_end, part_end_raw)
        bt_seq_base = seq_idx * max_blocks_per_seq

        # ═══ Runtime main loop over BLOCK_N_TILE-sized chunks ═══
        # Loop-carried state: [running_max, running_sum]
        # LDS accum is shared memory — updated via side effects, not carried.
        _loop_start = arith.index_cast(T.index, arith.unwrap(partition_token_start))
        _loop_stop = arith.index_cast(T.index, arith.unwrap(partition_token_end))
        _loop_step = arith.index(BLOCK_N_TILE)

        init_max = arith.constant(-1e30, type=T.f32)
        init_sum = arith.constant(0.0, type=T.f32)

        for iv, state in range(_loop_start, _loop_stop, _loop_step,
                               init=[init_max, init_sum]):
            running_max = state[0]
            running_sum = state[1]
            token_base = arith.index_cast(T.i32, iv)

            # --- Load K tile ---
            logical_block_idx = token_base // kv_block_size
            token_in_block_base = token_base % kv_block_size

            bt_offset = bt_seq_base + logical_block_idx
            physical_block = buffer_ops.buffer_load(
                rsrc_bt, bt_offset, vec_width=1, dtype=T.i32)
            k_page_base = physical_block * K_STRIDE_BLOCK + k_head_offset

            for e in range_constexpr(K_ELEMS_PER_THREAD):
                flat_idx = tid * K_ELEMS_PER_THREAD + e
                token_local = flat_idx // HEAD_SIZE
                head_dim = flat_idx % HEAD_SIZE
                token_in_block = token_in_block_base + token_local

                h_split = head_dim // X_PACK
                h_elem = head_dim % X_PACK
                k_within_page = (h_split * K_STRIDE_HSPLIT
                                + token_in_block * K_STRIDE_TOKEN
                                + h_elem)
                k_addr = k_page_base + k_within_page

                val = buffer_ops.buffer_load(rsrc_K, k_addr,
                                             vec_width=1, dtype=T_ELEM)
                memref_store(val, k_staging_1d, flat_idx)

            gpu.barrier()

            # --- QK GEMM ---
            tileK = fx.make_tile(BLOCK_N_TILE, HEAD_SIZE)
            bK = fx.slice(fx.zipped_divide(k_staging_lds, tileK), (None, 0))
            copy_src_K = thr_copy_B_lds.partition_S(bK)
            frag_K = thr_mma.make_fragment_B(bK)
            copy_frag_K = thr_copy_B_lds.retile(frag_K)

            tileC = fx.make_tile(NUM_QUERIES, BLOCK_N_TILE)
            bC_lds = fx.slice(fx.zipped_divide(scores_lds, tileC), (None, 0))
            copy_dst_C_lds = thr_copy_C_lds.partition_S(bC_lds)
            frag_C = thr_mma.make_fragment_C(bC_lds)
            copy_frag_C_lds = thr_copy_C_lds.retile(frag_C)

            # Zero-init scores_lds so frag_C starts at zero for QK GEMM
            # (MFMA does D = A*B + C; we need C = 0)
            for e in range_constexpr(ELEMS_PER_THREAD_SM):
                col = sm_col * ELEMS_PER_THREAD_SM + e
                flat_idx = sm_row * BLOCK_N_TILE + col
                memref_store(arith.constant(0.0, type=T.f32), scores_1d, flat_idx)
            gpu.barrier()

            fx.copy(copy_atom_lds_half, copy_src_Q, copy_frag_Q, pred=None)
            fx.copy(copy_atom_lds_half, copy_src_K, copy_frag_K, pred=None)
            fx.copy(copy_atom_lds_f32, copy_dst_C_lds, copy_frag_C_lds, pred=None)
            fx.gemm(mma_atom, frag_C, frag_Q, frag_K, frag_C)
            fx.copy(copy_atom_lds_f32, copy_frag_C_lds, copy_dst_C_lds, pred=None)

            gpu.barrier()

            # --- Online softmax with sliding window masking ---
            chunk_max = arith.constant(-1e30, type=T.f32)
            for e in range_constexpr(ELEMS_PER_THREAD_SM):
                col = sm_col * ELEMS_PER_THREAD_SM + e
                flat_idx = sm_row * BLOCK_N_TILE + col
                s = memref_load(scores_1d, flat_idx)
                s = s * SCALE_C

                # Sliding window masking: mask tokens outside the window
                if _sliding_window > 0:
                    global_token = token_base + col
                    in_window = (global_token >= seq_start) & (global_token < seq_end)
                    s = arith.select(in_window, s, NEG_INF_C)
                else:
                    # Standard masking: mask tokens beyond context length
                    global_token = token_base + col
                    valid = global_token < context_len
                    s = arith.select(valid, s, NEG_INF_C)

                memref_store(s, scores_1d, flat_idx)
                chunk_max = chunk_max.maximumf(s)

            for sh in SM_SHUFFLE_BITS:
                peer = chunk_max.shuffle_xor(arith.constant(sh, type=T.i32),
                                              arith.constant(64, type=T.i32))
                chunk_max = chunk_max.maximumf(peer)

            new_max = running_max.maximumf(chunk_max)
            rescale = ((running_max - new_max) * LOG2E_C).exp2(
                fastmath=arith.FastMathFlags.fast)
            running_sum = running_sum * rescale

            chunk_sum = arith.constant(0.0, type=T.f32)
            for e in range_constexpr(ELEMS_PER_THREAD_SM):
                col = sm_col * ELEMS_PER_THREAD_SM + e
                flat_idx = sm_row * BLOCK_N_TILE + col
                s = memref_load(scores_1d, flat_idx)
                p = ((s - new_max) * LOG2E_C).exp2(
                    fastmath=arith.FastMathFlags.fast)
                chunk_sum = chunk_sum + p
                p_half = arith.trunc_f(T_ELEM, p)
                memref_store(p_half, probs_1d, flat_idx)

            for sh in SM_SHUFFLE_BITS:
                peer = chunk_sum.shuffle_xor(arith.constant(sh, type=T.i32),
                                              arith.constant(64, type=T.i32))
                chunk_sum = chunk_sum + peer

            running_sum = running_sum + chunk_sum
            running_max = new_max

            # Rescale accum
            memref_store(rescale, scores_1d, sm_row)
            gpu.barrier()

            accum_row = (tid * ACCUM_PER_THREAD) // HEAD_SIZE
            my_rescale = memref_load(scores_1d, accum_row)
            for e in range_constexpr(ACCUM_PER_THREAD):
                flat_idx = tid * ACCUM_PER_THREAD + e
                old_val = memref_load(accum_1d, flat_idx)
                new_val = old_val * my_rescale
                memref_store(new_val, accum_1d, flat_idx)

            gpu.barrier()

            # --- PV GEMM ---
            tileP = fx.make_tile(NUM_QUERIES, BLOCK_N_TILE)
            bP = fx.slice(fx.zipped_divide(probs_lds, tileP), (None, 0))
            copy_src_P = thr_copy_PV_A.partition_S(bP)
            frag_P = thr_mma.make_fragment_A(bP)
            copy_frag_P = thr_copy_PV_A.retile(frag_P)

            for pv_iter in range_constexpr(PV_N_ITERS):
                hd_base = pv_iter * BLOCK_N_TILE

                for e in range_constexpr(V_CHUNK_ELEMS_PER_THREAD):
                    flat_idx = tid * V_CHUNK_ELEMS_PER_THREAD + e
                    hd_local = flat_idx // BLOCK_N_TILE
                    token_local = flat_idx % BLOCK_N_TILE
                    head_dim = hd_base + hd_local
                    token = token_base + token_local

                    logical_block_v = token // kv_block_size
                    token_in_block_v = token % kv_block_size
                    bt_offset_v = bt_seq_base + logical_block_v
                    v_phys_block = buffer_ops.buffer_load(
                        rsrc_bt, bt_offset_v, vec_width=1, dtype=T.i32)
                    v_page_base = v_phys_block * V_STRIDE_BLOCK + v_head_offset
                    v_addr = (v_page_base
                              + head_dim * V_STRIDE_HD
                              + token_in_block_v * V_STRIDE_TOKEN)

                    val = buffer_ops.buffer_load(rsrc_V, v_addr,
                                                 vec_width=1, dtype=T_ELEM)
                    memref_store(val, v_staging_1d, flat_idx)

                gpu.barrier()

                tileV = fx.make_tile(BLOCK_N_TILE, BLOCK_N_TILE)
                bV = fx.slice(fx.zipped_divide(v_staging_lds, tileV), (None, 0))
                copy_src_V = thr_copy_B_lds.partition_S(bV)
                frag_V = thr_mma.make_fragment_B(bV)
                copy_frag_V = thr_copy_B_lds.retile(frag_V)

                tileAcc = fx.make_tile(NUM_QUERIES, BLOCK_N_TILE)
                bAcc = fx.slice(fx.zipped_divide(accum_lds, tileAcc),
                                 (None, pv_iter))
                copy_src_acc = thr_copy_C_lds.partition_S(bAcc)
                frag_Acc = thr_mma.make_fragment_C(bAcc)
                copy_frag_Acc = thr_copy_C_lds.retile(frag_Acc)

                fx.copy(copy_atom_lds_f32, copy_src_acc, copy_frag_Acc, pred=None)
                fx.copy(copy_atom_lds_half, copy_src_P, copy_frag_P, pred=None)
                fx.copy(copy_atom_lds_half, copy_src_V, copy_frag_V, pred=None)
                fx.gemm(mma_atom, frag_Acc, frag_P, frag_V, frag_Acc)
                fx.copy(copy_atom_lds_f32, copy_frag_Acc, copy_src_acc, pred=None)

                gpu.barrier()

            results = yield [running_max, running_sum]

        running_max = results[0]
        running_sum = results[1]

        # ═══ Post-loop: sinks + output ═══
        if _one_shot:
            # ONE_SHOT: normalize and write final output directly

            # Sinks contribution (if enabled)
            if _use_sinks:
                rsrc_sinks = buffer_ops.create_buffer_resource(Sinks)
                sinks_offset = kv_head_idx * NUM_QUERIES + sm_row
                sink_val = buffer_ops.buffer_load(rsrc_sinks, sinks_offset,
                                                   vec_width=1, dtype=T.f32)
                sink_contrib = ((sink_val - running_max) * LOG2E_C).exp2(
                    fastmath=arith.FastMathFlags.fast)
                running_sum = running_sum + sink_contrib

            # Broadcast running_sum from sm_row threads to accum_row threads
            # via LDS. sm_row = tid // THREADS_PER_ROW_SM, but
            # accum_row = (tid * ACCUM_PER_THREAD) // HEAD_SIZE, so they differ.
            memref_store(running_sum, scores_1d, sm_row)
            gpu.barrier()

            accum_row_out = (tid * ACCUM_PER_THREAD) // HEAD_SIZE
            my_inv_sum = arith.constant(1.0, type=T.f32) / memref_load(
                scores_1d, accum_row_out)

            rsrc_out = buffer_ops.create_buffer_resource(Out)
            out_base = ((seq_idx * num_kv_heads + kv_head_idx)
                        * (NUM_QUERIES * HEAD_SIZE))

            for e in range_constexpr(ACCUM_PER_THREAD):
                flat_idx = tid * ACCUM_PER_THREAD + e
                val = memref_load(accum_1d, flat_idx)
                val = val * my_inv_sum
                val_half = arith.trunc_f(T_ELEM, val)
                buffer_ops.buffer_store(val_half, rsrc_out, out_base + flat_idx)
        else:
            # Split-K: store partial results
            rsrc_partial = buffer_ops.create_buffer_resource(Partial_out)
            rsrc_maxlog = buffer_ops.create_buffer_resource(Max_logits)
            rsrc_expsum = buffer_ops.create_buffer_resource(Exp_sums)

            out_base = (((seq_idx * num_kv_heads + kv_head_idx) * num_partitions
                         + partition_idx) * (NUM_QUERIES * HEAD_SIZE))
            for e in range_constexpr(ACCUM_PER_THREAD):
                flat_idx = tid * ACCUM_PER_THREAD + e
                val = memref_load(accum_1d, flat_idx)
                buffer_ops.buffer_store(val, rsrc_partial, out_base + flat_idx)

            meta_base = (((seq_idx * num_kv_heads + kv_head_idx) * num_partitions
                          + partition_idx) * NUM_QUERIES)
            buffer_ops.buffer_store(running_max, rsrc_maxlog, meta_base + sm_row)
            buffer_ops.buffer_store(running_sum, rsrc_expsum, meta_base + sm_row)

    @flyc.kernel
    def pa_reduce_kernel(
        Partial_out: fx.Tensor,
        Max_logits: fx.Tensor,
        Exp_sums: fx.Tensor,
        Out: fx.Tensor,
        Sinks: fx.Tensor,
        NUM_QUERIES: fx.Constexpr[int],
        HEAD_SIZE: fx.Constexpr[int],
        NUM_PARTITIONS: fx.Constexpr[int],
    ):
        """Combine split-K partitions, write f16/bf16 output."""
        tid = fx.thread_idx.x
        kv_head_idx = fx.block_idx.x
        seq_idx = fx.block_idx.y

        T_ELEM = T.bf16 if dtype == "bf16" else T.f16

        rsrc_partial = buffer_ops.create_buffer_resource(Partial_out)
        rsrc_maxlog = buffer_ops.create_buffer_resource(Max_logits)
        rsrc_expsum = buffer_ops.create_buffer_resource(Exp_sums)
        rsrc_out = buffer_ops.create_buffer_resource(Out)

        LOG2E_C = arith.constant(LOG2E, type=T.f32)

        out_row = (tid * ACCUM_PER_THREAD) // HEAD_SIZE

        meta_head_base = (seq_idx * num_kv_heads + kv_head_idx) * NUM_PARTITIONS

        global_max = arith.constant(-1e30, type=T.f32)
        for p in range_constexpr(num_partitions):
            meta_off = (meta_head_base + p) * NUM_QUERIES + out_row
            p_max = buffer_ops.buffer_load(rsrc_maxlog, meta_off,
                                            vec_width=1, dtype=T.f32)
            global_max = global_max.maximumf(p_max)

        total_sum = arith.constant(0.0, type=T.f32)
        out_vals = []
        for e in range_constexpr(ACCUM_PER_THREAD):
            out_vals.append(arith.constant(0.0, type=T.f32))

        partial_head_base = (seq_idx * num_kv_heads + kv_head_idx) * NUM_PARTITIONS

        for p in range_constexpr(num_partitions):
            meta_off = (meta_head_base + p) * NUM_QUERIES + out_row
            p_max = buffer_ops.buffer_load(rsrc_maxlog, meta_off,
                                            vec_width=1, dtype=T.f32)
            p_sum = buffer_ops.buffer_load(rsrc_expsum, meta_off,
                                            vec_width=1, dtype=T.f32)
            rescale = ((p_max - global_max) * LOG2E_C).exp2(
                fastmath=arith.FastMathFlags.fast)
            total_sum = total_sum + p_sum * rescale

            partial_base = (partial_head_base + p) * (NUM_QUERIES * HEAD_SIZE)
            for e in range_constexpr(ACCUM_PER_THREAD):
                flat_idx = tid * ACCUM_PER_THREAD + e
                pval = buffer_ops.buffer_load(rsrc_partial,
                                               partial_base + flat_idx,
                                               vec_width=1, dtype=T.f32)
                out_vals[e] = out_vals[e] + pval * rescale

        # Sinks contribution in reduce path
        if _use_sinks:
            rsrc_sinks = buffer_ops.create_buffer_resource(Sinks)
            sinks_offset = kv_head_idx * NUM_QUERIES + out_row
            sink_val = buffer_ops.buffer_load(rsrc_sinks, sinks_offset,
                                               vec_width=1, dtype=T.f32)
            sink_contrib = ((sink_val - global_max) * LOG2E_C).exp2(
                fastmath=arith.FastMathFlags.fast)
            total_sum = total_sum + sink_contrib

        # Normalize and store
        inv_sum = arith.constant(1.0, type=T.f32) / total_sum
        out_base = ((seq_idx * num_kv_heads + kv_head_idx)
                    * (NUM_QUERIES * HEAD_SIZE))
        for e in range_constexpr(ACCUM_PER_THREAD):
            flat_idx = tid * ACCUM_PER_THREAD + e
            final_val = out_vals[e] * inv_sum
            final_half = arith.trunc_f(T_ELEM, final_val)
            buffer_ops.buffer_store(final_half, rsrc_out, out_base + flat_idx)

    @flyc.jit
    def launch_splitk(Q, K_cache, V_cache, Block_table, Context_lens,
                       Partial_out, Max_logits, Exp_sums, Out, Sinks,
                       gx, gy, gz,
                       stream: fx.Stream = fx.Stream(None)):
        pa_splitk_kernel(
            Q, K_cache, V_cache, Block_table, Context_lens,
            Partial_out, Max_logits, Exp_sums, Out, Sinks,
            NUM_QUERIES, HEAD_SIZE, BLOCK_N_TILE,
            PV_N_ITERS, partition_size,
        ).launch(
            grid=(gx, gy, gz),
            block=(NUM_THREADS, 1, 1),
            smem=TOTAL_LDS,
            stream=stream,
        )

    @flyc.jit
    def launch_reduce(Partial_out, Max_logits, Exp_sums, Out, Sinks,
                       gx, gy,
                       stream: fx.Stream = fx.Stream(None)):
        pa_reduce_kernel(
            Partial_out, Max_logits, Exp_sums, Out, Sinks,
            NUM_QUERIES, HEAD_SIZE, num_partitions,
        ).launch(
            grid=(gx, gy, 1),
            block=(NUM_THREADS, 1, 1),
            smem=0,
            stream=stream,
        )

    return {
        'launch_splitk': launch_splitk,
        'launch_reduce': launch_reduce,
    }


def pa_decode_tile_sw_launch(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float,
    *,
    max_context_partition_num: int = 0,
    kv_block_size: int = 0,
    context_partition_size: int = 256,
    sliding_window: int = 0,
    sinks: torch.Tensor = None,
    stream=None,
):
    """
    Launch PA decode tile kernel with sliding window support (f16/bf16).

    Args:
        output:           [batch_size, num_query_heads, head_size] f16/bf16
        query:            [batch_size, num_query_heads, head_size] f16/bf16
        key_cache:        [N, nkv, hd//x, bs, x] f16/bf16 (x=8)
        value_cache:      [N, nkv, hd, bs] f16/bf16
        context_lengths:  [batch_size] i32
        block_tables:     [batch_size, max_blocks_per_seq] i32
        softmax_scale:    float
        sliding_window:   int, 0 = disabled
        sinks:            [num_query_heads] f32, or None
    """
    batch_size = query.shape[0]
    num_query_heads = query.shape[1]
    head_size = query.shape[2]
    num_kv_heads = key_cache.shape[1]
    num_qh_per_kvh = num_query_heads // num_kv_heads

    if kv_block_size == 0:
        kv_block_size = key_cache.shape[3]

    max_blocks_per_seq = block_tables.shape[1]

    # Determine dtype from input tensor
    if query.dtype == torch.float16:
        dtype = "f16"
    elif query.dtype == torch.bfloat16:
        dtype = "bf16"
    else:
        raise ValueError(f"Unsupported query dtype: {query.dtype}")

    max_context = int(context_lengths.max().item())

    # Compute effective context for partition count
    if sliding_window > 0:
        effective_context = min(max_context, sliding_window)
    else:
        effective_context = max_context

    if max_context_partition_num > 0:
        num_partitions = max_context_partition_num
    else:
        num_partitions = (effective_context + context_partition_size - 1) // context_partition_size

    # One-shot detection (matches Gluon logic)
    one_shot = (num_partitions <= 1) and not (
        sliding_window > 0 and kv_block_size == 1024)

    use_sinks = sinks is not None

    dev = query.device
    s = stream or torch.cuda.current_stream()

    # Flatten Q to 2D: [batch * nqh, hd]
    Q_flat = query.reshape(-1, head_size).contiguous()
    Out_flat = output.reshape(-1).contiguous()

    # Sinks tensor (or dummy)
    if sinks is not None:
        sinks_flat = sinks.contiguous().view(-1)
    else:
        sinks_flat = torch.empty(1, dtype=torch.float32, device=dev)

    grid_x = 1 if one_shot else num_partitions
    grid_y = num_kv_heads
    grid_z = batch_size

    # Intermediate buffers (not used in one_shot mode, but need valid tensors)
    if one_shot:
        partial_out = torch.empty(1, dtype=torch.float32, device=dev)
        max_logits = torch.empty(1, dtype=torch.float32, device=dev)
        exp_sums = torch.empty(1, dtype=torch.float32, device=dev)
    else:
        total_kv_slots = batch_size * num_kv_heads
        partial_size = total_kv_slots * num_partitions * NUM_QUERIES * head_size
        meta_size = total_kv_slots * num_partitions * NUM_QUERIES
        partial_out = torch.zeros(partial_size, dtype=torch.float32, device=dev)
        max_logits = torch.full((meta_size,), -1e30, dtype=torch.float32, device=dev)
        exp_sums = torch.zeros(meta_size, dtype=torch.float32, device=dev)

    compiled = compile_pa_decode_tile_sw(
        num_kv_heads=num_kv_heads,
        num_qh_per_kvh=num_qh_per_kvh,
        kv_block_size=kv_block_size,
        partition_size=context_partition_size,
        max_blocks_per_seq=max_blocks_per_seq,
        num_partitions=num_partitions,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        one_shot=one_shot,
        use_sinks=use_sinks,
        dtype=dtype,
    )

    K_flat = key_cache.contiguous().view(-1)
    V_flat = value_cache.contiguous().view(-1)
    BT_flat = block_tables.contiguous().view(-1)
    CL_flat = context_lengths.contiguous().view(-1)

    compiled['launch_splitk'](
        Q_flat, K_flat, V_flat, BT_flat, CL_flat,
        partial_out, max_logits, exp_sums, Out_flat, sinks_flat,
        grid_x, grid_y, grid_z, s)

    if not one_shot:
        compiled['launch_reduce'](
            partial_out, max_logits, exp_sums, Out_flat, sinks_flat,
            grid_y, grid_z, s)


def test():
    """Self-test with bf16 and f16, with and without sliding window."""
    import sys

    KV_BLOCK_SIZE = 1024
    CONTEXT_LEN = 8192
    NUM_KV_HEADS = 1
    NUM_QH_PER_KVH_T = 16
    NUM_QUERY_HEADS_T = NUM_KV_HEADS * NUM_QH_PER_KVH_T
    PARTITION_SIZE = 256
    SCALE = 1.0 / math.sqrt(HEAD_SIZE)

    batch_size = 2

    for test_dtype in [torch.bfloat16, torch.float16]:
        for sw in [0, 2048]:
            dtype_name = "bf16" if test_dtype == torch.bfloat16 else "f16"
            sw_name = f"sw={sw}" if sw > 0 else "no_sw"
            print(f"\n=== Test: {dtype_name}, {sw_name}, ctx={CONTEXT_LEN} ===")

            num_logical_blocks = CONTEXT_LEN // KV_BLOCK_SIZE
            num_blocks = num_logical_blocks + 2
            phys_blocks = list(range(num_logical_blocks - 1, -1, -1))

            Q = torch.randn(batch_size, NUM_QUERY_HEADS_T, HEAD_SIZE,
                             dtype=test_dtype).cuda()
            K_flat = torch.randn(CONTEXT_LEN, HEAD_SIZE, dtype=test_dtype).cuda()
            V_flat = torch.randn(CONTEXT_LEN, HEAD_SIZE, dtype=test_dtype).cuda()

            key_cache = torch.zeros(num_blocks, NUM_KV_HEADS, HEAD_SIZE // X_PACK,
                                    KV_BLOCK_SIZE, X_PACK, dtype=test_dtype).cuda()
            val_cache = torch.zeros(num_blocks, NUM_KV_HEADS, HEAD_SIZE,
                                    KV_BLOCK_SIZE, dtype=test_dtype).cuda()

            for lb in range(num_logical_blocks):
                pb = phys_blocks[lb]
                token_start = lb * KV_BLOCK_SIZE
                token_end = token_start + KV_BLOCK_SIZE
                K_block = K_flat[token_start:token_end]
                V_block = V_flat[token_start:token_end]

                K_reshaped = K_block.view(KV_BLOCK_SIZE, HEAD_SIZE // X_PACK, X_PACK)
                K_packed = K_reshaped.permute(1, 0, 2)
                key_cache[pb, 0] = K_packed
                val_cache[pb, 0] = V_block.T

            block_table = torch.zeros(batch_size, num_logical_blocks,
                                       dtype=torch.int32).cuda()
            for lb in range(num_logical_blocks):
                block_table[0, lb] = phys_blocks[lb]
                block_table[1, lb] = phys_blocks[lb]

            context_lengths = torch.tensor([CONTEXT_LEN] * batch_size,
                                            dtype=torch.int32).cuda()
            output = torch.zeros(batch_size, NUM_QUERY_HEADS_T, HEAD_SIZE,
                                  dtype=test_dtype).cuda()

            pa_decode_tile_sw_launch(
                output, Q, key_cache, val_cache,
                context_lengths, block_table,
                softmax_scale=SCALE,
                context_partition_size=PARTITION_SIZE,
                sliding_window=sw,
                stream=torch.cuda.Stream(),
            )
            torch.cuda.synchronize()

            # Reference computation
            if sw > 0:
                start_idx = max(0, CONTEXT_LEN - sw)
                K_ref = K_flat[start_idx:].float()
                V_ref = V_flat[start_idx:].float()
            else:
                K_ref = K_flat.float()
                V_ref = V_flat.float()

            max_diff_all = 0.0
            for b in range(batch_size):
                for qh in range(NUM_QUERY_HEADS_T):
                    q_h = Q[b, qh:qh+1].float()
                    scores_h = q_h @ K_ref.T
                    probs_h = torch.softmax(scores_h * SCALE, dim=-1).to(test_dtype).float()
                    exp_h = probs_h @ V_ref
                    d = (output[b, qh].float() - exp_h.squeeze()).abs().max().item()
                    max_diff_all = max(max_diff_all, d)
                    if qh < 2 or d > 0.1:
                        print(f"  Batch {b}, Head {qh}: max_diff={d:.6f}")

            print(f"  Overall max diff: {max_diff_all:.6f}")
            passed = max_diff_all < 0.15
            print(f"  PASSED: {passed}")
            if not passed:
                print(f"  FAILED! max_diff={max_diff_all}")
                sys.exit(1)

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    test()
