"""
PA Decode Tile (BF16): FlyDSL paged attention decode using tile programming.

CuTe-style layout algebra with TiledMma + TiledCopy + fx.gemm().
Online softmax, split-K partitioning, GQA, paged KV cache.

Supports:
  - K cache: [num_blocks, num_kv_heads, head_size//x, kv_block_size, x] (x=8 for bf16)
  - V cache: [num_blocks, num_kv_heads, head_size, kv_block_size]
  - Query:   [batch_size, num_query_heads, head_size]
  - Output:  [batch_size, num_query_heads, head_size]

Grid mapping:
  - grid.x = num_partitions (split-K)
  - grid.y = num_kv_heads
  - grid.z = batch_size
  Each CTA handles one KV head, processing NUM_QUERIES=16 query heads that share it.
"""
import functools
import math
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, gpu
from flydsl._mlir.dialects.fly import PointerType, AddressSpace
from flydsl.expr.primitive import (get_dyn_shared, recast_iter, make_view,
                                    add_offset, memref_load, memref_store)
from flydsl.expr.typing import T
from flydsl.expr import buffer_ops


# Fixed architecture constants
NUM_QUERIES = 16  # = query_group_size, fixed
HEAD_SIZE = 128
X_PACK = 8        # bf16 packing factor (same as f16: 2 bytes per elem)
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
PROBS_TILE_BYTES = PROBS_TILE_ELEMS * 2  # bf16 = 2 bytes

K_TILE_ELEMS = BLOCK_N_TILE * HEAD_SIZE
V_CHUNK_ELEMS = BLOCK_N_TILE * BLOCK_N_TILE
STAGING_ELEMS = max(K_TILE_ELEMS, V_CHUNK_ELEMS)
STAGING_BYTES = STAGING_ELEMS * 2  # bf16 = 2 bytes

ACCUM_ELEMS = NUM_QUERIES * HEAD_SIZE
ACCUM_BYTES = ACCUM_ELEMS * 4

STAGING_OFFSET_BYTES = QK_SCORES_BYTES + PROBS_TILE_BYTES
STAGING_OFFSET_BF16 = STAGING_OFFSET_BYTES // 2

ACCUM_OFFSET_BYTES = STAGING_OFFSET_BYTES + STAGING_BYTES
ACCUM_OFFSET_F32 = ACCUM_OFFSET_BYTES // 4

K_ELEMS_PER_THREAD = K_TILE_ELEMS // NUM_THREADS
V_CHUNK_ELEMS_PER_THREAD = V_CHUNK_ELEMS // NUM_THREADS
ACCUM_PER_THREAD = ACCUM_ELEMS // NUM_THREADS

TOTAL_LDS = QK_SCORES_BYTES + PROBS_TILE_BYTES + STAGING_BYTES + ACCUM_BYTES


@functools.lru_cache(maxsize=64)
def compile_pa_decode_tile(
    num_kv_heads,
    num_qh_per_kvh,
    kv_block_size,
    partition_size,
    max_blocks_per_seq,
    num_partitions,
    softmax_scale,
):
    """Compile split-K and reduce kernels for PA decode tile (bf16)."""
    assert num_qh_per_kvh == NUM_QUERIES, \
        f"query_group_size ({num_qh_per_kvh}) must equal NUM_QUERIES ({NUM_QUERIES})"

    ITERS_PER_PARTITION = partition_size // BLOCK_N_TILE

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
        NUM_QUERIES: fx.Constexpr[int],
        HEAD_SIZE: fx.Constexpr[int],
        BLOCK_N_TILE: fx.Constexpr[int],
        ITERS_PER_PARTITION: fx.Constexpr[int],
        PV_N_ITERS: fx.Constexpr[int],
        PARTITION_SIZE: fx.Constexpr[int],
    ):
        tid = fx.thread_idx.x
        partition_idx = fx.block_idx.x
        kv_head_idx = fx.block_idx.y
        seq_idx = fx.block_idx.z

        # KV cache head offsets
        k_head_offset = kv_head_idx * K_STRIDE_KVHEAD
        v_head_offset = kv_head_idx * V_STRIDE_KVHEAD

        # ═══ MMA and Copy Setup (BF16) ═══
        # BF16 MFMA: 16x16x16 (CDNA3 _1k variant, same K as f16)
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((1, 4, 1), (1, 1, 0))
        )
        thr_mma = tiled_mma.thr_slice(tid)

        copy_atom_in = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.BFloat16)
        tiled_copy_A = fx.make_tiled_copy_A(copy_atom_in, tiled_mma)
        thr_copy_A = tiled_copy_A.get_slice(tid)

        copy_atom_lds_bf16 = fx.make_copy_atom(fx.UniversalCopy(32), fx.BFloat16)
        tiled_copy_B_lds = fx.make_tiled_copy_B(copy_atom_lds_bf16, tiled_mma)
        thr_copy_B_lds = tiled_copy_B_lds.get_slice(tid)
        tiled_copy_PV_A = fx.make_tiled_copy_A(copy_atom_lds_bf16, tiled_mma)
        thr_copy_PV_A = tiled_copy_PV_A.get_slice(tid)

        copy_atom_lds_f32 = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
        tiled_copy_C_lds = fx.make_tiled_copy_C(copy_atom_lds_f32, tiled_mma)
        thr_copy_C_lds = tiled_copy_C_lds.get_slice(tid)

        copy_atom_out = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        tiled_copy_C_out = fx.make_tiled_copy_C(copy_atom_out, tiled_mma)
        thr_copy_C_out = tiled_copy_C_out.get_slice(tid)

        # ═══ LDS Views ═══
        lds_raw = get_dyn_shared()

        scores_ptr_ty = PointerType.get(elem_ty=T.f32,
                                         address_space=int(AddressSpace.Shared))
        scores_ptr = recast_iter(scores_ptr_ty, lds_raw)
        scores_lds = make_view(scores_ptr,
                                fx.make_layout((NUM_QUERIES, BLOCK_N_TILE),
                                                (BLOCK_N_TILE, 1)))
        scores_1d = make_view(scores_ptr, fx.make_layout(QK_SCORES_ELEMS, 1))

        probs_ptr_ty = PointerType.get(elem_ty=T.bf16,
                                        address_space=int(AddressSpace.Shared))
        probs_base_ptr = recast_iter(probs_ptr_ty, lds_raw)
        probs_ptr = add_offset(probs_base_ptr, QK_SCORES_BYTES // 2)
        probs_lds = make_view(probs_ptr,
                               fx.make_layout((NUM_QUERIES, BLOCK_N_TILE),
                                                (BLOCK_N_TILE, 1)))
        probs_1d = make_view(probs_ptr, fx.make_layout(PROBS_TILE_ELEMS, 1))

        staging_base_ptr = recast_iter(probs_ptr_ty, lds_raw)
        staging_ptr = add_offset(staging_base_ptr, STAGING_OFFSET_BF16)
        k_staging_lds = make_view(staging_ptr,
                                   fx.make_layout((BLOCK_N_TILE, HEAD_SIZE),
                                                    (HEAD_SIZE, 1)))
        k_staging_1d = make_view(staging_ptr, fx.make_layout(K_TILE_ELEMS, 1))
        v_staging_lds = make_view(staging_ptr,
                                   fx.make_layout((BLOCK_N_TILE, BLOCK_N_TILE),
                                                    (BLOCK_N_TILE, 1)))
        v_staging_1d = make_view(staging_ptr, fx.make_layout(V_CHUNK_ELEMS, 1))

        accum_ptr_ty = PointerType.get(elem_ty=T.f32,
                                        address_space=int(AddressSpace.Shared))
        accum_base_ptr = recast_iter(accum_ptr_ty, lds_raw)
        accum_ptr_v = add_offset(accum_base_ptr, ACCUM_OFFSET_F32)
        accum_lds = make_view(accum_ptr_v,
                               fx.make_layout((NUM_QUERIES, HEAD_SIZE),
                                                (HEAD_SIZE, 1)))
        accum_1d = make_view(accum_ptr_v, fx.make_layout(ACCUM_ELEMS, 1))

        rsrc_bt = buffer_ops.create_buffer_resource(Block_table)

        # Initialize accum to zero
        for e in range_constexpr(ACCUM_PER_THREAD):
            flat_idx = tid * ACCUM_PER_THREAD + e
            memref_store(arith.constant(0.0, type=T.f32), accum_1d, flat_idx)
        gpu.barrier()

        # Q: flat [batch * num_kv_heads * NUM_QUERIES, HEAD_SIZE]
        Q_buf = fx.rocdl.make_buffer_tensor(Q)
        rsrc_K = buffer_ops.create_buffer_resource(K_cache)
        rsrc_V = buffer_ops.create_buffer_resource(V_cache)

        q_tile_idx = seq_idx * num_kv_heads + kv_head_idx

        tileQ = fx.make_tile(NUM_QUERIES, HEAD_SIZE)
        bQ = fx.slice(fx.zipped_divide(Q_buf, tileQ), (None, q_tile_idx))
        copy_src_Q = thr_copy_A.partition_S(bQ)
        partition_Q = thr_mma.partition_A(bQ)
        frag_Q = thr_mma.make_fragment_A(partition_Q)
        copy_frag_Q = thr_copy_A.retile(frag_Q)

        sm_row = tid // THREADS_PER_ROW_SM
        sm_col = tid % THREADS_PER_ROW_SM

        running_max = arith.constant(-1e30, type=T.f32)
        running_sum = arith.constant(0.0, type=T.f32)

        SCALE_C = arith.constant(SCALE, type=T.f32)
        LOG2E_C = arith.constant(LOG2E, type=T.f32)

        partition_token_start = partition_idx * PARTITION_SIZE
        bt_seq_base = seq_idx * max_blocks_per_seq

        for ctx_iter in range_constexpr(ITERS_PER_PARTITION):
            token_base = partition_token_start + ctx_iter * BLOCK_N_TILE

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
                                             vec_width=1, dtype=T.bf16)
                memref_store(val, k_staging_1d, flat_idx)

            gpu.barrier()

            # --- QK GEMM ---
            tileK = fx.make_tile(BLOCK_N_TILE, HEAD_SIZE)
            bK = fx.slice(fx.zipped_divide(k_staging_lds, tileK), (None, 0))
            copy_src_K = thr_copy_B_lds.partition_S(bK)
            partition_K = thr_mma.partition_B(bK)
            frag_K = thr_mma.make_fragment_B(partition_K)
            copy_frag_K = thr_copy_B_lds.retile(frag_K)

            tileC = fx.make_tile(NUM_QUERIES, BLOCK_N_TILE)
            bC_lds = fx.slice(fx.zipped_divide(scores_lds, tileC), (None, 0))
            copy_dst_C_lds = thr_copy_C_lds.partition_S(bC_lds)
            partition_C = thr_mma.partition_C(bC_lds)
            frag_C = thr_mma.make_fragment_C(partition_C)
            copy_frag_C_lds = thr_copy_C_lds.retile(frag_C)

            fx.copy(copy_atom_in, copy_src_Q, copy_frag_Q, pred=None)
            fx.copy(copy_atom_lds_bf16, copy_src_K, copy_frag_K, pred=None)
            fx.gemm(mma_atom, frag_C, frag_Q, frag_K, frag_C)
            fx.copy(copy_atom_lds_f32, copy_frag_C_lds, copy_dst_C_lds, pred=None)

            gpu.barrier()

            # --- Online softmax ---
            chunk_max = arith.constant(-1e30, type=T.f32)
            for e in range_constexpr(ELEMS_PER_THREAD_SM):
                col = sm_col * ELEMS_PER_THREAD_SM + e
                flat_idx = sm_row * BLOCK_N_TILE + col
                s = memref_load(scores_1d, flat_idx)
                s = s * SCALE_C
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
                p_bf16 = arith.trunc_f(T.bf16, p)
                memref_store(p_bf16, probs_1d, flat_idx)

            for sh in SM_SHUFFLE_BITS:
                peer = chunk_sum.shuffle_xor(arith.constant(sh, type=T.i32),
                                              arith.constant(64, type=T.i32))
                chunk_sum = chunk_sum + peer

            running_sum = running_sum + chunk_sum
            running_max = new_max

            # Rescale accum (broadcast rescale per row via LDS)
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
            partition_P = thr_mma.partition_A(bP)
            frag_P = thr_mma.make_fragment_A(partition_P)
            copy_frag_P = thr_copy_PV_A.retile(frag_P)

            for pv_iter in range_constexpr(PV_N_ITERS):
                hd_base = pv_iter * BLOCK_N_TILE

                # V: [N, nkv, hd, bs] → addr = block*BLOCK + head*KVHEAD + hd*HD + tok
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
                                                 vec_width=1, dtype=T.bf16)
                    memref_store(val, v_staging_1d, flat_idx)

                gpu.barrier()

                tileV = fx.make_tile(BLOCK_N_TILE, BLOCK_N_TILE)
                bV = fx.slice(fx.zipped_divide(v_staging_lds, tileV), (None, 0))
                copy_src_V = thr_copy_B_lds.partition_S(bV)
                partition_V = thr_mma.partition_B(bV)
                frag_V = thr_mma.make_fragment_B(partition_V)
                copy_frag_V = thr_copy_B_lds.retile(frag_V)

                tileAcc = fx.make_tile(NUM_QUERIES, BLOCK_N_TILE)
                bAcc = fx.slice(fx.zipped_divide(accum_lds, tileAcc),
                                 (None, pv_iter))
                copy_src_acc = thr_copy_C_lds.partition_S(bAcc)
                partition_Acc = thr_mma.partition_C(bAcc)
                frag_Acc = thr_mma.make_fragment_C(partition_Acc)
                copy_frag_Acc = thr_copy_C_lds.retile(frag_Acc)

                fx.copy(copy_atom_lds_f32, copy_src_acc, copy_frag_Acc, pred=None)
                fx.copy(copy_atom_lds_bf16, copy_src_P, copy_frag_P, pred=None)
                fx.copy(copy_atom_lds_bf16, copy_src_V, copy_frag_V, pred=None)
                fx.gemm(mma_atom, frag_Acc, frag_P, frag_V, frag_Acc)
                fx.copy(copy_atom_lds_f32, copy_frag_Acc, copy_src_acc, pred=None)

                gpu.barrier()

        # ═══ Store partial results ═══
        rsrc_partial = buffer_ops.create_buffer_resource(Partial_out)
        rsrc_maxlog = buffer_ops.create_buffer_resource(Max_logits)
        rsrc_expsum = buffer_ops.create_buffer_resource(Exp_sums)

        # Partials: [batch, nkv, num_parts, NUM_Q * HEAD_SIZE]
        out_base = (((seq_idx * num_kv_heads + kv_head_idx) * num_partitions
                     + partition_idx) * (NUM_QUERIES * HEAD_SIZE))
        for e in range_constexpr(ACCUM_PER_THREAD):
            flat_idx = tid * ACCUM_PER_THREAD + e
            val = memref_load(accum_1d, flat_idx)
            buffer_ops.buffer_store(val, rsrc_partial, out_base + flat_idx)

        # Meta: [batch, nkv, num_parts, NUM_Q]
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
        NUM_QUERIES: fx.Constexpr[int],
        HEAD_SIZE: fx.Constexpr[int],
        NUM_PARTITIONS: fx.Constexpr[int],
    ):
        """Combine split-K partitions, write bf16 output."""
        tid = fx.thread_idx.x
        kv_head_idx = fx.block_idx.x
        seq_idx = fx.block_idx.y

        rsrc_partial = buffer_ops.create_buffer_resource(Partial_out)
        rsrc_maxlog = buffer_ops.create_buffer_resource(Max_logits)
        rsrc_expsum = buffer_ops.create_buffer_resource(Exp_sums)
        rsrc_out = buffer_ops.create_buffer_resource(Out)

        LOG2E_C = arith.constant(LOG2E, type=T.f32)

        out_row = (tid * ACCUM_PER_THREAD) // HEAD_SIZE

        # Meta: [batch, nkv, num_parts, NUM_Q]
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

        # Normalize and store as bf16
        inv_sum = arith.constant(1.0, type=T.f32) / total_sum
        out_base = ((seq_idx * num_kv_heads + kv_head_idx)
                    * (NUM_QUERIES * HEAD_SIZE))
        for e in range_constexpr(ACCUM_PER_THREAD):
            flat_idx = tid * ACCUM_PER_THREAD + e
            final_val = out_vals[e] * inv_sum
            final_bf16 = arith.trunc_f(T.bf16, final_val)
            buffer_ops.buffer_store(final_bf16, rsrc_out, out_base + flat_idx)

    @flyc.jit
    def launch_splitk(Q, K_cache, V_cache, Block_table, Context_lens,
                       Partial_out, Max_logits, Exp_sums,
                       gx, gy, gz,
                       stream: fx.Stream = fx.Stream(None)):
        pa_splitk_kernel(
            Q, K_cache, V_cache, Block_table, Context_lens,
            Partial_out, Max_logits, Exp_sums,
            NUM_QUERIES, HEAD_SIZE, BLOCK_N_TILE,
            ITERS_PER_PARTITION, PV_N_ITERS, partition_size,
        ).launch(
            grid=(gx, gy, gz),
            block=(NUM_THREADS, 1, 1),
            smem=TOTAL_LDS,
            stream=stream,
        )

    @flyc.jit
    def launch_reduce(Partial_out, Max_logits, Exp_sums, Out,
                       gx, gy,
                       stream: fx.Stream = fx.Stream(None)):
        pa_reduce_kernel(
            Partial_out, Max_logits, Exp_sums, Out,
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


def pa_decode_tile_launch(
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
    stream=None,
):
    """
    Launch PA decode tile kernel (bf16).

    Args:
        output:           [batch_size, num_query_heads, head_size] bf16
        query:            [batch_size, num_query_heads, head_size] bf16
        key_cache:        [N, nkv, hd//x, bs, x] bf16 (x=8)
        value_cache:      [N, nkv, hd, bs] bf16
        context_lengths:  [batch_size] i32
        block_tables:     [batch_size, max_blocks_per_seq] i32
        softmax_scale:    float
    """
    batch_size = query.shape[0]
    num_query_heads = query.shape[1]
    head_size = query.shape[2]
    num_kv_heads = key_cache.shape[1]
    num_qh_per_kvh = num_query_heads // num_kv_heads

    if kv_block_size == 0:
        kv_block_size = key_cache.shape[3]

    max_blocks_per_seq = block_tables.shape[1]

    max_context = int(context_lengths.max().item())
    if max_context_partition_num > 0:
        num_partitions = max_context_partition_num
    else:
        num_partitions = (max_context + context_partition_size - 1) // context_partition_size

    dev = query.device
    s = stream or torch.cuda.current_stream()

    # Flatten Q to 2D: [batch * nqh, hd]
    Q_flat = query.reshape(-1, head_size).contiguous()
    Out_flat = output.reshape(-1).contiguous()

    # Grid: (num_partitions, num_kv_heads, batch_size)
    grid_x = num_partitions
    grid_y = num_kv_heads
    grid_z = batch_size

    # Intermediate buffers
    total_kv_slots = batch_size * num_kv_heads
    partial_size = total_kv_slots * num_partitions * NUM_QUERIES * head_size
    meta_size = total_kv_slots * num_partitions * NUM_QUERIES

    partial_out = torch.zeros(partial_size, dtype=torch.float32, device=dev)
    max_logits = torch.full((meta_size,), -1e30, dtype=torch.float32, device=dev)
    exp_sums = torch.zeros(meta_size, dtype=torch.float32, device=dev)

    compiled = compile_pa_decode_tile(
        num_kv_heads=num_kv_heads,
        num_qh_per_kvh=num_qh_per_kvh,
        kv_block_size=kv_block_size,
        partition_size=context_partition_size,
        max_blocks_per_seq=max_blocks_per_seq,
        num_partitions=num_partitions,
        softmax_scale=softmax_scale,
    )

    K_flat = key_cache.contiguous().view(-1)
    V_flat = value_cache.contiguous().view(-1)
    BT_flat = block_tables.contiguous().view(-1)
    CL_flat = context_lengths.contiguous().view(-1)

    compiled['launch_splitk'](
        Q_flat, K_flat, V_flat, BT_flat, CL_flat,
        partial_out, max_logits, exp_sums,
        grid_x, grid_y, grid_z, s)

    compiled['launch_reduce'](
        partial_out, max_logits, exp_sums, Out_flat,
        grid_y, grid_z, s)


def test():
    """Self-test with bf16, kv_block_size=1024."""
    KV_BLOCK_SIZE = 1024
    CONTEXT_LEN = 8192
    NUM_KV_HEADS = 1
    NUM_QH_PER_KVH_T = 16
    NUM_QUERY_HEADS_T = NUM_KV_HEADS * NUM_QH_PER_KVH_T
    PARTITION_SIZE = 256
    SCALE = 1.0 / math.sqrt(HEAD_SIZE)

    batch_size = 2
    num_logical_blocks = CONTEXT_LEN // KV_BLOCK_SIZE
    # Allocate extra physical blocks so logical-to-physical mapping is non-trivial
    num_blocks = num_logical_blocks + 2
    MAX_NUM_BLOCKS_PER_SEQ = num_logical_blocks
    # Shuffle physical block indices to stress-test block table indirection
    phys_blocks = list(range(num_logical_blocks - 1, -1, -1))  # reversed

    Q = torch.randn(batch_size, NUM_QUERY_HEADS_T, HEAD_SIZE,
                     dtype=torch.bfloat16).cuda()
    K_flat = torch.randn(CONTEXT_LEN, HEAD_SIZE, dtype=torch.bfloat16).cuda()
    V_flat = torch.randn(CONTEXT_LEN, HEAD_SIZE, dtype=torch.bfloat16).cuda()

    key_cache = torch.zeros(num_blocks, NUM_KV_HEADS, HEAD_SIZE // X_PACK,
                            KV_BLOCK_SIZE, X_PACK, dtype=torch.bfloat16).cuda()
    val_cache = torch.zeros(num_blocks, NUM_KV_HEADS, HEAD_SIZE,
                            KV_BLOCK_SIZE, dtype=torch.bfloat16).cuda()

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

    block_table = torch.zeros(batch_size, MAX_NUM_BLOCKS_PER_SEQ,
                               dtype=torch.int32).cuda()
    for lb in range(num_logical_blocks):
        block_table[0, lb] = phys_blocks[lb]
        block_table[1, lb] = phys_blocks[lb]

    context_lengths = torch.tensor([CONTEXT_LEN] * batch_size,
                                    dtype=torch.int32).cuda()
    output = torch.zeros(batch_size, NUM_QUERY_HEADS_T, HEAD_SIZE,
                          dtype=torch.bfloat16).cuda()

    print(f"=== BF16 Tile Kernel Test (kv_block_size={KV_BLOCK_SIZE}) ===")
    print(f"Q: {Q.shape} {Q.dtype}")
    print(f"K cache: {key_cache.shape} {key_cache.dtype}")
    print(f"V cache: {val_cache.shape} {val_cache.dtype}")
    print(f"Block table: {block_table}")
    print(f"Context: {CONTEXT_LEN}, Block size: {KV_BLOCK_SIZE}")
    print(f"Query heads: {NUM_QUERY_HEADS_T}, KV heads: {NUM_KV_HEADS}")
    print(f"Batch size: {batch_size}")
    print(f"Partition size: {PARTITION_SIZE}")
    print(f"Num partitions: {(CONTEXT_LEN + PARTITION_SIZE - 1) // PARTITION_SIZE}")
    print(f"LDS: {TOTAL_LDS} bytes")

    pa_decode_tile_launch(
        output, Q, key_cache, val_cache,
        context_lengths, block_table,
        softmax_scale=SCALE,
        context_partition_size=PARTITION_SIZE,
        stream=torch.cuda.Stream(),
    )
    torch.cuda.synchronize()

    # Reference
    max_diff_all = 0.0
    for b in range(batch_size):
        for qh in range(NUM_QUERY_HEADS_T):
            q_h = Q[b, qh:qh+1].float()
            scores_h = q_h @ K_flat.float().T
            probs_h = torch.softmax(scores_h * SCALE, dim=-1).to(torch.bfloat16).float()
            exp_h = probs_h @ V_flat.float()
            d = (output[b, qh].float() - exp_h.squeeze()).abs().max().item()
            max_diff_all = max(max_diff_all, d)
            if qh < 4 or d > 0.1:
                print(f"  Batch {b}, Head {qh}: max_diff={d:.6f}")

    print(f"Overall max diff: {max_diff_all:.6f}")
    print(f"All correct (atol=0.1): {max_diff_all < 0.1}")
    print(f"All correct (atol=0.5): {max_diff_all < 0.5}")


if __name__ == "__main__":
    test()
