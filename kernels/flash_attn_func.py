"""flash_attn_func kernel builder for FlyDSL.

Aggressive flash_attn_func path:
- True MFMA32 remap: `mfma_f32_32x32x8f16` for both GEMM stages.
- Tile shape: BLOCK_M=128, BLOCK_N=32, 4 waves (256 threads).
- Per-wave Q rows: 32.
- GEMM1 uses `K @ Q^T` so S/P live in MFMA32 register layout.
- Online softmax over KV dimension is done in registers.
- P is kept in registers and fed directly to GEMM2 (`V^T @ P`) without LDS roundtrip.
- K and V^T use separate LDS regions (single-buffered per iteration).

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,) where num_q_tiles = seq_len / BLOCK_M.
Block:  (256,) -- 4 waves of 64 on AMD (wave64).

Requires: head_dim % 32 == 0, head_dim >= 64, seq_len % 128 == 0.
"""

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.utils import env
from flydsl._mlir import ir
from flydsl._mlir.dialects import memref as _memref, scf, fly as _fly, llvm as _llvm


KERNEL_NAME = "flash_attn_func_kernel"


def select_flash_attn_func_path(num_heads, head_dim, causal=True, dtype_str="f16"):
    """Select active flash_attn_func path tag for build-time specialization."""
    override = os.getenv("FLYDSL_FLASH_ATTN_FUNC_PATH", "auto").strip().lower()
    if override in ("fallback", "fallback_n32", "n32"):
        return "fallback_n32"
    if override in ("fastpath", "ck_n128_fastpath", "n128"):
        return "ck_n128_fastpath"
    # Keep N128 path feature-gated by default due current occupancy/perf risk.
    enable_n128 = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_N128", "0") == "1"
    if (
        enable_n128
        and dtype_str == "f16"
        and causal
        and num_heads == 64
        and head_dim == 128
    ):
        return "ck_n128_fastpath"
    return "fallback_n32"


def build_flash_attn_func_module_primary(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="f16",
    sm_scale=None,
    waves_per_eu=3,
    flat_work_group_size=256,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
):
    """Build the flash_attn_func launcher using the post-refactor FlyDSL API."""
    env.compile.unsafe_fp_math = unsafe_fp_math
    env.compile.fast_fp_math = fast_fp_math
    gpu_arch = get_hip_arch()

    # Aggressive MFMA32 configuration for target B=1, H=64, S=8192, D=128.
    BLOCK_M = 128
    BLOCK_N = 32
    NUM_WAVES = 4
    WARP_SIZE = 64
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE  # 256
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES  # 32
    PATH_TAG = select_flash_attn_func_path(
        num_heads, head_dim, causal=causal, dtype_str=dtype_str
    )
    BLOCK_N_OUT = 128 if PATH_TAG == "ck_n128_fastpath" else BLOCK_N
    N_SUBTILES = BLOCK_N_OUT // BLOCK_N
    ENABLE_PREFETCH_3BUF = (
        os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_PREFETCH3", "0") == "1"
    )
    ENABLE_LDS_VEC16 = (
        os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    )
    REDUCE_MODE = os.getenv("FLYDSL_FLASH_ATTN_FUNC_REDUCE_MODE", "xor").strip().lower()
    if REDUCE_MODE not in ("xor", "ds_bpermute"):
        REDUCE_MODE = "xor"
    NUM_PREFETCH_K = 3 if ENABLE_PREFETCH_3BUF else 1
    NUM_PREFETCH_V = 3 if ENABLE_PREFETCH_3BUF else 1
    CK_LDS_SEQ = (1, 2, 0, 1, 0, 1, 2, 0) if ENABLE_PREFETCH_3BUF else (0,)

    # MFMA32 K-dimension is 8.
    K_STEP_QK = 8
    K_STEPS_QK = head_dim // K_STEP_QK
    # PV stage computes 32 output columns per accumulator chunk.
    D_CHUNK = 32
    D_CHUNKS = head_dim // D_CHUNK
    PV_K_STEP = 8
    PV_K_STEPS = BLOCK_N // PV_K_STEP  # 4 for BN=32

    assert BLOCK_M % NUM_WAVES == 0
    assert head_dim % 32 == 0, f"head_dim ({head_dim}) must be divisible by 32"
    assert head_dim >= 64, f"head_dim ({head_dim}) must be >= 64"
    assert dtype_str == "f16", "flash_attn_func currently only supports f16"
    assert BLOCK_N % 32 == 0
    assert BLOCK_N_OUT % BLOCK_N == 0

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    # Bank-conflict-friendly LDS strides.
    K_STRIDE = HEAD_DIM + 2
    VT_STRIDE = BLOCK_N + 2

    # Vectorized cooperative load constants.
    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8
    assert HEAD_DIM % VEC_WIDTH == 0
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    assert BLOCK_SIZE % THREADS_PER_ROW_LOAD == 0
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        assert BLOCK_N % ROWS_PER_BATCH_LOAD == 0
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    # K/VT circular buffers; defaults to 1/1, optional 3/3 with CK-like LDS sequence.
    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_VT_TILE_SIZE = HEAD_DIM * VT_STRIDE
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_VT_BASE = LDS_K_TOTAL_SIZE
    LDS_VT_TOTAL_SIZE = NUM_PREFETCH_V * LDS_VT_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_VT_TOTAL_SIZE

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"flash_attn_func_smem_{PATH_TAG}",
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE * 2

    _cache_tag = (
        PATH_TAG,
        NUM_HEADS,
        HEAD_DIM,
        CAUSAL,
        ENABLE_PREFETCH_3BUF,
        ENABLE_LDS_VEC16,
        REDUCE_MODE,
        NUM_PREFETCH_K,
        NUM_PREFETCH_V,
        waves_per_eu,
        flat_work_group_size,
        unsafe_fp_math,
        fast_fp_math,
        daz,
    )

    @flyc.kernel
    def flash_attn_func_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        seq_len: fx.Int32,
    ):
        elem_type = T.f16
        compute_type = T.f32
        llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
        q_ptr = _fly.extract_aligned_pointer_as_index(llvm_ptr_ty, Q.value)
        k_ptr = _fly.extract_aligned_pointer_as_index(llvm_ptr_ty, K.value)
        v_ptr = _fly.extract_aligned_pointer_as_index(llvm_ptr_ty, V.value)
        o_ptr = _fly.extract_aligned_pointer_as_index(llvm_ptr_ty, O.value)

        fm_fast = arith.FastMathFlags.fast
        v4f16_type = T.vec(4, elem_type)
        vxf16_type = T.vec(VEC_WIDTH, elem_type)
        v8f16_type = T.vec(8, elem_type)
        v16f32_type = T.vec(16, compute_type)

        seq_len_v = arith.index_cast(T.index, seq_len)

        # ---- LDS view ----
        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(
            base_ptr,
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()

        # ---- Thread / block indices ----
        block_id = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)

        # ---- Wave decomposition ----
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32  # 0/1

        # ---- Wave offsets ----
        wave_q_offset = wave_id * ROWS_PER_WAVE

        # ---- Decompose block_id ----
        head_idx = block_id % NUM_HEADS
        temp = block_id // NUM_HEADS
        num_q_tiles = seq_len_v // BLOCK_M
        q_tile_idx = temp % num_q_tiles
        batch_idx = temp // num_q_tiles
        q_start = q_tile_idx * BLOCK_M

        # ---- Cooperative load decomposition ----
        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        # ---- Helper: global flat index ----
        def global_idx(token_idx, col):
            token = batch_idx * seq_len_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

        _GEP_DYNAMIC = -2147483648  # LLVM's kDynamicIndex sentinel (0x80000000 as signed i32)

        def _gep_load(base_ptr, elem_idx, vec_type):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(llvm_ptr_ty, base_ptr, [idx_i64],
                              rawConstantIndices=[_GEP_DYNAMIC],
                              elem_type=elem_type,
                              noWrapFlags=0)
            return _llvm.LoadOp(vec_type, gep.result).result

        def _gep_store(val, base_ptr, elem_idx):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(llvm_ptr_ty, base_ptr, [idx_i64],
                              rawConstantIndices=[_GEP_DYNAMIC],
                              elem_type=elem_type,
                              noWrapFlags=0)
            _llvm.StoreOp(val, gep.result)

        def load_global_f16x4(base_ptr, base_idx):
            return _gep_load(base_ptr, base_idx, v4f16_type)

        def load_global_f16xN(base_ptr, base_idx):
            return _gep_load(base_ptr, base_idx, vxf16_type)

        def k_buf_base(buf_id):
            return arith.index(buf_id * LDS_K_TILE_SIZE)

        def vt_buf_base(buf_id):
            return arith.index(LDS_VT_BASE + buf_id * LDS_VT_TILE_SIZE)

        # ---- Cooperative K load (row-major, padded stride) ----
        def coop_load_k(tile_start, buf_id=0):
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    if row_valid:
                        g_idx = global_idx(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        lds_idx = k_base + lds_row * K_STRIDE + load_col_base
                        vec = load_global_f16xN(k_ptr, g_idx)
                        vector.store(vec, lds_kv, [lds_idx])
                else:
                    g_idx = global_idx(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    lds_idx = k_base + lds_row * K_STRIDE + load_col_base
                    vec = load_global_f16xN(k_ptr, g_idx)
                    vector.store(vec, lds_kv, [lds_idx])

        # ---- Cooperative V load (transposed, padded stride) ----
        def coop_load_v_transposed(tile_start, buf_id=0):
            vt_base = vt_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    if row_valid:
                        g_idx = global_idx(row_idx, load_col_base)
                        load_row = load_row_in_batch + row_offset
                        vec = load_global_f16xN(v_ptr, g_idx)
                        for e in range_constexpr(VEC_WIDTH):
                            elem = vector.extract(
                                vec,
                                static_position=[e],
                                dynamic_position=[],
                            )
                            col_e = load_col_base + e
                            lds_idx = vt_base + col_e * VT_STRIDE + load_row
                            _memref.StoreOp(elem, lds_kv, [lds_idx])
                else:
                    g_idx = global_idx(row_idx, load_col_base)
                    load_row = load_row_in_batch + row_offset
                    vec = load_global_f16xN(v_ptr, g_idx)
                    for e in range_constexpr(VEC_WIDTH):
                        elem = vector.extract(
                            vec,
                            static_position=[e],
                            dynamic_position=[],
                        )
                        col_e = load_col_base + e
                        lds_idx = vt_base + col_e * VT_STRIDE + load_row
                        _memref.StoreOp(elem, lds_kv, [lds_idx])

        # ---- Preload Q^T B-operand packs once (register-resident) ----
        # B operand uses j = lane_mod_32, k-subblock = lane_div_32*4.
        q_row = q_start + wave_q_offset + lane_mod_32
        q_row_i64 = arith.index_cast(T.i64, q_row)
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = arith.index(ks * K_STEP_QK) + lane_div_32 * 4
            g_idx = global_idx(q_row, q_col)
            q_b_packs.append(load_global_f16x4(q_ptr, g_idx))

        # ---- Constants ----
        c_neg_inf = arith.constant(float("-inf"), type=compute_type)
        c_zero_f = arith.constant(0.0, type=compute_type)
        c_one_f = arith.constant(1.0, type=compute_type)
        c_sm_scale = arith.constant(sm_scale, type=compute_type)
        c_log2e = arith.constant(1.4426950408889634, type=compute_type)
        c_zero_v16f32 = arith.constant_vector(0.0, v16f32_type)
        width_i32 = arith.constant(WARP_SIZE, type=T.i32)
        shuf_32_i32 = arith.constant(32, type=T.i32)
        c4_i32 = arith.constant(4, type=T.i32)
        lane_i32 = arith.index_cast(T.i32, lane)
        lane_xor_32_i32 = arith.XOrIOp(lane_i32, shuf_32_i32).result
        lane_xor_32_byte = arith.MulIOp(lane_xor_32_i32, c4_i32).result

        def reduction_peer(v_f32):
            if REDUCE_MODE == "ds_bpermute":
                v_i32 = arith.ArithValue(v_f32).bitcast(T.i32)
                peer_i32 = rocdl.ds_bpermute(T.i32, lane_xor_32_byte, v_i32)
                return arith.ArithValue(peer_i32).bitcast(compute_type)
            return arith.ArithValue(v_f32).shuffle_xor(shuf_32_i32, width_i32)

        # ---- KV loop upper bound ----
        kv_upper = q_start + BLOCK_M if CAUSAL else seq_len_v

        # Loop-carried: [m_old, l_old, o_acc_chunks...]
        init_args = [c_neg_inf, c_zero_f]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)

        for kv_block_start, inner_iter_args, loop_results in scf.for_(
            arith.index(0),
            kv_upper,
            arith.index(BLOCK_N_OUT),
            iter_args=init_args,
        ):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [
                inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)
            ]
            preload_k_count = (
                NUM_PREFETCH_K if NUM_PREFETCH_K < N_SUBTILES else N_SUBTILES
            )

            if ENABLE_PREFETCH_3BUF:
                for pre_k in range_constexpr(preload_k_count):
                    pre_k_slot = CK_LDS_SEQ[pre_k % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                    pre_k_start = kv_block_start + pre_k * BLOCK_N
                    coop_load_k(pre_k_start, pre_k_slot)
                rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, 0)
                gpu.barrier()

            for kv_sub in range_constexpr(N_SUBTILES):
                kv_start = kv_block_start + kv_sub * BLOCK_N

                if ENABLE_PREFETCH_3BUF:
                    k_slot = CK_LDS_SEQ[kv_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                else:
                    k_slot = 0
                    coop_load_k(kv_start, k_slot)
                    gpu.barrier()
                k_base = k_buf_base(k_slot)

                # ==== GEMM1: S = K @ Q^T (MFMA32), S in v16f32 ====
                s_acc = c_zero_v16f32
                for ks in range_constexpr(K_STEPS_QK):
                    k_idx = (
                        k_base
                        + lane_mod_32 * K_STRIDE
                        + ks * K_STEP_QK
                        + lane_div_32 * 4
                    )
                    k_pack = vector.load_op(v4f16_type, lds_kv, [k_idx])
                    q_pack = q_b_packs[ks]
                    s_acc = rocdl.mfma_f32_32x32x8f16(
                        v16f32_type,
                        [k_pack, q_pack, s_acc, 0, 0, 0],
                    )

                # ==== Online softmax over KV dimension (register only) ====
                s_vals = []
                for r in range_constexpr(16):
                    s_val = vector.extract(
                        s_acc,
                        static_position=[r],
                        dynamic_position=[],
                    )
                    s_val = arith.MulFOp(
                        s_val,
                        c_sm_scale,
                        fastmath=fm_fast,
                    ).result
                    if CAUSAL:
                        kv_row_rel = lane_div_32 * 4 + (r // 4) * 8 + (r % 4)
                        kv_col = kv_start + kv_row_rel
                        kv_col_i64 = arith.index_cast(T.i64, kv_col)
                        is_masked = arith.cmpi(
                            arith.CmpIPredicate.ugt,
                            kv_col_i64,
                            q_row_i64,
                        )
                        s_val = arith.select(is_masked, c_neg_inf, s_val)
                    s_vals.append(s_val)

                local_max = s_vals[0]
                for r in range_constexpr(15):
                    local_max = arith.MaxNumFOp(local_max, s_vals[r + 1]).result
                peer_max = reduction_peer(local_max)
                row_max = arith.MaxNumFOp(local_max, peer_max).result
                m_new = arith.MaxNumFOp(m_running, row_max).result

                diff_m = arith.SubFOp(
                    m_running,
                    m_new,
                    fastmath=fm_fast,
                ).result
                diff_m_s = arith.MulFOp(
                    diff_m,
                    c_log2e,
                    fastmath=fm_fast,
                ).result
                corr = arith.ArithValue(diff_m_s).exp2(fastmath=fm_fast)

                p_vals = []
                local_sum = c_zero_f
                for r in range_constexpr(16):
                    diff = arith.SubFOp(
                        s_vals[r],
                        m_new,
                        fastmath=fm_fast,
                    ).result
                    diff_s = arith.MulFOp(
                        diff,
                        c_log2e,
                        fastmath=fm_fast,
                    ).result
                    p = arith.ArithValue(diff_s).exp2(fastmath=fm_fast)
                    p_vals.append(p)
                    local_sum = arith.AddFOp(
                        local_sum,
                        p,
                        fastmath=fm_fast,
                    ).result

                peer_sum = reduction_peer(local_sum)
                tile_sum = arith.AddFOp(
                    local_sum,
                    peer_sum,
                    fastmath=fm_fast,
                ).result
                l_corr = arith.MulFOp(
                    corr,
                    l_running,
                    fastmath=fm_fast,
                ).result
                l_new = arith.AddFOp(
                    l_corr,
                    tile_sum,
                    fastmath=fm_fast,
                ).result

                # ==== Rescale O accumulators ====
                corr_vec = vector.broadcast(v16f32_type, corr)
                for dc in range_constexpr(D_CHUNKS):
                    o_accs[dc] = arith.MulFOp(
                        o_accs[dc],
                        corr_vec,
                        fastmath=fm_fast,
                    ).result

                if ENABLE_PREFETCH_3BUF and (kv_sub + preload_k_count) < N_SUBTILES:
                    next_k_sub = kv_sub + preload_k_count
                    next_k_start = kv_block_start + next_k_sub * BLOCK_N
                    next_k_slot = (
                        CK_LDS_SEQ[next_k_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_K
                    )
                    coop_load_k(next_k_start, next_k_slot)

                if ENABLE_PREFETCH_3BUF:
                    v_slot = CK_LDS_SEQ[kv_sub % len(CK_LDS_SEQ)] % NUM_PREFETCH_V
                else:
                    v_slot = 0
                v_base = vt_buf_base(v_slot)

                # ==== Load V^T for current tile into LDS_KV ====
                coop_load_v_transposed(kv_start, v_slot)
                rocdl.sched_group_barrier(rocdl.mask_dswr, 1, 0)
                gpu.barrier()

                # ==== Build P packs in MFMA32 B-input format from register S ====
                p_f16 = []
                for r in range_constexpr(16):
                    p_f16.append(arith.trunc_f(elem_type, p_vals[r]))
                p_packs = []
                for pks in range_constexpr(PV_K_STEPS):
                    p_base = pks * 4
                    p_packs.append(
                        vector.from_elements(
                            v4f16_type,
                            [
                                p_f16[p_base + 0],
                                p_f16[p_base + 1],
                                p_f16[p_base + 2],
                                p_f16[p_base + 3],
                            ],
                        )
                    )

                # ==== GEMM2: O^T += V^T @ P (MFMA32) ====
                for dc in range_constexpr(D_CHUNKS):
                    for pks in range_constexpr(PV_K_STEPS):
                        v_idx = (
                            v_base
                            + (dc * D_CHUNK + lane_mod_32) * VT_STRIDE
                            + pks * PV_K_STEP
                            + lane_div_32 * 4
                        )
                        v_pack = vector.load_op(v4f16_type, lds_kv, [v_idx])
                        o_accs[dc] = rocdl.mfma_f32_32x32x8f16(
                            v16f32_type,
                            [v_pack, p_packs[pks], o_accs[dc], 0, 0, 0],
                        )

                m_running = m_new
                l_running = l_new

            yield [m_running, l_running] + o_accs

        # ---- Normalize and store O ----
        l_final = loop_results[1]
        o_finals = [
            loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)
        ]

        inv_l = arith.DivFOp(
            c_one_f,
            l_final,
            fastmath=fm_fast,
        ).result
        inv_l_vec = vector.broadcast(v16f32_type, inv_l)

        for dc in range_constexpr(D_CHUNKS):
            o_norm_vec = arith.MulFOp(
                o_finals[dc],
                inv_l_vec,
                fastmath=fm_fast,
            ).result
            for r in range_constexpr(16):
                o_val = vector.extract(
                    o_norm_vec,
                    static_position=[r],
                    dynamic_position=[],
                )
                o_f16 = arith.trunc_f(elem_type, o_val)

                d_row_rel = lane_div_32 * 4 + (r // 4) * 8 + (r % 4)
                d_col = arith.index(dc * D_CHUNK) + d_row_rel
                o_global = global_idx(q_row, d_col)
                _gep_store(o_f16, o_ptr, o_global)

    @flyc.jit
    def launch_flash_attn_func(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _cache_tag
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = arith.index_cast(T.index, batch_size)
        sl_idx = arith.index_cast(T.index, seq_len)
        num_q_tiles = sl_idx // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS

        launcher = flash_attn_func_kernel(Q, K, V, O, seq_len)

        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32,
                            _wpe,
                        )
        if flat_work_group_size is not None:
            _fwgs = int(flat_work_group_size)
            if _fwgs >= 1:
                flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                for op in ctx.gpu_module_body.operations:
                    if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                        op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

        passthrough_entries = []
        if daz:
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("denormal-fp-math-f32"),
                ir.StringAttr.get("preserve-sign,preserve-sign"),
            ]))
            # passthrough_entries.append(ir.ArrayAttr.get([
            #     ir.StringAttr.get("no-nans-fp-math"),
            #     ir.StringAttr.get("true"),
            # ]))
            # passthrough_entries.append(ir.ArrayAttr.get([
            #     ir.StringAttr.get("unsafe-fp-math"),
            #     ir.StringAttr.get("true"),
            # ]))
        passthrough_entries.append(ir.ArrayAttr.get([
            ir.StringAttr.get("amdgpu-gemm-schedule-opt"),
            ir.StringAttr.get("true"),
        ]))
        for op in ctx.gpu_module_body.operations:
            if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_flash_attn_func


build_flash_attn_func_module = build_flash_attn_func_module_primary
