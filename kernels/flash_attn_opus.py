# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""OPUS-style flash_attn fast path for FlyDSL (D=128 bf16 gfx950).

Adopts OPUS's high-impact structural optimizations on top of the proven
FlyDSL flash_attn_func BLOCK_M=256 algorithm. The dispatcher will only
select this path when:

    head_dim == 128, dtype == bf16, gpu_arch >= gfx950,
    seq_len % 256 == 0, seq_len >= 384.

OPUS optimizations included:
    * 3D grid launch (H, num_q_blocks, B): better workload distribution
      across CUs vs. 1D grid (block_id_x decomposition arithmetic stays
      in scalar registers from the launcher rather than per-thread).
    * Double-buffered K and V LDS with DMA async loads.
    * Online softmax with **lazy rescaling** (OPUS lines 476-484, 540-548):
      skip ``O *= corr`` when no lane's row_max changed beyond
      RESCALE_THRESHOLD (= 8.0), saving 32 v_pk_mul per skipped tile.
    * ``s_setprio(1)`` raised before GEMM2/rescale, lowered after
      (OPUS lines 471, 493, 535, 557).
    * Inline-asm causal mask: ``v_cmp_lt_i32 + v_cndmask_b32`` pairs
      with immediate K-position thresholds, replacing the 32-element
      select chain (OPUS lines 233-249).
    * ``s_nop 15; s_nop 7`` yield window after s_setprio(0) to let the
      other wave-group seize the MFMA/VALU units.

Layout (LDS, MFMA, Q/K/V/O addressing) matches existing
``flash_attn_func.py`` BLOCK_M=256 path to inherit its proven correctness.
"""

import math as host_math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf, vector
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import ArithValue, _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.kernels_common import dtype_to_elem_type

KERNEL_NAME = "flash_attn_opus_kernel"
_LOG2E = host_math.log2(host_math.e)

# s_waitcnt bitfield encoding
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3
_LDS_ALIAS_DOMAIN = '#llvm.alias_scope_domain<id = "flydsl.opus.lds">'


def _llvm_value(value):
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _ds_read_tr16_b64_imm(result_type, addr_i32, imm_offset=0):
    """gfx950 ds_read_b64_tr_b16 with OPUS-style immediate byte offset."""
    imm = int(imm_offset)
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    raw = llvm.inline_asm(
        raw_type,
        [_llvm_value(addr_i32)],
        f"ds_read_b64_tr_b16 $0, $1 offset:{imm}\n"
        "s_waitcnt lgkmcnt(0)",
        "=v,v,~{memory}",
        has_side_effects=True,
    )
    return vector.BitCastOp(result_type, raw).result


def _extract_aligned_pointer(tensor, address_space=None) -> ir.Value:
    from flydsl._mlir.dialects import fly as _fly

    ptr_type = ir.Type.parse(
        "!llvm.ptr" if address_space is None else f"!llvm.ptr<{address_space}>"
    )
    return _fly.extract_aligned_pointer_as_index(ptr_type, _llvm_value(tensor))


def _pointer_load(result_type, ptr):
    return llvm.LoadOp(result_type, _llvm_value(ptr)).result


def _pointer_store(value, ptr):
    return llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (
        (n & _VMCNT_LO_MASK)
        | _LGKMCNT_EXPCNT_BASE
        | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    )
    rocdl.s_waitcnt(val)


def _lds_alias_scope_array(names):
    attrs = [
        f'#llvm.alias_scope<id = "{name}", domain = {_LDS_ALIAS_DOMAIN}>'
        for name in names
    ]
    return ir.Attribute.parse(f"[{', '.join(attrs)}]")


def build_flash_attn_opus_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    num_kv_heads=None,
    waves_per_eu=2,
    daz=True,
):
    """Build an OPUS-style flash_attn launcher for D=128 bf16 on gfx950.

    Launcher signature: ``launcher(Q, K, V, O, batch_size, seq_len, stride_kv_n=None, stride_q_n=None, head_dim_runtime=None, *, stream=None)``
    """
    gpu_arch = get_hip_arch()

    if not gpu_arch.startswith("gfx950"):
        raise RuntimeError(
            f"flash_attn_opus requires gfx950+ (uses ds_read_tr16_b64), got {gpu_arch}"
        )
    if head_dim != 128:
        raise RuntimeError(f"flash_attn_opus is D=128 only, got head_dim={head_dim}")
    if dtype_str != "bf16":
        raise RuntimeError(f"flash_attn_opus is bf16 only, got dtype={dtype_str}")

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0

    # ──────────────────────────── Tile constants ────────────────────────────
    # Match existing flash_attn_func BLOCK_M=256 path for layout compatibility.
    BLOCK_M = 256
    BLOCK_N = 64
    BLOCK_N_OUT = 64           # single sub-tile per outer iter (=BLOCK_N)
    N_SUBTILES = BLOCK_N_OUT // BLOCK_N
    K_SUB_N = 32               # MFMA W_N
    WARP_SIZE = 64
    NUM_WAVES = 8              # BLOCK_M / 32
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE   # 512
    ROWS_PER_WAVE = 32

    HEAD_DIM = head_dim
    K_STEP_QK = 16             # W_K
    K_STEPS_QK = HEAD_DIM // K_STEP_QK    # 8
    D_CHUNK = 32
    D_CHUNKS = HEAD_DIM // D_CHUNK    # 4
    PV_K_STEP = 16
    PV_K_STEPS = K_SUB_N // PV_K_STEP    # 2
    MFMA_LANE_K = 8

    NUM_HEADS_Q = num_heads
    NUM_HEADS_KV = num_kv_heads
    GQA_GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_KV
    CAUSAL = causal
    DEFAULT_STRIDE_Q_N = NUM_HEADS_Q * HEAD_DIM
    DEFAULT_STRIDE_KV_N = NUM_HEADS_KV * HEAD_DIM

    # ── OPUS LDS trait constants (matches gqa_d128_kernel_template.hpp §4-5) ──
    # K/V LDS layout: interleaved double-buffer K0, V0, K1, V1.
    # Per-warp slab line stride: smem_linear_wave + smem_padding.
    #   K: 512 + 8 = 520 bf16 per line (smem_padding_16B = 16 B = 8 bf16)
    #   V: 512 + 32 = 544 bf16 per line (smem_padding_64B = 64 B = 32 bf16)
    # Per-buffer: smem_n_rpt * smem_d_rpt * line_stride = 8 * 2 * line_stride
    #   K: 8320 bf16 (16640 B), V: 8704 bf16 (17408 B)
    # Total LDS (2 K + 2 V): 68096 B
    BF16_BYTES = 2
    D_128B_SIZE = 64                            # = 128 B / sizeof(bf16) = 64 bf16
    VEC_KV = 8                                  # bf16 per ds_read pack (also MFMA pack_a/pack_b)
    VEC_TR_V = 4                                # bf16 per ds_read_tr16_b64
    SMEM_LINEAR_WAVE = WARP_SIZE * 16 // BF16_BYTES   # 64 * 8 = 512 bf16 per wave per "line"
    SMEM_N_PER_WAVE = SMEM_LINEAR_WAVE // D_128B_SIZE  # 8 KV rows per wave per line
    SMEM_N_RPT = BLOCK_N // SMEM_N_PER_WAVE      # 64 / 8 = 8 lines along N
    SMEM_D_RPT = HEAD_DIM // D_128B_SIZE         # 128 / 64 = 2 lines along D
    SMEM_K_PAD = 16 // BF16_BYTES                # 8 bf16 (= 16 B padding)
    SMEM_V_PAD = 64 // BF16_BYTES                # 32 bf16 (= 64 B padding)
    SMEM_K_LINE_STRIDE = SMEM_LINEAR_WAVE + SMEM_K_PAD   # 520 bf16
    SMEM_V_LINE_STRIDE = SMEM_LINEAR_WAVE + SMEM_V_PAD   # 544 bf16
    SMEM_K_TILE_ELEMS = SMEM_N_RPT * SMEM_D_RPT * SMEM_K_LINE_STRIDE   # 8 * 2 * 520 = 8320
    SMEM_V_TILE_ELEMS = SMEM_N_RPT * SMEM_D_RPT * SMEM_V_LINE_STRIDE   # 8 * 2 * 544 = 8704
    NUM_PREFETCH_K = 2     # OPUS double-buffer
    NUM_PREFETCH_V = 2
    # OPUS interleaved layout: [K0][V0][K1][V1]
    OPUS_KV_PER_BUFFER = SMEM_K_TILE_ELEMS + SMEM_V_TILE_ELEMS   # 17024 bf16 per (K, V) pair
    LDS_KV_TOTAL_SIZE = NUM_PREFETCH_K * OPUS_KV_PER_BUFFER       # 34048 bf16 = 68096 B
    # K and V buffer bases (bf16 element offsets within the unified LDS region).
    OPUS_K_BUF_BASE = (0, OPUS_KV_PER_BUFFER)                     # K[0]=0, K[1]=17024
    OPUS_V_BUF_BASE = (SMEM_K_TILE_ELEMS,                         # V[0]=8320
                       SMEM_K_TILE_ELEMS + OPUS_KV_PER_BUFFER)    # V[1]=25344
    # u_rk OPUS strides (per derived element strides for the 8-axis u_rk layout).
    #   N-grp y-axis (axis 2)  : stride 256 bf16 (between v_s_lo and v_s_hi)
    #   K-step axis (axes 4, 5): inner stride 16 (i_5 step), outer 4160 (i_4 d_rpt)
    OPUS_URK_N_STRIP_STRIDE = 256       # bf16 offset to add for v_s_hi (n_strip=1)
    OPUS_URK_KSTEP_INNER = 16            # bf16 stride between consecutive K-steps within a d_rpt
    OPUS_URK_KSTEP_OUTER = SMEM_N_RPT * SMEM_K_LINE_STRIDE   # 4160 bf16 between d_rpt=0/1 arrays
    # u_rv OPUS per-lane base coefficients and step strides.
    #   base_per_lane(lane) = (lane/32)*OPUS_URV_GRPK + ((lane%16)/4)*OPUS_URV_LANE_HI
    #                       + ((lane/16)%2)*OPUS_URV_GRP_N + (lane%4)*OPUS_URV_LANE_LO
    OPUS_URV_GRPK     = 2176             # = 4 * 544 (grp_k stride, axes 2)
    OPUS_URV_LANE_HI  = SMEM_V_LINE_STRIDE   # 544 (lane_hi stride, axes 3)
    OPUS_URV_GRP_N    = 16               # 4 (lane_lo) * 4 (VEC_TR_V) = grp_n stride
    OPUS_URV_LANE_LO  = 4                # VEC_TR_V (lane_lo stride)
    OPUS_URV_STEP_K_STRIDE = 128         # = 2 * 64 = lane_hi_y * D_128B_SIZE (axis 4 element stride)
    OPUS_URV_DC_AXIS0 = SMEM_N_RPT * SMEM_V_LINE_STRIDE   # 4352 (d_rpt array, axis 0 element stride)
    OPUS_URV_DC_AXIS1 = 32               # axis 1 element stride (within half-D sub-row)
    OPUS_URV_I5_STRIDE = D_128B_SIZE     # 64 (axis 5 element stride within a step_k)

    # DMA load chunking
    VEC_WIDTH = 16
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD
    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    PATH_TAG = "OPUS"
    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"flash_attn_opus_smem_{PATH_TAG}",
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE * BF16_BYTES   # 68096 B for OPUS K0/V0/K1/V1

    # OPUS lazy-rescale threshold (line 374)
    OPUS_RESCALE_THRESHOLD = 8.0

    # Enable / disable individual OPUS optimizations via env vars (debug).
    OPUS_LAZY_RESCALE = os.getenv("FLYDSL_OPUS_LAZY_RESCALE", "1") == "1"
    OPUS_SETPRIO = os.getenv("FLYDSL_OPUS_SETPRIO", "1") == "1"
    # P5 stagger (`if (warp_id/4) s_barrier;` in prologue + reverse in
    # pre-store) is now functionally CORRECT in this port because all 6 V
    # LDS read sites have been hoisted into the cluster immediately
    # preceding their consumer (Clusters 2/6/10/12), mirroring the C++
    # template. With V already in VGPRs before each cluster boundary
    # barrier, the dual-group phase shift cannot race against the next
    # async_load that overwrites the V LDS buffer.
    #
    # Default ON: the P1-P6 OPUS path requires this flag set to truly
    # mirror gqa_d128_kernel_template.hpp end-to-end. Setting
    # `FLYDSL_OPUS_STAGGER=0` falls back to a symmetric (lockstep)
    # barrier — useful for A/B testing only.
    OPUS_ENABLE_STAGGER = os.getenv("FLYDSL_OPUS_STAGGER", "1") == "1"
    OPUS_YIELD_NOP = os.getenv("FLYDSL_OPUS_YIELD_NOP", "1") == "1"

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_opus_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        seq_len: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
    ):
        #     using T = opus::remove_cvref_t<Traits>;
        #     using D_ATTN = typename T::D_ATTN;
        #     using D_ACC = typename T::D_ACC;
        #
        elem_dtype = dtype_to_elem_type(dtype_str)
        elem_type = elem_dtype.ir_type
        compute_type = fx.Float32.ir_type
        fm_fast = fx.arith.FastMathFlags.fast
        v4i32_type = Vec.make_type(4, fx.Int32)
        v4f16_type = Vec.make_type(4, elem_dtype)
        v8f16_type = Vec.make_type(8, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)
        mfma_pack_type = v8f16_type

        _MFMA_MASK = 0x008
        _VALU_MASK = 0x002
        _EXP_MASK = 0x400

        seq_len_v = fx.Index(seq_len)
        stride_q_n_v = fx.Index(stride_q_n)
        stride_kv_n_v = fx.Index(stride_kv_n)
        stride_kv_n_bytes = stride_kv_n_v * fx.Index(BF16_BYTES)

        #     // Shared memory for K and V tiles
        #     __shared__ char smem_buf[T::smem_size_bytes()];
        #     smem<D_ATTN> s_k[2] = {
        #         make_smem(reinterpret_cast<D_ATTN*>(smem_buf)),
        #         make_smem(reinterpret_cast<D_ATTN*>(smem_buf) + T::smem_buffer_elems)
        #     };
        #     smem<D_ATTN> s_v[2] = {
        #         make_smem(reinterpret_cast<D_ATTN*>(smem_buf) + T::smem_k_tile_elems),
        #         make_smem(reinterpret_cast<D_ATTN*>(smem_buf) + T::smem_buffer_elems + T::smem_k_tile_elems)
        #     };
        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(
            base_ptr,
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()
        lds_kv_base_idx = buffer_ops.extract_base_index(lds_kv, address_space=3)
        lds_kv_base_ptr = buffer_ops.create_llvm_ptr(lds_kv_base_idx, address_space=3)

        lds_scope_names = ("lds_k0", "lds_k1", "lds_v0", "lds_v1")

        def lds_scope(kind, buf_id):
            return f"lds_{kind}{buf_id}"

        def lds_alias_scopes(name):
            return _lds_alias_scope_array([name])

        def lds_noalias_scopes(name):
            return _lds_alias_scope_array(
                [scope_name for scope_name in lds_scope_names if scope_name != name]
            )

        #     const int workgroup_x = block_id_x();
        #     const int q_block_idx = block_id_y();
        #     const int b = block_id_z();
        #     const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
        #     const int lane_id = thread_id_x() % T::WARP_SIZE;
        #     const int stagger = warp_id / 4;
        h_idx = fx.Index(gpu.block_idx.x)
        q_block_idx = fx.Index(gpu.block_idx.y)
        batch_idx = fx.Index(gpu.block_idx.z)
        tid = fx.Index(gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32

        _tid_i32 = arith.index_cast(T.i32, _raw(tid))
        _wave_id_uni_i32 = rocdl.readfirstlane(
            T.i32,
            arith.divsi(_tid_i32, arith.constant(WARP_SIZE, type=T.i32)),
        )
        _stagger_i32 = arith.divsi(
            _wave_id_uni_i32, arith.constant(4, type=T.i32)
        )
        wave_id_uni = fx.Index(arith.index_cast(T.index, _wave_id_uni_i32))
        stagger_is_one_i1 = arith.cmpi(
            arith.CmpIPredicate.ne, _stagger_i32, arith.constant(0, type=T.i32)
        )
        stagger_is_zero_i1 = arith.cmpi(
            arith.CmpIPredicate.eq, _stagger_i32, arith.constant(0, type=T.i32)
        )

        tr_k_group = (lane % 16) // 4
        tr_col_sub = lane % 4
        tr_col_half = (lane % 32) // 16

        #     const int q_block_size = T::NUM_WARPS * T::Q_TILE_SIZE;
        #     const int q_block_start = q_block_idx * q_block_size;
        wave_q_offset = wave_id * ROWS_PER_WAVE
        q_block_size = BLOCK_M
        q_start = q_block_idx * q_block_size

        #     const int group_size = kargs.H / kargs.H_KV;
        #     const int h = (workgroup_x % kargs.H_KV) * group_size + (workgroup_x / kargs.H_KV);
        #     const int h_kv = h / group_size;
        #     const int q_block_size = T::NUM_WARPS * T::Q_TILE_SIZE;
        #     const int q_block_start = q_block_idx * q_block_size;
        #     const int qo_gmem_offset = b * kargs.stride_q_b + q_block_start * kargs.stride_q_n + h * kargs.stride_q_h;
        #     const int kv_gmem_offset = b * kargs.stride_kv_b + h_kv * kargs.stride_kv_h;
        h_kv_idx = h_idx % NUM_HEADS_KV
        group_id = h_idx // NUM_HEADS_KV
        q_head_idx = h_kv_idx * GQA_GROUP_SIZE + group_id
        kv_head_idx = h_kv_idx

        #     // Global memory tensors
        #     auto g_q = make_gmem(reinterpret_cast<const D_ATTN*>(kargs.ptr_q) + qo_gmem_offset);
        #     auto g_k = make_gmem(reinterpret_cast<const D_ATTN*>(kargs.ptr_k) + kv_gmem_offset);
        #     auto g_v = make_gmem(reinterpret_cast<const D_ATTN*>(kargs.ptr_v) + kv_gmem_offset);
        #     auto g_o = make_gmem(reinterpret_cast<D_ATTN*>(kargs.ptr_o) + qo_gmem_offset);
        q_gmem_byte_offset = (
            (batch_idx * seq_len_v + q_start) * stride_q_n_v
            + q_head_idx * fx.Index(HEAD_DIM)
        ) * fx.Index(BF16_BYTES)
        kv_gmem_byte_offset = (
            batch_idx * seq_len_v * stride_kv_n_v
            + kv_head_idx * fx.Index(HEAD_DIM)
        ) * fx.Index(BF16_BYTES)
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=True, base_byte_offset=q_gmem_byte_offset
        )
        k_rsrc = buffer_ops.create_buffer_resource(
            K, max_size=True, base_byte_offset=kv_gmem_byte_offset
        )
        v_rsrc = buffer_ops.create_buffer_resource(
            V, max_size=True, base_byte_offset=kv_gmem_byte_offset
        )
        o_base_ptr = _extract_aligned_pointer(O)
        o_base_i64 = llvm.PtrToIntOp(T.i64, o_base_ptr).result
        o_base_lo = ArithValue(o_base_i64).trunci(T.i32)
        o_base_hi = ArithValue(ArithValue(o_base_i64).shrui(fx.Int64(32))).trunci(T.i32)
        o_rsrc = Vec.from_elements(
            [
                o_base_lo,
                o_base_hi,
                buffer_ops._create_i32_constant(0xFFFFFFFF),
                buffer_ops._create_i32_constant(buffer_ops._get_buffer_flags()),
            ],
            fx.Int32,
        ).ir_value()

        DMA_BYTES = 16
        NUM_DMA_K = SMEM_D_RPT
        NUM_DMA_V = SMEM_D_RPT

        _dma_size = fx.Int32(DMA_BYTES)
        _dma_off = fx.Int32(0)
        _dma_aux = fx.Int32(0)

        lane_in_warp = tid % fx.Index(WARP_SIZE)
        n_in_warp = lane_in_warp // fx.Index(VEC_KV)
        d_bucket = lane_in_warp % fx.Index(VEC_KV)

        #     // Scaling constants and online softmax state
        #     constexpr D_ACC RESCALE_THRESHOLD = D_ACC(8.0f);
        #     constexpr float LOG2_E = 1.44269504089f;
        #     const float temperature_scale = (1.0f / sqrtf(static_cast<float>(kargs.D))) * LOG2_E;
        #
        #     D_ACC m_row = opus::numeric_limits<D_ACC>::lowest();
        #     D_ACC l_row = 0.0f;
        #     D_ACC rescale_m = 1.0f;
        c_neg_inf = fx.Float32(float("-inf"))
        # c_neg_inf = fx.Float32(float(-1e30))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        head_dim_f32 = fx.Float32(fx.Int32(head_dim_runtime))
        c_log2e_f = fx.Float32(_LOG2E)
        c_sm_scale_log2e = fx.Float32(
            arith.mulf(
                _raw(fmath.rsqrt(head_dim_f32, fastmath=fm_fast)),
                _raw(c_log2e_f),
                fastmath=fm_fast,
            )
        )
        c_eight_f = fx.Float32(OPUS_RESCALE_THRESHOLD)
        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_32_i32 = fx.Int32(32)
        c4_i32 = fx.Int32(4)
        lane_i32 = fx.Int32(lane)
        v64bf16_type = Vec.make_type(K_STEPS_QK * MFMA_LANE_K, elem_dtype)
        v64f32_type = Vec.make_type(K_STEPS_QK * MFMA_LANE_K, fx.Float32)

        #     // Tile traversal helpers
        #     const int kv_tile_stride = T::KV_TILE_SIZE * kargs.stride_kv_n;
        #     const int num_kv_tiles = ceil_div(kargs.N, T::KV_TILE_SIZE);
        #     int max_num_tiles = num_kv_tiles;
        #     if constexpr (T::CAUSAL) {
        #         const int q_block_end = q_block_start + q_block_size;
        #         const int causal_num_tiles = ceil_div(q_block_end, T::KV_TILE_SIZE);
        #         max_num_tiles = causal_num_tiles < max_num_tiles ? causal_num_tiles : max_num_tiles;
        #     }
        #     auto kv_tile = [&](int tile_idx) { return tile_idx * kv_tile_stride; };
        kv_tile_size = fx.Index(BLOCK_N)
        num_kv_tiles = (seq_len_v + kv_tile_size - fx.Index(1)) // kv_tile_size
        if const_expr(CAUSAL):
            q_block_end = q_start + fx.Index(BLOCK_M)
            causal_num_tiles = (q_block_end + kv_tile_size - fx.Index(1)) // kv_tile_size
            max_num_tiles = fx.Index(
                ArithValue(causal_num_tiles < num_kv_tiles).select(
                    causal_num_tiles, num_kv_tiles
                )
            )
        else:
            max_num_tiles = num_kv_tiles

        #     // Partition layouts
        #     auto u_q  = make_layout_q<T>(warp_id, lane_id, kargs.stride_q_n);
        #     auto u_gk = make_layout_gk_gv<T>(warp_id, lane_id, kargs.stride_kv_n);
        #     auto u_sk = make_layout_sk_sv<T, T::smem_padding_16B>(warp_id);
        #     auto u_rk = make_layout_rk<T>(lane_id);
        #     auto u_gv = make_layout_gk_gv<T>(warp_id, lane_id, kargs.stride_kv_n);
        #     auto u_sv = make_layout_sk_sv<T, T::smem_padding_64B>(warp_id);
        #     auto u_rv = make_layout_rv<T>(lane_id);
        urk_base_per_lane = (
            (lane_mod_32 % fx.Index(8)) * fx.Index(SMEM_K_LINE_STRIDE)
            + (lane_mod_32 // fx.Index(8)) * fx.Index(D_128B_SIZE)
            + lane_div_32 * fx.Index(VEC_KV)
        )

        urv_base_per_lane = (
            lane_div_32 * fx.Index(OPUS_URV_GRPK)
            + ((lane % fx.Index(16)) // fx.Index(4)) * fx.Index(OPUS_URV_LANE_HI)
            + ((lane // fx.Index(16)) % fx.Index(2)) * fx.Index(OPUS_URV_GRP_N)
            + (lane % fx.Index(4)) * fx.Index(OPUS_URV_LANE_LO)
        )

        #     // Causal masking helpers
        #     [[maybe_unused]] const int q_start_pos = q_block_start + warp_id * T::Q_TILE_SIZE;
        #     [[maybe_unused]] const opus::u32_t neg_inf_v = std::bit_cast<opus::u32_t>(-opus::numeric_limits<D_ACC>::infinity());
        _NEG_INF_F32_BITS = 0xFF800000

        _LGKMCNT_0_ONLY = 0xC07F


        def _mfma(mfma_fn, a, b, c):
            return mfma_fn(v16f32_type, [a, b, c])

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        def mfma_acc(a, b, c):
            return _mfma(rocdl.mfma_f32_32x32x16_bf16, a, b, c)

        def _sched_barrier_pairs(pairs, valu_cnt, group):
            """Emit `pairs` × {1 MFMA + valu_cnt VALU} sched_group_barrier groups.

            Matches gqa_d128_kernel_template.hpp's
            `sched_barrier_pairs<Pairs, VALU_CNT, Group>()` (lines 18-23).
            """
            for _ in range(pairs):
                rocdl.sched_group_barrier(_MFMA_MASK, 1, group)
                rocdl.sched_group_barrier(_VALU_MASK, valu_cnt, group)

        def _sched_barrier_exp_pairs(pairs, exp_cnt, group):
            """Emit `pairs` × {1 MFMA + exp_cnt EXP} sched_group_barrier groups.

            Matches gqa_d128_kernel_template.hpp's
            `sched_barrier_exp_pairs<Pairs, EXP_CNT, Group>()` (lines 25-30).
            """
            for _ in range(pairs):
                rocdl.sched_group_barrier(_MFMA_MASK, 1, group)
                rocdl.sched_group_barrier(_EXP_MASK, exp_cnt, group)

        def ds_read_tr_v4f16_imm(lds_base_elem_idx, imm_bytes):
            byte_offset = lds_base_elem_idx * 2 + lds_kv_offset
            addr_i32 = fx.Int32(byte_offset)
            return _ds_read_tr16_b64_imm(v4f16_type, addr_i32, imm_bytes)

        def global_idx_q(token_idx, col):
            token = batch_idx * seq_len_v + token_idx
            return token * stride_q_n_v + q_head_idx * HEAD_DIM + col

        def _concat_vectors(lhs, rhs):
            lhs_vec = Vec(lhs)
            rhs_vec = Vec(rhs)
            return lhs_vec.shuffle(
                rhs_vec,
                list(range(lhs_vec.numel))
                + [lhs_vec.numel + i for i in range(rhs_vec.numel)],
            )

        def _raw_buffer_load_bytes(result_type, rsrc, byte_offset_i32):
            zero = buffer_ops._create_i32_constant(0)
            aux = buffer_ops._create_i32_constant(0)
            return buffer_ops.rocdl.RawPtrBufferLoadOp(
                result_type,
                rsrc,
                _raw(byte_offset_i32),
                zero,
                aux,
            ).result

        def load_q_all(q_row_in_block):
            q_raw_packs = []
            for ks in range_constexpr(K_STEPS_QK):
                q_col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                g_idx = q_row_in_block * stride_q_n_v + q_col
                q_byte_offset = fx.Int32(g_idx) << fx.Int32(1)
                q_i32_pack = _raw_buffer_load_bytes(v4i32_type, q_rsrc, q_byte_offset)
                q_raw_packs.append(Vec(q_i32_pack, (4,), fx.Int32).bitcast(elem_dtype).ir_value())
            q_16_packs = []
            for pair in range_constexpr(K_STEPS_QK // 2):
                q_16_packs.append(
                    _concat_vectors(q_raw_packs[pair * 2], q_raw_packs[pair * 2 + 1])
                )

            q_32_packs = []
            for pair in range_constexpr(K_STEPS_QK // 4):
                q_32_packs.append(
                    _concat_vectors(q_16_packs[pair * 2], q_16_packs[pair * 2 + 1])
                )

            q_all = _concat_vectors(q_32_packs[0], q_32_packs[1])
            return Vec(q_all, (K_STEPS_QK * MFMA_LANE_K,), elem_dtype)

        def scale_q_all(q_all_bf16):
            fm_fast_attr = ir.Attribute.parse("#llvm.fastmath<fast>")
            q_all_f32_op = llvm.FPExtOp(v64f32_type, _raw(q_all_bf16))
            q_all_f32_op.operation.attributes["fastmathFlags"] = fm_fast_attr
            q_all_f32 = q_all_f32_op.result
            scale_vec = Vec.from_elements([c_sm_scale_log2e], fx.Float32).broadcast_to(
                K_STEPS_QK * MFMA_LANE_K
            )
            q_all_scaled_f32 = arith.mulf(
                _raw(scale_vec),
                _raw(q_all_f32),
                fastmath=fm_fast,
            )
            q_all_scaled_bf16_op = llvm.FPTruncOp(v64bf16_type, q_all_scaled_f32)
            q_all_scaled_bf16_op.operation.attributes["fastmathFlags"] = fm_fast_attr
            q_all_scaled_bf16 = q_all_scaled_bf16_op.result
            return Vec(q_all_scaled_bf16, (K_STEPS_QK * MFMA_LANE_K,), elem_dtype)

        def get_q_pack(q_all_scaled_bf16, ks):
            q_vec = Vec(q_all_scaled_bf16)
            base = ks * MFMA_LANE_K
            return q_vec.shuffle(q_vec, [base + i for i in range(MFMA_LANE_K)]).ir_value()

        def _make_raw_buffer_rsrc(tensor):
            base_ptr = _extract_aligned_pointer(tensor)
            base_i64 = llvm.PtrToIntOp(T.i64, base_ptr).result
            base_lo = ArithValue(base_i64).trunci(T.i32)
            base_hi = ArithValue(ArithValue(base_i64).shrui(fx.Int64(32))).trunci(T.i32)
            return Vec.from_elements(
                [
                    base_lo,
                    base_hi,
                    buffer_ops._create_i32_constant(0xFFFFFFFF),
                    buffer_ops._create_i32_constant(buffer_ops._get_buffer_flags()),
                ],
                fx.Int32,
            ).ir_value()

        def _bitcast_i32(value):
            return fx.Int32(ArithValue(value).bitcast(fx.Int32.ir_type))

        def _bitcast_f32(value):
            return fx.Float32(ArithValue(value).bitcast(fx.Float32.ir_type))

        def _attn_mask_vec2_imm(rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32):
            """OPUS-style pair mask asm: 2 compares followed by 2 cndmasks."""
            asm_str = (
                f"v_cmp_lt_i32_e64 $0, $6, {int(thr_x)}\n\t"
                f"v_cmp_lt_i32_e64 $1, $6, {int(thr_y)}\n\t"
                "v_cndmask_b32_e64 $2, $4, $7, $0\n\t"
                "v_cndmask_b32_e64 $3, $5, $7, $1"
            )
            ret_struct_ty = ir.Type.parse("!llvm.struct<(i64, i64, i32, i32)>")
            ret = llvm.inline_asm(
                ret_struct_ty,
                [
                    _llvm_value(x_ref_i32),
                    _llvm_value(y_ref_i32),
                    _llvm_value(rel_i32),
                    _llvm_value(neg_inf_i32),
                ],
                asm_str,
                "=s,=s,=v,=v,2,3,v,v,~{vcc}",
                has_side_effects=True,
            )
            return llvm.extractvalue(T.i32, ret, [2]), llvm.extractvalue(T.i32, ret, [3])

        def _anchor_vec(value):
            val_ir = _llvm_value(value)
            return llvm.inline_asm(
                val_ir.type,
                [val_ir],
                "",
                "=v,0",
                has_side_effects=True,
            )

        def _anchor_pair(lo, hi):
            lo_ir = _llvm_value(lo)
            hi_ir = _llvm_value(hi)
            ret_ty = ir.Type.parse("!llvm.struct<(vector<16xf32>, vector<16xf32>)>")
            ret = llvm.inline_asm(
                ret_ty,
                [lo_ir, hi_ir],
                "",
                "=v,=v,0,1",
                has_side_effects=True,
            )
            return (
                llvm.extractvalue(lo_ir.type, ret, [0]),
                llvm.extractvalue(hi_ir.type, ret, [1]),
            )

        def _anchor_packs(packs):
            return [_anchor_vec(p) for p in packs]

        def _stagger_extra_barrier_if_one():
            """Emit `sched_barrier(0); s_barrier;` only when stagger == 1.

            Matches C++ template gqa_d128_kernel_template.hpp lines 415-418.
            The body runs on warps 4-7 only, advancing their s_barrier ordinal
            by one relative to warps 0-3 → starts the dual-group phase shift.

            Emit real CFG + ROCDL barrier ops instead of opaque inline asm, so
            LLVM sees the `llvm.amdgcn.s.barrier` intrinsic and keeps the
            conditional barrier as a scheduling boundary.
            """
            if_op = scf.IfOp(stagger_is_one_i1, [], has_else=False, loc=ir.Location.unknown())
            with ir.InsertionPoint(if_op.regions[0].blocks[0]):
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                scf.YieldOp([])

        def _stagger_extra_barrier_if_zero():
            """Emit `s_barrier;` only when stagger == 0.

            Matches C++ template gqa_d128_kernel_template.hpp lines 748-750.
            The body runs on warps 0-3 only, letting them catch up by one
            s_barrier ordinal before the final global store → closes the
            dual-group phase shift opened in the prologue.
            """
            llvm.inline_asm(
                ir.Type.parse("!llvm.void"),
                [_stagger_i32],
                (
                    "s_cmp_eq_u32 $0, 0\n\t"
                    "s_cbranch_scc0 1f\n\t"
                    "s_barrier\n\t"
                    "1:"
                ),
                "s",
                has_side_effects=True,
            )

        def bf16_trunc_pack_v8(f32_vals):
            pairs = []
            for j in range_constexpr(4):
                pairs.append(
                    rocdl.cvt_pk_bf16_f32(f32_vals[j * 2], f32_vals[j * 2 + 1])
                )
            return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()

        def k_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(OPUS_K_BUF_BASE[buf_id])
            # runtime buf_id (rare): K0=0, K1=OPUS_KV_PER_BUFFER
            return buf_id * fx.Index(OPUS_KV_PER_BUFFER)

        def v_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(OPUS_V_BUF_BASE[buf_id])
            # runtime buf_id (rare): V0=SMEM_K_TILE_ELEMS, V1=SMEM_K_TILE_ELEMS+OPUS_KV_PER_BUFFER
            return fx.Index(SMEM_K_TILE_ELEMS) + buf_id * fx.Index(OPUS_KV_PER_BUFFER)

        def async_load_k(tile_start, buf_id):
            k_lds_byte_base = lds_kv_base_idx + k_buf_base(buf_id) * fx.Index(BF16_BYTES)
            for d in range_constexpr(NUM_DMA_K):
                lds_addr = (
                    k_lds_byte_base
                    + wave_id_uni * fx.Index(SMEM_K_LINE_STRIDE * BF16_BYTES)
                    + fx.Index(d * SMEM_N_RPT * SMEM_K_LINE_STRIDE * BF16_BYTES)
                )
                lds_ptr = buffer_ops.create_llvm_ptr(lds_addr, address_space=3)

                n_in_tile = n_in_warp * fx.Index(NUM_WAVES) + wave_id
                global_d = d_bucket * fx.Index(VEC_KV) + fx.Index(d * D_128B_SIZE)
                uniform_byte = tile_start * stride_kv_n_bytes
                lane_byte = (
                    n_in_tile * stride_kv_n_bytes
                    + global_d * fx.Index(BF16_BYTES)
                )
                voffset = fx.Int32(lane_byte)
                soffset = fx.Int32(uniform_byte)
                scope_name = lds_scope("k", buf_id)
                rocdl.raw_ptr_buffer_load_lds(
                    k_rsrc,
                    lds_ptr,
                    _dma_size,
                    voffset,
                    soffset,
                    _dma_off,
                    _dma_aux,
                    alias_scopes=lds_alias_scopes(scope_name),
                    noalias_scopes=lds_noalias_scopes(scope_name),
                )

        def async_load_v(tile_start, buf_id):
            v_lds_byte_base = lds_kv_base_idx + v_buf_base(buf_id) * fx.Index(BF16_BYTES)
            for d in range_constexpr(NUM_DMA_V):
                lds_addr = (
                    v_lds_byte_base
                    + wave_id_uni * fx.Index(SMEM_V_LINE_STRIDE * BF16_BYTES)
                    + fx.Index(d * SMEM_N_RPT * SMEM_V_LINE_STRIDE * BF16_BYTES)
                )
                lds_ptr = buffer_ops.create_llvm_ptr(lds_addr, address_space=3)

                n_in_tile = n_in_warp * fx.Index(NUM_WAVES) + wave_id
                global_d = d_bucket * fx.Index(VEC_KV) + fx.Index(d * D_128B_SIZE)
                uniform_byte = tile_start * stride_kv_n_bytes
                lane_byte = (
                    n_in_tile * stride_kv_n_bytes
                    + global_d * fx.Index(BF16_BYTES)
                )
                voffset = fx.Int32(lane_byte)
                soffset = fx.Int32(uniform_byte)
                scope_name = lds_scope("v", buf_id)
                rocdl.raw_ptr_buffer_load_lds(
                    v_rsrc,
                    lds_ptr,
                    _dma_size,
                    voffset,
                    soffset,
                    _dma_off,
                    _dma_aux,
                    alias_scopes=lds_alias_scopes(scope_name),
                    noalias_scopes=lds_noalias_scopes(scope_name),
                )

        def reduction_pair(v_f32):
            v_i32 = _raw(_bitcast_i32(v_f32))
            pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
            swapped = rocdl.permlane32_swap(pair_ty, v_i32, v_i32, False, True)
            lhs_i32 = llvm.extractvalue(T.i32, swapped, [0])
            rhs_i32 = llvm.extractvalue(T.i32, swapped, [1])
            return _raw(_bitcast_f32(lhs_i32)), _raw(_bitcast_f32(rhs_i32))

        def async_load_k_from_lds_to_vgpr(buf_id, urk_base):
            """Read all 16 K MFMA packs from LDS buffer `buf_id` (OPUS u_rk)."""
            k_base = k_buf_base(buf_id)
            k_lo = [None] * K_STEPS_QK
            k_hi = [None] * K_STEPS_QK

            def load_k_pack_aligned(elem_idx):
                scope_name = lds_scope("k", buf_id)
                byte_offset = elem_idx * fx.Index(BF16_BYTES)
                ptr = buffer_ops.get_element_ptr(
                    lds_kv_base_ptr, byte_offset=byte_offset, elem_type=T.i8
                )
                return llvm.LoadOp(
                    mfma_pack_type,
                    ptr,
                    alignment=16,
                    alias_scopes=lds_alias_scopes(scope_name),
                    noalias_scopes=lds_noalias_scopes(scope_name),
                ).result

            for ks in range_constexpr(K_STEPS_QK):
                ks_offset = (ks // 4) * OPUS_URK_KSTEP_OUTER + (ks % 4) * OPUS_URK_KSTEP_INNER
                idx_lo = k_base + urk_base + fx.Index(ks_offset)
                idx_hi = idx_lo + fx.Index(OPUS_URK_N_STRIP_STRIDE)
                k_lo[ks] = load_k_pack_aligned(idx_lo)
                k_hi[ks] = load_k_pack_aligned(idx_hi)
            return k_lo, k_hi

        def _read_v_packs_for_buf(buf_id, urv_base):
            """Read all V packs from LDS buffer `buf_id` in OPUS issue order.

            Returns packs indexed as [k_substep][dc], but emits the ds_read_tr16_b64
            sequence as dc outer / k_substep inner to mirror OPUS's tr_load layout
            issue order.
            """
            v_base = v_buf_base(buf_id)
            lds_base = v_base + urv_base
            packs = [[None] * D_CHUNKS for _ in range(4)]
            for dc in range_constexpr(D_CHUNKS):
                i_0 = dc // 2     # axes 0 selection: 0 → D < 64, 1 → D >= 64 (d_rpt)
                i_1 = dc % 2      # axes 1 selection: half-D sub-row group
                dc_off = i_0 * OPUS_URV_DC_AXIS0 + i_1 * OPUS_URV_DC_AXIS1
                for k_substep in range_constexpr(4):
                    step_k_off = k_substep * OPUS_URV_STEP_K_STRIDE
                    imm_lo = (step_k_off + dc_off) * BF16_BYTES
                    # axis 5 = 0 and axis 5 = 1 reads (in-register K stride 64 bf16)
                    a = ds_read_tr_v4f16_imm(lds_base, imm_lo)
                    b = ds_read_tr_v4f16_imm(
                        lds_base,
                        imm_lo + OPUS_URV_I5_STRIDE * BF16_BYTES,
                    )
                    packs[k_substep][dc] = Vec(a).shuffle(
                        Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]
                    ).ir_value()
            return packs

        def _gemm0(k_lo, k_hi):
            v_s_lo = c_zero_v16f32
            v_s_hi = c_zero_v16f32
            for ks in range_constexpr(K_STEPS_QK):
                q_pack = get_q_pack(q_all_scaled_bf16, ks)
                v_s_lo = mfma_acc(k_lo[ks], q_pack, v_s_lo)
                v_s_hi = mfma_acc(k_hi[ks], q_pack, v_s_hi)
            return v_s_lo, v_s_hi

        def _causal_mask_inplace(s_lo, s_hi, tile_idx):
            """Apply causal mask using OPUS inline-asm attn_mask_vec2_imm (OPUS u_rk path).

            This mirrors C++ attn_mask_causal_tile:
              k_pos = kv_start + i_n * W_N + lane_group * 4
              thr(r) = (r//4) * 8 + (r%4)

            For MFMA-C output lane k (lane_group = lane_id / 32):
              v_s_lo[r] of lane k → N = lane_group*4 + (r//4)*8 + (r%4)
              v_s_hi[r] of lane k → N = 32 + lane_group*4 + (r//4)*8 + (r%4)

            Mask if M = q_row < kv_start + N → set to -inf.
            Rewrite as `rel < thr` with:
              rel_lo = q_row - kv_start - lane_group*4
              rel_hi = rel_lo - 32
              thr(r) = (r//4)*8 + (r%4)
            """
            kv_tile_start = tile_idx * fx.Index(BLOCK_N)
            kv_start_i32 = fx.Int32(kv_tile_start)
            lane_off_i32 = fx.Int32(lane_div_32) * fx.Int32(4)
            rel_lo_i32 = fx.Int32(q_row_i32 - kv_start_i32 - lane_off_i32)
            # v_s_hi: i_n=1, so N += W_N = 32
            rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(32))
            neg_inf_i32 = fx.Int32(_NEG_INF_F32_BITS)

            # 8 (thr_x, thr_y) pairs matching C++ attn_mask_causal_tile.
            # For r=0..15: thr(r) = (r//4)*8 + (r%4):
            #   r= 0: 0  r= 1: 1   r= 2: 2  r= 3: 3
            #   r= 4: 8  r= 5: 9   r= 6:10  r= 7:11
            #   r= 8:16  r= 9:17   r=10:18  r=11:19
            #   r=12:24  r=13:25   r=14:26  r=15:27
            pair_thresholds = [
                (0, 1), (2, 3),         # r=0,1  r=2,3
                (8, 9), (10, 11),       # r=4,5  r=6,7
                (16, 17), (18, 19),     # r=8,9  r=10,11
                (24, 25), (26, 27),     # r=12,13 r=14,15
            ]
            for p in range_constexpr(len(pair_thresholds)):
                thr_x, thr_y = pair_thresholds[p]
                idx_x = p * 2
                idx_y = p * 2 + 1

                # s_lo pair (n_strip = 0)
                x_lo_bits = _raw(_bitcast_i32(s_lo[idx_x]))
                y_lo_bits = _raw(_bitcast_i32(s_lo[idx_y]))
                new_x_lo, new_y_lo = _attn_mask_vec2_imm(
                    rel_lo_i32, neg_inf_i32, thr_x, thr_y,
                    x_lo_bits, y_lo_bits,
                )
                s_lo[idx_x] = _raw(_bitcast_f32(new_x_lo))
                s_lo[idx_y] = _raw(_bitcast_f32(new_y_lo))

            for p in range_constexpr(len(pair_thresholds)):
                thr_x, thr_y = pair_thresholds[p]
                idx_x = p * 2
                idx_y = p * 2 + 1

                # s_hi pair (n_strip = 1, rel shifted by 4)
                x_hi_bits = _raw(_bitcast_i32(s_hi[idx_x]))
                y_hi_bits = _raw(_bitcast_i32(s_hi[idx_y]))
                new_x_hi, new_y_hi = _attn_mask_vec2_imm(
                    rel_hi_i32, neg_inf_i32, thr_x, thr_y,
                    x_hi_bits, y_hi_bits,
                )
                s_hi[idx_x] = _raw(_bitcast_f32(new_x_hi))
                s_hi[idx_y] = _raw(_bitcast_f32(new_y_hi))

        def _causal_mask_prologue_if_needed(s_lo, s_hi, tile_idx=fx.Index(0), kv_end_pos=fx.Index(BLOCK_N)):
            """Return masked score vectors when OPUS's causal guard is active."""
            acc_values = [_raw(v) for v in (s_lo + s_hi)]
            result_types = [v.type for v in acc_values]
            mask_needed = arith.cmpi(
                arith.CmpIPredicate.slt,
                q_start_pos_i32,
                _raw(fx.Int32(kv_end_pos)),
            )
            if_op = scf.IfOp(mask_needed, result_types, has_else=True, loc=ir.Location.unknown())

            with ir.InsertionPoint(if_op.regions[0].blocks[0]):
                then_lo = list(s_lo)
                then_hi = list(s_hi)
                _causal_mask_inplace(then_lo, then_hi, tile_idx)
                scf.YieldOp([_raw(v) for v in (then_lo + then_hi)])

            if len(if_op.regions[1].blocks) == 0:
                if_op.regions[1].blocks.append(*[])
            with ir.InsertionPoint(if_op.regions[1].blocks[0]):
                scf.YieldOp(acc_values)

            results = list(if_op.results)
            return results[:16], results[16:]

        def attn_row_max(s_lo, s_hi):
            m = c_neg_inf
            for r in range_constexpr(16):
                m = _fmax(m, s_lo[r])
            for r in range_constexpr(16):
                m = _fmax(m, s_hi[r])
            lhs, rhs = reduction_pair(m)
            return _fmax(lhs, rhs)

        def attn_sub_row(s_lo, s_hi, m_row):
            lo_sub = []
            hi_sub = []
            for r in range_constexpr(16):
                lo_sub.append(_fsub(s_lo[r], m_row))
            for r in range_constexpr(16):
                hi_sub.append(_fsub(s_hi[r], m_row))
            return lo_sub, hi_sub

        def attn_exp2_first_half_slice(s_lo, s_hi):
            lo_partial = []
            for r in range_constexpr(16):
                lo_partial.append(rocdl.exp2(T.f32, _raw(s_lo[r])))
            return lo_partial, list(s_hi)

        def attn_sub_row_first_half_exp2_slice(s_lo, s_hi, m_row):
            lo_partial = []
            hi_partial = []
            for r in range_constexpr(16):
                diff_lo = _fsub(s_lo[r], m_row)
                lo_partial.append(rocdl.exp2(T.f32, _raw(diff_lo)))
            for r in range_constexpr(16):
                diff_hi = _fsub(s_hi[r], m_row)
                hi_partial.append(diff_hi)
            return lo_partial, hi_partial

        def attn_exp2_second_half_slice(hi_partial_vec):
            hi_full = []
            for r in range_constexpr(16):
                hi_full.append(
                    rocdl.exp2(T.f32, _raw(Vec(hi_partial_vec)[r]))
                )
            return hi_full

        def attn_sum_parts(lo_partial_list, hi_full):
            local_sum = c_zero_f
            for r in range_constexpr(16):
                local_sum = _fadd(local_sum, lo_partial_list[r])
            for r in range_constexpr(16):
                local_sum = _fadd(local_sum, hi_full[r])
            lhs_sum, rhs_sum = reduction_pair(local_sum)
            return _fadd(lhs_sum, rhs_sum)

        def cast_p_parts(lo_partial_list, hi_full):
            p_lo_packs = []
            p_hi_packs = []
            for pks in range_constexpr(PV_K_STEPS):
                p_base = pks * 8
                lo_slice = [lo_partial_list[p_base + s] for s in range_constexpr(8)]
                p_lo_packs.append(bf16_trunc_pack_v8(lo_slice))
                hi_slice = hi_full[p_base:p_base + 8]
                p_hi_packs.append(bf16_trunc_pack_v8(hi_slice))
            return p_lo_packs, p_hi_packs

        def _scale_o(o_accs, scale_scalar):
            scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = _fmul(Vec(o_accs[dc]), scale_vec)

        def _waitcnt_lgkm_0_vm_n(n):
            """Combined: lgkmcnt(0) + vmcnt(n)."""
            val = (
                (n & _VMCNT_LO_MASK)
                | (7 << 4)
                | (0 << 8)
                | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
            )
            rocdl.s_waitcnt(val)

        #     // Prologue
        #     async_load<T::VEC_KV>(g_k, s_k[0].ptr, u_gk, u_sk, kv_tile(0));
        async_load_k(fx.Index(0), 0)

        #     __builtin_amdgcn_s_waitcnt(0);
        rocdl.s_waitcnt(0)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     __builtin_amdgcn_s_barrier();
        rocdl.s_barrier()

        #     v_q = load<T::VEC_Q>(g_q, u_q);
        #     auto v_q_f32 = opus::cast<float>(v_q);
        #     static_for<q_len>([&](auto i) { v_q_f32[i.value] *= temperature_scale; });
        #     v_q = opus::cast<D_ATTN>(v_q_f32);
        q_row_in_block = wave_q_offset + lane_mod_32
        q_start_pos_i32 = fx.Int32(q_start + wave_id_uni * fx.Index(ROWS_PER_WAVE))
        q_row = q_start + q_row_in_block
        q_row_i32 = fx.Int32(q_row)
        q_in_bounds = q_row < seq_len_v
        q_all_bf16 = load_q_all(q_row_in_block)
        q_all_scaled_bf16 = scale_q_all(q_all_bf16)

        #     async_load<T::VEC_KV>(g_k, s_k[1].ptr, u_gk, u_sk, kv_tile(1));
        async_load_k(fx.Index(BLOCK_N), 1)
        #     async_load<T::VEC_KV>(g_v, s_v[0].ptr, u_gv, u_sv, kv_tile(0));
        async_load_v(fx.Index(0), 0)

        #     v_k = load<T::VEC_KV>(s_k[0], u_rk);
        k_pl_pro, k_ph_pro = async_load_k_from_lds_to_vgpr(0, urk_base_per_lane)

        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     s_waitcnt_lgkmcnt(0_I);
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        #     s_waitcnt_vmcnt(number<T::v_buffer_load_insts>{});
        _waitcnt_vm_n(NUM_DMA_V)

        #     if (stagger) {
        #         __builtin_amdgcn_sched_barrier(0);
        #         __builtin_amdgcn_s_barrier();
        #     }
        if const_expr(OPUS_ENABLE_STAGGER):
            _stagger_extra_barrier_if_one()
        else:
            rocdl.sched_barrier(0)
            rocdl.s_barrier()

        #     v_s[0] = mma0(v_q, v_k);
        v_s_0_lo_raw, v_s_0_hi_raw = _gemm0(k_pl_pro, k_ph_pro)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     if constexpr (T::CAUSAL) {
        #         const int kv_end_pos = T::KV_TILE_SIZE;
        #         if (q_start_pos < kv_end_pos) {
        #             attn_mask_causal_tile<T>(v_s[0], q_start_pos, 0, neg_inf_v, lane_id);
        #         }
        #     }
        s_lo_pro = [Vec(v_s_0_lo_raw)[r] for r in range_constexpr(16)]
        s_hi_pro = [Vec(v_s_0_hi_raw)[r] for r in range_constexpr(16)]
        if const_expr(CAUSAL):
            s_lo_pro, s_hi_pro = _causal_mask_prologue_if_needed(s_lo_pro, s_hi_pro)

        #     m_row = attn_row_max<T>(v_s[0]);
        m_row_pro = attn_row_max(s_lo_pro, s_hi_pro)

        #     attn_sub_row<T>(v_s[0], m_row);
        s_lo_pro, s_hi_pro = attn_sub_row(s_lo_pro, s_hi_pro, m_row_pro)
        v_s_0_lo_init = Vec.from_elements(s_lo_pro, fx.Float32).ir_value()
        v_s_0_hi_init = Vec.from_elements(s_hi_pro, fx.Float32).ir_value()
        #     asm volatile("" : "+v"(v_s[0]) ::);
        v_s_0_lo_init, v_s_0_hi_init = _anchor_pair(v_s_0_lo_init, v_s_0_hi_init)

        #     attn_exp2_slice<T, 0, s_half_len>(v_s[0]);
        s_lo_pro = [Vec(v_s_0_lo_init)[r] for r in range_constexpr(16)]
        s_hi_pro = [Vec(v_s_0_hi_init)[r] for r in range_constexpr(16)]
        lo_pro, hi_pro = attn_exp2_first_half_slice(s_lo_pro, s_hi_pro)
        v_s_0_lo_init = Vec.from_elements(lo_pro, fx.Float32).ir_value()
        v_s_0_hi_init = Vec.from_elements(hi_pro, fx.Float32).ir_value()

        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     __builtin_amdgcn_s_barrier();
        rocdl.s_barrier()
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     async_load<T::VEC_KV>(g_k, s_k[0].ptr, u_gk, u_sk, kv_tile(2));
        async_load_k(fx.Index(2 * BLOCK_N), 0)

        l_row_init = c_zero_f
        init_args = [m_row_pro, l_row_init]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)
        init_args.append(v_s_0_lo_init)
        init_args.append(v_s_0_hi_init)

        #     // Main loop
        #     for (int j = 3; j < max_num_tiles - 1; j += 2) {
        loop_results = init_args
        for j, loop_args in range(
            fx.Index(3),
            max_num_tiles - fx.Index(1),
            fx.Index(2),
            init=init_args,
        ):
            m_row = loop_args[0]
            l_row = loop_args[1]
            o_accs = [loop_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            v_s_0_lo_partial = loop_args[2 + D_CHUNKS]
            v_s_0_hi_partial = loop_args[3 + D_CHUNKS]

            j_idx = j

            #         // Cluster 0:
            #         async_load<T::VEC_KV>(g_v, s_v[1].ptr, u_gv, u_sv, kv_tile(j - 2));
            async_load_v((j_idx - fx.Index(2)) * fx.Index(BLOCK_N), 1)
            #         v_k = load<T::VEC_KV>(s_k[1], u_rk);
            k_pl_a, k_ph_a = async_load_k_from_lds_to_vgpr(1, urk_base_per_lane)
            #         s_waitcnt_lgkmcnt(0_I);
            rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
            #         s_waitcnt_vmcnt(number<T::k_buffer_load_insts + T::v_buffer_load_insts>{});
            _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)
            #         __builtin_amdgcn_s_barrier();
            rocdl.s_barrier()
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)

            #         // Cluster 1:
            #         v_s[1] = mma0(v_q, v_k);
            v_s_1_lo_raw, v_s_1_hi_raw = _gemm0(k_pl_a, k_ph_a)
            #         attn_exp2_slice<T, s_half_len, s_half_len>(v_s[0]);
            #         l_row += attn_sum<T>(v_s[0]);
            #         v_p = opus::cast<D_ATTN>(v_s[0]);
            #         asm volatile("" : "+v"(v_p) ::);
            v_s_0_hi_full = attn_exp2_second_half_slice(v_s_0_hi_partial)
            v_s_0_lo_list = [Vec(v_s_0_lo_partial)[r] for r in range_constexpr(16)]
            tile_sum_a = attn_sum_parts(v_s_0_lo_list, v_s_0_hi_full)
            l_row = _fadd(l_row, tile_sum_a)

            v_p_lo_a, v_p_hi_a = cast_p_parts(v_s_0_lo_list, v_s_0_hi_full)
            v_p_lo_a = _anchor_packs(v_p_lo_a)
            v_p_hi_a = _anchor_packs(v_p_hi_a)

            #         sched_barrier_exp_pairs<6, 3, 1>();
            _sched_barrier_exp_pairs(6, 3, 1)
            #         sched_barrier_pairs<10, 5, 1>();
            _sched_barrier_pairs(10, 5, 1)
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)
            #         __builtin_amdgcn_s_barrier();
            rocdl.s_barrier()
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)

            #         // Cluster 2:
            #         async_load<T::VEC_KV>(g_k, s_k[1].ptr, u_gk, u_sk, kv_tile(j));
            async_load_k(j_idx * fx.Index(BLOCK_N), 1)
            #         v_v = tr_load<T::VEC_TR_V>(s_v[0], u_rv);
            v_packs_a = _read_v_packs_for_buf(0, urv_base_per_lane)
            #         s_waitcnt_lgkmcnt(0_I);
            rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
            #         s_waitcnt_vmcnt(number<T::k_buffer_load_insts + T::v_buffer_load_insts>{});
            _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)
            #         __builtin_amdgcn_s_barrier();
            rocdl.s_barrier()
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)

            #         // Cluster 3:
            #         __builtin_amdgcn_s_setprio(1);
            if const_expr(OPUS_SETPRIO):
                rocdl.s_setprio(1)

            #         v_o = mma1.step_k(0_I, v_p, v_v, v_o);
            v_pk = v_packs_a[0]
            p_pk = v_p_lo_a[0]
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

            s_lo_a = [Vec(v_s_1_lo_raw)[r] for r in range_constexpr(16)]
            s_hi_a = [Vec(v_s_1_hi_raw)[r] for r in range_constexpr(16)]

            #         D_ACC row_max = attn_row_max<T>(v_s[1]);
            m_tile_max_a = attn_row_max(s_lo_a, s_hi_a)

            #         sched_barrier_pairs<4, 5, 2>();
            _sched_barrier_pairs(4, 5, 2)

            #         bool below_thresh = ((row_max - m_row) <= RESCALE_THRESHOLD);
            #         bool all_below = (__builtin_amdgcn_ballot_w64(below_thresh) == __builtin_amdgcn_read_exec());
            #         if (__builtin_expect(all_below, 1)) {
            #             row_max = m_row;
            #         } else {
            #             rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
            #             scale_output_tile<T>(v_o, rescale_m);
            #             l_row *= rescale_m;
            #             m_row = row_max;
            #         }
            m_diff_a = _fsub(m_tile_max_a, m_row)
            if const_expr(OPUS_LAZY_RESCALE):
                below_a = ArithValue(fx.Float32(m_diff_a) <= c_eight_f)
                ballot_a = rocdl.ballot(T.i64, _raw(below_a))
                all_below_a = fx.Int64(ballot_a) == fx.Int64(-1)
                ab_a = ArithValue(all_below_a)
                m_new_a = ab_a.select(m_row, _fmax(m_row, m_tile_max_a))
                corr_a = rocdl.exp2(T.f32, _raw(_fsub(m_row, m_new_a)))
                eff_corr_a = ab_a.select(c_one_f, corr_a)
            else:
                m_new_a = _fmax(m_row, m_tile_max_a)
                corr_a = rocdl.exp2(T.f32, _raw(_fsub(m_row, m_new_a)))
                eff_corr_a = corr_a

            _scale_o(o_accs, eff_corr_a)
            l_row = _fmul(l_row, corr_a)
            m_row = m_new_a

            #         v_o = mma1.step_k(1_I, v_p, v_v, v_o);
            #         v_o = mma1.step_k(2_I, v_p, v_v, v_o);
            #         v_o = mma1.step_k(3_I, v_p, v_v, v_o);
            for kss in range_constexpr(3):
                actual = kss + 1
                v_pk = v_packs_a[actual]
                if const_expr(actual < 2):
                    p_pk = v_p_lo_a[actual]
                else:
                    p_pk = v_p_hi_a[actual - 2]
                for dc in range_constexpr(D_CHUNKS):
                    o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

            #         attn_sub_row<T>(v_s[1], row_max);
            #         asm volatile("" : "+v"(v_s[1]) ::);
            #         attn_exp2_slice<T, 0, s_half_len>(v_s[1]);
            s_lo_a, s_hi_a = attn_sub_row(s_lo_a, s_hi_a, m_new_a)
            v_s_1_lo_partial = Vec.from_elements(s_lo_a, fx.Float32).ir_value()
            v_s_1_hi_partial = Vec.from_elements(s_hi_a, fx.Float32).ir_value()
            v_s_1_lo_partial, v_s_1_hi_partial = _anchor_pair(
                v_s_1_lo_partial, v_s_1_hi_partial
            )
            s_lo_a = [Vec(v_s_1_lo_partial)[r] for r in range_constexpr(16)]
            s_hi_a = [Vec(v_s_1_hi_partial)[r] for r in range_constexpr(16)]
            lo_part_a, hi_part_a = attn_exp2_first_half_slice(s_lo_a, s_hi_a)
            v_s_1_lo_partial = Vec.from_elements(lo_part_a, fx.Float32).ir_value()
            v_s_1_hi_partial = Vec.from_elements(hi_part_a, fx.Float32).ir_value()

            #         sched_barrier_pairs<6, 5, 2>();
            _sched_barrier_pairs(6, 5, 2)
            #         sched_barrier_exp_pairs<6, 3, 2>();
            _sched_barrier_exp_pairs(6, 3, 2)
            #         __builtin_amdgcn_s_setprio(0);
            if const_expr(OPUS_SETPRIO):
                rocdl.s_setprio(0)
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)
            #         __builtin_amdgcn_s_barrier();
            rocdl.s_barrier()
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)

            #         // Cluster 4:
            #         async_load<T::VEC_KV>(g_v, s_v[0].ptr, u_gv, u_sv, kv_tile(j - 1));
            async_load_v((j_idx - fx.Index(1)) * fx.Index(BLOCK_N), 0)
            #         v_k = load<T::VEC_KV>(s_k[0], u_rk);
            k_pl_b, k_ph_b = async_load_k_from_lds_to_vgpr(0, urk_base_per_lane)
            #         s_waitcnt_lgkmcnt(0_I);
            rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
            #         s_waitcnt_vmcnt(number<T::k_buffer_load_insts + T::v_buffer_load_insts>{});
            _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)
            #         __builtin_amdgcn_s_barrier();
            rocdl.s_barrier()
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)

            #         // Cluster 5:
            #         v_s[0] = mma0(v_q, v_k);
            v_s_0_lo_raw_b, v_s_0_hi_raw_b = _gemm0(k_pl_b, k_ph_b)
            #         attn_exp2_slice<T, s_half_len, s_half_len>(v_s[1]);
            #         l_row += attn_sum<T>(v_s[1]);
            #         v_p = opus::cast<D_ATTN>(v_s[1]);
            #         asm volatile("" : "+v"(v_p) ::);
            v_s_1_hi_full = attn_exp2_second_half_slice(v_s_1_hi_partial)
            v_s_1_lo_list = [Vec(v_s_1_lo_partial)[r] for r in range_constexpr(16)]
            tile_sum_b = attn_sum_parts(v_s_1_lo_list, v_s_1_hi_full)
            l_row = _fadd(l_row, tile_sum_b)

            v_p_lo_b, v_p_hi_b = cast_p_parts(v_s_1_lo_list, v_s_1_hi_full)
            v_p_lo_b = _anchor_packs(v_p_lo_b)
            v_p_hi_b = _anchor_packs(v_p_hi_b)

            #         sched_barrier_exp_pairs<6, 3, 3>();
            _sched_barrier_exp_pairs(6, 3, 3)
            #         sched_barrier_pairs<10, 5, 3>();
            _sched_barrier_pairs(10, 5, 3)
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)
            #         __builtin_amdgcn_s_barrier();
            rocdl.s_barrier()
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)

            #         // Cluster 6:
            #         async_load<T::VEC_KV>(g_k, s_k[0].ptr, u_gk, u_sk, kv_tile(j + 1));
            async_load_k((j_idx + fx.Index(1)) * fx.Index(BLOCK_N), 0)
            #         v_v = tr_load<T::VEC_TR_V>(s_v[1], u_rv);
            v_packs_b = _read_v_packs_for_buf(1, urv_base_per_lane)
            #         if constexpr (T::CAUSAL) {
            #             const int kv_end_pos = j * T::KV_TILE_SIZE;
            #             if (q_start_pos < kv_end_pos) {
            #                 attn_mask_causal_tile<T>(v_s[0], q_start_pos, j - 1, neg_inf_v, lane_id);
            #             }
            #         }
            s_lo_b = [Vec(v_s_0_lo_raw_b)[r] for r in range_constexpr(16)]
            s_hi_b = [Vec(v_s_0_hi_raw_b)[r] for r in range_constexpr(16)]
            if const_expr(CAUSAL):
                s_lo_b, s_hi_b = _causal_mask_prologue_if_needed(
                    s_lo_b,
                    s_hi_b,
                    j_idx - fx.Index(1),
                    j_idx * fx.Index(BLOCK_N),
                )
            #         s_waitcnt_lgkmcnt(0_I);
            rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
            #         s_waitcnt_vmcnt(number<T::k_buffer_load_insts + T::v_buffer_load_insts>{});
            _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)
            #         __builtin_amdgcn_s_barrier();
            rocdl.s_barrier()
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)

            #         // Cluster 7:
            #         __builtin_amdgcn_s_setprio(1);
            if const_expr(OPUS_SETPRIO):
                rocdl.s_setprio(1)

            #         v_o = mma1.step_k(0_I, v_p, v_v, v_o);
            v_pk = v_packs_b[0]
            p_pk = v_p_lo_b[0]
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

            #         row_max = attn_row_max<T>(v_s[0]);
            m_tile_max_b = attn_row_max(s_lo_b, s_hi_b)

            #         sched_barrier_pairs<4, 5, 4>();
            _sched_barrier_pairs(4, 5, 4)

            #         below_thresh = ((row_max - m_row) <= RESCALE_THRESHOLD);
            #         all_below = (__builtin_amdgcn_ballot_w64(below_thresh) == __builtin_amdgcn_read_exec());
            #         if (__builtin_expect(all_below, 1)) {
            #             row_max = m_row;
            #         } else {
            #             rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
            #             scale_output_tile<T>(v_o, rescale_m);
            #             l_row *= rescale_m;
            #             m_row = row_max;
            #         }
            m_diff_b = _fsub(m_tile_max_b, m_row)
            if const_expr(OPUS_LAZY_RESCALE):
                below_b = ArithValue(fx.Float32(m_diff_b) <= c_eight_f)
                ballot_b = rocdl.ballot(T.i64, _raw(below_b))
                all_below_b = fx.Int64(ballot_b) == fx.Int64(-1)
                ab_b = ArithValue(all_below_b)
                m_new_b = ab_b.select(m_row, _fmax(m_row, m_tile_max_b))
                corr_b = rocdl.exp2(T.f32, _raw(_fsub(m_row, m_new_b)))
                eff_corr_b = ab_b.select(c_one_f, corr_b)
            else:
                m_new_b = _fmax(m_row, m_tile_max_b)
                corr_b = rocdl.exp2(T.f32, _raw(_fsub(m_row, m_new_b)))
                eff_corr_b = corr_b

            _scale_o(o_accs, eff_corr_b)
            l_row = _fmul(l_row, corr_b)
            m_row = m_new_b

            #         v_o = mma1.step_k(1_I, v_p, v_v, v_o);
            #         v_o = mma1.step_k(2_I, v_p, v_v, v_o);
            #         v_o = mma1.step_k(3_I, v_p, v_v, v_o);
            for kss in range_constexpr(3):
                actual = kss + 1
                v_pk = v_packs_b[actual]
                if const_expr(actual < 2):
                    p_pk = v_p_lo_b[actual]
                else:
                    p_pk = v_p_hi_b[actual - 2]
                for dc in range_constexpr(D_CHUNKS):
                    o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

            #         attn_sub_row<T>(v_s[0], row_max);
            #         asm volatile("" : "+v"(v_s[0]) ::);
            #         attn_exp2_slice<T, 0, s_half_len>(v_s[0]);
            s_lo_b, s_hi_b = attn_sub_row(s_lo_b, s_hi_b, m_new_b)
            v_s_0_lo_yield = Vec.from_elements(s_lo_b, fx.Float32).ir_value()
            v_s_0_hi_yield = Vec.from_elements(s_hi_b, fx.Float32).ir_value()
            v_s_0_lo_yield, v_s_0_hi_yield = _anchor_pair(
                v_s_0_lo_yield, v_s_0_hi_yield
            )
            s_lo_b = [Vec(v_s_0_lo_yield)[r] for r in range_constexpr(16)]
            s_hi_b = [Vec(v_s_0_hi_yield)[r] for r in range_constexpr(16)]
            lo_part_b, hi_part_b = attn_exp2_first_half_slice(s_lo_b, s_hi_b)
            v_s_0_lo_yield = Vec.from_elements(lo_part_b, fx.Float32).ir_value()
            v_s_0_hi_yield = Vec.from_elements(hi_part_b, fx.Float32).ir_value()

            #         sched_barrier_pairs<6, 5, 4>();
            _sched_barrier_pairs(6, 5, 4)
            #         sched_barrier_exp_pairs<6, 3, 4>();
            _sched_barrier_exp_pairs(6, 3, 4)
            #         __builtin_amdgcn_s_setprio(0);
            if const_expr(OPUS_SETPRIO):
                rocdl.s_setprio(0)
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)
            #         __builtin_amdgcn_s_barrier();
            rocdl.s_barrier()
            #         __builtin_amdgcn_sched_barrier(0);
            rocdl.sched_barrier(0)

            yield_args = [m_row, l_row] + o_accs + [v_s_0_lo_yield, v_s_0_hi_yield]
            loop_results = yield yield_args

        #     // Epilogue
        m_row = loop_results[0]
        l_row = loop_results[1]
        o_accs = [loop_results[2 + i] for i in range_constexpr(D_CHUNKS)]
        v_s_0_lo_partial = loop_results[2 + D_CHUNKS]
        v_s_0_hi_partial = loop_results[3 + D_CHUNKS]

        max_m3 = max_num_tiles - fx.Index(3)
        max_m2 = max_num_tiles - fx.Index(2)
        max_m1 = max_num_tiles - fx.Index(1)

        #     // Cluster 0:
        #     async_load<T::VEC_KV>(g_v, s_v[1].ptr, u_gv, u_sv, kv_tile(max_num_tiles - 3));
        async_load_v(max_m3 * fx.Index(BLOCK_N), 1)
        #     v_k = load<T::VEC_KV>(s_k[1], u_rk);
        k_pl_e0, k_ph_e0 = async_load_k_from_lds_to_vgpr(1, urk_base_per_lane)
        #     s_waitcnt_lgkmcnt(0_I);
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        #     s_waitcnt_vmcnt(number<T::k_buffer_load_insts + T::v_buffer_load_insts>{});
        _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     __builtin_amdgcn_s_barrier();
        rocdl.s_barrier()
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     // Cluster 1:
        #     v_s[1] = mma0(v_q, v_k);
        v_s_1_lo_e, v_s_1_hi_e = _gemm0(k_pl_e0, k_ph_e0)
        #     attn_exp2_slice<T, s_half_len, s_half_len>(v_s[0]);
        #     v_p = opus::cast<D_ATTN>(v_s[0]);
        #     asm volatile("" : "+v"(v_p) ::);
        v_s_0_hi_full_e1 = attn_exp2_second_half_slice(v_s_0_hi_partial)
        v_s_0_lo_list_e1 = [Vec(v_s_0_lo_partial)[r] for r in range_constexpr(16)]
        tile_sum_e1 = attn_sum_parts(v_s_0_lo_list_e1, v_s_0_hi_full_e1)
        #     l_row += attn_sum<T>(v_s[0]);
        l_row = _fadd(l_row, tile_sum_e1)

        v_p_lo_e1, v_p_hi_e1 = cast_p_parts(v_s_0_lo_list_e1, v_s_0_hi_full_e1)
        v_p_lo_e1 = _anchor_packs(v_p_lo_e1)
        v_p_hi_e1 = _anchor_packs(v_p_hi_e1)

        #     sched_barrier_exp_pairs<6, 3, 5>();
        _sched_barrier_exp_pairs(6, 3, 5)
        #     sched_barrier_pairs<10, 5, 5>();
        _sched_barrier_pairs(10, 5, 5)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     __builtin_amdgcn_s_barrier();
        rocdl.s_barrier()
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     // Cluster 2:
        #     async_load<T::VEC_KV>(g_k, s_k[1].ptr, u_gk, u_sk, kv_tile(max_num_tiles - 1));
        async_load_k(max_m1 * fx.Index(BLOCK_N), 1)
        #     v_v = tr_load<T::VEC_TR_V>(s_v[0], u_rv);
        v_packs_e3 = _read_v_packs_for_buf(0, urv_base_per_lane)
        #     if constexpr (T::CAUSAL) {
        #         const int kv_end_pos = (max_num_tiles - 2) * T::KV_TILE_SIZE;
        #         if (q_start_pos < kv_end_pos) {
        #             attn_mask_causal_tile<T>(v_s[1], q_start_pos, max_num_tiles - 3, neg_inf_v, lane_id);
        #         }
        #     }
        s_lo_e1 = [Vec(v_s_1_lo_e)[r] for r in range_constexpr(16)]
        s_hi_e1 = [Vec(v_s_1_hi_e)[r] for r in range_constexpr(16)]
        if const_expr(CAUSAL):
            s_lo_e1, s_hi_e1 = _causal_mask_prologue_if_needed(
                s_lo_e1,
                s_hi_e1,
                max_m3,
                max_m2 * fx.Index(BLOCK_N),
            )
        #     s_waitcnt_lgkmcnt(0_I);
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        #     s_waitcnt_vmcnt(number<T::k_buffer_load_insts + T::v_buffer_load_insts>{});
        _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     __builtin_amdgcn_s_barrier();
        rocdl.s_barrier()
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     // Cluster 3:
        #     __builtin_amdgcn_s_setprio(1);
        if const_expr(OPUS_SETPRIO):
            rocdl.s_setprio(1)

        #     v_o = mma1(v_p, v_v, v_o);
        for kss in range_constexpr(4):
            v_pk = v_packs_e3[kss]
            if const_expr(kss < 2):
                p_pk = v_p_lo_e1[kss]
            else:
                p_pk = v_p_hi_e1[kss - 2]
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

        #     D_ACC row_max = max(m_row, attn_row_max<T>(v_s[1]));
        m_tile_max_e3 = attn_row_max(s_lo_e1, s_hi_e1)
        row_max_e3 = _fmax(m_row, m_tile_max_e3)
        #     rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
        rescale_e3 = rocdl.exp2(T.f32, _raw(_fsub(m_row, row_max_e3)))
        #     m_row = row_max;
        m_row = row_max_e3
        #     attn_sub_row<T>(v_s[1], row_max);
        #     asm volatile("" : "+v"(v_s[1]) ::);
        #     attn_exp2_slice<T, 0, s_half_len>(v_s[1]);
        s_lo_e1, s_hi_e1 = attn_sub_row(s_lo_e1, s_hi_e1, row_max_e3)
        v_s_1_lo_e_partial = Vec.from_elements(s_lo_e1, fx.Float32).ir_value()
        v_s_1_hi_e_partial = Vec.from_elements(s_hi_e1, fx.Float32).ir_value()
        v_s_1_lo_e_partial, v_s_1_hi_e_partial = _anchor_pair(
            v_s_1_lo_e_partial, v_s_1_hi_e_partial
        )
        s_lo_e1 = [Vec(v_s_1_lo_e_partial)[r] for r in range_constexpr(16)]
        s_hi_e1 = [Vec(v_s_1_hi_e_partial)[r] for r in range_constexpr(16)]
        lo_e3, hi_e3 = attn_exp2_first_half_slice(s_lo_e1, s_hi_e1)
        v_s_1_lo_e_partial = Vec.from_elements(lo_e3, fx.Float32).ir_value()
        v_s_1_hi_e_partial = Vec.from_elements(hi_e3, fx.Float32).ir_value()

        #     sched_barrier_pairs<10, 5, 6>();
        _sched_barrier_pairs(10, 5, 6)
        #     sched_barrier_exp_pairs<6, 3, 6>();
        _sched_barrier_exp_pairs(6, 3, 6)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     scale_output_tile<T>(v_o, rescale_m);
        #     auto* v_o_pin = reinterpret_cast<vector_t<fp32_t, 16>*>(&v_o);
        #     asm volatile("" : "+v"(v_o_pin[0]), "+v"(v_o_pin[1]), "+v"(v_o_pin[2]), "+v"(v_o_pin[3]) ::);
        _scale_o(o_accs, rescale_e3)

        #     __builtin_amdgcn_s_setprio(0);
        if const_expr(OPUS_SETPRIO):
            rocdl.s_setprio(0)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     __builtin_amdgcn_s_barrier();
        rocdl.s_barrier()
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     // Cluster 4:
        #     async_load<T::VEC_KV>(g_v, s_v[0].ptr, u_gv, u_sv, kv_tile(max_num_tiles - 2));
        async_load_v(max_m2 * fx.Index(BLOCK_N), 0)
        #     v_k = load<T::VEC_KV>(s_k[0], u_rk);
        k_pl_e4, k_ph_e4 = async_load_k_from_lds_to_vgpr(0, urk_base_per_lane)
        #     s_waitcnt_lgkmcnt(0_I);
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        #     s_waitcnt_vmcnt(number<T::k_buffer_load_insts + T::v_buffer_load_insts>{});
        _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     __builtin_amdgcn_s_barrier();
        rocdl.s_barrier()
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     // Cluster 5:
        #     v_s[0] = mma0(v_q, v_k);
        v_s_0_lo_e5, v_s_0_hi_e5 = _gemm0(k_pl_e4, k_ph_e4)
        #     l_row *= rescale_m;
        l_row = _fmul(l_row, rescale_e3)
        #     attn_exp2_slice<T, s_half_len, s_half_len>(v_s[1]);
        #     v_p = opus::cast<D_ATTN>(v_s[1]);
        #     asm volatile("" : "+v"(v_p) ::);
        v_s_1_hi_full_e5 = attn_exp2_second_half_slice(v_s_1_hi_e_partial)
        v_s_1_lo_list_e5 = [Vec(v_s_1_lo_e_partial)[r] for r in range_constexpr(16)]
        tile_sum_e5 = attn_sum_parts(v_s_1_lo_list_e5, v_s_1_hi_full_e5)
        #     l_row += attn_sum<T>(v_s[1]);
        l_row = _fadd(l_row, tile_sum_e5)

        v_p_lo_e5, v_p_hi_e5 = cast_p_parts(v_s_1_lo_list_e5, v_s_1_hi_full_e5)
        v_p_lo_e5 = _anchor_packs(v_p_lo_e5)
        v_p_hi_e5 = _anchor_packs(v_p_hi_e5)

        #     sched_barrier_exp_pairs<6, 3, 7>();
        _sched_barrier_exp_pairs(6, 3, 7)
        #     sched_barrier_pairs<10, 5, 7>();
        _sched_barrier_pairs(10, 5, 7)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     __builtin_amdgcn_s_barrier();
        rocdl.s_barrier()
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     // Cluster 6:
        #     v_v = tr_load<T::VEC_TR_V>(s_v[1], u_rv);
        v_packs_e7 = _read_v_packs_for_buf(1, urv_base_per_lane)
        #     if constexpr (T::CAUSAL) {
        #         const int kv_end_pos = (max_num_tiles - 1) * T::KV_TILE_SIZE;
        #         if (q_start_pos < kv_end_pos) {
        #             attn_mask_causal_tile<T>(v_s[0], q_start_pos, max_num_tiles - 2, neg_inf_v, lane_id);
        #         }
        #     }
        s_lo_e5 = [Vec(v_s_0_lo_e5)[r] for r in range_constexpr(16)]
        s_hi_e5 = [Vec(v_s_0_hi_e5)[r] for r in range_constexpr(16)]
        if const_expr(CAUSAL):
            s_lo_e5, s_hi_e5 = _causal_mask_prologue_if_needed(
                s_lo_e5,
                s_hi_e5,
                max_m2,
                max_m1 * fx.Index(BLOCK_N),
            )
        #     s_waitcnt_lgkmcnt(0_I);
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        #     s_waitcnt_vmcnt(number<T::v_buffer_load_insts>{});
        _waitcnt_vm_n(NUM_DMA_V)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     __builtin_amdgcn_s_barrier();
        rocdl.s_barrier()
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     // Cluster 7:
        #     __builtin_amdgcn_s_setprio(1);
        if const_expr(OPUS_SETPRIO):
            rocdl.s_setprio(1)

        #     v_o = mma1(v_p, v_v, v_o);
        for kss in range_constexpr(4):
            v_pk = v_packs_e7[kss]
            if const_expr(kss < 2):
                p_pk = v_p_lo_e5[kss]
            else:
                p_pk = v_p_hi_e5[kss - 2]
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

        #     row_max = max(m_row, attn_row_max<T>(v_s[0]));
        m_tile_max_e7 = attn_row_max(s_lo_e5, s_hi_e5)
        row_max_e7 = _fmax(m_row, m_tile_max_e7)
        #     rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
        rescale_e7 = rocdl.exp2(T.f32, _raw(_fsub(m_row, row_max_e7)))
        #     m_row = row_max;
        m_row = row_max_e7
        #     attn_sub_row<T>(v_s[0], row_max);
        #     asm volatile("" : "+v"(v_s[0]) ::);
        #     attn_exp2_slice<T, 0, s_half_len>(v_s[0]);
        s_lo_e5, s_hi_e5 = attn_sub_row(s_lo_e5, s_hi_e5, row_max_e7)
        v_s_0_lo_e_partial = Vec.from_elements(s_lo_e5, fx.Float32).ir_value()
        v_s_0_hi_e_partial = Vec.from_elements(s_hi_e5, fx.Float32).ir_value()
        v_s_0_lo_e_partial, v_s_0_hi_e_partial = _anchor_pair(
            v_s_0_lo_e_partial, v_s_0_hi_e_partial
        )
        s_lo_e5 = [Vec(v_s_0_lo_e_partial)[r] for r in range_constexpr(16)]
        s_hi_e5 = [Vec(v_s_0_hi_e_partial)[r] for r in range_constexpr(16)]
        lo_e7, hi_e7 = attn_exp2_first_half_slice(s_lo_e5, s_hi_e5)
        v_s_0_lo_e_partial = Vec.from_elements(lo_e7, fx.Float32).ir_value()
        v_s_0_hi_e_partial = Vec.from_elements(hi_e7, fx.Float32).ir_value()

        #     sched_barrier_pairs<10, 5, 8>();
        _sched_barrier_pairs(10, 5, 8)
        #     sched_barrier_exp_pairs<6, 3, 8>();
        _sched_barrier_exp_pairs(6, 3, 8)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     scale_output_tile<T>(v_o, rescale_m);
        #     asm volatile("" : "+v"(v_o_pin[0]), "+v"(v_o_pin[1]), "+v"(v_o_pin[2]), "+v"(v_o_pin[3]) ::);
        _scale_o(o_accs, rescale_e7)

        #     __builtin_amdgcn_s_setprio(0);
        if const_expr(OPUS_SETPRIO):
            rocdl.s_setprio(0)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     __builtin_amdgcn_s_barrier();
        rocdl.s_barrier()
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     // Cluster 8:
        #     async_load<T::VEC_KV>(g_v, s_v[1].ptr, u_gv, u_sv, kv_tile(max_num_tiles - 1));
        async_load_v(max_m1 * fx.Index(BLOCK_N), 1)
        #     v_k = load<T::VEC_KV>(s_k[1], u_rk);
        k_pl_e8, k_ph_e8 = async_load_k_from_lds_to_vgpr(1, urk_base_per_lane)
        #     s_waitcnt_lgkmcnt(0_I);
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        #     s_waitcnt_vmcnt(number<T::v_buffer_load_insts>{});
        _waitcnt_vm_n(NUM_DMA_V)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     __builtin_amdgcn_s_barrier();
        rocdl.s_barrier()
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     // Cluster 9:
        #     v_s[1] = mma0(v_q, v_k);
        v_s_1_lo_e9, v_s_1_hi_e9 = _gemm0(k_pl_e8, k_ph_e8)
        #     l_row *= rescale_m;
        l_row = _fmul(l_row, rescale_e7)
        #     attn_exp2_slice<T, s_half_len, s_half_len>(v_s[0]);
        #     v_p = opus::cast<D_ATTN>(v_s[0]);
        #     asm volatile("" : "+v"(v_p) ::);
        v_s_0_hi_full_e9 = attn_exp2_second_half_slice(v_s_0_hi_e_partial)
        v_s_0_lo_list_e9 = [Vec(v_s_0_lo_e_partial)[r] for r in range_constexpr(16)]
        tile_sum_e9 = attn_sum_parts(v_s_0_lo_list_e9, v_s_0_hi_full_e9)
        #     l_row += attn_sum<T>(v_s[0]);
        l_row = _fadd(l_row, tile_sum_e9)

        v_p_lo_e9, v_p_hi_e9 = cast_p_parts(v_s_0_lo_list_e9, v_s_0_hi_full_e9)
        v_p_lo_e9 = _anchor_packs(v_p_lo_e9)
        v_p_hi_e9 = _anchor_packs(v_p_hi_e9)

        #     sched_barrier_exp_pairs<6, 3, 9>();
        _sched_barrier_exp_pairs(6, 3, 9)
        #     sched_barrier_pairs<10, 5, 9>();
        _sched_barrier_pairs(10, 5, 9)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     __builtin_amdgcn_s_barrier();
        rocdl.s_barrier()
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     // Cluster 10:
        #     v_v = tr_load<T::VEC_TR_V>(s_v[0], u_rv);
        v_packs_e11 = _read_v_packs_for_buf(0, urv_base_per_lane)
        #     if constexpr (T::CAUSAL) {
        #         const int kv_end_pos = max_num_tiles * T::KV_TILE_SIZE;
        #         if (q_start_pos < kv_end_pos) {
        #             attn_mask_causal_tile<T>(v_s[1], q_start_pos, max_num_tiles - 1, neg_inf_v, lane_id);
        #         }
        #     }
        s_lo_e9 = [Vec(v_s_1_lo_e9)[r] for r in range_constexpr(16)]
        s_hi_e9 = [Vec(v_s_1_hi_e9)[r] for r in range_constexpr(16)]
        if const_expr(CAUSAL):
            s_lo_e9, s_hi_e9 = _causal_mask_prologue_if_needed(
                s_lo_e9,
                s_hi_e9,
                max_m1,
                max_num_tiles * fx.Index(BLOCK_N),
            )
        #     s_waitcnt_lgkmcnt(0_I);
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        #     s_waitcnt_vmcnt(0_I);
        _waitcnt_vm_n(0)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     __builtin_amdgcn_s_barrier();
        rocdl.s_barrier()
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     // Cluster 11:
        #     v_o = mma1(v_p, v_v, v_o);
        for kss in range_constexpr(4):
            v_pk = v_packs_e11[kss]
            if const_expr(kss < 2):
                p_pk = v_p_lo_e9[kss]
            else:
                p_pk = v_p_hi_e9[kss - 2]
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

        #     row_max = max(m_row, attn_row_max<T>(v_s[1]));
        m_tile_max_e11 = attn_row_max(s_lo_e9, s_hi_e9)
        row_max_e11 = _fmax(m_row, m_tile_max_e11)
        #     rescale_m = __builtin_amdgcn_exp2f(m_row - row_max);
        rescale_e11 = rocdl.exp2(T.f32, _raw(_fsub(m_row, row_max_e11)))
        #     m_row = row_max;
        m_row = row_max_e11

        #     attn_sub_row<T>(v_s[1], row_max);
        #     asm volatile("" : "+v"(v_s[1]) ::);
        #     attn_exp2_slice<T, 0, s_half_len>(v_s[1]);
        s_lo_e9, s_hi_e9 = attn_sub_row(s_lo_e9, s_hi_e9, row_max_e11)
        v_s_1_lo_e11 = Vec.from_elements(s_lo_e9, fx.Float32).ir_value()
        v_s_1_hi_e11 = Vec.from_elements(s_hi_e9, fx.Float32).ir_value()
        v_s_1_lo_e11, v_s_1_hi_e11 = _anchor_pair(
            v_s_1_lo_e11, v_s_1_hi_e11
        )
        s_lo_e9 = [Vec(v_s_1_lo_e11)[r] for r in range_constexpr(16)]
        s_hi_e9 = [Vec(v_s_1_hi_e11)[r] for r in range_constexpr(16)]
        lo_e11, hi_e11 = attn_exp2_first_half_slice(s_lo_e9, s_hi_e9)

        #     sched_barrier_pairs<10, 5, 10>();
        _sched_barrier_pairs(10, 5, 10)
        #     sched_barrier_exp_pairs<6, 3, 10>();
        _sched_barrier_exp_pairs(6, 3, 10)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     attn_exp2_slice<T, s_half_len, s_half_len>(v_s[1]);
        hi_e11_full = []
        for r in range_constexpr(16):
            hi_e11_full.append(rocdl.exp2(T.f32, _raw(hi_e11[r])))
        #     l_row *= rescale_m;
        l_row = _fmul(l_row, rescale_e11)
        tile_sum_e11 = attn_sum_parts(lo_e11, hi_e11_full)
        #     l_row += attn_sum<T>(v_s[1]);
        l_row = _fadd(l_row, tile_sum_e11)

        #     v_p = opus::cast<D_ATTN>(v_s[1]);
        #     asm volatile("" : "+v"(v_p) ::);
        v_p_lo_e11, v_p_hi_e11 = cast_p_parts(lo_e11, hi_e11_full)
        v_p_lo_e11 = _anchor_packs(v_p_lo_e11)
        v_p_hi_e11 = _anchor_packs(v_p_hi_e11)

        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     scale_output_tile<T>(v_o, rescale_m);
        #     asm volatile("" : "+v"(v_o_pin[0]), "+v"(v_o_pin[1]), "+v"(v_o_pin[2]), "+v"(v_o_pin[3]) ::);
        _scale_o(o_accs, rescale_e11)
        #     __builtin_amdgcn_s_barrier();
        rocdl.s_barrier()
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     // Cluster 12:
        #     v_v = tr_load<T::VEC_TR_V>(s_v[1], u_rv);
        v_packs_e13 = _read_v_packs_for_buf(1, urv_base_per_lane)
        #     s_waitcnt_lgkmcnt(0_I);
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)
        #     __builtin_amdgcn_s_barrier();
        rocdl.s_barrier()
        #     __builtin_amdgcn_sched_barrier(0);
        rocdl.sched_barrier(0)

        #     // Cluster 13:
        #     v_o = mma1(v_p, v_v, v_o);
        for kss in range_constexpr(4):
            v_pk = v_packs_e13[kss]
            if const_expr(kss < 2):
                p_pk = v_p_lo_e11[kss]
            else:
                p_pk = v_p_hi_e11[kss - 2]
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

        #     // ──── Normalize O and store to gmem ────
        #     D_ACC l_inv = (l_row > D_ACC(0.0f)) ? (D_ACC(1.0f) / l_row) : D_ACC(0.0f);
        #     static_for<o_len>([&](auto i) { v_o[i.value] *= l_inv; });
        inv_l = rocdl.rcp(T.f32, _raw(l_row))
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(16)

        #     if (!stagger) {
        #         __builtin_amdgcn_s_barrier();
        #     }
        if const_expr(OPUS_ENABLE_STAGGER):
            _stagger_extra_barrier_if_zero()
        else:
            rocdl.s_barrier()

        #     auto u_o = make_layout_o<T>(warp_id, lane_id, kargs.stride_q_n);
        #     auto v_o_bf16 = opus::cast<D_ATTN>(v_o);
        #     store<T::VEC_O>(g_o, v_o_bf16, u_o);
        if q_in_bounds:
            for dc in range_constexpr(D_CHUNKS):
                o_norm_vec = Vec(o_accs[dc]) * inv_l_vec
                for store_group in range_constexpr(4):
                    r_base = store_group * 4
                    lo = rocdl.cvt_pk_bf16_f32(
                        Vec(o_norm_vec)[r_base],
                        Vec(o_norm_vec)[r_base + 1],
                    )
                    hi = rocdl.cvt_pk_bf16_f32(
                        Vec(o_norm_vec)[r_base + 2],
                        Vec(o_norm_vec)[r_base + 3],
                    )
                    o_pack = Vec.from_elements([lo, hi], fx.Int32).ir_value()
                    d_row_rel = lane_div_32 * 4 + store_group * 8
                    d_col = fx.Index(dc * D_CHUNK) + d_row_rel
                    o_global = global_idx_q(q_row, d_col)
                    o_byte_offset = fx.Int32(o_global * fx.Index(BF16_BYTES))
                    rocdl.raw_buffer_store(
                        o_pack,
                        o_rsrc,
                        _llvm_value(o_byte_offset),
                        _llvm_value(fx.Int32(0)),
                        _llvm_value(fx.Int32(0)),
                    )

    @flyc.jit
    def launch_flash_attn_opus(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_blocks = (sl_idx + BLOCK_M - 1) // BLOCK_M

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else None
        )
        flash_attn_opus_kernel(
            Q, K, V, O, seq_len, stride_q_n, stride_kv_n, head_dim_runtime,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": f"{BLOCK_SIZE},{BLOCK_SIZE}",
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(NUM_HEADS_Q, num_q_blocks, bs_idx),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _opus_compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    def _launch(
        Q, K, V, O, batch_size, seq_len, stride_kv_n=None, stride_q_n=None,
        head_dim_runtime=None, *, stream=None
    ):
        if stride_kv_n is None:
            stride_kv_n = DEFAULT_STRIDE_KV_N
        if stride_q_n is None:
            stride_q_n = DEFAULT_STRIDE_Q_N
        if head_dim_runtime is None:
            head_dim_runtime = HEAD_DIM
        with CompilationContext.compile_hints(_opus_compile_hints):
            if stream is None:
                return launch_flash_attn_opus(
                    Q, K, V, O, batch_size, seq_len,
                    stride_q_n, stride_kv_n, head_dim_runtime
                )
            return launch_flash_attn_opus(
                Q, K, V, O, batch_size, seq_len,
                stride_q_n, stride_kv_n, head_dim_runtime, stream=stream
            )

    def _compile(
        Q, K, V, O, batch_size, seq_len, stride_kv_n=None, stride_q_n=None,
        head_dim_runtime=None, *, stream=None
    ):
        if stride_kv_n is None:
            stride_kv_n = DEFAULT_STRIDE_KV_N
        if stride_q_n is None:
            stride_q_n = DEFAULT_STRIDE_Q_N
        if head_dim_runtime is None:
            head_dim_runtime = HEAD_DIM
        with CompilationContext.compile_hints(_opus_compile_hints):
            return flyc.compile(
                launch_flash_attn_opus, Q, K, V, O, batch_size, seq_len,
                stride_q_n, stride_kv_n, head_dim_runtime,
                fx.Stream(stream))

    _launch.compile = _compile

    return _launch
