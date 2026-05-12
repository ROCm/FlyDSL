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
from flydsl._mlir.dialects import llvm, scf
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


def _llvm_value(value):
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


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

    Launcher signature: ``launcher(Q, K, V, O, batch_size, seq_len, *, stream=None)``
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

    SM_SCALE = 1.0 / host_math.sqrt(head_dim)
    NUM_HEADS_Q = num_heads
    NUM_HEADS_KV = num_kv_heads
    GQA_GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_KV
    CAUSAL = causal
    STRIDE_TOKEN_Q = NUM_HEADS_Q * HEAD_DIM
    STRIDE_TOKEN_KV = NUM_HEADS_KV * HEAD_DIM

    # K/V LDS double-buffered, XOR-swizzled (16B = 8 bf16 swizzle granularity).
    K_STRIDE = HEAD_DIM
    V_STRIDE = HEAD_DIM
    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    NUM_PREFETCH_K = 2     # OPUS double-buffer
    NUM_PREFETCH_V = 2
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

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
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE * 2   # bf16 = 2 bytes

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
    ):
        elem_dtype = dtype_to_elem_type(dtype_str)
        elem_type = elem_dtype.ir_type
        compute_type = fx.Float32.ir_type
        q_ptr = _extract_aligned_pointer(Q)
        k_ptr = _extract_aligned_pointer(K)
        v_ptr = _extract_aligned_pointer(V)
        o_ptr = _extract_aligned_pointer(O)

        fm_fast = fx.arith.FastMathFlags.fast
        v4f16_type = Vec.make_type(4, elem_dtype)
        v8f16_type = Vec.make_type(8, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)
        mfma_pack_type = v8f16_type

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

        # ── P3: sched_group_barrier discipline (matches C++ template lines 14-30) ──
        # __builtin_amdgcn_sched_group_barrier(mask, count, group_id)
        # masks (LLVM AMDGPU convention):
        _MFMA_MASK = 0x008
        _VALU_MASK = 0x002
        _EXP_MASK = 0x400

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

        seq_len_v = fx.Index(seq_len)

        # LDS view
        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(
            base_ptr,
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()
        lds_kv_base_idx = buffer_ops.extract_base_index(lds_kv, address_space=3)

        # ── 3D grid block indices (OPUS layout) ──
        h_idx = fx.Index(gpu.block_idx.x)
        q_block_idx = fx.Index(gpu.block_idx.y)
        batch_idx = fx.Index(gpu.block_idx.z)
        tid = fx.Index(gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32

        # P5 stagger: warps 0-3 → stagger=0, warps 4-7 → stagger=1.
        # Matches C++ template gqa_d128_kernel_template.hpp lines 306-308:
        #     const int warp_id = __builtin_amdgcn_readfirstlane(thread_id_x() / WARP_SIZE);
        #     const int stagger = warp_id / 4;
        # readfirstlane forces the value into an SGPR so the resulting
        # conditional `if (stagger ...) { s_barrier }` is a SCALAR
        # (wave-uniform) branch — that is essential for the conditional
        # s_barrier to be skipped per-wave rather than masked per-lane.
        #
        # The V LDS reads have been hoisted out of Clusters 3/7/11/13 into
        # the immediately preceding cluster (2/6/10/12) so V[*] is held in
        # registers BEFORE the cluster boundary barrier. This mirrors the
        # C++ template's `tr_load V into v_v` before the s_barrier and is
        # what makes the asymmetric stagger barrier safe: even when warps
        # 4-7 are one phase behind warps 0-3, the lagging group has already
        # captured V into VGPRs in its own preceding cluster, so a peer
        # wave overwriting `s_v[buf]` via a subsequent async_load is
        # harmless.
        _tid_i32 = arith.index_cast(T.i32, _raw(tid))
        _wave_id_uni_i32 = rocdl.readfirstlane(
            T.i32,
            arith.divsi(_tid_i32, arith.constant(WARP_SIZE, type=T.i32)),
        )
        _stagger_i32 = arith.divsi(
            _wave_id_uni_i32, arith.constant(4, type=T.i32)
        )
        stagger_is_one_i1 = arith.cmpi(
            arith.CmpIPredicate.ne, _stagger_i32, arith.constant(0, type=T.i32)
        )
        stagger_is_zero_i1 = arith.cmpi(
            arith.CmpIPredicate.eq, _stagger_i32, arith.constant(0, type=T.i32)
        )

        # HW transpose decomp for ds_read_tr16_b64 (gfx950)
        tr_k_group = (lane % 16) // 4
        tr_col_sub = lane % 4
        tr_col_half = (lane % 32) // 16

        # ds_read_tr_v4f16 helper
        def ds_read_tr_v4f16(lds_elem_idx):
            byte_offset = lds_elem_idx * 2 + lds_kv_offset
            byte_i64 = fx.Int64(byte_offset)
            ptr = buffer_ops.create_llvm_ptr(byte_i64, address_space=3)
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

        # ── Wave / tile bookkeeping ──
        wave_q_offset = wave_id * ROWS_PER_WAVE
        q_block_size = BLOCK_M
        q_start = q_block_idx * q_block_size

        # GQA mapping mirrors OPUS lines 310-312:
        # h = (h_idx % H_KV) * group_size + (h_idx / H_KV)
        # h_kv = h / group_size = h_idx % H_KV
        h_kv_idx = h_idx % NUM_HEADS_KV
        group_id = h_idx // NUM_HEADS_KV
        q_head_idx = h_kv_idx * GQA_GROUP_SIZE + group_id
        kv_head_idx = h_kv_idx

        def global_idx_q(token_idx, col):
            token = batch_idx * seq_len_v + token_idx
            return token * STRIDE_TOKEN_Q + q_head_idx * HEAD_DIM + col

        def _load_global_half_vec(ptr, base_idx, vec_elems):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            return _pointer_load(Vec.make_type(vec_elems, elem_dtype), gep)

        def _store_global_half(ptr, base_idx, val):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=elem_type)
            _pointer_store(val, gep)

        def load_global_mfma_pack(rsrc, base_idx):
            return _load_global_half_vec(rsrc, base_idx, MFMA_LANE_K)

        def _bitcast_i32(value):
            return fx.Int32(ArithValue(value).bitcast(fx.Int32.ir_type))

        def _bitcast_f32(value):
            return fx.Float32(ArithValue(value).bitcast(fx.Float32.ir_type))

        # ── P4: opus::attn_mask_vec2_imm inline asm (matches C++ lines 233-249) ──
        # We split the C++ 4-op block into two 2-op pairs because MLIR's
        # llvm.inline_asm with struct return + multiple "=s" outputs has
        # proven brittle. The resulting ISA is the same {cmp_lt + cndmask}
        # pattern that OPUS depends on.
        def _attn_mask_imm_single(rel_i32, neg_inf_i32, thr, x_ref_i32):
            """Single asm: x_new = (rel < thr) ? neg_inf : x_ref.

            asm:
              v_cmp_lt_i32_e64 sgpr_mask, rel, thr
              v_cndmask_b32_e64 vdst, x_ref, neg_inf, sgpr_mask
            """
            asm_str = (
                f"v_cmp_lt_i32_e64 $1, $2, {int(thr)}\n\t"
                "v_cndmask_b32_e64 $0, $3, $4, $1"
            )
            # $0 = new_x (=v), $1 = mask (=s, early-clobber), $2 = rel, $3 = x_ref, $4 = neg_inf
            constraints = "=v,=&s,v,v,v"
            ret_struct_ty = ir.Type.parse("!llvm.struct<(i32, i64)>")
            ret = llvm.inline_asm(
                ret_struct_ty,
                [_llvm_value(rel_i32), _llvm_value(x_ref_i32),
                 _llvm_value(neg_inf_i32)],
                asm_str,
                constraints,
                has_side_effects=False,
            )
            return llvm.extractvalue(T.i32, ret, [0])

        def _attn_mask_vec2_imm(rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32):
            new_x = _attn_mask_imm_single(rel_i32, neg_inf_i32, thr_x, x_ref_i32)
            new_y = _attn_mask_imm_single(rel_i32, neg_inf_i32, thr_y, y_ref_i32)
            return new_x, new_y

        # ── P4: register anchor (matches C++ `asm volatile("" : "+v"(v) ::)`) ──
        # Forces the scheduler to treat `value` as both used and defined at this
        # point, preventing it from being reordered across the anchor. We use a
        # tied "=v,0" constraint so the input and output occupy the same VGPR(s);
        # has_side_effects=True keeps the asm anchored through DCE.
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
            return _anchor_vec(lo), _anchor_vec(hi)

        def _anchor_packs(packs):
            return [_anchor_vec(p) for p in packs]

        def _stagger_extra_barrier_if_one():
            """Emit `sched_barrier(0); s_barrier;` only when stagger == 1.

            Matches C++ template gqa_d128_kernel_template.hpp lines 415-418.
            The body runs on warps 4-7 only, advancing their s_barrier ordinal
            by one relative to warps 0-3 → starts the dual-group phase shift.

            Implemented via inline assembly with the stagger value forced into
            an SGPR via the `s` constraint, so the conditional branch is a
            scalar `s_cbranch_scc*` (per-wave) rather than a per-lane VGPR
            predicate. This mirrors how the C++ builtin compiles.
            """
            rocdl.sched_barrier(0)
            llvm.inline_asm(
                ir.Type.parse("!llvm.void"),
                [_stagger_i32],
                (
                    "s_cmp_eq_u32 $0, 0\n\t"
                    "s_cbranch_scc1 1f\n\t"
                    "s_barrier\n\t"
                    "1:"
                ),
                "s",
                has_side_effects=True,
            )

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

        def _pack_bf16_pair(lo, hi, shift, mask):
            lo_i32 = _bitcast_i32(lo)
            hi_i32 = _bitcast_i32(hi)
            return (hi_i32 & mask) | lo_i32.shrui(shift)

        def bf16_trunc_pack_v8(f32_vals):
            _c16 = fx.Int32(16)
            _cmask = fx.Int32(0xFFFF0000)
            pairs = []
            for j in range_constexpr(4):
                pairs.append(_pack_bf16_pair(f32_vals[j * 2], f32_vals[j * 2 + 1], _c16, _cmask))
            return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()

        def k_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(buf_id * LDS_K_TILE_SIZE)
            return buf_id * fx.Index(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)
            return fx.Index(LDS_V_BASE) + buf_id * fx.Index(LDS_V_TILE_SIZE)

        # ── DMA loaders (buffer_load_dwordx4_lds, gfx950) ──
        k_rsrc = buffer_ops.create_buffer_resource(K, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(V, max_size=True)
        DMA_BYTES = 16
        DMA_BATCH_BYTES = BLOCK_SIZE * DMA_BYTES
        K_TILE_BYTES = BLOCK_N * K_STRIDE * 2
        NUM_DMA_K = K_TILE_BYTES // DMA_BATCH_BYTES
        LANES_PER_K_ROW = HEAD_DIM * 2 // DMA_BYTES
        ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (HEAD_DIM * 2)
        V_TILE_BYTES = BLOCK_N * V_STRIDE * 2
        NUM_DMA_V = V_TILE_BYTES // DMA_BATCH_BYTES
        LANES_PER_V_ROW = HEAD_DIM * 2 // DMA_BYTES

        _dma_size = fx.Int32(DMA_BYTES)
        _dma_soff = fx.Int32(0)
        _dma_off = fx.Int32(0)
        _dma_aux = fx.Int32(1)

        def coop_dma_k(tile_start, buf_id):
            k_lds_byte_base = lds_kv_base_idx + k_buf_base(buf_id) * fx.Index(2)
            for d in range_constexpr(NUM_DMA_K):
                lds_addr = (
                    k_lds_byte_base
                    + wave_id * fx.Index(WARP_SIZE * DMA_BYTES)
                    + fx.Index(d * DMA_BATCH_BYTES)
                )
                lds_i64 = fx.Int64(lds_addr)
                lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, lds_i64)
                lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)

                row_in_tile = tid // LANES_PER_K_ROW + fx.Index(d * ROWS_PER_DMA_BATCH)
                swiz_col_f16 = (tid % LANES_PER_K_ROW) * (DMA_BYTES // 2)
                xor_mask = (row_in_tile & fx.Index(0x7)) << fx.Index(4)
                unsw_col_f16 = swiz_col_f16 ^ xor_mask
                col_byte = unsw_col_f16 * 2
                global_row = batch_idx * seq_len_v + tile_start + row_in_tile
                global_byte = (
                    global_row * fx.Index(STRIDE_TOKEN_KV * 2)
                    + kv_head_idx * fx.Index(HEAD_DIM * 2)
                    + col_byte
                )
                voffset = fx.Int32(global_byte)
                rocdl.raw_ptr_buffer_load_lds(
                    k_rsrc, lds_ptr, _dma_size, voffset, _dma_soff, _dma_off, _dma_aux
                )

        def coop_dma_v(tile_start, buf_id):
            v_lds_byte_base = lds_kv_base_idx + v_buf_base(buf_id) * fx.Index(2)
            for d in range_constexpr(NUM_DMA_V):
                lds_addr = (
                    v_lds_byte_base
                    + wave_id * fx.Index(WARP_SIZE * DMA_BYTES)
                    + fx.Index(d * DMA_BATCH_BYTES)
                )
                lds_i64 = fx.Int64(lds_addr)
                lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, lds_i64)
                lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)

                row_in_tile = tid // LANES_PER_V_ROW + fx.Index(d * (DMA_BATCH_BYTES // (HEAD_DIM * 2)))
                swiz_col_f16 = (tid % LANES_PER_V_ROW) * (DMA_BYTES // 2)
                xor_mask = (row_in_tile & fx.Index(0x3)) << fx.Index(4)
                unsw_col_f16 = swiz_col_f16 ^ xor_mask
                col_byte = unsw_col_f16 * 2
                global_row = batch_idx * seq_len_v + tile_start + row_in_tile
                global_byte = (
                    global_row * fx.Index(STRIDE_TOKEN_KV * 2)
                    + kv_head_idx * fx.Index(HEAD_DIM * 2)
                    + col_byte
                )
                voffset = fx.Int32(global_byte)
                rocdl.raw_ptr_buffer_load_lds(
                    v_rsrc, lds_ptr, _dma_size, voffset, _dma_soff, _dma_off, _dma_aux
                )

        # ── Constants ──
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_sm_scale_log2e = fx.Float32(SM_SCALE * _LOG2E)
        c_eight_f = fx.Float32(OPUS_RESCALE_THRESHOLD)
        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_32_i32 = fx.Int32(32)
        c4_i32 = fx.Int32(4)
        lane_i32 = fx.Int32(lane)
        v8f32_type = Vec.make_type(MFMA_LANE_K, fx.Float32)

        # ── Q preload (B-operand for MFMA, register-resident) ──
        # P2: pre-multiply Q by temperature_scale = (1/sqrt(D)) * log2(e)
        # so that softmax can operate directly in log2 space without per-FMA
        # multiplications. Matches OPUS C++ lines 404-406 (kernel template).
        q_row = q_start + wave_q_offset + lane_mod_32
        q_row_i32 = fx.Int32(q_row)
        q_in_bounds = q_row < seq_len_v
        q_row_safe = fx.Index(ArithValue(q_in_bounds).select(q_row, fx.Index(0)))
        c_zero_mfma_pack = Vec.filled(MFMA_LANE_K, 0.0, elem_dtype).ir_value()
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            g_idx = global_idx_q(q_row_safe, q_col)
            raw = load_global_mfma_pack(q_ptr, g_idx)
            raw_safe = ArithValue(q_in_bounds).select(raw, c_zero_mfma_pack)
            # bf16x8 → f32x8 → multiply by c_sm_scale_log2e → bf16x8
            pack_f32 = Vec(raw_safe).extf(v8f32_type)
            scaled_elems = []
            for k in range_constexpr(MFMA_LANE_K):
                scaled_elems.append(
                    _fmul(Vec(pack_f32)[k], c_sm_scale_log2e)
                )
            pack_scaled_f32 = Vec.from_elements(scaled_elems, fx.Float32)
            pack_scaled_bf16 = pack_scaled_f32.truncf(v8f16_type)
            q_b_packs.append(pack_scaled_bf16.ir_value())

        # Use shuffle_xor by 32 for the cross-half reduction (== permlane32_swap+max in OPUS).
        def reduction_peer(v_f32):
            return fx.Float32(v_f32).shuffle_xor(shuf_32_i32, width_i32)

        # ──────────────────────────────────────────────────────────────────────
        # P1: Restructure to match opus_attn/gqa_d128_kernel_template.hpp
        # exactly at the cluster level. Each labelled section maps to the
        # corresponding C++ source range.
        # ──────────────────────────────────────────────────────────────────────

        # ── Compute max_num_tiles (matches C++ lines 383-390) ──
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

        # ── Helper: K LDS → register packs (8 lo + 8 hi) ──
        def _read_k_packs_for_buf(buf_id):
            """Read all 16 K MFMA packs from LDS buffer `buf_id`."""
            k_base = k_buf_base(buf_id)
            k_hi_offset = K_SUB_N * K_STRIDE
            k_swz_mask = (lane_mod_32 & fx.Index(0x7)) << fx.Index(4)
            k_lo = [None] * K_STEPS_QK
            k_hi = [None] * K_STEPS_QK
            for ks in range_constexpr(K_STEPS_QK):
                col = fx.Index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                idx_lo = (
                    k_base + lane_mod_32 * fx.Index(K_STRIDE) + (col ^ k_swz_mask)
                )
                idx_hi = (
                    k_base
                    + fx.Index(k_hi_offset)
                    + lane_mod_32 * fx.Index(K_STRIDE)
                    + (col ^ k_swz_mask)
                )
                k_lo[ks] = Vec.load(mfma_pack_type, lds_kv, [idx_lo])
                k_hi[ks] = Vec.load(mfma_pack_type, lds_kv, [idx_hi])
            return k_lo, k_hi

        # ── Helper: V LDS → register packs for ONE K-substep (4 packs / D-chunk) ──
        # k_substep ∈ {0, 1, 2, 3}, mapping to MFMA W_K=16 sub-step of GEMM2:
        #   0 → lo N-strip, pks=0, KV cols 0..15
        #   1 → lo N-strip, pks=1, KV cols 16..31
        #   2 → hi N-strip, pks=0, KV cols 32..47
        #   3 → hi N-strip, pks=1, KV cols 48..63
        # Mirrors C++ mma1.step_k(K_idx)'s V operand requirement.
        def _read_v_packs_for_k_substep(buf_id, k_substep):
            v_base = v_buf_base(buf_id)
            is_hi = k_substep // 2
            pks = k_substep % 2
            packs = []
            for dc in range_constexpr(D_CHUNKS):
                d_col = (
                    fx.Index(dc * D_CHUNK)
                    + tr_col_half * fx.Index(16)
                    + tr_col_sub * fx.Index(4)
                )
                if const_expr(is_hi == 0):
                    k_row = (
                        fx.Index(pks * PV_K_STEP)
                        + lane_div_32 * fx.Index(4)
                        + tr_k_group
                    )
                else:
                    k_row = (
                        fx.Index(pks * PV_K_STEP + K_SUB_N)
                        + lane_div_32 * fx.Index(4)
                        + tr_k_group
                    )
                v_xor_mask = (k_row & fx.Index(0x3)) << fx.Index(4)
                d_col_eff = d_col ^ v_xor_mask
                lds = v_base + k_row * fx.Index(V_STRIDE) + d_col_eff
                a = ds_read_tr_v4f16(lds)
                b = ds_read_tr_v4f16(lds + fx.Index(8 * V_STRIDE))
                packs.append(
                    Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                )
            return packs

        # ── Helper: GEMM0 (S = Q @ K^T) – 16 MFMAs ──
        def _gemm0(k_lo, k_hi):
            v_s_lo = c_zero_v16f32
            v_s_hi = c_zero_v16f32
            for ks in range_constexpr(K_STEPS_QK):
                v_s_lo = mfma_acc(k_lo[ks], q_b_packs[ks], v_s_lo)
                v_s_hi = mfma_acc(k_hi[ks], q_b_packs[ks], v_s_hi)
            return v_s_lo, v_s_hi

        # ── Helper: causal mask in-place on extracted s_lo[16] / s_hi[16] ──
        # P4: -inf as i32 bitpattern for the cndmask source.
        _NEG_INF_F32_BITS = 0xFF800000

        def _causal_mask_inplace(s_lo, s_hi, tile_idx):
            """Apply causal mask using OPUS inline-asm attn_mask_vec2_imm.

            Mirrors gqa_d128_kernel_template.hpp::attn_mask_causal_tile
            (lines 251-289) with GEMM0_E_N=2 (BLOCK_N=64 → s_lo + s_hi).

            Each MFMA C-output lane holds elements at:
              kv_col_in_strip = lane_div_32*4 + (r//4)*8 + (r%4)
            giving threshold pairs {(0,1),(2,3),(8,9),(10,11),(16,17),
            (18,19),(24,25),(26,27)} within each W_N=32 strip.

            For s_hi the strip shifts by W_N=32, so the per-lane absolute K
            column is (kv_start + 32 + lane_div_32*4 + offset). The mask
            condition `q_pos < absolute_K_col` is rewritten as
            `rel < thr` where rel = q_pos - (kv_start + i_n*W_N + lane_group*c_pack)
            and thr is the per-element offset (immediate).
            """
            kv_tile_start = tile_idx * fx.Index(BLOCK_N)
            kv_start_i32 = fx.Int32(kv_tile_start)
            lane_off_i32 = fx.Int32(lane_div_32) * c4_i32
            # rel = q_pos - (kv_start + i_n*W_N + lane_group*c_pack)
            rel_lo_i32 = fx.Int32(q_row_i32 - kv_start_i32 - lane_off_i32)
            rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(K_SUB_N))
            neg_inf_i32 = fx.Int32(_NEG_INF_F32_BITS)

            # 8 (thr_x, thr_y) pairs, matching C++ static_for nest:
            #   i_rept in [0,4), i_pair in [0, c_pack/2) = [0, 2)
            #   thr_x = i_rept * c_rept_stride + i_pair * 2
            #   thr_y = thr_x + 1
            #   c_rept_stride = (WARP_SIZE / W_M) * c_pack = 8
            pair_thresholds = [
                (0, 1), (2, 3),
                (8, 9), (10, 11),
                (16, 17), (18, 19),
                (24, 25), (26, 27),
            ]
            for p in range_constexpr(len(pair_thresholds)):
                thr_x, thr_y = pair_thresholds[p]
                idx_x = p * 2
                idx_y = p * 2 + 1

                # s_lo pair (i_n = 0)
                x_lo_bits = _raw(_bitcast_i32(s_lo[idx_x]))
                y_lo_bits = _raw(_bitcast_i32(s_lo[idx_y]))
                new_x_lo, new_y_lo = _attn_mask_vec2_imm(
                    rel_lo_i32, neg_inf_i32, thr_x, thr_y,
                    x_lo_bits, y_lo_bits,
                )
                s_lo[idx_x] = _raw(_bitcast_f32(new_x_lo))
                s_lo[idx_y] = _raw(_bitcast_f32(new_y_lo))

                # s_hi pair (i_n = 1, rel shifted by W_N)
                x_hi_bits = _raw(_bitcast_i32(s_hi[idx_x]))
                y_hi_bits = _raw(_bitcast_i32(s_hi[idx_y]))
                new_x_hi, new_y_hi = _attn_mask_vec2_imm(
                    rel_hi_i32, neg_inf_i32, thr_x, thr_y,
                    x_hi_bits, y_hi_bits,
                )
                s_hi[idx_x] = _raw(_bitcast_f32(new_x_hi))
                s_hi[idx_y] = _raw(_bitcast_f32(new_y_hi))

        # ── Helper: wave-wide row_max over a 32-element extended vector ──
        def _wave_row_max(s_lo, s_hi):
            m = c_neg_inf
            for r in range_constexpr(16):
                m = _fmax(m, s_lo[r])
                m = _fmax(m, s_hi[r])
            return _fmax(m, reduction_peer(m))

        # ── Helper: sub_row + first-half exp2 (matches C++ attn_sub_row + exp2_slice<0,16>) ──
        # P2: Q has been pre-multiplied by sm_scale*log2e in the prologue, so s_lo, s_hi
        # and m_row are already in log2 space. The FMA collapses to a plain subtract.
        # Returns:
        #   lo_partial[16] = exp2(s_lo - m_row)
        #   hi_partial[16] = s_hi - m_row              (NOT yet exp'd)
        def _sub_row_first_half_exp(s_lo, s_hi, m_row):
            lo_partial = []
            hi_partial = []
            for r in range_constexpr(16):
                diff_lo = _fsub(s_lo[r], m_row)
                lo_partial.append(ArithValue(diff_lo).exp2(fastmath=fm_fast))
            for r in range_constexpr(16):
                diff_hi = _fsub(s_hi[r], m_row)
                hi_partial.append(diff_hi)
            return lo_partial, hi_partial

        # ── Helper: finish second-half exp + sum + cast → P bf16 packs ──
        # Matches C++ cluster 1 / cluster 5 body:
        #   attn_exp2_slice<T, s_half_len, s_half_len>(v_s[?]);
        #   l_row += attn_sum<T>(v_s[?]);
        #   v_p = cast<bf16>(v_s[?]);
        def _finish_softmax_cast_p(lo_partial_list, hi_partial_vec):
            hi_full = []
            for r in range_constexpr(16):
                hi_full.append(
                    ArithValue(Vec(hi_partial_vec)[r]).exp2(fastmath=fm_fast)
                )
            local_sum = c_zero_f
            for r in range_constexpr(16):
                local_sum = _fadd(local_sum, lo_partial_list[r])
            for r in range_constexpr(16):
                local_sum = _fadd(local_sum, hi_full[r])
            peer_sum = reduction_peer(local_sum)
            tile_sum = _fadd(local_sum, peer_sum)
            p_lo_packs = []
            p_hi_packs = []
            for pks in range_constexpr(PV_K_STEPS):
                p_base = pks * 8
                lo_slice = [lo_partial_list[p_base + s] for s in range_constexpr(8)]
                p_lo_packs.append(bf16_trunc_pack_v8(lo_slice))
                hi_slice = hi_full[p_base:p_base + 8]
                p_hi_packs.append(bf16_trunc_pack_v8(hi_slice))
            return p_lo_packs, p_hi_packs, tile_sum

        # ── Helper: scale o_accs by a scalar f32 (broadcast 16-wide vec mul) ──
        def _scale_o(o_accs, scale_scalar):
            scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = _fmul(Vec(o_accs[dc]), scale_vec)

        # s_waitcnt encoding helpers
        _LGKMCNT_0_ONLY = 0xC07F   # vmcnt=63, expcnt=7, lgkmcnt=0, vmcnt_hi=3

        def _waitcnt_lgkm_0_vm_n(n):
            """Combined: lgkmcnt(0) + vmcnt(n)."""
            val = (
                (n & _VMCNT_LO_MASK)
                | (7 << 4)
                | (0 << 8)
                | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
            )
            rocdl.s_waitcnt(val)

        # Causal-mask q_start_pos (C++ line 394). Each warp computes its first Q row
        # position so that causal_mask_tile can decide whether to apply.
        # For P1 we apply causal mask unconditionally inside the const_expr branch;
        # the runtime gate is deferred to later phases.

        # ─────────────────────── PROLOGUE (C++ lines 397-436) ───────────────────────

        # [P1] async_load K[0] → s_k[0]   (C++ line 398)
        coop_dma_k(fx.Index(0), 0)

        # [P2] s_waitcnt(0); sched_barrier(0); s_barrier   (C++ lines 399-401)
        rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)
        gpu.barrier()

        # [P3] Q is already preloaded above (q_b_packs).
        # P2 (in-flight Q scaling): Q packs were pre-multiplied by
        # (sm_scale * log2e) during the load loop above, matching the C++
        # template (gqa_d128_kernel_template.hpp lines 404-406). The softmax
        # path now operates directly in log2 space (no per-FMA scale).

        # [P4] async_load K[1] → s_k[1], V[0] → s_v[0]   (C++ lines 408-409)
        coop_dma_k(fx.Index(BLOCK_N), 1)
        coop_dma_v(fx.Index(0), 0)

        # [P5] v_k = load(s_k[0], u_rk)   (C++ line 410)
        k_pl_pro, k_ph_pro = _read_k_packs_for_buf(0)

        # [P6] sched_barrier; s_waitcnt_lgkmcnt(0); s_waitcnt_vmcnt(v_buffer_load_insts)
        #     (C++ lines 411-413)
        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(NUM_DMA_V)   # vmcnt(2)

        # [P7] stagger barrier — extra s_barrier for warps 4-7 only.
        #       Mirrors C++ lines 415-418:
        #           if (stagger) {
        #               __builtin_amdgcn_sched_barrier(0);
        #               __builtin_amdgcn_s_barrier();
        #           }
        # The asymmetric (per-wave-group) version is gated by
        # OPUS_ENABLE_STAGGER (default ON). The path is correctness-safe
        # thanks to V LDS reads being hoisted one cluster earlier (see
        # Clusters 2/6/10/12). With the flag OFF we fall back to an
        # unconditional `gpu.barrier()` so both wave groups stay in
        # lockstep.
        if const_expr(OPUS_ENABLE_STAGGER):
            _stagger_extra_barrier_if_one()
        else:
            rocdl.sched_barrier(0)
            gpu.barrier()

        # [P8] v_s[0] = mma0(v_q, v_k)   (C++ line 420)
        v_s_0_lo_raw, v_s_0_hi_raw = _gemm0(k_pl_pro, k_ph_pro)
        rocdl.sched_barrier(0)

        # [P9] Causal mask for tile 0 (C++ lines 422-427)
        s_lo_pro = [Vec(v_s_0_lo_raw)[r] for r in range_constexpr(16)]
        s_hi_pro = [Vec(v_s_0_hi_raw)[r] for r in range_constexpr(16)]
        if const_expr(CAUSAL):
            _causal_mask_inplace(s_lo_pro, s_hi_pro, fx.Index(0))

        # [P10] m_row = attn_row_max<T>(v_s[0])   (C++ line 428)
        m_row_pro = _wave_row_max(s_lo_pro, s_hi_pro)

        # [P11] attn_sub_row(v_s[0], m_row); first-half exp2(v_s[0])  (C++ lines 429-431)
        lo_pro, hi_pro = _sub_row_first_half_exp(s_lo_pro, s_hi_pro, m_row_pro)
        v_s_0_lo_init = Vec.from_elements(lo_pro, fx.Float32).ir_value()
        v_s_0_hi_init = Vec.from_elements(hi_pro, fx.Float32).ir_value()
        # P4 anchor #1: matches C++ line 430 `asm volatile("" : "+v"(v_s[0]) ::);`
        v_s_0_lo_init, v_s_0_hi_init = _anchor_pair(v_s_0_lo_init, v_s_0_hi_init)

        # [P12] sched_barrier(0); s_barrier; sched_barrier(0)   (C++ lines 432-434)
        rocdl.sched_barrier(0)
        gpu.barrier()
        rocdl.sched_barrier(0)

        # [P13] async_load K[2] → s_k[0]   (C++ line 436)
        coop_dma_k(fx.Index(2 * BLOCK_N), 0)

        # ─────────────────── MAIN LOOP (C++ lines 439-560) ──────────────────────────
        # for j = 3; j < max_num_tiles - 1; j += 2
        # Each iter processes 2 KV tiles in 8 clusters.
        #
        # Loop state: [m_row, l_row, o_accs[0..3], v_s_0_lo_partial, v_s_0_hi_partial]
        l_row_init = c_zero_f
        init_args = [m_row_pro, l_row_init]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)
        init_args.append(v_s_0_lo_init)
        init_args.append(v_s_0_hi_init)

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

            j_idx = j  # fx.Index value in {3, 5, 7, ...}

            # ─── Cluster 0 (C++ lines 441-447) ───
            # V[j-2] async + K[j-2] ds_read + wait + barrier
            coop_dma_v((j_idx - fx.Index(2)) * fx.Index(BLOCK_N), 1)
            k_pl_a, k_ph_a = _read_k_packs_for_buf(1)
            rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
            _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
            rocdl.sched_barrier(0)
            gpu.barrier()
            rocdl.sched_barrier(0)

            # ─── Cluster 1 (C++ lines 449-459) ───
            # v_s[1] = mma0(v_q, K[j-2]); finish exp v_s[0]; sum; cast v_p = P[j-3]
            v_s_1_lo_raw, v_s_1_hi_raw = _gemm0(k_pl_a, k_ph_a)
            v_p_lo_a, v_p_hi_a, tile_sum_a = _finish_softmax_cast_p(
                [Vec(v_s_0_lo_partial)[r] for r in range_constexpr(16)],
                v_s_0_hi_partial,
            )
            l_row = _fadd(l_row, tile_sum_a)
            # P4 anchor #2: matches C++ line 454 `asm volatile("" : "+v"(v_p) ::);`
            v_p_lo_a = _anchor_packs(v_p_lo_a)
            v_p_hi_a = _anchor_packs(v_p_hi_a)
            # P3 schedule hints (C++ lines 455-456): tell scheduler about
            # the 16-MFMA mma0 + interleaved softmax pipeline.
            _sched_barrier_exp_pairs(6, 3, 1)
            _sched_barrier_pairs(10, 5, 1)
            rocdl.sched_barrier(0)
            gpu.barrier()
            rocdl.sched_barrier(0)

            # ─── Cluster 2 (C++ lines 461-468) ───
            # K[j] async + tr_load V[j-3] (4 substeps) from s_v[0] into registers + wait.
            # Mirrors C++ kernel line 466 `v_v = tr_load<...>(s_v[0], u_rv);` which
            # happens BEFORE the cluster-2 s_barrier. Hoisting the V reads out of
            # Cluster 3 is required for the P5 stagger path to be correct: with
            # the phase-shifted execution, warps 4-7 would otherwise read
            # `s_v[0]` AFTER warps 0-3 have already issued the Cluster-4
            # async_load that overwrites it. With the reads here, V[j-3] is
            # held in VGPRs across the barrier, so the next async_load into
            # `s_v[0]` is harmless.
            coop_dma_k(j_idx * fx.Index(BLOCK_N), 1)
            v_packs_a = []
            for kss in range_constexpr(4):
                v_packs_a.append(_read_v_packs_for_k_substep(0, kss))
            rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
            _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
            rocdl.sched_barrier(0)
            gpu.barrier()
            rocdl.sched_barrier(0)

            # ─── Cluster 3 (C++ lines 470-496) ───
            # GEMM2 P[j-3] @ V[j-3] via step_k(0..3) + lazy rescale + softmax v_s[1] head
            if const_expr(OPUS_SETPRIO):
                rocdl.s_setprio(1)

            # step_k(0): 4 MFMAs (one per D-chunk), KV cols 0-15 of V[j-3] (from s_v[0])
            v_pk = v_packs_a[0]
            p_pk = v_p_lo_a[0]
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

            # row_max + lazy rescale on v_s[1] (= S[j-2])
            s_lo_a = [Vec(v_s_1_lo_raw)[r] for r in range_constexpr(16)]
            s_hi_a = [Vec(v_s_1_hi_raw)[r] for r in range_constexpr(16)]
            # NOTE: main loop's v_s[1] (S[j-2]) is "deep below diagonal" — no causal mask needed.
            m_tile_max_a = _wave_row_max(s_lo_a, s_hi_a)
            # P3 schedule hint (C++ line 474): 4 MFMA + VALU pairs around
            # the lazy-rescale row_max computation, between step_k(0) and
            # the subsequent step_k(1..3).
            _sched_barrier_pairs(4, 5, 2)
            # P2: m_row and m_tile_max_a are already log2-scaled (Q was pre-scaled).
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

            # step_k(1..3): 12 more MFMAs (V packs were pre-loaded in Cluster 2)
            for kss in range_constexpr(3):
                actual = kss + 1
                v_pk = v_packs_a[actual]
                if const_expr(actual < 2):
                    p_pk = v_p_lo_a[actual]
                else:
                    p_pk = v_p_hi_a[actual - 2]
                for dc in range_constexpr(D_CHUNKS):
                    o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

            # sub_row + first-half exp on v_s[1]
            lo_part_a, hi_part_a = _sub_row_first_half_exp(s_lo_a, s_hi_a, m_new_a)
            v_s_1_lo_partial = Vec.from_elements(lo_part_a, fx.Float32).ir_value()
            v_s_1_hi_partial = Vec.from_elements(hi_part_a, fx.Float32).ir_value()
            # P4 anchor #3: matches C++ line 489 `asm volatile("" : "+v"(v_s[1]) ::);`
            v_s_1_lo_partial, v_s_1_hi_partial = _anchor_pair(
                v_s_1_lo_partial, v_s_1_hi_partial
            )

            # P3 schedule hints (C++ lines 491-492): finish GEMM2 + softmax head.
            _sched_barrier_pairs(6, 5, 2)
            _sched_barrier_exp_pairs(6, 3, 2)
            if const_expr(OPUS_SETPRIO):
                rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            gpu.barrier()
            rocdl.sched_barrier(0)

            # ─── Cluster 4 (C++ lines 498-505) ───
            # V[j-1] async + K[j-1] ds_read + wait
            coop_dma_v((j_idx - fx.Index(1)) * fx.Index(BLOCK_N), 0)
            k_pl_b, k_ph_b = _read_k_packs_for_buf(0)
            rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
            _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
            rocdl.sched_barrier(0)
            gpu.barrier()
            rocdl.sched_barrier(0)

            # ─── Cluster 5 (C++ lines 507-517) ───
            # v_s[0] = mma0(v_q, K[j-1]); finish exp v_s[1]; sum; cast v_p = P[j-2]
            v_s_0_lo_raw_b, v_s_0_hi_raw_b = _gemm0(k_pl_b, k_ph_b)
            v_p_lo_b, v_p_hi_b, tile_sum_b = _finish_softmax_cast_p(
                [Vec(v_s_1_lo_partial)[r] for r in range_constexpr(16)],
                v_s_1_hi_partial,
            )
            l_row = _fadd(l_row, tile_sum_b)
            # P4 anchor #4: matches C++ line 512 `asm volatile("" : "+v"(v_p) ::);`
            v_p_lo_b = _anchor_packs(v_p_lo_b)
            v_p_hi_b = _anchor_packs(v_p_hi_b)
            # P3 schedule hints (C++ lines 513-514).
            _sched_barrier_exp_pairs(6, 3, 3)
            _sched_barrier_pairs(10, 5, 3)
            rocdl.sched_barrier(0)
            gpu.barrier()
            rocdl.sched_barrier(0)

            # ─── Cluster 6 (C++ lines 519-532) ───
            # K[j+1] async + tr_load V[j-2] (4 substeps) from s_v[1] into registers
            # + causal mask on v_s[0] = S[j-1] + wait.
            # Hoisted V reads mirror C++ line 522 `v_v = tr_load<...>(s_v[1], u_rv);`
            # before the cluster-6 s_barrier. Required for P5 stagger correctness:
            # without the hoist warps 4-7 would read `s_v[1]` after warps 0-3
            # have already issued the next iteration's Cluster-0 async_load
            # into the same buffer.
            coop_dma_k((j_idx + fx.Index(1)) * fx.Index(BLOCK_N), 0)
            v_packs_b = []
            for kss in range_constexpr(4):
                v_packs_b.append(_read_v_packs_for_k_substep(1, kss))
            s_lo_b = [Vec(v_s_0_lo_raw_b)[r] for r in range_constexpr(16)]
            s_hi_b = [Vec(v_s_0_hi_raw_b)[r] for r in range_constexpr(16)]
            if const_expr(CAUSAL):
                _causal_mask_inplace(s_lo_b, s_hi_b, j_idx - fx.Index(1))
            rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
            _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
            rocdl.sched_barrier(0)
            gpu.barrier()
            rocdl.sched_barrier(0)

            # ─── Cluster 7 (C++ lines 534-560) ───
            # GEMM2 P[j-2] @ V[j-2] via step_k(0..3) + lazy rescale + softmax v_s[0] head
            if const_expr(OPUS_SETPRIO):
                rocdl.s_setprio(1)

            # step_k(0): V packs were pre-loaded in Cluster 6
            v_pk = v_packs_b[0]
            p_pk = v_p_lo_b[0]
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

            # row_max + lazy rescale on v_s[0] (= S[j-1], already masked)
            m_tile_max_b = _wave_row_max(s_lo_b, s_hi_b)
            # P3 schedule hint (C++ line 538).
            _sched_barrier_pairs(4, 5, 4)
            # P2: m_row and m_tile_max_b are already log2-scaled (Q was pre-scaled).
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

            # step_k(1..3): V packs were pre-loaded in Cluster 6
            for kss in range_constexpr(3):
                actual = kss + 1
                v_pk = v_packs_b[actual]
                if const_expr(actual < 2):
                    p_pk = v_p_lo_b[actual]
                else:
                    p_pk = v_p_hi_b[actual - 2]
                for dc in range_constexpr(D_CHUNKS):
                    o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

            # sub_row + first-half exp on v_s[0]
            lo_part_b, hi_part_b = _sub_row_first_half_exp(s_lo_b, s_hi_b, m_new_b)
            v_s_0_lo_yield = Vec.from_elements(lo_part_b, fx.Float32).ir_value()
            v_s_0_hi_yield = Vec.from_elements(hi_part_b, fx.Float32).ir_value()
            # P4 anchor #5: matches C++ line 553 `asm volatile("" : "+v"(v_s[0]) ::);`
            v_s_0_lo_yield, v_s_0_hi_yield = _anchor_pair(
                v_s_0_lo_yield, v_s_0_hi_yield
            )

            # P3 schedule hints (C++ lines 555-556).
            _sched_barrier_pairs(6, 5, 4)
            _sched_barrier_exp_pairs(6, 3, 4)
            if const_expr(OPUS_SETPRIO):
                rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            gpu.barrier()
            rocdl.sched_barrier(0)

            yield_args = [m_row, l_row] + o_accs + [v_s_0_lo_yield, v_s_0_hi_yield]
            loop_results = yield yield_args

        # ─────────────────── EPILOGUE (C++ lines 564-742) ───────────────────────────
        # Drains the last 3 KV tiles + completes the partial tile carried from the loop.
        # 13 clusters total.
        m_row = loop_results[0]
        l_row = loop_results[1]
        o_accs = [loop_results[2 + i] for i in range_constexpr(D_CHUNKS)]
        v_s_0_lo_partial = loop_results[2 + D_CHUNKS]
        v_s_0_hi_partial = loop_results[3 + D_CHUNKS]

        max_m3 = max_num_tiles - fx.Index(3)
        max_m2 = max_num_tiles - fx.Index(2)
        max_m1 = max_num_tiles - fx.Index(1)

        # ─── Cluster 0 (epi, C++ lines 565-571) ───
        # V[max-3] async + K[max-3] ds_read + wait
        coop_dma_v(max_m3 * fx.Index(BLOCK_N), 1)
        k_pl_e0, k_ph_e0 = _read_k_packs_for_buf(1)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
        rocdl.sched_barrier(0)
        gpu.barrier()
        rocdl.sched_barrier(0)

        # ─── Cluster 1 (epi, C++ lines 573-583) ───
        # v_s[1] = mma0(v_q, K[max-3]); finish exp v_s[0]; sum; cast v_p = P[max-4]
        v_s_1_lo_e, v_s_1_hi_e = _gemm0(k_pl_e0, k_ph_e0)
        v_p_lo_e1, v_p_hi_e1, tile_sum_e1 = _finish_softmax_cast_p(
            [Vec(v_s_0_lo_partial)[r] for r in range_constexpr(16)],
            v_s_0_hi_partial,
        )
        l_row = _fadd(l_row, tile_sum_e1)
        # P4 anchor #6: matches C++ line 578 `asm volatile("" : "+v"(v_p) ::);`
        v_p_lo_e1 = _anchor_packs(v_p_lo_e1)
        v_p_hi_e1 = _anchor_packs(v_p_hi_e1)
        # P3 schedule hints (C++ lines 579-580).
        _sched_barrier_exp_pairs(6, 3, 5)
        _sched_barrier_pairs(10, 5, 5)
        rocdl.sched_barrier(0)
        gpu.barrier()
        rocdl.sched_barrier(0)

        # ─── Cluster 2 (epi, C++ lines 585-598) ───
        # K[max-1] async + tr_load V[max-4] (4 substeps) from s_v[0] into registers
        # + causal mask on v_s[1] (= S[max-3]) + wait.
        # Hoisted V reads mirror C++ epilogue line 588 `v_v = tr_load<...>(s_v[0], u_rv);`
        # before the cluster-2 s_barrier. Required for P5 stagger correctness:
        # the symmetric epilogue Cluster-4 async_load overwrites `s_v[0]`,
        # so V[max-4] must be held in registers across the cluster boundary.
        coop_dma_k(max_m1 * fx.Index(BLOCK_N), 1)
        v_packs_e3 = []
        for kss in range_constexpr(4):
            v_packs_e3.append(_read_v_packs_for_k_substep(0, kss))
        s_lo_e1 = [Vec(v_s_1_lo_e)[r] for r in range_constexpr(16)]
        s_hi_e1 = [Vec(v_s_1_hi_e)[r] for r in range_constexpr(16)]
        if const_expr(CAUSAL):
            _causal_mask_inplace(s_lo_e1, s_hi_e1, max_m3)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
        rocdl.sched_barrier(0)
        gpu.barrier()
        rocdl.sched_barrier(0)

        # ─── Cluster 3 (epi, C++ lines 600-618) ───
        # FULL GEMM2 (no lazy) + softmax start v_s[1] + scale o
        if const_expr(OPUS_SETPRIO):
            rocdl.s_setprio(1)

        # mma1 (full 16 MFMAs): GEMM2 with v_p = P[max-4] and V[max-4] (from s_v[0])
        # V packs were pre-loaded in Cluster 2 (epi).
        for kss in range_constexpr(4):
            v_pk = v_packs_e3[kss]
            if const_expr(kss < 2):
                p_pk = v_p_lo_e1[kss]
            else:
                p_pk = v_p_hi_e1[kss - 2]
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

        # row_max + rescale_m + sub_row + first-half exp on v_s[1]
        # P2: m_row and m_tile_max_e3 are already log2-scaled (Q pre-scaled).
        m_tile_max_e3 = _wave_row_max(s_lo_e1, s_hi_e1)
        row_max_e3 = _fmax(m_row, m_tile_max_e3)
        rescale_e3 = rocdl.exp2(T.f32, _raw(_fsub(m_row, row_max_e3)))
        m_row = row_max_e3
        lo_e3, hi_e3 = _sub_row_first_half_exp(s_lo_e1, s_hi_e1, row_max_e3)
        v_s_1_lo_e_partial = Vec.from_elements(lo_e3, fx.Float32).ir_value()
        v_s_1_hi_e_partial = Vec.from_elements(hi_e3, fx.Float32).ir_value()
        # P4 anchor #7: matches C++ line 607 `asm volatile("" : "+v"(v_s[1]) ::);`
        v_s_1_lo_e_partial, v_s_1_hi_e_partial = _anchor_pair(
            v_s_1_lo_e_partial, v_s_1_hi_e_partial
        )

        # P3 schedule hints (C++ lines 609-610).
        _sched_barrier_pairs(10, 5, 6)
        _sched_barrier_exp_pairs(6, 3, 6)
        rocdl.sched_barrier(0)
        _scale_o(o_accs, rescale_e3)

        if const_expr(OPUS_SETPRIO):
            rocdl.s_setprio(0)
        rocdl.sched_barrier(0)
        gpu.barrier()
        rocdl.sched_barrier(0)

        # ─── Cluster 4 (epi, C++ lines 620-627) ───
        # V[max-2] async + K[max-2] ds_read + wait
        coop_dma_v(max_m2 * fx.Index(BLOCK_N), 0)
        k_pl_e4, k_ph_e4 = _read_k_packs_for_buf(0)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(NUM_DMA_K + NUM_DMA_V)
        rocdl.sched_barrier(0)
        gpu.barrier()
        rocdl.sched_barrier(0)

        # ─── Cluster 5 (epi, C++ lines 629-640) ───
        # v_s[0] = mma0(v_q, K[max-2]); l_row *= rescale_e3; finish exp v_s[1]; sum; cast P[max-3]
        v_s_0_lo_e5, v_s_0_hi_e5 = _gemm0(k_pl_e4, k_ph_e4)
        l_row = _fmul(l_row, rescale_e3)
        v_p_lo_e5, v_p_hi_e5, tile_sum_e5 = _finish_softmax_cast_p(
            [Vec(v_s_1_lo_e_partial)[r] for r in range_constexpr(16)],
            v_s_1_hi_e_partial,
        )
        l_row = _fadd(l_row, tile_sum_e5)
        # P4 anchor #8: matches C++ line 635 `asm volatile("" : "+v"(v_p) ::);`
        v_p_lo_e5 = _anchor_packs(v_p_lo_e5)
        v_p_hi_e5 = _anchor_packs(v_p_hi_e5)
        # P3 schedule hints (C++ lines 636-637).
        _sched_barrier_exp_pairs(6, 3, 7)
        _sched_barrier_pairs(10, 5, 7)
        rocdl.sched_barrier(0)
        gpu.barrier()
        rocdl.sched_barrier(0)

        # ─── Cluster 6 (epi, C++ lines 642-654) ───
        # tr_load V[max-3] (4 substeps) from s_v[1] into registers + causal mask
        # on v_s[0] (= S[max-2]) + wait (lgkm(0), vmcnt(v_buffer_load_insts)).
        # Hoisted V reads mirror C++ epilogue line 645 `v_v = tr_load<...>(s_v[1], u_rv);`
        # before the cluster-6 s_barrier. Required for P5 stagger correctness:
        # epilogue Cluster-8 async_load overwrites `s_v[1]`, so V[max-3] must
        # be held in registers across the cluster boundary.
        v_packs_e7 = []
        for kss in range_constexpr(4):
            v_packs_e7.append(_read_v_packs_for_k_substep(1, kss))
        s_lo_e5 = [Vec(v_s_0_lo_e5)[r] for r in range_constexpr(16)]
        s_hi_e5 = [Vec(v_s_0_hi_e5)[r] for r in range_constexpr(16)]
        if const_expr(CAUSAL):
            _causal_mask_inplace(s_lo_e5, s_hi_e5, max_m2)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(NUM_DMA_V)
        rocdl.sched_barrier(0)
        gpu.barrier()
        rocdl.sched_barrier(0)

        # ─── Cluster 7 (epi, C++ lines 656-673) ───
        # FULL GEMM2 (no lazy) + softmax start v_s[0] + scale o
        if const_expr(OPUS_SETPRIO):
            rocdl.s_setprio(1)

        # mma1 (full 16 MFMAs): GEMM2 with v_p = P[max-3] and V[max-3] (from s_v[1])
        # V packs were pre-loaded in Cluster 6 (epi).
        for kss in range_constexpr(4):
            v_pk = v_packs_e7[kss]
            if const_expr(kss < 2):
                p_pk = v_p_lo_e5[kss]
            else:
                p_pk = v_p_hi_e5[kss - 2]
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

        # P2: m_row and m_tile_max_e7 are already log2-scaled.
        m_tile_max_e7 = _wave_row_max(s_lo_e5, s_hi_e5)
        row_max_e7 = _fmax(m_row, m_tile_max_e7)
        rescale_e7 = rocdl.exp2(T.f32, _raw(_fsub(m_row, row_max_e7)))
        m_row = row_max_e7
        lo_e7, hi_e7 = _sub_row_first_half_exp(s_lo_e5, s_hi_e5, row_max_e7)
        v_s_0_lo_e_partial = Vec.from_elements(lo_e7, fx.Float32).ir_value()
        v_s_0_hi_e_partial = Vec.from_elements(hi_e7, fx.Float32).ir_value()

        # P3 schedule hints (C++ lines 665-666).
        _sched_barrier_pairs(10, 5, 8)
        _sched_barrier_exp_pairs(6, 3, 8)
        rocdl.sched_barrier(0)
        _scale_o(o_accs, rescale_e7)

        if const_expr(OPUS_SETPRIO):
            rocdl.s_setprio(0)
        rocdl.sched_barrier(0)
        gpu.barrier()
        rocdl.sched_barrier(0)

        # ─── Cluster 8 (epi, C++ lines 675-682) ───
        # V[max-1] async + K[max-1] ds_read + wait (lgkm(0), vmcnt(v_buffer_load_insts))
        coop_dma_v(max_m1 * fx.Index(BLOCK_N), 1)
        k_pl_e8, k_ph_e8 = _read_k_packs_for_buf(1)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(NUM_DMA_V)
        rocdl.sched_barrier(0)
        gpu.barrier()
        rocdl.sched_barrier(0)

        # ─── Cluster 9 (epi, C++ lines 684-695) ───
        # v_s[1] = mma0(v_q, K[max-1]); l_row *= rescale_e7; finish exp v_s[0]; sum; cast P[max-2]
        v_s_1_lo_e9, v_s_1_hi_e9 = _gemm0(k_pl_e8, k_ph_e8)
        l_row = _fmul(l_row, rescale_e7)
        v_p_lo_e9, v_p_hi_e9, tile_sum_e9 = _finish_softmax_cast_p(
            [Vec(v_s_0_lo_e_partial)[r] for r in range_constexpr(16)],
            v_s_0_hi_e_partial,
        )
        l_row = _fadd(l_row, tile_sum_e9)
        # P3 schedule hints (C++ lines 691-692).
        _sched_barrier_exp_pairs(6, 3, 9)
        _sched_barrier_pairs(10, 5, 9)
        rocdl.sched_barrier(0)
        gpu.barrier()
        rocdl.sched_barrier(0)

        # ─── Cluster 10 (epi, C++ lines 697-709) ───
        # tr_load V[max-2] (4 substeps) from s_v[0] into registers + causal mask
        # on v_s[1] (= S[max-1]) + wait vmcnt(0).
        # Hoisted V reads mirror C++ epilogue line 700 `v_v = tr_load<...>(s_v[0], u_rv);`
        # before the cluster-10 s_barrier. While no further async_load
        # overwrites s_v[0] in this epilogue, hoisting here keeps the V-read
        # placement uniformly one cluster before consumption — matching the
        # C++ kernel exactly and removing all V LDS reads from the MFMA
        # window for cleaner scheduling under stagger.
        v_packs_e11 = []
        for kss in range_constexpr(4):
            v_packs_e11.append(_read_v_packs_for_k_substep(0, kss))
        s_lo_e9 = [Vec(v_s_1_lo_e9)[r] for r in range_constexpr(16)]
        s_hi_e9 = [Vec(v_s_1_hi_e9)[r] for r in range_constexpr(16)]
        if const_expr(CAUSAL):
            _causal_mask_inplace(s_lo_e9, s_hi_e9, max_m1)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(0)
        rocdl.sched_barrier(0)
        gpu.barrier()
        rocdl.sched_barrier(0)

        # ─── Cluster 11 (epi, C++ lines 711-732) ───
        # FULL GEMM2 (P[max-2] @ V[max-2]) + row_max + sub_row + FULL exp v_s[1] + cast P[max-1] + scale o
        # mma1: GEMM2 with v_p = P[max-2] and V[max-2] (from s_v[0])
        # V packs were pre-loaded in Cluster 10 (epi).
        for kss in range_constexpr(4):
            v_pk = v_packs_e11[kss]
            if const_expr(kss < 2):
                p_pk = v_p_lo_e9[kss]
            else:
                p_pk = v_p_hi_e9[kss - 2]
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

        # P2: m_row and m_tile_max_e11 are already log2-scaled.
        m_tile_max_e11 = _wave_row_max(s_lo_e9, s_hi_e9)
        row_max_e11 = _fmax(m_row, m_tile_max_e11)
        rescale_e11 = rocdl.exp2(T.f32, _raw(_fsub(m_row, row_max_e11)))
        m_row = row_max_e11

        # sub_row + first-half exp v_s[1]
        lo_e11, hi_e11 = _sub_row_first_half_exp(s_lo_e9, s_hi_e9, row_max_e11)
        # P3 schedule hints (C++ lines 719-720).
        _sched_barrier_pairs(10, 5, 10)
        _sched_barrier_exp_pairs(6, 3, 10)
        rocdl.sched_barrier(0)

        # Second-half exp v_s[1] (FULL exp now); l_row *= rescale_e11; sum; cast P[max-1]
        hi_e11_full = []
        for r in range_constexpr(16):
            hi_e11_full.append(ArithValue(hi_e11[r]).exp2(fastmath=fm_fast))
        l_row = _fmul(l_row, rescale_e11)
        local_sum_e11 = c_zero_f
        for r in range_constexpr(16):
            local_sum_e11 = _fadd(local_sum_e11, lo_e11[r])
        for r in range_constexpr(16):
            local_sum_e11 = _fadd(local_sum_e11, hi_e11_full[r])
        peer_sum_e11 = reduction_peer(local_sum_e11)
        tile_sum_e11 = _fadd(local_sum_e11, peer_sum_e11)
        l_row = _fadd(l_row, tile_sum_e11)

        v_p_lo_e11 = []
        v_p_hi_e11 = []
        for pks in range_constexpr(PV_K_STEPS):
            p_base = pks * 8
            lo_slice = [lo_e11[p_base + s] for s in range_constexpr(8)]
            v_p_lo_e11.append(bf16_trunc_pack_v8(lo_slice))
            hi_slice = hi_e11_full[p_base:p_base + 8]
            v_p_hi_e11.append(bf16_trunc_pack_v8(hi_slice))

        rocdl.sched_barrier(0)
        _scale_o(o_accs, rescale_e11)
        gpu.barrier()
        rocdl.sched_barrier(0)

        # ─── Cluster 12 (epi, C++ lines 735-739) ───
        # tr_load V[max-1] (4 substeps) from s_v[1] into registers + s_waitcnt_lgkmcnt(0)
        # + barrier. Hoisted V reads mirror C++ epilogue line 736
        # `v_v = tr_load<...>(s_v[1], u_rv);` before the cluster-12 s_barrier.
        # Keeps V-read placement uniformly one cluster before consumption.
        v_packs_e13 = []
        for kss in range_constexpr(4):
            v_packs_e13.append(_read_v_packs_for_k_substep(1, kss))
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        rocdl.sched_barrier(0)
        gpu.barrier()
        rocdl.sched_barrier(0)

        # ─── Cluster 13 (epi, C++ line 742) ───
        # FINAL mma1: GEMM2 with v_p = P[max-1] and V[max-1] (from s_v[1])
        # V packs were pre-loaded in Cluster 12 (epi).
        for kss in range_constexpr(4):
            v_pk = v_packs_e13[kss]
            if const_expr(kss < 2):
                p_pk = v_p_lo_e11[kss]
            else:
                p_pk = v_p_hi_e11[kss - 2]
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = mfma_acc(v_pk[dc], p_pk, o_accs[dc])

        # ─── Normalize O and store to gmem (C++ lines 744-754) ───
        # l_inv = (l_row > 0) ? 1/l_row : 0
        inv_l = rocdl.rcp(T.f32, _raw(l_row))
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(16)

        # P5 stagger closeout — extra s_barrier for warps 0-3 only,
        # mirrors C++ lines 748-750:
        #     if (!stagger) {
        #         __builtin_amdgcn_s_barrier();
        #     }
        # Same gating as the prologue half: with OPUS_ENABLE_STAGGER ON
        # (default) warps 0-3 hit one extra barrier here so that across
        # the whole kernel both groups observe the same total barrier
        # count. With the flag OFF we fall back to an unconditional
        # barrier on every wave.
        if const_expr(OPUS_ENABLE_STAGGER):
            _stagger_extra_barrier_if_zero()
        else:
            gpu.barrier()

        if q_in_bounds:
            for dc in range_constexpr(D_CHUNKS):
                o_norm_vec = Vec(o_accs[dc]) * inv_l_vec
                for r in range_constexpr(16):
                    o_val = Vec(o_norm_vec)[r]
                    o_f16 = fx.Float32(o_val).to(elem_dtype)
                    d_row_rel = lane_div_32 * 4 + (r // 4) * 8 + (r % 4)
                    d_col = fx.Index(dc * D_CHUNK) + d_row_rel
                    o_global = global_idx_q(q_row, d_col)
                    _store_global_half(o_ptr, o_global, o_f16)

    @flyc.jit
    def launch_flash_attn_opus(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
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
            Q, K, V, O, seq_len,
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

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_opus_compile_hints):
            return launch_flash_attn_opus(*args, **kwargs)

    def _compile(Q, K, V, O, batch_size, seq_len, stream=None):
        with CompilationContext.compile_hints(_opus_compile_hints):
            return flyc.compile(
                launch_flash_attn_opus, Q, K, V, O, batch_size, seq_len,
                fx.Stream(stream))

    _launch.compile = _compile

    return _launch
