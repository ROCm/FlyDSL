# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""FlyDSL port of aiter PR #3470 ``gemm2_a4w4`` (MXFP4 MoE down-proj, gfx950).

Parametrized over the launch_atomic specialization:
    ``launch_atomic<MAX_M=655360, NE=385, K=512, N_OUT=7168, TOPK=9, BM, kUseNT>``
Supported instances (atomic path):
  * BM=32, kUseNT=false -> ``...TOPK9_BM32_ATOMIC``        (compile_gemm2_a4w4_port(BM=32))
  * BM=16, kUseNT=true  -> ``...TOPK9_BM16_ATOMIC_NT``     (compile_gemm2_a4w4_port(BM=16, use_nt=True))

The port mirrors gemm2_a4w4.cuh's atomic path instruction-for-instruction:
  * 4 ``make.buffer.rsrc`` (A_q, A_scale, B_q, B_scale) with exact num_bytes.
  * A -> LDS via ``raw.ptr.buffer.load.lds`` (2 slots), swizzled (BM16: 2 waves).
  * B / scales via ``raw.ptr.buffer.load.v4i32`` / ``.i32`` (NT: B aux=2).
  * ``s_waitcnt vmcnt(23/22)`` + ``s_barrier`` cross-wave fences.
  * K=512 = 2 K-tiles fully unrolled; 32 (BM32) / 16 (BM16) MFMAs.
  * atomic bf16 epilog: LDS cshuffle -> ``global.atomic.fadd.v2bf16`` * topk weight.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

# ── shape constants (BM-independent) ─────────────────────────────────────────
MAX_M = 655360
NE = 385
K = 512  # gemm2 contraction = inter_dim
N_OUT = 7168  # gemm2 output dim = model_dim
TOPK = 9

BN = 256
BK = 256
K_HALF = K // 2  # 256 packed-fp4 bytes along K
KH_TILE = BK // 2  # 128 packed bytes per K-tile
NUM_N_BLOCKS = N_OUT // 256  # 28
K_TILES_TOTAL = K // BK  # 2
kStages = 2
_A_ROWS_PER_WAVE = 8  # each loading wave streams 8 A rows into LDS

# scale-layout consts (mirror gemm2_a4w4.cuh)
kBS_c_k1 = (K // 32) // 4 // 2  # 2
kBS_stride_k0_dw = 64
kBS_stride_n0_dw = kBS_c_k1 * 64  # 128
kBS_c_n1 = N_OUT // 16 // 2  # 224
kBS_per_expert_dw = kBS_c_n1 * kBS_stride_n0_dw  # 28672
kAS_c_k1 = (K // 32) // 4 // 2  # 2
kAS_per_chunk_dw = kAS_c_k1 * 64  # 128

# BM-independent buffer resource sizes (bytes) — must match HIP make.buffer.rsrc
AQ_BYTES = MAX_M * K_HALF  # 167772160
BQ_BYTES = NE * N_OUT * K_HALF  # 706478080
BSCALE_BYTES = NE * kBS_per_expert_dw * 4  # 44154880


def ascale_bytes(BM):
    """A_scale buffer-resource num_bytes for a given BM (kAS_bound_div=BM in
    atomic mode): (MAX_M/BM) * kAS_per_chunk_dw * 4."""
    return (MAX_M // BM) * kAS_per_chunk_dw * 4


def saq_slot_bytes(BM):
    return BM * KH_TILE  # s_Aq[slot] = BM rows x KH_TILE bytes


def lds_bytes(BM):
    return BM * BN * 4  # union max: lds_acc[BM*BN] f32 (>= 2*saq_slot_bytes)


def kmchunks(BM):
    return 1 if BM == 16 else BM // 16


# Back-compat module constants (BM32 defaults; the test imports BM/ASCALE_BYTES).
BM = 32
kMChunks = kmchunks(BM)
SAQ_SLOT_BYTES = saq_slot_bytes(BM)
LDS_ACC_FLOATS = BM * BN
LDS_BYTES = lds_bytes(BM)
ASCALE_BYTES = ascale_bytes(BM)


_PTR3 = "!llvm.ptr<3>"


def _raw(v):
    """Unwrap an fx value to a raw ir.Value for raw llvm/arith ops."""
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _lds_ptr3(base_i32, byte_off_i32):
    """ptr<3> = inttoptr(i64(base_i32 + byte_off_i32))."""
    addr_i64 = fx.Int64(base_i32 + byte_off_i32)
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(addr_i64))


def _lds_base_ptr3(lds_view):
    """One ptr<3> for the LDS base; offsets via GEP. (extract_aligned_pointer ->
    inttoptr is forced by FlyDSL's memref.global LDS model.)"""
    base_i32 = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(lds_view))
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(fx.Int64(base_i32)))


def _gep3(base_ptr, byte_off_i32):
    """getelementptr i8, base_ptr, byte_off_i32  (ptr<3>)."""
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8)


def _s_barrier_bare():
    """Bare ``s_barrier`` (no surrounding memory fence), matching HIP's K-loop
    ``__builtin_amdgcn_s_barrier()`` cross-wave fence after the vmcnt wait."""
    llvm.inline_asm(res=None, operands_=[], asm_string="s_barrier", constraints="", has_side_effects=True)


def _global_base_ptr1(arg):
    """One ptr<1> base for a global tensor (single memref->ptr conversion)."""
    base_idx = buffer_ops.extract_base_index(arg, address_space=1)
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(base_idx)))


def _gep1(base_ptr, byte_off_i32):
    """getelementptr i8, base_ptr, byte_off_i32  (ptr<1>)."""
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8)


def _global_ptr1(arg, byte_off_i32):
    return _gep1(_global_base_ptr1(arg), byte_off_i32)


def _lds_swizzle_mask(row):
    """lds_swizzle_mask<ROW_BYTES=BK/2=128>(row): mask = (row & 14) << 3."""
    return (row & fx.Int32(14)) << fx.Int32(3)


def _issue_a_load_lds(aq_rsrc, saq, slot, kt, car0, lane, wave, slot_bytes):
    """Issue one A->LDS tile load: ``raw.ptr.buffer.load.lds`` into s_Aq[slot].
    Identical formula for BM16/BM32 (lds_row = wave*8); BM16 callers gate this on
    ``wave < BM/8``. Side-effecting, so it can be issued before the cumsum branch
    without the compiler sinking it back."""
    lane_div_8 = lane // fx.Int32(8)
    lane_mod_8 = lane % fx.Int32(8)
    lds_row = wave * fx.Int32(_A_ROWS_PER_WAVE)
    mask = _lds_swizzle_mask(lds_row + lane_div_8)
    voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + car0 * fx.Int32(K // 2)
    base_i32 = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(saq.get()))
    off_i32 = fx.Int32(slot * slot_bytes) + lds_row * fx.Int32(KH_TILE)
    lds_ptr = _lds_ptr3(base_i32, off_i32)
    rocdl.raw_ptr_buffer_load_lds(
        aq_rsrc, lds_ptr, fx.Int32(16), voffset, fx.Int32(kt * KH_TILE), fx.Int32(0), fx.Int32(0)
    )


def compile_gemm2_a4w4_port(BM=32, use_nt=False):
    """Compile the gemm2 a4w4 atomic port for the given BM / kUseNT specialization.

    BM=32, use_nt=False  -> mirrors ...BM32_ATOMIC
    BM=16, use_nt=True   -> mirrors ...BM16_ATOMIC_NT (production fused-moe pick)
    """
    _kMChunks = kmchunks(BM)
    _slot_bytes = saq_slot_bytes(BM)
    _lds_acc_floats = BM * BN
    _lds_bytes = lds_bytes(BM)
    _ascale_bytes = ascale_bytes(BM)
    _n_load_waves = BM // _A_ROWS_PER_WAVE  # BM16: 2, BM32: 4
    _name = f"gemm2_a4w4_port_bm{BM}{'_nt' if use_nt else ''}_atomic"

    allocator = SmemAllocator(None, arch="gfx950", global_sym_name=f"gemm2port_smem_bm{BM}{'_nt' if use_nt else ''}")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + _lds_bytes

    @flyc.kernel(name=_name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_aq: fx.Tensor,
        arg_ascale: fx.Tensor,
        arg_bq: fx.Tensor,
        arg_bscale: fx.Tensor,
        arg_eids: fx.Tensor,
        arg_cumsum: fx.Tensor,
        arg_stids: fx.Tensor,
        arg_sweights: fx.Tensor,
        i32_M: fx.Int32,
        arg_out: fx.Tensor,
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = arith.index_cast(T.i32, tx)
        bx_i32 = arith.index_cast(T.i32, bx)

        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))  # wave == wave_n

        # ── issue A->LDS as early as possible, BEFORE the cumsum-gated branch ──
        # raw.ptr.buffer.load.lds is side-effecting (writes LDS), so the compiler
        # cannot sink it back into the then-block. Issuing it here overlaps the
        # A->LDS HBM latency with the cumsum load + bound check. A->LDS depends
        # only on bx/lane (not cumsum/eids); padding blocks load harmlessly and
        # the early-return below still skips all compute. BM16 loads only 16 rows
        # (waves 0,1), so gate the issue on wave < BM/8.
        m_row0 = (bx_i32 // fx.Int32(NUM_N_BLOCKS)) * fx.Int32(BM)
        car0 = m_row0 + wave * fx.Int32(_A_ROWS_PER_WAVE) + (lane // fx.Int32(8))
        aq_rsrc = buffer_ops.create_buffer_resource(arg_aq, max_size=False, num_records_bytes=fx.Index(AQ_BYTES))
        saq = SmemPtr(allocator.get_base(), lds_off, T.i8, shape=(kStages * _slot_bytes,))

        def _issue_both_a_loads():
            _issue_a_load_lds(aq_rsrc, saq, 0, 0, car0, lane, wave, _slot_bytes)
            _issue_a_load_lds(aq_rsrc, saq, 1, 1, car0, lane, wave, _slot_bytes)

        if const_expr(_n_load_waves < 4):  # BM16: only waves 0,1 hold A rows
            a_pred = arith.cmpi(arith.CmpIPredicate.slt, wave, fx.Int32(_n_load_waves))
            a_if = scf.IfOp(a_pred, [], has_else=False)
            with ir.InsertionPoint(a_if.then_block):
                _issue_both_a_loads()
                scf.YieldOp([])
        else:
            _issue_both_a_loads()
        rocdl.sched_barrier(0)

        # total_m_blocks = cumsum[0] / BM ; bound = total_m_blocks * NUM_N_BLOCKS
        cumsum0 = llvm.load(T.i32, _global_ptr1(arg_cumsum, fx.Int32(0)))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(NUM_N_BLOCKS)

        in_range = arith.cmpi(arith.CmpIPredicate.slt, bx_i32, bound)
        if_op = scf.IfOp(in_range, [], has_else=False)
        with ir.InsertionPoint(if_op.then_block):
            _gemm2_body(
                allocator,
                lds_off,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                i32_M,
                arg_out,
                bx_i32,
                lane,
                wave,
                BM,
                use_nt,
            )
            scf.YieldOp([])

    @flyc.jit
    def launch_gemm2(
        arg_aq: fx.Tensor,
        arg_ascale: fx.Tensor,
        arg_bq: fx.Tensor,
        arg_bscale: fx.Tensor,
        arg_eids: fx.Tensor,
        arg_cumsum: fx.Tensor,
        arg_stids: fx.Tensor,
        arg_sweights: fx.Tensor,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Tensor,
        stream: fx.Stream,
    ):
        from flydsl.compiler.kernel_function import CompilationContext

        ctx = CompilationContext.get_current()
        allocator.finalized = False
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        grid_x = arith.index_cast(T.index, i32_max_m_blocks) * fx.Index(NUM_N_BLOCKS)
        gemm2_kernel(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            arg_out,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2


def _gemm2_body(
    allocator,
    lds_off,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_stids,
    arg_sweights,
    i32_M,
    arg_out,
    bx_i32,
    lane,
    wave,
    BM,
    use_nt,
):
    _kMChunks = kmchunks(BM)
    _slot_bytes = saq_slot_bytes(BM)
    _lds_acc_floats = BM * BN
    _ascale_bytes = ascale_bytes(BM)
    b_aux = 2 if use_nt else 0  # NT: B_q loads carry aux=2 (non-temporal hint)

    # block -> (m_block_idx, n_block_idx) ; e = sorted_expert_ids[m_block_idx]
    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    e = llvm.load(T.i32, _global_ptr1(arg_eids, m_block_idx * fx.Int32(4)))
    e = rocdl.readfirstlane(T.i32, e)
    m_row = m_block_idx * fx.Int32(BM)

    # ── buffer resources (exact num_bytes) ──────────────────────────────────
    # (A_q resource + A->LDS loads are issued by the kernel before the branch.)
    ascale_rsrc = buffer_ops.create_buffer_resource(
        arg_ascale, max_size=False, num_records_bytes=fx.Index(_ascale_bytes)
    )
    bq_rsrc = buffer_ops.create_buffer_resource(arg_bq, max_size=False, num_records_bytes=fx.Index(BQ_BYTES))
    bscale_rsrc = buffer_ops.create_buffer_resource(
        arg_bscale, max_size=False, num_records_bytes=fx.Index(BSCALE_BYTES)
    )

    # ── LDS base ────────────────────────────────────────────────────────────
    lds_base = allocator.get_base()
    saq = SmemPtr(lds_base, lds_off, T.i8, shape=(kStages * _slot_bytes,))
    lds_acc = SmemPtr(lds_base, lds_off, T.f32, shape=(_lds_acc_floats,))

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    # ── s_base computations (readfirstlane'd, uniform per wave) ──────────────
    b_load_s_base = []
    for j in range_constexpr(4):
        v = (e * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + fx.Int32(j * 16)) * fx.Int32(
            K_HALF
        )
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    mni_base = n_block_idx * fx.Int32(BN // 16 // 2) + wave * fx.Int32(BN // 64 // 2)
    b_scale_s_base = []
    for mw in range_constexpr(2):
        v = (e * fx.Int32(kBS_per_expert_dw) + (mni_base + fx.Int32(mw)) * fx.Int32(kBS_stride_n0_dw)) * fx.Int32(4)
        b_scale_s_base.append(rocdl.readfirstlane(T.i32, v))

    # a_scale_s_base[0]: chunk_base = m_row / BM (atomic kAS_bound_div = BM); sub=0
    chunk_base = m_row // fx.Int32(BM)
    a_scale_s_base0 = rocdl.readfirstlane(T.i32, chunk_base * fx.Int32(kAS_per_chunk_dw) * fx.Int32(4))

    # ── a_scale (atomic) : v_voff = ((lane/16)*16 + lane%16)*4 ───────────────
    v_voff_scale = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)
    a_scale_v = []
    for ku in range_constexpr(2):
        v = buffer_ops.buffer_load(
            ascale_rsrc,
            (v_voff_scale + fx.Int32(ku * 256)) // fx.Int32(4),
            vec_width=1,
            dtype=T.i32,
            soffset_bytes=a_scale_s_base0,
        )
        a_scale_v.append(v)

    # ── b_scale ku0/ku1 ──────────────────────────────────────────────────────
    b_scale_v = [[None, None], [None, None]]
    for ku in range_constexpr(2):
        imm = ku * (kBS_stride_k0_dw * 4)
        for mw in range_constexpr(2):
            v = buffer_ops.buffer_load(
                bscale_rsrc,
                (v_voff_scale + fx.Int32(imm)) // fx.Int32(4),
                vec_width=1,
                dtype=T.i32,
                soffset_bytes=b_scale_s_base[mw],
            )
            b_scale_v[ku][mw] = v

    # ── B loads (NT: cache_modifier=2) : v_voff = (lane/16)*256 + (lane%16)*16 + K_BYTE
    b = [[[None, None] for _ in range(4)] for _ in range(2)]
    for kc in range_constexpr(2):
        k_byte = kc * 2048
        v_voff_b = (lane_div_16 * fx.Int32(256)) + (lane_mod_16 * fx.Int32(16)) + fx.Int32(k_byte)
        for j in range_constexpr(4):
            for half in range_constexpr(2):
                imm = half * 1024
                frag = buffer_ops.buffer_load(
                    bq_rsrc,
                    (v_voff_b + fx.Int32(imm)) // fx.Int32(4),
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=b_aux,
                    soffset_bytes=b_load_s_base[j],
                )
                b[kc][j][half] = Vec(frag)

    # ── ds_read(slot) -> a[i][k] (i32x4) ; i in [0,kMChunks) ─────────────────
    def issue_a_ds_read(slot):
        lane_row = lane_mod_16
        lane_col = lane_div_16 * fx.Int32(16)
        mask = _lds_swizzle_mask(lane_row)
        base_ptr = _lds_base_ptr3(saq.get())
        a = [[None, None] for _ in range(_kMChunks)]
        for k in range_constexpr(2):
            lds_col = (lane_col + fx.Int32(k * 64)) ^ mask
            for i in range_constexpr(_kMChunks):
                lds_row = lane_row + fx.Int32(i * 16)
                byte_off = fx.Int32(slot * _slot_bytes) + lds_row * fx.Int32(KH_TILE) + lds_col
                a[i][k] = llvm.load(T.vec(4, T.i32), _gep3(base_ptr, byte_off))  # ds_read_b128
        return a

    # ── MFMA cluster (BM16: kMChunks=1 -> i0 only) ───────────────────────────
    mfma_res_ty = T.f32x4
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    accm = [[None, None, None, None] for _ in range(_kMChunks)]

    def mfma_cluster(slot, a, sa, b_scale_slot, init):
        for J in range_constexpr(4):
            mni = J // 2
            in_b = J % 2
            sb = b_scale_slot[mni]
            b_J0 = b[slot][J][0]
            b_J1 = b[slot][J][1]
            if const_expr(init):
                accm[0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_res_ty, [a[0][0], b_J0, zero4, 4, 4, 0, sa, 0 + in_b, sb]
                )
                if const_expr(_kMChunks > 1):
                    accm[1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty, [a[1][0], b_J0, zero4, 4, 4, 1, sa, 0 + in_b, sb]
                    )
            else:
                accm[0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_res_ty, [a[0][0], b_J0, accm[0][J], 4, 4, 0, sa, 0 + in_b, sb]
                )
                if const_expr(_kMChunks > 1):
                    accm[1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty, [a[1][0], b_J0, accm[1][J], 4, 4, 1, sa, 0 + in_b, sb]
                    )
            accm[0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                mfma_res_ty, [a[0][1], b_J1, accm[0][J], 4, 4, 2, sa, 2 + in_b, sb]
            )
            if const_expr(_kMChunks > 1):
                accm[1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_res_ty, [a[1][1], b_J1, accm[1][J], 4, 4, 3, sa, 2 + in_b, sb]
                )

    # ── K loop (2 stages, fully unrolled) ────────────────────────────────────
    for S in range_constexpr(kStages):
        kt = K_TILES_TOTAL - kStages + S
        slot = kt % kStages
        vmcnt = 23 if S == 0 else 22
        llvm.inline_asm(
            res=None, operands_=[], asm_string=f"s_waitcnt vmcnt({vmcnt})", constraints="", has_side_effects=True
        )
        _s_barrier_bare()
        a = issue_a_ds_read(slot)
        mfma_cluster(slot, a, a_scale_v[kt], b_scale_v[slot], init=(S == 0))

    # ── epilog: apply_atomic_bf16_epilog ─────────────────────────────────────
    saq._view_cache = None
    lds_acc._view_cache = None
    _atomic_bf16_epilog(lds_acc, accm, arg_out, arg_stids, arg_sweights, m_row, n_block_idx, wave, lane, i32_M, BM)


def _atomic_bf16_epilog(lds_acc, accm, arg_out, arg_stids, arg_sweights, m_row, n_block_idx, wave, lane, i32_M, BM):
    _kMChunks = kmchunks(BM)
    M_REPS = BM // 8  # BM32: 4, BM16: 2
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_base_ptr3(lds_acc.get())

    tx_i32 = arith.index_cast(T.i32, gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    stids_base = _global_base_ptr1(arg_stids)
    sweights_base = _global_base_ptr1(arg_sweights)
    out_base = _global_base_ptr1(arg_out)

    # Prefetch sorted_token_ids / sorted_weights BEFORE the cshuffle stores and
    # both LDS barriers (invariant => freely hoistable), overlapping their global
    # latency with the store + barriers instead of exposing it in the atomic loop.
    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(llvm.load(T.i32, _gep1(stids_base, sorted_pos * fx.Int32(4)), invariant=True))
        weight.append(llvm.load(T.f32, _gep1(sweights_base, sorted_pos * fx.Int32(4)), invariant=True))

    # pre-store fence+barrier (HIP run_one __syncthreads() before the epilog).
    rocdl.barrier()

    # write accm -> lds_acc cshuffle (scalar f32 stores, as HIP does)
    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            col = wave * fx.Int32(64) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))

    rocdl.barrier()

    # read back + weighted atomic add (token_id / weight prefetched above)
    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        valid = arith.cmpi(arith.CmpIPredicate.slt, token_id, i32_M)
        if_op = scf.IfOp(valid, [], has_else=False)
        with ir.InsertionPoint(if_op.then_block):
            row_base_addr = token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            for s in range_constexpr(4):
                # adjacent ee=0,1 are contiguous -> one <2xf32> load (as HIP vectorizes)
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(llvm.load(T.vec(2, T.f32), _gep3(lds_base, idx0 * fx.Int32(4))))
                pk = Vec.from_elements([v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32).to(fx.BFloat16)
                off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)  # bf16 byte offset
                out_ptr = _gep1(out_base, off)
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    out_ptr,
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
            scf.YieldOp([])
