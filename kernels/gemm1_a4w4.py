"""FlyDSL port of aiter MXFP4 MoE GEMM-1 (fused SwiGLU): ``gemm1_a4w4``.

This is a 1:1 instruction-shape port of the HIP kernel at
``csrc/kernels/mxfp4_moe/gemm_a4w4/gemm1_a4w4.cuh`` for the

    BM=32, kInlineQuant=False, kUseNT=False, kXcdSwizzle=0

instance targeting gfx950 (CDNA4, wave64, MXFP4 MFMA).

The kernel computes a per-expert MXFP4 x MXFP4 GEMM producing the fused
gate/up intermediate, applies SwiGLU (silu(gate) * up) and re-quantises the
result back to MXFP4 (packed fp4 + e8m0 scales).

The body mirrors the HIP instruction shape: ``buffer_load_lds`` global->LDS
DMA for A_q / A_scale, raw ``buffer_load_b128`` / ``buffer_load_b32`` for the
B operands (kept in registers), the ``mfma.scale.f32.16x16x128.f8f6f4``
(cbsz=4 blgp=4) FP4xFP4 MFMA, ``sched_barrier`` / ``s_setprio`` scheduling
fences, the DPP quad amax + ``cvt.scalef32.pk.fp4.f32`` requant epilog.

Only the layout/byte-math constants are computed in Python (``const_expr``);
all data movement maps to the same hardware instructions HIP emits.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, memref
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator

# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------


def _i32(v):
    """Materialise an i32 ir.Value from a Python int / DSL Numeric / ir.Value."""
    if isinstance(v, int):
        return arith.unwrap(arith.constant(v, type=T.i32))
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "ir_value"):
        return v.ir_value()
    if hasattr(v, "value"):
        return v.value
    return v


def _f32(v):
    if isinstance(v, float):
        return arith.unwrap(arith.constant(v, type=T.f32))
    return _i32(v)


def _lds_base_addr_i32(base_memref):
    """Return the 32-bit LDS byte address of the start of ``base_memref``."""
    idx = memref.extract_aligned_pointer_as_index(base_memref)
    return arith.unwrap(arith.index_cast(T.i32, idx))


def _lds_ptr(base_addr_i32, byte_off_i32):
    """Form an ``!llvm.ptr<3>`` at ``base + byte_off`` (both i32)."""
    addr = arith.unwrap(arith.addi(base_addr_i32, byte_off_i32))
    ptr_ty = ir.Type.parse("!llvm.ptr<3>")
    return llvm.inttoptr(ptr_ty, addr)


def _lds_load_vec(base_addr_i32, byte_off_i32, vec_ty, align):
    ptr = _lds_ptr(base_addr_i32, byte_off_i32)
    return llvm.load(vec_ty, ptr, alignment=align)


def _lds_store(base_addr_i32, byte_off_i32, val, align, nontemporal=False):
    ptr = _lds_ptr(base_addr_i32, byte_off_i32)
    llvm.store(val, ptr, alignment=align, nontemporal=nontemporal)


def _global_ptr(base_ptr_i64_addr, byte_off_i32):
    off64 = arith.unwrap(arith.extsi(T.i64, byte_off_i32))
    base = arith.unwrap(arith.addi(base_ptr_i64_addr, off64))
    ptr_ty = ir.Type.parse("!llvm.ptr<1>")
    return llvm.inttoptr(ptr_ty, base)


def _readfirstlane(v):
    return rocdl.readfirstlane(T.i32, _i32(v))


def _sched_barrier():
    rocdl.sched_barrier(0)


def _setprio(p):
    rocdl.s_setprio(p)


def _raw_buffer_load_b128(rsrc, voffset_i32, soffset_i32, aux=0):
    v4i32 = ir.VectorType.get([4], T.i32)
    return rocdl.raw_ptr_buffer_load(v4i32, rsrc, _i32(voffset_i32), _i32(soffset_i32), _i32(aux))


def _raw_buffer_load_b32(rsrc, voffset_i32, soffset_i32, aux=0):
    return rocdl.raw_ptr_buffer_load(T.i32, rsrc, _i32(voffset_i32), _i32(soffset_i32), _i32(aux))


def _buffer_load_lds(rsrc, lds_ptr, size, voffset_i32, soffset_i32, offset=0, aux=0):
    rocdl.raw_ptr_buffer_load_lds(
        rsrc, lds_ptr, _i32(size), _i32(voffset_i32), _i32(soffset_i32), _i32(offset), _i32(aux)
    )


def _make_rsrc(arg, num_bytes):
    """Make an AMD buffer descriptor (!llvm.ptr<8>) from a fly.Tensor arg.

    ``num_bytes`` is the buffer ``num_records`` (in bytes) used for HW OOB
    checking; mirrors the HIP ``make_buffer_rsrc`` second argument.
    """
    mref = arg.value if hasattr(arg, "value") else arg
    if hasattr(num_bytes, "ir_value"):
        num_bytes = num_bytes.ir_value()
    return buffer_ops.create_buffer_resource(mref, max_size=False, num_records_bytes=num_bytes)


def _tensor_addr_i64(arg):
    """Extract the device base address of a tensor arg as i64."""
    mref = arg.value if hasattr(arg, "value") else arg
    idx = buffer_ops.extract_base_index(mref, address_space=1)
    return arith.unwrap(arith.index_cast(T.i64, idx))


# ---------------------------------------------------------------------------
# vector packing helpers
# ---------------------------------------------------------------------------

_V4I32 = lambda: ir.VectorType.get([4], T.i32)
_V8I32 = lambda: ir.VectorType.get([8], T.i32)
_V4F32 = lambda: ir.VectorType.get([4], T.f32)


def _vec_extract(vec, i, ty):
    pos = arith.unwrap(arith.constant(i, type=T.i64))
    return llvm.extractelement(vec, pos)


def _bitcast(res_ty, v):
    """Bit-reinterpret ``v`` (i32 / vector dword) to ``res_ty``.

    ``arith.bitcast`` rejects scalar<->vector reinterprets ("same shape" check);
    use ``llvm.bitcast`` which allows them (same total bit-width).  Mirrors the
    HIP ``__builtin_bit_cast`` / reinterpret on the packed dwords.
    """
    return llvm.bitcast(res_ty, _i32(v))


def _vec_insert(vec, val, i):
    pos = arith.unwrap(arith.constant(i, type=T.i64))
    return llvm.insertelement(vec, val, pos)


def _pack_v8i32_from_v4i32(v4):
    """Build vector<8xi32> = [v4[0..3], 0, 0, 0, 0] (two zero MFMA slots).

    Mirrors HIP passing i32x4 to the v4i32 intrinsic; FlyDSL's op takes
    vector<8xi32> with the high half zero (the backend only reads the used half).
    """
    zero = _i32(0)
    out = llvm.mlir_undef(_V8I32())
    for i in range(4):
        out = _vec_insert(out, _vec_extract(v4, i, T.i32), i)
    for i in range(4, 8):
        out = _vec_insert(out, zero, i)
    return out


def _mfma_scale(a_v8, b_v8, c_v4f32, opsel_a, opsel_b, sa, sb):
    return rocdl.mfma_scale_f32_16x16x128_f8f6f4(
        _V4F32(),
        [a_v8, b_v8, c_v4f32, 4, 4, int(opsel_a), _i32(sa), int(opsel_b), _i32(sb)],
    )


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def compile_gemm1_a4w4(
    *,
    MAX_M: int = 655360,
    NUM_EXPERTS: int = 385,
    K: int = 7168,
    N_OUT: int = 1024,
    BM: int = 32,
    kUseNT: bool = False,
    kInlineQuant: bool = False,
    kXcdSwizzle: int = 0,
):
    # ---- static_asserts mirroring the HIP kernel (gemm1_a4w4.cuh:41-47). ----
    assert K == 7168, "static_assert(K == 7168)"
    assert N_OUT % 256 == 0, "static_assert(N_OUT % 256 == 0)"
    assert BM in (16, 32, 64, 128), "static_assert(BM in {16,32,64,128})"
    assert (not kInlineQuant) or BM in (16, 32), "kInlineQuant supports BM=16/32"
    assert (not kInlineQuant) or BM != 128, "kInlineQuant not supported at BM=128"
    assert kXcdSwizzle == 0, "no CSV gemm1 variant uses xcd swizzle"
    # The CSV-required variants are:
    #   (BM=16,  kUseNT=True,  kInlineQuant=True ),
    #   (BM=32,  kUseNT=True,  kInlineQuant=False),
    #   (BM=32,  kUseNT=False, kInlineQuant=False),
    #   (BM=128, kUseNT=False, kInlineQuant=False).
    # The kInlineQuant && BM==32 remap_xcd branch in HIP is dead code here and is
    # intentionally NOT ported.
    if kInlineQuant and BM == 32:
        raise NotImplementedError("kInlineQuant && BM==32 (remap_xcd path) not required by CSV")

    # ---- derived compile-time constants (pure Python ints) ----
    BN = 256
    BK = 256
    K_HALF = K // 2
    K_TILES_TOTAL = K // BK
    kStages = 2
    # BM-dependent (gemm1_a4w4.cuh:49-62):
    kAStages = 2 if BM == 128 else 3
    kLoopIter = K_TILES_TOTAL - kStages
    kUnroll = kLoopIter
    kSubBlocks = 1 if BM < 32 else BM // 32
    kMChunks = BM // 16
    num_n_blocks = N_OUT // 256

    kBS_c_n1 = N_OUT // 16 // 2
    kBS_c_k1 = (K // 32) // 4 // 2
    kBS_stride_k0_dw = 64
    kBS_stride_n0_dw = kBS_c_k1 * 64
    kBS_per_expert_dw = kBS_c_n1 * kBS_stride_n0_dw

    kAS_c_k1 = (K // 32) // 4 // 2
    assert kAS_c_k1 == 28
    kAS_per_chunk_dw = 1 * kAS_c_k1 * 64

    # ---- LDS union sizing ----
    s_Aq_bytes = kAStages * BM * (BK // 2)  # 3 * 32 * 128 = 12288
    s_Ascale_off = s_Aq_bytes
    s_Ascale_bytes = kSubBlocks * K_TILES_TOTAL * 256  # 1 * 28 * 256 = 7168
    lds_acc_bytes = BM * BN * 4  # 32 * 256 * 4 = 32768
    lds_total_bytes = max(s_Aq_bytes + s_Ascale_bytes, lds_acc_bytes)

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="gemm1_a4w4_lds")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + lds_total_bytes

    # epilog-only constants
    BN_INT = BN // 2
    N_INTER = N_OUT // 2
    K_G2_HALF = N_INTER // 2  # 256
    epi_kAS_c_k1 = (N_INTER // 32) // 4 // 2
    epi_kAS_per_chunk_dw = 1 * epi_kAS_c_k1 * 64  # 128

    _tag = f"BM{BM}" + ("_NT" if kUseNT else "") + ("_IQ" if kInlineQuant else "")
    KERNEL_NAME = f"gemm1_a4w4_{_tag}_K{K}_N{N_OUT}_E{NUM_EXPERTS}"

    # =====================================================================
    @flyc.kernel
    def kernel_gemm1(
        arg_A_q: fx.Tensor,
        arg_A_scale: fx.Tensor,
        arg_B_q: fx.Tensor,
        arg_B_scale: fx.Tensor,
        arg_sorted_expert_ids: fx.Tensor,
        arg_cumsum: fx.Tensor,
        arg_m_indices: fx.Tensor,
        i32_n_tokens: fx.Int32,
        arg_A_q_out: fx.Tensor,
        arg_A_scale_out: fx.Tensor,
        arg_hidden_states: fx.Tensor,
    ):
        pid = fx.Int32(gpu.block_id("x"))
        tid = fx.Int32(gpu.thread_id("x"))
        wave = fx.Int32(_readfirstlane((tid // 64).ir_value()))
        wave_n = wave
        lane = tid % 64

        # ---- buffer descriptors (mirror HIP make_buffer_rsrc, bytes) ----
        n_tokens = i32_n_tokens
        A_q_rsrc = _make_rsrc(arg_A_q, (n_tokens * K_HALF).ir_value())
        B_q_rsrc = _make_rsrc(arg_B_q, _i32(NUM_EXPERTS * N_OUT * K_HALF))
        A_scale_rsrc = _make_rsrc(arg_A_scale, _i32((MAX_M // 32) * kAS_per_chunk_dw * 4))
        B_scale_rsrc = _make_rsrc(arg_B_scale, _i32(NUM_EXPERTS * kBS_per_expert_dw * 4))
        # hidden_states rsrc (live only for kInlineQuant; mirrors HIP):
        #   base = hidden_states (IQ) / B_q (else, size=0 ⇒ OOB-trap), bytes =
        #   n_tokens * K * sizeof(bf16) when IQ else 0.
        if const_expr(kInlineQuant):
            hidden_rsrc = _make_rsrc(arg_hidden_states, (n_tokens * (K * 2)).ir_value())
        else:
            hidden_rsrc = _make_rsrc(arg_B_q, _i32(0))

        # ---- scalar tile setup (read cumsum/sorted_expert_ids) ----
        cumsum_addr = _tensor_addr_i64(arg_cumsum)
        cumsum0 = llvm.load(T.i32, _global_ptr(cumsum_addr, _i32(0)), alignment=4)
        total_m_blocks = fx.Int32(cumsum0) // BM
        total_tiles = total_m_blocks * num_n_blocks

        in_range = arith.unwrap(arith.cmpi(CmpIPredicate.slt, pid.ir_value(), total_tiles.ir_value()))
        if_op = scf_if_begin(in_range)
        with ir.InsertionPoint(if_op.then_block):
            m_block_idx = pid // num_n_blocks
            n_block_idx = pid % num_n_blocks

            sorted_addr = _tensor_addr_i64(arg_sorted_expert_ids)
            e_ptr = _global_ptr(sorted_addr, (m_block_idx * 4).ir_value())
            e = fx.Int32(llvm.load(T.i32, e_ptr, alignment=4))

            m_row = m_block_idx * BM

            # ---- LDS base address ----
            lds_base = allocator.get_base()
            lds_addr = _lds_base_addr_i32(lds_base)
            s_Aq_addr = arith.unwrap(arith.addi(lds_addr, _i32(lds_off)))
            s_Ascale_addr = arith.unwrap(arith.addi(s_Aq_addr, _i32(s_Ascale_off)))
            lds_acc_addr = s_Aq_addr  # union: lds_acc aliases s_Aq

            m_indices_addr = _tensor_addr_i64(arg_m_indices)

            # ---- cached row gather (gemm1_a4w4.cuh:382-410) ----
            # Non-inline-quant path: cached_actual_row[sub] (BM>=32; BM==16 wave<2
            # special is not in the CSV).  Inline-quant path: cached_row_inline[s].
            cached_actual_row = []
            cached_row_inline = []
            if const_expr(not kInlineQuant):
                row_off = lane // 8  # lane / 8
                lds_row0 = wave * (BM // 4)  # wave * 8
                for sub in range_constexpr(kSubBlocks):
                    idx = m_row + lds_row0 + sub * 8 + row_off
                    ptr = _global_ptr(m_indices_addr, (idx * 4).ir_value())
                    cached_actual_row.append(fx.Int32(llvm.load(T.i32, ptr, alignment=4)))
            else:
                kCachedInline = 1 if BM == 16 else 2
                rcls = wave * 4 + (lane // 16)
                for s in range_constexpr(kCachedInline):
                    idx = m_row + s * 16 + rcls
                    ptr = _global_ptr(m_indices_addr, (idx * 4).ir_value())
                    cached_row_inline.append(fx.Int32(llvm.load(T.i32, ptr, alignment=4)))

            # ---- b_load_s_base[0..3], b_scale_s_base[0..1], *_hi ----
            b_load_s_base = []
            for j in range_constexpr(4):
                base = (e * N_OUT + n_block_idx * BN + wave_n * (BN // 4) + j * 16) * (K // 2)
                b_load_s_base.append(fx.Int32(_readfirstlane(base.ir_value())))

            mni_base = n_block_idx * (BN // 16 // 2) + wave_n * (BN // 64 // 2)
            b_scale_s_base = []
            b_scale_s_base_hi = []
            for mw in range_constexpr(2):
                base = (e * kBS_per_expert_dw + (mni_base + mw) * kBS_stride_n0_dw) * 4
                b0 = fx.Int32(_readfirstlane(base.ir_value()))
                b_scale_s_base.append(b0)
                bhi = b0 + 16 * (kBS_stride_k0_dw * 4)
                b_scale_s_base_hi.append(fx.Int32(_readfirstlane(bhi.ir_value())))

            # ----------------------------------------------------------------
            # issue_a_scale_load(m_row): direct A-scale global -> LDS
            # ----------------------------------------------------------------
            def issue_a_scale_load():
                kAS_chunk_bytes = kAS_per_chunk_dw * 4
                chunk_base_BM32 = m_row // 32
                v_voff_dx4 = (wave * 64 + lane) * 16
                v_voff_dw = (wave * 64 + lane) * 4
                for sub in range_constexpr(kSubBlocks):
                    s_chunk_base = fx.Int32(_readfirstlane(((chunk_base_BM32 + sub) * kAS_per_chunk_dw * 4).ir_value()))
                    lds_sub_off = sub * kAS_chunk_bytes
                    dst = _lds_ptr(s_Ascale_addr, (lds_sub_off + wave * 1024).ir_value())
                    _buffer_load_lds(A_scale_rsrc, dst, 16, v_voff_dx4.ir_value(), s_chunk_base.ir_value())
                    for d in range_constexpr(3):
                        byte_off = 4096 + d * 1024
                        s_off = fx.Int32(_readfirstlane((s_chunk_base + byte_off).ir_value()))
                        dst2 = _lds_ptr(s_Ascale_addr, (lds_sub_off + byte_off + wave * 256).ir_value())
                        _buffer_load_lds(A_scale_rsrc, dst2, 4, v_voff_dw.ir_value(), s_off.ir_value())

            # ----------------------------------------------------------------
            # issue_a_load_lds(slot, kt): direct A_q global -> LDS (BM=32)
            # ----------------------------------------------------------------
            def issue_a_load_lds(slot, kt):
                kRowsPerChunk = 8
                kLanesPerRow = 8
                row_off_ = lane // kLanesPerRow
                for sub in range_constexpr(kSubBlocks):
                    lds_row = wave * (BM // 4) + sub * kRowsPerChunk
                    # lds_swizzle_mask<BK/2=128>(lds_row + row_off)
                    mask = ((lds_row + row_off_) & 0xE) << 3
                    voffset = ((lane % kLanesPerRow) * 16 ^ mask) + cached_actual_row[sub] * (K // 2)
                    # dest byte offset: &s_Aq[slot][lds_row][0]
                    dst_byte = slot * (BM * (BK // 2)) + lds_row * (BK // 2)
                    dst = _lds_ptr(s_Aq_addr, _i32(dst_byte) if isinstance(dst_byte, int) else dst_byte.ir_value())
                    _buffer_load_lds(A_q_rsrc, dst, 16, voffset.ir_value(), _i32(kt * (BK // 2)))

            # ----------------------------------------------------------------
            # issue_a_ds_read(lds_slot) -> a[i][k] (kMChunks x 2) of v4i32
            # ----------------------------------------------------------------
            def issue_a_ds_read(lds_slot):
                lane_row = lane % 16
                lane_col = (lane // 16) * 16
                mask = (lane_row & 0xE) << 3
                a = [[None, None] for _ in range(kMChunks)]
                for k in range_constexpr(2):
                    lds_col = (lane_col + k * 64) ^ mask
                    for i in range_constexpr(kMChunks):
                        lds_row = lane_row + i * 16
                        byte_off = lds_slot * (BM * (BK // 2)) + lds_row * (BK // 2) + lds_col
                        v = _lds_load_vec(s_Aq_addr, byte_off.ir_value(), _V4I32(), 16)
                        a[i][k] = v
                return a

            # ----------------------------------------------------------------
            # issue_a_scale_ds_read(kt) -> a_scale_aiter[sub] (i32)
            # ----------------------------------------------------------------
            def issue_a_scale_ds_read(kt):
                out = []
                for sub in range_constexpr(kSubBlocks):
                    lds_dw = sub * kAS_per_chunk_dw + kt * 64 + (lane // 16) * 16 + (lane % 16)
                    byte_off = lds_dw * 4
                    v = _lds_load_vec(s_Ascale_addr, byte_off.ir_value(), T.i32, 4)
                    out.append(v)
                return out

            # ----------------------------------------------------------------
            # issue_b_load_j(K_C, j) -> (b[j][0], b[j][1]) v4i32 in registers
            # ----------------------------------------------------------------
            def issue_b_load_j(K_C, j):
                K_BYTE = K_C * 2048
                v_voff = (lane // 16) * 256 + (lane % 16) * 16 + K_BYTE
                aux = 2 if kUseNT else 0
                b0 = _raw_buffer_load_b128(B_q_rsrc, v_voff.ir_value(), b_load_s_base[j].ir_value(), aux)
                v_voff1 = v_voff + 1024  # 12-bit MUBUF inst_offset folded into voffset
                b1 = _raw_buffer_load_b128(B_q_rsrc, v_voff1.ir_value(), b_load_s_base[j].ir_value(), aux)
                return [b0, b1]

            # ----------------------------------------------------------------
            # issue_b_scale_load(K_C) -> bs[0..1] (i32)
            # ----------------------------------------------------------------
            def issue_b_scale_load(K_C):
                v_voff = ((lane // 16) * 16 + (lane % 16)) * 4
                K_C_HI = K_C // 16
                IMM = (K_C - K_C_HI * 16) * (kBS_stride_k0_dw * 4)
                out = []
                for mw in range_constexpr(2):
                    s_base = b_scale_s_base[mw] if K_C_HI == 0 else b_scale_s_base_hi[mw]
                    vo = v_voff + IMM
                    out.append(_raw_buffer_load_b32(B_scale_rsrc, vo.ir_value(), s_base.ir_value()))
                return out

            # ----------------------------------------------------------------
            # inline-quant producer (kInlineQuant, BM=16): bf16 hidden -> fp4+e8m0
            # ----------------------------------------------------------------
            def inline_quant_load_kt(B128_IDX, kt, row_token):
                # gemm1_a4w4.cuh:197-207 — bf16 b128 load from hidden_rsrc.
                v_voff = row_token * (K * 2) + ((lane // 4) % 4) * 64 + (lane % 4) * 16
                # ``kt`` and ``B128_IDX`` come from range_constexpr / Python ints, so
                # the offset is a plain Python int — _readfirstlane/_i32 handle that.
                s_soff = fx.Int32(_readfirstlane(kt * (BK * 2) + B128_IDX * 256))
                return _raw_buffer_load_b128(hidden_rsrc, v_voff.ir_value(), s_soff.ir_value(), 0)

            def _inline_quant_body(B128_IDX, SUB, slot, kt, h_v, scale_accum):
                # gemm1_a4w4.cuh:222-258 (steps 2-8). SUB is always 0 for BM=16.
                h_dw = [_vec_extract(h_v, j, T.i32) for j in range(4)]
                hm = [arith.unwrap(arith.andi(h_dw[j], _i32(0x7FFF7FFF))) for j in range(4)]
                m01 = _pkmax_u16(hm[0], hm[1])
                m23 = _pkmax_u16(hm[2], hm[3])
                m0123 = _pkmax_u16(m01, m23)
                lo = arith.unwrap(arith.andi(m0123, _i32(0xFFFF)))
                hi = arith.unwrap(arith.shrui(m0123, _i32(16)))
                local_amax = _max_u32(lo, hi)
                amax_u32 = _dpp_quad_amax_u32(local_amax)
                e8m0 = _encode_e8m0(amax_u32)  # fx.Int32
                qs_bits = (e8m0 << 23).ir_value()
                qs = arith.unwrap(arith.bitcast(T.f32, qs_bits))

                v2bf16 = ir.VectorType.get([2], T.bf16)
                pk = _i32(0)
                for d in range_constexpr(4):
                    src = _bitcast(v2bf16, h_dw[d])
                    pk = rocdl.cvt_scalef32_pk_fp4_bf16(T.i32, pk, src, qs, d)

                lib = lane % 4
                r_in_chunk = wave * 4 + (lane // 16)
                r = SUB * 16 + r_in_chunk
                kb_in_kt = B128_IDX * 4 + ((lane // 4) % 4)
                mask_r = (r & 0xE) << 3
                b_off = lib * 4
                dst_byte = slot * (BM * (BK // 2)) + r * (BK // 2) + (((kb_in_kt * 16) ^ mask_r) + b_off)
                _lds_store(s_Aq_addr, dst_byte.ir_value(), pk, 4)

                # kPackScale=True always for BM=16: scale_accum |= e8m0 << (pack_byte*8)
                pack_byte = B128_IDX * 2 + SUB
                contrib = (e8m0 & 0xFF) << (pack_byte * 8)
                return scale_accum | contrib

            def inline_quant_kt(B128_IDX, SUB, slot, kt, row_token, scale_accum):
                h_v = inline_quant_load_kt(B128_IDX, kt, row_token)
                return _inline_quant_body(B128_IDX, SUB, slot, kt, h_v, scale_accum)

            def inline_quant_finish_kt(B128_IDX, SUB, slot, kt, h_v, scale_accum):
                return _inline_quant_body(B128_IDX, SUB, slot, kt, h_v, scale_accum)

            def inline_quant_pack_write(kt, scale_accum):
                # gemm1_a4w4.cuh:261-266
                r_in_chunk = wave * 4 + (lane // 16)
                lane_tgt = ((lane // 4) % 4) * 16 + r_in_chunk
                byte_off = kt * 256 + lane_tgt * 4
                _lds_store(s_Ascale_addr, byte_off.ir_value(), _i32(scale_accum), 4)

            # ----------------------------------------------------------------
            # issue_mfma_cluster(J, kInit, slot): 4 MFMA per J (BM=32 VGPR)
            #   accm[i][J] is f32x4 SSA; a is list, b_slot is [J][half], bs slot
            # ----------------------------------------------------------------
            def issue_mfma_cluster(J, kInit, a, b_slot, a_scale, b_scale_slot, accm):
                in_b = J % 2
                mni = J // 2
                if const_expr(BM == 16):
                    # gemm1_a4w4.cuh:338-345 single-chunk branch: op_sel_a in {0,2}.
                    sa = a_scale[0]
                    sb = b_scale_slot[mni]
                    b0_v8 = _pack_v8i32_from_v4i32(b_slot[J][0])
                    a_0 = _pack_v8i32_from_v4i32(a[0][0])
                    c0 = _zero_v4f32() if kInit else accm[0][J]
                    accm[0][J] = _mfma_scale(a_0, b0_v8, c0, 0, 0 + in_b, sa, sb)
                    b1_v8 = _pack_v8i32_from_v4i32(b_slot[J][1])
                    a_1 = _pack_v8i32_from_v4i32(a[0][1])
                    accm[0][J] = _mfma_scale(a_1, b1_v8, accm[0][J], 2, 2 + in_b, sa, sb)
                    return
                # BM in {32, 64, 128}: 4 MFMA per (sub, J).  For BM==128 HIP pins C/D
                # into AccVGPRs via inline asm; we use the VGPR intrinsic (option A) —
                # same hardware MFMA, the backend keeps acc in (A)VGPRs at occ 1.
                for sub in range_constexpr(kSubBlocks):
                    sa = a_scale[sub]
                    sb = b_scale_slot[mni]
                    i0 = sub * 2 + 0
                    i1 = sub * 2 + 1
                    b0_v8 = _pack_v8i32_from_v4i32(b_slot[J][0])
                    a_i0_0 = _pack_v8i32_from_v4i32(a[i0][0])
                    a_i1_0 = _pack_v8i32_from_v4i32(a[i1][0])
                    # ``kInit`` is a plain Python bool (computed from the
                    # range_constexpr OFFSET *outside* this call), so a single
                    # straight-line conditional expression always binds c0/c1.
                    c0 = _zero_v4f32() if kInit else accm[i0][J]
                    c1 = _zero_v4f32() if kInit else accm[i1][J]
                    accm[i0][J] = _mfma_scale(a_i0_0, b0_v8, c0, 0, 0 + in_b, sa, sb)
                    accm[i1][J] = _mfma_scale(a_i1_0, b0_v8, c1, 1, 0 + in_b, sa, sb)
                    b1_v8 = _pack_v8i32_from_v4i32(b_slot[J][1])
                    a_i0_1 = _pack_v8i32_from_v4i32(a[i0][1])
                    a_i1_1 = _pack_v8i32_from_v4i32(a[i1][1])
                    accm[i0][J] = _mfma_scale(a_i0_1, b1_v8, accm[i0][J], 2, 2 + in_b, sa, sb)
                    accm[i1][J] = _mfma_scale(a_i1_1, b1_v8, accm[i1][J], 3, 2 + in_b, sa, sb)

            # ================================================================
            # run_one body
            # ================================================================
            accm = [[_zero_v4f32() for _ in range(4)] for _ in range(kMChunks)]
            b_reg = [[None, None, None, None] for _ in range(kStages)]
            b_scale_v = [None for _ in range(kStages)]

            if const_expr(not kInlineQuant):
                issue_a_scale_load()

            # prologue: kStages (gemm1_a4w4.cuh:434-463)
            for K_C in range_constexpr(kStages):
                if const_expr(kInlineQuant):
                    # BM=16 IQ else-branch (:447-455): two inline-quant calls (SUB=0,
                    # B128_IDX 0,1) interleaved with the four B-loads.
                    scale_accum = fx.Int32(0)
                    scale_accum = inline_quant_kt(0, 0, K_C, K_C, cached_row_inline[0], scale_accum)
                    b_reg[K_C][0] = issue_b_load_j(K_C, 0)
                    b_reg[K_C][1] = issue_b_load_j(K_C, 1)
                    scale_accum = inline_quant_kt(1, 0, K_C, K_C, cached_row_inline[0], scale_accum)
                    b_reg[K_C][2] = issue_b_load_j(K_C, 2)
                    b_reg[K_C][3] = issue_b_load_j(K_C, 3)
                    inline_quant_pack_write(K_C, scale_accum)
                else:
                    issue_a_load_lds(K_C, K_C)
                    for j in range_constexpr(4):
                        b_reg[K_C][j] = issue_b_load_j(K_C, j)
                b_scale_v[K_C] = issue_b_scale_load(K_C)

            # main loop: kUnroll (fully unrolled to match HIP static_for)
            for OFFSET in range_constexpr(kUnroll):
                K_C = kStages + OFFSET
                read_slot = OFFSET % kAStages
                write_slot = K_C % kAStages
                slot_b = OFFSET % kStages

                gpu.barrier()
                a = issue_a_ds_read(read_slot)
                a_scale = issue_a_scale_ds_read(K_C - kStages)
                if const_expr(not kInlineQuant):
                    issue_a_load_lds(write_slot, K_C)

                if const_expr(kInlineQuant):
                    # BM=16 IQ else-branch (:522-551): prefetch bf16, MFMA, then
                    # finish-quant overlapped with the MFMA pipeline.
                    h_v0 = inline_quant_load_kt(0, K_C, cached_row_inline[0])
                    h_v1 = inline_quant_load_kt(1, K_C, cached_row_inline[0])
                    _sched_barrier()
                    for J in range_constexpr(4):
                        _sched_barrier()
                        _setprio(1)
                        issue_mfma_cluster(J, OFFSET == 0, a, b_reg[slot_b], a_scale, b_scale_v[slot_b], accm)
                        _setprio(0)
                        _sched_barrier()
                        b_reg[slot_b][J] = issue_b_load_j(K_C, J)
                        _sched_barrier()
                    b_scale_v[slot_b] = issue_b_scale_load(K_C)
                    scale_accum = fx.Int32(0)
                    scale_accum = inline_quant_finish_kt(0, 0, write_slot, K_C, h_v0, scale_accum)
                    scale_accum = inline_quant_finish_kt(1, 0, write_slot, K_C, h_v1, scale_accum)
                    inline_quant_pack_write(K_C, scale_accum)
                else:
                    # gemm1_a4w4.cuh:529-543: the leading sched_barrier + s_setprio
                    # fences around the MFMA cluster are emitted only for BM != 128.
                    for J in range_constexpr(4):
                        if const_expr(BM != 128):
                            _sched_barrier()
                            _setprio(1)
                        issue_mfma_cluster(J, OFFSET == 0, a, b_reg[slot_b], a_scale, b_scale_v[slot_b], accm)
                        if const_expr(BM != 128):
                            _setprio(0)
                        _sched_barrier()
                        b_reg[slot_b][J] = issue_b_load_j(K_C, J)
                        _sched_barrier()
                    b_scale_v[slot_b] = issue_b_scale_load(K_C)

            # drain: kStages
            for S in range_constexpr(kStages):
                kt = K_TILES_TOTAL - kStages + S
                read_slot_a = kt % kAStages
                slot_b_drain = kt % kStages
                gpu.barrier()
                a = issue_a_ds_read(read_slot_a)
                a_scale = issue_a_scale_ds_read(kt)
                for J in range_constexpr(4):
                    issue_mfma_cluster(J, False, a, b_reg[slot_b_drain], a_scale, b_scale_v[slot_b_drain], accm)

            gpu.barrier()

            # ----------------------------------------------------------------
            # epilog: apply_cshuffle_quant_epilog<N_OUT, BM>
            # ----------------------------------------------------------------
            _epilog(
                accm,
                arg_A_q_out,
                arg_A_scale_out,
                m_block_idx,
                m_row,
                n_block_idx,
                wave,
                wave_n,
                lane,
                tid,
                lds_acc_addr,
            )

            scf_yield()

    # =====================================================================
    # epilog helper (lives in the kernel-trace scope via closure constants)
    # =====================================================================
    def _epilog(
        accm, arg_aq_out, arg_ascale_out, m_block_idx, m_row, n_block_idx, wave, wave_n, lane, tid, lds_acc_addr
    ):
        EVec = 8
        M_REPS = BM // 16
        kSubBlocks_epi = 1 if BM < 32 else BM // 32

        # store accumulators into lds_acc[(row)*BN + col]
        for i in range_constexpr(BM // 16):
            row_base = i * 16 + (lane // 16) * 4
            for J in range_constexpr(4):
                is_up = (J % 2) == 1
                J_local = J // 2
                col_local = wave_n * 32 + J_local * 16 + (lane % 16)
                lds_col = (col_local + 128) if is_up else col_local
                for v in range_constexpr(4):
                    fval = _vec_extract(accm[i][J], v, T.f32)
                    byte_off = ((row_base + v) * BN + lds_col) * 4
                    _lds_store(lds_acc_addr, byte_off.ir_value(), fval, 4)

        gpu.barrier()

        NLane = 16
        m_lane = tid // NLane
        n_lane = tid % NLane
        wave_grp = n_lane // 4
        kk = n_lane % 4

        aq_out_addr = _tensor_addr_i64(arg_aq_out)
        ascale_out_addr = _tensor_addr_i64(arg_ascale_out)

        scales_per_mr = []
        for mr in range_constexpr(M_REPS):
            row_local = mr * 16 + m_lane

            results = []
            for e in range_constexpr(EVec):
                col_in_grp = 8 * kk + e
                gate_col = wave_grp * 32 + col_in_grp
                up_col = 128 + gate_col
                g = _lds_load_vec(lds_acc_addr, ((row_local * BN + gate_col) * 4).ir_value(), T.f32, 4)
                u = _lds_load_vec(lds_acc_addr, ((row_local * BN + up_col) * 4).ir_value(), T.f32, 4)
                results.append(_silu_mul_fast(g, u))

            # local_max = max_e |result[e]|
            local_max = _fabs(results[0])
            for e in range(1, EVec):
                local_max = _fmax(local_max, _fabs(results[e]))
            # DPP quad amax
            lm_i32 = arith.unwrap(arith.bitcast(T.i32, local_max))
            peer1 = _update_dpp(lm_i32, lm_i32, 0xB1)
            local_max = _fmax(local_max, arith.unwrap(arith.bitcast(T.f32, peer1)))
            lm_i32 = arith.unwrap(arith.bitcast(T.i32, local_max))
            peer2 = _update_dpp(lm_i32, lm_i32, 0x4E)
            local_max = _fmax(local_max, arith.unwrap(arith.bitcast(T.f32, peer2)))

            amax_i32 = fx.Int32(arith.unwrap(arith.bitcast(T.i32, local_max)))
            qscale_bits = amax_i32 + 0x200000
            quant_scale_f = arith.unwrap(arith.bitcast(T.f32, qscale_bits.ir_value()))
            quant_scale = _fmul(quant_scale_f, _f32(0.25))
            qs_bits = fx.Int32(arith.unwrap(arith.bitcast(T.i32, quant_scale)))
            sb_raw = qs_bits >> 23
            sb = _min_u(sb_raw, 254)
            scales_per_mr.append(sb)

            packed = _i32(0)
            packed = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, packed, results[0], results[1], quant_scale, 0)
            packed = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, packed, results[2], results[3], quant_scale, 1)
            packed = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, packed, results[4], results[5], quant_scale, 2)
            packed = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, packed, results[6], results[7], quant_scale, 3)

            byte_pos = n_block_idx * (BN_INT // 2) + wave_grp * 16 + kk * 4
            out_row = m_row + row_local
            out_byte = out_row * K_G2_HALF + byte_pos
            optr = _global_ptr(aq_out_addr, out_byte.ir_value())
            llvm.store(packed, optr, alignment=4, nontemporal=True)

        # scale store (kk == 0)
        kk_is0 = arith.unwrap(arith.cmpi(CmpIPredicate.eq, kk.ir_value(), _i32(0)))
        if_kk = scf_if_begin(kk_is0)
        with ir.InsertionPoint(if_kk.then_block):
            ku = n_block_idx >> 1
            ikxdl = n_block_idx & 1
            if BM == 16:
                # mxfp4_epilogs.hpp:127-133 — single LOW-byte store, chunk=m_block_idx.
                chunk = m_block_idx
                dword_off = chunk * epi_kAS_per_chunk_dw + ku * 64 + wave_grp * 16 + m_lane
                byte = scales_per_mr[0] & 0xFF
                byte8 = arith.unwrap(arith.trunci(T.i8, byte.ir_value()))
                addr_bytes = dword_off * 4 + ikxdl * 2
                sptr = _global_ptr(ascale_out_addr, addr_bytes.ir_value())
                llvm.store(byte8, sptr, alignment=1)
            else:
                for sub in range_constexpr(kSubBlocks_epi):
                    chunk = m_block_idx * kSubBlocks_epi + sub
                    dword_off = chunk * epi_kAS_per_chunk_dw + ku * 64 + wave_grp * 16 + m_lane
                    lo = scales_per_mr[sub * 2 + 0]
                    hi = scales_per_mr[sub * 2 + 1]
                    pair = (lo & 0xFF) | ((hi & 0xFF) << 8)
                    pair16 = arith.unwrap(arith.trunci(T.i16, pair.ir_value()))
                    addr_bytes = dword_off * 4 + ikxdl * 2
                    sptr = _global_ptr(ascale_out_addr, addr_bytes.ir_value())
                    llvm.store(pair16, sptr, alignment=2)
            scf_yield()

    # ---- Host launcher ----
    @flyc.jit
    def launch_gemm1(
        arg_A_q: fx.Tensor,
        arg_A_scale: fx.Tensor,
        arg_B_q: fx.Tensor,
        arg_B_scale: fx.Tensor,
        arg_sorted_expert_ids: fx.Tensor,
        arg_cumsum: fx.Tensor,
        arg_m_indices: fx.Tensor,
        i32_n_tokens: fx.Int32,
        arg_A_q_out: fx.Tensor,
        arg_A_scale_out: fx.Tensor,
        arg_hidden_states: fx.Tensor,
        i32_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # The test passes the BM-correct grid (HIP launch():595-620):
        #   TOPK=9, num_n_blocks=N_OUT/256
        #   BM==128 : max_m_blocks = (n*TOPK + NE*(BM-1) + BM-1)/BM
        #   else    : active = min(n*TOPK, NE)
        #             max_m_blocks = (n*TOPK + active*(BM-1) + BM-1)/BM
        #   grid    = max_m_blocks * num_n_blocks
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        kernel_gemm1._func.__name__ = KERNEL_NAME
        launcher = kernel_gemm1(
            arg_A_q,
            arg_A_scale,
            arg_B_q,
            arg_B_scale,
            arg_sorted_expert_ids,
            arg_cumsum,
            arg_m_indices,
            i32_n_tokens,
            arg_A_q_out,
            arg_A_scale_out,
            arg_hidden_states,
        )
        launcher.launch(grid=(i32_grid, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm1


# ---------------------------------------------------------------------------
# math / scf helpers used inside the kernel trace
# ---------------------------------------------------------------------------


def _zero_v4f32():
    z = arith.unwrap(arith.constant(0.0, type=T.f32))
    out = llvm.mlir_undef(_V4F32())
    for i in range(4):
        pos = arith.unwrap(arith.constant(i, type=T.i64))
        out = llvm.insertelement(out, z, pos)
    return out


def _silu_mul_fast(g, u):
    # silu_mul_fast(g,u) = g * rcp(1 + exp(-g)) * u, using exp2(x*log2e) and HW rcp.
    log2e = _f32(1.4426950408889634)
    neg_g = arith.unwrap(arith.subf(_f32(0.0), g))
    x = arith.unwrap(arith.mulf(neg_g, log2e))
    e = rocdl.exp2(T.f32, x)
    one = _f32(1.0)
    denom = arith.unwrap(arith.addf(one, e))
    inv = rocdl.rcp(T.f32, denom)
    t = arith.unwrap(arith.mulf(g, inv))
    return arith.unwrap(arith.mulf(t, u))


def _fabs(v):
    return llvm.intr_fabs(v)


def _fmax(a, b):
    return llvm.intr_maxnum(a, b)


def _fmul(a, b):
    return arith.unwrap(arith.mulf(a, b))


def _min_u(a, b_const):
    bc = fx.Int32(b_const)
    cond = arith.unwrap(arith.cmpi(CmpIPredicate.ult, a.ir_value(), bc.ir_value()))
    sel = arith.unwrap(arith.select(cond, a.ir_value(), bc.ir_value()))
    return fx.Int32(sel)


def _update_dpp(old, src, ctrl, row_mask=0xF, bank_mask=0xF, bound_ctrl=True):
    return llvm.call_intrinsic(
        T.i32,
        "llvm.amdgcn.update.dpp.i32",
        [
            _i32(old),
            _i32(src),
            arith.unwrap(arith.constant(ctrl, type=T.i32)),
            arith.unwrap(arith.constant(row_mask, type=T.i32)),
            arith.unwrap(arith.constant(bank_mask, type=T.i32)),
            arith.unwrap(arith.constant(1 if bound_ctrl else 0, type=ir.IntegerType.get_signless(1))),
        ],
        [],
        [],
    )


# ---------------------------------------------------------------------------
# inline-quant (bf16 -> fp4 + e8m0) helpers (mxfp4_gemm_common.hpp:76-94)
# ---------------------------------------------------------------------------


def _pkmax_u16(a, b):
    """v_pk_max_u16: element-wise max of two packed vector<2xi16> in i32 dwords.

    HIP uses inline asm ``v_pk_max_u16``; ``arith.maxui`` on ``vector<2xi16>``
    lowers to the same instruction on gfx950.
    """
    v2i16 = ir.VectorType.get([2], T.i16)
    av = _bitcast(v2i16, a)
    bv = _bitcast(v2i16, b)
    mv = arith.unwrap(arith.maxui(av, bv))
    return llvm.bitcast(T.i32, mv)


def _max_u32(a, b):
    cond = arith.unwrap(arith.cmpi(CmpIPredicate.ugt, _i32(a), _i32(b)))
    return arith.unwrap(arith.select(cond, _i32(a), _i32(b)))


def _dpp_quad_amax_u32(a32):
    """inline_quant_dpp_quad_amax: two update.dpp (0xB1, 0x4E) + unsigned max."""
    s1 = _update_dpp(a32, a32, 0xB1)
    a32 = _max_u32(a32, s1)
    s2 = _update_dpp(a32, a32, 0x4E)
    return _max_u32(a32, s2)


def _encode_e8m0(amax_u16):
    """inline_quant_encode_e8m0: amax(u16)->e8m0 byte (min(254,max(0,bexp-2)))."""
    amax = fx.Int32(_i32(amax_u16)) & 0xFFFF
    f32bits = amax << 16
    bexp = ((f32bits + 0x200000) >> 23) & 0xFF
    # max(0, bexp - 2)
    bm2 = bexp - 2
    zero = fx.Int32(0)
    c0 = arith.unwrap(arith.cmpi(CmpIPredicate.sgt, bm2.ir_value(), zero.ir_value()))
    m0 = fx.Int32(arith.unwrap(arith.select(c0, bm2.ir_value(), zero.ir_value())))
    return _min_u(m0, 254)


# ---------------------------------------------------------------------------
# scf if/yield convenience
# ---------------------------------------------------------------------------


def scf_if_begin(cond_i1):
    from flydsl._mlir.dialects import scf

    return scf.IfOp(cond_i1, [], has_else=False)


def scf_yield():
    from flydsl._mlir.dialects import scf

    scf.YieldOp([])
