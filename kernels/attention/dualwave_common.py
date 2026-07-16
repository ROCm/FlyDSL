# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared module-level helpers for the gfx950 dual-wave, software-pipelined
flash-attention kernels.

These MLIR-dialect-facing free functions and the ``s_waitcnt`` bit-field
constants were previously duplicated verbatim across ``flash_attn_gfx950``
(bf16/f16) and ``flash_attn_fp8_gfx950`` (fp8); ``_LOG2E`` / ``_waitcnt_vm_n``
are also shared with ``flash_attn_generic``. Moving them here changes nothing
about the emitted IR/ISA -- it only removes the duplication.
"""

import math as host_math
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, vector
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace as _TargetAddressSpace
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from kernels.common.kernels_common import dtype_to_elem_type

_LOG2E = host_math.log2(host_math.e)
# s_waitcnt bitfield encoding
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3
scf_if_dispatch = ReplaceIfWithDispatch.scf_if_dispatch

_LDS_ALIAS_DOMAIN = '#llvm.alias_scope_domain<id = "flydsl.dualwave_swp.lds">'


@dataclass(frozen=True)
class DualwaveSwpTraits:
    """Pure compile-time tile/layout constants for gfx950 DUALWAVE_SWP."""

    BLOCK_M: int
    BLOCK_N: int
    BLOCK_N_OUT: int
    K_SUB_N: int
    WARP_SIZE: int
    NUM_WAVES: int
    BLOCK_SIZE: int
    ROWS_PER_WAVE: int
    HEAD_DIM: int
    K_STEP_QK: int
    K_STEPS_QK: int
    D_CHUNK: int
    D_CHUNKS: int
    PV_K_STEP: int
    PV_K_STEPS: int
    MFMA_LANE_K: int
    NUM_HEADS_Q: int
    NUM_HEADS_KV: int
    GQA_GROUP_SIZE: int
    CAUSAL: bool
    DTYPE_STR: str
    WAVES_PER_EU: int
    DAZ: bool
    DUALWAVE_SWP_LAZY_RESCALE: bool
    DUALWAVE_SWP_SETPRIO: bool
    DUALWAVE_SWP_DEBUG_LAZY_COUNTS: bool
    DUALWAVE_SWP_ENABLE_STAGGER: bool
    NUM_KV_SPLITS: int
    SPLITK: bool
    PAGED: bool
    VARLEN: bool
    CROSS_SEQLEN: bool
    KV_CACHE_LAYOUT: str
    KV_VECTORIZED: bool
    DEFAULT_STRIDE_Q_N: int
    DEFAULT_STRIDE_KV_N: int
    DMA_BYTES: int
    BF16_BYTES: int
    D_128B_SIZE: int
    VEC_KV: int
    SMEM_LINEAR_WAVE: int
    SMEM_N_PER_WAVE: int
    SMEM_N_RPT: int
    SMEM_D_RPT: int
    SMEM_K_PAD: int
    SMEM_V_PAD: int
    SMEM_K_LINE_STRIDE: int
    SMEM_V_LINE_STRIDE: int
    SMEM_K_TILE_ELEMS: int
    SMEM_V_TILE_ELEMS: int
    NUM_PREFETCH_K: int
    DUALWAVE_SWP_KV_PER_BUFFER: int
    LDS_KV_TOTAL_SIZE: int
    DUALWAVE_SWP_K_BUF_BASE: tuple[int, int]
    DUALWAVE_SWP_V_BUF_BASE: tuple[int, int]
    DUALWAVE_SWP_URK_N_STRIP_STRIDE: int
    DUALWAVE_SWP_URK_KSTEP_INNER: int
    DUALWAVE_SWP_URK_KSTEP_OUTER: int
    DUALWAVE_SWP_URV_GRPK: int
    DUALWAVE_SWP_URV_LANE_HI: int
    DUALWAVE_SWP_URV_GRP_N: int
    DUALWAVE_SWP_URV_LANE_LO: int
    DUALWAVE_SWP_URV_STEP_K_STRIDE: int
    DUALWAVE_SWP_URV_DC_AXIS0: int
    DUALWAVE_SWP_URV_DC_AXIS1: int
    DUALWAVE_SWP_URV_I5_STRIDE: int
    PAGED_BT_LDS_SIZE: int
    DUALWAVE_SWP_RESCALE_THRESHOLD: float
    KV_VEC_SIZE: int
    VEC_V_ROW_STRIDE: int
    SCHED_MFMA_MASK: int
    SCHED_VALU_MASK: int
    SCHED_EXP_MASK: int
    LDS_SCOPE_NAMES: tuple[str, str, str, str]
    NEG_INF_F32_BITS: int
    LGKMCNT_0_ONLY: int


def _make_dualwave_swp_traits(
    num_heads,
    num_kv_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    waves_per_eu=2,
    daz=True,
    dualwave_swp_lazy_rescale=True,
    dualwave_swp_setprio=True,
    dualwave_swp_debug_lazy_counts=False,
    dualwave_swp_enable_stagger=True,
    num_kv_splits=1,
    varlen=False,
    cross_seqlen=False,
    paged=False,
    kv_cache_layout="linear",
    kv_vectorized=None,
):
    """Build gfx950 DUALWAVE_SWP compile-time layout traits."""
    block_m = 256
    block_n = 64
    block_n_out = 64
    k_sub_n = 32
    warp_size = 64
    num_waves = 8
    block_size = num_waves * warp_size
    rows_per_wave = 32

    k_step_qk = 16
    k_steps_qk = head_dim // k_step_qk
    d_chunk = 32
    d_chunks = head_dim // d_chunk
    pv_k_step = 16
    pv_k_steps = k_sub_n // pv_k_step
    mfma_lane_k = 8

    gqa_group_size = num_heads // num_kv_heads
    default_stride_q_n = num_heads * head_dim
    default_stride_kv_n = num_kv_heads * head_dim

    dma_bytes = 16
    bf16_bytes = 2
    d_128b_size = 64
    vec_kv = 8
    smem_linear_wave = warp_size * 16 // bf16_bytes
    smem_n_per_wave = smem_linear_wave // d_128b_size
    smem_n_rpt = block_n // smem_n_per_wave
    smem_d_rpt = head_dim // d_128b_size
    smem_k_pad = 16 // bf16_bytes
    smem_v_pad = 64 // bf16_bytes
    smem_k_line_stride = smem_linear_wave + smem_k_pad
    smem_v_line_stride = smem_linear_wave + smem_v_pad
    smem_k_tile_elems = smem_n_rpt * smem_d_rpt * smem_k_line_stride
    smem_v_tile_elems = smem_n_rpt * smem_d_rpt * smem_v_line_stride
    num_prefetch_k = 2
    dualwave_swp_kv_per_buffer = smem_k_tile_elems + smem_v_tile_elems
    lds_kv_total_size = num_prefetch_k * dualwave_swp_kv_per_buffer
    dualwave_swp_k_buf_base = (0, dualwave_swp_kv_per_buffer)
    dualwave_swp_v_buf_base = (
        smem_k_tile_elems,
        smem_k_tile_elems + dualwave_swp_kv_per_buffer,
    )

    dualwave_swp_urk_n_strip_stride = 256
    dualwave_swp_urk_kstep_inner = 16
    dualwave_swp_urk_kstep_outer = smem_n_rpt * smem_k_line_stride
    dualwave_swp_urv_grpk = 2176
    dualwave_swp_urv_lane_hi = smem_v_line_stride
    dualwave_swp_urv_grp_n = 16
    dualwave_swp_urv_lane_lo = 4
    dualwave_swp_urv_step_k_stride = 128
    dualwave_swp_urv_dc_axis0 = smem_n_rpt * smem_v_line_stride
    dualwave_swp_urv_dc_axis1 = 32
    dualwave_swp_urv_i5_stride = d_128b_size
    kv_vec_size = 16 // bf16_bytes
    vec_v_row_stride = smem_v_line_stride
    splitk = num_kv_splits > 1
    paged = bool(paged)
    varlen = bool(varlen)
    cross_seqlen = bool(cross_seqlen)

    return DualwaveSwpTraits(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_N_OUT=block_n_out,
        K_SUB_N=k_sub_n,
        WARP_SIZE=warp_size,
        NUM_WAVES=num_waves,
        BLOCK_SIZE=block_size,
        ROWS_PER_WAVE=rows_per_wave,
        HEAD_DIM=head_dim,
        K_STEP_QK=k_step_qk,
        K_STEPS_QK=k_steps_qk,
        D_CHUNK=d_chunk,
        D_CHUNKS=d_chunks,
        PV_K_STEP=pv_k_step,
        PV_K_STEPS=pv_k_steps,
        MFMA_LANE_K=mfma_lane_k,
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        CAUSAL=causal,
        DTYPE_STR=dtype_str,
        WAVES_PER_EU=waves_per_eu,
        DAZ=bool(daz),
        DUALWAVE_SWP_LAZY_RESCALE=bool(dualwave_swp_lazy_rescale),
        DUALWAVE_SWP_SETPRIO=bool(dualwave_swp_setprio),
        DUALWAVE_SWP_DEBUG_LAZY_COUNTS=bool(dualwave_swp_debug_lazy_counts),
        DUALWAVE_SWP_ENABLE_STAGGER=bool(dualwave_swp_enable_stagger),
        NUM_KV_SPLITS=num_kv_splits,
        SPLITK=splitk,
        PAGED=paged,
        VARLEN=varlen,
        CROSS_SEQLEN=cross_seqlen,
        KV_CACHE_LAYOUT=kv_cache_layout,
        KV_VECTORIZED=kv_vectorized,
        DEFAULT_STRIDE_Q_N=default_stride_q_n,
        DEFAULT_STRIDE_KV_N=default_stride_kv_n,
        DMA_BYTES=dma_bytes,
        BF16_BYTES=bf16_bytes,
        D_128B_SIZE=d_128b_size,
        VEC_KV=vec_kv,
        SMEM_LINEAR_WAVE=smem_linear_wave,
        SMEM_N_PER_WAVE=smem_n_per_wave,
        SMEM_N_RPT=smem_n_rpt,
        SMEM_D_RPT=smem_d_rpt,
        SMEM_K_PAD=smem_k_pad,
        SMEM_V_PAD=smem_v_pad,
        SMEM_K_LINE_STRIDE=smem_k_line_stride,
        SMEM_V_LINE_STRIDE=smem_v_line_stride,
        SMEM_K_TILE_ELEMS=smem_k_tile_elems,
        SMEM_V_TILE_ELEMS=smem_v_tile_elems,
        NUM_PREFETCH_K=num_prefetch_k,
        DUALWAVE_SWP_KV_PER_BUFFER=dualwave_swp_kv_per_buffer,
        LDS_KV_TOTAL_SIZE=lds_kv_total_size,
        DUALWAVE_SWP_K_BUF_BASE=dualwave_swp_k_buf_base,
        DUALWAVE_SWP_V_BUF_BASE=dualwave_swp_v_buf_base,
        DUALWAVE_SWP_URK_N_STRIP_STRIDE=dualwave_swp_urk_n_strip_stride,
        DUALWAVE_SWP_URK_KSTEP_INNER=dualwave_swp_urk_kstep_inner,
        DUALWAVE_SWP_URK_KSTEP_OUTER=dualwave_swp_urk_kstep_outer,
        DUALWAVE_SWP_URV_GRPK=dualwave_swp_urv_grpk,
        DUALWAVE_SWP_URV_LANE_HI=dualwave_swp_urv_lane_hi,
        DUALWAVE_SWP_URV_GRP_N=dualwave_swp_urv_grp_n,
        DUALWAVE_SWP_URV_LANE_LO=dualwave_swp_urv_lane_lo,
        DUALWAVE_SWP_URV_STEP_K_STRIDE=dualwave_swp_urv_step_k_stride,
        DUALWAVE_SWP_URV_DC_AXIS0=dualwave_swp_urv_dc_axis0,
        DUALWAVE_SWP_URV_DC_AXIS1=dualwave_swp_urv_dc_axis1,
        DUALWAVE_SWP_URV_I5_STRIDE=dualwave_swp_urv_i5_stride,
        PAGED_BT_LDS_SIZE=2048,
        DUALWAVE_SWP_RESCALE_THRESHOLD=8.0,
        KV_VEC_SIZE=kv_vec_size,
        VEC_V_ROW_STRIDE=vec_v_row_stride,
        SCHED_MFMA_MASK=0x008,
        SCHED_VALU_MASK=0x002,
        SCHED_EXP_MASK=0x400,
        LDS_SCOPE_NAMES=("lds_k0", "lds_k1", "lds_v0", "lds_v1"),
        NEG_INF_F32_BITS=0xFF800000,
        LGKMCNT_0_ONLY=0xC07F,
    )


def _fadd(a, b, fm_fast):
    return arith.addf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fsub(a, b, fm_fast):
    return arith.subf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fmul(a, b, fm_fast):
    return arith.mulf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fmax(a, b, fm_fast):
    return arith.MaxNumFOp(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast).result


def _mfma_acc(a, b, c, _mma_atom, mfma_acc_vec_type):
    return fly.mma_atom_call_ssa([mfma_acc_vec_type], _mma_atom, a, b, c)


def _ds_read_tr16_b64_imm(result_type, addr_i32, imm_offset=0):
    """gfx950 ds_read_b64_tr_b16 with DUALWAVE_SWP immediate byte offset."""
    imm = int(imm_offset)
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    raw = llvm.inline_asm(
        raw_type,
        [as_mlir_value(addr_i32)],
        f"ds_read_b64_tr_b16 $0, $1 offset:{imm}\n",
        "=v,v,~{memory}",
        has_side_effects=True,
    )
    return vector.BitCastOp(result_type, raw).result


def _ds_read_tr8_b64_imm(result_type, addr_i32, imm_offset=0):
    """gfx950 ds_read_b64_tr_b8 (8-bit transpose) with immediate byte offset.

    Returns 64 bits = 8 fp8 (the fp8 analog of ds_read_b64_tr_b16's 4 bf16),
    used for the fp8 V transpose load.
    """
    imm = int(imm_offset)
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    raw = llvm.inline_asm(
        raw_type,
        [as_mlir_value(addr_i32)],
        f"ds_read_b64_tr_b8 $0, $1 offset:{imm}\n",
        "=v,v,~{memory}",
        has_side_effects=True,
    )
    return vector.BitCastOp(result_type, raw).result


def _extract_aligned_pointer(tensor, address_space=None) -> ir.Value:
    from flydsl._mlir.dialects import fly as _fly

    ptr_type = ir.Type.parse("!llvm.ptr" if address_space is None else f"!llvm.ptr<{address_space}>")
    return _fly.extract_aligned_pointer_as_index(ptr_type, tensor)


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (n & _VMCNT_LO_MASK) | _LGKMCNT_EXPCNT_BASE | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    rocdl.s_waitcnt(val)


def _s_waitcnt(val):
    rocdl.s_waitcnt(val)


def _sched_barrier(val):
    rocdl.sched_barrier(val)


def _s_barrier():
    rocdl.s_barrier()


def _s_setprio(val):
    rocdl.s_setprio(val)


def _read_exec_i64():
    """Read the current wave exec mask, matching Clang's builtin lowering."""
    true_i1 = fx.Boolean(True).ir_value()
    return rocdl.ballot(T.i64, true_i1)


def _lds_alias_scope_array(names):
    attrs = [f'#llvm.alias_scope<id = "{name}", domain = {_LDS_ALIAS_DOMAIN}>' for name in names]
    return ir.Attribute.parse(f"[{', '.join(attrs)}]")


def _dualwave_lds_scope(kind, buf_id):
    return f"lds_{kind}{buf_id}"


def _dualwave_lds_alias_scopes(name):
    return _lds_alias_scope_array([name])


def _dualwave_lds_noalias_scopes(name, scope_names):
    return _lds_alias_scope_array([scope_name for scope_name in scope_names if scope_name != name])


def _cu_load(div, idx, cu_atom, cu_v1i32):
    v = fly.copy_atom_call_ssa([cu_v1i32], cu_atom, fx.slice(div, (None, fx.Int32(idx))))
    return fx.Index(Vec(v, (1,), fx.Int32)[0])


def _make_page_view(
    base_iter, base_iter_ty, align, page_id, page_byte_stride, page_nrec_bytes, page_layout, elem_ir, buf_flags_i32
):
    base_i64 = fx.Int64(fx.ptrtoint(base_iter))
    off_i64 = fx.Int64(page_id * page_byte_stride)
    shifted = fx.inttoptr(base_iter_ty, base_i64 + off_i64)
    buf_ptr_ty = fx.PointerType.get(elem_ty=elem_ir, address_space=_TargetAddressSpace.BufferDesc, alignment=align)
    buf_ptr = fx.make_ptr(
        buf_ptr_ty,
        [shifted, fx.Int16(0).ir_value(), page_nrec_bytes.ir_value(), buf_flags_i32.ir_value()],
    )
    return fx.logical_divide(fx.make_view(buf_ptr, page_layout), fx.make_layout(1, 1))


def _make_v_page_rsrc(v_base_i64, page_id, page_byte_stride, page_nrec_bytes):
    addr = as_mlir_value(v_base_i64 + fx.Int64(page_id * page_byte_stride))
    return buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=as_mlir_value(page_nrec_bytes))


def _vec_v_elem(n, d, kv_head_idx, TRAITS_BLOCK_N, TRAITS_KV_VEC_SIZE, TRAITS_HEAD_DIM):
    return (
        kv_head_idx * (TRAITS_BLOCK_N // TRAITS_KV_VEC_SIZE) * TRAITS_HEAD_DIM * TRAITS_KV_VEC_SIZE
        + (n // TRAITS_KV_VEC_SIZE) * TRAITS_HEAD_DIM * TRAITS_KV_VEC_SIZE
        + d * TRAITS_KV_VEC_SIZE
        + (n % TRAITS_KV_VEC_SIZE)
    )


def _make_ws_rsrc(ws_base_i64, byte_offset, nrec_bytes):
    addr_i64 = as_mlir_value(ws_base_i64 + fx.Int64(byte_offset))
    return buffer_ops.create_buffer_resource_from_addr(addr_i64, num_records_bytes=as_mlir_value(fx.Int64(nrec_bytes)))


def _read_v8f16_off(traits, v_base_ptr, const_off, kv_mfma_pack_type):
    ptr = buffer_ops.get_element_ptr(
        v_base_ptr, byte_offset=as_mlir_value(fx.Int32(const_off * traits.BF16_BYTES)), elem_type=T.i8
    )
    return llvm.LoadOp(kv_mfma_pack_type, ptr, alignment=16).result


def _load_k_pack_aligned(traits, lds_kv_base_ptr, elem_idx, buf_id, kv_mfma_pack_type):
    scope_name = _dualwave_lds_scope("k", buf_id)
    byte_offset = elem_idx * traits.BF16_BYTES
    ptr = buffer_ops.get_element_ptr(lds_kv_base_ptr, byte_offset=byte_offset, elem_type=T.i8)
    return llvm.LoadOp(
        kv_mfma_pack_type,
        ptr,
        alignment=16,
        alias_scopes=_dualwave_lds_alias_scopes(scope_name),
        noalias_scopes=_dualwave_lds_noalias_scopes(scope_name, traits.LDS_SCOPE_NAMES),
    ).result


def _scale_sched_pairs(pairs, head_dim):
    return max(1, (pairs + 1) // 2) if head_dim == 64 else pairs


def _concat_vectors(lhs, rhs):
    lhs_vec = Vec(lhs)
    rhs_vec = Vec(rhs)
    return lhs_vec.shuffle(
        rhs_vec,
        list(range(lhs_vec.numel)) + [lhs_vec.numel + i for i in range(rhs_vec.numel)],
    )


def _bitcast_i32(value):
    return as_mlir_value(ArithValue(value).bitcast(fx.Int32.ir_type))


def _bitcast_f32(value):
    return as_mlir_value(ArithValue(value).bitcast(fx.Float32.ir_type))


def _attn_mask_vec2_imm(rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32):
    """DUALWAVE_SWP pair mask asm: 2 compares followed by 2 cndmasks."""
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
            as_mlir_value(x_ref_i32),
            as_mlir_value(y_ref_i32),
            as_mlir_value(rel_i32),
            as_mlir_value(neg_inf_i32),
        ],
        asm_str,
        "=s,=s,=v,=v,2,3,v,v,~{vcc}",
        has_side_effects=True,
    )
    return llvm.extractvalue(T.i32, ret, [2]), llvm.extractvalue(T.i32, ret, [3])


def _anchor_pair(v_s):
    lo, hi = v_s
    lo_ir = as_mlir_value(lo)
    hi_ir = as_mlir_value(hi)
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


def _anchor_scalar_f32(x):
    """Pin a scalar f32 at the current source position (no-op asm)."""
    x_ir = as_mlir_value(x)
    return llvm.inline_asm(
        x_ir.type,
        [x_ir],
        "",
        "=v,0",
        has_side_effects=True,
    )


def _stagger_extra_barrier_if_zero(stagger_i32):
    """Emit `s_barrier;` only when stagger == 0."""
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [stagger_i32],
        ("s_cmp_eq_u32 $0, 0\n\ts_cbranch_scc0 1f\n\ts_barrier\n\t1:"),
        "s",
        has_side_effects=True,
    )


def _reduction_pair(v_f32):
    v_i32 = _bitcast_i32(v_f32)
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_ty, v_i32, v_i32, False, True)
    lhs_i32 = llvm.extractvalue(T.i32, swapped, [0])
    rhs_i32 = llvm.extractvalue(T.i32, swapped, [1])
    return _bitcast_f32(lhs_i32), _bitcast_f32(rhs_i32)


def _make_rebased_view(base_iter, byte_off, nrec_bytes, layout, _buf_flags_i32, _elem_ir):
    base_i64 = fx.Int64(fx.ptrtoint(base_iter))
    shifted = fx.inttoptr(base_iter.type, base_i64 + fx.Int64(byte_off))
    buf_ptr_ty = fx.PointerType.get(
        elem_ty=_elem_ir,
        address_space=_TargetAddressSpace.BufferDesc,
        alignment=base_iter.alignment,
    )
    buf_ptr = fx.make_ptr(
        buf_ptr_ty,
        [shifted, fx.Int16(0).ir_value(), fx.Int64(nrec_bytes).ir_value(), _buf_flags_i32.ir_value()],
    )
    return fx.logical_divide(fx.make_view(buf_ptr, layout), fx.make_layout(1, 1))


def _ws_store_f32(f32_val, local_elem_index, rsrc):
    """32-bit f32 store into a per-split-z workspace region via raw buffer descriptor."""
    f32_ir = as_mlir_value(fx.Float32(f32_val))
    buffer_ops.buffer_store(f32_ir, rsrc, as_mlir_value(fx.Int32(local_elem_index)))


def _ws_store_quad_i32(dwords, local_elem_index, rsrc):
    """128-bit i32x4 store (buffer_store_dwordx4) into a per-split-z workspace region."""
    vec_ir = Vec.from_elements([fx.Int32(v) for v in dwords], fx.Int32).ir_value()
    buffer_ops.buffer_store(vec_ir, rsrc, as_mlir_value(fx.Int32(local_elem_index)))


def _buffer_load_128(elem_index, _load_atom_128, q_div, q_load_i32x4_type):
    """128-bit global->register load (buffer_load_dwordx4) from Q."""
    return fly.copy_atom_call_ssa([q_load_i32x4_type], _load_atom_128, fx.slice(q_div, (None, fx.Int32(elem_index))))


def _buffer_load_lds_128(src_div, lds_byte_addr, src_elem, soffset_elems, _dma_atom, _lds_ptr_ty):
    """128-bit global->LDS DMA; `src_elem` is voffset, `soffset_elems` is scaled by the atom."""
    lds_ptr = fx.inttoptr(_lds_ptr_ty, fx.Int32(lds_byte_addr))
    dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
    src = fx.slice(src_div, (None, fx.Int32(src_elem)))
    fx.copy(_dma_atom, src, dst, soffset=fx.Int32(soffset_elems))


def _buffer_store_64(pack_i32_vec, elem_index, _o_store_reg, _store_atom_64, o_div):
    """64-bit register->global store (buffer_store_dwordx2) into O."""
    fx.memref_store_vec(pack_i32_vec, _o_store_reg)
    fx.copy(_store_atom_64, _o_store_reg, fx.slice(o_div, (None, fx.Int32(elem_index))))


def _buffer_store_128(pack_i32_vec, elem_index, _o_store_reg_128, _store_atom_128, o_div):
    """128-bit register->global store (buffer_store_dwordx4) into O."""
    fx.memref_store_vec(pack_i32_vec, _o_store_reg_128)
    fx.copy(_store_atom_128, _o_store_reg_128, fx.slice(o_div, (None, fx.Int32(elem_index))))


def _splitk_workspace_split_z(traits, batch_idx, split_idx):
    return batch_idx * traits.NUM_KV_SPLITS + split_idx


def _splitk_workspace_resources(
    ws_base_i64,
    split_z,
    ws_opart_per_split_bytes,
    ws_ml_per_split_bytes,
    ws_mrow_abs_bytes,
    ws_lrow_abs_bytes,
):
    opart_rsrc = _make_ws_rsrc(ws_base_i64, split_z * ws_opart_per_split_bytes, ws_opart_per_split_bytes)
    mrow_rsrc = _make_ws_rsrc(ws_base_i64, ws_mrow_abs_bytes + split_z * ws_ml_per_split_bytes, ws_ml_per_split_bytes)
    lrow_rsrc = _make_ws_rsrc(ws_base_i64, ws_lrow_abs_bytes + split_z * ws_ml_per_split_bytes, ws_ml_per_split_bytes)
    return opart_rsrc, mrow_rsrc, lrow_rsrc


def _splitk_local_opart_row_base(traits, q_head_idx, seq_len_v, q_row):
    return (q_head_idx * seq_len_v + q_row) * fx.Index(traits.HEAD_DIM // 2)


def _splitk_local_ml_idx(q_head_idx, seq_len_v, q_row):
    return q_head_idx * seq_len_v + q_row


def _splitk_o_partial_dword_col(traits, dc, g, lane_div_32):
    return dc * (traits.D_CHUNK // 2) + (2 * g + lane_div_32) * 4


def _store_empty_splitk_o_partial_row(traits, local_opart_base, lane_div_32, opart_rsrc):
    c_zero_i = fx.Int32(0)
    for dc in range_constexpr(traits.D_CHUNKS):
        for g in range_constexpr(2):
            _ws_store_quad_i32(
                [c_zero_i, c_zero_i, c_zero_i, c_zero_i],
                local_opart_base + _splitk_o_partial_dword_col(traits, dc, g, lane_div_32),
                opart_rsrc,
            )


def _store_splitk_ml_row(m_row, l_row, local_ml_idx, mrow_rsrc, lrow_rsrc):
    _ws_store_f32(m_row, local_ml_idx, mrow_rsrc)
    _ws_store_f32(l_row, local_ml_idx, lrow_rsrc)


def _o_pack_2dw(traits, v_o, dc, store_group, elem_dtype):
    r_base = store_group * 4
    if const_expr(traits.DTYPE_STR == "bf16"):
        lo = rocdl.cvt_pk_bf16_f32(
            Vec(v_o[dc])[r_base],
            Vec(v_o[dc])[r_base + 1],
        )
        hi = rocdl.cvt_pk_bf16_f32(
            Vec(v_o[dc])[r_base + 2],
            Vec(v_o[dc])[r_base + 3],
        )
        return lo, hi

    o_f16 = []
    for i in range_constexpr(4):
        o_f16.append(fx.Float32(Vec(v_o[dc])[r_base + i]).to(elem_dtype))
    pack = Vec.from_elements(o_f16, elem_dtype).bitcast(fx.Int32)
    return as_mlir_value(pack[0]), as_mlir_value(pack[1])


def _swap_halves(dw):
    pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False)
    lo_res = llvm.extractvalue(T.i32, swapped, [0])
    hi_res = llvm.extractvalue(T.i32, swapped, [1])
    return lo_res, hi_res


def _fused_o_128_dwords(lane_div_32, d0_a, d1_a, d0_b, d1_b):
    is_hi_half = ArithValue(lane_div_32 != fx.Index(0))
    y0_a_lo, y0_a_hi = _swap_halves(d0_a)
    y1_a_lo, y1_a_hi = _swap_halves(d1_a)
    y0_b_lo, y0_b_hi = _swap_halves(d0_b)
    y1_b_lo, y1_b_hi = _swap_halves(d1_b)
    y0_a, y1_a = is_hi_half.select(y0_a_lo, y0_a_hi), is_hi_half.select(y1_a_lo, y1_a_hi)
    y0_b, y1_b = is_hi_half.select(y0_b_lo, y0_b_hi), is_hi_half.select(y1_b_lo, y1_b_hi)
    w0 = is_hi_half.select(y0_b, as_mlir_value(d0_a))
    w1 = is_hi_half.select(y1_b, as_mlir_value(d1_a))
    w2 = is_hi_half.select(as_mlir_value(d0_b), y0_a)
    w3 = is_hi_half.select(as_mlir_value(d1_b), y1_a)
    return w0, w1, w2, w3


def _packed_o_128_dwords(traits, v_o, dc, g, lane_div_32, elem_dtype):
    d0_a, d1_a = _o_pack_2dw(traits, v_o, dc, 2 * g, elem_dtype)
    d0_b, d1_b = _o_pack_2dw(traits, v_o, dc, 2 * g + 1, elem_dtype)
    return _fused_o_128_dwords(lane_div_32, d0_a, d1_a, d0_b, d1_b)


def _packed_o_128_vec(traits, v_o, dc, g, lane_div_32, elem_dtype):
    return Vec.from_elements(
        [fx.Int32(w) for w in _packed_o_128_dwords(traits, v_o, dc, g, lane_div_32, elem_dtype)],
        fx.Int32,
    )


def _final_o_base(traits, q_row, stride_q_n_v, q_head_idx, lane_div_32):
    return q_row * stride_q_n_v + q_head_idx * traits.HEAD_DIM + lane_div_32 * 8


def _final_o_global(traits, o_base, dc, g):
    return o_base + (dc * traits.D_CHUNK + 2 * g * 8)


def _store_final_o_128(traits, v_o, dc, g, o_base, lane_div_32, elem_dtype, _o_store_reg_128, _store_atom_128, o_div):
    _buffer_store_128(
        _packed_o_128_vec(traits, v_o, dc, g, lane_div_32, elem_dtype),
        _final_o_global(traits, o_base, dc, g),
        _o_store_reg_128=_o_store_reg_128,
        _store_atom_128=_store_atom_128,
        o_div=o_div,
    )


def _store_final_o_row(
    traits, v_o, q_row, stride_q_n_v, q_head_idx, lane_div_32, elem_dtype, _o_store_reg_128, _store_atom_128, o_div
):
    o_base = _final_o_base(traits, q_row, stride_q_n_v, q_head_idx, lane_div_32)
    for dc in range_constexpr(traits.D_CHUNKS):
        for g in range_constexpr(2):
            _store_final_o_128(
                traits,
                v_o,
                dc,
                g,
                o_base,
                lane_div_32,
                elem_dtype,
                _o_store_reg_128,
                _store_atom_128,
                o_div,
            )


def _store_splitk_partial_o_quad(traits, v_o, dc, g, local_opart_row_base, lane_div_32, opart_rsrc, elem_dtype):
    w0, w1, w2, w3 = _packed_o_128_dwords(traits, v_o, dc, g, lane_div_32, elem_dtype)
    _ws_store_quad_i32(
        [w0, w1, w2, w3],
        local_opart_row_base + _splitk_o_partial_dword_col(traits, dc, g, lane_div_32),
        opart_rsrc,
    )


def _store_splitk_partial_o_row(traits, v_o, local_opart_row_base, lane_div_32, opart_rsrc, elem_dtype):
    for dc in range_constexpr(traits.D_CHUNKS):
        for g in range_constexpr(2):
            _store_splitk_partial_o_quad(
                traits,
                v_o,
                dc,
                g,
                local_opart_row_base,
                lane_div_32,
                opart_rsrc,
                elem_dtype,
            )


def _sched_barrier_pairs(traits, pairs, valu_cnt, group):
    """Emit `pairs` × {1 MFMA + valu_cnt VALU} sched_group_barrier groups."""
    pairs = _scale_sched_pairs(pairs, traits.HEAD_DIM)
    for _ in range_constexpr(pairs):
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, group)
        rocdl.sched_group_barrier(traits.SCHED_VALU_MASK, valu_cnt, group)


def _sched_barrier_exp_pairs(traits, pairs, exp_cnt, group):
    """Emit `pairs` × {1 MFMA + exp_cnt EXP} sched_group_barrier groups."""
    pairs = _scale_sched_pairs(pairs, traits.HEAD_DIM)
    for _ in range_constexpr(pairs):
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, group)
        rocdl.sched_group_barrier(traits.SCHED_EXP_MASK, exp_cnt, group)


def _ds_read_tr_v4f16_imm(lds_base_elem_idx, imm_bytes, lds_kv_base_idx, v_lds_read_vec4_type):
    byte_offset = lds_base_elem_idx * 2 + lds_kv_base_idx
    addr_i32 = fx.Int32(byte_offset)
    return _ds_read_tr16_b64_imm(v_lds_read_vec4_type, addr_i32, imm_bytes)


def _get_q_pack(traits, q_all_scaled_bf16, ks):
    q_vec = Vec(q_all_scaled_bf16)
    base = ks * traits.MFMA_LANE_K
    return q_vec.shuffle(q_vec, [base + i for i in range(traits.MFMA_LANE_K)]).ir_value()


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


def _anchor_v_p(traits, v_p, elem_dtype):
    p_lo, p_hi = v_p
    p_lo_all = _concat_vectors(p_lo[0], p_lo[1])
    p_hi_all = _concat_vectors(p_hi[0], p_hi[1])
    p_all = _concat_vectors(p_lo_all, p_hi_all)
    p_all_ir = as_mlir_value(p_all)
    p_all_anchored = llvm.inline_asm(
        p_all_ir.type,
        [p_all_ir],
        "",
        "=v,0",
        has_side_effects=True,
    )
    p_vec = Vec(p_all_anchored, (traits.PV_K_STEPS * 2 * 8,), elem_dtype)
    anchored_lo = []
    anchored_hi = []
    for pks in range_constexpr(traits.PV_K_STEPS):
        lo_base = pks * 8
        hi_base = traits.PV_K_STEPS * 8 + pks * 8
        anchored_lo.append(p_vec.shuffle(p_vec, [lo_base + i for i in range(8)]).ir_value())
        anchored_hi.append(p_vec.shuffle(p_vec, [hi_base + i for i in range(8)]).ir_value())
    return anchored_lo, anchored_hi


def _v_p_to_vec32(v_p):
    p_lo, p_hi = v_p
    p_lo_all = _concat_vectors(p_lo[0], p_lo[1])
    p_hi_all = _concat_vectors(p_hi[0], p_hi[1])
    return _concat_vectors(p_lo_all, p_hi_all).ir_value()


def _v_vec32_to_p(traits, v_p_all, elem_dtype):
    p_vec = Vec(v_p_all, (traits.PV_K_STEPS * 2 * 8,), elem_dtype)
    p_lo = []
    p_hi = []
    for pks in range_constexpr(traits.PV_K_STEPS):
        lo_base = pks * 8
        hi_base = traits.PV_K_STEPS * 8 + pks * 8
        p_lo.append(p_vec.shuffle(p_vec, [lo_base + i for i in range(8)]).ir_value())
        p_hi.append(p_vec.shuffle(p_vec, [hi_base + i for i in range(8)]).ir_value())
    return p_lo, p_hi


def _rescale_value_types(traits, elem_dtype):
    v32bf16_type = Vec.make_type(traits.PV_K_STEPS * 2 * 8, elem_dtype)
    v32f32_type = Vec.make_type(traits.PV_K_STEPS * 2 * 8, fx.Float32)
    return v32bf16_type, v32f32_type


def _scale_v_p(traits, v_p, scale_scalar, elem_dtype, fm_fast):
    v32bf16_type, v32f32_type = _rescale_value_types(traits, elem_dtype)
    fm_fast_attr = ir.Attribute.parse("#llvm.fastmath<fast>")
    p_all = _v_p_to_vec32(v_p)
    p_all_f32_op = llvm.FPExtOp(v32f32_type, as_mlir_value(p_all))
    p_all_f32_op.operation.attributes["fastmathFlags"] = fm_fast_attr
    scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(traits.PV_K_STEPS * 2 * 8)
    p_scaled_f32 = arith.mulf(
        as_mlir_value(scale_vec),
        as_mlir_value(p_all_f32_op.result),
        fastmath=fm_fast,
    )
    p_scaled_bf16_op = llvm.FPTruncOp(v32bf16_type, p_scaled_f32)
    p_scaled_bf16_op.operation.attributes["fastmathFlags"] = fm_fast_attr
    return _v_vec32_to_p(traits, p_scaled_bf16_op.result, elem_dtype=elem_dtype)


@flyc.jit
def _stagger_extra_barrier_if_one(stagger_i32):
    """Emit `sched_barrier(0); s_barrier;` only when stagger == 1."""
    if fx.Int32(stagger_i32) != fx.Int32(0):
        rocdl.sched_barrier(0)
        rocdl.s_barrier()


def _bf16_trunc_pack_v8(traits, f32_vals, elem_dtype):
    if const_expr(traits.DTYPE_STR == "bf16"):
        pairs = []
        for j in range_constexpr(4):
            pairs.append(rocdl.cvt_pk_bf16_f32(f32_vals[j * 2], f32_vals[j * 2 + 1]))
        return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()
    # fp16: truncate each f32 -> f16 (RNE) and build the v8 pack directly.
    f16_vals = []
    for i in range_constexpr(8):
        f16_vals.append(fx.Float32(f32_vals[i]).to(elem_dtype))
    return Vec.from_elements(f16_vals, elem_dtype).ir_value()


def _k_buf_base(traits, buf_id):
    if const_expr(isinstance(buf_id, int)):
        return traits.DUALWAVE_SWP_K_BUF_BASE[buf_id]
    # runtime buf_id (rare): K0=0, K1=DUALWAVE_SWP_KV_PER_BUFFER
    return buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER


def _v_buf_base(traits, buf_id):
    if const_expr(isinstance(buf_id, int)):
        return traits.DUALWAVE_SWP_V_BUF_BASE[buf_id]
    return traits.SMEM_K_TILE_ELEMS + buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER


def _kv_tile_addr(traits, tile_start, kv_gmem_elem_offset, kv_head_elem_offset, stride_kv_n_v):
    """Return (src_base, soffset): dense uses tile_start*stride; paged folds page offset into the descriptor."""
    if const_expr(traits.PAGED):
        return kv_head_elem_offset, 0
    return kv_gmem_elem_offset, tile_start * stride_kv_n_v


def _linear_kv_src_elem(traits, src_base, d, n_in_warp, wave_id, d_bucket, stride_kv_n_v):
    """Global element index for lane's d-th 128-bit chunk in a linear KV tile."""
    n_in_tile = n_in_warp * traits.NUM_WAVES + wave_id
    global_d = d_bucket * traits.VEC_KV + d * traits.D_128B_SIZE
    return src_base + n_in_tile * stride_kv_n_v + global_d


def _vec_k_dma_oct_idx(traits, d, wave_id_uni, lane_in_warp):
    """Flat octet index for this wave/lane's d-th DMA slot in vectorized K layout."""
    return wave_id_uni * (traits.WARP_SIZE * traits.SMEM_D_RPT) + d * traits.WARP_SIZE + lane_in_warp


def _sigma_k_tile_n(ni):
    """Sigma permutation applied to K tile-n during vectorized DMA (bit-shuffle)."""
    return (ni & 3) | ((ni & 8) >> 1) | ((ni & 4) << 1) | (ni & ~15)


def _vec_k_src_elem(traits, d, wave_id_uni, lane_in_warp, kv_head_idx):
    """Global element index for vectorized K DMA slot d (sigma remap applied)."""
    oct_idx = _vec_k_dma_oct_idx(traits, d, wave_id_uni=wave_id_uni, lane_in_warp=lane_in_warp)
    ni, dg = oct_idx % traits.BLOCK_N, oct_idx // traits.BLOCK_N
    src_oct = dg * traits.BLOCK_N + _sigma_k_tile_n(ni)
    return (
        kv_head_idx * (traits.HEAD_DIM // traits.KV_VEC_SIZE) * traits.BLOCK_N * traits.KV_VEC_SIZE
        + src_oct * traits.KV_VEC_SIZE
    )


def _vec_v_src_elem(traits, d, wave_id_uni, lane_in_warp, kv_head_idx):
    """Global element index for vectorized V DMA slot d (NO-major wave rows)."""
    row = wave_id_uni * traits.SMEM_D_RPT + d
    no = lane_in_warp // traits.SMEM_N_PER_WAVE
    d_col = row * traits.SMEM_N_PER_WAVE + lane_in_warp % traits.SMEM_N_PER_WAVE
    return _vec_v_elem(no * traits.KV_VEC_SIZE, d_col, kv_head_idx, traits.BLOCK_N, traits.KV_VEC_SIZE, traits.HEAD_DIM)


def _paged_bt_byte_offset(tile_idx, split_t0):
    """Byte offset of `tile_idx`'s page-id entry in the LDS block-table cache."""
    return fx.Int32((tile_idx - split_t0) * fx.Index(4))


def _q_pack_col(traits, ks, lane_div_32):
    """K-dimension column for Q pack at MFMA k-step `ks` for this lane."""
    return ks * traits.K_STEP_QK + lane_div_32 * traits.MFMA_LANE_K


def _q_pack_global_idx(traits, q_row_in_block, ks, lane_div_32, stride_q_n_v):
    """Flat global element index for Q pack (q_row, ks)."""
    return q_row_in_block * stride_q_n_v + _q_pack_col(traits, ks, lane_div_32=lane_div_32)


def _vec_k_lds_idx_lo(traits, k_base, ks, lane_div_32, lane_mod_32):
    """LDS element index for vectorized K pack lo at k-step `ks` for this lane."""
    return k_base + (ks * 2 + lane_div_32) * (traits.BLOCK_N * traits.KV_VEC_SIZE) + lane_mod_32 * traits.KV_VEC_SIZE


def _swizzled_ks_offset(traits, ks):
    """Non-vectorized K LDS offset for k-step `ks` (outer/inner swizzle pattern)."""
    return (ks // 4) * traits.DUALWAVE_SWP_URK_KSTEP_OUTER + (ks % 4) * traits.DUALWAVE_SWP_URK_KSTEP_INNER


def _k_lds_read_base_per_lane(traits, lane_mod_32, lane_div_32):
    return (
        (lane_mod_32 % 8) * traits.SMEM_K_LINE_STRIDE
        + (lane_mod_32 // 8) * traits.D_128B_SIZE
        + lane_div_32 * traits.VEC_KV
    )


def _v_lds_read_base_per_lane(traits, lane, lane_div_32):
    return (
        lane_div_32 * traits.DUALWAVE_SWP_URV_GRPK
        + ((lane % 16) // 4) * traits.DUALWAVE_SWP_URV_LANE_HI
        + ((lane // 16) % 2) * traits.DUALWAVE_SWP_URV_GRP_N
        + (lane % 4) * traits.DUALWAVE_SWP_URV_LANE_LO
    )


def _vec_v_lds_addr_base(traits, v_base, lane_div_32, lane_mod_32):
    """Per-lane LDS element base (in elements) for vectorized V reads."""
    lm = lane_mod_32
    return (
        v_base
        + (lm // traits.SMEM_N_PER_WAVE) * traits.VEC_V_ROW_STRIDE
        + lane_div_32 * traits.D_128B_SIZE
        + (lm % traits.SMEM_N_PER_WAVE) * traits.KV_VEC_SIZE
    )


def _vec_v_const_off(traits, dc, k_substep):
    """Constant element offset from lane base for vectorized V pack (dc, k_substep)."""
    return dc * (traits.D_CHUNK // traits.SMEM_N_PER_WAVE) * traits.VEC_V_ROW_STRIDE + k_substep * (
        2 * traits.D_128B_SIZE
    )


def _swizzled_v_dc_off(traits, dc):
    """Non-vectorized V LDS dc-axis offset (swizzled axis0/axis1 decomposition)."""
    return (dc // 2) * traits.DUALWAVE_SWP_URV_DC_AXIS0 + (dc % 2) * traits.DUALWAVE_SWP_URV_DC_AXIS1


def _swizzled_v_imm_lo(traits, dc, k_substep):
    """Non-vectorized V LDS byte offset for (dc, k_substep) in bf16 elements."""
    return (k_substep * traits.DUALWAVE_SWP_URV_STEP_K_STRIDE + _swizzled_v_dc_off(traits, dc)) * traits.BF16_BYTES


def _seq_pad_col_base(traits, tile_idx, lane_div_32):
    """Base KV column for this lane's lo-half scores at `tile_idx`."""
    _lane_n_off = 8 if traits.KV_VECTORIZED else 4
    return fx.Int32(tile_idx * traits.BLOCK_N) + fx.Int32(lane_div_32) * fx.Int32(_lane_n_off)


def _seq_pad_score_threshold(traits, r):
    """Column threshold offset for score row `r` (layout-dependent swizzle)."""
    if const_expr(traits.KV_VECTORIZED):
        return (r // 8) * 16 + (r % 8)
    return (r // 4) * 8 + (r % 4)


class DualwaveKernelContext:
    """Shared per-kernel state for the gfx950 dualwave attention helpers."""

    def __init__(
        self,
        traits_or_ctx,
        Q=None,
        K=None,
        V=None,
        O=None,  # noqa: E741
        DebugCounts=None,
        CuSeqQ=None,
        CuSeqKv=None,
        BlockTable=None,
        seq_len=None,
        seq_len_kv=None,
        stride_q_n=None,
        stride_kv_n=None,
        head_dim_runtime=None,
        block_table_stride=None,
    ):
        if isinstance(traits_or_ctx, DualwaveKernelContext):
            self.__dict__.update(traits_or_ctx.__dict__)
            self.ctx_ref = getattr(traits_or_ctx, "ctx_ref", traits_or_ctx)
            return

        self.ctx_ref = self
        self.traits = traits_or_ctx
        self.Q = Q
        self.K = K
        self.V = V
        self.O = O
        self.DebugCounts = DebugCounts
        self.CuSeqQ = CuSeqQ
        self.CuSeqKv = CuSeqKv
        self.BlockTable = BlockTable
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.stride_q_n = stride_q_n
        self.stride_kv_n = stride_kv_n
        self.head_dim_runtime = head_dim_runtime
        self.block_table_stride = block_table_stride

    def init_types_and_constants(self, head_dim_runtime=None):
        if head_dim_runtime is None:
            head_dim_runtime = self.head_dim_runtime
        traits = self.traits
        self.NUM_DMA_K = traits.SMEM_D_RPT
        self.NUM_DMA_V = traits.SMEM_D_RPT

        self.fm_fast = fx.arith.FastMathFlags.fast
        self.elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)
        self.q_load_i32x4_type = Vec.make_type(4, fx.Int32)
        self.v_lds_read_vec4_type = Vec.make_type(4, self.elem_dtype)
        self.kv_mfma_pack_type = Vec.make_type(8, self.elem_dtype)
        self.mfma_acc_vec_type = Vec.make_type(16, fx.Float32)

        self.c_neg_inf = fx.Float32(float("-inf"))
        self.c_neg_floor = fx.Float32(-3.0e38)
        self.c_zero_f = fx.Float32(0.0)
        self.c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        head_dim_f32 = fx.Float32(fx.Int32(head_dim_runtime))
        c_log2e_f = fx.Float32(_LOG2E)
        self.c_sm_scale_log2e = fx.Float32(
            arith.mulf(
                as_mlir_value(fmath.rsqrt(head_dim_f32, fastmath=self.fm_fast)),
                as_mlir_value(c_log2e_f),
                fastmath=self.fm_fast,
            )
        )

    def init_runtime_indices(self, seq_len=None, seq_len_kv=None, stride_q_n=None, stride_kv_n=None):
        if seq_len is None:
            seq_len = self.seq_len
        if seq_len_kv is None:
            seq_len_kv = self.seq_len_kv
        if stride_q_n is None:
            stride_q_n = self.stride_q_n
        if stride_kv_n is None:
            stride_kv_n = self.stride_kv_n
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.seq_len_v = fx.Index(seq_len)
        self.seq_len_kv_v = fx.Index(seq_len_kv)
        self.stride_q_n_v = fx.Index(stride_q_n)
        self.stride_kv_n_v = fx.Index(stride_kv_n)

    def init_lds(self, shared_storage):
        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.kv.ptr))
        self.lds_kv_base_ptr = buffer_ops.create_llvm_ptr(self.lds_kv_base_idx, address_space=3)
        if const_expr(self.traits.PAGED):
            self.lds_bt_base_idx = fx.Index(fx.ptrtoint(lds.bt.ptr))
            self.lds_bt_base_ptr = buffer_ops.create_llvm_ptr(self.lds_bt_base_idx, address_space=3)
        else:
            self.lds_bt_base_ptr = None

    def init_thread_mapping(self):
        traits = self.traits
        self.h_idx = fx.Index(gpu.block_idx.x)
        self.q_block_idx = fx.Index(gpu.block_idx.y)
        if const_expr(traits.SPLITK):
            self.bz_idx = fx.Index(gpu.block_idx.z)
            self.batch_idx = self.bz_idx // traits.NUM_KV_SPLITS
            self.split_idx = self.bz_idx % traits.NUM_KV_SPLITS
        else:
            self.batch_idx = fx.Index(gpu.block_idx.z)
            self.split_idx = None
        self.tid = fx.Index(gpu.thread_idx.x)

        self.wave_id = self.tid // traits.WARP_SIZE
        self.lane = self.tid % traits.WARP_SIZE
        self.lane_mod_32 = self.lane % 32
        self.lane_div_32 = self.lane // 32

        _tid_i32 = as_mlir_value(fx.Int32(self.tid))
        _wave_id_uni_i32 = rocdl.readfirstlane(
            T.i32,
            arith.divsi(_tid_i32, as_mlir_value(fx.Int32(traits.WARP_SIZE))),
        )
        self.stagger_i32 = arith.divsi(_wave_id_uni_i32, as_mlir_value(fx.Int32(4)))
        self.wave_id_uni = fx.Index(_wave_id_uni_i32)

        self.wave_q_offset = self.wave_id * traits.ROWS_PER_WAVE
        self.q_start = self.q_block_idx * traits.BLOCK_M

        self.h_kv_idx = self.h_idx % traits.NUM_HEADS_KV
        self.group_id = self.h_idx // traits.NUM_HEADS_KV
        self.q_head_idx = self.h_kv_idx * traits.GQA_GROUP_SIZE + self.group_id
        self.kv_head_idx = self.h_kv_idx

    def init_sequence_lengths(self, CuSeqQ=None, CuSeqKv=None):
        if CuSeqQ is None:
            CuSeqQ = self.CuSeqQ
        if CuSeqKv is None:
            CuSeqKv = self.CuSeqKv
        traits = self.traits
        if const_expr(traits.VARLEN):
            _cuq_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(CuSeqQ), fx.make_layout(1, 1))
            _cuk_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(CuSeqKv), fx.make_layout(1, 1))
            _cu_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            _cu_v1i32 = Vec.make_type(1, fx.Int32)

            self.q_tok_base = _cu_load(_cuq_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.q_tok_end = _cu_load(_cuq_div, self.batch_idx + fx.Index(1), _cu_atom, _cu_v1i32)
            self.kv_tok_base = _cu_load(_cuk_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.kv_tok_end = _cu_load(_cuk_div, self.batch_idx + fx.Index(1), _cu_atom, _cu_v1i32)
            self.seqlen_q_v = self.q_tok_end - self.q_tok_base
            self.seqlen_kv_v = self.kv_tok_end - self.kv_tok_base
            self.seqlen_kv_i32 = fx.Int32(self.seqlen_kv_v)
        else:
            self.q_tok_base = self.batch_idx * self.seq_len_v
            self.kv_tok_base = self.batch_idx * self.seq_len_kv_v
            self.q_tok_end = (self.batch_idx + fx.Index(1)) * self.seq_len_v
            self.kv_tok_end = (self.batch_idx + fx.Index(1)) * self.seq_len_kv_v
            self.seqlen_q_v = self.seq_len_v
            self.seqlen_kv_v = self.seq_len_kv_v
            self.seqlen_kv_i32 = self.seq_len_kv

    def init_descriptors(
        self,
        q_tensor=None,
        k_tensor=None,
        v_tensor=None,
        o_tensor=None,
        block_table=None,
        block_table_stride=None,
    ):
        if q_tensor is None:
            q_tensor = self.Q
        if k_tensor is None:
            k_tensor = self.K
        if v_tensor is None:
            v_tensor = self.V
        if o_tensor is None:
            o_tensor = self.O
        if block_table is None:
            block_table = self.BlockTable
        if block_table_stride is None:
            block_table_stride = self.block_table_stride
        traits = self.traits
        if const_expr(traits.PAGED):
            self.block_table_stride_v = fx.Index(block_table_stride)
            self.bt_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(block_table), fx.make_layout(1, 1))
            self.bt_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            self.bt_v1i32 = Vec.make_type(1, fx.Int32)
            self.kv_head_elem_offset = self.kv_head_idx * traits.HEAD_DIM
        else:
            self.block_table_stride_v = None
            self.bt_div = None
            self.bt_atom = None
            self.bt_v1i32 = None
            self.kv_head_elem_offset = None

        self.delta_i32 = fx.Int32(self.seqlen_kv_i32 - fx.Int32(self.seqlen_q_v))
        self.q_gmem_elem_offset = self.q_start * self.stride_q_n_v + self.q_head_idx * traits.HEAD_DIM
        self.kv_gmem_elem_offset = self.kv_head_idx * traits.HEAD_DIM

        self.buf_flags_i32 = fx.Int32(buffer_ops._get_buffer_flags())
        self.elem_ir = self.elem_dtype.ir_type
        qo_per_batch_elems = self.seqlen_q_v * self.stride_q_n_v
        qo_nrec_bytes = qo_per_batch_elems * fx.Index(traits.BF16_BYTES)
        qo_layout = fx.make_layout(fx.Int32(qo_per_batch_elems), fx.Int32(1))
        q_batch_byte_off = self.q_tok_base * self.stride_q_n_v * fx.Index(traits.BF16_BYTES)
        self.q_div = _make_rebased_view(
            fx.get_iter(q_tensor),
            q_batch_byte_off,
            qo_nrec_bytes,
            qo_layout,
            _buf_flags_i32=self.buf_flags_i32,
            _elem_ir=self.elem_ir,
        )
        self.o_div = _make_rebased_view(
            fx.get_iter(o_tensor),
            q_batch_byte_off,
            qo_nrec_bytes,
            qo_layout,
            _buf_flags_i32=self.buf_flags_i32,
            _elem_ir=self.elem_ir,
        )

        if const_expr(traits.PAGED):
            self.k_div = None
            self.v_div = None
            page_elems = fx.Index(traits.BLOCK_N) * self.stride_kv_n_v
            self.page_byte_stride = page_elems * fx.Index(traits.BF16_BYTES)
            self.page_nrec_bytes = fx.Int64(self.page_byte_stride)
            self.page_layout = fx.make_layout(fx.Int32(page_elems), fx.Int32(1))
        else:
            kv_per_batch_elems = self.seqlen_kv_v * self.stride_kv_n_v
            kv_nrec_bytes = kv_per_batch_elems * fx.Index(traits.BF16_BYTES)
            kv_layout = fx.make_layout(fx.Int32(kv_per_batch_elems), fx.Int32(1))
            kv_batch_byte_off = self.kv_tok_base * self.stride_kv_n_v * fx.Index(traits.BF16_BYTES)
            self.k_div = _make_rebased_view(
                fx.get_iter(k_tensor),
                kv_batch_byte_off,
                kv_nrec_bytes,
                kv_layout,
                _buf_flags_i32=self.buf_flags_i32,
                _elem_ir=self.elem_ir,
            )
            self.v_div = _make_rebased_view(
                fx.get_iter(v_tensor),
                kv_batch_byte_off,
                kv_nrec_bytes,
                kv_layout,
                _buf_flags_i32=self.buf_flags_i32,
                _elem_ir=self.elem_ir,
            )
            self.page_byte_stride = None
            self.page_nrec_bytes = None
            self.page_layout = None
        self.debug_counts_rsrc = (
            _make_raw_buffer_rsrc(self.DebugCounts) if traits.DUALWAVE_SWP_DEBUG_LAZY_COUNTS else None
        )

    def init_workspace(self, DebugCounts=None):
        if DebugCounts is None:
            DebugCounts = self.DebugCounts
        traits = self.traits
        if const_expr(traits.SPLITK):
            self.ws_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(DebugCounts)))
            self.ws_opart_per_split_elems = (
                fx.Index(traits.NUM_HEADS_Q) * self.seq_len_v * fx.Index(traits.HEAD_DIM // 2)
            )
            self.ws_ml_per_split_elems = fx.Index(traits.NUM_HEADS_Q) * self.seq_len_v
            self.ws_opart_per_split_bytes = self.ws_opart_per_split_elems * fx.Index(4)
            self.ws_ml_per_split_bytes = self.ws_ml_per_split_elems * fx.Index(4)
            self.ws_grid_z = fx.Index(gpu.grid_dim.z)
            self.ws_mrow_abs_bytes = self.ws_grid_z * self.ws_opart_per_split_bytes
            self.ws_lrow_abs_bytes = self.ws_mrow_abs_bytes + self.ws_grid_z * self.ws_ml_per_split_bytes
        else:
            self.ws_base_i64 = None
            self.ws_opart_per_split_bytes = None
            self.ws_ml_per_split_bytes = None
            self.ws_mrow_abs_bytes = None
            self.ws_lrow_abs_bytes = None

    def init_atoms_and_lds_ptrs(self):
        self.load_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.store_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, self.elem_dtype))
        self.o_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        self.lds_ptr_ty = fx.PointerType.get(self.elem_dtype.ir_type, 2, self.traits.DMA_BYTES)

    def init_dma_thread_offsets(self):
        self.lane_in_warp = self.tid % self.traits.WARP_SIZE
        self.n_in_warp = self.lane_in_warp // self.traits.VEC_KV
        self.d_bucket = self.lane_in_warp % self.traits.VEC_KV

    def init_q_row(self):
        traits = self.traits
        self.q_row_in_block = self.wave_q_offset + self.lane_mod_32
        self.q_start_pos_i32 = fx.Int32(self.q_start + self.wave_id_uni * traits.ROWS_PER_WAVE)
        self.q_row = self.q_start + self.q_row_in_block
        self.q_row_i32 = fx.Int32(self.q_row)

    def k_dma_base(self, buf_id, d):
        return _k_dma_m0_base(
            self.traits,
            buf_id,
            d,
            lane_in_warp=self.lane_in_warp,
            lds_kv_base_idx=self.lds_kv_base_idx,
            wave_id_uni=self.wave_id_uni,
        )

    def v_dma_base(self, buf_id, d):
        return _v_dma_m0_base(
            self.traits,
            buf_id,
            d,
            lane_in_warp=self.lane_in_warp,
            lds_kv_base_idx=self.lds_kv_base_idx,
            wave_id_uni=self.wave_id_uni,
        )

    def dma_m0_table(self, base_fn, count):
        return tuple(tuple(base_fn(buf, d) for d in range(count)) for buf in range(2))

    def init_dma_m0_tables(self):
        self.k_dma_m0 = self.dma_m0_table(self.k_dma_base, self.NUM_DMA_K)
        self.v_dma_m0 = self.dma_m0_table(self.v_dma_base, self.NUM_DMA_V)

    def init_tile_bounds(self):
        traits = self.traits
        self.kv_tile_size = traits.BLOCK_N
        self.num_kv_tiles = (self.seqlen_kv_v + self.kv_tile_size - 1) // self.kv_tile_size
        if const_expr(traits.CAUSAL):
            self.causal_end_raw_i32 = fx.Int32(self.q_start + traits.BLOCK_M) + self.delta_i32
            self.causal_end_i32 = fx.Int32(
                ArithValue(self.causal_end_raw_i32 > fx.Int32(0)).select(self.causal_end_raw_i32, fx.Int32(0))
            )
            self.causal_num_tiles = (fx.Index(self.causal_end_i32) + self.kv_tile_size - 1) // self.kv_tile_size
            self.max_num_tiles = fx.Index(
                ArithValue(self.causal_num_tiles < self.num_kv_tiles).select(self.causal_num_tiles, self.num_kv_tiles)
            )
        else:
            self.causal_end_raw_i32 = None
            self.max_num_tiles = self.num_kv_tiles

        self.max_num_tiles = ((self.max_num_tiles + fx.Index(1)) // fx.Index(2)) * fx.Index(2)
        self.max_num_tiles = fx.Index(
            ArithValue(self.max_num_tiles < fx.Index(4)).select(fx.Index(4), self.max_num_tiles)
        )

        if const_expr(traits.SPLITK):
            chunk = ((self.max_num_tiles + (traits.NUM_KV_SPLITS - 1)) // traits.NUM_KV_SPLITS + 1) // 2 * 2
            chunk = fx.Index(ArithValue(chunk < fx.Index(6)).select(fx.Index(6), chunk))
            self.split_t0 = self.split_idx * chunk
            self.split_t_end = self.split_t0 + chunk
            self.split_t_end = fx.Index(
                ArithValue(self.split_t_end < self.max_num_tiles).select(self.split_t_end, self.max_num_tiles)
            )
            self.split_t_end = fx.Index(
                ArithValue(self.max_num_tiles - self.split_t_end < fx.Index(4)).select(
                    self.max_num_tiles, self.split_t_end
                )
            )
            self.split_nonempty = self.split_t0 + fx.Index(4) <= self.max_num_tiles
        else:
            self.split_t0 = 0
            self.split_t_end = self.max_num_tiles

    def compute_active_guard(self):
        traits = self.traits
        if const_expr(traits.SPLITK):
            return self.split_nonempty
        if const_expr(traits.VARLEN):
            if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
                return ArithValue(self.q_start < self.seqlen_q_v) & (self.causal_end_raw_i32 > fx.Int32(0))
            return ArithValue(self.q_start < self.seqlen_q_v)
        if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
            return ArithValue(self.causal_end_raw_i32 > fx.Int32(0))
        return None

    def init_active_guard(self):
        self.active = self.compute_active_guard()

    def init_lds_read_bases(self):
        self.k_lds_read_base_per_lane = _k_lds_read_base_per_lane(self.traits, self.lane_mod_32, self.lane_div_32)
        self.v_lds_read_base_per_lane = _v_lds_read_base_per_lane(self.traits, self.lane, self.lane_div_32)

    def split_tile(self, offset_tiles=0):
        return self.split_t0 + fx.Index(offset_tiles)

    def tile_start(self, tile_idx):
        return tile_idx * self.traits.BLOCK_N


class DualwavePageIdLoader(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_block_table_to_lds(self):
        traits = self.traits
        tid = self.tid
        split_t0 = self.split_t0
        split_t_end = self.split_t_end
        num_kv_tiles = self.num_kv_tiles
        batch_idx = self.batch_idx
        block_table_stride_v = self.block_table_stride_v
        lds_bt_base_ptr = self.lds_bt_base_ptr
        bt_div = self.bt_div
        bt_atom = self.bt_atom
        bt_v1i32 = self.bt_v1i32

        @flyc.jit
        def _load_block_table_to_lds():
            segment_tiles = split_t_end - split_t0
            for pass_id in range_constexpr(traits.PAGED_BT_LDS_SIZE // traits.BLOCK_SIZE):
                local_tile = tid + fx.Index(pass_id * traits.BLOCK_SIZE)
                if local_tile < segment_tiles:
                    tile_idx = split_t0 + local_tile
                    byte_off = as_mlir_value(fx.Int32(local_tile * fx.Index(4)))
                    dst = buffer_ops.get_element_ptr(lds_bt_base_ptr, byte_offset=byte_off, elem_type=T.i8)
                    llvm.StoreOp(as_mlir_value(fx.Int32(0)), dst)
                    if tile_idx < num_kv_tiles:
                        row_idx = batch_idx * block_table_stride_v + tile_idx
                        v = fly.copy_atom_call_ssa([bt_v1i32], bt_atom, fx.slice(bt_div, (None, fx.Int32(row_idx))))
                        page_id_i32 = as_mlir_value(fx.Int32(Vec(v, (1,), fx.Int32)[0]))
                        llvm.StoreOp(page_id_i32, dst)

        _load_block_table_to_lds()

    def load_page_id_lds(self, tile_idx):
        src = buffer_ops.get_element_ptr(
            self.lds_bt_base_ptr,
            byte_offset=as_mlir_value(_paged_bt_byte_offset(tile_idx, split_t0=self.split_t0)),
            elem_type=T.i8,
        )
        return llvm.LoadOp(T.i32, src).result

    def finish_page_id(self, v):
        rocdl.s_waitcnt(self.traits.LGKMCNT_0_ONLY)
        v = rocdl.readfirstlane(T.i32, v)
        return fx.Index(fx.Int32(v))

    def async_load_tile_page_id(self, tile_idx, page_id_override=None):
        if const_expr(self.traits.PAGED):
            if const_expr(page_id_override is not None):
                return page_id_override
            page_id = self.load_page_id_lds(tile_idx)
            return self.finish_page_id(page_id)
        return fx.Index(0)

    def async_load_split_page(self, offset_tiles=0, page_id_override=None):
        return self.async_load_tile_page_id(self.split_tile(offset_tiles), page_id_override=page_id_override)

    def async_load_page_id(self, tile_start, page_id_override=None):
        if const_expr(self.traits.PAGED):
            if const_expr(page_id_override is not None):
                return page_id_override
            return self.async_load_tile_page_id(tile_start // fx.Index(self.traits.BLOCK_N))
        return fx.Index(0)


class DualwaveQLoader(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_pack(self, q_row_in_block, ks):
        q_i32_pack = _buffer_load_128(
            self.q_gmem_elem_offset
            + _q_pack_global_idx(
                self.traits,
                q_row_in_block,
                ks,
                lane_div_32=self.lane_div_32,
                stride_q_n_v=self.stride_q_n_v,
            ),
            _load_atom_128=self.load_atom_128,
            q_div=self.q_div,
            q_load_i32x4_type=self.q_load_i32x4_type,
        )
        return Vec(q_i32_pack, (4,), fx.Int32).bitcast(self.elem_dtype).ir_value()

    def load_all(self):
        traits = self.traits
        ctx = self.ctx_ref
        ctx.init_q_row()

        q_raw_packs = []
        for ks in range_constexpr(traits.K_STEPS_QK):
            q_raw_packs.append(self.load_pack(ctx.q_row_in_block, ks))
        q_16_packs = []
        for pair in range_constexpr(traits.K_STEPS_QK // 2):
            q_16_packs.append(_concat_vectors(q_raw_packs[pair * 2], q_raw_packs[pair * 2 + 1]))

        q_32_packs = []
        for pair in range_constexpr(traits.K_STEPS_QK // 4):
            q_32_packs.append(_concat_vectors(q_16_packs[pair * 2], q_16_packs[pair * 2 + 1]))

        q_all = q_32_packs[0] if const_expr(traits.K_STEPS_QK == 4) else _concat_vectors(q_32_packs[0], q_32_packs[1])
        return Vec(q_all, (traits.K_STEPS_QK * traits.MFMA_LANE_K,), self.elem_dtype)

    def scale_all(self, q_all_bf16):
        traits = self.traits
        fm_fast_attr = ir.Attribute.parse("#llvm.fastmath<fast>")
        v64bf16_type = Vec.make_type(traits.K_STEPS_QK * traits.MFMA_LANE_K, self.elem_dtype)
        v64f32_type = Vec.make_type(traits.K_STEPS_QK * traits.MFMA_LANE_K, fx.Float32)
        q_all_f32_op = llvm.FPExtOp(v64f32_type, as_mlir_value(q_all_bf16))
        q_all_f32_op.operation.attributes["fastmathFlags"] = fm_fast_attr
        q_all_f32 = q_all_f32_op.result
        scale_vec = Vec.from_elements([self.c_sm_scale_log2e], fx.Float32).broadcast_to(
            traits.K_STEPS_QK * traits.MFMA_LANE_K
        )
        q_all_scaled_f32 = arith.mulf(
            as_mlir_value(scale_vec),
            as_mlir_value(q_all_f32),
            fastmath=self.fm_fast,
        )
        q_all_scaled_bf16_op = llvm.FPTruncOp(v64bf16_type, q_all_scaled_f32)
        q_all_scaled_bf16_op.operation.attributes["fastmathFlags"] = fm_fast_attr
        q_all_scaled_bf16 = q_all_scaled_bf16_op.result
        return Vec(q_all_scaled_bf16, (traits.K_STEPS_QK * traits.MFMA_LANE_K,), self.elem_dtype)


class DualwaveGemmHelper(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def qk(self, v_k, q_all_scaled_bf16):
        k_lo, k_hi = v_k
        v_s_lo = self.c_zero_v16f32
        v_s_hi = self.c_zero_v16f32
        for ks in range_constexpr(self.traits.K_STEPS_QK):
            q_pack = _get_q_pack(self.traits, q_all_scaled_bf16, ks)
            v_s_lo = _mfma_acc(k_lo[ks], q_pack, v_s_lo, self.mma_atom, self.mfma_acc_vec_type)
            v_s_hi = _mfma_acc(k_hi[ks], q_pack, v_s_hi, self.mma_atom, self.mfma_acc_vec_type)
        return (v_s_lo, v_s_hi)

    def pv_step_k(self, step, v_p, v_v, v_o):
        v_p_lo, v_p_hi = v_p
        v_pk = v_v[step]
        if const_expr(step < 2):
            p_pk = v_p_lo[step]
        else:
            p_pk = v_p_hi[step - 2]
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_o[dc] = _mfma_acc(v_pk[dc], p_pk, v_o[dc], self.mma_atom, self.mfma_acc_vec_type)
        return v_o

    def pv(self, v_p, v_v, v_o):
        for step in range_constexpr(4):
            v_o = self.pv_step_k(step, v_p, v_v, v_o)
        return v_o


class DualwaveSoftmaxHelper(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def reduce_max(self, v_s):
        s_lo, s_hi = v_s
        m = self.c_neg_inf
        for r in range_constexpr(16):
            m = _fmax(m, s_lo[r], self.fm_fast)
        for r in range_constexpr(16):
            m = _fmax(m, s_hi[r], self.fm_fast)
        lhs, rhs = _reduction_pair(m)
        return _fmax(lhs, rhs, self.fm_fast)

    def floor_masked_max(self, row_max):
        return _fmax(row_max, self.c_neg_floor, self.fm_fast)

    def rescale_from_tile_max(self, m_row, m_tile_max):
        row_max = _fmax(m_row, m_tile_max, self.fm_fast)
        rescale = rocdl.exp2(T.f32, as_mlir_value(_fsub(m_row, row_max, self.fm_fast)))
        return row_max, rescale

    def apply_l_rescale(self, l_row, rescale):
        return _fmul(l_row, rescale, self.fm_fast)

    def exp2(self, v_s, start, length):
        if const_expr(start == 0):
            s_lo = [Vec(v_s[0])[r] for r in range_constexpr(16)]
            lo_partial = []
            for r in range_constexpr(16):
                lo_partial.append(rocdl.exp2(T.f32, as_mlir_value(s_lo[r])))
            return Vec.from_elements(lo_partial, fx.Float32).ir_value(), v_s[1]

        lo_partial = [Vec(v_s[0])[r] for r in range_constexpr(16)]
        hi_full = []
        for r in range_constexpr(16):
            hi_full.append(rocdl.exp2(T.f32, as_mlir_value(Vec(v_s[1])[r])))
        return lo_partial, hi_full

    def reduce_sum(self, l_row, v_p):
        lo_partial_list, hi_full = v_p
        local_sum = self.c_zero_f
        for r in range_constexpr(16):
            local_sum = _fadd(local_sum, lo_partial_list[r], self.fm_fast)
        for r in range_constexpr(16):
            local_sum = _fadd(local_sum, hi_full[r], self.fm_fast)
        lhs_sum, rhs_sum = _reduction_pair(local_sum)
        return _fadd(l_row, _fadd(lhs_sum, rhs_sum, self.fm_fast), self.fm_fast)

    def sub_m(self, v_s, row_max):
        s_lo, s_hi = v_s
        lo_sub = []
        hi_sub = []
        for r in range_constexpr(16):
            lo_sub.append(_fsub(s_lo[r], row_max, self.fm_fast))
        for r in range_constexpr(16):
            hi_sub.append(_fsub(s_hi[r], row_max, self.fm_fast))
        lo_vec = Vec.from_elements(lo_sub, fx.Float32).ir_value()
        hi_vec = Vec.from_elements(hi_sub, fx.Float32).ir_value()
        return lo_vec, hi_vec

    def cast_p(self, v_p):
        lo_partial_list, hi_full = v_p
        p_lo_packs = []
        p_hi_packs = []
        for pks in range_constexpr(self.traits.PV_K_STEPS):
            p_base = pks * 8
            lo_slice = [lo_partial_list[p_base + s] for s in range_constexpr(8)]
            hi_slice = hi_full[p_base : p_base + 8]
            p_lo_packs.append(_bf16_trunc_pack_v8(self.traits, lo_slice, elem_dtype=self.elem_dtype))
            p_hi_packs.append(_bf16_trunc_pack_v8(self.traits, hi_slice, elem_dtype=self.elem_dtype))
        return p_lo_packs, p_hi_packs

    def safe_l_inv(self, l_row):
        l_inv = rocdl.rcp(T.f32, as_mlir_value(l_row))
        return ArithValue(fx.Float32(l_row) > self.c_zero_f).select(l_inv, self.c_zero_f)

    def scale_o(self, v_o, scale_scalar):
        scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_o[dc] = _fmul(Vec(v_o[dc]), scale_vec, self.fm_fast)

    def rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        m_new = _fmax(m_row, m_tile_max, self.fm_fast)
        corr = rocdl.exp2(T.f32, as_mlir_value(_fsub(m_row, m_new, self.fm_fast)))
        self.scale_o(v_o, corr)
        v_o = _anchor_v_o(self.traits, v_o)
        v_p = _scale_v_p(
            self.traits,
            v_p,
            corr,
            elem_dtype=self.elem_dtype,
            fm_fast=self.fm_fast,
        )
        l_row = _fmul(l_row, corr, self.fm_fast)
        return v_o, m_new, l_row, v_p

    def _lazy_rescale_o_rescale(self, _n, *_st, v_o, m_row, l_row, m_tile_max, v_p):
        corr = rocdl.exp2(T.f32, as_mlir_value(_fsub(m_row, m_tile_max, self.fm_fast)))
        scaled_accs = list(v_o)
        self.scale_o(scaled_accs, corr)
        out = [as_mlir_value(scaled_accs[dc]) for dc in range(self.traits.D_CHUNKS)]
        scaled_p = _scale_v_p(
            self.traits,
            v_p,
            corr,
            elem_dtype=self.elem_dtype,
            fm_fast=self.fm_fast,
        )
        out.append(_v_p_to_vec32(scaled_p))
        out.append(as_mlir_value(_fmul(l_row, corr, self.fm_fast)))
        out.append(_anchor_scalar_f32(m_tile_max))
        return out

    def lazy_rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        traits = self.traits
        lane = self.lane
        debug_counts_rsrc = self.debug_counts_rsrc

        @flyc.jit
        def _lazy_rescale_o(v_o, m_row, l_row, m_tile_max, v_p):
            c_eight_f = fx.Float32(traits.DUALWAVE_SWP_RESCALE_THRESHOLD)
            m_diff = _fsub(m_tile_max, m_row, self.fm_fast)
            below = ArithValue(fx.Float32(m_diff) <= c_eight_f)
            ballot = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = arith.cmpi(arith.CmpIPredicate.eq, as_mlir_value(ballot), _read_exec_i64())
            all_below = llvm.intr_expect(all_below, arith.constant(1, type=ir.IntegerType.get_signless(1)))
            _debug_count_lazy_branch(
                traits,
                all_below,
                debug_counts_rsrc=debug_counts_rsrc,
                lane=lane,
            )

            _state = [as_mlir_value(v_o[dc]) for dc in range(traits.D_CHUNKS)]
            _state += [_v_p_to_vec32(v_p), as_mlir_value(l_row), as_mlir_value(m_row)]
            _names = tuple("_lr%d" % i for i in range(traits.D_CHUNKS + 3))

            _rescale = lambda _n, *_st: self._lazy_rescale_o_rescale(
                _n,
                *_st,
                v_o=v_o,
                m_row=m_row,
                l_row=l_row,
                m_tile_max=m_tile_max,
                v_p=v_p,
            )

            _res = scf_if_dispatch(all_below, lambda *_a: None, _rescale, state_names=_names, state_values=_state)
            o_out = list(_res[0 : traits.D_CHUNKS])
            vp_out = _res[traits.D_CHUNKS]
            l_out = _res[traits.D_CHUNKS + 1]
            m_out = _res[traits.D_CHUNKS + 2]
            return (o_out, m_out, l_out, _v_vec32_to_p(traits, vp_out, elem_dtype=self.elem_dtype))

        return _lazy_rescale_o(v_o, m_row, l_row, m_tile_max, v_p)

    def v_s_vec_to_lists(self, v_s):
        s_lo, s_hi = v_s
        return (
            [Vec(s_lo)[r] for r in range_constexpr(16)],
            [Vec(s_hi)[r] for r in range_constexpr(16)],
        )

    def causal_mask_prologue_if_needed(
        self,
        v_s,
        tile_idx=None,
        kv_end_pos=None,
        q_start_pos_i32=None,
        q_row_i32=None,
        *,
        kv_end_tile=None,
    ):
        if tile_idx is None:
            tile_idx = fx.Index(0)
        if kv_end_pos is None:
            end_tile = tile_idx + fx.Index(1) if kv_end_tile is None else kv_end_tile
            kv_end_pos = self.tile_start(end_tile)
        if q_start_pos_i32 is None:
            q_start_pos_i32 = self.ctx_ref.q_start_pos_i32
        if q_row_i32 is None:
            q_row_i32 = self.ctx_ref.q_row_i32
        traits = self.traits
        delta_i32 = self.delta_i32
        lane_div_32 = self.lane_div_32

        @flyc.jit
        def _causal_mask_prologue_if_needed(v_s, tile_idx, kv_end_pos, q_start_pos_i32, q_row_i32):
            s_lo, s_hi = v_s
            if q_start_pos_i32 + delta_i32 < fx.Int32(kv_end_pos):
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                _causal_mask_inplace(
                    traits,
                    (lo_list, hi_list),
                    tile_idx,
                    delta_i32=delta_i32,
                    lane_div_32=lane_div_32,
                    q_row_i32=q_row_i32,
                )
                s_lo = Vec.from_elements([as_mlir_value(v) for v in lo_list], fx.Float32).ir_value()
                s_hi = Vec.from_elements([as_mlir_value(v) for v in hi_list], fx.Float32).ir_value()
            return s_lo, s_hi

        return _causal_mask_prologue_if_needed(v_s, tile_idx, kv_end_pos, q_start_pos_i32, q_row_i32)

    def causal_mask_split_prologue_if_needed(self, v_s, offset_tiles=0, end_offset_tiles=1):
        return self.causal_mask_prologue_if_needed(
            v_s,
            self.split_tile(offset_tiles),
            kv_end_tile=self.split_tile(end_offset_tiles),
        )

    def seq_pad_mask_inplace(self, v_s_lists, tile_idx):
        s_lo, s_hi = v_s_lists
        col_base = _seq_pad_col_base(self.traits, tile_idx, lane_div_32=self.lane_div_32)
        for r in range_constexpr(16):
            col_lo = col_base + fx.Int32(_seq_pad_score_threshold(self.traits, r))
            col_hi = col_lo + fx.Int32(32)
            s_lo[r] = ArithValue(col_lo < self.seqlen_kv_i32).select(s_lo[r], self.c_neg_inf)
            s_hi[r] = ArithValue(col_hi < self.seqlen_kv_i32).select(s_hi[r], self.c_neg_inf)

    def seq_pad_mask_if_needed(self, v_s, tile_idx):
        traits = self.traits
        seqlen_kv_i32 = self.seqlen_kv_i32

        @flyc.jit
        def _seq_pad_mask_if_needed(v_s, tile_idx):
            s_lo, s_hi = v_s
            kv_tile_end = (tile_idx + fx.Index(1)) * traits.BLOCK_N
            if fx.Int32(kv_tile_end) > seqlen_kv_i32:
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self.seq_pad_mask_inplace((lo_list, hi_list), tile_idx)
                s_lo = Vec.from_elements([as_mlir_value(v) for v in lo_list], fx.Float32).ir_value()
                s_hi = Vec.from_elements([as_mlir_value(v) for v in hi_list], fx.Float32).ir_value()
            return s_lo, s_hi

        return _seq_pad_mask_if_needed(v_s, tile_idx)


def _k_dma_m0_base(traits, buf_id, d, lane_in_warp, lds_kv_base_idx, wave_id_uni):
    k_lds_byte_base = lds_kv_base_idx + _k_buf_base(traits, buf_id) * traits.BF16_BYTES
    if const_expr(traits.KV_VECTORIZED):
        oct_idx = wave_id_uni * (traits.WARP_SIZE * traits.SMEM_D_RPT) + d * traits.WARP_SIZE + lane_in_warp
        lds_addr = k_lds_byte_base + oct_idx * (traits.KV_VEC_SIZE * traits.BF16_BYTES)
    else:
        lds_addr = (
            k_lds_byte_base
            + wave_id_uni * (traits.SMEM_K_LINE_STRIDE * traits.BF16_BYTES)
            + (d * traits.SMEM_N_RPT * traits.SMEM_K_LINE_STRIDE * traits.BF16_BYTES)
        )
    return rocdl.readfirstlane(T.i32, as_mlir_value(fx.Int32(lds_addr)))


def _v_dma_m0_base(traits, buf_id, d, lane_in_warp, lds_kv_base_idx, wave_id_uni):
    v_lds_byte_base = lds_kv_base_idx + _v_buf_base(traits, buf_id) * traits.BF16_BYTES
    if const_expr(traits.KV_VECTORIZED):
        row = wave_id_uni * traits.SMEM_D_RPT + d
        lds_elem = row * traits.VEC_V_ROW_STRIDE + lane_in_warp * traits.KV_VEC_SIZE
        lds_addr = v_lds_byte_base + lds_elem * traits.BF16_BYTES
    else:
        lds_addr = (
            v_lds_byte_base
            + wave_id_uni * (traits.SMEM_V_LINE_STRIDE * traits.BF16_BYTES)
            + (d * traits.SMEM_N_RPT * traits.SMEM_V_LINE_STRIDE * traits.BF16_BYTES)
        )
    return rocdl.readfirstlane(T.i32, as_mlir_value(fx.Int32(lds_addr)))


class DualwaveKvGmemToLdsLoader(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.num_dma_k = self.NUM_DMA_K
        self.num_dma_v = self.NUM_DMA_V

    def _issue_kv_dma(self, src_div, lds_addr, src_elem, soffset):
        _buffer_load_lds_128(
            src_div,
            lds_addr,
            src_elem,
            soffset,
            _dma_atom=self.dma_atom,
            _lds_ptr_ty=self.lds_ptr_ty,
        )

    def _kv_src_div(self, tensor, dense_div, page_id, name):
        if const_expr(self.traits.PAGED):
            if const_expr(page_id is None):
                raise ValueError(f"{name} requires page_id when PAGED=True")
            base_iter = fx.get_iter(tensor)
            return _make_page_view(
                base_iter,
                base_iter.type,
                base_iter.alignment,
                page_id,
                self.page_byte_stride,
                self.page_nrec_bytes,
                self.page_layout,
                self.elem_ir,
                self.buf_flags_i32,
            )
        return dense_div

    def _async_load_kv_linear(self, dma_m0, buf_id, src_div, src_base, soffset, num_dma):
        for d in range_constexpr(num_dma):
            self._issue_kv_dma(
                src_div,
                dma_m0[buf_id][d],
                _linear_kv_src_elem(
                    self.traits,
                    src_base,
                    d,
                    n_in_warp=self.n_in_warp,
                    wave_id=self.wave_id,
                    d_bucket=self.d_bucket,
                    stride_kv_n_v=self.stride_kv_n_v,
                ),
                soffset,
            )

    def load_k(self, tile_start, buf_id, page_id=None):
        ctx = self.ctx_ref
        src_base, soffset = _kv_tile_addr(
            self.traits,
            tile_start,
            kv_gmem_elem_offset=self.kv_gmem_elem_offset,
            kv_head_elem_offset=self.kv_head_elem_offset,
            stride_kv_n_v=self.stride_kv_n_v,
        )
        src_div = self._kv_src_div(self.K, self.k_div, page_id, "DualwaveKvGmemToLdsLoader.load_k")
        if const_expr(self.traits.KV_VECTORIZED):
            for d in range_constexpr(self.num_dma_k):
                self._issue_kv_dma(
                    src_div,
                    ctx.k_dma_m0[buf_id][d],
                    _vec_k_src_elem(
                        self.traits,
                        d,
                        wave_id_uni=self.wave_id_uni,
                        lane_in_warp=self.lane_in_warp,
                        kv_head_idx=self.kv_head_idx,
                    ),
                    soffset,
                )
        else:
            self._async_load_kv_linear(ctx.k_dma_m0, buf_id, src_div, src_base, soffset, self.num_dma_k)

    def load_k_tile(self, tile_idx, buf_id, page_id=None):
        self.load_k(self.tile_start(tile_idx), buf_id, page_id=page_id)

    def load_k_split(self, offset_tiles, buf_id, page_id=None):
        self.load_k_tile(self.split_tile(offset_tiles), buf_id, page_id=page_id)

    def load_v(self, tile_start, buf_id, page_id=None):
        ctx = self.ctx_ref
        src_base, soffset = _kv_tile_addr(
            self.traits,
            tile_start,
            kv_gmem_elem_offset=self.kv_gmem_elem_offset,
            kv_head_elem_offset=self.kv_head_elem_offset,
            stride_kv_n_v=self.stride_kv_n_v,
        )
        src_div = self._kv_src_div(self.V, self.v_div, page_id, "DualwaveKvGmemToLdsLoader.load_v")
        if const_expr(self.traits.KV_VECTORIZED):
            for d in range_constexpr(self.num_dma_v):
                self._issue_kv_dma(
                    src_div,
                    ctx.v_dma_m0[buf_id][d],
                    _vec_v_src_elem(
                        self.traits,
                        d,
                        wave_id_uni=self.wave_id_uni,
                        lane_in_warp=self.lane_in_warp,
                        kv_head_idx=self.kv_head_idx,
                    ),
                    soffset,
                )
        else:
            self._async_load_kv_linear(ctx.v_dma_m0, buf_id, src_div, src_base, soffset, self.num_dma_v)

    def load_v_tile(self, tile_idx, buf_id, page_id=None):
        self.load_v(self.tile_start(tile_idx), buf_id, page_id=page_id)

    def load_v_split(self, offset_tiles, buf_id, page_id=None):
        self.load_v_tile(self.split_tile(offset_tiles), buf_id, page_id=page_id)


class DualwaveKvLdsToVgprLoader(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _load_k_pair(self, buf_id, idx_lo):
        lo = _load_k_pack_aligned(
            self.traits,
            self.lds_kv_base_ptr,
            idx_lo,
            buf_id,
            self.kv_mfma_pack_type,
        )
        hi = _load_k_pack_aligned(
            self.traits,
            self.lds_kv_base_ptr,
            idx_lo + self.traits.DUALWAVE_SWP_URK_N_STRIP_STRIDE,
            buf_id,
            self.kv_mfma_pack_type,
        )
        return lo, hi

    def load_k(self, buf_id, urk_base=None):
        if urk_base is None:
            urk_base = self.k_lds_read_base_per_lane
        k_base = _k_buf_base(self.traits, buf_id)
        k_lo = [None] * self.traits.K_STEPS_QK
        k_hi = [None] * self.traits.K_STEPS_QK

        if const_expr(self.traits.KV_VECTORIZED):
            for ks in range_constexpr(self.traits.K_STEPS_QK):
                k_lo[ks], k_hi[ks] = self._load_k_pair(
                    buf_id,
                    _vec_k_lds_idx_lo(
                        self.traits,
                        k_base,
                        ks,
                        lane_div_32=self.lane_div_32,
                        lane_mod_32=self.lane_mod_32,
                    ),
                )
            return (k_lo, k_hi)

        for ks in range_constexpr(self.traits.K_STEPS_QK):
            k_lo[ks], k_hi[ks] = self._load_k_pair(buf_id, k_base + urk_base + _swizzled_ks_offset(self.traits, ks))
        return (k_lo, k_hi)

    def load_v(self, buf_id, urv_base=None):
        if urv_base is None:
            urv_base = self.v_lds_read_base_per_lane
        v_base = _v_buf_base(self.traits, buf_id)
        packs = [[None] * self.traits.D_CHUNKS for _ in range(4)]
        if const_expr(self.traits.KV_VECTORIZED):
            v_base_ptr = buffer_ops.get_element_ptr(
                self.lds_kv_base_ptr,
                byte_offset=as_mlir_value(
                    fx.Int32(
                        _vec_v_lds_addr_base(
                            self.traits,
                            v_base,
                            lane_div_32=self.lane_div_32,
                            lane_mod_32=self.lane_mod_32,
                        )
                        * self.traits.BF16_BYTES
                    )
                ),
                elem_type=T.i8,
            )
            for dc in range_constexpr(self.traits.D_CHUNKS):
                for k_substep in range_constexpr(4):
                    packs[k_substep][dc] = _read_v8f16_off(
                        self.traits,
                        v_base_ptr,
                        _vec_v_const_off(self.traits, dc, k_substep),
                        self.kv_mfma_pack_type,
                    )
            return packs

        lds_base = v_base + urv_base
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for k_substep in range_constexpr(4):
                imm_lo = _swizzled_v_imm_lo(self.traits, dc, k_substep)
                a = _ds_read_tr_v4f16_imm(
                    lds_base,
                    imm_lo,
                    lds_kv_base_idx=self.lds_kv_base_idx,
                    v_lds_read_vec4_type=self.v_lds_read_vec4_type,
                )
                b = _ds_read_tr_v4f16_imm(
                    lds_base,
                    imm_lo + self.traits.DUALWAVE_SWP_URV_I5_STRIDE * self.traits.BF16_BYTES,
                    lds_kv_base_idx=self.lds_kv_base_idx,
                    v_lds_read_vec4_type=self.v_lds_read_vec4_type,
                )
                packs[k_substep][dc] = Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
        return packs


class DualwaveStoreHelper(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def zero_o_block_if_needed(self, causal_end_raw_i32=None):
        if causal_end_raw_i32 is None:
            causal_end_raw_i32 = self.causal_end_raw_i32
        traits = self.traits
        q_start = self.q_start
        wave_q_offset = self.wave_q_offset
        lane_mod_32 = self.lane_mod_32
        seq_len_v = self.seq_len_v
        stride_q_n_v = self.stride_q_n_v
        q_head_idx = self.q_head_idx
        lane_div_32 = self.lane_div_32
        o_store_reg_128 = self.o_store_reg_128
        store_atom_128 = self.store_atom_128
        o_div = self.o_div

        @flyc.jit
        def _zero_o_block_if_needed():
            if causal_end_raw_i32 <= fx.Int32(0):
                q_row_z = q_start + wave_q_offset + lane_mod_32
                c_zero_i = fx.Int32(0)
                zero_pack = Vec.from_elements([c_zero_i, c_zero_i, c_zero_i, c_zero_i], fx.Int32)
                if q_row_z < seq_len_v:
                    o_base_z = q_row_z * stride_q_n_v + q_head_idx * traits.HEAD_DIM + lane_div_32 * 8
                    for dc in range_constexpr(traits.D_CHUNKS):
                        for g in range_constexpr(2):
                            o_global_z = o_base_z + (dc * traits.D_CHUNK + 2 * g * 8)
                            _buffer_store_128(
                                zero_pack,
                                o_global_z,
                                _o_store_reg_128=o_store_reg_128,
                                _store_atom_128=store_atom_128,
                                o_div=o_div,
                            )

        _zero_o_block_if_needed()

    def _splitk_workspace_resources(self):
        split_z = _splitk_workspace_split_z(self.traits, self.batch_idx, self.split_idx)
        return _splitk_workspace_resources(
            self.ws_base_i64,
            split_z,
            self.ws_opart_per_split_bytes,
            self.ws_ml_per_split_bytes,
            self.ws_mrow_abs_bytes,
            self.ws_lrow_abs_bytes,
        )

    def store_empty_split(self):
        traits = self.traits
        batch_idx = self.batch_idx
        split_idx = self.split_idx
        seq_len_v = self.seq_len_v
        q_start = self.q_start
        q_head_idx = self.q_head_idx
        wave_q_offset = self.wave_q_offset
        lane_mod_32 = self.lane_mod_32
        lane_div_32 = self.lane_div_32
        lane = self.lane
        max_num_tiles = self.max_num_tiles
        split_t0 = self.split_t0
        c_zero_f = self.c_zero_f
        ws_base_i64 = self.ws_base_i64
        ws_opart_per_split_bytes = self.ws_opart_per_split_bytes
        ws_ml_per_split_bytes = self.ws_ml_per_split_bytes
        ws_mrow_abs_bytes = self.ws_mrow_abs_bytes
        ws_lrow_abs_bytes = self.ws_lrow_abs_bytes

        @flyc.jit
        def _store_empty_split():
            if max_num_tiles < split_t0 + fx.Index(4):
                q_row_e = q_start + wave_q_offset + lane_mod_32
                split_z_e = _splitk_workspace_split_z(traits, batch_idx, split_idx)
                _opart_rsrc_e, _mrow_rsrc_e, _lrow_rsrc_e = _splitk_workspace_resources(
                    ws_base_i64,
                    split_z_e,
                    ws_opart_per_split_bytes,
                    ws_ml_per_split_bytes,
                    ws_mrow_abs_bytes,
                    ws_lrow_abs_bytes,
                )
                local_opart_base_e = _splitk_local_opart_row_base(traits, q_head_idx, seq_len_v, q_row_e)
                local_ml_e = _splitk_local_ml_idx(q_head_idx, seq_len_v, q_row_e)
                if q_row_e < seq_len_v:
                    _store_empty_splitk_o_partial_row(traits, local_opart_base_e, lane_div_32, _opart_rsrc_e)
                    if lane < fx.Index(32):
                        _store_splitk_ml_row(fx.Float32(-1e30), c_zero_f, local_ml_e, _mrow_rsrc_e, _lrow_rsrc_e)

        _store_empty_split()

    def store_final_o(self, v_o, q_row):
        _store_final_o_row(
            self.traits,
            v_o,
            q_row,
            self.stride_q_n_v,
            self.q_head_idx,
            self.lane_div_32,
            self.elem_dtype,
            self.o_store_reg_128,
            self.store_atom_128,
            self.o_div,
        )

    def store_splitk_partial_o(self, v_o, m_row, l_row, q_row):
        _opart_rsrc, _mrow_rsrc, _lrow_rsrc = self._splitk_workspace_resources()
        local_opart_row_base = _splitk_local_opart_row_base(self.traits, self.q_head_idx, self.seq_len_v, q_row)
        local_ml_idx = _splitk_local_ml_idx(self.q_head_idx, self.seq_len_v, q_row)
        traits = self.traits
        seq_len_v = self.seq_len_v
        lane_div_32 = self.lane_div_32
        elem_dtype = self.elem_dtype
        lane = self.lane

        @flyc.jit
        def _store_splitk_partial_if_qrow():
            if q_row < seq_len_v:
                _store_splitk_partial_o_row(
                    traits,
                    v_o,
                    local_opart_row_base,
                    lane_div_32,
                    _opart_rsrc,
                    elem_dtype,
                )
                if lane < fx.Index(32):
                    _store_splitk_ml_row(m_row, l_row, local_ml_idx, _mrow_rsrc, _lrow_rsrc)

        _store_splitk_partial_if_qrow()


def _causal_mask_inplace(traits, v_s, tile_idx, delta_i32, lane_div_32, q_row_i32):
    """Apply causal mask using DUALWAVE_SWP inline-asm attn_mask_vec2_imm (DUALWAVE_SWP u_rk path)."""
    s_lo, s_hi = v_s
    kv_tile_start = tile_idx * traits.BLOCK_N
    kv_start_i32 = fx.Int32(kv_tile_start)
    # lane>=32 holds n offset by +8 in the K-permuted P layout (vs +4 in the
    # interleaved layout); thresholds below are the lane-independent n part.
    _lane_n_off = 8 if traits.KV_VECTORIZED else 4
    lane_off_i32 = fx.Int32(lane_div_32) * fx.Int32(_lane_n_off)
    # Bottom-right causal: keep key col <= q_row + delta (delta=seqlen_kv-seqlen_q).
    rel_lo_i32 = fx.Int32(q_row_i32 + delta_i32 - kv_start_i32 - lane_off_i32)
    # v_s_hi: i_n=1, so N += W_N = 32
    rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(32))
    neg_inf_i32 = fx.Int32(traits.NEG_INF_F32_BITS)

    if const_expr(traits.KV_VECTORIZED):
        # Vectorized K-read makes P land in consecutive-n groups: {0..7, 16..23}.
        pair_thresholds = [
            (0, 1),
            (2, 3),  # r=0,1  r=2,3
            (4, 5),
            (6, 7),  # r=4,5  r=6,7
            (16, 17),
            (18, 19),  # r=8,9  r=10,11
            (20, 21),
            (22, 23),  # r=12,13 r=14,15
        ]
    else:
        pair_thresholds = [
            (0, 1),
            (2, 3),  # r=0,1  r=2,3
            (8, 9),
            (10, 11),  # r=4,5  r=6,7
            (16, 17),
            (18, 19),  # r=8,9  r=10,11
            (24, 25),
            (26, 27),  # r=12,13 r=14,15
        ]
    for p in range_constexpr(len(pair_thresholds)):
        thr_x, thr_y = pair_thresholds[p]
        idx_x = p * 2
        idx_y = p * 2 + 1

        # s_lo pair (n_strip = 0)
        x_lo_bits = _bitcast_i32(s_lo[idx_x])
        y_lo_bits = _bitcast_i32(s_lo[idx_y])
        new_x_lo, new_y_lo = _attn_mask_vec2_imm(
            rel_lo_i32,
            neg_inf_i32,
            thr_x,
            thr_y,
            x_lo_bits,
            y_lo_bits,
        )
        s_lo[idx_x] = _bitcast_f32(new_x_lo)
        s_lo[idx_y] = _bitcast_f32(new_y_lo)

    for p in range_constexpr(len(pair_thresholds)):
        thr_x, thr_y = pair_thresholds[p]
        idx_x = p * 2
        idx_y = p * 2 + 1
        # s_hi pair (n_strip = 1, rel shifted by 4)
        x_hi_bits = _bitcast_i32(s_hi[idx_x])
        y_hi_bits = _bitcast_i32(s_hi[idx_y])
        new_x_hi, new_y_hi = _attn_mask_vec2_imm(
            rel_hi_i32,
            neg_inf_i32,
            thr_x,
            thr_y,
            x_hi_bits,
            y_hi_bits,
        )
        s_hi[idx_x] = _bitcast_f32(new_x_hi)
        s_hi[idx_y] = _bitcast_f32(new_y_hi)


def _v_pair_to_vec32(v):
    return _concat_vectors(v[0], v[1]).ir_value()


def _v_vec32_to_pair(v):
    v_vec = Vec(v, (32,), fx.Float32)
    v_lo = v_vec.shuffle(v_vec, [i for i in range(16)]).ir_value()
    v_hi = v_vec.shuffle(v_vec, [16 + i for i in range(16)]).ir_value()
    return v_lo, v_hi


def _anchor_v_o(traits, v_o):
    """Pin v_o accumulators at the current source position."""
    acc_irs = [as_mlir_value(v_o[dc]) for dc in range_constexpr(traits.D_CHUNKS)]
    ret_ty = ir.Type.parse(f"!llvm.struct<({', '.join(['vector<16xf32>'] * traits.D_CHUNKS)})>")
    constraints = ",".join(["=v"] * traits.D_CHUNKS + [str(i) for i in range(traits.D_CHUNKS)])
    ret = llvm.inline_asm(
        ret_ty,
        acc_irs,
        "",
        constraints,
        has_side_effects=True,
    )
    return [llvm.extractvalue(acc_irs[dc].type, ret, [dc]) for dc in range_constexpr(traits.D_CHUNKS)]


def _debug_atomic_inc_lazy_count(byte_offset, debug_counts_rsrc):
    rocdl.raw_buffer_atomic_fadd(
        as_mlir_value(fx.Float32(1.0)),
        debug_counts_rsrc,
        as_mlir_value(fx.Int32(byte_offset)),
        as_mlir_value(fx.Int32(0)),
        as_mlir_value(fx.Int32(0)),
    )


@flyc.jit
def _debug_count_lazy_branch(traits, all_below, debug_counts_rsrc, lane):
    if const_expr(traits.DUALWAVE_SWP_DEBUG_LAZY_COUNTS):
        if fx.Int32(lane) == fx.Int32(0):
            if fx.Boolean(all_below):
                _debug_atomic_inc_lazy_count(0, debug_counts_rsrc=debug_counts_rsrc)
            else:
                _debug_atomic_inc_lazy_count(4, debug_counts_rsrc=debug_counts_rsrc)


def _dualwave_sync_barrier():
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def dualwave_splitk_workspace_elems(batch_size, num_heads, seq_len, num_kv_splits, head_dim=128):
    """fp32 elements needed for the split-K workspace: O_partial + Mrow + Lrow.

    O_partial is stored as kernel-native 16-bit (bf16/fp16), two columns per
    fp32 slot; Mrow/Lrow stay fp32.
    """
    rows = batch_size * num_kv_splits * num_heads * seq_len
    return rows * (head_dim // 2) + 2 * rows
