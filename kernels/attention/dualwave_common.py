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
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

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


def make_dualwave_swp_traits(
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
def _stagger_extra_barrier_if_one(_stagger_i32):
    """Emit `sched_barrier(0); s_barrier;` only when stagger == 1."""
    if fx.Int32(_stagger_i32) != fx.Int32(0):
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
