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
from flydsl._mlir.dialects import scf as _scf
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace as _TargetAddressSpace
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from kernels.common.kernels_common import _if_then, dtype_to_elem_type

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


def _read_v8f16_off(v_base_ptr, const_off, TRAITS_BF16_BYTES, mfma_pack_type):
    ptr = buffer_ops.get_element_ptr(
        v_base_ptr, byte_offset=as_mlir_value(fx.Int32(const_off * TRAITS_BF16_BYTES)), elem_type=T.i8
    )
    return llvm.LoadOp(mfma_pack_type, ptr, alignment=16).result


def _load_k_pack_aligned(
    lds_kv_base_ptr,
    elem_idx,
    buf_id,
    mfma_pack_type,
    lds_scope_names,
    TRAITS_BF16_BYTES,
):
    scope_name = _dualwave_lds_scope("k", buf_id)
    byte_offset = elem_idx * TRAITS_BF16_BYTES
    ptr = buffer_ops.get_element_ptr(lds_kv_base_ptr, byte_offset=byte_offset, elem_type=T.i8)
    return llvm.LoadOp(
        mfma_pack_type,
        ptr,
        alignment=16,
        alias_scopes=_dualwave_lds_alias_scopes(scope_name),
        noalias_scopes=_dualwave_lds_noalias_scopes(scope_name, lds_scope_names),
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


# Helpers moved out of flash_attn_gfx950.py to keep the kernel body definition-free.


def _load_block_table_to_lds(
    traits,
    *,
    _bt_atom,
    _bt_div,
    _bt_v1i32,
    batch_idx,
    block_table_stride_v,
    lds_bt_base_ptr,
    num_kv_tiles,
    split_t0,
    split_t_end,
    tid,
):
    segment_tiles = split_t_end - split_t0
    for pass_id in range_constexpr(traits.PAGED_BT_LDS_SIZE // traits.BLOCK_SIZE):
        local_tile = tid + fx.Index(pass_id * traits.BLOCK_SIZE)
        with _if_then(_scf.IfOp(as_mlir_value(ArithValue(local_tile < segment_tiles)))):
            tile_idx = split_t0 + local_tile
            byte_off = as_mlir_value(fx.Int32(local_tile * fx.Index(4)))
            dst = buffer_ops.get_element_ptr(lds_bt_base_ptr, byte_offset=byte_off, elem_type=T.i8)
            llvm.StoreOp(as_mlir_value(fx.Int32(0)), dst)
            with _if_then(_scf.IfOp(as_mlir_value(ArithValue(tile_idx < num_kv_tiles)))):
                row_idx = batch_idx * block_table_stride_v + tile_idx
                v = fly.copy_atom_call_ssa([_bt_v1i32], _bt_atom, fx.slice(_bt_div, (None, fx.Int32(row_idx))))
                page_id_i32 = as_mlir_value(fx.Int32(Vec(v, (1,), fx.Int32)[0]))
                llvm.StoreOp(page_id_i32, dst)


def _finish_page_id(v, *, traits):
    rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
    v = rocdl.readfirstlane(T.i32, v)
    return fx.Index(fx.Int32(v))


def _make_rebased_view(
    base_iter,
    byte_off,
    nrec_bytes,
    layout,
    *,
    _buf_flags_i32,
    _elem_ir,
):
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


def _buffer_load_128(
    elem_index,
    *,
    _load_atom_128,
    q_div,
    v4i32_type,
):
    """128-bit global->register load (buffer_load_dwordx4) from Q."""
    return fly.copy_atom_call_ssa([v4i32_type], _load_atom_128, fx.slice(q_div, (None, fx.Int32(elem_index))))


def _buffer_load_lds_128(
    src_div,
    lds_byte_addr,
    src_elem,
    soffset_elems,
    *,
    _dma_atom,
    _lds_ptr_ty,
):
    """128-bit global->LDS DMA; `src_elem` is voffset, `soffset_elems` is scaled by the atom."""
    lds_ptr = fx.inttoptr(_lds_ptr_ty, fx.Int32(lds_byte_addr))
    dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
    src = fx.slice(src_div, (None, fx.Int32(src_elem)))
    fx.copy(_dma_atom, src, dst, soffset=fx.Int32(soffset_elems))


def _buffer_store_64(
    pack_i32_vec,
    elem_index,
    *,
    _o_store_reg,
    _store_atom_64,
    o_div,
):
    """64-bit register->global store (buffer_store_dwordx2) into O."""
    fx.memref_store_vec(pack_i32_vec, _o_store_reg)
    fx.copy(_store_atom_64, _o_store_reg, fx.slice(o_div, (None, fx.Int32(elem_index))))


def _buffer_store_128(
    pack_i32_vec,
    elem_index,
    *,
    _o_store_reg_128,
    _store_atom_128,
    o_div,
):
    """128-bit register->global store (buffer_store_dwordx4) into O."""
    fx.memref_store_vec(pack_i32_vec, _o_store_reg_128)
    fx.copy(_store_atom_128, _o_store_reg_128, fx.slice(o_div, (None, fx.Int32(elem_index))))


def _fadd(
    a,
    b,
    *,
    fm_fast,
):
    return arith.addf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fsub(
    a,
    b,
    *,
    fm_fast,
):
    return arith.subf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fmul(
    a,
    b,
    *,
    fm_fast,
):
    return arith.mulf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fmax(
    a,
    b,
    *,
    fm_fast,
):
    return arith.MaxNumFOp(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast).result


def _mfma_acc(
    a,
    b,
    c,
    *,
    _mma_atom,
    v16f32_type,
):
    return fly.mma_atom_call_ssa([v16f32_type], _mma_atom, a, b, c)


def _sched_barrier_pairs(
    pairs,
    valu_cnt,
    group,
    traits,
):
    """Emit `pairs` × {1 MFMA + valu_cnt VALU} sched_group_barrier groups."""
    pairs = _scale_sched_pairs(pairs, traits.HEAD_DIM)
    for _ in range_constexpr(pairs):
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, group)
        rocdl.sched_group_barrier(traits.SCHED_VALU_MASK, valu_cnt, group)


def _sched_barrier_exp_pairs(
    pairs,
    exp_cnt,
    group,
    traits,
):
    """Emit `pairs` × {1 MFMA + exp_cnt EXP} sched_group_barrier groups."""
    pairs = _scale_sched_pairs(pairs, traits.HEAD_DIM)
    for _ in range_constexpr(pairs):
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, group)
        rocdl.sched_group_barrier(traits.SCHED_EXP_MASK, exp_cnt, group)


def _ds_read_tr_v4f16_imm(
    lds_base_elem_idx,
    imm_bytes,
    *,
    lds_kv_base_idx,
    v4f16_type,
):
    byte_offset = lds_base_elem_idx * 2 + lds_kv_base_idx
    addr_i32 = fx.Int32(byte_offset)
    return _ds_read_tr16_b64_imm(v4f16_type, addr_i32, imm_bytes)


def _get_q_pack(q_all_scaled_bf16, ks, traits):
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


def _anchor_v_p(
    v_p,
    traits,
    *,
    elem_dtype,
):
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


def _v_vec32_to_p(
    v_p_all,
    traits,
    *,
    elem_dtype,
):
    p_vec = Vec(v_p_all, (traits.PV_K_STEPS * 2 * 8,), elem_dtype)
    p_lo = []
    p_hi = []
    for pks in range_constexpr(traits.PV_K_STEPS):
        lo_base = pks * 8
        hi_base = traits.PV_K_STEPS * 8 + pks * 8
        p_lo.append(p_vec.shuffle(p_vec, [lo_base + i for i in range(8)]).ir_value())
        p_hi.append(p_vec.shuffle(p_vec, [hi_base + i for i in range(8)]).ir_value())
    return p_lo, p_hi


def _scale_v_p(
    v_p,
    scale_scalar,
    traits,
    *,
    elem_dtype,
    fm_fast,
    v32bf16_type,
    v32f32_type,
):
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
    return _v_vec32_to_p(p_scaled_bf16_op.result, traits, elem_dtype=elem_dtype)


def _rescale_params(traits):
    elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)
    fm_fast = fx.arith.FastMathFlags.fast
    v32bf16_type = Vec.make_type(traits.PV_K_STEPS * 2 * 8, elem_dtype)
    v32f32_type = Vec.make_type(traits.PV_K_STEPS * 2 * 8, fx.Float32)
    return elem_dtype, fm_fast, v32bf16_type, v32f32_type


@flyc.jit
def _stagger_extra_barrier_if_one(*, _stagger_i32):
    """Emit `sched_barrier(0); s_barrier;` only when stagger == 1."""
    if fx.Int32(_stagger_i32) != fx.Int32(0):
        rocdl.sched_barrier(0)
        rocdl.s_barrier()


def _bf16_trunc_pack_v8(
    f32_vals,
    *,
    dtype_str,
    elem_dtype,
):
    if const_expr(dtype_str == "bf16"):
        pairs = []
        for j in range_constexpr(4):
            pairs.append(rocdl.cvt_pk_bf16_f32(f32_vals[j * 2], f32_vals[j * 2 + 1]))
        return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()
    # fp16: truncate each f32 -> f16 (RNE) and build the v8 pack directly.
    f16_vals = []
    for i in range_constexpr(8):
        f16_vals.append(fx.Float32(f32_vals[i]).to(elem_dtype))
    return Vec.from_elements(f16_vals, elem_dtype).ir_value()


def _k_buf_base(buf_id, TRAITS_DUALWAVE_SWP_K_BUF_BASE, TRAITS_DUALWAVE_SWP_KV_PER_BUFFER):
    if const_expr(isinstance(buf_id, int)):
        return TRAITS_DUALWAVE_SWP_K_BUF_BASE[buf_id]
    # runtime buf_id (rare): K0=0, K1=DUALWAVE_SWP_KV_PER_BUFFER
    return buf_id * TRAITS_DUALWAVE_SWP_KV_PER_BUFFER


def _v_buf_base(
    buf_id,
    TRAITS_DUALWAVE_SWP_V_BUF_BASE,
    TRAITS_SMEM_K_TILE_ELEMS,
    TRAITS_DUALWAVE_SWP_KV_PER_BUFFER,
):
    if const_expr(isinstance(buf_id, int)):
        return TRAITS_DUALWAVE_SWP_V_BUF_BASE[buf_id]
    return TRAITS_SMEM_K_TILE_ELEMS + buf_id * TRAITS_DUALWAVE_SWP_KV_PER_BUFFER


def _kv_tile_addr(
    tile_start,
    *,
    PAGED,
    kv_gmem_elem_offset,
    kv_head_elem_offset,
    stride_kv_n_v,
):
    """Return (src_base, soffset): dense uses tile_start*stride; paged folds page offset into the descriptor."""
    if const_expr(PAGED):
        return kv_head_elem_offset, 0
    return kv_gmem_elem_offset, tile_start * stride_kv_n_v


def _k_dma_m0_base(
    buf_id,
    d,
    traits,
    *,
    lane_in_warp,
    lds_kv_base_idx,
    wave_id_uni,
):
    k_lds_byte_base = (
        lds_kv_base_idx
        + _k_buf_base(buf_id, traits.DUALWAVE_SWP_K_BUF_BASE, traits.DUALWAVE_SWP_KV_PER_BUFFER) * traits.BF16_BYTES
    )
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


def _v_dma_m0_base(
    buf_id,
    d,
    traits,
    *,
    lane_in_warp,
    lds_kv_base_idx,
    wave_id_uni,
):
    v_lds_byte_base = (
        lds_kv_base_idx
        + _v_buf_base(
            buf_id,
            traits.DUALWAVE_SWP_V_BUF_BASE,
            traits.SMEM_K_TILE_ELEMS,
            traits.DUALWAVE_SWP_KV_PER_BUFFER,
        )
        * traits.BF16_BYTES
    )
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


def _mma0(
    v_k,
    *,
    traits,
    _mma_atom,
    c_zero_v16f32,
    q_all_scaled_bf16,
    v16f32_type,
):
    k_lo, k_hi = v_k
    v_s_lo = c_zero_v16f32
    v_s_hi = c_zero_v16f32
    for ks in range_constexpr(traits.K_STEPS_QK):
        q_pack = _get_q_pack(q_all_scaled_bf16, ks, traits)
        v_s_lo = _mfma_acc(k_lo[ks], q_pack, v_s_lo, _mma_atom=_mma_atom, v16f32_type=v16f32_type)
        v_s_hi = _mfma_acc(k_hi[ks], q_pack, v_s_hi, _mma_atom=_mma_atom, v16f32_type=v16f32_type)
    return (v_s_lo, v_s_hi)


def _causal_mask_inplace(
    v_s,
    tile_idx,
    traits,
    *,
    delta_i32,
    lane_div_32,
    q_row_i32,
):
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


def _attn_row_max(
    v_s,
    *,
    c_neg_inf,
    fm_fast,
):
    s_lo, s_hi = v_s
    m = c_neg_inf
    for r in range_constexpr(16):
        m = _fmax(m, s_lo[r], fm_fast=fm_fast)
    for r in range_constexpr(16):
        m = _fmax(m, s_hi[r], fm_fast=fm_fast)
    lhs, rhs = _reduction_pair(m)
    return _fmax(lhs, rhs, fm_fast=fm_fast)


def _mma1_step_k(
    step,
    v_p,
    v_v,
    v_o,
    TRAITS_D_CHUNKS,
    *,
    _mma_atom,
    v16f32_type,
):
    v_p_lo, v_p_hi = v_p
    v_pk = v_v[step]
    if const_expr(step < 2):
        p_pk = v_p_lo[step]
    else:
        p_pk = v_p_hi[step - 2]
    for dc in range_constexpr(TRAITS_D_CHUNKS):
        v_o[dc] = _mfma_acc(v_pk[dc], p_pk, v_o[dc], _mma_atom=_mma_atom, v16f32_type=v16f32_type)
    return v_o


def _mma1(
    v_p,
    v_v,
    v_o,
    *,
    D_CHUNKS,
    _mma_atom,
    v16f32_type,
):
    for step in range_constexpr(4):
        v_o = _mma1_step_k(step, v_p, v_v, v_o, D_CHUNKS, _mma_atom=_mma_atom, v16f32_type=v16f32_type)
    return v_o


def _attn_sub_row(
    v_s,
    row_max,
    *,
    fm_fast,
):
    s_lo, s_hi = v_s
    lo_sub = []
    hi_sub = []
    for r in range_constexpr(16):
        lo_sub.append(_fsub(s_lo[r], row_max, fm_fast=fm_fast))
    for r in range_constexpr(16):
        hi_sub.append(_fsub(s_hi[r], row_max, fm_fast=fm_fast))
    lo_vec = Vec.from_elements(lo_sub, fx.Float32).ir_value()
    hi_vec = Vec.from_elements(hi_sub, fx.Float32).ir_value()
    return lo_vec, hi_vec


def _attn_exp2_slice(v_s, start, length):
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


def _attn_sum(
    v_p,
    *,
    c_zero_f,
    fm_fast,
):
    lo_partial_list, hi_full = v_p
    local_sum = c_zero_f
    for r in range_constexpr(16):
        local_sum = _fadd(local_sum, lo_partial_list[r], fm_fast=fm_fast)
    for r in range_constexpr(16):
        local_sum = _fadd(local_sum, hi_full[r], fm_fast=fm_fast)
    lhs_sum, rhs_sum = _reduction_pair(local_sum)
    return _fadd(lhs_sum, rhs_sum, fm_fast=fm_fast)


def _cast_p(
    v_p,
    traits,
    *,
    elem_dtype,
):
    lo_partial_list, hi_full = v_p
    p_lo_packs = []
    p_hi_packs = []
    # Vectorized QK^T already emits P in the order consumed by V reads.
    for pks in range_constexpr(traits.PV_K_STEPS):
        p_base = pks * 8
        lo_slice = [lo_partial_list[p_base + s] for s in range_constexpr(8)]
        hi_slice = hi_full[p_base : p_base + 8]
        p_lo_packs.append(_bf16_trunc_pack_v8(lo_slice, dtype_str=traits.DTYPE_STR, elem_dtype=elem_dtype))
        p_hi_packs.append(_bf16_trunc_pack_v8(hi_slice, dtype_str=traits.DTYPE_STR, elem_dtype=elem_dtype))
    return p_lo_packs, p_hi_packs


def _scale_o(
    v_o,
    scale_scalar,
    TRAITS_D_CHUNKS,
    *,
    fm_fast,
):
    scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
    for dc in range_constexpr(TRAITS_D_CHUNKS):
        v_o[dc] = _fmul(Vec(v_o[dc]), scale_vec, fm_fast=fm_fast)


def _anchor_v_o(v_o, TRAITS_D_CHUNKS):
    """Pin v_o accumulators at the current source position."""
    acc_irs = [as_mlir_value(v_o[dc]) for dc in range_constexpr(TRAITS_D_CHUNKS)]
    ret_ty = ir.Type.parse(f"!llvm.struct<({', '.join(['vector<16xf32>'] * TRAITS_D_CHUNKS)})>")
    constraints = ",".join(["=v"] * TRAITS_D_CHUNKS + [str(i) for i in range(TRAITS_D_CHUNKS)])
    ret = llvm.inline_asm(
        ret_ty,
        acc_irs,
        "",
        constraints,
        has_side_effects=True,
    )
    return [llvm.extractvalue(acc_irs[dc].type, ret, [dc]) for dc in range_constexpr(TRAITS_D_CHUNKS)]


def _debug_atomic_inc_lazy_count(byte_offset, *, debug_counts_rsrc):
    rocdl.raw_buffer_atomic_fadd(
        as_mlir_value(fx.Float32(1.0)),
        debug_counts_rsrc,
        as_mlir_value(fx.Int32(byte_offset)),
        as_mlir_value(fx.Int32(0)),
        as_mlir_value(fx.Int32(0)),
    )


@flyc.jit
def _debug_count_lazy_branch(
    all_below,
    *,
    DUALWAVE_SWP_DEBUG_LAZY_COUNTS,
    debug_counts_rsrc,
    lane,
):
    if const_expr(DUALWAVE_SWP_DEBUG_LAZY_COUNTS):
        if fx.Int32(lane) == fx.Int32(0):
            if fx.Boolean(all_below):
                _debug_atomic_inc_lazy_count(0, debug_counts_rsrc=debug_counts_rsrc)
            else:
                _debug_atomic_inc_lazy_count(4, debug_counts_rsrc=debug_counts_rsrc)


def _rescale_o(
    v_o,
    m_row,
    l_row,
    m_tile_max,
    v_p,
    traits,
):
    elem_dtype, fm_fast, v32bf16_type, v32f32_type = _rescale_params(traits)
    m_new = _fmax(m_row, m_tile_max, fm_fast=fm_fast)
    corr = rocdl.exp2(T.f32, as_mlir_value(_fsub(m_row, m_new, fm_fast=fm_fast)))
    _scale_o(v_o, corr, traits.D_CHUNKS, fm_fast=fm_fast)
    v_o = _anchor_v_o(v_o, traits.D_CHUNKS)
    v_p = _scale_v_p(
        v_p,
        corr,
        traits,
        elem_dtype=elem_dtype,
        fm_fast=fm_fast,
        v32bf16_type=v32bf16_type,
        v32f32_type=v32f32_type,
    )
    l_row = _fmul(l_row, corr, fm_fast=fm_fast)
    return v_o, m_new, l_row, v_p


def _lazy_rescale_o_rescale(
    _n,
    *_st,
    v_o,
    m_row,
    l_row,
    m_tile_max,
    v_p,
    traits,
):
    elem_dtype, fm_fast, v32bf16_type, v32f32_type = _rescale_params(traits)
    corr = rocdl.exp2(T.f32, as_mlir_value(_fsub(m_row, m_tile_max, fm_fast=fm_fast)))
    scaled_accs = list(v_o)
    _scale_o(scaled_accs, corr, traits.D_CHUNKS, fm_fast=fm_fast)
    out = [as_mlir_value(scaled_accs[dc]) for dc in range(traits.D_CHUNKS)]
    scaled_p = _scale_v_p(
        v_p,
        corr,
        traits,
        elem_dtype=elem_dtype,
        fm_fast=fm_fast,
        v32bf16_type=v32bf16_type,
        v32f32_type=v32f32_type,
    )
    out.append(_v_p_to_vec32(scaled_p))
    out.append(as_mlir_value(_fmul(l_row, corr, fm_fast=fm_fast)))
    out.append(_anchor_scalar_f32(m_tile_max))
    return out


@flyc.jit
def _lazy_rescale_o(
    v_o,
    m_row,
    l_row,
    m_tile_max,
    v_p,
    traits,
    *,
    debug_counts_rsrc,
    lane,
):
    """DUALWAVE_SWP lazy rescale before the remaining MMA1 steps."""
    elem_dtype, fm_fast, _, _ = _rescale_params(traits)
    c_eight_f = fx.Float32(traits.DUALWAVE_SWP_RESCALE_THRESHOLD)
    m_diff = _fsub(m_tile_max, m_row, fm_fast=fm_fast)
    below = ArithValue(fx.Float32(m_diff) <= c_eight_f)
    ballot = rocdl.ballot(T.i64, as_mlir_value(below))
    all_below = arith.cmpi(
        arith.CmpIPredicate.eq,
        as_mlir_value(ballot),
        _read_exec_i64(),
    )
    all_below = llvm.intr_expect(all_below, arith.constant(1, type=ir.IntegerType.get_signless(1)))
    _debug_count_lazy_branch(
        all_below,
        DUALWAVE_SWP_DEBUG_LAZY_COUNTS=traits.DUALWAVE_SWP_DEBUG_LAZY_COUNTS,
        debug_counts_rsrc=debug_counts_rsrc,
        lane=lane,
    )

    # Drive scf.if with explicit accumulator/P/l/m state; all_below keeps it unchanged.
    _state = [as_mlir_value(v_o[dc]) for dc in range(traits.D_CHUNKS)]
    _state += [_v_p_to_vec32(v_p), as_mlir_value(l_row), as_mlir_value(m_row)]
    _names = tuple("_lr%d" % i for i in range(traits.D_CHUNKS + 3))

    _rescale = lambda _n, *_st: _lazy_rescale_o_rescale(
        _n,
        *_st,
        v_o=v_o,
        m_row=m_row,
        l_row=l_row,
        m_tile_max=m_tile_max,
        v_p=v_p,
        traits=traits,
    )

    _res = scf_if_dispatch(all_below, lambda *_a: None, _rescale, state_names=_names, state_values=_state)
    o_out = list(_res[0 : traits.D_CHUNKS])
    vp_out = _res[traits.D_CHUNKS]
    l_out = _res[traits.D_CHUNKS + 1]
    m_out = _res[traits.D_CHUNKS + 2]
    return (o_out, m_out, l_out, _v_vec32_to_p(vp_out, traits, elem_dtype=elem_dtype))


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
