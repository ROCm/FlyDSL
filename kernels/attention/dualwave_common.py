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

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, vector
from flydsl.expr import rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw

_LOG2E = host_math.log2(host_math.e)
# s_waitcnt bitfield encoding
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3
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
    DEFAULT_STRIDE_Q_N: int
    DEFAULT_STRIDE_KV_N: int
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


def make_dualwave_swp_traits(num_heads, num_kv_heads, head_dim):
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
        DEFAULT_STRIDE_Q_N=default_stride_q_n,
        DEFAULT_STRIDE_KV_N=default_stride_kv_n,
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
    )


def _ds_read_tr16_b64_imm(result_type, addr_i32, imm_offset=0):
    """gfx950 ds_read_b64_tr_b16 with DUALWAVE_SWP immediate byte offset."""
    imm = int(imm_offset)
    raw_type = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    raw = llvm.inline_asm(
        raw_type,
        [_raw(addr_i32)],
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
        [_raw(addr_i32)],
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
    return _raw(ArithValue(value).bitcast(fx.Int32.ir_type))


def _bitcast_f32(value):
    return _raw(ArithValue(value).bitcast(fx.Float32.ir_type))


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
            _raw(x_ref_i32),
            _raw(y_ref_i32),
            _raw(rel_i32),
            _raw(neg_inf_i32),
        ],
        asm_str,
        "=s,=s,=v,=v,2,3,v,v,~{vcc}",
        has_side_effects=True,
    )
    return llvm.extractvalue(T.i32, ret, [2]), llvm.extractvalue(T.i32, ret, [3])


def _anchor_pair(v_s):
    lo, hi = v_s
    lo_ir = _raw(lo)
    hi_ir = _raw(hi)
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
    x_ir = _raw(x)
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


def dualwave_splitk_workspace_elems(batch_size, num_heads, seq_len, num_kv_splits, head_dim=128):
    """fp32 elements needed for the split-K workspace: O_partial + Mrow + Lrow.

    O_partial is stored as kernel-native 16-bit (bf16/fp16), two columns per
    fp32 slot; Mrow/Lrow stay fp32.
    """
    rows = batch_size * num_kv_splits * num_heads * seq_len
    return rows * (head_dim // 2) + 2 * rows
