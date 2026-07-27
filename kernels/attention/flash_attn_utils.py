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
import os
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, vector
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace as _TargetAddressSpace
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from flydsl.utils.smem_allocator import SmemPtr
from kernels.common import buffer_ops
from kernels.common.kernels_common import dtype_to_elem_type

_LOG2E = host_math.log2(host_math.e)
# s_waitcnt bitfield encoding
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3
scf_if_dispatch = ReplaceIfWithDispatch.scf_if_dispatch

_LDS_ALIAS_DOMAIN = '#llvm.alias_scope_domain<id = "flydsl.dualwave_swp.lds">'


# Wait and low-level ROCDL wrappers


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


def _dualwave_sync_barrier():
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _s_nop(x):
    if not isinstance(x, int) or not 0 <= x <= 15:
        raise ValueError("s_nop immediate must be a Python int in [0, 15]")
    llvm.inline_asm(ir.Type.parse("!llvm.void"), [], f"s_nop {x}", "", has_side_effects=True)


def _read_exec_i64():
    """Read the current wave exec mask, matching Clang's builtin lowering."""
    true_i1 = fx.Boolean(True).ir_value()
    return rocdl.ballot(T.i64, true_i1)


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


# Arithmetic and inline-asm primitives


def _fadd(a, b, fm_fast):
    return arith.addf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fsub(a, b, fm_fast):
    return arith.subf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fmul(a, b, fm_fast):
    return arith.mulf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _tree_reduce(vals, binop):
    items = list(vals)
    while len(items) > 1:
        nxt = [binop(items[i], items[i + 1]) for i in range(0, len(items) - 1, 2)]
        if len(items) % 2 == 1:
            nxt.append(items[-1])
        items = nxt
    return items[0]


def _fmax(a, b, fm_fast):
    return arith.MaxNumFOp(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast).result


def _mfma_acc(a, b, c, _mma_atom, mfma_acc_vec_type):
    return fly.mma_atom_call_ssa([mfma_acc_vec_type], _mma_atom, a, b, c)


def _concat_vectors(lhs, rhs):
    lhs_vec = Vec(lhs)
    rhs_vec = Vec(rhs)
    return lhs_vec.shuffle(
        rhs_vec,
        list(range(lhs_vec.numel)) + [lhs_vec.numel + i for i in range(rhs_vec.numel)],
    )


def _bitcast_i32(value):
    return as_mlir_value(fx.Float32(value).bitcast(fx.Int32).ir_value())


def _bitcast_f32(value):
    return as_mlir_value(fx.Int32(value).bitcast(fx.Float32).ir_value())


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


def _reduction_pair(v_f32):
    v_i32 = _bitcast_i32(v_f32)
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_ty, v_i32, v_i32, False, True)
    lhs_i32 = llvm.extractvalue(T.i32, swapped, [0])
    rhs_i32 = llvm.extractvalue(T.i32, swapped, [1])
    return _bitcast_f32(lhs_i32), _bitcast_f32(rhs_i32)


def _swap_halves(dw):
    pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False)
    lo_res = llvm.extractvalue(T.i32, swapped, [0])
    hi_res = llvm.extractvalue(T.i32, swapped, [1])
    return lo_res, hi_res


def _fused_o_128_dwords(lane_div_32, d0_a, d1_a, d0_b, d1_b):
    is_hi_half = lane_div_32 != fx.Index(0)
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


def _packed_o_128_dwords(traits, v_o, dc, g, lane_div_32, elem_dtype):
    d0_a, d1_a = _o_pack_2dw(traits, v_o, dc, 2 * g, elem_dtype)
    d0_b, d1_b = _o_pack_2dw(traits, v_o, dc, 2 * g + 1, elem_dtype)
    return _fused_o_128_dwords(lane_div_32, d0_a, d1_a, d0_b, d1_b)


def _packed_o_128_vec(traits, v_o, dc, g, lane_div_32, elem_dtype):
    return Vec.from_elements(
        [fx.Int32(w) for w in _packed_o_128_dwords(traits, v_o, dc, g, lane_div_32, elem_dtype)],
        fx.Int32,
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


def _v_pair_to_vec32(v):
    return _concat_vectors(v[0], v[1]).ir_value()


def _v_vec32_to_pair(v):
    v_vec = Vec(v, (32,), fx.Float32)
    v_lo = v_vec.shuffle(v_vec, [i for i in range(16)]).ir_value()
    v_hi = v_vec.shuffle(v_vec, [16 + i for i in range(16)]).ir_value()
    return v_lo, v_hi


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


def _score_pair_to_lists(v_s):
    s_lo, s_hi = v_s
    return (
        [Vec(s_lo)[r] for r in range_constexpr(16)],
        [Vec(s_hi)[r] for r in range_constexpr(16)],
    )


def _score_lists_to_vecs(v_s_lists):
    s_lo, s_hi = v_s_lists
    return (
        Vec.from_elements([as_mlir_value(v) for v in s_lo], fx.Float32).ir_value(),
        Vec.from_elements([as_mlir_value(v) for v in s_hi], fx.Float32).ir_value(),
    )


def _reduce_score_pair(v_s, initial, reducer, fm_fast):
    s_lo, s_hi = v_s
    acc = initial
    for r in range_constexpr(16):
        acc = reducer(acc, s_lo[r], fm_fast)
    for r in range_constexpr(16):
        acc = reducer(acc, s_hi[r], fm_fast)
    return acc


def _lane_pair_reduce(v, reducer, fm_fast):
    lhs, rhs = _reduction_pair(v)
    return reducer(lhs, rhs, fm_fast)


def _score_pair_max(v_s, neg_inf, fm_fast):
    return _lane_pair_reduce(_reduce_score_pair(v_s, neg_inf, _fmax, fm_fast), _fmax, fm_fast)


def _score_pair_sum(v_s, zero_f, fm_fast):
    return _lane_pair_reduce(_reduce_score_pair(v_s, zero_f, _fadd, fm_fast), _fadd, fm_fast)


def _sub_score_pair(v_s, row_max, fm_fast):
    s_lo, s_hi = v_s
    lo_sub = []
    hi_sub = []
    for r in range_constexpr(16):
        lo_sub.append(_fsub(s_lo[r], row_max, fm_fast))
    for r in range_constexpr(16):
        hi_sub.append(_fsub(s_hi[r], row_max, fm_fast))
    return Vec.from_elements(lo_sub, fx.Float32).ir_value(), Vec.from_elements(hi_sub, fx.Float32).ir_value()


def _exp2_score_slice(v_s, start, length):
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


def _pack_p_v8_slices(traits, v_p, pack_v8_fn):
    lo_partial_list, hi_full = v_p
    p_lo_packs = []
    p_hi_packs = []
    for pks in range_constexpr(traits.PV_K_STEPS):
        p_base = pks * 8
        lo_slice = [lo_partial_list[p_base + s] for s in range_constexpr(8)]
        hi_slice = hi_full[p_base : p_base + 8]
        p_lo_packs.append(pack_v8_fn(lo_slice))
        p_hi_packs.append(pack_v8_fn(hi_slice))
    return p_lo_packs, p_hi_packs


def _safe_l_inv(l_row, zero_f):
    l_inv = rocdl.rcp(T.f32, as_mlir_value(l_row))
    return (fx.Float32(l_row) > zero_f).select(l_inv, zero_f)


def _rescale_from_tile_max(m_row, m_tile_max, fm_fast):
    row_max = _fmax(m_row, m_tile_max, fm_fast)
    rescale = rocdl.exp2(T.f32, as_mlir_value(_fsub(m_row, row_max, fm_fast)))
    return row_max, rescale


def _scale_o_accs(v_o, scale_scalar, traits, fm_fast):
    scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
    for dc in range_constexpr(traits.D_CHUNKS):
        v_o[dc] = _fmul(Vec(v_o[dc]), scale_vec, fm_fast)


def _causal_pair_thresholds(kv_vectorized):
    if const_expr(kv_vectorized):
        return [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (16, 17),
            (18, 19),
            (20, 21),
            (22, 23),
        ]
    return [
        (0, 1),
        (2, 3),
        (8, 9),
        (10, 11),
        (16, 17),
        (18, 19),
        (24, 25),
        (26, 27),
    ]


def _apply_dualwave_causal_mask_pair(s_values, rel_i32, neg_inf_i32, pair_thresholds):
    for p in range_constexpr(len(pair_thresholds)):
        thr_x, thr_y = pair_thresholds[p]
        idx_x = p * 2
        idx_y = p * 2 + 1
        x_bits = _bitcast_i32(s_values[idx_x])
        y_bits = _bitcast_i32(s_values[idx_y])
        new_x, new_y = _attn_mask_vec2_imm(rel_i32, neg_inf_i32, thr_x, thr_y, x_bits, y_bits)
        s_values[idx_x] = _bitcast_f32(new_x)
        s_values[idx_y] = _bitcast_f32(new_y)


# Descriptor, LDS, and buffer helpers


def _llvm_value(value):
    """Unwrap FlyDSL values before passing them to LLVM dialect ops."""
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _extract_aligned_pointer(tensor, address_space=None) -> ir.Value:
    from flydsl._mlir.dialects import fly as _fly

    ptr_type = ir.Type.parse("!llvm.ptr" if address_space is None else f"!llvm.ptr<{address_space}>")
    return _fly.extract_aligned_pointer_as_index(ptr_type, _llvm_value(tensor))


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return llvm.LoadOp(result_type, _llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


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


def _make_ws_rsrc(ws_base_i64, byte_offset, nrec_bytes):
    addr_i64 = as_mlir_value(ws_base_i64 + fx.Int64(byte_offset))
    return buffer_ops.create_buffer_resource_from_addr(addr_i64, num_records_bytes=as_mlir_value(fx.Int64(nrec_bytes)))


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


def _make_raw_buffer_rsrc(tensor):
    base_ptr = _extract_aligned_pointer(tensor)
    base_i64 = llvm.PtrToIntOp(T.i64, base_ptr).result
    base_lo = arith.trunci(T.i32, fx.Int64(base_i64).ir_value())
    base_hi = arith.trunci(T.i32, fx.Int64(base_i64).shrui(fx.Int64(32)).ir_value())
    return Vec.from_elements(
        [
            base_lo,
            base_hi,
            buffer_ops._create_i32_constant(0xFFFFFFFF),
            buffer_ops._create_i32_constant(buffer_ops._get_buffer_flags()),
        ],
        fx.Int32,
    ).ir_value()


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


def _buffer_store_128(pack_i32_vec, elem_index, _o_store_reg_128, _store_atom_128, o_div):
    """128-bit register->global store (buffer_store_dwordx4) into O."""
    fx.memref_store_vec(pack_i32_vec, _o_store_reg_128)
    fx.copy(_store_atom_128, _o_store_reg_128, fx.slice(o_div, (None, fx.Int32(elem_index))))


# Mapping and address helpers


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


def _vec_v_elem(n, d, kv_head_idx, TRAITS_BLOCK_N, TRAITS_KV_VEC_SIZE, TRAITS_HEAD_DIM):
    return (
        kv_head_idx * (TRAITS_BLOCK_N // TRAITS_KV_VEC_SIZE) * TRAITS_HEAD_DIM * TRAITS_KV_VEC_SIZE
        + (n // TRAITS_KV_VEC_SIZE) * TRAITS_HEAD_DIM * TRAITS_KV_VEC_SIZE
        + d * TRAITS_KV_VEC_SIZE
        + (n % TRAITS_KV_VEC_SIZE)
    )


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


def _get_q_pack(traits, q_all_scaled_bf16, ks):
    q_vec = Vec(q_all_scaled_bf16)
    base = ks * traits.MFMA_LANE_K
    return q_vec.shuffle(q_vec, [base + i for i in range(traits.MFMA_LANE_K)]).ir_value()


def _vec_k_lds_idx_lo(traits, k_base, ks, lane_div_32, lane_mod_32):
    """LDS element index for vectorized K pack lo at k-step `ks` for this lane."""
    return k_base + (ks * 2 + lane_div_32) * (traits.BLOCK_N * traits.KV_VEC_SIZE) + lane_mod_32 * traits.KV_VEC_SIZE


def _swizzled_ks_offset(traits, ks):
    """Non-vectorized K LDS offset for k-step `ks` (outer/inner swizzle pattern)."""
    return (ks // 4) * traits.K_LDS_TO_REG_KSTEP_OUTER_STRIDE + (ks % 4) * traits.K_LDS_TO_REG_KSTEP_INNER_STRIDE


def _k_lds_read_base_per_lane(traits, lane_mod_32, lane_div_32):
    return (
        (lane_mod_32 % 8) * traits.SMEM_K_LINE_STRIDE
        + (lane_mod_32 // 8) * traits.D_128B_SIZE
        + lane_div_32 * traits.VEC_KV
    )


def _v_lds_read_base_per_lane(traits, lane, lane_div_32):
    return (
        lane_div_32 * traits.V_LDS_TO_REG_HALF_WAVE_STRIDE
        + ((lane % 16) // 4) * traits.V_LDS_TO_REG_LANE_QUAD_STRIDE
        + ((lane // 16) % 2) * traits.V_LDS_TO_REG_N_GROUP_STRIDE
        + (lane % 4) * traits.V_LDS_TO_REG_LANE_IN_QUAD_STRIDE
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
    return (dc // 2) * traits.V_LDS_TO_REG_DCHUNK_PAIR_STRIDE + (dc % 2) * traits.V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE


def _swizzled_v_imm_lo(traits, dc, k_substep):
    """Non-vectorized V LDS byte offset for (dc, k_substep) in bf16 elements."""
    return (k_substep * traits.V_LDS_TO_REG_K_SUBSTEP_STRIDE + _swizzled_v_dc_off(traits, dc)) * traits.BF16_BYTES


def _ds_read_tr_v4f16_imm(lds_base_elem_idx, imm_bytes, lds_kv_base_idx, v_lds_read_vec4_type):
    byte_offset = lds_base_elem_idx * 2 + lds_kv_base_idx
    addr_i32 = fx.Int32(byte_offset)
    return _ds_read_tr16_b64_imm(v_lds_read_vec4_type, addr_i32, imm_bytes)


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


def _init_dualwave_thread_mapping(ctx):
    """Set block/wave/lane/head indices on a dualwave-style context.

    Shared verbatim by DualwaveKernelContext and DualwaveFp8KernelContext."""
    traits = ctx.traits
    ctx.h_idx = fx.Index(gpu.block_idx.x)
    ctx.q_block_idx = fx.Index(gpu.block_idx.y)
    if const_expr(traits.SPLITK):
        ctx.bz_idx = fx.Index(gpu.block_idx.z)
        ctx.batch_idx = ctx.bz_idx // traits.NUM_KV_SPLITS
        ctx.split_idx = ctx.bz_idx % traits.NUM_KV_SPLITS
    else:
        ctx.batch_idx = fx.Index(gpu.block_idx.z)
        ctx.split_idx = None
    ctx.tid = fx.Index(gpu.thread_idx.x)

    ctx.wave_id = ctx.tid // traits.WARP_SIZE
    ctx.lane = ctx.tid % traits.WARP_SIZE
    ctx.lane_mod_32 = ctx.lane % 32
    ctx.lane_div_32 = ctx.lane // 32

    _tid_i32 = as_mlir_value(fx.Int32(ctx.tid))
    _wave_id_uni_i32 = rocdl.readfirstlane(
        T.i32,
        arith.divsi(_tid_i32, as_mlir_value(fx.Int32(traits.WARP_SIZE))),
    )
    ctx.stagger_i32 = arith.divsi(_wave_id_uni_i32, as_mlir_value(fx.Int32(4)))
    ctx.wave_id_uni = fx.Index(_wave_id_uni_i32)

    ctx.wave_q_offset = ctx.wave_id * traits.ROWS_PER_WAVE
    ctx.q_start = ctx.q_block_idx * traits.BLOCK_M

    ctx.h_kv_idx = ctx.h_idx % traits.NUM_HEADS_KV
    ctx.group_id = ctx.h_idx // traits.NUM_HEADS_KV
    ctx.q_head_idx = ctx.h_kv_idx * traits.GQA_GROUP_SIZE + ctx.group_id
    ctx.kv_head_idx = ctx.h_kv_idx


def _init_dualwave_q_row(ctx):
    """Set q_row / q_row_i32 / q_start_pos_i32 on a dualwave-style context."""
    traits = ctx.traits
    ctx.q_row_in_block = ctx.wave_q_offset + ctx.lane_mod_32
    ctx.q_start_pos_i32 = fx.Int32(ctx.q_start + ctx.wave_id_uni * traits.ROWS_PER_WAVE)
    ctx.q_row = ctx.q_start + ctx.q_row_in_block
    ctx.q_row_i32 = fx.Int32(ctx.q_row)


# Traits and factory


@dataclass(frozen=True)
class FlashAttnGenericTraits:
    """Compile-time constants for the generic flash-attention kernel."""

    NUM_HEADS_Q: int
    NUM_HEADS_KV: int
    GQA_GROUP_SIZE: int
    HEAD_DIM: int
    DTYPE_STR: str
    CAUSAL: bool
    VARLEN: bool
    CROSS_SEQLEN: bool
    PAGED: bool
    KV_CACHE_LAYOUT: str
    KV_VECTORIZED: bool
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_N_OUT: int
    K_SUB_N: int
    WARP_SIZE: int
    NUM_WAVES: int
    BLOCK_SIZE: int
    ROWS_PER_WAVE: int
    PATH_TAG: str
    N_SUBTILES: int
    ENABLE_PREFETCH_3BUF: bool
    ENABLE_DMA: bool
    ENABLE_LDS_VEC16: bool
    REDUCE_MODE: str
    NUM_PREFETCH_K: int
    NUM_PREFETCH_V: int
    CK_LDS_SEQ: tuple[int, ...]
    USE_HW_TR: bool
    USE_K16: bool
    USE_PERMLANE_OSTORE: bool
    K_STEP_QK: int
    K_STEPS_QK: int
    D_CHUNK: int
    D_CHUNKS: int
    PV_K_STEP: int
    PV_K_STEPS: int
    MFMA_LANE_K: int
    KV_VEC_SIZE: int
    PAGE_SIZE: int
    PAGE_STRIDE_VEC: int
    HEAD_STRIDE_VEC: int
    STRIDE_TOKEN_Q: int
    STRIDE_TOKEN_KV: int
    K_STRIDE: int
    K_SWZ_ROWMASK: int
    VT_STRIDE: int
    V_STRIDE: int
    VEC_V_LINE: int
    VEC_V_D128: int
    VEC_V_NGROUPS: int
    TOTAL_V8: int
    NV8_PER_THREAD: int
    V_NOMAJOR_DMA: bool
    VEC_WIDTH: int
    THREADS_PER_ROW_LOAD: int
    ROWS_PER_BATCH_LOAD: int
    NUM_BATCHES_KV: int
    KV_NEEDS_GUARD: bool
    LDS_K_TILE_SIZE: int
    LDS_V_TILE_SIZE: int
    LDS_K_TOTAL_SIZE: int
    LDS_V_BASE: int
    LDS_V_TOTAL_SIZE: int
    LDS_KV_TOTAL_SIZE: int
    WAVES_PER_EU: int
    DAZ: bool
    FAST_FP_MATH: bool
    UNSAFE_FP_MATH: bool
    SM_SCALE: float | None
    SKIP_KV_PAD_MASK: bool
    ENABLE_GFX942_DMA: bool
    ENABLE_GFX942_KV_GPFETCH: bool
    ENABLE_GFX942_VEC_K: bool
    V_PERM_TR: bool
    K_VEC_SIZE: int
    K_VEC_N_STRIDE: int
    K_VEC_HI_N_OFFSET: int
    QK_PREFETCH_DEPTH: int

    @property
    def cache_tag(self):
        return (
            self.NUM_HEADS_Q,
            self.NUM_HEADS_KV,
            self.GQA_GROUP_SIZE,
            self.HEAD_DIM,
            self.DTYPE_STR,
            self.CAUSAL,
            self.VARLEN,
            self.CROSS_SEQLEN,
            self.PAGED,
            self.KV_CACHE_LAYOUT,
            self.KV_VECTORIZED,
            False,  # SPLITK is not supported by the generic builder.
            1,
            self.BLOCK_M,
            self.BLOCK_N,
            self.BLOCK_N_OUT,
            self.K_SUB_N,
            self.WARP_SIZE,
            self.NUM_WAVES,
            self.BLOCK_SIZE,
            self.ROWS_PER_WAVE,
            self.PATH_TAG,
            self.N_SUBTILES,
            self.ENABLE_PREFETCH_3BUF,
            self.ENABLE_DMA,
            self.ENABLE_LDS_VEC16,
            self.REDUCE_MODE,
            self.NUM_PREFETCH_K,
            self.NUM_PREFETCH_V,
            self.CK_LDS_SEQ,
            self.USE_HW_TR,
            self.USE_K16,
            self.USE_PERMLANE_OSTORE,
            self.K_STEP_QK,
            self.K_STEPS_QK,
            self.D_CHUNK,
            self.D_CHUNKS,
            self.PV_K_STEP,
            self.PV_K_STEPS,
            self.MFMA_LANE_K,
            self.KV_VEC_SIZE,
            self.PAGE_SIZE,
            self.PAGE_STRIDE_VEC,
            self.HEAD_STRIDE_VEC,
            self.STRIDE_TOKEN_Q,
            self.STRIDE_TOKEN_KV,
            self.K_STRIDE,
            self.K_SWZ_ROWMASK,
            self.VT_STRIDE,
            self.V_STRIDE,
            self.VEC_V_LINE,
            self.VEC_V_D128,
            self.VEC_V_NGROUPS,
            self.TOTAL_V8,
            self.NV8_PER_THREAD,
            self.V_NOMAJOR_DMA,
            self.VEC_WIDTH,
            self.THREADS_PER_ROW_LOAD,
            self.ROWS_PER_BATCH_LOAD,
            self.NUM_BATCHES_KV,
            self.KV_NEEDS_GUARD,
            self.LDS_K_TILE_SIZE,
            self.LDS_V_TILE_SIZE,
            self.LDS_K_TOTAL_SIZE,
            self.LDS_V_BASE,
            self.LDS_V_TOTAL_SIZE,
            self.LDS_KV_TOTAL_SIZE,
            self.WAVES_PER_EU,
            self.DAZ,
            self.FAST_FP_MATH,
            self.UNSAFE_FP_MATH,
            self.SM_SCALE,
            self.SKIP_KV_PAD_MASK,
            self.ENABLE_GFX942_DMA,
            self.ENABLE_GFX942_KV_GPFETCH,
            self.ENABLE_GFX942_VEC_K,
            self.V_PERM_TR,
            self.K_VEC_SIZE,
            self.K_VEC_N_STRIDE,
            self.K_VEC_HI_N_OFFSET,
            self.QK_PREFETCH_DEPTH,
        )


def _make_flash_attn_generic_traits(
    num_heads,
    num_kv_heads,
    head_dim,
    gpu_arch,
    causal=True,
    dtype_str="f16",
    flat_work_group_size=None,
    block_m=None,
    path_tag="auto",
    varlen=False,
    cross_seqlen=False,
    paged=False,
    kv_cache_layout="linear",
    waves_per_eu=2,
    daz=True,
    fast_fp_math=True,
    unsafe_fp_math=True,
    sm_scale=None,
    skip_kv_pad_mask=None,
):
    """Build compile-time traits for ``flash_attn_generic``."""
    block_n = 64
    k_sub_n = 32
    warp_size = 64

    block_m = 128 if block_m is None else block_m
    if flat_work_group_size is None:
        flat_work_group_size = 256 if block_m <= 128 else 512
    num_waves = flat_work_group_size // warp_size
    block_size = flat_work_group_size
    rows_per_wave = block_m // num_waves

    if path_tag.upper() in ("N32", "N128"):
        path = path_tag.upper()
    elif dtype_str in ("f16", "bf16") and causal and head_dim == 128:
        path = "N128"
    else:
        path = "N32"
    block_n_out = 128 if path == "N128" else block_n
    n_subtiles = block_n_out // block_n

    enable_prefetch_3buf = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_PREFETCH3", "0") == "1"
    has_lds_load_b128 = not gpu_arch.startswith("gfx942")
    enable_gfx942_dma = gpu_arch.startswith("gfx942") and (
        os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_GFX942_DMA", "0") == "1"
    )
    enable_dma = enable_gfx942_dma or (
        has_lds_load_b128 and (path == "N128" or (os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_DMA", "0") == "1"))
    )
    enable_gfx942_kv_gpfetch = (
        gpu_arch.startswith("gfx942")
        and not enable_prefetch_3buf
        and (not enable_dma or enable_gfx942_dma)
        and os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_KV_GPFETCH", "1") == "1"
    )
    enable_gfx942_vec_k = (
        gpu_arch.startswith("gfx942")
        and enable_dma
        and os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_GFX942_VEC_K", "1") == "1"
    )
    enable_lds_vec16 = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    reduce_mode = os.getenv("FLYDSL_FLASH_ATTN_FUNC_REDUCE_MODE", "xor").strip().lower()
    if reduce_mode not in ("xor", "ds_bpermute"):
        reduce_mode = "xor"
    if skip_kv_pad_mask is None:
        skip_kv_pad_mask = os.getenv("FLYDSL_FLASH_ATTN_FUNC_SKIP_KV_PAD_MASK", "0") == "1"
    else:
        skip_kv_pad_mask = bool(skip_kv_pad_mask)
    num_prefetch_k = 3 if enable_prefetch_3buf else (2 if (enable_dma and not enable_gfx942_dma) else 1)
    num_prefetch_v = 3 if enable_prefetch_3buf else 1
    ck_lds_seq = (1, 2, 0, 1, 0, 1, 2, 0) if enable_prefetch_3buf else (0,)

    use_hw_tr = gpu_arch.startswith("gfx950")
    use_k16 = gpu_arch.startswith("gfx950")
    use_permlane_ostore = gpu_arch.startswith("gfx950")
    k_step_qk = 16 if use_k16 else 8
    k_steps_qk = head_dim // k_step_qk
    d_chunk = 32
    d_chunks = head_dim // d_chunk
    pv_k_step = 16 if use_k16 else 8
    pv_k_steps = k_sub_n // pv_k_step
    mfma_lane_k = 8 if use_k16 else 4

    paged = bool(paged)
    kv_vectorized = paged and (kv_cache_layout == "vectorized")
    kv_vec_size = 8
    page_size = block_n
    page_stride_vec = num_kv_heads * head_dim * page_size
    head_stride_vec = head_dim * page_size

    stride_token_q = num_heads * head_dim
    stride_token_kv = num_kv_heads * head_dim
    _k_pad_default = 4 if (gpu_arch.startswith("gfx942") and not enable_dma) else 0
    _k_pad = int(os.getenv("FLYDSL_FLASH_ATTN_FUNC_K_PAD", str(_k_pad_default)))
    k_stride = head_dim + _k_pad
    k_swz_rowmask = head_dim // 16 - 1
    if use_hw_tr:
        vt_stride = 0
        v_stride = head_dim if enable_dma else head_dim + 4
    else:
        vt_stride = block_n + 2
        v_stride = vt_stride

    vec_v_line = 544
    vec_v_d128 = 64
    vec_v_ngroups = block_n // 8
    if kv_vectorized:
        total_v8 = vec_v_ngroups * head_dim
        assert total_v8 % block_size == 0
        nv8_per_thread = total_v8 // block_size
        assert not enable_prefetch_3buf, "KV_VECTORIZED no-major V unsupported with 3-buffer prefetch"
        v_nomajor_dma = os.getenv("FLYDSL_FLASH_ATTN_FUNC_VEC_V_DMA", "1") == "1"
    else:
        total_v8 = 0
        nv8_per_thread = 0
        v_nomajor_dma = False

    vec_width = 16 if enable_lds_vec16 else 8
    assert head_dim % vec_width == 0
    threads_per_row_load = head_dim // vec_width
    assert block_size % threads_per_row_load == 0
    rows_per_batch_load = block_size // threads_per_row_load
    if rows_per_batch_load >= block_n:
        num_batches_kv = 1
        kv_needs_guard = rows_per_batch_load > block_n
    else:
        assert block_n % rows_per_batch_load == 0
        num_batches_kv = block_n // rows_per_batch_load
        kv_needs_guard = False

    v_perm_tr_env = os.getenv("FLYDSL_FLASH_ATTN_FUNC_V_PERM_TR", "1") == "1"
    v_perm_tr = (
        v_perm_tr_env
        and (not use_hw_tr)
        and (not enable_dma)
        and (not enable_prefetch_3buf)
        and dtype_str == "bf16"
        and vec_width == 16
        and num_batches_kv == 1
        and (not kv_needs_guard)
        and (vt_stride == block_n + 2)
        and (block_n % 2 == 0)
    )
    k_vec_size = k_step_qk
    k_vec_n_stride = block_n * k_vec_size
    k_vec_hi_n_offset = k_sub_n * k_vec_size
    if enable_gfx942_kv_gpfetch and not enable_gfx942_vec_k:
        qk_prefetch_depth = int(os.getenv("FLYDSL_FLASH_ATTN_FUNC_QK_PF_DEPTH", "3"))
    else:
        qk_prefetch_depth = 4 if (enable_gfx942_kv_gpfetch or enable_gfx942_vec_k) else (3 if enable_dma else 2)

    lds_k_tile_size = block_n * k_stride
    if kv_vectorized:
        lds_v_tile_size = (head_dim // 8) * vec_v_line
    elif use_hw_tr:
        lds_v_tile_size = block_n * v_stride
    else:
        lds_v_tile_size = head_dim * vt_stride
    lds_k_total_size = num_prefetch_k * lds_k_tile_size
    lds_v_base = lds_k_total_size
    lds_v_total_size = num_prefetch_v * lds_v_tile_size
    lds_kv_total_size = lds_k_total_size + lds_v_total_size

    return FlashAttnGenericTraits(
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=num_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        DTYPE_STR=dtype_str,
        CAUSAL=bool(causal),
        VARLEN=bool(varlen),
        CROSS_SEQLEN=bool(cross_seqlen),
        PAGED=paged,
        KV_CACHE_LAYOUT=kv_cache_layout,
        KV_VECTORIZED=kv_vectorized,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_N_OUT=block_n_out,
        K_SUB_N=k_sub_n,
        WARP_SIZE=warp_size,
        NUM_WAVES=num_waves,
        BLOCK_SIZE=block_size,
        ROWS_PER_WAVE=rows_per_wave,
        PATH_TAG=path,
        N_SUBTILES=n_subtiles,
        ENABLE_PREFETCH_3BUF=enable_prefetch_3buf,
        ENABLE_DMA=enable_dma,
        ENABLE_LDS_VEC16=enable_lds_vec16,
        REDUCE_MODE=reduce_mode,
        NUM_PREFETCH_K=num_prefetch_k,
        NUM_PREFETCH_V=num_prefetch_v,
        CK_LDS_SEQ=ck_lds_seq,
        USE_HW_TR=use_hw_tr,
        USE_K16=use_k16,
        USE_PERMLANE_OSTORE=use_permlane_ostore,
        K_STEP_QK=k_step_qk,
        K_STEPS_QK=k_steps_qk,
        D_CHUNK=d_chunk,
        D_CHUNKS=d_chunks,
        PV_K_STEP=pv_k_step,
        PV_K_STEPS=pv_k_steps,
        MFMA_LANE_K=mfma_lane_k,
        KV_VEC_SIZE=kv_vec_size,
        PAGE_SIZE=page_size,
        PAGE_STRIDE_VEC=page_stride_vec,
        HEAD_STRIDE_VEC=head_stride_vec,
        STRIDE_TOKEN_Q=stride_token_q,
        STRIDE_TOKEN_KV=stride_token_kv,
        K_STRIDE=k_stride,
        K_SWZ_ROWMASK=k_swz_rowmask,
        VT_STRIDE=vt_stride,
        V_STRIDE=v_stride,
        VEC_V_LINE=vec_v_line,
        VEC_V_D128=vec_v_d128,
        VEC_V_NGROUPS=vec_v_ngroups,
        TOTAL_V8=total_v8,
        NV8_PER_THREAD=nv8_per_thread,
        V_NOMAJOR_DMA=v_nomajor_dma,
        VEC_WIDTH=vec_width,
        THREADS_PER_ROW_LOAD=threads_per_row_load,
        ROWS_PER_BATCH_LOAD=rows_per_batch_load,
        NUM_BATCHES_KV=num_batches_kv,
        KV_NEEDS_GUARD=kv_needs_guard,
        LDS_K_TILE_SIZE=lds_k_tile_size,
        LDS_V_TILE_SIZE=lds_v_tile_size,
        LDS_K_TOTAL_SIZE=lds_k_total_size,
        LDS_V_BASE=lds_v_base,
        LDS_V_TOTAL_SIZE=lds_v_total_size,
        LDS_KV_TOTAL_SIZE=lds_kv_total_size,
        WAVES_PER_EU=waves_per_eu,
        DAZ=bool(daz),
        FAST_FP_MATH=bool(fast_fp_math),
        UNSAFE_FP_MATH=bool(unsafe_fp_math),
        SM_SCALE=sm_scale,
        SKIP_KV_PAD_MASK=skip_kv_pad_mask,
        ENABLE_GFX942_DMA=enable_gfx942_dma,
        ENABLE_GFX942_KV_GPFETCH=enable_gfx942_kv_gpfetch,
        ENABLE_GFX942_VEC_K=enable_gfx942_vec_k,
        V_PERM_TR=v_perm_tr,
        K_VEC_SIZE=k_vec_size,
        K_VEC_N_STRIDE=k_vec_n_stride,
        K_VEC_HI_N_OFFSET=k_vec_hi_n_offset,
        QK_PREFETCH_DEPTH=qk_prefetch_depth,
    )


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
    K_LDS_TO_REG_N_STRIP_STRIDE: int
    K_LDS_TO_REG_KSTEP_INNER_STRIDE: int
    K_LDS_TO_REG_KSTEP_OUTER_STRIDE: int
    V_LDS_TO_REG_HALF_WAVE_STRIDE: int
    V_LDS_TO_REG_LANE_QUAD_STRIDE: int
    V_LDS_TO_REG_N_GROUP_STRIDE: int
    V_LDS_TO_REG_LANE_IN_QUAD_STRIDE: int
    V_LDS_TO_REG_K_SUBSTEP_STRIDE: int
    V_LDS_TO_REG_DCHUNK_PAIR_STRIDE: int
    V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE: int
    V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE: int
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

    @property
    def cache_tag(self):
        return (
            self.NUM_HEADS_Q,
            self.NUM_HEADS_KV,
            self.HEAD_DIM,
            self.CAUSAL,
            self.DTYPE_STR,
            self.WAVES_PER_EU,
            self.DAZ,
            self.DUALWAVE_SWP_LAZY_RESCALE,
            self.DUALWAVE_SWP_SETPRIO,
            self.DUALWAVE_SWP_DEBUG_LAZY_COUNTS,
            self.DUALWAVE_SWP_ENABLE_STAGGER,
            self.NUM_KV_SPLITS,
            self.SPLITK,
            self.PAGED,
            self.VARLEN,
            self.CROSS_SEQLEN,
            self.KV_CACHE_LAYOUT,
            self.KV_VECTORIZED,
        )


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
    # Tile shape and wave geometry follow the gfx950 dual-wave 8-wave CTA.
    block_m = 256
    block_n = 64
    block_n_out = 64
    k_sub_n = 32
    warp_size = 64
    num_waves = 8
    block_size = num_waves * warp_size
    rows_per_wave = 32

    # QK walks D in 16-wide MFMA K steps; PV consumes K_SUB_N in two 16-token steps.
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

    # Global K/V DMA is 16B per lane; D_128B_SIZE is one 128B row in bf16 elements.
    dma_bytes = 16
    bf16_bytes = 2
    d_128b_size = 64
    vec_kv = 8
    smem_linear_wave = warp_size * 16 // bf16_bytes
    smem_n_per_wave = smem_linear_wave // d_128b_size
    smem_n_rpt = block_n // smem_n_per_wave
    smem_d_rpt = head_dim // d_128b_size
    # K/V LDS tiles are wave-linear rows with padding to avoid bank-aligned repeats.
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

    # K LDS->VGPR reads: hi half jumps one N strip; ks uses inner stride then d_rpt stride.
    k_lds_to_reg_n_strip_stride = 256
    k_lds_to_reg_kstep_inner_stride = 16
    k_lds_to_reg_kstep_outer_stride = smem_n_rpt * smem_k_line_stride
    # V LDS->VGPR base is decomposed by half-wave, lane quad, N group, and lane-in-quad.
    v_lds_to_reg_half_wave_stride = 2176
    v_lds_to_reg_lane_quad_stride = smem_v_line_stride
    v_lds_to_reg_n_group_stride = 16
    v_lds_to_reg_lane_in_quad_stride = 4
    # V read immediates step across K substeps, D-chunk pairs, and transpose-load pairs.
    v_lds_to_reg_k_substep_stride = 128
    v_lds_to_reg_dchunk_pair_stride = smem_n_rpt * smem_v_line_stride
    v_lds_to_reg_dchunk_in_pair_stride = 32
    v_lds_to_reg_transpose_pair_stride = d_128b_size
    # Vectorized KV path keeps one 16B vector per lane and reuses the V LDS row stride.
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
        K_LDS_TO_REG_N_STRIP_STRIDE=k_lds_to_reg_n_strip_stride,
        K_LDS_TO_REG_KSTEP_INNER_STRIDE=k_lds_to_reg_kstep_inner_stride,
        K_LDS_TO_REG_KSTEP_OUTER_STRIDE=k_lds_to_reg_kstep_outer_stride,
        V_LDS_TO_REG_HALF_WAVE_STRIDE=v_lds_to_reg_half_wave_stride,
        V_LDS_TO_REG_LANE_QUAD_STRIDE=v_lds_to_reg_lane_quad_stride,
        V_LDS_TO_REG_N_GROUP_STRIDE=v_lds_to_reg_n_group_stride,
        V_LDS_TO_REG_LANE_IN_QUAD_STRIDE=v_lds_to_reg_lane_in_quad_stride,
        V_LDS_TO_REG_K_SUBSTEP_STRIDE=v_lds_to_reg_k_substep_stride,
        V_LDS_TO_REG_DCHUNK_PAIR_STRIDE=v_lds_to_reg_dchunk_pair_stride,
        V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE=v_lds_to_reg_dchunk_in_pair_stride,
        V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE=v_lds_to_reg_transpose_pair_stride,
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


@dataclass(frozen=True)
class DualwaveSwpFp8Traits:
    """Pure compile-time tile/layout constants for the gfx950 DUALWAVE_SWP fp8 kernel.

    fp8 runs a single path: WIDE QK (32x32x64 mfma_scale) feeding HIPREC PV (fp8 V
    dequantized into a bf16 ``vt`` LDS scratch, then a bf16 PV MMA). The ``*_BF``
    fields describe that bf16 vt layout; ``ELEM_BYTES`` is 1 (Q/K/V are fp8)."""

    BLOCK_M: int
    BLOCK_N: int
    K_SUB_N: int
    WARP_SIZE: int
    NUM_WAVES: int
    BLOCK_SIZE: int
    ROWS_PER_WAVE: int
    HEAD_DIM: int
    D_CHUNK: int
    D_CHUNKS: int
    PV_K_STEPS: int
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
    VARLEN: bool
    CROSS_SEQLEN: bool
    DEFAULT_STRIDE_Q_N: int
    DEFAULT_STRIDE_KV_N: int
    DMA_BYTES: int
    ELEM_BYTES: int
    OUT_ELEM_BYTES: int
    D_128B_SIZE: int
    VEC_KV: int
    LANE_SPLIT_KV: int
    SMEM_N_RPT: int
    SMEM_D_RPT: int
    SMEM_K_LINE_STRIDE: int
    SMEM_K_TILE_ELEMS: int
    NUM_PREFETCH_K: int
    DUALWAVE_SWP_KV_PER_BUFFER: int
    LDS_KV_TOTAL_SIZE: int
    DUALWAVE_SWP_K_BUF_BASE: tuple[int, int]
    DUALWAVE_SWP_V_BUF_BASE: tuple[int, int]
    # bf16 vt scratch layout (HIPREC V dequant target + transpose read strides).
    EB_BF: int
    D128_BF: int
    VEC_BF: int
    SDRPT_BF: int
    SNRPT_BF: int
    VLS_BF: int
    VT_BF16_ELEMS: int
    VT_BF16_TOTAL: int
    URV_GRPK_BF: int
    URV_GRP_N_BF: int
    URV_LANE_LO_BF: int
    URV_LANE_HI_BF: int
    URV_STEPK_BF: int
    URV_DC_AXIS0_BF: int
    URV_DC_AXIS1_BF: int
    URV_I5_BF: int
    DUALWAVE_SWP_RESCALE_THRESHOLD: float
    SCHED_MFMA_MASK: int
    SCHED_VALU_MASK: int
    SCHED_EXP_MASK: int
    LDS_SCOPE_NAMES: tuple[str, str, str, str]
    NEG_INF_F32_BITS: int
    LGKMCNT_0_ONLY: int

    @property
    def cache_tag(self):
        return (
            self.NUM_HEADS_Q,
            self.NUM_HEADS_KV,
            self.HEAD_DIM,
            self.CAUSAL,
            self.DTYPE_STR,
            self.WAVES_PER_EU,
            self.DAZ,
            self.DUALWAVE_SWP_LAZY_RESCALE,
            self.DUALWAVE_SWP_SETPRIO,
            self.DUALWAVE_SWP_DEBUG_LAZY_COUNTS,
            self.DUALWAVE_SWP_ENABLE_STAGGER,
            self.NUM_KV_SPLITS,
            self.SPLITK,
            self.VARLEN,
            self.CROSS_SEQLEN,
            "fp8_wide_qk_hiprec_pv",
            self.ELEM_BYTES,
            self.OUT_ELEM_BYTES,
            self.LANE_SPLIT_KV,
            self.VT_BF16_ELEMS,
            self.VT_BF16_TOTAL,
        )


def _make_dualwave_swp_fp8_traits(
    num_heads,
    num_kv_heads,
    head_dim,
    causal=True,
    waves_per_eu=2,
    daz=True,
    dualwave_swp_lazy_rescale=True,
    dualwave_swp_setprio=True,
    dualwave_swp_debug_lazy_counts=False,
    dualwave_swp_enable_stagger=True,
    num_kv_splits=1,
    varlen=False,
    cross_seqlen=False,
):
    """Build gfx950 DUALWAVE_SWP fp8 compile-time layout traits (dtype fixed to fp8)."""
    # Tile shape and wave geometry follow the gfx950 dual-wave 8-wave CTA.
    block_m = 256
    block_n = 64
    k_sub_n = 32
    warp_size = 64
    num_waves = 8
    block_size = num_waves * warp_size
    rows_per_wave = 32

    d_chunk = 32
    d_chunks = head_dim // d_chunk
    pv_k_step = 16
    pv_k_steps = k_sub_n // pv_k_step

    gqa_group_size = num_heads // num_kv_heads
    default_stride_q_n = num_heads * head_dim
    default_stride_kv_n = num_kv_heads * head_dim

    # fp8: Q/K/V are 1B; O is bf16 (2B). ELEM_BYTES=1 drives the fp8 address math.
    elem_bytes = 1
    out_elem_bytes = 2
    d_128b_size = 128 // elem_bytes
    vec_kv = 16 // elem_bytes
    lane_split_kv = 8
    smem_linear_wave = warp_size * 16 // elem_bytes
    smem_n_per_wave = smem_linear_wave // d_128b_size
    smem_n_rpt = block_n // smem_n_per_wave
    smem_d_rpt = head_dim // d_128b_size
    smem_k_pad = 16 // elem_bytes
    smem_v_pad = 64 // elem_bytes
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

    # bf16 vt scratch layout: HIPREC dequantizes fp8 V into these positions so the
    # proven bf16 V transpose read (ds_read_tr16) + bf16 PV MMA are reused unchanged.
    eb_bf = 2
    d128_bf = 128 // eb_bf
    vec_bf = 16 // eb_bf
    slw_bf = warp_size * 16 // eb_bf
    snrpt_bf = block_n // (slw_bf // d128_bf)
    sdrpt_bf = head_dim // d128_bf
    vls_bf = slw_bf + 64 // eb_bf
    vt_bf16_elems = snrpt_bf * sdrpt_bf * vls_bf
    vt_bf16_total = num_prefetch_k * vt_bf16_elems

    splitk = num_kv_splits > 1

    return DualwaveSwpFp8Traits(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        K_SUB_N=k_sub_n,
        WARP_SIZE=warp_size,
        NUM_WAVES=num_waves,
        BLOCK_SIZE=block_size,
        ROWS_PER_WAVE=rows_per_wave,
        HEAD_DIM=head_dim,
        D_CHUNK=d_chunk,
        D_CHUNKS=d_chunks,
        PV_K_STEPS=pv_k_steps,
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        CAUSAL=causal,
        DTYPE_STR="fp8",
        WAVES_PER_EU=waves_per_eu,
        DAZ=bool(daz),
        DUALWAVE_SWP_LAZY_RESCALE=bool(dualwave_swp_lazy_rescale),
        DUALWAVE_SWP_SETPRIO=bool(dualwave_swp_setprio),
        DUALWAVE_SWP_DEBUG_LAZY_COUNTS=bool(dualwave_swp_debug_lazy_counts),
        DUALWAVE_SWP_ENABLE_STAGGER=bool(dualwave_swp_enable_stagger),
        NUM_KV_SPLITS=num_kv_splits,
        SPLITK=splitk,
        VARLEN=bool(varlen),
        CROSS_SEQLEN=bool(cross_seqlen),
        DEFAULT_STRIDE_Q_N=default_stride_q_n,
        DEFAULT_STRIDE_KV_N=default_stride_kv_n,
        DMA_BYTES=16,
        ELEM_BYTES=elem_bytes,
        OUT_ELEM_BYTES=out_elem_bytes,
        D_128B_SIZE=d_128b_size,
        VEC_KV=vec_kv,
        LANE_SPLIT_KV=lane_split_kv,
        SMEM_N_RPT=smem_n_rpt,
        SMEM_D_RPT=smem_d_rpt,
        SMEM_K_LINE_STRIDE=smem_k_line_stride,
        SMEM_K_TILE_ELEMS=smem_k_tile_elems,
        NUM_PREFETCH_K=num_prefetch_k,
        DUALWAVE_SWP_KV_PER_BUFFER=dualwave_swp_kv_per_buffer,
        LDS_KV_TOTAL_SIZE=lds_kv_total_size,
        DUALWAVE_SWP_K_BUF_BASE=dualwave_swp_k_buf_base,
        DUALWAVE_SWP_V_BUF_BASE=dualwave_swp_v_buf_base,
        EB_BF=eb_bf,
        D128_BF=d128_bf,
        VEC_BF=vec_bf,
        SDRPT_BF=sdrpt_bf,
        SNRPT_BF=snrpt_bf,
        VLS_BF=vls_bf,
        VT_BF16_ELEMS=vt_bf16_elems,
        VT_BF16_TOTAL=vt_bf16_total,
        URV_GRPK_BF=4 * vls_bf,
        URV_GRP_N_BF=16,
        URV_LANE_LO_BF=4,
        URV_LANE_HI_BF=vls_bf,
        URV_STEPK_BF=128,
        URV_DC_AXIS0_BF=snrpt_bf * vls_bf,
        URV_DC_AXIS1_BF=32,
        URV_I5_BF=d128_bf,
        DUALWAVE_SWP_RESCALE_THRESHOLD=8.0,
        SCHED_MFMA_MASK=0x008,
        SCHED_VALU_MASK=0x002,
        SCHED_EXP_MASK=0x400,
        LDS_SCOPE_NAMES=("lds_k0", "lds_k1", "lds_v0", "lds_v1"),
        NEG_INF_F32_BITS=0xFF800000,
        LGKMCNT_0_ONLY=0xC07F,
    )


# Kernel context


class GenericFlashAttnContext:
    """Runtime setup state for the generic flash-attention kernel."""

    def __init__(self, traits, K, V, seq_len, seq_len_kv, allocator, lds_kv_offset):
        self.traits = traits
        self.K = K
        self.V = V
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.allocator = allocator
        self.lds_kv_offset = lds_kv_offset

    def init_types_and_pointers(self):
        traits = self.traits
        self.elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)
        self.elem_type = self.elem_dtype.ir_type
        self.compute_type = fx.Float32.ir_type
        self.k_ptr = _extract_aligned_pointer(self.K)
        self.v_ptr = _extract_aligned_pointer(self.V)
        self.load_atom_32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        self.v1i32_type = Vec.make_type(1, fx.Int32)

        self.fm_fast = fx.arith.FastMathFlags.fast
        self.v4f16_type = Vec.make_type(4, self.elem_dtype)
        self.v8f16_type = Vec.make_type(8, self.elem_dtype)
        self.v16f32_type = Vec.make_type(16, fx.Float32)
        self.mfma_pack_type = self.v8f16_type if traits.USE_K16 else self.v4f16_type
        self.MFMA_LANE_K = traits.MFMA_LANE_K
        self.mma_atom_k16 = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, self.elem_dtype))

    def init_sequence_indices(self):
        self.seq_len_v = fx.Index(self.seq_len)
        self.seq_len_kv_v = fx.Index(self.seq_len_kv)

    def init_lds_view(self):
        self.base_ptr = self.allocator.get_base()
        self.lds_kv = SmemPtr(
            self.base_ptr,
            self.lds_kv_offset,
            self.elem_type,
            shape=(self.traits.LDS_KV_TOTAL_SIZE,),
        ).get()

    def init_thread_mapping(self):
        traits = self.traits
        self.block_id = fx.Index(gpu.block_idx.x)
        self.tid = fx.Index(gpu.thread_idx.x)

        self.wave_id = self.tid // traits.WARP_SIZE
        self.lane = self.tid % traits.WARP_SIZE
        self.lane_mod_32 = self.lane % 32
        self.lane_div_32 = self.lane // 32

        self.tr_k_group = (self.lane % 16) // 4
        self.tr_col_sub = self.lane % 4
        self.tr_col_half = (self.lane % 32) // 16
        self.wave_q_offset = self.wave_id * traits.ROWS_PER_WAVE

    def init_block_mapping(self):
        traits = self.traits
        self.q_head_idx = self.block_id % traits.NUM_HEADS_Q
        self.batch_q_tile_id = self.block_id // traits.NUM_HEADS_Q
        self.num_q_tiles = (self.seq_len_v + traits.BLOCK_M - 1) // traits.BLOCK_M
        self.q_tile_idx = self.batch_q_tile_id % self.num_q_tiles
        self.batch_idx = self.batch_q_tile_id // self.num_q_tiles
        self.q_start = self.q_tile_idx * traits.BLOCK_M
        self.kv_head_idx = self.q_head_idx if traits.GQA_GROUP_SIZE == 1 else self.q_head_idx // traits.GQA_GROUP_SIZE

    def init_load_mapping(self):
        traits = self.traits
        self.load_row_in_batch = self.tid // traits.THREADS_PER_ROW_LOAD
        self.load_lane_in_row = self.tid % traits.THREADS_PER_ROW_LOAD
        self.load_col_base = self.load_lane_in_row * traits.VEC_WIDTH
        if const_expr(traits.V_PERM_TR):
            self.vp_rows_per_thread = 2
            self.vp_num_row_groups = traits.BLOCK_N // self.vp_rows_per_thread
            self.vp_active_threads = traits.THREADS_PER_ROW_LOAD * self.vp_num_row_groups
            self.vp_col_group = self.tid % traits.THREADS_PER_ROW_LOAD
            self.vp_row_group = self.tid // traits.THREADS_PER_ROW_LOAD
            self.vp_col_base = self.vp_col_group * traits.VEC_WIDTH
            self.vp_row_base = self.vp_row_group * self.vp_rows_per_thread
            self.vp_sel_lo = fx.Int32(0x05040100)
            self.vp_sel_hi = fx.Int32(0x07060302)

    def init_constants(self, sm_scale):
        traits = self.traits
        self.c_neg_inf = fx.Float32(float("-inf"))
        self.c_neg_floor = fx.Float32(-3.0e38)
        self.c_zero_f = fx.Float32(0.0)
        self.c_zero_i32 = fx.Int32(0)
        self.c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        self.c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        self.width_i32 = fx.Int32(traits.WARP_SIZE)
        self.shuf_32_i32 = fx.Int32(32)
        self.lane_xor_32_byte = (fx.Int32(self.lane) ^ self.shuf_32_i32) * fx.Int32(4)

    def init_sequence_lengths(self, CuSeqQ, CuSeqKv):
        traits = self.traits
        if const_expr(traits.VARLEN):
            cuq_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(CuSeqQ), fx.make_layout(1, 1))
            cuk_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(CuSeqKv), fx.make_layout(1, 1))
            self.q_tok_base = _cu_load(cuq_div, self.batch_idx, self.load_atom_32, self.v1i32_type)
            self.kv_tok_base = _cu_load(cuk_div, self.batch_idx, self.load_atom_32, self.v1i32_type)
            self.seqlen_q_b = (
                _cu_load(cuq_div, self.batch_idx + fx.Index(1), self.load_atom_32, self.v1i32_type) - self.q_tok_base
            )
            self.seqlen_kv_b = (
                _cu_load(cuk_div, self.batch_idx + fx.Index(1), self.load_atom_32, self.v1i32_type) - self.kv_tok_base
            )
        else:
            self.q_tok_base = self.batch_idx * self.seq_len_v
            self.kv_tok_base = self.batch_idx * self.seq_len_kv_v
            self.seqlen_q_b = self.seq_len_v
            self.seqlen_kv_b = self.seq_len_kv_v

    def init_kv_batch_pointers(self):
        # Dense/varlen fold batch into raw K/V pointers; paged adds page_id per tile.
        traits = self.traits
        if const_expr(not traits.PAGED):
            off = self.kv_tok_base * fx.Index(traits.STRIDE_TOKEN_KV)
            self.k_ptr = buffer_ops.get_element_ptr(self.k_ptr, off, elem_type=self.elem_type)
            self.v_ptr = buffer_ops.get_element_ptr(self.v_ptr, off, elem_type=self.elem_type)

    def init_descriptors(self, Q, O):  # noqa: E741
        # Per-batch descriptors keep global indices 0-based and bounded to one batch,
        # keeping 32-bit offsets small while preserving arbitrary-seqlen OOB behavior.
        traits = self.traits
        self.kv_nrec_bytes = as_mlir_value(self.seqlen_kv_b * fx.Index(traits.STRIDE_TOKEN_KV * 2))
        self.kv_batch_byte_off = as_mlir_value(self.kv_tok_base * fx.Index(traits.STRIDE_TOKEN_KV * 2))
        q_nrec_bytes = as_mlir_value(self.seqlen_q_b * fx.Index(traits.STRIDE_TOKEN_Q * 2))
        q_batch_byte_off = as_mlir_value(self.q_tok_base * fx.Index(traits.STRIDE_TOKEN_Q * 2))
        self.q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=q_nrec_bytes, base_byte_offset=q_batch_byte_off
        )
        self.o_rsrc = buffer_ops.create_buffer_resource(
            O, max_size=False, num_records_bytes=q_nrec_bytes, base_byte_offset=q_batch_byte_off
        )

    def init_q_row(self):
        # B operand: j = lane_mod_32, k-subblock = lane_div_32*MFMA_LANE_K. Q is
        # num_records-bounded (q_rsrc) so OOB rows read 0 -- no q_in_bounds select.
        self.q_row = self.q_start + self.wave_q_offset + self.lane_mod_32
        self.q_row_i32 = fx.Int32(self.q_row)

    def init_kv_bounds(self):
        traits = self.traits
        q_end = self.q_start + traits.BLOCK_M
        if const_expr(traits.CAUSAL):
            self.delta_i32 = fx.Int32(self.seqlen_kv_b) - fx.Int32(self.seqlen_q_b)
            self.causal_end_raw_i32 = fx.Int32(q_end) + self.delta_i32
            causal_end_i32 = fx.Int32(
                (self.causal_end_raw_i32 > fx.Int32(0)).select(self.causal_end_raw_i32, fx.Int32(0))
            )
            causal_end = fx.Index(causal_end_i32)
            self.kv_upper = fx.Index((causal_end < self.seqlen_kv_b).select(causal_end, self.seqlen_kv_b))
        else:
            self.kv_upper = self.seqlen_kv_b

    def global_idx_q(self, token_idx, col):
        traits = self.traits
        return token_idx * traits.STRIDE_TOKEN_Q + self.q_head_idx * traits.HEAD_DIM + col

    def global_idx_kv(self, token_idx, col):
        traits = self.traits
        return token_idx * traits.STRIDE_TOKEN_KV + self.kv_head_idx * traits.HEAD_DIM + col

    def kv_row_clamp(self, row_idx):
        last = self.seqlen_kv_b - fx.Index(1)
        return fx.Index((row_idx < self.seqlen_kv_b).select(row_idx, last))

    def load_global_half_vec(self, ptr, base_idx, vec_elems: int):
        gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=self.elem_type)
        return _pointer_load(Vec.make_type(vec_elems, self.elem_dtype), gep)

    def store_global_half(self, ptr, base_idx, val):
        gep = buffer_ops.get_element_ptr(ptr, fx.Int64(base_idx), elem_type=self.elem_type)
        _pointer_store(val, gep)

    def load_global_f16x4(self, rsrc, base_idx):
        return self.load_global_half_vec(rsrc, base_idx, 4)

    def load_global_mfma_pack(self, rsrc, base_idx):
        return self.load_global_half_vec(rsrc, base_idx, self.traits.MFMA_LANE_K)

    def load_global_f16xN(self, rsrc, base_idx):
        return self.load_global_half_vec(rsrc, base_idx, self.traits.VEC_WIDTH)

    def k_buf_base(self, buf_id):
        if const_expr(isinstance(buf_id, int)):
            return fx.Index(buf_id * self.traits.LDS_K_TILE_SIZE)
        return buf_id * fx.Index(self.traits.LDS_K_TILE_SIZE)

    def v_buf_base(self, buf_id):
        return fx.Index(self.traits.LDS_V_BASE + buf_id * self.traits.LDS_V_TILE_SIZE)

    def k_swizzle(self, row_idx, col_idx):
        mask = (row_idx & fx.Index(self.traits.K_SWZ_ROWMASK)) << fx.Index(4)
        return col_idx ^ mask

    def v_swizzle(self, row_idx, col_idx):
        mask = (row_idx & fx.Index(0x3)) << fx.Index(4)
        return col_idx ^ mask

    def sigma_kv(self, n):
        return (
            (n & fx.Index(3))
            | ((n & fx.Index(8)) >> fx.Index(1))
            | ((n & fx.Index(4)) << fx.Index(1))
            | (n & fx.Index(-16))
        )

    def k_vec_elem_idx(self, n, col_f16):
        traits = self.traits
        d_group = col_f16 // fx.Index(traits.K_VEC_SIZE)
        d_local = col_f16 % fx.Index(traits.K_VEC_SIZE)
        return d_group * fx.Index(traits.K_VEC_N_STRIDE) + n * fx.Index(traits.K_VEC_SIZE) + d_local


class GenericPageIdLoader:
    def __init__(self, ctx, BlockTable, block_table_stride):
        self.ctx = ctx
        self.BlockTable = BlockTable
        self.block_table_stride = block_table_stride
        if const_expr(ctx.traits.PAGED):
            self.bt_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(BlockTable), fx.make_layout(1, 1))
            self.bt_stride_v = fx.Index(block_table_stride)

    def page_id(self, tile_start):
        ctx = self.ctx
        tile_idx = tile_start // fx.Index(ctx.traits.PAGE_SIZE)
        bt_off = ctx.batch_idx * self.bt_stride_v + tile_idx
        v = fly.copy_atom_call_ssa(
            [ctx.v1i32_type],
            ctx.load_atom_32,
            fx.slice(self.bt_div, (None, fx.Int32(bt_off))),
        )
        return fx.Index(Vec(v, (1,), fx.Int32)[0])


class GenericKvGmemToLdsLoader:
    """KV address/load leaf helper; the caller keeps pipeline scheduling local."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.page_ids = None

    def global_idx(self, token_idx, col):
        return self.ctx.global_idx_kv(token_idx, col)

    def row_clamp(self, row_idx):
        return self.ctx.kv_row_clamp(row_idx)

    def load_half_vec(self, ptr, base_idx, vec_elems: int):
        return self.ctx.load_global_half_vec(ptr, base_idx, vec_elems)

    def load_f16xN(self, rsrc, base_idx):
        return self.ctx.load_global_f16xN(rsrc, base_idx)

    def load_vectorized_k(self, k_ptr, page_id, kv_row, col):
        ctx = self.ctx
        traits = ctx.traits
        base = (
            page_id * fx.Index(traits.PAGE_STRIDE_VEC)
            + ctx.kv_head_idx * fx.Index(traits.HEAD_STRIDE_VEC)
            + kv_row * fx.Index(traits.KV_VEC_SIZE)
        )
        dg = col // fx.Index(traits.KV_VEC_SIZE)
        lo = ctx.load_global_half_vec(
            k_ptr,
            base + dg * fx.Index(traits.PAGE_SIZE * traits.KV_VEC_SIZE),
            traits.KV_VEC_SIZE,
        )
        hi = ctx.load_global_half_vec(
            k_ptr,
            base + (dg + fx.Index(1)) * fx.Index(traits.PAGE_SIZE * traits.KV_VEC_SIZE),
            traits.KV_VEC_SIZE,
        )
        return Vec(lo).shuffle(Vec(hi), list(range(traits.VEC_WIDTH))).ir_value()

    def load_vectorized_v(self, v_ptr, page_id, kv_row, col):
        ctx = self.ctx
        traits = ctx.traits
        kg = kv_row // fx.Index(traits.KV_VEC_SIZE)
        kr = kv_row % fx.Index(traits.KV_VEC_SIZE)
        base = (
            page_id * fx.Index(traits.PAGE_STRIDE_VEC)
            + ctx.kv_head_idx * fx.Index(traits.HEAD_STRIDE_VEC)
            + kg * fx.Index(traits.HEAD_DIM * traits.KV_VEC_SIZE)
            + kr
        )
        elems = []
        for i in range_constexpr(traits.VEC_WIDTH):
            v1 = ctx.load_global_half_vec(v_ptr, base + (col + fx.Index(i)) * fx.Index(traits.KV_VEC_SIZE), 1)
            elems.append(Vec(v1)[0])
        return Vec.from_elements(elems, ctx.elem_dtype).ir_value()

    def k_buf_base(self, buf_id):
        return self.ctx.k_buf_base(buf_id)

    def v_buf_base(self, buf_id):
        return self.ctx.v_buf_base(buf_id)

    def k_swizzle(self, row_idx, col_idx):
        return self.ctx.k_swizzle(row_idx, col_idx)

    def v_swizzle(self, row_idx, col_idx):
        return self.ctx.v_swizzle(row_idx, col_idx)

    def sigma_kv(self, n):
        return self.ctx.sigma_kv(n)

    def _tile_page_id(self, tile_start):
        return self.page_ids.page_id(tile_start)

    def coop_load_k(self, tile_start, buf_id=0):
        """Cooperative K load (row-major, XOR-swizzled)."""
        ctx = self.ctx
        traits = ctx.traits
        lds_kv = ctx.lds_kv
        row = ctx.load_row_in_batch
        col = ctx.load_col_base
        k_base = self.k_buf_base(buf_id)
        if const_expr(traits.PAGED):
            pid = self._tile_page_id(tile_start)
        for batch in range_constexpr(traits.NUM_BATCHES_KV):
            row_offset = batch * traits.ROWS_PER_BATCH_LOAD
            if const_expr(traits.PAGED):
                row_idx = pid * fx.Index(traits.PAGE_SIZE) + row + row_offset
            else:
                row_idx = self.row_clamp(tile_start + row + row_offset)
            if const_expr(traits.KV_NEEDS_GUARD):
                if row < fx.Index(traits.BLOCK_N):
                    g_idx = self.global_idx(row_idx, col)
                    lds_row = row + row_offset
                    lds_idx = k_base + lds_row * traits.K_STRIDE + self.k_swizzle(lds_row, col)
                    Vec(self.load_f16xN(ctx.k_ptr, g_idx)).store(lds_kv, [lds_idx])
            else:
                lds_row = row + row_offset
                lds_idx = k_base + lds_row * traits.K_STRIDE + self.k_swizzle(lds_row, col)
                if const_expr(traits.KV_VECTORIZED):
                    vec = self.load_vectorized_k(ctx.k_ptr, pid, self.sigma_kv(lds_row), col)
                else:
                    vec = self.load_f16xN(ctx.k_ptr, self.global_idx(row_idx, col))
                Vec(vec).store(lds_kv, [lds_idx])

    def coop_load_k_global(self, tile_start):
        ctx = self.ctx
        traits = ctx.traits
        vecs = []
        if const_expr(traits.PAGED):
            pid = self._tile_page_id(tile_start)
        for batch in range_constexpr(traits.NUM_BATCHES_KV):
            row_offset = batch * traits.ROWS_PER_BATCH_LOAD
            if const_expr(traits.PAGED):
                row_idx = pid * fx.Index(traits.PAGE_SIZE) + ctx.load_row_in_batch + row_offset
            else:
                row_idx = self.row_clamp(tile_start + ctx.load_row_in_batch + row_offset)
            vecs.append(self.load_f16xN(ctx.k_ptr, self.global_idx(row_idx, ctx.load_col_base)))
        return vecs

    def coop_store_k_lds(self, vecs, buf_id=0):
        ctx = self.ctx
        traits = ctx.traits
        k_base = self.k_buf_base(buf_id)
        for batch in range_constexpr(traits.NUM_BATCHES_KV):
            row_offset = batch * traits.ROWS_PER_BATCH_LOAD
            if const_expr(traits.KV_NEEDS_GUARD):
                if ctx.load_row_in_batch < fx.Index(traits.BLOCK_N):
                    lds_row = ctx.load_row_in_batch + row_offset
                    lds_idx = k_base + lds_row * traits.K_STRIDE + self.k_swizzle(lds_row, ctx.load_col_base)
                    Vec(vecs[batch]).store(ctx.lds_kv, [lds_idx])
            else:
                lds_row = ctx.load_row_in_batch + row_offset
                lds_idx = k_base + lds_row * traits.K_STRIDE + self.k_swizzle(lds_row, ctx.load_col_base)
                Vec(vecs[batch]).store(ctx.lds_kv, [lds_idx])

    def _v_store_to_lds(self, v_base, lds_row, vec):
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(traits.USE_HW_TR):
            lds_idx = v_base + lds_row * traits.V_STRIDE + ctx.load_col_base
            Vec(vec).store(ctx.lds_kv, [lds_idx])
        else:
            for _e in range_constexpr(traits.VEC_WIDTH):
                elem = Vec(vec)[_e]
                vt_d = ctx.load_col_base + _e
                vt_idx = v_base + vt_d * traits.VT_STRIDE + lds_row
                Vec.from_elements([elem], ctx.elem_dtype).store(ctx.lds_kv, [vt_idx])

    def coop_load_v(self, tile_start, buf_id=0):
        """Cooperative V load, storing row-major or transposed per USE_HW_TR."""
        ctx = self.ctx
        traits = ctx.traits
        row = ctx.load_row_in_batch
        col = ctx.load_col_base
        v_base = self.v_buf_base(buf_id)
        if const_expr(traits.PAGED):
            pid = self._tile_page_id(tile_start)
        for batch in range_constexpr(traits.NUM_BATCHES_KV):
            row_offset = batch * traits.ROWS_PER_BATCH_LOAD
            if const_expr(traits.PAGED):
                row_idx = pid * fx.Index(traits.PAGE_SIZE) + row + row_offset
            else:
                row_idx = self.row_clamp(tile_start + row + row_offset)
            if const_expr(traits.KV_NEEDS_GUARD):
                if row < fx.Index(traits.BLOCK_N):
                    g_idx = self.global_idx(row_idx, col)
                    self._v_store_to_lds(v_base, row + row_offset, self.load_f16xN(ctx.v_ptr, g_idx))
            else:
                lds_row = row + row_offset
                if const_expr(traits.KV_VECTORIZED):
                    vec = self.load_vectorized_v(ctx.v_ptr, pid, self.sigma_kv(lds_row), col)
                else:
                    vec = self.load_f16xN(ctx.v_ptr, self.global_idx(row_idx, col))
                self._v_store_to_lds(v_base, lds_row, vec)

    def coop_load_v_global(self, tile_start):
        """Issue global V loads; vectorized mode returns no-major v8 rows."""
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(traits.KV_VECTORIZED):
            pid = self._tile_page_id(tile_start)
            base_v = pid * fx.Index(traits.PAGE_STRIDE_VEC) + ctx.kv_head_idx * fx.Index(traits.HEAD_STRIDE_VEC)
            vecs = []
            for j in range_constexpr(traits.NV8_PER_THREAD):
                flat = ctx.tid + fx.Index(j * traits.BLOCK_SIZE)
                d = flat // fx.Index(traits.VEC_V_NGROUPS)
                ng = flat % fx.Index(traits.VEC_V_NGROUPS)
                src = base_v + ng * fx.Index(traits.HEAD_DIM * traits.KV_VEC_SIZE) + d * fx.Index(traits.KV_VEC_SIZE)
                vecs.append(self.load_half_vec(ctx.v_ptr, src, traits.KV_VEC_SIZE))
            return vecs
        vecs = []
        if const_expr(traits.PAGED):
            pid = self._tile_page_id(tile_start)
        for batch in range_constexpr(traits.NUM_BATCHES_KV):
            row_offset = batch * traits.ROWS_PER_BATCH_LOAD
            if const_expr(traits.PAGED):
                row_idx = pid * fx.Index(traits.PAGE_SIZE) + ctx.load_row_in_batch + row_offset
            else:
                row_idx = self.row_clamp(tile_start + ctx.load_row_in_batch + row_offset)
            vecs.append(self.load_f16xN(ctx.v_ptr, self.global_idx(row_idx, ctx.load_col_base)))
        return vecs

    def coop_store_v_lds(self, vecs, buf_id=0):
        """Write V vectors to LDS; vectorized mode uses no-major rows."""
        ctx = self.ctx
        traits = ctx.traits
        v_base = self.v_buf_base(buf_id)
        if const_expr(traits.KV_VECTORIZED):
            for j in range_constexpr(traits.NV8_PER_THREAD):
                flat = ctx.tid + fx.Index(j * traits.BLOCK_SIZE)
                d = flat // fx.Index(traits.VEC_V_NGROUPS)
                ng = flat % fx.Index(traits.VEC_V_NGROUPS)
                dst = (
                    v_base
                    + (d // fx.Index(8)) * fx.Index(traits.VEC_V_LINE)
                    + ng * fx.Index(traits.VEC_V_D128)
                    + (d % fx.Index(8)) * fx.Index(8)
                )
                Vec(vecs[j]).store(ctx.lds_kv, [dst])
            return
        for batch in range_constexpr(traits.NUM_BATCHES_KV):
            row_offset = batch * traits.ROWS_PER_BATCH_LOAD
            if const_expr(traits.KV_NEEDS_GUARD):
                if ctx.load_row_in_batch < fx.Index(traits.BLOCK_N):
                    self._v_store_to_lds(v_base, ctx.load_row_in_batch + row_offset, vecs[batch])
            else:
                self._v_store_to_lds(v_base, ctx.load_row_in_batch + row_offset, vecs[batch])

    def coop_load_v_global_perm(self, tile_start):
        ctx = self.ctx
        traits = ctx.traits
        vecs = []
        if const_expr(traits.PAGED):
            pid = self._tile_page_id(tile_start)
        for r in range_constexpr(ctx.vp_rows_per_thread):
            if const_expr(traits.PAGED):
                row_idx = pid * fx.Index(traits.PAGE_SIZE) + ctx.vp_row_base + r
            else:
                row_idx = self.row_clamp(tile_start + ctx.vp_row_base + r)
            vecs.append(self.load_f16xN(ctx.v_ptr, self.global_idx(row_idx, ctx.vp_col_base)))
        return vecs

    def coop_store_v_lds_perm(self, vecs, buf_id=0):
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(ctx.vp_active_threads < traits.BLOCK_SIZE):
            active = ctx.tid < fx.Index(ctx.vp_active_threads)

            def _store_active():
                self._coop_store_v_lds_perm_body(vecs, buf_id)

            scf_if_dispatch(active, _store_active)
        else:
            self._coop_store_v_lds_perm_body(vecs, buf_id)

    def _coop_store_v_lds_perm_body(self, vecs, buf_id):
        ctx = self.ctx
        traits = ctx.traits
        v_base = self.v_buf_base(buf_id)
        dwords = [Vec(vecs[r]).bitcast(fx.Int32) for r in range_constexpr(ctx.vp_rows_per_thread)]
        ndw = traits.VEC_WIDTH // 2
        for w in range_constexpr(ndw):
            a = dwords[0][w]
            b = dwords[1][w]
            dl = ctx.vp_col_base + fx.Index(2 * w)
            dh = dl + fx.Index(1)
            lo_01 = rocdl.perm_b32(b, a, ctx.vp_sel_lo)
            v_lo = Vec.from_elements([lo_01], fx.Int32).bitcast(ctx.elem_dtype)
            v_lo.store(ctx.lds_kv, [v_base + dl * fx.Index(traits.VT_STRIDE) + ctx.vp_row_base])
            hi_01 = rocdl.perm_b32(b, a, ctx.vp_sel_hi)
            v_hi = Vec.from_elements([hi_01], fx.Int32).bitcast(ctx.elem_dtype)
            v_hi.store(ctx.lds_kv, [v_base + dh * fx.Index(traits.VT_STRIDE) + ctx.vp_row_base])

    def init_dma_nomajor(self):
        # KV_VECTORIZED V: no-major GM->LDS DMA constants (one aligned v8 per lane).
        ctx = self.ctx
        traits = ctx.traits
        self._v_dma_base_i64 = fx.Int64(buffer_ops.extract_base_index(ctx.V, address_space=1))
        self._v_dma_page_bytes = fx.Int64(traits.PAGE_STRIDE_VEC * 2)
        self._v_dma_lds_base = buffer_ops.extract_base_index(ctx.lds_kv, address_space=3)
        self._v_dma_sz = fx.Int32(16)
        self._v_dma_z = fx.Int32(0)
        self._v_dma_aux = fx.Int32(1)
        self._v_dma_no = ctx.lane // fx.Index(8)
        self._v_dma_dloc = ctx.lane % fx.Index(8)

    def coop_dma_v_nomajor(self, tile_start, buf_id=0):
        # Each lane DMAs one contiguous v8 into the no-major V line.
        ctx = self.ctx
        traits = ctx.traits
        pid = self._tile_page_id(tile_start)
        paddr = as_mlir_value(self._v_dma_base_i64 + fx.Int64(pid) * self._v_dma_page_bytes)
        rsrc = buffer_ops.create_buffer_resource_from_addr(
            paddr, num_records_bytes=as_mlir_value(self._v_dma_page_bytes)
        )
        if const_expr(isinstance(buf_id, int)):
            vb = self._v_dma_lds_base + fx.Index((traits.LDS_V_BASE + buf_id * traits.LDS_V_TILE_SIZE) * 2)
        else:
            vb = self._v_dma_lds_base + fx.Index(traits.LDS_V_BASE * 2) + buf_id * fx.Index(traits.LDS_V_TILE_SIZE * 2)
        for d in range_constexpr(traits.NV8_PER_THREAD):
            row = ctx.wave_id * fx.Index(traits.NV8_PER_THREAD) + fx.Index(d)
            lds_b = vb + row * fx.Index(traits.VEC_V_LINE * 2)
            lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(lds_b))
            lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)
            dcol = row * fx.Index(8) + self._v_dma_dloc
            voff_e = (
                ctx.kv_head_idx * fx.Index(traits.HEAD_STRIDE_VEC)
                + self._v_dma_no * fx.Index(traits.HEAD_DIM * 8)
                + dcol * fx.Index(8)
            )
            voff = fx.Int32(voff_e * fx.Index(2))
            rocdl.raw_ptr_buffer_load_lds(
                rsrc, lds_ptr, self._v_dma_sz, voff, self._v_dma_z, self._v_dma_z, self._v_dma_aux
            )

    def init_dma(self):
        # buffer_load_dwordx4 GM->LDS DMA constants + K/V per-batch resources.
        ctx = self.ctx
        traits = ctx.traits
        self.DMA_BYTES = 4 if traits.ENABLE_GFX942_DMA else 16
        self.DMA_BATCH_BYTES = traits.BLOCK_SIZE * self.DMA_BYTES
        self.lds_kv_base_idx = buffer_ops.extract_base_index(ctx.lds_kv, address_space=3)
        self._dma_size = fx.Int32(self.DMA_BYTES)
        self._dma_soff = fx.Int32(0)
        self._dma_off = fx.Int32(0)
        self._dma_aux = fx.Int32(1)
        self.k_rsrc = buffer_ops.create_buffer_resource(
            ctx.K, max_size=False, num_records_bytes=ctx.kv_nrec_bytes, base_byte_offset=ctx.kv_batch_byte_off
        )
        self.NUM_DMA_K = (traits.BLOCK_N * traits.K_STRIDE * 2) // self.DMA_BATCH_BYTES
        self.LANES_PER_K_ROW = traits.HEAD_DIM * 2 // self.DMA_BYTES
        self.ROWS_PER_DMA_BATCH = self.DMA_BATCH_BYTES // (traits.HEAD_DIM * 2)
        self.v_rsrc = buffer_ops.create_buffer_resource(
            ctx.V, max_size=False, num_records_bytes=ctx.kv_nrec_bytes, base_byte_offset=ctx.kv_batch_byte_off
        )
        self.NUM_DMA_V = (traits.BLOCK_N * traits.V_STRIDE * 2) // self.DMA_BATCH_BYTES
        self.LANES_PER_V_ROW = traits.HEAD_DIM * 2 // self.DMA_BYTES
        self.ROWS_PER_DMA_BATCH_V = self.DMA_BATCH_BYTES // (traits.HEAD_DIM * 2)

    def _coop_dma_row(self, rsrc, lds_byte_base, num_dma, lanes_per_row, rows_per_batch, tile_start, xor_and):
        ctx = self.ctx
        traits = ctx.traits
        for d in range_constexpr(num_dma):
            lds_addr = (
                lds_byte_base
                + ctx.wave_id * fx.Index(traits.WARP_SIZE * self.DMA_BYTES)
                + fx.Index(d * self.DMA_BATCH_BYTES)
            )
            lds_lane0 = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(lds_addr))
            lds_ptr = buffer_ops.create_llvm_ptr(lds_lane0, address_space=3)
            row_in_tile = ctx.tid // lanes_per_row + fx.Index(d * rows_per_batch)
            swiz_col_f16 = (ctx.tid % lanes_per_row) * (self.DMA_BYTES // 2)
            xor_mask = (row_in_tile & fx.Index(xor_and)) << fx.Index(4)
            col_byte = (swiz_col_f16 ^ xor_mask) * 2
            global_row = tile_start + row_in_tile  # 0-based: batch base folded into k/v_rsrc
            global_byte = (
                global_row * fx.Index(traits.STRIDE_TOKEN_KV * 2)
                + ctx.kv_head_idx * fx.Index(traits.HEAD_DIM * 2)
                + col_byte
            )
            rocdl.raw_ptr_buffer_load_lds(
                rsrc, lds_ptr, self._dma_size, fx.Int32(global_byte), self._dma_soff, self._dma_off, self._dma_aux
            )

    def coop_dma_k(self, tile_start, buf_id=0):
        """Load K tile via DMA with XOR-swizzled global fetch."""
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(isinstance(buf_id, int)):
            k_lds_byte_base = self.lds_kv_base_idx + fx.Index(buf_id * traits.LDS_K_TILE_SIZE * 2)
        else:
            k_lds_byte_base = self.lds_kv_base_idx + buf_id * fx.Index(traits.LDS_K_TILE_SIZE * 2)
        if const_expr(traits.ENABLE_GFX942_VEC_K):
            for d in range_constexpr(self.NUM_DMA_K):
                row_in_tile = ctx.tid // self.LANES_PER_K_ROW + fx.Index(d * self.ROWS_PER_DMA_BATCH)
                col_f16 = (ctx.tid % self.LANES_PER_K_ROW) * (self.DMA_BYTES // 2)
                lds_elem = ctx.k_vec_elem_idx(row_in_tile, col_f16)
                lds_addr = k_lds_byte_base + lds_elem * 2
                global_row = tile_start + ctx.sigma_kv(row_in_tile)
                col_byte = col_f16 * 2
                lds_ptr = buffer_ops.create_llvm_ptr(fx.Int64(lds_addr), address_space=3)
                global_byte = (
                    global_row * fx.Index(traits.STRIDE_TOKEN_KV * 2)
                    + ctx.kv_head_idx * fx.Index(traits.HEAD_DIM * 2)
                    + col_byte
                )
                rocdl.raw_ptr_buffer_load_lds(
                    self.k_rsrc,
                    lds_ptr,
                    self._dma_size,
                    fx.Int32(global_byte),
                    self._dma_soff,
                    self._dma_off,
                    self._dma_aux,
                )
        else:
            self._coop_dma_row(
                self.k_rsrc,
                k_lds_byte_base,
                self.NUM_DMA_K,
                self.LANES_PER_K_ROW,
                self.ROWS_PER_DMA_BATCH,
                tile_start,
                0x7,
            )

    def coop_dma_v(self, tile_start, buf_id=0):
        """Load V tile via DMA with XOR-swizzled global fetch."""
        traits = self.ctx.traits
        v_lds_byte_base = self.lds_kv_base_idx + fx.Index((traits.LDS_V_BASE + buf_id * traits.LDS_V_TILE_SIZE) * 2)
        self._coop_dma_row(
            self.v_rsrc,
            v_lds_byte_base,
            self.NUM_DMA_V,
            self.LANES_PER_V_ROW,
            self.ROWS_PER_DMA_BATCH_V,
            tile_start,
            0x3,
        )


class GenericKvLdsToVgprLoader:
    def __init__(self, ctx):
        self.ctx = ctx
        self._pv_steps = [(dc, pks) for dc in range(ctx.traits.D_CHUNKS) for pks in range(ctx.traits.PV_K_STEPS)]
        self.total_pv = len(self._pv_steps)

    def ds_read_tr_v4f16(self, lds_elem_idx):
        ctx = self.ctx
        byte_offset = lds_elem_idx * 2 + ctx.lds_kv_offset
        ptr = buffer_ops.create_llvm_ptr(fx.Int64(byte_offset), address_space=3)
        return rocdl.ds_read_tr16_b64(ctx.v4f16_type, ptr).result

    def load_k_packs(self, k_base):
        ctx = self.ctx
        traits = ctx.traits
        k_hi_offset = traits.K_SUB_N * traits.K_STRIDE
        k_swz_mask = (ctx.lane_mod_32 & fx.Index(traits.K_SWZ_ROWMASK)) << fx.Index(4)

        def _idx(ks, hi):
            if const_expr(traits.ENABLE_GFX942_VEC_K):
                base = k_base + (fx.Index(traits.K_VEC_HI_N_OFFSET) if hi else fx.Index(0))
                return (
                    base
                    + fx.Index(ks) * fx.Index(traits.K_VEC_N_STRIDE)
                    + ctx.lane_mod_32 * fx.Index(traits.K_VEC_SIZE)
                    + ctx.lane_div_32 * ctx.MFMA_LANE_K
                )
            col = fx.Index(ks * traits.K_STEP_QK) + ctx.lane_div_32 * ctx.MFMA_LANE_K
            base = k_base + (k_hi_offset if hi else fx.Index(0)) + ctx.lane_mod_32 * traits.K_STRIDE
            return base + (col ^ k_swz_mask)

        depth = traits.QK_PREFETCH_DEPTH
        lo = [None] * traits.K_STEPS_QK
        hi = [None] * traits.K_STEPS_QK
        for p in range_constexpr(depth):
            lo[p] = Vec.load(ctx.mfma_pack_type, ctx.lds_kv, [_idx(p, False)]).ir_value()
            hi[p] = Vec.load(ctx.mfma_pack_type, ctx.lds_kv, [_idx(p, True)]).ir_value()
        if const_expr(traits.ENABLE_GFX942_VEC_K or traits.ENABLE_GFX942_KV_GPFETCH):
            rocdl.sched_group_barrier(rocdl.mask_dsrd, depth * 2, 0)
        self._k_idx = _idx
        self._k_depth = depth
        return lo, hi

    def load_k_pack_at(self, ks):
        ctx = self.ctx
        lo = Vec.load(ctx.mfma_pack_type, ctx.lds_kv, [self._k_idx(ks, False)]).ir_value()
        hi = Vec.load(ctx.mfma_pack_type, ctx.lds_kv, [self._k_idx(ks, True)]).ir_value()
        return lo, hi

    def read_v_pack(self, step_idx, v_base):
        ctx = self.ctx
        traits = ctx.traits
        dc, pks = self._pv_steps[step_idx]
        if const_expr(traits.KV_VECTORIZED):
            # No-major V: one aligned v8 per (dc,pks) half. Lane l reads
            # V[d=dc*32+l%32, n=pks*16+(l//32)*8+0..7] (lo) / +32 (hi).
            lm = ctx.lane_mod_32
            v_lane_base = (
                v_base
                + (lm // fx.Index(8)) * fx.Index(traits.VEC_V_LINE)
                + ctx.lane_div_32 * fx.Index(traits.VEC_V_D128)
                + (lm % fx.Index(8)) * fx.Index(8)
            )
            lo_off = dc * (traits.D_CHUNK // 8) * traits.VEC_V_LINE + pks * (traits.PV_K_STEP // 8) * traits.VEC_V_D128
            hi_off = lo_off + (traits.K_SUB_N // 8) * traits.VEC_V_D128
            vl = Vec.load(ctx.mfma_pack_type, ctx.lds_kv, [v_lane_base + fx.Index(lo_off)])
            vh = Vec.load(ctx.mfma_pack_type, ctx.lds_kv, [v_lane_base + fx.Index(hi_off)])
            return vl, vh
        if const_expr(traits.USE_HW_TR):
            d_col = fx.Index(dc * traits.D_CHUNK) + ctx.tr_col_half * 16 + ctx.tr_col_sub * 4
            k_row = fx.Index(pks * traits.PV_K_STEP) + ctx.lane_div_32 * 4 + ctx.tr_k_group
            d_col_eff = ctx.v_swizzle(k_row, d_col) if traits.ENABLE_DMA else d_col
            lds_lo = v_base + k_row * traits.V_STRIDE + d_col_eff
            lds_hi = lds_lo + fx.Index(traits.K_SUB_N * traits.V_STRIDE)
            if const_expr(traits.USE_K16):
                vl_a = self.ds_read_tr_v4f16(lds_lo)
                vl_b = self.ds_read_tr_v4f16(lds_lo + fx.Index(8 * traits.V_STRIDE))
                vl = Vec(vl_a).shuffle(Vec(vl_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                vh_a = self.ds_read_tr_v4f16(lds_hi)
                vh_b = self.ds_read_tr_v4f16(lds_hi + fx.Index(8 * traits.V_STRIDE))
                vh = Vec(vh_a).shuffle(Vec(vh_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
            else:
                vl = self.ds_read_tr_v4f16(lds_lo)
                vh = self.ds_read_tr_v4f16(lds_hi)
            return vl, vh
        d_pos = fx.Index(dc * traits.D_CHUNK) + ctx.lane_mod_32
        k_col = fx.Index(pks * traits.PV_K_STEP) + ctx.lane_div_32 * 4
        v_lo_idx = v_base + d_pos * traits.VT_STRIDE + k_col
        v_hi_idx = v_lo_idx + fx.Index(traits.K_SUB_N)
        vl = Vec.load(ctx.v4f16_type, ctx.lds_kv, [v_lo_idx])
        vh = Vec.load(ctx.v4f16_type, ctx.lds_kv, [v_hi_idx])
        return vl, vh


class GenericQLoader:
    def __init__(self, ctx):
        self.ctx = ctx

    def load_all(self, q_rsrc, q_row):
        ctx = self.ctx
        traits = ctx.traits
        q_b_packs = []
        for ks in range_constexpr(traits.K_STEPS_QK):
            q_col = fx.Index(ks * traits.K_STEP_QK) + ctx.lane_div_32 * traits.MFMA_LANE_K
            q_b_packs.append(
                buffer_ops.buffer_load(
                    q_rsrc,
                    ctx.global_idx_q(q_row, q_col),
                    vec_width=traits.MFMA_LANE_K,
                    dtype=ctx.elem_dtype,
                )
            )
        return q_b_packs


class GenericGemmHelper:
    def __init__(self, ctx):
        self.ctx = ctx

    def _mfma(self, mfma_fn, a, b, c):
        return mfma_fn(self.ctx.v16f32_type, [a, b, c])

    def mfma_acc(self, a, b, c):
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(traits.USE_K16):
            return fly.mma_atom_call_ssa([ctx.v16f32_type], ctx.mma_atom_k16, a, b, c)
        if const_expr(traits.DTYPE_STR == "bf16"):
            a = Vec(a).bitcast(fx.Int16)
            b = Vec(b).bitcast(fx.Int16)
            return self._mfma(rocdl.mfma_f32_32x32x8bf16_1k, a, b, c)
        return self._mfma(rocdl.mfma_f32_32x32x8f16, a, b, c)

    def gemm1_accumulate(self, kv_lds_to_vgpr, k_lo, k_hi, q_b_packs, kv_gmem_to_lds=None, kv_start=None):
        ctx = self.ctx
        traits = ctx.traits
        depth = traits.QK_PREFETCH_DEPTH
        s_lo = ctx.c_zero_v16f32
        s_hi = ctx.c_zero_v16f32
        for ks in range_constexpr(traits.K_STEPS_QK):
            if const_expr(
                traits.ENABLE_DMA
                and traits.USE_HW_TR
                and not traits.ENABLE_PREFETCH_3BUF
                and ks == traits.K_STEPS_QK // 2
            ):
                kv_gmem_to_lds.coop_dma_v(kv_start, 0)
                rocdl.sched_barrier(0)
            s_lo = self.mfma_acc(k_lo[ks], q_b_packs[ks], s_lo)
            s_hi = self.mfma_acc(k_hi[ks], q_b_packs[ks], s_hi)
            if const_expr(ks + depth < traits.K_STEPS_QK):
                k_lo[ks + depth], k_hi[ks + depth] = kv_lds_to_vgpr.load_k_pack_at(ks + depth)
        return s_lo, s_hi

    def gemm2_pv(self, kv_lds_to_vgpr, o_accs, p_packs_lo, p_packs_hi, v_base, corr_vec):
        # O += V^T_lo @ P_lo + V^T_hi @ P_hi with interleaved V prefetch.
        traits = self.ctx.traits
        steps = kv_lds_to_vgpr._pv_steps
        total = kv_lds_to_vgpr.total_pv
        v_lo_cur, v_hi_cur = kv_lds_to_vgpr.read_v_pack(0, v_base)
        for si in range_constexpr(total):
            dc, pks = steps[si]
            if const_expr(si + 1 < total):
                v_lo_nxt, v_hi_nxt = kv_lds_to_vgpr.read_v_pack(si + 1, v_base)
            o_accs[dc] = self.mfma_acc(v_lo_cur, p_packs_lo[pks], o_accs[dc])
            o_accs[dc] = self.mfma_acc(v_hi_cur, p_packs_hi[pks], o_accs[dc])
            if const_expr(not traits.USE_HW_TR and dc == 0 and pks < traits.D_CHUNKS - 1):
                o_accs[pks + 1] = Vec(o_accs[pks + 1]) * corr_vec
            if const_expr(si + 1 < total):
                v_lo_cur = v_lo_nxt
                v_hi_cur = v_hi_nxt
        return o_accs


class GenericSoftmaxHelper:
    def __init__(self, ctx):
        self.ctx = ctx

    def reduction_peer(self, v_f32):
        ctx = self.ctx
        if const_expr(ctx.traits.REDUCE_MODE == "ds_bpermute"):
            v_i32 = fx.Float32(v_f32).bitcast(fx.Int32)
            peer_i32 = rocdl.ds_bpermute(fx.Int32.ir_type, ctx.lane_xor_32_byte, v_i32)
            return fx.Int32(peer_i32).bitcast(ctx.compute_type)
        return fx.Float32(v_f32).shuffle_xor(ctx.shuf_32_i32, ctx.width_i32)

    def split_scores(self, s_acc_lo, s_acc_hi):
        return _score_pair_to_lists((s_acc_lo, s_acc_hi))

    def _kv_mask_lane_off(self, kv_start_i32):
        # Physical KV column base per lane; KV_VECTORIZED applies sigma(kv) in the K load.
        ctx = self.ctx
        if const_expr(ctx.traits.KV_VECTORIZED):
            lane_off = fx.Int32(ctx.lane_div_32) * fx.Int32(8)
            moff = (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23)
        else:
            lane_off = fx.Int32(ctx.lane_div_32) * fx.Int32(4)
            moff = (0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27)
        return kv_start_i32 + lane_off, moff

    def apply_kv_mask(self, s_raw_lo, s_raw_hi, kv_start):
        ctx = self.ctx
        traits = ctx.traits
        kv_start_i32 = fx.Int32(kv_start)
        if const_expr(traits.CAUSAL):
            # Keep the runtime tile_needs_mask guard (below-diagonal tiles skip the 32
            # selects) but drive the scf.if with the 32 scalar scores as explicit state
            # (a Python list cannot cross a dynamic `if`) -> byte-identical to the unrolled
            # form. The score at logical n_pos holds physical kv = kv_start + sigma(n_pos).
            q_start_i32 = fx.Int32(ctx.q_start) + ctx.delta_i32
            q_mask_limit_i32 = ctx.q_row_i32 + ctx.delta_i32
            max_kv_col_i32 = kv_start_i32 + fx.Int32(traits.BLOCK_N - 1)
            tile_needs_mask = max_kv_col_i32 > q_start_i32
            col_base_i32, moff = self._kv_mask_lane_off(kv_start_i32)
            c_neg_inf = ctx.c_neg_inf

            def _apply_causal_mask(_names, *scores):
                out = []
                for r in range_constexpr(16):
                    kv_col = col_base_i32 + fx.Int32(moff[r])
                    out.append((kv_col > q_mask_limit_i32).select(c_neg_inf, scores[2 * r]))
                    out.append(
                        (kv_col + fx.Int32(traits.K_SUB_N) > q_mask_limit_i32).select(c_neg_inf, scores[2 * r + 1])
                    )
                return out

            mask_names = tuple("_sm%d" % i for i in range(32))
            interleaved = [v for r in range(16) for v in (s_raw_lo[r], s_raw_hi[r])]
            masked = scf_if_dispatch(
                tile_needs_mask, _apply_causal_mask, state_names=mask_names, state_values=interleaved
            )
            return [masked[2 * r] for r in range(16)], [masked[2 * r + 1] for r in range(16)]

        # Non-causal: mask physical KV columns outside seqlen so tail rows stay out of softmax.
        seq_len_i32 = fx.Int32(ctx.seqlen_kv_b)
        if const_expr(not traits.SKIP_KV_PAD_MASK):
            col_base_i32, moff = self._kv_mask_lane_off(kv_start_i32)
            kv_tile_end = kv_start + fx.Index(traits.BLOCK_N)
            needs_pad_mask = fx.Int32(kv_tile_end) > seq_len_i32
            for r in range_constexpr(16):
                kv_col = col_base_i32 + fx.Int32(moff[r])
                masked_lo = (kv_col >= seq_len_i32).select(ctx.c_neg_inf, s_raw_lo[r])
                masked_hi = (kv_col + fx.Int32(traits.K_SUB_N) >= seq_len_i32).select(ctx.c_neg_inf, s_raw_hi[r])
                s_raw_lo[r] = (needs_pad_mask).select(masked_lo, s_raw_lo[r])
                s_raw_hi[r] = (needs_pad_mask).select(masked_hi, s_raw_hi[r])
        return s_raw_lo, s_raw_hi

    def _exp2(self, x):
        ctx = self.ctx
        if const_expr(os.getenv("FLYDSL_FLASH_ATTN_FUNC_NATIVE_EXP2", "1") == "1"):
            return rocdl.exp2(T.f32, as_mlir_value(x))
        return fx.Float32(x).exp2(fastmath=ctx.fm_fast)

    def online_softmax_stats(self, m_running, s_raw_lo, s_raw_hi):
        ctx = self.ctx
        traits = ctx.traits
        fm_fast = ctx.fm_fast

        if const_expr(os.getenv("FLYDSL_FLASH_ATTN_FUNC_TREE_REDUCE", "0") == "1"):

            def _max_pair(a, b):
                return _fmax(a, b, fm_fast)

            local_max = _tree_reduce(list(s_raw_lo) + list(s_raw_hi), _max_pair)
        else:
            local_max = s_raw_lo[0]
            for r in range_constexpr(15):
                local_max = _fmax(local_max, s_raw_lo[r + 1], fm_fast)
            for r in range_constexpr(16):
                local_max = _fmax(local_max, s_raw_hi[r], fm_fast)
        row_max = _fmax(local_max, self.reduction_peer(local_max), fm_fast)
        m_new_raw = _fmax(m_running, row_max, fm_fast)
        if const_expr(traits.CAUSAL):
            m_new_raw = _fmax(m_new_raw, ctx.c_neg_floor, fm_fast)

        diff_m_scaled = _fmul(_fsub(m_running, m_new_raw, fm_fast), ctx.c_sm_scale_log2e, fm_fast)
        corr = self._exp2(diff_m_scaled)
        neg_scaled_max = _fsub(ctx.c_zero_f, _fmul(ctx.c_sm_scale_log2e, m_new_raw, fm_fast), fm_fast)
        return m_new_raw, corr, neg_scaled_max

    def online_softmax(self, m_running, l_running, s_raw_lo, s_raw_hi):
        ctx = self.ctx
        fm_fast = ctx.fm_fast
        m_new_raw, corr, neg_scaled_max = self.online_softmax_stats(m_running, s_raw_lo, s_raw_hi)

        p_vals_lo = []
        p_vals_hi = []
        local_sum = ctx.c_zero_f
        for r in range_constexpr(16):
            diff_lo = fmath.fma(s_raw_lo[r], ctx.c_sm_scale_log2e, neg_scaled_max, fastmath=ctx.fm_fast)
            p_lo = self._exp2(diff_lo)
            p_vals_lo.append(p_lo)
            local_sum = _fadd(local_sum, p_lo, fm_fast)
        for r in range_constexpr(16):
            diff_hi = fmath.fma(s_raw_hi[r], ctx.c_sm_scale_log2e, neg_scaled_max, fastmath=ctx.fm_fast)
            p_hi = self._exp2(diff_hi)
            p_vals_hi.append(p_hi)
            local_sum = _fadd(local_sum, p_hi, fm_fast)

        tile_sum = _fadd(local_sum, self.reduction_peer(local_sum), fm_fast)
        l_new = _fadd(_fmul(corr, l_running, fm_fast), tile_sum, fm_fast)
        return m_new_raw, l_new, corr, p_vals_lo, p_vals_hi

    def rescale_o_accs(self, o_accs, corr):
        ctx = self.ctx
        traits = ctx.traits
        corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(16)
        if const_expr(not traits.USE_HW_TR):
            o_accs[0] = _fmul(Vec(o_accs[0]), corr_vec, ctx.fm_fast)
        else:
            for dc in range_constexpr(traits.D_CHUNKS):
                o_accs[dc] = _fmul(Vec(o_accs[dc]), corr_vec, ctx.fm_fast)
        return o_accs, corr_vec

    def build_p_packs(self, p_vals):
        ctx = self.ctx
        traits = ctx.traits
        packs = []
        if const_expr(traits.DTYPE_STR == "bf16"):
            for pks in range_constexpr(traits.PV_K_STEPS):
                p_base = pks * traits.MFMA_LANE_K
                chunk = p_vals[p_base : p_base + traits.MFMA_LANE_K]
                if const_expr(traits.USE_K16):
                    packs.append(self.bf16_trunc_pack_v8(chunk))
                else:
                    packs.append(self.bf16_trunc_pack_v4(chunk))
            return packs

        p_f16 = []
        for r in range_constexpr(16):
            p_f16.append(fx.Float32(p_vals[r]).to(ctx.elem_dtype))
        for pks in range_constexpr(traits.PV_K_STEPS):
            p_base = pks * traits.MFMA_LANE_K
            packs.append(Vec.from_elements(p_f16[p_base : p_base + traits.MFMA_LANE_K], ctx.elem_dtype).ir_value())
        return packs

    def _bitcast_i32(self, value):
        return fx.Float32(value).bitcast(fx.Int32)

    def _pack_bf16_pair(self, lo, hi, shift, mask):
        lo_i32 = self._bitcast_i32(lo)
        hi_i32 = self._bitcast_i32(hi)
        if const_expr(os.getenv("FLYDSL_FLASH_ATTN_FUNC_PERM_PACK", "1") == "1"):
            return fx.Int32(rocdl.perm_b32(hi_i32, lo_i32, fx.Int32(0x07060302)))
        return (hi_i32 & mask) | lo_i32.shrui(shift)

    def bf16_trunc_pack_v4(self, f32_vals):
        c16 = fx.Int32(16)
        cmask = fx.Int32(0xFFFF0000)
        packed = [
            self._pack_bf16_pair(f32_vals[0], f32_vals[1], c16, cmask),
            self._pack_bf16_pair(f32_vals[2], f32_vals[3], c16, cmask),
        ]
        return Vec.from_elements(packed, fx.Int32).bitcast(self.ctx.elem_dtype).ir_value()

    def bf16_trunc_pack_v8(self, f32_vals):
        c16 = fx.Int32(16)
        cmask = fx.Int32(0xFFFF0000)
        pairs = []
        for j in range_constexpr(4):
            pairs.append(self._pack_bf16_pair(f32_vals[j * 2], f32_vals[j * 2 + 1], c16, cmask))
        return Vec.from_elements(pairs, fx.Int32).bitcast(self.ctx.elem_dtype).ir_value()

    def gemm2_gpfetch_fused(
        self,
        gemm_helper,
        kv_lds_to_vgpr,
        o_accs,
        corr_vec,
        corr,
        l_running,
        s_raw_lo,
        s_raw_hi,
        neg_scaled_max,
        v_base,
    ):
        ctx = self.ctx
        traits = ctx.traits
        fm_fast = ctx.fm_fast
        local_sum = ctx.c_zero_f
        if const_expr(not traits.USE_HW_TR):
            for dc in range_constexpr(1, traits.D_CHUNKS):
                o_accs[dc] = Vec(o_accs[dc]) * corr_vec
        for pks in range_constexpr(traits.PV_K_STEPS):
            p_base = pks * traits.MFMA_LANE_K
            p_exp_lo = []
            p_exp_hi = []
            for j in range_constexpr(traits.MFMA_LANE_K):
                diff_lo = fmath.fma(s_raw_lo[p_base + j], ctx.c_sm_scale_log2e, neg_scaled_max, fastmath=ctx.fm_fast)
                p_exp_lo.append(self._exp2(diff_lo))
                diff_hi = fmath.fma(s_raw_hi[p_base + j], ctx.c_sm_scale_log2e, neg_scaled_max, fastmath=ctx.fm_fast)
                p_exp_hi.append(self._exp2(diff_hi))
            for j in range_constexpr(traits.MFMA_LANE_K):
                local_sum = _fadd(local_sum, p_exp_lo[j], fm_fast)
                local_sum = _fadd(local_sum, p_exp_hi[j], fm_fast)
            p_lo = self.bf16_trunc_pack_v4(p_exp_lo)
            p_hi = self.bf16_trunc_pack_v4(p_exp_hi)
            v_lo = [None] * traits.D_CHUNKS
            v_hi = [None] * traits.D_CHUNKS
            for dc in range_constexpr(traits.D_CHUNKS):
                v_lo[dc], v_hi[dc] = kv_lds_to_vgpr.read_v_pack(dc * traits.PV_K_STEPS + pks, v_base)
            for dc in range_constexpr(traits.D_CHUNKS):
                o_accs[dc] = gemm_helper.mfma_acc(v_lo[dc], p_lo, o_accs[dc])
            for dc in range_constexpr(traits.D_CHUNKS):
                o_accs[dc] = gemm_helper.mfma_acc(v_hi[dc], p_hi, o_accs[dc])
        tile_sum = _fadd(local_sum, self.reduction_peer(local_sum), fm_fast)
        l_new = _fadd(_fmul(corr, l_running, fm_fast), tile_sum, fm_fast)
        return o_accs, l_new


class GenericStoreHelper:
    def __init__(self, ctx):
        self.ctx = ctx

    def zero_o_block(self, o_rsrc, q_row):
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(traits.USE_PERMLANE_OSTORE):
            zero_pack = Vec.filled(4, 0, fx.Int32)
            for dc in range_constexpr(traits.D_CHUNKS):
                for g in range_constexpr(2):
                    d_col = fx.Index(dc * traits.D_CHUNK) + (fx.Index(2 * g) + ctx.lane_div_32) * fx.Index(8)
                    o_global = ctx.global_idx_q(q_row, d_col)
                    buffer_ops.buffer_store(zero_pack, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True)
        else:
            zero_pack = Vec.filled(2, 0, fx.Int32)
            for dc in range_constexpr(traits.D_CHUNKS):
                for grp in range_constexpr(4):
                    d_col = fx.Index(dc * traits.D_CHUNK) + ctx.lane_div_32 * fx.Index(4) + fx.Index(grp * 8)
                    o_global = ctx.global_idx_q(q_row, d_col)
                    buffer_ops.buffer_store(zero_pack, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True)

    def finalize_o(self, loop_results):
        # Normalize and store O (128-bit buffer_store_dwordx4); cross-seqlen zeroes
        # fully-masked q-blocks (causal_end_raw <= 0) instead of dividing by l==0.
        ctx = self.ctx
        traits = ctx.traits
        if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):

            @flyc.jit
            def _store_cross_o():
                if ctx.causal_end_raw_i32 <= ctx.c_zero_i32:
                    self.zero_o_block(ctx.o_rsrc, ctx.q_row)
                else:
                    self.normalize_and_store_o(loop_results, ctx.o_rsrc, ctx.q_row)

            _store_cross_o()
        else:
            self.normalize_and_store_o(loop_results, ctx.o_rsrc, ctx.q_row)

    def normalize_and_store_o(self, loop_results, o_rsrc, q_row):
        ctx = self.ctx
        traits = ctx.traits
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(traits.D_CHUNKS)]

        inv_l_rcp = rocdl.rcp(T.f32, l_final)
        if const_expr(traits.CAUSAL):
            inv_l = (fx.Float32(l_final) > ctx.c_zero_f).select(inv_l_rcp, ctx.c_zero_f)
        else:
            inv_l = inv_l_rcp
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(16)
        v_o = [Vec(o_finals[dc]) * inv_l_vec for dc in range_constexpr(traits.D_CHUNKS)]

        if const_expr(traits.USE_PERMLANE_OSTORE):
            self._store_permlane_o(v_o, o_rsrc, q_row)
        else:
            self._store_fallback_o(v_o, o_rsrc, q_row)

    def _store_permlane_o(self, v_o, o_rsrc, q_row):
        ctx = self.ctx
        traits = ctx.traits
        pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
        is_hi_half = ctx.lane_div_32 != fx.Index(0)

        def _o_pack_2dw(dc, store_group):
            r_base = store_group * 4
            if const_expr(traits.DTYPE_STR == "bf16"):
                lo = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base], Vec(v_o[dc])[r_base + 1])
                hi = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base + 2], Vec(v_o[dc])[r_base + 3])
                return lo, hi
            o_f16 = [fx.Float32(Vec(v_o[dc])[r_base + i]).to(ctx.elem_dtype) for i in range_constexpr(4)]
            pack = Vec.from_elements(o_f16, ctx.elem_dtype).bitcast(fx.Int32)
            return as_mlir_value(pack[0]), as_mlir_value(pack[1])

        def _swap_halves(dw):
            swapped = rocdl.permlane32_swap(pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False)
            lo_res = llvm.extractvalue(T.i32, swapped, [0])
            hi_res = llvm.extractvalue(T.i32, swapped, [1])
            return is_hi_half.select(lo_res, hi_res)

        for dc in range_constexpr(traits.D_CHUNKS):
            for g in range_constexpr(2):
                d0_a, d1_a = _o_pack_2dw(dc, 2 * g)
                d0_b, d1_b = _o_pack_2dw(dc, 2 * g + 1)
                y0_a, y1_a = _swap_halves(d0_a), _swap_halves(d1_a)
                y0_b, y1_b = _swap_halves(d0_b), _swap_halves(d1_b)
                w0 = is_hi_half.select(y0_b, as_mlir_value(d0_a))
                w1 = is_hi_half.select(y1_b, as_mlir_value(d1_a))
                w2 = is_hi_half.select(as_mlir_value(d0_b), y0_a)
                w3 = is_hi_half.select(as_mlir_value(d1_b), y1_a)
                o_pack = Vec.from_elements([fx.Int32(w0), fx.Int32(w1), fx.Int32(w2), fx.Int32(w3)], fx.Int32)
                d_col = fx.Index(dc * traits.D_CHUNK) + (fx.Index(2 * g) + ctx.lane_div_32) * fx.Index(8)
                o_global = ctx.global_idx_q(q_row, d_col)
                buffer_ops.buffer_store(o_pack, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True)

    def _store_fallback_o(self, v_o, o_rsrc, q_row):
        ctx = self.ctx
        traits = ctx.traits
        for dc in range_constexpr(traits.D_CHUNKS):
            for grp in range_constexpr(4):
                r0 = grp * 4
                o_f16 = [fx.Float32(Vec(v_o[dc])[r0 + i]).to(ctx.elem_dtype) for i in range_constexpr(4)]
                pack = Vec.from_elements(o_f16, ctx.elem_dtype).bitcast(fx.Int32)
                o2 = Vec.from_elements([as_mlir_value(pack[0]), as_mlir_value(pack[1])], fx.Int32)
                d_col = fx.Index(dc * traits.D_CHUNK) + ctx.lane_div_32 * fx.Index(4) + fx.Index(grp * 8)
                o_global = ctx.global_idx_q(q_row, d_col)
                buffer_ops.buffer_store(o2, o_rsrc, o_global * fx.Index(2), offset_is_bytes=True)


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
        _init_dualwave_thread_mapping(self)

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
        _init_dualwave_q_row(self)

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
            causal_end_i32 = fx.Int32(
                (self.causal_end_raw_i32 > fx.Int32(0)).select(self.causal_end_raw_i32, fx.Int32(0))
            )
            causal_num_tiles = (fx.Index(causal_end_i32) + self.kv_tile_size - 1) // self.kv_tile_size
            self.max_num_tiles = fx.Index(
                (causal_num_tiles < self.num_kv_tiles).select(causal_num_tiles, self.num_kv_tiles)
            )
        else:
            self.causal_end_raw_i32 = None
            self.max_num_tiles = self.num_kv_tiles

        self.max_num_tiles = ((self.max_num_tiles + fx.Index(1)) // fx.Index(2)) * fx.Index(2)
        self.max_num_tiles = fx.Index((self.max_num_tiles < fx.Index(4)).select(fx.Index(4), self.max_num_tiles))

        if const_expr(traits.SPLITK):
            chunk = ((self.max_num_tiles + (traits.NUM_KV_SPLITS - 1)) // traits.NUM_KV_SPLITS + 1) // 2 * 2
            chunk = fx.Index((chunk < fx.Index(6)).select(fx.Index(6), chunk))
            self.split_t0 = self.split_idx * chunk
            self.split_t_end = self.split_t0 + chunk
            self.split_t_end = fx.Index(
                (self.split_t_end < self.max_num_tiles).select(self.split_t_end, self.max_num_tiles)
            )
            self.split_t_end = fx.Index(
                (self.max_num_tiles - self.split_t_end < fx.Index(4)).select(self.max_num_tiles, self.split_t_end)
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
                return (self.q_start < self.seqlen_q_v) & (self.causal_end_raw_i32 > fx.Int32(0))
            return self.q_start < self.seqlen_q_v
        if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
            return self.causal_end_raw_i32 > fx.Int32(0)
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


# Pipeline helpers


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
        return _score_pair_max(v_s, self.c_neg_inf, self.fm_fast)

    def floor_masked_max(self, row_max):
        return _fmax(row_max, self.c_neg_floor, self.fm_fast)

    def rescale_from_tile_max(self, m_row, m_tile_max):
        return _rescale_from_tile_max(m_row, m_tile_max, self.fm_fast)

    def apply_l_rescale(self, l_row, rescale):
        return _fmul(l_row, rescale, self.fm_fast)

    def exp2(self, v_s, start, length):
        return _exp2_score_slice(v_s, start, length)

    def reduce_sum(self, l_row, v_p):
        return _fadd(l_row, _score_pair_sum(v_p, self.c_zero_f, self.fm_fast), self.fm_fast)

    def sub_m(self, v_s, row_max):
        return _sub_score_pair(v_s, row_max, self.fm_fast)

    def cast_p(self, v_p):
        return _pack_p_v8_slices(
            self.traits,
            v_p,
            lambda vals: _bf16_trunc_pack_v8(self.traits, vals, elem_dtype=self.elem_dtype),
        )

    def safe_l_inv(self, l_row):
        return _safe_l_inv(l_row, self.c_zero_f)

    def scale_o(self, v_o, scale_scalar):
        _scale_o_accs(v_o, scale_scalar, self.traits, self.fm_fast)

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
            below = fx.Float32(m_diff) <= c_eight_f
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
        return _score_pair_to_lists(v_s)

    def _causal_mask_inplace(self, v_s, tile_idx, q_row_i32=None):
        """Apply causal mask using DUALWAVE_SWP inline-asm attn_mask_vec2_imm."""
        if q_row_i32 is None:
            q_row_i32 = self.ctx_ref.q_row_i32
        traits = self.traits
        s_lo, s_hi = v_s
        kv_tile_start = tile_idx * traits.BLOCK_N
        kv_start_i32 = fx.Int32(kv_tile_start)
        # lane>=32 has a larger n offset in the K-permuted P layout.
        lane_n_off = 8 if traits.KV_VECTORIZED else 4
        lane_off_i32 = fx.Int32(self.lane_div_32) * fx.Int32(lane_n_off)
        rel_lo_i32 = fx.Int32(q_row_i32 + self.delta_i32 - kv_start_i32 - lane_off_i32)
        rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(32))
        neg_inf_i32 = fx.Int32(traits.NEG_INF_F32_BITS)

        pair_thresholds = _causal_pair_thresholds(traits.KV_VECTORIZED)
        _apply_dualwave_causal_mask_pair(s_lo, rel_lo_i32, neg_inf_i32, pair_thresholds)
        _apply_dualwave_causal_mask_pair(s_hi, rel_hi_i32, neg_inf_i32, pair_thresholds)

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

        @flyc.jit
        def _causal_mask_prologue_if_needed(v_s, tile_idx, kv_end_pos, q_start_pos_i32, q_row_i32):
            s_lo, s_hi = v_s
            if q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos):
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self._causal_mask_inplace((lo_list, hi_list), tile_idx, q_row_i32=q_row_i32)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
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
            s_lo[r] = (col_lo < self.seqlen_kv_i32).select(s_lo[r], self.c_neg_inf)
            s_hi[r] = (col_hi < self.seqlen_kv_i32).select(s_hi[r], self.c_neg_inf)

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
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _seq_pad_mask_if_needed(v_s, tile_idx)


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
            idx_lo + self.traits.K_LDS_TO_REG_N_STRIP_STRIDE,
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
                    imm_lo + self.traits.V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE * self.traits.BF16_BYTES,
                    lds_kv_base_idx=self.lds_kv_base_idx,
                    v_lds_read_vec4_type=self.v_lds_read_vec4_type,
                )
                packs[k_substep][dc] = Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
        return packs


class DualwaveStoreHelper(DualwaveKernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _final_o_base(self, q_row):
        return q_row * self.stride_q_n_v + self.q_head_idx * self.traits.HEAD_DIM + self.lane_div_32 * 8

    def _final_o_global(self, o_base, dc, g):
        return o_base + (dc * self.traits.D_CHUNK + 2 * g * 8)

    def _store_final_o_128(self, v_o, dc, g, o_base):
        _buffer_store_128(
            _packed_o_128_vec(self.traits, v_o, dc, g, self.lane_div_32, self.elem_dtype),
            self._final_o_global(o_base, dc, g),
            _o_store_reg_128=self.o_store_reg_128,
            _store_atom_128=self.store_atom_128,
            o_div=self.o_div,
        )

    def _store_final_o_row(self, v_o, q_row):
        o_base = self._final_o_base(q_row)
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for g in range_constexpr(2):
                self._store_final_o_128(v_o, dc, g, o_base)

    def _store_splitk_partial_o_quad(self, v_o, dc, g, local_opart_row_base, opart_rsrc):
        w0, w1, w2, w3 = _packed_o_128_dwords(self.traits, v_o, dc, g, self.lane_div_32, self.elem_dtype)
        _ws_store_quad_i32(
            [w0, w1, w2, w3],
            local_opart_row_base + _splitk_o_partial_dword_col(self.traits, dc, g, self.lane_div_32),
            opart_rsrc,
        )

    def _store_splitk_partial_o_row(self, v_o, local_opart_row_base, opart_rsrc):
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for g in range_constexpr(2):
                self._store_splitk_partial_o_quad(v_o, dc, g, local_opart_row_base, opart_rsrc)

    def zero_o_block_if_needed(self, causal_end_raw_i32=None):
        if causal_end_raw_i32 is None:
            causal_end_raw_i32 = self.causal_end_raw_i32
        traits = self.traits
        q_start = self.q_start
        wave_q_offset = self.wave_q_offset
        lane_mod_32 = self.lane_mod_32
        seq_len_v = self.seq_len_v

        @flyc.jit
        def _zero_o_block_if_needed():
            if causal_end_raw_i32 <= fx.Int32(0):
                q_row_z = q_start + wave_q_offset + lane_mod_32
                c_zero_i = fx.Int32(0)
                zero_pack = Vec.from_elements([c_zero_i, c_zero_i, c_zero_i, c_zero_i], fx.Int32)
                if q_row_z < seq_len_v:
                    o_base_z = self._final_o_base(q_row_z)
                    for dc in range_constexpr(traits.D_CHUNKS):
                        for g in range_constexpr(2):
                            _buffer_store_128(
                                zero_pack,
                                self._final_o_global(o_base_z, dc, g),
                                _o_store_reg_128=self.o_store_reg_128,
                                _store_atom_128=self.store_atom_128,
                                o_div=self.o_div,
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
        self._store_final_o_row(v_o, q_row)

    def store_splitk_partial_o(self, v_o, m_row, l_row, q_row):
        _opart_rsrc, _mrow_rsrc, _lrow_rsrc = self._splitk_workspace_resources()
        local_opart_row_base = _splitk_local_opart_row_base(self.traits, self.q_head_idx, self.seq_len_v, q_row)
        local_ml_idx = _splitk_local_ml_idx(self.q_head_idx, self.seq_len_v, q_row)
        seq_len_v = self.seq_len_v
        lane = self.lane

        @flyc.jit
        def _store_splitk_partial_if_qrow():
            if q_row < seq_len_v:
                self._store_splitk_partial_o_row(v_o, local_opart_row_base, _opart_rsrc)
                if lane < fx.Index(32):
                    _store_splitk_ml_row(m_row, l_row, local_ml_idx, _mrow_rsrc, _lrow_rsrc)

        _store_splitk_partial_if_qrow()


class DualwaveFp8KernelContext:
    """Shared per-kernel state for the gfx950 dualwave fp8 attention helpers.

    Mirrors ``DualwaveKernelContext`` but for the fp8 single path: raw fp8 Q/K/V
    (i8 buffer views), per-tensor Q/K/V descale scalars applied to the fp32 logits,
    and a bf16 ``vt`` LDS scratch for HIPREC PV."""

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
        QDescale=None,
        KDescale=None,
        VDescale=None,
        seq_len=None,
        seq_len_kv=None,
        stride_q_n=None,
        stride_kv_n=None,
        head_dim_runtime=None,
    ):
        if isinstance(traits_or_ctx, DualwaveFp8KernelContext):
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
        self.QDescale = QDescale
        self.KDescale = KDescale
        self.VDescale = VDescale
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.stride_q_n = stride_q_n
        self.stride_kv_n = stride_kv_n
        self.head_dim_runtime = head_dim_runtime

    def init_types_and_constants(self):
        traits = self.traits
        self.elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)
        self.fm_fast = fx.arith.FastMathFlags.fast
        self.v4i32_type = Vec.make_type(4, fx.Int32)
        self.v4f16_type = Vec.make_type(4, self.elem_dtype)
        self.v16f32_type = Vec.make_type(16, fx.Float32)
        self.v2i32_type = Vec.make_type(2, fx.Int32)
        self.p_elem = fx.BFloat16
        self.v4bf16_type = Vec.make_type(4, fx.BFloat16)
        self.NUM_DMA_K = traits.SMEM_D_RPT
        self.NUM_DMA_V = traits.SMEM_D_RPT
        self.c_neg_inf = fx.Float32(float("-inf"))
        self.c_neg_floor = fx.Float32(-3.0e38)
        self.c_zero_f = fx.Float32(0.0)
        self.c_eight_f = fx.Float32(traits.DUALWAVE_SWP_RESCALE_THRESHOLD)
        self.c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)

    def init_runtime_indices(self):
        self.seq_len_v = fx.Index(self.seq_len)
        self.seq_len_kv_v = fx.Index(self.seq_len_kv)
        self.stride_q_n_v = fx.Index(self.stride_q_n)
        self.stride_kv_n_v = fx.Index(self.stride_kv_n)

    def init_lds(self, shared_storage):
        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.kv.ptr))
        self.lds_kv_base_ptr = buffer_ops.create_llvm_ptr(self.lds_kv_base_idx, address_space=3)
        self.lds_vt_base_idx = fx.Index(fx.ptrtoint(lds.vt.ptr))
        self.lds_vt_base_ptr = buffer_ops.create_llvm_ptr(self.lds_vt_base_idx, address_space=3)

    def init_thread_mapping(self):
        _init_dualwave_thread_mapping(self)

    def init_dma_thread_offsets(self):
        # Emitted after descriptors/atoms (matching the original schedule) so the
        # d_bucket ``v_and`` lands at the same ISA position.
        traits = self.traits
        self.lane_in_warp = self.tid % traits.WARP_SIZE
        self.n_in_warp = self.lane_in_warp // traits.LANE_SPLIT_KV
        self.d_bucket = self.lane_in_warp % traits.LANE_SPLIT_KV

    def init_sequence_lengths(self):
        traits = self.traits
        if const_expr(traits.VARLEN):
            _cuq_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(self.CuSeqQ), fx.make_layout(1, 1))
            _cuk_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(self.CuSeqKv), fx.make_layout(1, 1))
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
        self.delta_i32 = fx.Int32(self.seqlen_kv_i32 - fx.Int32(self.seqlen_q_v))
        self.q_gmem_elem_offset = (
            self.q_tok_base + self.q_start
        ) * self.stride_q_n_v + self.q_head_idx * traits.HEAD_DIM
        self.kv_gmem_elem_offset = self.kv_tok_base * self.stride_kv_n_v + self.kv_head_idx * traits.HEAD_DIM

    def init_descriptors(self):
        traits = self.traits
        eb = traits.ELEM_BYTES
        q_nrec_bytes = as_mlir_value(self.q_tok_end * self.stride_q_n_v * eb)
        kv_nrec_bytes = as_mlir_value(self.kv_tok_end * self.stride_kv_n_v * eb)
        o_nrec_bytes = as_mlir_value(self.q_tok_end * self.stride_q_n_v * traits.OUT_ELEM_BYTES)

        def _make_buf_div(tensor, nrec_bytes):
            # fp8 Q/K/V buffer views are i8-typed so DMA and register loads share one
            # byte view.
            bt = fx.rocdl.make_buffer_tensor(tensor, num_records_bytes=nrec_bytes)
            it = fx.get_iter(bt)
            i8_ptr_ty = fx.PointerType.get(
                elem_ty=fx.Int8.ir_type,
                address_space=fx.PointerType(it.type).address_space,
                alignment=fx.PointerType(it.type).alignment,
            )
            bt = fx.Tensor(fx.make_view(fx.recast_iter(i8_ptr_ty, it), fx.get_layout(bt)))
            return fx.logical_divide(bt, fx.make_layout(1, 1))

        self.q_div = _make_buf_div(self.Q, q_nrec_bytes)
        self.k_div = _make_buf_div(self.K, kv_nrec_bytes)
        self.v_div = _make_buf_div(self.V, kv_nrec_bytes)
        self.o_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(self.O, num_records_bytes=o_nrec_bytes), fx.make_layout(1, 1)
        )

    def init_atoms_and_lds_ptrs(self):
        traits = self.traits
        self.load_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.load_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
        self.store_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
        self.store_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.o_store_reg = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
        self.o_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        # fp8 global->LDS DMA uses i8 destination typing; K/V LDS reads are byte-addressed.
        self.lds_ptr_ty = fx.PointerType.get(fx.Int8.ir_type, 2, traits.DMA_BYTES)
        self.bf16_mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.BFloat16))
        self.v_fp8_load64_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)

    def init_descale(self):
        def _load_scale_scalar(tensor):
            _div = fx.logical_divide(fx.rocdl.make_buffer_tensor(tensor), fx.make_layout(1, 1))
            _atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            _v = fly.copy_atom_call_ssa([Vec.make_type(1, fx.Float32)], _atom, fx.slice(_div, (None, fx.Int32(0))))
            return fx.Float32(Vec(_v, (1,), fx.Float32)[0])

        head_dim_f32 = fx.Float32(fx.Int32(self.head_dim_runtime))
        c_log2e_f = fx.Float32(_LOG2E)
        c_sm_scale_log2e = fx.Float32(
            arith.mulf(
                as_mlir_value(fmath.rsqrt(head_dim_f32, fastmath=self.fm_fast)),
                as_mlir_value(c_log2e_f),
                fastmath=self.fm_fast,
            )
        )
        _qd = _load_scale_scalar(self.QDescale)
        _kd = _load_scale_scalar(self.KDescale)
        self.vd_fp8 = _load_scale_scalar(self.VDescale)
        # fp8 feeds raw Q/K into the MFMA, so q/k descale * softmax scale multiplies
        # the fp32 logits after QK.
        self.c_logit_scale = fx.Float32(
            arith.mulf(
                as_mlir_value(c_sm_scale_log2e),
                as_mlir_value(arith.mulf(as_mlir_value(_qd), as_mlir_value(_kd), fastmath=self.fm_fast)),
                fastmath=self.fm_fast,
            )
        )

    def init_tile_bounds(self):
        traits = self.traits
        kv_tile_size = traits.BLOCK_N
        num_kv_tiles = (self.seqlen_kv_v + kv_tile_size - 1) // kv_tile_size
        if const_expr(traits.CAUSAL):
            causal_end_i32 = fx.Int32(self.q_start + traits.BLOCK_M) + self.delta_i32
            causal_end_i32 = fx.Int32((causal_end_i32 > fx.Int32(0)).select(causal_end_i32, fx.Int32(0)))
            causal_num_tiles = (fx.Index(causal_end_i32) + kv_tile_size - 1) // kv_tile_size
            max_num_tiles = fx.Index((causal_num_tiles < num_kv_tiles).select(causal_num_tiles, num_kv_tiles))
        else:
            max_num_tiles = num_kv_tiles
        # Pipeline needs an EVEN tile count >= 4; extra tiles read 0 (num_records) and are masked.
        max_num_tiles = ((max_num_tiles + fx.Index(1)) // fx.Index(2)) * fx.Index(2)
        max_num_tiles = fx.Index((max_num_tiles < fx.Index(4)).select(fx.Index(4), max_num_tiles))
        self.max_num_tiles = max_num_tiles
        if const_expr(traits.SPLITK):
            chunk = ((max_num_tiles + (traits.NUM_KV_SPLITS - 1)) // traits.NUM_KV_SPLITS + 1) // 2 * 2
            chunk = fx.Index((chunk < fx.Index(6)).select(fx.Index(6), chunk))
            split_t0 = self.split_idx * chunk
            split_t_end = split_t0 + chunk
            split_t_end = fx.Index((split_t_end < max_num_tiles).select(split_t_end, max_num_tiles))
            split_t_end = fx.Index((max_num_tiles - split_t_end < fx.Index(4)).select(max_num_tiles, split_t_end))
            self.split_nonempty = split_t0 + fx.Index(4) <= max_num_tiles
        else:
            split_t0 = 0
            split_t_end = max_num_tiles
            self.split_nonempty = None
        self.split_t0 = split_t0
        self.split_t_end = split_t_end

    def init_workspace_io(self):
        if const_expr(self.traits.SPLITK):
            self.ws_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(self.DebugCounts), fx.make_layout(1, 1))
            self.ws_store_atom_32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            self.ws_store_reg_32 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            self.ws_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)

    def ws_store_f32(self, f32_val, elem_index):
        pack = Vec.from_elements([fx.Float32(f32_val)], fx.Float32).bitcast(fx.Int32)
        fx.memref_store_vec(pack, self.ws_store_reg_32)
        fx.copy(self.ws_store_atom_32, self.ws_store_reg_32, fx.slice(self.ws_div, (None, fx.Int32(elem_index))))

    def ws_store_quad_i32(self, dwords, elem_index):
        pack = Vec.from_elements([fx.Int32(v) for v in dwords], fx.Int32)
        fx.memref_store_vec(pack, self.ws_store_reg_128)
        fx.copy(self.store_atom_128, self.ws_store_reg_128, fx.slice(self.ws_div, (None, fx.Int32(elem_index))))

    def init_q_row(self):
        _init_dualwave_q_row(self)

    def k_buf_base(self, buf_id):
        traits = self.traits
        if const_expr(isinstance(buf_id, int)):
            return traits.DUALWAVE_SWP_K_BUF_BASE[buf_id]
        return buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER

    def v_buf_base(self, buf_id):
        traits = self.traits
        if const_expr(isinstance(buf_id, int)):
            return traits.DUALWAVE_SWP_V_BUF_BASE[buf_id]
        return traits.SMEM_K_TILE_ELEMS + buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER

    def v_pair_to_vec32(self, v):
        return _v_pair_to_vec32(v)

    def v_vec32_to_pair(self, v):
        return _v_vec32_to_pair(v)

    def bf16_trunc_pack_v8(self, f32_vals):
        # HIPREC carries P/V as v8 bf16 regardless of the fp8 element dtype: pack
        # 8 f32 -> 4 cvt_pk_bf16 dwords. (The generic _bf16_trunc_pack_v8 branches on
        # DTYPE_STR and would take the fp16 path for fp8, so keep this bf16-only pack.)
        pairs = []
        for j in range_constexpr(4):
            pairs.append(rocdl.cvt_pk_bf16_f32(f32_vals[j * 2], f32_vals[j * 2 + 1]))
        return Vec.from_elements(pairs, fx.Int32).bitcast(fx.BFloat16).ir_value()

    def buffer_load_128(self, elem_index):
        return _buffer_load_128(elem_index, self.load_atom_128, self.q_div, self.v4i32_type)

    def buffer_load_lds_128(self, src_div, lds_byte_addr, src_elem, soffset_elems):
        _buffer_load_lds_128(
            src_div, lds_byte_addr, src_elem, soffset_elems, _dma_atom=self.dma_atom, _lds_ptr_ty=self.lds_ptr_ty
        )

    def buffer_store_128(self, pack_i32_vec, elem_index):
        _buffer_store_128(pack_i32_vec, elem_index, self.o_store_reg_128, self.store_atom_128, self.o_div)

    def global_idx_q(self, token_idx, col):
        return (self.q_tok_base + token_idx) * self.stride_q_n_v + self.q_head_idx * self.traits.HEAD_DIM + col

    def read_i32x8_lds(self, base_ptr, byte_row):
        # Read 32 contiguous fp8 (= 8 i32 words) from `base_ptr` -> one i32x8 operand.
        words = []
        for w in range_constexpr(8):
            p = buffer_ops.get_element_ptr(base_ptr, byte_offset=fx.Int32(byte_row + w * 4), elem_type=T.i8)
            words.append(fx.Int32(llvm.LoadOp(T.i32, p, alignment=1).result))
        return Vec.from_elements(words, fx.Int32).ir_value()


class DualwaveFp8QLoader(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_all_wide(self, q_row_in_block):
        # Wide QK Q reads 32 contiguous fp8 D values per lane and wide step.
        # head_dim=128 gives two K=64 steps, selected by ws and lane//32.
        traits = self.traits
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            q_col = ws * 64 + d_base
            g_idx = self.q_gmem_elem_offset + q_row_in_block * self.stride_q_n_v + q_col
            # 32 contiguous fp8 = 256 bits = two 128-bit loads (i32x4 each) -> i32x8.
            lo = Vec(self.buffer_load_128(g_idx), (4,), fx.Int32)
            hi = Vec(self.buffer_load_128(g_idx + 16), (4,), fx.Int32)
            packs.append(lo.shuffle(hi, [0, 1, 2, 3, 4, 5, 6, 7]).ir_value())
        return packs


class DualwaveFp8GemmHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _mfma_acc_fp8_wide(self, a_i32x8, b_i32x8, c_v16):
        # Wide fp8 QK: mfma_scale (32x32x64) with unit E8M0 scales, i32x8 operands.
        return rocdl.mfma_scale_f32_32x32x64_f8f6f4(
            self.v16f32_type,
            as_mlir_value(a_i32x8),
            as_mlir_value(b_i32x8),
            as_mlir_value(c_v16),
            0,
            0,
            0,
            as_mlir_value(fx.Int32(0x7F7F7F7F)),
            0,
            as_mlir_value(fx.Int32(0x7F7F7F7F)),
        ).result

    def _mfma_acc_bf16(self, a_v8, b_v8, c_v16):
        return fly.mma_atom_call_ssa([self.v16f32_type], self.bf16_mma_atom, a_v8, b_v8, c_v16)

    def qk(self, v_k, q_all_wide):
        # Wide QK: two 32x32x64 MFMAs (one per head-dim half). A=K, B=Q.
        traits = self.traits
        k_lo, k_hi = v_k
        v_s_lo = self.c_zero_v16f32
        v_s_hi = self.c_zero_v16f32
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            q_w = q_all_wide[ws]
            v_s_lo = self._mfma_acc_fp8_wide(k_lo[ws], q_w, v_s_lo)
            v_s_hi = self._mfma_acc_fp8_wide(k_hi[ws], q_w, v_s_hi)
        scale_vec = Vec.from_elements([self.c_logit_scale], fx.Float32).broadcast_to(16)
        v_s_lo = _fmul(Vec(v_s_lo), scale_vec, self.fm_fast)
        v_s_hi = _fmul(Vec(v_s_hi), scale_vec, self.fm_fast)
        return (v_s_lo, v_s_hi)

    def pv_step_k(self, step, v_p, v_v, v_o):
        # HIPREC PV: P and V are both v8 bf16, accumulated by a bf16 MMA.
        v_p_lo, v_p_hi = v_p
        v_pk = v_v[step]
        if const_expr(step < 2):
            p_pk = v_p_lo[step]
        else:
            p_pk = v_p_hi[step - 2]
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_o[dc] = self._mfma_acc_bf16(v_pk[dc], p_pk, v_o[dc])
        return v_o

    def pv(self, v_p, v_v, v_o):
        for step in range_constexpr(4):
            v_o = self.pv_step_k(step, v_p, v_v, v_o)
        return v_o


class DualwaveFp8KvGmemToLdsLoader(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_k(self, tile_start, buf_id):
        traits = self.traits
        eb = traits.ELEM_BYTES
        k_lds_byte_base = self.lds_kv_base_idx + self.k_buf_base(buf_id) * eb
        for d in range_constexpr(self.NUM_DMA_K):
            lds_addr = (
                k_lds_byte_base
                + self.wave_id_uni * (traits.SMEM_K_LINE_STRIDE * eb)
                + (d * traits.SMEM_N_RPT * traits.SMEM_K_LINE_STRIDE * eb)
            )
            n_in_tile = self.n_in_warp * traits.NUM_WAVES + self.wave_id
            global_d = self.d_bucket * traits.VEC_KV + (d * traits.D_128B_SIZE)
            src_elem = self.kv_gmem_elem_offset + n_in_tile * self.stride_kv_n_v + global_d
            self.buffer_load_lds_128(self.k_div, lds_addr, src_elem, tile_start * self.stride_kv_n_v)

    def load_v(self, tile_start, buf_id):
        # HIPREC dequantizes fp8 V into the bf16 vt scratch under the K DMA schedule.
        self._stage_vt_dequant_fp8(tile_start, buf_id)

    def _stage_vt_dequant_fp8(self, tile_start, buf_id):
        # Dequantize fp8 V into the exact bf16 V staging positions. The two d-iters
        # load 8 fp8 at D offsets 64 apart; a contiguous 16B load would gather wrong.
        traits = self.traits
        vt_buf = buf_id * traits.VT_BF16_ELEMS
        n_in_tile = self.n_in_warp * traits.NUM_WAVES + self.wave_id
        for d in range_constexpr(traits.SDRPT_BF):
            global_d = self.d_bucket * traits.VEC_BF + (d * traits.D128_BF)
            src_elem = (
                self.kv_gmem_elem_offset + n_in_tile * self.stride_kv_n_v + global_d + tile_start * self.stride_kv_n_v
            )
            v_i32x2 = fly.copy_atom_call_ssa(
                [self.v2i32_type], self.v_fp8_load64_atom, fx.slice(self.v_div, (None, fx.Int32(src_elem)))
            )
            v_words = Vec(v_i32x2, (2,), fx.Int32)
            bf = []
            for w in range_constexpr(2):
                word = as_mlir_value(fx.Int32(v_words[w]))
                lo2 = Vec(rocdl.cvt_pk_f32_fp8(Vec.make_type(2, fx.Float32), word, False), (2,), fx.Float32)
                hi2 = Vec(rocdl.cvt_pk_f32_fp8(Vec.make_type(2, fx.Float32), word, True), (2,), fx.Float32)
                for e in (lo2[0], lo2[1], hi2[0], hi2[1]):
                    bf.append(fx.Float32(e) * self.vd_fp8)
            v8bf = self.bf16_trunc_pack_v8(bf)
            byte_off = (
                vt_buf
                + self.wave_id_uni * traits.VLS_BF
                + d * traits.SNRPT_BF * traits.VLS_BF
                + self.lane * traits.VEC_BF
            ) * traits.EB_BF
            lds_ptr = buffer_ops.get_element_ptr(self.lds_vt_base_ptr, byte_offset=byte_off, elem_type=T.i8)
            llvm.StoreOp(as_mlir_value(v8bf), lds_ptr, alignment=16)


class DualwaveFp8KvLdsToVgprLoader(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_k(self, buf_id):
        # Read K in the wide 32x32x64 QK operand layout (32 contiguous head-dim/lane,
        # two N-strips, two head-dim halves).
        traits = self.traits
        k_base = self.k_buf_base(buf_id)
        d_base = self.lane_div_32 * 32
        n_lo = self.lane_mod_32
        n_hi = self.lane_mod_32 + 32

        def _read_strip(key):
            row = (key % 8) * traits.SMEM_K_LINE_STRIDE + (key // 8) * traits.D_128B_SIZE
            return [
                self.read_i32x8_lds(self.lds_kv_base_ptr, k_base + row + ws * 64 + d_base)
                for ws in range_constexpr(traits.HEAD_DIM // 64)
            ]

        return (_read_strip(n_lo), _read_strip(n_hi))

    def load_v(self, buf_id):
        # Read all V packs from the bf16 vt scratch for buffer `buf_id`.
        traits = self.traits
        urv = (
            self.lane_div_32 * traits.URV_GRPK_BF
            + ((self.lane % 16) // 4) * traits.URV_LANE_HI_BF
            + ((self.lane // 16) % 2) * traits.URV_GRP_N_BF
            + (self.lane % 4) * traits.URV_LANE_LO_BF
        )
        packs = [[None] * traits.D_CHUNKS for _ in range(4)]
        for dc in range_constexpr(traits.D_CHUNKS):
            dc_off = (dc // 2) * traits.URV_DC_AXIS0_BF + (dc % 2) * traits.URV_DC_AXIS1_BF
            for k_substep in range_constexpr(4):
                imm_lo = (k_substep * traits.URV_STEPK_BF + dc_off) * traits.EB_BF
                byte0 = (urv + buf_id * traits.VT_BF16_ELEMS) * traits.EB_BF + self.lds_vt_base_idx
                a = _ds_read_tr16_b64_imm(self.v4bf16_type, fx.Int32(byte0), imm_lo)
                b = _ds_read_tr16_b64_imm(self.v4bf16_type, fx.Int32(byte0), imm_lo + traits.URV_I5_BF * traits.EB_BF)
                packs[k_substep][dc] = Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
        return packs


class DualwaveFp8SoftmaxHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _attn_mask_vec2_imm(self, rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32):
        return _attn_mask_vec2_imm(rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32)

    def v_s_vec_to_lists(self, v_s):
        return _score_pair_to_lists(v_s)

    def _causal_mask_inplace(self, v_s, tile_idx):
        traits = self.traits
        s_lo, s_hi = v_s
        kv_tile_start = tile_idx * traits.BLOCK_N
        kv_start_i32 = fx.Int32(kv_tile_start)
        lane_off_i32 = fx.Int32(self.lane_div_32) * fx.Int32(4)
        # q_row_i32 is set by init_q_row (called after helper construction), so read
        # it from the live ctx.
        rel_lo_i32 = fx.Int32(self.ctx_ref.q_row_i32 + self.delta_i32 - kv_start_i32 - lane_off_i32)
        rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(32))
        neg_inf_i32 = fx.Int32(traits.NEG_INF_F32_BITS)
        pair_thresholds = _causal_pair_thresholds(False)
        _apply_dualwave_causal_mask_pair(s_lo, rel_lo_i32, neg_inf_i32, pair_thresholds)
        _apply_dualwave_causal_mask_pair(s_hi, rel_hi_i32, neg_inf_i32, pair_thresholds)

    def causal_mask_prologue_if_needed(self, v_s, tile_idx=None, kv_end_pos=None):
        if tile_idx is None:
            tile_idx = fx.Index(0)
        if kv_end_pos is None:
            kv_end_pos = self.traits.BLOCK_N

        @flyc.jit
        def _run(v_s, tile_idx=tile_idx, kv_end_pos=kv_end_pos):
            s_lo, s_hi = v_s
            if self.ctx_ref.q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos):
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self._causal_mask_inplace((lo_list, hi_list), tile_idx)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _run(v_s)

    def _seq_pad_mask_inplace(self, v_s_lists, tile_idx):
        traits = self.traits
        s_lo, s_hi = v_s_lists
        kv_tile_start = tile_idx * traits.BLOCK_N
        col_base = fx.Int32(kv_tile_start) + fx.Int32(self.lane_div_32) * fx.Int32(4)
        for r in range_constexpr(16):
            thr = (r // 4) * 8 + (r % 4)
            col_lo = col_base + fx.Int32(thr)
            col_hi = col_lo + fx.Int32(32)
            s_lo[r] = (col_lo < self.seqlen_kv_i32).select(s_lo[r], self.c_neg_inf)
            s_hi[r] = (col_hi < self.seqlen_kv_i32).select(s_hi[r], self.c_neg_inf)

    def seq_pad_mask_if_needed(self, v_s, tile_idx=None):
        if tile_idx is None:
            tile_idx = fx.Index(0)

        @flyc.jit
        def _run(v_s, tile_idx=tile_idx):
            s_lo, s_hi = v_s
            kv_tile_end = (tile_idx + fx.Index(1)) * self.traits.BLOCK_N
            if fx.Int32(kv_tile_end) > self.seqlen_kv_i32:
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self._seq_pad_mask_inplace((lo_list, hi_list), tile_idx)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _run(v_s)

    def reduce_max(self, v_s):
        return _score_pair_max(v_s, self.c_neg_inf, self.fm_fast)

    def floor_masked_max(self, row_max):
        return _fmax(row_max, self.c_neg_floor, self.fm_fast)

    def sub_m(self, v_s, row_max):
        return _sub_score_pair(v_s, row_max, self.fm_fast)

    def exp2(self, v_s, start, length):
        return _exp2_score_slice(v_s, start, length)

    def tile_sum(self, v_p):
        return _score_pair_sum(v_p, self.c_zero_f, self.fm_fast)

    def reduce_sum(self, l_row, v_p):
        return _fadd(l_row, self.tile_sum(v_p), self.fm_fast)

    def cast_p(self, v_p):
        # Pack the finished softmax probabilities into v8 bf16 P packs for PV.
        return _pack_p_v8_slices(self.traits, v_p, self.bf16_trunc_pack_v8)

    def scale_o(self, v_o, scale_scalar):
        _scale_o_accs(v_o, scale_scalar, self.traits, self.fm_fast)

    def scale_v_p(self, v_p, scale_scalar):
        # P is v8 bf16 (HIPREC): ext to f32, scale, repack bf16.
        p_lo, p_hi = v_p
        out_lo, out_hi = [], []
        for src, dst in ((p_lo, out_lo), (p_hi, out_hi)):
            for pk in src:
                f32 = Vec(llvm.FPExtOp(Vec.make_type(8, fx.Float32), as_mlir_value(pk)).result, (8,), fx.Float32)
                scaled = [fx.Float32(f32[i]) * scale_scalar for i in range(8)]
                dst.append(self.bf16_trunc_pack_v8(scaled))
        return out_lo, out_hi

    def anchor_v_p(self, v_p):
        return _anchor_v_p(self.traits, v_p, elem_dtype=self.p_elem)

    def anchor_v_o(self, v_o):
        return _anchor_v_o(self.traits, v_o)

    def anchor_scalar_f32(self, x):
        return _anchor_scalar_f32(x)

    def safe_l_inv(self, l_row):
        return _safe_l_inv(l_row, self.c_zero_f)

    def rescale_from_tile_max(self, m_row, m_tile_max):
        return _rescale_from_tile_max(m_row, m_tile_max, self.fm_fast)

    def apply_l_rescale(self, l_row, rescale):
        return _fmul(l_row, rescale, self.fm_fast)

    def rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        m_new, corr = self.rescale_from_tile_max(m_row, m_tile_max)
        self.scale_o(v_o, corr)
        v_o = self.anchor_v_o(v_o)
        v_p = self.scale_v_p(v_p, corr)
        l_row = self.apply_l_rescale(l_row, corr)
        return v_o, m_new, l_row, v_p

    def v_p_to_vec32(self, v_p):
        # P packs are (p_lo[0..1], p_hi[0..1]) v8 bf16; concat into one v32 SSA value
        # for the scf.if loop-carry.
        return _v_p_to_vec32(v_p)

    def v_vec32_to_p(self, v_p_all):
        return _v_vec32_to_p(self.traits, v_p_all, elem_dtype=self.p_elem)

    def lazy_rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        @flyc.jit
        def _run(v_o, m_row, l_row, m_tile_max, v_p):
            m_diff = _fsub(m_tile_max, m_row, self.fm_fast)
            below = fx.Float32(m_diff) <= self.c_eight_f
            ballot = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = arith.cmpi(arith.CmpIPredicate.eq, as_mlir_value(ballot), _read_exec_i64())
            all_below = llvm.intr_expect(all_below, arith.constant(1, type=ir.IntegerType.get_signless(1)))

            o0, o1, o2, o3 = (
                as_mlir_value(v_o[0]),
                as_mlir_value(v_o[1]),
                as_mlir_value(v_o[2]),
                as_mlir_value(v_o[3]),
            )
            m_out = as_mlir_value(m_row)
            l_out = as_mlir_value(l_row)
            vp_out = self.v_p_to_vec32(v_p)
            if fx.Boolean(all_below):
                pass
            else:
                corr = rocdl.exp2(T.f32, as_mlir_value(_fsub(m_row, m_tile_max, self.fm_fast)))
                scaled_accs = list(v_o)
                self.scale_o(scaled_accs, corr)
                o0, o1, o2, o3 = (
                    as_mlir_value(scaled_accs[0]),
                    as_mlir_value(scaled_accs[1]),
                    as_mlir_value(scaled_accs[2]),
                    as_mlir_value(scaled_accs[3]),
                )
                vp_out = self.v_p_to_vec32(self.scale_v_p(v_p, corr))
                l_out = as_mlir_value(_fmul(l_row, corr, self.fm_fast))
                m_out = self.anchor_scalar_f32(m_tile_max)
            return ([o0, o1, o2, o3], m_out, l_out, self.v_vec32_to_p(vp_out))

        return _run(v_o, m_row, l_row, m_tile_max, v_p)


class DualwaveFp8StoreHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _o_pack_2dw(self, v_o, dc, store_group):
        r_base = store_group * 4
        lo = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base], Vec(v_o[dc])[r_base + 1])
        hi = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base + 2], Vec(v_o[dc])[r_base + 3])
        return lo, hi

    def _swap_half_partner(self, dw):
        pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
        swapped = rocdl.permlane32_swap(pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False)
        lo_res = llvm.extractvalue(T.i32, swapped, [0])
        hi_res = llvm.extractvalue(T.i32, swapped, [1])
        return (self.lane_div_32 != fx.Index(0)).select(lo_res, hi_res)

    def _packed_o_128_dwords(self, v_o, dc, g):
        is_hi_half = self.lane_div_32 != fx.Index(0)
        d0_a, d1_a = self._o_pack_2dw(v_o, dc, 2 * g)
        d0_b, d1_b = self._o_pack_2dw(v_o, dc, 2 * g + 1)
        y0_a, y1_a = self._swap_half_partner(d0_a), self._swap_half_partner(d1_a)
        y0_b, y1_b = self._swap_half_partner(d0_b), self._swap_half_partner(d1_b)
        w0 = is_hi_half.select(y0_b, as_mlir_value(d0_a))
        w1 = is_hi_half.select(y1_b, as_mlir_value(d1_a))
        w2 = is_hi_half.select(as_mlir_value(d0_b), y0_a)
        w3 = is_hi_half.select(as_mlir_value(d1_b), y1_a)
        return w0, w1, w2, w3

    def _packed_o_128_vec(self, v_o, dc, g):
        return Vec.from_elements([fx.Int32(w) for w in self._packed_o_128_dwords(v_o, dc, g)], fx.Int32)

    def store_final_o(self, v_o, q_row):
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for g in range_constexpr(2):
                o_pack = self._packed_o_128_vec(v_o, dc, g)
                d_col = (dc * self.traits.D_CHUNK) + (2 * g + self.lane_div_32) * 8
                o_global = self.global_idx_q(q_row, d_col)
                self.buffer_store_128(o_pack, o_global)

    def store_splitk_partial_o(self, v_o, m_row, l_row, q_row):
        split_z = self.batch_idx * self.traits.NUM_KV_SPLITS + self.split_idx
        o_part_row_base = ((split_z * self.traits.NUM_HEADS_Q + self.q_head_idx) * self.seq_len_v + q_row) * (
            self.traits.HEAD_DIM // 2
        )
        grid_z = fx.Index(gpu.grid_dim.z)
        mrow_base = grid_z * self.traits.NUM_HEADS_Q * self.seq_len_v * (self.traits.HEAD_DIM // 2)
        lrow_base = mrow_base + grid_z * self.traits.NUM_HEADS_Q * self.seq_len_v
        ml_row_idx = (split_z * self.traits.NUM_HEADS_Q + self.q_head_idx) * self.seq_len_v + q_row

        @flyc.jit
        def _store_splitk_partial_if_qrow():
            if q_row < self.seq_len_v:
                for dc in range_constexpr(self.traits.D_CHUNKS):
                    for g in range_constexpr(2):
                        dw_col = dc * (self.traits.D_CHUNK // 2) + (2 * g + self.lane_div_32) * 4
                        self.ws_store_quad_i32(self._packed_o_128_dwords(v_o, dc, g), o_part_row_base + dw_col)
                if self.lane < fx.Index(32):
                    self.ws_store_f32(m_row, mrow_base + ml_row_idx)
                    self.ws_store_f32(l_row, lrow_base + ml_row_idx)

        _store_splitk_partial_if_qrow()

    def store_empty_split(self):
        @flyc.jit
        def _store_empty_split():
            if self.max_num_tiles < self.split_t0 + fx.Index(4):
                q_row_e = self.q_start + self.wave_q_offset + self.lane_mod_32
                split_z_e = self.batch_idx * self.traits.NUM_KV_SPLITS + self.split_idx
                o_row_base_e = ((split_z_e * self.traits.NUM_HEADS_Q + self.q_head_idx) * self.seq_len_v + q_row_e) * (
                    self.traits.HEAD_DIM // 2
                )
                grid_z_e = fx.Index(gpu.grid_dim.z)
                mrow_base_e = grid_z_e * self.traits.NUM_HEADS_Q * self.seq_len_v * (self.traits.HEAD_DIM // 2)
                lrow_base_e = mrow_base_e + grid_z_e * self.traits.NUM_HEADS_Q * self.seq_len_v
                ml_row_e = (split_z_e * self.traits.NUM_HEADS_Q + self.q_head_idx) * self.seq_len_v + q_row_e
                if q_row_e < self.seq_len_v:
                    c_zero_i = fx.Int32(0)
                    for dc in range_constexpr(self.traits.D_CHUNKS):
                        for g in range_constexpr(2):
                            dw_col = dc * (self.traits.D_CHUNK // 2) + (2 * g + self.lane_div_32) * 4
                            self.ws_store_quad_i32([c_zero_i, c_zero_i, c_zero_i, c_zero_i], o_row_base_e + dw_col)
                    if self.lane < fx.Index(32):
                        self.ws_store_f32(fx.Float32(-1e30), mrow_base_e + ml_row_e)
                        self.ws_store_f32(self.c_zero_f, lrow_base_e + ml_row_e)

        _store_empty_split()


class DualwaveSplitKCombineContext:
    """Shared per-kernel state for the split-K combine pass."""

    def __init__(
        self,
        traits_or_ctx,
        O=None,  # noqa: E741
        WS=None,
        batch_size=None,
        seq_len=None,
        stride_q_n=None,
    ):
        if isinstance(traits_or_ctx, DualwaveSplitKCombineContext):
            self.__dict__.update(traits_or_ctx.__dict__)
            self.ctx_ref = getattr(traits_or_ctx, "ctx_ref", traits_or_ctx)
            return

        self.ctx_ref = self
        self.traits = traits_or_ctx
        self.O = O
        self.WS = WS
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.stride_q_n = stride_q_n

    def init_types_and_constants(self):
        self.elem_dtype = dtype_to_elem_type(self.traits.DTYPE_STR)
        self.fm_fast = fx.arith.FastMathFlags.fast
        self.c_zero_f = fx.Float32(0.0)
        self.c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)

    def init_runtime_indices(self):
        self.seq_len_v = fx.Index(self.seq_len)
        self.stride_q_n_v = fx.Index(self.stride_q_n)
        self.batch_size_v = fx.Index(self.batch_size)

    def init_thread_mapping(self, combine_rows_per_block, combine_lanes_per_row):
        traits = self.traits
        self.tid = fx.Index(gpu.thread_idx.x)
        self.blk = fx.Index(gpu.block_idx.x)
        self.row = self.blk * combine_rows_per_block + self.tid // combine_lanes_per_row
        self.col = (self.tid % combine_lanes_per_row) * 4
        heads_per_batch = self.seq_len_v * traits.NUM_HEADS_Q
        self.batch_idx = self.row // heads_per_batch
        rem = self.row % heads_per_batch
        self.q_head_idx = rem // self.seq_len_v
        self.seq_idx = rem % self.seq_len_v

    def init_workspace(self):
        traits = self.traits
        z_total = self.batch_size_v * traits.NUM_KV_SPLITS
        self.ws_opart_per_split_elems = fx.Index(traits.NUM_HEADS_Q) * self.seq_len_v * fx.Index(traits.HEAD_DIM // 2)
        self.ws_ml_per_split_elems = fx.Index(traits.NUM_HEADS_Q) * self.seq_len_v
        self.ws_opart_per_split_bytes = self.ws_opart_per_split_elems * fx.Index(4)
        self.ws_ml_per_split_bytes = self.ws_ml_per_split_elems * fx.Index(4)
        self.ws_mrow_abs_bytes = z_total * self.ws_opart_per_split_bytes
        self.ws_lrow_abs_bytes = self.ws_mrow_abs_bytes + z_total * self.ws_ml_per_split_bytes
        self.local_ml_idx = self.q_head_idx * self.seq_len_v + self.seq_idx
        self.local_o_base = (self.q_head_idx * self.seq_len_v + self.seq_idx) * fx.Index(traits.HEAD_DIM // 2)
        self.ws_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(self.WS)))

    def init_descriptors(self):
        per_batch_elems = self.seq_len_v * self.stride_q_n_v
        batch_byte_off = self.batch_idx * per_batch_elems * fx.Index(2)
        self.o_rsrc = buffer_ops.create_buffer_resource_from_addr(
            as_mlir_value(fx.Int64(fx.ptrtoint(fx.get_iter(self.O))) + fx.Int64(batch_byte_off)),
            num_records_bytes=as_mlir_value(fx.Int64(per_batch_elems * fx.Index(2))),
        )
        self.load_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)

    def workspace_resource(self, byte_offset, nrec_bytes):
        return _make_ws_rsrc(self.ws_base_i64, byte_offset, nrec_bytes)

    def split_z(self, split_i):
        return self.batch_idx * self.traits.NUM_KV_SPLITS + split_i

    def opart_resource(self, split_z):
        return self.workspace_resource(split_z * self.ws_opart_per_split_bytes, self.ws_opart_per_split_bytes)

    def mrow_resource(self, split_z):
        return self.workspace_resource(
            self.ws_mrow_abs_bytes + split_z * self.ws_ml_per_split_bytes,
            self.ws_ml_per_split_bytes,
        )

    def lrow_resource(self, split_z):
        return self.workspace_resource(
            self.ws_lrow_abs_bytes + split_z * self.ws_ml_per_split_bytes,
            self.ws_ml_per_split_bytes,
        )


class DualwaveSplitKCombineHelper(DualwaveSplitKCombineContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_ml_rows(self):
        m_s = []
        l_s = []
        for i in range_constexpr(self.traits.NUM_KV_SPLITS):
            split_z_i = self.split_z(i)
            m_f32 = buffer_ops.buffer_load(
                self.mrow_resource(split_z_i),
                as_mlir_value(fx.Int32(self.local_ml_idx)),
                vec_width=1,
                dtype=T.f32,
            )
            l_f32 = buffer_ops.buffer_load(
                self.lrow_resource(split_z_i),
                as_mlir_value(fx.Int32(self.local_ml_idx)),
                vec_width=1,
                dtype=T.f32,
            )
            m_s.append(m_f32)
            l_s.append(l_f32)
        return m_s, l_s

    def reduce_m_max(self, m_s):
        m_max = m_s[0]
        for i in range_constexpr(self.traits.NUM_KV_SPLITS - 1):
            m_max = _fmax(m_max, m_s[i + 1], self.fm_fast)
        return m_max

    def init_accumulators(self):
        return as_mlir_value(self.c_zero_v4f32), as_mlir_value(self.c_zero_f)

    def accumulate_split(self, acc, den, split_i, m_i, l_i, m_max):
        orsrc_i = self.opart_resource(self.split_z(split_i))
        local_o_idx_i = self.local_o_base + self.col // 2

        @flyc.jit
        def _accum_split(acc, den):
            if fx.Float32(l_i) > fx.Float32(0.0):
                w = rocdl.exp2(T.f32, as_mlir_value(_fsub(m_i, m_max, self.fm_fast)))
                wl = _fmul(w, l_i, self.fm_fast)
                den = _fadd(den, wl, self.fm_fast)
                o2_raw = buffer_ops.buffer_load(
                    orsrc_i,
                    as_mlir_value(fx.Int32(local_o_idx_i)),
                    vec_width=2,
                    dtype=T.i32,
                )
                o2_i32 = ir.Value(o2_raw)
                o4 = Vec(o2_i32, (2,), fx.Int32).bitcast(self.elem_dtype).to(fx.Float32)
                w4 = Vec.from_elements([fx.Float32(wl)], fx.Float32).broadcast_to(4)
                acc = _fadd(acc, _fmul(w4, o4, self.fm_fast), self.fm_fast)
            return acc, den

        return _accum_split(acc, den)

    def accumulate_splits(self, m_s, l_s, m_max):
        acc, den = self.init_accumulators()
        for i in range_constexpr(self.traits.NUM_KV_SPLITS):
            acc, den = self.accumulate_split(acc, den, i, m_s[i], l_s[i], m_max)
        return acc, den

    def pack_output(self, acc, den):
        inv_rcp = rocdl.rcp(T.f32, den)
        inv = (fx.Float32(den) > self.c_zero_f).select(inv_rcp, self.c_zero_f)
        inv4 = Vec.from_elements([fx.Float32(inv)], fx.Float32).broadcast_to(4)
        out4 = Vec(_fmul(acc, inv4, self.fm_fast), (4,), fx.Float32)
        if const_expr(self.traits.DTYPE_STR == "bf16"):
            lo = rocdl.cvt_pk_bf16_f32(out4[0], out4[1])
            hi = rocdl.cvt_pk_bf16_f32(out4[2], out4[3])
        else:
            o_f16 = []
            for i in range_constexpr(4):
                o_f16.append(fx.Float32(out4[i]).to(self.elem_dtype))
            pack = Vec.from_elements(o_f16, self.elem_dtype).bitcast(fx.Int32)
            lo, hi = as_mlir_value(pack[0]), as_mlir_value(pack[1])
        return Vec.from_elements([fx.Int32(lo), fx.Int32(hi)], fx.Int32)

    def store_output(self, o_pack):
        o_global = self.seq_idx * self.stride_q_n_v + self.q_head_idx * self.traits.HEAD_DIM + self.col
        buffer_ops.buffer_store(
            o_pack.ir_value(),
            self.o_rsrc,
            as_mlir_value(fx.Int32(o_global * fx.Index(2))),
            offset_is_bytes=True,
        )


# Misc debug and public helpers


def _scale_sched_pairs(pairs, head_dim):
    return max(1, (pairs + 1) // 2) if head_dim == 64 else pairs


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


def _stagger_extra_barrier_if_zero(stagger_i32):
    """Emit `s_barrier;` only when stagger == 0."""
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [stagger_i32],
        ("s_cmp_eq_u32 $0, 0\n\ts_cbranch_scc0 1f\n\ts_barrier\n\t1:"),
        "s",
        has_side_effects=True,
    )


@flyc.jit
def _stagger_extra_barrier_if_one(stagger_i32):
    """Emit `sched_barrier(0); s_barrier;` only when stagger == 1."""
    if fx.Int32(stagger_i32) != fx.Int32(0):
        rocdl.sched_barrier(0)
        rocdl.s_barrier()


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


def dualwave_splitk_workspace_elems(batch_size, num_heads, seq_len, num_kv_splits, head_dim=128):
    """fp32 elements needed for the split-K workspace: O_partial + Mrow + Lrow.

    O_partial is stored as kernel-native 16-bit (bf16/fp16), two columns per
    fp32 slot; Mrow/Lrow stay fp32.
    """
    rows = batch_size * num_kv_splits * num_heads * seq_len
    return rows * (head_dim // 2) + 2 * rows
