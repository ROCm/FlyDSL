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
