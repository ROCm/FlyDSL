# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Self-contained helpers for the a16w4/a16wi4/a16w16 fused MoE kernels."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.typing import T
from kernels.common import buffer_ops

_PTR3 = "!llvm.ptr<3>"
LOG2E = 1.4426950408889634


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _udiv(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.divui(_raw(a), _raw(cc)))


def _umod(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.remui(_raw(a), _raw(cc)))


def _global_i32_buffer_view(addr_i64, num_bytes):
    # fx.copy's BufferCopy/BufferCopyLDS atoms take soffset as an element count, not
    # the bytes buffer_ops.buffer_load's soffset_bytes expected.
    # make_layout's dynamic-shape leaf must be i32/i64, not fx.Index.
    num_bytes_i64 = fx.Int64(num_bytes)
    ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
    ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
    view = fx.Tensor(fx.make_view(ptr, fx.make_layout(num_bytes_i64 // fx.Int64(4), 1)))
    return fx.rocdl.make_buffer_tensor(view, max_size=False, num_records_bytes=num_bytes_i64)


def _global_i32_buffer_tiles(addr_i64, num_bytes, tile_elems):
    return fx.logical_divide(_global_i32_buffer_view(addr_i64, num_bytes), fx.make_layout(tile_elems, 1))


def _lds_ptr3(base_i32, byte_off_i32):
    addr_i64 = fx.Int64(base_i32 + byte_off_i32)
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(addr_i64))


def _gep3(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8)


def _global_base_ptr1(addr_i64):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr_i64)))


def _gep1(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8)


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _global_i32_at(addr_i64, idx):
    return _global_i32_ptr(addr_i64)[idx]


def _buffer_rsrc(addr_i64, num_records_bytes):
    return buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(addr_i64)), num_records_bytes=num_records_bytes)


def _silu_mul_batch(gs, us):
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def _sigmoid_f32(g):
    e = fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E))))
    return fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + e)))


def _tanh_f32(x):
    # tanh(x) via exp2/rcp, sign-restored (mirrors aiter mixed_moe tanh_elem):
    #   t = (1 - exp(-2|x|)) / (1 + exp(-2|x|)),  tanh(x) = sign(x) * t
    neg_two_log2e = fx.Float32(-2.0 * LOG2E)
    abs_x = x.maximumf(-x)
    e = fx.Float32(rocdl.exp2(T.f32, _raw(abs_x * neg_two_log2e)))
    recip = fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + e)))
    tanh_abs = (fx.Float32(1.0) - e) * recip
    is_pos = arith.cmpf(arith.CmpFPredicate.OGT, _raw(x), _raw(fx.Float32(0.0)))
    return fx.Float32(arith.select(is_pos, _raw(tanh_abs), _raw(-tanh_abs)))


def _situ_mul_batch(gs, us, situ_beta=1.0, situ_linear_beta=1.0, clamp_limit=7.0):
    """SiTUv2 activation (aiter mixed_moe situ_mul_vec4):
        situ(g)    = beta * tanh(g / beta) * sigmoid(g)
        situ_up(u) = linear_beta * tanh(u / linear_beta)
        out        = situ(clamp_gate(g)) * situ_up(clamp_lin(u))
    clamp_gate: g <= +limit (upper only); clamp_lin: u in [-limit, +limit].
    """
    neg_lim = fx.Float32(-clamp_limit)
    beta = fx.Float32(situ_beta)
    beta_rcp = fx.Float32(1.0 / situ_beta)
    lbeta = fx.Float32(situ_linear_beta)
    lbeta_rcp = fx.Float32(1.0 / situ_linear_beta)

    out = []
    for i in range(len(gs)):
        # clamp_gate: g <= +lim  (min(g, lim) == -max(-g, -lim), upper bound only).
        g = -((-gs[i]).maximumf(neg_lim))
        # clamp_lin: u in [-lim, +lim].
        u = (-((-us[i]).maximumf(neg_lim))).maximumf(neg_lim)
        situ_g = beta * _tanh_f32(g * beta_rcp) * _sigmoid_f32(g)
        situ_u = lbeta * _tanh_f32(u * lbeta_rcp)
        out.append(situ_g * situ_u)
    return out


def _int4_nibble_to_bf16x8(raw_i32, scale_f32):
    """int4 (signed) -> bf16 upconvert for one MFMA K32 step (8 nibbles -> v8bf16).

    ``raw_i32`` holds 8 packed signed-int4 nibbles in ``bits[4n+3:4n]`` order (the
    SAME K ordering the mxfp4 path uses via ``cvt_scalef32_pk_bf16_fp4`` sel 0..3).
    Each nibble uses the gfx950 ``v_cvt_off_f32_i4`` fast path: it reads the nibble
    as unsigned [0,15], subtracts 8 -> signed [-8,7], and multiplies the mantissa by
    16, so the ×16 correction is folded into the effective per-group scale
    (``eff = scale * 16``). ``scale_f32`` is the per-group dequant scale.
    """
    eff = fx.Float32(scale_f32 * fx.Float32(16.0))
    # Shift each nibble n (bits [4n:4n+3]) into the low nibble; v_cvt_off_f32_i4 reads bits[3:0].
    bf16s = []
    for n in range_constexpr(8):
        shifted = _raw(fx.Int32(raw_i32).shrui(fx.Int32(4 * n))) if n > 0 else _raw(raw_i32)
        v = fx.Float32(rocdl.cvt_off_f32_i4(shifted)) * eff
        bf16s.append(v.to(fx.BFloat16))
    return fx.Vector.from_elements(bf16s, fx.BFloat16)  # v8bf16


def kmchunks_for(BM):
    return BM // 16


def lds_acc_bytes_for(rows, BN):
    return rows * BN * 4
