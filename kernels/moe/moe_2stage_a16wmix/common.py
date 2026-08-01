# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Self-contained helpers for the a16w4/a16wi4/a16w16 fused MoE kernels."""

import os

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from kernels.common import buffer_ops

_PTR3 = "!llvm.ptr<3>"
LOG2E = 1.4426950408889634


def a16wmix_use_k16(arch=None):
    """Decide the a16wmix MFMA K-size + int4 dequant path (compile-time, host side).

    gfx950 (CDNA4) has the K=32 ``mfma_f32_16x16x32_bf16`` and the VALU-lean
    ``v_cvt_pk_bf16_f32`` int4 dequant pack; gfx942 (CDNA3) has NEITHER -- it only has
    the K=16 ``mfma_f32_16x16x16bf16_1k`` and must fall back to the scalar-trunc int4
    dequant. This returns True for the gfx942 (K=16 + scalar dequant) codepath.

    ``FLYDSL_A16WMIX_FORCE_K16=1`` forces the gfx942 codepath regardless of the
    detected arch, so the K=16 path can be validated on a gfx950 box (which runs the
    K=16 MFMA + scalar dequant fine -- it is a strict subset of its ISA).
    """
    if os.environ.get("FLYDSL_A16WMIX_FORCE_K16", "0") not in ("0", "", "false", "False"):
        return True
    if arch is None:
        arch = get_rocm_arch() or ""
    return "gfx95" not in str(arch)


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


def _cvt_pk_bf16_f32_se(src_a_f32, src_b_f32):
    """Side-effecting v_cvt_pk_bf16_f32 (pack 2 f32 -> 2xbf16 in i32).

    The stateless ``rocdl.cvt_pk_bf16_f32`` (has_side_effects=False) miscompiles in
    the a16wi4 gemm1 hot loop (garbage output) — the 4 identical-shaped packed
    converts per K-step get CSE-merged / reordered across K iterations. Marking the
    inline asm side-effecting pins each call to its K-step.
    """
    return llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [_raw(src_a_f32), _raw(src_b_f32)],
        "v_cvt_pk_bf16_f32 $0, $1, $2",
        "=v,v,v",
        has_side_effects=True,
    )


def _int4_nibble_to_bf16x8(raw_i32, scale_f32, *, use_k16=False):
    """int4 (signed) -> bf16 upconvert for one MFMA K32 step (8 nibbles -> v8bf16).

    ``raw_i32`` holds 8 packed signed-int4 nibbles in ``bits[4n+3:4n]`` order (the
    SAME K ordering the mxfp4 path uses via ``cvt_scalef32_pk_bf16_fp4`` sel 0..3).
    Each nibble uses the gfx9xx ``v_cvt_off_f32_i4`` fast path: it reads the nibble
    as unsigned [0,15], subtracts 8 -> signed [-8,7], and multiplies the mantissa by
    16, so the ×16 correction is folded into the effective per-group scale
    (``eff = scale * 16``). ``scale_f32`` is the per-group dequant scale.

    ``use_k16`` (gfx942/CDNA3): the VALU-lean ``v_cvt_pk_bf16_f32`` pack is gfx950-only,
    so fall back to a scalar ``.to(BFloat16)`` truncation per nibble (arch-agnostic).
    """
    eff = fx.Float32(scale_f32 * fx.Float32(16.0))
    raw_even = fx.Int32(raw_i32)
    raw_odd = raw_even.shrui(fx.Int32(4))
    if use_k16:
        # gfx942 fallback: scalar f32 -> bf16 truncation (no v_cvt_pk_bf16_f32).
        bf16s = []
        for j in range_constexpr(4):
            f_lo = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_even), byte_sel=j)) * eff
            f_hi = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_odd), byte_sel=j)) * eff
            bf16s.append(f_lo.to(fx.BFloat16))
            bf16s.append(f_hi.to(fx.BFloat16))
        return fx.Vector.from_elements([_raw(x) for x in bf16s], fx.BFloat16)  # v8bf16
    # byte_sel loads (1 shift total instead of 7); side-effecting pk-convert.
    i32s = []
    for j in range_constexpr(4):
        f_lo = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_even), byte_sel=j)) * eff
        f_hi = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_odd), byte_sel=j)) * eff
        i32s.append(fx.Int32(_cvt_pk_bf16_f32_se(_raw(f_lo), _raw(f_hi))))
    v4i32 = fx.Vector.from_elements([_raw(x) for x in i32s], fx.Int32)
    return v4i32.bitcast(fx.BFloat16)  # v8bf16


def kmchunks_for(BM):
    return BM // 16


def lds_acc_bytes_for(rows, BN):
    return rows * BN * 4
