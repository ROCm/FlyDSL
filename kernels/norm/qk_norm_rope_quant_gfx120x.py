# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x RMSNorm, RoPE on the tail, then fp8 or int8 quantization.

bf16 and fp16 stay on ``rms_rope_gfx120x``. This kernel is the quant path.
One scale covers each row. One wave holds the row in registers and reduces
there. The first ``head_dim - rope_head_dim`` elements are not rotated.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch


def build_qk_norm_rope_quant_module(
    head_dim: int,
    rope_head_dim: int,
    quant: str,
    eps: float = 1e-6,
) -> Callable[..., None]:
    """Build one-row RMSNorm + tail RoPE + per-row quant."""
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"build_qk_norm_rope_quant_module requires gfx120x, got {arch}"
    if quant not in ("fp8_e4m3fn", "fp8_e5m2", "int8"):
        raise ValueError(f"quant dtype {quant!r} is not fp8 or int8")
    if head_dim < 1:
        raise ValueError(f"head_dim {head_dim} must be positive")
    if rope_head_dim < 0 or rope_head_dim > head_dim:
        raise ValueError(f"rope_head_dim {rope_head_dim} must be in 0..{head_dim}")
    # One wave. Each thread keeps a strided set of pairs in registers.
    wave = 32
    n_pairs = (int(head_dim) + 1) // 2
    pairs_per = (n_pairs + wave - 1) // wave
    nope = head_dim - rope_head_dim
    qmax = {"fp8_e4m3fn": 448.0, "fp8_e5m2": 57344.0, "int8": 127.0}[quant]
    is_int8 = quant == "int8"
    is_e5m2 = quant == "fp8_e5m2"

    @flyc.kernel(known_block_size=[wave, 1, 1])
    def qk_norm_rope_quant_kernel(
        X: fx.Pointer,
        Weight: fx.Pointer,
        Cos: fx.Pointer,
        Sin: fx.Pointer,
        Positions: fx.Pointer,
        Out: fx.Pointer,
        Scale: fx.Pointer,
        row_stride: fx.Int32,
    ) -> None:
        x_ptr = fx.recast_iter(fx.PointerType.get(fx.BFloat16.ir_type, X.address_space), X)
        w_ptr = fx.recast_iter(fx.PointerType.get(fx.BFloat16.ir_type, Weight.address_space), Weight)
        cos_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, Cos.address_space), Cos)
        sin_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, Sin.address_space), Sin)
        pos_ptr = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, Positions.address_space), Positions)
        sc_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, Scale.address_space), Scale)
        row = fx.Int32(gpu.block_idx.x)
        tid = fx.Int32(gpu.thread_idx.x)
        base = row * fx.Int32(row_stride)
        pos = fx.Int32(fx.ptr_load(pos_ptr + row))
        xs0 = []
        xs1 = []
        ws0 = []
        ws1 = []
        es0 = []
        es1 = []
        ok0 = []
        ok1 = []
        reduced = fx.Float32(0.0)
        for p in range_constexpr(pairs_per):
            pair = fx.Int32(p) * fx.Int32(wave) + tid
            e0 = pair * fx.Int32(2)
            e1 = e0 + fx.Int32(1)
            valid0 = e0 < fx.Int32(head_dim)
            valid1 = e1 < fx.Int32(head_dim)
            i0 = valid0.select(e0, fx.Int32(0))
            i1 = valid1.select(e1, fx.Int32(0))
            x0 = valid0.select(fx.Float32(fx.ptr_load(x_ptr + base + i0)), fx.Float32(0.0))
            x1 = valid1.select(fx.Float32(fx.ptr_load(x_ptr + base + i1)), fx.Float32(0.0))
            w0 = valid0.select(fx.Float32(fx.ptr_load(w_ptr + i0)), fx.Float32(0.0))
            w1 = valid1.select(fx.Float32(fx.ptr_load(w_ptr + i1)), fx.Float32(0.0))
            reduced = reduced + x0 * x0 + x1 * x1
            xs0.append(x0)
            xs1.append(x1)
            ws0.append(w0)
            ws1.append(w1)
            es0.append(e0)
            es1.append(e1)
            ok0.append(valid0)
            ok1.append(valid1)
        for step in range_constexpr(5):
            reduced = reduced + gpu.shuffle_xor(reduced, fx.Int32(1 << step), fx.Int32(wave))
        # Hardware rsqrt. The precise sequence uses vcc, and a later compare
        # was scheduled between its divide steps.
        mean_sq = reduced * fx.Float32(1.0 / float(head_dim)) + fx.Float32(eps)
        inv = fmath.rsqrt(mean_sq, fastmath="fast")
        amax = fx.Float32(0.0)
        ys0 = []
        ys1 = []
        for p in range_constexpr(pairs_per):
            y0 = xs0[p] * inv * ws0[p]
            y1 = xs1[p] * inv * ws1[p]
            if const_expr(rope_head_dim > 0):
                e0 = es0[p]
                e1 = es1[p]
                in0 = e0 >= fx.Int32(nope)
                in1 = e1 >= fx.Int32(nope)
                safe = in0.select(
                    (e0 - fx.Int32(nope)) // fx.Int32(2),
                    in1.select((e1 - fx.Int32(nope)) // fx.Int32(2), fx.Int32(0)),
                )
                cos = fx.Float32(fx.ptr_load(cos_ptr + pos * fx.Int32((rope_head_dim + 1) // 2) + safe))
                sin = fx.Float32(fx.ptr_load(sin_ptr + pos * fx.Int32((rope_head_dim + 1) // 2) + safe))
                ry0 = y0 * cos - y1 * sin
                ry1 = y0 * sin + y1 * cos
                y0 = in0.select(ry0, y0)
                y1 = in1.select(ry1, y1)
            amax = fx.max(amax, fx.max(fx.absf(y0), fx.absf(y1)))
            ys0.append(y0)
            ys1.append(y1)
        for step in range_constexpr(5):
            amax = fx.max(amax, gpu.shuffle_xor(amax, fx.Int32(1 << step), fx.Int32(wave)))
        scale = fx.max(amax, fx.Float32(1e-12)) * fx.Float32(1.0 / qmax)
        if tid == fx.Int32(0):
            fx.ptr_store(scale, sc_ptr + row)
        # One hardware reciprocal. Two software divides share vcc and corrupt
        # the second quotient.
        inv_s = fx.Float32(fx.rocdl.rcp(T.f32, scale))
        o8 = fx.recast_iter(fx.PointerType.get(fx.Int8.ir_type, Out.address_space), Out)
        for p in range_constexpr(pairs_per):
            s0 = ys0[p] * inv_s
            s1 = ys1[p] * inv_s
            e0 = es0[p]
            e1 = es1[p]
            if const_expr(is_int8):
                q0 = fmath.roundeven(s0).to(fx.Int32)
                q1 = fmath.roundeven(s1).to(fx.Int32)
                q0 = fx.max(fx.min(q0, fx.Int32(127)), fx.Int32(-128))
                q1 = fx.max(fx.min(q1, fx.Int32(127)), fx.Int32(-128))
                b0 = fx.Int8(q0)
                b1 = fx.Int8(q1)
            else:
                # A direct f8 store does not legalize. Pack two values and keep the low bytes.
                if const_expr(is_e5m2):
                    packed = fx.rocdl.cvt_pk_bf8_f32(T.i32, s0, s1, fx.Int32(0), False)
                else:
                    packed = fx.rocdl.cvt_pk_fp8_f32(T.i32, s0, s1, fx.Int32(0), False)
                bits = fx.Int32(packed)
                b0 = fx.Int8(bits)
                b1 = fx.Int8(bits >> fx.Int32(8))
            if ok0[p]:
                fx.ptr_store(b0, o8 + row * fx.Int32(head_dim) + e0)
            if ok1[p]:
                fx.ptr_store(b1, o8 + row * fx.Int32(head_dim) + e1)

    @flyc.jit
    def launch(
        X: fx.Pointer,
        Weight: fx.Pointer,
        Cos: fx.Pointer,
        Sin: fx.Pointer,
        Positions: fx.Pointer,
        Out: fx.Pointer,
        Scale: fx.Pointer,
        rows: fx.Int32,
        row_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        qk_norm_rope_quant_kernel(X, Weight, Cos, Sin, Positions, Out, Scale, row_stride).launch(
            grid=(fx.Uint64(rows), 1, 1),
            block=(wave, 1, 1),
            stream=stream,
        )

    return launch


def qk_norm_rope_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    quant: str,
    eps: float = 1e-6,
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize ``x`` ``[rows, head_dim]``, rotate its tail, and quantize the row."""
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"qk_norm_rope_quant requires gfx120x, got {arch}"
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("qk_norm_rope_quant expects bf16 activations and weight")
    if x.dim() != 2 or int(x.shape[-1]) != int(weight.numel()):
        raise ValueError(f"x {tuple(x.shape)} does not match weight {tuple(weight.shape)}")
    rows, head_dim = (int(v) for v in x.shape)
    rope_dim = int(cos.shape[-1]) * 2
    out_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2, "int8": torch.int8}[quant]
    out = torch.empty(rows, head_dim, device=x.device, dtype=out_dtype)
    scale = torch.empty(rows, device=x.device, dtype=torch.float32)
    launch = build_qk_norm_rope_quant_module(head_dim, rope_dim, quant, eps)
    launch_stream = torch.cuda.current_stream(x.device) if stream is None else stream
    out_ty = {"fp8_e4m3fn": fx.Float8E4M3FN, "fp8_e5m2": fx.Float8E5M2, "int8": fx.Int8}[quant]
    pos = positions.to(torch.int32).contiguous()
    launch(
        flyc.from_c_void_p(fx.BFloat16, x.data_ptr()),
        flyc.from_c_void_p(fx.BFloat16, weight.data_ptr()),
        flyc.from_c_void_p(fx.Float32, cos.data_ptr()),
        flyc.from_c_void_p(fx.Float32, sin.data_ptr()),
        flyc.from_c_void_p(fx.Int32, pos.data_ptr()),
        flyc.from_c_void_p(out_ty, out.data_ptr()),
        flyc.from_c_void_p(fx.Float32, scale.data_ptr()),
        rows,
        int(x.stride(0)),
        launch_stream,
    )
    return out, scale
