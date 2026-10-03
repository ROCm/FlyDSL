# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""SiLU(gate)*up and chunk-2 SwiGLU for gfx120x (RDNA4).

Two host builders:

* ``build_silu_mul_module`` -- flat elementwise ``silu(gate) * up``
* ``build_swiglu_chunk_module`` -- last-dim chunk-2 SwiGLU (gate/up halves)

Plain SiLU*mul (no magnitude clamp). Distinct from interleaved/clamped SwiGLU
kernels used elsewhere. For the zero-LDS in-register MLP fuse, see
``kernels.gemm.rdna4_fused_mlp_nmajor``.
"""

from collections.abc import Callable
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, kernel_signature, ptr_buf_tensor

KERNEL_NAME = "swiglu_gfx120x"
BLOCK = 256
VEC_HALF = 8
VEC_F32 = 4


@lru_cache(maxsize=8)
def build_silu_mul_module(dtype: str = "bfloat16", vec: int | None = None) -> Callable[..., None]:
    """Flat elementwise silu(gate)*up."""
    ElemTy = {"float32": fx.Float32, "float16": fx.Float16, "bfloat16": fx.BFloat16}[dtype]
    elem_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[dtype]
    if vec is None:
        vec = VEC_F32 if dtype == "float32" else VEC_HALF
    tile = BLOCK * vec
    is_f32 = dtype == "float32"
    sig = kernel_signature(block=BLOCK, dtype=dtype, vec=vec, op="silu_mul")

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def silu_mul_kernel(Gate: fx.Pointer, Up: fx.Pointer, Out: fx.Pointer, n_elems: fx.Int32) -> None:
        bid = gpu.block_id("x")
        tid = gpu.thread_id("x")
        unit = fx.Int32(bid) * fx.Int32(BLOCK) + fx.Int32(tid)
        g_buf = ptr_buf_tensor(
            Gate,
            elem=ElemTy,
            n=0x3FFFFFFF,
            unit_elems=vec,
            num_records_bytes=fx.Int64(n_elems) * fx.Int64(elem_bytes),
        )
        u_buf = ptr_buf_tensor(
            Up,
            elem=ElemTy,
            n=0x3FFFFFFF,
            unit_elems=vec,
            num_records_bytes=fx.Int64(n_elems) * fx.Int64(elem_bytes),
        )
        o_buf = ptr_buf_tensor(
            Out,
            elem=ElemTy,
            n=0x3FFFFFFF,
            unit_elems=vec,
            num_records_bytes=fx.Int64(n_elems) * fx.Int64(elem_bytes),
        )
        gv = buf_copy_load(g_buf, unit, elem=ElemTy, unit_elems=vec)
        uv = buf_copy_load(u_buf, unit, elem=ElemTy, unit_elems=vec)
        if not is_f32:
            gv = gv.extf(T.vec(vec, T.f32))
            uv = uv.extf(T.vec(vec, T.f32))
        one = fx.Float32(1.0)
        neg_log2e = fx.Float32(-1.4426950408889634)
        outs = []
        for i in range_constexpr(vec):
            g = fx.Float32(gv[i])
            u = fx.Float32(uv[i])
            sigv = one / (one + fmath.exp2(g * neg_log2e))
            y = g * sigv * u
            outs.append(y if is_f32 else y.to(ElemTy))
        buf_copy_store(
            o_buf,
            unit,
            fx.Vector.from_elements(outs, ElemTy),
            elem=ElemTy,
            unit_elems=vec,
        )

    silu_mul_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        Gate: fx.Pointer,
        Up: fx.Pointer,
        Out: fx.Pointer,
        n_elems: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        n64 = fx.Int64(n_elems)
        grid_x = (n64 + fx.Int64(tile - 1)) // fx.Int64(tile)
        silu_mul_kernel(Gate, Up, Out, n_elems).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


@lru_cache(maxsize=8)
def build_swiglu_chunk_module(dtype: str = "bfloat16", vec: int | None = None) -> Callable[..., None]:
    """Contiguous last-dim chunk-2 SwiGLU (interleaved gate/up loads)."""
    ElemTy = {"float32": fx.Float32, "float16": fx.Float16, "bfloat16": fx.BFloat16}[dtype]
    elem_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[dtype]
    if vec is None:
        vec = VEC_F32 if dtype == "float32" else VEC_HALF
    tile = BLOCK * vec
    is_f32 = dtype == "float32"
    sig = kernel_signature(block=BLOCK, dtype=dtype, vec=vec, op="swiglu_chunk")

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def swiglu_kernel(X: fx.Pointer, Out: fx.Pointer, n_out: fx.Int32, half_w: fx.Int32) -> None:
        bid = gpu.block_id("x")
        tid = gpu.thread_id("x")
        unit = fx.Int32(bid) * fx.Int32(BLOCK) + fx.Int32(tid)
        out_base = unit * fx.Int32(vec)
        row = out_base // half_w
        col = out_base - row * half_w
        full_w = half_w + half_w
        gate_elem = row * full_w + col
        up_elem = gate_elem + half_w
        x_buf = ptr_buf_tensor(
            X,
            elem=ElemTy,
            n=0x3FFFFFFF,
            unit_elems=vec,
            unit_stride=1,
            num_records_bytes=fx.Int64(n_out) * fx.Int64(2) * fx.Int64(elem_bytes),
        )
        o_buf = ptr_buf_tensor(
            Out,
            elem=ElemTy,
            n=0x3FFFFFFF,
            unit_elems=vec,
            num_records_bytes=fx.Int64(n_out) * fx.Int64(elem_bytes),
        )
        gv = buf_copy_load(x_buf, gate_elem, elem=ElemTy, unit_elems=vec)
        uv = buf_copy_load(x_buf, up_elem, elem=ElemTy, unit_elems=vec)
        if not is_f32:
            gv = gv.extf(T.vec(vec, T.f32))
            uv = uv.extf(T.vec(vec, T.f32))
        one = fx.Float32(1.0)
        neg_log2e = fx.Float32(-1.4426950408889634)
        outs = []
        for i in range_constexpr(vec):
            g = fx.Float32(gv[i])
            u = fx.Float32(uv[i])
            sigv = one / (one + fmath.exp2(g * neg_log2e))
            y = g * sigv * u
            outs.append(y if is_f32 else y.to(ElemTy))
        buf_copy_store(
            o_buf,
            unit,
            fx.Vector.from_elements(outs, ElemTy),
            elem=ElemTy,
            unit_elems=vec,
        )

    swiglu_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        X: fx.Pointer,
        Out: fx.Pointer,
        n_out: fx.Int32,
        half_w: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        n64 = fx.Int64(n_out)
        grid_x = (n64 + fx.Int64(tile - 1)) // fx.Int64(tile)
        swiglu_kernel(X, Out, n_out, half_w).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch
