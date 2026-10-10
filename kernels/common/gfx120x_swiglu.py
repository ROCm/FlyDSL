# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""SiLU(gate)*up and chunk-2 SwiGLU for gfx120x (RDNA4).

Two host builders plus product hosts:

* ``build_silu_mul_module`` -- flat elementwise ``silu(gate) * up``
* ``build_swiglu_chunk_module`` -- last-dim chunk-2 SwiGLU (gate/up halves).
  ``elem_tail == 0`` expects ``half_w % vec == 0``. A nonzero tail scalar-loads.
* ``silu_mul`` / ``swiglu_chunk`` -- product hosts. A short length stays in the kernel.

Plain SiLU*mul (no magnitude clamp). Distinct from interleaved/clamped SwiGLU
kernels used elsewhere. For the zero-LDS in-register MLP fuse, see
``kernels.gemm.rdna4_fused_mlp_nmajor``.
"""

from collections.abc import Callable
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import (
    buf_base_i64,
    buf_copy_load,
    buf_copy_store,
    kernel_signature,
    ptr_buf_tensor,
)

KERNEL_NAME = "swiglu_gfx120x"
BLOCK = 256
VEC_HALF = 8
VEC_F32 = 4


def _elem_ptr(ptr: fx.Pointer, elem_ty: object, align: int) -> fx.Pointer:
    ptr_type = fx.PointerType.get(elem_ty.ir_type, address_space=fx.AddressSpace.Global, alignment=align)
    return fx.inttoptr(ptr_type, buf_base_i64(ptr))


def _load1(ptr: fx.Pointer, idx: fx.Int64 | int) -> object:
    return fx.Vector(fx.make_view(fx.add_offset(ptr, fx.Int64(idx)), fx.make_layout(1, 1)).load())[0]


def _store1(ptr: fx.Pointer, idx: fx.Int64 | int, val: object, elem_ty: object) -> None:
    fx.make_view(fx.add_offset(ptr, fx.Int64(idx)), fx.make_layout(1, 1)).store(fx.Vector.from_elements([val], elem_ty))


@lru_cache(maxsize=8)
def build_silu_mul_module(dtype: str = "bfloat16", vec: int | None = None, elem_tail: int = 0) -> Callable[..., None]:
    """Flat elementwise silu(gate)*up.

    Full ``vec`` groups stay on the wide buffer copy. ``elem_tail`` is the
    leftover element count (``n % vec``). That specialization scalar-loads
    the tail. ``elem_tail == 0`` keeps the previous kernel.
    """
    ElemTy = {"float32": fx.Float32, "float16": fx.Float16, "bfloat16": fx.BFloat16}[dtype]
    elem_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[dtype]
    if vec is None:
        vec = VEC_F32 if dtype == "float32" else VEC_HALF
    if elem_tail < 0 or elem_tail >= vec:
        raise ValueError(f"elem_tail={elem_tail} must be in 0..{vec - 1}")
    is_f32 = dtype == "float32"
    sig_kwargs = {"block": BLOCK, "dtype": dtype, "vec": vec, "op": "silu_mul"}
    if elem_tail:
        sig_kwargs["tail"] = elem_tail
    sig = kernel_signature(**sig_kwargs)
    one = fx.Float32(1.0)
    neg_log2e = fx.Float32(-1.4426950408889634)
    tail_units = 1 if elem_tail else 0

    if is_f32:

        @flyc.kernel(known_block_size=[BLOCK, 1, 1])
        def silu_mul_kernel(Gate: fx.Pointer, Up: fx.Pointer, Out: fx.Pointer, n_elems: fx.Int32) -> None:
            bid = gpu.block_id("x")
            tid = gpu.thread_id("x")
            unit = fx.Int32(bid) * fx.Int32(BLOCK) + fx.Int32(tid)
            n_full = n_elems // fx.Int32(vec)
            nbytes = fx.Int64(n_elems) * fx.Int64(elem_bytes)
            g_buf = ptr_buf_tensor(Gate, elem=ElemTy, n=0x3FFFFFFF, unit_elems=vec, num_records_bytes=nbytes)
            u_buf = ptr_buf_tensor(Up, elem=ElemTy, n=0x3FFFFFFF, unit_elems=vec, num_records_bytes=nbytes)
            o_buf = ptr_buf_tensor(Out, elem=ElemTy, n=0x3FFFFFFF, unit_elems=vec, num_records_bytes=nbytes)
            if unit < n_full:
                gv = buf_copy_load(g_buf, unit, elem=ElemTy, unit_elems=vec)
                uv = buf_copy_load(u_buf, unit, elem=ElemTy, unit_elems=vec)
                outs = []
                for i in range_constexpr(vec):
                    g = fx.Float32(gv[i])
                    u = fx.Float32(uv[i])
                    sigv = one / (one + fmath.exp2(g * neg_log2e))
                    outs.append(g * sigv * u)
                buf_copy_store(o_buf, unit, fx.Vector.from_elements(outs, ElemTy), elem=ElemTy, unit_elems=vec)
            if const_expr(elem_tail != 0):
                if unit == n_full:
                    base = fx.Int64(n_full) * fx.Int64(vec)
                    g_ptr = _elem_ptr(Gate, ElemTy, elem_bytes)
                    u_ptr = _elem_ptr(Up, ElemTy, elem_bytes)
                    o_ptr = _elem_ptr(Out, ElemTy, elem_bytes)
                    for i in range_constexpr(elem_tail):
                        idx = base + fx.Int64(i)
                        g = fx.Float32(_load1(g_ptr, idx))
                        u = fx.Float32(_load1(u_ptr, idx))
                        sigv = one / (one + fmath.exp2(g * neg_log2e))
                        _store1(o_ptr, idx, g * sigv * u, ElemTy)

    else:

        @flyc.kernel(known_block_size=[BLOCK, 1, 1])
        def silu_mul_kernel(Gate: fx.Pointer, Up: fx.Pointer, Out: fx.Pointer, n_elems: fx.Int32) -> None:
            bid = gpu.block_id("x")
            tid = gpu.thread_id("x")
            unit = fx.Int32(bid) * fx.Int32(BLOCK) + fx.Int32(tid)
            n_full = n_elems // fx.Int32(vec)
            nbytes = fx.Int64(n_elems) * fx.Int64(elem_bytes)
            g_buf = ptr_buf_tensor(Gate, elem=ElemTy, n=0x3FFFFFFF, unit_elems=vec, num_records_bytes=nbytes)
            u_buf = ptr_buf_tensor(Up, elem=ElemTy, n=0x3FFFFFFF, unit_elems=vec, num_records_bytes=nbytes)
            o_buf = ptr_buf_tensor(Out, elem=ElemTy, n=0x3FFFFFFF, unit_elems=vec, num_records_bytes=nbytes)
            if unit < n_full:
                gv = buf_copy_load(g_buf, unit, elem=ElemTy, unit_elems=vec)
                uv = buf_copy_load(u_buf, unit, elem=ElemTy, unit_elems=vec)
                gv = gv.extf(T.vec(vec, T.f32))
                uv = uv.extf(T.vec(vec, T.f32))
                outs = []
                for i in range_constexpr(vec):
                    g = fx.Float32(gv[i])
                    u = fx.Float32(uv[i])
                    sigv = one / (one + fmath.exp2(g * neg_log2e))
                    outs.append((g * sigv * u).to(ElemTy))
                buf_copy_store(o_buf, unit, fx.Vector.from_elements(outs, ElemTy), elem=ElemTy, unit_elems=vec)
            if const_expr(elem_tail != 0):
                if unit == n_full:
                    base = fx.Int64(n_full) * fx.Int64(vec)
                    g_ptr = _elem_ptr(Gate, ElemTy, elem_bytes)
                    u_ptr = _elem_ptr(Up, ElemTy, elem_bytes)
                    o_ptr = _elem_ptr(Out, ElemTy, elem_bytes)
                    for i in range_constexpr(elem_tail):
                        idx = base + fx.Int64(i)
                        g = fx.Float32(_load1(g_ptr, idx))
                        u = fx.Float32(_load1(u_ptr, idx))
                        sigv = one / (one + fmath.exp2(g * neg_log2e))
                        _store1(o_ptr, idx, (g * sigv * u).to(ElemTy), ElemTy)

    silu_mul_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        Gate: fx.Pointer,
        Up: fx.Pointer,
        Out: fx.Pointer,
        n_elems: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        n64 = fx.Int64(n_elems)
        n_full = n64 // fx.Int64(vec)
        n_work = n_full + fx.Int64(tail_units)
        grid_x = (n_work + fx.Int64(BLOCK - 1)) // fx.Int64(BLOCK)
        silu_mul_kernel(Gate, Up, Out, n_elems).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


@lru_cache(maxsize=8)
def build_swiglu_chunk_module(
    dtype: str = "bfloat16", vec: int | None = None, elem_tail: int = 0
) -> Callable[..., None]:
    """Contiguous last-dim chunk-2 SwiGLU (interleaved gate/up loads).

    ``elem_tail == 0`` keeps the wide kernel and requires ``half_w % vec == 0``.
    A nonzero tail scalar-loads every output element, so a short half-width
    does not need a padded row.
    """
    ElemTy = {"float32": fx.Float32, "float16": fx.Float16, "bfloat16": fx.BFloat16}[dtype]
    elem_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[dtype]
    if vec is None:
        vec = VEC_F32 if dtype == "float32" else VEC_HALF
    if elem_tail < 0 or elem_tail >= vec:
        raise ValueError(f"elem_tail={elem_tail} must be in 0..{vec - 1}")
    is_f32 = dtype == "float32"
    sig_kwargs = {"block": BLOCK, "dtype": dtype, "vec": vec, "op": "swiglu_chunk"}
    if elem_tail:
        sig_kwargs["tail"] = elem_tail
    sig = kernel_signature(**sig_kwargs)
    one = fx.Float32(1.0)
    neg_log2e = fx.Float32(-1.4426950408889634)

    if elem_tail:

        @flyc.kernel(known_block_size=[BLOCK, 1, 1])
        def swiglu_kernel(X: fx.Pointer, Out: fx.Pointer, n_out: fx.Int32, half_w: fx.Int32) -> None:
            bid = gpu.block_id("x")
            tid = gpu.thread_id("x")
            unit = fx.Int32(bid) * fx.Int32(BLOCK) + fx.Int32(tid)
            if unit < n_out:
                row = unit // half_w
                col = unit - row * half_w
                full_w = half_w + half_w
                x_ptr = _elem_ptr(X, ElemTy, elem_bytes)
                o_ptr = _elem_ptr(Out, ElemTy, elem_bytes)
                g = fx.Float32(_load1(x_ptr, fx.Int64(row) * fx.Int64(full_w) + fx.Int64(col)))
                u = fx.Float32(_load1(x_ptr, fx.Int64(row) * fx.Int64(full_w) + fx.Int64(col) + fx.Int64(half_w)))
                sigv = one / (one + fmath.exp2(g * neg_log2e))
                val = g * sigv * u
                if not is_f32:
                    val = val.to(ElemTy)
                _store1(o_ptr, fx.Int64(unit), val, ElemTy)

        swiglu_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

        @flyc.jit
        def launch(
            X: fx.Pointer,
            Out: fx.Pointer,
            n_out: fx.Int32,
            half_w: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ) -> None:
            n64 = fx.Int64(n_out)
            grid_x = (n64 + fx.Int64(BLOCK - 1)) // fx.Int64(BLOCK)
            swiglu_kernel(X, Out, n_out, half_w).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

        launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
        return launch

    if is_f32:

        @flyc.kernel(known_block_size=[BLOCK, 1, 1])
        def swiglu_kernel(X: fx.Pointer, Out: fx.Pointer, n_out: fx.Int32, half_w: fx.Int32) -> None:
            bid = gpu.block_id("x")
            tid = gpu.thread_id("x")
            unit = fx.Int32(bid) * fx.Int32(BLOCK) + fx.Int32(tid)
            n_full = n_out // fx.Int32(vec)
            x_nbytes = fx.Int64(n_out) * fx.Int64(2) * fx.Int64(elem_bytes)
            o_nbytes = fx.Int64(n_out) * fx.Int64(elem_bytes)
            if unit < n_full:
                out_base = unit * fx.Int32(vec)
                row = out_base // half_w
                col = out_base - row * half_w
                full_w = half_w + half_w
                gate_elem = row * full_w + col
                up_elem = gate_elem + half_w
                x_buf = ptr_buf_tensor(
                    X, elem=ElemTy, n=0x3FFFFFFF, unit_elems=vec, unit_stride=1, num_records_bytes=x_nbytes
                )
                o_buf = ptr_buf_tensor(Out, elem=ElemTy, n=0x3FFFFFFF, unit_elems=vec, num_records_bytes=o_nbytes)
                gv = buf_copy_load(x_buf, gate_elem, elem=ElemTy, unit_elems=vec)
                uv = buf_copy_load(x_buf, up_elem, elem=ElemTy, unit_elems=vec)
                outs = []
                for i in range_constexpr(vec):
                    g = fx.Float32(gv[i])
                    u = fx.Float32(uv[i])
                    sigv = one / (one + fmath.exp2(g * neg_log2e))
                    outs.append(g * sigv * u)
                buf_copy_store(o_buf, unit, fx.Vector.from_elements(outs, ElemTy), elem=ElemTy, unit_elems=vec)

    else:

        @flyc.kernel(known_block_size=[BLOCK, 1, 1])
        def swiglu_kernel(X: fx.Pointer, Out: fx.Pointer, n_out: fx.Int32, half_w: fx.Int32) -> None:
            bid = gpu.block_id("x")
            tid = gpu.thread_id("x")
            unit = fx.Int32(bid) * fx.Int32(BLOCK) + fx.Int32(tid)
            n_full = n_out // fx.Int32(vec)
            x_nbytes = fx.Int64(n_out) * fx.Int64(2) * fx.Int64(elem_bytes)
            o_nbytes = fx.Int64(n_out) * fx.Int64(elem_bytes)
            if unit < n_full:
                out_base = unit * fx.Int32(vec)
                row = out_base // half_w
                col = out_base - row * half_w
                full_w = half_w + half_w
                gate_elem = row * full_w + col
                up_elem = gate_elem + half_w
                x_buf = ptr_buf_tensor(
                    X, elem=ElemTy, n=0x3FFFFFFF, unit_elems=vec, unit_stride=1, num_records_bytes=x_nbytes
                )
                o_buf = ptr_buf_tensor(Out, elem=ElemTy, n=0x3FFFFFFF, unit_elems=vec, num_records_bytes=o_nbytes)
                gv = buf_copy_load(x_buf, gate_elem, elem=ElemTy, unit_elems=vec)
                uv = buf_copy_load(x_buf, up_elem, elem=ElemTy, unit_elems=vec)
                gv = gv.extf(T.vec(vec, T.f32))
                uv = uv.extf(T.vec(vec, T.f32))
                outs = []
                for i in range_constexpr(vec):
                    g = fx.Float32(gv[i])
                    u = fx.Float32(uv[i])
                    sigv = one / (one + fmath.exp2(g * neg_log2e))
                    outs.append((g * sigv * u).to(ElemTy))
                buf_copy_store(o_buf, unit, fx.Vector.from_elements(outs, ElemTy), elem=ElemTy, unit_elems=vec)

    swiglu_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        X: fx.Pointer,
        Out: fx.Pointer,
        n_out: fx.Int32,
        half_w: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        n64 = fx.Int64(n_out)
        n_full = n64 // fx.Int64(vec)
        grid_x = (n_full + fx.Int64(BLOCK - 1)) // fx.Int64(BLOCK)
        swiglu_kernel(X, Out, n_out, half_w).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


_DTYPE_NAME = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}


def _ptr(t: torch.Tensor):
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def silu_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Product silu(gate)*up. A length that is not a vector multiple stays in the kernel."""
    require_gfx120x(gate.device, what="silu_mul (gfx120x)")
    if gate.dtype not in _DTYPE_NAME or up.dtype != gate.dtype:
        raise ValueError(f"silu_mul requires matching bf16/fp16/f32, got {gate.dtype}/{up.dtype}")
    if gate.shape != up.shape:
        raise ValueError(f"silu_mul shape mismatch {tuple(gate.shape)} vs {tuple(up.shape)}")
    dtype_name = _DTYPE_NAME[gate.dtype]
    vec = VEC_F32 if dtype_name == "float32" else VEC_HALF
    flat_g = gate.reshape(-1)
    flat_u = up.reshape(-1)
    n = int(flat_g.numel())
    if n == 0:
        if out is None:
            return gate.reshape(gate.shape)
        if stream is None:
            out.copy_(gate)  # empty shape; no launch
        else:
            with torch.cuda.stream(stream):
                out.copy_(gate)
        return out
    from kernels.common.gfx120x_pad import ensure_contiguous

    g_c = ensure_contiguous(flat_g, stream=stream)
    u_c = ensure_contiguous(flat_u, stream=stream)
    scratch = torch.empty((n,), device=gate.device, dtype=gate.dtype)
    launch = build_silu_mul_module(dtype_name, elem_tail=n % vec)
    if stream is None:
        launch(_ptr(g_c), _ptr(u_c), _ptr(scratch), n)
    else:
        launch(_ptr(g_c), _ptr(u_c), _ptr(scratch), n, stream)
    y = scratch.reshape(gate.shape)
    if out is None:
        return y
    if stream is None:
        out.copy_(y)
    else:
        with torch.cuda.stream(stream):
            out.copy_(y)
    return out


def swiglu_chunk(
    x: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Product chunk-2 SwiGLU: ``x[..., 2*H]`` → ``silu(gate)*up`` with shape ``[..., H]``.

    A half-width that is not a multiple of the vector width (8 for bf16/fp16,
    4 for f32) is scalar-loaded in the kernel. ``stream`` is
    ``torch.cuda.Stream`` (host); the kernel launch uses ``fx.Stream``.
    """
    require_gfx120x(x.device, what="swiglu_chunk (gfx120x)")
    if x.dtype not in _DTYPE_NAME:
        raise ValueError(f"swiglu_chunk requires bf16/fp16/f32, got {x.dtype}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"swiglu_chunk last dim must be even (2*H), got {x.shape[-1]}")
    half_w = int(x.shape[-1]) // 2
    dtype_name = _DTYPE_NAME[x.dtype]
    vec = VEC_F32 if dtype_name == "float32" else VEC_HALF
    out_shape = x.shape[:-1] + (half_w,)
    if half_w == 0 or int(x.numel()) == 0:
        if out is None:
            return torch.empty(out_shape, device=x.device, dtype=x.dtype)
        if tuple(out.shape) != out_shape or out.dtype != x.dtype or out.device != x.device:
            raise ValueError(
                f"swiglu_chunk prealloc out mismatch: got {tuple(out.shape)}/{out.dtype}/{out.device}, "
                f"want {out_shape}/{x.dtype}/{x.device}"
            )
        return out
    from kernels.common.gfx120x_pad import ensure_contiguous

    x_work = ensure_contiguous(x, stream=stream)

    if out is None:
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    elif tuple(out.shape) != out_shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError(
            f"swiglu_chunk prealloc out mismatch: got {tuple(out.shape)}/{out.dtype}/{out.device}, "
            f"want {out_shape}/{x.dtype}/{x.device}"
        )

    rows = int(x_work.numel() // (2 * half_w))
    # The kernel stores packed row-major from data_ptr(). A non-contiguous
    # `out` (a column slice, for example) would be written as if it were packed.
    launch_out = out
    if not out.is_contiguous():
        launch_out = torch.empty(out_shape, device=out.device, dtype=out.dtype)
    dst = launch_out.reshape(rows, half_w)
    launch = build_swiglu_chunk_module(dtype_name, elem_tail=half_w % vec)
    n_out = rows * half_w
    if stream is None:
        launch(_ptr(x_work), _ptr(dst), n_out, half_w)
    else:
        launch(_ptr(x_work), _ptr(dst), n_out, half_w, stream)
    if launch_out is not out:
        if stream is None:
            out.copy_(launch_out)
        else:
            with torch.cuda.stream(stream):
                out.copy_(launch_out)
    return out
