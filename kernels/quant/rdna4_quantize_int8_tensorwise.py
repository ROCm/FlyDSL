# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Tensorwise absmax INT8 quantize for gfx120x (``quantize_int8_tensorwise``).

One scale for the whole tensor, HIP-compatible with reference
``quantize_int8_tensorwise`` / ``TensorWiseINT8Layout`` (weight path,
``is_weight=True``, ``per_channel=False``)::

    scale = max(amax / 127, 1e-30)     # scalar fp32
    q = roundeven(x * rcp(scale))      # clamp to [-128, 127]

``rcp`` matches HIP ``amdgcn_rcpf`` (same INT8 scale exception as rowwise).
A caller-supplied ``scale`` uses the eager path. ``stochastic_rounding``
raises. The device kernel is the absmax path only.

Does **not** require the gfx120x iu8 WMMA atom. Public:
``build_quantize_int8_tensorwise_module``, ``quantize_int8_tensorwise``.
"""

import math
from collections.abc import Callable
from functools import lru_cache
from typing import Optional, Tuple, Union

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_argument import PointerJitArg
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor
from kernels.common.tensor_shim import _run_compiled

KERNEL_NAME = "rdna4_quantize_int8_tensorwise"
WARP = 32
BLOCK = 256
INV_127 = 1.0 / 127.0


def _vec_width(dtype_str: str) -> int:
    return 4 if dtype_str == "float32" else 8


def _elem(dtype_str: str) -> tuple[type, int]:
    return {
        "float32": (fx.Float32, 4),
        "float16": (fx.Float16, 2),
        "bfloat16": (fx.BFloat16, 2),
    }[dtype_str]


@lru_cache(maxsize=8)
def build_quantize_int8_tensorwise_module(dtype_str: str) -> Callable[..., None]:
    """Three launches: block partial amax, tree-reduce amax, quantize.

    Cached per dtype. Element count is a runtime ``fx.Int32`` (pointer args),
    so one specialization covers every shape of that dtype.
    """
    elem_dtype, elem_bytes = _elem(dtype_str)
    vec = _vec_width(dtype_str)
    out_bytes = vec  # int8
    reduction_slots = BLOCK // WARP
    RedTy = fx.Array[fx.Float32, reduction_slots, 16]

    @fx.struct
    class SharedStorage:
        reduction_buffer: RedTy

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def partial_kernel(In: fx.Pointer, Partial: fx.Pointer, nvec: fx.Int32, npartial: fx.Int32) -> None:
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        c0 = fx.Float32(0.0)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        red = lds.reduction_buffer.view(fx.make_layout(reduction_slots, 1))

        def wave_reduce_max(w0: object) -> fx.Float32:
            w = w0
            for _sh in range_constexpr(int(math.log2(WARP))):
                off = WARP // (2 << _sh)
                w = fx.max(w, gpu.shuffle_xor(w, off, WARP))
            return w

        def block_reduce_max(val: object) -> fx.Float32:
            lane = tid % WARP
            wave = tid // WARP
            w = wave_reduce_max(val)
            if lane == 0:
                fx.memref_store(w, red, wave)
            gpu.barrier()
            if wave == 0:
                lane_ok = lane < reduction_slots
                lane_safe = lane_ok.select(lane, 0)
                vv = red[lane_safe]
                ww = lane_ok.select(vv, c0)
                ww = wave_reduce_max(ww)
                if lane == 0:
                    fx.memref_store(ww, red, 0)
            gpu.barrier()
            return fx.Float32(red[0])

        unit = fx.Int32(bid) * fx.Int32(BLOCK) + fx.Int32(tid)
        in_buf = ptr_buf_tensor(
            In,
            elem=elem_dtype,
            n=0x3FFFFFFF,
            unit_elems=vec,
            num_records_bytes=fx.Int64(nvec) * fx.Int64(vec * elem_bytes),
        )
        part_buf = ptr_buf_tensor(
            Partial,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(npartial) * fx.Int64(4),
        )
        in_range = unit < nvec
        safe = in_range.select(unit, fx.Int32(0))
        vec_v = buf_copy_load(in_buf, safe, elem=elem_dtype, unit_elems=vec)
        xv = vec_v.to(fx.Float32)
        thread_amax = c0
        for ei in range_constexpr(vec):
            ae = fmath.absf(xv[ei])
            thread_amax = fx.max(thread_amax, in_range.select(ae, c0))
        amax = block_reduce_max(thread_amax)
        if tid == 0:
            buf_copy_store(part_buf, fx.Int32(bid), amax, elem=fx.Float32, unit_elems=1)

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def reduce_kernel(Src: fx.Pointer, Dst: fx.Pointer, n_in: fx.Int32, n_out: fx.Int32) -> None:
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        c0 = fx.Float32(0.0)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        red = lds.reduction_buffer.view(fx.make_layout(reduction_slots, 1))

        def wave_reduce_max(w0: object) -> fx.Float32:
            w = w0
            for _sh in range_constexpr(int(math.log2(WARP))):
                off = WARP // (2 << _sh)
                w = fx.max(w, gpu.shuffle_xor(w, off, WARP))
            return w

        def block_reduce_max(val: object) -> fx.Float32:
            lane = tid % WARP
            wave = tid // WARP
            w = wave_reduce_max(val)
            if lane == 0:
                fx.memref_store(w, red, wave)
            gpu.barrier()
            if wave == 0:
                lane_ok = lane < reduction_slots
                lane_safe = lane_ok.select(lane, 0)
                vv = red[lane_safe]
                ww = lane_ok.select(vv, c0)
                ww = wave_reduce_max(ww)
                if lane == 0:
                    fx.memref_store(ww, red, 0)
            gpu.barrier()
            return fx.Float32(red[0])

        idx = fx.Int32(bid) * fx.Int32(BLOCK) + fx.Int32(tid)
        src_buf = ptr_buf_tensor(
            Src,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_in) * fx.Int64(4),
        )
        dst_buf = ptr_buf_tensor(
            Dst,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_out) * fx.Int64(4),
        )
        in_range = idx < n_in
        safe = in_range.select(idx, fx.Int32(0))
        v = fx.Float32(buf_copy_load(src_buf, safe, elem=fx.Float32, unit_elems=1))
        v = in_range.select(v, c0)
        amax = block_reduce_max(v)
        if tid == 0:
            buf_copy_store(dst_buf, fx.Int32(bid), amax, elem=fx.Float32, unit_elems=1)

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def quant_kernel(
        In: fx.Pointer,
        Out: fx.Pointer,
        Amax: fx.Pointer,
        Scale: fx.Pointer,
        nvec: fx.Int32,
    ) -> None:
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        unit = fx.Int32(bid) * fx.Int32(BLOCK) + fx.Int32(tid)
        in_buf = ptr_buf_tensor(
            In,
            elem=elem_dtype,
            n=0x3FFFFFFF,
            unit_elems=vec,
            num_records_bytes=fx.Int64(nvec) * fx.Int64(vec * elem_bytes),
        )
        out_buf = ptr_buf_tensor(
            Out,
            elem=fx.Int8,
            n=0x3FFFFFFF,
            unit_elems=vec,
            num_records_bytes=fx.Int64(nvec) * fx.Int64(out_bytes),
        )
        amax_buf = ptr_buf_tensor(Amax, elem=fx.Float32, n=1, unit_elems=1, num_records_bytes=4)
        scale_buf = ptr_buf_tensor(Scale, elem=fx.Float32, n=1, unit_elems=1, num_records_bytes=4)
        amax = fx.Float32(buf_copy_load(amax_buf, fx.Int32(0), elem=fx.Float32, unit_elems=1))
        scale = fx.max(amax * fx.Float32(INV_127), fx.Float32(1e-30))
        inv = fx.Float32(fx.rocdl.rcp(T.f32, scale))
        if tid == 0 and bid == 0:
            buf_copy_store(scale_buf, fx.Int32(0), scale, elem=fx.Float32, unit_elems=1)
        in_range = unit < nvec
        if in_range:
            vec_v = buf_copy_load(in_buf, unit, elem=elem_dtype, unit_elems=vec)
            xv = vec_v.to(fx.Float32)
            qs = []
            for ei in range_constexpr(vec):
                qf = fmath.roundeven(xv[ei] * inv)
                qf = fx.max(fx.min(qf, fx.Float32(127.0)), fx.Float32(-128.0))
                qs.append(qf.to(fx.Int8))
            buf_copy_store(
                out_buf,
                unit,
                fx.Vector.from_elements(qs, fx.Int8),
                elem=fx.Int8,
                unit_elems=vec,
            )

    partial_kernel.__name__ = f"{KERNEL_NAME}_partial_{dtype_str}"
    reduce_kernel.__name__ = f"{KERNEL_NAME}_reduce_{dtype_str}"
    quant_kernel.__name__ = f"{KERNEL_NAME}_quant_{dtype_str}"

    @flyc.jit
    def launch_partial(
        In: fx.Pointer,
        Partial: fx.Pointer,
        nvec: fx.Int32,
        npartial: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        grid_x = (fx.Int64(nvec) + fx.Int64(BLOCK - 1)) // fx.Int64(BLOCK)
        partial_kernel(In, Partial, nvec, npartial).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    @flyc.jit
    def launch_reduce(
        Src: fx.Pointer,
        Dst: fx.Pointer,
        n_in: fx.Int32,
        n_out: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        grid_x = (fx.Int64(n_in) + fx.Int64(BLOCK - 1)) // fx.Int64(BLOCK)
        reduce_kernel(Src, Dst, n_in, n_out).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    @flyc.jit
    def launch_quant(
        In: fx.Pointer,
        Out: fx.Pointer,
        Amax: fx.Pointer,
        Scale: fx.Pointer,
        nvec: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        grid_x = (fx.Int64(nvec) + fx.Int64(BLOCK - 1)) // fx.Int64(BLOCK)
        quant_kernel(In, Out, Amax, Scale, nvec).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch_partial.__name__ = f"launch_{KERNEL_NAME}_partial_{dtype_str}"
    launch_reduce.__name__ = f"launch_{KERNEL_NAME}_reduce_{dtype_str}"
    launch_quant.__name__ = f"launch_{KERNEL_NAME}_quant_{dtype_str}"
    return launch_partial, launch_reduce, launch_quant


def _ptr(tensor: torch.Tensor) -> PointerJitArg:
    return flyc.from_c_void_p(fx.Uint8, tensor.data_ptr())


def _dtype_name(dt: torch.dtype) -> str:
    return {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }[dt]


def _eager_supplied_scale(x: torch.Tensor, scale: torch.Tensor | float | None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager reference path for a caller-supplied scale (not the absmax kernel)."""
    if not isinstance(scale, torch.Tensor):
        scale_t = torch.tensor(scale, device=x.device, dtype=torch.float32)
    else:
        scale_t = scale.to(device=x.device, dtype=torch.float32)
    scale_min = torch.finfo(x.dtype).tiny
    scale_math = torch.where(scale_t == 0, torch.full_like(scale_t, scale_min), scale_t)
    q = torch.round(x / scale_math.to(dtype=x.dtype)).clamp(-128, 127).to(torch.int8)
    if scale_t.ndim != 0:
        scale_t = scale_t.reshape(())
    return q, scale_t


def quantize_int8_tensorwise(
    x: torch.Tensor,
    scale: Union[torch.Tensor, float, str, None] = None,
    stochastic_rounding: Optional[int] = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tensorwise absmax int8 quant → ``(q, scale)``.

    ``q`` matches ``x.shape`` (int8). ``scale`` is a 0-dim float32 tensor on
    ``x.device`` (reference ``quantize_int8_tensorwise``).
    """
    require_gfx120x(x.device, what="quantize_int8_tensorwise (gfx120x)")
    if stochastic_rounding:
        raise ValueError("quantize_int8_tensorwise: stochastic_rounding not supported")
    if isinstance(scale, str):
        if scale != "recalculate":
            raise ValueError(f"quantize_int8_tensorwise: unsupported scale={scale!r}")
        scale = None
    if scale is not None:
        return _eager_supplied_scale(x, scale)
    if x.device.type != "cuda" or x.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError("quantize_int8_tensorwise: need CUDA/ROCm f32/f16/bf16")
    if x.numel() == 0:
        raise ValueError("quantize_int8_tensorwise: empty tensor")

    dtype_str = _dtype_name(x.dtype)
    vec = _vec_width(dtype_str)
    flat = x.reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    numel = flat.numel()
    pad = (vec - (numel % vec)) % vec
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)], dim=0)
    nvec = flat.numel() // vec
    npartial = (nvec + BLOCK - 1) // BLOCK
    qpad = torch.empty(flat.numel(), dtype=torch.int8, device=x.device)
    partial = torch.empty(npartial, dtype=torch.float32, device=x.device)
    scale_t = torch.empty((), dtype=torch.float32, device=x.device)
    launch_partial, launch_reduce, launch_quant = build_quantize_int8_tensorwise_module(dtype_str)
    stream = torch.cuda.current_stream(device=x.device)
    _run_compiled(launch_partial, _ptr(flat), _ptr(partial), nvec, npartial, stream)
    curr = partial
    n = npartial
    while n > 1:
        n2 = (n + BLOCK - 1) // BLOCK
        nxt = torch.empty(n2, dtype=torch.float32, device=x.device)
        _run_compiled(launch_reduce, _ptr(curr), _ptr(nxt), n, n2, stream)
        curr = nxt
        n = n2
    _run_compiled(launch_quant, _ptr(flat), _ptr(qpad), _ptr(curr), _ptr(scale_t), nvec, stream)
    return qpad[:numel].reshape(x.shape), scale_t
