# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Tensorwise absmax INT8 quantize for gfx120x (``quantize_int8_tensorwise``).

One scale for the whole tensor, HIP-compatible with reference
``quantize_int8_tensorwise`` / ``TensorWiseINT8Layout`` (weight path,
``is_weight=True``, ``per_channel=False``)::

    scale = max(amax / 127, 1e-30)     # scalar fp32
    q = roundeven(x * rcp(scale))      # clamp to [-128, 127]

``rcp`` matches HIP ``amdgcn_rcpf`` (same INT8 scale exception as rowwise).
A caller-supplied ``scale`` is the same quant kernel with that scale, not an
eager round. ``stochastic_rounding`` raises.

Does **not** require the gfx120x iu8 WMMA atom. Public:
``build_quantize_int8_tensorwise_module``, ``quantize_int8_tensorwise``.
"""

import math
from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional, Tuple, Union

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import Config, autotune
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor
from kernels.common.gfx120x_pad import ensure_contiguous

KERNEL_NAME = "rdna4_quantize_int8_tensorwise"
WARP = 32
BLOCK = 256
# Fixed coverage tile so every wave32 block reduces the same elements.
# 1024 is divisible by 32, 64, 128, 256, 512 and 1024.
TILE = 1024
TUNING_SCHEMA = 1
_BLOCK_CHOICES = (32, 64, 128, 256, 512, 1024)
INV_127 = 1.0 / 127.0


def _vec_width(dtype_str: str) -> int:
    return 4 if dtype_str == "float32" else 8


def _elem(dtype_str: str) -> tuple[type, int]:
    return {
        "float32": (fx.Float32, 4),
        "float16": (fx.Float16, 2),
        "bfloat16": (fx.BFloat16, 2),
    }[dtype_str]


@lru_cache(maxsize=16)
def build_quantize_int8_tensorwise_module(
    dtype_str: str, block_threads: int = BLOCK, elem_tail: int = 0
) -> Callable[..., None]:
    """Three launches: block partial amax, tree-reduce amax, quantize.

    Cached per dtype, block, and ``elem_tail`` (``numel % vec``). Zero keeps
    the full-vector buffer record. A non-zero tail sizes that record to the
    real element count so the last vector zero-fills past the allocation.
    """
    if block_threads not in _BLOCK_CHOICES:
        raise ValueError(f"block_threads={block_threads} is not a legal wave32 block")
    elem_dtype, elem_bytes = _elem(dtype_str)
    vec = _vec_width(dtype_str)
    out_bytes = vec  # int8
    reduction_slots = block_threads // WARP
    vecs_per_thread = TILE // block_threads
    RedTy = fx.Array[fx.Float32, reduction_slots, 16]

    @fx.struct
    class SharedStorage:
        reduction_buffer: RedTy

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def partial_kernel(In: fx.Tensor, Partial: fx.Tensor, nvec: fx.Int32, npartial: fx.Int32) -> None:
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

        if const_expr(elem_tail != 0):
            in_records = (fx.Int64(nvec) - fx.Int64(1)) * fx.Int64(vec * elem_bytes) + fx.Int64(elem_tail * elem_bytes)
        else:
            in_records = fx.Int64(nvec) * fx.Int64(vec * elem_bytes)
        in_buf = ptr_buf_tensor(
            In,
            elem=elem_dtype,
            n=0x3FFFFFFF,
            unit_elems=vec,
            num_records_bytes=in_records,
        )
        part_buf = ptr_buf_tensor(
            Partial,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(npartial) * fx.Int64(4),
        )
        if const_expr(elem_tail != 0):
            in_elem = ptr_buf_tensor(
                In,
                elem=elem_dtype,
                n=0x3FFFFFFF,
                unit_elems=1,
                num_records_bytes=in_records,
            )
        unit_base = fx.Int32(bid) * fx.Int32(TILE)
        thread_amax = c0
        for step in range_constexpr(vecs_per_thread):
            unit = unit_base + fx.Int32(tid) + fx.Int32(step * block_threads)
            in_range = unit < nvec
            if const_expr(elem_tail != 0):
                # A wide load of the last vector drops the last in-range element.
                if unit == (fx.Int32(nvec) - fx.Int32(1)):
                    base = unit * fx.Int32(vec)
                    for ei in range_constexpr(elem_tail):
                        one = buf_copy_load(in_elem, base + fx.Int32(ei), elem=elem_dtype, unit_elems=1)
                        thread_amax = fx.max(thread_amax, fmath.absf(fx.Float32(one)))
                else:
                    safe = in_range.select(unit, fx.Int32(0))
                    vec_v = buf_copy_load(in_buf, safe, elem=elem_dtype, unit_elems=vec)
                    xv = vec_v.to(fx.Float32)
                    for ei in range_constexpr(vec):
                        ae = fmath.absf(xv[ei])
                        thread_amax = fx.max(thread_amax, in_range.select(ae, c0))
            else:
                safe = in_range.select(unit, fx.Int32(0))
                vec_v = buf_copy_load(in_buf, safe, elem=elem_dtype, unit_elems=vec)
                xv = vec_v.to(fx.Float32)
                for ei in range_constexpr(vec):
                    ae = fmath.absf(xv[ei])
                    thread_amax = fx.max(thread_amax, in_range.select(ae, c0))
        amax = block_reduce_max(thread_amax)
        if tid == 0:
            buf_copy_store(part_buf, fx.Int32(bid), amax, elem=fx.Float32, unit_elems=1)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def reduce_kernel(Src: fx.Tensor, Dst: fx.Tensor, n_in: fx.Int32, n_out: fx.Int32) -> None:
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
        idx_base = fx.Int32(bid) * fx.Int32(TILE)
        thread_amax = c0
        for step in range_constexpr(vecs_per_thread):
            idx = idx_base + fx.Int32(tid) + fx.Int32(step * block_threads)
            in_range = idx < n_in
            safe = in_range.select(idx, fx.Int32(0))
            v = fx.Float32(buf_copy_load(src_buf, safe, elem=fx.Float32, unit_elems=1))
            v = in_range.select(v, c0)
            thread_amax = fx.max(thread_amax, v)
        amax = block_reduce_max(thread_amax)
        if tid == 0:
            buf_copy_store(dst_buf, fx.Int32(bid), amax, elem=fx.Float32, unit_elems=1)

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def quant_kernel(
        In: fx.Tensor,
        Out: fx.Tensor,
        Amax: fx.Tensor,
        Scale: fx.Tensor,
        nvec: fx.Int32,
    ) -> None:
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        if const_expr(elem_tail != 0):
            in_records = (fx.Int64(nvec) - fx.Int64(1)) * fx.Int64(vec * elem_bytes) + fx.Int64(elem_tail * elem_bytes)
            out_records = (fx.Int64(nvec) - fx.Int64(1)) * fx.Int64(vec) + fx.Int64(elem_tail)
        else:
            in_records = fx.Int64(nvec) * fx.Int64(vec * elem_bytes)
            out_records = fx.Int64(nvec) * fx.Int64(out_bytes)
        in_buf = ptr_buf_tensor(
            In,
            elem=elem_dtype,
            n=0x3FFFFFFF,
            unit_elems=vec,
            num_records_bytes=in_records,
        )
        out_buf = ptr_buf_tensor(
            Out,
            elem=fx.Int8,
            n=0x3FFFFFFF,
            unit_elems=vec,
            num_records_bytes=out_records,
        )
        if const_expr(elem_tail != 0):
            in_elem = ptr_buf_tensor(In, elem=elem_dtype, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=in_records)
            out_elem = ptr_buf_tensor(Out, elem=fx.Int8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=out_records)
        amax_buf = ptr_buf_tensor(Amax, elem=fx.Float32, n=1, unit_elems=1, num_records_bytes=4)
        scale_buf = ptr_buf_tensor(Scale, elem=fx.Float32, n=1, unit_elems=1, num_records_bytes=4)
        amax = fx.Float32(buf_copy_load(amax_buf, fx.Int32(0), elem=fx.Float32, unit_elems=1))
        scale = fx.max(amax * fx.Float32(INV_127), fx.Float32(1e-30))
        inv = fx.Float32(fx.rocdl.rcp(T.f32, scale))
        if tid == 0 and bid == 0:
            buf_copy_store(scale_buf, fx.Int32(0), scale, elem=fx.Float32, unit_elems=1)
        unit_base = fx.Int32(bid) * fx.Int32(TILE)
        for step in range_constexpr(vecs_per_thread):
            unit = unit_base + fx.Int32(tid) + fx.Int32(step * block_threads)
            in_range = unit < nvec
            if const_expr(elem_tail != 0):
                if unit == (fx.Int32(nvec) - fx.Int32(1)):
                    base = unit * fx.Int32(vec)
                    for ei in range_constexpr(elem_tail):
                        idx = base + fx.Int32(ei)
                        one = fx.Float32(buf_copy_load(in_elem, idx, elem=elem_dtype, unit_elems=1))
                        qf = fmath.roundeven(one * inv)
                        qf = fx.max(fx.min(qf, fx.Float32(127.0)), fx.Float32(-128.0))
                        buf_copy_store(out_elem, idx, qf.to(fx.Int8), elem=fx.Int8, unit_elems=1)
                elif in_range:
                    vec_v = buf_copy_load(in_buf, unit, elem=elem_dtype, unit_elems=vec)
                    xv = vec_v.to(fx.Float32)
                    qs = []
                    for ei in range_constexpr(vec):
                        qf = fmath.roundeven(xv[ei] * inv)
                        qf = fx.max(fx.min(qf, fx.Float32(127.0)), fx.Float32(-128.0))
                        qs.append(qf.to(fx.Int8))
                    buf_copy_store(out_buf, unit, fx.Vector.from_elements(qs, fx.Int8), elem=fx.Int8, unit_elems=vec)
            elif in_range:
                vec_v = buf_copy_load(in_buf, unit, elem=elem_dtype, unit_elems=vec)
                xv = vec_v.to(fx.Float32)
                qs = []
                for ei in range_constexpr(vec):
                    qf = fmath.roundeven(xv[ei] * inv)
                    qf = fx.max(fx.min(qf, fx.Float32(127.0)), fx.Float32(-128.0))
                    qs.append(qf.to(fx.Int8))
                buf_copy_store(out_buf, unit, fx.Vector.from_elements(qs, fx.Int8), elem=fx.Int8, unit_elems=vec)

    partial_kernel.__name__ = f"{KERNEL_NAME}_partial_{dtype_str}"
    reduce_kernel.__name__ = f"{KERNEL_NAME}_reduce_{dtype_str}"
    quant_kernel.__name__ = f"{KERNEL_NAME}_quant_{dtype_str}"

    @flyc.jit
    def launch_partial(
        In: fx.Tensor,
        Partial: fx.Tensor,
        nvec: fx.Int32,
        npartial: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        grid_x = (fx.Int64(nvec) + fx.Int64(TILE - 1)) // fx.Int64(TILE)
        partial_kernel(In, Partial, nvec, npartial).launch(
            grid=(grid_x, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_reduce(
        Src: fx.Tensor,
        Dst: fx.Tensor,
        n_in: fx.Int32,
        n_out: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        grid_x = (fx.Int64(n_in) + fx.Int64(TILE - 1)) // fx.Int64(TILE)
        reduce_kernel(Src, Dst, n_in, n_out).launch(grid=(grid_x, 1, 1), block=(block_threads, 1, 1), stream=stream)

    @flyc.jit
    def launch_quant(
        In: fx.Tensor,
        Out: fx.Tensor,
        Amax: fx.Tensor,
        Scale: fx.Tensor,
        nvec: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        grid_x = (fx.Int64(nvec) + fx.Int64(TILE - 1)) // fx.Int64(TILE)
        quant_kernel(In, Out, Amax, Scale, nvec).launch(grid=(grid_x, 1, 1), block=(block_threads, 1, 1), stream=stream)

    launch_partial.__name__ = f"launch_{KERNEL_NAME}_partial_{dtype_str}"
    launch_reduce.__name__ = f"launch_{KERNEL_NAME}_reduce_{dtype_str}"
    launch_quant.__name__ = f"launch_{KERNEL_NAME}_quant_{dtype_str}"
    return launch_partial, launch_reduce, launch_quant


def _dtype_name(dt: torch.dtype) -> str:
    return {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }[dt]


def _default_block(*_args, **_kwargs) -> Config:
    return Config(BLOCK_THREADS=128)


@contextmanager
def _validate_named(sig_args, names: tuple[str, ...]):
    import torch

    poisoned = []
    for name in names:
        value = sig_args[name]
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.numel():
            value.fill_(float("nan"))
            poisoned.append(value)
    yield
    for value in poisoned:
        if not bool(torch.isfinite(value).all()):
            raise ValueError("candidate produced non-finite output")


@contextmanager
def _validate_partial(sig_args):
    with _validate_named(sig_args, ("Partial",)):
        yield


@contextmanager
def _validate_reduce(sig_args):
    with _validate_named(sig_args, ("Dst",)):
        yield


@contextmanager
def _validate_quant(sig_args):
    with _validate_named(sig_args, ("Scale",)):
        yield


@flyc.jit
def quantize_int8_tensorwise_partial_direct(
    In: fx.Tensor,
    Partial: fx.Tensor,
    nvec: fx.Int32,
    npartial: fx.Int32,
    dtype_str: fx.Constexpr[str],
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    elem_tail: fx.Constexpr[int] = 0,
    stream: fx.Stream = fx.Stream(None),
):
    launch_partial, _launch_reduce, _launch_quant = build_quantize_int8_tensorwise_module(
        dtype_str, block_threads=BLOCK_THREADS, elem_tail=elem_tail
    )
    launch_partial(In, Partial, nvec, npartial, stream)


@flyc.jit
def quantize_int8_tensorwise_reduce_direct(
    Src: fx.Tensor,
    Dst: fx.Tensor,
    n_in: fx.Int32,
    n_out: fx.Int32,
    dtype_str: fx.Constexpr[str],
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    _launch_partial, launch_reduce, _launch_quant = build_quantize_int8_tensorwise_module(
        dtype_str, block_threads=BLOCK_THREADS
    )
    launch_reduce(Src, Dst, n_in, n_out, stream)


@flyc.jit
def quantize_int8_tensorwise_quant_direct(
    In: fx.Tensor,
    Out: fx.Tensor,
    Amax: fx.Tensor,
    Scale: fx.Tensor,
    nvec: fx.Int32,
    dtype_str: fx.Constexpr[str],
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    elem_tail: fx.Constexpr[int] = 0,
    stream: fx.Stream = fx.Stream(None),
):
    _launch_partial, _launch_reduce, launch_quant = build_quantize_int8_tensorwise_module(
        dtype_str, block_threads=BLOCK_THREADS, elem_tail=elem_tail
    )
    launch_quant(In, Out, Amax, Scale, nvec, stream)


_partial_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["dtype_str", "tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_quantize_int8_tensorwise_partial",
    validate_hook=_validate_partial,
)(quantize_int8_tensorwise_partial_direct)

_reduce_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["dtype_str", "tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_quantize_int8_tensorwise_reduce",
    validate_hook=_validate_reduce,
)(quantize_int8_tensorwise_reduce_direct)

_quant_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["dtype_str", "tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_quantize_int8_tensorwise_quant",
    validate_hook=_validate_quant,
)(quantize_int8_tensorwise_quant_direct)


def _eager_supplied_scale(
    x: torch.Tensor,
    scale: torch.Tensor | float | None,
    *,
    stream: torch.cuda.Stream | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Device quant with a caller-supplied scale: ``roundeven(x * rcp(scale))``."""
    if not isinstance(scale, torch.Tensor):
        scale_t = torch.tensor(scale, device=x.device, dtype=torch.float32)
    else:
        scale_t = scale.to(device=x.device, dtype=torch.float32).reshape(())
    if scale_t.numel() != 1:
        raise ValueError(f"supplied scale must be scalar, got {tuple(scale_t.shape)}")
    # Kernel uses rcp of the signed scale, magnitude clamped at 1e-30.
    # 0-d tensors have no stride-1 axis, so the jit cannot wrap them. The kernel
    # reads one f32 at index 0; a 1-element view is the same storage.
    launch_scale = scale_t.reshape(1)
    if x.device.type != "cuda" or x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("quantize_int8_tensorwise: need CUDA/ROCm f32/f16/bf16")
    if x.numel() == 0:
        raise ValueError("quantize_int8_tensorwise: empty tensor")
    dtype_str = _dtype_name(x.dtype)
    vec = _vec_width(dtype_str)
    flat = ensure_contiguous(x.reshape(-1), stream=stream)
    numel = int(flat.numel())
    elem_tail = numel % vec
    nvec = (numel + vec - 1) // vec
    qpad = torch.empty(numel, dtype=torch.int8, device=x.device)
    _given_tuned(
        flat,
        qpad,
        launch_scale,
        nvec,
        dtype_str=dtype_str,
        tuning_schema=TUNING_SCHEMA,
        elem_tail=elem_tail,
    )
    return qpad[:numel].reshape(x.shape), scale_t


@lru_cache(maxsize=16)
def build_quantize_int8_given_scale_module(
    dtype_str: str, block_threads: int = BLOCK, elem_tail: int = 0
) -> Callable[..., None]:
    """Quantize with a scalar scale already chosen by the caller."""
    if block_threads not in _BLOCK_CHOICES:
        raise ValueError(f"block_threads={block_threads} is not a legal wave32 block")
    elem_dtype, elem_bytes = _elem(dtype_str)
    vec = _vec_width(dtype_str)
    out_bytes = vec
    vecs_per_thread = TILE // block_threads

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def given_scale_kernel(In: fx.Tensor, Out: fx.Tensor, Scale: fx.Tensor, nvec: fx.Int32) -> None:
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        if const_expr(elem_tail != 0):
            in_records = (fx.Int64(nvec) - fx.Int64(1)) * fx.Int64(vec * elem_bytes) + fx.Int64(elem_tail * elem_bytes)
            out_records = (fx.Int64(nvec) - fx.Int64(1)) * fx.Int64(vec) + fx.Int64(elem_tail)
        else:
            in_records = fx.Int64(nvec) * fx.Int64(vec * elem_bytes)
            out_records = fx.Int64(nvec) * fx.Int64(out_bytes)
        in_buf = ptr_buf_tensor(
            In,
            elem=elem_dtype,
            n=0x3FFFFFFF,
            unit_elems=vec,
            num_records_bytes=in_records,
        )
        out_buf = ptr_buf_tensor(
            Out,
            elem=fx.Int8,
            n=0x3FFFFFFF,
            unit_elems=vec,
            num_records_bytes=out_records,
        )
        if const_expr(elem_tail != 0):
            in_elem = ptr_buf_tensor(In, elem=elem_dtype, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=in_records)
            out_elem = ptr_buf_tensor(Out, elem=fx.Int8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=out_records)
        scale_buf = ptr_buf_tensor(Scale, elem=fx.Float32, n=1, unit_elems=1, num_records_bytes=4)
        raw_scale = fx.Float32(buf_copy_load(scale_buf, fx.Int32(0), elem=fx.Float32, unit_elems=1))
        mag = fx.max(fmath.absf(raw_scale), fx.Float32(1e-30))
        # Keep the caller sign. abs() would quantize x and -x to the same codes.
        signed = (raw_scale < fx.Float32(0.0)).select(-mag, mag)
        inv = fx.Float32(fx.rocdl.rcp(T.f32, signed))
        unit_base = fx.Int32(bid) * fx.Int32(TILE)
        for step in range_constexpr(vecs_per_thread):
            unit = unit_base + fx.Int32(tid) + fx.Int32(step * block_threads)
            in_range = unit < nvec
            if const_expr(elem_tail != 0):
                if unit == (fx.Int32(nvec) - fx.Int32(1)):
                    base = unit * fx.Int32(vec)
                    for ei in range_constexpr(elem_tail):
                        idx = base + fx.Int32(ei)
                        one = fx.Float32(buf_copy_load(in_elem, idx, elem=elem_dtype, unit_elems=1))
                        qf = fmath.roundeven(one * inv)
                        qf = fx.max(fx.min(qf, fx.Float32(127.0)), fx.Float32(-128.0))
                        buf_copy_store(out_elem, idx, qf.to(fx.Int8), elem=fx.Int8, unit_elems=1)
                elif in_range:
                    vec_v = buf_copy_load(in_buf, unit, elem=elem_dtype, unit_elems=vec)
                    xv = vec_v.to(fx.Float32)
                    qs = []
                    for ei in range_constexpr(vec):
                        qf = fmath.roundeven(xv[ei] * inv)
                        qf = fx.max(fx.min(qf, fx.Float32(127.0)), fx.Float32(-128.0))
                        qs.append(qf.to(fx.Int8))
                    buf_copy_store(out_buf, unit, fx.Vector.from_elements(qs, fx.Int8), elem=fx.Int8, unit_elems=vec)
            elif in_range:
                vec_v = buf_copy_load(in_buf, unit, elem=elem_dtype, unit_elems=vec)
                xv = vec_v.to(fx.Float32)
                qs = []
                for ei in range_constexpr(vec):
                    qf = fmath.roundeven(xv[ei] * inv)
                    qf = fx.max(fx.min(qf, fx.Float32(127.0)), fx.Float32(-128.0))
                    qs.append(qf.to(fx.Int8))
                buf_copy_store(out_buf, unit, fx.Vector.from_elements(qs, fx.Int8), elem=fx.Int8, unit_elems=vec)

    given_scale_kernel.__name__ = f"{KERNEL_NAME}_given_scale_{dtype_str}_b{block_threads}"

    @flyc.jit
    def launch(
        In: fx.Tensor,
        Out: fx.Tensor,
        Scale: fx.Tensor,
        nvec: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        grid_x = (fx.Int64(nvec) + fx.Int64(TILE - 1)) // fx.Int64(TILE)
        given_scale_kernel(In, Out, Scale, nvec).launch(grid=(grid_x, 1, 1), block=(block_threads, 1, 1), stream=stream)

    launch.__name__ = f"launch_{KERNEL_NAME}_given_scale_{dtype_str}_b{block_threads}"
    return launch


@flyc.jit
def quantize_int8_given_scale_direct(
    In: fx.Tensor,
    Out: fx.Tensor,
    Scale: fx.Tensor,
    nvec: fx.Int32,
    dtype_str: fx.Constexpr[str],
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    elem_tail: fx.Constexpr[int] = 0,
    stream: fx.Stream = fx.Stream(None),
):
    launch = build_quantize_int8_given_scale_module(dtype_str, block_threads=BLOCK_THREADS, elem_tail=elem_tail)
    launch(In, Out, Scale, nvec, stream)


@contextmanager
def _validate_given(sig_args):
    """Reject candidates that disagree with host given-scale int8 math.

    Int8 has no NaN encoding, so a finite-cast check on ``Out`` is a tautology.
    Compare against round-even clamp of ``In / Scale`` on the live launch shape.
    """
    import torch

    out = sig_args["Out"]
    inp = sig_args["In"]
    scale = sig_args["Scale"]
    out.fill_(0)
    yield
    if not out.numel():
        return
    scale_f = float(scale.reshape(-1)[0].item())
    inv = 1.0 / max(scale_f, 1e-30)
    # Match kernel: roundeven + clamp to int8 range, compare only the live elements.
    n = min(int(out.numel()), int(inp.numel()))
    ref = torch.round(inp.reshape(-1)[:n].float() * inv).clamp(-128, 127).to(torch.int8)
    got = out.reshape(-1)[:n]
    if not torch.equal(got, ref):
        raise ValueError("candidate disagreed with host int8 given-scale reference")


_given_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["dtype_str", "tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_quantize_int8_tensorwise_given_scale",
    validate_hook=_validate_given,
)(quantize_int8_given_scale_direct)


def quantize_int8_tensorwise(
    x: torch.Tensor,
    scale: Union[torch.Tensor, float, str, None] = None,
    stochastic_rounding: Optional[int] = 0,
    *,
    stream: torch.cuda.Stream | None = None,
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
        return _eager_supplied_scale(x, scale, stream=stream)
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
        flat = ensure_contiguous(flat, stream=stream)
    numel = int(flat.numel())
    elem_tail = numel % vec
    nvec = (numel + vec - 1) // vec
    npartial = (nvec + TILE - 1) // TILE
    qpad = torch.empty(numel, dtype=torch.int8, device=x.device)
    partial = torch.empty(npartial, dtype=torch.float32, device=x.device)
    scale_buf = torch.empty((1,), dtype=torch.float32, device=x.device)
    _partial_tuned(flat, partial, nvec, npartial, dtype_str=dtype_str, tuning_schema=TUNING_SCHEMA, elem_tail=elem_tail)
    curr = partial
    n = npartial
    while n > 1:
        n2 = (n + TILE - 1) // TILE
        nxt = torch.empty(n2, dtype=torch.float32, device=x.device)
        _reduce_tuned(curr, nxt, n, n2, dtype_str=dtype_str, tuning_schema=TUNING_SCHEMA)
        curr = nxt
        n = n2
    _quant_tuned(
        flat, qpad, curr, scale_buf, nvec, dtype_str=dtype_str, tuning_schema=TUNING_SCHEMA, elem_tail=elem_tail
    )
    return qpad[:numel].reshape(x.shape), scale_buf.reshape(())


def dequantize_int8_tensorwise(
    q: torch.Tensor,
    scale: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Inverse of ``quantize_int8_tensorwise``: ``q * scale``."""
    require_gfx120x(q.device, what="dequantize_int8_tensorwise (gfx120x)")
    if q.dtype != torch.int8:
        raise ValueError(f"q must be int8, got {q.dtype}")
    names = {
        torch.bfloat16: (fx.BFloat16, 2),
        torch.float16: (fx.Float16, 2),
        torch.float32: (fx.Float32, 4),
    }
    if out_dtype not in names:
        raise ValueError(f"out_dtype must be bf16, fp16, or fp32, got {out_dtype}")
    flat = ensure_contiguous(q.reshape(-1), stream=stream)
    n = int(flat.numel())
    scale_v = ensure_contiguous(scale.to(device=q.device, dtype=torch.float32).reshape(1), stream=stream)
    out = torch.empty(q.shape, device=q.device, dtype=out_dtype)
    out_ty, out_bytes = names[out_dtype]
    block = 256

    @flyc.kernel(known_block_size=[block, 1, 1])
    def dequant_kernel(Q: fx.Pointer, Scale: fx.Pointer, Out: fx.Pointer, n_elems: fx.Int32) -> None:
        idx = fx.block_idx.x * fx.Int32(block) + fx.thread_idx.x
        q_buf = ptr_buf_tensor(Q, elem=fx.Int8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_elems))
        s_buf = ptr_buf_tensor(Scale, elem=fx.Float32, n=1, unit_elems=1, num_records_bytes=4)
        o_buf = ptr_buf_tensor(
            Out, elem=out_ty, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_elems) * fx.Int64(out_bytes)
        )
        if idx < n_elems:
            raw = fx.Int32(buf_copy_load(q_buf, idx, elem=fx.Int8, unit_elems=1)) & fx.Int32(255)
            raw = (raw >= fx.Int32(128)).select(raw - fx.Int32(256), raw)
            val = fx.Float32(raw) * fx.Float32(buf_copy_load(s_buf, fx.Int32(0), elem=fx.Float32, unit_elems=1))
            stored = val if out_dtype == torch.float32 else val.to(out_ty)
            buf_copy_store(o_buf, idx, stored, elem=out_ty, unit_elems=1)

    @flyc.jit
    def launch(
        Q: fx.Pointer, Scale: fx.Pointer, Out: fx.Pointer, n_elems: fx.Int32, stream: fx.Stream = fx.Stream(None)
    ) -> None:
        grid = (fx.Int64(n_elems) + fx.Int64(block - 1)) // fx.Int64(block)
        dequant_kernel(Q, Scale, Out, n_elems).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    launch(flat, scale_v, out.reshape(-1), n, stream if stream is not None else torch.cuda.current_stream())
    return out
