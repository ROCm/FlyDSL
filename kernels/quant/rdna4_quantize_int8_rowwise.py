# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Rowwise absmax INT8 quantize for gfx120x (``quantize_int8_rowwise``).

Does **not** require the gfx120x iu8 WMMA atom. Scale uses ``rocdl.rcp`` to
match HIP ``amdgcn_rcpf`` (INT8 scale exception). On R9700 (idle microbench): q bit-identical;
scale_maxdiff=0; tiny ~1.13x / large ~3.63x vs HIP.

Public: ``build_quantize_int8_rowwise_module``, ``quantize_int8_rowwise``.
"""

import math
from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import Config, autotune
from kernels.common.gfx120x_arch import require_gfx120x

KERNEL_NAME = "rdna4_quantize_int8_rowwise"
WARP = 32
BLOCK = 256
TUNING_SCHEMA = 1
_BLOCK_CHOICES = (32, 64, 128, 256, 512, 1024)


def _block_threads(k: int) -> int:
    """Measured gfx1201 BT gate (2026-09-30) — see gfx120x_autotune_tables."""
    from kernels.common.gfx120x_autotune_tables import pick_quantize_int8_rowwise_block_threads

    return pick_quantize_int8_rowwise_block_threads(k)


@lru_cache(maxsize=64)
def build_quantize_int8_rowwise_module(
    K: int, dtype_str: str, block_threads: Optional[int] = None
) -> Callable[..., None]:
    """Build the rowwise absmax int8 quant kernel."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import const_expr, gpu, range_constexpr
    from flydsl.expr import math as fmath
    from flydsl.expr.typing import T
    from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor

    if block_threads is None:
        block_threads = _block_threads(K)
    if block_threads not in _BLOCK_CHOICES:
        raise ValueError(f"block_threads={block_threads} is not a legal wave32 block")
    elem_dtype, elem_bits = {
        "float32": (fx.Float32, 32),
        "float16": (fx.Float16, 16),
        "bfloat16": (fx.BFloat16, 16),
    }[dtype_str]
    vec_width = 128 // elem_bits
    if K < 0:
        raise ValueError(f"K={K} must be non-negative")
    # int8 pack: store vec_width i8 via 64b (bf16/f16 vec=8) or 128b (f32 vec=4)
    # bf16: 8 i8 = 64-bit → BufferCopy64b unit 8
    # f32: 4 i8 = 32-bit → BufferCopy32b
    out_copy_bits = vec_width * 8
    reduction_slots = (block_threads + WARP - 1) // WARP
    if reduction_slots > WARP:
        raise AssertionError("reduction_slots > WARP")
    full_vecs = K // vec_width
    vec_steps = (full_vecs + block_threads - 1) // block_threads if K > 0 else 0
    tail = K - full_vecs * vec_width
    scalar_steps = 0 if tail == 0 else (K + block_threads - 1) // block_threads
    INV_127 = 1.0 / 127.0

    RedTy = fx.Array[fx.Float32, reduction_slots, 16]

    @fx.struct
    class SharedStorage:
        reduction_buffer: RedTy

    def _load_vec(copy_atom: object, div_tensor: fx.Tensor, idx: fx.Int32 | int) -> fx.Vector:
        r = fx.make_rmem_tensor(vec_width, elem_dtype)
        fx.copy(copy_atom, div_tensor[None, idx], r)
        return r.load()

    if out_copy_bits == 64:
        CopyOutAtom = fx.rocdl.BufferCopy64b
    elif out_copy_bits == 32:
        CopyOutAtom = fx.rocdl.BufferCopy32b
    else:
        CopyOutAtom = fx.rocdl.BufferCopy128b

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def rowwise_kernel(
        Input: fx.Tensor,
        OutQ: fx.Tensor,
        OutScale: fx.Tensor,
    ) -> None:
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        c0 = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        red = lds.reduction_buffer.view(fx.make_layout(reduction_slots, 1))

        def wave_reduce_max(val: object) -> fx.Float32:
            w = val
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
                in_range = lane < reduction_slots
                lane_safe = in_range.select(lane, 0)
                v = red[lane_safe]
                ww = in_range.select(v, c0)
                ww = wave_reduce_max(ww)
                if lane == 0:
                    fx.memref_store(ww, red, 0)
            gpu.barrier()
            return fx.Float32(red[0])

        In_buf = fx.rocdl.make_buffer_tensor(Input)
        Q_buf = fx.rocdl.make_buffer_tensor(OutQ)
        Sc_buf = fx.rocdl.make_buffer_tensor(OutScale)

        row_in = In_buf[bid, None]
        row_q = Q_buf[bid, None]

        thread_amax = c0
        in_local = []
        if const_expr(tail == 0 and K > 0):
            copy_in = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_out = fx.make_copy_atom(CopyOutAtom(), 8)
            in_div = fx.logical_divide(row_in, fx.make_layout(vec_width, 1))
            q_div = fx.logical_divide(row_q, fx.make_layout(vec_width, 1))
            for step in range_constexpr(vec_steps):
                vec_idx = tid + step * block_threads
                is_valid = vec_idx < full_vecs
                safe = is_valid.select(vec_idx, 0)
                vec = _load_vec(copy_in, in_div, safe)
                in_local.append(vec)
                xv = vec.to(fx.Float32)
                for ei in range_constexpr(vec_width):
                    ae = fmath.absf(xv[ei])
                    thread_amax = fx.max(thread_amax, is_valid.select(ae, c0))
        else:
            in_s = ptr_buf_tensor(Input, elem=elem_dtype, unit_elems=1)
            for step in range_constexpr(scalar_steps):
                col = tid + fx.Int32(step * block_threads)
                inb = col < fx.Int32(K)
                raw = buf_copy_load(
                    in_s,
                    inb.select(fx.Int64(bid) * fx.Int64(K) + fx.Int64(col), fx.Int64(0)),
                    elem=elem_dtype,
                    unit_elems=1,
                )
                if const_expr(dtype_str != "float32"):
                    xv_t = fx.Float32(elem_dtype(raw).to(fx.Float32))
                else:
                    xv_t = fx.Float32(raw)
                thread_amax = fx.max(thread_amax, inb.select(fmath.absf(xv_t), c0))

        amax = block_reduce_max(thread_amax)
        # Match HIP binary: scale = max(amax/127, 1e-30); inv = amdgcn_rcpf(scale)
        # (HIP lowers `1.0f/scale` to approx v_rcp_f32 — IEEE div breaks bf16/f16 half-ties).
        scale = fx.max(amax * fx.Float32(INV_127), fx.Float32(1e-30))
        inv = fx.Float32(fx.rocdl.rcp(T.f32, scale))
        if tid == 0:
            sc_r = fx.make_rmem_tensor(1, fx.Float32)
            sc_r.store(fx.Vector.from_elements([scale], fx.Float32))
            sc_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
            sc_div = fx.logical_divide(Sc_buf, fx.make_layout(1, 1))
            fx.copy(sc_atom, sc_r, sc_div[None, bid])

        gpu.barrier()

        if const_expr(tail == 0 and K > 0):
            for step in range_constexpr(vec_steps):
                vec_idx = tid + step * block_threads
                if vec_idx < full_vecs:
                    xv = in_local[step].to(fx.Float32)
                    qs = []
                    for ei in range_constexpr(vec_width):
                        # HIP: round-half-to-even (banker's); no fastmath
                        qf = fmath.roundeven(xv[ei] * inv)
                        qf = fx.max(fx.min(qf, fx.Float32(127.0)), fx.Float32(-128.0))
                        qs.append(qf.to(fx.Int8))
                    r = fx.make_rmem_tensor(vec_width, fx.Int8)
                    r.store(fx.Vector.from_elements(qs, fx.Int8))
                    fx.copy(copy_out, r, q_div[None, vec_idx])
        else:
            in_s = ptr_buf_tensor(Input, elem=elem_dtype, unit_elems=1)
            q_s = ptr_buf_tensor(OutQ, elem=fx.Int8, unit_elems=1)
            for step in range_constexpr(scalar_steps):
                col = tid + fx.Int32(step * block_threads)
                if col < fx.Int32(K):
                    raw = buf_copy_load(
                        in_s,
                        fx.Int64(bid) * fx.Int64(K) + fx.Int64(col),
                        elem=elem_dtype,
                        unit_elems=1,
                    )
                    if const_expr(dtype_str != "float32"):
                        xv_t = fx.Float32(elem_dtype(raw).to(fx.Float32))
                    else:
                        xv_t = fx.Float32(raw)
                    qf = fmath.roundeven(xv_t * inv)
                    qf = fx.max(fx.min(qf, fx.Float32(127.0)), fx.Float32(-128.0))
                    buf_copy_store(
                        q_s,
                        fx.Int64(bid) * fx.Int64(K) + fx.Int64(col),
                        qf.to(fx.Int8),
                        elem=fx.Int8,
                        unit_elems=1,
                    )

    @flyc.jit
    def launch(
        Input: fx.Tensor,
        OutQ: fx.Tensor,
        OutScale: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        rowwise_kernel(Input, OutQ, OutScale).launch(grid=(rows, 1, 1), block=(block_threads, 1, 1), stream=stream)

    return launch


# Back-compat alias for older call sites.
build_int8_quant_module = build_quantize_int8_rowwise_module


# ---------------------------------------------------------------------------
# Host API
# ---------------------------------------------------------------------------

from typing import Optional, Tuple  # noqa: E402

import torch  # noqa: E402


def _dtype_name(dt: torch.dtype) -> str:
    return {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }[dt]


def _layout_ok(x: torch.Tensor) -> bool:
    if x.device.type != "cuda" or x.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        return False
    return x.ndim >= 1


def _default_block(*_args, **_kwargs) -> Config:
    return Config(BLOCK_THREADS=128)


def _check_out_scale(scale, rows: int) -> None:
    """Kernel writes one f32 per row into a rank-1 ``[rows]`` buffer."""
    import torch

    if not isinstance(scale, torch.Tensor):
        return
    if scale.ndim != 1 or int(scale.numel()) != int(rows):
        raise ValueError(f"OutScale must be rank-1 float32 [rows={rows}], got shape={tuple(scale.shape)}")
    if scale.dtype != torch.float32:
        raise ValueError(f"OutScale must be float32, got {scale.dtype}")


@contextmanager
def _validate_rowwise(sig_args):
    import torch

    scale = sig_args["OutScale"]
    rows = int(sig_args["Input"].shape[0]) if hasattr(sig_args["Input"], "shape") else int(scale.numel())
    _check_out_scale(scale, rows)
    scale.fill_(float("nan"))
    yield
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("candidate produced non-finite output")


@flyc.jit
def quantize_int8_rowwise_direct(
    Input: fx.Tensor,
    OutQ: fx.Tensor,
    OutScale: fx.Tensor,
    rows: fx.Int32,
    K: fx.Constexpr[int],
    dtype_str: fx.Constexpr[str],
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    """Direct JIT entry. ``tuning_schema`` partitions the autotune cache."""
    launch = build_quantize_int8_rowwise_module(K, dtype_str, block_threads=BLOCK_THREADS)
    launch(Input, OutQ, OutScale, rows, stream)


_quantize_int8_rowwise_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["K", "dtype_str", "tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_quantize_int8_rowwise",
    validate_hook=_validate_rowwise,
)(quantize_int8_rowwise_direct)


def quantize_int8_rowwise(
    x: torch.Tensor,
    stochastic_rounding: Optional[int] = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rowwise absmax int8 quant → (q, scale).

    Kernel scratch for scales is rank-1 ``[rows]``; the returned ``scale`` is
    reshaped to ``(*x.shape[:-1], 1)`` (HIP-compatible trailing singleton).
    Direct/AOT callers must pass ``OutScale`` as float32 ``[rows]`` (not ``[rows, 1]``).
    """
    require_gfx120x(x.device, what="quantize_int8_rowwise (gfx120x)")
    if stochastic_rounding:
        raise ValueError("quantize_int8_rowwise: stochastic_rounding not supported")
    if not _layout_ok(x):
        raise ValueError("quantize_int8_rowwise: unsupported layout/dtype/K")
    x_shape = x.shape
    k = int(x_shape[-1])
    try:
        x2d = x.reshape(-1, k)
    except RuntimeError:
        x2d = x.contiguous().reshape(-1, k)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    rows = int(x2d.shape[0])
    q = torch.empty((rows, k), dtype=torch.int8, device=x.device)
    scales = torch.empty((rows,), dtype=torch.float32, device=x.device)
    _quantize_int8_rowwise_tuned(
        x2d,
        q,
        scales,
        rows,
        K=k,
        dtype_str=_dtype_name(x.dtype),
        tuning_schema=TUNING_SCHEMA,
    )
    return q.reshape(x_shape), scales.reshape(*x_shape[:-1], 1)


def dequantize_int8_rowwise(
    q: torch.Tensor,
    scale: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    stream: Optional[torch.cuda.Stream] = None,
) -> tuple:
    """Inverse of ``quantize_int8_rowwise``: ``q * scale`` per row."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(q.device, what="dequantize_int8_rowwise (gfx120x)")
    if q.dtype != torch.int8:
        raise ValueError(f"q must be int8, got {q.dtype}")
    names = {
        torch.bfloat16: ("bfloat16", fx.BFloat16, 2),
        torch.float16: ("float16", fx.Float16, 2),
        torch.float32: ("float32", fx.Float32, 4),
    }
    if out_dtype not in names:
        raise ValueError(f"out_dtype must be bf16, fp16, or fp32, got {out_dtype}")
    k = int(q.shape[-1])
    q2 = ensure_contiguous(q.reshape(-1, k), stream=stream)
    rows = int(q2.shape[0])
    scale_v = ensure_contiguous(scale.to(device=q.device, dtype=torch.float32).reshape(-1), stream=stream)
    if scale_v.numel() != rows:
        raise ValueError(f"scale must have {rows} rows, got {scale_v.numel()}")
    out = torch.empty(q.shape, device=q.device, dtype=out_dtype)
    out2 = out.reshape(rows, k)
    _, out_ty, out_bytes = names[out_dtype]
    block = 256

    @flyc.kernel(known_block_size=[block, 1, 1])
    def dequant_kernel(Q: fx.Pointer, Scale: fx.Pointer, Out: fx.Pointer, n_rows: fx.Int32) -> None:
        idx = fx.block_idx.x * fx.Int32(block) + fx.thread_idx.x
        n = n_rows * fx.Int32(k)
        q_buf = ptr_buf_tensor(Q, elem=fx.Int8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n))
        s_buf = ptr_buf_tensor(
            Scale, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_rows) * fx.Int64(4)
        )
        o_buf = ptr_buf_tensor(
            Out, elem=out_ty, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n) * fx.Int64(out_bytes)
        )
        if idx < n:
            row = idx // fx.Int32(k)
            raw = fx.Int32(buf_copy_load(q_buf, idx, elem=fx.Int8, unit_elems=1)) & fx.Int32(255)
            raw = (raw >= fx.Int32(128)).select(raw - fx.Int32(256), raw)
            val = fx.Float32(raw) * fx.Float32(buf_copy_load(s_buf, row, elem=fx.Float32, unit_elems=1))
            stored = val if out_dtype == torch.float32 else val.to(out_ty)
            buf_copy_store(o_buf, idx, stored, elem=out_ty, unit_elems=1)

    @flyc.jit
    def launch(
        Q: fx.Pointer, Scale: fx.Pointer, Out: fx.Pointer, n_rows: fx.Int32, stream: fx.Stream = fx.Stream(None)
    ) -> None:
        n = fx.Int64(n_rows) * fx.Int64(k)
        grid = (n + fx.Int64(block - 1)) // fx.Int64(block)
        dequant_kernel(Q, Scale, Out, n_rows).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    launch(q2, scale_v, out2, rows, stream if stream is not None else torch.cuda.current_stream())
    return out
