# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Rowwise absmax INT8 quantize for gfx120x (``quantize_int8_rowwise``).

Does **not** require the gfx120x iu8 WMMA atom. Scale uses ``rocdl.rcp`` to
match HIP ``amdgcn_rcpf`` (INT8 scale exception). On R9700 (idle microbench): q bit-identical;
scale_maxdiff=0; tiny ~1.13x / large ~3.63x vs HIP.

Public: ``build_quantize_int8_rowwise_module``, ``quantize_int8_rowwise``.
"""

import math
from collections.abc import Callable  # noqa: E402
from functools import lru_cache
from typing import Optional

from kernels.common.gfx120x_arch import require_gfx120x

KERNEL_NAME = "rdna4_quantize_int8_rowwise"
WARP = 32
BLOCK = 256


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
    from flydsl.expr import gpu, range_constexpr
    from flydsl.expr import math as fmath
    from flydsl.expr.typing import T

    if block_threads is None:
        block_threads = _block_threads(K)
    elem_dtype, elem_bits = {
        "float32": (fx.Float32, 32),
        "float16": (fx.Float16, 16),
        "bfloat16": (fx.BFloat16, 16),
    }[dtype_str]
    vec_width = 128 // elem_bits
    if K % vec_width != 0:
        raise ValueError(f"K={K} bad for vec={vec_width}")
    # int8 pack: store vec_width i8 via 64b (bf16/f16 vec=8) or 128b (f32 vec=4 → use 32b×1? )
    # bf16: 8 i8 = 64-bit → BufferCopy64b unit 8
    # f32: 4 i8 = 32-bit → BufferCopy32b
    out_copy_bits = vec_width * 8
    reduction_slots = (block_threads + WARP - 1) // WARP
    if reduction_slots > WARP:
        raise AssertionError("reduction_slots > WARP")
    full_vecs = K // vec_width
    vec_steps = (full_vecs + block_threads - 1) // block_threads
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

        copy_in = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        copy_out = fx.make_copy_atom(CopyOutAtom(), 8)

        in_div = fx.logical_divide(row_in, fx.make_layout(vec_width, 1))
        q_div = fx.logical_divide(row_q, fx.make_layout(vec_width, 1))

        thread_amax = c0
        in_local = []
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

    @flyc.jit
    def launch(
        Input: fx.Tensor,
        OutQ: fx.Tensor,
        OutScale: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
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

from kernels.common.tensor_shim import _run_compiled  # noqa: E402


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
    if x.ndim < 1:
        return False
    k = x.shape[-1]
    bits = {torch.float32: 32, torch.float16: 16, torch.bfloat16: 16}[x.dtype]
    vec = 128 // bits
    if k % vec != 0:
        return False
    if x.stride(-1) != 1:
        return False
    return True


def quantize_int8_rowwise(
    x: torch.Tensor,
    stochastic_rounding: Optional[int] = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rowwise absmax int8 quant → (q, scale). scale shape (*x.shape[:-1], 1)."""
    require_gfx120x(x.device, what="quantize_int8_rowwise (gfx120x)")
    if stochastic_rounding:
        raise ValueError("quantize_int8_rowwise: stochastic_rounding not supported")
    if not _layout_ok(x):
        raise ValueError("quantize_int8_rowwise: unsupported layout/dtype/K")
    x_shape = x.shape
    k = x_shape[-1]
    x2d = x.reshape(-1, k)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    rows = x2d.shape[0]
    q = torch.empty((rows, k), dtype=torch.int8, device=x.device)
    scales = torch.empty((rows,), dtype=torch.float32, device=x.device)
    launch = build_quantize_int8_rowwise_module(k, _dtype_name(x.dtype))
    stream = torch.cuda.current_stream(device=x.device)
    _run_compiled(launch, x2d, q, scales, rows, stream)
    return q.reshape(x_shape), scales.reshape(*x_shape[:-1], 1)
