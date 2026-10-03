# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""AdaLN / RMS-AdaLN fused modulation for gfx120x (RDNA4).

Fuses RMS or LayerNorm with affine modulation from a conditioning vector
(shift/scale), matching the DiT-style AdaLN pattern. Specialization knobs:

* ``subtract_mean=True`` -- LayerNorm AdaLN (center then scale)
* ``subtract_mean=False`` -- RMS AdaLN (scale only)

Block thread count grows with hidden size N (256 / 512 / 1024). This module
is FlyDSL-native; call the builders here directly (no third-party attention/norm package).
"""

import math
from collections.abc import Callable
from functools import lru_cache
from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from kernels.common.gfx120x_buf_helpers import kernel_signature

KERNEL_NAME = "adaln_gfx120x"
WARP = 32


def _block_threads(n: int) -> int:
    """Measured gfx1201 BT gate (2026-09-30) — see gfx120x_autotune_tables."""
    from kernels.common.gfx120x_autotune_tables import pick_adaln_block_threads

    return pick_adaln_block_threads(n)


@lru_cache(maxsize=64)
def build_adaln_module(
    N: int, dtype_str: str, subtract_mean: bool, block_threads: Optional[int] = None
) -> Callable[..., None]:
    """Specialize fused AdaLN kernel on (N, dtype, subtract_mean)."""

    if block_threads is None:
        block_threads = _block_threads(N)
    sig = kernel_signature(
        n=N,
        dtype=dtype_str,
        subtract_mean=int(subtract_mean),
        block=block_threads,
        op="adaln",
    )
    elem_dtype, elem_bits = {
        "float32": (fx.Float32, 32),
        "float16": (fx.Float16, 16),
        "bfloat16": (fx.BFloat16, 16),
    }[dtype_str]
    vec_width = 128 // elem_bits
    if N % vec_width != 0:
        raise ValueError(f"N={N} not divisible by vec={vec_width}")
    reduction_slots = (block_threads + WARP - 1) // WARP
    if reduction_slots > WARP:
        raise AssertionError("reduction_slots > WARP")
    slots_total = reduction_slots * 2  # sumsq + sum (LN); RMS uses first half only
    full_vecs = N // vec_width
    vec_steps = (full_vecs + block_threads - 1) // block_threads
    n_float = float(N)
    do_mean = bool(subtract_mean)

    @fx.struct
    class SharedStorage:
        reduction_buffer: fx.Array[fx.Float32, slots_total, 16]

    def _load_vec(copy_atom: object, div_tensor: fx.Tensor, idx: fx.Int32 | int) -> fx.Vector:
        r = fx.make_rmem_tensor(vec_width, elem_dtype)
        fx.copy(copy_atom, div_tensor[None, idx], r)
        return r.load()

    def _store_vec(copy_atom: object, val: fx.Vector, div_tensor: fx.Tensor, idx: fx.Int32 | int) -> None:
        r = fx.make_rmem_tensor(vec_width, elem_dtype)
        r.store(val)
        fx.copy(copy_atom, r, div_tensor[None, idx])

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def adaln_kernel(
        Input: fx.Tensor,
        Scale: fx.Tensor,
        Shift: fx.Tensor,
        Output: fx.Tensor,
        ScaleGroup: fx.Int32,
        ShiftGroup: fx.Int32,
        Eps: fx.Float32,
    ) -> None:
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        eps_c = Eps
        c0 = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        red = lds.reduction_buffer.view(fx.make_layout(slots_total, 1))

        def wave_reduce_add(val: object) -> fx.Float32:
            w = val
            for _sh in range_constexpr(int(math.log2(WARP))):
                off = WARP // (2 << _sh)
                w += gpu.shuffle_xor(w, off, WARP)
            return w

        def block_reduce_add(val: object, base: object) -> fx.Float32:
            lane = tid % WARP
            wave = tid // WARP
            w = wave_reduce_add(val)
            if lane == 0:
                fx.memref_store(w, red, base + wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < reduction_slots
                lane_safe = in_range.select(lane, 0)
                v = red[base + lane_safe]
                ww = in_range.select(v, 0.0)
                ww = wave_reduce_add(ww)
                if lane == 0:
                    fx.memref_store(ww, red, base)
            gpu.barrier()
            return fx.Float32(red[base])

        In_buf = fx.rocdl.make_buffer_tensor(Input)
        Sc_buf = fx.rocdl.make_buffer_tensor(Scale)
        Sh_buf = fx.rocdl.make_buffer_tensor(Shift)
        Out_buf = fx.rocdl.make_buffer_tensor(Output)

        row_in = In_buf[bid, None]
        row_sc = Sc_buf[bid // ScaleGroup, None]
        row_sh = Sh_buf[bid // ShiftGroup, None]
        row_out = Out_buf[bid, None]

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        in_div = fx.logical_divide(row_in, fx.make_layout(vec_width, 1))
        sc_div = fx.logical_divide(row_sc, fx.make_layout(vec_width, 1))
        sh_div = fx.logical_divide(row_sh, fx.make_layout(vec_width, 1))
        out_div = fx.logical_divide(row_out, fx.make_layout(vec_width, 1))

        thread_sum = c0
        thread_sumsq = c0
        in_local = []

        for step in range_constexpr(vec_steps):
            vec_idx = tid + step * block_threads
            is_valid = vec_idx < full_vecs
            vec_idx_safe = is_valid.select(vec_idx, 0)
            vec = _load_vec(copy_atom, in_div, vec_idx_safe)
            in_local.append(vec)
            xv = vec.to(fx.Float32)
            for ei in range_constexpr(vec_width):
                xe = xv[ei]
                thread_sumsq = thread_sumsq + is_valid.select(xe * xe, c0)
                if const_expr(do_mean):
                    thread_sum = thread_sum + is_valid.select(xe, c0)

        sumsq = block_reduce_add(thread_sumsq, 0)
        if const_expr(do_mean):
            sumv = block_reduce_add(thread_sum, reduction_slots)
            mean = sumv / fx.Float32(n_float)
            var = sumsq / fx.Float32(n_float) - mean * mean
            var = fx.max(var, c0)
            rstd = fmath.rsqrt(var + eps_c, fastmath="fast")
        else:
            mean = c0
            rstd = fmath.rsqrt(sumsq / fx.Float32(n_float) + eps_c, fastmath="fast")

        one = fx.Float32(1.0)
        for step in range_constexpr(vec_steps):
            vec_idx = tid + step * block_threads
            if vec_idx < full_vecs:
                scv = _load_vec(copy_atom, sc_div, vec_idx).to(fx.Float32)
                shv = _load_vec(copy_atom, sh_div, vec_idx).to(fx.Float32)
                xv = in_local[step].to(fx.Float32)
                outs = []
                for ei in range_constexpr(vec_width):
                    y = (xv[ei] - mean) * rstd * (one + scv[ei]) + shv[ei]
                    if const_expr(dtype_str == "float32"):
                        outs.append(y)
                    else:
                        outs.append(y.to(elem_dtype))
                _store_vec(
                    copy_atom,
                    fx.Vector.from_elements(outs, elem_dtype),
                    out_div,
                    vec_idx,
                )

    @flyc.jit
    def launch(
        Input: fx.Tensor,
        Scale: fx.Tensor,
        Shift: fx.Tensor,
        Output: fx.Tensor,
        rows: fx.Int32,
        scale_group: fx.Int32,
        shift_group: fx.Int32,
        eps: fx.Float32,
        stream: fx.Stream,
    ) -> None:
        adaln_kernel(Input, Scale, Shift, Output, scale_group, shift_group, eps).launch(
            grid=(rows, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    adaln_kernel.__name__ = f"{KERNEL_NAME}_{sig}"
    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch
