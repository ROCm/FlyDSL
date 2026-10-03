# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Fused RMSNorm + RoPE for gfx120x (RDNA4).

Applies RMSNorm over the last dimension, then rotary embeddings on the Q/K
layout expected by attention. Fusing the two passes saves a global memory
round-trip versus separate RMS and RoPE launches.

Supports split-half and interleaved pair layouts via host specialization.
Measured wins come from removing the intermediate write on gfx120x
activation shapes used by DiT / video diffusion workloads.
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

KERNEL_NAME = "rms_rope_gfx120x"
WARP = 32


def _block_threads(hd: int) -> int:
    """Measured gfx1201 BT gate (2026-09-30) — see gfx120x_autotune_tables."""
    from kernels.common.gfx120x_autotune_tables import pick_rms_rope_block_threads

    return pick_rms_rope_block_threads(hd)


@lru_cache(maxsize=64)
def build_rms_rope_module(HD: int, dtype_str: str, block_threads: Optional[int] = None) -> Callable[..., None]:
    """Build the RMSNorm-then-RoPE kernel for one tensor."""
    if block_threads is None:
        block_threads = _block_threads(HD)
    sig = kernel_signature(hd=HD, dtype=dtype_str, block=block_threads, op="rms_rope")
    elem_dtype, elem_bits = {
        "float32": (fx.Float32, 32),
        "float16": (fx.Float16, 16),
        "bfloat16": (fx.BFloat16, 16),
    }[dtype_str]
    vec_width = 128 // elem_bits
    if HD % vec_width != 0 or HD % 2 != 0:
        raise ValueError(f"HD={HD} bad for vec={vec_width}")
    reduction_slots = (block_threads + WARP - 1) // WARP
    if reduction_slots > WARP:
        raise AssertionError("reduction_slots > WARP")
    full_vecs = HD // vec_width
    vec_steps = (full_vecs + block_threads - 1) // block_threads
    n_float = float(HD)

    @fx.struct
    class SharedStorage:
        reduction_buffer: fx.Array[fx.Float32, reduction_slots, 16]

    def _load_vec(copy_atom: object, div_tensor: fx.Tensor, idx: fx.Int32 | int) -> fx.Vector:
        r = fx.make_rmem_tensor(vec_width, elem_dtype)
        fx.copy(copy_atom, div_tensor[None, idx], r)
        return r.load()

    def _store_vec(copy_atom: object, val: fx.Vector, div_tensor: fx.Tensor, idx: fx.Int32 | int) -> None:
        r = fx.make_rmem_tensor(vec_width, elem_dtype)
        r.store(val)
        fx.copy(copy_atom, r, div_tensor[None, idx])

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def rms_rope_kernel(
        Input: fx.Tensor,
        Scale: fx.Tensor,
        Freqs: fx.Tensor,
        Output: fx.Tensor,
        FreqGroup: fx.Int32,
        Eps: fx.Float32,
    ) -> None:
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        c0 = fx.Float32(0.0)
        eps_c = Eps

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        red = lds.reduction_buffer.view(fx.make_layout(reduction_slots, 1))

        def wave_reduce_add(val: object) -> fx.Float32:
            w = val
            for _sh in range_constexpr(int(math.log2(WARP))):
                off = WARP // (2 << _sh)
                w += gpu.shuffle_xor(w, off, WARP)
            return w

        def block_reduce_add(val: object) -> fx.Float32:
            lane = tid % WARP
            wave = tid // WARP
            w = wave_reduce_add(val)
            if lane == 0:
                fx.memref_store(w, red, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < reduction_slots
                lane_safe = in_range.select(lane, 0)
                v = red[lane_safe]
                ww = in_range.select(v, 0.0)
                ww = wave_reduce_add(ww)
                if lane == 0:
                    fx.memref_store(ww, red, 0)
            gpu.barrier()
            return fx.Float32(red[0])

        In_buf = fx.rocdl.make_buffer_tensor(Input)
        Sc_buf = fx.rocdl.make_buffer_tensor(Scale)
        Fr_buf = fx.rocdl.make_buffer_tensor(Freqs)
        Out_buf = fx.rocdl.make_buffer_tensor(Output)

        row_in = In_buf[bid, None]
        row_out = Out_buf[bid, None]
        f_row = bid // FreqGroup

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        in_div = fx.logical_divide(row_in, fx.make_layout(vec_width, 1))
        sc_div = fx.logical_divide(Sc_buf, fx.make_layout(vec_width, 1))
        out_div = fx.logical_divide(row_out, fx.make_layout(vec_width, 1))

        thread_sumsq = c0
        in_local = []
        sc_local = []

        for step in range_constexpr(vec_steps):
            vec_idx = tid + step * block_threads
            is_valid = vec_idx < full_vecs
            safe = is_valid.select(vec_idx, 0)
            vec = _load_vec(copy_atom, in_div, safe)
            scv = _load_vec(copy_atom, sc_div, safe)
            in_local.append(vec)
            sc_local.append(scv)
            xv = vec.to(fx.Float32)
            for ei in range_constexpr(vec_width):
                xe = xv[ei]
                thread_sumsq = thread_sumsq + is_valid.select(xe * xe, c0)

        sumsq = block_reduce_add(thread_sumsq)
        rstd = fmath.rsqrt(sumsq / fx.Float32(n_float) + eps_c, fastmath="fast")

        for step in range_constexpr(vec_steps):
            vec_idx = tid + step * block_threads
            if vec_idx < full_vecs:
                xv = in_local[step].to(fx.Float32)
                scv = sc_local[step].to(fx.Float32)
                outs = []
                for pi in range_constexpr(vec_width // 2):
                    e0 = 2 * pi
                    e1 = e0 + 1
                    n0 = xv[e0] * rstd * scv[e0]
                    n1 = xv[e1] * rstd * scv[e1]
                    if const_expr(dtype_str != "float32"):
                        n0 = n0.to(elem_dtype).to(fx.Float32)
                        n1 = n1.to(elem_dtype).to(fx.Float32)
                    pair = vec_idx * (vec_width // 2) + pi
                    f00 = fx.Float32(Fr_buf[f_row, pair, 0, 0])
                    f01 = fx.Float32(Fr_buf[f_row, pair, 0, 1])
                    f10 = fx.Float32(Fr_buf[f_row, pair, 1, 0])
                    f11 = fx.Float32(Fr_buf[f_row, pair, 1, 1])
                    y0 = f00 * n0 + f01 * n1
                    y1 = f10 * n0 + f11 * n1
                    if const_expr(dtype_str == "float32"):
                        outs.append(y0)
                        outs.append(y1)
                    else:
                        outs.append(y0.to(elem_dtype))
                        outs.append(y1.to(elem_dtype))
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
        Freqs: fx.Tensor,
        Output: fx.Tensor,
        rows: fx.Int32,
        freq_group: fx.Int32,
        eps: fx.Float32,
        stream: fx.Stream,
    ) -> None:
        rms_rope_kernel(Input, Scale, Freqs, Output, freq_group, eps).launch(
            grid=(rows, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    rms_rope_kernel.__name__ = f"{KERNEL_NAME}_{sig}"
    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


@lru_cache(maxsize=64)
def build_rms_rope_split_module(HD: int, dtype_str: str, block_threads: Optional[int] = None) -> Callable[..., None]:
    """Build RMSNorm-then-RoPE with split-half rotary pairs."""
    if block_threads is None:
        block_threads = _block_threads(HD)
    sig = kernel_signature(hd=HD, dtype=dtype_str, block=block_threads, op="rms_rope_split")
    elem_dtype, elem_bits = {
        "float32": (fx.Float32, 32),
        "float16": (fx.Float16, 16),
        "bfloat16": (fx.BFloat16, 16),
    }[dtype_str]
    vec_width = 128 // elem_bits
    if HD % vec_width != 0 or HD % 2 != 0:
        raise ValueError(f"HD={HD} bad for vec={vec_width}")
    n_pairs = HD // 2
    if n_pairs % vec_width != 0:
        raise ValueError(f"n_pairs={n_pairs} not divisible by vec={vec_width}")
    reduction_slots = (block_threads + WARP - 1) // WARP
    if reduction_slots > WARP:
        raise AssertionError("reduction_slots > WARP")
    full_vecs = HD // vec_width
    half_vecs = n_pairs // vec_width
    vec_steps = (full_vecs + block_threads - 1) // block_threads
    half_steps = (half_vecs + block_threads - 1) // block_threads
    n_float = float(HD)

    RedTy = fx.Array[fx.Float32, reduction_slots, 16]
    NormTy = fx.Array[fx.Float32, HD, 16]

    @fx.struct
    class SharedStorage:
        reduction_buffer: RedTy
        norm_row: NormTy

    def _load_vec(copy_atom: object, div_tensor: fx.Tensor, idx: fx.Int32 | int) -> fx.Vector:
        r = fx.make_rmem_tensor(vec_width, elem_dtype)
        fx.copy(copy_atom, div_tensor[None, idx], r)
        return r.load()

    def _store_vec(copy_atom: object, val: fx.Vector, div_tensor: fx.Tensor, idx: fx.Int32 | int) -> None:
        r = fx.make_rmem_tensor(vec_width, elem_dtype)
        r.store(val)
        fx.copy(copy_atom, r, div_tensor[None, idx])

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def rms_rope_split_kernel(
        Input: fx.Tensor,
        Scale: fx.Tensor,
        Freqs: fx.Tensor,
        Output: fx.Tensor,
        FreqGroup: fx.Int32,
        Eps: fx.Float32,
    ) -> None:
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        c0 = fx.Float32(0.0)
        eps_c = Eps

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        red = lds.reduction_buffer.view(fx.make_layout(reduction_slots, 1))
        norm = lds.norm_row.view(fx.make_layout(HD, 1))

        def wave_reduce_add(val: object) -> fx.Float32:
            w = val
            for _sh in range_constexpr(int(math.log2(WARP))):
                off = WARP // (2 << _sh)
                w += gpu.shuffle_xor(w, off, WARP)
            return w

        def block_reduce_add(val: object) -> fx.Float32:
            lane = tid % WARP
            wave = tid // WARP
            w = wave_reduce_add(val)
            if lane == 0:
                fx.memref_store(w, red, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < reduction_slots
                lane_safe = in_range.select(lane, 0)
                v = red[lane_safe]
                ww = in_range.select(v, 0.0)
                ww = wave_reduce_add(ww)
                if lane == 0:
                    fx.memref_store(ww, red, 0)
            gpu.barrier()
            return fx.Float32(red[0])

        In_buf = fx.rocdl.make_buffer_tensor(Input)
        Sc_buf = fx.rocdl.make_buffer_tensor(Scale)
        Fr_buf = fx.rocdl.make_buffer_tensor(Freqs)
        Out_buf = fx.rocdl.make_buffer_tensor(Output)

        row_in = In_buf[bid, None]
        row_out = Out_buf[bid, None]
        f_row = bid // FreqGroup

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
        in_div = fx.logical_divide(row_in, fx.make_layout(vec_width, 1))
        sc_div = fx.logical_divide(Sc_buf, fx.make_layout(vec_width, 1))
        out_div = fx.logical_divide(row_out, fx.make_layout(vec_width, 1))

        thread_sumsq = c0
        in_local = []
        sc_local = []

        for step in range_constexpr(vec_steps):
            vec_idx = tid + step * block_threads
            is_valid = vec_idx < full_vecs
            safe = is_valid.select(vec_idx, 0)
            vec = _load_vec(copy_atom, in_div, safe)
            scv = _load_vec(copy_atom, sc_div, safe)
            in_local.append(vec)
            sc_local.append(scv)
            xv = vec.to(fx.Float32)
            for ei in range_constexpr(vec_width):
                xe = xv[ei]
                thread_sumsq = thread_sumsq + is_valid.select(xe * xe, c0)

        sumsq = block_reduce_add(thread_sumsq)
        rstd = fmath.rsqrt(sumsq / fx.Float32(n_float) + eps_c, fastmath="fast")

        # Normalize into LDS (full row as f32, matching HIP cast-through dtype)
        for step in range_constexpr(vec_steps):
            vec_idx = tid + step * block_threads
            if vec_idx < full_vecs:
                xv = in_local[step].to(fx.Float32)
                scv = sc_local[step].to(fx.Float32)
                base = vec_idx * vec_width
                for ei in range_constexpr(vec_width):
                    n = xv[ei] * rstd * scv[ei]
                    if const_expr(dtype_str != "float32"):
                        n = n.to(elem_dtype).to(fx.Float32)
                    fx.memref_store(n, norm, base + ei)

        gpu.barrier()

        # split_half RoPE from LDS: pair p uses norm[p] and norm[p + n_pairs]
        for step in range_constexpr(half_steps):
            vec_idx = tid + step * block_threads
            if vec_idx < half_vecs:
                y0s = []
                y1s = []
                for ei in range_constexpr(vec_width):
                    pair = vec_idx * vec_width + ei
                    n0 = fx.Float32(norm[pair])
                    n1 = fx.Float32(norm[pair + n_pairs])
                    f00 = fx.Float32(Fr_buf[f_row, pair, 0, 0])
                    f01 = fx.Float32(Fr_buf[f_row, pair, 0, 1])
                    f10 = fx.Float32(Fr_buf[f_row, pair, 1, 0])
                    f11 = fx.Float32(Fr_buf[f_row, pair, 1, 1])
                    y0 = f00 * n0 + f01 * n1
                    y1 = f10 * n0 + f11 * n1
                    if const_expr(dtype_str == "float32"):
                        y0s.append(y0)
                        y1s.append(y1)
                    else:
                        y0s.append(y0.to(elem_dtype))
                        y1s.append(y1.to(elem_dtype))
                _store_vec(
                    copy_atom,
                    fx.Vector.from_elements(y0s, elem_dtype),
                    out_div,
                    vec_idx,
                )
                _store_vec(
                    copy_atom,
                    fx.Vector.from_elements(y1s, elem_dtype),
                    out_div,
                    vec_idx + half_vecs,
                )

    @flyc.jit
    def launch(
        Input: fx.Tensor,
        Scale: fx.Tensor,
        Freqs: fx.Tensor,
        Output: fx.Tensor,
        rows: fx.Int32,
        freq_group: fx.Int32,
        eps: fx.Float32,
        stream: fx.Stream,
    ) -> None:
        rms_rope_split_kernel(Input, Scale, Freqs, Output, freq_group, eps).launch(
            grid=(rows, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    rms_rope_split_kernel.__name__ = f"{KERNEL_NAME}_{sig}"
    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


@lru_cache(maxsize=64)
def build_rms_rope_qk_fused_module(HD: int, dtype_str: str, block_threads: Optional[int] = None) -> Callable[..., None]:
    """Build RMSNorm-then-RoPE for Q and K in one launch."""
    if block_threads is None:
        block_threads = _block_threads(HD)
    sig = kernel_signature(hd=HD, dtype=dtype_str, block=block_threads, op="rms_rope_qk")
    elem_dtype, elem_bits = {
        "float32": (fx.Float32, 32),
        "float16": (fx.Float16, 16),
        "bfloat16": (fx.BFloat16, 16),
    }[dtype_str]
    vec_width = 128 // elem_bits
    if HD % vec_width != 0 or HD % 2 != 0:
        raise ValueError(f"HD={HD} bad for vec={vec_width}")
    reduction_slots = (block_threads + WARP - 1) // WARP
    if reduction_slots > WARP:
        raise AssertionError("reduction_slots > WARP")
    full_vecs = HD // vec_width
    vec_steps = (full_vecs + block_threads - 1) // block_threads
    n_float = float(HD)

    @fx.struct
    class SharedStorage:
        reduction_buffer: fx.Array[fx.Float32, reduction_slots, 16]

    def _load_vec(copy_atom: object, div_tensor: fx.Tensor, idx: fx.Int32 | int) -> fx.Vector:
        r = fx.make_rmem_tensor(vec_width, elem_dtype)
        fx.copy(copy_atom, div_tensor[None, idx], r)
        return r.load()

    def _store_vec(copy_atom: object, val: fx.Vector, div_tensor: fx.Tensor, idx: fx.Int32 | int) -> None:
        r = fx.make_rmem_tensor(vec_width, elem_dtype)
        r.store(val)
        fx.copy(copy_atom, r, div_tensor[None, idx])

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def rms_rope_qk_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        ScaleQ: fx.Tensor,
        ScaleK: fx.Tensor,
        Freqs: fx.Tensor,
        OutQ: fx.Tensor,
        OutK: fx.Tensor,
        FreqGroup: fx.Int32,
        Eps: fx.Float32,
    ) -> None:
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        c0 = fx.Float32(0.0)
        eps_c = Eps

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        red = lds.reduction_buffer.view(fx.make_layout(reduction_slots, 1))

        def wave_reduce_add(val: object) -> fx.Float32:
            w = val
            for _sh in range_constexpr(int(math.log2(WARP))):
                off = WARP // (2 << _sh)
                w += gpu.shuffle_xor(w, off, WARP)
            return w

        def block_reduce_add(val: object) -> fx.Float32:
            lane = tid % WARP
            wave = tid // WARP
            w = wave_reduce_add(val)
            if lane == 0:
                fx.memref_store(w, red, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < reduction_slots
                lane_safe = in_range.select(lane, 0)
                v = red[lane_safe]
                ww = in_range.select(v, 0.0)
                ww = wave_reduce_add(ww)
                if lane == 0:
                    fx.memref_store(ww, red, 0)
            gpu.barrier()
            return fx.Float32(red[0])

        Q_buf = fx.rocdl.make_buffer_tensor(Q)
        K_buf = fx.rocdl.make_buffer_tensor(K)
        ScQ_buf = fx.rocdl.make_buffer_tensor(ScaleQ)
        ScK_buf = fx.rocdl.make_buffer_tensor(ScaleK)
        Fr_buf = fx.rocdl.make_buffer_tensor(Freqs)
        OutQ_buf = fx.rocdl.make_buffer_tensor(OutQ)
        OutK_buf = fx.rocdl.make_buffer_tensor(OutK)

        f_row = bid // FreqGroup
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

        # ---- process one stream (Q or K) ----
        def process_one(In_buf: object, Sc_buf: object, Out_buf: object) -> None:
            row_in = In_buf[bid, None]
            row_out = Out_buf[bid, None]
            in_div = fx.logical_divide(row_in, fx.make_layout(vec_width, 1))
            sc_div = fx.logical_divide(Sc_buf, fx.make_layout(vec_width, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(vec_width, 1))

            thread_sumsq = c0
            in_local = []
            sc_local = []

            for step in range_constexpr(vec_steps):
                vec_idx = tid + step * block_threads
                is_valid = vec_idx < full_vecs
                safe = is_valid.select(vec_idx, 0)
                vec = _load_vec(copy_atom, in_div, safe)
                scv = _load_vec(copy_atom, sc_div, safe)
                in_local.append(vec)
                sc_local.append(scv)
                xv = vec.to(fx.Float32)
                for ei in range_constexpr(vec_width):
                    xe = xv[ei]
                    thread_sumsq = thread_sumsq + is_valid.select(xe * xe, c0)

            sumsq = block_reduce_add(thread_sumsq)
            rstd = fmath.rsqrt(sumsq / fx.Float32(n_float) + eps_c, fastmath="fast")

            for step in range_constexpr(vec_steps):
                vec_idx = tid + step * block_threads
                if vec_idx < full_vecs:
                    xv = in_local[step].to(fx.Float32)
                    scv = sc_local[step].to(fx.Float32)
                    outs = []
                    for pi in range_constexpr(vec_width // 2):
                        e0 = 2 * pi
                        e1 = e0 + 1
                        n0 = xv[e0] * rstd * scv[e0]
                        n1 = xv[e1] * rstd * scv[e1]
                        if const_expr(dtype_str != "float32"):
                            n0 = n0.to(elem_dtype).to(fx.Float32)
                            n1 = n1.to(elem_dtype).to(fx.Float32)
                        pair = vec_idx * (vec_width // 2) + pi
                        f00 = fx.Float32(Fr_buf[f_row, pair, 0, 0])
                        f01 = fx.Float32(Fr_buf[f_row, pair, 0, 1])
                        f10 = fx.Float32(Fr_buf[f_row, pair, 1, 0])
                        f11 = fx.Float32(Fr_buf[f_row, pair, 1, 1])
                        y0 = f00 * n0 + f01 * n1
                        y1 = f10 * n0 + f11 * n1
                        if const_expr(dtype_str == "float32"):
                            outs.append(y0)
                            outs.append(y1)
                        else:
                            outs.append(y0.to(elem_dtype))
                            outs.append(y1.to(elem_dtype))
                    _store_vec(
                        copy_atom,
                        fx.Vector.from_elements(outs, elem_dtype),
                        out_div,
                        vec_idx,
                    )

        process_one(Q_buf, ScQ_buf, OutQ_buf)
        # barrier so LDS reduction buffer is free for K
        gpu.barrier()
        process_one(K_buf, ScK_buf, OutK_buf)

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        ScaleQ: fx.Tensor,
        ScaleK: fx.Tensor,
        Freqs: fx.Tensor,
        OutQ: fx.Tensor,
        OutK: fx.Tensor,
        rows: fx.Int32,
        freq_group: fx.Int32,
        eps: fx.Float32,
        stream: fx.Stream,
    ) -> None:
        rms_rope_qk_kernel(Q, K, ScaleQ, ScaleK, Freqs, OutQ, OutK, freq_group, eps).launch(
            grid=(rows, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    rms_rope_qk_kernel.__name__ = f"{KERNEL_NAME}_{sig}"
    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


@lru_cache(maxsize=64)
def build_rms_rope_split_qk_fused_module(
    HD: int, dtype_str: str, block_threads: Optional[int] = None
) -> Callable[..., None]:
    """Q then K in one block — reuse LDS norm+red, shared freqs."""

    if block_threads is None:
        block_threads = _block_threads(HD)
    sig = kernel_signature(hd=HD, dtype=dtype_str, block=block_threads, op="rms_rope_sh_qk")
    elem_dtype, elem_bits = {
        "float32": (fx.Float32, 32),
        "float16": (fx.Float16, 16),
        "bfloat16": (fx.BFloat16, 16),
    }[dtype_str]
    vec_width = 128 // elem_bits
    if HD % vec_width != 0 or HD % 2 != 0:
        raise ValueError(f"HD={HD} bad for vec={vec_width}")
    n_pairs = HD // 2
    if n_pairs % vec_width != 0:
        raise ValueError(f"n_pairs={n_pairs} not divisible by vec={vec_width}")
    reduction_slots = (block_threads + WARP - 1) // WARP
    if reduction_slots > WARP:
        raise AssertionError("reduction_slots > WARP")
    full_vecs = HD // vec_width
    half_vecs = n_pairs // vec_width
    vec_steps = (full_vecs + block_threads - 1) // block_threads
    half_steps = (half_vecs + block_threads - 1) // block_threads
    n_float = float(HD)

    RedTy = fx.Array[fx.Float32, reduction_slots, 16]
    NormTy = fx.Array[fx.Float32, HD, 16]

    @fx.struct
    class SharedStorage:
        reduction_buffer: RedTy
        norm_row: NormTy

    def _load_vec(copy_atom: object, div_tensor: fx.Tensor, idx: fx.Int32 | int) -> fx.Vector:
        r = fx.make_rmem_tensor(vec_width, elem_dtype)
        fx.copy(copy_atom, div_tensor[None, idx], r)
        return r.load()

    def _store_vec(copy_atom: object, val: fx.Vector, div_tensor: fx.Tensor, idx: fx.Int32 | int) -> None:
        r = fx.make_rmem_tensor(vec_width, elem_dtype)
        r.store(val)
        fx.copy(copy_atom, r, div_tensor[None, idx])

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def rms_rope_split_qk_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        ScaleQ: fx.Tensor,
        ScaleK: fx.Tensor,
        Freqs: fx.Tensor,
        OutQ: fx.Tensor,
        OutK: fx.Tensor,
        FreqGroup: fx.Int32,
        Eps: fx.Float32,
    ) -> None:
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        c0 = fx.Float32(0.0)
        eps_c = Eps

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        red = lds.reduction_buffer.view(fx.make_layout(reduction_slots, 1))
        norm = lds.norm_row.view(fx.make_layout(HD, 1))

        def wave_reduce_add(val: object) -> fx.Float32:
            w = val
            for _sh in range_constexpr(int(math.log2(WARP))):
                off = WARP // (2 << _sh)
                w += gpu.shuffle_xor(w, off, WARP)
            return w

        def block_reduce_add(val: object) -> fx.Float32:
            lane = tid % WARP
            wave = tid // WARP
            w = wave_reduce_add(val)
            if lane == 0:
                fx.memref_store(w, red, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < reduction_slots
                lane_safe = in_range.select(lane, 0)
                v = red[lane_safe]
                ww = in_range.select(v, 0.0)
                ww = wave_reduce_add(ww)
                if lane == 0:
                    fx.memref_store(ww, red, 0)
            gpu.barrier()
            return fx.Float32(red[0])

        Q_buf = fx.rocdl.make_buffer_tensor(Q)
        K_buf = fx.rocdl.make_buffer_tensor(K)
        ScQ_buf = fx.rocdl.make_buffer_tensor(ScaleQ)
        ScK_buf = fx.rocdl.make_buffer_tensor(ScaleK)
        Fr_buf = fx.rocdl.make_buffer_tensor(Freqs)
        OutQ_buf = fx.rocdl.make_buffer_tensor(OutQ)
        OutK_buf = fx.rocdl.make_buffer_tensor(OutK)

        f_row = bid // FreqGroup
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

        def process_one(In_buf: object, Sc_buf: object, Out_buf: object) -> None:
            row_in = In_buf[bid, None]
            row_out = Out_buf[bid, None]
            in_div = fx.logical_divide(row_in, fx.make_layout(vec_width, 1))
            sc_div = fx.logical_divide(Sc_buf, fx.make_layout(vec_width, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(vec_width, 1))

            thread_sumsq = c0
            in_local = []
            sc_local = []

            for step in range_constexpr(vec_steps):
                vec_idx = tid + step * block_threads
                is_valid = vec_idx < full_vecs
                safe = is_valid.select(vec_idx, 0)
                vec = _load_vec(copy_atom, in_div, safe)
                scv = _load_vec(copy_atom, sc_div, safe)
                in_local.append(vec)
                sc_local.append(scv)
                xv = vec.to(fx.Float32)
                for ei in range_constexpr(vec_width):
                    xe = xv[ei]
                    thread_sumsq = thread_sumsq + is_valid.select(xe * xe, c0)

            sumsq = block_reduce_add(thread_sumsq)
            rstd = fmath.rsqrt(sumsq / fx.Float32(n_float) + eps_c, fastmath="fast")

            for step in range_constexpr(vec_steps):
                vec_idx = tid + step * block_threads
                if vec_idx < full_vecs:
                    xv = in_local[step].to(fx.Float32)
                    scv = sc_local[step].to(fx.Float32)
                    base = vec_idx * vec_width
                    for ei in range_constexpr(vec_width):
                        n = xv[ei] * rstd * scv[ei]
                        if const_expr(dtype_str != "float32"):
                            n = n.to(elem_dtype).to(fx.Float32)
                        fx.memref_store(n, norm, base + ei)

            gpu.barrier()

            for step in range_constexpr(half_steps):
                vec_idx = tid + step * block_threads
                if vec_idx < half_vecs:
                    y0s = []
                    y1s = []
                    for ei in range_constexpr(vec_width):
                        pair = vec_idx * vec_width + ei
                        n0 = fx.Float32(norm[pair])
                        n1 = fx.Float32(norm[pair + n_pairs])
                        f00 = fx.Float32(Fr_buf[f_row, pair, 0, 0])
                        f01 = fx.Float32(Fr_buf[f_row, pair, 0, 1])
                        f10 = fx.Float32(Fr_buf[f_row, pair, 1, 0])
                        f11 = fx.Float32(Fr_buf[f_row, pair, 1, 1])
                        y0 = f00 * n0 + f01 * n1
                        y1 = f10 * n0 + f11 * n1
                        if const_expr(dtype_str == "float32"):
                            y0s.append(y0)
                            y1s.append(y1)
                        else:
                            y0s.append(y0.to(elem_dtype))
                            y1s.append(y1.to(elem_dtype))
                    _store_vec(
                        copy_atom,
                        fx.Vector.from_elements(y0s, elem_dtype),
                        out_div,
                        vec_idx,
                    )
                    _store_vec(
                        copy_atom,
                        fx.Vector.from_elements(y1s, elem_dtype),
                        out_div,
                        vec_idx + half_vecs,
                    )

        process_one(Q_buf, ScQ_buf, OutQ_buf)
        gpu.barrier()
        process_one(K_buf, ScK_buf, OutK_buf)

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        ScaleQ: fx.Tensor,
        ScaleK: fx.Tensor,
        Freqs: fx.Tensor,
        OutQ: fx.Tensor,
        OutK: fx.Tensor,
        rows: fx.Int32,
        freq_group: fx.Int32,
        eps: fx.Float32,
        stream: fx.Stream,
    ) -> None:
        rms_rope_split_qk_kernel(Q, K, ScaleQ, ScaleK, Freqs, OutQ, OutK, freq_group, eps).launch(
            grid=(rows, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    rms_rope_split_qk_kernel.__name__ = f"{KERNEL_NAME}_{sig}"
    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch
