# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Rotary position embedding (RoPE) for gfx120x (RDNA4).

Applies cos/sin rotary embeddings to Q and K. Q and K may use different
head counts (GQA): pass ``k_n_pairs_total`` and ``k_dim1``. Batch, sequence,
and head dim stay shared. Freqs broadcast the same way as the equal-shape path.
pair layout (split-half vs interleaved), and compile ``block`` size.
Default ``block`` comes from ``pick_rope_block_threads(n_pairs_total)``
(pair-count heuristic; see ``gfx120x_autotune_tables``).

FlyDSL-native. Prefer ``rms_rope_gfx120x`` when RMSNorm
precedes RoPE in the same layer.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu
from flydsl.expr.typing import T
from kernels.common.gfx120x_buf_helpers import (
    buf_copy_load,
    buf_copy_store,
    kernel_signature,
    ptr_buf_tensor,
)

KERNEL_NAME = "rope_gfx120x"


def pick_rope_block_threads(n_pairs_total: int) -> int:
    """Public n_pairs_total → compile block (gfx120x). Delegates to autotune tables."""
    from kernels.common.gfx120x_autotune_tables import pick_rope_block_threads as _pick

    return _pick(n_pairs_total)


def resolve_rope_block(n_pairs_total: int, block: Optional[int] = None) -> int:
    """Default host path: use ``block`` override or ``pick_rope_block_threads``."""
    if block is not None:
        return int(block)
    return pick_rope_block_threads(int(n_pairs_total))


def _resolve_build_block(block: Optional[int], n_pairs_total: Optional[int]) -> int:
    if block is None and n_pairs_total is None:
        raise TypeError("RoPE build requires block= or n_pairs_total= (default via pick_rope_block_threads)")
    return resolve_rope_block(int(n_pairs_total or 0), block)


def build_rope_module(
    x_name: str,
    block: Optional[int] = None,
    *,
    n_pairs_total: Optional[int] = None,
) -> Callable[..., None]:
    """Specialize RoPE kernel; default block via pick_rope_block_threads."""
    block = _resolve_build_block(block, n_pairs_total)
    return _build_rope_module_cached(x_name, block)


@lru_cache(maxsize=16)
def _build_rope_module_cached(x_name: str, block: int) -> Callable[..., None]:
    """Interleaved-pair RoPE with in-kernel freqs broadcast (no host expand)."""

    XTy = {"float32": fx.Float32, "float16": fx.Float16, "bfloat16": fx.BFloat16}[x_name]
    x_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[x_name]
    tile = block

    sig = kernel_signature(block=block, dtype=x_name, op="rope")

    if x_name == "float32":

        @flyc.kernel(known_block_size=[block, 1, 1])
        def rope_kernel(
            X: fx.Pointer,
            Freqs: fx.Pointer,
            Out: fx.Pointer,
            n_pairs_total: fx.Int32,
            n_pairs: fx.Int32,
            dim2: fx.Int32,
            dim1: fx.Int32,
            f_dim2: fx.Int32,
            f_dim1: fx.Int32,
            f_batch: fx.Int32,
            freqs_n_pairs_total: fx.Int32,
        ) -> None:
            bid = gpu.block_id("x")
            tid = gpu.thread_id("x")
            p = fx.Int32(bid) * fx.Int32(block) + fx.Int32(tid)
            x_buf = ptr_buf_tensor(
                X,
                elem=XTy,
                n=0x3FFFFFFF,
                unit_elems=2,
                num_records_bytes=fx.Int64(n_pairs_total) * fx.Int64(2 * x_bytes),
            )
            out_buf = ptr_buf_tensor(
                Out,
                elem=XTy,
                n=0x3FFFFFFF,
                unit_elems=2,
                num_records_bytes=fx.Int64(n_pairs_total) * fx.Int64(2 * x_bytes),
            )
            f_buf = ptr_buf_tensor(
                Freqs,
                elem=fx.Float32,
                n=0x3FFFFFFF,
                unit_elems=4,
                num_records_bytes=fx.Int64(freqs_n_pairs_total) * fx.Int64(16),
            )
            if p < n_pairs_total:
                pair = p % n_pairs
                tmp = p // n_pairs
                d2 = tmp % dim2
                tmp = tmp // dim2
                d1 = tmp % dim1
                b = tmp // dim1
                fb = (f_batch == fx.Int32(1)).select(fx.Int32(0), b)
                fd1 = (f_dim1 == fx.Int32(1)).select(fx.Int32(0), d1)
                fd2 = (f_dim2 == fx.Int32(1)).select(fx.Int32(0), d2)
                fu = ((fb * f_dim1 + fd1) * f_dim2 + fd2) * n_pairs + pair
                xv = buf_copy_load(x_buf, p, elem=XTy, unit_elems=2)
                fv = buf_copy_load(f_buf, fu, elem=fx.Float32, unit_elems=4)
                x0 = fx.Float32(xv[0])
                x1 = fx.Float32(xv[1])
                y0 = fx.Float32(fv[0]) * x0 + fx.Float32(fv[1]) * x1
                y1 = fx.Float32(fv[2]) * x0 + fx.Float32(fv[3]) * x1
                buf_copy_store(
                    out_buf,
                    p,
                    fx.Vector.from_elements([y0, y1], XTy),
                    elem=XTy,
                    unit_elems=2,
                )

    else:

        @flyc.kernel(known_block_size=[block, 1, 1])
        def rope_kernel(
            X: fx.Pointer,
            Freqs: fx.Pointer,
            Out: fx.Pointer,
            n_pairs_total: fx.Int32,
            n_pairs: fx.Int32,
            dim2: fx.Int32,
            dim1: fx.Int32,
            f_dim2: fx.Int32,
            f_dim1: fx.Int32,
            f_batch: fx.Int32,
            freqs_n_pairs_total: fx.Int32,
        ) -> None:
            bid = gpu.block_id("x")
            tid = gpu.thread_id("x")
            p = fx.Int32(bid) * fx.Int32(block) + fx.Int32(tid)
            x_buf = ptr_buf_tensor(
                X,
                elem=XTy,
                n=0x3FFFFFFF,
                unit_elems=2,
                num_records_bytes=fx.Int64(n_pairs_total) * fx.Int64(2 * x_bytes),
            )
            out_buf = ptr_buf_tensor(
                Out,
                elem=XTy,
                n=0x3FFFFFFF,
                unit_elems=2,
                num_records_bytes=fx.Int64(n_pairs_total) * fx.Int64(2 * x_bytes),
            )
            f_buf = ptr_buf_tensor(
                Freqs,
                elem=fx.Float32,
                n=0x3FFFFFFF,
                unit_elems=4,
                num_records_bytes=fx.Int64(freqs_n_pairs_total) * fx.Int64(16),
            )
            if p < n_pairs_total:
                pair = p % n_pairs
                tmp = p // n_pairs
                d2 = tmp % dim2
                tmp = tmp // dim2
                d1 = tmp % dim1
                b = tmp // dim1
                fb = (f_batch == fx.Int32(1)).select(fx.Int32(0), b)
                fd1 = (f_dim1 == fx.Int32(1)).select(fx.Int32(0), d1)
                fd2 = (f_dim2 == fx.Int32(1)).select(fx.Int32(0), d2)
                fu = ((fb * f_dim1 + fd1) * f_dim2 + fd2) * n_pairs + pair
                xv = buf_copy_load(x_buf, p, elem=XTy, unit_elems=2)
                fv = buf_copy_load(f_buf, fu, elem=fx.Float32, unit_elems=4)
                xvf = xv.extf(T.vec(2, T.f32))
                x0 = fx.Float32(xvf[0])
                x1 = fx.Float32(xvf[1])
                y0 = fx.Float32(fv[0]) * x0 + fx.Float32(fv[1]) * x1
                y1 = fx.Float32(fv[2]) * x0 + fx.Float32(fv[3]) * x1
                buf_copy_store(
                    out_buf,
                    p,
                    fx.Vector.from_elements([y0.to(XTy), y1.to(XTy)], XTy),
                    elem=XTy,
                    unit_elems=2,
                )

    @flyc.jit
    def launch(
        X: fx.Pointer,
        Freqs: fx.Pointer,
        Out: fx.Pointer,
        n_pairs_total: fx.Int32,
        n_pairs: fx.Int32,
        dim2: fx.Int32,
        dim1: fx.Int32,
        f_dim2: fx.Int32,
        f_dim1: fx.Int32,
        f_batch: fx.Int32,
        freqs_n_pairs_total: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        n64 = fx.Int64(n_pairs_total)
        grid_x = (n64 + fx.Int64(tile - 1)) // fx.Int64(tile)
        rope_kernel(
            X,
            Freqs,
            Out,
            n_pairs_total,
            n_pairs,
            dim2,
            dim1,
            f_dim2,
            f_dim1,
            f_batch,
            freqs_n_pairs_total,
        ).launch(grid=(grid_x, 1, 1), block=(block, 1, 1), stream=stream)

    rope_kernel.__name__ = f"{KERNEL_NAME}_{sig}"
    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


def build_rope_split_module(
    x_name: str,
    block: Optional[int] = None,
    *,
    n_pairs_total: Optional[int] = None,
) -> Callable[..., None]:
    """Specialize RoPE kernel; default block via pick_rope_block_threads."""
    block = _resolve_build_block(block, n_pairs_total)
    return _build_rope_split_module_cached(x_name, block)


@lru_cache(maxsize=16)
def _build_rope_split_module_cached(x_name: str, block: int) -> Callable[..., None]:
    """split_half RoPE: pair k uses elems [k] and [k + n_pairs]."""
    XTy = {"float32": fx.Float32, "float16": fx.Float16, "bfloat16": fx.BFloat16}[x_name]
    x_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[x_name]
    tile = block
    sig = kernel_signature(block=block, dtype=x_name, op="rope_split")

    @flyc.kernel(known_block_size=[block, 1, 1])
    def rope_kernel(
        X: fx.Pointer,
        Freqs: fx.Pointer,
        Out: fx.Pointer,
        n_pairs_total: fx.Int32,
        n_pairs: fx.Int32,
        dim2: fx.Int32,
        dim1: fx.Int32,
        f_dim2: fx.Int32,
        f_dim1: fx.Int32,
        f_batch: fx.Int32,
        freqs_n_pairs_total: fx.Int32,
        head_dim: fx.Int32,
    ) -> None:
        bid = gpu.block_id("x")
        tid = gpu.thread_id("x")
        p = fx.Int32(bid) * fx.Int32(block) + fx.Int32(tid)
        n_elems = n_pairs_total * fx.Int32(2)
        x_buf = ptr_buf_tensor(
            X,
            elem=XTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_elems) * fx.Int64(x_bytes),
        )
        out_buf = ptr_buf_tensor(
            Out,
            elem=XTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_elems) * fx.Int64(x_bytes),
        )
        f_buf = ptr_buf_tensor(
            Freqs,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=4,
            num_records_bytes=fx.Int64(freqs_n_pairs_total) * fx.Int64(16),
        )
        if p < n_pairs_total:
            pair = p % n_pairs
            tmp = p // n_pairs
            d2 = tmp % dim2
            tmp = tmp // dim2
            d1 = tmp % dim1
            b = tmp // dim1
            fb = (f_batch == fx.Int32(1)).select(fx.Int32(0), b)
            fd1 = (f_dim1 == fx.Int32(1)).select(fx.Int32(0), d1)
            fd2 = (f_dim2 == fx.Int32(1)).select(fx.Int32(0), d2)
            fu = ((fb * f_dim1 + fd1) * f_dim2 + fd2) * n_pairs + pair
            row = (b * dim1 + d1) * dim2 + d2
            idx0 = row * head_dim + pair
            idx1 = idx0 + n_pairs
            x0 = fx.Float32(buf_copy_load(x_buf, idx0, elem=XTy, unit_elems=1))
            x1 = fx.Float32(buf_copy_load(x_buf, idx1, elem=XTy, unit_elems=1))
            fv = buf_copy_load(f_buf, fu, elem=fx.Float32, unit_elems=4)
            y0 = (fx.Float32(fv[0]) * x0 + fx.Float32(fv[1]) * x1).to(XTy)
            y1 = (fx.Float32(fv[2]) * x0 + fx.Float32(fv[3]) * x1).to(XTy)
            buf_copy_store(out_buf, idx0, y0, elem=XTy, unit_elems=1)
            buf_copy_store(out_buf, idx1, y1, elem=XTy, unit_elems=1)

    @flyc.jit
    def launch(
        X: fx.Pointer,
        Freqs: fx.Pointer,
        Out: fx.Pointer,
        n_pairs_total: fx.Int32,
        n_pairs: fx.Int32,
        dim2: fx.Int32,
        dim1: fx.Int32,
        f_dim2: fx.Int32,
        f_dim1: fx.Int32,
        f_batch: fx.Int32,
        freqs_n_pairs_total: fx.Int32,
        head_dim: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        n64 = fx.Int64(n_pairs_total)
        grid_x = (n64 + fx.Int64(tile - 1)) // fx.Int64(tile)
        rope_kernel(
            X,
            Freqs,
            Out,
            n_pairs_total,
            n_pairs,
            dim2,
            dim1,
            f_dim2,
            f_dim1,
            f_batch,
            freqs_n_pairs_total,
            head_dim,
        ).launch(grid=(grid_x, 1, 1), block=(block, 1, 1), stream=stream)

    rope_kernel.__name__ = f"{KERNEL_NAME}_{sig}"
    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


def build_rope_qk_fused_module(
    x_name: str,
    block: Optional[int] = None,
    *,
    n_pairs_total: Optional[int] = None,
) -> Callable[..., None]:
    """Specialize RoPE kernel; default block via pick_rope_block_threads."""
    block = _resolve_build_block(block, n_pairs_total)
    return _build_rope_qk_fused_module_cached(x_name, block)


@lru_cache(maxsize=16)
def _build_rope_qk_fused_module_cached(x_name: str, block: int) -> Callable[..., None]:
    XTy = {"float32": fx.Float32, "float16": fx.Float16, "bfloat16": fx.BFloat16}[x_name]
    x_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[x_name]
    is_f32 = x_name == "float32"
    sig = kernel_signature(block=block, dtype=x_name, op="rope_qk")

    @flyc.kernel(known_block_size=[block, 1, 1])
    def rope_qk_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        Freqs: fx.Pointer,
        OutQ: fx.Pointer,
        OutK: fx.Pointer,
        n_pairs_total: fx.Int32,
        n_pairs: fx.Int32,
        dim2: fx.Int32,
        dim1: fx.Int32,
        f_dim2: fx.Int32,
        f_dim1: fx.Int32,
        f_batch: fx.Int32,
        freqs_n_pairs_total: fx.Int32,
        k_n_pairs_total: fx.Int32,
        k_dim1: fx.Int32,
    ) -> None:
        bid = gpu.block_id("x")
        tid = gpu.thread_id("x")
        p = fx.Int32(bid) * fx.Int32(block) + fx.Int32(tid)
        q_nbytes = fx.Int64(n_pairs_total) * fx.Int64(2 * x_bytes)
        k_nbytes = fx.Int64(k_n_pairs_total) * fx.Int64(2 * x_bytes)
        q_buf = ptr_buf_tensor(Q, elem=XTy, n=0x3FFFFFFF, unit_elems=2, num_records_bytes=q_nbytes)
        k_buf = ptr_buf_tensor(K, elem=XTy, n=0x3FFFFFFF, unit_elems=2, num_records_bytes=k_nbytes)
        oq_buf = ptr_buf_tensor(OutQ, elem=XTy, n=0x3FFFFFFF, unit_elems=2, num_records_bytes=q_nbytes)
        ok_buf = ptr_buf_tensor(OutK, elem=XTy, n=0x3FFFFFFF, unit_elems=2, num_records_bytes=k_nbytes)
        f_buf = ptr_buf_tensor(
            Freqs,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=4,
            num_records_bytes=fx.Int64(freqs_n_pairs_total) * fx.Int64(16),
        )

        def rotate(buf, out_buf, n_lim, axis_dim1):
            if p < n_lim:
                pair = p % n_pairs
                tmp = p // n_pairs
                d2 = tmp % dim2
                tmp = tmp // dim2
                d1 = tmp % axis_dim1
                b = tmp // axis_dim1
                fb = (f_batch == fx.Int32(1)).select(fx.Int32(0), b)
                fd1 = (f_dim1 == fx.Int32(1)).select(fx.Int32(0), d1)
                fd2 = (f_dim2 == fx.Int32(1)).select(fx.Int32(0), d2)
                fu = ((fb * f_dim1 + fd1) * f_dim2 + fd2) * n_pairs + pair
                xv = buf_copy_load(buf, p, elem=XTy, unit_elems=2)
                fv = buf_copy_load(f_buf, fu, elem=fx.Float32, unit_elems=4)
                if not is_f32:
                    xv = xv.extf(T.vec(2, T.f32))
                x0 = fx.Float32(xv[0])
                x1 = fx.Float32(xv[1])
                f0 = fx.Float32(fv[0])
                f1 = fx.Float32(fv[1])
                f2 = fx.Float32(fv[2])
                f3 = fx.Float32(fv[3])
                y0 = f0 * x0 + f1 * x1
                y1 = f2 * x0 + f3 * x1
                if is_f32:
                    buf_copy_store(out_buf, p, fx.Vector.from_elements([y0, y1], XTy), elem=XTy, unit_elems=2)
                else:
                    buf_copy_store(
                        out_buf,
                        p,
                        fx.Vector.from_elements([y0.to(XTy), y1.to(XTy)], XTy),
                        elem=XTy,
                        unit_elems=2,
                    )

        rotate(q_buf, oq_buf, n_pairs_total, dim1)
        rotate(k_buf, ok_buf, k_n_pairs_total, k_dim1)

    @flyc.jit
    def launch(
        Q: fx.Pointer,
        K: fx.Pointer,
        Freqs: fx.Pointer,
        OutQ: fx.Pointer,
        OutK: fx.Pointer,
        n_pairs_total: fx.Int32,
        n_pairs: fx.Int32,
        dim2: fx.Int32,
        dim1: fx.Int32,
        f_dim2: fx.Int32,
        f_dim1: fx.Int32,
        f_batch: fx.Int32,
        freqs_n_pairs_total: fx.Int32,
        k_n_pairs_total: fx.Int32,
        k_dim1: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        nq = fx.Int64(n_pairs_total)
        nk = fx.Int64(k_n_pairs_total)
        nwork = (nq >= nk).select(nq, nk)
        grid_x = (nwork + fx.Int64(block - 1)) // fx.Int64(block)
        rope_qk_kernel(
            Q,
            K,
            Freqs,
            OutQ,
            OutK,
            n_pairs_total,
            n_pairs,
            dim2,
            dim1,
            f_dim2,
            f_dim1,
            f_batch,
            freqs_n_pairs_total,
            k_n_pairs_total,
            k_dim1,
        ).launch(grid=(grid_x, 1, 1), block=(block, 1, 1), stream=stream)

    rope_qk_kernel.__name__ = f"{KERNEL_NAME}_{sig}"
    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


def build_rope_split_half_qk_fused_module(
    x_name: str,
    block: Optional[int] = None,
    *,
    n_pairs_total: Optional[int] = None,
) -> Callable[..., None]:
    """Specialize RoPE kernel; default block via pick_rope_block_threads."""
    block = _resolve_build_block(block, n_pairs_total)
    return _build_rope_split_half_qk_fused_module_cached(x_name, block)


@lru_cache(maxsize=16)
def _build_rope_split_half_qk_fused_module_cached(x_name: str, block: int) -> Callable[..., None]:
    XTy = {"float32": fx.Float32, "float16": fx.Float16, "bfloat16": fx.BFloat16}[x_name]
    x_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[x_name]
    sig = kernel_signature(block=block, dtype=x_name, op="rope_sh_qk")

    @flyc.kernel(known_block_size=[block, 1, 1])
    def rope_sh_qk_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        Freqs: fx.Pointer,
        OutQ: fx.Pointer,
        OutK: fx.Pointer,
        n_pairs_total: fx.Int32,
        n_pairs: fx.Int32,
        dim2: fx.Int32,
        dim1: fx.Int32,
        f_dim2: fx.Int32,
        f_dim1: fx.Int32,
        f_batch: fx.Int32,
        freqs_n_pairs_total: fx.Int32,
        head_dim: fx.Int32,
        k_n_pairs_total: fx.Int32,
        k_dim1: fx.Int32,
    ) -> None:
        bid = gpu.block_id("x")
        tid = gpu.thread_id("x")
        p = fx.Int32(bid) * fx.Int32(block) + fx.Int32(tid)
        q_nbytes = fx.Int64(n_pairs_total * fx.Int32(2)) * fx.Int64(x_bytes)
        k_nbytes = fx.Int64(k_n_pairs_total * fx.Int32(2)) * fx.Int64(x_bytes)
        q_buf = ptr_buf_tensor(Q, elem=XTy, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=q_nbytes)
        k_buf = ptr_buf_tensor(K, elem=XTy, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=k_nbytes)
        oq_buf = ptr_buf_tensor(OutQ, elem=XTy, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=q_nbytes)
        ok_buf = ptr_buf_tensor(OutK, elem=XTy, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=k_nbytes)
        f_buf = ptr_buf_tensor(
            Freqs,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=4,
            num_records_bytes=fx.Int64(freqs_n_pairs_total) * fx.Int64(16),
        )

        def rotate(buf, out_buf, n_lim, axis_dim1):
            if p < n_lim:
                pair = p % n_pairs
                tmp = p // n_pairs
                d2 = tmp % dim2
                tmp = tmp // dim2
                d1 = tmp % axis_dim1
                b = tmp // axis_dim1
                fb = (f_batch == fx.Int32(1)).select(fx.Int32(0), b)
                fd1 = (f_dim1 == fx.Int32(1)).select(fx.Int32(0), d1)
                fd2 = (f_dim2 == fx.Int32(1)).select(fx.Int32(0), d2)
                fu = ((fb * f_dim1 + fd1) * f_dim2 + fd2) * n_pairs + pair
                row = (b * axis_dim1 + d1) * dim2 + d2
                idx0 = row * head_dim + pair
                idx1 = idx0 + n_pairs
                x0 = fx.Float32(buf_copy_load(buf, idx0, elem=XTy, unit_elems=1))
                x1 = fx.Float32(buf_copy_load(buf, idx1, elem=XTy, unit_elems=1))
                fv = buf_copy_load(f_buf, fu, elem=fx.Float32, unit_elems=4)
                f0 = fx.Float32(fv[0])
                f1 = fx.Float32(fv[1])
                f2 = fx.Float32(fv[2])
                f3 = fx.Float32(fv[3])
                y0 = (f0 * x0 + f1 * x1).to(XTy)
                y1 = (f2 * x0 + f3 * x1).to(XTy)
                buf_copy_store(out_buf, idx0, y0, elem=XTy, unit_elems=1)
                buf_copy_store(out_buf, idx1, y1, elem=XTy, unit_elems=1)

        rotate(q_buf, oq_buf, n_pairs_total, dim1)
        rotate(k_buf, ok_buf, k_n_pairs_total, k_dim1)

    @flyc.jit
    def launch(
        Q: fx.Pointer,
        K: fx.Pointer,
        Freqs: fx.Pointer,
        OutQ: fx.Pointer,
        OutK: fx.Pointer,
        n_pairs_total: fx.Int32,
        n_pairs: fx.Int32,
        dim2: fx.Int32,
        dim1: fx.Int32,
        f_dim2: fx.Int32,
        f_dim1: fx.Int32,
        f_batch: fx.Int32,
        freqs_n_pairs_total: fx.Int32,
        head_dim: fx.Int32,
        k_n_pairs_total: fx.Int32,
        k_dim1: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        nq = fx.Int64(n_pairs_total)
        nk = fx.Int64(k_n_pairs_total)
        nwork = (nq >= nk).select(nq, nk)
        grid_x = (nwork + fx.Int64(block - 1)) // fx.Int64(block)
        rope_sh_qk_kernel(
            Q,
            K,
            Freqs,
            OutQ,
            OutK,
            n_pairs_total,
            n_pairs,
            dim2,
            dim1,
            f_dim2,
            f_dim1,
            f_batch,
            freqs_n_pairs_total,
            head_dim,
            k_n_pairs_total,
            k_dim1,
        ).launch(grid=(grid_x, 1, 1), block=(block, 1, 1), stream=stream)

    rope_sh_qk_kernel.__name__ = f"{KERNEL_NAME}_{sig}"
    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch
