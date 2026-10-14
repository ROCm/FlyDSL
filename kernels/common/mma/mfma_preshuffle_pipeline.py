# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared B-preshuffle layout and workgroup remapping helpers."""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir


@dataclass(frozen=True)
class PreshuffleBLayout:
    """Container returned by `make_preshuffle_b_layout`."""

    layout_b: object
    kpack_bytes: int


def make_preshuffle_b_layout(
    *,
    c_n: ir.Value,
    c_k: ir.Value,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
    k_major: bool = False,
) -> PreshuffleBLayout:
    """Build B layout matching aiter/CK preshuffle for A8 MFMA kernels.

    When *k_major* is True the block-level order is K-major (``k_blk`` outermost),
    matching the ``(0,3,1,4,2,5)`` shuffle permutation.  The default N-major
    order (``k_major=False``) matches the legacy ``(0,1,3,4,2,5)`` permutation.
    """
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")

    c16 = fx.Index(16)
    c_kpack = fx.Index(kpack_bytes)

    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    c_k_bytes = c_k * fx.Index(int(elem_bytes))
    n0 = c_n // c16

    c_kpack_elems = c_kpack if elem_bytes == 1 else (c_kpack // fx.Index(int(elem_bytes)))

    stride_nlane = c_kpack_elems

    if k_major:
        c32 = fx.Index(32)
        c2 = fx.Index(2)
        c_k0 = c_k_bytes // c32
        klane_dim = 2
        stride_klane = c16 * stride_nlane
        stride_n0 = c2 * stride_klane
        stride_k0 = n0 * stride_n0
    else:
        c64 = fx.Index(64)
        c4 = fx.Index(4)
        c_k0 = c_k_bytes // c64
        klane_dim = 4
        stride_klane = c16 * stride_nlane
        stride_k0 = c4 * stride_klane
        stride_n0 = c_k0 * stride_k0

    kpack_elems_static = kpack_bytes if elem_bytes == 1 else kpack_bytes // elem_bytes
    n0_i32 = fx.Int32(n0)
    c_k0_i32 = fx.Int32(c_k0)
    stride_n0_i32 = fx.Int32(stride_n0)
    stride_k0_i32 = fx.Int32(stride_k0)
    stride_klane_i32 = fx.Int32(stride_klane)
    stride_nlane_i32 = fx.Int32(stride_nlane)

    stride_b = (stride_n0_i32, stride_k0_i32, stride_klane_i32, stride_nlane_i32, 1)
    layout_b = fx.make_layout((n0_i32, c_k0_i32, klane_dim, 16, kpack_elems_static), stride_b)
    return PreshuffleBLayout(layout_b=layout_b, kpack_bytes=kpack_bytes)


@flyc.jit
def xcd_remap_bx_by(
    bx,
    by,
    c_m,
    *,
    tile_m: int,
    tile_n: int,
    N: int,
    xcd_swizzle: int,
    num_xcds: int = 8,
):
    if xcd_swizzle <= 0:
        return bx, by

    # Keep the whole remap in i32 (grid dims fit): gpu.block_id yields index, so
    # cast the block ids so every derived value (and both ternary branches) is i32.
    bx = fx.Int32(bx)
    by = fx.Int32(by)

    gx = N // tile_n
    gy = (c_m + tile_m - 1) // tile_m

    linear_id = bx * gx + by
    num_wgs = gx * gy

    q = num_wgs // num_xcds
    r = num_wgs % num_xcds
    xcd = linear_id % num_xcds
    in_xcd = linear_id // num_xcds
    xcd_lt_r = xcd < r
    clip = xcd if xcd_lt_r else r
    wgid = xcd * q + clip + in_xcd

    num_wgid_in_group = xcd_swizzle * gx
    group_id = wgid // num_wgid_in_group
    first_pid_m = group_id * xcd_swizzle
    remaining_m = gy - first_pid_m
    cmp_m = remaining_m < xcd_swizzle
    group_size_m = remaining_m if cmp_m else fx.Int32(xcd_swizzle)

    wgid_in_group = wgid % num_wgid_in_group
    new_bx = first_pid_m + (wgid_in_group % group_size_m)
    new_by = wgid_in_group // group_size_m
    return new_bx, new_by


__all__ = ["PreshuffleBLayout", "make_preshuffle_b_layout", "xcd_remap_bx_by"]
