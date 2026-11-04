# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx1250 TDM (tensor data mover) loads over raw addresses for the dsv41 mono
kernels: the weights are kernel arguments as 64-bit integers, so a descriptor's
GROUP0 is packed from an i64 global address and an LDS byte address by hand (the
layout of aiter ``mega_moe_gfx1250.tdm_prims``):

  GROUP0  pred | LDS byte address | global lo | global hi (bit 31: type field)
  GROUP1  s0 data_size[17:16]   s1 dim0 << 16   s2 dim0 hi | dim1 << 16
          s3 dim1 hi | tile_dim0 << 16   s4 tile_dim1   s5 row stride
          s6 stride hi | dim1 stride << 16   s7 dim1 stride hi

dim0 is the contiguous extent of a row (elements), dim1 the row count;
``tensor_dim == tile_dim`` turns OOB checking off (the caller guarantees the run
is in bounds); the dim1 stride is set to dim1 (zero with dim1 == 1 hangs the
engine). No padding, multicast or atomic barrier.

Rules (kernel-wiki mi400-tdm-programming / -sync, aiter comments):
  * a wave keeps at most ``MAX_IN_FLIGHT`` tensor ops in flight; loads and
    stores share TENSORcnt and complete in issue order (``wait``);
  * the direct (cache-bypassing) path wants 128 B aligned global addresses and
    a row of exactly 128 B or >= 256 B (``check_block``);
  * the descriptor operands must be wave-uniform: ``readfirstlane`` here, and
    every caller is in a wave-uniform branch (lane 0 active);
  * TDM ops have no context save: a step drains them (``wait(0)``) before its
    consumers, and no TDM op stays in flight at kernel end.
"""

import math

import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.rocdl import readfirstlane, tdm_ops
from flydsl.expr.typing import T

from kernels.monokernel.sync_gfx1250 import wait_tensor

TDM_MAX_DIM = 0xFFFF
MAX_IN_FLIGHT = 3  # TENSOR_WAVE_CNT: a wave's ops between issue and XACK
ROW_BYTES = 4096  # a contiguous block's descriptor row (1024 dwords)


def _i32(v):
    v &= 0xFFFFFFFF
    return fx.Int32(v - (1 << 32) if v >= 1 << 31 else v)


def lds_addr(ptr):
    """The LDS byte address of an LDS pointer (a struct field's ``.ptr``) as i32.
    ``fly.ptrtoint``'s result type is not documented in flydsl 0.3.3 (index for
    aiter's main): handled whichever integer it is."""
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import arith as std_arith

    v = fx.ptrtoint(ptr)
    raw = v.ir_value() if hasattr(v, "ir_value") else v
    ty = raw.type
    if isinstance(ty, ir.IndexType):
        return fx.Int32(arith.index_cast(T.i32, raw))
    if ir.IntegerType(ty).width > 32:
        return fx.Int32(std_arith.TruncIOp(ir.IntegerType.get_signless(32), raw).result)
    return fx.Int32(raw)


def group0(lds_byte_addr, global_addr):
    lo = fx.Int32(global_addr)
    hi = fx.Int32(fx.Uint64(global_addr) >> 32) | _i32(1 << 31)
    return fx.Vector.from_elements(
        [
            _i32(1),
            readfirstlane(T.i32, fx.Int32(lds_byte_addr).ir_value()),
            readfirstlane(T.i32, lo.ir_value()),
            readfirstlane(T.i32, hi.ir_value()),
        ],
        fx.Int32,
    ).ir_value()


def group1(dim0, dim1, elem_bytes, stride=None):
    """A ``dim1`` x ``dim0`` tile of ``elem_bytes`` elements, rows ``stride``
    elements apart (default dense); compile-time, eight SGPR constants."""
    assert elem_bytes in (1, 2, 4, 8), elem_bytes
    assert 1 <= dim0 <= TDM_MAX_DIM and 1 <= dim1 <= TDM_MAX_DIM, (dim1, dim0)
    stride = dim0 if stride is None else stride
    ds = int(math.log2(elem_bytes))
    return fx.Vector.from_elements(
        [
            _i32(ds << 16),
            _i32((dim0 & 0xFFFF) << 16),
            _i32(((dim0 >> 16) & 0xFFFF) | ((dim1 & 0xFFFF) << 16)),
            _i32(((dim1 >> 16) & 0xFFFF) | ((dim0 & 0xFFFF) << 16)),
            _i32(dim1 & 0xFFFF),
            _i32(stride & 0xFFFFFFFF),
            _i32(((stride >> 32) & 0xFFFF) | ((dim1 & 0xFFFF) << 16)),
            _i32((dim1 >> 16) & 0xFFFFFFFF),
        ],
        fx.Int32,
    ).ir_value()


def check_block(nbytes: int) -> tuple:
    """A contiguous block's (row dwords, rows): one row up to ``ROW_BYTES``, else
    the widest power-of-two row that divides it (5120 B: 5 x 1024). Raises
    unless the block is a whole number of 128 B lines."""
    if nbytes % 128:
        raise ValueError(f"a TDM block must be whole 128 B lines, got {nbytes} B")
    row = nbytes if nbytes <= ROW_BYTES else next((r for r in (4096, 2048, 1024, 512, 256) if nbytes % r == 0), 128)
    if nbytes % row or nbytes // row > TDM_MAX_DIM:
        raise ValueError(f"a TDM block of {nbytes} B is not whole {row} B rows")
    if row != 128 and row < 256:
        raise ValueError(f"a TDM row of {row} B misses the direct path (128 B or >= 256 B)")
    return row // 4, nbytes // row


def load_block(lds_ptr, global_addr, nbytes: int, cache_policy: int = 0):
    """Async copy of ``nbytes`` contiguous bytes (128 B aligned) at the i64
    ``global_addr`` into LDS at ``lds_ptr``: one tensor op. Issue from one wave
    (a wave-uniform branch); observe completion with ``wait``."""
    dim0, rows = check_block(nbytes)
    tdm_ops.tensor_load_2d(
        tdm_ops.TDMDescriptor2D(group0(lds_addr(lds_ptr), fx.Int64(global_addr)), group1(dim0, rows, 4)),
        cache_policy,
    )


def wait(count: int = 0):
    """This wave's tensor ops down to ``count`` in flight (the oldest retire)."""
    wait_tensor(count)


# ---------------------------------------------------------------- gather mode
GATHER_MAX = 8  # row indices an op takes in 32-bit index mode (16 in 16-bit mode)


def load_gather(lds_ptr, global_addr, rows, row_bytes: int, row_stride_bytes: int, total_rows):
    """One tensor op gathering rows ``rows`` (up to GATHER_MAX traced i32 row
    numbers, each < ``total_rows``) of ``row_bytes`` from a matrix at the i64
    ``global_addr`` with rows ``row_stride_bytes`` apart, into LDS at ``lds_ptr``
    back to back (ISA 4.10.3.2 gather mode; FlyDSL ``make_tensor_gather_descriptor``'s
    groups over raw addresses). A gather is one FIFO entry and one descriptor slot
    however many rows it moves. Issue from one wave (wave-uniform operands)."""
    from flydsl.expr.rocdl.tdm_ops import TDMGatherDescriptor, tensor_load_gather

    assert 1 <= len(rows) <= GATHER_MAX and row_bytes % 4 == 0 and row_stride_bytes % 4 == 0
    width, stride = row_bytes // 4, row_stride_bytes // 4  # 4-byte elements
    assert width <= TDM_MAX_DIM
    addr = fx.Int64(global_addr)
    lo = fx.Int32(addr)
    hi = fx.Int32(fx.Uint64(addr) >> 32) | _i32(1 << 31)
    rf = lambda v: readfirstlane(T.i32, fx.Int32(v).ir_value())  # noqa: E731
    g0 = fx.Vector.from_elements(
        [_i32(1 | (1 << 30) | (1 << 31)), rf(lds_addr(lds_ptr)), rf(lo), rf(hi)], fx.Int32
    ).ir_value()  # pred, gather with 32-bit indices
    n_rows = rf(total_rows)
    g1 = fx.Vector.from_elements(
        [
            _i32(2 << 16),  # data_size 4 B
            _i32((width & 0xFFFF) << 16),  # tensor_dim0 = the row
            fx.Int32((width >> 16) & 0xFFFF) | ((fx.Int32(n_rows) & 0xFFFF) << 16),
            ((fx.Int32(n_rows) >> 16) & 0xFFFF) | _i32(width << 16),  # tensor_dim1 hi | tile_dim0
            _i32(len(rows)),  # tile_dim1: the indices used
            _i32(stride),
            _i32(0),
            _i32(0),
        ],
        fx.Int32,
    ).ir_value()
    idx = [rf(r) for r in rows] + [_i32(0)] * (GATHER_MAX - len(rows))
    g2 = fx.Vector.from_elements(idx[:4], fx.Int32).ir_value()
    g3 = fx.Vector.from_elements(idx[4:], fx.Int32).ir_value()
    tensor_load_gather(TDMGatherDescriptor(dgroup0=g0, dgroup1=g1, dgroup2=g2, dgroup3=g3))
