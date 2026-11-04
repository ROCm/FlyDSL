# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Wave32 device primitives of the gfx1250 mono kernels.

The gfx950 helpers (``kernels.monokernel.ops``) assume 64 lanes: permlane32,
``row_bcast`` DPP (removed in gfx12), i64 ballots, lane 63. Here a wave is 32
lanes = two 16-lane DPP rows:

* xor offsets 1..8 stay in a row: one DPP ``row_xmask`` each;
* xor 16 crosses the rows: ``v_permlane16_swap``;
* the MI350 kernels' xor-32 step joins the two halves of an MI350 wave, which
  are now two waves (``2w``, ``2w + 1``): ``butterfly64`` exchanges through LDS,
  reproducing the 64-lane association bit for bit where a stage's numerical
  contract needs it.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value

from kernels.common import buffer_ops as bo
from kernels.common.dpp_utils import update_dpp_i32
from kernels.monokernel.execution_gfx1250 import THREADS, WAVE, WAVES
from kernels.monokernel.sync_gfx1250 import SCOPE_DEV, f32_pinned, rsrc, to_bf16, traced

DPP_ROW_XMASK = 0x160  # row_xmask:m -> lane ^ m within a 16-lane row


# --- lanes ---------------------------------------------------------------------
def uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def uniform_f32(v):
    return uniform(fx.Float32(v).bitcast(fx.Int32)).bitcast(fx.Float32)


def readlane(v, src_lane):
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    r = fx.Int32(rocdl.readlane(T.i32, bits, fx.Int32(src_lane)))
    return r.bitcast(fx.Float32) if isinstance(v, fx.Float32) else r


def lane_gather(v, src_lane):
    """Lane ``src_lane``'s value of v, a source lane a lane (ds_bpermute)."""
    return fx.Int32(rocdl.ds_bpermute(T.i32, fx.Int32(src_lane) * 4, fx.Int32(v).ir_value()))


def ballot(pred):
    """The wave's lanes where ``pred`` holds, a bit a lane (32 bits)."""
    return fx.Int32(rocdl.ballot(T.i32, as_ir_value(pred)))


def popcount(x):
    return fx.Int32(fx.ctpop(x))


def lanes_below(bal, lane):
    return popcount(bal & ((fx.Int32(1) << fx.Int32(lane)) - 1))


def permlane16_swap(x, y):
    """v_permlane16_swap on (x, y): x's row 1 trades places with y's row 0.
    With x = y = v this returns (row 0 in both rows, row 1 in both rows)."""
    pr = rocdl.permlane16_swap(
        llvm.StructType.get_literal([T.i32, T.i32]),
        as_ir_value(fx.Int32(x)),
        as_ir_value(fx.Int32(y)),
        False,
        False,
    )
    return tuple(fx.Int32(llvm.extractvalue(T.i32, pr, [j])) for j in range(2))


def _bits(v):
    return v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)


def _same_kind(v, bits):
    return bits.bitcast(fx.Float32) if isinstance(v, fx.Float32) else type(v)(bits)


def xshfl(v, off, lane=None):
    """Value of lane ``lane ^ off``. ``off`` 16 needs ``lane``."""
    assert off in (1, 2, 4, 8, 16), off
    x = _bits(v)
    if off == 16:
        lo, hi = permlane16_swap(x, x)
        return _same_kind(v, ((lane & 16) != 0).select(lo, hi))
    return _same_kind(v, fx.Int32(update_dpp_i32(x, x, DPP_ROW_XMASK + off, 0xF, 0xF, False)))


def xred(v, off, op):
    """op(v, lane ^ off) for a commutative op (16: both halves of one swap)."""
    if off < 16:
        return op(v, xshfl(v, off))
    x = _bits(v)
    a, b = permlane16_swap(x, x)
    return op(_same_kind(v, a), _same_kind(v, b))


def _add(a, b):
    return a + b


def butterfly(v, offsets, op=_add):
    """Butterfly reduction over xor ``offsets`` in the given order (the order is
    the numerical contract)."""
    for off in offsets:
        v = xred(v, off, op)
    return v


def wave_sum(v):
    return butterfly(v, (16, 8, 4, 2, 1))


def wave_max(v):
    return butterfly(v, (16, 8, 4, 2, 1), fx.max)


@traced
def butterfly64(v, offsets, lane, wave, xbuf, op=_add):
    """The MI350 64-lane butterfly over ``offsets`` (each a power of two <= 32) on a
    pair of waves (2w, 2w + 1), bit-exact: the xor-32 step adds the partner wave's
    same lane through LDS (``xbuf``: THREADS floats; every wave of the CTA must call
    this, it has barriers). Commutative ops only, as on MI350."""
    for off in offsets:
        if const_expr(off == 32):
            gpu.barrier()
            fx.ptr_store(fx.Float32(v), xbuf + (wave * WAVE + lane))
            gpu.barrier()
            other = fx.ptr_load(xbuf + ((wave ^ 1) * WAVE + lane))
            v = op(v, other)
        else:
            v = xred(v, off, op)
    return v


def wave_incl_scan(v, lane):
    """Inclusive prefix sum of an Int32 over the wave, in lane order."""
    blk = v
    for off in (1, 2, 4, 8, 16):
        o = xshfl(blk, off, lane)
        v = v + ((lane & off) != 0).select(o, 0)
        blk = blk + o
    return v


@traced
def block_excl_scan(v, lane, wave, buf):
    """(exclusive prefix of v over the CTA in thread order, CTA total); ``buf``:
    WAVES words."""
    incl = wave_incl_scan(v, lane)
    gpu.barrier()
    if lane == WAVE - 1:
        fx.ptr_store(incl, buf + wave)
    gpu.barrier()
    off = fx.Int32(0)
    tot = fx.Int32(0)
    for w in range_constexpr(WAVES):
        cnt = fx.ptr_load(buf + w)
        off = off + (w < wave).select(cnt, 0)
        tot = tot + cnt
    return incl - v + off, tot


@traced
def block_max(v, lane, wave, red):
    """Max of v over the CTA's waves; ``red``: WAVES words."""
    wv = wave_max(v)
    gpu.barrier()
    if lane == 0:
        fx.ptr_store(wv, red + wave)
    gpu.barrier()
    t = fx.ptr_load(red + 0)
    for w in range_constexpr(1, WAVES):
        t = fx.max(t, fx.ptr_load(red + w))
    gpu.barrier()
    return t


# --- scheduling hints ----------------------------------------------------------
def fresh(v):
    """``v`` through an opaque move: the lane math built on it is computed at its
    use instead of being hoisted out of a task loop into long-lived registers."""
    return fx.Int32(
        llvm.inline_asm(T.i32, [fx.Int32(v).ir_value()], "v_mov_b32 $0, $1", "=v,v", has_side_effects=True)
    )


def unlikely(cond):
    """``cond`` marked cold (llvm.expect), laid out of the hot path."""
    return fx.Boolean(
        llvm.call_intrinsic(
            ir.IntegerType.get_signless(1),
            "llvm.expect.i1",
            [as_ir_value(cond), as_ir_value(fx.Boolean(False))],
            [],
            [],
        )
    )


# --- math ----------------------------------------------------------------------
def _hw_f32(name, x):
    return fx.Float32(llvm.call_intrinsic(T.f32, name, [fx.Float32(x).ir_value()], [], []))


def hw_rsq(x):
    return _hw_f32("llvm.amdgcn.rsq.f32", x)


def hw_rcp(x):
    return _hw_f32("llvm.amdgcn.rcp.f32", x)


def hw_exp2(x):
    return _hw_f32("llvm.amdgcn.exp2.f32", x)


def div_rn(x, d, r):
    """``x / d`` rounded to nearest given ``r = 1.0 / d`` (Markstein)."""
    q = x * r
    return fx.Float32(fmath.fma(fx.Float32(fmath.fma(-q, d, x)), r, q))


def bf16_round(a):
    """``a`` rounded to fp32 then bf16, back as f32 (``sync_gfx1250.to_bf16``: no
    fused v_fma_mix*_bf16 single rounding)."""
    return fx.Float32(to_bf16(a))


def bf16_pair(a, b):
    """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
    return fx.Vector.from_elements([f32_pinned(a), f32_pinned(b)], fx.Float32).to(fx.BFloat16).bitcast(fx.Float32)[0]


def bf_lo(w):
    return (w << 16).bitcast(fx.Float32)


def bf_hi(w):
    return (w & fx.Int32(-65536)).bitcast(fx.Float32)


def fp8_pack4(a, b, c, d):
    """Four f32 (already clamped to the E4M3 range) -> one dword of E4M3, RNE.
    The CLAMP bit stays off: on MI450-B0 it selects the E5M3 conversion."""
    lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False))
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, c, d, lo, True))


# --- memory --------------------------------------------------------------------
def global_load(addr, words=1, nt=False):
    """``words`` i32 (1, 2 or 4) at the 64-bit global address ``addr`` (a pool past
    4 GiB, beyond a buffer load's 32-bit offset)."""
    i32 = ir.IntegerType.get_signless(32)
    ty = i32 if words == 1 else ir.VectorType.get([words], i32)
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(addr)))
    return llvm.LoadOp(ty, ptr.result, alignment=4 * words, nontemporal=nt or None).result


def ld_i32(ptr, i):
    return fx.Int32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32))


def ld_dev(ptr, i, words=1):
    """``words`` i32 at word ``i``, read at device scope (past the WGP cache)."""
    return bo.buffer_load(rsrc(ptr), i, vec_width=words, dtype=T.i32, cache_modifier=SCOPE_DEV)


def s_load_i32(ptr, i):
    """Word ``i`` (wave-uniform) at ``ptr`` as an ``s_buffer_load``: an SGPR, waited
    on the scalar counter -- not a VMEM load made uniform, whose ``s_wait_loadcnt``
    would also drain every vector load in flight (aiter ``buffer_load(is_scalar)``)."""
    i128 = llvm.ptrtoint(ir.IntegerType.get_signless(128), rocdl.get_buffer_rsrc(rsrc(ptr)))
    v4 = llvm.bitcast(ir.VectorType.get([4], ir.IntegerType.get_signless(32)), i128)
    off = as_ir_value(fx.Int32(i) * 4)
    return fx.Int32(llvm.call_intrinsic(T.i32, "llvm.amdgcn.s.buffer.load.i32", [v4, off, as_ir_value(fx.Int32(0))], [], []))


def s_load_ptr64(table, i):
    """Entry ``i`` of an i64 table by scalar loads (``s_load_i32``)."""
    lo, hi = s_load_i32(table, 2 * i), s_load_i32(table, 2 * i + 1)
    return (fx.Int64(hi) << 32) | fx.Int64(fx.Uint32(lo))


def load_ptr64(table, i):
    """Entry ``i`` of an i64 table, wave-uniform: a peer buffer base."""
    pw = fx.Vector(bo.buffer_load(rsrc(table), i * 2, vec_width=2, dtype=T.i32))
    return (fx.Int64(uniform(pw[1])) << 32) | fx.Int64(fx.Uint32(uniform(pw[0])))


LOAD_BATCH = 4


def first_rounds(tid, n, load, batch=LOAD_BATCH):
    rounds = -(-n // THREADS)
    return [load(fx.min(tid + THREADS * i, n - 1)) for i in range(min(batch, rounds))]


@traced
def batched_rounds(tid, n, load, use, batch=LOAD_BATCH, first=None):
    """A CTA's strided loop over v = tid + THREADS i < n with ``batch`` rounds of
    loads issued before any ``use``; a round past n loads element n - 1."""
    rounds = -(-n // THREADS)
    for b0 in range_constexpr(0, rounds, batch):
        idx = range(b0, min(b0 + batch, rounds))
        if const_expr(b0 == 0 and first is not None):
            got = first
        else:
            got = [load(fx.min(tid + THREADS * i, n - 1)) for i in idx]
        for i, g in zip(idx, got):
            v = tid + THREADS * i
            if v < n:
                use(v, g)
