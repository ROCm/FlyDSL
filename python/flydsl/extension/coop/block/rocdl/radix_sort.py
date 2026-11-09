# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 ``BlockRadixSort``: packed-counter and ballot ranking, one scatter per digit pass.

The portable sort ranks a one-bit digit with a full ``BlockScan`` (DPP ladder,
warp-total exchange, barrier) and each wider digit with one block scan per bin.
On a wave64 CDNA part a digit pass needs neither:

- **Digits of two or three bits ranked by one packed counter.** Every bin owns
  an ``fw``-bit field (``fw`` = bit length of ``wave_threads * items - 1``) of a
  one- or two-dword counter; each thread adds ``1 << (d * fw)`` per item, so one
  DPP lane scan per dword is every bin's lane prefix at once, and one bit-field
  extract per item reads its own. The wave's per-bin counts (``v_readlane`` of
  the last lane) go to a digit-major table; after one barrier its exclusive
  scan across lanes is every wave's base for every bin, and ``ds_bpermute``
  hands each item the base of its digit.
- **One-bit digits ranked by ballots.** ``ballot(bit == 0)`` per item,
  ``s_bcnt1`` for the wave's count and a ``v_mbcnt_lo/hi`` chain for the lanes
  below: no DPP ladder.
- **Two barriers per pass.** The digit pass scatters every 32-bit word of key
  and payload at once (the portable ``_exchange``); the next pass's count
  barrier already orders this read-back before the next scatter. A single-wave
  block needs no count exchange. The sort synchronizes once before returning.

Results are bit-identical to the portable sort: both are stable LSD radix sorts
over the same transformed digits, and an LSD sort's result does not depend on
its digit width -- so the width here is a performance choice (``_pass_bits``):
the widest the packed counter takes, unless ``radix_bits`` names one that ranks
well. Other targets keep the portable implementation and its storage.
"""

from ....._mlir import ir
from ....._mlir.dialects import arith as _arith
from ....._mlir.dialects import math as _math
from ....._mlir.dialects import rocdl as _rocdl
from .....compiler import jit
from .....compiler.backends import current_target
from .....expr.gpu import barrier
from .....expr.numeric import Int32, Uint32, Uint64
from .....expr.primitive import ptr_load
from .....expr.rocdl import ballot, ds_bpermute, readfirstlane, readlane
from .....expr.struct import Struct
from .....expr.typing import Array, ReductionOp, Vector
from ...._dispatch import override
from ..._common import _linear_thread_id
from ..._sorting import _pack, _unpack
from ...warp.rocdl.scan import _inclusive_scan
from .. import radix_sort as _portable
from ..radix_sort import BlockRadixSortAlgorithm, _columns, _exchange

__all__ = ["BlockRadixSort"]

_WAVE = 64
# Widest digit ranked in one pass; wider requests run at the widest width the
# packed counter takes (identical results).
_MAX_BITS = 3
# One-bit passes rank by ballots, whose per-item cost overtakes the portable
# block scan beyond this many items per thread (0.92x latency at 6 items); an
# explicit radix_bits=1 above it runs at the packed width instead.
_BALLOT_MAX_ITEMS = 4


def _applies():
    target = current_target()
    return target.arch.startswith("gfx9") and target.warp_size == _WAVE


def _i32():
    return ir.IntegerType.get_signless(32)


def _i64():
    return ir.IntegerType.get_signless(64)


def _ballot(pred):
    """Uniform 64-bit mask of the lanes whose *pred* holds (an SGPR pair)."""
    return ballot(_i64(), pred.ir_value())


def _popc(mask):
    """Lanes set in a uniform mask: ``s_bcnt1_i32_b64``."""
    return Int32(_arith.trunci(_i32(), _math.ctpop(mask)))


def _mbcnt(masks, acc):
    """``acc`` plus the lanes below this one set in each mask (``v_mbcnt_lo/hi`` chain)."""
    value = acc.ir_value()
    shift = _arith.constant(_i64(), 32)
    for mask in masks:
        lo = _arith.trunci(_i32(), mask)
        hi = _arith.trunci(_i32(), _arith.shrui(mask, shift))
        value = _rocdl.mbcnt_lo(_i32(), lo, value)
        value = _rocdl.mbcnt_hi(_i32(), hi, value)
    return Int32(value)


def _storage(dtype, threads, warp, count):
    sort = _portable._storage(dtype, threads, warp, count)
    if not _applies():
        return sort
    # A digit-major count table of at most one entry per lane of a wave.
    return Struct["sort":sort, "counts" : Array[Int32, _WAVE, 16]]


# -- one-bit digits ---------------------------------------------------------------


@jit
def _publish_count(counts, warp, lane, value):
    if lane == 0:
        counts[warp] = value


def _ballot_ranks(digits, storage, warp, lane, waves, wave_threads, count):
    """Blocked-order stable ranks of one-bit *digits*."""
    zero = [d == 0 for d in digits]
    masks = [_ballot(z) for z in zero]
    total = _popc(masks[0])
    for mask in masks[1:]:
        total = total + _popc(mask)
    if waves > 1:
        _publish_count(storage.counts, warp, lane, total)
    # Count-independent part, before the barrier: zeros ahead of each item in the wave.
    below = _mbcnt(masks, Int32(0))
    ahead, zeros = [], Int32(0)
    for z in zero:
        ahead.append(below + zeros)
        zeros = zeros + Int32(z)
    if waves > 1:
        barrier()
        table = ptr_load(storage.counts.ptr, Vector[Int32, waves])
        before, everything = Int32(0), table[waves - 1]
        for w in range(waves - 1):
            before = before + (warp > w).select(table[w], Int32(0))
            everything = everything + table[w]
    else:
        before, everything = Int32(0), total
    ones = everything + warp * (wave_threads * count) - before + lane * count
    return [z.select(before + x, ones + i - x) for i, (z, x) in enumerate(zip(zero, ahead))]


# -- wider digits -----------------------------------------------------------------


@jit
def _publish_table(table, index, lane, bins, value):
    if lane < bins:
        table[index] = value


def _packed_layout(bins, wave_threads, count):
    """``(field_bits, dwords)`` of the packed bin counter, or ``None`` when it does not fit.

    A field holds a lane's exclusive prefix plus its own earlier items, at most
    ``wave_threads * count - 1``. One dword takes any width that fits; two
    dwords use 8- or 16-bit fields so that no field straddles the dwords, which
    are scanned separately.
    """
    need = max(1, (wave_threads * count - 1).bit_length())
    if bins * need <= 32:
        return need, 1
    for fw in (8, 16):
        if need <= fw and bins * fw <= 64:
            return fw, 2
    return None


def _halves(value):
    return [(value >> (32 * h)).to(Uint32).bitcast(Int32) for h in range(2)]


def _join(lo, hi):
    return lo.bitcast(Uint32).to(Uint64) | (hi.bitcast(Uint32).to(Uint64) << 32)


def _lane_scan_exclusive(total, dwords, wave_threads):
    if dwords == 1:
        return _inclusive_scan(total, ReductionOp.ADD, wave_threads) - total
    return _join(*[_inclusive_scan(x, ReductionOp.ADD, wave_threads) - x for x in _halves(total)])


def _lane_value(value, dwords, lane):
    """``value`` as held by *lane* (uniform: v_readlane into SGPRs)."""
    if dwords == 1:
        return Int32(readlane(_i32(), value.ir_value(), lane))
    return _join(*[Int32(readlane(_i32(), x.ir_value(), lane)) for x in _halves(value)])


def _packed_ranks(digits, bins, storage, warp, lane, waves, wave_threads, count):
    """Blocked-order stable ranks of *digits* (Int32 in ``[0, bins)``), all bins in one counter."""
    fw, dwords = _packed_layout(bins, wave_threads, count)
    packed = Int32 if dwords == 1 else Uint64
    mask = (1 << fw) - 1
    shifts = [d * fw for d in digits]
    amounts = shifts if dwords == 1 else [s.bitcast(Uint32).to(Uint64) for s in shifts]
    running, total = [], packed(0)
    for s in amounts:
        running.append(total)
        total = total + (packed(1) << s)
    excl = _lane_scan_exclusive(total, dwords, wave_threads)
    last = wave_threads - 1
    e_last, t_last = _lane_value(excl, dwords, last), _lane_value(total, dwords, last)
    field = (lane & (bins - 1)) * fw
    if dwords == 2:
        field = field.bitcast(Uint32).to(Uint64)
    # Lane k holds the wave's bin-k count; the fields are added apart so none carries.
    mine = ((e_last >> field) & mask) + ((t_last >> field) & mask)
    if dwords == 2:
        mine = mine.to(Uint32).bitcast(Int32)
    entries = bins * waves
    if waves > 1:
        _publish_table(storage.counts, lane * waves + warp, lane, bins, mine)
    # Count-independent part, before the barrier.
    inner = []
    for r, s in zip(running, amounts):
        local = ((excl + r) >> s) & mask
        inner.append(local if dwords == 1 else local.to(Uint32).bitcast(Int32))
    sources = [(d * waves + warp) * 4 for d in digits]
    if waves > 1:
        barrier()
        mine = storage.counts[(lane < entries).select(lane, Int32(0))]
    width = 1 << max(0, (entries - 1).bit_length())
    table = (_inclusive_scan(mine, ReductionOp.ADD, width) - mine).ir_value()
    return [Int32(ds_bpermute(_i32(), src.ir_value(), table)) + local for src, local in zip(sources, inner)]


class _BlockRadixSortMeta(_portable._BlockRadixSortMeta):
    _shared_storage = {algorithm: _storage for algorithm in BlockRadixSortAlgorithm}


class BlockRadixSort(_portable.BlockRadixSort, metaclass=_BlockRadixSortMeta):
    __doc__ = _portable.BlockRadixSort.__doc__

    @classmethod
    def _geometry(cls):
        wave_threads = min(cls.block_threads, _WAVE)
        return cls.block_threads // wave_threads, wave_threads

    @classmethod
    def _packed_fits(cls, bits):
        waves, wave_threads = cls._geometry()
        bins = 1 << bits
        return _packed_layout(bins, wave_threads, cls.items_per_thread) is not None and bins * waves <= wave_threads

    @classmethod
    def _auto_bits(cls):
        for bits in range(_MAX_BITS, 1, -1):
            if cls._packed_fits(bits):
                return bits
        return 1

    @classmethod
    def _pass_bits(cls, requested):
        """Digit width used: *requested* when it ranks well, else the widest that does."""
        if requested is None:
            return cls._auto_bits()
        if requested == 1:
            return 1 if cls.items_per_thread <= _BALLOT_MAX_ITEMS else cls._auto_bits()
        if requested <= _MAX_BITS and cls._packed_fits(requested):
            return requested
        return min(requested, cls._auto_bits())

    @override
    @classmethod
    def _sort(cls, keys, values, storage, descending, begin_bit, end_bit, decomposer, radix_bits, striped):
        valid = radix_bits is None or (
            isinstance(radix_bits, int) and not isinstance(radix_bits, bool) and 1 <= radix_bits < 31
        )
        if not _applies() or not valid:
            sort = storage.sort if _applies() else storage
            return super()._sort(keys, values, sort, descending, begin_bit, end_bit, decomposer, radix_bits, striped)
        cls._check(keys)
        items, payload, vector = _unpack(keys, values, cls.value_dtype, cls.dtype)
        if payload is not None and cls.value_dtype is None:
            raise TypeError("specialize [(key_dtype, value_dtype), block_size, items_per_thread] for pairs")
        width = sum(field.dtype.width for field in _portable._fields(items[0], decomposer))
        end_bit = _portable._bit_range(width, begin_bit, end_bit)
        bits = cls._pass_bits(radix_bits)
        passes = list(range(begin_bit, end_bit, bits))
        if not passes and not striped:
            return _pack(items, payload, vector, keys, values)

        count, threads = cls.items_per_thread, cls.block_threads
        tid = _linear_thread_id(cls.block_size)
        waves, wave_threads = cls._geometry()
        lane = tid % wave_threads if waves > 1 else tid
        warp = Int32(readfirstlane(_i32(), (tid // wave_threads).ir_value())) if waves > 1 else Int32(0)

        key_columns, rebuild_keys = _columns(items)
        value_columns, rebuild_values = _columns(payload) if payload is not None else ([], None)
        # An empty bit range still moves the tile once when striped output is requested.
        for index, bit in enumerate(passes or [None]):
            if bit is None:
                ranks = [tid * count + i for i in range(count)]
            else:
                digit_bits = min(bits, end_bit - bit)
                digits = []
                for item in items:
                    digit = _portable._digit(item, bit, digit_bits, True, decomposer)
                    digits.append(digit ^ ((1 << digit_bits) - 1) if descending else digit)
                if digit_bits == 1:
                    ranks = _ballot_ranks(digits, storage, warp, lane, waves, wave_threads, count)
                else:
                    ranks = _packed_ranks(digits, 1 << digit_bits, storage, warp, lane, waves, wave_threads, count)
            # The last pass reads the requested layout straight out of its scatter.
            last = index == max(len(passes), 1) - 1
            columns = _exchange(key_columns + value_columns, ranks, tid, storage.sort, threads, count, striped and last)
            key_columns, value_columns = columns[: len(key_columns)], columns[len(key_columns) :]
            items = rebuild_keys(key_columns)
        # Every thread has read the tile back before any thread returns.
        barrier()
        if payload is not None:
            payload = rebuild_values(value_columns)
        return _pack(items, payload, vector, keys, values)
