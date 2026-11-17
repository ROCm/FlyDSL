# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 warp bitonic sort.

The network is the portable one, compare-exchange for compare-exchange: the
same pairs in the same order, the same strict ``before`` on each side, so
keys, payloads and the order of equal keys come out bit-identical to
``fx.coop.universal.warp_bitonic_sort``. What changes is how a step is
computed:

- **Moving the partner.** The portable step moves key, validity and payload
  with ``gpu.shuffle`` xor, which gfx942 runs on the LDS pipe
  (``ds_swizzle``/``ds_bpermute`` + ``s_waitcnt``). Here xor 1, 2 and 8 are
  single DPP moves (``quad_perm[1,0,3,2]``, ``quad_perm[2,3,0,1]``,
  ``row_ror:8``) and xor 4 is two (``row_half_mirror`` is xor 7 and
  ``quad_perm[3,2,1,0]`` xor 3). Only xor 16 (``ds_swizzle``) and xor 32
  (``ds_bpermute``) still cross a 16-lane row and stay on the LDS pipe --
  except on gfx950, where they are a ``v_permlane16_swap`` /
  ``v_permlane32_swap`` and one select.
- **No validity when there is none.** With ``valid_items=None`` and a
  power-of-two item count every slot is valid, and the portable validity word
  -- shuffled and compared at every step -- is dropped altogether.
- **One comparator call per lane.** The lane keeping the smaller asks
  ``before(partner, key)``, the other ``before(key, partner)``; selecting the
  operands per lane first makes one call the take predicate. The portable
  step calls the comparator for both orders and picks one with an i1 select,
  which the backend widens into 0/1 VGPRs.
- **min/max for integer keys.** With 32-bit integer keys and the default
  comparator, a compare-exchange is ``smaller ? min : max`` and the DPP move
  fuses into ``v_min_*_dpp``/``v_max_*_dpp``. Equal integer keys are equal
  bits, so a lane took its partner exactly when its key changed, and a
  payload follows ``new != old``. Invalid slots of a key-only sort become the
  dtype's maximum: they sort last, and the first ``valid_items`` outputs --
  the only specified ones -- are the sorted valid keys bit for bit.
"""

from .....compiler import jit
from .....expr import arith as _arith
from .....expr.gpu import lane_id, shuffle_xor
from .....expr.numeric import Int32, Integer, Numeric
from ..._common import _resolve_warp_width, _validate_valid_items
from ..._sorting import _before, _pack, _prepare, _unpack
from ..._values import _is_items
from ..bitonic_sort import warp_bitonic_sort as _portable_warp_bitonic_sort
from ._common import (
    _QUAD_REVERSE,
    _ROW_HALF_MIRROR,
    _SWIZZLE_XOR16,
    _XOR_DPP,
    _dpp,
    _has_permlane_swap,
    _is_gfx9,
    _permlane_swap,
    _swizzle,
)

__all__ = ["warp_bitonic_sort"]


def _move(value, ctrl):
    """One exact in-row move of a 32-bit value.

    Every lane reads an in-range lane, so ``bound_ctrl`` never changes a
    result; setting it is what lets ``GCNDPPCombine`` fold the move into the
    min/max that consumes it.
    """
    return _dpp(value, ctrl, old=value, bound_ctrl=True)


def _xor(value, distance):
    """Lane ``l`` reads lane ``l ^ distance`` of a 32-bit value."""
    if distance in _XOR_DPP:
        return _move(value, _XOR_DPP[distance])
    if distance == 4:
        return _move(_move(value, _ROW_HALF_MIRROR), _QUAD_REVERSE)
    if distance >= 16 and _has_permlane_swap():
        # gfx950 crosses rows and halves in the VALU: the swap hands every lane
        # both values of its pair, and the lane keeps the one that is not its own.
        from_lower, from_upper = _permlane_swap(value, value, distance)
        return ((lane_id() & distance) == 0).select(from_upper, from_lower)
    if distance == 16:
        return _swizzle(value, _SWIZZLE_XOR16)
    return value.dtype(shuffle_xor(value, distance, 64))


def _is_word(value):
    return isinstance(value, Numeric) and value.dtype.width == 32


def _applies(items, payload):
    """gfx9 DPP controls, and keys/payloads that are one 32-bit word each."""
    if not _is_gfx9():
        return False
    return all(_is_word(v) for v in items) and (payload is None or all(_is_word(v) for v in payload))


@jit
def _guarded_before(a, b, both, compare_op=None):
    # The comparator runs only where both entries are valid, as in the portable
    # ``_valid_before``. Its other arm is the constant false, so the merge is an
    # i1 AND instead of an i1 select the backend would widen into VGPRs.
    before = both & False
    if both:
        before = _before(a, b, False, compare_op)
    return before


def _take(x, y, vx, vy, compare_op):
    """``_valid_before(x, y, vx, vy)``: invalid entries follow every valid one."""
    if vx is None:
        return _before(x, y, False, compare_op)
    x_ok, y_ok = vx != 0, vy != 0
    return (x_ok & ~y_ok) | _guarded_before(x, y, x_ok & y_ok, compare_op)


def _sentinel(dtype):
    return dtype(2**32 - 1 if not dtype.signed else 2**31 - 1)


def _network(items, payload, valid, width, compare_op, minmax):
    lane = lane_id() % width
    count = len(items)
    total = width * count
    span = 2
    while span <= total:
        distance = span // 2
        while distance:
            out, carried, out_valid = [], [], []
            for i, key in enumerate(items):
                own_valid = None if valid is None else valid[i]
                own_payload = None if payload is None else payload[i]
                if distance < width:
                    partner = _xor(key, distance)
                    other_valid = None if valid is None else _xor(own_valid, distance)
                    other_payload = None if payload is None else _xor(own_payload, distance)
                    index = lane + i * width
                    smaller = ((index & span) == 0) == ((index & distance) == 0)
                else:
                    j = i ^ (distance // width)
                    partner = items[j]
                    other_valid = None if valid is None else valid[j]
                    other_payload = None if payload is None else payload[j]
                    # Both bits lie above the lane bits, so the side is static.
                    index = i * width
                    smaller = ((index & span) == 0) == ((index & distance) == 0)
                if minmax:
                    low = key.dtype(_arith.min(key, partner))
                    high = key.dtype(_arith.max(key, partner))
                    new = smaller.select(low, high) if not isinstance(smaller, bool) else (low if smaller else high)
                    out.append(new)
                    if payload is not None:
                        carried.append((new != key).select(other_payload, own_payload))
                    continue
                if isinstance(smaller, bool):
                    x, y = (partner, key) if smaller else (key, partner)
                    vx, vy = (other_valid, own_valid) if smaller else (own_valid, other_valid)
                else:
                    x, y = smaller.select(partner, key), smaller.select(key, partner)
                    vx = None if valid is None else smaller.select(other_valid, own_valid)
                    vy = None if valid is None else smaller.select(own_valid, other_valid)
                take = _take(x, y, vx, vy, compare_op)
                out.append(take.select(partner, key))
                if valid is not None:
                    out_valid.append(take.select(other_valid, own_valid))
                if payload is not None:
                    carried.append(take.select(other_payload, own_payload))
            items = out
            if valid is not None:
                valid = out_valid
            if payload is not None:
                payload = carried
            distance //= 2
        span *= 2
    return items, payload


def warp_bitonic_sort(
    keys,
    values=None,
    *,
    width: int | None = None,
    compare_op=None,
    valid_items: int | Integer | None = None,
    _dtype=None,
    _value_dtype=None,
):
    if compare_op is not None and not callable(compare_op):
        raise TypeError("compare_op must be a strict ordering callable")
    if not _is_items(keys, _dtype):
        raise TypeError(
            "keys must be a nonempty outer item sequence; wrap one complete key in [key] or (key,). "
            "A Vector represents an item sequence only for a Numeric key dtype"
        )
    width = _resolve_warp_width(width, "warp_bitonic_sort width")
    items, payload, vector = _unpack(keys, values, _value_dtype, _dtype)
    if not _applies(items, payload):
        return _portable_warp_bitonic_sort(
            keys,
            values,
            width=width,
            compare_op=compare_op,
            valid_items=valid_items,
            _dtype=_dtype,
            _value_dtype=_value_dtype,
        )
    count = len(items)
    _validate_valid_items(valid_items, width * count)
    lane = lane_id() % width
    items, payload, valid = _prepare(items, payload, [lane + i * width for i in range(count)], valid_items)
    padded = 1 << (count - 1).bit_length()
    # Padding reuses a well-typed value and is invalid, exactly as portable.
    items += [items[-1]] * (padded - count)
    valid += [Int32(0)] * (padded - count)
    if payload is not None:
        payload += [payload[-1]] * (padded - count)
    all_valid = valid_items is None and padded == count
    minmax = compare_op is None and issubclass(items[0].dtype, Integer)
    if minmax and not all_valid and payload is None:
        # Key-only: an invalid slot becomes the maximum, so it sorts last and
        # the valid prefix is exactly the sorted valid keys.
        top = _sentinel(items[0].dtype)
        items = [(v != 0).select(key, top) for key, v in zip(items, valid)]
        all_valid = True
    elif minmax and not all_valid:
        minmax = False
    items, payload = _network(items, payload, None if all_valid else valid, width, compare_op, minmax)
    return _pack(items[:count], payload[:count] if payload is not None else None, vector, keys, values)


warp_bitonic_sort.__doc__ = _portable_warp_bitonic_sort.__doc__
