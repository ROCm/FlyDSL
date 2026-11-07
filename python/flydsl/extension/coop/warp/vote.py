# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Warp-wide predicate voting, ballot masks, and compaction ranks — the portable form."""

from ....expr.gpu import lane_id
from ....expr.numeric import Boolean, Int32, Integer, Uint32, Uint64
from ....expr.struct import Empty
from ....expr.typing import ReductionOp
from .._common import _resolve_warp_width, _shuffle_value
from .._values import _is_items, _normalize_value
from ._spec import WarpPrimitive
from .reduce import warp_reduce as _portable_warp_reduce
from .scan import warp_scan_with_aggregate as _portable_warp_scan_with_aggregate

__all__ = [
    "WarpVote",
    "warp_any",
    "warp_all",
    "warp_ballot",
    "warp_count",
    "warp_ballot_rank",
]


def _as_predicate(predicate: object) -> Boolean:
    """Convert a Boolean or integer lane flag into a ``Boolean`` predicate."""
    predicate = _normalize_value(predicate)
    if isinstance(predicate, Boolean):
        return predicate
    if isinstance(predicate, Integer):
        return predicate != predicate.dtype(0)
    if isinstance(predicate, (bool, int)):
        return Boolean(bool(predicate))
    raise TypeError(f"expected a Boolean or integer predicate, got {type(predicate).__name__}")


def warp_any(
    predicate: bool | int | Integer,
    *,
    width: int | None = None,
) -> Boolean:
    """Return ``True`` in every lane if any lane in the logical warp has a nonzero predicate.

    Every lane in each logical warp must participate.

    Args:
        predicate: Lane predicate (``Boolean`` or integer flag where nonzero is true).
        width: Compile-time power-of-two logical width, at most the native warp width.
            ``None`` uses the native width.

    Returns:
        ``Boolean`` uniform across each logical warp.
    """
    width = _resolve_warp_width(width, "warp_any width")
    pred = _as_predicate(predicate)
    if width == 1:
        return pred
    return _portable_warp_reduce(pred, ReductionOp.ADD, width=width)


def warp_all(
    predicate: bool | int | Integer,
    *,
    width: int | None = None,
) -> Boolean:
    """Return ``True`` in every lane if every lane in the logical warp has a nonzero predicate.

    Every lane in each logical warp must participate.

    Args:
        predicate: Lane predicate (``Boolean`` or integer flag where nonzero is true).
        width: Compile-time power-of-two logical width, at most the native warp width.
            ``None`` uses the native width.

    Returns:
        ``Boolean`` uniform across each logical warp.
    """
    width = _resolve_warp_width(width, "warp_all width")
    pred = _as_predicate(predicate)
    if width == 1:
        return pred
    return _portable_warp_reduce(pred, ReductionOp.MUL, width=width)


def warp_ballot(
    predicate: bool | int | Integer,
    *,
    width: int | None = None,
) -> Uint64:
    """Collect lane predicates into a logical-warp bitmask returned to every lane.

    Bit ``k`` (``0 <= k < width``) of the returned ``Uint64`` is ``1`` if and only if
    logical lane ``k`` (``lane_id() % width == k``) supplied a nonzero predicate; all
    bits at or above ``width`` are ``0``. Returning ``Uint64`` uniformly keeps the
    result type identical across wave32 and wave64 targets.

    Args:
        predicate: Lane predicate (``Boolean`` or integer flag where nonzero is true).
        width: Compile-time power-of-two logical width, at most the native warp width.
            ``None`` uses the native width.

    Returns:
        ``Uint64`` bitmask in ``[0, 2**width)`` in every lane of the logical warp.
    """
    width = _resolve_warp_width(width, "warp_ballot width")
    pred = _as_predicate(predicate)
    if width == 1:
        return pred.select(Uint64(1), Uint64(0))
    in_group = lane_id() % width
    if width <= 32:
        mask32 = pred.select(Uint32(1) << Uint32(in_group), Uint32(0))
        offset = 1
        while offset < width:
            mask32 = mask32 | _shuffle_value(mask32, offset, width, mode="xor")
            offset <<= 1
        return Uint64(mask32)
    mask64 = pred.select(Uint64(1) << Uint64(in_group), Uint64(0))
    offset = 1
    while offset < width:
        mask64 = mask64 | _shuffle_value(mask64, offset, width, mode="xor")
        offset <<= 1
    return mask64


def warp_count(
    predicate: bool | int | Integer,
    *,
    width: int | None = None,
) -> Int32:
    """Count lanes with a nonzero predicate in each logical warp.

    Every lane in each logical warp must participate.

    Args:
        predicate: Lane predicate (``Boolean`` or integer flag where nonzero is true).
        width: Compile-time power-of-two logical width, at most the native warp width.
            ``None`` uses the native width.

    Returns:
        ``Int32`` active-lane count in ``[0, width]`` in every lane of the logical warp.
    """
    width = _resolve_warp_width(width, "warp_count width")
    pred = _as_predicate(predicate)
    one = pred.select(Int32(1), Int32(0))
    if width == 1:
        return one
    return _portable_warp_reduce(one, ReductionOp.ADD, width=width)


def warp_ballot_rank(
    predicate: bool | int | Integer,
    *,
    width: int | None = None,
) -> tuple[Int32, Int32]:
    """Compute each lane's exclusive active-lane rank and the logical-warp total count.

    Useful for register-level stream compaction: lanes with ``predicate`` true write
    to ``base_offset + rank`` and advance ``base_offset`` by ``total``.

    Args:
        predicate: Lane predicate (``Boolean`` or integer flag where nonzero is true).
        width: Compile-time power-of-two logical width, at most the native warp width.
            ``None`` uses the native width.

    Returns:
        Tuple ``(exclusive_rank, total_count)`` of ``Int32`` values in every lane of
        the logical warp, where ``0 <= exclusive_rank <= total_count <= width``.
    """
    width = _resolve_warp_width(width, "warp_ballot_rank width")
    pred = _as_predicate(predicate)
    one = pred.select(Int32(1), Int32(0))
    if width == 1:
        return Int32(0), one
    _, exclusive, total = _portable_warp_scan_with_aggregate(one, ReductionOp.ADD, width=width)
    return exclusive, total


class WarpVote(WarpPrimitive):
    """Vote, ballot, and compaction-rank collectives over a logical warp.

    Specialize with ``WarpVote[dtype, width]``. The parameters below are compile-time
    positional arguments to ``WarpVote[...]``, in bracket order.

    Args:
        dtype: Required scalar predicate type (``Boolean`` or any scalar ``Integer``
            type; nonzero integer values evaluate as true).
        width: Optional; defaults to ``None``. A positive power-of-two Python ``int``
            no larger than the target physical warp width, or ``None`` to use that
            physical width.

    Every lane in each logical warp must participate. ``SharedStorage`` is ``Empty``.

    Examples:
        V = fx.coop.WarpVote[fx.Boolean, 8]
        mask = V.ballot(x > 0)
        rank, count = V.ballot_rank(x > 0)
    """

    @classmethod
    def _validate_specialization(cls) -> None:
        if not issubclass(cls.dtype, Integer):
            raise TypeError(f"WarpVote requires a Boolean or integer scalar dtype, got {cls.dtype.__name__}")

    @classmethod
    def _prepare_predicate(cls, predicate: object) -> Boolean:
        value = cls._prepare(predicate)
        if _is_items(value, cls.dtype):
            raise TypeError("WarpVote methods expect one scalar predicate per lane")
        return _as_predicate(value)

    @classmethod
    def any(cls, predicate: bool | int | Integer, *, storage: Empty | None = None) -> Boolean:
        """Return ``True`` if any lane in the logical warp has a nonzero predicate."""
        pred = cls._prepare_predicate(predicate)
        return cls._invoke(warp_any, pred, storage=storage)

    @classmethod
    def all(cls, predicate: bool | int | Integer, *, storage: Empty | None = None) -> Boolean:
        """Return ``True`` if every lane in the logical warp has a nonzero predicate."""
        pred = cls._prepare_predicate(predicate)
        return cls._invoke(warp_all, pred, storage=storage)

    @classmethod
    def ballot(cls, predicate: bool | int | Integer, *, storage: Empty | None = None) -> Uint64:
        """Return the ``Uint64`` logical-warp ballot bitmask in ``[0, 2**width)``."""
        pred = cls._prepare_predicate(predicate)
        return cls._invoke(warp_ballot, pred, storage=storage)

    @classmethod
    def count(cls, predicate: bool | int | Integer, *, storage: Empty | None = None) -> Int32:
        """Return the ``Int32`` count of nonzero predicates in the logical warp."""
        pred = cls._prepare_predicate(predicate)
        return cls._invoke(warp_count, pred, storage=storage)

    @classmethod
    def ballot_rank(cls, predicate: bool | int | Integer, *, storage: Empty | None = None) -> tuple[Int32, Int32]:
        """Return ``(exclusive_rank, total_count)`` as ``Int32`` values."""
        pred = cls._prepare_predicate(predicate)
        return cls._invoke(warp_ballot_rank, pred, storage=storage)
