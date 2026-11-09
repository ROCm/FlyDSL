# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx9 ``BlockRadixRank``: warp match masks from one ``ballot`` per digit bit.

The portable MATCH policies build each item's match mask with one shuffle per
lane -- 64 ``v_readlane`` and about five VALU each on wave64 -- before
counting the lanes below with a 64-bit popcount. Here:

- **match** (``_match_any``) -- a lane's mask is the set of lanes that differ
  from it in no digit bit. Per bit: the bit spread to 0/-1 (``v_bfe_i32``) is
  the ballot predicate (``v_cmp`` into an SGPR pair) and also the XOR that
  turns the ballot into the differing lanes; two ``v_xor`` and two ``v_or``
  accumulate them. ``radix_bits`` ballots instead of ``width`` shuffles.
- **rank in mask** (``_lanes_below``) -- ``v_mbcnt_lo/hi`` count the mask
  bits below the lane.
- **early counts** (``_early_masks``) -- MATCH_EARLY_COUNTS_ANY forms every
  item's mask up front (two VGPRs each), so each digit's first lane adds the
  digit's whole count to the warp histogram -- one LDS atomic per distinct
  digit instead of one per lane, which serializes on skewed digits -- and the
  rounds reuse the masks.

All rely on the logical warp being the physical wave: radix rank warps are
whole waves, or the single sub-wave block, whose lanes start at lane 0. The
policy, storage, synchronization and results are the portable class's;
another target falls back to it. MATCH_EARLY_COUNTS_ATOMIC_OR keeps its
shared atomic-OR masks, and BASIC/MEMOIZE need no override.
"""

from ....._mlir import ir
from ....._mlir.dialects import rocdl as _rocdl
from .....expr.numeric import Int32, Int64, Uint32, Uint64
from .....expr.rocdl import ballot
from ...._dispatch import override
from ...warp.rocdl._common import _is_gfx9
from ..radix_rank import BlockRadixRank as _PortableBlockRadixRank

__all__ = ["BlockRadixRank"]


def _ballot_match(digit, width, bits):
    """Lanes whose digit equals this lane's: no lane differs from it in any bit."""
    i64 = ir.IntegerType.get_signless(64)
    differ = None
    for b in range(bits):
        spread = (digit << (31 - b)) >> 31
        votes = Uint64(ballot(i64, (spread != 0).ir_value()))
        lanes = votes ^ spread.to(Int64).to(Uint64)
        differ = lanes if differ is None else differ | lanes
    mask = Uint64((1 << 64) - 1) if differ is None else ~differ
    # Lanes outside a sub-wave block are inactive and vote 0, which reads as
    # a match for a 0 bit: keep only the block's lanes.
    return mask & Uint64((1 << width) - 1) if width < 64 else mask


def _mbcnt(mask):
    i32 = ir.IntegerType.get_signless(32)
    low = _rocdl.mbcnt_lo(i32, mask.to(Uint32).ir_value(), Int32(0).ir_value())
    return Int32(_rocdl.mbcnt_hi(i32, (mask >> 32).to(Uint32).ir_value(), low))


class BlockRadixRank(_PortableBlockRadixRank):
    __doc__ = _PortableBlockRadixRank.__doc__

    @override
    @classmethod
    def _match_any(cls, digit, width, bits):
        if not _is_gfx9():
            return super()._match_any(digit, width, bits)
        return _ballot_match(digit, width, bits)

    @override
    @classmethod
    def _lanes_below(cls, mask, lane):
        if not _is_gfx9():
            return super()._lanes_below(mask, lane)
        return _mbcnt(mask)

    @override
    @classmethod
    def _early_masks(cls, digits, width, bits):
        if not _is_gfx9():
            return super()._early_masks(digits, width, bits)
        return [_ballot_match(digit, width, bits) for digit in digits]
