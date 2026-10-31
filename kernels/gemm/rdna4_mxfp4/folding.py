# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Exact FP4 magnitude tables generated in registers, without a memory lookup."""

import flydsl.expr as fx


def magnitude_words(shift, mode, normal_only):
    # Give code zero a temporary .5 encoding to prevent cross-byte borrow,
    # then clear byte zero after the independent exponent adjustments.
    shift = fx.Int32(shift)
    if mode == "e4m3":
        offset = fx.Uint32(shift) * fx.Uint32(0x08080808)
        lo = (fx.Uint32(0x3C383030) + offset) & fx.Uint32(0xFFFFFF00)
        hi = fx.Uint32(0x4C484440) + offset
        if not normal_only:
            lo = (shift == -6).select(fx.Uint32(0x0C080400), lo)
            hi = (shift == -6).select(fx.Uint32(0x1C181410), hi)
            lo = (shift == -7).select(fx.Uint32(0x06040200), lo)
            hi = (shift == -7).select(fx.Uint32(0x14100C08), hi)
            lo = (shift == -8).select(fx.Uint32(0x03020100), lo)
            hi = (shift == -8).select(fx.Uint32(0x0C080604), hi)
    else:
        offset = fx.Uint32(shift) * fx.Uint32(0x04040404)
        lo = (fx.Uint32(0x3E3C3838) + offset) & fx.Uint32(0xFFFFFF00)
        hi = fx.Uint32(0x46444240) + offset
        if not normal_only:
            lo = (shift == -14).select(fx.Uint32(0x06040200), lo)
            hi = (shift == -14).select(fx.Uint32(0x0E0C0A08), hi)
            lo = (shift == -15).select(fx.Uint32(0x03020100), lo)
            hi = (shift == -15).select(fx.Uint32(0x0A080604), hi)
    return lo, hi
