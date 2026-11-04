# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Execution model of the gfx1250 (MI455X) mono kernels.

One CTA per WGP, every CTA co-resident: ``BLOCKS`` = 256 = the WGPs MI455X
exposes (``cu_num``). A CTA is 512 threads = 16 wave32 waves = 4 waves a SIMD,
which leaves each lane 1024 / 4 = 256 VGPRs, the per-thread budget of the
gfx950 kernels' 512-thread CTA. MI350's wave ``w`` maps to waves ``2w`` and
``2w + 1`` here, so a thread keeps its MI350 index and the element maps of the
elementwise stages carry over unchanged.

Stages hand off through tagged mailboxes; there is no grid barrier. A stage's
task ``t`` runs on CTA ``(base + t) % BLOCKS`` (``first_task``).
"""

from __future__ import annotations

BLOCKS = 256
THREADS = 512
WAVE = 32
WAVES = THREADS // WAVE

# The LDS a CTA requests. Above half the 320 KiB WGP maximum, so the dispatcher
# cannot pack two CTAs into one WGP and leave another idle; ``check_device``
# confirms the partition actually grants it.
LDS_BYTES = 192 * 1024
LDS_MAX = 320 * 1024  # a WGP's LDS (the 384 KB pool less the >= 64 KB WGP cache)
LDS_GRANULE = 2048  # SPG: 2 KiB allocation blocks (another section says 1 KiB)
VGPR_MAX = 1024 // (WAVES // 4)  # 256: 16 waves over 4 SIMDs of 1024 VGPRs

assert WAVES == 16 and BLOCKS & (BLOCKS - 1) == 0


def first_task(bid, base):
    """The first task of a stage placed from CTA ``base`` that CTA ``bid`` runs; it
    then takes every ``BLOCKS``-th task. ``base`` may pass BLOCKS (a stage after a
    long one): reduced here, since the device remainder of a negative is negative."""
    return (bid + (BLOCKS - base % BLOCKS)) % BLOCKS


def lds_alloc(nbytes: int, granule: int = LDS_GRANULE) -> int:
    """The LDS a request of ``nbytes`` occupies once rounded to whole blocks."""
    return -(-nbytes // granule) * granule


def check_device(cu_num: int, max_lds_per_block: int, arch: str) -> None:
    """Refuse a device the schedule cannot be co-resident on (host side)."""
    if not arch.startswith("gfx1250"):
        raise RuntimeError(f"gfx1250 mono kernels need gfx1250, got {arch}")
    if cu_num != BLOCKS:
        raise RuntimeError(f"{BLOCKS} co-resident CTAs need {BLOCKS} WGPs, device has {cu_num}")
    if max_lds_per_block < LDS_BYTES:
        raise RuntimeError(f"LDS partition grants {max_lds_per_block} B a block, the kernels need {LDS_BYTES}")
