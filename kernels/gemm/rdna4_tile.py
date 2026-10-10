# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Shared WMMA tile record for the gfx120x GEMM hosts."""

from dataclasses import dataclass

_WARP = 32


@dataclass(frozen=True)
class TileConfig:
    """WMMA launch tile. ``threads`` is the block size. ``name`` is the tile label."""

    bm: int
    bn: int
    bk: int
    warps_m: int
    warps_n: int
    tm: int
    tn: int

    @property
    def threads(self) -> int:
        """Threads in the block: warps_m * warps_n * 32."""
        return self.warps_m * self.warps_n * _WARP

    @property
    def name(self) -> str:
        """Tile label, for logs and the autotune table."""
        return f"{self.bm}x{self.bn}x{self.bk}_w{self.warps_m}x{self.warps_n}_t{self.tm}x{self.tn}"
