# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""FlyDSL port of the mxfp4 activation scale-swizzle (drop-in for the Triton fused_mxfp4_swizzle).

Reorders the row-major e8m0 block-scale into the MFMA-consumed tiled layout, identical byte layout
to ``fused_swizzle.fused_mxfp4_swizzle`` (so the fused group-GEMM consumes it unchanged).  Assumes the
input is already expert-major (identity sorted_ids, TOPK=1) — the deployed fused-stage1 case where the
sort is folded into dispatch.

Output layout (matches Triton): uint32 [M/32, N/8, 4, 16], where each u32[a,b] packs 4 e8m0 bytes
  byte0=(row R+a,   col C+b)   byte1=(row R+16+a, col C+b)
  byte2=(row R+a, col C+4+b)   byte3=(row R+16+a, col C+4+b)     (R=mt*32, C=nt*8)
"""
from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import T, arith, range_constexpr, vector
from flydsl.expr.buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from flydsl.expr.typing import Stream

fp8_e8m0 = getattr(torch, "float8_e8m0fnu", torch.uint8)


def make_fused_swizzle_jit(*, num_valid_max, n_blocks, block_num=304, warp_num_per_block=4):
    """Returns (launch_jit, m_tiles, n_tiles).  n_blocks = scale blocks/row (= model_dim/32)."""
    assert n_blocks % 8 == 0, f"n_blocks({n_blocks}) must be a multiple of 8"
    n_tiles = n_blocks // 8
    n_i32 = n_blocks // 4
    NVMAX = int(num_valid_max)
    m_tiles = (NVMAX + 31) // 32
    total_tiles = m_tiles * n_tiles

    @flyc.kernel(known_block_size=[warp_num_per_block * 64, 1, 1])
    def swizzle(addr_in: fx.Int64, addr_nv: fx.Int64, addr_out: fx.Int64):
        # Each lane owns one row-within-half `a` of one tile; loops b=0..3 (constexpr) to emit the
        # tile's 4 output u32 for that row.  Loads the row's 8 scale blocks as one vec2 i32 (vs 16
        # uncoalesced byte loads), and the byte shift b*8 is compile-time -> no runtime shift.
        # 64 lanes => 4 tiles/warp (sub = lane>>4).
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        gwarp = bid * warp_num_per_block + warp
        gwarp_num = block_num * warp_num_per_block
        nv = buffer_load(create_buffer_resource_from_addr(addr_nv), 0, vec_width=1, dtype=T.i32())
        r_in = create_buffer_resource_from_addr(addr_in)
        r_out = create_buffer_resource_from_addr(addr_out)
        a = lane & 15            # row-within-half 0..15
        sub = lane >> 4          # which of the 4 tiles in this warp 0..3
        c8 = arith.constant(8, type=T.i32())
        c16 = arith.constant(16, type=T.i32())
        c24 = arith.constant(24, type=T.i32())
        m8 = arith.constant(0xFF, type=T.i32())
        nvmax_m1 = arith.constant(NVMAX - 1)
        c_nvmax = arith.constant(NVMAX)
        tot_m1 = arith.constant(total_tiles - 1)
        c_tot = arith.constant(total_tiles)
        m_valid = (nv + 31) >> 5                 # ceil(nv/32)
        n_groups = (m_valid * n_tiles + 3) >> 2  # ceil(valid_tiles/4)
        for g in range(gwarp, n_groups, gwarp_num):
            t = (g << 2) + sub
            t = arith.select(arith.cmpi(arith.CmpIPredicate.ult, t, c_tot), t, tot_m1)
            mt = t // n_tiles
            nt = t - mt * n_tiles
            c_i32 = nt * 2                        # (nt*8)//4
            row_lo = mt * 32 + a
            row_hi = row_lo + 16
            row_lo = arith.select(arith.cmpi(arith.CmpIPredicate.ult, row_lo, c_nvmax), row_lo, nvmax_m1)
            row_hi = arith.select(arith.cmpi(arith.CmpIPredicate.ult, row_hi, c_nvmax), row_hi, nvmax_m1)
            vlo = buffer_load(r_in, row_lo * n_i32 + c_i32, vec_width=2, dtype=T.i32())  # cols C..C+7
            vhi = buffer_load(r_in, row_hi * n_i32 + c_i32, vec_width=2, dtype=T.i32())
            i_lo0 = vector.extract(vlo, static_position=[0], dynamic_position=[])  # row_lo cols C..C+3
            i_lo1 = vector.extract(vlo, static_position=[1], dynamic_position=[])  # row_lo cols C+4..C+7
            i_hi0 = vector.extract(vhi, static_position=[0], dynamic_position=[])  # row_hi cols C..C+3
            i_hi1 = vector.extract(vhi, static_position=[1], dynamic_position=[])  # row_hi cols C+4..C+7
            out_tile_base = mt * (n_tiles * 64) + nt * 64 + a
            for b in range_constexpr(4):
                sh = arith.constant(b * 8, type=T.i32())
                byte0 = arith.andi(arith.shrui(i_lo0, sh), m8)   # (row_lo, C+b)
                byte1 = arith.andi(arith.shrui(i_hi0, sh), m8)   # (row_hi, C+b)
                byte2 = arith.andi(arith.shrui(i_lo1, sh), m8)   # (row_lo, C+4+b)
                byte3 = arith.andi(arith.shrui(i_hi1, sh), m8)   # (row_hi, C+4+b)
                packed = arith.ori(arith.ori(byte0, arith.shli(byte1, c8)),
                                   arith.ori(arith.shli(byte2, c16), arith.shli(byte3, c24)))
                buffer_store(packed, r_out, out_tile_base + b * 16)

    @flyc.jit
    def launch(addr_in: fx.Int64, addr_nv: fx.Int64, addr_out: fx.Int64, stream: Stream = Stream(None)):
        swizzle(addr_in, addr_nv, addr_out).launch(
            grid=(block_num, 1, 1), block=(warp_num_per_block * 64, 1, 1), stream=stream)

    return launch, m_tiles, n_tiles


class FusedScaleSwizzle:
    """Stateful drop-in for the Triton ``fused_mxfp4_swizzle`` (fused/expert-major, identity ids).

    Holds the compiled FlyDSL kernel + the swizzled-scale output buffer.  Call with the raw
    row-major e8m0 ``scale_em`` and the ``num_valid`` tensor; returns the swizzled scale (uint8 view)
    the fused group-GEMM consumes (same byte layout as the Triton kernel)."""

    def __init__(self, *, num_valid_max, n_blocks, device, block_num=304, warp_num_per_block=4):
        self.n_blocks = int(n_blocks)
        self._launch, self.m_tiles, self.n_tiles = make_fused_swizzle_jit(
            num_valid_max=num_valid_max, n_blocks=n_blocks,
            block_num=block_num, warp_num_per_block=warp_num_per_block)
        self.out = torch.zeros((self.m_tiles, self.n_tiles, 4, 16), dtype=torch.int32, device=device)
        self._compiled = None

    def __call__(self, scale_em, num_valid, stream=None):
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())
        ip = scale_em.data_ptr(); nvp = num_valid.data_ptr(); op = self.out.data_ptr()
        if self._compiled is None:
            self._compiled = flyc.compile(self._launch, fx.Int64(ip), fx.Int64(nvp), fx.Int64(op), stream)
        else:
            self._compiled(ip, nvp, op, stream)
        return self.out.view(fp8_e8m0).view(-1, self.n_blocks).view(torch.uint8)


def make_fused_swizzle_sparse_jit(*, num_valid_max, n_blocks, rows_per_tile,
                                  block_num=304, warp_num_per_block=4):
    """SPARSE-aware variant of make_fused_swizzle_jit (unified fixed-slot stage-1).

    The unified dispatch leaves the raw e8m0 scale at SPARSE expert slots (row = le*cap + idx) and
    emits, per OCCUPIED compact tile bx, its sparse row base ``tile_row_base[bx]`` (= le*cap +
    t*tile_m).  This kernel, per compact 32-row scale m-tile ``mt``, reads ``scale_em`` at the
    SPARSE rows and writes the MFMA-swizzled scale to the COMPACT output tile ``mt`` (same byte
    layout as the contiguous swizzle, so the fused group-GEMM consumes it unchanged when it indexes
    the A-scale by the COMPACT tile counter bx).

    A GEMM tile spans ``ratio = rows_per_tile/32`` consecutive 32-row scale m-tiles, so scale m-tile
    ``mt`` belongs to GEMM tile ``bx = mt//ratio`` at sub-block ``sub = mt%ratio``; its sparse rows
    are ``[tile_row_base[bx] + sub*32 .. +32)``.  With identity tile_row_base (bx -> bx*rows_per_tile)
    this reduces EXACTLY to the contiguous swizzle (mt -> rows [mt*32 .. +32)).
    """
    assert n_blocks % 8 == 0, f"n_blocks({n_blocks}) must be a multiple of 8"
    n_tiles = n_blocks // 8
    n_i32 = n_blocks // 4
    NVMAX = int(num_valid_max)
    m_tiles = (NVMAX + 31) // 32
    total_tiles = m_tiles * n_tiles
    ratio = max(1, int(rows_per_tile) // 32)   # 32-row scale m-tiles per GEMM (tile_row_base) tile

    @flyc.kernel(known_block_size=[warp_num_per_block * 64, 1, 1])
    def swizzle(addr_in: fx.Int64, addr_nv: fx.Int64, addr_trb: fx.Int64, addr_out: fx.Int64):
        # Identical layout/repack to make_fused_swizzle_jit; ONLY the input row base differs: it is
        # gathered from tile_row_base (sparse) instead of the compact mt*32.  Output stays compact.
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        gwarp = bid * warp_num_per_block + warp
        gwarp_num = block_num * warp_num_per_block
        nv = buffer_load(create_buffer_resource_from_addr(addr_nv), 0, vec_width=1, dtype=T.i32())
        r_in = create_buffer_resource_from_addr(addr_in)
        r_trb = create_buffer_resource_from_addr(addr_trb)
        r_out = create_buffer_resource_from_addr(addr_out)
        a = lane & 15            # row-within-half 0..15
        sub = lane >> 4          # which of the 4 tiles in this warp 0..3
        c8 = arith.constant(8, type=T.i32())
        c16 = arith.constant(16, type=T.i32())
        c24 = arith.constant(24, type=T.i32())
        c32 = arith.constant(32, type=T.i32())
        m8 = arith.constant(0xFF, type=T.i32())
        tot_m1 = arith.constant(total_tiles - 1)
        c_tot = arith.constant(total_tiles)
        m_valid = (nv + 31) >> 5                 # ceil(nv/32) valid 32-row scale m-tiles
        m_valid_m1 = m_valid - arith.constant(1)
        n_groups = (m_valid * n_tiles + 3) >> 2  # ceil(valid_tiles/4)
        for g in range(gwarp, n_groups, gwarp_num):
            t = (g << 2) + sub
            t = arith.select(arith.cmpi(arith.CmpIPredicate.ult, t, c_tot), t, tot_m1)
            mt = t // n_tiles
            nt = t - mt * n_tiles
            c_i32 = nt * 2                        # (nt*8)//4
            # compact tile mt -> sparse row base via tile_row_base (clamp padding mt to a valid tile
            # so the gathered base + read stay in-bounds; padded output tiles are never GEMM-read).
            mt_in = arith.select(arith.cmpi(arith.CmpIPredicate.ult, mt, m_valid), mt, m_valid_m1)
            bx = mt_in // ratio
            sub_m = mt_in - bx * ratio
            base = buffer_load(r_trb, bx, vec_width=1, dtype=T.i32()) + sub_m * c32
            row_lo = base + a
            row_hi = base + a + c16
            vlo = buffer_load(r_in, row_lo * n_i32 + c_i32, vec_width=2, dtype=T.i32())  # cols C..C+7
            vhi = buffer_load(r_in, row_hi * n_i32 + c_i32, vec_width=2, dtype=T.i32())
            i_lo0 = vector.extract(vlo, static_position=[0], dynamic_position=[])  # row_lo cols C..C+3
            i_lo1 = vector.extract(vlo, static_position=[1], dynamic_position=[])  # row_lo cols C+4..C+7
            i_hi0 = vector.extract(vhi, static_position=[0], dynamic_position=[])  # row_hi cols C..C+3
            i_hi1 = vector.extract(vhi, static_position=[1], dynamic_position=[])  # row_hi cols C+4..C+7
            out_tile_base = mt * (n_tiles * 64) + nt * 64 + a
            for b in range_constexpr(4):
                sh = arith.constant(b * 8, type=T.i32())
                byte0 = arith.andi(arith.shrui(i_lo0, sh), m8)   # (row_lo, C+b)
                byte1 = arith.andi(arith.shrui(i_hi0, sh), m8)   # (row_hi, C+b)
                byte2 = arith.andi(arith.shrui(i_lo1, sh), m8)   # (row_lo, C+4+b)
                byte3 = arith.andi(arith.shrui(i_hi1, sh), m8)   # (row_hi, C+4+b)
                packed = arith.ori(arith.ori(byte0, arith.shli(byte1, c8)),
                                   arith.ori(arith.shli(byte2, c16), arith.shli(byte3, c24)))
                buffer_store(packed, r_out, out_tile_base + b * 16)

    @flyc.jit
    def launch(addr_in: fx.Int64, addr_nv: fx.Int64, addr_trb: fx.Int64, addr_out: fx.Int64,
               stream: Stream = Stream(None)):
        swizzle(addr_in, addr_nv, addr_trb, addr_out).launch(
            grid=(block_num, 1, 1), block=(warp_num_per_block * 64, 1, 1), stream=stream)

    return launch, m_tiles, n_tiles


class FusedSparseScaleSwizzle:
    """Sparse-aware drop-in for FusedScaleSwizzle (unified fixed-slot stage-1).

    Reads the raw row-major e8m0 ``scale_em`` at the SPARSE expert slots indicated by
    ``tile_row_base`` and writes the MFMA-swizzled scale to a COMPACT output indexed by the compact
    occupied-tile counter (the index the fused group-GEMM uses for the pre-swizzled A-scale)."""

    def __init__(self, *, num_valid_max, n_blocks, rows_per_tile, device,
                 block_num=304, warp_num_per_block=4):
        self.n_blocks = int(n_blocks)
        self._launch, self.m_tiles, self.n_tiles = make_fused_swizzle_sparse_jit(
            num_valid_max=num_valid_max, n_blocks=n_blocks, rows_per_tile=rows_per_tile,
            block_num=block_num, warp_num_per_block=warp_num_per_block)
        self.out = torch.zeros((self.m_tiles, self.n_tiles, 4, 16), dtype=torch.int32, device=device)
        self._compiled = None

    def __call__(self, scale_em, num_valid, tile_row_base, stream=None):
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())
        ip = scale_em.data_ptr(); nvp = num_valid.data_ptr()
        tbp = tile_row_base.data_ptr(); op = self.out.data_ptr()
        if self._compiled is None:
            self._compiled = flyc.compile(self._launch, fx.Int64(ip), fx.Int64(nvp),
                                          fx.Int64(tbp), fx.Int64(op), stream)
        else:
            self._compiled(ip, nvp, tbp, op, stream)
        return self.out.view(fp8_e8m0).view(-1, self.n_blocks).view(torch.uint8)
