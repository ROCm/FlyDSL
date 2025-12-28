"""
Shared MFMA FP8 preshuffle "pipeline" helpers.

Goal: de-duplicate the common building blocks used by:
- `test_mfma_gemm_fp8_rocir_preshuffle.py`
- `test_moe_gemm_stage1_fp8_rocir_preshuffle.py`

This is not a full generic GEMM generator; it provides reusable primitives:
- CK-style xor swizzle on K at 16B granularity
- Preshuffle B layout builder (N0,K0,KLane,NLane,KPack)
- Split global dwordx4 loads for FP8 tiles
- Load B packs (K32 or K64 micro-step) from preshuffled layout
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from _mlir import ir
from _mlir.dialects import vector


@dataclass(frozen=True)
class PreshuffleBLayout:
    """Container returned by `make_preshuffle_b_layout`."""
    layout_b: object
    c16: ir.Value
    c64: ir.Value
    c1024: ir.Value


def _unwrap(v):
    """Best-effort unwrap for ArithValue-like wrappers used in tests."""
    while hasattr(v, "_value") or hasattr(v, "value"):
        v = getattr(v, "_value", getattr(v, "value", v))
    return v


def swizzle_xor_16b(flir, _arith_mlir, row_idx: ir.Value, col_idx: ir.Value, *, tile_k: int) -> ir.Value:
    """CK-style xor-with-modulo swizzle on K at 16B granularity (index-typed)."""
    row_idx = _unwrap(row_idx)
    col_idx = _unwrap(col_idx)
    c16 = flir.const_index(16)
    k_blocks16 = flir.const_index(tile_k // 16)
    row_mod = _unwrap(_arith_mlir.RemUIOp(row_idx, k_blocks16).result)
    xor_mask = _unwrap(_arith_mlir.MulIOp(row_mod, c16).result)
    return _unwrap(_arith_mlir.XOrIOp(col_idx, xor_mask).result)


def make_preshuffle_b_layout(flir, _arith_mlir, *, c_n: ir.Value, c_k: ir.Value) -> PreshuffleBLayout:
    """Build B layout matching aiter/CK preshuffle for fp8 mfma kernels.

    Shape: (N0, K0, KLane, NLane, KPack) = (N/16, K/64, 4, 16, 16)
    """
    c_n = _unwrap(c_n)
    c_k = _unwrap(c_k)

    c16 = flir.const_index(16)
    c64 = flir.const_index(64)
    c1024 = flir.const_index(1024)

    c_k0 = _unwrap(_arith_mlir.DivUIOp(c_k, c64).result)
    stride_n0 = _unwrap(_arith_mlir.MulIOp(c_k0, c1024).result)
    stride_b = (
        stride_n0,  # n0
        c1024,      # k0
        flir.const_index(256),  # k1 (KLane)
        c16,        # n1
        flir.const_index(1),    # k2
    )
    n0 = _unwrap(_arith_mlir.DivUIOp(c_n, c16).result)
    layout_b = flir.make_layout(
        (n0, c_k0, flir.const_index(4), c16, c16),
        stride=stride_b,
    )
    return PreshuffleBLayout(layout_b=layout_b, c16=c16, c64=c64, c1024=c1024)


def compute_split_load_lens(bytes_per_thread: int, *, max_bytes_per_load: int = 16) -> List[int]:
    """Return per-part byte lengths for split dwordx4 loads (python ints)."""
    lens: List[int] = []
    remaining = int(bytes_per_thread)
    while remaining > 0:
        curr = min(remaining, max_bytes_per_load)
        lens.append(curr)
        remaining -= curr
    return lens


def load_fp8_tile_split_dwordx4(buffer_ops, flir, _arith_mlir, *, rsrc, idx_div4: ir.Value, lens: Sequence[int], i32_type, mask=None):
    """Load FP8 tile fragment via global dwordx4 loads; returns list of vector<4xi32>."""
    rsrc = _unwrap(rsrc)
    idx_div4 = _unwrap(idx_div4)
    mask = _unwrap(mask) if mask is not None else None
    parts = []
    off_i32 = 0
    for curr_bytes in lens:
        curr_idx = idx_div4 if off_i32 == 0 else _unwrap(_arith_mlir.AddIOp(idx_div4, flir.const_index(off_i32)).result)
        parts.append(buffer_ops.buffer_load(rsrc, curr_idx, vec_width=4, dtype=i32_type, mask=mask))
        off_i32 += (curr_bytes // 4)
    return parts


def load_b_pack_k32(buffer_ops, flir, _arith_mlir, *, b_rsrc, layout_b, base_k: ir.Value, ki_step: int, n_blk: ir.Value, n_intra: ir.Value, lane_div_16: ir.Value, i32_type) -> ir.Value:
    """Load one 8B (i64) B pack for one MFMA(x32) step."""
    b_rsrc = _unwrap(b_rsrc)
    layout_b = _unwrap(layout_b)
    base_k = _unwrap(base_k)
    n_blk = _unwrap(n_blk)
    n_intra = _unwrap(n_intra)
    lane_div_16 = _unwrap(lane_div_16)
    c4 = flir.const_index(4)
    c64 = flir.const_index(64)
    k0_base = _unwrap(_arith_mlir.DivUIOp(base_k, c64).result)
    k0 = _unwrap(_arith_mlir.AddIOp(k0_base, flir.const_index(ki_step // 2)).result)
    k1 = lane_div_16
    half = ki_step % 2
    k2_base = flir.const_index(half * 8)

    coord_b = flir.make_coord(n_blk, k0, k1, n_intra, k2_base)
    idx_bytes = flir.crd2idx(coord_b, layout_b)
    idx_i32 = _unwrap(_arith_mlir.DivUIOp(_unwrap(idx_bytes), c4).result)
    b8 = buffer_ops.buffer_load(b_rsrc, idx_i32, vec_width=2, dtype=i32_type)
    vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
    b_vec64 = _unwrap(vector.BitCastOp(vec1_i64, _unwrap(b8)).result)
    return _unwrap(vector.ExtractOp(b_vec64, static_position=[0], dynamic_position=[]).result)


def load_b_packs_k64(buffer_ops, flir, _arith_mlir, *, b_rsrc, layout_b, base_k: ir.Value, ki_step: int, n_blk: ir.Value, n_intra: ir.Value, lane_div_16: ir.Value, i32_type) -> Tuple[ir.Value, ir.Value]:
    """Load 16B (two i64 packs) for one K64 micro-step."""
    b_rsrc = _unwrap(b_rsrc)
    layout_b = _unwrap(layout_b)
    base_k = _unwrap(base_k)
    n_blk = _unwrap(n_blk)
    n_intra = _unwrap(n_intra)
    lane_div_16 = _unwrap(lane_div_16)
    c4 = flir.const_index(4)
    c64 = flir.const_index(64)
    k0_base = _unwrap(_arith_mlir.DivUIOp(base_k, c64).result)
    k0 = _unwrap(_arith_mlir.AddIOp(k0_base, flir.const_index(ki_step)).result)
    k1 = lane_div_16
    k2_base = flir.const_index(0)
    coord_b = flir.make_coord(n_blk, k0, k1, n_intra, k2_base)
    idx_bytes = flir.crd2idx(coord_b, layout_b)
    idx_i32 = _unwrap(_arith_mlir.DivUIOp(_unwrap(idx_bytes), c4).result)
    b16 = buffer_ops.buffer_load(b_rsrc, idx_i32, vec_width=4, dtype=i32_type)
    vec2_i64 = ir.VectorType.get([2], ir.IntegerType.get_signless(64))
    b_vec128 = _unwrap(vector.BitCastOp(vec2_i64, _unwrap(b16)).result)
    b0 = _unwrap(vector.ExtractOp(b_vec128, static_position=[0], dynamic_position=[]).result)
    b1 = _unwrap(vector.ExtractOp(b_vec128, static_position=[1], dynamic_position=[]).result)
    return b0, b1



__all__ = [
    "PreshuffleBLayout",
    "compute_split_load_lens",
    "load_fp8_tile_split_dwordx4",
    "load_b_pack_k32",
    "load_b_packs_k64",
    "make_preshuffle_b_layout",
    "swizzle_xor_16b",
]


