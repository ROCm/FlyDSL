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
from pyflir.dialects.ext import arith, vector


@dataclass(frozen=True)
class PreshuffleBLayout:
    """Container returned by `make_preshuffle_b_layout`."""
    layout_b: object
    c16: ir.Value
    c64: ir.Value
    c1024: ir.Value


def swizzle_xor_16b(flir, _arith_mlir, row_idx: ir.Value, col_idx: ir.Value, *, tile_k: int) -> ir.Value:
    """CK-style XOR16 swizzle on K at 16B granularity (index-typed).

    This now routes through the dedicated `flir.swizzle_xor16` op so lowering can
    optimize to bitwise ops when `kBlocks16` is a const power-of-two.
    """
    # Accept ArithValue / Value / int and rely on ext unwrapping.
    k_blocks16 = arith.constant(tile_k // 16, index=True)
    return flir.swizzle_xor16(row_idx, col_idx, k_blocks16)


def make_preshuffle_b_layout(flir, _arith_mlir, *, c_n: ir.Value, c_k: ir.Value) -> PreshuffleBLayout:
    """Build B layout matching aiter/CK preshuffle for fp8 mfma kernels.

    Shape: (N0, K0, KLane, NLane, KPack) = (N/16, K/64, 4, 16, 16)
    """
    c16 = arith.constant(16, index=True)
    c64 = arith.constant(64, index=True)
    c1024 = arith.constant(1024, index=True)

    # Index math: use operator overloading to avoid low-level op result access.
    c_k0 = arith.ArithValue(c_k) / c64
    stride_n0 = c_k0 * c1024
    stride_b = (
        stride_n0,  # n0
        c1024,      # k0
        arith.constant(256, index=True),  # k1 (KLane)
        c16,        # n1
        arith.constant(1, index=True),    # k2
    )
    n0 = arith.ArithValue(c_n) / c16
    layout_b = flir.make_layout(
        (n0, c_k0, arith.constant(4, index=True), c16, c16),
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
    """Load FP8 tile fragment via global dwordx4 loads; returns list of vector<4xi32>.

    This is written in the same style as the preshuffle GEMM kernel:
    - Use `flir.copy(load-only)` with `src_buffer_resource` so lowering selects
      `buffer_load_dwordx4`.
    - Allow scalar broadcast predication (`mask`) for padded/sentinel tokens.

    NOTE: We intentionally load 16 fp8 bytes per split (dwordx4). Callers pad
    the underlying storage so this is safe even when the last split is < 16B.
    """
    # Callers may pass ArithValue/int; ext wrappers handle unwrapping.

    # 16 fp8 bytes -> vector<16xf8> -> bitcast to vector<4xi32>
    f8 = ir.Float8E4M3FNType.get()
    atom = flir.make_copy_atom(f8, vector_size=16)
    vec4_i32 = ir.VectorType.get([4], i32_type)

    parts = []
    off_i32 = 0
    for curr_bytes in lens:
        curr_idx = idx_div4 if off_i32 == 0 else (arith.ArithValue(idx_div4) + arith.constant(off_i32, index=True))
        src_view = flir.TensorView(
            None,
            (16,),
            strides=(1,),
            base_indices=(curr_idx,),
            element_type=f8,
        )
        v16 = flir.copy(
            atom,
            src_view,
            None,
            pred=mask,
            return_vector=True,
            src_buffer_resource=rsrc,
            src_buffer_offset_in_bytes=False,  # idx_div4 is in dword units
            alignment=16,
        )
        parts.append(vector.bitcast(vec4_i32, v16))
        off_i32 += (curr_bytes // 4)
    return parts


def load_b_pack_k32(buffer_ops, flir, _arith_mlir, *, b_rsrc, layout_b, base_k: ir.Value, ki_step: int, n_blk: ir.Value, n_intra: ir.Value, lane_div_16: ir.Value, i32_type) -> ir.Value:
    """Load one 8B (i64) B pack for one MFMA(x32) step.

    Uses `flir.copy(load-only)` + `src_buffer_resource` to generate buffer loads,
    matching the preshuffle GEMM path.
    """
    # Accept ArithValue/Value/int and rely on ext wrappers to unwrap where needed.
    c64 = arith.constant(64, index=True)
    k0_base = arith.ArithValue(base_k) / c64
    k0 = k0_base + arith.constant(ki_step // 2, index=True)
    k1 = lane_div_16
    half = ki_step % 2
    k2_base = arith.constant(half * 8, index=True)

    coord_b = flir.make_coord(n_blk, k0, k1, n_intra, k2_base)
    idx_bytes = flir.crd2idx(coord_b, layout_b)
    f8 = ir.Float8E4M3FNType.get()
    atom = flir.make_copy_atom(f8, vector_size=8)
    b_view = flir.TensorView(
        None,
        (8,),
        strides=(1,),
        base_indices=(idx_bytes,),
        element_type=f8,
    )
    b8_f8 = flir.copy(
        atom,
        b_view,
        None,
        alignment=8,
        return_vector=True,
        src_buffer_resource=b_rsrc,
        src_buffer_offset_in_bytes=True,
    )
    vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
    b_vec64 = vector.bitcast(vec1_i64, b8_f8)
    return vector.extract(b_vec64, static_position=[0], dynamic_position=[])


def load_b_packs_k64(buffer_ops, flir, _arith_mlir, *, b_rsrc, layout_b, base_k: ir.Value, ki_step: int, n_blk: ir.Value, n_intra: ir.Value, lane_div_16: ir.Value, i32_type) -> Tuple[ir.Value, ir.Value]:
    """Load 16B (two i64 packs) for one K64 micro-step."""
    # Accept ArithValue/Value/int and rely on ext wrappers to unwrap where needed.
    c4 = arith.constant(4, index=True)
    c64 = arith.constant(64, index=True)
    k0_base = arith.ArithValue(base_k) / c64
    k0 = k0_base + arith.constant(ki_step, index=True)
    k1 = lane_div_16
    k2_base = arith.constant(0, index=True)
    coord_b = flir.make_coord(n_blk, k0, k1, n_intra, k2_base)
    idx_bytes = flir.crd2idx(coord_b, layout_b)
    idx_i32 = arith.ArithValue(idx_bytes) / c4
    b16 = buffer_ops.buffer_load(b_rsrc, idx_i32, vec_width=4, dtype=i32_type)
    vec2_i64 = ir.VectorType.get([2], ir.IntegerType.get_signless(64))
    b_vec128 = vector.bitcast(vec2_i64, b16)
    b0 = vector.extract(b_vec128, static_position=[0], dynamic_position=[])
    b1 = vector.extract(b_vec128, static_position=[1], dynamic_position=[])
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


