"""Shared MFMA preshuffle helpers (used by preshuffle GEMM + MoE kernels).

This module consolidates the common building blocks that were previously duplicated
across:
- `kernels/preshuffle_gemm.py`
- `kernels/moe_gemm_2stage.py`

Key primitives:
- B preshuffle layout builder (supports byte-packed element types, incl. packed int4)
- B pack load for MFMA K32 micro-steps (8B output pack; optional int4->int8 unpack)
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from _mlir import ir

# Perf experiment flags (results will be numerically wrong):
# SKIP_DEQUANT=1  — skip sitofp+truncf+scale, cheap bitcast only.
# SKIP_LOAD_B=1   — skip VMEM weight load entirely, return constant i32.
# SKIP_SCALE=1    — skip scale VMEM load, use constant scale (isolate weight load perf).
_SKIP_DEQUANT = os.environ.get("SKIP_DEQUANT", "0") == "1"
_SKIP_LOAD_B = os.environ.get("SKIP_LOAD_B", "0") == "1"
_SKIP_SCALE = os.environ.get("SKIP_SCALE", "0") == "1"

@dataclass(frozen=True)
class PreshuffleBLayout:
    """Container returned by `make_preshuffle_b_layout`."""

    layout_b: object
    kpack_bytes: int


def make_preshuffle_b_layout(
    flir,
    arith,
    *,
    c_n: ir.Value,
    c_k: ir.Value,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
) -> PreshuffleBLayout:
    """Build B layout matching aiter/CK preshuffle for A8 MFMA kernels.

    Shape: (N0, K0, KLane, NLane, KPackBytes) = (N/16, K/64, 4, 16, kpack_bytes)

    Notes:
    - For FP8/INT8: kpack_bytes=16 (one byte per element).
    - For packed INT4 (W4A8): kpack_bytes=8 (two 4-bit values per byte).
    """
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")

    c16 = arith.constant(16, index=True)
    c64 = arith.constant(64, index=True)
    c4 = arith.constant(4, index=True)
    c_kpack = arith.constant(kpack_bytes, index=True)

    # This layout is fundamentally byte-addressed along K:
    # - For 1B types (fp8/i8): KBytes == K
    # - For 2B types (fp16/bf16): KBytes == 2*K
    #
    # We keep the same 64B K0 "macro-step" used by CK/aiter preshuffle.
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    c_k_bytes = c_k * arith.constant(int(elem_bytes), index=True)
    c_k0 = c_k_bytes / c64
    n0 = c_n / c16

    # Layout is expressed in ELEMENT units (not bytes). Convert KPackBytes -> KPackElems.
    c_kpack_elems = c_kpack if elem_bytes == 1 else (c_kpack / arith.constant(int(elem_bytes), index=True))

    # Strides derived from the layout shape:
    # - KPack stride = 1
    # - NLane stride = KPackElems
    # - KLane stride = NLane * KPackElems = 16 * KPackElems
    # - K0   stride = KLane * NLane * KPackElems = 4 * 16 * KPackElems
    stride_nlane = c_kpack_elems
    stride_klane = c16 * stride_nlane
    stride_k0 = c4 * stride_klane
    stride_n0 = c_k0 * stride_k0

    stride_b = (
        stride_n0,      # n0
        stride_k0,      # k0
        stride_klane,   # k1 (KLane)
        stride_nlane,   # n1
        arith.constant(1, index=True),  # k2
    )
    layout_b = flir.make_layout((n0, c_k0, c4, c16, c_kpack_elems), stride=stride_b)
    return PreshuffleBLayout(layout_b=layout_b, kpack_bytes=kpack_bytes)


def make_preshuffle_scale_layout(
    flir,
    arith,
    *,
    c_mn: ir.Value,
    c_k: ir.Value,
    mn_pack: int = 2,
    k_pack: int = 2,
    elem_bytes: int = 4,
    scale_block_size: int = 32,
) -> object:
    """Build scale layout matching aiter/CK preshuffle for MXFP4 MFMA kernels.
    scale dtype is e8m0
    the scale shuffle to [K_Pack, N_Pack], pack to int32

    Shape: (N1, K1, KLane, NLane, [K_Pack, N_Pack]) = (N/32, K/8, 4, 16, [2, 2])
    """
    c16 = arith.constant(16, index=True)
    c32 = arith.constant(32, index=True)
    c4 = arith.constant(4, index=True)

    c_mn_pack = arith.constant(mn_pack, index=True)
    c_k_pack = arith.constant(k_pack, index=True)
    c_k_scale = c_k / scale_block_size

    c_mn1 = c_mn / c16 / c_mn_pack
    c_k1 = c_k_scale / c4 / c_k_pack

    # We keep the same 64B K0 "macro-step" used by CK/aiter preshuffle.
    if elem_bytes != mn_pack * k_pack:
        raise ValueError(f"elem_bytes of scale must be {mn_pack} * {k_pack}, got {elem_bytes!r}")

    # Strides derived from the layout shape:
    # - KPack stride = 1
    # - NLane stride = KPackElems
    # - KLane stride = NLane * KPackElems = 16 * KPackElems
    # - K0   stride = KLane * NLane * KPackElems = 4 * 16 * KPackElems
    stride_nlane = arith.constant(1, index=True)
    stride_klane = c16
    stride_k0 = c4 * stride_klane
    stride_n0 = c_k1 * stride_k0

    stride_b_scale = (
        stride_n0,      # n0
        stride_k0,      # k0
        stride_klane,   # KLane
        stride_nlane,   # NLane
    )
    layout_b = flir.make_layout((c_mn1, c_k1, c4, c16), stride=stride_b_scale)
    return layout_b


def load_b_pack_k32(
    buffer_ops,
    flir,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: ir.Value,
    ki_step: int,
    n_blk: ir.Value,
    n_intra: ir.Value,
    lane_div_16: ir.Value,
    elem_type: ir.Type,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
    unpack_int4: bool = False,
) -> ir.Value:
    """Load one B pack for one MFMA(x32) micro-step.

    Returns an i64 Value containing 8 bytes consumed by MFMA.

    - For FP8/INT8: loads 16 bytes (one full KPack) and extracts the 8 bytes used by
      this micro-step.
    - For packed INT4 (W4A8): loads 4 bytes (8 int4 values) and unpacks to 8 int8 bytes
      using the 7-op sequence (no v_perm).
    """
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")
    if unpack_int4 and kpack_bytes != 8:
        raise ValueError("unpack_int4 requires kpack_bytes=8 (packed int4 layout)")

    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")

    c64 = arith.constant(64, index=True)
    base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
    k0_base = base_k_bytes / c64
    k0 = k0_base + arith.constant(ki_step // 2, index=True)
    k1 = lane_div_16
    half_bytes = kpack_bytes // 2
    k2_base = arith.constant((ki_step % 2) * half_bytes, index=True)

    # Always compute the *pack base* index (k2=0). Layout is in ELEMENT units.
    # add/sub on the address path and keeps the load address stable across the
    # two half-steps.
    coord_pack = flir.make_coord(n_blk, k0, k1, n_intra, arith.constant(0, index=True))
    idx_pack = flir.crd2idx(coord_pack, layout_b)

    if unpack_int4:
        # Load 4 bytes -> i32 -> unpack to i64 (8 i8 bytes).
        atom = flir.make_copy_atom(elem_type, vector_size=4)
        # packed int4 is byte-addressed (elem_bytes==1)
        idx_bytes = idx_pack + k2_base
        b_view = flir.TensorView(
            arg_b,
            (4,),
            strides=(1,),
            base_indices=(idx_bytes,),
            element_type=elem_type,
        )
        b4 = flir.copy(
            atom,
            b_view,
            None,
            alignment=4,
            return_vector=True,
            src_buffer_resource=b_rsrc,
            src_buffer_offset_in_bytes=True,
        )
        vec1_i32 = ir.VectorType.get([1], ir.IntegerType.get_signless(32))
        packed32 = vector.extract(
            vector.bitcast(vec1_i32, b4),
            static_position=[0],
            dynamic_position=[],
        )

        # 7-op unpack (and + mul + and_or + shifts). Requires prepacked nibble layout:
        # bytes: [ (v4<<4)|v0, (v5<<4)|v1, (v6<<4)|v2, (v7<<4)|v3 ]
        c_08080808 = arith.constant(0x08080808, type=ir.IntegerType.get_signless(32))
        c_0f0f0f0f = arith.constant(0x0F0F0F0F, type=ir.IntegerType.get_signless(32))
        c_1e = arith.constant(0x1E, type=ir.IntegerType.get_signless(32))
        c_4_i32 = arith.constant(4, type=ir.IntegerType.get_signless(32))

        s0 = (packed32 & c_08080808) * c_1e
        even = (packed32 & c_0f0f0f0f) | s0

        t = packed32 >> c_4_i32
        s1 = (t & c_08080808) * c_1e
        odd = (t & c_0f0f0f0f) | s1

        vec2_i32 = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
        v2 = vector.from_elements(vec2_i32, [even, odd])
        vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
        v64 = vector.bitcast(vec1_i64, v2)
        return vector.extract(v64, static_position=[0], dynamic_position=[])

    # FP8/INT8: load 16 bytes (one full KPack) and extract half (8B) as i64.
    #
    # This keeps the original semantics (return the same 8B i64 used by MFMA for
    # this `ki_step`), but makes the intended 16B buffer-load (dwordx4) explicit
    # in the IR instead of relying on backend vectorization.
    vec_elems = kpack_bytes // int(elem_bytes)
    atom = flir.make_copy_atom(elem_type, vector_size=vec_elems)
    b_view = flir.TensorView(
        arg_b,
        (vec_elems,),
        strides=(1,),
        base_indices=(idx_pack,),
        element_type=elem_type,
    )
    b16 = flir.copy(
        atom,
        b_view,
        None,
        # Keep conservative alignment here: some layouts/launchers may only guarantee 8B.
        # This is still compatible with 16B buffer loads; it just avoids overstating
        # alignment to the compiler.
        alignment=8,
        return_vector=True,
        src_buffer_resource=b_rsrc,
        # Only 1B element types can safely treat the base index as bytes.
        src_buffer_offset_in_bytes=(elem_bytes == 1),
    )
    # Extract the needed 8B half as an i64 while keeping the other half dead.
    #
    # NOTE: We intentionally build the i64 from the selected 2 dwords, instead of
    # `bitcast -> i64x2 -> extract`, to help the backend shorten live ranges and
    # avoid unnecessary VGPR pressure on some schedules.
    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)
    vec4_i32 = ir.VectorType.get([4], i32)
    b_i32x4 = vector.bitcast(vec4_i32, b16)

    half = ki_step % 2
    if half == 0:
        d0 = vector.extract(b_i32x4, static_position=[0], dynamic_position=[])
        d1 = vector.extract(b_i32x4, static_position=[1], dynamic_position=[])
    else:
        d0 = vector.extract(b_i32x4, static_position=[2], dynamic_position=[])
        d1 = vector.extract(b_i32x4, static_position=[3], dynamic_position=[])

    vec2_i32 = ir.VectorType.get([2], i32)
    v2 = vector.from_elements(vec2_i32, [d0, d1])
    vec1_i64 = ir.VectorType.get([1], i64)
    v64 = vector.bitcast(vec1_i64, v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def load_b_pack_w4a16(
    buffer_ops,
    flir,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: ir.Value,
    ku: int,
    n_blk: ir.Value,
    n_intra: ir.Value,
    lane_div_16: ir.Value,
    elem_type: ir.Type,
    kpack_bytes: int = 8,
    act_elem_bytes: int = 2,
) -> tuple[ir.Value, ir.Value]:
    """Load B pack for W4A16: returns (b0, b1) for two MFMA_BF16_K16 micro-steps.

    For W4A16 (bf16 activations, packed int4 weights):
    - K64 bytes of activation = 32 bf16 elements = 2x MFMA_BF16_K16
    - Each MFMA needs 4 bf16 per thread
    - One 4-byte load provides 8 int4 -> 8 bf16 -> (even=4, odd=4)

    The preshuffle layout was designed for int8 MFMA (K=32), where each k1
    (lane_div_16) group covers 16 K elements. For bf16 MFMA (K=16), we need
    to remap addresses so B provides the same K positions that A reads.

    Args:
        ku: K unroll index (0, 1, 2, ...), NOT ki_step.
            Each ku corresponds to one K64-byte block (in activation bytes).
        act_elem_bytes: Activation element bytes (2 for bf16).

    Returns:
        (b0, b1): Two i64 values, each containing 4 bf16 for one MFMA.
    """
    if kpack_bytes != 8:
        raise ValueError(f"W4A16 requires kpack_bytes=8, got {kpack_bytes!r}")

    # Address calculation with K-dimension remapping for bf16 MFMA.
    #
    # The preshuffle layout shape: (N0, K0, KLane=4, NLane=16, KPackBytes=8)
    # - K0 = K_elements / 64 (w_elem_bytes=1 for packed int4)
    # - Each (k0, k1) covers a specific range of K elements
    #
    # Mapping for bf16 (K=16 per MFMA, 4 k1_mfma groups of 4 elements each):
    #   total_k1 = ku * 2 + lane_div_16 // 2
    #   k0_bf16  = k0_base + total_k1 // 4
    #   k1_local = total_k1 % 4
    #   k2_base  = (lane_div_16 % 2) * 4   (selects first or second 4 bytes)
    #
    # The even/odd nibble selection (for b0 vs b1) is done after loading.
    c64 = arith.constant(64, index=True)
    half_bytes = kpack_bytes // 2  # = 4

    # k0_base from base_k (B layout uses w_elem_bytes=1)
    k0_base = base_k / c64

    # Remap addresses based on ku and lane_div_16
    k1_layout_offset = ku * 2  # compile-time
    c2_idx = arith.constant(2, index=True)
    c4_idx = arith.constant(4, index=True)

    # lane_div_16 // 2: maps lanes 0,1 -> 0 and lanes 2,3 -> 1
    lane_div_32 = lane_div_16 / c2_idx
    total_k1 = arith.constant(k1_layout_offset, index=True) + lane_div_32

    k0 = k0_base + (total_k1 / c4_idx)
    k1_local = total_k1 % c4_idx

    # (lane_div_16 % 2): selects first or second 4 bytes within KPack
    lane_odd = lane_div_16 % c2_idx
    k2_base = lane_odd * arith.constant(half_bytes, index=True)

    coord_pack = flir.make_coord(n_blk, k0, k1_local, n_intra, arith.constant(0, index=True))
    idx_pack = flir.crd2idx(coord_pack, layout_b)
    idx_bytes = idx_pack + k2_base

    # Load 4 bytes (8 packed int4)
    atom = flir.make_copy_atom(elem_type, vector_size=4)
    b_view = flir.TensorView(
        arg_b,
        (4,),
        strides=(1,),
        base_indices=(idx_bytes,),
        element_type=elem_type,
    )
    b4 = flir.copy(
        atom,
        b_view,
        None,
        alignment=4,
        return_vector=True,
        src_buffer_resource=b_rsrc,
        src_buffer_offset_in_bytes=True,
    )
    vec1_i32 = ir.VectorType.get([1], ir.IntegerType.get_signless(32))
    packed32 = vector.extract(
        vector.bitcast(vec1_i32, b4),
        static_position=[0],
        dynamic_position=[],
    )

    # 7-op unpack to get 8 signed int8 values
    # Output layout: even=[v0,v1,v2,v3], odd=[v4,v5,v6,v7]
    c_08080808 = arith.constant(0x08080808, type=ir.IntegerType.get_signless(32))
    c_0f0f0f0f = arith.constant(0x0F0F0F0F, type=ir.IntegerType.get_signless(32))
    c_1e = arith.constant(0x1E, type=ir.IntegerType.get_signless(32))
    c_4_i32 = arith.constant(4, type=ir.IntegerType.get_signless(32))

    s0 = (packed32 & c_08080808) * c_1e
    even = (packed32 & c_0f0f0f0f) | s0

    t = packed32 >> c_4_i32
    s1 = (t & c_08080808) * c_1e
    odd = (t & c_0f0f0f0f) | s1

    # Convert int8 to bf16 using shift-based f32→bf16 truncation
    b0 = _i8x4_in_i32_to_bf16x4_i64(even, arith, vector)
    b1 = _i8x4_in_i32_to_bf16x4_i64(odd, arith, vector)

    return (b0, b1)


def _i8x4_in_i32_to_bf16x4_i64(val_i32, arith, vector, scale_val=None):
    """Convert one i32 (4 signed int8 bytes) → 4 bf16 packed as i64.

    Uses shift-based f32→bf16 truncation (``v_lshrrev_b32 dst, 16, src``)
    instead of ``arith.truncf`` which on gfx942 expands to ~5 VALU per
    element (software RNE).  The shift is exact for unscaled int8 values
    and introduces <0.5 ULP error for scaled values — negligible for
    inference.

    If *scale_val* (an f32 SSA value) is provided, each f32 is multiplied
    by the scale before truncation (groupwise path).
    """
    i8 = ir.IntegerType.get_signless(8)
    i32 = ir.IntegerType.get_signless(32)
    f32 = ir.F32Type.get()
    vec1_i32_t = ir.VectorType.get([1], i32)
    vec2_i32 = ir.VectorType.get([2], i32)
    vec4_i8 = ir.VectorType.get([4], i8)
    vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))

    v1 = vector.from_elements(vec1_i32_t, [val_i32])
    i8x4 = vector.bitcast(vec4_i8, v1)

    # 4× scalar sitofp (no packed hw instruction for int8→f32)
    f32_vals = []
    for i in range(4):
        val_i8 = vector.extract(i8x4, static_position=[i], dynamic_position=[])
        v = arith.sitofp(f32, val_i8)
        if scale_val is not None:
            v = v * scale_val
        f32_vals.append(v)

    # Shift-based f32→bf16: take upper 16 bits of each f32 (truncation toward zero).
    # ~3 VALU per pair vs ~10 VALU for software RNE truncf on gfx942.
    c16 = arith.constant(16, type=i32)
    c_ffff0000 = arith.constant(0xFFFF0000, type=i32)
    bits0 = arith.bitcast(i32, f32_vals[0])
    bits1 = arith.bitcast(i32, f32_vals[1])
    bits2 = arith.bitcast(i32, f32_vals[2])
    bits3 = arith.bitcast(i32, f32_vals[3])
    i32_lo = arith.ori(arith.shrui(bits0, c16), arith.andi(bits1, c_ffff0000))
    i32_hi = arith.ori(arith.shrui(bits2, c16), arith.andi(bits3, c_ffff0000))

    v2 = vector.from_elements(vec2_i32, [i32_lo, i32_hi])
    v64 = vector.bitcast(vec1_i64, v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def tile_chunk_coord_i32(
    flir,
    arith,
    *,
    tx_i32_base: ir.Value,
    i: int,
    total_threads: int,
    layout_tile_div4,
    chunk_i32: int = 4,
):
    """Map (thread, chunk_id) -> (row_local, col_local_i32) for X/A loads.

    General form (dword-granularity):
      chunk_linear   = tx + i*total_threads
      chunk_i32_base = chunk_linear * chunk_i32

    Where chunk_i32 is the number of dwords per chunk:
      - 4  -> 16B (dwordx4)
      - 2  ->  8B (dwordx2)
      - 1  ->  4B (dword)

    NOTE: `layout_tile_div4` is expressed in dword elements along K (K/4),
    matching the existing GEMM/MoE mapping.
    """
    if chunk_i32 not in (1, 2, 4):
        raise ValueError(f"chunk_i32 must be one of (1,2,4), got {chunk_i32!r}")
    chunk_off_i32 = arith.constant(i * total_threads * chunk_i32, index=True)
    tile_idx_i32 = tx_i32_base + chunk_off_i32
    coord_local = flir.idx2crd(tile_idx_i32, layout_tile_div4)
    row_local = flir.get(coord_local, 0)
    col_local_i32 = flir.get(coord_local, 1)
    return row_local, col_local_i32


def buffer_copy_gmem16_dwordx4(
    flir,
    *,
    arg,
    elem_type,
    idx_i32: ir.Value,
    atom_g2r16,
    rsrc,
    vec_elems: int = 16,
):
    """Copy 16 bytes from global memory into regs via buffer-load dwordx4 lowering.

    `idx_i32` is a dword element offset (not bytes), so `src_buffer_offset_in_bytes=False`.
    """
    if int(vec_elems) <= 0:
        raise ValueError(f"vec_elems must be > 0, got {vec_elems!r}")
    view = flir.TensorView(
        arg,
        (int(vec_elems),),
        strides=(1,),
        base_indices=(idx_i32,),
        element_type=elem_type,
    )
    return flir.copy(
        atom_g2r16,
        view,
        None,
        alignment=16,
        return_vector=True,
        src_buffer_resource=rsrc,
        src_buffer_offset_in_bytes=False,
    )


def lds_store_16b_xor16(
    flir,
    arith,
    vector,
    *,
    lds_memref,
    vec16_ty,
    elem_type,
    atom_s16,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x4: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 16B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = flir.swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else (col_swz_bytes / arith.constant(2, index=True))
    coord_store = flir.make_coord(row_local, col_swz)
    idx0 = flir.crd2idx(coord_store, layout_lds)
    idx0 = idx0 + lds_base
    v16 = vector.bitcast(vec16_ty, vec_part_i32x4)
    extent_elems = 16 if elem_bytes == 1 else 8
    s_view = flir.TensorView(
        lds_memref,
        (extent_elems,),
        strides=(1,),
        base_indices=(idx0,),
        element_type=elem_type,
    )
    flir.copy(atom_s16, v16, s_view, alignment=16)


def lds_store_8b_xor16(
    flir,
    arith,
    vector,
    *,
    lds_memref,
    vec8_ty,
    elem_type,
    atom_s8,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x2: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 8B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = flir.swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else (col_swz_bytes / arith.constant(2, index=True))
    coord_store = flir.make_coord(row_local, col_swz)
    idx0 = flir.crd2idx(coord_store, layout_lds)
    idx0 = idx0 + lds_base
    v8 = vector.bitcast(vec8_ty, vec_part_i32x2)
    extent_elems = 8 if elem_bytes == 1 else 4
    s_view = flir.TensorView(
        lds_memref,
        (extent_elems,),
        strides=(1,),
        base_indices=(idx0,),
        element_type=elem_type,
    )
    flir.copy(atom_s8, v8, s_view, alignment=8)


def lds_store_4b_xor16(
    flir,
    arith,
    vector,
    *,
    lds_memref,
    vec4_ty,
    elem_type,
    atom_s4,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x1: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 4B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = flir.swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else (col_swz_bytes / arith.constant(2, index=True))
    coord_store = flir.make_coord(row_local, col_swz)
    idx0 = flir.crd2idx(coord_store, layout_lds)
    idx0 = idx0 + lds_base
    v4 = vector.bitcast(vec4_ty, vec_part_i32x1)
    extent_elems = 4 if elem_bytes == 1 else 2
    s_view = flir.TensorView(
        lds_memref,
        (extent_elems,),
        strides=(1,),
        base_indices=(idx0,),
        element_type=elem_type,
    )
    flir.copy(atom_s4, v4, s_view, alignment=4)


def lds_load_pack_k32(
    flir,
    arith,
    vector,
    *,
    lds_memref,
    layout_lds,
    k_blocks16: ir.Value,
    curr_row_a_lds: ir.Value,
    col_base: ir.Value,
    half: int,
    lds_base: ir.Value,
    ck_lds128: bool,
    vec16_ty,
    vec8_ty,
    vec2_i64_ty,
    vec1_i64_ty,
):
    """Load one i64 A-pack for an MFMA K32 micro-step from LDS.

    - ck_lds128=True: load 16B and extract half (8B) as i64
    - ck_lds128=False: load 8B directly as i64
    """
    col_base_swz = flir.swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
    if ck_lds128:
        coord_a16 = flir.make_coord(curr_row_a_lds, col_base_swz)
        idx_a16 = flir.crd2idx(coord_a16, layout_lds)
        idx_a16 = idx_a16 + lds_base
        loaded_a16 = vector.load_op(vec16_ty, lds_memref, [idx_a16])
        a_vec128 = vector.bitcast(vec2_i64_ty, loaded_a16)
        return vector.extract(a_vec128, static_position=[half], dynamic_position=[])
    else:
        col_swizzled = col_base_swz + arith.constant(int(half) * 8, index=True)
        coord_a = flir.make_coord(curr_row_a_lds, col_swizzled)
        idx_a = flir.crd2idx(coord_a, layout_lds)
        idx_a = idx_a + lds_base
        loaded_a8 = vector.load_op(vec8_ty, lds_memref, [idx_a])
        a_vec64 = vector.bitcast(vec1_i64_ty, loaded_a8)
        return vector.extract(a_vec64, static_position=[0], dynamic_position=[])


__all__ = [
    "PreshuffleBLayout",
    "buffer_copy_gmem16_dwordx4",
    "lds_load_pack_k32",
    "lds_store_4b_xor16",
    "lds_store_8b_xor16",
    "lds_store_16b_xor16",
    "make_preshuffle_b_layout",
    "make_preshuffle_scale_layout",
    "load_b_pack_k32",
    "tile_chunk_coord_i32",
]



def load_b_pack_w4a16_groupwise(
    buffer_ops,
    flir,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: "ir.Value",
    ku: int,
    n_blk: "ir.Value",
    n_intra: "ir.Value",
    lane_div_16: "ir.Value",
    elem_type: "ir.Type",
    # Group-wise scale parameters
    scale_rsrc,
    expert_offset: "ir.Value",  # expert_id * N_per_expert (runtime SSA value)
    num_groups: int,  # K // group_size
    group_size: int,
    n_per_expert: int,  # N per expert (2*inter_dim for stage1, model_dim for stage2)
    kpack_bytes: int = 8,
    act_elem_bytes: int = 2,
) -> tuple:
    """Load B pack for W4A16 with group-wise scale applied in-kernel.

    Similar to load_b_pack_w4a16 but applies per-group scale during dequant.

    For W4A16 with group_size=G:
    - scale layout: [E, K//G, N] (Opt 0: cache-friendly, adjacent threads read adjacent N)
    - Flat index = expert_offset*(G-1) + n_global + group_idx * N_per_expert
    - For each loaded int4, compute k_pos // G to index the correct scale group

    Args:
        scale_rsrc: Buffer resource for scale tensor (flattened [E, K//G, N])
        expert_offset: expert_id * N_per_expert (SSA value for index computation)
        num_groups: K // group_size (compile-time constant)
        group_size: Number of K elements sharing one scale value
        n_per_expert: N dimension per expert (compile-time constant)

    Returns:
        (b0, b1): Two i64 values, each containing 4 scaled bf16 for one MFMA.
    """
    if kpack_bytes != 8:
        raise ValueError(f"W4A16 requires kpack_bytes=8, got {kpack_bytes!r}")

    # Same address calculation as load_b_pack_w4a16
    c64 = arith.constant(64, index=True)
    half_bytes = kpack_bytes // 2

    k0_base = base_k / c64
    k1_layout_offset = ku * 2
    c2_idx = arith.constant(2, index=True)
    c4_idx = arith.constant(4, index=True)

    lane_div_32 = lane_div_16 / c2_idx
    total_k1 = arith.constant(k1_layout_offset, index=True) + lane_div_32

    k0 = k0_base + (total_k1 / c4_idx)
    k1_local = total_k1 % c4_idx

    lane_odd = lane_div_16 % c2_idx
    k2_base = lane_odd * arith.constant(half_bytes, index=True)

    coord_pack = flir.make_coord(n_blk, k0, k1_local, n_intra, arith.constant(0, index=True))
    idx_pack = flir.crd2idx(coord_pack, layout_b)
    idx_bytes = idx_pack + k2_base

    # Load 4 bytes (8 packed int4)
    atom = flir.make_copy_atom(elem_type, vector_size=4)
    b_view = flir.TensorView(
        arg_b,
        (4,),
        strides=(1,),
        base_indices=(idx_bytes,),
        element_type=elem_type,
    )
    b4 = flir.copy(
        atom,
        b_view,
        None,
        alignment=4,
        return_vector=True,
        src_buffer_resource=b_rsrc,
        src_buffer_offset_in_bytes=True,
    )
    vec1_i32 = ir.VectorType.get([1], ir.IntegerType.get_signless(32))
    packed32 = vector.extract(
        vector.bitcast(vec1_i32, b4),
        static_position=[0],
        dynamic_position=[],
    )

    # 7-op unpack to get 8 signed int8 values
    c_08080808 = arith.constant(0x08080808, type=ir.IntegerType.get_signless(32))
    c_0f0f0f0f = arith.constant(0x0F0F0F0F, type=ir.IntegerType.get_signless(32))
    c_1e = arith.constant(0x1E, type=ir.IntegerType.get_signless(32))
    c_4_i32 = arith.constant(4, type=ir.IntegerType.get_signless(32))

    s0 = (packed32 & c_08080808) * c_1e
    even = (packed32 & c_0f0f0f0f) | s0

    t = packed32 >> c_4_i32
    s1 = (t & c_08080808) * c_1e
    odd = (t & c_0f0f0f0f) | s1

    i32 = ir.IntegerType.get_signless(32)
    f32 = ir.F32Type.get()

    # Calculate K position for group index
    # NOTE: base_k is already element count (not bytes)
    c_ku_elems = arith.constant(ku * 32, index=True)  # 32 bf16 per ku
    k_pos_base = base_k + c_ku_elems

    # Group index = k_pos // group_size
    c_group_size = arith.constant(group_size, index=True)
    group_idx = k_pos_base / c_group_size

    # N position for scale indexing (includes expert offset)
    c16 = arith.constant(16, index=True)
    n_global = n_blk * c16 + n_intra

    # Scale index for [E, K//G, N] layout (Opt 0):
    #   flat_idx = expert_offset * (G-1) + n_global + group_idx * N_per_expert
    # Proof: e*N*(G-1) + (e*N+n) + g*N = e*G*N + g*N + n  (correct for [E,G,N] flat)
    c_gm1 = arith.constant(num_groups - 1, index=True)
    c_npe = arith.constant(n_per_expert, index=True)
    scale_idx = expert_offset * c_gm1 + n_global + group_idx * c_npe
    scale_idx_i32 = arith.index_cast(i32, scale_idx)
    scale_val = buffer_ops.buffer_load(scale_rsrc, scale_idx_i32, vec_width=1, dtype=f32)

    # Convert even/odd int8x4 to bf16x4 with scale (shift-based truncation)
    b0 = _i8x4_in_i32_to_bf16x4_i64(even, arith, vector, scale_val=scale_val)
    b1 = _i8x4_in_i32_to_bf16x4_i64(odd, arith, vector, scale_val=scale_val)

    return (b0, b1)


# ---------------------------------------------------------------------------
# Opt 1: Split load / unpack for W4A16 latency hiding
# ---------------------------------------------------------------------------
# The fused functions above (load_b_pack_w4a16, load_b_pack_w4a16_groupwise)
# issue a buffer_load and immediately unpack+dequant in the same call, causing
# a VMEM stall before the VALU work begins.  The split versions below separate
# the buffer_load (Phase 1) from the unpack/dequant (Phase 2) so that the
# caller can issue ALL loads first and then unpack, overlapping VMEM latency.
# ---------------------------------------------------------------------------


def load_b_raw_w4a16(
    buffer_ops,
    flir,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: "ir.Value",
    ku: int,
    n_blk: "ir.Value",
    n_intra: "ir.Value",
    lane_div_16: "ir.Value",
    elem_type: "ir.Type",
    kpack_bytes: int = 8,
    act_elem_bytes: int = 2,
):
    """Phase 1 of W4A16 B load: issue buffer_load_dword, return raw packed i32.

    Same address calculation as load_b_pack_w4a16 but stops after the load.
    The returned ``packed32`` must be passed to :func:`unpack_b_w4a16` later.
    """
    if kpack_bytes != 8:
        raise ValueError(f"W4A16 requires kpack_bytes=8, got {kpack_bytes!r}")

    c64 = arith.constant(64, index=True)
    half_bytes = kpack_bytes // 2

    k0_base = base_k / c64
    k1_layout_offset = ku * 2
    c2_idx = arith.constant(2, index=True)
    c4_idx = arith.constant(4, index=True)

    lane_div_32 = lane_div_16 / c2_idx
    total_k1 = arith.constant(k1_layout_offset, index=True) + lane_div_32

    k0 = k0_base + (total_k1 / c4_idx)
    k1_local = total_k1 % c4_idx

    lane_odd = lane_div_16 % c2_idx
    k2_base = lane_odd * arith.constant(half_bytes, index=True)

    coord_pack = flir.make_coord(n_blk, k0, k1_local, n_intra, arith.constant(0, index=True))
    idx_pack = flir.crd2idx(coord_pack, layout_b)
    idx_bytes = idx_pack + k2_base

    atom = flir.make_copy_atom(elem_type, vector_size=4)
    b_view = flir.TensorView(
        arg_b, (4,), strides=(1,),
        base_indices=(idx_bytes,),
        element_type=elem_type,
    )
    b4 = flir.copy(
        atom, b_view, None,
        alignment=4,
        return_vector=True,
        src_buffer_resource=b_rsrc,
        src_buffer_offset_in_bytes=True,
    )
    vec1_i32 = ir.VectorType.get([1], ir.IntegerType.get_signless(32))
    packed32 = vector.extract(
        vector.bitcast(vec1_i32, b4),
        static_position=[0],
        dynamic_position=[],
    )
    return packed32


def unpack_b_w4a16(packed32, arith, vector):
    """Phase 2 of W4A16 B load: 7-op unpack + int8->bf16 conversion.

    Takes the raw ``packed32`` from :func:`load_b_raw_w4a16` and produces
    ``(b0, b1)`` -- two i64 values each containing 4 bf16 for one MFMA.

    Uses shift-based f32→bf16 truncation (exact for int4 range).
    """
    c_08080808 = arith.constant(0x08080808, type=ir.IntegerType.get_signless(32))
    c_0f0f0f0f = arith.constant(0x0F0F0F0F, type=ir.IntegerType.get_signless(32))
    c_1e = arith.constant(0x1E, type=ir.IntegerType.get_signless(32))
    c_4_i32 = arith.constant(4, type=ir.IntegerType.get_signless(32))

    s0 = (packed32 & c_08080808) * c_1e
    even = (packed32 & c_0f0f0f0f) | s0

    t = packed32 >> c_4_i32
    s1 = (t & c_08080808) * c_1e
    odd = (t & c_0f0f0f0f) | s1

    b0 = _i8x4_in_i32_to_bf16x4_i64(even, arith, vector)
    b1 = _i8x4_in_i32_to_bf16x4_i64(odd, arith, vector)

    return (b0, b1)


def load_b_raw_w4a16_groupwise(
    buffer_ops,
    flir,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: "ir.Value",
    ku: int,
    n_blk: "ir.Value",
    n_intra: "ir.Value",
    lane_div_16: "ir.Value",
    elem_type: "ir.Type",
    scale_rsrc,
    expert_offset: "ir.Value",
    num_groups: int,
    group_size: int,
    n_per_expert: int,
    kpack_bytes: int = 8,
    act_elem_bytes: int = 2,
):
    """Phase 1 of W4A16 groupwise B load: issue buffer_loads for weight + scale.

    Returns ``(packed32, scale_val)`` to be passed to
    :func:`unpack_b_w4a16_groupwise` later.
    """
    if kpack_bytes != 8:
        raise ValueError(f"W4A16 requires kpack_bytes=8, got {kpack_bytes!r}")

    c64 = arith.constant(64, index=True)
    half_bytes = kpack_bytes // 2

    k0_base = base_k / c64
    k1_layout_offset = ku * 2
    c2_idx = arith.constant(2, index=True)
    c4_idx = arith.constant(4, index=True)

    lane_div_32 = lane_div_16 / c2_idx
    total_k1 = arith.constant(k1_layout_offset, index=True) + lane_div_32

    k0 = k0_base + (total_k1 / c4_idx)
    k1_local = total_k1 % c4_idx

    lane_odd = lane_div_16 % c2_idx
    k2_base = lane_odd * arith.constant(half_bytes, index=True)

    coord_pack = flir.make_coord(n_blk, k0, k1_local, n_intra, arith.constant(0, index=True))
    idx_pack = flir.crd2idx(coord_pack, layout_b)
    idx_bytes = idx_pack + k2_base

    atom = flir.make_copy_atom(elem_type, vector_size=4)
    b_view = flir.TensorView(
        arg_b, (4,), strides=(1,),
        base_indices=(idx_bytes,),
        element_type=elem_type,
    )
    b4 = flir.copy(
        atom, b_view, None,
        alignment=4,
        return_vector=True,
        src_buffer_resource=b_rsrc,
        src_buffer_offset_in_bytes=True,
    )
    vec1_i32 = ir.VectorType.get([1], ir.IntegerType.get_signless(32))
    packed32 = vector.extract(
        vector.bitcast(vec1_i32, b4),
        static_position=[0],
        dynamic_position=[],
    )

    i32 = ir.IntegerType.get_signless(32)
    f32 = ir.F32Type.get()

    c_ku_elems = arith.constant(ku * 32, index=True)
    k_pos_base = base_k + c_ku_elems

    c_group_size = arith.constant(group_size, index=True)
    group_idx = k_pos_base / c_group_size

    c16 = arith.constant(16, index=True)
    n_global = n_blk * c16 + n_intra

    c_gm1 = arith.constant(num_groups - 1, index=True)
    c_npe = arith.constant(n_per_expert, index=True)
    scale_idx = expert_offset * c_gm1 + n_global + group_idx * c_npe
    scale_idx_i32 = arith.index_cast(i32, scale_idx)

    scale_val = buffer_ops.buffer_load(scale_rsrc, scale_idx_i32, vec_width=1, dtype=f32)

    return (packed32, scale_val)


def unpack_b_w4a16_groupwise(packed32, scale_val, arith, vector):
    """Phase 2 of W4A16 groupwise B load: unpack + scale + convert to bf16.

    Takes ``(packed32, scale_val)`` from :func:`load_b_raw_w4a16_groupwise`
    and produces ``(b0, b1)`` -- two i64 values each containing 4 scaled bf16.

    Uses shift-based f32→bf16 truncation (<0.5 ULP, fine for inference).
    """
    c_08080808 = arith.constant(0x08080808, type=ir.IntegerType.get_signless(32))
    c_0f0f0f0f = arith.constant(0x0F0F0F0F, type=ir.IntegerType.get_signless(32))
    c_1e = arith.constant(0x1E, type=ir.IntegerType.get_signless(32))
    c_4_i32 = arith.constant(4, type=ir.IntegerType.get_signless(32))

    s0 = (packed32 & c_08080808) * c_1e
    even = (packed32 & c_0f0f0f0f) | s0

    t = packed32 >> c_4_i32
    s1 = (t & c_08080808) * c_1e
    odd = (t & c_0f0f0f0f) | s1

    b0 = _i8x4_in_i32_to_bf16x4_i64(even, arith, vector, scale_val=scale_val)
    b1 = _i8x4_in_i32_to_bf16x4_i64(odd, arith, vector, scale_val=scale_val)

    return (b0, b1)


# ---------------------------------------------------------------------------
#  W8A16 (int8_bf16) helpers: int8 weight load + convert to bf16
# ---------------------------------------------------------------------------

def _i32_to_bf16x4_i64(val_i32, arith, vector):
    """Convert one i32 (4 signed int8) to 4 bf16, returned as i64."""
    i32 = ir.IntegerType.get_signless(32)
    vec2_i32 = ir.VectorType.get([2], i32)
    vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))

    if _SKIP_DEQUANT:
        v = vector.from_elements(vec2_i32, [val_i32, val_i32])
        v64 = vector.bitcast(vec1_i64, v)
        return vector.extract(v64, static_position=[0], dynamic_position=[])

    i8 = ir.IntegerType.get_signless(8)
    f32 = ir.F32Type.get()
    vec1_i32_t = ir.VectorType.get([1], i32)
    vec4_i8 = ir.VectorType.get([4], i8)

    v1 = vector.from_elements(vec1_i32_t, [val_i32])
    i8x4 = vector.bitcast(vec4_i8, v1)

    f32_vals = []
    for i in range(4):
        val_i8 = vector.extract(i8x4, static_position=[i], dynamic_position=[])
        f32_vals.append(arith.sitofp(f32, val_i8))

    # Shift-based f32→bf16: upper 16 bits of f32 = bf16 (exact for int8 range).
    c16 = arith.constant(16, type=i32)
    c_ffff0000 = arith.constant(0xFFFF0000, type=i32)
    bits0 = arith.bitcast(i32, f32_vals[0])
    bits1 = arith.bitcast(i32, f32_vals[1])
    bits2 = arith.bitcast(i32, f32_vals[2])
    bits3 = arith.bitcast(i32, f32_vals[3])
    i32_lo = arith.ori(arith.shrui(bits0, c16), arith.andi(bits1, c_ffff0000))
    i32_hi = arith.ori(arith.shrui(bits2, c16), arith.andi(bits3, c_ffff0000))

    v2 = vector.from_elements(vec2_i32, [i32_lo, i32_hi])
    v64 = vector.bitcast(vec1_i64, v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def _i32_to_bf16x4_scaled_i64(val_i32, scale_val, arith, vector):
    """Convert one i32 (4 signed int8) to 4 bf16 with scale multiply, returned as i64."""
    i32 = ir.IntegerType.get_signless(32)
    vec2_i32 = ir.VectorType.get([2], i32)
    vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))

    if _SKIP_DEQUANT:
        v = vector.from_elements(vec2_i32, [val_i32, val_i32])
        v64 = vector.bitcast(vec1_i64, v)
        return vector.extract(v64, static_position=[0], dynamic_position=[])

    i8 = ir.IntegerType.get_signless(8)
    f32 = ir.F32Type.get()
    vec1_i32_t = ir.VectorType.get([1], i32)
    vec4_i8 = ir.VectorType.get([4], i8)

    v1 = vector.from_elements(vec1_i32_t, [val_i32])
    i8x4 = vector.bitcast(vec4_i8, v1)

    f32_scaled = []
    for i in range(4):
        val_i8 = vector.extract(i8x4, static_position=[i], dynamic_position=[])
        val_f32 = arith.sitofp(f32, val_i8)
        f32_scaled.append(val_f32 * scale_val)

    # Shift-based f32→bf16: <0.5 ULP error after scale multiply (fine for inference).
    c16 = arith.constant(16, type=i32)
    c_ffff0000 = arith.constant(0xFFFF0000, type=i32)
    bits0 = arith.bitcast(i32, f32_scaled[0])
    bits1 = arith.bitcast(i32, f32_scaled[1])
    bits2 = arith.bitcast(i32, f32_scaled[2])
    bits3 = arith.bitcast(i32, f32_scaled[3])
    i32_lo = arith.ori(arith.shrui(bits0, c16), arith.andi(bits1, c_ffff0000))
    i32_hi = arith.ori(arith.shrui(bits2, c16), arith.andi(bits3, c_ffff0000))

    v2 = vector.from_elements(vec2_i32, [i32_lo, i32_hi])
    v64 = vector.bitcast(vec1_i64, v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def load_b_raw_w8a16(
    buffer_ops,
    flir,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: "ir.Value",
    ku: int,
    n_blk: "ir.Value",
    n_intra: "ir.Value",
    lane_div_16: "ir.Value",
    elem_type: "ir.Type",
    kpack_bytes: int = 16,
    act_elem_bytes: int = 2,
):
    """Phase 1 of W8A16 B load: load 8 int8 bytes from preshuffle layout.

    Uses kpack_bytes=16 (same as fp8/int8 paths).  Each KPack is 16 bytes
    holding 16 int8 values.  Two lanes share one KPack: ``lane_odd`` selects
    the first or second 8-byte half, giving 8 int8 values per lane.

    Returns ``(i32_low, i32_high)`` -- two i32 values each containing 4 int8.
    Pass to :func:`convert_b_w8a16` for int8->bf16 conversion.
    """
    if kpack_bytes != 16:
        raise ValueError(f"W8A16 requires kpack_bytes=16, got {kpack_bytes!r}")

    if _SKIP_LOAD_B:
        # Fast path: skip VMEM load entirely, return constant i32 (garbage values).
        _i32 = ir.IntegerType.get_signless(32)
        c42 = arith.constant(42, type=_i32)
        return (c42, c42)

    c64 = arith.constant(64, index=True)
    c2_idx = arith.constant(2, index=True)
    half_bytes = kpack_bytes // 2  # 8

    k0 = base_k / c64  # k0_base; no offset needed (total_k1 < 4)

    # Simplified: total_k1 = ku*2 + lane_div_32, max=3 < 4
    # → k0_offset = 0 (eliminated div), k1_local = total_k1 (eliminated mod)
    lane_div_32 = lane_div_16 / c2_idx
    k1_local = arith.constant(ku * 2, index=True) + lane_div_32

    lane_odd = lane_div_16 % c2_idx
    k2_base = lane_odd * arith.constant(half_bytes, index=True)

    coord_pack = flir.make_coord(n_blk, k0, k1_local, n_intra, arith.constant(0, index=True))
    idx_pack = flir.crd2idx(coord_pack, layout_b)
    idx_bytes = idx_pack + k2_base

    # Single 8-byte load (buffer_load_dwordx2)
    atom8 = flir.make_copy_atom(elem_type, vector_size=8)
    b_view = flir.TensorView(
        arg_b, (8,), strides=(1,),
        base_indices=(idx_bytes,),
        element_type=elem_type,
    )
    b8 = flir.copy(
        atom8, b_view, None,
        alignment=4,
        return_vector=True,
        src_buffer_resource=b_rsrc,
        src_buffer_offset_in_bytes=True,
    )

    vec2_i32 = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    i32x2 = vector.bitcast(vec2_i32, b8)
    i32_low = vector.extract(i32x2, static_position=[0], dynamic_position=[])
    i32_high = vector.extract(i32x2, static_position=[1], dynamic_position=[])

    return (i32_low, i32_high)


def convert_b_w8a16(raw_pair, arith, vector):
    """Phase 2 of W8A16 B load: int8 -> bf16 conversion (no nibble unpack).

    Takes ``(i32_low, i32_high)`` from :func:`load_b_raw_w8a16`.
    Returns ``(b0, b1)`` -- two i64 values each containing 4 bf16 for one MFMA.
    """
    i32_low, i32_high = raw_pair
    b0 = _i32_to_bf16x4_i64(i32_low, arith, vector)
    b1 = _i32_to_bf16x4_i64(i32_high, arith, vector)
    return (b0, b1)


def convert_b_w8a16_single(val_i32, arith, vector):
    """Convert a single i32 (4 int8) to 4 bf16 packed as i64.

    Finer-grained than :func:`convert_b_w8a16` which converts both halves at
    once.  Use this for interleaving individual dequant operations between
    MFMA calls.
    """
    return _i32_to_bf16x4_i64(val_i32, arith, vector)


def convert_b_w8a16_single_scaled(val_i32, scale_val, arith, vector):
    """Convert a single i32 (4 int8) × scale to 4 bf16 packed as i64.

    Groupwise variant of :func:`convert_b_w8a16_single`.
    """
    return _i32_to_bf16x4_scaled_i64(val_i32, scale_val, arith, vector)


def load_b_raw_w8a16_groupwise(
    buffer_ops,
    flir,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: "ir.Value",
    ku: int,
    n_blk: "ir.Value",
    n_intra: "ir.Value",
    lane_div_16: "ir.Value",
    elem_type: "ir.Type",
    scale_rsrc,
    expert_offset: "ir.Value",
    num_groups: int,
    group_size: int,
    n_per_expert: int,
    kpack_bytes: int = 16,
    act_elem_bytes: int = 2,
):
    """Phase 1 of W8A16 groupwise B load: load int8 weight + groupwise scale.

    Returns ``((i32_low, i32_high), scale_val)`` to be passed to
    :func:`convert_b_w8a16_groupwise`.
    """
    if kpack_bytes != 16:
        raise ValueError(f"W8A16 requires kpack_bytes=16, got {kpack_bytes!r}")

    if _SKIP_LOAD_B:
        # Fast path: skip VMEM load + scale load entirely.
        _i32_t = ir.IntegerType.get_signless(32)
        _f32_t = ir.F32Type.get()
        c42 = arith.constant(42, type=_i32_t)
        c1f = arith.constant(1.0, type=_f32_t)
        return ((c42, c42), c1f)

    c64 = arith.constant(64, index=True)
    c2_idx = arith.constant(2, index=True)
    half_bytes = kpack_bytes // 2  # 8

    k0 = base_k / c64  # k0_base; no offset needed (total_k1 < 4)

    # Simplified: total_k1 = ku*2 + lane_div_32, max=3 < 4
    # → k0_offset = 0 (eliminated div), k1_local = total_k1 (eliminated mod)
    lane_div_32 = lane_div_16 / c2_idx
    k1_local = arith.constant(ku * 2, index=True) + lane_div_32

    lane_odd = lane_div_16 % c2_idx
    k2_base = lane_odd * arith.constant(half_bytes, index=True)

    coord_pack = flir.make_coord(n_blk, k0, k1_local, n_intra, arith.constant(0, index=True))
    idx_pack = flir.crd2idx(coord_pack, layout_b)
    idx_bytes = idx_pack + k2_base

    # Single 8-byte load (buffer_load_dwordx2)
    atom8 = flir.make_copy_atom(elem_type, vector_size=8)
    b_view = flir.TensorView(
        arg_b, (8,), strides=(1,),
        base_indices=(idx_bytes,),
        element_type=elem_type,
    )
    b8 = flir.copy(
        atom8, b_view, None,
        alignment=4,
        return_vector=True,
        src_buffer_resource=b_rsrc,
        src_buffer_offset_in_bytes=True,
    )

    vec2_i32 = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    i32x2 = vector.bitcast(vec2_i32, b8)
    i32_low = vector.extract(i32x2, static_position=[0], dynamic_position=[])
    i32_high = vector.extract(i32x2, static_position=[1], dynamic_position=[])

    # Scale addressing: same group for all values within one half-KPack.
    # k_pos_base = base_k + ku * 32 (each ku covers 32 K-positions, same as int4).
    i32 = ir.IntegerType.get_signless(32)
    f32 = ir.F32Type.get()

    if _SKIP_SCALE:
        # Fast path: skip scale VMEM load, return constant scale.
        scale_val = arith.constant(1.0, type=f32)
    else:
        c_ku_elems = arith.constant(ku * 32, index=True)
        k_pos_base = base_k + c_ku_elems

        c_group_size = arith.constant(group_size, index=True)
        group_idx = k_pos_base / c_group_size

        c16 = arith.constant(16, index=True)
        n_global = n_blk * c16 + n_intra

        c_gm1 = arith.constant(num_groups - 1, index=True)
        c_npe = arith.constant(n_per_expert, index=True)
        scale_idx = expert_offset * c_gm1 + n_global + group_idx * c_npe
        scale_idx_i32 = arith.index_cast(i32, scale_idx)

        scale_val = buffer_ops.buffer_load(scale_rsrc, scale_idx_i32, vec_width=1, dtype=f32)

    return ((i32_low, i32_high), scale_val)


def convert_b_w8a16_groupwise(raw_pair, scale_val, arith, vector):
    """Phase 2 of W8A16 groupwise B load: int8 * scale -> bf16.

    Takes ``((i32_low, i32_high), scale_val)`` from :func:`load_b_raw_w8a16_groupwise`.
    Returns ``(b0, b1)`` -- two i64 values each containing 4 scaled bf16.
    """
    i32_low, i32_high = raw_pair
    b0 = _i32_to_bf16x4_scaled_i64(i32_low, scale_val, arith, vector)
    b1 = _i32_to_bf16x4_scaled_i64(i32_high, scale_val, arith, vector)
    return (b0, b1)


# ---------------------------------------------------------------------------
#  Precomputed-address W8A16 load: 16B dwordx4 + register select
# ---------------------------------------------------------------------------

def w8a16_precomp_lane_constants(arith, lane_div_16):
    """Precompute per-thread constants for W8A16 optimised load.

    Call once at kernel start (before K loop).  Returns a dict of values
    that should be passed to :func:`load_b_raw_w8a16_precomp`.

    For k_unroll=2 (tile_k_bytes=128): total_k1 = ku*2 + lane_div_32,
    max total_k1 = 3 < 4, so k0_offset is always 0 and k1_local = total_k1.
    """
    c2 = arith.constant(2, index=True)
    lane_div_32 = lane_div_16 / c2
    # lane_odd: 0 for even groups of 16 threads, 1 for odd → selects KPack half
    lane_odd_idx = lane_div_16 % c2
    # Convert to i1 for arith.select
    is_odd = arith.cmpu(lane_odd_idx, arith.constant(0, index=True), "ugt")
    return {
        "lane_div_32": lane_div_32,
        "is_odd": is_odd,
    }


def load_b_raw_w8a16_precomp(
    flir,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    k0: "ir.Value",
    k1_local: "ir.Value",
    n_blk: "ir.Value",
    n_intra: "ir.Value",
    is_odd: "ir.Value",
    elem_type: "ir.Type",
):
    """W8A16 B load with precomputed k0/k1/is_odd.

    Loads a full 16-byte KPack (buffer_load_dwordx4) and selects the
    correct 8-byte half via register-level ``select``, avoiding the
    expensive per-call address arithmetic of :func:`load_b_raw_w8a16`.

    Returns ``(i32_low, i32_high)`` — same format as the original.
    """
    if _SKIP_LOAD_B:
        _i32_t = ir.IntegerType.get_signless(32)
        c42 = arith.constant(42, type=_i32_t)
        return (c42, c42)

    coord = flir.make_coord(n_blk, k0, k1_local, n_intra, arith.constant(0, index=True))
    idx = flir.crd2idx(coord, layout_b)

    # 16-byte load (buffer_load_dwordx4)
    atom16 = flir.make_copy_atom(elem_type, vector_size=16)
    b_view = flir.TensorView(
        arg_b, (16,), strides=(1,),
        base_indices=(idx,),
        element_type=elem_type,
    )
    b16 = flir.copy(
        atom16, b_view, None,
        alignment=4,
        return_vector=True,
        src_buffer_resource=b_rsrc,
        src_buffer_offset_in_bytes=True,
    )

    # Split into 4 x i32
    i32 = ir.IntegerType.get_signless(32)
    vec4_i32 = ir.VectorType.get([4], i32)
    b_i32x4 = vector.bitcast(vec4_i32, b16)
    d0 = vector.extract(b_i32x4, static_position=[0], dynamic_position=[])
    d1 = vector.extract(b_i32x4, static_position=[1], dynamic_position=[])
    d2 = vector.extract(b_i32x4, static_position=[2], dynamic_position=[])
    d3 = vector.extract(b_i32x4, static_position=[3], dynamic_position=[])

    # is_odd=0 → (d0, d1);  is_odd=1 → (d2, d3)
    i32_low = arith.select(is_odd, d2, d0)
    i32_high = arith.select(is_odd, d3, d1)
    return (i32_low, i32_high)


def load_b_raw_w8a16_groupwise_precomp(
    buffer_ops,
    flir,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    k0: "ir.Value",
    k1_local: "ir.Value",
    n_blk: "ir.Value",
    n_intra: "ir.Value",
    is_odd: "ir.Value",
    elem_type: "ir.Type",
    scale_rsrc,
    scale_idx_base: "ir.Value",
    n_global: "ir.Value",
    group_idx: "ir.Value",
    n_per_expert: int,
):
    """W8A16 groupwise B load with precomputed addresses.

    Same functionality as :func:`load_b_raw_w8a16_groupwise` but all
    lane-dependent and k-tile-dependent arithmetic is precomputed by the
    caller, reducing the critical path before the VMEM load.

    ``scale_idx_base`` = expert_offset * (num_groups - 1)
    ``n_global``       = n_blk * 16 + n_intra       (per ni)
    ``group_idx``      = (base_k + ku * 32) / group_size  (per K-tile, per ku)

    Returns ``((i32_low, i32_high), scale_val)``.
    """
    if _SKIP_LOAD_B:
        _i32_t = ir.IntegerType.get_signless(32)
        _f32_t = ir.F32Type.get()
        c42 = arith.constant(42, type=_i32_t)
        c1f = arith.constant(1.0, type=_f32_t)
        return ((c42, c42), c1f)

    # Weight load (same as non-groupwise precomp)
    i32_low, i32_high = load_b_raw_w8a16_precomp(
        flir, arith, vector,
        arg_b=arg_b, b_rsrc=b_rsrc, layout_b=layout_b,
        k0=k0, k1_local=k1_local,
        n_blk=n_blk, n_intra=n_intra,
        is_odd=is_odd, elem_type=elem_type,
    )

    # Scale load — precomputed components, only a few ops left
    i32_ty = ir.IntegerType.get_signless(32)
    f32 = ir.F32Type.get()
    c_npe = arith.constant(n_per_expert, index=True)
    scale_idx = scale_idx_base + n_global + group_idx * c_npe
    scale_idx_i32 = arith.index_cast(i32_ty, scale_idx)
    scale_val = buffer_ops.buffer_load(scale_rsrc, scale_idx_i32, vec_width=1, dtype=f32)

    return ((i32_low, i32_high), scale_val)


# ---------------------------------------------------------------------------
#  W8A16 16B-per-lane-group load (new preshuffle layout)
# ---------------------------------------------------------------------------

def load_b_raw_w8a16_16B(
    flir,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: "ir.Value",
    ku_pair: int,
    n_blk: "ir.Value",
    n_intra: "ir.Value",
    lane_div_16: "ir.Value",
    elem_type: "ir.Type",
    kpack_bytes: int = 16,
):
    """Load full 16B kpack (buffer_load_dwordx4) for W8A16 new preshuffle layout.

    In the new layout each kpack belongs to a single lane group (k1 = lane_div_16)
    and contains data for two adjacent ku values (first 8B and second 8B) within
    one K64-byte block.  One load covers 2 ku's worth of MFMA B-operands, halving
    weight VMEM instructions vs the old 8B-per-call approach.

    ``ku_pair``: index of the K64 block pair (0, 1, ... for k_unroll//2 pairs).
      - ku_pair=0 → covers ku=0 and ku=1 (k0 = base_k/64 + 0)
      - ku_pair=1 → covers ku=2 and ku=3 (k0 = base_k/64 + 1)

    Returns ``(i32_0, i32_1, i32_2, i32_3)`` — 4 i32 values each holding 4 int8.
      - ``(i32_0, i32_1)`` correspond to the first ku in the pair (first 8B)
      - ``(i32_2, i32_3)`` correspond to the second ku in the pair (second 8B)
    """
    if kpack_bytes != 16:
        raise ValueError(f"W8A16 16B requires kpack_bytes=16, got {kpack_bytes!r}")

    if _SKIP_LOAD_B:
        _i32_t = ir.IntegerType.get_signless(32)
        c42 = arith.constant(42, type=_i32_t)
        return (c42, c42, c42, c42)

    # Compute k0 = base_k / 64 + ku_pair (the K64-block index).
    # Use ku_pair * 4 offset on the k1 dimension (stride overflow wraps to next k0)
    # to avoid creating extra arith.constant ops.
    c64 = arith.constant(64, index=True)
    k0 = base_k / c64
    k1_offset = ku_pair * 4  # 0 for ku_pair=0, 4 for ku_pair=1, etc.

    # k1 = lane_div_16 + k1_offset: lane group's kpack in the right K64 block
    k1 = lane_div_16
    if k1_offset > 0:
        k1 = k1 + arith.constant(k1_offset, index=True)

    coord = flir.make_coord(n_blk, k0, k1, n_intra, arith.constant(0, index=True))
    idx = flir.crd2idx(coord, layout_b)

    # 16-byte load (buffer_load_dwordx4)
    atom16 = flir.make_copy_atom(elem_type, vector_size=16)
    b_view = flir.TensorView(
        arg_b, (16,), strides=(1,),
        base_indices=(idx,),
        element_type=elem_type,
    )
    b16 = flir.copy(
        atom16, b_view, None,
        alignment=4,
        return_vector=True,
        src_buffer_resource=b_rsrc,
        src_buffer_offset_in_bytes=True,
    )

    # Split 16B into 4 x i32
    i32 = ir.IntegerType.get_signless(32)
    vec4_i32 = ir.VectorType.get([4], i32)
    b_i32x4 = vector.bitcast(vec4_i32, b16)
    i32_0 = vector.extract(b_i32x4, static_position=[0], dynamic_position=[])
    i32_1 = vector.extract(b_i32x4, static_position=[1], dynamic_position=[])
    i32_2 = vector.extract(b_i32x4, static_position=[2], dynamic_position=[])
    i32_3 = vector.extract(b_i32x4, static_position=[3], dynamic_position=[])

    return (i32_0, i32_1, i32_2, i32_3)


def load_b_raw_w8a16_16B_groupwise(
    buffer_ops,
    flir,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: "ir.Value",
    ku_pair: int,
    n_blk: "ir.Value",
    n_intra: "ir.Value",
    lane_div_16: "ir.Value",
    elem_type: "ir.Type",
    scale_rsrc,
    expert_offset: "ir.Value",
    num_groups: int,
    group_size: int,
    n_per_expert: int,
    kpack_bytes: int = 16,
):
    """Load full 16B kpack + two groupwise scales for W8A16 new preshuffle layout.

    One call replaces two consecutive ku calls.  ``ku_pair`` selects which K64
    block within the tile to load (same semantics as in :func:`load_b_raw_w8a16_16B`).

    Returns ``((i32_0, i32_1, i32_2, i32_3), (scale_ku0, scale_ku1))``.
    """
    # Weight load (reuse the non-groupwise 16B loader)
    i32_0, i32_1, i32_2, i32_3 = load_b_raw_w8a16_16B(
        flir, arith, vector,
        arg_b=arg_b, b_rsrc=b_rsrc, layout_b=layout_b,
        base_k=base_k, ku_pair=ku_pair,
        n_blk=n_blk, n_intra=n_intra,
        lane_div_16=lane_div_16, elem_type=elem_type,
        kpack_bytes=kpack_bytes,
    )

    # Scale addressing — two scales, one per ku in the pair.
    # ku_pair covers ku = (ku_pair*2) and (ku_pair*2 + 1).
    # Each ku spans 32 K-elements, so:
    #   k_pos_first  = base_k + ku_pair * 64
    #   k_pos_second = base_k + ku_pair * 64 + 32
    i32_ty = ir.IntegerType.get_signless(32)
    f32 = ir.F32Type.get()

    if _SKIP_SCALE:
        scale_ku0 = arith.constant(1.0, type=f32)
        scale_ku1 = arith.constant(1.0, type=f32)
    else:
        c16 = arith.constant(16, index=True)
        n_global = n_blk * c16 + n_intra
        c_gm1 = arith.constant(num_groups - 1, index=True)
        c_npe = arith.constant(n_per_expert, index=True)
        c_gs = arith.constant(group_size, index=True)

        _k_off_val = ku_pair * 64

        # first ku in pair: k_pos = base_k + ku_pair * 64
        k_pos_base = base_k if _k_off_val == 0 else (base_k + arith.constant(_k_off_val, index=True))
        group_idx_ku0 = k_pos_base / c_gs
        scale_idx_ku0 = expert_offset * c_gm1 + n_global + group_idx_ku0 * c_npe
        scale_idx_ku0_i32 = arith.index_cast(i32_ty, scale_idx_ku0)
        scale_ku0 = buffer_ops.buffer_load(scale_rsrc, scale_idx_ku0_i32, vec_width=1, dtype=f32)

        # second ku in pair: k_pos = base_k + ku_pair * 64 + 32
        c_k_off2 = arith.constant(ku_pair * 64 + 32, index=True)
        group_idx_ku1 = (base_k + c_k_off2) / c_gs
        scale_idx_ku1 = expert_offset * c_gm1 + n_global + group_idx_ku1 * c_npe
        scale_idx_ku1_i32 = arith.index_cast(i32_ty, scale_idx_ku1)
        scale_ku1 = buffer_ops.buffer_load(scale_rsrc, scale_idx_ku1_i32, vec_width=1, dtype=f32)

    return ((i32_0, i32_1, i32_2, i32_3), (scale_ku0, scale_ku1))
