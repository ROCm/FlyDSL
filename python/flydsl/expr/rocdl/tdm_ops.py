"""TDM (Tensor Data Mover) operations for gfx1250.

High-level Python API that encapsulates TDM descriptor construction,
analogous to how buffer_ops.py wraps buffer resource descriptors.

The TDM hardware on gfx1250 provides descriptor-driven DMA for
Global <-> LDS transfers. This module hides the bitfield packing
behind a clean API:

    desc = tdm_ops.make_tensor_descriptor_2d(
        global_ptr=arg_a, lds_memref=lds_a_mem,
        global_offset=(blk_m, k_base),
        tensor_shape=(tile_m, K), strides=(K, 1),
        tile_shape=(tile_m, tile_k),
        elem_bytes=2,
        pad_interval=64, pad_amount=8,
        num_warps=8,
    )
    tdm_ops.tensor_load_2d(desc)
    tdm_ops.tensor_wait(0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

from ..._mlir import ir
from ..._mlir.dialects import (
    arith as std_arith,
    llvm as llvm_dialect,
    memref as memref_dialect,
    rocdl,
)
from .. import arith, vector
from ..arith import _to_raw as _raw
from ..typing import T
from ..utils.arith import ArithValue as _ArithValue

__all__ = [
    "TDMDescriptor2D",
    "TDMGatherDescriptor",
    "make_tensor_descriptor_2d",
    "make_tensor_gather_dgroup0",
    "make_tensor_gather_descriptor",
    "tensor_load_2d",
    "tensor_load_gather",
    "tensor_store_gather",
    "tensor_store_2d",
    "tensor_wait",
    "compute_padding_encoding",
    "compute_warp_distribution",
    "l2_prefetch_tile",
]


# ---------------------------------------------------------------------------
# Pure-Python helpers (compile-time, no IR emission)
# ---------------------------------------------------------------------------

def compute_padding_encoding(
    pad_interval_elems: int,
    pad_amount_elems: int,
    elem_bits: int = 16,
) -> Tuple[int, int]:
    """Compute TDM descriptor padding bitfield values.

    Follows Triton TDMUtility.cpp convention:
      padIntervalInDwords = pad_interval_elems * elem_bits / 32
      padAmountInDwords   = pad_amount_elems   * elem_bits / 32
      encoded_interval    = log2(padIntervalInDwords) - 1
      encoded_amount      = padAmountInDwords - 1

    Args:
        pad_interval_elems: Padding interval in elements (e.g. tile_k = 64).
        pad_amount_elems:   Padding amount in elements (e.g. LDS_PAD = 8).
        elem_bits:          Bits per element (16 for f16/bf16, 32 for f32).

    Returns:
        (encoded_interval, encoded_amount) ready for descriptor bits.
    """
    dword_bits = 32
    interval_dw = pad_interval_elems * elem_bits // dword_bits
    amount_dw = pad_amount_elems * elem_bits // dword_bits
    if interval_dw <= 0 or amount_dw <= 0:
        return (0, 0)
    assert interval_dw & (interval_dw - 1) == 0, (
        f"padIntervalInDwords must be power-of-2, got {interval_dw}"
    )
    encoded_interval = int(math.log2(interval_dw)) - 1
    encoded_amount = amount_dw - 1
    return (encoded_interval, encoded_amount)


def compute_warp_distribution(
    block_shape: Sequence[int],
    num_warps: int,
) -> Tuple[list, list]:
    """Compute per-warp block sub-tile after distributing warps.

    Mirrors Triton's tdmGetWarpDistribution + tdmGetAdjustedBlockShape
    from TDMCommon.h.

    Args:
        block_shape: Full tile shape, e.g. [tile_m, tile_k].
        num_warps:   Total number of warps in the workgroup.

    Returns:
        (warps_per_dim, block_per_warp) — how many warps along each dim
        and the sub-tile size each warp handles.
    """
    ndims = len(block_shape)
    warps = [1] * ndims
    remaining = num_warps
    for i in range(ndims):
        while remaining > 1 and warps[i] * 2 <= block_shape[i]:
            warps[i] *= 2
            remaining //= 2
    if remaining > 1:
        warps[-1] *= remaining
    block_per_warp = [
        (block_shape[i] + warps[i] - 1) // warps[i]
        for i in range(ndims)
    ]
    return warps, block_per_warp


# ---------------------------------------------------------------------------
# Descriptor data class
# ---------------------------------------------------------------------------

@dataclass
class TDMDescriptor2D:
    """Holds constructed GROUP0 and GROUP1 vectors for tensor_load_to_lds_d2."""
    dgroup0: object  # vector<4xi32> MLIR Value
    dgroup1: object  # vector<8xi32> MLIR Value


@dataclass
class TDMGatherDescriptor:
    """Holds GROUP0, GROUP1, GROUP2, GROUP3 for TDM gather mode.

    In gather mode, groups 2 and 3 carry row indices instead of
    higher-dimension tensor metadata.

    - 32-bit index mode: up to 8 row indices (4 per group)
    - 16-bit index mode: up to 16 row indices (8 per group)
    """
    dgroup0: object  # vector<4xi32> MLIR Value
    dgroup1: object  # vector<8xi32> MLIR Value
    dgroup2: object  # vector<4xi32> MLIR Value — row indices [0..3] or [0..7]
    dgroup3: object  # vector<4xi32> MLIR Value — row indices [4..7] or [8..15]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unwrap(value):
    """Unwrap ArithValue wrappers to get raw ir.Value."""
    max_depth = 10
    depth = 0
    while depth < max_depth and not isinstance(value, ir.Value):
        if hasattr(value, "_value"):
            value = value._value
        elif hasattr(value, "value"):
            value = value.value
        else:
            break
        depth += 1
    return value


def _i32_const(v: int) -> ir.Value:
    """Emit an i32 constant, handling negative / unsigned values."""
    i32 = ir.IntegerType.get_signless(32)
    if v > 0x7FFFFFFF:
        v = int(v - 2**32)
    return _unwrap(std_arith.ConstantOp(i32, ir.IntegerAttr.get(i32, v)).result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_tensor_descriptor_2d(
    global_ptr,
    lds_memref,
    global_offset: Tuple,
    tensor_shape: Tuple[int, int],
    strides: Tuple[int, int],
    tile_shape: Tuple[int, int],
    elem_bytes: int = 2,
    pad_interval: int = 0,
    pad_amount: int = 0,
    num_warps: int = 1,
    cache_policy: int = 0,
    pred: int = 1,
    workgroup_mask: Union[int, "ir.Value"] = 0,
    lds_byte_offset=None,
    for_store: bool = False,
    atomic_barrier_enable: bool = False,
) -> TDMDescriptor2D:
    """Build a 2D TDM descriptor for tensor_load_to_lds_d2.

    Convention (matching ISA):
      dim0 = innermost (fastest-varying, e.g. K for row-major A)
      dim1 = outermost (e.g. M for row-major A)
      tensor_shape = (outer_size, inner_size) in user order
      strides       = (outer_stride, inner_stride)
      tile_shape    = (outer_tile, inner_tile)
      global_offset is (outer_offset, inner_offset) — MLIR index Values

    Per-warp distribution is handled internally when num_warps > 1:
    each wave computes its own LDS and global offsets so that all waves
    collectively cover the full tile.

    Padding params are in ELEMENTS (converted to dwords for encoding).

    Args:
        global_ptr:    The global tensor (fx.Tensor or fly memref value).
        lds_memref:    The LDS memref value (already the correct buffer slot).
        global_offset: (outer_idx, inner_idx) as MLIR index values.
        tensor_shape:  (outer_size, inner_size) as Python ints.
        strides:       (outer_stride, inner_stride) as Python ints.
        tile_shape:    (outer_tile, inner_tile) as Python ints.
        elem_bytes:    Element size in bytes (2 for f16/bf16, 4 for f32).
        pad_interval:  Padding interval in elements (0 to disable).
        pad_amount:    Padding amount in elements (0 to disable).
        num_warps:     Total warps in the workgroup.
        cache_policy:  Cache policy (0 = default).
        pred:          Predicate (1 = enabled).
        workgroup_mask: MCAST workgroup mask [15:0] for TDM GROUP1 descriptor.
                       int: compile-time constant folded into descriptor.
                       ir.Value (i32 SGPR): runtime mask, ORed with upper config bits.
                       0 = no multicast (default).
        lds_byte_offset: Optional extra LDS byte offset applied after the per-wave
                       LDS address is computed. Use this when multiple descriptors
                       share the same LDS backing allocation.
        for_store:      Build a descriptor for the LDS->global store path. When
                       enabled, any LDS padding is folded into the tile extent
                       because stores do not perform an implicit de-padding step.
        atomic_barrier_enable: Set the descriptor's hardware auto-barrier bit.
                       Leave this disabled unless the kernel is intentionally
                       relying on TDM atomic-barrier semantics; this helper keeps
                       the encoded atomic-barrier address at zero, so all
                       participating waves must agree on that protocol.

    Returns:
        TDMDescriptor2D with dgroup0 and dgroup1 ready for tensor_load_2d.
    """
    from ..._mlir.dialects import fly as _fly_d

    outer_size, inner_size = tensor_shape
    outer_stride, inner_stride = strides
    outer_tile, inner_tile = tile_shape
    outer_off, inner_off = global_offset

    # -- Warp distribution --
    warps_per_dim, block_per_warp = compute_warp_distribution(
        [outer_tile, inner_tile], num_warps,
    )
    bpw_outer, bpw_inner = block_per_warp
    warps_dim0 = warps_per_dim[0]

    if num_warps > 1:
        # Auto-acquire SGPR wave_id via hardware register (TTMP8[29:25]).
        # This keeps the entire descriptor address chain in SALU,
        from .. import rocdl as _rocdl_ext
        _wid_i32 = _rocdl_ext.wave_id()
        wave_id = arith.index_cast(T.index, _wid_i32)
        warp_coord_outer = wave_id % arith.index(warps_dim0)
        warp_coord_inner = wave_id / arith.index(warps_dim0)
        warp_off_outer = warp_coord_outer * arith.index(bpw_outer)
        warp_off_inner = warp_coord_inner * arith.index(bpw_inner)
    else:
        warp_off_outer = arith.index(0)
        warp_off_inner = arith.index(0)

    # -- Global address (byte address for descriptor) --
    glb_ptr_type = ir.Type.parse("!llvm.ptr<1>")
    i64 = ir.IntegerType.get_signless(64)
    a_raw = global_ptr.__fly_values__()[0]
    glb_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, a_raw)
    glb_base_i64 = _ArithValue(llvm_dialect.ptrtoint(i64, glb_ptr))
    glb_elem_off = (
        (outer_off + warp_off_outer) * arith.index(outer_stride)
        + (inner_off + warp_off_inner) * arith.index(inner_stride)
    )
    glb_byte_off = glb_elem_off * arith.index(elem_bytes)
    glb_byte_off_i64 = arith.index_cast(T.i64, glb_byte_off)
    glb_addr_i64 = glb_base_i64 + glb_byte_off_i64

    # -- LDS address (byte address within shared memory) --
    lds_base_idx = _ArithValue(memref_dialect.extract_aligned_pointer_as_index(lds_memref))
    # Compute padded LDS stride (elements) for the outer dim
    if pad_interval > 0 and pad_amount > 0:
        lds_inner_stride = inner_tile + pad_amount  # padded row width
    else:
        lds_inner_stride = inner_tile
    lds_warp_elem_off = (
        warp_off_outer * arith.index(lds_inner_stride) + warp_off_inner
    )
    lds_warp_byte_off = lds_warp_elem_off * arith.index(elem_bytes)
    lds_total_off = lds_base_idx + lds_warp_byte_off
    if lds_byte_offset is not None:
        lds_total_off = lds_total_off + lds_byte_offset
    lds_addr_i32 = arith.index_cast(T.i32, lds_total_off)

    # ================================================================
    # GROUP0 (vector<4xi32>): pred, lds_addr, global_addr_lo/hi
    # ================================================================
    g0_s0 = arith.constant(pred, type=T.i32)
    g0_s1 = lds_addr_i32
    i32 = ir.IntegerType.get_signless(32)
    g0_s2 = _ArithValue(std_arith.TruncIOp(i32, _raw(glb_addr_i64)).result)
    hi_raw = _ArithValue(_raw(glb_addr_i64)).shrui(arith.constant(32, type=T.i64))
    g0_s3 = (
        _ArithValue(std_arith.TruncIOp(i32, _raw(hi_raw)).result)
        | arith.constant(1 << 31, type=T.i32)  # type field = 2 in [31:30]
    )
    dgroup0 = vector.from_elements(
        T.vec(4, T.i32), [g0_s0, g0_s1, g0_s2, g0_s3]
    )

    # ================================================================
    # GROUP1 (vector<8xi32>): config + tensor dims + strides + tile
    # ================================================================
    # Descriptor dim ordering: dim0=innermost, dim1=outermost
    tdim0 = bpw_inner    # innermost extent per warp
    tdim1 = bpw_outer    # outermost extent per warp
    tile_d0 = bpw_inner  # block dim0 per warp
    tile_d1 = bpw_outer  # block dim1 per warp

    # Padding can be applied to the LDS address when copying from memory to LDS,
    #  but not when copying from LDS to memory
    #  (there is no "de-padding" operation; padding is ignored).
    if for_store and pad_interval > 0 and pad_amount > 0:
        tile_d0 += pad_amount
        pad_interval = 0
        pad_amount = 0

    # stride_dim0 in descriptor = outermost stride in elements
    stride0 = outer_stride

    # data_size = log2(elem_bytes)
    data_size_code = int(math.log2(elem_bytes))

    # Padding encoding
    if pad_interval > 0 and pad_amount > 0:
        elem_bits = elem_bytes * 8
        enc_interval, enc_amount = compute_padding_encoding(
            pad_interval, pad_amount, elem_bits
        )
        pad_enable = 1
    else:
        enc_interval, enc_amount = 0, 0
        pad_enable = 0

    # sgpr0: config bitfields
    _abe = 1 if atomic_barrier_enable else 0
    g1_s0_upper = (
        (data_size_code << 16)      # data_size [17:16]
        | (_abe << 18)                # atomic_barrier_enable
        | (0 << 19)                   # iterate_enable
        | (pad_enable << 20)          # pad_enable
        | (0 << 21)                   # early_timeout
        | (enc_interval << 22)        # pad_interval [24:22]
        | (enc_amount << 25)          # pad_amount [31:25]
    )

    if isinstance(workgroup_mask, int):
        g1_s0_val = (workgroup_mask & 0xFFFF) | g1_s0_upper
        g1_s0 = arith.constant(g1_s0_val, type=T.i32)
    else:
        upper_const = arith.constant(g1_s0_upper, type=T.i32)
        mask_i32 = arith.andi(workgroup_mask, arith.constant(0xFFFF, type=T.i32))
        g1_s0 = arith.ori(upper_const, mask_i32)

    # sgpr1: atomic_barrier_addr[15:0]=0 | tensor_dim0_lo[31:16]
    g1_s1 = arith.constant((tdim0 & 0xFFFF) << 16, type=T.i32)

    # sgpr2: tensor_dim0_hi[15:0] | tensor_dim1_lo[31:16]
    g1_s2 = arith.constant(
        ((tdim0 >> 16) & 0xFFFF) | ((tdim1 & 0xFFFF) << 16),
        type=T.i32,
    )

    # sgpr3: tensor_dim1_hi[15:0] | tile_dim0[31:16]
    g1_s3 = arith.constant(
        ((tdim1 >> 16) & 0xFFFF) | (tile_d0 << 16),
        type=T.i32,
    )

    # sgpr4: tile_dim1[15:0] | tile_dim2[31:16]=0
    g1_s4 = arith.constant(tile_d1 & 0xFFFF, type=T.i32)

    # sgpr5: tensor_dim0_stride (low 32 bits) — stride of outermost dim
    g1_s5 = arith.constant(stride0 & 0xFFFFFFFF, type=T.i32)

    # sgpr6-7: for 2D, no higher-dim strides
    g1_s6 = arith.constant(0, type=T.i32)
    g1_s7 = arith.constant(0, type=T.i32)

    dgroup1 = vector.from_elements(
        T.vec(8, T.i32),
        [g1_s0, g1_s1, g1_s2, g1_s3, g1_s4, g1_s5, g1_s6, g1_s7],
    )

    return TDMDescriptor2D(dgroup0=dgroup0, dgroup1=dgroup1)


def make_tensor_gather_descriptor(
    global_ptr,
    lds_memref,
    row_indices,
    row_width: int,
    tensor_dim0: int,
    tensor_dim1,
    stride: int,
    elem_bytes: int = 1,
    pad_interval: int = 0,
    pad_amount: int = 0,
    index_size: int = 32,
    gather_tile_dim1=None,
    lds_byte_offset=None,
    global_byte_offset=None,
    workgroup_mask: Union[int, "ir.Value"] = 0,
) -> TDMGatherDescriptor:
    """Build a TDM gather descriptor for loading arbitrary rows from global to LDS.

    In gather mode the TDM fetches rows specified by explicit indices in
    descriptor groups 2 and 3, rather than iterating over contiguous dim1.

    Args:
        global_ptr:    The global tensor pointer (fx.Tensor).
        lds_memref:    The LDS memref base (SmemAllocator base).
        row_indices:   List of row index MLIR i32 Values.  Max 8 for 32-bit
                       mode, max 16 for 16-bit mode.
        row_width:     Width of each row in data_size elements (= tile_dim0).
                       Must be a multiple of 4 bytes.
        tensor_dim0:   Full tensor dimension 0 (row width) for OOB check.
        tensor_dim1:   Full tensor dimension 1 (num rows) for OOB check.
                       Accepts a Python int (compile-time) or an MLIR i32
                       Value / SGPR (runtime).  Per ISA spec §4.10.3.2,
                       row indices >= tensor_dim1 are treated as OOB, so
                       this MUST be >= the actual number of rows (tokens).
        stride:        Stride of dim0 in elements (row stride of the global
                       matrix).
        elem_bytes:    Element size in bytes (1, 2, 4, or 8).
        pad_interval:  Padding interval in elements (0 to disable).
        pad_amount:    Padding amount in elements (0 to disable).
        index_size:    Row index width in bits (16 or 32).
        gather_tile_dim1:
                      Optional override for gather-mode tile_dim1 (the number
                      of valid indices to consume from groups 2/3). Accepts a
                      Python int or runtime MLIR i32 Value / SGPR. Defaults to
                      len(row_indices), preserving the historical behavior.
        lds_byte_offset: Additional LDS byte offset.
        global_byte_offset: Additional global memory byte offset (MLIR index).
                           Used for K-tile column offsets.
        workgroup_mask: Multicast mask.

    Returns:
        TDMGatherDescriptor with groups 0-3 ready for tensor_load_gather.
    """
    assert index_size in (16, 32), f"index_size must be 16 or 32, got {index_size}"
    max_indices = 8 if index_size == 32 else 16
    num_indices = len(row_indices)
    assert 0 < num_indices <= max_indices, (
        f"row_indices length {num_indices} exceeds max {max_indices} for {index_size}-bit mode"
    )
    assert row_width * elem_bytes % 4 == 0, (
        f"row_width * elem_bytes must be multiple of 4, got {row_width * elem_bytes}"
    )

    dgroup0 = make_tensor_gather_dgroup0(
        global_ptr=global_ptr,
        lds_memref=lds_memref,
        index_size=index_size,
        lds_byte_offset=lds_byte_offset,
        global_byte_offset=global_byte_offset,
    )

    # ================================================================
    # GROUP 1: config + tensor dims + tile + stride
    # ================================================================
    data_size_code = int(math.log2(elem_bytes))

    if pad_interval > 0 and pad_amount > 0:
        elem_bits = elem_bytes * 8
        enc_interval, enc_amount = compute_padding_encoding(
            pad_interval, pad_amount, elem_bits
        )
        pad_enable = 1
    else:
        enc_interval, enc_amount = 0, 0
        pad_enable = 0

    if isinstance(workgroup_mask, int):
        g1_s0_val = (
            (workgroup_mask & 0xFFFF)
            | (data_size_code << 16)
            | (0 << 18)               # atomic_barrier_enable
            | (0 << 19)               # iterate_enable (ignored in gather)
            | (pad_enable << 20)
            | (0 << 21)               # early_timeout
            | (enc_interval << 22)
            | (enc_amount << 25)
        )
        g1_s0 = arith.constant(g1_s0_val, type=T.i32)
    else:
        upper = (
            (data_size_code << 16)
            | (pad_enable << 20)
            | (enc_interval << 22)
            | (enc_amount << 25)
        )
        g1_s0 = arith.ori(
            arith.constant(upper, type=T.i32),
            arith.andi(workgroup_mask, arith.constant(0xFFFF, type=T.i32)),
        )

    # tensor_dim0 (32 bits) packed into sgpr1[31:16] and sgpr2[15:0]
    # tensor_dim1 (32 bits) packed into sgpr2[31:16] and sgpr3[15:0]
    #
    # tensor_dim1 may be a runtime MLIR i32 value (e.g. num_tokens) —
    # the TDM hardware uses it for OOB checking on gather row indices.
    _td1_is_runtime = not isinstance(tensor_dim1, int)

    g1_s1 = arith.constant((tensor_dim0 & 0xFFFF) << 16, type=T.i32)

    if _td1_is_runtime:
        _td0_hi = arith.constant((tensor_dim0 >> 16) & 0xFFFF, type=T.i32)
        _td1_lo = arith.andi(tensor_dim1, arith.constant(0xFFFF, type=T.i32))
        _td1_lo_shifted = arith.shli(_td1_lo, arith.constant(16, type=T.i32))
        g1_s2 = arith.ori(_td0_hi, _td1_lo_shifted)

        _td1_hi = arith.andi(
            arith.shrui(tensor_dim1, arith.constant(16, type=T.i32)),
            arith.constant(0xFFFF, type=T.i32),
        )
        g1_s3 = arith.ori(_td1_hi, arith.constant(row_width << 16, type=T.i32))
    else:
        g1_s2 = arith.constant(
            ((tensor_dim0 >> 16) & 0xFFFF) | ((tensor_dim1 & 0xFFFF) << 16),
            type=T.i32,
        )
        g1_s3 = arith.constant(
            ((tensor_dim1 >> 16) & 0xFFFF) | (row_width << 16),
            type=T.i32,
        )

    # sgpr4: tile_dim1[15:0] — in gather mode, this is the number of valid
    # indices consumed from descriptor groups 2/3. Allow kernels to override it
    # at runtime so they can keep a fixed index vector while shrinking the valid
    # prefix for padded MoE tiles.
    if gather_tile_dim1 is None:
        g1_s4 = arith.constant(num_indices & 0xFFFF, type=T.i32)
    elif isinstance(gather_tile_dim1, int):
        g1_s4 = arith.constant(gather_tile_dim1 & 0xFFFF, type=T.i32)
    else:
        g1_s4 = arith.andi(gather_tile_dim1, arith.constant(0xFFFF, type=T.i32))

    # sgpr5: tensor_dim0_stride (dim0 stride = row stride in elements)
    g1_s5 = arith.constant(stride & 0xFFFFFFFF, type=T.i32)

    # sgpr6-7: tensor_dim1_stride (ignored in gather mode)
    g1_s6 = arith.constant(0, type=T.i32)
    g1_s7 = arith.constant(0, type=T.i32)

    dgroup1 = vector.from_elements(
        T.vec(8, T.i32),
        [g1_s0, g1_s1, g1_s2, g1_s3, g1_s4, g1_s5, g1_s6, g1_s7],
    )

    # ================================================================
    # GROUP 2 & 3: row indices
    # ================================================================
    zero = arith.constant(0, type=T.i32)

    if index_size == 32:
        # 32-bit mode: group2 has indices [0..3], group3 has [4..7]
        g2_vals = [row_indices[i] if i < num_indices else zero for i in range(4)]
        g3_vals = [row_indices[i + 4] if (i + 4) < num_indices else zero for i in range(4)]
    else:
        # 16-bit mode: pack 2 x 16-bit indices per 32-bit word
        # Group 2: indices [0..7] packed into 4 x i32
        g2_vals = []
        for w in range(4):
            lo_idx = w * 2
            hi_idx = w * 2 + 1
            lo = row_indices[lo_idx] if lo_idx < num_indices else zero
            hi = row_indices[hi_idx] if hi_idx < num_indices else zero
            lo_masked = arith.andi(lo, arith.constant(0xFFFF, type=T.i32))
            hi_shifted = arith.shli(arith.andi(hi, arith.constant(0xFFFF, type=T.i32)),
                                     arith.constant(16, type=T.i32))
            g2_vals.append(arith.ori(lo_masked, hi_shifted))
        # Group 3: indices [8..15] packed into 4 x i32
        g3_vals = []
        for w in range(4):
            lo_idx = 8 + w * 2
            hi_idx = 8 + w * 2 + 1
            lo = row_indices[lo_idx] if lo_idx < num_indices else zero
            hi = row_indices[hi_idx] if hi_idx < num_indices else zero
            lo_masked = arith.andi(lo, arith.constant(0xFFFF, type=T.i32))
            hi_shifted = arith.shli(arith.andi(hi, arith.constant(0xFFFF, type=T.i32)),
                                     arith.constant(16, type=T.i32))
            g3_vals.append(arith.ori(lo_masked, hi_shifted))

    dgroup2 = vector.from_elements(T.vec(4, T.i32), g2_vals)
    dgroup3 = vector.from_elements(T.vec(4, T.i32), g3_vals)

    return TDMGatherDescriptor(
        dgroup0=dgroup0, dgroup1=dgroup1,
        dgroup2=dgroup2, dgroup3=dgroup3,
    )



def make_tensor_gather_dgroup0(
    global_ptr,
    lds_memref,
    *,
    index_size: int = 32,
    lds_byte_offset=None,
    global_byte_offset=None,
):
    """Build gather descriptor GROUP0 only.

    This is the dynamic address-bearing portion of a TDM gather descriptor.
    Separating it lets kernels hoist static GROUP1/GROUP2/GROUP3 state and
    only rebuild the per-issue address group close to the TDM instruction.
    """
    from ..._mlir.dialects import fly as _fly_d

    assert index_size in (16, 32), f"index_size must be 16 or 32, got {index_size}"

    glb_ptr_type = ir.Type.parse("!llvm.ptr<1>")
    i64 = ir.IntegerType.get_signless(64)
    a_raw = global_ptr.__fly_values__()[0]
    glb_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, a_raw)
    glb_base_i64 = _ArithValue(llvm_dialect.ptrtoint(i64, glb_ptr))
    if global_byte_offset is not None:
        glb_byte_off_i64 = arith.index_cast(T.i64, global_byte_offset)
        glb_base_i64 = glb_base_i64 + glb_byte_off_i64

    lds_base_idx = _ArithValue(memref_dialect.extract_aligned_pointer_as_index(lds_memref))
    lds_total_off = lds_base_idx
    if lds_byte_offset is not None:
        lds_total_off = lds_total_off + lds_byte_offset
    lds_addr_i32 = arith.index_cast(T.i32, lds_total_off)

    gather_index_bit = 1 if index_size == 32 else 0
    g0_pred = (
        1
        | (gather_index_bit << 30)
        | (1 << 31)
    )
    g0_s0 = arith.constant(g0_pred, type=T.i32)
    g0_s1 = lds_addr_i32

    i32 = ir.IntegerType.get_signless(32)
    g0_s2 = _ArithValue(std_arith.TruncIOp(i32, _raw(glb_base_i64)).result)
    hi_raw = _ArithValue(_raw(glb_base_i64)).shrui(arith.constant(32, type=T.i64))
    g0_s3 = (
        _ArithValue(std_arith.TruncIOp(i32, _raw(hi_raw)).result)
        | arith.constant(1 << 31, type=T.i32)
    )
    return vector.from_elements(
        T.vec(4, T.i32), [g0_s0, g0_s1, g0_s2, g0_s3]
    )

def tensor_load_gather(
    desc: TDMGatherDescriptor,
    cache_policy: int = 0,
) -> None:
    """Issue a TDM gather load (Global -> LDS) using row indices.

    Uses the 5-group tensor_load_to_lds intrinsic with groups 2 and 3
    carrying the gather row indices.

    Args:
        desc:         TDMGatherDescriptor from make_tensor_gather_descriptor.
        cache_policy: Cache policy (0 = default).
    """
    dg4 = _raw(_zero_dgroup_v8i32())
    rocdl.tensor_load_to_lds(
        _raw(desc.dgroup0), _raw(desc.dgroup1),
        _raw(desc.dgroup2), _raw(desc.dgroup3),
        dg4, cache_policy,
    )


def tensor_store_gather(
    desc: TDMGatherDescriptor,
    cache_policy: int = 0,
) -> None:
    """Issue a TDM gather store (LDS -> Global) using row indices.

    Uses the 5-group tensor_store_from_lds intrinsic with groups 2 and 3
    carrying the gather row indices.

    Args:
        desc:         TDMGatherDescriptor from make_tensor_gather_descriptor.
        cache_policy: Cache policy (0 = default).
    """
    dg4 = _raw(_zero_dgroup_v8i32())
    rocdl.tensor_store_from_lds(
        _raw(desc.dgroup0), _raw(desc.dgroup1),
        _raw(desc.dgroup2), _raw(desc.dgroup3),
        dg4, cache_policy,
    )


def _zero_dgroup_v4i32():
    """Create a zero vector<4xi32> for unused descriptor groups."""
    z = arith.constant(0, type=T.i32)
    return vector.from_elements(T.vec(4, T.i32), [z, z, z, z])


def _zero_dgroup_v8i32():
    """Create a zero vector<8xi32> for unused descriptor groups."""
    z = arith.constant(0, type=T.i32)
    return vector.from_elements(T.vec(8, T.i32), [z, z, z, z, z, z, z, z])


def tensor_load_2d(
    desc: TDMDescriptor2D,
    cache_policy: int = 0,
) -> None:
    """Issue a TDM 2D async load (Global -> LDS).

    Each wave in the workgroup calls this with its own descriptor
    (as built by make_tensor_descriptor_2d). All waves together
    cover the full tile.

    Uses the unified 5-group intrinsic with dgroup2/dgroup3/dgroup4
    zero-initialized for 2D tensors.

    Args:
        desc:         TDMDescriptor2D from make_tensor_descriptor_2d.
        cache_policy: Cache policy (0 = default).
    """
    dg2 = _raw(_zero_dgroup_v4i32())
    dg3 = _raw(_zero_dgroup_v4i32())
    dg4 = _raw(_zero_dgroup_v8i32())
    rocdl.tensor_load_to_lds(
        _raw(desc.dgroup0), _raw(desc.dgroup1), dg2, dg3, dg4, cache_policy
    )


def tensor_store_2d(
    desc: TDMDescriptor2D,
    cache_policy: int = 0,
) -> None:
    """Issue a TDM 2D async store (LDS -> Global).

    Uses the unified 5-group intrinsic with dgroup2/dgroup3/dgroup4
    zero-initialized for 2D tensors.

    Args:
        desc:         TDMDescriptor2D (with LDS source and global destination).
        cache_policy: Cache policy (0 = default).
    """
    dg2 = _raw(_zero_dgroup_v4i32())
    dg3 = _raw(_zero_dgroup_v4i32())
    dg4 = _raw(_zero_dgroup_v8i32())
    rocdl.tensor_store_from_lds(
        _raw(desc.dgroup0), _raw(desc.dgroup1), dg2, dg3, dg4, cache_policy
    )


def tensor_wait(count: int = 0) -> None:
    """Wait for outstanding TDM tensor operations.

    Issues s_wait_tensorcnt.

    Args:
        count: Number of outstanding operations to allow (0 = wait for all).
    """
    rocdl.s_wait_tensorcnt(count)


# ---------------------------------------------------------------------------
# L2 prefetch
# ---------------------------------------------------------------------------

# Scope constants for global_prefetch
PREFETCH_SCOPE_SE = 8       # SE scope = L2 cache
PREFETCH_SCOPE_DEVICE = 16  # Device scope

def l2_prefetch_tile(
    global_ptr,
    global_offset: Tuple,
    tile_shape: Tuple[int, int],
    strides: Tuple[int, int],
    elem_bytes: int = 2,
    num_warps: int = 1,
    wave_id=None,
    thread_id=None,
    block_threads: int = 256,
    scope: int = PREFETCH_SCOPE_SE,
) -> None:
    """Issue per-lane L2 cache prefetch hints for a 2D tile.

    Each lane in the workgroup prefetches 1 byte at a distinct global address
    within the tile, distributing prefetch coverage across the tile.

    For a tile of outer×inner elements, each lane covers a unique row offset.
    Multiple calls (from successive iterations) accumulate coverage.

    Args:
        global_ptr:    The global tensor (fx.Tensor).
        global_offset: (outer_idx, inner_idx) as MLIR index values.
        tile_shape:    (outer_size, inner_size) in elements.
        strides:       (outer_stride, inner_stride) in elements.
        elem_bytes:    Element size in bytes.
        num_warps:     Total warps in the workgroup.
        wave_id:       Current wave ID (MLIR index). Unused; thread_id used instead.
        thread_id:     Workgroup-local thread ID (MLIR index value).
        block_threads: Total threads in the workgroup.
        scope:         Prefetch scope (default: SE = L2).
    """
    from ..._mlir.dialects import (
        fly as _fly_d,
        llvm as llvm_dialect,
    )

    outer_size, inner_size = tile_shape
    outer_stride, inner_stride = strides
    outer_off, inner_off = global_offset

    # Get global base address as i64
    glb_ptr_type = ir.Type.parse("!llvm.ptr<1>")
    i64 = ir.IntegerType.get_signless(64)
    a_raw = global_ptr.__fly_values__()[0]
    glb_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, a_raw)
    glb_base_i64 = _ArithValue(llvm_dialect.ptrtoint(i64, glb_ptr))

    # Each thread prefetches one row of the tile.
    # thread_id maps to an outer-dim offset within the tile.
    # Total rows = outer_size; if block_threads > outer_size, some threads
    # wrap and prefetch additional cachelines.
    # For simplicity, each thread prefetches row[tid % outer_size], col=0.
    tile_row = thread_id % arith.index(outer_size)

    elem_off = (
        (outer_off + tile_row) * arith.index(outer_stride)
        + inner_off * arith.index(inner_stride)
    )
    byte_off = elem_off * arith.index(elem_bytes)
    byte_off_i64 = arith.index_cast(T.i64, byte_off)
    addr_i64 = glb_base_i64 + byte_off_i64

    # Convert i64 address to pointer
    ptr_val = llvm_dialect.inttoptr(glb_ptr_type, _raw(addr_i64))

    # Issue prefetch hint via ROCDL dialect op.
    # NOTE: rocdl.global_prefetch lowers to llvm.amdgcn.global.prefetch, which
    # requires LLVM ISel support for gfx1250 global_prefetch_b8. If the LLVM
    # build lacks this pattern, the instruction will be silently dropped.
    rocdl.global_prefetch(ptr_val, scope)
