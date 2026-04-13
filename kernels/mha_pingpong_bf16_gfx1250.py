"""
FlyDSL kernel
Warps: 8, Threads/Warp: 32
"""

import math as _math
import operator
import flydsl.expr as fx
import flydsl.compiler as flyc
from flydsl.expr import arith, vector, gpu, rocdl, buffer_ops, math
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl._mlir.dialects import fly as _fly_d
from flydsl.expr.utils.arith import ArithValue as _ArithValue
from flydsl._mlir.dialects import scf as _scf_dialect
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.compiler.jit_function import CompilationContext

NUM_WARPS = 8
THREADS_PER_WARP = 32
TARGET = "hip:gfx1250"

def _addptr(base, offset, elem_bytes=2):
    """Triton-style pointer arithmetic: base + offset * elem_bytes.

    If base is a memref, extracts the raw LLVM pointer first.
    Returns an i64 byte address.
    """
    i64 = ir.IntegerType.get_signless(64)
    i32 = ir.IntegerType.get_signless(32)
    # Extract raw pointer from memref if needed
    base_val = base
    if hasattr(base, "type") and "memref" in str(base.type):
        glb_ptr_type = ir.Type.parse("!llvm.ptr<1>")
        base_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, base)
        base_val = _ArithValue(llvm_dialect.ptrtoint(i64, base_ptr))
    elif hasattr(base, "type") and str(base.type) == "i64":
        base_val = _ArithValue(base)
    else:
        # Already an i64 byte address
        base_val = _ArithValue(base) if not isinstance(base, _ArithValue) else base
    # Convert offset to i64 byte offset
    offset_val = offset
    if hasattr(offset, "type") and str(offset.type) == "i32":
        offset_val = arith.ExtSIOp(i64, offset).result
    elem_bytes_val = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, elem_bytes)).result
    byte_offset = _ArithValue(arith.MulIOp(offset_val, elem_bytes_val).result)
    return _ArithValue(arith.AddIOp(base_val, byte_offset).result)


def _make_tdm_desc(base_addr_i64, shapes, strides, elem_bytes=2, tile_shape=None, num_warps=1, pad_interval=0, pad_amount=0):
    """Construct a TDM descriptor from a raw i64 byte address + shapes + strides.

    tile_shape: (outer_tile, inner_tile) static tile dimensions for the TDM descriptor.
    num_warps: number of warps in the workgroup (for per-warp distribution).
    pad_interval: padded_shared interval in elements (0 = no padding).
    pad_amount: padded_shared pad amount in elements (0 = no padding).
    Returns a TDMDescriptor2D (dgroup0, dgroup1) with per-warp metadata.
    """
    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)

    from flydsl.expr import rocdl as _rocdl_ext
    from flydsl.expr.rocdl.tdm_ops import compute_warp_distribution
    outer_stride = strides[0]

    # Tile shape
    if tile_shape is not None:
        _tile_outer, _tile_inner = tile_shape
    else:
        _tile_outer = _tile_inner = 128

    # Per-warp distribution (matches tdm_ops.make_tensor_descriptor_2d)
    _warps_per_dim, _block_per_warp = compute_warp_distribution(
        [_tile_outer, _tile_inner], num_warps)
    bpw_outer, bpw_inner = _block_per_warp
    warps_dim0 = _warps_per_dim[0]

    # Get base address
    glb_addr_i64 = _ArithValue(base_addr_i64)

    # Per-warp global address offset via wave_id
    if num_warps > 1:
        _wid_i32 = _rocdl_ext.wave_id()
        _wid_idx = arith.index_cast(T.index, _wid_i32)
        _warp_coord_outer = _wid_idx % arith.index(warps_dim0)
        _warp_coord_inner = _wid_idx / arith.index(warps_dim0)
        _warp_off_outer = _warp_coord_outer * arith.index(bpw_outer)
        _warp_off_inner = _warp_coord_inner * arith.index(bpw_inner)
        # Global byte offset: (warp_off_outer * outer_stride + warp_off_inner) * elem_bytes
        if hasattr(outer_stride, "type") and str(outer_stride.type) == "i64":
            _stride_idx = arith.index_cast(T.index, arith.TruncIOp(i32, outer_stride).result)
        elif hasattr(outer_stride, "type") and str(outer_stride.type) == "i32":
            _stride_idx = arith.index_cast(T.index, outer_stride)
        else:
            _stride_idx = arith.index(int(outer_stride) if not hasattr(outer_stride, "type") else 1)
        _warp_elem_off = _warp_off_outer * _stride_idx + _warp_off_inner
        _warp_byte_off = _warp_elem_off * arith.index(elem_bytes)
        _warp_byte_off_i64 = arith.index_cast(T.i64, _warp_byte_off)
        glb_addr_i64 = _ArithValue(arith.AddIOp(glb_addr_i64, _warp_byte_off_i64).result)

    # LDS address placeholder (will be filled by tdm_copy)
    lds_addr_i32 = arith.constant(0, type=T.i32)

    # GROUP0: pred, lds_addr, global_addr_lo/hi
    g0_s0 = arith.constant(1, type=T.i32)  # pred = enabled
    g0_s1 = lds_addr_i32
    g0_s2 = _ArithValue(arith.TruncIOp(i32, glb_addr_i64).result)
    hi_raw = _ArithValue(glb_addr_i64).shrui(arith.constant(32, type=T.i64))
    g0_s3 = _ArithValue(arith.TruncIOp(i32, hi_raw).result) | arith.constant(1 << 31, type=T.i32)
    dgroup0 = vector.from_elements(T.vec(4, T.i32), [g0_s0, g0_s1, g0_s2, g0_s3])

    # GROUP1: config + tensor dims + tile (using per-warp dims)
    data_size_code = int(_math.log2(elem_bytes))
    if pad_interval > 0 and pad_amount > 0:
        elem_bits = elem_bytes * 8
        interval_dw = pad_interval * elem_bits // 32
        amount_dw = pad_amount * elem_bits // 32
        enc_interval = int(_math.log2(interval_dw)) - 1 if interval_dw > 0 else 0
        enc_amount = amount_dw - 1 if amount_dw > 0 else 0
        g1_s0_upper = (data_size_code << 16) | (1 << 20) | (enc_interval << 22) | (enc_amount << 25)
    else:
        g1_s0_upper = (data_size_code << 16)
    g1_s0 = arith.constant(g1_s0_upper, type=T.i32)
    # Per-warp dims: dim0=innermost=bpw_inner, dim1=outermost=bpw_outer
    # Matches tdm_ops.make_tensor_descriptor_2d encoding
    g1_s1 = arith.constant((bpw_inner & 0xFFFF) << 16, type=T.i32)   # tdim0_lo
    g1_s2 = arith.constant((bpw_outer & 0xFFFF) << 16, type=T.i32)   # tdim1_lo
    g1_s3 = arith.constant(bpw_inner << 16, type=T.i32)               # tile_d0
    g1_s4 = arith.constant(bpw_outer & 0xFFFF, type=T.i32)            # tile_d1
    # stride (outer stride in elements)
    if hasattr(outer_stride, "type") and str(outer_stride.type) == "i64":
        g1_s5 = _ArithValue(arith.TruncIOp(i32, outer_stride).result)
    else:
        g1_s5 = outer_stride
    g1_s6 = arith.constant(0, type=T.i32)
    g1_s7 = arith.constant(0, type=T.i32)
    dgroup1 = vector.from_elements(
        T.vec(8, T.i32), [g1_s0, g1_s1, g1_s2, g1_s3, g1_s4, g1_s5, g1_s6, g1_s7])

    desc = tdm_ops.TDMDescriptor2D(dgroup0=dgroup0, dgroup1=dgroup1)
    # Store per-warp metadata for _tdm_copy_to_lds
    desc._bpw_outer = bpw_outer
    desc._bpw_inner = bpw_inner
    desc._warps_dim0 = warps_dim0
    desc._tile_inner = _tile_inner
    desc._num_warps = num_warps
    desc._pad_amount = pad_amount
    return desc


# _lds_alloc removed — LDS allocation now uses SmemAllocator/SmemPtr


def _tdm_copy_to_lds(desc, offsets, dst, elem_bytes=2, pred=None):
    """TDM async copy from global to LDS.

    Creates an updated descriptor with:
    1. LDS target address from dst memref (with per-warp offset)
    2. Global source address offset by (outer_off * stride + inner_off)
    Then issues the async copy.
    """
    from flydsl._mlir.dialects import vector as _vec_d
    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)

    # Get LDS byte address from dst memref (or _LdsSubview)
    _sub_elem_off = None
    if isinstance(dst, _LdsSubview):
        _sub_elem_off = dst.elem_offset  # i32 element offset
        dst = dst.base_mr
    lds_base_idx = memref_dialect.extract_aligned_pointer_as_index(dst)
    # Add sub-buffer element offset (from _memdesc_index_subview) as bytes
    if _sub_elem_off is not None:
        _off_idx = arith.index_cast(T.index, _sub_elem_off)
        _off_bytes = _off_idx * arith.index(elem_bytes)
        lds_base_idx = arith.AddIOp(lds_base_idx, _off_bytes).result

    # Add per-warp LDS offset
    _nw = getattr(desc, "_num_warps", 1)
    if _nw > 1:
        from flydsl.expr import rocdl as _rocdl_ext
        _wid = arith.index_cast(T.index, _rocdl_ext.wave_id())
        _wco = _wid % arith.index(desc._warps_dim0)
        _wci = _wid / arith.index(desc._warps_dim0)
        _woo = _wco * arith.index(desc._bpw_outer)
        _woi = _wci * arith.index(desc._bpw_inner)
        # LDS inner stride = tile_inner + pad_amount (padded row width)
        _pad_amt = getattr(desc, "_pad_amount", 0)
        _lds_inner_stride = desc._tile_inner + _pad_amt
        _lds_warp_elem = _woo * arith.index(_lds_inner_stride) + _woi
        _lds_warp_bytes = _lds_warp_elem * arith.index(elem_bytes)
        lds_base_idx = arith.AddIOp(lds_base_idx,
            arith.IndexCastOp(ir.IndexType.get(), _lds_warp_bytes).result
            if not hasattr(_lds_warp_bytes, "type") or str(getattr(_lds_warp_bytes, "type", "")) != "index"
            else _lds_warp_bytes).result

    lds_addr_i32 = arith.IndexCastOp(i32, lds_base_idx).result

    # Build updated dgroup0 with correct LDS address
    g0_s0 = pred if pred is not None else _vec_d.extract(desc.dgroup0, [], [0])
    g0_s1 = lds_addr_i32  # LDS target address (includes per-warp offset)

    # Apply outer + inner offsets to global address
    # offsets = (outer_off, inner_off)
    outer_off_raw = offsets[0]
    if hasattr(outer_off_raw, "type") and str(outer_off_raw.type) == "i32":
        outer_off_i64 = arith.ExtSIOp(i64, outer_off_raw).result
    elif hasattr(outer_off_raw, "type") and str(outer_off_raw.type) == "i64":
        outer_off_i64 = outer_off_raw
    else:
        outer_off_i64 = arith.ExtSIOp(i64, outer_off_raw).result

    # Extract original global addr from dgroup0[2:3]
    orig_lo = _vec_d.extract(desc.dgroup0, [], [2])
    orig_hi = _vec_d.extract(desc.dgroup0, [], [3])
    # Reconstruct i64 global addr
    lo_i64 = arith.ExtUIOp(i64, orig_lo).result
    hi_masked = arith.AndIOp(orig_hi, arith.ConstantOp(i32, ir.IntegerAttr.get(i32, 0x7FFFFFFF)).result).result
    hi_i64 = arith.ExtUIOp(i64, hi_masked).result
    shift_32 = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, 32)).result
    hi_shifted = arith.ShLIOp(hi_i64, shift_32).result
    orig_addr = arith.AddIOp(lo_i64, hi_shifted).result

    # Extract stride from dgroup1[5] (outer stride in elements)
    stride_i32 = _vec_d.extract(desc.dgroup1, [], [5])
    stride_i64 = arith.ExtSIOp(i64, stride_i32).result

    # off_bytes = (outer_off * stride + inner_off) * elem_bytes
    # inner_stride is always 1 for TDM descriptors (contiguous innermost)
    off_elems = arith.MulIOp(outer_off_i64, stride_i64).result
    # Add inner offset if present
    if len(offsets) > 1:
        inner_off_raw = offsets[1]
        if hasattr(inner_off_raw, "type") and str(inner_off_raw.type) == "i32":
            inner_off_i64 = arith.ExtSIOp(i64, inner_off_raw).result
        elif hasattr(inner_off_raw, "type") and str(inner_off_raw.type) == "i64":
            inner_off_i64 = inner_off_raw
        else:
            inner_off_i64 = arith.ExtSIOp(i64, inner_off_raw).result
        off_elems = arith.AddIOp(off_elems, inner_off_i64).result
    eb = arith.ConstantOp(i64, ir.IntegerAttr.get(i64, elem_bytes)).result
    off_bytes = arith.MulIOp(off_elems, eb).result
    new_addr = arith.AddIOp(orig_addr, off_bytes).result

    # Split new addr to lo/hi
    g0_s2 = arith.TruncIOp(i32, new_addr).result
    hi_raw = arith.ShRUIOp(new_addr, shift_32).result
    hi_trunc = arith.TruncIOp(i32, hi_raw).result
    type_bit = arith.ConstantOp(i32, ir.IntegerAttr.get(i32, 1 << 31)).result
    g0_s3 = arith.OrIOp(hi_trunc, type_bit).result

    new_dg0 = vector.from_elements(T.vec(4, T.i32), [g0_s0, g0_s1, g0_s2, g0_s3])

    updated_desc = tdm_ops.TDMDescriptor2D(dgroup0=new_dg0, dgroup1=desc.dgroup1)
    tdm_ops.tensor_load_2d(updated_desc)
    return dst


def _to_index(val):
    """Convert a value to MLIR index type."""
    if isinstance(val, ir.Value):
        if val.type == ir.IndexType.get():
            return val
        return arith.IndexCastOp(ir.IndexType.get(), val).result
    if hasattr(val, "ir_value"):
        raw = val.ir_value()
        if isinstance(raw, ir.Value) and raw.type != ir.IndexType.get():
            return arith.IndexCastOp(ir.IndexType.get(), raw).result
        return raw
    if isinstance(val, int):
        return arith.ConstantOp(ir.IndexType.get(), val).result
    raise TypeError(f"_to_index: unexpected type {type(val)}")


class _SCFForCtx:
    """Context manager for scf.for loop."""
    def __init__(self, lb, ub, step, init_vals=None):
        start = _to_index(lb)
        stop = _to_index(ub)
        step_v = _to_index(step)
        raw_inits = []
        if init_vals:
            for v in init_vals:
                if isinstance(v, ir.Value):
                    raw_inits.append(v)
                elif hasattr(v, "ir_value"):
                    raw_inits.append(v.ir_value())
                else:
                    raw_inits.append(v)
        if raw_inits:
            self.for_op = _scf_dialect.ForOp(start, stop, step_v, raw_inits)
        else:
            self.for_op = _scf_dialect.ForOp(start, stop, step_v)
        self.ip = ir.InsertionPoint(self.for_op.body)

    def __enter__(self):
        self.ip.__enter__()
        iv = self.for_op.induction_variable
        iargs = list(self.for_op.inner_iter_args) if self.for_op.inner_iter_args else []
        return iv, iargs

    def __exit__(self, *args):
        self.ip.__exit__(*args)

    @property
    def results(self):
        return list(self.for_op.results)


def _scf_yield_(vals):
    """Emit scf.yield with the given values."""
    raw = []
    for v in vals:
        if isinstance(v, ir.Value):
            raw.append(v)
        elif hasattr(v, "ir_value"):
            raw.append(v.ir_value())
        else:
            raw.append(v)
    _scf_dialect.YieldOp(raw)


def _tree_reduce(vec, reduce_fn, start, count):
    """Tree-reduce `count` elements from `vec` starting at index `start`.

    Uses static vector.extract indices (Python ints) so FlyDSL JIT generates
    constant-index extractelement ops — no waterfall loops on AMDGPU.
    Binary tree structure enables LLVM DAG combine (e.g. max(max(a,b),c) → v_max3).

    reduce_fn: callable(a, b) -> reduced value. Can be arith.maxnumf,
    operator.add (for addf), etc.
    """
    from flydsl._mlir.dialects import vector as _vec_dialect
    vals = [_vec_dialect.extract(vec, [], [start + i]) for i in range(count)]
    while len(vals) > 1:
        nxt = []
        for j in range(0, len(vals) - 1, 2):
            nxt.append(reduce_fn(vals[j], vals[j + 1]))
        if len(vals) % 2:
            nxt.append(vals[-1])
        vals = nxt
    return vals[0]


class _LdsSubview:
    """Wrapper for a sub-buffer view into a multi-buffered LDS memref.

    Carries the base memref and an element offset so that both _tdm_copy_to_lds
    (which extracts the raw pointer) and _lds_load_gf2 (which loads by index)
    can correctly address the sub-buffer.
    """
    def __init__(self, base_mr, elem_offset_i32):
        self.base_mr = base_mr  # the full LDS memref
        self.elem_offset = elem_offset_i32  # i32 element offset within it
        # Expose .type so code expecting a memref-like object can inspect it
        self.type = base_mr.type


def _memdesc_index_subview(memref_or_tuple, idx, buffer_size, pad_interval=0, pad_amount=0):
    """Create a sub-view of a multi-buffered LDS memref at buffer idx.

    memref_or_tuple: the full LDS memref (or (memref, transposed) tuple).
    idx: buffer index (i32 or Python int).
    buffer_size: number of elements per buffer (unpadded).
    pad_interval: padded_shared interval in elements (0 = no padding).
    pad_amount: padded_shared pad amount in elements (0 = no padding).
    Returns the base memref (idx==0) or an _LdsSubview with element offset.
    """
    # Unwrap possible tuple
    if isinstance(memref_or_tuple, tuple):
        base_mr = memref_or_tuple[0]
    else:
        base_mr = memref_or_tuple

    # If idx is a compile-time 0, return the base memref (no offset needed)
    is_zero = False
    if isinstance(idx, int) and idx == 0:
        is_zero = True
    elif isinstance(idx, ir.Value):
        try:
            is_zero = ir.IntegerAttr(idx.owner.attributes["value"]).value == 0
        except:
            pass

    if is_zero:
        return base_mr

    # Compute element offset as i32 for downstream consumers
    if isinstance(idx, int):
        off_i32 = arith.constant(idx * buffer_size, type=T.i32)
    else:
        idx_i32 = idx if str(getattr(idx, "type", "")) == "i32" else arith.index_cast(T.i32, idx)
        off_i32 = idx_i32 * arith.constant(buffer_size, type=T.i32)

    # Apply padded_shared padding adjustment: off += (off >> log2(interval)) << log2(amount)
    if pad_interval > 0 and pad_amount > 0:
        _pi_shift = int(_math.log2(pad_interval))
        _pa_shift = int(_math.log2(pad_amount))
        pad_off = (off_i32 >> arith.constant(_pi_shift, type=T.i32)) << arith.constant(_pa_shift, type=T.i32)
        off_i32 = off_i32 + pad_off

    return _LdsSubview(base_mr, off_i32)


def _lds_load_gf2(lds_ref, reg_offsets, lane_bases, warp_bases, n_elems, elem_dtype,
                  vec_size=1, transpose_load=False, tr_lane_bases=None,
                  n_additive=0, pad_interval=0, pad_amount=0):
    """Load n_elems from LDS using GF(2) XOR-based addressing.

    When vec_size > 1, uses vectorized llvm.load (ds_read_b64/b128) for
    contiguous groups instead of per-element memref.load.
    When transpose_load=True, uses rocdl.ds_load_tr16_b128 for transposed
    V operand loads (8 bf16 per call via hardware transpose).

    Each thread's per-element offset = lane_contribution XOR warp_contribution XOR reg_offset[i].
    reg_offsets: list of per-register-element offsets (length = n_elems).
    lane_bases: GF(2) basis vectors for lane_id bits.
    warp_bases: GF(2) basis vectors for warp_id bits.
    vec_size: max contiguous elements per vectorized load (from C++ largestVectorisation).
    transpose_load: if True, use ds_load_tr16_b128 with remapped lane bases.
    tr_lane_bases: remapped lane bases for transpose loads (bits 0-2 → row offsets).
    pad_interval: padded_shared interval in elements (0 = no padding).
    pad_amount: padded_shared pad amount in elements (0 = no padding).
    """
    from flydsl._mlir.dialects import memref as _mr_d
    from flydsl._mlir.dialects import vector as _vec_dialect
    from flydsl._mlir.dialects import llvm as _llvm_d

    # Unwrap LDS memref from possible (memref, transposed) tuple or _LdsSubview
    _sub_elem_off = None  # i32 element offset for sub-buffer
    if isinstance(lds_ref, tuple):
        inner = lds_ref[0]
        if isinstance(inner, _LdsSubview):
            _sub_elem_off = inner.elem_offset
            lds_flat = inner.base_mr
        else:
            lds_flat = inner
    elif isinstance(lds_ref, _LdsSubview):
        _sub_elem_off = lds_ref.elem_offset
        lds_flat = lds_ref.base_mr
    else:
        lds_flat = lds_ref

    tid = arith.index_cast(T.i32, gpu.thread_id("x"))
    lane = tid & arith.constant(31, type=T.i32)
    warp = (tid >> arith.constant(5, type=T.i32)) & arith.constant(3, type=T.i32)

    # Compute GF(2) lane contribution: XOR of (lane_bit_i * lane_bases[i])
    lane_c = arith.constant(0, type=T.i32)
    for bit, basis in enumerate(lane_bases):
        if basis == 0:
            continue
        lane_bit = (lane >> arith.constant(bit, type=T.i32)) & arith.constant(1, type=T.i32)
        lane_c = lane_c ^ (lane_bit * arith.constant(basis, type=T.i32))

    # Compute GF(2) warp contribution
    warp_c = arith.constant(0, type=T.i32)
    for bit, basis in enumerate(warp_bases):
        if basis == 0:
            continue
        warp_bit = (warp >> arith.constant(bit, type=T.i32)) & arith.constant(1, type=T.i32)
        warp_c = warp_c ^ (warp_bit * arith.constant(basis, type=T.i32))

    thr_base = lane_c ^ warp_c

    # Load elements from LDS at offset = thr_base XOR reg_offsets[i]
    raw_dtype = elem_dtype
    if hasattr(raw_dtype, "ir_type"):
        raw_dtype = raw_dtype.ir_type()
    vec_type = ir.VectorType.get([n_elems], raw_dtype)
    _pad_i_shift = int(_math.log2(pad_interval)) if pad_interval > 0 else 0
    _pad_a_shift = int(_math.log2(pad_amount)) if pad_amount > 0 else 0

    def _apply_padding(elem_off):
        if pad_interval > 0 and pad_amount > 0:
            pad_off = (elem_off >> arith.constant(_pad_i_shift, type=T.i32)) << arith.constant(_pad_a_shift, type=T.i32)
            return elem_off + pad_off
        return elem_off

    # Compile-time padding: apply padding formula on a Python int constant
    def _ct_padding(off):
        if pad_interval <= 0 or pad_amount <= 0:
            return off
        return off + ((off >> _pad_i_shift) << _pad_a_shift)

    # Shared variables for pointer-based LDS access
    lds_ptr_ty = ir.Type.parse('!llvm.ptr<3>')
    lds_base_idx = _mr_d.extract_aligned_pointer_as_index(lds_flat)
    lds_base_i32 = arith.index_cast(T.i32, lds_base_idx)
    elem_bytes = (raw_dtype.width + 7) // 8

    # Transpose load path: use rocdl.ds_load_tr16_b128 for transposed V operand
    if transpose_load and tr_lane_bases is not None:
        from flydsl._mlir.dialects import rocdl as _rocdl_d
        tr_tile = 8  # ds_load_tr16_b128 loads 8 bf16 per call
        tr_vec_ty = ir.VectorType.get([tr_tile], raw_dtype)

        # Rebuild thr_base with transposed lane bases (bits 0-2 → row offsets)
        tr_lane_c = arith.constant(0, type=T.i32)
        for bit, basis in enumerate(tr_lane_bases):
            if basis == 0:
                continue
            lane_bit = (lane >> arith.constant(bit, type=T.i32)) & arith.constant(1, type=T.i32)
            tr_lane_c = tr_lane_c ^ (lane_bit * arith.constant(basis, type=T.i32))
        tr_thr_base = tr_lane_c ^ warp_c

        # Detect additive strides for transpose groups:
        # If group base offset bits are disjoint from lane/warp bits,
        # XOR == ADD and we can use compile-time inner offsets.
        _tr_addr_bits = set()
        for _b in (tr_lane_bases or []):
            for _i in range(32):
                if _b & (1 << _i): _tr_addr_bits.add(_i)
        for _b in (warp_bases or []):
            for _i in range(32):
                if _b & (1 << _i): _tr_addr_bits.add(_i)
        _group_bases = [reg_offsets[g] for g in range(0, n_elems, tr_tile)]
        _tr_additive = len(_group_bases) > 1 and all(
            all((gb >> bit) & 1 == 0 for bit in _tr_addr_bits)
            for gb in _group_bases)

        elems = []
        if _tr_additive:
            # Two-level: one outer XOR+padding, compile-time inner offsets
            outer_reg_off = reg_offsets[0]
            outer_off = tr_thr_base ^ arith.constant(outer_reg_off, type=T.i32)
            outer_off = _apply_padding(outer_off)
            if _sub_elem_off is not None:
                outer_off = outer_off + _sub_elem_off
            outer_byte = lds_base_i32 + outer_off * arith.constant(elem_bytes, type=T.i32)
            for g in range(0, n_elems, tr_tile):
                delta = reg_offsets[g] - outer_reg_off
                byte_delta = _ct_padding(delta) * elem_bytes
                if byte_delta == 0:
                    ptr_addr = outer_byte
                else:
                    ptr_addr = outer_byte + arith.constant(byte_delta, type=T.i32)
                ptr = _llvm_d.inttoptr(lds_ptr_ty, ptr_addr)
                loaded = _rocdl_d.ds_load_tr16_b128(tr_vec_ty, ptr)
                for j in range(tr_tile):
                    elems.append(_llvm_d.extractelement(loaded, arith.constant(j, type=T.i32)))
        else:
            # Fallback: per-group XOR + runtime padding
            for g in range(0, n_elems, tr_tile):
                group_base_off = reg_offsets[g]
                elem_off = tr_thr_base ^ arith.constant(group_base_off, type=T.i32)
                elem_off = _apply_padding(elem_off)
                if _sub_elem_off is not None:
                    elem_off = elem_off + _sub_elem_off
                byte_off = elem_off * arith.constant(elem_bytes, type=T.i32)
                total_byte = lds_base_i32 + byte_off
                ptr = _llvm_d.inttoptr(lds_ptr_ty, total_byte)
                loaded = _rocdl_d.ds_load_tr16_b128(tr_vec_ty, ptr)
                for j in range(tr_tile):
                    elems.append(_llvm_d.extractelement(loaded, arith.constant(j, type=T.i32)))
        return vector.from_elements(vec_type, elems)

    # Additive-strides path: two-level loop with compile-time inner offsets
    if vec_size > 1 and n_additive >= vec_size:
        elems = []
        for outer in range(0, n_elems, n_additive):
            # Outer: XOR-based address with runtime padding
            outer_reg_off = reg_offsets[outer]
            elem_off = thr_base ^ arith.constant(outer_reg_off, type=T.i32)
            elem_off = _apply_padding(elem_off)
            if _sub_elem_off is not None:
                elem_off = elem_off + _sub_elem_off
            byte_off = elem_off * arith.constant(elem_bytes, type=T.i32)
            outer_byte_addr = lds_base_i32 + byte_off
            for inner in range(0, n_additive, vec_size):
                idx = outer + inner
                if idx >= n_elems:
                    break
                inner_reg_off = reg_offsets[idx]
                inner_delta = inner_reg_off - outer_reg_off
                # Compile-time padding on delta (safe: bits are disjoint)
                inner_delta_padded = _ct_padding(inner_delta)
                inner_byte_delta = inner_delta_padded * elem_bytes
                if inner_byte_delta == 0:
                    ptr_addr = outer_byte_addr
                else:
                    ptr_addr = outer_byte_addr + arith.constant(inner_byte_delta, type=T.i32)
                ptr = _llvm_d.inttoptr(lds_ptr_ty, ptr_addr)
                ld_vec_ty = ir.VectorType.get([vec_size], raw_dtype)
                loaded = _llvm_d.load(ld_vec_ty, ptr)
                for j in range(vec_size):
                    elems.append(_llvm_d.extractelement(loaded, arith.constant(j, type=T.i32)))
        return vector.from_elements(vec_type, elems)

    # Vectorized path: use llvm.load for contiguous groups (no additive strides)
    if vec_size > 1:
        elems = []
        i = 0
        while i < n_elems:
            # Check if next vec_size offsets are contiguous
            group_len = 1
            if i + vec_size <= n_elems:
                is_contig = all(reg_offsets[i + j] == reg_offsets[i] + j for j in range(vec_size))
                if is_contig:
                    group_len = vec_size
            base_off = reg_offsets[i]
            elem_off = thr_base ^ arith.constant(base_off, type=T.i32)
            elem_off = _apply_padding(elem_off)
            if _sub_elem_off is not None:
                elem_off = elem_off + _sub_elem_off
            byte_off = elem_off * arith.constant(elem_bytes, type=T.i32)
            total_byte = arith.index_cast(T.i32, lds_base_idx) + byte_off
            ptr = _llvm_d.inttoptr(lds_ptr_ty, total_byte)
            if group_len > 1:
                ld_vec_ty = ir.VectorType.get([group_len], raw_dtype)
                loaded = _llvm_d.load(ld_vec_ty, ptr)
                for j in range(group_len):
                    elems.append(_llvm_d.extractelement(loaded, arith.constant(j, type=T.i32)))
            else:
                elems.append(_llvm_d.load(raw_dtype, ptr))
            i += group_len
        return vector.from_elements(vec_type, elems)

    # Scalar fallback: per-element memref.load (original path)
    elems = []
    for i in range(n_elems):
        elem_off = thr_base ^ arith.constant(reg_offsets[i], type=T.i32)
        elem_off = _apply_padding(elem_off)
        if _sub_elem_off is not None:
            elem_off = elem_off + _sub_elem_off
        idx = arith.index_cast(T.index, elem_off)
        val = _mr_d.load(lds_flat, [idx])
        elems.append(val)
    return vector.from_elements(vec_type, elems)


def _wmma_gemm_full(a_data, b_lds_ref, c_vec, ab_dtype,
                     tile_m=128, tile_n=128, tile_k=128,
                     warps_m=4, warps_n=1, WMMA_M=16, WMMA_N=16, WMMA_K=32,
                     a_pad_interval=0, a_pad_amount=0, b_pad_interval=0, b_pad_amount=0):
    """Complete tiled WMMA gemm: A (register vec) × B (LDS memref) → C (f32 vec).

    A is a flat per-thread register vector from buffer_load (dot_op opIdx=0 layout).
    B is a LDS memref from TDM copy, loaded with WMMA-aware per-element addressing.
    C is the per-thread accumulator vector (f32).

    warps_m/warps_n: how many warps split the M/N dimensions.
    For FA kernel: warps_m=4, warps_n=1 (4 warps along M, each does all N).

    Thread decomposition: lane16 = tid % 16, lane_kgrp = (tid // 16) % 2.
    A fragment: 16 consecutive elements from flat vec.
    B fragment: 16 elements loaded from LDS with WMMA addressing.
    a_pad_* / b_pad_* describe padded_shared row padding when fly.lds_load
    forwards an LDS memref rather than a pre-loaded register vector.
    """
    from flydsl._mlir.dialects import vector as _vec_dialect
    from flydsl._mlir.dialects import memref as _mr_d
    from flydsl.expr import rocdl

    raw_ab = ab_dtype
    if hasattr(raw_ab, "ir_type"):
        raw_ab = raw_ab.ir_type()
    is_bf16 = str(raw_ab) == "bf16"
    wmma_op = rocdl.wmma_f32_16x16x32_bf16 if is_bf16 else rocdl.wmma_f32_16x16x32_f16
    elem_ty = raw_ab

    # Unwrap values
    _a_sub_off = None  # i32 element offset for A sub-buffer
    if isinstance(a_data, _LdsSubview):
        _a_sub_off = a_data.elem_offset
        a_v = a_data.base_mr
    else:
        a_v = a_data.ir_value() if hasattr(a_data, "ir_value") else a_data
    c_v = c_vec.ir_value() if hasattr(c_vec, "ir_value") else c_vec

    # Check if a_data/b_lds_ref are flat register vectors (VectorType) vs LDS memref.
    # Both VectorType and MemRefType have .shape/.element_type, so use isinstance.
    a_is_vec = isinstance(a_v.type, ir.VectorType)

    # Check if B is already a pre-loaded register vector (from fly.lds_load with GF(2) offsets)
    b_raw = b_lds_ref.ir_value() if hasattr(b_lds_ref, "ir_value") else (b_lds_ref if isinstance(b_lds_ref, ir.Value) else None)
    b_is_vec = b_raw is not None and isinstance(b_raw.type, ir.VectorType)

    # ── CuTE layout definitions ──
    # Thread layout: (warps_m, warps_n, kgrp=2, lane=16)
    #   tid → (wave_m_idx, wave_n_idx, lane_kgrp, lane16)
    layout_thr = fx.make_layout(
        (warps_m, warps_n, 2, 16),
        (warps_n * 32, 32, 16, 1))

    # Thread decomposition via CuTE idx2crd
    tid_raw = gpu.thread_id("x")
    tid = arith.index_cast(T.i32, tid_raw)
    thr_crd = _idx2crd(tid, layout_thr)
    wave_m_idx = fx.get(thr_crd, 0)
    wave_n_idx = fx.get(thr_crd, 1)
    lane_kgrp  = fx.get(thr_crd, 2)
    lane16     = fx.get(thr_crd, 3)

    warp_tile_m = tile_m // warps_m
    warp_tile_n = tile_n // warps_n
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    k_wmma_steps = tile_k // WMMA_K
    n_accs = wmma_m_rep * wmma_n_rep

    # Warp-level M and N base offsets
    warp_m_off = wave_m_idx * arith.index(warp_tile_m)
    warp_n_off = wave_n_idx * arith.index(warp_tile_n)

    # Initialize accumulators from c_vec (8 f32 per WMMA result)
    c_per_wmma = 8
    wmma_result_type = ir.VectorType.get([c_per_wmma], ir.F32Type.get())
    accs = []
    for i in range(n_accs):
        c_slice = vector.extract_strided_slice(wmma_result_type, c_v, [i * c_per_wmma], [c_per_wmma], [1])
        accs.append(c_slice)

    # Get B operand: pre-loaded register vector or LDS memref
    # b_lds_ref may be a (memref, transposed=True) tuple from memdesc_trans
    # or an _LdsSubview from _memdesc_index_subview
    b_transposed = False
    _b_sub_off = None  # i32 element offset for sub-buffer
    if not b_is_vec:
        if isinstance(b_lds_ref, tuple):
            b_inner = b_lds_ref[0]
            b_transposed = b_lds_ref[1] if len(b_lds_ref) > 1 else False
            if isinstance(b_inner, _LdsSubview):
                _b_sub_off = b_inner.elem_offset
                b_lds_flat = b_inner.base_mr
            else:
                b_lds_flat = b_inner
        elif isinstance(b_lds_ref, _LdsSubview):
            _b_sub_off = b_lds_ref.elem_offset
            b_lds_flat = b_lds_ref.base_mr
        else:
            b_lds_flat = b_lds_ref

    a_frag_type = ir.VectorType.get([16], elem_ty)
    b_frag_type = ir.VectorType.get([16], elem_ty)
    b_half_type = ir.VectorType.get([8], elem_ty)

    # LDS layouts for crd2idx-based offset computation (only needed when B is memref)
    a_row_stride = (a_pad_interval + a_pad_amount) if a_pad_amount > 0 else tile_k
    if not b_is_vec:
        b_row_stride = (b_pad_interval + b_pad_amount) if b_pad_amount > 0 else (tile_k if b_transposed else tile_n)
        if b_transposed:
            layout_lds_b = fx.make_layout((tile_n, tile_k), (b_row_stride, 1))
    layout_lds_a = fx.make_layout((tile_m, tile_k), (a_row_stride, 1))

    for ks in range(k_wmma_steps):
        k_step = arith.index(ks * WMMA_K)

        # Load B fragments
        b_frags = []
        if b_is_vec:
            # B is pre-loaded register vector (dot_op opIdx=1 layout)
            # Layout: [wn * (k_wmma_steps * 16) + ks * 16 + 0..15]
            for wn in range(wmma_n_rep):
                b_start = wn * (k_wmma_steps * 16) + ks * 16
                b_frags.append(vector.extract_strided_slice(b_frag_type, b_raw, [b_start], [16], [1]))
        else:
            # B is LDS memref. For the regular GEMM fallback path this is
            # the WMMA opIdx=1 layout and must be read via transposed LDS
            # loads rather than naive scalar (k, n) indexing.
            if b_transposed:
                for wn in range(wmma_n_rep):
                    n_off = warp_n_off + arith.index(wn * WMMA_N)
                    vals = []
                    for k0 in range(2):
                        for k1 in range(8):
                            kk = k_step + (arith.index(k0 * 2) + lane_kgrp) * arith.index(8) + arith.index(k1)
                            off = _crd2idx((n_off + lane16, kk), layout_lds_b)
                            if _b_sub_off is not None:
                                off = off + arith.index_cast(T.index, _b_sub_off)
                            val = _mr_d.load(b_lds_flat, [off])
                            vals.append(val)
                    b_frags.append(vector.from_elements(b_frag_type, vals))
            else:
                lane8 = lane16 % arith.index(8)
                lane_ngrp = lane16 / arith.index(8)
                k_lane_off = (lane_kgrp * arith.index(8) + lane8) * arith.index(b_row_stride)
                n_lane_off = lane_ngrp * arith.index(8)
                for wn in range(wmma_n_rep):
                    n_col = warp_n_off + arith.index(wn * WMMA_N) + n_lane_off
                    b_lane_base = k_lane_off + n_col
                    if _b_sub_off is not None:
                        b_lane_base = b_lane_base + arith.index_cast(T.index, _b_sub_off)
                    halves = []
                    for k_half in range(2):
                        k_row_off = arith.index((ks * WMMA_K + k_half * 16) * b_row_stride)
                        elem_off = b_lane_base + k_row_off
                        halves.append(rocdl.lds_transpose_load(b_half_type, b_lds_flat, elem_off, 2))
                    b_frags.append(vector.shuffle(halves[0], halves[1], list(range(16))))

        # For each M-tile, extract A fragment and call WMMA
        for wm in range(wmma_m_rep):
            if a_is_vec:
                # A is in flat register vector (dot_op opIdx=0 layout)
                # Register layout: [m_tile_local * (k_wmma_steps * 16) + ks * 16 + 0..15]
                a_start = wm * (k_wmma_steps * 16) + ks * 16
                a_frag = vector.extract_strided_slice(a_frag_type, a_v, [a_start], [16], [1])
            else:
                # A is also in LDS — use crd2idx with warp M offset
                a_vals = []
                m_off = warp_m_off + arith.index(wm * WMMA_M)
                for k0 in range(2):
                    for k1 in range(8):
                        kk = k_step + (arith.index(k0 * 2) + lane_kgrp) * arith.index(8) + arith.index(k1)
                        off = _crd2idx((m_off + lane16, kk), layout_lds_a)
                        if _a_sub_off is not None:
                            off = off + arith.index_cast(T.index, _a_sub_off)
                        a_vals.append(_mr_d.load(a_v, [off]))
                a_frag = vector.from_elements(a_frag_type, a_vals)

            for wn in range(wmma_n_rep):
                acc_idx = wm * wmma_n_rep + wn
                accs[acc_idx] = wmma_op(
                    wmma_result_type,
                    b_frags[wn],
                    a_frag,
                    accs[acc_idx],
                    signA=False, signB=False, modC=0,
                    reuseA=False, reuseB=False,
                ).result

    # Pack accumulators back into a flat result vector
    result = c_v
    for i in range(n_accs):
        result = vector.insert_strided_slice(accs[i], result, [i * c_per_wmma], [1])

    return result


def _crd2idx(crd, layout):
    """CuTE crd2idx: (coord_or_index, layout) → index-typed scalar offset.

    Accepts a scalar i32 (auto-converts to int_tuple) or a tuple coordinate.
    """
    # Auto-convert scalar i32 to int_tuple for fx.crd2idx
    if isinstance(crd, ir.Value) and not str(crd.type).startswith("!fly.int_tuple"):
        crd_i32 = crd
        if str(crd.type) != "i32":
            crd_i32 = arith.IndexCastOp(ir.IntegerType.get_signless(32), crd).result
        IntTupleTy, dyncElems = _fly_d.infer_int_tuple_type(crd_i32)
        crd = _fly_d.make_int_tuple(IntTupleTy, dyncElems)
    result = fx.crd2idx(crd, layout)
    scalar = fx.get_scalar(result)
    if isinstance(scalar, ir.Value) and not isinstance(scalar.type, ir.IndexType):
        scalar = arith.IndexCastOp(ir.IndexType.get(), scalar).result
    return scalar


def _idx2crd(idx, layout):
    """CuTE idx2crd: (linear_index, layout) → coordinate tuple.

    Returns the raw fly.int_tuple; use fx.get(result, i) to extract mode i.
    """
    if isinstance(idx, ir.Value) and not str(idx.type).startswith("!fly.int_tuple"):
        idx_i32 = idx
        if str(idx.type) != "i32":
            idx_i32 = arith.IndexCastOp(ir.IntegerType.get_signless(32), idx).result
        IntTupleTy, dyncElems = _fly_d.infer_int_tuple_type(idx_i32)
        idx = _fly_d.make_int_tuple(IntTupleTy, dyncElems)
    return fx.idx2crd(idx, layout)


def _make_reg_offset_vec(base, reg_shape, reg_stride):
    """Build vector<n x i32> of register offsets from CuTE layout pattern.

    offset[i] = base + crd2idx(i, make_layout(reg_shape, reg_stride))

    Example: _make_reg_offset_vec(base, (8, 8), (1, 16))
    → vector<64 x i32> with [base, base+1, ..., base+7, base+16, ..., base+119]
    """
    n = 1
    for s in reg_shape:
        n *= s

    def _flat_offset(i):
        off = 0
        for s, d in zip(reg_shape, reg_stride):
            off += (i % s) * d
            i //= s
        return off

    elems = []
    for i in range(n):
        o = _flat_offset(i)
        elems.append(base + o if o != 0 else base)
    return vector.from_elements(T.vec(n, T.i32), elems)


def _expand_layout(shape, stride):
    """Expand CuTE layout to flat offset list.

    _expand_layout((8, 8, 4), (1, 16, 2048))
    → [0, 1, ..., 7, 16, 17, ..., 23, ..., 6256, ..., 6263]
    """
    n = 1
    for s in shape:
        n *= s
    offsets = []
    for i in range(n):
        off = 0
        idx = i
        for s, d in zip(shape, stride):
            off += (idx % s) * d
            idx //= s
        offsets.append(off)
    return offsets


def _shuffle_repeat_each(src, n_repeat):
    """Repeat each element n_repeat times: [a,b,...] → [a,a,..., b,b,..., ...]"""
    from flydsl._mlir.dialects import vector as _vec_dialect
    sv = src.ir_value() if hasattr(src, "ir_value") else src
    n = sv.type.shape[0]
    mask = []
    for i in range(n):
        mask.extend([i] * n_repeat)
    return _vec_dialect.shuffle(src, src, mask)


def _shuffle_repeat_block(src, n_repeat):
    """Repeat entire vector n_repeat times: [a,b,c] → [a,b,c, a,b,c, ...]"""
    from flydsl._mlir.dialects import vector as _vec_dialect
    sv = src.ir_value() if hasattr(src, "ir_value") else src
    n = sv.type.shape[0]
    mask = list(range(n)) * n_repeat
    return _vec_dialect.shuffle(src, src, mask)


def _shuffle_select_repeat(src, indices, n_repeat):
    """Select elements at given indices, repeat each n_repeat times.

    Example: _shuffle_select_repeat(v, [0, 16], n_repeat=64)
    → [v[0],v[0],...(64×), v[16],v[16],...(64×)]
    """
    from flydsl._mlir.dialects import vector as _vec_dialect
    mask = []
    for idx in indices:
        mask.extend([idx] * n_repeat)
    return _vec_dialect.shuffle(src, src, mask)


def _buffer_load_dot_op_a(memref_ptr, offsets, masks, n_elems, elem_dtype,
                           tile_m=128, tile_k=128, warps_m=4, warps_n=1,
                           stride_m=128, WMMA_M=16, WMMA_N=16, WMMA_K=32,
                           k_width=1):
    """Load A operand (Q) from global memory with dot_op opIdx=0 layout.

    The Triton dot_op<opIdx=0> encoding distributes the 128x128 A tile
    across threads according to the WMMA A-operand layout.
    Each thread loads 128 elements covering specific (M, K) positions.

    offsets is a per-element vector pre-computed by the MLIR lowering:
      offsets[flat_idx] = batch_head_off + M(flat_idx) * stride_m + K(flat_idx)
    where M and K follow the WMMA A-input linear layout (wm*64 + wave_id*16 + lane16).
    We use offsets[flat_idx] directly rather than recomputing M and K.

    k_width: number of contiguous K elements per register group (from dot_op kWidth).
    When k_width > 1, issues vectorized buffer_load instructions (e.g. buffer_load_b128
    for k_width=8 bf16) instead of per-element scalar loads.
    """
    from flydsl._mlir.dialects import vector as _vec_dialect
    rsrc = buffer_ops.create_buffer_resource(memref_ptr)
    elem_ir = elem_dtype() if callable(elem_dtype) else elem_dtype

    warp_tile_m = tile_m // warps_m
    k_wmma_steps = tile_k // WMMA_K
    wmma_m_rep = warp_tile_m // WMMA_M

    # Vectorization width: use k_width contiguous elements per load,
    # capped at hardware max (128-bit buffer load).
    elem_bits = getattr(elem_ir, 'width', 16)
    vec_width = min(k_width, 128 // elem_bits) if k_width > 1 else 1

    # Build result vector
    if isinstance(elem_ir, ir.FloatType):
        zero_attr = ir.FloatAttr.get(elem_ir, 0.0)
    else:
        zero_attr = ir.IntegerAttr.get(elem_ir, 0)
    zero_elem = arith.ConstantOp(elem_ir, zero_attr).result
    result_type = ir.VectorType.get([n_elems], elem_ir)
    result_vec = _vec_dialect.broadcast(result_type, zero_elem)

    for wm in range(wmma_m_rep):
        for ks in range(k_wmma_steps):
            for j in range(0, 16, vec_width):
                flat_idx = wm * (k_wmma_steps * 16) + ks * 16 + j
                # Use offset of first element in contiguous group
                off = _vec_dialect.extract(offsets, [], [flat_idx])
                # Mask from first element (all vec_width share same M position)
                mask_i = None
                if masks is not None:
                    mask_i = _vec_dialect.extract(masks, [], [flat_idx])
                if vec_width > 1:
                    loaded = buffer_ops.buffer_load(rsrc, off, vec_width=vec_width, dtype=elem_ir, mask=mask_i)
                    for vi in range(vec_width):
                        val = _vec_dialect.extract(loaded, [], [vi])
                        result_vec = _vec_dialect.insert(val, result_vec, [], [flat_idx + vi])
                else:
                    val_i = buffer_ops.buffer_load(rsrc, off, vec_width=1, dtype=elem_ir, mask=mask_i)
                    result_vec = _vec_dialect.insert(val_i, result_vec, [], [flat_idx])

    return result_vec


def _buffer_store_vec(data_vec, memref_ptr, offsets, masks, n_elems, elem_dtype,
                       contiguous_size=None):
    """Store per-thread vector to global memory via buffer_store.

    Args:
        data_vec: vector<NxType> data to store
        memref_ptr: memref value (global memory tensor pointer)
        offsets: vector<NxI32> of per-element offsets (in elements)
        masks: vector<NxI1> of per-element masks (or None)
        n_elems: number of elements in the result vector
        elem_dtype: element FlyDSL type class (e.g. T.bf16, T.f32)
        contiguous_size: number of contiguous elements per chunk for vectorized stores.
            When > 1, issues one wide buffer_store per chunk instead of per-element.
    """
    from flydsl._mlir.dialects import vector as _vec_dialect
    from flydsl._mlir.dialects import scf as _scf_d
    rsrc = buffer_ops.create_buffer_resource(memref_ptr)
    elem_ir = elem_dtype() if callable(elem_dtype) else elem_dtype
    if contiguous_size is not None and contiguous_size > 1:
        # Vectorized path: group contiguous elements into wide buffer stores.
        n_chunks = n_elems // contiguous_size
        chunk_ty = ir.VectorType.get([contiguous_size], elem_ir)
        for chunk_i in range(n_chunks):
            base_idx = chunk_i * contiguous_size
            # Build vector chunk from data_vec
            zero_elem = arith.ConstantOp(elem_ir,
                ir.FloatAttr.get(elem_ir, 0.0) if isinstance(elem_ir, ir.FloatType)
                else ir.IntegerAttr.get(elem_ir, 0)).result
            chunk = _vec_dialect.broadcast(chunk_ty, zero_elem)
            for j in range(contiguous_size):
                val_j = _vec_dialect.extract(data_vec, [], [base_idx + j])
                chunk = _vec_dialect.insert(val_j, chunk, [], [j])
            off_0 = _vec_dialect.extract(offsets, [], [base_idx])
            # Mask: all contiguous elements share the same M position
            if masks is not None:
                mask_i = _vec_dialect.extract(masks, [], [base_idx])
                if_op = _scf_d.IfOp(mask_i, [], has_else=False)
                with ir.InsertionPoint(if_op.then_block):
                    buffer_ops.buffer_store(chunk, rsrc, off_0)
                    _scf_d.YieldOp([])
            else:
                buffer_ops.buffer_store(chunk, rsrc, off_0)
    else:
        # Per-element scalar path
        for i in range(n_elems):
            val_i = _vec_dialect.extract(data_vec, [], [i])
            off_i = _vec_dialect.extract(offsets, [], [i])
            if masks is not None:
                mask_i = _vec_dialect.extract(masks, [], [i])
                if_op = _scf_d.IfOp(mask_i, [], has_else=False)
                with ir.InsertionPoint(if_op.then_block):
                    buffer_ops.buffer_store(val_i, rsrc, off_i)
                    _scf_d.YieldOp([])
            else:
                buffer_ops.buffer_store(val_i, rsrc, off_i)



# SmemAllocator: total LDS = 90048 bytes
_lds_allocator = SmemAllocator(None, arch="gfx1250", global_sym_name="kernel_smem")
_lds_allocator.ptr = 90048

_LDS_OFFSET__v36 = 0  # byte offset
_LDS_ELEMS__v36 = 17400  # total elements
_LDS_OFFSET__v39 = 34800  # byte offset
_LDS_ELEMS__v39 = 9208  # total elements
_LDS_OFFSET__v46 = 53216  # byte offset
_LDS_ELEMS__v46 = 18416  # total elements

@flyc.kernel
def attn_fwd_pingpong_pipelined_kernel(q_ptr: fx.Tensor,
                                       k_ptr: fx.Tensor,
                                       v_ptr: fx.Tensor,
                                       out_ptr: fx.Tensor,
                                       stride_qz: fx.Int32,
                                       stride_qh: fx.Int32,
                                       stride_qm: fx.Int32,
                                       stride_kz: fx.Int32,
                                       stride_kh: fx.Int32,
                                       stride_kn: fx.Int32,
                                       stride_vz: fx.Int32,
                                       stride_vh: fx.Int32,
                                       stride_vn: fx.Int32,
                                       stride_oz: fx.Int32,
                                       stride_oh: fx.Int32,
                                       stride_om: fx.Int32):
    stride_qz = stride_qz.ir_value()
    stride_qh = stride_qh.ir_value()
    stride_qm = stride_qm.ir_value()
    stride_kz = stride_kz.ir_value()
    stride_kh = stride_kh.ir_value()
    stride_kn = stride_kn.ir_value()
    stride_vz = stride_vz.ir_value()
    stride_vh = stride_vh.ir_value()
    stride_vn = stride_vn.ir_value()
    stride_oz = stride_oz.ir_value()
    stride_oh = stride_oh.ir_value()
    stride_om = stride_om.ir_value()
    
    tid = arith.index_cast(T.i32, gpu.thread_id("x"))
    
    # Get LDS base memref from SmemAllocator
    _lds_base = _lds_allocator.get_base()
    
    ONES_16 = arith.constant_vector(1.0, T.vec(16, T.f32))
    c2_i32 = arith.constant(2, type=T.i32)
    c192_i32 = arith.constant(192, type=T.i32)
    cst_0 = arith.constant_vector(0.10411755, T.vec(64, T.f32))
    cst_1 = arith.constant_vector(0.10411755, T.vec(2, T.f32))
    NEG_INF_64 = arith.constant_vector(float('-inf'), T.vec(64, T.f32))
    cst_3 = arith.constant_vector(512, T.vec(32, T.i32))
    ZEROS_64 = arith.constant_vector(0.0, T.vec(64, T.f32))
    c1_i32 = arith.constant(1, type=T.i32)
    c0_i32 = arith.constant(0, type=T.i32)
    c320_i32 = arith.constant(320, type=T.i32)
    ZEROS_128 = arith.constant_vector(0.0, T.vec(128, T.f32))
    ONES_2 = arith.constant_vector(1.0, T.vec(2, T.f32))
    NEG_INF_2 = arith.constant_vector(float('-inf'), T.vec(2, T.f32))
    cst_8 = arith.constant_vector(256, T.vec(16, T.i32))
    c64_i32 = arith.constant(64, type=T.i32)
    c1_i64 = arith.constant(1, type=T.i64)
    c128_i32 = arith.constant(128, type=T.i32)
    c512_i32 = arith.constant(512, type=T.i32)
    cst_9 = arith.constant_vector(256, T.vec(32, T.i32))
    cst_10 = arith.constant_vector(128, T.vec(32, T.i32))
    c256_i32 = arith.constant(256, type=T.i32)
    bid_x = arith.index_cast(T.i32, gpu.block_id("x"))
    bid_y = arith.index_cast(T.i32, gpu.block_id("y"))
    bid_z = arith.index_cast(T.i32, gpu.block_id("z"))
    _v3 = bid_z * c256_i32
    _v4 = stride_qz * bid_x
    _v5 = stride_qh * bid_y
    _v6 = _v4 + _v5
    _v8_layout = fx.make_layout((16, 2, 8), (1, 0, 16))
    _v8_base = arith.index_cast(T.i32, _crd2idx(tid % 256, _v8_layout))
    _v8 = vector.from_elements(T.vec(2, T.i32), [_v8_base, _v8_base + 128])
    _v9 = vector.broadcast(T.vec(2, T.i32), _v3)
    _v10 = _v9 + _v8
    _v11 = _shuffle_repeat_each(_v10, n_repeat=16)
    _v12 = vector.broadcast(T.vec(32, T.i32), stride_qm)
    _v13 = _v12 * _v11
    _v14 = vector.broadcast(T.vec(32, T.i32), _v6)
    _v15 = _v14 + _v13
    _v16_layout = fx.make_layout((16, 2, 8), (0, 8, 0))
    _v16_base = arith.index_cast(T.i32, _crd2idx(tid % 256, _v16_layout))
    _v16 = _make_reg_offset_vec(_v16_base, (8, 8), (1, 16))
    _v17 = _shuffle_select_repeat(_v15, [0, 16], n_repeat=64)
    _v18 = _shuffle_repeat_block(_v16, n_repeat=2)
    _v19 = _v17 + _v18
    _v20_layout = fx.make_layout((16, 2, 8), (0, 8, 0))
    _v20_base = arith.index_cast(T.i32, _crd2idx(tid % 256, _v20_layout))
    _v20 = _make_reg_offset_vec(_v20_base, (8, 4), (1, 16))
    _v21 = _v20 + cst_10
    _v22 = _shuffle_select_repeat(_v15, [0, 16], n_repeat=32)
    _v23 = _shuffle_repeat_block(_v21, n_repeat=2)
    _v24 = _v22 + _v23
    _v25 = _v11 < cst_9
    _v26 = _shuffle_select_repeat(_v25, [0, 16], n_repeat=64)
    _v27 = _buffer_load_dot_op_a(q_ptr, _v19, _v26, 128, T.bf16,
                                      tile_m=256, tile_k=128, warps_m=8, warps_n=1, stride_m=stride_qm, k_width=8)
    _v28 = _shuffle_select_repeat(_v25, [0, 16], n_repeat=32)
    _v29 = _buffer_load_dot_op_a(q_ptr, _v24, _v28, 64, T.bf16,
                                      tile_m=256, tile_k=64, warps_m=8, warps_n=1, stride_m=stride_qm, k_width=8)
    _v30 = stride_kz * bid_x
    _v31 = _addptr(k_ptr, _v30, elem_bytes=2)  # addptr
    _v32 = stride_kh * bid_y
    _v33 = _addptr(_v31, _v32, elem_bytes=2)  # addptr
    _v34 = arith.extsi(T.i64, stride_kn)
    _v35 = _make_tdm_desc(_v33,
                          [c512_i32, c128_i32],
                          [_v34, c1_i64],
                          elem_bytes=2,
                          tile_shape=(64, 128),
                          pad_interval=128,
                          pad_amount=8,
                          num_warps=1)
    _v36 = SmemPtr(_lds_base, _LDS_OFFSET__v36, T.bf16, shape=(_LDS_ELEMS__v36,)).get()  # LDS alloc via SmemAllocator
    _v37 = _addptr(_v33, c128_i32, elem_bytes=2)  # addptr
    _v38 = _make_tdm_desc(_v37,
                          [c512_i32, c64_i32],
                          [_v34, c1_i64],
                          elem_bytes=2,
                          tile_shape=(64, 64),
                          pad_interval=64,
                          pad_amount=8,
                          num_warps=1)
    _v39 = SmemPtr(_lds_base, _LDS_OFFSET__v39, T.bf16, shape=(_LDS_ELEMS__v39,)).get()  # LDS alloc via SmemAllocator
    _v40 = stride_vz * bid_x
    _v41 = _addptr(v_ptr, _v40, elem_bytes=2)  # addptr
    _v42 = stride_vh * bid_y
    _v43 = _addptr(_v41, _v42, elem_bytes=2)  # addptr
    _v44 = arith.extsi(T.i64, stride_vn)
    _v45 = _make_tdm_desc(_v43,
                          [c512_i32, c128_i32],
                          [_v44, c1_i64],
                          elem_bytes=2,
                          tile_shape=(64, 128),
                          pad_interval=128,
                          pad_amount=16,
                          num_warps=1)
    _v46 = SmemPtr(_lds_base, _LDS_OFFSET__v46, T.bf16, shape=(_LDS_ELEMS__v46,)).get()  # LDS alloc via SmemAllocator
    _v47 = stride_oz * bid_x
    _v48 = stride_oh * bid_y
    _v49 = _v47 + _v48
    _v50 = vector.shuffle(_v10, _v10, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _v51 = vector.broadcast(T.vec(16, T.i32), stride_om)
    _v52 = _v51 * _v50
    _v53 = vector.broadcast(T.vec(16, T.i32), _v49)
    _v54 = _v53 + _v52
    _v55 = _shuffle_select_repeat(_v54, [0, 8], n_repeat=64)
    _v56 = _v55 + _v18
    _v57 = _v50 < cst_8
    _v58 = _memdesc_index_subview(_v36, c0_i32, 8192, pad_interval=128, pad_amount=8)
    _tdm_copy_to_lds(_v35, (c0_i32, c0_i32), _v58, elem_bytes=2)
    _v60 = _memdesc_index_subview(_v39, c0_i32, 4096, pad_interval=64, pad_amount=8)
    _tdm_copy_to_lds(_v38, (c0_i32, c0_i32), _v60, elem_bytes=2)
    _v62 = _memdesc_index_subview(_v36, c1_i32, 8192, pad_interval=128, pad_amount=8)
    gpu.barrier()

    _tdm_copy_to_lds(_v35, (c64_i32, c0_i32), _v62, elem_bytes=2)
    _v64 = _memdesc_index_subview(_v39, c1_i32, 4096, pad_interval=64, pad_amount=8)
    _tdm_copy_to_lds(_v38, (c64_i32, c0_i32), _v64, elem_bytes=2)
    _v66 = _memdesc_index_subview(_v46, c0_i32, 8192, pad_interval=128, pad_amount=16)
    _tdm_copy_to_lds(_v45, (c0_i32, c0_i32), _v66, elem_bytes=2)
    tdm_ops.tensor_wait(0)
    gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

    _v69 = (_v58, True)  # memdesc_trans → (memref, transposed=True)
    # fly.lds_load with pre-computed LinearLayout offsets (256 elements, vec=8)
    _v70 = _lds_load_gf2(_v69,
                         _expand_layout((8, 8, 4), (1, 16, 2048)),
                         [128, 256, 512, 1024, 8],
                         [0, 0, 0],
                         256,
                         T.bf16,
                         vec_size=8,
                         n_additive=256,
                         pad_interval=128,
                         pad_amount=8)
    tdm_ops.tensor_wait(0)
    gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

    _v72 = (_v60, True)  # memdesc_trans → (memref, transposed=True)
    # fly.lds_load with pre-computed LinearLayout offsets (128 elements, vec=8)
    _v73 = _lds_load_gf2(_v72,
                         _expand_layout((8, 4, 4), (1, 16, 1024)),
                         [64, 128, 256, 512, 8],
                         [0, 0, 0],
                         128,
                         T.bf16,
                         vec_size=8,
                         n_additive=128,
                         pad_interval=64,
                         pad_amount=8)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=8 warps_n=1
    _v74 = _wmma_gemm_full(_v27, _v70, ZEROS_64, T.bf16, tile_m=256, tile_n=64, tile_k=128, warps_m=8, warps_n=1)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=8 warps_n=1
    _v75 = _wmma_gemm_full(_v29, _v73, _v74, T.bf16, tile_m=256, tile_n=64, tile_k=64, warps_m=8, warps_n=1)
    _v76 = _v20 < cst_3
    _v77 = _shuffle_repeat_block(_v76, n_repeat=2)
    _v78 = arith.select(_v77, _v75, NEG_INF_64)
    ZEROS_2 = arith.constant_vector(0.0, T.vec(2, T.f32))
    _v141 = _tree_reduce(_v78, arith.maxnumf, 0, 32)
    _v142 = vector.insert(_v141, ZEROS_2, static_position=[0], dynamic_position=[])
    _v205 = _tree_reduce(_v78, arith.maxnumf, 32, 32)
    _v206 = vector.insert(_v205, _v142, static_position=[1], dynamic_position=[])
    _v207 = vector.extract(_v206, static_position=[0])
    c32_i32 = arith.constant(32, type=T.i32)
    _v208 = tid % c32_i32
    c16_i32 = arith.constant(16, type=T.i32)
    _v209 = _v208 ^ c16_i32
    _v210 = _v209 << c2_i32
    _v211 = arith.bitcast(T.i32, _v207)
    _v212 = rocdl.ds_bpermute(T.i32, _v210, _v211)
    _v213 = arith.bitcast(T.f32, _v212)
    _v214 = arith.maxnumf(_v207, _v213)
    _v215 = vector.insert(_v214, _v206, static_position=[0], dynamic_position=[])
    _v216 = vector.extract(_v215, static_position=[1])
    _v217 = arith.bitcast(T.i32, _v216)
    _v218 = rocdl.ds_bpermute(T.i32, _v210, _v217)
    _v219 = arith.bitcast(T.f32, _v218)
    _v220 = arith.maxnumf(_v216, _v219)
    _v221 = vector.insert(_v220, _v215, static_position=[1], dynamic_position=[])
    _v222 = arith.maxnumf(_v221, NEG_INF_2)
    _v223 = _v222 * cst_1
    _v224 = _v78 * cst_0
    _v225 = vector.shuffle(_v223, _v223, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _v226 = _shuffle_select_repeat(_v225, [0, 8], n_repeat=32)
    _v227 = _v224 - _v226
    _v228 = math.exp2(_v227)
    _v229 = NEG_INF_2 - _v223
    _v230 = math.exp2(_v229)
    _v231 = _memdesc_index_subview(_v46, c1_i32, 8192, pad_interval=128, pad_amount=16)
    _tdm_copy_to_lds(_v45, (c64_i32, c0_i32), _v231, elem_bytes=2)
    _tdm_copy_to_lds(_v35, (c128_i32, c0_i32), _v58, elem_bytes=2)
    gpu.barrier()

    _tdm_copy_to_lds(_v38, (c128_i32, c0_i32), _v60, elem_bytes=2)
    tdm_ops.tensor_wait(0)
    gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

    _v236 = (_v62, True)  # memdesc_trans → (memref, transposed=True)
    # fly.lds_load with pre-computed LinearLayout offsets (256 elements, vec=8)
    _v237 = _lds_load_gf2(_v236,
                          _expand_layout((8, 8, 4), (1, 16, 2048)),
                          [128, 256, 512, 1024, 8],
                          [0, 0, 0],
                          256,
                          T.bf16,
                          vec_size=8,
                          n_additive=256,
                          pad_interval=128,
                          pad_amount=8)
    tdm_ops.tensor_wait(0)
    gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

    _v239 = (_v64, True)  # memdesc_trans → (memref, transposed=True)
    # fly.lds_load with pre-computed LinearLayout offsets (128 elements, vec=8)
    _v240 = _lds_load_gf2(_v239,
                          _expand_layout((8, 4, 4), (1, 16, 1024)),
                          [64, 128, 256, 512, 8],
                          [0, 0, 0],
                          128,
                          T.bf16,
                          vec_size=8,
                          n_additive=128,
                          pad_interval=64,
                          pad_amount=8)
    _for_ctx__v241_8 = _SCFForCtx(c0_i32,
                                  c320_i32,
                                  c64_i32,
                                  [_v222, ONES_2, ZEROS_128, _v237, _v240, _v228, _v230, c0_i32])
    with _for_ctx__v241_8 as (_iv_index, [arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24]):
        arg16 = arith.index_cast(T.i32, _iv_index)
        _v1050 = arg16 + c128_i32
        _v1051 = arg16 + c192_i32
        # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=8 warps_n=1
        _v1052 = _wmma_gemm_full(_v27, arg20, ZEROS_64, T.bf16, tile_m=256, tile_n=64, tile_k=128, warps_m=8, warps_n=1)
        # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=8 warps_n=1
        _v1053 = _wmma_gemm_full(_v29, arg21, _v1052, T.bf16, tile_m=256, tile_n=64, tile_k=64, warps_m=8, warps_n=1)
        _v1054 = _v1053
        _v1034_r0 = _v1050
        _v1034_r1 = _v1051
        _v1034_r2 = _v1054
        tdm_ops.tensor_wait(0)
        gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

        _v1050 = vector.extract(arg22, static_position=[0])
        _v1051 = vector.extract(arg22, static_position=[1])
        _v1052 = vector.extract(arg22, static_position=[2])
        _v1053 = vector.extract(arg22, static_position=[3])
        _v1054 = vector.extract(arg22, static_position=[4])
        _v1055 = vector.extract(arg22, static_position=[5])
        _v1056 = vector.extract(arg22, static_position=[6])
        _v1057 = vector.extract(arg22, static_position=[7])
        _v1058 = vector.extract(arg22, static_position=[8])
        _v1059 = vector.extract(arg22, static_position=[9])
        _v1060 = vector.extract(arg22, static_position=[10])
        _v1061 = vector.extract(arg22, static_position=[11])
        _v1062 = vector.extract(arg22, static_position=[12])
        _v1063 = vector.extract(arg22, static_position=[13])
        _v1064 = vector.extract(arg22, static_position=[14])
        _v1065 = vector.extract(arg22, static_position=[15])
        _v1066 = vector.extract(arg22, static_position=[16])
        _v1067 = vector.extract(arg22, static_position=[17])
        _v1068 = vector.extract(arg22, static_position=[18])
        _v1069 = vector.extract(arg22, static_position=[19])
        _v1070 = vector.extract(arg22, static_position=[20])
        _v1071 = vector.extract(arg22, static_position=[21])
        _v1072 = vector.extract(arg22, static_position=[22])
        _v1073 = vector.extract(arg22, static_position=[23])
        _v1074 = vector.extract(arg22, static_position=[24])
        _v1075 = vector.extract(arg22, static_position=[25])
        _v1076 = vector.extract(arg22, static_position=[26])
        _v1077 = vector.extract(arg22, static_position=[27])
        _v1078 = vector.extract(arg22, static_position=[28])
        _v1079 = vector.extract(arg22, static_position=[29])
        _v1080 = vector.extract(arg22, static_position=[30])
        _v1081 = vector.extract(arg22, static_position=[31])
        _v1082 = _v1050 + _v1051
        _v1083 = _v1052 + _v1053
        _v1084 = _v1054 + _v1055
        _v1085 = _v1056 + _v1057
        _v1086 = _v1058 + _v1059
        _v1087 = _v1060 + _v1061
        _v1088 = _v1062 + _v1063
        _v1089 = _v1064 + _v1065
        _v1090 = _v1066 + _v1067
        _v1091 = _v1068 + _v1069
        _v1092 = _v1070 + _v1071
        _v1093 = _v1072 + _v1073
        _v1094 = _v1074 + _v1075
        _v1095 = _v1076 + _v1077
        _v1096 = _v1078 + _v1079
        _v1097 = _v1080 + _v1081
        _v1098 = _v1082 + _v1083
        _v1099 = _v1084 + _v1085
        _v1100 = _v1086 + _v1087
        _v1101 = _v1088 + _v1089
        _v1102 = _v1090 + _v1091
        _v1103 = _v1092 + _v1093
        _v1104 = _v1094 + _v1095
        _v1105 = _v1096 + _v1097
        _v1106 = _v1098 + _v1099
        _v1107 = _v1100 + _v1101
        _v1108 = _v1102 + _v1103
        _v1109 = _v1104 + _v1105
        _v1110 = _v1106 + _v1107
        _v1111 = _v1108 + _v1109
        _v1112 = _v1110 + _v1111
        _v1113 = vector.insert(_v1112, ZEROS_2, static_position=[0], dynamic_position=[])
        _v1114 = vector.extract(arg22, static_position=[32])
        _v1115 = vector.extract(arg22, static_position=[33])
        _v1116 = vector.extract(arg22, static_position=[34])
        _v1117 = vector.extract(arg22, static_position=[35])
        _v1118 = vector.extract(arg22, static_position=[36])
        _v1119 = vector.extract(arg22, static_position=[37])
        _v1120 = vector.extract(arg22, static_position=[38])
        _v1121 = vector.extract(arg22, static_position=[39])
        _v1122 = vector.extract(arg22, static_position=[40])
        _v1123 = vector.extract(arg22, static_position=[41])
        _v1124 = vector.extract(arg22, static_position=[42])
        _v1125 = vector.extract(arg22, static_position=[43])
        _v1126 = vector.extract(arg22, static_position=[44])
        _v1127 = vector.extract(arg22, static_position=[45])
        _v1128 = vector.extract(arg22, static_position=[46])
        _v1129 = vector.extract(arg22, static_position=[47])
        _v1130 = vector.extract(arg22, static_position=[48])
        _v1131 = vector.extract(arg22, static_position=[49])
        _v1132 = vector.extract(arg22, static_position=[50])
        _v1133 = vector.extract(arg22, static_position=[51])
        _v1134 = vector.extract(arg22, static_position=[52])
        _v1135 = vector.extract(arg22, static_position=[53])
        _v1136 = vector.extract(arg22, static_position=[54])
        _v1137 = vector.extract(arg22, static_position=[55])
        _v1138 = vector.extract(arg22, static_position=[56])
        _v1139 = vector.extract(arg22, static_position=[57])
        _v1140 = vector.extract(arg22, static_position=[58])
        _v1141 = vector.extract(arg22, static_position=[59])
        _v1142 = vector.extract(arg22, static_position=[60])
        _v1143 = vector.extract(arg22, static_position=[61])
        _v1144 = vector.extract(arg22, static_position=[62])
        _v1145 = vector.extract(arg22, static_position=[63])
        _v1146 = _v1114 + _v1115
        _v1147 = _v1116 + _v1117
        _v1148 = _v1118 + _v1119
        _v1149 = _v1120 + _v1121
        _v1150 = _v1122 + _v1123
        _v1151 = _v1124 + _v1125
        _v1152 = _v1126 + _v1127
        _v1153 = _v1128 + _v1129
        _v1154 = _v1130 + _v1131
        _v1155 = _v1132 + _v1133
        _v1156 = _v1134 + _v1135
        _v1157 = _v1136 + _v1137
        _v1158 = _v1138 + _v1139
        _v1159 = _v1140 + _v1141
        _v1160 = _v1142 + _v1143
        _v1161 = _v1144 + _v1145
        _v1162 = _v1146 + _v1147
        _v1163 = _v1148 + _v1149
        _v1164 = _v1150 + _v1151
        _v1165 = _v1152 + _v1153
        _v1166 = _v1154 + _v1155
        _v1167 = _v1156 + _v1157
        _v1168 = _v1158 + _v1159
        _v1169 = _v1160 + _v1161
        _v1170 = _v1162 + _v1163
        _v1171 = _v1164 + _v1165
        _v1172 = _v1166 + _v1167
        _v1173 = _v1168 + _v1169
        _v1174 = _v1170 + _v1171
        _v1175 = _v1172 + _v1173
        _v1176 = _v1174 + _v1175
        _v1177 = vector.insert(_v1176, _v1113, static_position=[1], dynamic_position=[])
        _v1178 = vector.extract(_v1177, static_position=[0])
        _v1179 = arith.bitcast(T.i32, _v1178)
        _v1180 = rocdl.ds_bpermute(T.i32, _v210, _v1179)
        _v1181 = arith.bitcast(T.f32, _v1180)
        _v1182 = _v1178 + _v1181
        _v1183 = vector.insert(_v1182, _v1177, static_position=[0], dynamic_position=[])
        _v1184 = vector.extract(_v1183, static_position=[1])
        _v1185 = arith.bitcast(T.i32, _v1184)
        _v1186 = rocdl.ds_bpermute(T.i32, _v210, _v1185)
        _v1187 = arith.bitcast(T.f32, _v1186)
        _v1188 = _v1184 + _v1187
        _v1189 = vector.insert(_v1188, _v1183, static_position=[1], dynamic_position=[])
        _v1190 = vector.shuffle(arg23, arg23, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        _v1191 = _shuffle_select_repeat(_v1190, [0, 8], n_repeat=64)
        _v1192 = arg19 * _v1191
        _v1193 = _v1192
        _v1194 = arith.truncf(T.vec(64, T.bf16), arg22)
        _v1195 = _v1194
        _v1196 = arg18 * arg23
        _v1197 = _v1196 + _v1189
        _v1198 = _v1197
        _v1199 = arg24 % c2_i32
        _v1200 = _memdesc_index_subview(_v46, _v1199, 8192, pad_interval=128, pad_amount=16)
        # fly.lds_load with pre-computed LinearLayout offsets (256 elements, tr)
        _v1201 = _lds_load_gf2(_v1200,
                               _expand_layout((8, 4, 8), (128, 2048, 16)),
                               [1, 2, 4, 8, 1024],
                               [0, 0, 0],
                               256,
                               T.bf16,
                               transpose_load=True,
                               tr_lane_bases=[128, 256, 512, 8, 1024],
                               pad_interval=128,
                               pad_amount=16)
        _v1202 = _v1201
        _v1203 = arg24 + c1_i32
        _v1204 = _v1203 % c2_i32
        _v1205 = _memdesc_index_subview(_v36, _v1204, 8192, pad_interval=128, pad_amount=8)
        _v1206 = _tdm_copy_to_lds(_v35, (_v1034_r1, c0_i32), _v1205, elem_bytes=2)
        _v1207 = _memdesc_index_subview(_v39, _v1204, 4096, pad_interval=64, pad_amount=8)
        _v1208 = _tdm_copy_to_lds(_v38, (_v1034_r1, c0_i32), _v1207, elem_bytes=2)
        _v1037_r0 = _v1193
        _v1037_r1 = _v1195
        _v1037_r2 = _v1198
        _v1037_r3 = _v1199
        _v1037_r4 = _v1200
        _v1037_r5 = _v1202
        _v1037_r6 = _v1203
        _v1051 = _v1037_r1
        # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=8 warps_n=1
        _v1052 = _wmma_gemm_full(_v1051,
                                 _v1037_r5,
                                 _v1037_r0,
                                 T.bf16,
                                 tile_m=256,
                                 tile_n=128,
                                 tile_k=64,
                                 warps_m=8,
                                 warps_n=1)
        _v1053 = _v1052
        _v1041 = _v1053
        tdm_ops.tensor_wait(0)
        gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

        _v1112 = _tree_reduce(_v1034_r2, arith.maxnumf, 0, 32)
        _v1113 = vector.insert(_v1112, ZEROS_2, static_position=[0], dynamic_position=[])
        _v1176 = _tree_reduce(_v1034_r2, arith.maxnumf, 32, 32)
        _v1177 = vector.insert(_v1176, _v1113, static_position=[1], dynamic_position=[])
        _v1178 = vector.extract(_v1177, static_position=[0])
        _v1179 = arith.bitcast(T.i32, _v1178)
        _v1180 = rocdl.ds_bpermute(T.i32, _v210, _v1179)
        _v1181 = arith.bitcast(T.f32, _v1180)
        _v1182 = arith.maxnumf(_v1178, _v1181)
        _v1183 = vector.insert(_v1182, _v1177, static_position=[0], dynamic_position=[])
        _v1184 = vector.extract(_v1183, static_position=[1])
        _v1185 = arith.bitcast(T.i32, _v1184)
        _v1186 = rocdl.ds_bpermute(T.i32, _v210, _v1185)
        _v1187 = arith.bitcast(T.f32, _v1186)
        _v1188 = arith.maxnumf(_v1184, _v1187)
        _v1189 = vector.insert(_v1188, _v1183, static_position=[1], dynamic_position=[])
        _v1190 = arith.maxnumf(arg17, _v1189)
        _v1191 = _v1190
        _v1192 = _v1190 * cst_1
        _v1193 = _v1034_r2 * cst_0
        _v1194 = vector.shuffle(_v1192, _v1192, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        _v1195 = _shuffle_select_repeat(_v1194, [0, 8], n_repeat=32)
        _v1196 = _v1193 - _v1195
        _v1197 = math.exp2(_v1196)
        _v1198 = _v1197
        _v1199 = arg17 * cst_1
        _v1200 = _v1199 - _v1192
        _v1201 = math.exp2(_v1200)
        _v1202 = _v1201
        _v1203 = _memdesc_index_subview(_v36, _v1037_r3, 8192, pad_interval=128, pad_amount=8)
        _v1204 = (_v1203, True)  # memdesc_trans → (memref, transposed=True)
        # fly.lds_load with pre-computed LinearLayout offsets (256 elements, vec=8)
        _v1205 = _lds_load_gf2(_v1204,
                               _expand_layout((8, 8, 4), (1, 16, 2048)),
                               [128, 256, 512, 1024, 8],
                               [0, 0, 0],
                               256,
                               T.bf16,
                               vec_size=8,
                               n_additive=256,
                               pad_interval=128,
                               pad_amount=8)
        _v1206 = _v1205
        _v1207 = _memdesc_index_subview(_v39, _v1037_r3, 4096, pad_interval=64, pad_amount=8)
        _v1208 = (_v1207, True)  # memdesc_trans → (memref, transposed=True)
        # fly.lds_load with pre-computed LinearLayout offsets (128 elements, vec=8)
        _v1209 = _lds_load_gf2(_v1208,
                               _expand_layout((8, 4, 4), (1, 16, 1024)),
                               [64, 128, 256, 512, 8],
                               [0, 0, 0],
                               128,
                               T.bf16,
                               vec_size=8,
                               n_additive=128,
                               pad_interval=64,
                               pad_amount=8)
        _tdm_copy_to_lds(_v45, (_v1034_r0, c0_i32), _v1037_r4, elem_bytes=2)
        _v1044_r0 = _v1191
        _v1044_r1 = _v1198
        _v1044_r2 = _v1202
        _v1044_r3 = _v1206
        _scf_yield_([_v1044_r0, _v1037_r2, _v1041, _v1044_r3, _v1209, _v1044_r1, _v1044_r2, _v1037_r6])
    _v241_r0 = _for_ctx__v241_8.results[0]
    _v241_r1 = _for_ctx__v241_8.results[1]
    _v241_r2 = _for_ctx__v241_8.results[2]
    _v241_r3 = _for_ctx__v241_8.results[3]
    _v241_r4 = _for_ctx__v241_8.results[4]
    _v241_r5 = _for_ctx__v241_8.results[5]
    _v241_r6 = _for_ctx__v241_8.results[6]
    _v241_r7 = _for_ctx__v241_8.results[7]
    _v242 = _v241_r7 - c1_i32
    _v243 = _v242 * c64_i32
    _v244 = _v243 + c128_i32
    _v245 = _v243 + c192_i32
    _v308 = _tree_reduce(_v241_r5, operator.add, 0, 32)
    _v309 = vector.insert(_v308, ZEROS_2, static_position=[0], dynamic_position=[])
    _v372 = _tree_reduce(_v241_r5, operator.add, 32, 32)
    _v373 = vector.insert(_v372, _v309, static_position=[1], dynamic_position=[])
    _v374 = vector.extract(_v373, static_position=[0])
    _v375 = arith.bitcast(T.i32, _v374)
    _v376 = rocdl.ds_bpermute(T.i32, _v210, _v375)
    _v377 = arith.bitcast(T.f32, _v376)
    _v378 = _v374 + _v377
    _v379 = vector.insert(_v378, _v373, static_position=[0], dynamic_position=[])
    _v380 = vector.extract(_v379, static_position=[1])
    _v381 = arith.bitcast(T.i32, _v380)
    _v382 = rocdl.ds_bpermute(T.i32, _v210, _v381)
    _v383 = arith.bitcast(T.f32, _v382)
    _v384 = _v380 + _v383
    _v385 = vector.insert(_v384, _v379, static_position=[1], dynamic_position=[])
    _v386 = vector.shuffle(_v241_r6, _v241_r6, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _v387 = _shuffle_select_repeat(_v386, [0, 8], n_repeat=64)
    _v388 = _v241_r2 * _v387
    _v389 = arith.truncf(T.vec(64, T.bf16), _v241_r5)
    _v391 = _v241_r1 * _v241_r6
    _v392 = _v391 + _v385
    _v393 = _v241_r7 % c2_i32
    tdm_ops.tensor_wait(0)
    gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

    _v395 = _memdesc_index_subview(_v46, _v393, 8192, pad_interval=128, pad_amount=16)
    # fly.lds_load with pre-computed LinearLayout offsets (256 elements, tr)
    _v396 = _lds_load_gf2(_v395,
                          _expand_layout((8, 4, 8), (128, 2048, 16)),
                          [1, 2, 4, 8, 1024],
                          [0, 0, 0],
                          256,
                          T.bf16,
                          transpose_load=True,
                          tr_lane_bases=[128, 256, 512, 8, 1024],
                          pad_interval=128,
                          pad_amount=16)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=8 warps_n=1
    _v399 = _wmma_gemm_full(_v389, _v396, _v388, T.bf16, tile_m=256, tile_n=128, tile_k=64, warps_m=8, warps_n=1)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=8 warps_n=1
    _v400 = _wmma_gemm_full(_v27, _v241_r3, ZEROS_64, T.bf16, tile_m=256, tile_n=64, tile_k=128, warps_m=8, warps_n=1)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=8 warps_n=1
    _v401 = _wmma_gemm_full(_v29, _v241_r4, _v400, T.bf16, tile_m=256, tile_n=64, tile_k=64, warps_m=8, warps_n=1)
    _v402 = vector.broadcast(T.vec(32, T.i32), _v244)
    _v403 = _v402 + _v20
    _v404 = _v403 < cst_3
    _v405 = _shuffle_repeat_block(_v404, n_repeat=2)
    _v406 = arith.select(_v405, _v401, NEG_INF_64)
    _v469 = _tree_reduce(_v406, arith.maxnumf, 0, 32)
    _v470 = vector.insert(_v469, ZEROS_2, static_position=[0], dynamic_position=[])
    _v533 = _tree_reduce(_v406, arith.maxnumf, 32, 32)
    _v534 = vector.insert(_v533, _v470, static_position=[1], dynamic_position=[])
    _v535 = vector.extract(_v534, static_position=[0])
    _v536 = arith.bitcast(T.i32, _v535)
    _v537 = rocdl.ds_bpermute(T.i32, _v210, _v536)
    _v538 = arith.bitcast(T.f32, _v537)
    _v539 = arith.maxnumf(_v535, _v538)
    _v540 = vector.insert(_v539, _v534, static_position=[0], dynamic_position=[])
    _v541 = vector.extract(_v540, static_position=[1])
    _v542 = arith.bitcast(T.i32, _v541)
    _v543 = rocdl.ds_bpermute(T.i32, _v210, _v542)
    _v544 = arith.bitcast(T.f32, _v543)
    _v545 = arith.maxnumf(_v541, _v544)
    _v546 = vector.insert(_v545, _v540, static_position=[1], dynamic_position=[])
    _v547 = arith.maxnumf(_v241_r0, _v546)
    _v548 = _v547 * cst_1
    _v549 = _v406 * cst_0
    _v550 = vector.shuffle(_v548, _v548, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _v551 = _shuffle_select_repeat(_v550, [0, 8], n_repeat=32)
    _v552 = _v549 - _v551
    _v553 = math.exp2(_v552)
    _v554 = _v241_r0 * cst_1
    _v555 = _v554 - _v548
    _v556 = math.exp2(_v555)
    tdm_ops.tensor_wait(0)
    gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

    _v558 = _memdesc_index_subview(_v36, _v393, 8192, pad_interval=128, pad_amount=8)
    _v559 = (_v558, True)  # memdesc_trans → (memref, transposed=True)
    # fly.lds_load with pre-computed LinearLayout offsets (256 elements, vec=8)
    _v560 = _lds_load_gf2(_v559,
                          _expand_layout((8, 8, 4), (1, 16, 2048)),
                          [128, 256, 512, 1024, 8],
                          [0, 0, 0],
                          256,
                          T.bf16,
                          vec_size=8,
                          n_additive=256,
                          pad_interval=128,
                          pad_amount=8)
    tdm_ops.tensor_wait(0)
    gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

    _v562 = _memdesc_index_subview(_v39, _v393, 4096, pad_interval=64, pad_amount=8)
    _v563 = (_v562, True)  # memdesc_trans → (memref, transposed=True)
    # fly.lds_load with pre-computed LinearLayout offsets (128 elements, vec=8)
    _v564 = _lds_load_gf2(_v563,
                          _expand_layout((8, 4, 4), (1, 16, 1024)),
                          [64, 128, 256, 512, 8],
                          [0, 0, 0],
                          128,
                          T.bf16,
                          vec_size=8,
                          n_additive=128,
                          pad_interval=64,
                          pad_amount=8)
    _tdm_copy_to_lds(_v45, (_v245, c0_i32), _v395, elem_bytes=2)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=8 warps_n=1
    _v566 = _wmma_gemm_full(_v27, _v560, ZEROS_64, T.bf16, tile_m=256, tile_n=64, tile_k=128, warps_m=8, warps_n=1)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=8 warps_n=1
    _v567 = _wmma_gemm_full(_v29, _v564, _v566, T.bf16, tile_m=256, tile_n=64, tile_k=64, warps_m=8, warps_n=1)
    _v568 = vector.broadcast(T.vec(32, T.i32), _v245)
    _v569 = _v568 + _v20
    _v570 = _v569 < cst_3
    _v571 = _shuffle_repeat_block(_v570, n_repeat=2)
    _v572 = arith.select(_v571, _v567, NEG_INF_64)
    _v635 = _tree_reduce(_v553, operator.add, 0, 32)
    _v636 = vector.insert(_v635, ZEROS_2, static_position=[0], dynamic_position=[])
    _v699 = _tree_reduce(_v553, operator.add, 32, 32)
    _v700 = vector.insert(_v699, _v636, static_position=[1], dynamic_position=[])
    _v701 = vector.extract(_v700, static_position=[0])
    _v702 = arith.bitcast(T.i32, _v701)
    _v703 = rocdl.ds_bpermute(T.i32, _v210, _v702)
    _v704 = arith.bitcast(T.f32, _v703)
    _v705 = _v701 + _v704
    _v706 = vector.insert(_v705, _v700, static_position=[0], dynamic_position=[])
    _v707 = vector.extract(_v706, static_position=[1])
    _v708 = arith.bitcast(T.i32, _v707)
    _v709 = rocdl.ds_bpermute(T.i32, _v210, _v708)
    _v710 = arith.bitcast(T.f32, _v709)
    _v711 = _v707 + _v710
    _v712 = vector.insert(_v711, _v706, static_position=[1], dynamic_position=[])
    _v713 = vector.shuffle(_v556, _v556, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _v714 = _shuffle_select_repeat(_v713, [0, 8], n_repeat=64)
    _v715 = _v399 * _v714
    _v716 = arith.truncf(T.vec(64, T.bf16), _v553)
    _v718 = _v392 * _v556
    _v719 = _v718 + _v712
    _v720 = _v241_r7 + c1_i32
    _v721 = _v720 % c2_i32
    tdm_ops.tensor_wait(0)
    gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

    _v723 = _memdesc_index_subview(_v46, _v721, 8192, pad_interval=128, pad_amount=16)
    # fly.lds_load with pre-computed LinearLayout offsets (256 elements, tr)
    _v724 = _lds_load_gf2(_v723,
                          _expand_layout((8, 4, 8), (128, 2048, 16)),
                          [1, 2, 4, 8, 1024],
                          [0, 0, 0],
                          256,
                          T.bf16,
                          transpose_load=True,
                          tr_lane_bases=[128, 256, 512, 8, 1024],
                          pad_interval=128,
                          pad_amount=16)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=8 warps_n=1
    _v727 = _wmma_gemm_full(_v716, _v724, _v715, T.bf16, tile_m=256, tile_n=128, tile_k=64, warps_m=8, warps_n=1)
    _v790 = _tree_reduce(_v572, arith.maxnumf, 0, 32)
    _v791 = vector.insert(_v790, ZEROS_2, static_position=[0], dynamic_position=[])
    _v854 = _tree_reduce(_v572, arith.maxnumf, 32, 32)
    _v855 = vector.insert(_v854, _v791, static_position=[1], dynamic_position=[])
    _v856 = vector.extract(_v855, static_position=[0])
    _v857 = arith.bitcast(T.i32, _v856)
    _v858 = rocdl.ds_bpermute(T.i32, _v210, _v857)
    _v859 = arith.bitcast(T.f32, _v858)
    _v860 = arith.maxnumf(_v856, _v859)
    _v861 = vector.insert(_v860, _v855, static_position=[0], dynamic_position=[])
    _v862 = vector.extract(_v861, static_position=[1])
    _v863 = arith.bitcast(T.i32, _v862)
    _v864 = rocdl.ds_bpermute(T.i32, _v210, _v863)
    _v865 = arith.bitcast(T.f32, _v864)
    _v866 = arith.maxnumf(_v862, _v865)
    _v867 = vector.insert(_v866, _v861, static_position=[1], dynamic_position=[])
    _v868 = arith.maxnumf(_v547, _v867)
    _v869 = _v868 * cst_1
    _v870 = _v572 * cst_0
    _v871 = vector.shuffle(_v869, _v869, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _v872 = _shuffle_select_repeat(_v871, [0, 8], n_repeat=32)
    _v873 = _v870 - _v872
    _v874 = math.exp2(_v873)
    _v875 = _v548 - _v869
    _v876 = math.exp2(_v875)
    _v939 = _tree_reduce(_v874, operator.add, 0, 32)
    _v940 = vector.insert(_v939, ZEROS_2, static_position=[0], dynamic_position=[])
    _v1003 = _tree_reduce(_v874, operator.add, 32, 32)
    _v1004 = vector.insert(_v1003, _v940, static_position=[1], dynamic_position=[])
    _v1005 = vector.extract(_v1004, static_position=[0])
    _v1006 = arith.bitcast(T.i32, _v1005)
    _v1007 = rocdl.ds_bpermute(T.i32, _v210, _v1006)
    _v1008 = arith.bitcast(T.f32, _v1007)
    _v1009 = _v1005 + _v1008
    _v1010 = vector.insert(_v1009, _v1004, static_position=[0], dynamic_position=[])
    _v1011 = vector.extract(_v1010, static_position=[1])
    _v1012 = arith.bitcast(T.i32, _v1011)
    _v1013 = rocdl.ds_bpermute(T.i32, _v210, _v1012)
    _v1014 = arith.bitcast(T.f32, _v1013)
    _v1015 = _v1011 + _v1014
    _v1016 = vector.insert(_v1015, _v1010, static_position=[1], dynamic_position=[])
    _v1017 = vector.shuffle(_v876, _v876, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _v1018 = _shuffle_select_repeat(_v1017, [0, 8], n_repeat=64)
    _v1019 = _v727 * _v1018
    _v1020 = arith.truncf(T.vec(64, T.bf16), _v874)
    _v1022 = _v719 * _v876
    _v1023 = _v1022 + _v1016
    tdm_ops.tensor_wait(0)
    gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

    # fly.lds_load with pre-computed LinearLayout offsets (256 elements, tr)
    _v1025 = _lds_load_gf2(_v395,
                           _expand_layout((8, 4, 8), (128, 2048, 16)),
                           [1, 2, 4, 8, 1024],
                           [0, 0, 0],
                           256,
                           T.bf16,
                           transpose_load=True,
                           tr_lane_bases=[128, 256, 512, 8, 1024],
                           pad_interval=128,
                           pad_amount=16)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=8 warps_n=1
    _v1028 = _wmma_gemm_full(_v1020, _v1025, _v1019, T.bf16, tile_m=256, tile_n=128, tile_k=64, warps_m=8, warps_n=1)
    _v1029 = vector.shuffle(_v1023, _v1023, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _v1030 = ONES_16 / _v1029
    _v1031 = _shuffle_select_repeat(_v1030, [0, 8], n_repeat=64)
    _v1032 = _v1028 * _v1031
    _v1033 = _shuffle_select_repeat(_v57, [0, 8], n_repeat=64)

    _buffer_store_vec(_v1032, out_ptr, _v56, _v1033, 128, T.f32, contiguous_size=4)
    return
