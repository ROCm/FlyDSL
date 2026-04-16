"""
FlyDSL kernel
Warps: 4, Threads/Warp: 32
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

NUM_WARPS = 4
THREADS_PER_WARP = 32
BLOCK_THREADS = 128
TARGET = "hip:gfx1250"
WAVES_PER_EU = 1  # occupancy hint: 1 = max VGPRs, higher = more occupancy

def _set_kernel_attrs(waves_per_eu=WAVES_PER_EU):
    """Set GPU function attributes for register allocation.

    Call from @flyc.jit launcher after kernel() but before launcher.launch().
    """
    ctx = CompilationContext.get_current()
    for op in ctx.gpu_module_body.operations:
        if getattr(op, "OPERATION_NAME", None) == "gpu.func":
            op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32), waves_per_eu
            )


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


def _rocdl_exp2_vec(vec):
    """Element-wise exp2 using native v_exp_f32 (rocdl.exp2).

    Bypasses OCML's __ocml_exp2_f32 which adds unnecessary input
    range clamping (5 ISA instructions per element vs 1).
    """
    from flydsl._mlir.dialects import vector as _vec_dialect
    from flydsl.expr import rocdl
    vt = vec.type if isinstance(vec, ir.Value) else vec.ir_value().type
    n = vt.shape[0]
    result = vec
    for i in range(n):
        elem = _vec_dialect.extract(vec, [], [i])
        exp_elem = rocdl.exp2(T.f32, elem)
        result = _vec_dialect.insert(exp_elem, result, [], [i])
    return result


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

@flyc.kernel(known_block_size=[128, 1, 1])
def attn_fwd_pipelined_kernel(q_ptr: fx.Tensor,
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
                              stride_om: fx.Int32,
                              seqlen_k: fx.Int32):
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
    seqlen_k = seqlen_k.ir_value()
    
    tid = arith.index_cast(T.i32, gpu.thread_id("x"))
    
    # Get LDS base memref from SmemAllocator
    _lds_base = _lds_allocator.get_base()
    
    ONES_16 = arith.constant_vector(1.0, T.vec(16, T.f32))
    c2_i32 = arith.constant(2, type=T.i32)
    c192_i32 = arith.constant(192, type=T.i32)
    cst_0 = arith.constant_vector(0.10411755, T.vec(64, T.f32))
    cst_1 = arith.constant_vector(0.10411755, T.vec(2, T.f32))
    NEG_INF_64 = arith.constant_vector(float('-inf'), T.vec(64, T.f32))
    cst_3 = vector.broadcast(T.vec(32, T.i32), seqlen_k)
    ZEROS_64 = arith.constant_vector(0.0, T.vec(64, T.f32))
    c1_i32 = arith.constant(1, type=T.i32)
    c0_i32 = arith.constant(0, type=T.i32)
    loop_bound = seqlen_k - c192_i32  # SK - 3*BLOCK_N (pipeline depth 3)
    ZEROS_128 = arith.constant_vector(0.0, T.vec(128, T.f32))
    ONES_2 = arith.constant_vector(1.0, T.vec(2, T.f32))
    NEG_INF_2 = arith.constant_vector(float('-inf'), T.vec(2, T.f32))
    cst_8 = arith.constant_vector(128, T.vec(16, T.i32))
    c64_i32 = arith.constant(64, type=T.i32)
    c1_i64 = arith.constant(1, type=T.i64)
    # seqlen_k replaces the former c512_i32 constant
    cst_9 = arith.constant_vector(128, T.vec(32, T.i32))
    c128_i32 = arith.constant(128, type=T.i32)
    cst_12 = arith.constant_vector(16, T.vec(64, T.i32))
    bid_x = arith.index_cast(T.i32, gpu.block_id("x"))
    bid_y = arith.index_cast(T.i32, gpu.block_id("y"))
    bid_z = arith.index_cast(T.i32, gpu.block_id("z"))
    _v3 = bid_z * c128_i32
    _v4 = stride_qz * bid_x
    _v5 = stride_qh * bid_y
    _v6 = _v4 + _v5
    _v8_layout = fx.make_layout((16, 2, 4), (1, 0, 16))
    _v8_base = arith.index_cast(T.i32, _crd2idx(tid % 128, _v8_layout))
    _v8 = vector.from_elements(T.vec(2, T.i32), [_v8_base, _v8_base + 64])
    _v9 = vector.broadcast(T.vec(2, T.i32), _v3)
    _v10 = _v9 + _v8
    _v11 = _shuffle_repeat_each(_v10, n_repeat=16)
    _v12 = vector.broadcast(T.vec(32, T.i32), stride_qm)
    _v13 = _v12 * _v11
    _v14 = vector.broadcast(T.vec(32, T.i32), _v6)
    _v15 = _v14 + _v13
    _v16_layout = fx.make_layout((16, 2, 4), (0, 8, 0))
    _v16_base = arith.index_cast(T.i32, _crd2idx(tid % 128, _v16_layout))
    _v16 = _make_reg_offset_vec(_v16_base, (8, 8), (1, 16))
    _v17 = _shuffle_select_repeat(_v15, [0, 16], n_repeat=64)
    _v18 = _shuffle_repeat_block(_v16, n_repeat=2)
    _v19 = _v17 + _v18
    _v20_layout = fx.make_layout((16, 2, 4), (0, 8, 0))
    _v20_base = arith.index_cast(T.i32, _crd2idx(tid % 128, _v20_layout))
    _v20 = _make_reg_offset_vec(_v20_base, (8, 4), (1, 16))
    _v21 = _v20 + cst_9
    _v22 = _shuffle_select_repeat(_v15, [0, 16], n_repeat=32)
    _v23 = _shuffle_repeat_block(_v21, n_repeat=2)
    _v24 = _v22 + _v23
    _v25 = _v11 < cst_9
    _v26 = _shuffle_select_repeat(_v25, [0, 16], n_repeat=64)
    _v27 = _buffer_load_dot_op_a(q_ptr, _v19, _v26, 128, T.bf16,
                                      tile_m=128, tile_k=128, warps_m=4, warps_n=1, stride_m=stride_qm, k_width=8)
    _v28 = _shuffle_select_repeat(_v25, [0, 16], n_repeat=32)
    _v29 = _buffer_load_dot_op_a(q_ptr, _v24, _v28, 64, T.bf16,
                                      tile_m=128, tile_k=64, warps_m=4, warps_n=1, stride_m=stride_qm, k_width=8)
    _v30 = stride_kz * bid_x
    _v31 = _addptr(k_ptr, _v30, elem_bytes=2)  # addptr
    _v32 = stride_kh * bid_y
    _v33 = _addptr(_v31, _v32, elem_bytes=2)  # addptr
    _v34 = arith.extsi(T.i64, stride_kn)
    _v35 = _make_tdm_desc(_v33,
                          [seqlen_k, c128_i32],
                          [_v34, c1_i64],
                          elem_bytes=2,
                          tile_shape=(64, 128),
                          pad_interval=128,
                          pad_amount=8,
                          num_warps=1)
    _v36 = SmemPtr(_lds_base, _LDS_OFFSET__v36, T.bf16, shape=(_LDS_ELEMS__v36,)).get()  # LDS alloc via SmemAllocator
    _v37 = _addptr(_v33, c128_i32, elem_bytes=2)  # addptr
    _v38 = _make_tdm_desc(_v37,
                          [seqlen_k, c64_i32],
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
                          [seqlen_k, c128_i32],
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
    tdm_ops.tensor_wait(2)
    gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

    _v69 = (_v58, True)  # memdesc_trans → (memref, transposed=True)
    # fly.lds_load with pre-computed LinearLayout offsets (256 elements, vec=8)
    _v70 = _lds_load_gf2(_v69,
                         _expand_layout((8, 8, 4), (1, 16, 2048)),
                         [128, 256, 512, 1024, 8],
                         [0, 0],
                         256,
                         T.bf16,
                         vec_size=8,
                         n_additive=256,
                         pad_interval=128,
                         pad_amount=8)
    _v71 = (_v60, True)  # memdesc_trans → (memref, transposed=True)
    # fly.lds_load with pre-computed LinearLayout offsets (128 elements, vec=8)
    _v72 = _lds_load_gf2(_v71,
                         _expand_layout((8, 4, 4), (1, 16, 1024)),
                         [64, 128, 256, 512, 8],
                         [0, 0],
                         128,
                         T.bf16,
                         vec_size=8,
                         n_additive=128,
                         pad_interval=64,
                         pad_amount=8)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=4 warps_n=1
    _v73 = _wmma_gemm_full(_v27, _v70, ZEROS_64, T.bf16, tile_m=128, tile_n=64, tile_k=128, warps_m=4, warps_n=1)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=4 warps_n=1
    _v74 = _wmma_gemm_full(_v29, _v72, _v73, T.bf16, tile_m=128, tile_n=64, tile_k=64, warps_m=4, warps_n=1)
    _v75 = _v20 < cst_3
    _v76 = _shuffle_repeat_block(_v75, n_repeat=2)
    _v77 = arith.select(_v76, _v74, NEG_INF_64)
    ZEROS_2 = arith.constant_vector(0.0, T.vec(2, T.f32))
    _v140 = _tree_reduce(_v77, arith.maximumf, 0, 32)
    _v141 = vector.insert(_v140, ZEROS_2, static_position=[0], dynamic_position=[])
    _v204 = _tree_reduce(_v77, arith.maximumf, 32, 32)
    _v205 = vector.insert(_v204, _v141, static_position=[1], dynamic_position=[])
    _v206 = vector.extract(_v205, static_position=[0])
    c32_i32 = arith.constant(32, type=T.i32)
    _v207 = tid % c32_i32
    c16_i32 = arith.constant(16, type=T.i32)
    _v208 = _v207 ^ c16_i32
    _v209 = _v208 << c2_i32
    _v210 = arith.bitcast(T.i32, _v206)
    _v211 = rocdl.ds_bpermute(T.i32, _v209, _v210)
    _v212 = arith.bitcast(T.f32, _v211)
    _v213 = arith.maximumf(_v206, _v212)
    _v214 = vector.insert(_v213, _v205, static_position=[0], dynamic_position=[])
    _v215 = vector.extract(_v214, static_position=[1])
    _v216 = arith.bitcast(T.i32, _v215)
    _v217 = rocdl.ds_bpermute(T.i32, _v209, _v216)
    _v218 = arith.bitcast(T.f32, _v217)
    _v219 = arith.maximumf(_v215, _v218)
    _v220 = vector.insert(_v219, _v214, static_position=[1], dynamic_position=[])
    _v221 = arith.maximumf(_v220, NEG_INF_2)
    _v222 = _v221 * cst_1
    _v223 = _v77 * cst_0
    _v224 = vector.shuffle(_v222, _v222, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _v225 = _shuffle_select_repeat(_v224, [0, 8], n_repeat=32)
    _v226 = _v223 - _v225
    _v227 = _rocdl_exp2_vec(_v226)
    _v228 = NEG_INF_2 - _v222
    _v229 = _rocdl_exp2_vec(_v228)
    _v230 = _memdesc_index_subview(_v46, c1_i32, 8192, pad_interval=128, pad_amount=16)
    _tdm_copy_to_lds(_v45, (c64_i32, c0_i32), _v230, elem_bytes=2)
    gpu.barrier()

    _tdm_copy_to_lds(_v35, (c128_i32, c0_i32), _v58, elem_bytes=2)
    _tdm_copy_to_lds(_v38, (c128_i32, c0_i32), _v60, elem_bytes=2)
    tdm_ops.tensor_wait(3)
    gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

    _v235 = (_v62, True)  # memdesc_trans → (memref, transposed=True)
    # fly.lds_load with pre-computed LinearLayout offsets (256 elements, vec=8)
    _v236 = _lds_load_gf2(_v235,
                          _expand_layout((8, 8, 4), (1, 16, 2048)),
                          [128, 256, 512, 1024, 8],
                          [0, 0],
                          256,
                          T.bf16,
                          vec_size=8,
                          n_additive=256,
                          pad_interval=128,
                          pad_amount=8)
    _v237 = (_v64, True)  # memdesc_trans → (memref, transposed=True)
    # fly.lds_load with pre-computed LinearLayout offsets (128 elements, vec=8)
    _v238 = _lds_load_gf2(_v237,
                          _expand_layout((8, 4, 4), (1, 16, 1024)),
                          [64, 128, 256, 512, 8],
                          [0, 0],
                          128,
                          T.bf16,
                          vec_size=8,
                          n_additive=128,
                          pad_interval=64,
                          pad_amount=8)
    _for_ctx__v239_8 = _SCFForCtx(c0_i32,
                                  loop_bound,
                                  c64_i32,
                                  [_v221, ONES_2, ZEROS_128, _v236, _v238, _v227, _v229, c0_i32])
    with _for_ctx__v239_8 as (_iv_index, [arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24]):
        arg16 = arith.index_cast(T.i32, _iv_index)
        _v1040 = arg16 + c128_i32
        _v1041 = arg16 + c192_i32
        # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=4 warps_n=1
        _v1042 = _wmma_gemm_full(_v27, arg20, ZEROS_64, T.bf16, tile_m=128, tile_n=64, tile_k=128, warps_m=4, warps_n=1)
        # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=4 warps_n=1
        _v1043 = _wmma_gemm_full(_v29, arg21, _v1042, T.bf16, tile_m=128, tile_n=64, tile_k=64, warps_m=4, warps_n=1)
        _v1106 = _tree_reduce(arg22, operator.add, 0, 32)
        _v1107 = vector.insert(_v1106, ZEROS_2, static_position=[0], dynamic_position=[])
        _v1170 = _tree_reduce(arg22, operator.add, 32, 32)
        _v1171 = vector.insert(_v1170, _v1107, static_position=[1], dynamic_position=[])
        _v1172 = vector.extract(_v1171, static_position=[0])
        _v1173 = arith.bitcast(T.i32, _v1172)
        _v1174 = rocdl.ds_bpermute(T.i32, _v209, _v1173)
        _v1175 = arith.bitcast(T.f32, _v1174)
        _v1176 = _v1172 + _v1175
        _v1177 = vector.insert(_v1176, _v1171, static_position=[0], dynamic_position=[])
        _v1178 = vector.extract(_v1177, static_position=[1])
        _v1179 = arith.bitcast(T.i32, _v1178)
        _v1180 = rocdl.ds_bpermute(T.i32, _v209, _v1179)
        _v1181 = arith.bitcast(T.f32, _v1180)
        _v1182 = _v1178 + _v1181
        _v1183 = vector.insert(_v1182, _v1177, static_position=[1], dynamic_position=[])
        _v1184 = vector.shuffle(arg23, arg23, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        _v1185 = _shuffle_select_repeat(_v1184, [0, 8], n_repeat=64)
        _v1186 = arg19 * _v1185
        _v1187 = arith.bitcast(T.vec(64, T.i32), arg22)
        _v1188 = _v1187 >> cst_12
        _v1189 = arith.trunci(T.vec(64, T.i16), _v1188)
        _v1190 = arith.bitcast(T.vec(64, T.bf16), _v1189)
        _v1192 = arg18 * arg23
        _v1193 = _v1192 + _v1183
        _v1194 = arg24 % c2_i32
        tdm_ops.tensor_wait(2)
        gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

        _v1196 = _memdesc_index_subview(_v46, _v1194, 8192, pad_interval=128, pad_amount=16)
        # fly.lds_load with pre-computed LinearLayout offsets (256 elements, tr)
        _v1197 = _lds_load_gf2(_v1196,
                               _expand_layout((8, 4, 8), (128, 2048, 16)),
                               [1, 2, 4, 8, 1024],
                               [0, 0],
                               256,
                               T.bf16,
                               transpose_load=True,
                               tr_lane_bases=[128, 256, 512, 8, 1024],
                               pad_interval=128,
                               pad_amount=16)
        _v1198 = arg24 + c1_i32
        _v1199 = _v1198 % c2_i32
        _v1200 = _memdesc_index_subview(_v36, _v1199, 8192, pad_interval=128, pad_amount=8)
        _tdm_copy_to_lds(_v35, (_v1041, c0_i32), _v1200, elem_bytes=2)
        _v1202 = _memdesc_index_subview(_v39, _v1199, 4096, pad_interval=64, pad_amount=8)
        _tdm_copy_to_lds(_v38, (_v1041, c0_i32), _v1202, elem_bytes=2)
        # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=4 warps_n=1
        _v1206 = _wmma_gemm_full(_v1190,
                                 _v1197,
                                 _v1186,
                                 T.bf16,
                                 tile_m=128,
                                 tile_n=128,
                                 tile_k=64,
                                 warps_m=4,
                                 warps_n=1)
        _v1269 = _tree_reduce(_v1043, arith.maximumf, 0, 32)
        _v1270 = vector.insert(_v1269, ZEROS_2, static_position=[0], dynamic_position=[])
        _v1333 = _tree_reduce(_v1043, arith.maximumf, 32, 32)
        _v1334 = vector.insert(_v1333, _v1270, static_position=[1], dynamic_position=[])
        _v1335 = vector.extract(_v1334, static_position=[0])
        _v1336 = arith.bitcast(T.i32, _v1335)
        _v1337 = rocdl.ds_bpermute(T.i32, _v209, _v1336)
        _v1338 = arith.bitcast(T.f32, _v1337)
        _v1339 = arith.maximumf(_v1335, _v1338)
        _v1340 = vector.insert(_v1339, _v1334, static_position=[0], dynamic_position=[])
        _v1341 = vector.extract(_v1340, static_position=[1])
        _v1342 = arith.bitcast(T.i32, _v1341)
        _v1343 = rocdl.ds_bpermute(T.i32, _v209, _v1342)
        _v1344 = arith.bitcast(T.f32, _v1343)
        _v1345 = arith.maximumf(_v1341, _v1344)
        _v1346 = vector.insert(_v1345, _v1340, static_position=[1], dynamic_position=[])
        _v1347 = arith.maximumf(arg17, _v1346)
        _v1348 = _v1347 * cst_1
        _v1349 = _v1043 * cst_0
        _v1350 = vector.shuffle(_v1348, _v1348, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        _v1351 = _shuffle_select_repeat(_v1350, [0, 8], n_repeat=32)
        _v1352 = _v1349 - _v1351
        _v1353 = _rocdl_exp2_vec(_v1352)
        _v1354 = arg17 * cst_1
        _v1355 = _v1354 - _v1348
        _v1356 = _rocdl_exp2_vec(_v1355)
        tdm_ops.tensor_wait(2)
        gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

        _v1358 = _memdesc_index_subview(_v36, _v1194, 8192, pad_interval=128, pad_amount=8)
        _v1359 = (_v1358, True)  # memdesc_trans → (memref, transposed=True)
        # fly.lds_load with pre-computed LinearLayout offsets (256 elements, vec=8)
        _v1360 = _lds_load_gf2(_v1359,
                               _expand_layout((8, 8, 4), (1, 16, 2048)),
                               [128, 256, 512, 1024, 8],
                               [0, 0],
                               256,
                               T.bf16,
                               vec_size=8,
                               n_additive=256,
                               pad_interval=128,
                               pad_amount=8)
        _v1361 = _memdesc_index_subview(_v39, _v1194, 4096, pad_interval=64, pad_amount=8)
        _v1362 = (_v1361, True)  # memdesc_trans → (memref, transposed=True)
        # fly.lds_load with pre-computed LinearLayout offsets (128 elements, vec=8)
        _v1363 = _lds_load_gf2(_v1362,
                               _expand_layout((8, 4, 4), (1, 16, 1024)),
                               [64, 128, 256, 512, 8],
                               [0, 0],
                               128,
                               T.bf16,
                               vec_size=8,
                               n_additive=128,
                               pad_interval=64,
                               pad_amount=8)
        _tdm_copy_to_lds(_v45, (_v1040, c0_i32), _v1196, elem_bytes=2)
        _scf_yield_([_v1347, _v1193, _v1206, _v1360, _v1363, _v1353, _v1356, _v1198])
    _v239_r0 = _for_ctx__v239_8.results[0]
    _v239_r1 = _for_ctx__v239_8.results[1]
    _v239_r2 = _for_ctx__v239_8.results[2]
    _v239_r3 = _for_ctx__v239_8.results[3]
    _v239_r4 = _for_ctx__v239_8.results[4]
    _v239_r5 = _for_ctx__v239_8.results[5]
    _v239_r6 = _for_ctx__v239_8.results[6]
    _v239_r7 = _for_ctx__v239_8.results[7]
    _v240 = _v239_r7 - c1_i32
    _v241 = _v240 * c64_i32
    _v242 = _v241 + c128_i32
    _v243 = _v241 + c192_i32
    _v306 = _tree_reduce(_v239_r5, operator.add, 0, 32)
    _v307 = vector.insert(_v306, ZEROS_2, static_position=[0], dynamic_position=[])
    _v370 = _tree_reduce(_v239_r5, operator.add, 32, 32)
    _v371 = vector.insert(_v370, _v307, static_position=[1], dynamic_position=[])
    _v372 = vector.extract(_v371, static_position=[0])
    _v373 = arith.bitcast(T.i32, _v372)
    _v374 = rocdl.ds_bpermute(T.i32, _v209, _v373)
    _v375 = arith.bitcast(T.f32, _v374)
    _v376 = _v372 + _v375
    _v377 = vector.insert(_v376, _v371, static_position=[0], dynamic_position=[])
    _v378 = vector.extract(_v377, static_position=[1])
    _v379 = arith.bitcast(T.i32, _v378)
    _v380 = rocdl.ds_bpermute(T.i32, _v209, _v379)
    _v381 = arith.bitcast(T.f32, _v380)
    _v382 = _v378 + _v381
    _v383 = vector.insert(_v382, _v377, static_position=[1], dynamic_position=[])
    _v384 = vector.shuffle(_v239_r6, _v239_r6, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _v385 = _shuffle_select_repeat(_v384, [0, 8], n_repeat=64)
    _v386 = _v239_r2 * _v385
    _v387 = arith.bitcast(T.vec(64, T.i32), _v239_r5)
    cst_11 = arith.constant_vector(16, T.vec(64, T.i32))
    _v388 = _v387 >> cst_11
    _v389 = arith.trunci(T.vec(64, T.i16), _v388)
    _v390 = arith.bitcast(T.vec(64, T.bf16), _v389)
    _v392 = _v239_r1 * _v239_r6
    _v393 = _v392 + _v383
    _v394 = _v239_r7 % c2_i32
    tdm_ops.tensor_wait(2)
    gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

    _v396 = _memdesc_index_subview(_v46, _v394, 8192, pad_interval=128, pad_amount=16)
    # fly.lds_load with pre-computed LinearLayout offsets (256 elements, tr)
    _v397 = _lds_load_gf2(_v396,
                          _expand_layout((8, 4, 8), (128, 2048, 16)),
                          [1, 2, 4, 8, 1024],
                          [0, 0],
                          256,
                          T.bf16,
                          transpose_load=True,
                          tr_lane_bases=[128, 256, 512, 8, 1024],
                          pad_interval=128,
                          pad_amount=16)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=4 warps_n=1
    _v400 = _wmma_gemm_full(_v390, _v397, _v386, T.bf16, tile_m=128, tile_n=128, tile_k=64, warps_m=4, warps_n=1)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=4 warps_n=1
    _v401 = _wmma_gemm_full(_v27, _v239_r3, ZEROS_64, T.bf16, tile_m=128, tile_n=64, tile_k=128, warps_m=4, warps_n=1)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=4 warps_n=1
    _v402 = _wmma_gemm_full(_v29, _v239_r4, _v401, T.bf16, tile_m=128, tile_n=64, tile_k=64, warps_m=4, warps_n=1)
    _v403 = vector.broadcast(T.vec(32, T.i32), _v242)
    _v404 = _v403 + _v20
    _v405 = _v404 < cst_3
    _v406 = _shuffle_repeat_block(_v405, n_repeat=2)
    _v407 = arith.select(_v406, _v402, NEG_INF_64)
    _v470 = _tree_reduce(_v407, arith.maximumf, 0, 32)
    _v471 = vector.insert(_v470, ZEROS_2, static_position=[0], dynamic_position=[])
    _v534 = _tree_reduce(_v407, arith.maximumf, 32, 32)
    _v535 = vector.insert(_v534, _v471, static_position=[1], dynamic_position=[])
    _v536 = vector.extract(_v535, static_position=[0])
    _v537 = arith.bitcast(T.i32, _v536)
    _v538 = rocdl.ds_bpermute(T.i32, _v209, _v537)
    _v539 = arith.bitcast(T.f32, _v538)
    _v540 = arith.maximumf(_v536, _v539)
    _v541 = vector.insert(_v540, _v535, static_position=[0], dynamic_position=[])
    _v542 = vector.extract(_v541, static_position=[1])
    _v543 = arith.bitcast(T.i32, _v542)
    _v544 = rocdl.ds_bpermute(T.i32, _v209, _v543)
    _v545 = arith.bitcast(T.f32, _v544)
    _v546 = arith.maximumf(_v542, _v545)
    _v547 = vector.insert(_v546, _v541, static_position=[1], dynamic_position=[])
    _v548 = arith.maximumf(_v239_r0, _v547)
    _v549 = _v548 * cst_1
    _v550 = _v407 * cst_0
    _v551 = vector.shuffle(_v549, _v549, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _v552 = _shuffle_select_repeat(_v551, [0, 8], n_repeat=32)
    _v553 = _v550 - _v552
    _v554 = _rocdl_exp2_vec(_v553)
    _v555 = _v239_r0 * cst_1
    _v556 = _v555 - _v549
    _v557 = _rocdl_exp2_vec(_v556)
    tdm_ops.tensor_wait(1)
    gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

    _v559 = _memdesc_index_subview(_v36, _v394, 8192, pad_interval=128, pad_amount=8)
    _v560 = (_v559, True)  # memdesc_trans → (memref, transposed=True)
    # fly.lds_load with pre-computed LinearLayout offsets (256 elements, vec=8)
    _v561 = _lds_load_gf2(_v560,
                          _expand_layout((8, 8, 4), (1, 16, 2048)),
                          [128, 256, 512, 1024, 8],
                          [0, 0],
                          256,
                          T.bf16,
                          vec_size=8,
                          n_additive=256,
                          pad_interval=128,
                          pad_amount=8)
    _v562 = _memdesc_index_subview(_v39, _v394, 4096, pad_interval=64, pad_amount=8)
    _v563 = (_v562, True)  # memdesc_trans → (memref, transposed=True)
    # fly.lds_load with pre-computed LinearLayout offsets (128 elements, vec=8)
    _v564 = _lds_load_gf2(_v563,
                          _expand_layout((8, 4, 4), (1, 16, 1024)),
                          [64, 128, 256, 512, 8],
                          [0, 0],
                          128,
                          T.bf16,
                          vec_size=8,
                          n_additive=128,
                          pad_interval=64,
                          pad_amount=8)
    _tdm_copy_to_lds(_v45, (_v243, c0_i32), _v396, elem_bytes=2)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=4 warps_n=1
    _v566 = _wmma_gemm_full(_v27, _v561, ZEROS_64, T.bf16, tile_m=128, tile_n=64, tile_k=128, warps_m=4, warps_n=1)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=4 warps_n=1
    _v567 = _wmma_gemm_full(_v29, _v564, _v566, T.bf16, tile_m=128, tile_n=64, tile_k=64, warps_m=4, warps_n=1)
    _v568 = vector.broadcast(T.vec(32, T.i32), _v243)
    _v569 = _v568 + _v20
    _v570 = _v569 < cst_3
    _v571 = _shuffle_repeat_block(_v570, n_repeat=2)
    _v572 = arith.select(_v571, _v567, NEG_INF_64)
    _v635 = _tree_reduce(_v554, operator.add, 0, 32)
    _v636 = vector.insert(_v635, ZEROS_2, static_position=[0], dynamic_position=[])
    _v699 = _tree_reduce(_v554, operator.add, 32, 32)
    _v700 = vector.insert(_v699, _v636, static_position=[1], dynamic_position=[])
    _v701 = vector.extract(_v700, static_position=[0])
    _v702 = arith.bitcast(T.i32, _v701)
    _v703 = rocdl.ds_bpermute(T.i32, _v209, _v702)
    _v704 = arith.bitcast(T.f32, _v703)
    _v705 = _v701 + _v704
    _v706 = vector.insert(_v705, _v700, static_position=[0], dynamic_position=[])
    _v707 = vector.extract(_v706, static_position=[1])
    _v708 = arith.bitcast(T.i32, _v707)
    _v709 = rocdl.ds_bpermute(T.i32, _v209, _v708)
    _v710 = arith.bitcast(T.f32, _v709)
    _v711 = _v707 + _v710
    _v712 = vector.insert(_v711, _v706, static_position=[1], dynamic_position=[])
    _v713 = vector.shuffle(_v557, _v557, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _v714 = _shuffle_select_repeat(_v713, [0, 8], n_repeat=64)
    _v715 = _v400 * _v714
    _v716 = arith.bitcast(T.vec(64, T.i32), _v554)
    _v717 = _v716 >> cst_11
    _v718 = arith.trunci(T.vec(64, T.i16), _v717)
    _v719 = arith.bitcast(T.vec(64, T.bf16), _v718)
    _v721 = _v393 * _v557
    _v722 = _v721 + _v712
    _v723 = _v239_r7 + c1_i32
    _v724 = _v723 % c2_i32
    tdm_ops.tensor_wait(1)
    gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

    _v726 = _memdesc_index_subview(_v46, _v724, 8192, pad_interval=128, pad_amount=16)
    # fly.lds_load with pre-computed LinearLayout offsets (256 elements, tr)
    _v727 = _lds_load_gf2(_v726,
                          _expand_layout((8, 4, 8), (128, 2048, 16)),
                          [1, 2, 4, 8, 1024],
                          [0, 0],
                          256,
                          T.bf16,
                          transpose_load=True,
                          tr_lane_bases=[128, 256, 512, 8, 1024],
                          pad_interval=128,
                          pad_amount=16)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=4 warps_n=1
    _v730 = _wmma_gemm_full(_v719, _v727, _v715, T.bf16, tile_m=128, tile_n=128, tile_k=64, warps_m=4, warps_n=1)
    _v793 = _tree_reduce(_v572, arith.maximumf, 0, 32)
    _v794 = vector.insert(_v793, ZEROS_2, static_position=[0], dynamic_position=[])
    _v857 = _tree_reduce(_v572, arith.maximumf, 32, 32)
    _v858 = vector.insert(_v857, _v794, static_position=[1], dynamic_position=[])
    _v859 = vector.extract(_v858, static_position=[0])
    _v860 = arith.bitcast(T.i32, _v859)
    _v861 = rocdl.ds_bpermute(T.i32, _v209, _v860)
    _v862 = arith.bitcast(T.f32, _v861)
    _v863 = arith.maximumf(_v859, _v862)
    _v864 = vector.insert(_v863, _v858, static_position=[0], dynamic_position=[])
    _v865 = vector.extract(_v864, static_position=[1])
    _v866 = arith.bitcast(T.i32, _v865)
    _v867 = rocdl.ds_bpermute(T.i32, _v209, _v866)
    _v868 = arith.bitcast(T.f32, _v867)
    _v869 = arith.maximumf(_v865, _v868)
    _v870 = vector.insert(_v869, _v864, static_position=[1], dynamic_position=[])
    _v871 = arith.maximumf(_v548, _v870)
    _v872 = _v871 * cst_1
    _v873 = _v572 * cst_0
    _v874 = vector.shuffle(_v872, _v872, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _v875 = _shuffle_select_repeat(_v874, [0, 8], n_repeat=32)
    _v876 = _v873 - _v875
    _v877 = _rocdl_exp2_vec(_v876)
    _v878 = _v549 - _v872
    _v879 = _rocdl_exp2_vec(_v878)
    _v942 = _tree_reduce(_v877, operator.add, 0, 32)
    _v943 = vector.insert(_v942, ZEROS_2, static_position=[0], dynamic_position=[])
    _v1006 = _tree_reduce(_v877, operator.add, 32, 32)
    _v1007 = vector.insert(_v1006, _v943, static_position=[1], dynamic_position=[])
    _v1008 = vector.extract(_v1007, static_position=[0])
    _v1009 = arith.bitcast(T.i32, _v1008)
    _v1010 = rocdl.ds_bpermute(T.i32, _v209, _v1009)
    _v1011 = arith.bitcast(T.f32, _v1010)
    _v1012 = _v1008 + _v1011
    _v1013 = vector.insert(_v1012, _v1007, static_position=[0], dynamic_position=[])
    _v1014 = vector.extract(_v1013, static_position=[1])
    _v1015 = arith.bitcast(T.i32, _v1014)
    _v1016 = rocdl.ds_bpermute(T.i32, _v209, _v1015)
    _v1017 = arith.bitcast(T.f32, _v1016)
    _v1018 = _v1014 + _v1017
    _v1019 = vector.insert(_v1018, _v1013, static_position=[1], dynamic_position=[])
    _v1020 = vector.shuffle(_v879, _v879, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _v1021 = _shuffle_select_repeat(_v1020, [0, 8], n_repeat=64)
    _v1022 = _v730 * _v1021
    _v1023 = arith.bitcast(T.vec(64, T.i32), _v877)
    _v1024 = _v1023 >> cst_11
    _v1025 = arith.trunci(T.vec(64, T.i16), _v1024)
    _v1026 = arith.bitcast(T.vec(64, T.bf16), _v1025)
    _v1028 = _v722 * _v879
    _v1029 = _v1028 + _v1019
    tdm_ops.tensor_wait(0)
    gpu.barrier()  # TDM is async; barrier ensures LDS visible to all waves

    # fly.lds_load with pre-computed LinearLayout offsets (256 elements, tr)
    _v1031 = _lds_load_gf2(_v396,
                           _expand_layout((8, 4, 8), (128, 2048, 16)),
                           [1, 2, 4, 8, 1024],
                           [0, 0],
                           256,
                           T.bf16,
                           transpose_load=True,
                           tr_lane_bases=[128, 256, 512, 8, 1024],
                           pad_interval=128,
                           pad_amount=16)
    # GEMM: wmma_16x16x32 (16x16x32) — tiled WMMA, warps_m=4 warps_n=1
    _v1034 = _wmma_gemm_full(_v1026, _v1031, _v1022, T.bf16, tile_m=128, tile_n=128, tile_k=64, warps_m=4, warps_n=1)
    _v1035 = vector.shuffle(_v1029, _v1029, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _v1036 = ONES_16 / _v1035
    _v1037 = _shuffle_select_repeat(_v1036, [0, 8], n_repeat=64)
    _v1038 = _v1034 * _v1037
    _v1039 = _shuffle_select_repeat(_v57, [0, 8], n_repeat=64)

    _buffer_store_vec(_v1038, out_ptr, _v56, _v1039, 128, T.f32, contiguous_size=4)
    return
