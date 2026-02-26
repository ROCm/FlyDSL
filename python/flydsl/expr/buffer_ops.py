"""AMD Buffer Load/Store Operations - High-level Python API

This module provides high-level Python wrappers for AMD CDNA3/CDNA4 buffer operations.
Buffer operations use a scalar base pointer and per-thread offsets for efficient memory access.

Example:
    >>> from flydsl._mlir_helpers import buffer_ops
    >>> from flydsl._mlir_helpers import arith
    >>> import _mlir.extras.types as T
    >>> 
    >>> # Create buffer resource from memref
    >>> rsrc = buffer_ops.create_buffer_resource(A)
    >>> 
    >>> # Compute offset
    >>> offset = row * arith.index(4096) + col
    >>> 
    >>> # Buffer load (4xf32)
    >>> data = buffer_ops.buffer_load(rsrc, offset, vec_width=4)
    >>> 
    >>> # Buffer store
    >>> buffer_ops.buffer_store(data, rsrc, offset)
"""

from .._mlir import ir
from .._mlir.dialects import llvm, rocdl, arith as std_arith
from .._mlir.extras import types as T
from typing import Optional, Union

__all__ = [
    'create_buffer_resource',
    'buffer_load',
    'buffer_store',
    'BufferResourceDescriptor',
]


def _unwrap_value(value):
    """Recursively unwrap ArithValue or similar wrappers to get the actual MLIR value.
    
    flir's ArithValue can be nested (double-wrapped), so we need to unwrap recursively
    until we get to a real ir.Value (OpResult or BlockArgument).
    """
    max_depth = 10  # Safety limit
    depth = 0
    
    while depth < max_depth and not isinstance(value, ir.Value):
        if hasattr(value, '_value'):
            value = value._value
        elif hasattr(value, 'value'):
            value = value.value
        else:
            break
        depth += 1
    
    return value


def _create_i64_constant(value: int) -> ir.Value:
    """Create i64 constant using standard MLIR arith dialect."""
    i64_type = ir.IntegerType.get_signless(64)
    attr = ir.IntegerAttr.get(i64_type, value)
    op = std_arith.ConstantOp(i64_type, attr)
    return op.result


def _create_i32_constant(value: int) -> ir.Value:
    """Create i32 constant using standard MLIR arith dialect."""
    i32_type = ir.IntegerType.get_signless(32)
    if value > 0x7FFFFFFF:
        value = int(value - 2**32)
    attr = ir.IntegerAttr.get(i32_type, value)
    op = std_arith.ConstantOp(i32_type, attr)
    return _unwrap_value(op.result)


def _create_i16_constant(value: int) -> ir.Value:
    """Create i16 constant using standard MLIR arith dialect."""
    i16_type = ir.IntegerType.get_signless(16)
    attr = ir.IntegerAttr.get(i16_type, value)
    op = std_arith.ConstantOp(i16_type, attr)
    return _unwrap_value(op.result)


def _create_i64_constant(value: int) -> ir.Value:
    """Create i64 constant using standard MLIR arith dialect."""
    i64_type = ir.IntegerType.get_signless(64)
    attr = ir.IntegerAttr.get(i64_type, value)
    op = std_arith.ConstantOp(i64_type, attr)
    return _unwrap_value(op.result)


class BufferResourceDescriptor:
    """AMD Buffer Resource Descriptor
    
    A buffer resource descriptor contains:
    - base_pointer: Scalar base pointer (wave-uniform, stored in SGPRs)
    - stride: Stride for structured buffers (typically 0 for contiguous)
    - num_records: Buffer size in bytes
    - flags: Data format and access flags
    
    The descriptor is stored in a special LLVM pointer type (!llvm.ptr<8>)
    """
    
    def __init__(self, rsrc: ir.Value):
        """Initialize with ROCDL resource descriptor value."""
        self.rsrc = rsrc
    
    @staticmethod
    def from_memref(memref_val: ir.Value, 
                    stride: int = 0, 
                    max_size: bool = True,
                    data_format: str = 'f32',
                    num_records_bytes: Optional[Union[int, ir.Value]] = None) -> 'BufferResourceDescriptor':
        """Create buffer resource descriptor from memref.
        
        Args:
            memref_val: Memref value to create descriptor for
            stride: Stride in elements (0 for contiguous)
            max_size: If True, use max buffer size for flexibility
            num_records_bytes: Override buffer size (in BYTES) used by hardware OOB checking.
                              If provided, this takes precedence over `max_size`.
            data_format: Data format ('f32', 'f16', 'i32', etc.)
            
        Returns:
            BufferResourceDescriptor instance
            
        Example:
            >>> rsrc = BufferResourceDescriptor.from_memref(A)
        """
        # Extract raw pointer from fly.memref.
        raw_val = _unwrap_value(memref_val)
        from .._mlir.dialects import fly as _fly
        ptr_type = ir.Type.parse('!llvm.ptr')
        base_ptr = _fly.extract_aligned_pointer_as_index(raw_val, results=[ptr_type])
        
        # Create buffer resource descriptor
        flags_val = (7 << 12) | (4 << 15)  # data_format=7 (float), num_format=4 (32bit)
        flags = _create_i32_constant(flags_val)
        stride_val = _create_i16_constant(stride)
        
        def _num_records_from_memref_type() -> Optional[int]:
            """Best-effort: derive logical buffer size (in bytes) from static memref type."""
            try:
                mt = ir.MemRefType(_unwrap_value(memref_val).type)
                shape = list(mt.shape)
                if any(int(d) < 0 for d in shape):
                    return None
                # Compute element size in bytes (scalar element type).
                elem_t = mt.element_type
                elem_bits = getattr(elem_t, "width", None)
                if elem_bits is None:
                    return None
                elem_bytes = int(elem_bits) // 8
                if elem_bytes <= 0:
                    return None
                num_elems = 1
                for d in shape:
                    num_elems *= int(d)
                return int(num_elems) * int(elem_bytes)
            except Exception:
                return None

        if num_records_bytes is not None:
            # Caller-provided size in BYTES (preferred for exact hardware OOB behavior).
            if isinstance(num_records_bytes, int):
                nbytes = int(num_records_bytes)
                if nbytes <= 0:
                    nbytes = 0
                # Descriptor uses i32 bytes; clamp to the max representable.
                if nbytes > 0xFFFFFFFF:
                    nbytes = 0xFFFFFFFF
                num_records = _create_i64_constant(nbytes)
            else:
                v = _unwrap_value(num_records_bytes)
                i64_type = ir.IntegerType.get_signless(64)
                if not isinstance(v.type, ir.IntegerType) or v.type.width != 64:
                    op = std_arith.IndexCastOp(i64_type, v)
                    v = _unwrap_value(op.result)
                num_records = v
        elif max_size:
            # Use max for flexibility (hardware will check actual bounds)
            # Note: flir's rocdl.make.buffer.rsrc requires i32, not i64
            num_records = _create_i64_constant(0xFFFFFFFF)  # FALLBACK_MAX_SIZE
        else:
            # Use the logical memref size (in bytes) for hardware OOB checking.
            nbytes = _num_records_from_memref_type()
            if nbytes is None:
                # Fall back to max-size if we can't infer statically.
                num_records = _create_i64_constant(0xFFFFFFFF)
            else:
                if nbytes > 0xFFFFFFFF:
                    nbytes = 0xFFFFFFFF
                num_records = _create_i64_constant(int(nbytes))
        
        # Create resource descriptor (returns !llvm.ptr<8>)
        rsrc_type = ir.Type.parse('!llvm.ptr<8>')
        rsrc = rocdl.MakeBufferRsrcOp(rsrc_type, base_ptr, stride_val, num_records, flags).result
        
        return BufferResourceDescriptor(rsrc)


def create_buffer_resource(memref_val: ir.Value, 
                           stride: int = 0,
                           max_size: bool = True,
                           *,
                           num_records_bytes: Optional[Union[int, ir.Value]] = None) -> ir.Value:
    """Create AMD buffer resource descriptor from memref.
    
    This is a simplified wrapper around BufferResourceDescriptor.from_memref()
    that returns the raw ROCDL resource value.
    
    Args:
        memref_val: Memref value
        stride: Buffer stride (0 for contiguous)
        max_size: Use maximum buffer size
        
    Returns:
        ROCDL buffer resource descriptor (!llvm.ptr<8>)
        
    Example:
        >>> rsrc = create_buffer_resource(A)
        >>> data = buffer_load(rsrc, offset)
    """
    desc = BufferResourceDescriptor.from_memref(
        memref_val, stride, max_size, num_records_bytes=num_records_bytes
    )
    return desc.rsrc


def buffer_load(rsrc: ir.Value,
                offset: ir.Value,
                vec_width: int = 4,
                dtype = None,
                mask: Optional[ir.Value] = None,
                cache_modifier: int = 0,
                soffset_bytes: Optional[Union[int, ir.Value]] = None) -> ir.Value:
    """AMD buffer load operation.
    
    Load data from global memory using buffer descriptor and offset.
    Uses hardware-level bounds checking and vectorization.
    
    Args:
        rsrc: Buffer resource descriptor (!llvm.ptr<8>)
        offset: Offset in elements (i32 type)
        vec_width: Vector width (1, 2, or 4)
        dtype: Element data type (None for f32, or ir.F32Type, etc.)
        mask: Optional mask for predicated load (i1 type)
        cache_modifier: Cache control flags (0 for default)
        soffset_bytes: Optional scalar offset (in BYTES) added by the buffer instruction (soffset).
                      Use this to fold small constant deltas into the instruction instead of emitting
                      extra VGPR address arithmetic.
        
    Returns:
        Loaded data (scalar or vector depending on vec_width)
        
    Example:
        >>> # Load 4xf32
        >>> data = buffer_load(rsrc, offset, vec_width=4)
        >>> 
        >>> # Load with mask
        >>> data = buffer_load(rsrc, offset, vec_width=4, mask=valid)
    """
    # Default dtype to f32
    if dtype is None:
        dtype = ir.F32Type.get()
    
    # Unwrap offset first
    offset = _unwrap_value(offset)
    
    # Convert offset to i32 if needed
    if not isinstance(offset.type, ir.IntegerType) or offset.type.width != 32:
        op = std_arith.IndexCastOp(ir.IntegerType.get_signless(32), offset)
        offset = _unwrap_value(op.result)
    
    # IMPORTANT: Buffer load offset is in BYTES, not elements!
    # For vec4xf32, each element is 4 bytes, so multiply offset by 4
    element_bytes = dtype.width // 8
    bytes_const = _create_i32_constant(element_bytes)
    op = std_arith.MulIOp(offset, bytes_const)
    offset = _unwrap_value(op.result)
    
    # Apply mask by setting invalid offsets to max
    if mask is not None:
        mask = _unwrap_value(mask)
        max_offset = _create_i32_constant(0x7FFFFFFF)
        op = std_arith.SelectOp(mask, offset, max_offset)
        offset = _unwrap_value(op.result)
    
    # Create vector type
    if vec_width == 1:
        result_type = dtype
    else:
        result_type = ir.VectorType.get([vec_width], dtype)
    
    # Create instruction offset and aux flags
    if soffset_bytes is None:
        soffset = _create_i32_constant(0)
    else:
        if isinstance(soffset_bytes, int):
            soffset = _create_i32_constant(soffset_bytes)
        else:
            soffset = _unwrap_value(soffset_bytes)
            if not isinstance(soffset.type, ir.IntegerType) or soffset.type.width != 32:
                op = std_arith.IndexCastOp(ir.IntegerType.get_signless(32), soffset)
                soffset = _unwrap_value(op.result)
    aux_flags = _create_i32_constant(cache_modifier)
    
    # Emit buffer load
    load_op = rocdl.RawPtrBufferLoadOp(
        result_type, 
        rsrc, 
        offset, 
        soffset,   # soffset (scalar byte offset)
        aux_flags  # aux (cache modifiers)
    )
    
    return load_op.result


def buffer_store(data: ir.Value,
                 rsrc: ir.Value,
                 offset: ir.Value,
                 mask: Optional[ir.Value] = None,
                 cache_modifier: int = 0,
                 *,
                 soffset_bytes: Optional[Union[int, ir.Value]] = None,
                 offset_is_bytes: bool = False):
    """AMD buffer store operation.
    
    Store data to global memory using buffer descriptor and offset.
    
    Args:
        data: Data to store (scalar or vector)
        rsrc: Buffer resource descriptor (!llvm.ptr<8>)
        offset: Offset in elements (i32 type)
        mask: Optional mask for predicated store (i1 type)
        cache_modifier: Cache control flags (0 for default)
        
    Example:
        >>> buffer_store(data, rsrc, offset)
        >>> 
        >>> # Store with mask
        >>> buffer_store(data, rsrc, offset, mask=valid)
    """
    # Unwrap all inputs
    data = _unwrap_value(data)
    rsrc = _unwrap_value(rsrc)
    offset = _unwrap_value(offset)
    
    # Convert offset to i32 if needed
    if not isinstance(offset.type, ir.IntegerType) or offset.type.width != 32:
        op = std_arith.IndexCastOp(ir.IntegerType.get_signless(32), offset)
        offset = _unwrap_value(op.result)
    
    # IMPORTANT: RawPtrBufferStoreOp offset is in BYTES.
    # For backward compat, `buffer_store()` accepts element offsets by default
    # and scales them to bytes. Set `offset_is_bytes=True` to skip scaling.
    if not offset_is_bytes:
        # Get element size from data type
        data_type = data.type
        if hasattr(data_type, 'element_type'):  # Vector type
            element_type = data_type.element_type
        else:  # Scalar type
            element_type = data_type
        element_bytes = element_type.width // 8
        bytes_const = _create_i32_constant(element_bytes)
        op = std_arith.MulIOp(offset, bytes_const)
        offset = _unwrap_value(op.result)
    
    # Apply mask by setting invalid offsets to max
    if mask is not None:
        mask = _unwrap_value(mask)
        max_offset = _create_i32_constant(0x7FFFFFFF)
        op = std_arith.SelectOp(mask, offset, max_offset)
        offset = _unwrap_value(op.result)
    
    # Create instruction offset (soffset) and aux flags
    if soffset_bytes is None:
        soffset = _create_i32_constant(0)
    else:
        if isinstance(soffset_bytes, int):
            soffset = _create_i32_constant(int(soffset_bytes))
        else:
            soffset = _unwrap_value(soffset_bytes)
            if not isinstance(soffset.type, ir.IntegerType) or soffset.type.width != 32:
                op = std_arith.IndexCastOp(ir.IntegerType.get_signless(32), soffset)
                soffset = _unwrap_value(op.result)
    aux_flags = _create_i32_constant(cache_modifier)
    
    # Emit buffer store
    rocdl.RawPtrBufferStoreOp(
        data,
        rsrc,
        offset,
        soffset,   # soffset (scalar byte offset)
        aux_flags  # aux (cache modifiers)
    )



