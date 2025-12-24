"""AMD Buffer Load/Store Operations - High-level Python API

This module provides high-level Python wrappers for AMD CDNA3/CDNA4 buffer operations.
Buffer operations use a scalar base pointer and per-thread offsets for efficient memory access.

Example:
    >>> from pyflir.dialects.ext import buffer_ops
    >>> from pyflir.dialects.ext import arith
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

from _mlir import ir
from _mlir.dialects import llvm, rocdl, arith as std_arith, memref
import _mlir.extras.types as T
from typing import Optional, Union

__all__ = [
    'create_llvm_ptr',
    'create_buffer_resource',
    'buffer_load',
    'buffer_store',
    'buffer_load_2d',
    'buffer_store_2d',
    'BufferResourceDescriptor',
    'index_cast_to_i32',
    'i32_mul',
    'i32_add',
    'i32_select',
]


def create_llvm_ptr(value, address_space: int = 0) -> ir.Value:
    """Convert an index value to LLVM pointer.
    
    Args:
        value: Index value (typically from memref.extract_aligned_pointer_as_index)
               Can be ir.Value or ArithValue wrapper
        address_space: LLVM address space (0=generic, 3=LDS, 8=buffer descriptor)
        
    Returns:
        LLVM pointer value
        
    Example:
        >>> ptr_idx = memref.extract_aligned_pointer_as_index(A)
        >>> ptr = create_llvm_ptr(ptr_idx)
    """
    # Extract actual MLIR value from wrapper
    value = _unwrap_value(value)
    
    # Convert index to i64 first (llvm.inttoptr requires signless integer, not index)
    if isinstance(value.type, ir.IndexType):
        i64_type = ir.IntegerType.get_signless(64)
        op = std_arith.IndexCastOp(i64_type, value)
        value = _unwrap_value(op.result)
    
    # Use opaque pointer syntax (LLVM 15+)
    if address_space == 0:
        ptr_type = ir.Type.parse('!llvm.ptr')
    else:
        ptr_type = ir.Type.parse(f'!llvm.ptr<{address_space}>')
    return llvm.IntToPtrOp(ptr_type, value).result


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


def _create_i32_constant(value: int) -> ir.Value:
    """Create i32 constant using standard MLIR arith dialect."""
    i32_type = ir.IntegerType.get_signless(32)
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
                    data_format: str = 'f32') -> 'BufferResourceDescriptor':
        """Create buffer resource descriptor from memref.
        
        Args:
            memref_val: Memref value to create descriptor for
            stride: Stride in elements (0 for contiguous)
            max_size: If True, use max buffer size for flexibility
            data_format: Data format ('f32', 'f16', 'i32', etc.)
            
        Returns:
            BufferResourceDescriptor instance
            
        Example:
            >>> rsrc = BufferResourceDescriptor.from_memref(A)
        """
        # Extract base pointer as index
        extract_op = memref.ExtractAlignedPointerAsIndexOp(memref_val)
        ptr_idx = extract_op.result if hasattr(extract_op, 'result') else extract_op
        
        # Convert to LLVM pointer
        base_ptr = create_llvm_ptr(ptr_idx, address_space=0)
        
        # Create buffer resource descriptor
        flags_val = (7 << 12) | (4 << 15)  # data_format=7 (float), num_format=4 (32bit)
        flags = _create_i32_constant(flags_val)
        stride_val = _create_i16_constant(stride)
        
        if max_size:
            # Use max for flexibility (hardware will check actual bounds)
            # Note: flir's rocdl.make.buffer.rsrc requires i32, not i64
            num_records = _create_i32_constant(0x7FFFFFFE)
        else:
            # TODO: Extract actual size from memref type
            num_records = _create_i32_constant(0x7FFFFFFE)
        
        # Create resource descriptor (returns !llvm.ptr<8>)
        rsrc_type = ir.Type.parse('!llvm.ptr<8>')
        rsrc = rocdl.MakeBufferRsrcOp(rsrc_type, base_ptr, stride_val, num_records, flags).result
        
        return BufferResourceDescriptor(rsrc)


def create_buffer_resource(memref_val: ir.Value, 
                           stride: int = 0,
                           max_size: bool = True) -> ir.Value:
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
    desc = BufferResourceDescriptor.from_memref(memref_val, stride, max_size)
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
                 cache_modifier: int = 0):
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
    
    # IMPORTANT: Buffer store offset is in BYTES, not elements!
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
    
    # Create instruction offset and aux flags
    zero_i32 = _create_i32_constant(0)
    aux_flags = _create_i32_constant(cache_modifier)
    
    # Emit buffer store
    rocdl.RawPtrBufferStoreOp(
        data,
        rsrc,
        offset,
        zero_i32,  # soffset (instruction offset)
        aux_flags  # aux (cache modifiers)
    )


# Convenience functions for common patterns

def buffer_load_f32x4(rsrc: ir.Value, offset: ir.Value, mask: Optional[ir.Value] = None) -> ir.Value:
    """Load vector<4xf32> using buffer operation."""
    return buffer_load(rsrc, offset, vec_width=4, dtype=ir.F32Type.get(), mask=mask)


def buffer_load_f16x4(rsrc: ir.Value, offset: ir.Value, mask: Optional[ir.Value] = None) -> ir.Value:
    """Load vector<4xf16> using buffer operation (stored as 2xi32)."""
    # For f16, we load 4 elements but they're packed into 2xi32
    i32_data = buffer_load(rsrc, offset, vec_width=2, dtype=ir.IntegerType.get_signless(32), mask=mask)
    # TODO: Add bitcast to 4xf16 if needed
    return i32_data


def buffer_store_f32x4(data: ir.Value, rsrc: ir.Value, offset: ir.Value, mask: Optional[ir.Value] = None):
    """Store vector<4xf32> using buffer operation."""
    buffer_store(data, rsrc, offset, mask=mask)


def index_cast_to_i32(value) -> ir.Value:
    """Cast index value to i32.
    
    Args:
        value: Index value (can be ArithValue or ir.Value)
        
    Returns:
        i32 value
        
    Example:
        >>> row_i32 = index_cast_to_i32(row_index)
    """
    value = _unwrap_value(value)
    i32_type = ir.IntegerType.get_signless(32)
    op = std_arith.IndexCastOp(i32_type, value)
    return _unwrap_value(op.result)


def i32_mul(lhs, rhs) -> ir.Value:
    """Multiply two i32 values.
    
    Args:
        lhs, rhs: i32 values (will auto-unwrap if needed)
        
    Returns:
        i32 product
    """
    lhs = _unwrap_value(lhs)
    rhs = _unwrap_value(rhs)
    op = std_arith.MulIOp(lhs, rhs)
    return _unwrap_value(op.result)


def i32_add(lhs, rhs) -> ir.Value:
    """Add two i32 values.
    
    Args:
        lhs, rhs: i32 values (will auto-unwrap if needed)
        
    Returns:
        i32 sum
    """
    lhs = _unwrap_value(lhs)
    rhs = _unwrap_value(rhs)
    op = std_arith.AddIOp(lhs, rhs)
    return _unwrap_value(op.result)


def i32_select(cond, true_val, false_val) -> ir.Value:
    """Select between two i32 values based on condition.
    
    Args:
        cond: i1 condition (will auto-unwrap if needed)
        true_val: Value if cond is true
        false_val: Value if cond is false
        
    Returns:
        Selected value
    """
    cond = _unwrap_value(cond)
    true_val = _unwrap_value(true_val)
    false_val = _unwrap_value(false_val)
    op = std_arith.SelectOp(cond, true_val, false_val)
    return _unwrap_value(op.result)


def buffer_load_2d(rsrc, row, col, stride, vec_width=4, dtype=None, mask=None) -> ir.Value:
    """High-level 2D buffer load with automatic offset calculation.
    
    Args:
        rsrc: Buffer resource descriptor
        row: Row index (index or ArithValue)
        col: Column index (index or ArithValue)
        stride: Row stride (index or ArithValue)
        vec_width: Vector width (1, 2, or 4)
        dtype: Element data type (defaults to f32)
        mask: Optional mask for predicated load
        
    Returns:
        Loaded data (scalar or vector)
        
    Example:
        >>> rsrc = create_buffer_resource(A)
        >>> data = buffer_load_2d(rsrc, row, col, N, vec_width=4)
    """
    # Compute offset: row * stride + col
    row_i32 = index_cast_to_i32(row)
    col_i32 = index_cast_to_i32(col)
    stride_i32 = index_cast_to_i32(stride)
    
    offset = i32_add(i32_mul(row_i32, stride_i32), col_i32)
    
    return buffer_load(rsrc, offset, vec_width, dtype, mask)


def buffer_store_2d(data, rsrc, row, col, stride, mask=None):
    """High-level 2D buffer store with automatic offset calculation.
    
    Args:
        data: Data to store (scalar or vector)
        rsrc: Buffer resource descriptor
        row: Row index (index or ArithValue)
        col: Column index (index or ArithValue)
        stride: Row stride (index or ArithValue)
        mask: Optional mask for predicated store
        
    Example:
        >>> rsrc = create_buffer_resource(B)
        >>> buffer_store_2d(data, rsrc, row, col, M)
    """
    # Compute offset: row * stride + col
    row_i32 = index_cast_to_i32(row)
    col_i32 = index_cast_to_i32(col)
    stride_i32 = index_cast_to_i32(stride)
    
    offset = i32_add(i32_mul(row_i32, stride_i32), col_i32)
    
    buffer_store(data, rsrc, offset, mask)

