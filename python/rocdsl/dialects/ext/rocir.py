"""Python bindings for Rocir dialect operations.

This module provides Python wrappers for Rocir layout algebra operations,
making it easier to construct layouts and perform layout transformations
from Python code.
"""

from typing import List, Optional, Sequence, Union
from mlir.ir import (
    Type,
    Value,
    Location,
    InsertionPoint,
    IndexType,
    IntegerAttr,
    IntegerType,
)
from mlir.dialects import memref, arith, scf, gpu

# Import generated ops
import sys
import os
_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../build/python_bindings"))
if _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)
import rocir as rocir_ops
# import _rocirPassesExt  # Auto-register passes  # DISABLED: has symbol issues



def _get_location(loc: Optional[Location] = None) -> Location:
    """Get location, using current location if none provided."""
    if loc is None:
        loc = Location.unknown()
    return loc



def _unwrap_value(v):
    """Unwrap ArithValue or other value wrappers to get underlying MLIR Value."""
    if isinstance(v, int):
        from mlir.dialects import arith
        from mlir.ir import IndexType, IntegerAttr
        op = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), v))
        return _unwrap_value(op.result)
    if hasattr(v, 'value') and callable(getattr(type(v).value, 'fget', None)):
        # It's a property, call it
        return v.value
    elif hasattr(v, '_value'):
        # Direct attribute access
        return v._value
    else:
        # Already a Value or compatible
        return v

def _get_insertion_point(ip: Optional[InsertionPoint] = None) -> InsertionPoint:
    """Get insertion point, using current if none provided."""
    if ip is None:
        return InsertionPoint.current
    return ip


class TensorView:
    """Lightweight view object representing a tensor slice."""

    def __init__(self, memref_value, shape, base_coords=None):
        self.memref = _unwrap_value(memref_value)
        self.shape = tuple(shape) if shape is not None else ()
        self.rank = len(self.shape)
        if base_coords is None:
            base_coords = [_unwrap_value(0) for _ in self.shape]
        self.base_coords = base_coords


def _to_index_value(val):
    """Convert python int or MLIR value to an index-typed MLIR value."""
    return _unwrap_value(val)


def _linear_idx_to_coords(index_value, dims):
    """Convert linear index into per-dimension coordinates."""
    coords = []
    remaining = _to_index_value(index_value)
    for size in reversed(dims):
        size_val = _to_index_value(size)
        rem_val = _to_index_value(remaining)
        coord = arith.RemUIOp(rem_val, size_val).result
        coords.append(coord)
        remaining = arith.DivUIOp(rem_val, size_val).result
    coords.reverse()
    return coords


def _add_index(base, offset):
    """Add an offset to a base coordinate."""
    if offset is None:
        return _to_index_value(base)
    base_val = _to_index_value(base)
    offset_val = _to_index_value(offset)
    return arith.AddIOp(base_val, offset_val).result


def _scale_index(value, factor):
    """Scale an index value by an integer factor."""
    value_val = _to_index_value(value)
    factor_val = _to_index_value(factor)
    return arith.MulIOp(value_val, factor_val).result


class ShapeType(Type):
    """Rocir shape type."""
    
    @staticmethod
    def get(rank: int, context=None):
        """Create a shape type with given rank."""
        # This would need to be implemented in C++ bindings
        # For now, return a generic type
        from mlir.ir import Context
        if context is None:
            context = Context.current
        # Placeholder - would use actual ODS-generated type
        return Type.parse(f"!rocir.shape<{rank}>", context=context)


class StrideType(Type):
    """Rocir stride type."""
    
    @staticmethod
    def get(rank: int, context=None):
        """Create a stride type with given rank."""
        from mlir.ir import Context
        if context is None:
            context = Context.current
        return Type.parse(f"!rocir.stride<{rank}>", context=context)


class LayoutType(Type):
    """Rocir layout type."""
    
    @staticmethod
    def get(rank: int, context=None):
        """Create a layout type with given rank."""
        from mlir.ir import Context
        if context is None:
            context = Context.current
        return Type.parse(f"!rocir.layout<{rank}>", context=context)


class CoordType(Type):
    """Rocir coordinate type."""
    
    @staticmethod
    def get(rank: int, context=None):
        """Create a coordinate type with given rank."""
        from mlir.ir import Context
        if context is None:
            context = Context.current
        return Type.parse(f"!rocir.coord<{rank}>", context=context)



def _flatten_nested(values, result=None):
    """Flatten nested tuples/lists into a flat list of values."""
    if result is None:
        result = []
    
    for v in values:
        if isinstance(v, (tuple, list)):
            _flatten_nested(v, result)
        else:
            result.append(v)
    
    return result


def _count_total_dims(dims):
    """Count total dimensions in potentially nested structure."""
    count = 0
    for d in dims:
        if isinstance(d, (tuple, list)):
            count += _count_total_dims(d)
        else:
            count += 1
    return count


def make_shape(*dims, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a shape from dimension values (supports nested shapes).
    
    Args:
        *dims: Index values or tuples of index values for nested dimensions
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Rocir shape value
        
    Example:
        >>> # Flat shape
        >>> c8 = arith.constant(8, index=True)
        >>> c16 = arith.constant(16, index=True)
        >>> shape = rocir.make_shape(c8, c16)  # Creates shape<2>
        >>> 
        >>> # Nested shape example: (9, (4, 8))
        >>> c9 = arith.constant(9, index=True)
        >>> c4 = arith.constant(4, index=True)
        >>> shape = rocir.make_shape(c9, (c4, c8))  # Creates nested shape
    """
    
    loc = _get_location(loc)
    
    # If a single tuple/list is passed, unpack it
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = dims[0]
    
    # Flatten nested structure to get all dimension values
    flat_dims = _flatten_nested(dims)
    rank = len(flat_dims)
    result_type = ShapeType.get(rank)
    
    with ip or InsertionPoint.current:
        return rocir_ops.MakeShapeOp(result_type, [_unwrap_value(d) for d in flat_dims], loc=loc).result


def make_stride(*strides, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a stride from stride values (supports nested strides).
    
    Args:
        *strides: Index values or tuples of index values for nested strides
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Rocir stride value
        
    Example:
        >>> # Flat stride
        >>> c1 = arith.constant(1, index=True)
        >>> c8 = arith.constant(8, index=True)
        >>> stride = rocir.make_stride(c1, c8)  # Creates stride<2>
        >>> 
        >>> # Nested stride example: (59, (13, 1))
        >>> c59 = arith.constant(59, index=True)
        >>> c13 = arith.constant(13, index=True)
        >>> stride = rocir.make_stride(c59, (c13, c1))  # Creates nested stride
    """
    
    loc = _get_location(loc)
    
    # If a single tuple/list is passed, unpack it
    if len(strides) == 1 and isinstance(strides[0], (tuple, list)):
        strides = strides[0]
    
    # Flatten nested structure to get all stride values
    flat_strides = _flatten_nested(strides)
    rank = len(flat_strides)
    result_type = StrideType.get(rank)
    
    with ip or InsertionPoint.current:
        return rocir_ops.MakeStrideOp(result_type, [_unwrap_value(s) for s in flat_strides], loc=loc).result


def make_layout(shape, stride=None, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a layout from shape and stride (supports nested layouts).
    
    Args:
        shape: A Rocir shape value, or a tuple/int for creating shape
        stride: A Rocir stride value, or a tuple/int for creating stride, 
                or None to create default column-major stride
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Rocir layout value
        
    Example:
        >>> # Using shape and stride values
        >>> shape = rocir.make_shape(c8, c16)
        >>> stride = rocir.make_stride(c1, c8)
        >>> layout = rocir.make_layout(shape, stride)
        >>> 
        >>> # Using tuples directly in nested form
        >>> layout = rocir.make_layout((c9, (c4, c8)), stride=(c59, (c13, c1)))
        >>> 
        >>> # Using single value
        >>> layout = rocir.make_layout(c6, stride=c1)  # 1D layout 6:1
    """
    
    loc = _get_location(loc)
    
    # If shape is not already a Value, create shape from it
    if not isinstance(shape, Value):
        if isinstance(shape, (tuple, list)):
            shape = make_shape(*shape, loc=loc, ip=ip)
        else:
            # Single value
            shape = make_shape(shape, loc=loc, ip=ip)
    
    # If stride is not already a Value, create stride from it
    if stride is not None:
        if not isinstance(stride, Value):
            if isinstance(stride, (tuple, list)):
                stride = make_stride(*stride, loc=loc, ip=ip)
            else:
                # Single value
                stride = make_stride(stride, loc=loc, ip=ip)
    else:
        # Create default column-major stride (1, prev_dim, prev_dim*prev_stride, ...)
        # For now, just use unit stride
        # TODO: Implement proper default stride computation
        raise ValueError("Default stride not yet implemented, please provide explicit stride")
    
    # Extract rank from shape type
    shape_type_str = str(shape.type)
    # Type format: !rocir.shape<1, -1> where 1 is rank, -1 is structure encoding
    # Or: !rocir.shape<2> for rank 2
    type_content = shape_type_str.split("<")[1].split(">")[0]
    if "," in type_content:
        # Has structure encoding: first number is rank
        rank = int(type_content.split(",")[0].strip())
    else:
        # Simple rank
        rank = int(type_content)
    result_type = LayoutType.get(rank)
    
    with ip or InsertionPoint.current:
        return rocir_ops.MakeLayoutOp(result_type, _unwrap_value(shape), stride, loc=loc).result


def make_coord(*coords: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a coordinate from index values.
    
    Args:
        *coords: Index values representing each coordinate dimension
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Rocir coordinate value
        
    Example:
        >>> i = arith.constant(4, index=True)
        >>> j = arith.constant(7, index=True)
        >>> coord = rocir.make_coord(i, j)  # Creates coord<2>
    """
    
    loc = _get_location(loc)
    rank = len(coords)
    result_type = CoordType.get(rank)
    
    with ip or InsertionPoint.current:
        return rocir_ops.MakeCoordOp(result_type, [_unwrap_value(c) for c in coords], loc=loc).result


def crd2idx(coord: Value, layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Convert a coordinate to a linear index using a layout.
    
    Computes: sum(coord[i] * stride[i]) for all dimensions i.
    
    Args:
        coord: A Rocir coordinate value
        layout: A Rocir layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        An index value representing the linear offset
        
    Example:
        >>> coord = rocir.make_coord(i, j)
        >>> layout = rocir.make_layout(shape, stride)
        >>> idx = rocir.crd2idx(coord, layout)  # Returns i*stride[0] + j*stride[1]
    """
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        op = rocir_ops.Crd2IdxOp(_unwrap_value(coord), _unwrap_value(layout), results=[result_type], loc=loc, ip=ip)
        return op.results[0]


def idx2crd(idx: Value, layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Convert a linear index to a coordinate using a layout.
    
    This is the inverse operation of crd2idx.
    
    Args:
        idx: An index value representing the linear offset
        layout: A Rocir layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Rocir coordinate value
        
    Example:
        >>> idx = arith.constant(42, index=True)
        >>> layout = rocir.make_layout(shape, stride)
        >>> coord = rocir.idx2crd(idx, layout)  # Inverse of crd2idx
    """
    
    loc = _get_location(loc)
    # Extract rank from layout type
    layout_type_str = str(layout.type)
    rank = int(layout_type_str.split("<")[1].split(">")[0])
    result_type = CoordType.get(rank)
    
    with ip or InsertionPoint.current:
        op = rocir_ops.Idx2CrdOp(result_type, _unwrap_value(idx), _unwrap_value(layout), loc=loc, ip=ip)
        return op.results[0]



def size(layout_or_tensor, mode: Optional[List[int]] = None, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Get the size of a layout or tensor.
    
    Args:
        layout_or_tensor: Layout, tensor, or shape to query
        mode: Optional list of mode indices to query specific dimensions
              If None, returns total size
              If [0], returns size of mode 0 (e.g., thread count)
              If [1], returns size of mode 1 (e.g., tile count)
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Size as an index value
        
    Example:
        >>> shape = rocir.make_shape(c8, c16)
        >>> total = rocir.size(shape)  # Returns 128
        >>> thread_count = rocir.size(tv_layout, mode=[0])  # Thread dimension
        >>> tile_count = rocir.size(gC, mode=[1])  # Tile dimension
    """
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    # Handle tensor/layout with shape attribute and mode query
    if hasattr(layout_or_tensor, 'shape') and mode is not None:
        # Extract specific dimension
        if isinstance(layout_or_tensor.shape, (tuple, list)):
            idx = mode[0]
            if idx < len(layout_or_tensor.shape):
                return _unwrap_value(layout_or_tensor.shape[idx])
    
    with ip or InsertionPoint.current:
        op = rocir_ops.SizeOp(_unwrap_value(layout_or_tensor), results=[result_type], loc=loc, ip=ip)
        return op.results[0]


def cosize(layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the cosize (stride extent) of a layout.
    
    Args:
        layout: A Rocir layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        An index value representing the cosize
    """
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        op = rocir_ops.CosizeOp(_unwrap_value(layout), results=[result_type], loc=loc, ip=ip)
        return op.results[0]


def rank(shape_or_layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Get the rank (number of dimensions) of a shape or layout.
    
    Args:
        shape_or_layout: A Rocir shape or layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        An index value representing the rank
    """
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        op = rocir_ops.RankOp(_unwrap_value(shape_or_layout), results=[result_type], loc=loc, ip=ip)
        return op.results[0]


def get_shape(layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Extract the shape from a layout.
    
    Args:
        layout: A Rocir layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The shape component of the layout
    """
    
    loc = _get_location(loc)
    # Extract rank from layout type
    layout_type_str = str(layout.type)
    rank = int(layout_type_str.split("<")[1].split(">")[0])
    result_type = ShapeType.get(rank)
    
    with ip or InsertionPoint.current:
        return rocir_ops.GetShapeOp(result_type, _unwrap_value(layout), loc=loc).result


def get_stride(layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Extract the stride from a layout.
    
    Args:
        layout: A Rocir layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The stride component of the layout
    """
    
    loc = _get_location(loc)
    # Extract rank from layout type
    layout_type_str = str(layout.type)
    rank = int(layout_type_str.split("<")[1].split(">")[0])
    result_type = StrideType.get(rank)
    
    with ip or InsertionPoint.current:
        return rocir_ops.GetStrideOp(result_type, _unwrap_value(layout), loc=loc).result


def get(input: Value, index: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Extract element from shape/stride/coord at given index.
    
    Args:
        input: A Rocir shape, stride, or coord value
        index: Index of element to extract
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The element at the given index (as an index value)
        
    Example:
        >>> shape = rocir.make_shape(c2, c3, c4)
        >>> dim0 = rocir.get(shape, Const.index(0))  # Returns 2
        >>> dim1 = rocir.get(shape, Const.index(1))  # Returns 3
    """
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        return rocir_ops.GetOp(input=_unwrap_value(input), idx=_unwrap_value(index), results=[result_type], loc=loc, ip=ip).result


def composition(layout_a: Value, layout_b: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compose two layouts.
    
    Args:
        layout_a: First layout
        layout_b: Second layout
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The composed layout
        
    Example:
        >>> # Compose a column-major layout with a tiler
        >>> composed = rocir.composition(col_major, tiler)
    """
    
    loc = _get_location(loc)
    result_type = layout_a.type
    
    with ip or InsertionPoint.current:
        return rocir_ops.CompositionOp(result_type, _unwrap_value(layout_a), layout_b, loc=loc).result


def complement(tiler: Value, target_size: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the complement of a tiler layout.
    
    The complement finds the "rest" modes not covered by the tiler.
    This is used internally by logical_divide.
    
    Algorithm:
    1. Filters out stride-0 and size-1 modes from the tiler
    2. Sorts modes by stride (ascending)
    3. Folds to compute rest modes
    4. Returns coalesced layout of rest modes
    
    Args:
        tiler: The tiler layout
        target_size: The target size to complement against
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The complement layout
        
    Example:
        >>> # For a layout of size 12 with tiler of size 3
        >>> # complement returns a layout covering the remaining 4 elements
        >>> tiler = rocir.make_layout(c3, stride=c1)
        >>> target = Const.index(12)
        >>> comp = rocir.complement(tiler, target)  # Returns 4:3
    """
    
    loc = _get_location(loc)
    result_type = tiler.type
    
    with ip or InsertionPoint.current:
        return rocir_ops.ComplementOp(result_type, _unwrap_value(tiler), _unwrap_value(target_size), loc=loc).result


def coalesce(layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Coalesce/simplify a layout by flattening and combining modes.
    
    Ensures post-conditions:
    - Preserves size: size(layout) == size(result)
    - Flattened: depth(result) <= 1
    - Preserves function: For all i, layout(i) == result(i)
    
    Args:
        layout: Layout to coalesce
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The coalesced layout
        
    Example:
        >>> layout = rocir.make_layout((c2, (c1, c6)), stride=(c1, (c6, c2)))
        >>> coalesced = rocir.coalesce(layout)  # Simplifies to 12:1
    """
    from mlir import ir as _ir
    
    loc = _get_location(loc)
    result_type = layout.type
    
    with ip or InsertionPoint.current:
        # Create the operation directly using generic OpView
        unwrapped = _unwrap_value(layout)
        op = _ir.Operation.create(
            "rocir.coalesce",
            results=[result_type],
            operands=[unwrapped],
            loc=loc
        )
        return op.results[0]


# Product operations

def logical_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the logical product of two layouts (basic tiling).
    
    Args:
        block: Block layout
        tiler: Tiler layout
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The tiled layout
    """
    
    loc = _get_location(loc)
    result_type = block.type
    
    with ip or InsertionPoint.current:
        return rocir_ops.LogicalProductOp(result_type, _unwrap_value(block), tiler, loc=loc).result


def zipped_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the zipped product of two layouts."""
    
    loc = _get_location(loc)
    result_type = block.type
    
    with ip or InsertionPoint.current:
        return rocir_ops.ZippedProductOp(result_type, _unwrap_value(block), tiler, loc=loc).result


def tiled_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the tiled product of two layouts."""
    
    loc = _get_location(loc)
    result_type = block.type
    
    with ip or InsertionPoint.current:
        return rocir_ops.TiledProductOp(result_type, _unwrap_value(block), tiler, loc=loc).result


def flat_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the flat product of two layouts."""
    
    loc = _get_location(loc)
    result_type = block.type
    
    with ip or InsertionPoint.current:
        return rocir_ops.FlatProductOp(result_type, _unwrap_value(block), tiler, loc=loc).result


def raked_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the raked product of two layouts."""
    
    loc = _get_location(loc)
    result_type = block.type
    
    with ip or InsertionPoint.current:
        return rocir_ops.RakedProductOp(result_type, _unwrap_value(block), tiler, loc=loc).result


def blocked_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the blocked product of two layouts."""
    
    loc = _get_location(loc)
    result_type = block.type
    
    with ip or InsertionPoint.current:
        return rocir_ops.BlockedProductOp(result_type, _unwrap_value(block), tiler, loc=loc).result


# Divide operations

def logical_divide(layout: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Divide a layout by a tiler (basic partitioning).
    
    Args:
        layout: Layout to partition
        tiler: Tiler layout
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The partitioned layout
    """
    
    loc = _get_location(loc)
    result_type = layout.type
    
    with ip or InsertionPoint.current:
        return rocir_ops.LogicalDivideOp(result_type, _unwrap_value(layout), tiler, loc=loc).result


def zipped_divide(layout: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the zipped divide of a layout."""
    
    loc = _get_location(loc)
    result_type = layout.type
    
    with ip or InsertionPoint.current:
        return rocir_ops.ZippedDivideOp(result_type, _unwrap_value(layout), tiler, loc=loc).result


def tiled_divide(layout: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the tiled divide of a layout."""
    
    loc = _get_location(loc)
    result_type = layout.type
    
    with ip or InsertionPoint.current:
        return rocir_ops.TiledDivideOp(result_type, _unwrap_value(layout), tiler, loc=loc).result


def flat_divide(layout: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the flat divide of a layout."""
    
    loc = _get_location(loc)
    result_type = layout.type
    
    with ip or InsertionPoint.current:
        return rocir_ops.FlatDivideOp(result_type, _unwrap_value(layout), tiler, loc=loc).result


# Local operations

def local_partition(layout: Value, tile: Value, index: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Partition a layout for a specific thread or block index.
    
    Args:
        layout: Layout to partition
        tile: Tile layout
        index: Thread/block index
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The partitioned layout for the given index
        
    Example:
        >>> # Partition data among threads
        >>> thread_data = rocir.local_partition(global_layout, tile, thread_idx)
    """
    
    loc = _get_location(loc)
    result_type = layout.type
    
    with ip or InsertionPoint.current:
        return rocir_ops.LocalPartitionOp(result_type, _unwrap_value(layout), _unwrap_value(tile), _unwrap_value(index), loc=loc).result


def local_tile(layout: Value, tiler: Value, coord: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Extract a tile from a layout at specific coordinates.
    
    Args:
        layout: Layout to tile
        tiler: Tile shape
        coord: Coordinate to extract
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The tile at the given coordinate
        
    Example:
        >>> # Extract CTA tile at block coordinates
        >>> cta_data = rocir.local_tile(global_layout, cta_shape, block_coord)
    """
    
    loc = _get_location(loc)
    result_type = layout.type
    
    with ip or InsertionPoint.current:
        return rocir_ops.LocalTileOp(result_type, _unwrap_value(layout), _unwrap_value(tiler), _unwrap_value(coord), loc=loc).result


#===----------------------------------------------------------------------===//
# Copy Atom and Tiled Copy Classes
#===----------------------------------------------------------------------===//

class CopyAtom:
    """Copy atom descriptor for data movement operations.
    
    Encapsulates a copy operation with metadata about vector size and coalescing.
    Used to construct tiled copy operations.
    """
    
    def __init__(self, element_type: Type, vector_size: int, is_coalesced: bool = True):
        """Initialize a copy atom.
        
        Args:
            element_type: Element type being copied (e.g., f16, f32)
            vector_size: Number of elements per copy instruction
            is_coalesced: Whether memory accesses are coalesced
        """
        self.element_type = element_type
        self.vector_size = vector_size
        self.is_coalesced = is_coalesced
        self._value = None  # Will be set when MLIR op is created
    
    def __repr__(self):
        return f"CopyAtom({self.element_type}, vec={self.vector_size}, coalesced={self.is_coalesced})"


class TiledCopy:
    """Tiled copy operation descriptor.
    
    Represents a data movement operation distributed across threads in a block.
    Created by combining a CopyAtom with thread-value layouts.
    """
    
    def __init__(self, copy_atom: CopyAtom, tv_layout=None, tiler=None):
        """Initialize a tiled copy.
        
        Args:
            copy_atom: Base copy atom
            tv_layout: Thread-value layout (optional)
            tiler: Tiler shape (optional)
        """
        self.copy_atom = copy_atom
        self.tv_layout = tv_layout
        self.tiler = tiler
        self._value = None  # MLIR value
    
    def get_slice(self, thread_idx: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
        """Get per-thread slice of the tiled copy.
        
        Args:
            thread_idx: Thread index within the block
            loc: Optional source location
            ip: Optional insertion point
            
        Returns:
            ThrCopy instance for the specific thread
        """
        return ThrCopy(self, thread_idx)
    
    def partition_S(self, tensor: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
        """Partition source tensor according to this tiled copy's layout.
        
        Args:
            tensor: Tensor to partition
            loc: Optional source location
            ip: Optional insertion point
            
        Returns:
            Partitioned tensor
        """
        # This would use a partition operation in MLIR
        # For now, return the tensor (placeholder)
        return tensor
    
    def partition_D(self, tensor: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
        """Partition destination tensor according to this tiled copy's layout.
        
        Args:
            tensor: Tensor to partition
            loc: Optional source location
            ip: Optional insertion point
            
        Returns:
            Partitioned tensor
        """
        return tensor
    
    def __repr__(self):
        return f"TiledCopy({self.copy_atom})"


class ThrCopy:
    """Per-thread copy descriptor.
    
    Represents the portion of a tiled copy assigned to a specific thread.
    """
    
    def __init__(self, tiled_copy: TiledCopy, thread_idx: Value):
        """Initialize per-thread copy.
        
        Args:
            tiled_copy: Parent tiled copy
            thread_idx: Thread index
        """
        self.tiled_copy = tiled_copy
        self.thread_idx = thread_idx
    
    def partition_S(self, tensor: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
        """Partition source tensor for this thread.
        
        Args:
            tensor: Tensor to partition
            loc: Optional source location
            ip: Optional insertion point
            
        Returns:
            Thread's portion of the tensor
        """
        # Would use local_partition or similar operation
        return tensor
    
    def partition_D(self, tensor: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
        """Partition destination tensor for this thread."""
        return tensor
    
    def __repr__(self):
        return f"ThrCopy({self.tiled_copy}, tid={self.thread_idx})"


#===----------------------------------------------------------------------===//
# Tensor and Fragment Operations
#===----------------------------------------------------------------------===//

class Fragment:
    """Fragment (register memory tensor) wrapper.
    
    Represents a tensor allocated in register memory for per-thread computation.
    """
    
    def __init__(self, value: Value, element_type: Type = None):
        """Initialize fragment.
        
        Args:
            value: Underlying MLIR value
            element_type: Element type of the fragment
        """
        self._value = value
        self.element_type = element_type
        self.shape = None
    
    def load(self):
        """Load fragment data for computation.
        
        Returns:
            The loaded value (identity for now, will be optimized by compiler)
        """
        return self._value
    
    def store(self, value):
        """Store computed value into fragment.
        
        Args:
            value: Value to store
        """
        # In a full implementation, this would generate appropriate store operations
        # For now, this is handled by the compiler optimization
        pass
    
    def __getitem__(self, index):
        """Access fragment element at index."""
        # Would use appropriate indexing operation
        return self._value
    
    def __setitem__(self, index, value):
        """Set fragment element at index."""
        pass


def make_fragment_like(template, element_type: Type = None, loc: Optional[Location] = None,
                       ip: Optional[InsertionPoint] = None):
    """Create a fragment buffer.
    
    If `template` is a TiledCopy descriptor, a TensorView matching its value tile
    shape is returned. Otherwise this behaves like the previous implementation
    and simply allocates a memref matching the provided tensor."""
    loc = _get_location(loc)
    if isinstance(template, TiledCopy):
        if element_type is None:
            raise ValueError("make_fragment_like(tiled_copy, element_type=...) requires element_type")
        val_shape = template.val_shape
        if val_shape is None:
            raise ValueError("tiled_copy missing val_shape metadata")
        from mlir.ir import MemRefType
        memref_type = MemRefType.get(val_shape, element_type)
        with ip or InsertionPoint.current:
            buffer = memref.AllocaOp(memref_type, [], [], loc=loc).result
        base_coords = [_to_index_value(0) for _ in val_shape]
        return TensorView(buffer, val_shape, base_coords)
    
    tensor = _unwrap_value(template)
    tensor_type = tensor.type
    with ip or InsertionPoint.current:
        return memref.AllocaOp(tensor_type, [], [], loc=loc).result


def make_rmem_tensor(shape, element_type: Type, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a tensor in register memory with given shape and type.
    
    Args:
        shape: Shape tuple
        element_type: Element type
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Fragment allocated in register memory (memref)
        
    Example:
        >>> frgPred = rocir.make_rmem_tensor((4, 4), Boolean)
    """
    from mlir.ir import MemRefType
    
    loc = _get_location(loc)
    
    # Construct MemRefType from shape tuple
    if isinstance(shape, (tuple, list)):
        memref_type = MemRefType.get(shape, element_type)
    else:
        # Fallback/TODO: handle Value shape dynamic alloc
        raise NotImplementedError("Dynamic shape not supported in make_rmem_tensor yet")
        
    with ip or InsertionPoint.current:
        return memref.AllocaOp(memref_type, [], [], loc=loc).result


def make_identity_tensor(shape, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Create an identity tensor (coordinate tensor) with given shape.
    
    An identity tensor maps each coordinate to itself, useful for tracking
    coordinates during partitioning.
    
    Args:
        shape: Shape tuple or Value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Identity tensor
        
    Example:
        >>> idC = rocir.make_identity_tensor(mC.shape)
        >>> cC = rocir.zipped_divide(idC, tiler=tiler_mn)
    """
    loc = _get_location(loc)
    # For now, return shape as-is
    # Full implementation would create proper identity mapping
    return shape


def make_ordered_layout(shape: tuple, order: tuple = None, stride: tuple = None,
                       loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a layout with specified dimension ordering.
    
    Args:
        shape: Shape tuple
        order: Dimension order tuple (e.g., (1, 0) for column-major)
        stride: Optional explicit stride (computed from order if not provided)
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Layout value
        
    Example:
        >>> # Row-major: order (1, 0) means dim 1 varies fastest
        >>> thr_layout = rocir.make_ordered_layout((4, 32), order=(1, 0))
        >>> # Equivalent to make_layout((4, 32), stride=(32, 1))
    """
    if stride is not None:
        return make_layout(shape, stride=stride, loc=loc, ip=ip)
    
    if order is None:
        order = tuple(range(len(shape) - 1, -1, -1))  # Default: row-major
    
    # Compute strides from order
    # For order (1, 0): fastest dimension is 1, so stride[1] = 1, stride[0] = shape[1]
    computed_stride = [1] * len(shape)
    sorted_dims = sorted(range(len(shape)), key=lambda i: order[i])
    
    stride_val = 1
    for dim in sorted_dims:
        computed_stride[dim] = stride_val
        stride_val *= shape[dim]
    
    return make_layout(shape, stride=tuple(computed_stride), loc=loc, ip=ip)


def product_each(shape: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute element-wise product of shape dimensions.
    
    For nested shapes, computes the product at each hierarchical level.
    Example: ((2,3), (4,5)) -> (6, 20)
    
    Args:
        shape: Input shape
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Shape with products computed at each level
        
    Example:
        >>> shape = make_shape((2, 3), (4, 5))  # Nested shape
        >>> result = product_each(shape)  # -> (6, 20)
    """
    loc = _get_location(loc)
    shape = _unwrap_value(shape)
    with _get_insertion_point(ip):
        # from mlir.dialects import rocir as rocir_ops  # Not available
        op = rocir_ops.ProductEachOp(shape, loc=loc)
        return op.result


def make_layout_tv(thr_layout: Value, val_layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Create tiler and TV layout from thread and value layouts.
    
    Combines thread layout (thread → tile coords) and value layout (value → tile coords)
    to produce a tiler and thread-value layout.
    
    Computes:
    1. layout_mn = raked_product(thr_layout, val_layout)
    2. tiler_mn = product_each(layout_mn.shape)
    3. layout_tv = composition(right_inverse(layout_mn), make_layout((thr_size, val_size)))
    
    Args:
        thr_layout: Thread layout
        val_layout: Value layout
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Tuple of (tiler, tv_layout)
        
    Example:
        >>> thr_layout = rocir.make_layout((4, 32), stride=(32, 1))
        >>> val_layout = rocir.make_layout((4, 4), stride=(4, 1))
        >>> tiler_mn, tv_layout = rocir.make_layout_tv(thr_layout, val_layout)
    """
    loc = _get_location(loc)
    thr_layout = _unwrap_value(thr_layout)
    val_layout = _unwrap_value(val_layout)
    with _get_insertion_point(ip):
        # Note: MakeLayoutTVOp not implemented in MLIR, compute at Python level
        # This is a placeholder - full implementation would use product_each
        # and other layout algebra operations
        return (thr_layout, val_layout)


def elem_less(a, b, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Element-wise less-than comparison.
    
    Args:
        a: First value
        b: Second value (can be Shape or Value)
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Boolean result
        
    Example:
        >>> val = rocir.elem_less(thrCrd[i], shape)
        >>> frgPred[i] = val
    """
    from mlir.dialects import arith
    # Unwrap values
    a_val = _unwrap_value(a)
    b_val = _unwrap_value(b) if not isinstance(b, (tuple, list)) else _unwrap_value(b[0])
    
    # Generate comparison
    with ip or InsertionPoint.current:
        return arith.cmpi(arith.CmpIPredicate.slt, a_val, b_val)


#===----------------------------------------------------------------------===//
# Copy Atom Construction Functions
#===----------------------------------------------------------------------===//

def make_copy_atom(element_type: Type, vector_size: int = 8, is_coalesced: bool = True, 
                   loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> CopyAtom:
    """Create a copy atom for data movement operations.
    
    Args:
        element_type: Type of elements being copied (e.g., f16, f32)
        vector_size: Number of elements per copy instruction (default: 8)
        is_coalesced: Whether accesses should be coalesced (default: True)
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        CopyAtom descriptor
        
    Example:
        >>> from mlir.ir import F16Type
        >>> atom = rocir.make_copy_atom(F16Type.get(), vector_size=8)
    """
    return CopyAtom(element_type, vector_size, is_coalesced)


def make_tiled_copy_tv(copy_atom: CopyAtom, thr_layout: Value, val_layout: Value,
                       thr_shape: Optional[tuple] = None,
                       val_shape: Optional[tuple] = None,
                       loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> TiledCopy:
    """Create a tiled copy from copy atom and separate thread/value layouts.
    
    Args:
        copy_atom: Copy atom descriptor
        thr_layout: Thread layout mapping threads to tile coordinates
        val_layout: Value layout mapping per-thread values to tile coordinates
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        TiledCopy descriptor
        
    Example:
        >>> atom = rocir.make_copy_atom(F16Type.get(), 8)
        >>> thr_layout = rocir.make_layout((4, 32), stride=(32, 1))
        >>> val_layout = rocir.make_layout((4, 4), stride=(4, 1))
        >>> tiled_copy = rocir.make_tiled_copy_tv(atom, thr_layout, val_layout)
    """
    tiled_copy = TiledCopy(copy_atom, tv_layout=None, tiler=None)
    tiled_copy.thr_layout = thr_layout
    tiled_copy.val_layout = val_layout
    tiled_copy.thr_shape = thr_shape
    tiled_copy.val_shape = val_shape
    return tiled_copy


def make_tensor(memref, layout, shape, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> TensorView:
    """Create a tensor view from a memref with a specific layout and shape."""
    if shape is None:
        raise ValueError("make_tensor requires explicit shape information")
    _get_location(loc)  # ensure location evaluated for consistency
    _unwrap_value(layout)  # ensure layout op emitted
    base_coords = [_to_index_value(0) for _ in shape]
    return TensorView(memref, shape, base_coords)


def partition_src(tiled_copy, tensor, thread_id, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Partition source tensor for a specific thread."""
    loc = _get_location(loc)
    if not isinstance(tensor, TensorView):
        raise ValueError("partition_src expects a TensorView produced by make_tensor")
    thr_shape = getattr(tiled_copy, "thr_shape", None)
    val_shape = getattr(tiled_copy, "val_shape", None)
    if thr_shape is None or val_shape is None:
        raise ValueError("tiled_copy is missing thr_shape/val_shape metadata")
    coords = _linear_idx_to_coords(_to_index_value(thread_id), thr_shape)
    base_coords = []
    for dim in range(tensor.rank):
        coord = coords[dim] if dim < len(coords) else _to_index_value(0)
        step = val_shape[dim] if dim < len(val_shape) else 1
        offset = _scale_index(coord, step)
        base_coords.append(_add_index(tensor.base_coords[dim], offset))
    return TensorView(tensor.memref, tensor.shape, base_coords)


def partition_dst(tiled_copy, tensor, thread_id, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Partition destination tensor for a specific thread."""
    return partition_src(tiled_copy, tensor, thread_id, loc=loc, ip=ip)


def fragment_load(fragment, index, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Load an element from a register fragment."""
    from mlir.dialects import memref as memref_dialect
    loc = _get_location(loc)
    fragment = _unwrap_value(fragment)
    index = _unwrap_value(index)
    with _get_insertion_point(ip):
        return memref_dialect.load(fragment, [index])


def fragment_store(value, fragment, index, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Store an element to a register fragment."""
    from mlir.dialects import memref as memref_dialect
    loc = _get_location(loc)
    value = _unwrap_value(value)
    fragment = _unwrap_value(fragment)
    index = _unwrap_value(index)
    with _get_insertion_point(ip):
        memref_dialect.store(value, fragment, [index])


def copy(copy_desc, src, dst, 
         src_indices: Optional[List[Value]] = None,
         dst_indices: Optional[List[Value]] = None,
         pred: Optional[Value] = None,
         loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> None:
    """Execute a copy operation using the given copy atom.
    
    Args:
        copy_desc: Copy atom or tiled copy descriptor
        src: Source tensor (memref)
        dst: Destination tensor (memref)
        src_indices: Indices for source access
        dst_indices: Indices for destination access
        pred: Optional predicate mask for conditional copying
        loc: Optional source location
        ip: Optional insertion point
        
    Example:
        >>> rocir.copy(atom, src, dst, src_indices=[i,j], dst_indices=[k])
    """
    from mlir.dialects import memref as memref_dialect
    loc = _get_location(loc)
    if isinstance(copy_desc, TiledCopy) and isinstance(src, TensorView) and isinstance(dst, TensorView):
        val_shape = copy_desc.val_shape
        if val_shape is None:
            raise ValueError("tiled_copy missing val_shape metadata")
        offsets = [_to_index_value(0) for _ in val_shape]
        def recurse(dim):
            if dim == len(val_shape):
                src_coords = [_to_index_value(_add_index(src.base_coords[i], offsets[i])) for i in range(src.rank)]
                dst_coords = [_to_index_value(_add_index(dst.base_coords[i], offsets[i])) for i in range(dst.rank)]
                val = memref_dialect.load(src.memref, src_coords)
                val = _unwrap_value(val)
                memref_dialect.store(val, dst.memref, dst_coords)
                return
            for i in range(val_shape[dim]):
                offsets[dim] = _to_index_value(i)
                recurse(dim + 1)
        with ip or InsertionPoint.current:
            recurse(0)
        return
    
    copy_atom = copy_desc
    src_val = _unwrap_value(src)
    dst_val = _unwrap_value(dst)
    with ip or InsertionPoint.current:
        if src_indices is not None and dst_indices is not None:
            s_idx = [_unwrap_value(i) for i in src_indices]
            d_idx = [_unwrap_value(i) for i in dst_indices]
            val_op = memref_dialect.load(src_val, s_idx)
            if hasattr(val_op, 'result'):
                val = val_op.result
            elif hasattr(val_op, 'results') and len(val_op.results) > 0:
                val = val_op.results[0]
            else:
                val = val_op
            val = _unwrap_value(val)
            memref_dialect.store(val, dst_val, d_idx)
        else:
            pass


#===----------------------------------------------------------------------===//
# Printing operations
#===----------------------------------------------------------------------===//

# Use Python's built-in print for static compile-time values
# This mirrors the behavior where print shows compile-time information
print = print  # Re-export Python's built-in print


def printf(format_str: str, *args, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Print formatted output at runtime (dynamic values).
    
    This function prints dynamic values that are only known at runtime.
    It uses GPU printf to display values during kernel execution.
    
    Args:
        format_str: Format string (e.g., "value: {}")
        *args: Values to print (can be dynamic runtime values)
        loc: Optional source location
        ip: Optional insertion point
        
    Example:
        >>> # Print static value (compile time)
        >>> rocir.print(">>>", b)  # Shows static value
        >>> rocir.print(">>>", a)  # Shows "?" for dynamic value
        >>> 
        >>> # Print dynamic value (runtime)
        >>> rocir.printf(">?? {}", a)  # Shows actual runtime value
        >>> rocir.printf(">?? {}", b)  # Also works for static values
        >>> 
        >>> # Print layout
        >>> layout = rocir.make_layout(shape, stride)
        >>> rocir.print(">>>", layout)  # Shows layout with "?" for dynamic parts
        >>> rocir.printf(">?? {}", layout)  # Shows actual runtime values
    
    Note:
        - Use `rocir.print` (Python's print) for compile-time/static values
        - Use `rocir.printf` for runtime/dynamic values
        - Format strings use "{}" as placeholders (similar to Python f-strings)
    """
    from mlir.dialects import gpu as _gpu
    
    loc = _get_location(loc)
    
    # Unwrap all argument values
    unwrapped_args = [_unwrap_value(arg) for arg in args]
    
    with ip or InsertionPoint.current:
        return _gpu.printf(format=format_str, args=unwrapped_args, loc=loc, ip=ip)


__all__ = [
    # Types
    "ShapeType",
    "StrideType",
    "LayoutType",
    "CoordType",
    # Basic operations
    "make_shape",
    "make_stride",
    "make_layout",
    "make_coord",
    "crd2idx",
    "idx2crd",
    "size",
    "cosize",
    "rank",
    "get",
    "get_shape",
    "get_stride",
    "composition",
    "coalesce",
    # Product operations
    "logical_product",
    "zipped_product",
    "tiled_product",
    "flat_product",
    "raked_product",
    "blocked_product",
    # Divide operations
    "logical_divide",
    "zipped_divide",
    "tiled_divide",
    "flat_divide",
    # Local operations
    "local_partition",
    "local_tile",
    # Copy atom classes and operations
    "CopyAtom",
    "TiledCopy",
    "ThrCopy",
    "make_copy_atom",
    "make_tiled_copy_tv",
    "copy",
    # Tensor and fragment operations
    "Fragment",
    "make_fragment_like",
    "make_rmem_tensor",
    "make_identity_tensor",
    "make_ordered_layout",
    "make_layout_tv",
    "elem_less",
    # Printing operations
    "print",
    "printf",
]
