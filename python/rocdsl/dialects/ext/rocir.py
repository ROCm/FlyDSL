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

# Import generated ops
import sys
import os
_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../build/python_bindings"))
if _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)
import rocir as rocir_ops
import _rocirPassesExt  # Auto-register passes



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
        >>> # Nested shape like CuTe: (9, (4, 8))
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
        >>> # Nested stride like CuTe: (59, (13, 1))
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
        >>> # Using tuples directly (CuTe-style)
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



def size(shape_or_layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the total size of a shape or layout.
    
    Args:
        shape_or_layout: A Rocir shape or layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        An index value representing the total size
        
    Example:
        >>> shape = rocir.make_shape(c8, c16)
        >>> total = rocir.size(shape)  # Returns 128
    """
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        op = rocir_ops.SizeOp(_unwrap_value(shape_or_layout), results=[result_type], loc=loc, ip=ip)
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


# Printing operations

# Use Python's built-in print for static compile-time values
# This is consistent with CuTeDSL where print shows compile-time information
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
    # Printing operations
    "print",
    "printf",
]
