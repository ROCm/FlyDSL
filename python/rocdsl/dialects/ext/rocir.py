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
    Operation,
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



def make_shape(*dims: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a shape from dimension values.
    
    Args:
        *dims: Index values representing each dimension
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Rocir shape value
        
    Example:
        >>> c8 = arith.constant(8, index=True)
        >>> c16 = arith.constant(16, index=True)
        >>> shape = rocir.make_shape(c8, c16)  # Creates shape<2>
    """
    
    loc = _get_location(loc)
    rank = len(dims)
    result_type = ShapeType.get(rank)
    
    with ip or InsertionPoint.current:
        return rocir_ops.MakeShapeOp(result_type, [_unwrap_value(d) for d in dims], loc=loc).result


def make_stride(*strides: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a stride from stride values.
    
    Args:
        *strides: Index values representing each stride
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Rocir stride value
        
    Example:
        >>> c1 = arith.constant(1, index=True)
        >>> c8 = arith.constant(8, index=True)
        >>> stride = rocir.make_stride(c1, c8)  # Creates stride<2>
    """
    
    loc = _get_location(loc)
    rank = len(strides)
    result_type = StrideType.get(rank)
    
    with ip or InsertionPoint.current:
        return rocir_ops.MakeStrideOp(result_type, [_unwrap_value(s) for s in strides], loc=loc).result


def make_layout(shape: Value, stride: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a layout from shape and stride.
    
    Args:
        shape: A Rocir shape value
        stride: A Rocir stride value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Rocir layout value
        
    Example:
        >>> shape = rocir.make_shape(c8, c16)
        >>> stride = rocir.make_stride(c1, c8)
        >>> layout = rocir.make_layout(shape, stride)
    """
    
    loc = _get_location(loc)
    # Extract rank from shape type
    shape_type_str = str(shape.type)
    rank = int(shape_type_str.split("<")[1].split(">")[0])
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
    
    with ip or InsertionPoint.current as insertion_point:
        op = Operation.create(
            rocir_ops.Crd2IdxOp.OPERATION_NAME,
            results=[result_type],
            operands=[_unwrap_value(coord), _unwrap_value(layout)],
            loc=loc
        )
        # insertion_point.insert(op)
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
    
    with ip or InsertionPoint.current as insertion_point:
        op = Operation.create(
            rocir_ops.SizeOp.OPERATION_NAME,
            results=[result_type],
            operands=[_unwrap_value(shape_or_layout)],
            loc=loc
        )
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
    
    with ip or InsertionPoint.current as insertion_point:
        op = Operation.create(
            rocir_ops.CosizeOp.OPERATION_NAME,
            results=[result_type],
            operands=[_unwrap_value(layout)],
            loc=loc
        )
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
    
    with ip or InsertionPoint.current as insertion_point:
        op = Operation.create(
            rocir_ops.RankOp.OPERATION_NAME,
            results=[result_type],
            operands=[_unwrap_value(shape_or_layout)],
            loc=loc
        )
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
    "get_shape",
    "get_stride",
    "composition",
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
]
