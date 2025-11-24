"""Python bindings for CuTe dialect operations.

This module provides Python wrappers for CuTe layout algebra operations,
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


def _get_location(loc: Optional[Location] = None) -> Location:
    """Get location, using current location if none provided."""
    if loc is None:
        loc = Location.unknown()
    return loc


def _get_insertion_point(ip: Optional[InsertionPoint] = None) -> InsertionPoint:
    """Get insertion point, using current if none provided."""
    if ip is None:
        return InsertionPoint.current
    return ip


class ShapeType(Type):
    """CuTe shape type."""
    
    @staticmethod
    def get(rank: int, context=None):
        """Create a shape type with given rank."""
        # This would need to be implemented in C++ bindings
        # For now, return a generic type
        from mlir.ir import Context
        if context is None:
            context = Context.current
        # Placeholder - would use actual ODS-generated type
        return Type.parse(f"!cute.shape<{rank}>", context=context)


class StrideType(Type):
    """CuTe stride type."""
    
    @staticmethod
    def get(rank: int, context=None):
        """Create a stride type with given rank."""
        from mlir.ir import Context
        if context is None:
            context = Context.current
        return Type.parse(f"!cute.stride<{rank}>", context=context)


class LayoutType(Type):
    """CuTe layout type."""
    
    @staticmethod
    def get(rank: int, context=None):
        """Create a layout type with given rank."""
        from mlir.ir import Context
        if context is None:
            context = Context.current
        return Type.parse(f"!cute.layout<{rank}>", context=context)


def make_shape(*dims: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a shape from dimension values.
    
    Args:
        *dims: Index values representing each dimension
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A CuTe shape value
        
    Example:
        >>> c8 = arith.constant(8, index=True)
        >>> c16 = arith.constant(16, index=True)
        >>> shape = cute.make_shape(c8, c16)  # Creates shape<2>
    """
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    rank = len(dims)
    result_type = ShapeType.get(rank)
    
    with ip or InsertionPoint.current:
        return cute_ops.MakeShapeOp(result_type, list(dims), loc=loc).result


def make_stride(*strides: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a stride from stride values.
    
    Args:
        *strides: Index values representing each stride
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A CuTe stride value
        
    Example:
        >>> c1 = arith.constant(1, index=True)
        >>> c8 = arith.constant(8, index=True)
        >>> stride = cute.make_stride(c1, c8)  # Creates stride<2>
    """
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    rank = len(strides)
    result_type = StrideType.get(rank)
    
    with ip or InsertionPoint.current:
        return cute_ops.MakeStrideOp(result_type, list(strides), loc=loc).result


def make_layout(shape: Value, stride: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a layout from shape and stride.
    
    Args:
        shape: A CuTe shape value
        stride: A CuTe stride value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A CuTe layout value
        
    Example:
        >>> shape = cute.make_shape(c8, c16)
        >>> stride = cute.make_stride(c1, c8)
        >>> layout = cute.make_layout(shape, stride)
    """
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    # Extract rank from shape type
    shape_type_str = str(shape.type)
    rank = int(shape_type_str.split("<")[1].split(">")[0])
    result_type = LayoutType.get(rank)
    
    with ip or InsertionPoint.current:
        return cute_ops.MakeLayoutOp(result_type, shape, stride, loc=loc).result


def size(shape_or_layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the total size of a shape or layout.
    
    Args:
        shape_or_layout: A CuTe shape or layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        An index value representing the total size
        
    Example:
        >>> shape = cute.make_shape(c8, c16)
        >>> total = cute.size(shape)  # Returns 128
    """
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        return cute_ops.SizeOp(result_type, shape_or_layout, loc=loc).result


def cosize(layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the cosize (stride extent) of a layout.
    
    Args:
        layout: A CuTe layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        An index value representing the cosize
    """
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        return cute_ops.CosizeOp(result_type, layout, loc=loc).result


def rank(shape_or_layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Get the rank (number of dimensions) of a shape or layout.
    
    Args:
        shape_or_layout: A CuTe shape or layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        An index value representing the rank
    """
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        return cute_ops.RankOp(result_type, shape_or_layout, loc=loc).result


def get_shape(layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Extract the shape from a layout.
    
    Args:
        layout: A CuTe layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The shape component of the layout
    """
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    # Extract rank from layout type
    layout_type_str = str(layout.type)
    rank = int(layout_type_str.split("<")[1].split(">")[0])
    result_type = ShapeType.get(rank)
    
    with ip or InsertionPoint.current:
        return cute_ops.GetShapeOp(result_type, layout, loc=loc).result


def get_stride(layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Extract the stride from a layout.
    
    Args:
        layout: A CuTe layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The stride component of the layout
    """
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    # Extract rank from layout type
    layout_type_str = str(layout.type)
    rank = int(layout_type_str.split("<")[1].split(">")[0])
    result_type = StrideType.get(rank)
    
    with ip or InsertionPoint.current:
        return cute_ops.GetStrideOp(result_type, layout, loc=loc).result


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
        >>> composed = cute.composition(col_major, tiler)
    """
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = layout_a.type
    
    with ip or InsertionPoint.current:
        return cute_ops.CompositionOp(result_type, layout_a, layout_b, loc=loc).result


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
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = block.type
    
    with ip or InsertionPoint.current:
        return cute_ops.LogicalProductOp(result_type, block, tiler, loc=loc).result


def zipped_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the zipped product of two layouts."""
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = block.type
    
    with ip or InsertionPoint.current:
        return cute_ops.ZippedProductOp(result_type, block, tiler, loc=loc).result


def tiled_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the tiled product of two layouts."""
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = block.type
    
    with ip or InsertionPoint.current:
        return cute_ops.TiledProductOp(result_type, block, tiler, loc=loc).result


def flat_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the flat product of two layouts."""
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = block.type
    
    with ip or InsertionPoint.current:
        return cute_ops.FlatProductOp(result_type, block, tiler, loc=loc).result


def raked_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the raked product of two layouts."""
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = block.type
    
    with ip or InsertionPoint.current:
        return cute_ops.RakedProductOp(result_type, block, tiler, loc=loc).result


def blocked_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the blocked product of two layouts."""
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = block.type
    
    with ip or InsertionPoint.current:
        return cute_ops.BlockedProductOp(result_type, block, tiler, loc=loc).result


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
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = layout.type
    
    with ip or InsertionPoint.current:
        return cute_ops.LogicalDivideOp(result_type, layout, tiler, loc=loc).result


def zipped_divide(layout: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the zipped divide of a layout."""
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = layout.type
    
    with ip or InsertionPoint.current:
        return cute_ops.ZippedDivideOp(result_type, layout, tiler, loc=loc).result


def tiled_divide(layout: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the tiled divide of a layout."""
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = layout.type
    
    with ip or InsertionPoint.current:
        return cute_ops.TiledDivideOp(result_type, layout, tiler, loc=loc).result


def flat_divide(layout: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the flat divide of a layout."""
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = layout.type
    
    with ip or InsertionPoint.current:
        return cute_ops.FlatDivideOp(result_type, layout, tiler, loc=loc).result


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
        >>> thread_data = cute.local_partition(global_layout, tile, thread_idx)
    """
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = layout.type
    
    with ip or InsertionPoint.current:
        return cute_ops.LocalPartitionOp(result_type, layout, tile, index, loc=loc).result


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
        >>> cta_data = cute.local_tile(global_layout, cta_shape, block_coord)
    """
    from mlir.dialects import cute as cute_ops
    
    loc = _get_location(loc)
    result_type = layout.type
    
    with ip or InsertionPoint.current:
        return cute_ops.LocalTileOp(result_type, layout, tiler, coord, loc=loc).result


__all__ = [
    # Types
    "ShapeType",
    "StrideType",
    "LayoutType",
    # Basic operations
    "make_shape",
    "make_stride",
    "make_layout",
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
