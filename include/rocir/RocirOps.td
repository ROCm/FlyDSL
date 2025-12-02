//===- RocirOps.td - rocir operation definitions ------------*- tablegen -*-===//

#ifndef ROCIR_OPS
#define ROCIR_OPS

include "RocirDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Rocir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Rocir_Dialect, mnemonic, traits>;

// Type constraints
def Rocir_IntType : Type<CPred<"llvm::isa<::mlir::rocir::IntType>($_self)">, "rocir.int">;
def Rocir_ShapeType : Type<CPred<"llvm::isa<::mlir::rocir::ShapeType>($_self)">, "rocir.shape">;
def Rocir_StrideType : Type<CPred<"llvm::isa<::mlir::rocir::StrideType>($_self)">, "rocir.stride">;
def Rocir_LayoutType : Type<CPred<"llvm::isa<::mlir::rocir::LayoutType>($_self)">, "rocir.layout">;
def Rocir_CoordType : Type<CPred<"llvm::isa<::mlir::rocir::CoordType>($_self)">, "rocir.coord">;

//===----------------------------------------------------------------------===//
// Construction Operations
//===----------------------------------------------------------------------===//

def Rocir_MakeShapeOp : Rocir_Op<"make_shape", []> {
  let summary = "Create a rocir shape from index values or nested shapes";
  let description = [{
    Creates a shape value from runtime index values or nested shapes.
    Supports both flat and nested (tuple) layouts.
    Examples: 
      - Flat: rocir.make_shape %i, %j : (index, index) -> !rocir.shape<2>
      - Nested: rocir.make_shape %s1, %s2 : (!rocir.shape<2>, !rocir.shape<3>) -> !rocir.shape<2>
  }];
  let arguments = (ins Variadic<AnyTypeOf<[Index, Rocir_ShapeType]>>:$values);
  let results = (outs Rocir_ShapeType:$result);
  let assemblyFormat = "$values attr-dict `:` functional-type($values, $result)";
}

def Rocir_MakeStrideOp : Rocir_Op<"make_stride", []> {
  let summary = "Create a rocir stride from index values or nested strides";
  let description = [{
    Creates a stride value from runtime index values or nested strides.
    Supports both flat and nested (tuple) layouts.
    Examples:
      - Flat: rocir.make_stride %i, %j : (index, index) -> !rocir.stride<2>
      - Nested: rocir.make_stride %s1, %s2 : (!rocir.stride<2>, !rocir.stride<3>) -> !rocir.stride<2>
  }];
  let arguments = (ins Variadic<AnyTypeOf<[Index, Rocir_StrideType]>>:$values);
  let results = (outs Rocir_StrideType:$result);
  let assemblyFormat = "$values attr-dict `:` functional-type($values, $result)";
}

def Rocir_MakeLayoutOp : Rocir_Op<"make_layout", []> {
  let summary = "Create a rocir layout from shape and stride";
  let description = [{
    Creates a layout value by combining a shape and stride.
    Example: rocir.make_layout %shape, %stride : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_ShapeType:$shape, Rocir_StrideType:$stride);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$shape `,` $stride attr-dict `:` functional-type(operands, $result)";
}

def Rocir_MakeCoordOp : Rocir_Op<"make_coord", []> {
  let summary = "Create a rocir coordinate from index values";
  let description = [{
    Creates a coordinate value from runtime index values.
    Example: rocir.make_coord %i, %j : (index, index) -> !rocir.coord<2>
  }];
  let arguments = (ins Variadic<Index>:$values);
  let results = (outs Rocir_CoordType:$result);
  let assemblyFormat = "$values attr-dict `:` functional-type($values, $result)";
}

//===----------------------------------------------------------------------===//
// Query Operations
//===----------------------------------------------------------------------===//

def Rocir_SizeOp : Rocir_Op<"size", []> {
  let summary = "Get the total size of a shape or layout";
  let description = [{
    Computes the product of all dimensions in a shape or the size of a layout.
    For shape (M, N), returns M * N.
    Example: %s = rocir.size %shape : !rocir.shape<2> -> index
  }];
  let arguments = (ins AnyTypeOf<[Rocir_ShapeType, Rocir_LayoutType]>:$input);
  let results = (outs Index:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Rocir_CosizeOp : Rocir_Op<"cosize", []> {
  let summary = "Get the cosize of a layout";
  let description = [{
    Computes the cosize of a layout (codomain size).
    For a layout with shape and stride, returns the span of the mapping.
    Example: %cs = rocir.cosize %layout : !rocir.layout<2> -> index
  }];
  let arguments = (ins Rocir_LayoutType:$layout);
  let results = (outs Index:$result);
  let assemblyFormat = "$layout attr-dict `:` type($layout) `->` type($result)";
}

def Rocir_GetOp : Rocir_Op<"get", []> {
  let summary = "Extract element from shape/stride/coord at given index";
  let description = [{
    Extracts the value at a specific position from shape, stride, or coord.
    Example: %val = rocir.get %shape, %idx : !rocir.shape<3>, index -> index
  }];
  let arguments = (ins 
    AnyTypeOf<[Rocir_ShapeType, Rocir_StrideType, Rocir_CoordType]>:$input,
    Index:$idx
  );
  let results = (outs Index:$result);
  let assemblyFormat = "$input `,` $idx attr-dict `:` type($input) `,` type($idx) `->` type($result)";
}

def Rocir_RankOp : Rocir_Op<"rank", []> {
  let summary = "Get the rank (number of dimensions) of a type";
  let description = [{
    Returns the number of dimensions in a shape, stride, coord, or layout.
    Example: %r = rocir.rank %shape : !rocir.shape<3> -> index
  }];
  let arguments = (ins AnyTypeOf<[Rocir_ShapeType, Rocir_StrideType, Rocir_CoordType, Rocir_LayoutType]>:$input);
  let results = (outs Index:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Coordinate Mapping Operations
//===----------------------------------------------------------------------===//

def Rocir_Crd2IdxOp : Rocir_Op<"crd2idx", []> {
  let summary = "Convert coordinate to linear index using layout";
  let description = [{
    Converts a multi-dimensional coordinate to a linear index using a layout.
    Computes: sum(coord[i] * stride[i]) for all dimensions i.
    Example: %idx = rocir.crd2idx %coord, %layout : (!rocir.coord<2>, !rocir.layout<2>) -> index
  }];
  let arguments = (ins Rocir_CoordType:$coord, Rocir_LayoutType:$layout);
  let results = (outs Index:$result);
  let assemblyFormat = "$coord `,` $layout attr-dict `:` functional-type(operands, $result)";
}

def Rocir_Idx2CrdOp : Rocir_Op<"idx2crd", []> {
  let summary = "Convert linear index to coordinate using layout";
  let description = [{
    Converts a linear index to a multi-dimensional coordinate using a layout.
    This is the inverse of crd2idx.
    Example: %coord = rocir.idx2crd %idx, %layout : (index, !rocir.layout<2>) -> !rocir.coord<2>
  }];
  let arguments = (ins Index:$idx, Rocir_LayoutType:$layout);
  let results = (outs Rocir_CoordType:$result);
  let assemblyFormat = "$idx `,` $layout attr-dict `:` functional-type(operands, $result)";
}

//===----------------------------------------------------------------------===//
// Layout Manipulation Operations
//===----------------------------------------------------------------------===//

def Rocir_GetShapeOp : Rocir_Op<"get_shape", []> {
  let summary = "Extract shape from layout";
  let description = [{
    Extracts the shape component from a layout.
    Example: %shape = rocir.get_shape %layout : !rocir.layout<2> -> !rocir.shape<2>
  }];
  let arguments = (ins Rocir_LayoutType:$layout);
  let results = (outs Rocir_ShapeType:$result);
  let assemblyFormat = "$layout attr-dict `:` type($layout) `->` type($result)";
}

def Rocir_GetStrideOp : Rocir_Op<"get_stride", []> {
  let summary = "Extract stride from layout";
  let description = [{
    Extracts the stride component from a layout.
    Example: %stride = rocir.get_stride %layout : !rocir.layout<2> -> !rocir.stride<2>
  }];
  let arguments = (ins Rocir_LayoutType:$layout);
  let results = (outs Rocir_StrideType:$result);
  let assemblyFormat = "$layout attr-dict `:` type($layout) `->` type($result)";
}

def Rocir_CompositionOp : Rocir_Op<"composition", []> {
  let summary = "Compose two layouts";
  let description = [{
    Computes the composition of two layouts: layoutA ∘ layoutB.
    The result maps indices through layoutB then through layoutA.
    Example: %comp = rocir.composition %layoutA, %layoutB : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_LayoutType:$layoutA, Rocir_LayoutType:$layoutB);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$layoutA `,` $layoutB attr-dict `:` functional-type(operands, $result)";
}

def Rocir_CoalesceOp : Rocir_Op<"coalesce", []> {
  let summary = "Coalesce/simplify a layout";
  let description = [{
    Coalesces a layout by flattening and combining modes when possible.
    Ensures post-conditions:
    - Preserves size: rocir.size(layout) == rocir.size(result)
    - Flattened: depth(result) <= 1
    - Preserves function: For all i, 0 <= i < rocir.size(layout), layout(i) == result(i)
    Example: %coalesced = rocir.coalesce %layout : !rocir.layout<3> -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_LayoutType:$layout);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$layout attr-dict `:` type($layout) `->` type($result)";
}

def Rocir_ComplementOp : Rocir_Op<"complement", []> {
  let summary = "Compute complement of a tiler layout";
  let description = [{
    Computes the complement of a tiler layout with respect to a target size.
    This finds the "rest" modes that are not covered by the tiler.
    Used internally by logical_divide: logical_divide(L, T) = composition(L, make_layout(T, complement(T, size(L)))).
    
    The complement operation:
    1. Filters out stride-0 and size-1 modes from the tiler
    2. Sorts modes by stride (ascending)
    3. Folds to compute rest modes: at each step, removes the minimum stride mode
    4. Returns a coalesced layout of the rest modes
    
    Example: %comp = rocir.complement %tiler, %target_size : (!rocir.layout<2>, index) -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_LayoutType:$tiler, Index:$target_size);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$tiler `,` $target_size attr-dict `:` functional-type(operands, $result)";
}

#endif // ROCIR_OPS

//===----------------------------------------------------------------------===//
// Layout Product Operations
//===----------------------------------------------------------------------===//

def Rocir_LogicalProductOp : Rocir_Op<"logical_product", []> {
  let summary = "Compute logical product of layout with tiler";
  let description = [{
    Computes the logical product of a layout and a tiler layout.
    This applies the tiler to the layout in a mode-wise fashion.
    Example: %result = rocir.logical_product %input, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_LayoutType:$input, Rocir_LayoutType:$tiler);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
}

def Rocir_ZippedProductOp : Rocir_Op<"zipped_product", []> {
  let summary = "Compute zipped product of layout with tiler";
  let description = [{
    Computes the zipped product of a layout and a tiler layout.
    This interleaves the modes of the input and tiler.
    Example: %result = rocir.zipped_product %input, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_LayoutType:$input, Rocir_LayoutType:$tiler);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
}

def Rocir_TiledProductOp : Rocir_Op<"tiled_product", []> {
  let summary = "Compute tiled product of layout with tiler";
  let description = [{
    Computes the tiled product of a layout and a tiler layout.
    This creates a hierarchical tiled layout.
    Example: %result = rocir.tiled_product %input, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_LayoutType:$input, Rocir_LayoutType:$tiler);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
}

def Rocir_FlatProductOp : Rocir_Op<"flat_product", []> {
  let summary = "Compute flat product of layout with tiler";
  let description = [{
    Computes the flat product of a layout and a tiler layout.
    This produces a flattened layout from the product.
    Example: %result = rocir.flat_product %input, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_LayoutType:$input, Rocir_LayoutType:$tiler);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
}

def Rocir_RakedProductOp : Rocir_Op<"raked_product", []> {
  let summary = "Compute raked product of layout with tiler";
  let description = [{
    Computes the raked product of a layout and a tiler layout.
    This creates a raked memory access pattern.
    Example: %result = rocir.raked_product %input, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_LayoutType:$input, Rocir_LayoutType:$tiler);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
}

def Rocir_BlockedProductOp : Rocir_Op<"blocked_product", []> {
  let summary = "Compute blocked product of layout with tiler";
  let description = [{
    Computes the blocked product of a layout and a tiler layout.
    This creates a blocked memory access pattern.
    Example: %result = rocir.blocked_product %input, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_LayoutType:$input, Rocir_LayoutType:$tiler);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
}


//===----------------------------------------------------------------------===//
// Layout Divide Operations
//===----------------------------------------------------------------------===//

def Rocir_LogicalDivideOp : Rocir_Op<"logical_divide", []> {
  let summary = "Logically divide a layout by a tiler";
  let description = [{
    Divides a layout logically by a tiler, creating a partitioned view.
    This is commonly used for tiling tensor operations.
    Example: %result = rocir.logical_divide %input, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_LayoutType:$input, Rocir_LayoutType:$tiler);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
}

def Rocir_ZippedDivideOp : Rocir_Op<"zipped_divide", []> {
  let summary = "Divide a layout by a tiler with zipped semantics";
  let description = [{
    Divides a layout by a tiler using zipped division semantics.
    This interleaves the division across modes.
    Example: %result = rocir.zipped_divide %input, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_LayoutType:$input, Rocir_LayoutType:$tiler);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
}

def Rocir_TiledDivideOp : Rocir_Op<"tiled_divide", []> {
  let summary = "Divide a layout by a tiler with tiled semantics";
  let description = [{
    Divides a layout by a tiler creating a hierarchical tiled division.
    This is the dual of tiled_product.
    Example: %result = rocir.tiled_divide %input, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_LayoutType:$input, Rocir_LayoutType:$tiler);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
}

def Rocir_FlatDivideOp : Rocir_Op<"flat_divide", []> {
  let summary = "Divide a layout by a tiler with flat semantics";
  let description = [{
    Divides a layout by a tiler creating a flattened division.
    This produces a flat result from the division.
    Example: %result = rocir.flat_divide %input, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_LayoutType:$input, Rocir_LayoutType:$tiler);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
}


//===----------------------------------------------------------------------===//
// Local Partition and Tile Operations
//===----------------------------------------------------------------------===//

def Rocir_LocalPartitionOp : Rocir_Op<"local_partition", []> {
  let summary = "Partition a layout for a specific thread/block index";
  let description = [{
    Partitions a layout by a tiler for a specific thread or block index.
    This is used to assign thread-local portions of a tensor.
    
    Similar to: outer_partition(tensor, tile, coord_to_index(tile, index))
    
    Example: %result = rocir.local_partition %tensor, %tile, %index : (!rocir.layout<2>, !rocir.layout<2>, index) -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_LayoutType:$input, Rocir_LayoutType:$tiler, Index:$index);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler `,` $index attr-dict `:` functional-type(operands, $result)";
}

def Rocir_LocalTileOp : Rocir_Op<"local_tile", []> {
  let summary = "Extract a tile from a layout at specific coordinates";
  let description = [{
    Extracts a tile of a specific shape from a layout at given coordinates.
    This is used for block-level tiling (e.g., CTA tiles in GEMM).
    
    The operation tiles the input tensor and selects the tile at the given coordinate.
    
    Example: %result = rocir.local_tile %tensor, %tiler, %coord : (!rocir.layout<2>, !rocir.shape<2>, !rocir.shape<2>) -> !rocir.layout<2>
  }];
  let arguments = (ins Rocir_LayoutType:$input, Rocir_ShapeType:$tiler, Rocir_ShapeType:$coord);
  let results = (outs Rocir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler `,` $coord attr-dict `:` functional-type(operands, $result)";
}

