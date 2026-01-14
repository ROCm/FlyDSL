//===- FlirOps.td - flir operation definitions ------------*- tablegen -*-===//

#ifndef FLIR_OPS
#define FLIR_OPS

include "flir/FlirDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Flir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Flir_Dialect, mnemonic, traits>;

// Type constraints
def Flir_IntType : Type<CPred<"llvm::isa<::mlir::flir::IntType>($_self)">, "flir.int">;
def Flir_ShapeType : Type<CPred<"llvm::isa<::mlir::flir::ShapeType>($_self)">, "flir.shape">;
def Flir_StrideType : Type<CPred<"llvm::isa<::mlir::flir::StrideType>($_self)">, "flir.stride">;
def Flir_LayoutType : Type<CPred<"llvm::isa<::mlir::flir::LayoutType>($_self)">, "flir.layout">;
def Flir_CoordType : Type<CPred<"llvm::isa<::mlir::flir::CoordType>($_self)">, "flir.coord">;

//===----------------------------------------------------------------------===//
// Construction Operations
//===----------------------------------------------------------------------===//

def Flir_MakeShapeOp : Flir_Op<"make_shape", [Pure]> {
  let summary = "Create a flir shape (type-mode: structure/spec in type, dynamic leaves in operands)";
  let description = [{
    Creates a shape value.
    This op is **type-mode only**:
      - The tuple structure and any compile-time constants are encoded in the result type
        (`!flir.shape<...>` / structure+dims).
      - The operands contain **only the dynamic leaf index values** (in left-to-right leaf order).
  }];
  let arguments = (ins Variadic<Index>:$values);
  let results = (outs Flir_ShapeType:$result);
  let assemblyFormat = "$values attr-dict `:` functional-type($values, $result)";
  let hasVerifier = 1;
}

def Flir_MakeStrideOp : Flir_Op<"make_stride", [Pure]> {
  let summary = "Create a flir stride (type-mode: structure/spec in type, dynamic leaves in operands)";
  let description = [{
    Creates a stride value.
    This op is **type-mode only**:
      - The tuple structure and any compile-time constants are encoded in the result type
        (`!flir.stride<...>` / structure+dims).
      - The operands contain **only the dynamic leaf index values** (in left-to-right leaf order).
  }];
  let arguments = (ins Variadic<Index>:$values);
  let results = (outs Flir_StrideType:$result);
  let assemblyFormat = "$values attr-dict `:` functional-type($values, $result)";
  let hasVerifier = 1;
}

def Flir_MakeLayoutOp : Flir_Op<"make_layout", [Pure]> {
  let summary = "Create a flir layout from shape and stride";
  let description = [{
    Creates a layout value by combining a shape and stride.
    Example: flir.make_layout %shape, %stride : (!flir.shape<2>, !flir.stride<2>) -> !flir.layout<2>
  }];
  let arguments = (ins Flir_ShapeType:$shape, Flir_StrideType:$stride);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$shape `,` $stride attr-dict `:` functional-type(operands, $result)";
}

def Flir_MakeCoordOp : Flir_Op<"make_coord", [Pure]> {
  let summary = "Create a flir coordinate from index values";
  let description = [{
    Creates a coordinate value from runtime index values.
    Example: flir.make_coord %i, %j : (index, index) -> !flir.coord<2>
  }];
  // Coordinates are represented as a flat list of index operands.
  // Nested coord SSA trees are not supported.
  let arguments = (ins Variadic<Index>:$values);
  let results = (outs Flir_CoordType:$result);
  let assemblyFormat = "$values attr-dict `:` functional-type($values, $result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Query Operations
//===----------------------------------------------------------------------===//

def Flir_SizeOp : Flir_Op<"size", []> {
  let summary = "Get the total size of a shape or layout";
  let description = [{
    Computes the product of all dimensions in a shape or the size of a layout.
    For shape (M, N), returns M * N.
    Example: %s = flir.size %shape : !flir.shape<2> -> index
  }];
  let arguments = (ins AnyTypeOf<[Flir_ShapeType, Flir_LayoutType]>:$input);
  let results = (outs Index:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Flir_CosizeOp : Flir_Op<"cosize", []> {
  let summary = "Get the cosize of a layout";
  let description = [{
    Computes the cosize of a layout (codomain size).
    For a layout with shape and stride, returns the span of the mapping.
    Example: %cs = flir.cosize %layout : !flir.layout<2> -> index
  }];
  let arguments = (ins Flir_LayoutType:$layout);
  let results = (outs Index:$result);
  let assemblyFormat = "$layout attr-dict `:` type($layout) `->` type($result)";
}

def Flir_GetOp : Flir_Op<"get", []> {
  let summary = "Extract element from shape/stride/coord at given index";
  let description = [{
    Extracts the value at a specific position from shape, stride, or coord.
    Example: %val = flir.get %shape, %idx : !flir.shape<3>, index -> index
  }];
  let arguments = (ins 
    AnyTypeOf<[Flir_ShapeType, Flir_StrideType, Flir_CoordType]>:$input,
    Index:$idx
  );
  let results = (outs Index:$result);
  let assemblyFormat = "$input `,` $idx attr-dict `:` type($input) `,` type($idx) `->` type($result)";
}

def Flir_RankOp : Flir_Op<"rank", []> {
  let summary = "Get the rank (number of dimensions) of a type";
  let description = [{
    Returns the number of dimensions in a shape, stride, coord, or layout.
    Example: %r = flir.rank %shape : !flir.shape<3> -> index
  }];
  let arguments = (ins AnyTypeOf<[Flir_ShapeType, Flir_StrideType, Flir_CoordType, Flir_LayoutType]>:$input);
  let results = (outs Index:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Coordinate Mapping Operations
//===----------------------------------------------------------------------===//

def Flir_Crd2IdxOp : Flir_Op<"crd2idx", []> {
  let summary = "Convert coordinate to linear index using layout";
  let description = [{
    Converts a multi-dimensional coordinate to a linear index using a layout.
    Computes: sum(coord[i] * stride[i]) for all dimensions i.
    Example: %idx = flir.crd2idx %coord, %layout : (!flir.coord<2>, !flir.layout<2>) -> index
  }];
  let arguments = (ins Flir_CoordType:$coord, Flir_LayoutType:$layout);
  let results = (outs Index:$result);
  let assemblyFormat = "$coord `,` $layout attr-dict `:` functional-type(operands, $result)";
}

def Flir_Idx2CrdOp : Flir_Op<"idx2crd", []> {
  let summary = "Convert linear index to coordinate using layout";
  let description = [{
    Converts a linear index to a multi-dimensional coordinate using a layout.
    This is the inverse of crd2idx.
    Example: %coord = flir.idx2crd %idx, %layout : (index, !flir.layout<2>) -> !flir.coord<2>
  }];
  let arguments = (ins Index:$idx, Flir_LayoutType:$layout);
  let results = (outs Flir_CoordType:$result);
  let assemblyFormat = "$idx `,` $layout attr-dict `:` functional-type(operands, $result)";
}

//===----------------------------------------------------------------------===//
// Swizzle Operations
//===----------------------------------------------------------------------===//

def Flir_SwizzleXor16Op : Flir_Op<"swizzle_xor16", [Pure]> {
  let summary = "XOR-with-row swizzle for LDS bank-conflict avoidance (16B granularity)";
  let description = [{
    Computes:

      %row_mod = %row remui %kBlocks16
      %mask    = %row_mod * 16
      %out     = %col xor %mask

    This matches the common CK/ROCDSL-style LDS swizzle used for fp8/f16 tiles where
    the K dimension is permuted at 16B granularity (preserving intra-16B order).
  }];
  let arguments = (ins Index:$row, Index:$col, Index:$kBlocks16);
  let results = (outs Index:$result);
  let assemblyFormat = "$row `,` $col `,` $kBlocks16 attr-dict `:` functional-type(operands, $result)";
}

//===----------------------------------------------------------------------===//
// Layout Manipulation Operations
//===----------------------------------------------------------------------===//

def Flir_GetShapeOp : Flir_Op<"get_shape", []> {
  let summary = "Extract shape from layout";
  let description = [{
    Extracts the shape component from a layout.
    Example: %shape = flir.get_shape %layout : !flir.layout<2> -> !flir.shape<2>
  }];
  let arguments = (ins Flir_LayoutType:$layout);
  let results = (outs Flir_ShapeType:$result);
  let assemblyFormat = "$layout attr-dict `:` type($layout) `->` type($result)";
}

def Flir_GetStrideOp : Flir_Op<"get_stride", []> {
  let summary = "Extract stride from layout";
  let description = [{
    Extracts the stride component from a layout.
    Example: %stride = flir.get_stride %layout : !flir.layout<2> -> !flir.stride<2>
  }];
  let arguments = (ins Flir_LayoutType:$layout);
  let results = (outs Flir_StrideType:$result);
  let assemblyFormat = "$layout attr-dict `:` type($layout) `->` type($result)";
}

def Flir_CompositionOp : Flir_Op<"composition", []> {
  let summary = "Compose two layouts";
  let description = [{
    Computes the composition of two layouts: layoutA ∘ layoutB.
    The result maps indices through layoutB then through layoutA.
    Example: %comp = flir.composition %layoutA, %layoutB : (!flir.layout<2>, !flir.layout<2>) -> !flir.layout<2>
  }];
  let arguments = (ins Flir_LayoutType:$layoutA, Flir_LayoutType:$layoutB);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$layoutA `,` $layoutB attr-dict `:` functional-type(operands, $result)";
  let builders = [
    OpBuilder<(ins "Value":$layoutA, "Value":$layoutB), [{
      auto lhsTy = llvm::dyn_cast<::mlir::flir::LayoutType>(layoutA.getType());
      auto rhsTy = llvm::dyn_cast<::mlir::flir::LayoutType>(layoutB.getType());
      ::mlir::Type resultTy = layoutA.getType();
      if (lhsTy && rhsTy) {
        auto inferred = ::mlir::flir::inferCompositionType($_builder.getContext(), lhsTy, rhsTy);
        if (::mlir::succeeded(inferred))
          resultTy = *inferred;
        else
          resultTy = ::mlir::flir::makeRankOnlyLayoutType(
              $_builder.getContext(),
              std::max(::mlir::flir::getPatternRank(lhsTy.getShapePattern()),
                       ::mlir::flir::getPatternRank(rhsTy.getShapePattern())));
      }
      build($_builder, $_state, resultTy, layoutA, layoutB);
    }]>
  ];
}

def Flir_CoalesceOp : Flir_Op<"coalesce", []> {
  let summary = "Coalesce/simplify a layout";
  let description = [{
    Coalesces a layout by flattening and combining modes when possible.
    Ensures post-conditions:
    - Preserves size: flir.size(layout) == flir.size(result)
    - Flattened: depth(result) <= 1
    - Preserves function: For all i, 0 <= i < flir.size(layout), layout(i) == result(i)
    Example: %coalesced = flir.coalesce %layout : !flir.layout<3> -> !flir.layout<2>
  }];
  let arguments = (ins Flir_LayoutType:$layout);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$layout attr-dict `:` type($layout) `->` type($result)";
}

def Flir_ComplementOp : Flir_Op<"complement", []> {
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
    
    Example: %comp = flir.complement %tiler, %target_size : (!flir.layout<2>, index) -> !flir.layout<2>
  }];
  let arguments = (ins Flir_LayoutType:$tiler, Index:$target_size);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$tiler `,` $target_size attr-dict `:` functional-type(operands, $result)";
}

//===----------------------------------------------------------------------===//
// Layout Product Operations
//===----------------------------------------------------------------------===//

def Flir_LogicalProductOp : Flir_Op<"logical_product", []> {
  let summary = "Compute logical product of layout with tiler";
  let description = [{
    Computes the logical product of a layout and a tiler layout.
    This applies the tiler to the layout in a mode-wise fashion.
    Example: %result = flir.logical_product %input, %tiler : (!flir.layout<2>, !flir.layout<2>) -> !flir.layout<2>
  }];
  let arguments = (ins Flir_LayoutType:$input, Flir_LayoutType:$tiler);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$tiler), [{
      auto aTy = llvm::dyn_cast<::mlir::flir::LayoutType>(input.getType());
      auto bTy = llvm::dyn_cast<::mlir::flir::LayoutType>(tiler.getType());
      ::mlir::Type resultTy = input.getType();
      if (aTy && bTy) {
        auto inferred = ::mlir::flir::inferLogicalProductType($_builder.getContext(), aTy, bTy);
        if (::mlir::succeeded(inferred))
          resultTy = *inferred;
        else
          resultTy = ::mlir::flir::makeRankOnlyLayoutType(
              $_builder.getContext(),
              ::mlir::flir::getPatternRank(aTy.getShapePattern()) +
                  ::mlir::flir::getPatternRank(bTy.getShapePattern()));
      }
      build($_builder, $_state, resultTy, input, tiler);
    }]>
  ];
}

def Flir_ZippedProductOp : Flir_Op<"zipped_product", []> {
  let summary = "Compute zipped product of layout with tiler";
  let description = [{
    Computes the zipped product of a layout and a tiler layout.
    This interleaves the modes of the input and tiler.
    Example: %result = flir.zipped_product %input, %tiler : (!flir.layout<2>, !flir.layout<2>) -> !flir.layout<2>
  }];
  let arguments = (ins Flir_LayoutType:$input, Flir_LayoutType:$tiler);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$tiler), [{
      // Current lowering treats zipped_product the same as logical_product.
      auto aTy = llvm::dyn_cast<::mlir::flir::LayoutType>(input.getType());
      auto bTy = llvm::dyn_cast<::mlir::flir::LayoutType>(tiler.getType());
      ::mlir::Type resultTy = input.getType();
      if (aTy && bTy) {
        auto inferred = ::mlir::flir::inferLogicalProductType($_builder.getContext(), aTy, bTy);
        if (::mlir::succeeded(inferred))
          resultTy = *inferred;
        else
          resultTy = ::mlir::flir::makeRankOnlyLayoutType(
              $_builder.getContext(),
              ::mlir::flir::getPatternRank(aTy.getShapePattern()) +
                  ::mlir::flir::getPatternRank(bTy.getShapePattern()));
      }
      build($_builder, $_state, resultTy, input, tiler);
    }]>
  ];
}

def Flir_TiledProductOp : Flir_Op<"tiled_product", []> {
  let summary = "Compute tiled product of layout with tiler";
  let description = [{
    Computes the tiled product of a layout and a tiler layout.
    This creates a hierarchical tiled layout.
    Example: %result = flir.tiled_product %input, %tiler : (!flir.layout<2>, !flir.layout<2>) -> !flir.layout<2>
  }];
  let arguments = (ins Flir_LayoutType:$input, Flir_LayoutType:$tiler);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$tiler), [{
      // Current lowering treats tiled_product the same as logical_product.
      auto aTy = llvm::dyn_cast<::mlir::flir::LayoutType>(input.getType());
      auto bTy = llvm::dyn_cast<::mlir::flir::LayoutType>(tiler.getType());
      ::mlir::Type resultTy = input.getType();
      if (aTy && bTy) {
        auto inferred = ::mlir::flir::inferLogicalProductType($_builder.getContext(), aTy, bTy);
        if (::mlir::succeeded(inferred))
          resultTy = *inferred;
        else
          resultTy = ::mlir::flir::makeRankOnlyLayoutType(
              $_builder.getContext(),
              ::mlir::flir::getPatternRank(aTy.getShapePattern()) +
                  ::mlir::flir::getPatternRank(bTy.getShapePattern()));
      }
      build($_builder, $_state, resultTy, input, tiler);
    }]>
  ];
}

def Flir_FlatProductOp : Flir_Op<"flat_product", []> {
  let summary = "Compute flat product of layout with tiler";
  let description = [{
    Computes the flat product of a layout and a tiler layout.
    This produces a flattened layout from the product.
    Example: %result = flir.flat_product %input, %tiler : (!flir.layout<2>, !flir.layout<2>) -> !flir.layout<2>
  }];
  let arguments = (ins Flir_LayoutType:$input, Flir_LayoutType:$tiler);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$tiler), [{
      // Current lowering treats flat_product the same as logical_product.
      auto aTy = llvm::dyn_cast<::mlir::flir::LayoutType>(input.getType());
      auto bTy = llvm::dyn_cast<::mlir::flir::LayoutType>(tiler.getType());
      ::mlir::Type resultTy = input.getType();
      if (aTy && bTy) {
        auto inferred = ::mlir::flir::inferLogicalProductType($_builder.getContext(), aTy, bTy);
        if (::mlir::succeeded(inferred))
          resultTy = *inferred;
        else
          resultTy = ::mlir::flir::makeRankOnlyLayoutType(
              $_builder.getContext(),
              ::mlir::flir::getPatternRank(aTy.getShapePattern()) +
                  ::mlir::flir::getPatternRank(bTy.getShapePattern()));
      }
      build($_builder, $_state, resultTy, input, tiler);
    }]>
  ];
}

def Flir_RakedProductOp : Flir_Op<"raked_product", []> {
  let summary = "Compute raked product of layout with tiler";
  let description = [{
    Computes the raked product of a layout and a tiler layout.
    This creates a raked memory access pattern.
    Example: %result = flir.raked_product %input, %tiler : (!flir.layout<2>, !flir.layout<2>) -> !flir.layout<2>
  }];
  let arguments = (ins Flir_LayoutType:$input, Flir_LayoutType:$tiler);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$tiler), [{
      // Current lowering treats raked_product the same as logical_product.
      auto aTy = llvm::dyn_cast<::mlir::flir::LayoutType>(input.getType());
      auto bTy = llvm::dyn_cast<::mlir::flir::LayoutType>(tiler.getType());
      ::mlir::Type resultTy = input.getType();
      if (aTy && bTy) {
        auto inferred = ::mlir::flir::inferLogicalProductType($_builder.getContext(), aTy, bTy);
        if (::mlir::succeeded(inferred))
          resultTy = *inferred;
        else
          resultTy = ::mlir::flir::makeRankOnlyLayoutType(
              $_builder.getContext(),
              ::mlir::flir::getPatternRank(aTy.getShapePattern()) +
                  ::mlir::flir::getPatternRank(bTy.getShapePattern()));
      }
      build($_builder, $_state, resultTy, input, tiler);
    }]>
  ];
}

def Flir_BlockedProductOp : Flir_Op<"blocked_product", []> {
  let summary = "Compute blocked product of layout with tiler";
  let description = [{
    Computes the blocked product of a layout and a tiler layout.
    This creates a blocked memory access pattern.
    Example: %result = flir.blocked_product %input, %tiler : (!flir.layout<2>, !flir.layout<2>) -> !flir.layout<2>
  }];
  let arguments = (ins Flir_LayoutType:$input, Flir_LayoutType:$tiler);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$tiler), [{
      // Current lowering treats blocked_product the same as logical_product.
      auto aTy = llvm::dyn_cast<::mlir::flir::LayoutType>(input.getType());
      auto bTy = llvm::dyn_cast<::mlir::flir::LayoutType>(tiler.getType());
      ::mlir::Type resultTy = input.getType();
      if (aTy && bTy) {
        auto inferred = ::mlir::flir::inferLogicalProductType($_builder.getContext(), aTy, bTy);
        if (::mlir::succeeded(inferred))
          resultTy = *inferred;
        else
          resultTy = ::mlir::flir::makeRankOnlyLayoutType(
              $_builder.getContext(),
              ::mlir::flir::getPatternRank(aTy.getShapePattern()) +
                  ::mlir::flir::getPatternRank(bTy.getShapePattern()));
      }
      build($_builder, $_state, resultTy, input, tiler);
    }]>
  ];
}


//===----------------------------------------------------------------------===//
// Layout Divide Operations
//===----------------------------------------------------------------------===//

def Flir_LogicalDivideOp : Flir_Op<"logical_divide", []> {
  let summary = "Logically divide a layout by a tiler";
  let description = [{
    Divides a layout logically by a tiler, creating a partitioned view.
    This is commonly used for tiling tensor operations.
    Example: %result = flir.logical_divide %input, %tiler : (!flir.layout<2>, !flir.layout<2>) -> !flir.layout<2>
  }];
  let arguments = (ins Flir_LayoutType:$input, Flir_LayoutType:$tiler);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$tiler), [{
      auto aTy = llvm::dyn_cast<::mlir::flir::LayoutType>(input.getType());
      auto bTy = llvm::dyn_cast<::mlir::flir::LayoutType>(tiler.getType());
      ::mlir::Type resultTy = input.getType();
      if (aTy && bTy) {
        auto inferred = ::mlir::flir::inferLogicalDivideType($_builder.getContext(), aTy, bTy);
        if (::mlir::succeeded(inferred))
          resultTy = *inferred;
        else
          resultTy = ::mlir::flir::makeRankOnlyLayoutType(
              $_builder.getContext(),
              std::max(::mlir::flir::getPatternRank(aTy.getShapePattern()),
                       ::mlir::flir::getPatternRank(bTy.getShapePattern())));
      }
      build($_builder, $_state, resultTy, input, tiler);
    }]>
  ];
}

def Flir_ZippedDivideOp : Flir_Op<"zipped_divide", []> {
  let summary = "Divide a layout by a tiler with zipped semantics";
  let description = [{
    Divides a layout by a tiler using zipped division semantics.
    This interleaves the division across modes.
    Example: %result = flir.zipped_divide %input, %tiler : (!flir.layout<2>, !flir.layout<2>) -> !flir.layout<2>
  }];
  let arguments = (ins Flir_LayoutType:$input, Flir_LayoutType:$tiler);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$tiler), [{
      // Current lowering treats zipped_divide the same as logical_divide.
      auto aTy = llvm::dyn_cast<::mlir::flir::LayoutType>(input.getType());
      auto bTy = llvm::dyn_cast<::mlir::flir::LayoutType>(tiler.getType());
      ::mlir::Type resultTy = input.getType();
      if (aTy && bTy) {
        auto inferred = ::mlir::flir::inferLogicalDivideType($_builder.getContext(), aTy, bTy);
        if (::mlir::succeeded(inferred))
          resultTy = *inferred;
        else
          resultTy = ::mlir::flir::makeRankOnlyLayoutType(
              $_builder.getContext(),
              std::max(::mlir::flir::getPatternRank(aTy.getShapePattern()),
                       ::mlir::flir::getPatternRank(bTy.getShapePattern())));
      }
      build($_builder, $_state, resultTy, input, tiler);
    }]>
  ];
}

def Flir_TiledDivideOp : Flir_Op<"tiled_divide", []> {
  let summary = "Divide a layout by a tiler with tiled semantics";
  let description = [{
    Divides a layout by a tiler creating a hierarchical tiled division.
    This is the dual of tiled_product.
    Example: %result = flir.tiled_divide %input, %tiler : (!flir.layout<2>, !flir.layout<2>) -> !flir.layout<2>
  }];
  let arguments = (ins Flir_LayoutType:$input, Flir_LayoutType:$tiler);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$tiler), [{
      // Current lowering treats tiled_divide the same as logical_divide.
      auto aTy = llvm::dyn_cast<::mlir::flir::LayoutType>(input.getType());
      auto bTy = llvm::dyn_cast<::mlir::flir::LayoutType>(tiler.getType());
      ::mlir::Type resultTy = input.getType();
      if (aTy && bTy) {
        auto inferred = ::mlir::flir::inferLogicalDivideType($_builder.getContext(), aTy, bTy);
        if (::mlir::succeeded(inferred))
          resultTy = *inferred;
        else
          resultTy = ::mlir::flir::makeRankOnlyLayoutType(
              $_builder.getContext(),
              std::max(::mlir::flir::getPatternRank(aTy.getShapePattern()),
                       ::mlir::flir::getPatternRank(bTy.getShapePattern())));
      }
      build($_builder, $_state, resultTy, input, tiler);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Advanced Layout Operations
//===----------------------------------------------------------------------===//

// NOTE: product_each and make_layout_tv are implemented at the Python level
// using existing low-level operations (raked_product, get_shape, size, composition, etc.)
// They don't need dedicated MLIR IR operations since they can be computed
// using combinations of existing operations.

def Flir_FlatDivideOp : Flir_Op<"flat_divide", []> {
  let summary = "Divide a layout by a tiler with flat semantics";
  let description = [{
    Divides a layout by a tiler creating a flattened division.
    This produces a flat result from the division.
    Example: %result = flir.flat_divide %input, %tiler : (!flir.layout<2>, !flir.layout<2>) -> !flir.layout<2>
  }];
  let arguments = (ins Flir_LayoutType:$input, Flir_LayoutType:$tiler);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler attr-dict `:` functional-type(operands, $result)";
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$tiler), [{
      // Current lowering treats flat_divide the same as logical_divide.
      auto aTy = llvm::dyn_cast<::mlir::flir::LayoutType>(input.getType());
      auto bTy = llvm::dyn_cast<::mlir::flir::LayoutType>(tiler.getType());
      ::mlir::Type resultTy = input.getType();
      if (aTy && bTy) {
        auto inferred = ::mlir::flir::inferLogicalDivideType($_builder.getContext(), aTy, bTy);
        if (::mlir::succeeded(inferred))
          resultTy = *inferred;
        else
          resultTy = ::mlir::flir::makeRankOnlyLayoutType(
              $_builder.getContext(),
              std::max(::mlir::flir::getPatternRank(aTy.getShapePattern()),
                       ::mlir::flir::getPatternRank(bTy.getShapePattern())));
      }
      build($_builder, $_state, resultTy, input, tiler);
    }]>
  ];
}

#endif // FLIR_OPS


//===----------------------------------------------------------------------===//
// Local Partition and Tile Operations
//===----------------------------------------------------------------------===//

def Flir_LocalPartitionOp : Flir_Op<"local_partition", []> {
  let summary = "Partition a layout for a specific thread/block index";
  let description = [{
    Partitions a layout by a tiler for a specific thread or block index.
    This is used to assign thread-local portions of a tensor.
    
    Similar to: outer_partition(tensor, tile, coord_to_index(tile, index))
    
    Example: %result = flir.local_partition %tensor, %tile, %index : (!flir.layout<2>, !flir.layout<2>, index) -> !flir.layout<2>
  }];
  let arguments = (ins Flir_LayoutType:$input, Flir_LayoutType:$tiler, Index:$index);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler `,` $index attr-dict `:` functional-type(operands, $result)";
}

def Flir_LocalTileOp : Flir_Op<"local_tile", []> {
  let summary = "Extract a tile from a layout at specific coordinates";
  let description = [{
    Extracts a tile of a specific shape from a layout at given coordinates.
    This is used for block-level tiling (e.g., CTA tiles in GEMM).
    
    The operation tiles the input tensor and selects the tile at the given coordinate.
    
    Example: %result = flir.local_tile %tensor, %tiler, %coord : (!flir.layout<2>, !flir.shape<2>, !flir.shape<2>) -> !flir.layout<2>
  }];
  let arguments = (ins Flir_LayoutType:$input, Flir_ShapeType:$tiler, Flir_ShapeType:$coord);
  let results = (outs Flir_LayoutType:$result);
  let assemblyFormat = "$input `,` $tiler `,` $coord attr-dict `:` functional-type(operands, $result)";
}

