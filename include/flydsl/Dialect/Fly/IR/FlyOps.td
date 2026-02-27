#ifndef FLY_OPS
#define FLY_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Bytecode/BytecodeOpInterface.td"

include "flydsl/Dialect/Fly/IR/FlyDialect.td"
include "flydsl/Dialect/Fly/IR/FlyTypeDefs.td"
include "flydsl/Dialect/Fly/IR/FlyAttrDefs.td"


//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

def Fly_StaticOp : Fly_Op<"static", [Pure]> {  
  let arguments = (ins);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeIntTupleOp : Fly_Op<"make_int_tuple", [Pure]> {
  let arguments = (ins Variadic<AnyTypeOf<[I32,I64]>>:$dyncElems);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "`(` $dyncElems `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeShapeOp : Fly_Op<"make_shape", [Pure]> {
  let arguments = (ins Variadic<AnyTypeOf<[I32,I64]>>:$dyncElems);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "`(` $dyncElems `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeStrideOp : Fly_Op<"make_stride", [Pure]> {
  let arguments = (ins Variadic<AnyTypeOf<[I32,I64]>>:$dyncElems);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "`(` $dyncElems `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeCoordOp : Fly_Op<"make_coord", [Pure]> {
  let arguments = (ins Variadic<AnyTypeOf<[I32,I64]>>:$dyncElems);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "`(` $dyncElems `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeLayoutOp : Fly_Op<"make_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$shape, Optional<Fly_IntTuple>:$stride);
  let results = (outs Fly_Layout:$result);
  let assemblyFormat = "`(` $shape (`,` $stride^)? `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeTileOp: Fly_Op<"make_tile", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Variadic<AnyTypeOf<[Fly_IntTuple, Fly_Layout]>>:$modes);
  let results = (outs Fly_Tile:$result);
  let assemblyFormat = "`(` $modes `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeViewOp : Fly_Op<"make_view", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins IteratorLikeType:$iter, AnyLayoutType:$layout);
  let results = (outs TensorLikeType:$result);
  let assemblyFormat = "`(` $iter `,` $layout `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeLayoutLikeOp : Fly_Op<"make_layout_like", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Layout:$src);
  let results = (outs Fly_Layout:$result);
  let assemblyFormat = "`(` $src `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeOrderedLayoutOp : Fly_Op<"make_ordered_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$shape, Fly_IntTuple:$order);
  let results = (outs Fly_Layout:$result);
  let assemblyFormat = "`(` $shape `,` $order `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeComposedLayoutOp : Fly_Op<"make_composed_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyLayoutType:$inner,
                       Fly_IntTuple:$offset,
                       Fly_Layout:$outer);
  let results = (outs Fly_ComposedLayout:$result);
  let assemblyFormat = "`(` $inner `,` $offset `,` $outer `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeIdentityLayoutOp : Fly_Op<"make_identity_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$shape);
  let results = (outs Fly_Layout:$result);
  let assemblyFormat = "`(` $shape `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeIdentityTensorOp : Fly_Op<"make_identity_tensor", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$shape);
  let results = (outs Fly_CoordTensor:$result);
  let assemblyFormat = "`(` $shape `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeFragmentLikeOp : Fly_Op<"make_fragment_like", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Layout:$src);
  let results = (outs Fly_MemRef:$result);
  let assemblyFormat = "`(` $src `)` attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Extractors
//===----------------------------------------------------------------------===//

def Fly_GetOp : Fly_Op<"get", [Pure]> {
  let arguments = (ins AnyTypeOf<[Fly_IntTuple, Fly_Layout, Fly_MemRef]>:$input,
                       OptionalAttr<I32Attr>:$mode);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`(` $input (`,` $mode^)? `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_GetScalarOp : Fly_Op<"get_scalar", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$int_tuple);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`(` $int_tuple `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_GetLeavesOp : Fly_Op<"get_leaves", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_IntTuple, Fly_Layout]>:$input);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "`(` $input `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_GetShapeOp : Fly_Op<"get_shape", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Layout:$layout);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "`(` $layout `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_GetStrideOp : Fly_Op<"get_stride", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Layout:$layout);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "`(` $layout `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_GetLayoutOp : Fly_Op<"get_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_MemRef:$memref);
  let results = (outs Fly_Layout:$result);
  let assemblyFormat = "`(` $memref `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_GetIterOp : Fly_Op<"get_iter", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_MemRef:$memref);
  let results = (outs Fly_Pointer:$result);
  let assemblyFormat = "`(` $memref `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_GetLayoutsFromTileOp : Fly_Op<"get_layouts_from_tile", [Pure]> {
  let arguments = (ins Fly_Tile:$tile);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`(` $tile `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_GetLeafOp : Fly_Op<"get_leaf", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Get a leaf element from an IntTuple or Layout";
  let description = [{}];

  let arguments = (ins AnyTypeOf<[Fly_IntTuple, Fly_Layout]>:$tuple, I32Attr:$leaf_idx);
  let results = (outs AnyTypeOf<[Fly_IntTuple, Fly_Layout]>:$leaf);
  let assemblyFormat = "`(` $tuple `,` $leaf_idx `)` attr-dict `:` functional-type(operands, $leaf)";
}

def Fly_ComposedGetInnerOp : Fly_Op<"composed_get_inner", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_ComposedLayout:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`(` $input `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_ComposedGetOffsetOp : Fly_Op<"composed_get_offset", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_ComposedLayout:$input);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "`(` $input `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_ComposedGetOuterOp : Fly_Op<"composed_get_outer", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_ComposedLayout:$input);
  let results = (outs Fly_Layout:$result);
  let assemblyFormat = "`(` $input `)` attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// IntTuple operations
//===----------------------------------------------------------------------===//

class Fly_IntTupleUnaryOp<string mnemonic>
    : Fly_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$input);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "`(` $input `)` attr-dict `:` functional-type(operands, results)";
}

class Fly_IntTupleBinaryOp<string mnemonic>
    : Fly_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$lhs, Fly_IntTuple:$rhs);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "`(` $lhs `,` $rhs `)` attr-dict `:` functional-type(operands, results)";
}

class Fly_IntTupleUnaryWithProfileOp<string mnemonic>
    : Fly_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$input, Optional<Fly_IntTuple>:$target_profile);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "`(` $input (`,` $target_profile^)? `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_IntTupleAddOp : Fly_IntTupleBinaryOp<"int_tuple_add">;
def Fly_IntTupleSubOp : Fly_IntTupleBinaryOp<"int_tuple_sub">;
def Fly_IntTupleMulOp : Fly_IntTupleBinaryOp<"int_tuple_mul">;
def Fly_IntTupleDivOp : Fly_IntTupleBinaryOp<"int_tuple_div">;
def Fly_IntTupleModOp : Fly_IntTupleBinaryOp<"int_tuple_mod">;

def Fly_IntTupleProductEachOp : Fly_IntTupleUnaryOp<"int_tuple_product_each">;
def Fly_IntTupleProductOp     : Fly_IntTupleUnaryOp<"int_tuple_product">;

def Fly_ShapeDivOp : Fly_IntTupleBinaryOp<"shape_div">;
def Fly_CeilDivOp  : Fly_IntTupleBinaryOp<"ceil_div">;
def Fly_ElemLessOp : Fly_IntTupleBinaryOp<"elem_less">;
def Fly_EqualOp    : Fly_IntTupleBinaryOp<"equal">;

def Fly_AppendOp : Fly_Op<"append", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Layout:$tuple,
                       Fly_Layout:$elem,
                       OptionalAttr<I32Attr>:$n);
  let results = (outs Fly_Layout:$result);
  let assemblyFormat = "(`<` $n^ `>`)? `(` $tuple `,` $elem `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_PrependOp : Fly_Op<"prepend", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Layout:$tuple,
                       Fly_Layout:$elem,
                       OptionalAttr<I32Attr>:$n);
  let results = (outs Fly_Layout:$result);
  let assemblyFormat = "(`<` $n^ `>`)? `(` $tuple `,` $elem `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_SelectOp : Fly_Op<"select", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Select elements from an IntTuple or Layout by indices";
  let description = [{}];

  let arguments = (ins IntTupleLikeType:$tuple, DenseI32ArrayAttr:$indices);
  let results = (outs IntTupleLikeType:$result);
  let assemblyFormat = "`(` $tuple `,` $indices `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_GroupOp : Fly_Op<"group", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Group elements in an IntTuple or Layout";
  let description = [{}];

  let arguments = (ins AnyTypeOf<[Fly_IntTuple, Fly_Layout, Fly_MemRef]>:$tuple, I32Attr:$begin, I32Attr:$end);
  let results = (outs AnyTypeOf<[Fly_IntTuple, Fly_Layout, Fly_MemRef]>:$result);
  let assemblyFormat = "`(` $tuple `,` $begin `,` $end `)` attr-dict `:` functional-type(operands, results)";
}


def Fly_SliceOp : Fly_Op<"slice", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_IntTuple, Fly_Layout, Fly_MemRef]>:$src, Fly_IntTuple:$coord);
  let results = (outs AnyTypeOf<[Fly_IntTuple, Fly_Layout, Fly_MemRef]>:$result);
  let assemblyFormat = "`(` $src `,` $coord  `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_DiceOp : Fly_Op<"dice", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_IntTuple, Fly_Layout, Fly_MemRef]>:$src, Fly_IntTuple:$coord);
  let results = (outs AnyTypeOf<[Fly_IntTuple, Fly_Layout, Fly_MemRef]>:$result);
  let assemblyFormat = "`(` $src `,` $coord `)` attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Layout operations
//===----------------------------------------------------------------------===//

def Fly_SizeOp : Fly_Op<"size", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_IntTuple, Fly_Layout, Fly_MemRef]>:$int_tuple);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "`(` $int_tuple `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_CosizeOp : Fly_Op<"cosize", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Layout:$layout);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "$layout attr-dict `:` functional-type(operands, results)";
}

def Fly_Crd2IdxOp : Fly_Op<"crd2idx", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$coord, Fly_Layout:$layout);
  let results = (outs Fly_IntTuple:$index);
  let assemblyFormat = "`(` $coord `,` $layout `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_Idx2CrdOp : Fly_Op<"idx2crd", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$coord, Fly_Layout:$layout);
  let results = (outs Fly_IntTuple:$index);
  let assemblyFormat = "`(` $coord `,` $layout `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_GetFlatCoordOp : Fly_Op<"get_flat_coord", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$index, AnyTypeOf<[Fly_IntTuple, Fly_Layout]>:$input);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "`(` $index `,` $input `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_GetHierCoordOp : Fly_Op<"get_hier_coord", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$index, AnyTypeOf<[Fly_IntTuple, Fly_Layout]>:$input);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "`(` $index `,` $input `)` attr-dict `:` functional-type(operands, results)";
}


def Fly_CoalesceOp : Fly_Op<"coalesce", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Layout:$layout, Optional<Fly_IntTuple>:$attr);
  let results = (outs Fly_Layout:$result);
  let assemblyFormat = "`(` $layout (`,` $attr^)? `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_CompositionOp : Fly_Op<"composition", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Layout:$outer, AnyTypeOf<[Fly_Layout, Fly_Tile]>:$inner);
  let results = (outs Fly_Layout:$result);
  let assemblyFormat = "`(` $outer `,` $inner `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_ComplementOp : Fly_Op<"complement", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Layout:$layout, Optional<Fly_IntTuple>:$codomain_size);
  let results = (outs Fly_Layout:$result);
  let assemblyFormat = "`(` $layout (`,` $codomain_size^)? `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_RightInverseOp : Fly_Op<"right_inverse", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Layout:$layout);
  let results = (outs Fly_Layout:$result);
  let assemblyFormat = "`(` $layout `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_LeftInverseOp : Fly_Op<"left_inverse", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Layout:$layout);
  let results = (outs Fly_Layout:$result);
  let assemblyFormat = "`(` $layout `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_RecastLayoutOp : Fly_Op<"recast_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins I32Attr:$new_type_bits,
                       I32Attr:$old_type_bits,
                       Fly_Layout:$src);
  let results = (outs Fly_Layout:$result);
  let assemblyFormat = "`(` $new_type_bits `,` $old_type_bits `,` $src `)` attr-dict `:` functional-type(operands, results)";
}

class Fly_LayoutDivideOp<string mnemonic>
    : Fly_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins LayoutLikeType:$layout, AnyTypeOf<[Fly_Layout, Fly_Tile]>:$divisor);
  let results = (outs LayoutLikeType:$result);
  let assemblyFormat = "`(` $layout `,` $divisor `)` attr-dict `:` functional-type(operands, results)";
}

class Fly_LayoutProductOp<string mnemonic>
    : Fly_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins LayoutLikeType:$layout, Fly_Layout:$tile);
  let results = (outs LayoutLikeType:$result);
  let assemblyFormat = "`(` $layout `,` $tile `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_LogicalDivideOp : Fly_LayoutDivideOp<"logical_divide">;
def Fly_ZippedDivideOp : Fly_LayoutDivideOp<"zipped_divide">;
def Fly_TiledDivideOp : Fly_LayoutDivideOp<"tiled_divide">;
def Fly_FlatDivideOp : Fly_LayoutDivideOp<"flat_divide">;

def Fly_LogicalProductOp : Fly_LayoutProductOp<"logical_product">;
def Fly_ZippedProductOp : Fly_LayoutProductOp<"zipped_product">;
def Fly_TiledProductOp : Fly_LayoutProductOp<"tiled_product">;
def Fly_FlatProductOp : Fly_LayoutProductOp<"flat_product">;
def Fly_BlockedProductOp : Fly_LayoutProductOp<"blocked_product">;
def Fly_RakedProductOp : Fly_LayoutProductOp<"raked_product">;

def Fly_TileToShapeOp : Fly_Op<"tile_to_shape", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_Layout, Fly_Tile]>:$block,
                       Fly_IntTuple:$trg_shape,
                       Fly_IntTuple:$ord_shape);
  let results = (outs Fly_Layout:$result);
  let assemblyFormat = "`(` $block `,` $trg_shape `,` $ord_shape `)` attr-dict `:` functional-type(operands, results)";
}


//===----------------------------------------------------------------------===//
// Atom and Tiled Mma/Copy ops
//===----------------------------------------------------------------------===//
def Fly_MakeMmaAtomOp : Fly_Op<"make_mma_atom", [Pure]> {
  let arguments = (ins);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeCopyAtomOp : Fly_Op<"make_copy_atom", [Pure]> {
  let arguments = (ins I32Attr:$valBits);
  let results = (outs Fly_CopyAtom:$result);
  let assemblyFormat = "attr-dict `:` functional-type(operands, results)";
}

def Fly_CopyAtomCall : Fly_Op<"copy_atom_call"> {
  let arguments = (ins AnyType:$copyAtom, Fly_MemRef:$src, Fly_MemRef:$dst);
  let results = (outs);
  let assemblyFormat = "`(` $copyAtom `,` $src `,` $dst `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_MmaAtomCall : Fly_Op<"mma_atom_call"> {
  let arguments = (ins AnyType:$mmaAtom, Fly_MemRef:$d, Fly_MemRef:$a, Fly_MemRef:$b, Fly_MemRef:$c);
  let results = (outs);
  let assemblyFormat = "`(` $mmaAtom `,` $d `,` $a `,` $b `,` $c `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MakeTiledCopyOp : Fly_Op<"make_tiled_copy", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyType:$copyAtom, Fly_Layout:$layoutThrVal, Fly_Tile:$tileMN);
  let results = (outs Fly_TiledCopy:$result);
  let assemblyFormat = "`(` $copyAtom `,` $layoutThrVal  `,` $tileMN `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_MakeTiledMmaOp : Fly_Op<"make_tiled_mma", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyType:$mmaAtom, Fly_Layout:$atomLayout, Optional<Fly_Tile>:$permutation);
  let results = (outs Fly_TiledMma:$result);
  let assemblyFormat = "`(` $mmaAtom `,` $atomLayout (`,` $permutation^)? `)` attr-dict `:` functional-type(operands, results)";
}


def Fly_TiledCopyPartitionSrcOp : Fly_Op<"tiled_copy.partition_src", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TiledCopy:$tiledCopy, Fly_MemRef:$src, Fly_IntTuple:$coord);
  let results = (outs Fly_MemRef:$result);
  let assemblyFormat = "`(` $tiledCopy `,` $src `,` $coord `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_TiledCopyPartitionDstOp : Fly_Op<"tiled_copy.partition_dst", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TiledCopy:$tiledCopy, Fly_MemRef:$dst, Fly_IntTuple:$coord);
  let results = (outs Fly_MemRef:$result);
  let assemblyFormat = "`(` $tiledCopy `,`  $dst `,` $coord `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_TiledCopyRetileOp : Fly_Op<"tiled_copy.retile", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TiledCopy:$tiledCopy, Fly_MemRef:$input);
  let results = (outs Fly_MemRef:$result);
  let assemblyFormat = "`(` $tiledCopy `,` $input `)` attr-dict `:` functional-type(operands, results)";
}


def Fly_CopyOp : Fly_Op<"copy"> {
  let arguments = (ins AnyType:$copyAtom, Fly_MemRef:$src, Fly_MemRef:$dst, OptionalAttr<I1Attr>:$pred);
  let results = (outs);
  let assemblyFormat = "`(` $copyAtom `,` $src `,` $dst (`,` $pred^)? `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_GemmOp : Fly_Op<"gemm"> {
  let arguments = (ins AnyType:$mmaAtom, Fly_MemRef:$d, Fly_MemRef:$a,
                       Fly_MemRef:$b, Fly_MemRef:$c);
  let results = (outs);
  let assemblyFormat = "`(` $mmaAtom `,` $d `,` $a `,` $b `,` $c `)` attr-dict `:` functional-type(operands, results)";
}



def Fly_MmaMakeFragmentOp : Fly_Op<"mma_make_fragment", [Pure]> {
  let arguments = (ins AnyTypeOf<[I32, I64]>:$operand_id, AnyType:$atom, AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`(` $operand_id `,` $atom `,` $input `)` attr-dict `:` functional-type(operands, results)";
}


def Fly_TiledMmaPartitionOp : Fly_Op<"tiled_mma.partition", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_MmaOperandAttr:$operand_id, AnyType:$tiled_mma,
                       Fly_MemRef:$input, Fly_IntTuple:$coord);
  let results = (outs Fly_MemRef:$result);
  let assemblyFormat = "`(` $operand_id `,` $tiled_mma `,` $input `,` $coord `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_TiledMmaPartitionShapeOp : Fly_Op<"tiled_mma_partition_shape", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_MmaOperandAttr:$operand_id, AnyType:$tiled_mma, Fly_MemRef:$input);
  let results = (outs Fly_IntTuple:$result);
  let assemblyFormat = "`(` $operand_id `,` $tiled_mma `,` $input `)` attr-dict `:` functional-type(operands, results)";
}


//===----------------------------------------------------------------------===//
// MemRef and Ptr operations
//===----------------------------------------------------------------------===//

def Fly_MemRefAllocaOp : Fly_Op<"memref.alloca", []> {
  let arguments = (ins Fly_Layout:$layout);
  let results = (outs Fly_MemRef:$result);
  let assemblyFormat = "`(` $layout `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MemRefLoadOp : Fly_Op<"memref.load", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_MemRef:$memref, Fly_IntTuple:$indices);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`(` $memref `,` $indices `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_MemRefStoreOp : Fly_Op<"memref.store", []> {
  let arguments = (ins AnyType:$value, Fly_MemRef:$memref, Fly_IntTuple:$indices);
  let assemblyFormat = "`(` $value `,` $memref `,` $indices `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_MemRefLoadVecOp : Fly_Op<"memref.load_vec", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_MemRef:$memref); // restrict to register tensor
  let results = (outs AnyType:$result);
  let assemblyFormat = "`(` $memref `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_MemRefStoreVecOp : Fly_Op<"memref.store_vec", []> {
  let arguments = (ins AnyType:$vector, Fly_MemRef:$memref); // restrict to register tensor
  let results = (outs);
  let assemblyFormat = "`(` $vector `,` $memref `)` attr-dict `:` functional-type(operands, results)";
}


def Fly_MakePtrOp : Fly_Op<"make_ptr", [Pure]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs Fly_Pointer:$result);
  let assemblyFormat = "`(` $args `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_PtrLoadOp : Fly_Op<"ptr.load"> {
  let arguments = (ins Fly_Pointer:$ptr);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`(` $ptr `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_PtrStoreOp : Fly_Op<"ptr.store"> {
  let arguments = (ins AnyType:$value, Fly_Pointer:$ptr);
  let assemblyFormat = "`(` $value `,` $ptr `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_ApplySwizzleOp : Fly_Op<"apply_swizzle", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Pointer:$ptr);
  let results = (outs Fly_Pointer:$result);
  let assemblyFormat = "`(` $ptr `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_RecastIterOp : Fly_Op<"recast_iter", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Pointer:$src);
  let results = (outs Fly_Pointer:$result);
  let assemblyFormat = "`(` $src `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_AddOffsetOp : Fly_Op<"add_offset", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins IteratorLikeType:$ptr, Fly_IntTuple:$offset);
  let results = (outs IteratorLikeType:$result);
  let assemblyFormat = "`(` $ptr `,` $offset `)` attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Utility ops
//===----------------------------------------------------------------------===//

def Fly_PrintOp : Fly_Op<"print"> {
  let arguments = (ins StrAttr:$format, Variadic<AnyType>:$values);
  let assemblyFormat = "`(` $values `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_AssumeOp : Fly_Op<"assume", [Pure]> {
  let arguments = (ins AnyType:$dst, AnyType:$src);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`(` $dst `,` $src `)` attr-dict `:` functional-type(operands, results)";
}

#endif // FLY_OPS
