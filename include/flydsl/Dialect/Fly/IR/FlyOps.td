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
// Type aliases
//===----------------------------------------------------------------------===//

def Fly_IteratorLikeType : AnyTypeOf<[Fly_IntTuple, Fly_Pointer]>;
def Fly_TensorLikeType : AnyTypeOf<[Fly_CoordTensor, Fly_MemRef]>;
def Fly_NarrowLayoutType : AnyTypeOf<[Fly_Layout, Fly_ComposedLayout]>;
def Fly_LayoutLikeType : AnyTypeOf<[Fly_NarrowLayoutType, Fly_TensorLikeType]>;
def Fly_IntTupleLikeType : AnyTypeOf<[Fly_IntTuple, Fly_LayoutLikeType]>;

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

def Fly_StaticOp : Fly_Op<"static", [Pure]> {  
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Fly_MakeIntTupleOp : Fly_Op<"make_int_tuple", [Pure]> {
  let arguments = (ins Variadic<AnyTypeOf<[I32,I64]>>:$dyncElems);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_MakeShapeOp : Fly_Op<"make_shape", [Pure]> {
  let arguments = (ins Variadic<AnyTypeOf<[I32,I64]>>:$dyncElems);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_MakeStrideOp : Fly_Op<"make_stride", [Pure]> {
  let arguments = (ins Variadic<AnyTypeOf<[I32,I64]>>:$dyncElems);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_MakeCoordOp : Fly_Op<"make_coord", [Pure]> {
  let arguments = (ins Variadic<AnyTypeOf<[I32,I64]>>:$dyncElems);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_MakeLayoutOp : Fly_Op<"make_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$shape, Fly_IntTuple:$stride);
  let results = (outs Fly_Layout:$result);
}

def Fly_MakeLayoutLikeOp : Fly_Op<"make_layout_like", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Layout:$ref);
  let results = (outs Fly_Layout:$result);
}

def Fly_MakeOrderedLayoutOp : Fly_Op<"make_ordered_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$shape, Fly_IntTuple:$order);
  let results = (outs Fly_Layout:$result);
}

def Fly_MakeComposedLayoutOp : Fly_Op<"make_composed_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_Layout, Fly_Swizzle, Fly_ComposedLayout]>:$inner,
                       Fly_IntTuple:$offset,
                       Fly_Layout:$outer);
  let results = (outs Fly_ComposedLayout:$result);
}

def Fly_MakeIdentityLayoutOp : Fly_Op<"make_identity_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$shape);
  let results = (outs Fly_Layout:$result);
}

def Fly_MakeViewOp : Fly_Op<"make_view", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IteratorLikeType:$iter, Fly_NarrowLayoutType:$layout);
  let results = (outs Fly_TensorLikeType:$result);
}

def Fly_MakeFragmentLikeOp : Fly_Op<"make_fragment_like", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_Layout, Fly_MemRef]>:$src, OptionalAttr<TypeAttr>:$dtype);
  let results = (outs Fly_MemRef:$result);
  let assemblyFormat = "`(` $src (`,` $dtype^)? `)` attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Extractors
//===----------------------------------------------------------------------===//

def Fly_GetScalarOp : Fly_Op<"get_scalar", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$int_tuple);
  let results = (outs AnyType:$result);
}

// TODO: return a list of leaves
def Fly_GetLeavesOp : Fly_Op<"get_leaves", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$input);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_GetShapeOp : Fly_Op<"get_shape", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$layout);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_GetStrideOp : Fly_Op<"get_stride", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$layout);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_GetLayoutOp : Fly_Op<"get_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TensorLikeType:$memref);
  let results = (outs Fly_NarrowLayoutType:$result);
}

def Fly_GetIterOp : Fly_Op<"get_iter", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TensorLikeType:$memref);
  let results = (outs Fly_IteratorLikeType:$result);
}


def Fly_ComposedGetInnerOp : Fly_Op<"composed_get_inner", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_ComposedLayout:$input);
  let results = (outs AnyType:$result);
}

def Fly_ComposedGetOffsetOp : Fly_Op<"composed_get_offset", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_ComposedLayout:$input);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_ComposedGetOuterOp : Fly_Op<"composed_get_outer", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_ComposedLayout:$input);
  let results = (outs Fly_Layout:$result);
}

//===----------------------------------------------------------------------===//
// IntTuple operations
//===----------------------------------------------------------------------===//

class Fly_IntTupleUnaryOp<string mnemonic>
    : Fly_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$input);
  let results = (outs Fly_IntTuple:$result);
}

class Fly_IntTupleBinaryOp<string mnemonic>
    : Fly_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$lhs, Fly_IntTuple:$rhs);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_IntTupleAddOp : Fly_IntTupleBinaryOp<"int_tuple_add">;
def Fly_IntTupleSubOp : Fly_IntTupleBinaryOp<"int_tuple_sub">;
def Fly_IntTupleMulOp : Fly_IntTupleBinaryOp<"int_tuple_mul">;
def Fly_IntTupleDivOp : Fly_IntTupleBinaryOp<"int_tuple_div">;
def Fly_IntTupleModOp : Fly_IntTupleBinaryOp<"int_tuple_mod">;

def Fly_IntTupleProductOp     : Fly_IntTupleUnaryOp<"int_tuple_product">;
def Fly_IntTupleProductEachOp : Fly_IntTupleUnaryOp<"int_tuple_product_each">;
def Fly_IntTupleProductLikeOp : Fly_IntTupleBinaryOp<"int_tuple_product_like">;

def Fly_ShapeDivOp : Fly_IntTupleBinaryOp<"shape_div">;
def Fly_CeilDivOp  : Fly_IntTupleBinaryOp<"ceil_div">;
def Fly_ElemLessOp : Fly_IntTupleBinaryOp<"elem_less">;
def Fly_EqualOp    : Fly_IntTupleBinaryOp<"equal">;

//===----------------------------------------------------------------------===//
// IntTupleLike operations
//===----------------------------------------------------------------------===//

def Fly_GetOp : Fly_Op<"get", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTupleLikeType:$input, DenseI32ArrayAttr:$mode);
  let results = (outs Fly_IntTupleLikeType:$result);
}
def Fly_TakeOp : Fly_Op<"take", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTupleLikeType:$tuple, I32Attr:$begin, I32Attr:$end);
  let results = (outs Fly_IntTupleLikeType:$result);
}
def Fly_SelectOp : Fly_Op<"select", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTupleLikeType:$tuple, DenseI32ArrayAttr:$indices);
  let results = (outs Fly_IntTupleLikeType:$result);
}
def Fly_GroupOp : Fly_Op<"group", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTupleLikeType:$tuple, I32Attr:$begin, I32Attr:$end);
  let results = (outs Fly_IntTupleLikeType:$result);
}

def Fly_AppendOp : Fly_Op<"append", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_IntTuple, Fly_NarrowLayoutType]>:$tuple,
                       AnyTypeOf<[Fly_IntTuple, Fly_Layout]>:$elem,
                       OptionalAttr<I32Attr>:$n);
  let results = (outs AnyTypeOf<[Fly_IntTuple, Fly_NarrowLayoutType]>:$result);
  let assemblyFormat = "(`<` $n^ `>`)? `(` operands `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_PrependOp : Fly_Op<"prepend", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_IntTuple, Fly_NarrowLayoutType]>:$tuple,
                       AnyTypeOf<[Fly_IntTuple, Fly_Layout]>:$elem,
                       OptionalAttr<I32Attr>:$n);
  let results = (outs AnyTypeOf<[Fly_IntTuple, Fly_NarrowLayoutType]>:$result);
  let assemblyFormat = "(`<` $n^ `>`)? `(` operands `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_SliceOp : Fly_Op<"slice", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTupleLikeType:$src, Fly_IntTuple:$coord);
  let results = (outs Fly_IntTupleLikeType:$result);
}
def Fly_DiceOp : Fly_Op<"dice", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTypeOf<[Fly_IntTuple, Fly_Layout]>:$src, Fly_IntTuple:$coord);
  let results = (outs AnyTypeOf<[Fly_IntTuple, Fly_Layout]>:$result);
}

//===----------------------------------------------------------------------===//
// LayoutLike operations
//===----------------------------------------------------------------------===//

def Fly_SizeOp : Fly_Op<"size", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTupleLikeType:$int_tuple);
  let results = (outs Fly_IntTuple:$result);
}
def Fly_CoprofileOp : Fly_Op<"coprofile", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$layout);
  let results = (outs Fly_IntTuple:$result);
}
def Fly_CoshapeOp : Fly_Op<"coshape", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$layout);
  let results = (outs Fly_IntTuple:$result);
}
def Fly_CosizeOp : Fly_Op<"cosize", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$layout);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_Crd2IdxOp : Fly_Op<"crd2idx", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$coord, Fly_NarrowLayoutType:$layout);
  let results = (outs Fly_IntTuple:$index);
}
def Fly_Idx2CrdOp : Fly_Op<"idx2crd", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$coord, Fly_Layout:$layout);
  let results = (outs Fly_IntTuple:$index);
}
def Fly_GetFlatCoordOp : Fly_Op<"get_flat_coord", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$index, Fly_Layout:$layout);
  let results = (outs Fly_IntTuple:$result);
}
def Fly_Get1DCoordOp : Fly_Op<"get_1d_coord", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_IntTuple:$index, Fly_Layout:$layout);
  let results = (outs Fly_IntTuple:$result);
}

def Fly_CoalesceOp : Fly_Op<"coalesce", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$layout, Optional<Fly_IntTuple>:$attr);
  let results = (outs Fly_LayoutLikeType:$result);
  let assemblyFormat = "`(` $layout (`,` $attr^)? `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_CompositionOp : Fly_Op<"composition", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$outer, AnyTypeOf<[Fly_Layout, Fly_Tile]>:$inner);
  let results = (outs Fly_LayoutLikeType:$result);
}
def Fly_ComplementOp : Fly_Op<"complement", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_NarrowLayoutType:$layout, Optional<Fly_IntTuple>:$codomain_size);
  let results = (outs Fly_NarrowLayoutType:$result);
  let assemblyFormat = "`(` $layout (`,` $codomain_size^)? `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_RightInverseOp : Fly_Op<"right_inverse", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_NarrowLayoutType:$layout);
  let results = (outs Fly_NarrowLayoutType:$result);
}
def Fly_LeftInverseOp : Fly_Op<"left_inverse", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_NarrowLayoutType:$layout);
  let results = (outs Fly_NarrowLayoutType:$result);
}

class Fly_LayoutDivideOp<string mnemonic>
    : Fly_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_LayoutLikeType:$layout, AnyTypeOf<[Fly_Layout, Fly_Tile]>:$divisor);
  let results = (outs Fly_LayoutLikeType:$result);
}
class Fly_LayoutProductOp<string mnemonic>
    : Fly_Op<mnemonic, [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_NarrowLayoutType:$layout, Fly_Layout:$tile);
  let results = (outs Fly_NarrowLayoutType:$result);
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

def Fly_RecastLayoutOp : Fly_Op<"recast_layout", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins I32Attr:$new_type_bits, I32Attr:$old_type_bits, Fly_NarrowLayoutType:$src);
  let results = (outs Fly_NarrowLayoutType:$result);
}

def Fly_TileToShapeOp : Fly_Op<"tile_to_shape", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_NarrowLayoutType:$block, Fly_IntTuple:$trg_shape, Fly_IntTuple:$ord_shape);
  let results = (outs Fly_NarrowLayoutType:$result);
}

//===----------------------------------------------------------------------===//
// Atom and Tiled Mma/Copy ops
//===----------------------------------------------------------------------===//

def Fly_MakeMmaAtomOp : Fly_Op<"make_mma_atom", [Pure]> {
  let arguments = (ins);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Fly_MakeCopyAtomOp : Fly_Op<"make_copy_atom", [Pure]> {
  let arguments = (ins I32Attr:$valBits);
  let results = (outs Fly_CopyAtom:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Fly_CopyAtomCall : Fly_Op<"copy_atom_call"> {
  let arguments = (ins Fly_CopyAtom:$copyAtom, Fly_MemRef:$src, Fly_MemRef:$dst, Optional<Fly_MemRef>:$pred);
}
def Fly_MmaAtomCall : Fly_Op<"mma_atom_call"> {
  let arguments = (ins AnyType:$mmaAtom, Fly_MemRef:$d, Fly_MemRef:$a, Fly_MemRef:$b, Fly_MemRef:$c);
}

def Fly_MakeTiledCopyOp : Fly_Op<"make_tiled_copy", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_CopyAtom:$copyAtom, Fly_Layout:$layoutThrVal, Fly_Tile:$tileMN);
  let results = (outs Fly_TiledCopy:$result);
}
def Fly_MakeTiledMmaOp : Fly_Op<"make_tiled_mma", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyType:$mmaAtom, Fly_Layout:$atomLayout, Optional<Fly_Tile>:$permutation);
  let results = (outs Fly_TiledMma:$result);
  let assemblyFormat = "`(` $mmaAtom `,` $atomLayout (`,` $permutation^)? `)` attr-dict `:` functional-type(operands, results)";
}

def Fly_TiledCopyPartitionSrcOp : Fly_Op<"tiled_copy.partition_src", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TiledCopy:$tiledCopy, Fly_MemRef:$src, Fly_IntTuple:$coord);
  let results = (outs Fly_MemRef:$result);
}
def Fly_TiledCopyPartitionDstOp : Fly_Op<"tiled_copy.partition_dst", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TiledCopy:$tiledCopy, Fly_MemRef:$dst, Fly_IntTuple:$coord);
  let results = (outs Fly_MemRef:$result);
}
def Fly_TiledCopyRetileOp : Fly_Op<"tiled_copy.retile", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_TiledCopy:$tiledCopy, Fly_MemRef:$input);
  let results = (outs Fly_MemRef:$result);
}

def Fly_TiledMmaPartitionOp : Fly_Op<"tiled_mma.partition", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_MmaOperandAttr:$operand_id, Fly_TiledMma:$tiled_mma,
                       Fly_MemRef:$input, Fly_IntTuple:$coord);
  let results = (outs Fly_MemRef:$result);
}
def Fly_TiledMmaPartitionShapeOp : Fly_Op<"tiled_mma.partition_shape", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_MmaOperandAttr:$operand_id, Fly_TiledMma:$tiled_mma, Fly_IntTuple:$shape);
  let results = (outs Fly_IntTuple:$result);
}
def Fly_MmaMakeFragmentOp : Fly_Op<"mma_atom.make_fragment", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_MmaOperandAttr:$operand_id, Fly_TiledMma:$tiled_mma, Fly_MemRef:$input);
  let results = (outs Fly_MemRef:$result);
}

def Fly_CopyOp : Fly_Op<"copy"> {
  let arguments = (ins AnyType:$copyAtom, Fly_MemRef:$src, Fly_MemRef:$dst, Optional<Fly_MemRef>:$pred);
  let assemblyFormat = "`(` $copyAtom `,` $src `,` $dst (`,` $pred^)? `)` attr-dict `:` functional-type(operands, results)";
}
def Fly_GemmOp : Fly_Op<"gemm"> {
  let arguments = (ins AnyType:$mmaAtom, Fly_MemRef:$d, Fly_MemRef:$a, Fly_MemRef:$b, Fly_MemRef:$c);
}

//===----------------------------------------------------------------------===//
// MemRef and Ptr operations
//===----------------------------------------------------------------------===//

def Fly_MemRefAllocaOp : Fly_Op<"memref.alloca", []> {
  let arguments = (ins Fly_Layout:$layout);
  let results = (outs Fly_MemRef:$result);
}

def Fly_MemRefLoadOp : Fly_Op<"memref.load", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_MemRef:$memref, Fly_IntTuple:$indices);
  let results = (outs AnyType:$result);
}
def Fly_MemRefStoreOp : Fly_Op<"memref.store", []> {
  let arguments = (ins AnyType:$value, Fly_MemRef:$memref, Fly_IntTuple:$indices);
}

def Fly_MemRefLoadVecOp : Fly_Op<"memref.load_vec", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_MemRef:$memref); // restrict to register tensor
  let results = (outs AnyType:$result);
}
def Fly_MemRefStoreVecOp : Fly_Op<"memref.store_vec", []> {
  let arguments = (ins AnyType:$vector, Fly_MemRef:$memref); // restrict to register tensor
  let results = (outs);
}


def Fly_MakePtrOp : Fly_Op<"make_ptr", [Pure]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs Fly_Pointer:$result);
}
def Fly_PtrLoadOp : Fly_Op<"ptr.load"> {
  let arguments = (ins Fly_Pointer:$ptr);
  let results = (outs AnyType:$result);
}
def Fly_PtrStoreOp : Fly_Op<"ptr.store"> {
  let arguments = (ins AnyType:$value, Fly_Pointer:$ptr);
}
def Fly_ApplySwizzleOp : Fly_Op<"apply_swizzle", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Pointer:$ptr, Fly_Swizzle:$swizzle);
  let results = (outs Fly_Pointer:$result);
}
def Fly_RecastIterOp : Fly_Op<"recast_iter", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Pointer:$src);
  let results = (outs Fly_Pointer:$result);
}

def Fly_AddOffsetOp : Fly_Op<"add_offset", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Fly_Pointer:$ptr, Fly_IntTuple:$offset);
  let results = (outs Fly_Pointer:$result);
}

//===----------------------------------------------------------------------===//
// Utility ops
//===----------------------------------------------------------------------===//

def Fly_PrintOp : Fly_Op<"print"> {
  let arguments = (ins StrAttr:$format, Variadic<AnyType>:$values);
}

def Fly_AssumeOp : Fly_Op<"assume", [Pure]> {
  let arguments = (ins AnyType:$dst, AnyType:$src);
  let results = (outs AnyType:$result);
}

//===----------------------------------------------------------------------===//
// Deprecated, removed in the future
//===----------------------------------------------------------------------===//

def Fly_ExtractAlignedPointerAsIndexOp : Fly_Op<"extract_aligned_pointer_as_index", [Pure]> {
  let summary = "Extract the raw pointer from a fly.memref";
  let arguments = (ins Fly_MemRef:$source);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

// TODO: TileAttr is always static, change to StaticOp
def Fly_MakeTileOp: Fly_Op<"make_tile", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Variadic<AnyTypeOf<[Fly_IntTuple, Fly_Layout]>>:$modes);
  let results = (outs Fly_Tile:$result);
}

#endif // FLY_OPS
