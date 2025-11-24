//===- CuteOps.td - Cute operation definitions ------------*- tablegen -*-===//

#ifndef CUTE_OPS
#define CUTE_OPS

include "CuteDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Cute_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cute_Dialect, mnemonic, traits>;

// Type constraints
def Cute_IntType : Type<CPred<"llvm::isa<::mlir::cute::IntType>($_self)">, "cute.int">;
def Cute_ShapeType : Type<CPred<"llvm::isa<::mlir::cute::ShapeType>($_self)">, "cute.shape">;
def Cute_StrideType : Type<CPred<"llvm::isa<::mlir::cute::StrideType>($_self)">, "cute.stride">;
def Cute_LayoutType : Type<CPred<"llvm::isa<::mlir::cute::LayoutType>($_self)">, "cute.layout">;
def Cute_CoordType : Type<CPred<"llvm::isa<::mlir::cute::CoordType>($_self)">, "cute.coord">;

def Cute_MakeShapeOp : Cute_Op<"make_shape", []> {
  let summary = "Create a CUTE shape from integer values";
  let arguments = (ins Variadic<Index>:$values);
  let results = (outs Cute_ShapeType:$result);
  let assemblyFormat = "$values attr-dict `:` functional-type($values, $result)";
}

def Cute_MakeStrideOp : Cute_Op<"make_stride", []> {
  let summary = "Create a CUTE stride from integer values";
  let arguments = (ins Variadic<Index>:$values);
  let results = (outs Cute_StrideType:$result);
  let assemblyFormat = "$values attr-dict `:` functional-type($values, $result)";
}

def Cute_MakeLayoutOp : Cute_Op<"make_layout", []> {
  let summary = "Create a CUTE layout from shape and stride";
  let arguments = (ins Cute_ShapeType:$shape, Cute_StrideType:$stride);
  let results = (outs Cute_LayoutType:$result);
  let assemblyFormat = "$shape `,` $stride attr-dict `:` functional-type(operands, $result)";
}

def Cute_MakeCoordOp : Cute_Op<"make_coord", []> {
  let summary = "Create a CUTE coordinate from integer values";
  let arguments = (ins Variadic<Index>:$values);
  let results = (outs Cute_CoordType:$result);
  let assemblyFormat = "$values attr-dict `:` functional-type($values, $result)";
}

def Cute_SizeOp : Cute_Op<"size", []> {
  let summary = "Get the total size of a CUTE shape";
  let arguments = (ins Cute_ShapeType:$shape);
  let results = (outs Index:$result);
  let assemblyFormat = "$shape attr-dict `:` type($shape) `->` type($result)";
}

def Cute_Crd2IdxOp : Cute_Op<"crd2idx", []> {
  let summary = "Convert coordinate to linear index using layout";
  let arguments = (ins Cute_CoordType:$coord, Cute_LayoutType:$layout);
  let results = (outs Index:$result);
  let assemblyFormat = "$coord `,` $layout attr-dict `:` functional-type(operands, $result)";
}

#endif // CUTE_OPS
