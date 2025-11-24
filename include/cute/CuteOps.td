//===- CuteOps.td - Cute operation definitions ------------*- tablegen -*-===//
//
// Simplified Cute operations for modern MLIR
//
//===----------------------------------------------------------------------===//

#ifndef CUTE_OPS
#define CUTE_OPS

include "CuteDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Cute operation definitions
//===----------------------------------------------------------------------===//

class Cute_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cute_Dialect, mnemonic, traits>;

// Type constraints - manual for now since we're not using TypeDef
def Cute_IntType : Type<CPred<"$_self.isa<::mlir::cute::IntType>()">, "cute.int">;
def Cute_ShapeType : Type<CPred<"$_self.isa<::mlir::cute::ShapeType>()">, "cute.shape">;
def Cute_StrideType : Type<CPred<"$_self.isa<::mlir::cute::StrideType>()">, "cute.stride">;
def Cute_LayoutType : Type<CPred<"$_self.isa<::mlir::cute::LayoutType>()">, "cute.layout">;
def Cute_CoordType : Type<CPred<"$_self.isa<::mlir::cute::CoordType>()">, "cute.coord">;

//===----------------------------------------------------------------------===//
// MakeShapeOp
//===----------------------------------------------------------------------===//

def Cute_MakeShapeOp : Cute_Op<"make_shape", []> {
  let summary = "Create a CUTE shape from integer values";
  let description = [{
    Creates a shape value from compile-time integer values.
  }];

  let arguments = (ins Variadic<Cute_IntType>:$values);
  let results = (outs Cute_ShapeType:$result);

  let assemblyFormat = "$values attr-dict `:` functional-type($values, $result)";
}

//===----------------------------------------------------------------------===//
// MakeStrideOp
//===----------------------------------------------------------------------===//

def Cute_MakeStrideOp : Cute_Op<"make_stride", []> {
  let summary = "Create a CUTE stride from integer values";
  let description = [{
    Creates a stride value from compile-time integer values.
  }];

  let arguments = (ins Variadic<Cute_IntType>:$values);
  let results = (outs Cute_StrideType:$result);

  let assemblyFormat = "$values attr-dict `:` functional-type($values, $result)";
}

//===----------------------------------------------------------------------===//
// MakeLayoutOp
//===----------------------------------------------------------------------===//

def Cute_MakeLayoutOp : Cute_Op<"make_layout", []> {
  let summary = "Create a CUTE layout from shape and stride";
  let description = [{
    Creates a layout value by combining a shape and stride.
  }];

  let arguments = (ins 
    Cute_ShapeType:$shape,
    Cute_StrideType:$stride
  );
  let results = (outs Cute_LayoutType:$result);

  let assemblyFormat = "$shape `,` $stride attr-dict `:` functional-type(operands, $result)";
}

//===----------------------------------------------------------------------===//
// MakeCoordOp
//===----------------------------------------------------------------------===//

def Cute_MakeCoordOp : Cute_Op<"make_coord", []> {
  let summary = "Create a CUTE coordinate from integer values";
  let description = [{
    Creates a coordinate value from integer values.
  }];

  let arguments = (ins Variadic<Cute_IntType>:$values);
  let results = (outs Cute_CoordType:$result);

  let assemblyFormat = "$values attr-dict `:` functional-type($values, $result)";
}

//===----------------------------------------------------------------------===//
// SizeOp
//===----------------------------------------------------------------------===//

def Cute_SizeOp : Cute_Op<"size", []> {
  let summary = "Get the total size of a CUTE shape";
  let description = [{
    Computes the product of all dimensions in a shape.
  }];

  let arguments = (ins Cute_ShapeType:$shape);
  let results = (outs Cute_IntType:$result);

  let assemblyFormat = "$shape attr-dict `:` type($shape) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Crd2IdxOp
//===----------------------------------------------------------------------===//

def Cute_Crd2IdxOp : Cute_Op<"crd2idx", []> {
  let summary = "Convert coordinate to linear index using layout";
  let description = [{
    Converts a multi-dimensional coordinate to a linear index using a layout.
  }];

  let arguments = (ins 
    Cute_CoordType:$coord,
    Cute_LayoutType:$layout
  );
  let results = (outs Cute_IntType:$result);

  let assemblyFormat = "$coord `,` $layout attr-dict `:` functional-type(operands, $result)";
}

#endif // CUTE_OPS
