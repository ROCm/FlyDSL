// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLY_INTERFACES 
#define FLY_INTERFACES

include "mlir/IR/OpBase.td"

def Fly_NestedInterfaceMethods {
  list<InterfaceMethod> methods = [
    InterfaceMethod<"", "bool", "isLeaf", (ins)>,
    InterfaceMethod<"", "int32_t", "rank", (ins)>,
    InterfaceMethod<"", "int32_t", "rank", (ins "int32_t":$idx)>,
    InterfaceMethod<"", "int32_t", "rank", (ins "::llvm::ArrayRef<int32_t>":$idxs)>,
    InterfaceMethod<"", "int32_t", "depth", (ins)>,
    InterfaceMethod<"", "int32_t", "depth", (ins "int32_t":$idx)>,
    InterfaceMethod<"", "int32_t", "depth", (ins "::llvm::ArrayRef<int32_t>":$idxs)>
  ];
}

def Fly_MayStaticInterface {
  list<InterfaceMethod> methods = [
    InterfaceMethod<"", "bool", "isStatic", (ins)>
  ];
}

def Fly_StatefulOpTypeInterface : TypeInterface<"StatefulOpTypeInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = [
    InterfaceMethod<
      "Return the converted type that holds the mutable state of this op.",
      "::mlir::Type", "getConvertedType", (ins "::mlir::MLIRContext *":$ctx)>,
    InterfaceMethod<
      "Build the default state value for this op.",
      "::mlir::Value", "getDefaultState",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc)>,
    InterfaceMethod<
      "Insert the given value to a field of this stateful op. "
      "Returns the updated struct value after setting the field.",
      "::mlir::Value",
      "setAtomState",
      (ins "::mlir::OpBuilder &":$builder,
           "::mlir::Location":$loc,
           "::mlir::Value":$atomStruct,
           "::mlir::Attribute":$fieldAttr,
           "::mlir::Value":$fieldValue)>
  ];
}

def Fly_NestedAttrInterface : AttrInterface<"NestedAttrInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = Fly_NestedInterfaceMethods.methods;
}
def Fly_NestedTypeInterface : TypeInterface<"NestedTypeInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = Fly_NestedInterfaceMethods.methods;
}

def Fly_MayStaticAttrInterface : AttrInterface<"MayStaticAttrInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = Fly_MayStaticInterface.methods;
}
def Fly_MayStaticTypeInterface : TypeInterface<"MayStaticTypeInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = !listconcat(Fly_MayStaticInterface.methods, [
    InterfaceMethod<
      "Rebuild to a normal form op that produces a value of this static type. "
      "Returns nullptr if the given value is already in normal form.",
      "::mlir::Value", 
      "rebuildStaticValue",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc, "::mlir::Value":$currentValue)>
  ]);
}

def Fly_CopyOpTypeInterface : TypeInterface<"CopyOpTypeInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = [
    InterfaceMethod<"", "::mlir::Attribute", "getThrLayout", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrBitLayoutSrc", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrBitLayoutDst", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrBitLayoutRef", (ins)>,
    InterfaceMethod<
      "Emit the lowering IR for a copy_atom_call with this CopyOp type.",
      "::mlir::LogicalResult",
      "emitAtomCall",
      (ins "::mlir::OpBuilder &":$builder,
           "::mlir::Location":$loc,
           "::mlir::Type":$copyAtomTy,
           "::mlir::Type":$srcMemTy,
           "::mlir::Type":$dstMemTy,
           "::mlir::Value":$atomVal,
           "::mlir::Value":$src,
           "::mlir::Value":$dst)>,
    InterfaceMethod<
      "Emit the lowering IR for a predicated copy_atom_call with this CopyOp type.",
      "::mlir::LogicalResult",
      "emitAtomCall",
      (ins "::mlir::OpBuilder &":$builder,
           "::mlir::Location":$loc,
           "::mlir::Type":$copyAtomTy,
           "::mlir::Type":$srcMemTy,
           "::mlir::Type":$dstMemTy,
           "::mlir::Type":$predMemTy,
           "::mlir::Value":$atomVal,
           "::mlir::Value":$src,
           "::mlir::Value":$dst,
           "::mlir::Value":$pred)>
  ];
}

def Fly_MmaOpTypeInterface : TypeInterface<"MmaOpTypeInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = [
    InterfaceMethod<"", "::mlir::Attribute", "getThrLayout", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getShapeMNK", (ins)>,
    InterfaceMethod<"", "::mlir::Type", "getValTypeA", (ins)>,
    InterfaceMethod<"", "::mlir::Type", "getValTypeB", (ins)>,
    InterfaceMethod<"", "::mlir::Type", "getValTypeC", (ins)>,
    InterfaceMethod<"", "::mlir::Type", "getValTypeD", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrValLayoutA", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrValLayoutB", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrValLayoutC", (ins)>,
    InterfaceMethod<
      "Emit the lowering IR for an mma_atom_call with this MmaOp type.",
      "::mlir::LogicalResult",
      "emitAtomCall",
      (ins "::mlir::OpBuilder &":$builder,
           "::mlir::Location":$loc,
           "::mlir::Type":$mmaAtomTy,
           "::mlir::Type":$dMemTy,
           "::mlir::Type":$aMemTy,
           "::mlir::Type":$bMemTy,
           "::mlir::Type":$cMemTy,
           "::mlir::Value":$atomVal,
           "::mlir::Value":$d,
           "::mlir::Value":$a,
           "::mlir::Value":$b,
           "::mlir::Value":$c)>
  ];
}

#endif // FLY_INTERFACES
