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


def Fly_CopyOpTypeInterface : TypeInterface<"CopyOpTypeInterface", [Fly_MayStaticTypeInterface]> {
  let cppNamespace = "::mlir::fly";
  let methods = [
    InterfaceMethod<"", "::mlir::Attribute", "getThrLayout", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrBitLayoutSrc", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrBitLayoutDst", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrBitLayoutRef", (ins)>
  ];
}

def Fly_MmaAtomTypeInterface : TypeInterface<"MmaAtomTypeInterface", [Fly_MayStaticTypeInterface]> {
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
    InterfaceMethod<"", "::mlir::Attribute", "getThrValLayoutC", (ins)>
  ];
}

#endif // FLY_INTERFACES
