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
  let methods = Fly_NestedInterfaceMethods.methods;
}
def Fly_NestedTypeInterface : TypeInterface<"NestedTypeInterface"> {
  let methods = Fly_NestedInterfaceMethods.methods;
}

def Fly_MayStaticAttrInterface : AttrInterface<"MayStaticAttrInterface"> {
  let methods = Fly_MayStaticInterface.methods;
}
def Fly_MayStaticTypeInterface : TypeInterface<"MayStaticTypeInterface"> {
  let methods = Fly_MayStaticInterface.methods;
}


def Fly_CopyAtomTypeInterface : TypeInterface<"CopyAtomTypeInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = [
    InterfaceMethod<"", "::mlir::Attribute", "getThrSize", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrValLayoutSrc", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrValLayoutDst", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrValLayoutRef", (ins)>
  ];
}

def Fly_MmaAtomTypeInterface : TypeInterface<"MmaAtomTypeInterface"> {
  let cppNamespace = "::mlir::fly";
  let methods = [
    InterfaceMethod<"", "::mlir::Attribute", "getThrSize", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrValLayoutA", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrValLayoutB", (ins)>,
    InterfaceMethod<"", "::mlir::Attribute", "getThrValLayoutC", (ins)>
  ];
}

#endif // FLY_INTERFACES
