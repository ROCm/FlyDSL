#ifndef FLY_INTERFACES 
#define FLY_INTERFACES

include "mlir/IR/OpBase.td"

def Fly_NestedInterfaceMethods {
  list<InterfaceMethod> methods = [
    InterfaceMethod<"", "bool", "isLeaf", (ins)>,
    InterfaceMethod<"", "int32_t", "rank", (ins)>,
    InterfaceMethod<"", "int32_t", "rank", (ins "int32_t":$idx)>,
    InterfaceMethod<"", "int32_t", "rank", (ins "ArrayRef<int32_t>":$idxs)>,
    InterfaceMethod<"", "int32_t", "depth", (ins)>,
    InterfaceMethod<"", "int32_t", "depth", (ins "int32_t":$idx)>,
    InterfaceMethod<"", "int32_t", "depth", (ins "ArrayRef<int32_t>":$idxs)>
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
  let methods = [
    InterfaceMethod<"", "Attribute", "getThrSize", (ins)>,
    InterfaceMethod<"", "Attribute", "getThrValLayoutSrc", (ins)>,
    InterfaceMethod<"", "Attribute", "getThrValLayoutDst", (ins)>
  ];
}

def Fly_MmaAtomTypeInterface : TypeInterface<"MmaAtomTypeInterface"> {
  let methods = [
    InterfaceMethod<"", "Attribute", "getThrSize", (ins)>,
    InterfaceMethod<"", "Attribute", "getThrValLayoutA", (ins)>,
    InterfaceMethod<"", "Attribute", "getThrValLayoutB", (ins)>,
    InterfaceMethod<"", "Attribute", "getThrValLayoutC", (ins)>
  ];
}

#endif // FLY_INTERFACES
