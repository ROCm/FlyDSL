// Copyright (c) 2025 FlyDSL Project Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
  let methods = Fly_MayStaticInterface.methods;
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
