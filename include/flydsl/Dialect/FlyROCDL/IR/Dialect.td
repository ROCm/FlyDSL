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

#ifndef FLYROCDL_DIALECT
#define FLYROCDL_DIALECT

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

include "flydsl/Dialect/Fly/IR/FlyInterfaces.td"

def FlyROCDL_Dialect : Dialect {
  let name = "fly_rocdl";
  let cppNamespace = "::mlir::fly_rocdl";
  
  let dependentDialects = [
    "ROCDL::ROCDLDialect"
  ];

  let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

class FlyxROCL_MmaAtom<string typeName, string typeMnemonic, list<Trait> traits = []> 
    : TypeDef<FlyROCDL_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>, 
        DeclareTypeInterfaceMethods<Fly_MmaAtomTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
}

class FlyxROCL_CopyOp<string typeName, string typeMnemonic, list<Trait> traits = []> 
    : TypeDef<FlyROCDL_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>, 
        DeclareTypeInterfaceMethods<Fly_CopyOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
}

def FlyROCDL_SchedGroup : I32BitEnumAttr<"SchedGroup", "", [
  I32BitEnumAttrCaseNone<"None">,
  I32BitEnumAttrCaseBit<"NonSideEffect", 0>,
  I32BitEnumAttrCaseBit<"VALU", 1>,
  I32BitEnumAttrCaseBit<"SALU", 2>,
  I32BitEnumAttrCaseBit<"MFMA", 3>,
  I32BitEnumAttrCaseBit<"VMEM", 4>,
  I32BitEnumAttrCaseBit<"VMEM_READ", 5>,
  I32BitEnumAttrCaseBit<"VMEM_WRITE", 6>,
  I32BitEnumAttrCaseBit<"LDS", 7>,
  I32BitEnumAttrCaseBit<"DS_READ", 8>,
  I32BitEnumAttrCaseBit<"DS_WRITE", 9>,
  I32BitEnumAttrCaseBit<"Transcendental", 10>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = FlyROCDL_Dialect.cppNamespace;
}

#endif // FLYROCDL_DIALECT
