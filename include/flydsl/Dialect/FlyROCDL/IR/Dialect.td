// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

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
}

class FlyROCL_MmaOp<string typeName, string typeMnemonic, list<Trait> traits = []> 
    : TypeDef<FlyROCDL_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>, 
        DeclareTypeInterfaceMethods<Fly_MmaOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
}

class FlyROCL_CopyOp<string typeName, string typeMnemonic, list<Trait> traits = []> 
    : TypeDef<FlyROCDL_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>, 
        DeclareTypeInterfaceMethods<Fly_CopyOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
}

class FlyROCL_StatefulCopyOp<string typeName, string typeMnemonic, list<Trait> traits = []> 
    : TypeDef<FlyROCDL_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_CopyOpTypeInterface>,
        DeclareTypeInterfaceMethods<Fly_StatefulOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
}

class FlyROCL_StatefulMmaOp<string typeName, string typeMnemonic, list<Trait> traits = []> 
    : TypeDef<FlyROCDL_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_MmaOpTypeInterface>,
        DeclareTypeInterfaceMethods<Fly_StatefulOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
}

#endif // FLYROCDL_DIALECT
