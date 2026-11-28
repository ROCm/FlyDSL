// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYNVVM_DIALECT
#define FLYNVVM_DIALECT

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

include "flydsl/Dialect/Fly/IR/FlyInterfaces.td"

def FlyNVVM_Dialect : Dialect {
  let name = "fly_nvvm";
  let cppNamespace = "::mlir::fly_nvvm";

  let dependentDialects = [
    "NVVM::NVVMDialect"
  ];

  let useDefaultTypePrinterParser = 1;
}

class FlyNVVM_MmaOp<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FlyNVVM_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>,
        DeclareTypeInterfaceMethods<Fly_MmaOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
}

class FlyNVVM_CopyOp<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FlyNVVM_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>,
        DeclareTypeInterfaceMethods<Fly_CopyOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
}

#endif // FLYNVVM_DIALECT
