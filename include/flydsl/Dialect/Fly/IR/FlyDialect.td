// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLY_DIALECT
#define FLY_DIALECT

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

include "flydsl/Dialect/Fly/IR/FlyInterfaces.td"

def Fly_Dialect : Dialect {
  let name = "fly";
  let cppNamespace = "::mlir::fly";
  
  // let hasConstantMaterializer = 1
  let hasConstantMaterializer = 0;
  
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class Fly_Type<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Fly_Dialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

class Fly_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Fly_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class Fly_I32EnumAttr<string enumName, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<enumName, summary, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = Fly_Dialect.cppNamespace;
}

class Fly_EnumAttr<Fly_I32EnumAttr enumInfo, string attrMnemonic, list<Trait> traits = []>
    : EnumAttr<Fly_Dialect, enumInfo, attrMnemonic, traits>;

class Fly_Op<string mnemonic, list<Trait> traits = []> 
    : Op<Fly_Dialect, mnemonic, traits> {
  // set default assembly format
  let assemblyFormat = "`(` operands `)` attr-dict `:` functional-type(operands, results)";
}


#endif // FLY_DIALECT
