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
  let usePropertiesForAttributes = 1;
}

class Fly_Type<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Fly_Dialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

class Fly_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Fly_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class Fly_Op<string mnemonic, list<Trait> traits = []> 
    : Op<Fly_Dialect, mnemonic, traits> {
  // set default assembly format
  let assemblyFormat = "`(` operands `)` attr-dict `:` functional-type(operands, results)";
}


#endif // FLY_DIALECT
