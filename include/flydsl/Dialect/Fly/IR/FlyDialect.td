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
    : Op<Fly_Dialect, mnemonic, traits>;


#endif // FLY_DIALECT
