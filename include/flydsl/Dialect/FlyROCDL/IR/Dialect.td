#ifndef FLYROCDL_DIALECT
#define FLYROCDL_DIALECT

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

include "flydsl/Dialect/Fly/IR/FlyInterfaces.td"

def FlyROCDL_Dialect : Dialect {
  let name = "fly_rocdl";
  let cppNamespace = "::mlir::fly_rocdl";
  
  let usePropertiesForAttributes = 1;
}

class FlyROCDL_Type<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FlyROCDL_Dialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

#endif // FLYROCDL_DIALECT
