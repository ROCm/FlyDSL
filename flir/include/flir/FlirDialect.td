//===- FlirDialect.td - Flir dialect definition ---------*- tablegen -*-===//

#ifndef FLIR_DIALECT
#define FLIR_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Flir_Dialect : Dialect {
  let name = "flir";
  let summary = "A MLIR dialect for Flir tensor operations";
  let description = [{
    The Flir dialect provides types and operations for hierarchical 
    hierarchical tensor layout manipulation.
  }];
  let cppNamespace = "::mlir::flir";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif // FLIR_DIALECT
