//===- RocirDialect.td - Rocir dialect definition ---------*- tablegen -*-===//

#ifndef ROCIR_DIALECT
#define ROCIR_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Rocir_Dialect : Dialect {
  let name = "rocir";
  let summary = "A MLIR dialect for Rocir tensor operations";
  let description = [{
    The Rocir dialect provides types and operations for hierarchical 
    hierarchical tensor layout manipulation.
  }];
  let cppNamespace = "::mlir::rocir";
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
  
  let extraClassDeclaration = [{
    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type type, DialectAsmPrinter &printer) const override;
  }];
}

#endif // ROCIR_DIALECT
