//===- CuteDialect.td - Cute dialect definition -----------*- tablegen -*-===//

#ifndef CUTE_DIALECT
#define CUTE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Cute_Dialect : Dialect {
  let name = "cute";
  let summary = "A MLIR dialect for CUTE tensor operations";
  let description = [{
    The Cute dialect provides types and operations for CUTE-style 
    hierarchical tensor layout manipulation.
  }];
  let cppNamespace = "::mlir::cute";
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 1;
  
  let extraClassDeclaration = [{
    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type type, DialectAsmPrinter &printer) const override;
  }];
}

#endif // CUTE_DIALECT
