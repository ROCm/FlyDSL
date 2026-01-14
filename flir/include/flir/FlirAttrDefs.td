#ifndef FLIR_ATTRDEFS
#define FLIR_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

include "flir/FlirDialect.td"

// A leaf wildcard in tuple patterns (printed as `*`).
def Flir_UnderscoreAttr : AttrDef<Flir_Dialect, "Underscore"> {
  let mnemonic = "underscore";
  let parameters = (ins);
  let assemblyFormat = "`*`";
}

// A leaf dynamic placeholder in tuple patterns (printed as `?`).
def Flir_DyncI32Attr : AttrDef<Flir_Dialect, "DyncI32"> {
  let mnemonic = "dync_i32";
  let parameters = (ins "int32_t":$dyncElemIdx, "int32_t":$divisibility);
  let assemblyFormat = "`?` $dyncElemIdx `{` `div` `=` $divisibility `}`";
}

#endif // FLIR_ATTRDEFS

