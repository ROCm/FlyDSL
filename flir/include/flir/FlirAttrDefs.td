#ifndef FLIR_ATTRDEFS
#define FLIR_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

include "flir/FlirDialect.td"

def Flir_UnderscoreAttr : Flir_Attr<"Underscore", "underscore", []> {
  let parameters = (ins);
  let assemblyFormat = "`*`";
}

def Flir_DyncI32Attr : Flir_Attr<"DyncI32", "dync_i32", []> {
  let parameters = (ins
    "int32_t":$dyncElemIdx,
    "int32_t":$divisibility
  );
  let assemblyFormat = "`?` $dyncElemIdx `{` `div` `=` $divisibility `}`";
}

def Flir_DyncI64Attr : Flir_Attr<"DyncI64", "dync_i64", []> {
  let parameters = (ins
    "int32_t":$dyncElemIdx,
    "int32_t":$divisibility
  );
  let assemblyFormat = "`?` $dyncElemIdx `{` `div` `=` $divisibility `}`";
}

def Flir_DyncIntAttr : AnyAttrOf<[Flir_DyncI32Attr, Flir_DyncI64Attr]> {
  let cppFunctionName = "isValidDyncIntAttr";
}

#endif // FLIR_ATTRDEFS

