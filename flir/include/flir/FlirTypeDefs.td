#ifndef FLIR_TYPEDEFS
#define FLIR_TYPEDEFS

include "flir/FlirDialect.td"
include "flir/FlirAttrDefs.td"

class Flir_Type<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Flir_Dialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

// `!flir.int` - legacy simple type (kept for compatibility with existing IR).
def Flir_Int : Flir_Type<"Int", "int">;

// `!flir.shape<...>` / `!flir.stride<...>` / `!flir.coord<...>`
// The payload is an Attribute tree:
// - IntegerAttr: constant leaf
// - #flir.dyn: dynamic leaf (runtime operand)
// - #flir.underscore: wildcard leaf
// - ArrayAttr: tuple node, recursively containing elements
def Flir_Shape : Flir_Type<"Shape", "shape"> {
  let parameters = (ins "Attribute":$pattern);
  let assemblyFormat = "`<` custom<IntTuplePattern>($pattern) `>`";
}

def Flir_Stride : Flir_Type<"Stride", "stride"> {
  let parameters = (ins "Attribute":$pattern);
  let assemblyFormat = "`<` custom<IntTuplePattern>($pattern) `>`";
}

def Flir_Coord : Flir_Type<"Coord", "coord"> {
  let parameters = (ins "Attribute":$pattern);
  let assemblyFormat = "`<` custom<IntTuplePattern>($pattern) `>`";
}

// `!flir.layout<shape:stride>` (each side is the same tuple pattern Attribute).
def Flir_Layout : Flir_Type<"Layout", "layout"> {
  let parameters = (ins "Attribute":$shape_pattern, "Attribute":$stride_pattern);
  let assemblyFormat = "`<` custom<IntTuplePattern>($shape_pattern) `` `:` `` custom<IntTuplePattern>($stride_pattern) `>`";
}

#endif // FLIR_TYPEDEFS

