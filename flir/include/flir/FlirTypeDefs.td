#ifndef FLIR_TYPEDEFS
#define FLIR_TYPEDEFS

include "flir/FlirDialect.td"
include "flir/FlirAttrDefs.td"

// Flyx-aligned type system:
// - Shape/Stride/Coord: a single `pattern` Attribute
// - Layout: `shape_pattern` + `stride_pattern` Attributes
// Pattern syntax is parsed/printed via custom assembly in C++.

class Flir_Type<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Flir_Dialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

def Flir_Int : Flir_Type<"Int", "int", []> {
}

def Flir_Shape : Flir_Type<"Shape", "shape", []> {
  let parameters = (ins "Attribute":$pattern);
  let extraClassDeclaration = [{
    int getRank() const;
  }];
  let hasCustomAssemblyFormat = 1;
}

def Flir_Stride : Flir_Type<"Stride", "stride", []> {
  let parameters = (ins "Attribute":$pattern);
  let extraClassDeclaration = [{
    int getRank() const;
  }];
  let hasCustomAssemblyFormat = 1;
}

def Flir_Coord : Flir_Type<"Coord", "coord", []> {
  let parameters = (ins "Attribute":$pattern);
  let extraClassDeclaration = [{
    int getRank() const;
  }];
  let hasCustomAssemblyFormat = 1;
}

def Flir_Layout : Flir_Type<"Layout", "layout", []> {
  let parameters = (ins
    "Attribute":$shape_pattern,
    "Attribute":$stride_pattern
  );
  let extraClassDeclaration = [{
    static LayoutType get(::mlir::MLIRContext *context, int rank);
    static LayoutType get(::mlir::MLIRContext *context, ::mlir::flir::ShapeType shape,
                          ::mlir::flir::StrideType stride);
    int getRank() const;
  }];
  let hasCustomAssemblyFormat = 1;
}

#endif // FLIR_TYPEDEFS

