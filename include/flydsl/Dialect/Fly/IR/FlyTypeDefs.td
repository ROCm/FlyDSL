#ifndef FLY_TYPEDEFS
#define FLY_TYPEDEFS

include "flydsl/Dialect/Fly/IR/FlyDialect.td"
include "flydsl/Dialect/Fly/IR/FlyAttrDefs.td"

def Fly_Basis : Fly_Type<"Basis", "basis", [
    DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>
]> {
  let parameters = (ins Fly_BasisAttr:$attr);
  let assemblyFormat = "`<` $attr `>`";
  
  let extraClassDeclaration = [{
    int32_t depth();
  }];
  
  let builders = [
    TypeBuilderWithInferredContext<(ins "BasisAttr":$attr), [{
      return $_get(attr.getContext(), attr);
    }]>
  ];
}

def Fly_IntTuple : Fly_Type<"IntTuple", "int_tuple", [
    DeclareTypeInterfaceMethods<Fly_NestedTypeInterface>,
    DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>
]> {
  let parameters = (ins Fly_IntTupleAttr:$attr);
  let assemblyFormat = "`<` $attr `>`";

  let extraClassDeclaration = [{
    IntTupleType at(int32_t idx) const;
    IntTupleType at(ArrayRef<int32_t> idxs) const;
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "IntTupleAttr":$attr), [{
      return $_get(attr.getContext(), attr);
    }]>
  ];
}

def Fly_Layout : Fly_Type<"Layout", "layout", [
    DeclareTypeInterfaceMethods<Fly_NestedTypeInterface>,
    DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>
]> {
  let parameters = (ins Fly_LayoutAttr:$attr);
  let assemblyFormat = "`<` $attr `>`";

  let extraClassDeclaration = [{
    bool isStaticShape() const;
    bool isStaticStride() const;

    LayoutType at(int32_t idx) const;
    LayoutType at(ArrayRef<int32_t> idxs) const;
  }];
  
  let builders = [
    TypeBuilderWithInferredContext<(ins "LayoutAttr":$attr), [{
      return $_get(attr.getContext(), attr);
    }]>
  ];
}

def Fly_Swizzle : Fly_Type<"Swizzle", "swizzle", []> {
  let parameters = (ins Fly_SwizzleAttr:$attr);
  let assemblyFormat = "`<` $attr `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "SwizzleAttr":$attr), [{
      return $_get(attr.getContext(), attr);
    }]>
  ];
}

def Fly_ComposedLayout : Fly_Type<"ComposedLayout", "composed_layout", [
    DeclareTypeInterfaceMethods<Fly_NestedTypeInterface>,
    DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>
]> {
  let parameters = (ins Fly_ComposedLayoutAttr:$attr);
  let assemblyFormat = "`<` $attr `>`";

  let extraClassDeclaration = [{
    bool isStaticOuter() const;
    bool isStaticInner() const;
    bool isStaticOffset() const;

    ComposedLayoutType at(int32_t idx) const;
    ComposedLayoutType at(ArrayRef<int32_t> idxs) const;
  }];
  
  let builders = [
    TypeBuilderWithInferredContext<(ins "ComposedLayoutAttr":$attr), [{
      return $_get(attr.getContext(), attr);
    }]>
  ];
}

def Fly_Tile : Fly_Type<"Tile", "tile", []> {
  let parameters = (ins Fly_TileAttr:$attr);
  let assemblyFormat = "`<` $attr `>`";

  let extraClassDeclaration = [{
    int32_t rank() const;
  }];
  
  let builders = [
    TypeBuilderWithInferredContext<(ins "TileAttr":$attr), [{
      return $_get(attr.getContext(), attr);
    }]>
  ];
}

def Fly_Pointer : Fly_Type<"Pointer", "ptr", []> {
  let parameters = (ins
    "Type":$elemTy,
    "AddressSpaceAttr":$addressSpace,
    DefaultValuedParameter<"AlignAttr","AlignAttr::getTrivialAlignment($_ctxt)">:$alignment,
    DefaultValuedParameter<"SwizzleAttr","SwizzleAttr::getTrivialSwizzle($_ctxt)">:$swizzle
  );
  let assemblyFormat = "`<` $elemTy `,` `` $addressSpace (`,` $alignment^)? (`,` $swizzle^)? `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "AddressSpaceAttr":$addressSpace), [{
      return $_get(elemTy.getContext(), elemTy, addressSpace,
                   AlignAttr::getTrivialAlignment(elemTy.getContext()),
                   SwizzleAttr::getTrivialSwizzle(elemTy.getContext()));
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "AddressSpaceAttr":$addressSpace, "AlignAttr":$alignment), [{
      return $_get(elemTy.getContext(), elemTy, addressSpace, alignment, 
                  SwizzleAttr::getTrivialSwizzle(elemTy.getContext()));
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "AddressSpaceAttr":$addressSpace, "AlignAttr":$alignment, "SwizzleAttr":$swizzle), [{
      return $_get(elemTy.getContext(), elemTy, addressSpace, alignment, swizzle);
    }]>
  ];
  let extraClassDeclaration = [{}];
}

def Fly_CoordTensor : Fly_Type<"CoordTensor", "coord_tensor", [
    DeclareTypeInterfaceMethods<Fly_NestedTypeInterface>,
    DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>
]> {
  let parameters = (ins Fly_IntTupleAttr:$base, Fly_LayoutAttr:$layout);
  let assemblyFormat = "`<` $base `,` $layout `>`";

  let extraClassDeclaration = [{
    CoordTensorType at(int32_t idx) const;
    CoordTensorType at(ArrayRef<int32_t> idxs) const;
  }];
  
  let builders = [
    TypeBuilderWithInferredContext<(ins "IntTupleAttr":$base, "LayoutAttr":$layout), [{
      return $_get(base.getContext(), base, layout);
    }]>
  ];
}

def Fly_MemRef : Fly_Type<"MemRef", "memref", []> {
  let parameters = (ins
    "Type":$elemTy,
    "AddressSpaceAttr":$addressSpace,
    Fly_LayoutAttr:$layout,
    DefaultValuedParameter<"AlignAttr","AlignAttr::getTrivialAlignment($_ctxt)">:$alignment,
    DefaultValuedParameter<"SwizzleAttr","SwizzleAttr::getTrivialSwizzle($_ctxt)">:$swizzle
  );
  let assemblyFormat = "`<` $elemTy `,` `` $addressSpace `,` $layout (`,` $alignment^)? (`,` $swizzle^)? `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "LayoutAttr":$layout), [{
      // default address space is Global
      return $_get(elemTy.getContext(), elemTy, AddressSpaceAttr::get(elemTy.getContext(), static_cast<::mlir::fly::AddressSpace>(0)), layout,
                   AlignAttr::getTrivialAlignment(elemTy.getContext()),
                   SwizzleAttr::getTrivialSwizzle(elemTy.getContext()));
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "AddressSpaceAttr":$addressSpace, "LayoutAttr":$layout), [{
      return $_get(elemTy.getContext(), elemTy, addressSpace, layout,
                   AlignAttr::getTrivialAlignment(elemTy.getContext()),
                   SwizzleAttr::getTrivialSwizzle(elemTy.getContext()));
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "AddressSpaceAttr":$addressSpace, "LayoutAttr":$layout, "AlignAttr":$alignment), [{
      return $_get(elemTy.getContext(), elemTy, addressSpace, layout, alignment,
                   SwizzleAttr::getTrivialSwizzle(elemTy.getContext()));
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "AddressSpaceAttr":$addressSpace, "LayoutAttr":$layout, "AlignAttr":$alignment, "SwizzleAttr":$swizzle), [{
      return $_get(elemTy.getContext(), elemTy, addressSpace, layout, alignment, swizzle);
    }]>
  ];
}

def IteratorLikeType : AnyTypeOf<[Fly_IntTuple, Fly_Pointer]>;
def TensorLikeType : AnyTypeOf<[Fly_CoordTensor, Fly_MemRef]>;
def AnyLayoutType : AnyTypeOf<[Fly_Layout, Fly_ComposedLayout]>;
def LayoutLikeType : AnyTypeOf<[AnyLayoutType, TensorLikeType]>;
def IntTupleLikeType : AnyTypeOf<[Fly_IntTuple, LayoutLikeType]>;

def Fly_TiledCopy : Fly_Type<"TiledCopy", "tiled_copy", []> {
  let parameters = (ins
    "Type":$copyAtom,
    Fly_Layout:$layoutThrVal,
    Fly_Tile:$tileMN
  );
  let assemblyFormat = "`<` $copyAtom `,` $layoutThrVal `,` $tileMN `>`";
}

def Fly_TiledMma : Fly_Type<"TiledMma", "tiled_mma", []> {
  let parameters = (ins
    "Type":$mmaAtom,
    Fly_Layout:$atomLayout,
    Fly_Tile:$permutation
  );
  let assemblyFormat = "`<` $mmaAtom `,` $atomLayout `,` $permutation `>`";
}

def Fly_CopyOpUniversalCopy : Fly_Type<"CopyOpUniversalCopy", "op.universal_copy", [
  DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>,
  DeclareTypeInterfaceMethods<Fly_CopyOpTypeInterface>
]> {
  let parameters = (ins
    "int32_t":$bitSize
  );
  let assemblyFormat = "`<` $bitSize `>`";
}

def Fly_CopyAtom : Fly_Type<"CopyAtom", "copy_atom", [
  DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>
]> {
  let parameters = (ins
    "Type":$copyOp,
    "int32_t":$valBits
  );
  let assemblyFormat = "`<` $copyOp `,` $valBits `>`";

  let extraClassDeclaration = [{
    ::mlir::Attribute getThrLayout();
    ::mlir::Attribute getThrValLayoutSrc();
    ::mlir::Attribute getThrValLayoutDst();
    ::mlir::Attribute getThrValLayoutRef();
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$copyOp, "int32_t":$valBits), [{
      return $_get(copyOp.getContext(), copyOp, valBits);
    }]>
  ];
}

def Fly_MmaAtomUniversalFMA : Fly_Type<"MmaAtomUniversalFMA", "atom.universal_fma", [
  DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>,
  DeclareTypeInterfaceMethods<Fly_MmaAtomTypeInterface>
]> {
  let parameters = (ins "Type":$elemTy);

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elemTy), [{
      return $_get(elemTy.getContext(), elemTy);
    }]>
  ];
  let hasCustomAssemblyFormat = 1;
}

#endif // FLY_TYPEDEFS
