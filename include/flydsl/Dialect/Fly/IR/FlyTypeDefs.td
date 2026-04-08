// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

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
    }]>,
    TypeBuilderWithInferredContext<(ins "IntTupleAttr":$shape, "IntTupleAttr":$stride), [{
      return $_get(shape.getContext(), LayoutAttr::get(shape, stride));
    }]>
  ];
}

def Fly_Swizzle : Fly_Type<"Swizzle", "swizzle", [
    DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>
]> {
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
    }]>,
    TypeBuilderWithInferredContext<(ins "Attribute":$inner, "IntTupleAttr":$offset, "LayoutAttr":$outer), [{
      return $_get(inner.getContext(), ComposedLayoutAttr::get(inner, offset, outer));
    }]>
  ];
}

def Fly_Tile : Fly_Type<"Tile", "tile", [
    DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>
]> {
  let parameters = (ins Fly_TileAttr:$attr);
  let assemblyFormat = "`<` $attr `>`";

  let extraClassDeclaration = [{
    int32_t rank() const;
  }];
  
  let builders = [
    TypeBuilderWithInferredContext<(ins "TileAttr":$attr), [{
      return $_get(attr.getContext(), attr);
    }]>,
    TypeBuilderWithInferredContext<(ins "ArrayAttr":$modes), [{
      return $_get(modes.getContext(), TileAttr::get(modes));
    }]>
  ];
}

def Fly_Pointer : Fly_Type<"Pointer", "ptr", []> {
  let parameters = (ins
    "Type":$elemTy,
    "AddressSpaceAttr":$addressSpace,
    "AlignAttr":$alignment,
    DefaultValuedParameter<"SwizzleAttr","SwizzleAttr::getTrivialSwizzle($_ctxt)">:$swizzle
  );
  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "AddressSpaceAttr":$addressSpace), [{
      return $_get(elemTy.getContext(), elemTy, addressSpace,
                   AlignAttr::getTrivialAlignment(elemTy),
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
  let extraClassDeclaration = [{
    int32_t getValueDivisibility() const;
  }];
}

def Fly_CoordTensor : Fly_Type<"CoordTensor", "coord_tensor", [
    DeclareTypeInterfaceMethods<Fly_NestedTypeInterface>,
    DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>
]> {
  let parameters = (ins Fly_IntTupleAttr:$base, "Attribute":$layout);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    CoordTensorType at(int32_t idx) const;
    CoordTensorType at(ArrayRef<int32_t> idxs) const;
  }];
  
  let builders = [
    TypeBuilderWithInferredContext<(ins "IntTupleAttr":$base, "Attribute":$layout), [{
      return $_get(base.getContext(), base, layout);
    }]>
  ];
}

def Fly_MemRef : Fly_Type<"MemRef", "memref", []> {
  let parameters = (ins
    "Type":$elemTy,
    "AddressSpaceAttr":$addressSpace,
    "Attribute":$layout,
    "AlignAttr":$alignment,
    DefaultValuedParameter<"SwizzleAttr","SwizzleAttr::getTrivialSwizzle($_ctxt)">:$swizzle
  );
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    MemRefType at(int32_t idx) const;
    MemRefType at(ArrayRef<int32_t> idxs) const;
    
    PointerType getPointerType() const;
    int32_t getValueDivisibility() const;
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "Attribute":$layout), [{
      // default address space is Global
      return $_get(elemTy.getContext(), elemTy, AddressSpaceAttr::get(elemTy.getContext(), ::mlir::fly::AddressSpace::Global), layout,
                   AlignAttr::getTrivialAlignment(elemTy),
                   SwizzleAttr::getTrivialSwizzle(elemTy.getContext()));
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "AddressSpaceAttr":$addressSpace, "Attribute":$layout), [{
      return $_get(elemTy.getContext(), elemTy, addressSpace, layout,
                   AlignAttr::getTrivialAlignment(elemTy),
                   SwizzleAttr::getTrivialSwizzle(elemTy.getContext()));
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "AddressSpaceAttr":$addressSpace, "Attribute":$layout, "AlignAttr":$alignment), [{
      return $_get(elemTy.getContext(), elemTy, addressSpace, layout, alignment,
                   SwizzleAttr::getTrivialSwizzle(elemTy.getContext()));
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "AddressSpaceAttr":$addressSpace, "Attribute":$layout, "AlignAttr":$alignment, "SwizzleAttr":$swizzle), [{
      return $_get(elemTy.getContext(), elemTy, addressSpace, layout, alignment, swizzle);
    }]>
  ];
}

def Fly_TiledCopy : Fly_Type<"TiledCopy", "tiled_copy", [
    DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>
]> {
  let parameters = (ins
    "Type":$copyAtom,
    Fly_Layout:$layoutThrVal,
    Fly_Tile:$tileMN
  );
  let assemblyFormat = "`<` $copyAtom `,` qualified($layoutThrVal) `,` qualified($tileMN) `>`";
}

def Fly_TiledMma : Fly_Type<"TiledMma", "tiled_mma", [
    DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>
]> {
  let parameters = (ins
    "Type":$mmaAtom,
    Fly_Layout:$atomLayout,
    DefaultValuedParameter<"TileType", "TiledMmaType::getDefaultPermutationMNK($_ctxt)">:$permutation
  );
  let assemblyFormat = "`<` $mmaAtom `,` qualified($atomLayout) (`,` qualified($permutation)^)? `>`";

  let extraClassDeclaration = [{
    static TileType getDefaultPermutationMNK(MLIRContext *ctx);
  }];
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
    ::mlir::LogicalResult emitAtomCall(OpBuilder &builder, Location loc, Type copyAtomTy,
                                       Type srcMemTy, Type dstMemTy, Value atomVal, Value src, Value dst) const;
    ::mlir::LogicalResult emitAtomCall(OpBuilder &builder, Location loc, Type copyAtomTy,
                                       Type srcMemTy, Type dstMemTy, Type predMemTy, Value atomVal, Value src, Value dst,
                                       Value pred) const;
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$copyOp, "int32_t":$valBits), [{
      return $_get(copyOp.getContext(), copyOp, valBits);
    }]>
  ];
}

def Fly_MmaAtom : Fly_Type<"MmaAtom", "mma_atom", [
  DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>
]> {
  let parameters = (ins
    "Type":$mmaOp
  );
  let assemblyFormat = "`<` $mmaOp `>`";

  let extraClassDeclaration = [{
    ::mlir::Attribute getThrLayout() const;
    ::mlir::Attribute getShapeMNK() const;
    ::mlir::Type getValTypeA() const;
    ::mlir::Type getValTypeB() const;
    ::mlir::Type getValTypeC() const;
    ::mlir::Type getValTypeD() const;
    ::mlir::Attribute getThrValLayoutA() const;
    ::mlir::Attribute getThrValLayoutB() const;
    ::mlir::Attribute getThrValLayoutC() const;
    ::mlir::LogicalResult emitAtomCall(OpBuilder &builder, Location loc, Type mmaAtomTy,
                                       Type dMemTy, Type aMemTy, Type bMemTy, Type cMemTy, Value atomVal, Value d,
                                       Value a, Value b, Value c) const;
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$mmaOp), [{
      return $_get(mmaOp.getContext(), mmaOp);
    }]>
  ];
}

def Fly_CopyOpUniversalCopy : Fly_Type<"CopyOpUniversalCopy", "universal_copy", [
  DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>,
  DeclareTypeInterfaceMethods<Fly_CopyOpTypeInterface>
]> {
  let parameters = (ins
    "int32_t":$bitSize
  );
  let assemblyFormat = "`<` $bitSize `>`";
}

def Fly_CopyOpUniversalAtomic : Fly_Type<"CopyOpUniversalAtomic", "universal_atomic", [
  DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>,
  DeclareTypeInterfaceMethods<Fly_CopyOpTypeInterface>
]> {
  let parameters = (ins
    "AtomicOpAttr":$atomicOp,
    "Type":$valType
  );
  let assemblyFormat = "`<` $atomicOp `,` $valType `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "AtomicOpAttr":$atomicOp, "Type":$valType), [{
      return $_get(atomicOp.getContext(), atomicOp, valType);
    }]>
  ];
}

def Fly_MmaOpUniversalFMA : Fly_Type<"MmaOpUniversalFMA", "universal_fma", [
  DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>,
  DeclareTypeInterfaceMethods<Fly_MmaOpTypeInterface>
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
