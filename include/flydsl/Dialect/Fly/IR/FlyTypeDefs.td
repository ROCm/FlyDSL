// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLY_TYPEDEFS
#define FLY_TYPEDEFS

include "flydsl/Dialect/Fly/IR/FlyDialect.td"
include "flydsl/Dialect/Fly/IR/FlyAttrDefs.td"

def Fly_Basis : Fly_Type<"Basis", "basis", [
    DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>
]> {
  let parameters = (ins Fly_LeafAttr:$attr);
  let assemblyFormat = "`` $attr";
  let genVerifyDecl = 1;
  
  let extraClassDeclaration = [{
    const core::Leaf<core::Trait> &getRef() const { return getAttr().getRef(); }
  }];
  
  let builders = [
    TypeBuilder<(ins "core::Leaf<core::Trait>":$value), [{
      auto attr = LeafAttr::get($_ctxt, std::move(value));
      return attr && attr.isBasis() ? $_get($_ctxt, attr) : BasisType{};
    }]>,
    TypeBuilderWithInferredContext<(ins "LeafAttr":$attr), [{
      return attr && attr.isBasis() ? $_get(attr.getContext(), attr) : BasisType{};
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
    const core::IntTuple<core::Trait> &getRef() const { return getAttr().getRef(); }
    IntTupleType at(int32_t idx) const;
    IntTupleType at(ArrayRef<int32_t> idxs) const;
  }];

  let builders = [
    TypeBuilder<(ins "core::IntTuple<core::Trait>":$value), [{
      auto attr = IntTupleAttr::get($_ctxt, std::move(value));
      return attr ? $_get($_ctxt, attr) : IntTupleType{};
    }]>,
    TypeBuilder<(ins "core::IntTupleRef<core::Trait>":$value), [{
      return get($_ctxt, core::IntTuple<core::Trait>(value));
    }]>,
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
    const core::Layout<core::Trait> &getRef() const { return getAttr().getRef(); }
    bool isStaticShape() const;
    bool isStaticStride() const;

    LayoutType at(int32_t idx) const;
    LayoutType at(ArrayRef<int32_t> idxs) const;
  }];
  
  let builders = [
    TypeBuilder<(ins "core::Layout<core::Trait>":$value), [{
      auto attr = LayoutAttr::get($_ctxt, std::move(value));
      return attr ? $_get($_ctxt, attr) : LayoutType{};
    }]>,
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

  let extraClassDeclaration = [{
    const core::Swizzle<core::Trait> &getRef() const { return getAttr().getRef(); }
  }];
  let builders = [
    TypeBuilder<(ins "core::Swizzle<core::Trait>":$value), [{
      auto attr = SwizzleAttr::get($_ctxt, std::move(value));
      return attr ? $_get($_ctxt, attr) : SwizzleType{};
    }]>,
    TypeBuilderWithInferredContext<(ins "SwizzleAttr":$attr), [{
      return $_get(attr.getContext(), attr);
    }]>
  ];
}

def Fly_CoordSwizzle : Fly_Type<"CoordSwizzle", "coord_swizzle", [
    DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>
]> {
  let parameters = (ins Fly_CoordSwizzleAttr:$attr);
  let assemblyFormat = "`<` $attr `>`";

  let extraClassDeclaration = [{
    const core::CoordSwizzle<core::Trait> &getRef() const { return getAttr().getRef(); }
  }];
  let builders = [
    TypeBuilder<(ins "core::CoordSwizzle<core::Trait>":$value), [{
      auto attr = CoordSwizzleAttr::get($_ctxt, std::move(value));
      return attr ? $_get($_ctxt, attr) : CoordSwizzleType{};
    }]>,
    TypeBuilderWithInferredContext<(ins "CoordSwizzleAttr":$attr), [{
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
    const core::ComposedLayout<core::Trait> &getRef() const { return getAttr().getRef(); }
    bool isStaticOuter() const;
    bool isStaticInner() const;
    bool isStaticOffset() const;

    ComposedLayoutType at(int32_t idx) const;
    ComposedLayoutType at(ArrayRef<int32_t> idxs) const;
  }];
  
  let builders = [
    TypeBuilder<(ins "core::ComposedLayout<core::Trait>":$value), [{
      auto attr = ComposedLayoutAttr::get($_ctxt, std::move(value));
      return attr ? $_get($_ctxt, attr) : ComposedLayoutType{};
    }]>,
    TypeBuilderWithInferredContext<(ins "ComposedLayoutAttr":$attr), [{
      return $_get(attr.getContext(), attr);
    }]>,
    TypeBuilderWithInferredContext<(ins "Attribute":$inner, "IntTupleAttr":$offset, "Attribute":$outer), [{
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
    const core::Tile<core::Trait> &getRef() const { return getAttr().getRef(); }
    int32_t rank() const;
  }];
  
  let builders = [
    TypeBuilder<(ins "core::Tile<core::Trait>":$value), [{
      auto attr = TileAttr::get($_ctxt, std::move(value));
      return attr ? $_get($_ctxt, attr) : TileType{};
    }]>,
    TypeBuilderWithInferredContext<(ins "TileAttr":$attr), [{
      return $_get(attr.getContext(), attr);
    }]>,
    // Compatibility with attribute-tree construction; see TileAttr.
    TypeBuilderWithInferredContext<(ins "ArrayAttr":$modes), [{
      auto attr = TileAttr::get(modes);
      return attr ? $_get(modes.getContext(), attr) : TileType{};
    }]>
  ];
}

def Fly_Pointer : Fly_Type<"Pointer", "ptr", []> {
  let parameters = (ins
    "Type":$elemTy,
    "Attribute":$addressSpace,
    "AlignAttr":$alignment,
    DefaultValuedParameter<"SwizzleAttr","SwizzleAttr::getTrivialSwizzle($_ctxt)">:$swizzle
  );
  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilder<(ins "const core::Pointer<core::Trait> &":$value), [{
      auto params = getCoreParameters($_ctxt, value);
      if (!params)
        return {};
      return $_get($_ctxt, params->elemTy, params->addressSpace,
                   params->alignment, params->swizzle);
    }]>,
    TypeBuilderWithInferredContext<(ins "const core::Pointer<core::Trait> &":$value), [{
      auto params = getCoreParameters(nullptr, value);
      if (!params)
        return {};
      return $_get(params->elemTy.getContext(), params->elemTy, params->addressSpace,
                   params->alignment, params->swizzle);
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "Attribute":$addressSpace), [{
      return $_get(elemTy.getContext(), elemTy, addressSpace,
                   AlignAttr::getTrivialAlignment(elemTy),
                   SwizzleAttr::getTrivialSwizzle(elemTy.getContext()));
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "Attribute":$addressSpace, "AlignAttr":$alignment), [{
      return $_get(elemTy.getContext(), elemTy, addressSpace, alignment,
                  SwizzleAttr::getTrivialSwizzle(elemTy.getContext()));
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "Attribute":$addressSpace, "AlignAttr":$alignment, "SwizzleAttr":$swizzle), [{
      return $_get(elemTy.getContext(), elemTy, addressSpace, alignment, swizzle);
    }]>
  ];
  let extraClassDeclaration = [{
    core::Pointer<core::Trait> getRef() const;

  private:
    struct CoreParameters {
      Type elemTy;
      Attribute addressSpace;
      AlignAttr alignment;
      SwizzleAttr swizzle;
    };
    // A null context infers the context from the element type.
    static std::optional<CoreParameters> getCoreParameters(
        MLIRContext *ctx, const core::Pointer<core::Trait> &value);
  public:
  }];
  let genVerifyDecl = 1;
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

    core::CoordTensor<core::Trait> getRef() const;
  }];
  let genVerifyDecl = 1;
  
  let builders = [
    TypeBuilder<(ins "const core::CoordTensor<core::Trait> &":$value), [{
      auto base = IntTupleAttr::get($_ctxt, value.data());
      auto layout = getLayoutAttr($_ctxt, value.layout());
      if (!base || !layout)
        return {};
      return $_get($_ctxt, base, layout);
    }]>,
    TypeBuilderWithInferredContext<(ins "IntTupleAttr":$base, "Attribute":$layout), [{
      return $_get(base.getContext(), base, layout);
    }]>
  ];
}

def Fly_MemRef : Fly_Type<"MemRef", "memref", []> {
  let parameters = (ins
    "Type":$elemTy,
    "Attribute":$addressSpace,
    "Attribute":$layout,
    "AlignAttr":$alignment,
    DefaultValuedParameter<"SwizzleAttr","SwizzleAttr::getTrivialSwizzle($_ctxt)">:$swizzle
  );
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    MemRefType at(int32_t idx) const;
    MemRefType at(ArrayRef<int32_t> idxs) const;

    PointerType getPointerType() const;

    core::MemRef<core::Trait> getRef() const;
  }];
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilder<(ins "const core::MemRef<core::Trait> &":$value), [{
      auto pointer = PointerType::get($_ctxt, value.data());
      if (!pointer)
        return {};
      auto layout = getLayoutAttr($_ctxt, value.layout());
      if (!layout)
        return {};
      return $_get($_ctxt, pointer.getElemTy(), pointer.getAddressSpace(), layout,
                   pointer.getAlignment(), pointer.getSwizzle());
    }]>,
    TypeBuilderWithInferredContext<(ins "const core::MemRef<core::Trait> &":$value), [{
      auto pointer = PointerType::get(value.data());
      if (!pointer)
        return {};
      auto layout = getLayoutAttr(pointer.getContext(), value.layout());
      if (!layout)
        return {};
      return $_get(pointer.getContext(), pointer.getElemTy(), pointer.getAddressSpace(), layout,
                   pointer.getAlignment(), pointer.getSwizzle());
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "Attribute":$layout), [{
      // default address space is Global
      return $_get(elemTy.getContext(), elemTy, AddressSpaceAttr::get(elemTy.getContext(), ::mlir::fly::AddressSpace::Global), layout,
                   AlignAttr::getTrivialAlignment(elemTy),
                   SwizzleAttr::getTrivialSwizzle(elemTy.getContext()));
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "Attribute":$addressSpace, "Attribute":$layout), [{
      return $_get(elemTy.getContext(), elemTy, addressSpace, layout,
                   AlignAttr::getTrivialAlignment(elemTy),
                   SwizzleAttr::getTrivialSwizzle(elemTy.getContext()));
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "Attribute":$addressSpace, "Attribute":$layout, "AlignAttr":$alignment), [{
      return $_get(elemTy.getContext(), elemTy, addressSpace, layout, alignment,
                   SwizzleAttr::getTrivialSwizzle(elemTy.getContext()));
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "Attribute":$addressSpace, "Attribute":$layout, "AlignAttr":$alignment, "SwizzleAttr":$swizzle), [{
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

  let extraClassDeclaration = [{
    core::TiledCopy<core::Trait> getRef() const;
    /// Whether getRef() is defined: the copy atom's layouts form an atom and
    /// the thread-value layout tiles whole atoms.
    bool isValidRef() const;
  }];
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
    core::TiledMMA<core::Trait> getRef() const;
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

    /// Core description keeping the full copy-op and atom type identities.
    core::CopyAtom<core::Trait> getRef() const;
  
    ::mlir::LogicalResult emitAtomCall(OpBuilder &builder, Location loc, Type copyAtomTy,
                                       Type srcMemTy, Type dstMemTy, Value atomVal, Value src, Value dst) const;
    ::mlir::LogicalResult emitAtomCall(OpBuilder &builder, Location loc, Type copyAtomTy,
                                       Type srcMemTy, Type dstMemTy, Type predMemTy, Value atomVal, Value src, Value dst,
                                       Value pred) const;

    ::mlir::FailureOr<::mlir::Value> emitAtomCallSSA(OpBuilder &builder, Location loc, Type resultTy,
                                                     Type copyAtomTy, Type srcTy, Type dstTy,
                                                     Value atomVal, Value src, Value dst) const;
    ::mlir::FailureOr<::mlir::Value> emitAtomCallSSA(OpBuilder &builder, Location loc, Type resultTy,
                                                     Type copyAtomTy, Type srcTy, Type dstTy, Type predTy,
                                                     Value atomVal, Value src, Value dst, Value pred) const;

    bool isStateful() const;
    ::mlir::Type getConvertedType(::mlir::MLIRContext *ctx) const;
    ::mlir::Value setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                               Attribute fieldAttr, Value fieldValue) const;
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
    core::MMAAtom<core::Trait> getRef() const;
    ::mlir::LogicalResult emitAtomCall(OpBuilder &builder, Location loc, Type mmaAtomTy,
                                       Type dMemTy, TypeRange aMemTy, TypeRange bMemTy, Type cMemTy, Value atomVal, Value d,
                                       ValueRange a, ValueRange b, Value c) const;
    ::mlir::FailureOr<::mlir::Value> emitAtomCallSSA(OpBuilder &builder, Location loc, Type resultTy,
                                                     Type mmaAtomTy, Type dTy, TypeRange aTy, TypeRange bTy, Type cTy,
                                                     Value atomVal, Value d, ValueRange a, ValueRange b, Value c) const;

    bool isStateful() const;
    ::mlir::Type getConvertedType(::mlir::MLIRContext *ctx) const;
    ::mlir::Value setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                               Attribute fieldAttr, Value fieldValue) const;
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
    "Type":$valType,
    StringRefParameter<"llvm atomicrmw sync scope", [{""}]>:$syncscope
  );
  let assemblyFormat = "`<` $atomicOp `,` $valType (`,` $syncscope^)? `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "AtomicOpAttr":$atomicOp, "Type":$valType), [{
      return $_get(atomicOp.getContext(), atomicOp, valType, ::llvm::StringRef(""));
    }]>,
    TypeBuilderWithInferredContext<(ins "AtomicOpAttr":$atomicOp, "Type":$valType, "::llvm::StringRef":$syncscope), [{
      return $_get(atomicOp.getContext(), atomicOp, valType, syncscope);
    }]>
  ];
}

def Fly_FastDivMod : Fly_Type<"FastDivMod", "fast_divmod", []> {
  let parameters = (ins "IntegerType":$divisorType);
  let assemblyFormat = "`<` $divisorType `>`";
  let genVerifyDecl = 1;
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
