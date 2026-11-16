// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLY_ATTRDEFS
#define FLY_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/Constraints.td"
include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"

include "flydsl/Dialect/Fly/IR/FlyDialect.td"

//===----------------------------------------------------------------------===//
// Enum attributes
//===----------------------------------------------------------------------===//
def Fly_CachePolicy : Fly_I32EnumAttr<"CachePolicy", "", [
  I32EnumAttrCase<"CacheGlobal", 0, "cache_global">,
  I32EnumAttrCase<"CacheAlways", 1, "cache_always">
]>;
def Fly_CachePolicyAttr : Fly_EnumAttr<Fly_CachePolicy, "cache_policy"> {}

def Fly_AddressSpace : Fly_I32EnumAttr<"AddressSpace", "", [
  I32EnumAttrCase<"Generic", 0, "generic">,
  I32EnumAttrCase<"Global", 1, "global">,
  I32EnumAttrCase<"Shared", 2, "shared">,
  I32EnumAttrCase<"Register", 3, "register">
]>;
def Fly_AddressSpaceAttr : Fly_EnumAttr<Fly_AddressSpace, "address_space"> {}

def Fly_MmaOperand : Fly_I32EnumAttr<"MmaOperand", "", [
  I32EnumAttrCase<"A", 0, "a">,
  I32EnumAttrCase<"B", 1, "b">,
  I32EnumAttrCase<"C", 2, "c">,
  I32EnumAttrCase<"D", 3, "d">
]>;
def Fly_MmaOperandAttr : Fly_EnumAttr<Fly_MmaOperand, "mma_operand"> {}

def Fly_AtomicOp : Fly_I32EnumAttr<"AtomicOp", "", [
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"Max", 1, "max">,
  I32EnumAttrCase<"Min", 2, "min">,
  I32EnumAttrCase<"And", 3, "and">,
  I32EnumAttrCase<"Or", 4, "or">,
  I32EnumAttrCase<"Inc", 5, "inc">,
  I32EnumAttrCase<"Dec", 6, "dec">
]>;
def Fly_AtomicOpAttr : Fly_EnumAttr<Fly_AtomicOp, "atomic_op"> {}

def Fly_GemmTraversalOrder : Fly_I32EnumAttr<"GemmTraversalOrder", "", [
  I32EnumAttrCase<"KMN", 0, "kmn">,
  I32EnumAttrCase<"KNM", 1, "knm">,
  I32EnumAttrCase<"MKN", 2, "mkn">,
  I32EnumAttrCase<"MNK", 3, "mnk">,
  I32EnumAttrCase<"NKM", 4, "nkm">,
  I32EnumAttrCase<"NMK", 5, "nmk">,
  I32EnumAttrCase<"KMN_Serpentine", 6, "kmn_serpentine">,
  I32EnumAttrCase<"KNM_Serpentine", 7, "knm_serpentine">,
  I32EnumAttrCase<"MKN_Serpentine", 8, "mkn_serpentine">,
  I32EnumAttrCase<"MNK_Serpentine", 9, "mnk_serpentine">,
  I32EnumAttrCase<"NKM_Serpentine", 10, "nkm_serpentine">,
  I32EnumAttrCase<"NMK_Serpentine", 11, "nmk_serpentine">
]>;
def Fly_GemmTraversalOrderAttr : Fly_EnumAttr<Fly_GemmTraversalOrder, "gemm_traversal_order"> {}

// Owning values in storage; callers receive an immutable borrowing reference.
class Fly_CoreParameter<string type> : AttrParameter<type, "Core algebra value"> {
  let cppAccessorType = "const " # type # " &";
}

//===----------------------------------------------------------------------===//
// Type attributes
//===----------------------------------------------------------------------===//

def Fly_AlignAttr : Fly_Attr<"Align", "align", []> {
  let parameters = (ins "int32_t":$alignment);
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins "int32_t":$alignment), [{
      if (alignment <= 0)
        return {};
      return $_get($_ctxt, alignment);
    }]>,
    AttrBuilderWithInferredContext<(ins "Type":$elemTy, "int32_t":$divisibility), [{
      auto bits = int64_t{elemTy.getIntOrFloatBitWidth()} * std::max(divisibility, 1);
      return getFromBitAlignment(elemTy.getContext(), bits);
    }]>
  ];
  let extraClassDeclaration = [{
    int64_t getBitAlignment() const { return int64_t{getAlignment()} * 8; }
    /// Returns null for non-positive, non-byte-aligned, or overflowing values.
    static AlignAttr getFromBitAlignment(MLIRContext *ctx, int64_t bits);
    static AlignAttr getTrivialAlignment(Type elemTy);
  }];
}

/// A Core scalar or basis, with static, dynamic, ratio, or None payload.
def Fly_LeafAttr : Fly_Attr<"Leaf", "leaf", [
    DeclareAttrInterfaceMethods<Fly_MayStaticAttrInterface>
]> {
  let parameters = (ins Fly_CoreParameter<"core::Leaf<core::Trait>">:$ref);
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    AttrBuilder<(ins "core::Leaf<core::Trait>":$ref), [{
      if (ref.isError())
        return {};
      return $_get($_ctxt, std::move(ref));
    }]>,
    AttrBuilder<(ins "LeafAttr":$scale, "ArrayRef<int32_t>":$modes), [{
      if (!scale || scale.isBasis() || scale.isNone() || modes.empty())
        return {};
      // Preserve Fly's explicit 0E... syntax while Core algebra may fold zero.
      auto value = scale.getRef();
      auto leaf = core::Leaf<core::Trait>::getBasis(
          value.isZero() ? core::Leaf<core::Trait>::getOne() : value, modes);
      if (value.isZero() && leaf.isBasis())
        leaf.payload64_ = 0;
      return get($_ctxt, leaf);
    }]>,
    AttrBuilderWithInferredContext<(ins "LeafAttr":$scale, "ArrayRef<int32_t>":$modes), [{
      return get(scale.getContext(), scale, modes);
    }]>
  ];

  let extraClassDeclaration = [{
    bool isBasis() const { return getRef().isBasis(); }
    bool isRatio() const;
    bool isNone() const;
    bool isStaticValue(int64_t value) const;

    int64_t getValue() const;
    int32_t getWidth() const;
    int64_t getDivisibility() const;
    int32_t getNumerator() const;
    int32_t getDenominator() const;
    LeafAttr getScale() const { return get(getContext(), getRef().scale()); }
    ArrayRef<uint8_t> getModes() const { return getRef().modes(); }

    static LeafAttr getRatio(MLIRContext *ctx, int32_t numerator, int32_t denominator);
    static LeafAttr getNone(MLIRContext *ctx);
    static LeafAttr getStatic(MLIRContext *ctx, int64_t value);
    static LeafAttr getDynamic(MLIRContext *ctx, int32_t width = 32, int32_t divisibility = 1);
  }];
}

def Fly_SwizzleAttr : Fly_Attr<"Swizzle", "swizzle", []> {
  let parameters = (ins Fly_CoreParameter<"core::Swizzle<core::Trait>">:$ref);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    AttrBuilder<(ins "int32_t":$mask, "int32_t":$base, "int32_t":$shift), [{
      if (!core::Swizzle<core::Trait>::isValid(mask, base, shift))
        return {};
      return $_get($_ctxt, core::Swizzle<core::Trait>(mask, base, shift));
    }]>
  ];

  let extraClassDeclaration = [{
    int32_t getMask() const { return getRef().mask(); }
    int32_t getBase() const { return getRef().base(); }
    int32_t getShift() const { return getRef().shift(); }
    
    bool isTrivialSwizzle() const;
    static SwizzleAttr getTrivialSwizzle(MLIRContext *context);
  }];
}

def Fly_CoordSwizzleAttr : Fly_Attr<"CoordSwizzle", "coord_swizzle", []> {
  let parameters = (ins Fly_CoreParameter<"core::CoordSwizzle<core::Trait>">:$ref);
  let builders = [
    AttrBuilder<(ins "int32_t":$mask, "int32_t":$baseRow,
                    "ArrayRef<int32_t>":$modeRow, "int32_t":$baseCol,
                    "ArrayRef<int32_t>":$modeCol)>
  ];
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    int32_t getMask() const { return getRef().mask(); }
    int32_t getBaseRow() const { return getRef().rowBase(); }
    int32_t getBaseCol() const { return getRef().colBase(); }
    ArrayRef<uint8_t> getModeRow() const { return getRef().rowModes(); }
    ArrayRef<uint8_t> getModeCol() const { return getRef().colModes(); }
  }];
}

def Fly_IntTupleAttr : Fly_Attr<"IntTuple", "int_tuple", [
    DeclareAttrInterfaceMethods<Fly_NestedAttrInterface>,
    DeclareAttrInterfaceMethods<Fly_MayStaticAttrInterface>
]> {
  let parameters = (ins Fly_CoreParameter<"core::IntTuple<core::Trait>">:$ref);
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilder<(ins "core::IntTuple<core::Trait>":$ref), [{
      if (ref.isError())
        return {};
      return $_get($_ctxt, std::move(ref));
    }]>,
    AttrBuilderWithInferredContext<(ins "LeafAttr":$leaf), [{
      if (!leaf)
        return {};
      return get(leaf.getContext(), core::IntTuple<core::Trait>::fromLeaf(leaf.getRef()));
    }]>,
    AttrBuilderWithInferredContext<(ins "LeafAttr":$value, "ArrayRef<int32_t>":$modes), [{
      return get(LeafAttr::get(value, modes));
    }]>,
    // Compatibility with attribute-tree construction: a LeafAttr, or an ArrayAttr
    // of int tuples and leaves. Null for anything else. New code builds the Core
    // value instead.
    AttrBuilder<(ins "Attribute":$value)>,
    AttrBuilderWithInferredContext<(ins "Attribute":$value), [{
      return value ? get(value.getContext(), value) : IntTupleAttr();
    }]>
  ];

  let extraClassDeclaration = [{
    static IntTupleAttr getLeafNone(MLIRContext *ctx);
    static IntTupleAttr getLeafStatic(MLIRContext *ctx, int64_t value);
    static IntTupleAttr getLeafDynamic(MLIRContext *ctx, int32_t width = 32, int32_t divisibility = 1);

    bool isLeafInt() const;
    bool isLeafBasis() const;
    bool isLeafStaticValue(int64_t value) const;
    LeafAttr getLeaf() const;

    IntTupleAttr at(int32_t idx) const;
    IntTupleAttr at(const ArrayRef<int32_t> idxs) const;
  }];
}

def Fly_LayoutAttr : Fly_Attr<"Layout", "layout", [
    DeclareAttrInterfaceMethods<Fly_NestedAttrInterface>,
    DeclareAttrInterfaceMethods<Fly_MayStaticAttrInterface>
]> {
  let parameters = (ins Fly_CoreParameter<"core::Layout<core::Trait>">:$ref);
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    AttrBuilder<(ins "core::Layout<core::Trait>":$ref), [{
      if (ref.isError())
        return {};
      return $_get($_ctxt, std::move(ref));
    }]>,
    AttrBuilder<(ins "IntTupleAttr":$shape, "IntTupleAttr":$stride), [{
      if (!shape || !stride)
        return {};
      return get($_ctxt, core::Layout<core::Trait>(shape.getRef(), stride.getRef()));
    }]>,
    AttrBuilderWithInferredContext<(ins "IntTupleAttr":$shape, "IntTupleAttr":$stride), [{
      return get(shape.getContext(), shape, stride);
    }]>
  ];

  let extraClassDeclaration = [{
    IntTupleAttr getShape() const;
    IntTupleAttr getStride() const;
    bool isStaticShape() const;
    bool isStaticStride() const;

    LayoutAttr at(int32_t idx) const;
    LayoutAttr at(const ArrayRef<int32_t> idxs) const;
  }];
}

def Fly_ComposedLayoutAttr : Fly_Attr<"ComposedLayout", "composed_layout", [
    DeclareAttrInterfaceMethods<Fly_NestedAttrInterface>,
    DeclareAttrInterfaceMethods<Fly_MayStaticAttrInterface>
]> {
  let parameters = (ins Fly_CoreParameter<"core::ComposedLayout<core::Trait>">:$ref);
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
    
  let builders = [
    AttrBuilder<(ins "core::ComposedLayout<core::Trait>":$ref), [{
      if (ref.isError())
        return {};
      return $_get($_ctxt, std::move(ref));
    }]>,
    AttrBuilder<(ins "Attribute":$inner, "IntTupleAttr":$offset, "Attribute":$outer)>,
    AttrBuilderWithInferredContext<(ins "Attribute":$inner, "IntTupleAttr":$offset, "Attribute":$outer), [{
      return get(inner.getContext(), inner, offset, outer);
    }]>
  ];

  let extraClassDeclaration = [{
    Attribute getInner() const;
    IntTupleAttr getOffset() const;
    Attribute getOuter() const;
    bool isStaticOuter() const;
    bool isStaticInner() const;
    bool isStaticOffset() const;

    ComposedLayoutAttr at(int32_t idx) const;
    ComposedLayoutAttr at(const ArrayRef<int32_t> idxs) const;

  }];
}

def Fly_TileAttr : Fly_Attr<"Tile", "tile", []> {
  let parameters = (ins Fly_CoreParameter<"core::Tile<core::Trait>">:$ref);
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  
  let builders = [
    AttrBuilder<(ins "core::Tile<core::Trait>":$ref), [{
      if (ref.isError())
        return {};
      return $_get($_ctxt, std::move(ref));
    }]>,
    // Compatibility with attribute-tree construction: a tile, a layout, a leaf
    // that is neither a basis nor a ratio, or an ArrayAttr of modes. Null for
    // anything else. New code builds the Core value instead.
    AttrBuilder<(ins "Attribute":$value)>,
    AttrBuilderWithInferredContext<(ins "Attribute":$value), [{
      return value ? get(value.getContext(), value) : TileAttr();
    }]>
  ];

  let extraClassDeclaration = [{
    int32_t rank() const { return getRef().rank(); }
    bool isLeaf() const { return getRef().isLeaf(); }
    /// Mode `idx` of a tuple tile.
    TileAttr at(int32_t idx) const { return get(getContext(), getRef().at(idx)); }
  }];
}

def Fly_ExplicitModuleAttr : Fly_Attr<"ExplicitModule", "explicit_module", [
    OffloadingTranslationAttrTrait
]> {
  let summary = "GPU binary offloading handler with explicit Python-side module loading";
  let description = [{
    Offloading handler for FlyDSL JIT.  During LLVM translation it embeds the
    selected GPU binary and emits externally lookup-able
    `flydsl_gpu_module_init` and `flydsl_gpu_module_load_to_device`
    functions instead of loading through a global ctor.
  }];
  let assemblyFormat = "``";
}

def Fly_AotModuleAttr : Fly_Attr<"AotModule", "aot_module", [
    OffloadingTranslationAttrTrait
]> {
  let summary = "GPU binary offloading handler for exported (AOT) host objects";
  let description = [{
    Selects and embeds a GPU binary with internal linkage and emits prefixed
    module init, load, and unload functions. Kernel launches use the common
    FlyDSL AOT runtime ABI and runtime-managed per-device state. The optional
    target follows `gpu.select_object`: an integer selects by index, another
    attribute selects the object with an equal target, and omission selects
    the first object.
  }];
  let parameters = (ins
    StringRefParameter<"exported symbol prefix">:$symbolPrefix,
    OptionalParameter<"Attribute", "GPU object target to select">:$target
  );
  let assemblyFormat = [{
    `<` $symbolPrefix (`,` $target^)? `>`
  }];
  let genVerifyDecl = 1;
}

#endif // FLY_ATTRDEFS
