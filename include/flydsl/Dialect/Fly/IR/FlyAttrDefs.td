#ifndef FLY_ATTRDEFS
#define FLY_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/Constraints.td"

include "flydsl/Dialect/Fly/IR/FlyDialect.td"

//===----------------------------------------------------------------------===//
// Enum attributes
//===----------------------------------------------------------------------===//
def Fly_CachePolicy : I32EnumAttr<"CachePolicy", "", [
  I32EnumAttrCase<"CacheGlobal", 0, "cache_global">,
  I32EnumAttrCase<"CacheAlways", 1, "cache_always">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = Fly_Dialect.cppNamespace;
}
def Fly_CachePolicyAttr : EnumAttr<Fly_Dialect, Fly_CachePolicy, "cache_policy"> {}

def Fly_AddressSpace : I32EnumAttr<"AddressSpace", "", [
  I32EnumAttrCase<"Global", 0, "global">,
  I32EnumAttrCase<"Shared", 1, "shared">,
  I32EnumAttrCase<"Register", 2, "register">,
  I32EnumAttrCase<"BufferDesc", 8, "buffer_desc">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = Fly_Dialect.cppNamespace;
}
def Fly_AddressSpaceAttr : EnumAttr<Fly_Dialect, Fly_AddressSpace, "address_space"> {}

def Fly_MmaOperand : I32EnumAttr<"MmaOperand", "", [
  I32EnumAttrCase<"A", 0, "a">,
  I32EnumAttrCase<"B", 1, "b">,
  I32EnumAttrCase<"C", 2, "c">,
  I32EnumAttrCase<"D", 3, "d">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = Fly_Dialect.cppNamespace;
}
def Fly_MmaOperandAttr : EnumAttr<Fly_Dialect, Fly_MmaOperand, "mma_operand"> {}

//===----------------------------------------------------------------------===//
// Type attributes
//===----------------------------------------------------------------------===//

def Fly_AlignAttr : Fly_Attr<"Align", "align", []> {
  let parameters = (ins "int32_t":$alignment);
  let assemblyFormat = "`align` `<` `` $alignment `` `>`";

  let extraClassDeclaration = [{
    static AlignAttr getTrivialAlignment(MLIRContext *context);
  }];

  let extraClassDefinition = [{
    AlignAttr $cppClass::getTrivialAlignment(MLIRContext *context) {
      return get(context, 1);
    }
  }];
}

/// Unified integer attribute that represents both static and dynamic integers.
/// - None : value == 0 and width == 0. it could be used as a Static 0 in the arithmetic operations.
/// - Static integer: value is the actual integer value (value != INT32_MIN)
/// - Dynamic integer: value == INT32_MIN (sentinel), width and divisibility describe the dynamic value
def Fly_IntAttr : Fly_Attr<"Int", "int", [
    DeclareAttrInterfaceMethods<Fly_MayStaticAttrInterface>
]> {
  let parameters = (ins 
    "int32_t":$value, // needs to be int64_t to support large integers in the future
    DefaultValuedParameter<"int32_t", "32">:$width, 
    DefaultValuedParameter<"int32_t", "1">:$divisibility);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    bool isNone() const;
    // value can't be INT32_MIN here
    bool isStaticValue(int32_t value) const;
    static IntAttr getNone(MLIRContext *ctx);
    static IntAttr getStatic(MLIRContext *ctx, int32_t value);
    static IntAttr getDynamic(MLIRContext *ctx, int32_t width = 32, int32_t divisibility = 1);
  }];

  let extraClassDefinition = [{
    bool $cppClass::isNone() const {
      return getValue() == 0 && getWidth() == 0;
    }
    bool $cppClass::isStaticValue(int32_t value) const {
      return getValue() == value;
    }
    IntAttr $cppClass::getNone(MLIRContext *ctx) {
      return get(ctx, 0, 0, 0);
    }
    IntAttr $cppClass::getStatic(MLIRContext *ctx, int32_t value) {
      return get(ctx, value, 32, value == 0 ? 1 : value);
    }
    IntAttr $cppClass::getDynamic(MLIRContext *ctx, int32_t width, int32_t divisibility) {
      return get(ctx, std::numeric_limits<int32_t>::min(), width, divisibility);
    }
  }];
}

def Fly_SwizzleAttr : Fly_Attr<"Swizzle", "swizzle", []> {
  let parameters = (ins "int32_t":$mask, "int32_t":$base, "int32_t":$shift);
  let assemblyFormat = "`` `S` `` `<` $mask `` `,` `` $base `` `,` `` $shift `` `>`";

  let extraClassDeclaration = [{
    bool isTrivialSwizzle() const;
    static SwizzleAttr getTrivialSwizzle(MLIRContext *context);
  }];

  let extraClassDefinition = [{
    bool $cppClass::isTrivialSwizzle() const {
      return getMask() == 0;
    }
    SwizzleAttr $cppClass::getTrivialSwizzle(MLIRContext *context) {
      return get(context, 0, 0, 0);
    }
  }];
}

def Fly_BasisAttr : Fly_Attr<"Basis", "basis", [
    DeclareAttrInterfaceMethods<Fly_MayStaticAttrInterface>
]> {
  let parameters = (ins
    "Attribute":$value,
    ArrayRefParameter<"int32_t">:$modes
  );
  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "Attribute":$value, "int32_t":$mode), [{
      ::llvm::SmallVector<int32_t, 1> modes;
      modes.push_back(mode);
      return $_get(value.getContext(), value, modes);
    }]>
  ];
  
  let extraClassDeclaration = [{
    int32_t depth();
  }];
}

def Fly_IntTupleAttr : Fly_Attr<"IntTuple", "int_tuple", [
    DeclareAttrInterfaceMethods<Fly_NestedAttrInterface>,
    DeclareAttrInterfaceMethods<Fly_MayStaticAttrInterface>
]> {
  let parameters = (ins
    "Attribute":$value
  );
  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilder<(ins "int32_t":$value), [{
      return $_get($_ctxt, IntAttr::getStatic($_ctxt, value));
    }]>,
    AttrBuilderWithInferredContext<(ins "Attribute":$value), [{
      return $_get(value.getContext(), value);
    }]>
  ];

  let extraClassDeclaration = [{
    static IntTupleAttr getLeafNone(MLIRContext *ctx);
    static IntTupleAttr getLeafStatic(MLIRContext *ctx, int32_t value);
    static IntTupleAttr getLeafDynamic(MLIRContext *ctx, int32_t width = 32, int32_t divisibility = 1);

    bool isLeafNone() const;
    bool isLeafStaticValue(int32_t value) const;
    IntAttr getLeafAsInt() const;
    BasisAttr getLeafAsBasis() const;

    IntTupleAttr at(int32_t idx) const;
    IntTupleAttr at(const ArrayRef<int32_t> idxs) const;

    int32_t dyncLeafCount() const;
  }];
}

def Fly_LayoutAttr : Fly_Attr<"Layout", "layout", [
    DeclareAttrInterfaceMethods<Fly_NestedAttrInterface>,
    DeclareAttrInterfaceMethods<Fly_MayStaticAttrInterface>
]> {
  let parameters = (ins
    Fly_IntTupleAttr:$shape,
    Fly_IntTupleAttr:$stride
  );
  let assemblyFormat = "`` $shape `` `:` `` $stride";
  
  let builders = [
    AttrBuilderWithInferredContext<(ins "IntTupleAttr":$shape, "IntTupleAttr":$stride), [{
      return $_get(shape.getContext(), shape, stride);
    }]>
  ];

  let extraClassDeclaration = [{
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
  let parameters = (ins
    "Attribute":$inner,
    Fly_IntTupleAttr:$offset,
    Fly_LayoutAttr:$outer
  );
  let assemblyFormat = "`<` $inner `o` $offset `o` $outer `>`";
    
  let builders = [
    AttrBuilderWithInferredContext<(ins "Attribute":$inner, "IntTupleAttr":$offset, "LayoutAttr":$outer), [{
      return $_get(inner.getContext(), inner, offset, outer);
    }]>
  ];

  let extraClassDeclaration = [{
    bool isStaticOuter() const;
    bool isStaticInner() const;
    bool isStaticOffset() const;

    ComposedLayoutAttr at(int32_t idx) const;
    ComposedLayoutAttr at(const ArrayRef<int32_t> idxs) const;
  }];
}

def Fly_TileAttr : Fly_Attr<"Tile", "tile", []> {
  let parameters = (ins
    "Attribute":$value
  );
  let hasCustomAssemblyFormat = 1;
  
  let builders = [
    AttrBuilderWithInferredContext<(ins "Attribute":$value), [{
      return $_get(value.getContext(), value);
    }]>
  ];

  let extraClassDeclaration = [{
    int32_t rank() const;

    bool isLeaf() const;
    bool isNoneMode() const;
    bool isNoneMode(int32_t idx) const;

    Attribute at(int32_t idx) const;
  }];
}

def Fly_LeafAttr : AnyAttrOf<[Fly_IntAttr, Fly_BasisAttr]> {
  let cppFunctionName = "isValidLeafAttr";
}
def Fly_AnyLayoutAttr : AnyAttrOf<[Fly_LayoutAttr, Fly_ComposedLayoutAttr]> {
  let cppFunctionName = "isAnyValidLayoutAttr";
}

#endif // FLY_ATTRDEFS
