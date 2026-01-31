#include "flydsl/Dialect/Fly/Utils/NormalForm.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace mlir::fly {

//===----------------------------------------------------------------------===//
// NormalBasis: (StaticOp)
// Note: MakeBasisOp is not currently defined, so only StaticOp is valid
//===----------------------------------------------------------------------===//
bool isNormalForm(TypedValue<BasisType> value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }
  // NormalBasis ::= (StaticOp)
  // return isa<MakeBasisOp>(defOp);
  return false;
}

//===----------------------------------------------------------------------===//
// NormalIntTuple: (StaticOp) | (MakeIntTupleOp $dyncElems)
//===----------------------------------------------------------------------===//
bool isNormalForm(TypedValue<IntTupleType> value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }
  // if (isa<StaticOp>(defOp)) {
  //   return true;
  // }
  if (isa<MakeIntTupleOp>(defOp)) {
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// NormalLayout: (StaticOp) | (MakeLayoutOp NormalIntTuple, NormalIntTuple)
//===----------------------------------------------------------------------===//
bool isNormalForm(TypedValue<LayoutType> value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }
  // NormalLayout ::= (StaticOp)
  // if (isa<StaticOp>(defOp)) {
  //   return true;
  // }
  // NormalLayout ::= (MakeLayoutOp NormalIntTuple, NormalIntTuple)
  if (auto makeLayoutOp = dyn_cast<MakeLayoutOp>(defOp)) {
    auto shape = makeLayoutOp.getShape();
    if (!isNormalForm(shape)) {
      return false;
    }
    // Stride is optional
    if (auto stride = makeLayoutOp.getStride()) {
      if (!isNormalForm(stride)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// NormalComposedLayout: (StaticOp)
//   | (MakeComposedLayoutOp (NormalSwizzle | NormalLayout | NormalComposedLayout),
//                           NormalIntTuple, NormalLayout)
//===----------------------------------------------------------------------===//

// Helper: Check if a Value is a valid inner for ComposedLayout
// Inner can be: SwizzleType (always static), LayoutType, or ComposedLayoutType
static bool isNormalInner(Value inner) {
  // auto innerType = inner.getType();

  // SwizzleAttr is embedded in PointerType, not a standalone type
  // Check if it's a LayoutType
  if (auto layoutTyped = dyn_cast<TypedValue<LayoutType>>(inner)) {
    return isNormalForm(layoutTyped);
  }
  // Check if it's a ComposedLayoutType
  if (auto composedTyped = dyn_cast<TypedValue<ComposedLayoutType>>(inner)) {
    return isNormalForm(composedTyped);
  }
  return false;
}

bool isNormalForm(TypedValue<ComposedLayoutType> value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }
  // NormalComposedLayout ::= (StaticOp)
  if (isa<StaticOp>(defOp)) {
    return true;
  }
  // NormalComposedLayout ::= (MakeComposedLayoutOp inner, offset, outer)
  if (auto makeComposedOp = dyn_cast<MakeComposedLayoutOp>(defOp)) {
    // Check inner: (NormalSwizzle | NormalLayout | NormalComposedLayout)
    if (!isNormalInner(makeComposedOp.getInner())) {
      return false;
    }
    // Check offset: NormalIntTuple
    if (!isNormalForm(makeComposedOp.getOffset())) {
      return false;
    }
    // Check outer: NormalLayout
    if (!isNormalForm(makeComposedOp.getOuter())) {
      return false;
    }
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// NormalTile: (StaticOp) | (MakeTileOp (NormalIntTuple | NormalLayout)+)
//===----------------------------------------------------------------------===//
bool isNormalForm(TypedValue<TileType> value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }
  // NormalTile ::= (StaticOp)
  if (isa<StaticOp>(defOp)) {
    return true;
  }
  // NormalTile ::= (MakeTileOp (NormalIntTuple | NormalLayout)+)
  if (auto makeTileOp = dyn_cast<MakeTileOp>(defOp)) {
    for (Value mode : makeTileOp.getModes()) {
      // Each mode can be IntTupleType or LayoutType
      if (auto intTupleTyped = dyn_cast<TypedValue<IntTupleType>>(mode)) {
        if (!isNormalForm(intTupleTyped)) {
          return false;
        }
      } else if (auto layoutTyped = dyn_cast<TypedValue<LayoutType>>(mode)) {
        if (!isNormalForm(layoutTyped)) {
          return false;
        }
      } else {
        // Unknown type
        return false;
      }
    }
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// NormalCoordTensor: (MakeCoordTensorOp NormalIntTuple, NormalAnyLayout)
// Note: Using MakeIdentityTensorOp as the closest equivalent
//===----------------------------------------------------------------------===//
bool isNormalForm(TypedValue<CoordTensorType> value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }
  // Static CoordTensor
  if (isa<StaticOp>(defOp)) {
    return true;
  }
  // NormalCoordTensor via MakeIdentityTensorOp
  if (auto makeIdentityTensorOp = dyn_cast<MakeIdentityTensorOp>(defOp)) {
    return isNormalForm(makeIdentityTensorOp.getShape());
  }
  return false;
}

//===----------------------------------------------------------------------===//
// NormalPointer and NormalMemRef
// These are typically created via operations and should be static or from
// well-formed construction operations
//===----------------------------------------------------------------------===//
bool isNormalForm(TypedValue<PointerType> value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    // Block arguments are considered normal form for pointers
    return true;
  }
  // StaticOp produces normal form
  if (isa<StaticOp>(defOp)) {
    return true;
  }
  // Other operations that produce pointers are considered normal
  // as long as they don't have structural requirements
  return true;
}

bool isNormalForm(TypedValue<MemRefType> value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    // Block arguments are considered normal form
    return true;
  }
  // StaticOp produces normal form
  if (isa<StaticOp>(defOp)) {
    return true;
  }
  // MakeFragmentLikeOp with normal layout source
  if (auto makeFragmentOp = dyn_cast<MakeFragmentLikeOp>(defOp)) {
    return isNormalForm(makeFragmentOp.getSrc());
  }
  return true;
}

bool isNormalLayout(Value value) {
  if (auto layoutTyped = dyn_cast<TypedValue<LayoutType>>(value)) {
    return isNormalForm(layoutTyped);
  }
  if (auto composedTyped = dyn_cast<TypedValue<ComposedLayoutType>>(value)) {
    return isNormalForm(composedTyped);
  }
  return false;
}

} // namespace mlir::fly
