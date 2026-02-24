#include "flydsl/Dialect/Fly/Utils/NormalForm.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace mlir::fly {

// Swizzle and Tile are always static type, only the attribute matters
bool isNormalForm(TypedValue<SwizzleType> value) { return true; }
bool isNormalForm(TypedValue<TileType> value) { return true; }

bool isNormalForm(TypedValue<BasisType> value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }
  // return isa<MakeBasisOp>(defOp);
  return false;
}

bool isNormalForm(TypedValue<IntTupleType> value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }

  if (isa<MakeIntTupleOp>(defOp)) {
    return true;
  }
  return false;
}

bool isNormalForm(TypedValue<LayoutType> value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }
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

bool isNormalInner(Value inner) {
  if (auto layoutTyped = dyn_cast<TypedValue<LayoutType>>(inner)) {
    return isNormalForm(layoutTyped);
  } else if (auto composedTyped = dyn_cast<TypedValue<ComposedLayoutType>>(inner)) {
    return isNormalForm(composedTyped);
  } else if (auto swizzleTyped = dyn_cast<TypedValue<SwizzleType>>(inner)) {
    return isNormalForm(swizzleTyped);
  }
  return false;
}

bool isNormalForm(TypedValue<ComposedLayoutType> value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }
  // NormalComposedLayout ::= (MakeComposedLayoutOp inner, offset, outer)
  if (auto makeComposedOp = dyn_cast<MakeComposedLayoutOp>(defOp)) {
    if (!isNormalInner(makeComposedOp.getInner())) {
      return false;
    }
    if (!isNormalForm(makeComposedOp.getOffset())) {
      return false;
    }
    if (!isNormalForm(makeComposedOp.getOuter())) {
      return false;
    }
    return true;
  }
  return false;
}

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

bool isNormalForm(TypedValue<TiledCopyType> value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }

  if (isa<MakeTiledCopyOp>(defOp)) {
    // LayoutThrVal and TileMN are required as static.
    return true;
  }
  return false;
}

bool isNormalForm(TypedValue<TiledMmaType> value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }
  if (isa<MakeTiledMmaOp>(defOp)) {
    // AtomLayout and Permutation are required as static.
    return true;
  }
  return false;
}

} // namespace mlir::fly
