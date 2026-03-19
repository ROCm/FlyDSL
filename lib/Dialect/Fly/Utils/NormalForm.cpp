// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl/Dialect/Fly/Utils/NormalForm.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace mlir::fly {

bool isNormalForm(TypedValue<IntTupleType> value) {
  Operation *defOp = value.getDefiningOp();
  assert(defOp && "DSLTyped value must have a defining op");
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
  assert(defOp && "DSLTyped value must have a defining op");
  if (!defOp) {
    return false;
  }
  if (auto makeLayoutOp = dyn_cast<MakeLayoutOp>(defOp)) {
    return isNormalForm(makeLayoutOp.getShape()) && isNormalForm(makeLayoutOp.getStride());
  }
  return false;
}

bool isNormalForm(TypedValue<ComposedLayoutType> value) {
  Operation *defOp = value.getDefiningOp();
  assert(defOp && "DSLTyped value must have a defining op");
  if (!defOp) {
    return false;
  }

  auto isNormalComposedInner = [](Value inner) -> bool {
    if (auto layoutTyped = dyn_cast<TypedValue<LayoutType>>(inner)) {
      return isNormalForm(layoutTyped);
    } else if (auto composedTyped = dyn_cast<TypedValue<ComposedLayoutType>>(inner)) {
      return isNormalForm(composedTyped);
    } else if (auto swizzleTyped = dyn_cast<TypedValue<SwizzleType>>(inner)) {
      return true;
    }
    return false;
  };
  if (auto makeComposedOp = dyn_cast<MakeComposedLayoutOp>(defOp)) {
    if (!isNormalComposedInner(makeComposedOp.getInner())) {
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

bool isWeaklyNormalForm(TypedValue<MemRefType> value) {
  Operation *defOp = value.getDefiningOp();
  assert(defOp && "DSLTyped value must have a defining op");
  if (!defOp) {
    return false;
  }

  if (auto makeViewOp = dyn_cast<MakeViewOp>(defOp)) {
    auto layout = makeViewOp.getLayout();
    if (auto layoutTyped = dyn_cast<TypedValue<LayoutType>>(layout)) {
      return isNormalForm(layoutTyped);
    } else if (auto composedTyped = dyn_cast<TypedValue<ComposedLayoutType>>(layout)) {
      return isNormalForm(composedTyped);
    } else {
      return false;
    }
  } else {
    return false;
  }
}

bool isWeaklyNormalForm(TypedValue<CoordTensorType> value) {
  Operation *defOp = value.getDefiningOp();
  assert(defOp && "DSLTyped value must have a defining op");
  if (!defOp) {
    return false;
  }

  if (auto makeViewOp = dyn_cast<MakeViewOp>(defOp)) {
    auto layout = makeViewOp.getLayout();
    if (auto layoutTyped = dyn_cast<TypedValue<LayoutType>>(layout)) {
      return isNormalForm(layoutTyped);
    } else if (auto composedTyped = dyn_cast<TypedValue<ComposedLayoutType>>(layout)) {
      return isNormalForm(composedTyped);
    } else {
      return false;
    }
  } else {
    return false;
  }
}

} // namespace mlir::fly
