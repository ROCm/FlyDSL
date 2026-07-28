// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "flydsl/Dialect/Fly/Utils/TypeUtils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::fly {

Type toSSAValueType(Type elemTy) {
  auto intTy = dyn_cast<IntegerType>(elemTy);
  if (!intTy || !intTy.isUnsigned())
    return elemTy;
  return IntegerType::get(elemTy.getContext(), intTy.getWidth());
}

} // namespace mlir::fly
