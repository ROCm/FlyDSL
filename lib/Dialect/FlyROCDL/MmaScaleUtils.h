// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_LIB_DIALECT_FLYROCDL_MMASCALEUTILS_H
#define FLYDSL_LIB_DIALECT_FLYROCDL_MMASCALEUTILS_H

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::fly_rocdl {

// Auxiliary scale fragments reach target lowering as either a promoted vector
// or a pointer. Their representation is a property of the scaled MMA, not GEMM.
inline FailureOr<Value> loadMmaScale(OpBuilder &builder, Location loc, Type originalType,
                                     Value value, Type scaleType) {
  Type elementType = originalType;
  int64_t size = 1;
  if (auto memref = dyn_cast<fly::MemRefType>(originalType)) {
    elementType = memref.getElemTy();
    auto layout = dyn_cast<fly::LayoutAttr>(memref.getLayout());
    if (!layout || !layout.isStaticShape() || !layout.isStaticStride()) {
      emitError(loc, "scale fragments must have static layouts");
      return failure();
    }
    fly::LayoutBuilder<fly::LayoutAttr> layoutBuilder(builder.getContext());
    size = fly::layoutSize(layoutBuilder, layout).getLeafAsInt().getValue();
  } else if (auto vector = dyn_cast<VectorType>(originalType)) {
    elementType = vector.getElementType();
    size = vector.getNumElements();
  }
  if (elementType != scaleType) {
    emitError(loc) << "scale fragments must have " << scaleType << " elements";
    return failure();
  }
  if (size != 1) {
    emitError(loc, "scale fragments must have mode-0 size 1");
    return failure();
  }
  if (isa<fly::MemRefType>(originalType))
    return LLVM::LoadOp::create(builder, loc, scaleType, value).getResult();
  if (auto vector = dyn_cast<VectorType>(originalType))
    return builder.createOrFold<vector::ExtractOp>(loc, value,
                                                   SmallVector<int64_t>(vector.getRank(), 0));
  return value;
}

} // namespace mlir::fly_rocdl

#endif
