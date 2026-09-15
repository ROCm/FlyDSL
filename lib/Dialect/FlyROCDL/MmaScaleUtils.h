// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_LIB_DIALECT_FLYROCDL_MMASCALEUTILS_H
#define FLYDSL_LIB_DIALECT_FLYROCDL_MMASCALEUTILS_H

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::fly_rocdl {

// Read an explicit scale fragment, falling back to atom state when omitted.
inline FailureOr<Value> getMmaScale(OpBuilder &builder, Location loc, TypeRange operandTypes,
                                    ValueRange operands, Type scaleType, Value atomState,
                                    int64_t stateIndex) {
  if (operands.size() == 1)
    return builder.createOrFold<LLVM::ExtractValueOp>(loc, atomState,
                                                      ArrayRef<int64_t>{stateIndex});

  Type originalType = operandTypes[1];
  Value value = operands[1];
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
