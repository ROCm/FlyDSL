// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyROCDL/Utils/BufferFatPtr.h"

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::fly_rocdl;

LogicalResult GetBufferRsrcOp::inferReturnTypes(MLIRContext *context,
                                                std::optional<Location> location,
                                                ValueRange operands, DictionaryAttr attributes,
                                                OpaqueProperties properties, RegionRange regions,
                                                SmallVectorImpl<Type> &inferredReturnTypes) {
  auto ptrTy = dyn_cast<PointerType>(operands[0].getType());
  if (!ptrTy)
    return emitOptionalError(location, "GetBufferRsrcOp: expected a fly.ptr, got ",
                             operands[0].getType());
  if (!isTargetAddressSpace<BufferDescAddressAttr>(ptrTy.getAddressSpace()))
    return emitOptionalError(location,
                             "GetBufferRsrcOp: expected a buffer_desc address space pointer, got ",
                             ptrTy.getAddressSpace());
  inferredReturnTypes.assign({BufferFatPtr::getRsrcPtrType(context)});
  return success();
}
