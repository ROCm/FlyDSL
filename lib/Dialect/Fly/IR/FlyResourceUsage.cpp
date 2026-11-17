// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Kernel resources that Fly itself knows: the generic resource attribute and the ops that
// use shared memory. A backend reports its target-specific resources from its own ops.

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

using namespace mlir;
using namespace mlir::fly;

StringRef KernelResourceAttr::getDescription() const {
  switch (getValue()) {
  case KernelResource::SharedMemory:
    return "bytes of shared memory";
  }
  llvm_unreachable("unknown kernel resource");
}

/// Each shared allocation becomes its own LDS symbol, so padding between them is not counted.
void MakePtrOp::getResourceUsage(SmallVectorImpl<ResourceUsage> &usage) {
  auto ptrTy = dyn_cast<PointerType>(getResult().getType());
  auto dictAttrs = getDictAttrs();
  if (!ptrTy || !isGenericAddressSpace<AddressSpace::Shared>(ptrTy.getAddressSpace()) || !dictAttrs)
    return;
  if (auto allocBytes = dictAttrs->getAs<IntegerAttr>("allocBytes"))
    usage.push_back({ResourceUsage::Allocate,
                     KernelResourceAttr::get(getContext(), KernelResource::SharedMemory),
                     allocBytes.getInt(), getResult()});
}
