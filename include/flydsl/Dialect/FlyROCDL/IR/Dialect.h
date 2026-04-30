// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLYROCDL_IR_DIALECT_H
#define FLYDSL_DIALECT_FLYROCDL_IR_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

#include "flydsl/Dialect/FlyROCDL/IR/AtomStateEnums.h.inc"
#include "flydsl/Dialect/FlyROCDL/IR/AttrEnums.h.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyROCDL/IR/Atom.h.inc"
#define GET_ATTRDEF_CLASSES
#include "flydsl/Dialect/FlyROCDL/IR/AttrDefs.h.inc"

namespace mlir::fly_rocdl {

template <TargetAddressSpace expectedAddressSpace> bool isTargetAddressSpace(Attribute attr) {
  auto addressSpace = llvm::dyn_cast_if_present<TargetAddressSpaceAttr>(attr);
  return addressSpace && addressSpace.getValue() == expectedAddressSpace;
}

} // namespace mlir::fly_rocdl

#endif // FLYDSL_DIALECT_FLYROCDL_IR_DIALECT_H
