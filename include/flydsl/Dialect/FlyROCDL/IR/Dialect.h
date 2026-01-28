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

#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Enums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyROCDL/IR/Atom.h.inc"

#endif // FLYDSL_DIALECT_FLYROCDL_IR_DIALECT_H
