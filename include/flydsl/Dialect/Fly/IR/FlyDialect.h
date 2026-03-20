// Copyright (c) 2025 FlyDSL Project Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FLYDSL_DIALECT_FLY_IR_DIALECT_H
#define FLYDSL_DIALECT_FLY_IR_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
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

#include "flydsl/Dialect/Fly/IR/FlyDialect.h.inc"
#include "flydsl/Dialect/Fly/IR/FlyEnums.h.inc"

#include "flydsl/Dialect/Fly/IR/FlyAttrInterfaces.h.inc"
#include "flydsl/Dialect/Fly/IR/FlyTypeInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "flydsl/Dialect/Fly/IR/FlyAttrDefs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/Fly/IR/FlyTypeDefs.h.inc"
#define GET_OP_CLASSES
#include "flydsl/Dialect/Fly/IR/FlyOps.h.inc"

namespace mlir::fly {
#include "flydsl/Dialect/Fly/IR/FlyAttrConstraints.h.inc"
#include "flydsl/Dialect/Fly/IR/FlyTypeConstraints.h.inc"

ParseResult parseMNKDimensionList(AsmParser &parser, int32_t &m, int32_t &n, int32_t &k);
void printMNKDimensionList(AsmPrinter &printer, int32_t m, int32_t n, int32_t k);

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_FLY_IR_DIALECT_H
