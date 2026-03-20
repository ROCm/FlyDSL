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

#ifndef FLYDSL_DIALECT_UTILS_NORMALFORM_H
#define FLYDSL_DIALECT_UTILS_NORMALFORM_H

#include "mlir/IR/Attributes.h"
#include "mlir/Support/LogicalResult.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"

namespace mlir::fly {

bool isNormalForm(TypedValue<IntTupleType> value);
bool isNormalForm(TypedValue<BasisType> value);
bool isNormalForm(TypedValue<LayoutType> value);
bool isNormalForm(TypedValue<SwizzleType> value);
bool isNormalForm(TypedValue<ComposedLayoutType> value);
bool isNormalForm(TypedValue<TileType> value);
bool isNormalForm(TypedValue<CoordTensorType> value);

bool isNormalForm(TypedValue<PointerType> value);
bool isNormalForm(TypedValue<MemRefType> value);
bool isNormalForm(TypedValue<TiledCopyType> value);
bool isNormalForm(TypedValue<TiledMmaType> value);

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_UTILS_NORMALFORM_H
