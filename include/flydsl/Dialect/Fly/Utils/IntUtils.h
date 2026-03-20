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

#ifndef FLYDSL_DIALECT_FLY_UTILS_INTUTILS_H
#define FLYDSL_DIALECT_FLY_UTILS_INTUTILS_H

#include "mlir/IR/Attributes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

#include <numeric>

namespace mlir::fly {
namespace utils {

inline bool isPowerOf2(int32_t value) { return value > 0 && (value & (value - 1)) == 0; }
inline int32_t divisibilityAdd(int32_t lhs, int32_t rhs) { return std::gcd(lhs, rhs); }
inline int32_t divisibilitySub(int32_t lhs, int32_t rhs) { return std::gcd(lhs, rhs); }
inline int32_t divisibilityMul(int32_t lhs, int32_t rhs) { return lhs * rhs; }
inline int32_t divisibilityDiv(int32_t lhs, int32_t rhs) { return 1; }
inline int32_t divisibilityCeilDiv(int32_t lhs, int32_t rhs) { return 1; }
inline int32_t divisibilityModulo(int32_t lhs, int32_t rhs) { return std::gcd(lhs, rhs); }
inline int32_t divisibilityMin(int32_t lhs, int32_t rhs) { return std::gcd(lhs, rhs); }
inline int32_t divisibilityMax(int32_t lhs, int32_t rhs) { return std::gcd(lhs, rhs); }
inline int32_t divisibilityApplySwizzle(int32_t lhs, SwizzleAttr swizzle) {
  return std::gcd(lhs, 1 << swizzle.getBase());
}

} // namespace utils

//===----------------------------------------------------------------------===//
// IntAttr operations
//===----------------------------------------------------------------------===//

IntAttr operator+(IntAttr lhs, IntAttr rhs);
IntAttr operator-(IntAttr lhs, IntAttr rhs);
IntAttr operator*(IntAttr lhs, IntAttr rhs);
IntAttr operator/(IntAttr lhs, IntAttr rhs);
IntAttr operator%(IntAttr lhs, IntAttr rhs);

IntAttr operator&&(IntAttr lhs, IntAttr rhs);
IntAttr operator||(IntAttr lhs, IntAttr rhs);
IntAttr operator!(IntAttr val);

IntAttr operator<(IntAttr lhs, IntAttr rhs);
IntAttr operator<=(IntAttr lhs, IntAttr rhs);
IntAttr operator>(IntAttr lhs, IntAttr rhs);
IntAttr operator>=(IntAttr lhs, IntAttr rhs);
IntAttr operator==(IntAttr lhs, IntAttr rhs);
IntAttr operator!=(IntAttr lhs, IntAttr rhs);

IntAttr intMin(IntAttr lhs, IntAttr rhs);
IntAttr intMax(IntAttr lhs, IntAttr rhs);
IntAttr intSafeDiv(IntAttr lhs, IntAttr rhs);
IntAttr intCeilDiv(IntAttr lhs, IntAttr rhs);
IntAttr intShapeDiv(IntAttr lhs, IntAttr rhs);
IntAttr intApplySwizzle(IntAttr v, SwizzleAttr swizzle);

//===----------------------------------------------------------------------===//
// BasisAttr operations
//===----------------------------------------------------------------------===//

BasisAttr operator*(BasisAttr lhs, IntAttr rhs);
BasisAttr operator*(IntAttr lhs, BasisAttr rhs);

IntAttr operator==(BasisAttr lhs, BasisAttr rhs);
IntAttr operator!=(BasisAttr lhs, BasisAttr rhs);

BasisAttr intSafeDiv(BasisAttr lhs, IntAttr rhs);
BasisAttr intCeilDiv(BasisAttr lhs, IntAttr rhs);

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_FLY_UTILS_INTUTILS_H
