#ifndef FLY_DIALECT_UTILS_INTUTILS_H
#define FLY_DIALECT_UTILS_INTUTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/Support/LogicalResult.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

#include <limits>

namespace mlir::fly {
namespace utils {

inline int32_t divisibilityAdd(int32_t lhs, int32_t rhs) { return std::gcd(lhs, rhs); }
inline int32_t divisibilitySub(int32_t lhs, int32_t rhs) { return std::gcd(lhs, rhs); }
inline int32_t divisibilityMul(int32_t lhs, int32_t rhs) { return lhs * rhs; }
inline int32_t divisibilityDiv(int32_t lhs, int32_t rhs) { return 1; }
inline int32_t divisibilityCeilDiv(int32_t lhs, int32_t rhs) { return 1; }
inline int32_t divisibilityModulo(int32_t lhs, int32_t rhs) { return std::gcd(lhs, rhs); }
inline int32_t divisibilityMin(int32_t lhs, int32_t rhs) { return std::gcd(lhs, rhs); }
inline int32_t divisibilityMax(int32_t lhs, int32_t rhs) { return std::gcd(lhs, rhs); }

} // namespace utils

/// Sentinel value for dynamic integers
constexpr int32_t kDynamicIntSentinel = std::numeric_limits<int32_t>::min();

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

} // namespace mlir::fly

#endif // FLY_DIALECT_UTILS_INTUTILS_H
