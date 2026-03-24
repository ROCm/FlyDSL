// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl/Dialect/Fly/Utils/IntUtils.h"

namespace mlir::fly {

IntAttr operator+(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    return IntAttr::getStatic(ctx, lhs.getValue() + rhs.getValue());
  }
  if (lhs.isStaticValue(0)) {
    return rhs;
  }
  if (rhs.isStaticValue(0)) {
    return lhs;
  }
  int32_t width = std::max(lhs.getWidth(), rhs.getWidth());
  int32_t lhsDiv = lhs.isStatic() ? lhs.getValue() : lhs.getDivisibility();
  int32_t rhsDiv = rhs.isStatic() ? rhs.getValue() : rhs.getDivisibility();
  return IntAttr::getDynamic(ctx, width, utils::divisibilityAdd(lhsDiv, rhsDiv));
}

IntAttr operator-(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    return IntAttr::getStatic(ctx, lhs.getValue() - rhs.getValue());
  }
  if (lhs.isStaticValue(0)) {
    return rhs;
  }
  if (rhs.isStaticValue(0)) {
    return lhs;
  }
  int32_t width = std::max(lhs.getWidth(), rhs.getWidth());
  int32_t lhsDiv = lhs.isStatic() ? lhs.getValue() : lhs.getDivisibility();
  int32_t rhsDiv = rhs.isStatic() ? rhs.getValue() : rhs.getDivisibility();
  return IntAttr::getDynamic(ctx, width, utils::divisibilitySub(lhsDiv, rhsDiv));
}

IntAttr operator*(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    return IntAttr::getStatic(ctx, lhs.getValue() * rhs.getValue());
  }
  if (lhs.isStaticValue(0)) {
    return IntAttr::getStatic(ctx, 0);
  }
  if (rhs.isStaticValue(0)) {
    return IntAttr::getStatic(ctx, 0);
  }
  int32_t width = std::max(lhs.getWidth(), rhs.getWidth());
  int32_t lhsDiv = lhs.isStatic() ? lhs.getValue() : lhs.getDivisibility();
  int32_t rhsDiv = rhs.isStatic() ? rhs.getValue() : rhs.getDivisibility();
  return IntAttr::getDynamic(ctx, width, utils::divisibilityMul(lhsDiv, rhsDiv));
}

IntAttr operator/(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    return IntAttr::getStatic(ctx, lhs.getValue() / rhs.getValue());
  }
  if (lhs.isStaticValue(0)) {
    return IntAttr::getStatic(ctx, 0);
  }
  int32_t width = std::max(lhs.getWidth(), rhs.getWidth());
  int32_t lhsDiv = lhs.isStatic() ? lhs.getValue() : lhs.getDivisibility();
  int32_t rhsDiv = rhs.isStatic() ? rhs.getValue() : rhs.getDivisibility();
  return IntAttr::getDynamic(ctx, width, utils::divisibilityDiv(lhsDiv, rhsDiv));
}

IntAttr operator%(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    return IntAttr::getStatic(ctx, lhs.getValue() % rhs.getValue());
  }
  if (rhs.isStaticValue(1)) {
    return IntAttr::getStatic(ctx, 0);
  }
  if (lhs.isStaticValue(0)) {
    return IntAttr::getStatic(ctx, 0);
  }
  int32_t width = std::max(lhs.getWidth(), rhs.getWidth());
  int32_t lhsDiv = lhs.isStatic() ? lhs.getValue() : lhs.getDivisibility();
  int32_t rhsDiv = rhs.isStatic() ? rhs.getValue() : rhs.getDivisibility();
  return IntAttr::getDynamic(ctx, width, utils::divisibilityModulo(lhsDiv, rhsDiv));
}

IntAttr operator&&(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStaticValue(0)) {
    return IntAttr::getStatic(ctx, 0);
  }
  if (rhs.isStaticValue(0)) {
    return IntAttr::getStatic(ctx, 0);
  }
  return IntAttr::getDynamic(ctx, 32, 1);
}

IntAttr operator||(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && lhs.getValue() != 0) {
    return IntAttr::getStatic(ctx, 1);
  }
  if (rhs.isStatic() && rhs.getValue() != 0) {
    return IntAttr::getStatic(ctx, 1);
  }
  return IntAttr::getDynamic(ctx, 32, 1);
}

IntAttr operator!(IntAttr val) {
  auto *ctx = val.getContext();
  if (val.isStatic()) {
    return IntAttr::getStatic(ctx, val.getValue() == 0 ? 1 : 0);
  }
  return IntAttr::getDynamic(ctx, 32, 1);
}

IntAttr operator<(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    return IntAttr::getStatic(ctx, lhs.getValue() < rhs.getValue() ? 1 : 0);
  }
  return IntAttr::getDynamic(ctx, 32, 1);
}

IntAttr operator<=(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    return IntAttr::getStatic(ctx, lhs.getValue() <= rhs.getValue() ? 1 : 0);
  }
  return IntAttr::getDynamic(ctx, 32, 1);
}

IntAttr operator>(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    return IntAttr::getStatic(ctx, lhs.getValue() > rhs.getValue() ? 1 : 0);
  }
  return IntAttr::getDynamic(ctx, 32, 1);
}

IntAttr operator>=(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    return IntAttr::getStatic(ctx, lhs.getValue() >= rhs.getValue() ? 1 : 0);
  }
  return IntAttr::getDynamic(ctx, 32, 1);
}

IntAttr operator==(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    return IntAttr::getStatic(ctx, lhs.getValue() == rhs.getValue() ? 1 : 0);
  }
  return IntAttr::getDynamic(ctx, 32, 1);
}

IntAttr operator!=(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    return IntAttr::getStatic(ctx, lhs.getValue() != rhs.getValue() ? 1 : 0);
  }
  return IntAttr::getDynamic(ctx, 32, 1);
}

IntAttr intMin(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    return IntAttr::getStatic(ctx, std::min(lhs.getValue(), rhs.getValue()));
  }
  int32_t width = std::max(lhs.getWidth(), rhs.getWidth());
  int32_t lhsDiv = lhs.isStatic() ? lhs.getValue() : lhs.getDivisibility();
  int32_t rhsDiv = rhs.isStatic() ? rhs.getValue() : rhs.getDivisibility();
  return IntAttr::getDynamic(ctx, width, utils::divisibilityMin(lhsDiv, rhsDiv));
}

IntAttr intMax(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    return IntAttr::getStatic(ctx, std::max(lhs.getValue(), rhs.getValue()));
  }
  int32_t width = std::max(lhs.getWidth(), rhs.getWidth());
  int32_t lhsDiv = lhs.isStatic() ? lhs.getValue() : lhs.getDivisibility();
  int32_t rhsDiv = rhs.isStatic() ? rhs.getValue() : rhs.getDivisibility();
  return IntAttr::getDynamic(ctx, width, utils::divisibilityMax(lhsDiv, rhsDiv));
}

IntAttr intSafeDiv(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    assert(lhs.getValue() % rhs.getValue() == 0);
    return IntAttr::getStatic(ctx, lhs.getValue() / rhs.getValue());
  }
  if (rhs.isStatic()) {
    assert(lhs.getDivisibility() % rhs.getValue() == 0);
  }
  if (lhs.isStaticValue(0)) {
    return lhs;
  }
  int32_t width = std::max(lhs.getWidth(), rhs.getWidth());
  int32_t lhsDiv = lhs.isStatic() ? lhs.getValue() : lhs.getDivisibility();
  int32_t rhsDiv = rhs.isStatic() ? rhs.getValue() : rhs.getDivisibility();
  return IntAttr::getDynamic(ctx, width, utils::divisibilityDiv(lhsDiv, rhsDiv));
}

IntAttr intCeilDiv(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    return IntAttr::getStatic(ctx, (lhs.getValue() + rhs.getValue() - 1) / rhs.getValue());
  }
  if (lhs.isStaticValue(0) || lhs.isStaticValue(1)) {
    return lhs;
  }
  int32_t width = std::max(lhs.getWidth(), rhs.getWidth());
  int32_t lhsDiv = lhs.isStatic() ? lhs.getValue() : lhs.getDivisibility();
  int32_t rhsDiv = rhs.isStatic() ? rhs.getValue() : rhs.getDivisibility();
  return IntAttr::getDynamic(ctx, width, utils::divisibilityCeilDiv(lhsDiv, rhsDiv));
}

IntAttr intShapeDiv(IntAttr lhs, IntAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.isStatic() && rhs.isStatic()) {
    assert((lhs.getValue() % rhs.getValue() == 0 || rhs.getValue() % lhs.getValue() == 0));
    return IntAttr::getStatic(ctx, (lhs.getValue() + rhs.getValue() - 1) / rhs.getValue());
  }
  if (lhs.isStaticValue(0) || lhs.isStaticValue(1)) {
    return lhs;
  }
  int32_t width = std::max(lhs.getWidth(), rhs.getWidth());
  int32_t lhsDiv = lhs.isStatic() ? lhs.getValue() : lhs.getDivisibility();
  int32_t rhsDiv = rhs.isStatic() ? rhs.getValue() : rhs.getDivisibility();
  return IntAttr::getDynamic(ctx, width, utils::divisibilityCeilDiv(lhsDiv, rhsDiv));
}

IntAttr intApplySwizzle(IntAttr v, SwizzleAttr swizzle) {
  auto *ctx = v.getContext();
  if (swizzle.isTrivialSwizzle()) {
    return v;
  }
  if (v.isStatic()) {
    int32_t S = swizzle.getShift();
    int32_t val = v.getValue();
    int32_t bitMsk = ((1 << swizzle.getMask()) - 1) << (swizzle.getBase() + S);
    int32_t shifted = (val & bitMsk) >> S;
    return IntAttr::getStatic(ctx, val ^ shifted);
  }
  return IntAttr::getDynamic(ctx, v.getWidth(),
                             utils::divisibilityApplySwizzle(v.getDivisibility(), swizzle));
}

BasisAttr operator*(BasisAttr lhs, IntAttr rhs) {
  return BasisAttr::get(lhs.getValue() * rhs, lhs.getModes());
}
BasisAttr operator*(IntAttr lhs, BasisAttr rhs) {
  return BasisAttr::get(lhs * rhs.getValue(), rhs.getModes());
}

IntAttr operator==(BasisAttr lhs, BasisAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.getModes() == rhs.getModes()) {
    return lhs.getValue() == rhs.getValue();
  }
  return IntAttr::getStatic(ctx, 0);
}
IntAttr operator!=(BasisAttr lhs, BasisAttr rhs) {
  auto *ctx = lhs.getContext();
  if (lhs.getModes() != rhs.getModes()) {
    return IntAttr::getStatic(ctx, 1);
  }
  return lhs.getValue() != rhs.getValue();
}

BasisAttr intSafeDiv(BasisAttr lhs, IntAttr rhs) {
  return BasisAttr::get(intSafeDiv(lhs.getValue(), rhs), lhs.getModes());
}
BasisAttr intCeilDiv(BasisAttr lhs, IntAttr rhs) {
  return BasisAttr::get(intCeilDiv(lhs.getValue(), rhs), lhs.getModes());
}

} // namespace mlir::fly
