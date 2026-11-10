// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "flydsl/Core/Trait.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>

namespace mlir::fly::core {
namespace trait {
namespace {

struct Range {
  bool has = false;
  int32_t lo = 0, hi = 0;
};

constexpr uint32_t kFdmBit = static_cast<uint32_t>(DynIntFlag::FastDivMod);
#if FLYDSL_CORE_ENABLE_RANGE_INFERENCE
constexpr uint32_t kRangeBit = static_cast<uint32_t>(DynIntFlag::Range);
#endif

constexpr int32_t kI32Min = std::numeric_limits<int32_t>::min();
constexpr int32_t kI32Max = std::numeric_limits<int32_t>::max();

bool isInInt32Range(int64_t v) { return v >= kI32Min && v <= kI32Max; }

LogWidth joinLogWidth(LogWidth a, LogWidth b) { return a > b ? a : b; }
LogWidth minLogWidth(int64_t v) { return isInInt32Range(v) ? LogWidth::I32 : LogWidth::I64; }

#if FLYDSL_CORE_ENABLE_RANGE_INFERENCE
int32_t minValue(LogWidth logWidth) {
  unsigned bits = 1u << logWidth;
  if (bits == 1)
    return 0;
  if (bits >= 32)
    return kI32Min;
  return static_cast<int32_t>(-(1 << (bits - 1)));
}

int32_t maxValue(LogWidth logWidth) {
  unsigned bits = 1u << logWidth;
  if (bits == 1)
    return 1;
  if (bits >= 32)
    return kI32Max;
  return static_cast<int32_t>((1 << (bits - 1)) - 1);
}
#endif

int32_t saturateInt32(int64_t value) {
  return static_cast<int32_t>(std::clamp<int64_t>(value, kI32Min, kI32Max));
}

/// An unrepresentable interval is unknown, never a saturated interval.
Range checkedRange(int64_t lo, int64_t hi) {
  if (!isInInt32Range(lo) || !isInInt32Range(hi))
    return {};
  return {true, static_cast<int32_t>(lo), static_cast<int32_t>(hi)};
}

int64_t ceilDivI64(int64_t a, int64_t b) {
  int64_t q = a / b, r = a % b;
  if (r != 0 && ((a < 0) == (b < 0)))
    ++q;
  return q;
}
// Optional range carried by a dynamic integer.
uint64_t modeWord(Leaf<Trait> l) {
  uint64_t w;
  std::memcpy(&w, l.modes_, sizeof(w));
  return w;
}
void setModeWord(Leaf<Trait> &l, uint64_t w) { std::memcpy(l.modes_, &w, sizeof(w)); }

Leaf<Trait> makeRatio(int64_t num, int64_t den) {
  if (den == 0)
    return Leaf<Trait>::getError(ErrorCode::DivisionByZero);
  if (den < 0) {
    num = -num;
    den = -den;
  }
  auto g = std::gcd(num, den);
  num /= g;
  den /= g;
  if (den == 1)
    return Leaf<Trait>::getSInt(num);
  if (num < kI32Min || num > kI32Max || den > kI32Max)
    return Leaf<Trait>::getError(ErrorCode::RatioOverflow);
  Leaf<Trait> ratio;
  ratio.kind_ = LeafKind::Ratio;
  ratio.logWidth_ = static_cast<uint32_t>(LogWidth::I32);
  ratio.payload32_ = {static_cast<int32_t>(num), static_cast<int32_t>(den)};
  return ratio;
}

Range rangeOf(Leaf<Trait> s) {
#if FLYDSL_CORE_ENABLE_RANGE_INFERENCE
  if (s.kind_ == LeafKind::StaticInt || s.kind_ == LeafKind::None)
    return isInInt32Range(s.payload64_)
               ? Range{true, static_cast<int32_t>(s.payload64_), static_cast<int32_t>(s.payload64_)}
               : Range{};
  if (s.kind_ == LeafKind::DynamicInt && (s.flags_ & kRangeBit) != 0)
    return {true, s.rangeMin(), s.rangeMax()};
#else
  (void)s;
#endif
  return {};
}

/// A dynamic integer known to be a multiple of `div`, within `range` when the
/// width can hold it.
Leaf<Trait> dynamic(LogWidth logWidth, int32_t div, Range range = {}) {
  Leaf<Trait> result = Leaf<Trait>::getDInt(logWidth, div);
#if FLYDSL_CORE_ENABLE_RANGE_INFERENCE
  if (!range.has)
    return result;
  if (range.lo < minValue(logWidth) || range.hi > maxValue(logWidth))
    return result;
  return result.withRange(range.lo, range.hi);
#else
  (void)range;
  return result;
#endif
}

/// The type width of an operand: a static integer has the type of its literal,
/// i32 unless it needs i64.
LogWidth typeLogWidth(Leaf<Trait> value) {
  return value.isSInt() ? minLogWidth(value.payload64_) : value.logWidth();
}

/// The width of a dynamic bitwise or shift result: the wider operand type.
LogWidth bitwiseLogWidth(Leaf<Trait> a, Leaf<Trait> b) {
  return joinLogWidth(typeLogWidth(a), typeLogWidth(b));
}

/// The width of any other dynamic result: the wider operand type, with i1
/// promoted to i32 as the DSL widens Boolean for arithmetic.
LogWidth resultLogWidth(Leaf<Trait> a, Leaf<Trait> b) {
  auto logWidth = bitwiseLogWidth(a, b);
  return logWidth == LogWidth::I1 ? LogWidth::I32 : logWidth;
}

/// `value`, which an identity returns unchanged, at the width of the result it
/// stands for; a static operand keeps the width of its literal.
Leaf<Trait> widen(Leaf<Trait> value, LogWidth logWidth) {
  auto scale = value.scale();
  if (!scale.isDInt() || scale.logWidth() == logWidth)
    return value;
  return dynamic(logWidth, scale.divisibility(), rangeOf(scale)).withModes(value);
}

/// What a dynamic result can rely on about the divisibility of an operand:
/// the divisibility of a dynamic integer, the magnitude of a static one.
int32_t divisibility(Leaf<Trait> value) {
  if (value.kind_ == LeafKind::DynamicInt)
    return value.div_;
  auto magnitude = value.payload64_ < 0 ? -value.payload64_ : value.payload64_;
  return magnitude > kI32Max ? 1 : static_cast<int32_t>(magnitude);
}

/// The largest power of two dividing `value`.
int32_t lowestBit(int32_t value) {
  auto bits = static_cast<uint32_t>(value);
  return static_cast<int32_t>(bits & (~bits + 1));
}

/// A static integer or ratio as its numerator and denominator.
std::pair<int64_t, int64_t> fraction(Leaf<Trait> value) {
  if (value.isRatio())
    return {value.num(), value.den()};
  return {value.payload64_, 1};
}

/// The order of static `a` and `b`, at least one of them a ratio, as -1, 0 or
/// 1.
int compareFractions(Leaf<Trait> a, Leaf<Trait> b) {
  auto [an, ad] = fraction(a);
  auto [bn, bd] = fraction(b);
  auto lhs = an * bd, rhs = bn * ad;
  return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
}

/// The error of an operation that has no rule for a ratio operand.
Leaf<Trait> ratioError(Leaf<Trait> a, Leaf<Trait> b) {
  return Leaf<Trait>::getError(a.isDInt() || b.isDInt() ? ErrorCode::DynamicRatioUnsupported
                                                        : ErrorCode::UnsupportedRatioOperation);
}

/// A comparison whose result is not known statically.
Leaf<Trait> unknownComparison() { return Leaf<Trait>::getDInt().withRange(0, 1); }

Leaf<Trait> shiftStatic(bool left, Leaf<Trait> value, int64_t amount) {
  if (amount < 0) {
    left = !left;
    amount = -amount;
  }
  if (amount == 0)
    return value;
  auto v = value.payload64_;
  if (!left)
    return Leaf<Trait>::getSInt(v >> std::min<int64_t>(amount, 63));
  return Leaf<Trait>::getSInt(amount < 64 ? static_cast<int64_t>(static_cast<uint64_t>(v) << amount)
                                          : 0);
}

//===----------------------------------------------------------------------===//
// Leaf rules
//===----------------------------------------------------------------------===//

// One rule per operation. A rule never sees an error operand: `apply` returns
// that before the rule runs.

using Rule = Leaf<Trait> (*)(Leaf<Trait>, Leaf<Trait>);

Leaf<Trait> mul(Leaf<Trait> a, Leaf<Trait> b);

Leaf<Trait> add(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis()) {
    if (b.isZero())
      return widen(a, resultLogWidth(a.scale(), b));
    if (a.isZero())
      return widen(b, resultLogWidth(a, b.scale()));
    return Leaf<Trait>::getError(ErrorCode::IncompatibleBasis);
  }
  if (a.isDInt() || b.isDInt()) {
    if (a.isRatio() || b.isRatio())
      return Leaf<Trait>::getError(ErrorCode::DynamicRatioUnsupported);
    auto ia = rangeOf(a), ib = rangeOf(b);
    Range range;
    if (ia.has && ib.has)
      range = checkedRange(int64_t{ia.lo} + ib.lo, int64_t{ia.hi} + ib.hi);
    return dynamic(resultLogWidth(a, b), std::gcd(divisibility(a), divisibility(b)), range);
  }
  if (a.isRatio() || b.isRatio()) {
    auto [an, ad] = fraction(a);
    auto [bn, bd] = fraction(b);
    return makeRatio(an * bd + bn * ad, ad * bd);
  }
  return Leaf<Trait>::getSInt(a.payload64_ + b.payload64_);
}

Leaf<Trait> sub(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis()) {
    if (b.isZero())
      return widen(a, resultLogWidth(a.scale(), b));
    if (a.isZero())
      return mul(b.scale(), Leaf<Trait>::getSInt(-1)).withModes(b);
    return Leaf<Trait>::getError(ErrorCode::IncompatibleBasis);
  }
  if (a.isDInt() || b.isDInt()) {
    if (a.isRatio() || b.isRatio())
      return Leaf<Trait>::getError(ErrorCode::DynamicRatioUnsupported);
    auto ia = rangeOf(a), ib = rangeOf(b);
    Range range;
    if (ia.has && ib.has)
      range = checkedRange(int64_t{ia.lo} - ib.hi, int64_t{ia.hi} - ib.lo);
    return dynamic(resultLogWidth(a, b), std::gcd(divisibility(a), divisibility(b)), range);
  }
  if (a.isRatio() || b.isRatio()) {
    auto [an, ad] = fraction(a);
    auto [bn, bd] = fraction(b);
    return makeRatio(an * bd - bn * ad, ad * bd);
  }
  return Leaf<Trait>::getSInt(a.payload64_ - b.payload64_);
}

Leaf<Trait> mul(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis()) {
    if (a.isZero() || b.isZero())
      return Leaf<Trait>::getSInt(0);
    if (!b.isBasis())
      return mul(a.scale(), b).withModes(a);
    if (!a.isBasis())
      return mul(a, b.scale()).withModes(b);
    return Leaf<Trait>::getError(ErrorCode::UnsupportedBasisOperation);
  }
  if (a.isDInt() || b.isDInt()) {
    if (a.isRatio() || b.isRatio()) {
      // A rational stride applied to a provably aligned dynamic coordinate is
      // still an integer. Divide first so the numerator does not overflow an
      // otherwise representable result; never truncate a fractional coordinate.
      auto ratio = a.isRatio() ? a : b;
      auto value = a.isRatio() ? b : a;
      if (value.divisibility() % ratio.den() != 0)
        return Leaf<Trait>::getError(ErrorCode::DynamicRatioUnsupported);
      return safe_div(value, Leaf<Trait>::getSInt(ratio.den())) * Leaf<Trait>::getSInt(ratio.num());
    }
    if (a.isZero() || b.isZero())
      return Leaf<Trait>::getSInt(0);
    if (a.isOne())
      return widen(b, resultLogWidth(a, b));
    if (b.isOne())
      return widen(a, resultLogWidth(a, b));
    auto da = divisibility(a), db = divisibility(b);
    auto product = static_cast<int64_t>(da) * db;
    return dynamic(resultLogWidth(a, b),
                   product <= kI32Max ? static_cast<int32_t>(product) : std::max(da, db));
  }
  if (a.isRatio() || b.isRatio()) {
    auto [an, ad] = fraction(a);
    auto [bn, bd] = fraction(b);
    return makeRatio(an * bn, ad * bd);
  }
  return Leaf<Trait>::getSInt(a.payload64_ * b.payload64_);
}

/// The basis rule of the divisions: zero divided by a basis is zero, and a
/// basis divided by a scalar divides its scale.
Leaf<Trait> divideBasis(Rule rule, Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isZero())
    return Leaf<Trait>::getSInt(0);
  if (!b.isBasis())
    return rule(a.scale(), b).withModes(a);
  return Leaf<Trait>::getError(ErrorCode::UnsupportedBasisOperation);
}

/// The rounded quotient of `a` and `b`, one of them dynamic.
Leaf<Trait> dynamicQuotient(Leaf<Trait> a, Leaf<Trait> b) {
  auto logWidth = resultLogWidth(a, b);
  if (!a.isDInt() || !b.isSInt())
    return dynamic(logWidth, 1);
  auto divisor = b.payload64_ < 0 ? -b.payload64_ : b.payload64_;
  // An exact quotient keeps the remaining divisibility; otherwise nothing
  // beyond 1 holds for either rounding.
  return dynamic(logWidth, a.div_ % divisor == 0 ? static_cast<int32_t>(a.div_ / divisor) : 1);
}

Leaf<Trait> div(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis())
    return divideBasis(div, a, b);
  if (a.isRatio() || b.isRatio())
    return ratioError(a, b);
  if (a.isDInt() || b.isDInt()) {
    if (b.isZero())
      return Leaf<Trait>::getError(ErrorCode::DivisionByZero);
    if (b.isOne())
      return widen(a, resultLogWidth(a, b));
    if (a.isZero())
      return Leaf<Trait>::getSInt(0);
    return dynamicQuotient(a, b);
  }
  if (b.isZero())
    return Leaf<Trait>::getError(ErrorCode::DivisionByZero);
  return Leaf<Trait>::getSInt(a.payload64_ / b.payload64_);
}

Leaf<Trait> ceilDiv(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis())
    return divideBasis(ceilDiv, a, b);
  if (a.isRatio() || b.isRatio())
    return ratioError(a, b);
  if (a.isDInt() || b.isDInt()) {
    if (b.isZero())
      return Leaf<Trait>::getError(ErrorCode::DivisionByZero);
    if (b.isOne())
      return widen(a, resultLogWidth(a, b));
    if (a.isZero())
      return Leaf<Trait>::getSInt(0);
    // ceil_div is defined only for b > 0 (Category.def), where
    // ceil_div(1, b) == 1; shape_div folds 0 and 1 the same way.
    if (a.isOne())
      return a;
    return dynamicQuotient(a, b);
  }
  if (b.isZero())
    return Leaf<Trait>::getError(ErrorCode::DivisionByZero);
  return Leaf<Trait>::getSInt(ceilDivI64(a.payload64_, b.payload64_));
}

Leaf<Trait> mod(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis())
    return divideBasis(mod, a, b);
  if (a.isRatio() || b.isRatio())
    return ratioError(a, b);
  if (a.isDInt() || b.isDInt()) {
    if (b.isZero())
      return Leaf<Trait>::getError(ErrorCode::DivisionByZero);
    if (b.isOne() || b.isSInt(-1) || a.isZero())
      return Leaf<Trait>::getSInt(0);
    auto logWidth = resultLogWidth(a, b);
    if (a.isDInt() && b.isSInt()) {
      auto divisor = b.payload64_ < 0 ? -b.payload64_ : b.payload64_;
      if (a.div_ % divisor == 0)
        return Leaf<Trait>::getSInt(0);
      Range range;
      if (divisor <= static_cast<int64_t>(kI32Max) + 1) {
        auto bound = divisor - 1;
        auto input = rangeOf(a);
        if (input.has && input.lo >= 0)
          range = {true, 0, saturateInt32(bound)};
        else if (input.has && input.hi <= 0)
          range = {true, saturateInt32(-bound), 0};
        else
          range = {true, saturateInt32(-bound), saturateInt32(bound)};
      }
      return dynamic(logWidth, static_cast<int32_t>(std::gcd<int64_t>(a.div_, divisor)), range);
    }
    if (a.isSInt() && b.isDInt()) {
      auto av = a.payload64_;
      Range range;
      if (av >= 0 && isInInt32Range(av))
        range = {true, 0, static_cast<int32_t>(av)};
      else if (av < 0 && isInInt32Range(av))
        range = {true, static_cast<int32_t>(av), 0};
      return dynamic(logWidth, 1, range);
    }
    return dynamic(logWidth, 1);
  }
  if (b.isZero())
    return Leaf<Trait>::getError(ErrorCode::DivisionByZero);
  return Leaf<Trait>::getSInt(a.payload64_ % b.payload64_);
}

Leaf<Trait> exactDiv(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis())
    return divideBasis(exactDiv, a, b);
  if (a.isDInt() || b.isDInt()) {
    if (a.isRatio() || b.isRatio())
      return Leaf<Trait>::getError(ErrorCode::DynamicRatioUnsupported);
    if (b.isZero())
      return Leaf<Trait>::getError(ErrorCode::DivisionByZero);
    if (b.isOne())
      return widen(a, resultLogWidth(a, b));
    if (a.isZero())
      return Leaf<Trait>::getSInt(0);
    if (!a.isDInt() || !b.isSInt())
      return Leaf<Trait>::getError(ErrorCode::InexactDivision);
    auto divisor = b.payload64_ < 0 ? -b.payload64_ : b.payload64_;
    if (a.div_ % divisor != 0)
      return Leaf<Trait>::getError(ErrorCode::InexactDivision);
    return dynamic(resultLogWidth(a, b), static_cast<int32_t>(a.div_ / divisor));
  }
  if (a.isRatio() || b.isRatio()) {
    auto [an, ad] = fraction(a);
    auto [bn, bd] = fraction(b);
    return makeRatio(an * bd, ad * bn);
  }
  return makeRatio(a.payload64_, b.payload64_);
}

Leaf<Trait> safeDiv(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis())
    return divideBasis(safeDiv, a, b);
  if (a.isRatio() || b.isRatio())
    return ratioError(a, b);
  // safe_div checks exactness only between static operands; with a dynamic
  // operand it is the plain truncating quotient.
  if (a.isDInt() || b.isDInt())
    return div(a, b);
  if (b.isZero())
    return Leaf<Trait>::getError(ErrorCode::DivisionByZero);
  if (a.payload64_ % b.payload64_ != 0)
    return Leaf<Trait>::getError(ErrorCode::InexactDivision);
  return Leaf<Trait>::getSInt(a.payload64_ / b.payload64_);
}

Leaf<Trait> shapeDiv(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis())
    return divideBasis(shapeDiv, a, b);
  if (a.isRatio() || b.isRatio())
    return ratioError(a, b);
  if (a.isDInt() || b.isDInt()) {
    if (b.isZero())
      return Leaf<Trait>::getError(ErrorCode::DivisionByZero);
    if (a.isZero() || a.isOne())
      return a;
    return dynamicQuotient(a, b);
  }
  if (b.isZero())
    return Leaf<Trait>::getError(ErrorCode::DivisionByZero);
  auto av = a.payload64_, bv = b.payload64_;
  if (av != 0 && av % bv != 0 && av != 1 && av != -1 && bv % av != 0)
    return Leaf<Trait>::getError(ErrorCode::IncompatibleShapeDivision);
  return Leaf<Trait>::getSInt(ceilDivI64(av, bv));
}

Leaf<Trait> minimum(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis())
    return Leaf<Trait>::getError(ErrorCode::UnsupportedBasisOperation);
  if (a.isRatio() || b.isRatio())
    return ratioError(a, b);
  if (a.isDInt() || b.isDInt()) {
    auto ia = rangeOf(a), ib = rangeOf(b);
    Range range;
    if (ia.has && ib.has)
      range = {true, std::min(ia.lo, ib.lo), std::min(ia.hi, ib.hi)};
    return dynamic(resultLogWidth(a, b), std::gcd(divisibility(a), divisibility(b)), range);
  }
  return Leaf<Trait>::getSInt(std::min(a.payload64_, b.payload64_));
}

Leaf<Trait> maximum(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis())
    return Leaf<Trait>::getError(ErrorCode::UnsupportedBasisOperation);
  if (a.isRatio() || b.isRatio())
    return ratioError(a, b);
  if (a.isDInt() || b.isDInt()) {
    auto ia = rangeOf(a), ib = rangeOf(b);
    Range range;
    if (ia.has && ib.has)
      range = {true, std::max(ia.lo, ib.lo), std::max(ia.hi, ib.hi)};
    return dynamic(resultLogWidth(a, b), std::gcd(divisibility(a), divisibility(b)), range);
  }
  return Leaf<Trait>::getSInt(std::max(a.payload64_, b.payload64_));
}

Leaf<Trait> greatestCommonDivisor(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis())
    return Leaf<Trait>::getError(ErrorCode::UnsupportedBasisOperation);
  if (a.isRatio() || b.isRatio())
    return ratioError(a, b);
  if (a.isDInt() || b.isDInt()) {
    if (a.isZero())
      return widen(abs(b), resultLogWidth(a, b));
    if (b.isZero())
      return widen(abs(a), resultLogWidth(a, b));
    if (a.isOne() || a.isSInt(-1) || b.isOne() || b.isSInt(-1))
      return Leaf<Trait>::getSInt(1);
    auto staticDividesDynamic = [](int64_t staticValue, int32_t div) {
      auto magnitude = staticValue < 0 ? -staticValue : staticValue;
      return magnitude > 0 && magnitude <= kI32Max && div % static_cast<int32_t>(magnitude) == 0;
    };
    if (a.isSInt() && b.isDInt() && staticDividesDynamic(a.payload64_, b.div_))
      return Leaf<Trait>::getSInt(a.payload64_ < 0 ? -a.payload64_ : a.payload64_);
    if (b.isSInt() && a.isDInt() && staticDividesDynamic(b.payload64_, a.div_))
      return Leaf<Trait>::getSInt(b.payload64_ < 0 ? -b.payload64_ : b.payload64_);
    auto ia = rangeOf(a), ib = rangeOf(b);
    Range range;
    if (ia.has && ib.has) {
      auto absBound = [](int64_t v) { return v < 0 ? -v : v; };
      range = checkedRange(0, std::min(std::max(absBound(ia.lo), absBound(ia.hi)),
                                       std::max(absBound(ib.lo), absBound(ib.hi))));
    }
    return dynamic(resultLogWidth(a, b), std::gcd(divisibility(a), divisibility(b)), range);
  }
  return Leaf<Trait>::getSInt(std::gcd(a.payload64_, b.payload64_));
}

Leaf<Trait> bitAnd(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis()) {
    if (a.isZero() || b.isZero())
      return Leaf<Trait>::getSInt(0);
    return Leaf<Trait>::getError(ErrorCode::UnsupportedBasisOperation);
  }
  if (a.isRatio() || b.isRatio())
    return ratioError(a, b);
  if (a.isDInt() || b.isDInt()) {
    if (a.isZero() || b.isZero())
      return Leaf<Trait>::getSInt(0);
    // A multiple of 2^k has no bit below 2^k, so a static mask below 2^k
    // clears every bit.
    auto clears = [](Leaf<Trait> value, Leaf<Trait> mask) {
      return value.isDInt() && mask.isSInt() && mask.payload64_ >= 0 &&
             mask.payload64_ < lowestBit(value.div_);
    };
    if (clears(a, b) || clears(b, a))
      return Leaf<Trait>::getSInt(0);
    return dynamic(bitwiseLogWidth(a, b),
                   std::max(lowestBit(divisibility(a)), lowestBit(divisibility(b))));
  }
  return Leaf<Trait>::getSInt(a.payload64_ & b.payload64_);
}

Leaf<Trait> bitOr(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis())
    return Leaf<Trait>::getError(ErrorCode::UnsupportedBasisOperation);
  if (a.isRatio() || b.isRatio())
    return ratioError(a, b);
  if (a.isDInt() || b.isDInt()) {
    if (a.isZero())
      return widen(b, bitwiseLogWidth(a, b));
    if (b.isZero())
      return widen(a, bitwiseLogWidth(a, b));
    return dynamic(bitwiseLogWidth(a, b),
                   std::min(lowestBit(divisibility(a)), lowestBit(divisibility(b))));
  }
  return Leaf<Trait>::getSInt(a.payload64_ | b.payload64_);
}

Leaf<Trait> bitXor(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis())
    return Leaf<Trait>::getError(ErrorCode::UnsupportedBasisOperation);
  if (a.isRatio() || b.isRatio())
    return ratioError(a, b);
  if (a.isDInt() || b.isDInt()) {
    if (a.isZero())
      return widen(b, bitwiseLogWidth(a, b));
    if (b.isZero())
      return widen(a, bitwiseLogWidth(a, b));
    return dynamic(bitwiseLogWidth(a, b),
                   std::min(lowestBit(divisibility(a)), lowestBit(divisibility(b))));
  }
  return Leaf<Trait>::getSInt(a.payload64_ ^ b.payload64_);
}

/// `a` shifted left, or arithmetically right, by `b`; a negative amount
/// shifts the other way.
Leaf<Trait> shift(bool left, Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis())
    return Leaf<Trait>::getError(ErrorCode::UnsupportedBasisOperation);
  if (a.isRatio() || b.isRatio())
    return ratioError(a, b);
  if (!a.isDInt() && !b.isDInt())
    return shiftStatic(left, a, b.payload64_);
  auto logWidth = bitwiseLogWidth(a, b);
  if (b.isZero())
    return widen(a, logWidth);
  if (!a.isDInt() || !b.isSInt())
    return dynamic(logWidth, 1);
  auto amount = b.payload64_;
  if (amount < 0) {
    left = !left;
    amount = -amount;
  }
  if (amount >= (int64_t{1} << logWidth))
    return Leaf<Trait>::getError(ErrorCode::UnsupportedBitwiseOperand);
  if (left)
    return dynamic(logWidth,
                   amount < 31 && a.div_ <= (kI32Max >> amount) ? a.div_ << amount : a.div_);
  if (amount >= 31)
    return dynamic(logWidth, 1);
  auto scale = int32_t{1} << amount;
  return dynamic(logWidth, a.div_ % scale == 0 ? a.div_ / scale : 1);
}

Leaf<Trait> shl(Leaf<Trait> a, Leaf<Trait> b) { return shift(/*left=*/true, a, b); }

Leaf<Trait> shr(Leaf<Trait> a, Leaf<Trait> b) { return shift(/*left=*/false, a, b); }

Leaf<Trait> lessThan(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() && b.isBasis())
    return Leaf<Trait>::getError(ErrorCode::UnsupportedBasisComparison);
  if (a.isBasis() || b.isBasis())
    return Leaf<Trait>::getError(ErrorCode::UnsupportedBasisOperation);
  if (a.isDInt() || b.isDInt()) {
    if (a.isRatio() || b.isRatio())
      return Leaf<Trait>::getError(ErrorCode::DynamicRatioUnsupported);
    auto ia = rangeOf(a), ib = rangeOf(b);
    if (ia.has && ib.has) {
      if (ia.hi < ib.lo)
        return Leaf<Trait>::getSInt(1);
      if (ia.lo >= ib.hi)
        return Leaf<Trait>::getSInt(0);
    }
    return unknownComparison();
  }
  if (a.isRatio() || b.isRatio()) {
    return Leaf<Trait>::getSInt(compareFractions(a, b) < 0);
  }
  return Leaf<Trait>::getSInt(a.payload64_ < b.payload64_);
}

Leaf<Trait> equal(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis()) {
    // modes_ and nmodes share one word, so identical mode paths is one compare.
    if (a.isBasis() && b.isBasis() && modeWord(a) == modeWord(b))
      return equal(a.scale(), b.scale());
    return Leaf<Trait>::getSInt(0);
  }
  if (a.isDInt() || b.isDInt()) {
    if (a.isRatio() || b.isRatio())
      return Leaf<Trait>::getError(ErrorCode::DynamicRatioUnsupported);
    auto ia = rangeOf(a), ib = rangeOf(b);
    if (ia.has && ib.has) {
      if (ia.hi < ib.lo || ib.hi < ia.lo)
        return Leaf<Trait>::getSInt(0);
      if (ia.lo == ia.hi && ib.lo == ib.hi && ia.lo == ib.lo)
        return Leaf<Trait>::getSInt(1);
    }
    return unknownComparison();
  }
  if (a.isRatio() || b.isRatio()) {
    return Leaf<Trait>::getSInt(compareFractions(a, b) == 0);
  }
  return Leaf<Trait>::getSInt(a.payload64_ == b.payload64_);
}

Leaf<Trait> notEqual(Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isBasis() || b.isBasis()) {
    if (a.isBasis() && b.isBasis() && modeWord(a) == modeWord(b))
      return notEqual(a.scale(), b.scale());
    return Leaf<Trait>::getSInt(1);
  }
  if (a.isDInt() || b.isDInt()) {
    if (a.isRatio() || b.isRatio())
      return Leaf<Trait>::getError(ErrorCode::DynamicRatioUnsupported);
    auto ia = rangeOf(a), ib = rangeOf(b);
    if (ia.has && ib.has) {
      if (ia.lo == ia.hi && ib.lo == ib.hi)
        return Leaf<Trait>::getSInt(ia.lo != ib.lo);
      if (ia.hi < ib.lo || ib.hi < ia.lo)
        return Leaf<Trait>::getSInt(1);
    }
    return unknownComparison();
  }
  if (a.isRatio() || b.isRatio()) {
    return Leaf<Trait>::getSInt(compareFractions(a, b) != 0);
  }
  return Leaf<Trait>::getSInt(a.payload64_ != b.payload64_);
}

/// `rule` applied to `a` and `b`. An error operand propagates unchanged; an
/// error that neither operand carried was detected by `op` itself.
Leaf<Trait> apply(AlgebraOp op, Rule rule, Leaf<Trait> a, Leaf<Trait> b) {
  if (a.isError())
    return a;
  if (b.isError())
    return b;
  auto result = rule(a, b);
  if (result.isError() && result.errorInfo().depth() == 0)
    return Leaf<Trait>::getError(result.errorInfo().withFrame(op));
  return result;
}

Leaf<Trait> applyTypeSuffix(Leaf<Trait> leaf, const detail::LeafSuffix &suffix) {
  if (!leaf.isDInt())
    return suffix.range || suffix.fastDivMod || suffix.divisibility != 1 ||
                   suffix.logWidth != LogWidth::I32
               ? Leaf<Trait>::getError(ErrorCode::InvalidDynamicRange)
               : leaf;
  leaf = Leaf<Trait>::getDInt(suffix.logWidth, suffix.divisibility);
  if (suffix.range)
    leaf = leaf.withRange(suffix.range->first, suffix.range->second);
  if (suffix.fastDivMod && !leaf.isError())
    leaf = leaf.withFastDivMod();
  return leaf;
}

} // namespace
} // namespace trait

//===----------------------------------------------------------------------===//
// Leaf<Trait>
//===----------------------------------------------------------------------===//

Leaf<Trait> Leaf<Trait>::getNone() {
  Leaf<Trait> leaf;
  leaf.kind_ = LeafKind::None;
  return leaf;
}

Leaf<Trait> Leaf<Trait>::getError(ErrorInfo info) {
  Leaf<Trait> leaf;
  leaf.kind_ = LeafKind::Error;
  leaf.payload64_ = static_cast<int64_t>(info.raw());
  return leaf;
}

Leaf<Trait> Leaf<Trait>::getSInt(int64_t value) {
  Leaf<Trait> leaf;
  leaf.kind_ = LeafKind::StaticInt;
  leaf.logWidth_ = static_cast<uint32_t>(trait::minLogWidth(value));
  leaf.payload64_ = value;
  return leaf;
}

Leaf<Trait> Leaf<Trait>::getDInt(LogWidth logWidth, int32_t div) {
  Leaf<Trait> leaf;
  leaf.kind_ = LeafKind::DynamicInt;
  leaf.logWidth_ = static_cast<uint32_t>(logWidth);
  leaf.div_ = div <= 0 ? 1 : div;
  return leaf;
}

Leaf<Trait> Leaf<Trait>::getRatio(int32_t num, int32_t den) { return trait::makeRatio(num, den); }

Leaf<Trait> Leaf<Trait>::getBasis(Leaf<Trait> scaleVal, llvm::ArrayRef<int32_t> modeList) {
  Leaf<Trait> l = scaleVal.scale();
  if (l.isError())
    return l;
  if (modeList.empty() || l.isSInt(0))
    return l;
  if (!(l.kind_ <= LeafKind::DynamicInt || l.kind_ == LeafKind::Ratio))
    return getError(ErrorCode::InvalidBasisScale);
  if (modeList.size() > kMaxBasisModes)
    return getError(ErrorCode::TooManyBasisModes);
  for (unsigned i = 0; i < modeList.size(); ++i) {
    if (modeList[i] < 0 || modeList[i] > 255)
      return getError(ErrorCode::BasisModeOutOfRange);
    l.modes_[i] = static_cast<uint8_t>(modeList[i]);
  }
  // A basis scale is a plain integer; a fast-divmod divisor degrades to it.
  l.flags_ &= ~trait::kFdmBit;
  l.kind_ |= LeafKind::Basis;
  l.nmodes_ = static_cast<uint8_t>(modeList.size());
  return l;
}

Leaf<Trait> Leaf<Trait>::scale() const {
  Leaf<Trait> r = *this;
  r.kind_ &= ~LeafKind::Basis;
  trait::setModeWord(r, 0);
  return r;
}

Leaf<Trait> Leaf<Trait>::withModes(Leaf<Trait> proto) const {
  Leaf<Trait> r = scale();
  const bool scalar = r.kind_ <= LeafKind::DynamicInt || r.kind_ == LeafKind::Ratio;
  if (!scalar || r.isZero() || (proto.kind_ & LeafKind::Basis) == 0)
    return r;
  trait::setModeWord(r, trait::modeWord(proto));
  r.flags_ &= ~trait::kFdmBit;
  r.kind_ |= LeafKind::Basis;
  return r;
}

Leaf<Trait> Leaf<Trait>::withRange(int32_t lo, int32_t hi) const {
#if FLYDSL_CORE_ENABLE_RANGE_INFERENCE
  if (!isDInt())
    return getError(ErrorCode::RangeRequiresDynamicInt);
  if (lo > hi)
    return getError(ErrorCode::InvalidDynamicRange);
  const LogWidth lw = logWidth();
  if (lo < trait::minValue(lw) || hi > trait::maxValue(lw))
    return getError(ErrorCode::RangeOutsideBitWidth);

  // if lo == hi, return getSInt(lo)

  Leaf<Trait> result = *this;
  if (bitWidth() <= 32 && lo == trait::minValue(lw) && hi == trait::maxValue(lw)) {
    result.flags_ &= ~trait::kRangeBit;
    result.payload64_ = 0;
    return result;
  }
  result.flags_ |= trait::kRangeBit;
  result.payload32_ = {lo, hi};
  return result;
#else
  (void)lo;
  (void)hi;
  return *this;
#endif
}

Leaf<Trait> Leaf<Trait>::withFastDivMod() const {
  // A fast-divmod divisor is bound to a handle, so a static one becomes the
  // dynamic integer known about it.
  if (isSInt() && staticValue() >= 1)
    return trait::dynamic(logWidth(), trait::divisibility(*this), trait::rangeOf(*this))
        .withFastDivMod();
  if (!isDInt())
    return getError(ErrorCode::FastDivModRequiresPositiveInt);
  if (logWidth() != LogWidth::I32 && logWidth() != LogWidth::I64)
    return getError(ErrorCode::FastDivModRequiresI32OrI64);
  Leaf<Trait> result = *this;
  result.flags_ |= trait::kFdmBit;
  return result;
}

std::optional<Leaf<Trait>> Leaf<Trait>::fromString(llvm::StringRef text) {
  if (text == "*")
    return getNone();
  if (auto error = detail::parseError(text))
    return getError(*error);
  auto suffix = detail::parseLeafSuffix(text);
  if (!suffix || suffix->value.empty())
    return std::nullopt;

  Leaf<Trait> leaf;
  if (suffix->value.consume_front("?")) {
    if (!suffix->value.empty())
      return std::nullopt;
    leaf = getDInt(suffix->logWidth, suffix->divisibility);
  } else {
    auto slash = suffix->value.find('/');
    if (slash != llvm::StringRef::npos) {
      auto num = int32_t{0};
      auto den = int32_t{0};
      if (suffix->value.take_front(slash).getAsInteger(10, num) ||
          suffix->value.drop_front(slash + 1).getAsInteger(10, den))
        return std::nullopt;
      leaf = getRatio(num, den);
    } else {
      auto value = int64_t{0};
      if (suffix->value.getAsInteger(10, value))
        return std::nullopt;
      leaf = getSInt(value);
    }
  }
  leaf = trait::applyTypeSuffix(leaf, *suffix);
  if (leaf.isError())
    return std::nullopt;
  if (!suffix->modes.empty())
    leaf = getBasis(leaf, suffix->modes);
  return leaf;
}

//===----------------------------------------------------------------------===//
// Leaf operations
//===----------------------------------------------------------------------===//

llvm::hash_code hash_value(Leaf<Trait> l) {
  return hash_value(static_cast<const LeafProperty &>(l));
}

Leaf<Trait> operator+(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Add, trait::add, a, b);
}

Leaf<Trait> operator-(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Sub, trait::sub, a, b);
}

Leaf<Trait> operator*(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Mul, trait::mul, a, b);
}

Leaf<Trait> operator/(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Div, trait::div, a, b);
}

Leaf<Trait> operator%(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Mod, trait::mod, a, b);
}

Leaf<Trait> operator&(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::BitAnd, trait::bitAnd, a, b);
}

Leaf<Trait> operator|(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::BitOr, trait::bitOr, a, b);
}

Leaf<Trait> operator^(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::BitXor, trait::bitXor, a, b);
}

Leaf<Trait> operator<<(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Shl, trait::shl, a, b);
}

Leaf<Trait> operator>>(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Shr, trait::shr, a, b);
}

// Within its domain the logical shift is the arithmetic one.
Leaf<Trait> lshr(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::LShr, trait::shr, a, b);
}

Leaf<Trait> operator<(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Lt, trait::lessThan, a, b);
}

Leaf<Trait> eq(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Eq, trait::equal, a, b);
}

Leaf<Trait> ne(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Ne, trait::notEqual, a, b);
}

Leaf<Trait> lt(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Lt, trait::lessThan, a, b);
}

Leaf<Trait> min(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Min, trait::minimum, a, b);
}

Leaf<Trait> max(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Max, trait::maximum, a, b);
}

Leaf<Trait> gcd(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Gcd, trait::greatestCommonDivisor, a, b);
}

Leaf<Trait> exact_div(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Exact, trait::exactDiv, a, b);
}

Leaf<Trait> safe_div(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Safe, trait::safeDiv, a, b);
}

Leaf<Trait> ceil_div(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Ceil, trait::ceilDiv, a, b);
}

Leaf<Trait> shape_div(Leaf<Trait> a, Leaf<Trait> b) {
  return trait::apply(AlgebraOp::Shape, trait::shapeDiv, a, b);
}

Leaf<Trait> operator-(Leaf<Trait> a) { return a * Leaf<Trait>::getSInt(-1); }

Leaf<Trait> operator~(Leaf<Trait> a) { return a ^ Leaf<Trait>::getSInt(-1); }

Leaf<Trait> abs(Leaf<Trait> a) {
  // A fast-divmod divisor is positive.
  if (a.isError() || a.hasFastDivMod())
    return a;

  auto scale = a.scale();
  auto result = scale;
  if (scale.isSInt()) {
    auto value = scale.staticValue();
    result = Leaf<Trait>::getSInt(value < 0 ? -value : value);
  } else if (scale.isRatio()) {
    auto num = scale.num();
    if (num < 0)
      result = trait::makeRatio(-int64_t{num}, scale.den());
  } else if (scale.isDInt()) {
#if FLYDSL_CORE_ENABLE_RANGE_INFERENCE
    if (scale.hasRange()) {
      auto lo = scale.rangeMin();
      auto hi = scale.rangeMax();
      if (lo == INT32_MIN) {
        result.flags_ &= ~trait::kRangeBit;
        result.payload64_ = 0;
      } else if (hi <= 0) {
        result = scale.withRange(-hi, -lo);
      } else if (lo < 0) {
        result = scale.withRange(0, std::max(-lo, hi));
      }
    }
#endif
  } else {
    return scale;
  }
  return result.withModes(a);
}

Leaf<Trait> signum(Leaf<Trait> a) {
  if (a.isError())
    return a;
  if (a.hasFastDivMod())
    return Leaf<Trait>::getSInt(1);

  auto scale = a.scale();
  if (scale.isSInt()) {
    auto value = scale.staticValue();
    return Leaf<Trait>::getSInt(value < 0 ? -1 : value > 0 ? 1 : 0);
  }
  if (scale.isRatio()) {
    auto num = scale.num();
    return Leaf<Trait>::getSInt(num < 0 ? -1 : num > 0 ? 1 : 0);
  }
  if (scale.isDInt()) {
#if FLYDSL_CORE_ENABLE_RANGE_INFERENCE
    if (scale.hasRange()) {
      auto lo = scale.rangeMin();
      auto hi = scale.rangeMax();
      if (hi < 0)
        return Leaf<Trait>::getSInt(-1);
      if (lo > 0)
        return Leaf<Trait>::getSInt(1);
      if (lo == 0 && hi == 0)
        return Leaf<Trait>::getSInt(0);
      if (lo == 0)
        return Leaf<Trait>::getDInt().withRange(0, 1);
      if (hi == 0)
        return Leaf<Trait>::getDInt().withRange(-1, 0);
    }
    return Leaf<Trait>::getDInt().withRange(-1, 1);
#else
    return Leaf<Trait>::getDInt();
#endif
  }
  return scale;
}

void print(Leaf<Trait> l, llvm::raw_ostream &os) {
  if (l.isNone()) {
    os << '*';
    return;
  }
  auto s = l.scale();
  if (s.isError()) {
    os << get_error_message(s.errorCode());
  } else if (s.isSInt()) {
    os << s.staticValue();
  } else if (s.isRatio()) {
    os << s.num() << '/' << s.den();
  } else {
    os << '?';
    detail::printDynIntSuffix(s, os);
  }
  detail::printModes(l, os);
}

//===----------------------------------------------------------------------===//
// Pointer operations
//===----------------------------------------------------------------------===//

Pointer<Trait> add_offset(const Pointer<Trait> &pointer, IntTupleRef<Trait> offset) {
  FLYDSL_CORE_ASSERT(offset.isLeaf());
  auto bitAlignment = detail::offsetBitAlignment(pointer, offset);
  return Pointer<Trait>(pointer.elementType(), pointer.elementBitWidth(),
                        static_cast<int32_t>((bitAlignment + 7) / 8), bitAlignment,
                        pointer.swizzle(), pointer.addressSpace(),
                        pointer.storageElementBitWidth());
}

bool operator==(const Pointer<Trait> &lhs, const Pointer<Trait> &rhs) {
  return static_cast<const PointerProperty &>(lhs) == static_cast<const PointerProperty &>(rhs);
}

} // namespace mlir::fly::core
