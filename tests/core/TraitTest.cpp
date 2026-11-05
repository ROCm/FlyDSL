//===- TraitTest.cpp - The Trait category ----------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Leaf<Trait> and Pointer<Trait> of lib/Core/Trait.cpp: the static inference
// every category carries and the Fly dialect types store.  The parity files pin
// what a reference shows (values, static-ness, divisibility); these cases check
// what no reference has: integer widths, fast divmod, value ranges, ratio and
// basis leaves, errors, identity, the leaf text, and the bit-level alignment of
// sub-byte, swizzled and byte-stored pointers.
//
//===----------------------------------------------------------------------===//

#include "flydsl/Core/Trait.hpp"

#include "gtest/gtest.h"

#include <climits>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

using namespace mlir::fly::core;
using std::nullopt;
using std::optional;

namespace {

Leaf<Trait> i(int64_t v) { return Leaf<Trait>::getSInt(v); }
Leaf<Trait> d(LogWidth logWidth = LogWidth::I32, int32_t div = 1,
              optional<std::pair<int32_t, int32_t>> r = nullopt, bool fdm = false) {
  Leaf<Trait> s = Leaf<Trait>::getDInt(logWidth, div);
  if (r)
    s = s.withRange(r->first, r->second);
  if (fdm && !s.isError())
    s = s.withFastDivMod();
  EXPECT_FALSE(s.isError());
  return s;
}
Leaf<Trait> rat(int32_t n, int32_t den) {
  Leaf<Trait> s = Leaf<Trait>::getRatio(n, den);
  EXPECT_FALSE(s.isError());
  return s;
}
Leaf<Trait> bs(Leaf<Trait> scale, std::initializer_list<int32_t> modes) {
  return Leaf<Trait>::getBasis(scale, llvm::ArrayRef<int32_t>(modes.begin(), modes.size()));
}

} // namespace

TEST(TraitCategory, DynamicDefaults) {
  EXPECT_TRUE(d().isDInt());
  EXPECT_EQ(d().logWidth(), LogWidth::I32);
  EXPECT_EQ(d().bitWidth(), 32);
  EXPECT_EQ(d().divisibility(), 1);
  EXPECT_FALSE(d().hasRange());
  EXPECT_FALSE(d().hasFastDivMod());
  EXPECT_EQ(d(LogWidth::I64).logWidth(), LogWidth::I64);
  EXPECT_EQ(d(LogWidth::I64).bitWidth(), 64);
  EXPECT_EQ(d(LogWidth::I32, 8).divisibility(), 8);
  EXPECT_EQ(d(), d(LogWidth::I32, 1));
#if FLYDSL_CORE_ENABLE_RANGE_INFERENCE
  EXPECT_TRUE(d(LogWidth::I32, 1, {{0, 63}}).hasRange());
  EXPECT_EQ(d(LogWidth::I32, 1, {{0, 63}}).rangeMin(), 0);
  EXPECT_EQ(d(LogWidth::I32, 1, {{0, 63}}).rangeMax(), 63);
  EXPECT_FALSE(d(LogWidth::I32, 1, {{INT32_MIN, INT32_MAX}}).hasRange());
  EXPECT_EQ(d(LogWidth::I64, 1, {{INT32_MIN, INT32_MAX}}).logWidth(), LogWidth::I64);
  EXPECT_TRUE(d(LogWidth::I64, 1, {{INT32_MIN, INT32_MAX}}).hasRange());
  EXPECT_EQ(d(LogWidth::I64, 1, {{INT32_MIN, INT32_MAX}}).rangeMin(), INT32_MIN);
  EXPECT_EQ(d(LogWidth::I64, 1, {{INT32_MIN, INT32_MAX}}).rangeMax(), INT32_MAX);
  EXPECT_EQ(Leaf<Trait>::getDInt(LogWidth::I8).withRange(1 << 16, 1 << 16).errorCode(),
            ErrorCode::RangeOutsideBitWidth);
#endif
  EXPECT_EQ(Leaf<Trait>::getDInt(LogWidth::I16).withFastDivMod().errorCode(),
            ErrorCode::FastDivModRequiresI32OrI64);
}

TEST(TraitCategory, DivisibilityClamp) {
  EXPECT_EQ(d(LogWidth::I32, 8).divisibility(), 8);
  EXPECT_EQ(d(LogWidth::I32, 0).divisibility(), 1);
  EXPECT_EQ(d(LogWidth::I32, -4).divisibility(), 1);
}

TEST(TraitCategory, StaticWidth) {
  EXPECT_EQ(i(4).staticValue(), 4);
  EXPECT_EQ(i(1ll << 40).staticValue(), 1099511627776ll);
  EXPECT_EQ(i(4).logWidth(), LogWidth::I32);
  EXPECT_EQ(i(4).bitWidth(), 32);
  EXPECT_EQ(i(INT32_MIN).logWidth(), LogWidth::I32);
  EXPECT_EQ(i(INT32_MAX).logWidth(), LogWidth::I32);
  EXPECT_EQ(i(1ll << 31).logWidth(), LogWidth::I64);
  EXPECT_EQ(i(1ll << 31).bitWidth(), 64);
  EXPECT_EQ(i(1ll << 40).logWidth(), LogWidth::I64);
}

TEST(TraitCategory, WidthJoin) {
  EXPECT_EQ((d() + d(LogWidth::I64)).logWidth(), LogWidth::I64);
  EXPECT_EQ((d() + d()).logWidth(), LogWidth::I32);
  EXPECT_EQ(d() + d(LogWidth::I64), d(LogWidth::I64) + d());

  auto wide = i(int64_t{1} << 40);
  auto expectI64 = [](Leaf<Trait> value) {
    EXPECT_TRUE(value.isDInt());
    EXPECT_EQ(value.logWidth(), LogWidth::I64);
  };
  expectI64(d() + wide);
  expectI64(wide + d());
  expectI64(d() - wide);
  expectI64(wide - d());
  expectI64(d() * wide);
  expectI64(wide * d());
  expectI64(d() / wide);
  expectI64(wide / d());
  expectI64(d() % wide);
  expectI64(wide % d());
  expectI64(min(d(), wide));
  expectI64(min(wide, d()));
  expectI64(max(d(), wide));
  expectI64(max(wide, d()));
}

TEST(TraitCategory, IdentityShortcutsKeepWidthAndFastDivMod) {
  EXPECT_TRUE((d() | i(-1)).isDInt());
  EXPECT_TRUE((i(-1) | d()).isDInt());
  EXPECT_TRUE((i(0) << d()).isDInt());
  EXPECT_TRUE((i(0) >> d()).isDInt());

  EXPECT_EQ(d() * i(0), i(0));
  EXPECT_EQ(i(0) * d(), i(0));
  EXPECT_EQ(i(0) / d(), i(0));
  EXPECT_EQ(d() % i(-1), i(0));
  EXPECT_EQ(i(0) % d(), i(0));
  EXPECT_EQ(d() & i(0), i(0));
  EXPECT_EQ(i(0) & d(), i(0));

  auto narrow = d(LogWidth::I8, 1, std::nullopt, false);
  EXPECT_EQ((narrow + i(0)).logWidth(), LogWidth::I32);
  EXPECT_EQ((i(0) + narrow).logWidth(), LogWidth::I32);
  EXPECT_EQ((narrow - i(0)).logWidth(), LogWidth::I32);

  auto fast = d(LogWidth::I32, 1, std::nullopt, true);
  EXPECT_TRUE((fast * i(1)).hasFastDivMod());
  EXPECT_TRUE((fast / i(1)).hasFastDivMod());
  EXPECT_TRUE((fast | i(0)).hasFastDivMod());
  EXPECT_TRUE((fast ^ i(0)).hasFastDivMod());
  EXPECT_TRUE((fast << i(0)).hasFastDivMod());
}

TEST(TraitCategory, WideArithmetic) {
  EXPECT_EQ(i(int64_t{1} << 40) + i(int64_t{1} << 39), i(int64_t{3} << 39));
  EXPECT_EQ(i(int64_t{1} << 40) - i(int64_t{1} << 39), i(int64_t{1} << 39));
  EXPECT_EQ(i(int64_t{1} << 30) * i(int64_t{1} << 20), i(int64_t{1} << 50));
  EXPECT_EQ(i(int64_t{1} << 50) / i(int64_t{1} << 20), i(int64_t{1} << 30));
  EXPECT_EQ(ceil_div(i((int64_t{1} << 50) + 1), i(int64_t{1} << 20)), i((int64_t{1} << 30) + 1));
  EXPECT_EQ(exact_div(i(int64_t{1} << 50), i(int64_t{1} << 20)), i(int64_t{1} << 30));
  EXPECT_EQ((rat(1, 2) + i(4)), rat(9, 2));
  EXPECT_EQ((i(6) * rat(2, 3)), i(4));
  EXPECT_EQ((rat(std::numeric_limits<int32_t>::max(), 2) + rat(1, 3)).errorCode(),
            ErrorCode::RatioOverflow);
  EXPECT_EQ((rat(std::numeric_limits<int32_t>::max(), 2) * i(3)).errorCode(),
            ErrorCode::RatioOverflow);

  auto wideDivisibility = d(LogWidth::I64, 65536) * d(LogWidth::I64, 65536);
  EXPECT_TRUE(wideDivisibility.isDInt());
  EXPECT_EQ(wideDivisibility.divisibility(), 65536);
}

TEST(TraitCategory, FastDivMod) {
  EXPECT_FALSE(Leaf<Trait>::getDInt(LogWidth::I32).withFastDivMod().isError());
  EXPECT_TRUE(d(LogWidth::I32, 1, nullopt, true).hasFastDivMod());
  EXPECT_TRUE(d(LogWidth::I64, 1, nullopt, true).hasFastDivMod());
  EXPECT_EQ(Leaf<Trait>::getDInt(LogWidth::I8).withFastDivMod().errorCode(),
            ErrorCode::FastDivModRequiresI32OrI64);
  // Only a dynamic integer is a divisor: a static one becomes dynamic,
  // keeping its divisibility (and range, with range inference).
  auto seven = i(7).withFastDivMod();
  ASSERT_TRUE(seven.isDInt());
  EXPECT_TRUE(seven.hasFastDivMod());
  EXPECT_EQ(seven.logWidth(), LogWidth::I32);
  EXPECT_EQ(seven.divisibility(), 7);
  EXPECT_EQ(seven.hasRange(), bool(FLYDSL_CORE_ENABLE_RANGE_INFERENCE));
  auto wide = i(int64_t{1} << 40).withFastDivMod();
  ASSERT_TRUE(wide.isDInt());
  EXPECT_EQ(wide.logWidth(), LogWidth::I64);
  EXPECT_EQ(wide.divisibility(), 1);
  for (auto invalid :
       {i(0), i(-3), rat(1, 2), Leaf<Trait>::getNone(), Leaf<Trait>::getBasis(i(2), {0})})
    EXPECT_EQ(invalid.withFastDivMod().errorCode(), ErrorCode::FastDivModRequiresPositiveInt);
  // No other leaf has the flag.
  for (auto other : {i(7), d(), rat(1, 2), Leaf<Trait>::getNone(), Leaf<Trait>::getBasis(d(), {1})})
    EXPECT_FALSE(other.hasFastDivMod());
  Leaf<Trait> fast = d().withFastDivMod();
  ASSERT_FALSE(fast.isError());
  EXPECT_TRUE(fast.hasFastDivMod());
  EXPECT_EQ(toString(fast), "?{fdm}");
  EXPECT_EQ(*Leaf<Trait>::fromString("?i64{div=4 fdm}"), d(LogWidth::I64, 4, nullopt, true));

  // Trait never makes a divisor; one survives only an identity.
  EXPECT_EQ(fast * i(1), fast);
  EXPECT_EQ(fast / i(1), fast);
  EXPECT_EQ(fast | i(0), fast);
  EXPECT_FALSE((fast * i(2)).hasFastDivMod());
  EXPECT_FALSE((fast + d()).hasFastDivMod());
  EXPECT_FALSE((d() / fast).hasFastDivMod());
  EXPECT_FALSE((d() % fast).hasFastDivMod());
  // A divisor is positive.
  EXPECT_EQ(abs(fast), fast);
  EXPECT_EQ(signum(fast), i(1));
  // A basis scale is a plain integer.
  auto basis = Leaf<Trait>::getBasis(fast, {1});
  ASSERT_TRUE(basis.isBasis());
  EXPECT_FALSE(basis.scale().hasFastDivMod());
  EXPECT_FALSE((Leaf<Trait>::getBasis(i(1), {0}) * fast).scale().hasFastDivMod());
}

TEST(TraitCategory, Top) {
  EXPECT_TRUE(d().isTop());
  EXPECT_FALSE(d(LogWidth::I32, 8).isTop());
  EXPECT_EQ(d(LogWidth::I32, 1, {{0, 7}}).isTop(), !bool(FLYDSL_CORE_ENABLE_RANGE_INFERENCE));
  EXPECT_FALSE(d(LogWidth::I32, 1, nullopt, true).isTop());
}

TEST(TraitCategory, Ranges) {
  EXPECT_FALSE(d().hasRange());
#if FLYDSL_CORE_ENABLE_RANGE_INFERENCE
  EXPECT_TRUE(d(LogWidth::I32, 1, {{0, 7}}).hasRange());
  Leaf<Trait> ranged = d().withRange(0, 7);
  ASSERT_FALSE(ranged.isError());
  EXPECT_TRUE(ranged.hasRange());
  EXPECT_EQ(ranged.rangeMin(), 0);
  EXPECT_EQ(ranged.rangeMax(), 7);
  EXPECT_EQ(d().withRange(7, 0).errorCode(), ErrorCode::InvalidDynamicRange);
  EXPECT_EQ(i(4).withRange(0, 7).errorCode(), ErrorCode::RangeRequiresDynamicInt);
#else
  for (auto value :
       {d(), d(LogWidth::I8, 4), d().withFastDivMod(), i(4), Leaf<Trait>::getNone(), rat(1, 2),
        Leaf<Trait>::getBasis(d(), {1}), Leaf<Trait>::getError(ErrorCode::DivisionByZero)}) {
    EXPECT_EQ(value.withRange(0, 7), value);
    EXPECT_EQ(value.withRange(7, 0), value);
    EXPECT_EQ(value.withRange(INT32_MIN, INT32_MAX), value);
  }
#endif
}

#if !FLYDSL_CORE_ENABLE_RANGE_INFERENCE
TEST(TraitCategory, ArithmeticDoesNotIntroduceRangeMetadata) {
  for (auto value : {d() % i(8), lt(d(), i(0)), signum(d())}) {
    ASSERT_TRUE(value.isDInt());
    EXPECT_FALSE(value.hasRange());
    EXPECT_FALSE(value.hasFastDivMod());
    EXPECT_EQ(value, d());
    EXPECT_EQ(hash_value(value), hash_value(d()));
  }
}
#endif

TEST(TraitCategory, ModRange) {
  Leaf<Trait> m = d() % i(8);
  ASSERT_TRUE(m.isDInt());
#if FLYDSL_CORE_ENABLE_RANGE_INFERENCE
  ASSERT_TRUE(m.hasRange());
  EXPECT_EQ(m.rangeMin(), -7);
  EXPECT_EQ(m.rangeMax(), 7);
#else
  EXPECT_EQ(m, d());
#endif
}

TEST(TraitCategory, Facts) {
  EXPECT_EQ(d(LogWidth::I32, 8) * d(LogWidth::I32, 4), d(LogWidth::I32, 32));
  EXPECT_EQ(d(LogWidth::I32, 8) + d(LogWidth::I32, 6), d(LogWidth::I32, 2));
  EXPECT_EQ(d(LogWidth::I32, 1, {{0, 7}}) + d(LogWidth::I32, 1, {{0, 3}}),
            d(LogWidth::I32, 1, {{0, 10}}));
  EXPECT_EQ(d(LogWidth::I32, 1, {{0, 7}}) + d(), d());
  EXPECT_EQ(d(LogWidth::I32, 8) * i(3), d(LogWidth::I32, 24));
  EXPECT_EQ(d(LogWidth::I32, 8) / i(4), d(LogWidth::I32, 2));
  EXPECT_EQ(d(LogWidth::I32, 8) / i(3), d());
  EXPECT_EQ(d() % i(8), d(LogWidth::I32, 1, {{-7, 7}}));
  EXPECT_EQ(d(LogWidth::I32, 8) % i(4), i(0));
}

TEST(TraitCategory, StaticZeroPreservesDynamicDivisibility) {
  auto zero = Leaf<Trait>::getSInt(0);
  auto dynamic = Leaf<Trait>::getDInt(LogWidth::I64, 8);
  EXPECT_EQ((zero + dynamic).divisibility(), 8);
  EXPECT_EQ((dynamic + zero).divisibility(), 8);
  EXPECT_EQ((zero - dynamic).divisibility(), 8);
  EXPECT_EQ((dynamic - zero).divisibility(), 8);
}

TEST(TraitCategory, SoundnessAndSpecialize) {
  Leaf<Trait> sum = d(LogWidth::I32, 8) + d(LogWidth::I32, 6);
  EXPECT_EQ(sum.divisibility(), 2);
  EXPECT_EQ(40 % 2, 0);

  Leaf<Trait> rng = d(LogWidth::I32, 1, {{0, 7}}) + d(LogWidth::I32, 1, {{0, 3}});
#if FLYDSL_CORE_ENABLE_RANGE_INFERENCE
  EXPECT_TRUE(rng.hasRange());
  EXPECT_EQ(rng.rangeMin(), 0);
  EXPECT_EQ(rng.rangeMax(), 10);
  EXPECT_GE(7 + 3, rng.rangeMin());
  EXPECT_LE(7 + 3, rng.rangeMax());
#else
  EXPECT_EQ(rng, d());
#endif

  Leaf<Trait> g3 = d(LogWidth::I32, 8) * i(3);
  Leaf<Trait> promoted = d(LogWidth::I32, 8) * d(LogWidth::I32, 3, {{3, 3}});
  EXPECT_EQ(g3.divisibility(), promoted.divisibility());
  Leaf<Trait> fd3 = d(LogWidth::I32, 8) / i(4);
  Leaf<Trait> fd2 = d(LogWidth::I32, 8) / d(LogWidth::I32, 4, {{4, 4}});
  EXPECT_EQ(fd3.divisibility() % fd2.divisibility(), 0);
  EXPECT_GE(fd3.divisibility(), fd2.divisibility());
}

TEST(TraitCategory, AddSubMul) {
  EXPECT_EQ(i(5) + i(4), i(9));
  EXPECT_EQ(rat(1, 2) + rat(1, 3), rat(5, 6));
  EXPECT_EQ(i(12) * rat(1, 4), i(3));
  EXPECT_EQ(d() * i(0), i(0));
  EXPECT_EQ(d(LogWidth::I64) * i(0), i(0));
  EXPECT_EQ(d() * i(1), d());
  EXPECT_EQ(d() + d(LogWidth::I64), d(LogWidth::I64));
  EXPECT_TRUE((rat(1, 2) * d()).isError());
  EXPECT_EQ((rat(1, 2) * d()).errorCode(), ErrorCode::DynamicRatioUnsupported);
  EXPECT_EQ(d(LogWidth::I32, 8) * i(3), d(LogWidth::I32, 24));
  EXPECT_EQ(i(9) - i(4), i(5));
  EXPECT_EQ(i(3) * i(4), i(12));
}

TEST(TraitCategory, ExactDiv) {
  EXPECT_EQ(exact_div(i(3), i(4)), rat(3, 4));
  EXPECT_EQ(exact_div(i(8), i(4)), i(2));
  EXPECT_EQ(exact_div(d(LogWidth::I32, 8), i(4)), d(LogWidth::I32, 2));
  EXPECT_EQ(exact_div(d(LogWidth::I32, 8), i(3)).errorCode(), ErrorCode::InexactDivision);
  EXPECT_EQ(exact_div(d(), i(2)).errorCode(), ErrorCode::InexactDivision);
  EXPECT_EQ(exact_div(i(1), d()).errorCode(), ErrorCode::InexactDivision);
  EXPECT_EQ(exact_div(i(0), d()), i(0));
  EXPECT_EQ(exact_div(i(3), i(0)).errorCode(), ErrorCode::DivisionByZero);
}

TEST(TraitCategory, SafeDiv) {
  EXPECT_EQ(safe_div(i(8), i(4)), i(2));
  EXPECT_EQ(safe_div(i(3), i(4)).errorCode(), ErrorCode::InexactDivision);
  EXPECT_EQ(safe_div(d(LogWidth::I32, 8), i(4)), d(LogWidth::I32, 2));
  // Exactness is only checked between static operands; a dynamic operand
  // makes safe_div the truncating quotient.
  EXPECT_EQ(safe_div(d(), i(4)), d() / i(4));
  EXPECT_EQ(safe_div(d(LogWidth::I32, 6), i(4)), d());
  EXPECT_EQ(safe_div(i(3), d()), i(3) / d());
  EXPECT_EQ(safe_div(d(), d()), d() / d());
  EXPECT_EQ(safe_div(d(), i(0)).errorCode(), ErrorCode::DivisionByZero);
}

TEST(TraitCategory, FloorCeilMod) {
  EXPECT_EQ(i(7) / i(2), i(3));
  EXPECT_EQ(ceil_div(i(7), i(2)), i(4));
  EXPECT_EQ(i(-7) / i(2), i(-3));
  EXPECT_EQ(i(-7) % i(2), i(-1));
  EXPECT_EQ(d() % i(-1), i(0));
  EXPECT_EQ((rat(1, 2) / i(2)).errorCode(), ErrorCode::UnsupportedRatioOperation);
  EXPECT_EQ(d() % i(1), i(0));
  EXPECT_EQ(i(0) / d(), i(0));
  EXPECT_EQ(d() % i(8), d(LogWidth::I32, 1, {{-7, 7}}));
  EXPECT_EQ(d(LogWidth::I32, 6) % i(8), d(LogWidth::I32, 2, {{-7, 7}}));
}

TEST(TraitCategory, ShapeDiv) {
  EXPECT_EQ(shape_div(i(8), i(4)), i(2));
  EXPECT_EQ(shape_div(i(4), i(8)), i(1));
  EXPECT_EQ(shape_div(i(6), i(4)).errorCode(), ErrorCode::IncompatibleShapeDivision);
  EXPECT_EQ(shape_div(i(0), i(3)), i(0));
  // An exact quotient keeps the remaining divisibility, as floor division does.
  EXPECT_EQ(shape_div(d(LogWidth::I32, 64), i(32)), d(LogWidth::I32, 2));
  EXPECT_EQ(ceil_div(d(LogWidth::I32, 64), i(48)), d(LogWidth::I32));
}

TEST(TraitCategory, MinMax) {
  EXPECT_EQ(min(i(3), i(5)), i(3));
  EXPECT_EQ(max(i(3), i(5)), i(5));
  EXPECT_EQ(min(d(), i(0)), d());
  EXPECT_EQ(min(d(LogWidth::I32, 1, {{-5, 5}}), i(0)), d(LogWidth::I32, 1, {{-5, 0}}));
  EXPECT_EQ(min(rat(1, 2), i(1)).errorCode(), ErrorCode::UnsupportedRatioOperation);
}

TEST(TraitCategory, Gcd) {
  EXPECT_EQ(gcd(i(12), i(8)), i(4));
  EXPECT_EQ(gcd(i(-12), i(8)), i(4));
  EXPECT_EQ(gcd(i(12), i(0)), i(12));
  EXPECT_EQ(gcd(i(0), i(-12)), i(12));
  EXPECT_EQ(gcd(i(1), d()), i(1));
  EXPECT_EQ(gcd(d(LogWidth::I32, 8), i(8)), i(8));
  EXPECT_EQ(gcd(d(LogWidth::I32, 8), i(12)), d(LogWidth::I32, 4));
  EXPECT_EQ(gcd(d(LogWidth::I32, 8), d(LogWidth::I32, 12)), d(LogWidth::I32, 4));
  EXPECT_EQ(gcd(rat(1, 2), i(1)).errorCode(), ErrorCode::UnsupportedRatioOperation);
}

TEST(TraitCategory, Abs) {
  EXPECT_EQ(abs(i(12)), i(12));
  EXPECT_EQ(abs(i(-12)), i(12));
  EXPECT_EQ(abs(i(0)), i(0));
  EXPECT_EQ(abs(rat(-3, 4)), rat(3, 4));
  EXPECT_EQ(abs(d()), d());
  EXPECT_EQ(abs(d(LogWidth::I32, 8, {{-5, 3}})), d(LogWidth::I32, 8, {{0, 5}}));
  EXPECT_EQ(abs(d(LogWidth::I32, 8, {{-9, -2}})), d(LogWidth::I32, 8, {{2, 9}}));
  EXPECT_EQ(abs(d(LogWidth::I32, 8, {{2, 9}})), d(LogWidth::I32, 8, {{2, 9}}));
  EXPECT_EQ(abs(bs(i(-4), {2})), bs(i(4), {2}));
  EXPECT_EQ(abs(Leaf<Trait>::getNone()), i(0));
}

TEST(TraitCategory, Signum) {
  EXPECT_EQ(signum(i(12)), i(1));
  EXPECT_EQ(signum(i(-12)), i(-1));
  EXPECT_EQ(signum(i(0)), i(0));
  EXPECT_EQ(signum(rat(-3, 4)), i(-1));
  EXPECT_EQ(signum(rat(3, 4)), i(1));
  EXPECT_EQ(signum(d()), d(LogWidth::I32, 1, {{-1, 1}}));
  EXPECT_EQ(signum(d(LogWidth::I32, 8, {{-5, 3}})), d(LogWidth::I32, 1, {{-1, 1}}));
#if FLYDSL_CORE_ENABLE_RANGE_INFERENCE
  EXPECT_EQ(signum(d(LogWidth::I32, 8, {{-9, -2}})), i(-1));
  EXPECT_EQ(signum(d(LogWidth::I32, 8, {{2, 9}})), i(1));
#else
  EXPECT_EQ(signum(d(LogWidth::I32, 8)), d());
#endif
  EXPECT_EQ(signum(d(LogWidth::I32, 8, {{0, 9}})), d(LogWidth::I32, 1, {{0, 1}}));
  EXPECT_EQ(signum(d(LogWidth::I32, 8, {{-9, 0}})), d(LogWidth::I32, 1, {{-1, 0}}));
  EXPECT_EQ(signum(bs(i(-4), {2})), i(-1));
  EXPECT_EQ(signum(Leaf<Trait>::getNone()), i(0));
}

TEST(TraitCategory, Bitwise) {
  EXPECT_EQ((i(6) & i(3)), i(2));
  EXPECT_EQ((i(6) | i(3)), i(7));
  EXPECT_EQ((i(6) ^ i(3)), i(5));

  EXPECT_EQ((d() & i(0)), i(0));
  EXPECT_EQ((d() & i(-1)), d());
  EXPECT_EQ((d() | i(0)), d());
  EXPECT_TRUE((d() | i(-1)).isDInt());
  EXPECT_EQ((d() ^ i(0)), d());
  EXPECT_EQ((d() & d(LogWidth::I64)).logWidth(), LogWidth::I64);
  // A multiple of 8 has no bit below 8: only a mask from bit 3 up can keep one.
  EXPECT_EQ((d(LogWidth::I32, 24) & i(7)), i(0));
  EXPECT_EQ((i(7) & d(LogWidth::I32, 8)), i(0));
  EXPECT_EQ((d(LogWidth::I32, 8) & i(8)), d(LogWidth::I32, 8));
  EXPECT_TRUE((d(LogWidth::I32, 8) & i(-8)).isDInt());

  EXPECT_EQ((i(3) << i(2)), i(12));
  EXPECT_EQ((i(12) >> i(2)), i(3));
  EXPECT_EQ((i(8) << i(-3)), i(1));
  EXPECT_EQ((i(8) >> i(-3)), i(64));
  // Static shifts are exact within int64, like Python integers.
  EXPECT_EQ((i(1) << i(32)), i(int64_t{1} << 32));
  EXPECT_EQ((i(5) << i(40)), i(int64_t{5} << 40));
  EXPECT_EQ((i(-5) >> i(40)), i(-1));
  EXPECT_EQ((i(-1) >> i(64)), i(-1));
  EXPECT_EQ((i(1) >> i(64)), i(0));
  EXPECT_EQ((i(0) << i(100)), i(0));
  EXPECT_EQ((i(-1) << i(63)), i(INT64_MIN));
  EXPECT_EQ((d() << i(0)), d());
  EXPECT_EQ((d() >> i(2)), d());
  EXPECT_EQ((d(LogWidth::I32, 8) << i(1)), d(LogWidth::I32, 16));
  EXPECT_EQ((d(LogWidth::I32, 8) >> i(1)), d(LogWidth::I32, 4));

  EXPECT_EQ((rat(1, 2) & i(1)).errorCode(), ErrorCode::UnsupportedRatioOperation);
  EXPECT_EQ((bs(i(4), {0}) & i(0)), i(0));
}

TEST(TraitCategory, RatioReduce) {
  EXPECT_TRUE(rat(2, 4).isRatio());
  EXPECT_EQ(rat(2, 4).num(), 1);
  EXPECT_EQ(rat(2, 4).den(), 2);
  EXPECT_TRUE(rat(8, 2).isSInt());
  EXPECT_EQ(rat(8, 2).staticValue(), 4);
  EXPECT_EQ(rat(1, -2).num(), -1);
  EXPECT_EQ(rat(1, -2).den(), 2);
  EXPECT_EQ(rat(-1, -2).num(), 1);
  EXPECT_EQ(rat(1, 2).logWidth(), LogWidth::I32);
  EXPECT_EQ(rat(1, 2).bitWidth(), 32);
  EXPECT_EQ(Leaf<Trait>::getRatio(1, 0).errorCode(), ErrorCode::DivisionByZero);
  EXPECT_EQ(Leaf<Trait>::getRatio(INT32_MIN, -1).staticValue(), 1ll << 31);
  EXPECT_EQ(hash_value(rat(2, 4)), hash_value(rat(1, 2)));
}

TEST(TraitCategory, AlignedDynamicRatioBasis) {
  auto aligned = d(LogWidth::I64, 8);
  auto scale = rat(3, 2);
  EXPECT_EQ(scale * aligned, d(LogWidth::I64, 12));
  EXPECT_EQ(aligned * scale, scale * aligned);
  EXPECT_EQ(bs(scale, {2, 1}) * aligned, bs(d(LogWidth::I64, 12), {2, 1}));
  EXPECT_EQ((scale * d()).errorCode(), ErrorCode::DynamicRatioUnsupported);
  EXPECT_EQ((scale + aligned).errorCode(), ErrorCode::DynamicRatioUnsupported);
}

TEST(TraitCategory, StaticRatioComparisons) {
  EXPECT_TRUE(eq(rat(1, 2), rat(2, 4)).isOne());
  EXPECT_TRUE(ne(rat(1, 2), rat(3, 4)).isOne());
  EXPECT_TRUE(lt(rat(1, 2), rat(3, 4)).isOne());
  EXPECT_TRUE(lt(rat(-1, 2), i(0)).isOne());
  EXPECT_TRUE(lt(rat(1, 3), i(int64_t{1} << 40)).isOne());
  EXPECT_TRUE(lt(i(-(int64_t{1} << 40)), rat(-1, 3)).isOne());
  EXPECT_TRUE(eq(i(int64_t{1} << 40), rat(1, 3)).isZero());
}

TEST(TraitCategory, BasisFactories) {
  EXPECT_EQ(Leaf<Trait>::getNone(), Leaf<Trait>::getNone());
  EXPECT_TRUE(Leaf<Trait>::getNone().isNone());
  EXPECT_TRUE(i(4).isInt());
  EXPECT_TRUE(bs(i(4), {2}).isBasis());
  EXPECT_EQ(bs(i(4), {2}).scale(), i(4));
  EXPECT_EQ(bs(i(4), {2}).modeCount(), 1);
  EXPECT_EQ(bs(i(4), {2}).mode(0), 2);
  EXPECT_TRUE(bs(i(1), {0, 1}).isBasis());
  EXPECT_EQ(bs(i(1), {0, 1}).modeCount(), 2);
  EXPECT_EQ(bs(i(1), {0, 1}).mode(0), 0);
  EXPECT_EQ(bs(i(1), {0, 1}).mode(1), 1);
  EXPECT_TRUE(Leaf<Trait>::getBasis(i(4), {}).isInt());
  EXPECT_EQ(Leaf<Trait>::getBasis(i(4), {}), i(4));
  EXPECT_EQ(Leaf<Trait>::getBasis(i(0), {2}), i(0));
  EXPECT_EQ(Leaf<Trait>::getBasis(i(4), {-1}).errorCode(), ErrorCode::BasisModeOutOfRange);
  EXPECT_EQ(Leaf<Trait>::getBasis(i(4), {256}).errorCode(), ErrorCode::BasisModeOutOfRange);
  EXPECT_EQ(Leaf<Trait>::getBasis(i(1), {0, 1, 2, 3, 4, 5, 6, 7}).errorCode(),
            ErrorCode::TooManyBasisModes);
  EXPECT_EQ(
      Leaf<Trait>::getBasis(Leaf<Trait>::getError(ErrorCode::DivisionByZero), {2}).errorCode(),
      ErrorCode::DivisionByZero);
  EXPECT_TRUE(
      Leaf<Trait>::getBasis(Leaf<Trait>::getError(ErrorCode::DivisionByZero), {}).isError());
  EXPECT_EQ(Leaf<Trait>::getBasis(Leaf<Trait>::getError(ErrorCode::DivisionByZero), {}).errorCode(),
            ErrorCode::DivisionByZero);
}

TEST(TraitCategory, Predicates) {
  EXPECT_TRUE(i(4).isInt());
  EXPECT_FALSE(i(4).isBasis(LeafKind::StaticInt));
  EXPECT_TRUE(d().isInt());
  EXPECT_FALSE(rat(1, 2).isInt());
  EXPECT_TRUE(Leaf<Trait>::getNone().isInt());
  EXPECT_TRUE(Leaf<Trait>::getNone().isSInt());
  EXPECT_TRUE(Leaf<Trait>::getNone().isZero());
  EXPECT_FALSE(Leaf<Trait>::getNone().isOne());
  EXPECT_FALSE(Leaf<Trait>::getNone().isSInt(4));
  EXPECT_EQ(Leaf<Trait>::getNone().staticValue(), 0);
  EXPECT_FALSE(Leaf<Trait>::getError(ErrorCode::InvalidBasisScale).isInt());
  EXPECT_FALSE(Leaf<Trait>::getError(ErrorCode::InvalidBasisScale).isBasis());

  Leaf<Trait> basis = bs(i(4), {2});
  EXPECT_TRUE(basis.isBasis());
  EXPECT_TRUE(basis.isBasis(LeafKind::StaticInt));
  EXPECT_FALSE(basis.isBasis(LeafKind::DynamicInt));
  EXPECT_FALSE(basis.isInt());
  EXPECT_FALSE(basis.isSInt());
  EXPECT_EQ(basis.kind(), LeafKind::StaticInt | LeafKind::Basis);
  EXPECT_TRUE(basis.scale().isSInt());

  Leaf<Trait> dynamicBasis = bs(d(), {1});
  EXPECT_TRUE(dynamicBasis.isBasis(LeafKind::DynamicInt));
  EXPECT_FALSE(dynamicBasis.isBasis(LeafKind::StaticInt));
  EXPECT_FALSE(dynamicBasis.isDInt());
  EXPECT_EQ(dynamicBasis.kind(), LeafKind::DynamicInt | LeafKind::Basis);
  EXPECT_EQ(dynamicBasis.scale(), d());

  Leaf<Trait> ratioBasis = bs(rat(1, 2), {0});
  EXPECT_TRUE(ratioBasis.isBasis(LeafKind::Ratio));
  EXPECT_FALSE(ratioBasis.isRatio());
  EXPECT_EQ(ratioBasis.kind(), LeafKind::Ratio | LeafKind::Basis);
  EXPECT_EQ(ratioBasis.scale(), rat(1, 2));
}

TEST(TraitCategory, BasisArithmetic) {
  EXPECT_EQ(i(3) * bs(i(4), {2}), bs(i(12), {2}));
  EXPECT_EQ(exact_div(bs(i(12), {2}), i(4)), bs(i(3), {2}));
  EXPECT_EQ(exact_div(bs(i(3), {2}), i(4)), bs(rat(3, 4), {2}));
  EXPECT_EQ(bs(i(4), {2}) + i(0), bs(i(4), {2}));
  EXPECT_EQ((bs(i(4), {2}) + bs(i(1), {0})).errorCode(), ErrorCode::IncompatibleBasis);
  EXPECT_EQ(Leaf<Trait>::getNone() + i(1), i(1));
  EXPECT_EQ(bs(d(LogWidth::I32, 8), {1}) * i(3), bs(d(LogWidth::I32, 24), {1}));
  EXPECT_EQ(bs(i(4), {2}) * i(0), i(0));
  EXPECT_EQ(i(0) + bs(i(4), {2}), bs(i(4), {2}));
  EXPECT_EQ(-bs(i(4), {2}), bs(i(-4), {2}));
  EXPECT_EQ(eq(bs(i(4), {2}), bs(i(4), {3})), i(0));
  EXPECT_EQ(ne(bs(i(4), {2}), bs(i(4), {3})), i(1));
  EXPECT_EQ(min(bs(i(4), {2}), i(0)).errorCode(), ErrorCode::UnsupportedBasisOperation);
  EXPECT_EQ(min(i(3), i(5)), i(3));
  EXPECT_EQ(safe_div(bs(i(12), {2}), i(4)), bs(i(3), {2}));
  EXPECT_EQ(shape_div(bs(i(8), {2}), i(4)), bs(i(2), {2}));
  EXPECT_EQ((bs(i(4), {2}) * i(3)), bs(i(12), {2}));
}

TEST(TraitCategory, BasisNeverEqualsScalar) {
  for (auto scalar : {Leaf<Trait>::getSInt(1), Leaf<Trait>::getDInt(LogWidth::I64, 8)}) {
    auto basis = Leaf<Trait>::getBasis(scalar, {0});
    EXPECT_TRUE(eq(basis, scalar).isZero());
    EXPECT_TRUE(eq(scalar, basis).isZero());
    EXPECT_TRUE(ne(basis, scalar).isOne());
  }
}

TEST(TraitCategory, Errors) {
  Leaf<Trait> ov = Leaf<Trait>::getError(ErrorCode::DivisionByZero);
  EXPECT_TRUE(ov.isError());
  EXPECT_EQ(ov.errorCode(), ErrorCode::DivisionByZero);
  EXPECT_FALSE(ov.isInt());
  EXPECT_FALSE(ov.isRatio());
  EXPECT_EQ(Leaf<Trait>::getError(ErrorCode::DynamicRatioUnsupported).errorCode(),
            ErrorCode::DynamicRatioUnsupported);
  EXPECT_EQ(Leaf<Trait>::getError(ErrorCode::InvalidBasisScale).errorCode(),
            ErrorCode::InvalidBasisScale);
  EXPECT_EQ(Leaf<Trait>::getError(ErrorCode::InexactDivision).errorCode(),
            ErrorCode::InexactDivision);
  EXPECT_EQ(Leaf<Trait>::getError(ErrorCode::IncompatibleShapeDivision).errorCode(),
            ErrorCode::IncompatibleShapeDivision);
  EXPECT_EQ(ov, Leaf<Trait>::getError(ErrorCode::DivisionByZero));
  EXPECT_NE(ov, Leaf<Trait>::getError(ErrorCode::InexactDivision));

  EXPECT_EQ(ov + i(1), ov);
  EXPECT_EQ((i(1) + Leaf<Trait>::getError(ErrorCode::InexactDivision)).errorCode(),
            ErrorCode::InexactDivision);
  EXPECT_EQ(ov + Leaf<Trait>::getError(ErrorCode::InexactDivision), ov);
  EXPECT_EQ(-ov, ov);
}

TEST(TraitCategory, ZeroDivAndNone) {
  EXPECT_EQ((i(4) / i(0)).errorCode(), ErrorCode::DivisionByZero);
  EXPECT_EQ(Leaf<Trait>::getNone() + i(1), i(1));
  EXPECT_EQ(eq(Leaf<Trait>::getNone(), i(0)), i(1));
  EXPECT_TRUE(Leaf<Trait>::getNone().isNone());
  EXPECT_NE(Leaf<Trait>::getNone(), i(0));
}

TEST(TraitCategory, Identity) {
  EXPECT_EQ(i(4), i(4));
  EXPECT_EQ(hash_value(i(4)), hash_value(i(4)));
  Leaf<Trait> a = d(LogWidth::I32, 8, {{0, 63}});
  Leaf<Trait> rhs = d(LogWidth::I32, 8, {{0, 63}});
  EXPECT_EQ(a, rhs);
  EXPECT_EQ(hash_value(a), hash_value(rhs));
  EXPECT_EQ(d(), d(LogWidth::I32, 1));
  EXPECT_EQ(d(LogWidth::I64), d(LogWidth::I64));
  EXPECT_EQ(d(LogWidth::I32, 8), d(LogWidth::I32, 8));
  EXPECT_EQ(d(LogWidth::I32, 1, {{0, 63}}), d(LogWidth::I32, 1, {{0, 63}}));
  EXPECT_EQ(d(LogWidth::I32, 1, nullopt, true), d(LogWidth::I32, 1, nullopt, true));
  EXPECT_EQ(rat(3, 4), rat(3, 4));
  EXPECT_TRUE(Leaf<Trait>::getNone().isNone());
  EXPECT_EQ(bs(i(4), {2}), bs(i(4), {2}));
}

TEST(TraitCategory, IdentityIgnoresInactiveStorage) {
  auto original = Leaf<Trait>::getDInt(LogWidth::I64, 8);
  auto dirty = original;
  dirty.payload64_ = 123;
  dirty.nmodes_ = 1;
  dirty.modes_[0] = 42;
  EXPECT_EQ(original, dirty);
  EXPECT_EQ(hash_value(original), hash_value(dirty));
  auto basis = Leaf<Trait>::getBasis(original, {2});
  auto dirtyBasis = basis;
  dirtyBasis.modes_[6] = 99;
  EXPECT_EQ(basis, dirtyBasis);
  EXPECT_EQ(hash_value(basis), hash_value(dirtyBasis));
  dirtyBasis.modes_[0] = 3;
  EXPECT_NE(basis, dirtyBasis);
}

TEST(TraitCategory, RatioAndBasisRatioIdentity) {
  auto ratio = Leaf<Trait>::getRatio(1, 2);
  auto equivalent = Leaf<Trait>::getRatio(2, 4);
  auto different = Leaf<Trait>::getRatio(1, 3);
  EXPECT_EQ(ratio, equivalent);
  EXPECT_EQ(hash_value(ratio), hash_value(equivalent));
  EXPECT_NE(ratio, different);
  auto basis = Leaf<Trait>::getBasis(ratio, {2, 1});
  auto equivalentBasis = Leaf<Trait>::getBasis(equivalent, {2, 1});
  EXPECT_EQ(basis, equivalentBasis);
  EXPECT_EQ(hash_value(basis), hash_value(equivalentBasis));
  EXPECT_NE(basis, Leaf<Trait>::getBasis(different, {2, 1}));
}

TEST(TraitCategory, RebuildFromAccessors) {
  auto rebuild = [](Leaf<Trait> s) -> Leaf<Trait> {
    if (s.isError())
      return Leaf<Trait>::getError(s.errorCode());
    if (s.isSInt())
      return Leaf<Trait>::getSInt(s.staticValue());
    if (s.isRatio())
      return Leaf<Trait>::getRatio(s.num(), s.den());
    optional<std::pair<int32_t, int32_t>> r;
    if (s.hasRange())
      r = {s.rangeMin(), s.rangeMax()};
    Leaf<Trait> rebuilt = Leaf<Trait>::getDInt(s.logWidth(), s.divisibility());
    if (r)
      rebuilt = rebuilt.withRange(r->first, r->second);
    if (s.hasFastDivMod())
      rebuilt = rebuilt.withFastDivMod();
    return rebuilt;
  };
  std::vector<Leaf<Trait>> samples = {
      i(0),
      i(4),
      i(-1),
      rat(1, 2),
      d(),
      d(LogWidth::I64),
      d(LogWidth::I32, 8),
      d(LogWidth::I32, 1, {{0, 63}}),
      d(LogWidth::I32, 1, nullopt, true),
      d(LogWidth::I64, 4, {{INT32_MIN, INT32_MAX}}),
      Leaf<Trait>::getError(ErrorCode::DivisionByZero),
      Leaf<Trait>::getError(ErrorCode::InvalidBasisScale),
  };
  for (Leaf<Trait> s : samples) {
    EXPECT_EQ(rebuild(s), s);
    if (s.isDInt()) {
      optional<std::pair<int32_t, int32_t>> r;
      if (s.hasRange())
        r = {s.rangeMin(), s.rangeMax()};
      Leaf<Trait> rebuilt = Leaf<Trait>::getDInt(s.logWidth(), s.divisibility());
      if (r)
        rebuilt = rebuilt.withRange(r->first, r->second);
      if (s.hasFastDivMod())
        rebuilt = rebuilt.withFastDivMod();
      EXPECT_EQ(rebuilt, s);
    }
  }
}

// Static arithmetic assumes that its results and intermediates fit i64; a
// result at the limits is still exact.
TEST(TraitCategory, StaticExtremes) {
  auto min = Leaf<Trait>::getSInt(INT64_MIN), max = Leaf<Trait>::getSInt(INT64_MAX);
  EXPECT_EQ(max - Leaf<Trait>::getOne() + Leaf<Trait>::getOne(), max);
  EXPECT_EQ(-max - Leaf<Trait>::getOne(), min);
  EXPECT_EQ(shape_div(Leaf<Trait>::getZero(), Leaf<Trait>::getSInt(3)), Leaf<Trait>::getZero());
  EXPECT_EQ(gcd(max, Leaf<Trait>::getSInt(7)), Leaf<Trait>::getSInt(7));
  EXPECT_EQ(exact_div(max, Leaf<Trait>::getSInt(7)), Leaf<Trait>::getSInt(INT64_MAX / 7));
  EXPECT_EQ(min / Leaf<Trait>::getSInt(-2), Leaf<Trait>::getSInt(int64_t{1} << 62));
}

TEST(TraitCategory, ErrorIdentityIgnoresFrames) {
  auto bare = Leaf<Trait>::getError(ErrorCode::IndexOutOfRange);
  auto located =
      Leaf<Trait>::getError(ErrorInfo(ErrorCode::IndexOutOfRange).withFrame(AlgebraOp::Take, 3));
  EXPECT_EQ(bare, located);
  EXPECT_EQ(hash_value(bare), hash_value(located));
  EXPECT_EQ(located.errorInfo().op(0), AlgebraOp::Take);
  EXPECT_NE(bare, Leaf<Trait>::getError(ErrorCode::IncompatibleProfile));
  // Text keeps only the reason, so every reason round-trips.
  for (auto raw = 0u; raw <= static_cast<unsigned>(ErrorCode::Last); ++raw) {
    auto leaf = Leaf<Trait>::getError(static_cast<ErrorCode>(raw));
    EXPECT_EQ(Leaf<Trait>::fromString(toString(leaf)), leaf) << toString(leaf);
  }
}

TEST(TraitCategory, Algebra) {
  using TypeTuple = IntTuple<Trait>;
  using TypeLayout = Layout<Trait>;

  TypeTuple shape =
      make_tuple(Leaf<Trait>::getSInt(2), Leaf<Trait>::getSInt(3), Leaf<Trait>::getSInt(4));
  TypeLayout layout = make_layout(shape);
  EXPECT_EQ(toString(layout), "(2,3,4):(1,2,6)");
  EXPECT_EQ(crd2idx(TypeTuple::getSInt(23), layout), TypeTuple::getSInt(23));
  EXPECT_EQ(toString(coalesce(layout)), "24:1");
  EXPECT_EQ(product(shape), TypeTuple::getSInt(24));

  auto composed = make_composed_layout(
      make_layout(make_tuple(Leaf<Trait>::getSInt(32)), make_tuple(Leaf<Trait>::getSInt(2))),
      TypeTuple::getSInt(3), layout);
  EXPECT_EQ(crd2idx(TypeTuple::getSInt(5), composed), TypeTuple::getSInt(16));
  EXPECT_TRUE(composed.isStatic());
}

// A leaf prints to text that parses back to the same leaf.
TEST(TraitCategory, TextRoundTrip) {
  auto dynamic = Leaf<Trait>::getDInt(LogWidth::I64, 8).withRange(-16, 64).withFastDivMod();
  for (auto leaf : {i(-7), d(), dynamic, Leaf<Trait>::getNone(), rat(-3, 5), bs(dynamic, {1, 3}),
                    bs(i(4), {2}), Leaf<Trait>::getError(ErrorCode::DivisionByZero)}) {
    auto parsed = Leaf<Trait>::fromString(toString(leaf));
    ASSERT_TRUE(parsed.has_value()) << toString(leaf);
    EXPECT_EQ(*parsed, leaf) << toString(leaf);
  }
}

// add_offset keeps the alignment that the offset preserves, in bits for sub-byte elements.
TEST(TraitCategory, PointerAlignment) {
  auto f32 = Pointer<Trait>(OpaqueValue::getType<float>(), 32, 32);
  EXPECT_EQ(add_offset(f32, i(0)).byteAlignment(), 32);
  EXPECT_EQ(add_offset(f32, i(6)).byteAlignment(), 8);
  EXPECT_EQ(add_offset(f32, d(LogWidth::I32, 4)).byteAlignment(), 16);
  EXPECT_EQ(add_offset(f32, i(0)), f32);

  // A non-power-of-two alignment keeps the part the offset preserves.
  auto twelve = Pointer<Trait>(OpaqueValue::getType<float>(), 32, 12);
  EXPECT_EQ(twelve.storageBitAlignment(), 96);
  EXPECT_EQ(add_offset(twelve, i(1)).byteAlignment(), 4);
  EXPECT_EQ(add_offset(twelve, i(1)).storageBitAlignment(), 32);

  auto i4 = Pointer<Trait>(OpaqueValue::getType<int>(), 4, 16);
  EXPECT_EQ(add_offset(i4, d()).byteAlignment(), 1);
  EXPECT_EQ(add_offset(i4, d()).bitAlignment(), 4);
  auto i3 = Pointer<Trait>(OpaqueValue::getType<int>(), 3, 16);
  EXPECT_EQ(add_offset(i3, i(1)).byteAlignment(), 1);
  EXPECT_EQ(add_offset(i3, i(1)).bitAlignment(), 1);

  auto i32 = Pointer<Trait>(OpaqueValue::getType<int>(), 32, 16);
  EXPECT_EQ(add_offset(i32, i(INT64_MIN)).storageBitAlignment(), 128);
  EXPECT_EQ(add_offset(i32, i(INT64_MAX)).storageBitAlignment(), 32);
  EXPECT_EQ(add_offset(i32, i(-2)).storageBitAlignment(), 64);
  EXPECT_EQ(add_offset(i32, d(LogWidth::I64, 4)).storageBitAlignment(), 128);
  auto i3Aligned = Pointer<Trait>(OpaqueValue::getType<int>(), 3, 3, int64_t{24});
  EXPECT_EQ(add_offset(i3Aligned, i(INT64_MIN)).storageBitAlignment(), 24);
  EXPECT_EQ(add_offset(i3Aligned, i(INT64_MAX)).storageBitAlignment(), 3);
}

// A swizzle acts on byte addresses, so it keeps the bytes below its base
// whatever the element type.
TEST(TraitCategory, SwizzledPointerAlignment) {
  for (auto bits : {8, 16, 32, 64}) {
    for (auto shift : {-1, 1}) {
      auto pointer =
          Pointer<Trait>(OpaqueValue::getType<int>(), bits, 128, Swizzle<Trait>(1, 2, shift));
      EXPECT_EQ(pointer.storageBitAlignment(), 1024);
      EXPECT_EQ(pointer.bitAlignment(), 32);
    }
  }
  auto pointer = Pointer<Trait>(OpaqueValue::getType<float>(), 32, 128, Swizzle<Trait>(1, 2, 1));
  auto shifted = add_offset(pointer, i(int64_t{1} << 30));
  EXPECT_EQ(shifted.byteAlignment(), 128);
  EXPECT_EQ(shifted.storageBitAlignment(), 1024);
  EXPECT_EQ(shifted.bitAlignment(), 32);
}

// A logical bool stored in bytes advances by the storage width.
TEST(TraitCategory, ByteStoredBoolPointer) {
  auto byteBool = Pointer<Trait>(OpaqueValue::getType<bool>(), 1, 16, int64_t{128},
                                 Swizzle<Trait>{}, OpaqueValue::getType<void>(), 8);
  auto packedBool = Pointer<Trait>(OpaqueValue::getType<bool>(), 1, 16);
  EXPECT_NE(byteBool, packedBool);
  auto shifted = add_offset(byteBool, i(1));
  EXPECT_EQ(shifted.elementBitWidth(), 1);
  EXPECT_EQ(shifted.storageElementBitWidth(), 8);
  EXPECT_EQ(shifted.storageBitAlignment(), 8);
}

// The element type and the address space are part of a pointer's identity, beyond its widths.
TEST(TraitCategory, PointerIdentity) {
  struct F16 {};
  struct BF16 {};
  EXPECT_NE(Pointer<Trait>(OpaqueValue::getType<F16>(), 16, 16),
            Pointer<Trait>(OpaqueValue::getType<BF16>(), 16, 16));
  EXPECT_EQ(Pointer<Trait>(OpaqueValue::getType<F16>(), 16, 16),
            Pointer<Trait>(OpaqueValue::getType<F16>(), 16, 16));
  auto shared =
      Pointer<Trait>(OpaqueValue::getType<float>(), 32, 16, OpaqueValue::get(int64_t{17}));
  EXPECT_NE(shared,
            Pointer<Trait>(OpaqueValue::getType<float>(), 32, 16, OpaqueValue::get(int64_t{18})));
  EXPECT_EQ(add_offset(shared, i(2)).addressSpace(), shared.addressSpace());
}
